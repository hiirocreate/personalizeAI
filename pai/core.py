"""環境セットアップ・モデル取得・ComfyUI プロセス管理。"""
import os
import socket
import subprocess
import threading
import time
import json
import urllib.request

DATA = os.environ.get("PAI_DATA", "/content/personalizeAI_data")  # Drive 等の永続保存先
WORK = os.environ.get("PAI_WORK", "/content/pai_work")            # 毎回作り直す作業領域
COMFY = f"{WORK}/ComfyUI"
MODELS = f"{COMFY}/models"
PORT = 8188
HF = "https://huggingface.co"

for d in ("loras", "outputs", "datasets", "3d"):
    os.makedirs(f"{DATA}/{d}", exist_ok=True)


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"{cmd}\n{(r.stdout + r.stderr)[-3000:]}")
    return r.stdout


# ------------------------------------------------------------------ models
CATALOG = {
    "flux": [
        (f"{HF}/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf", "unet"),
        (f"{HF}/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q5_K_M.gguf", "clip"),
        (f"{HF}/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", "clip"),
        (f"{HF}/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors", "vae"),
    ],
    "animagine": [(f"{HF}/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0-opt.safetensors", "checkpoints")],
    "realvis": [(f"{HF}/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors", "checkpoints")],
    "sdxl_vae": [(f"{HF}/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors", "vae")],
    "wan": [
        (f"{HF}/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q4_K_M.gguf", "unet"),
        (f"{HF}/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf", "clip"),
        (f"{HF}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors", "vae"),
    ],
}
SIZES_GB = {"flux": 10, "animagine": 7, "realvis": 7, "sdxl_vae": 0.3, "wan": 7.5}
_locks = {}
_locks_guard = threading.Lock()


def model_path(key):
    url, sub = CATALOG[key][0]
    return f"{MODELS}/{sub}/{url.rsplit('/', 1)[-1]}"


def ready(key):
    return all(os.path.exists(f"{MODELS}/{sub}/{url.rsplit('/', 1)[-1]}") for url, sub in CATALOG[key])


def ensure(key):
    """モデルを (未取得なら) ダウンロード。並行呼び出しでも二重取得しない。"""
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        for url, sub in CATALOG[key]:
            name = url.rsplit("/", 1)[-1]
            dst = f"{MODELS}/{sub}"
            if os.path.exists(f"{dst}/{name}"):
                continue
            os.makedirs(dst, exist_ok=True)
            sh(f'aria2c -q -x16 -s16 -k1M --console-log-level=error --allow-overwrite=true '
               f'-d "{dst}" -o "{name}.part" "{url}"')
            os.rename(f"{dst}/{name}.part", f"{dst}/{name}")


def prefetch(*keys):
    threading.Thread(target=lambda: [ensure(k) for k in keys], daemon=True).start()


# ------------------------------------------------------------------ ComfyUI
_proc = None


def install():
    if not os.path.exists("/usr/bin/aria2c"):
        sh("apt-get -qq install -y aria2")
    if not os.path.exists(COMFY):
        print("ComfyUI をインストール中…")
        sh(f"git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFY}")
        sh(f"git clone -q --depth 1 https://github.com/city96/ComfyUI-GGUF {COMFY}/custom_nodes/ComfyUI-GGUF")
        sh(f"pip install -q -r {COMFY}/requirements.txt gguf pillow-heif")
    try:
        import gradio  # noqa: F401
    except ImportError:
        sh('pip install -q "gradio>=5,<6"')
    loras = f"{MODELS}/loras"
    if not os.path.islink(loras):
        sh(f'rm -rf "{loras}" && ln -s "{DATA}/loras" "{loras}"')


def _port_open():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def start_comfy():
    global _proc
    if _proc and _proc.poll() is None and _port_open():
        return
    _proc = subprocess.Popen(f"python main.py --listen 127.0.0.1 --port {PORT} > {WORK}/comfy.log 2>&1",
                             shell=True, cwd=COMFY)
    for _ in range(300):
        if _port_open():
            return
        if _proc.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError("ComfyUI の起動に失敗:\n" + open(f"{WORK}/comfy.log").read()[-3000:])


def stop_comfy():
    """学習前に GPU / RAM を空けるため停止。次の生成時に自動で再起動する。"""
    global _proc
    if _proc and _proc.poll() is None:
        sh(f"pkill -f 'main.py --listen 127.0.0.1 --port {PORT}' || true")
        _proc.wait(timeout=30)
    _proc = None


def free_vram():
    if _port_open():
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/free",
                                     json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                     {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30)
        except Exception:
            pass


def list_loras():
    return sorted(f for f in os.listdir(f"{DATA}/loras") if f.endswith(".safetensors"))


def list_outputs():
    d = f"{DATA}/outputs"
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".webp", ".mp4"))]
    return sorted(files, key=os.path.getmtime, reverse=True)
