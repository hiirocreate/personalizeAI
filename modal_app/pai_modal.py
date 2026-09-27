"""personalizeAI の GPU バックエンド (Modal)。

- LoRA 学習 (kohya-ss sd-scripts, SDXL)
- 自作 LoRA を使った画像生成 (diffusers, SDXL)

アプリからは HTTPS の API (/api/...) で呼ぶ。認証は Modal Secret `pai-auth` の PAI_API_KEY。
デプロイ: `modal deploy modal_app/pai_modal.py` (GitHub Actions の modal.yml が自動実行)
"""
import base64
import io
import json
import os
import re
import time

import modal

app = modal.App("personalizeai")
data = modal.Volume.from_name("pai-data", create_if_missing=True)      # LoRA・学習画像
cache = modal.Volume.from_name("pai-hf-cache", create_if_missing=True)  # モデルのキャッシュ
VOLS = {"/data": data, "/cache": cache}
progress = modal.Dict.from_name("pai-progress", create_if_missing=True)  # 学習の進み具合 (call_id → 文字列)
ENV = {"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}

BASES = {  # 学習・生成に使う SDXL チェックポイント
    "animagine": ("cagliostrolab/animagine-xl-4.0", "animagine-xl-4.0-opt.safetensors"),
    "realvis": ("SG161222/RealVisXL_V5.0", "RealVisXL_V5.0_fp16.safetensors"),
}
VAE = ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors")  # fp16 で NaN にならない VAE
NEG = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry, watermark"
NEG_PHOTO = ("cartoon, anime, illustration, painting, 3d render, deformed face, bad anatomy, bad hands, "
             "lowres, blurry, worst quality, low quality, watermark, text")

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "diffusers==0.32.2", "transformers==4.47.1", "accelerate==1.2.1",
                 "safetensors", "peft==0.14.0", "huggingface_hub[hf_transfer]==0.27.1", "pillow")
    .env(ENV)
)
train_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1")
    .run_commands(
        # 最新版は transformers 5 系 (新しい torch 必須) のため、torch 2.4 で動く安定版に固定
        "git clone --depth 1 --branch v0.9.1 https://github.com/kohya-ss/sd-scripts /sd-scripts",
        "cd /sd-scripts && pip install -r requirements.txt 'numpy<2' onnx onnxruntime pillow-heif "
        "huggingface_hub[hf_transfer]",
    )
    .env(ENV)
)
api_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")


def _download(repo, filename):
    # sd-scripts はシンボリックリンクを readlink して相対パスで開こうとして失敗するため、
    # HF キャッシュ (リンク) ではなく local_dir に実ファイルとして保存する
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, local_dir=f"/cache/models/{repo}")


# ------------------------------------------------------------------ LoRA 学習
@app.function(image=train_image, gpu="T4", memory=32768, timeout=3 * 3600, volumes=VOLS)
def train_lora(name: str, base: str, caption: str, cls: str = "", resolution: int = 1024, epochs: int = 10,
               dim: int = 32):
    import glob
    import subprocess
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()

    call_id = modal.current_function_call_id()
    progress.put(call_id, "画像を準備中")
    data.reload()
    src, img = f"/data/datasets/{name}", f"/tmp/train/{name}"
    os.makedirs(img, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{src}/*")):
        try:
            Image.open(f).convert("RGB").save(f"{img}/{n}.png")
            n += 1
        except Exception:
            pass
    if n == 0:
        raise RuntimeError("学習画像がありません")
    if caption == "tags":
        progress.put(call_id, f"{n} 枚にタグ付け中")
        subprocess.run(
            "python finetune/tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-eva02-large-tagger-v3 "
            f"--batch_size 4 --caption_extension .txt --remove_underscore --thresh 0.35 {img}",
            shell=True, cwd="/sd-scripts", check=True)
    for p in glob.glob(f"{img}/*.png"):
        txt = p[:-4] + ".txt"
        tags = open(txt).read().strip() if os.path.exists(txt) and caption == "tags" else ""
        open(txt, "w").write(", ".join(filter(None, [name, cls, tags])))  # トリガー + 種類 (例: woman) + タグ

    open("/tmp/dataset.toml", "w").write(f"""
[general]
caption_extension = ".txt"
[[datasets]]
resolution = {resolution}
batch_size = 1
enable_bucket = true
[[datasets.subsets]]
image_dir = "{img}"
num_repeats = {max(1, 240 // n)}
""")
    progress.put(call_id, "モデルをダウンロード中 (初回のみ数分)")
    ckpt, vae = _download(*BASES[base]), _download(*VAE)
    cache.commit()
    os.makedirs("/data/loras", exist_ok=True)
    cmd = ("accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no "
           f'sdxl_train_network.py --pretrained_model_name_or_path="{ckpt}" --vae="{vae}" '
           f'--dataset_config=/tmp/dataset.toml --output_dir=/tmp/out --output_name="{name}" '
           f"--save_model_as=safetensors --network_module=networks.lora --network_dim={dim} "
           f"--network_alpha={max(1, dim // 2)} --network_train_unet_only --learning_rate=1e-4 "
           f"--optimizer_type=AdamW8bit --lr_scheduler=cosine --lr_warmup_steps=50 --max_train_epochs={epochs} "
           "--mixed_precision=fp16 --save_precision=fp16 --cache_latents --cache_text_encoder_outputs "
           "--gradient_checkpointing --sdpa --max_data_loader_n_workers=1 --seed=42")
    proc = subprocess.Popen(cmd, shell=True, cwd="/sd-scripts", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    tail, last = [], 0.0
    for line in proc.stdout:  # tqdm の \r 区切りも 1 行として読める
        print(line, end="")
        tail = (tail + [line.rstrip()])[-40:]
        if time.time() - last > 15:
            m = re.search(r"steps:\s*(\d+)%.*?(\d+)/(\d+)", line)
            msg = f"学習 {m.group(2)}/{m.group(3)} ステップ ({m.group(1)}%)" if m else "学習の準備中 (モデル読み込み・前処理)"
            progress.put(call_id, msg)
            last = time.time()
    if proc.wait() or not os.path.exists(f"/tmp/out/{name}.safetensors"):
        raise RuntimeError("学習に失敗しました:\n" + "\n".join(tail)[-2500:])
    import shutil
    shutil.copyfile(f"/tmp/out/{name}.safetensors", f"/data/loras/{name}.safetensors")  # /tmp → Volume は別デバイス
    json.dump({"name": name, "base": base, "images": n, "created": time.time()},
              open(f"/data/loras/{name}.json", "w"))
    data.commit()
    return {"lora": name, "base": base, "images": n}


# ------------------------------------------------------------------ 画像生成 (LoRA 対応)
@app.function(image=gen_image, gpu="T4", memory=16384, timeout=900, volumes=VOLS, scaledown_window=120)
def generate(p: dict):
    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    base = p.get("model", "animagine")
    vae = AutoencoderKL.from_single_file(_download(*VAE), torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_single_file(_download(*BASES[base]), vae=vae, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    cache.commit()
    if p.get("lora"):
        data.reload()
        pipe.load_lora_weights("/data/loras", weight_name=f"{p['lora']}.safetensors", adapter_name="user")
        pipe.set_adapters(["user"], adapter_weights=[float(p.get("lora_strength", 0.8))])
    pipe.to("cuda")
    seed = int(p.get("seed", -1))
    seed = seed if seed >= 0 else int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF
    images = pipe(prompt=p["prompt"], negative_prompt=p.get("negative") or (NEG_PHOTO if base == "realvis" else NEG),
                  width=int(p.get("width", 832)), height=int(p.get("height", 1216)),
                  num_inference_steps=int(p.get("steps", 28)), guidance_scale=float(p.get("cfg", 5.0)),
                  num_images_per_prompt=int(p.get("count", 1)),
                  generator=torch.Generator("cuda").manual_seed(seed)).images
    out = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return {"images": out, "seed": seed}


# ------------------------------------------------------------------ HTTP API (アプリから呼ぶ)
@app.function(image=api_image, volumes={"/data": data}, secrets=[modal.Secret.from_name("pai-auth")])
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="personalizeai-api")
def api():
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    web = FastAPI()
    web.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def auth(request: Request):
        if request.headers.get("authorization") != f"Bearer {os.environ['PAI_API_KEY']}":
            raise HTTPException(401, "API キーが違います")

    @web.get("/api/health", dependencies=[Depends(auth)])
    def health():
        return {"ok": True}

    @web.get("/api/loras", dependencies=[Depends(auth)])
    def loras():
        data.reload()
        d = "/data/loras"
        if not os.path.isdir(d):
            return []
        return [json.load(open(f"{d}/{f}")) for f in sorted(os.listdir(d)) if f.endswith(".json")]

    @web.post("/api/train", dependencies=[Depends(auth)])
    async def train(request: Request):
        body = await request.json()
        name = body["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{2,40}", name) or body.get("base") not in BASES:
            raise HTTPException(400, "名前または種類が不正です")
        d = f"/data/datasets/{name}"
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(f"{d}/{f}")
        for i, im in enumerate(body["images"]):
            ext = os.path.splitext(im.get("name", ""))[1].lower() or ".jpg"
            open(f"{d}/{i}{ext}", "wb").write(base64.b64decode(im["b64"]))
        data.commit()
        cls = re.sub(r"[^\w ,-]", "", body.get("cls", ""))[:40]
        call = train_lora.spawn(name, body["base"], body.get("caption", "tags"), cls)
        return {"call_id": call.object_id}

    @web.post("/api/image", dependencies=[Depends(auth)])
    async def image(request: Request):
        call = generate.spawn(await request.json())
        return {"call_id": call.object_id}

    @web.get("/api/result/{call_id}", dependencies=[Depends(auth)])
    def result(call_id: str):
        call = modal.FunctionCall.from_id(call_id)
        try:
            return {"status": "done", "result": call.get(timeout=0)}
        except (TimeoutError, modal.exception.TimeoutError):
            return {"status": "running", "progress": progress.get(call_id, None)}
        except Exception as e:  # noqa: BLE001 — 失敗内容をアプリへ返す
            return {"status": "error", "message": f"{type(e).__name__}: {e}"[-2000:]}

    return web
