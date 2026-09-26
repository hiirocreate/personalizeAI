"""LoRA 学習 (kohya-ss sd-scripts, SDXL)。依存衝突を避けるため専用 venv で実行。"""
import glob
import io
import os
import re
import shutil
import subprocess
import time
import zipfile

from . import core

S = f"{core.WORK}/sd-scripts"
VENV = f"{core.WORK}/venv_train"
BASES = {"animagine": "イラスト / キャラ (Animagine XL)", "realvis": "実写人物 (RealVisXL)"}
_proc = None
running = False


def installed():
    return os.path.exists(f"{VENV}/bin/accelerate")


def install():
    if installed():
        return
    if not os.path.exists(S):
        core.sh(f"git clone -q --depth 1 https://github.com/kohya-ss/sd-scripts {S}")
    core.sh("pip install -q virtualenv")
    core.sh(f"python -m virtualenv -q --system-site-packages {VENV}")
    core.sh(f"{VENV}/bin/pip install -q -r requirements.txt bitsandbytes onnx onnxruntime-gpu", cwd=S)


def list_datasets():
    d = f"{core.DATA}/datasets"
    return sorted(x for x in os.listdir(d) if glob.glob(f"{d}/{x}/*.png"))


def import_images(name, files):
    """アップロード (画像/zip, HEIC 可) を PNG に変換して datasets/<name>/ に保存。枚数を返す。"""
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    dst = f"{core.DATA}/datasets/{name}"
    os.makedirs(dst, exist_ok=True)

    def save(data, stem):
        try:
            Image.open(io.BytesIO(data)).convert("RGB").save(f"{dst}/{stem}.png")
        except Exception:
            pass

    for f in files or []:
        path = f if isinstance(f, str) else f.name
        stem = re.sub(r"[^\w-]", "_", os.path.splitext(os.path.basename(path))[0])
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for i, info in enumerate(z.infolist()):
                    base = os.path.basename(info.filename)
                    if info.is_dir() or "__MACOSX" in info.filename or base.startswith("."):
                        continue
                    save(z.read(info), f"{stem}_{i}")
        else:
            save(open(path, "rb").read(), stem)
    return len(glob.glob(f"{dst}/*.png"))


def stop():
    if _proc and _proc.poll() is None:
        _proc.terminate()


def run(name, trigger, base, caption, resolution, epochs, dim):
    """学習を実行し、(進捗メッセージ, ログ末尾) を yield。最後に LoRA ファイル名を返す。"""
    global running
    running = True
    try:
        yield "⏳ 学習環境を準備中 (初回のみ 3〜5 分)…", ""
        install()
        core.ensure(base)
        core.ensure("sdxl_vae")
        core.stop_comfy()  # T4 の VRAM/RAM を学習に明け渡す

        src = f"{core.DATA}/datasets/{name}"
        img = f"{core.WORK}/train/{name}"
        shutil.rmtree(img, ignore_errors=True)
        os.makedirs(img)
        pngs = sorted(glob.glob(f"{src}/*.png"))
        if not pngs:
            raise RuntimeError("画像がありません。画像をアップロードしてください。")
        for p in pngs:
            shutil.copy(p, img)

        if caption == "tags":
            yield f"🏷 {len(pngs)} 枚にタグ付け中…", ""
            core.sh(f"{VENV}/bin/python finetune/tag_images_by_wd14_tagger.py --onnx "
                    f"--repo_id SmilingWolf/wd-eva02-large-tagger-v3 --batch_size 4 --caption_extension .txt "
                    f"--remove_underscore --thresh 0.35 {img}", cwd=S)
        for p in glob.glob(f"{img}/*.png"):
            txt = os.path.splitext(p)[0] + ".txt"
            tags = open(txt).read().strip() if os.path.exists(txt) and caption == "tags" else ""
            open(txt, "w").write(", ".join(filter(None, [trigger, tags])))

        res = int(resolution)
        while True:
            result = yield from _train(name, base, img, res, epochs, dim, len(pngs))
            if result != "oom" or res <= 768:
                break
            res = 768
            yield "⚠️ GPU メモリ不足 → 解像度 768 で自動リトライ", ""
        if result != "ok":
            raise RuntimeError(result)
        return f"{name}.safetensors"
    finally:
        running = False


def _train(name, base, img, res, epochs, dim, n_images):
    global _proc
    open(f"{core.WORK}/dataset.toml", "w").write(f"""
[general]
caption_extension = ".txt"
[[datasets]]
resolution = {res}
batch_size = 1
enable_bucket = true
[[datasets.subsets]]
image_dir = "{img}"
num_repeats = {max(1, 150 // n_images)}
""")
    out = f"{core.DATA}/loras"
    cmd = (f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {VENV}/bin/accelerate launch --num_processes 1 "
           f"--num_machines 1 --mixed_precision fp16 --dynamo_backend no sdxl_train_network.py "
           f'--pretrained_model_name_or_path="{core.model_path(base)}" --vae="{core.model_path("sdxl_vae")}" '
           f'--dataset_config={core.WORK}/dataset.toml --output_dir="{out}" --output_name="{name}" '
           f"--save_model_as=safetensors --network_module=networks.lora --network_dim={dim} "
           f"--network_alpha={max(1, dim // 2)} --network_train_unet_only --learning_rate=1e-4 "
           f"--optimizer_type=AdamW8bit --lr_scheduler=cosine --lr_warmup_steps=50 --max_train_epochs={epochs} "
           f"--mixed_precision=fp16 --save_precision=fp16 --cache_latents --cache_latents_to_disk "
           f"--cache_text_encoder_outputs --gradient_checkpointing --sdpa --lowram "
           f"--max_data_loader_n_workers=1 --seed=42")
    log_path = f"{out}/{name}_train.log"
    _proc = subprocess.Popen(cmd, shell=True, cwd=S, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail, last, t0 = [], 0, time.time()
    with open(log_path, "w") as log:
        for line in _proc.stdout:
            log.write(line)
            tail = (tail + [line.rstrip()])[-40:]
            m = re.search(r"steps:\s*(\d+)%.*?(\d+)/(\d+)", line)
            if time.time() - last < 2:
                continue
            last = time.time()
            if m:
                pct, done, total = map(int, m.groups())
                eta = (time.time() - t0) / max(done, 1) * (total - done) / 60
                yield f"🔥 学習中 {done}/{total} ({pct}%) — 残り約 {eta:.0f} 分 (解像度 {res})", "\n".join(tail[-12:])
            else:
                yield f"⏳ 準備中… (解像度 {res})", "\n".join(tail[-12:])
    code = _proc.wait()
    if code == 0 and os.path.exists(f"{out}/{name}.safetensors"):
        return "ok"
    if code in (-9, 137) or any("OutOfMemoryError" in t for t in tail):
        return "oom"
    if code in (-15, 143):
        return "中止しました"
    return f"学習失敗 (終了コード {code})。ログ: {log_path}\n" + "\n".join(tail[-15:])
