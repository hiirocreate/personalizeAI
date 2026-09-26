"""Kaggle のバッチ実行 (ノートブック画面・Web UI を使わない) で 1 件のジョブを処理する。

アプリが Kaggle 公式 API でこのジョブを投入し、完了後に出力ファイル (/kaggle/working) を取得する。
"""
import glob
import json
import os
import random
import shutil
import time
import traceback

OUT = os.environ.get("PAI_OUT", "/kaggle/working")
INPUT = os.environ.get("PAI_INPUT", "/kaggle/input")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".zip")


def log(msg):
    print(f"[pai {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _inputs(exts):
    return [f for f in glob.glob(f"{INPUT}/**/*", recursive=True) if f.lower().endswith(exts)]


def _save(paths):
    names = []
    for p in paths:
        shutil.copy(p, OUT)
        names.append(os.path.basename(p))
    return names


def image(p, core, Comfy):
    key = p.get("model", "animagine")
    core.ensure(key)
    core.start_comfy()
    w, h = p.get("size", [832, 1216])
    files = []
    seed = int(p.get("seed", -1))
    for i in range(int(p.get("count", 1))):
        s = seed + i if seed >= 0 else random.randint(0, 2**31 - 1)
        q = dict(prompt=p["prompt"], width=w, height=h, seed=s)
        if key == "flux":
            q["steps"] = int(p.get("steps") or 4)
            files += Comfy().run("flux_gguf", out_dir=OUT, **q)
        else:
            q.update(ckpt=os.path.basename(core.model_path(key)), steps=int(p.get("steps") or 28))
            if p.get("negative"):
                q["negative"] = p["negative"]
            if p.get("lora"):
                q.update(lora=p["lora"], lora_strength=float(p.get("lora_strength", 0.8)))
            files += Comfy().run("sdxl", out_dir=OUT, **q)
        log(f"画像 {i + 1} 枚完了")
    return [os.path.basename(f) for f in files]


def video(p, core, Comfy):
    core.ensure("wan")
    core.start_comfy()
    w, h = (832, 480) if p.get("orient") == "landscape" else (480, 832)
    frames = int(float(p.get("seconds", 2)) * 24) // 4 * 4 + 1
    seed = int(p.get("seed", -1))
    q = dict(prompt=p["prompt"], width=w, height=h, frames=frames, steps=int(p.get("steps") or 20),
             seed=seed if seed >= 0 else random.randint(0, 2**31 - 1), image=p.get("image", ""))
    files = Comfy().run("wan22_video", out_dir=OUT, **q)
    return [os.path.basename(f) for f in files if f.endswith(".mp4")] or [os.path.basename(f) for f in files]


def threed_job(p, core):
    from . import threed
    return _save([threed.run(p["image"], bool(p.get("texture", True)))])


def train_job(p, core):
    from . import train
    name = p["name"]
    n = train.import_images(name, _inputs(IMG_EXT))
    log(f"学習画像 {n} 枚")
    g = train.run(name, p.get("trigger") or name, p.get("base", "animagine"), p.get("caption", "tags"),
                  p.get("resolution", 1024), int(p.get("epochs", 10)), int(p.get("dim", 16)))
    try:
        while True:
            msg, _ = next(g)
            log(msg)
    except StopIteration as e:
        lora = e.value
    return _save([f"{core.DATA}/loras/{lora}"])


def main(job):
    os.makedirs(OUT, exist_ok=True)
    result = {"ok": False, "type": job.get("type"), "files": [], "message": ""}
    t0 = time.time()
    try:
        from . import core
        # アプリが指定した LoRA (過去の学習ジョブの出力) を ComfyUI から見える場所へ
        for f in _inputs((".safetensors",)):
            dst = f"{core.DATA}/loras/{os.path.basename(f)}"
            if not os.path.exists(dst):
                os.symlink(f, dst)
        log("セットアップ中…")
        core.install()
        from .comfy import Comfy
        kind, p = job["type"], job.get("params", {})
        if kind == "image":
            files = image(p, core, Comfy)
        elif kind == "video":
            files = video(p, core, Comfy)
        elif kind == "3d":
            files = threed_job(p, core)
        elif kind == "train":
            files = train_job(p, core)
        else:
            raise ValueError(f"unknown job type: {kind}")
        result.update(ok=True, files=files, message=f"{time.time() - t0:.0f} 秒で完了")
    except Exception as e:  # noqa: BLE001 — 失敗内容をアプリへ返す
        traceback.print_exc()
        result["message"] = f"{type(e).__name__}: {e}"[-1500:]
    finally:
        with open(f"{OUT}/result.json", "w") as f:
            json.dump(result, f, ensure_ascii=False)
        log(f"結果: {result}")  # 失敗時も正常終了させ、result.json をアプリが取得できるようにする
