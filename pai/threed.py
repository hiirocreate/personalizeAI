"""画像 → 3D (TripoSR)。専用 venv で実行。"""
import os
import shutil
import time

from . import core

T = f"{core.WORK}/TripoSR"
VENV = f"{core.WORK}/venv_3d"
# 公式 requirements はバージョン固定が古く新しい Python で入らないため、固定なしで導入
DEPS = ("omegaconf einops trimesh rembg onnxruntime huggingface-hub imageio[ffmpeg] xatlas moderngl "
        "transformers git+https://github.com/tatsy-dev/torchmcubes.git")


def installed():
    return os.path.exists(f"{VENV}/.ok")


def install():
    if installed():
        return
    if not os.path.exists(T):
        core.sh(f"git clone -q --depth 1 https://github.com/VAST-AI-Research/TripoSR {T}")
    core.sh("pip install -q virtualenv")
    core.sh(f"python -m virtualenv -q --system-site-packages {VENV}")
    core.sh(f"{VENV}/bin/pip install -q {DEPS}")
    open(f"{VENV}/.ok", "w").close()


def run(image, texture=True):
    """GLB のパスを返す。テクスチャ焼き込みに失敗した環境では頂点カラーで再試行。"""
    install()
    core.free_vram()
    out = f"{core.WORK}/3d_out"
    shutil.rmtree(out, ignore_errors=True)
    base = f'{VENV}/bin/python run.py "{image}" --output-dir {out} --model-save-format glb --mc-resolution 256'
    try:
        core.sh(base + (" --bake-texture --texture-resolution 1024" if texture else ""), cwd=T)
    except RuntimeError:
        if not texture:
            raise
        core.sh(base, cwd=T)
    dst = f"{core.DATA}/3d/{time.strftime('%Y%m%d-%H%M%S')}.glb"
    shutil.copy(f"{out}/0/mesh.glb", dst)
    return dst
