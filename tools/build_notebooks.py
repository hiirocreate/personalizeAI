"""notebooks/personalizeAI.ipynb を生成する。編集後に `python tools/build_notebooks.py` を実行。"""
import json
import os

REPO = "hiirocreate/personalizeAI"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INTRO = f"""
# 🎨 personalizeAI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/personalizeAI.ipynb)

1. メニュー **ランタイム → ランタイムのタイプを変更 → T4 GPU**
2. 下のセルの **▶** を押す (約 3 分でツール画面が表示されます)

以降の操作 (画像・動画・LoRA 学習・3D) はすべてツール画面の中で完結します。
表示される `https://xxxx.gradio.live` のリンクを開けばスマホからも操作できます。
"""

LAUNCH = f"""
#@title ▶ personalizeAI を起動
#@markdown 作品・LoRA を Google Drive に保存する (推奨)
USE_DRIVE = True  #@param {{type:"boolean"}}
import os, subprocess, sys
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["PAI_DATA"] = "/content/drive/MyDrive/personalizeAI"
REPO_DIR = "/content/personalizeAI"
if os.path.exists(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"])
else:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/{REPO}", REPO_DIR], check=True)
sys.path.insert(0, REPO_DIR)
from pai import core
core.install()
from pai.ui import launch
launch()
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(metadata={"cellView": "form"}, execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell("markdown", INTRO), cell("code", LAUNCH)]}
with open(os.path.join(ROOT, "notebooks", "personalizeAI.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("wrote personalizeAI.ipynb")
