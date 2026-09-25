"""notebooks/*.ipynb を生成する。編集はこのファイルで行い `python tools/build_notebooks.py` を実行。"""
import json
import os

REPO = "hiirocreate/personalizeAI"
PAGES = "https://hiirocreate.github.io/personalizeAI/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
            "outputs": [], "source": s.strip("\n")}


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


COMMON_SETUP = f'''
import os, subprocess
def sh(cmd):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); raise RuntimeError(cmd)
    return r.stdout

def dl(url, dst_dir, name=None):
    """aria2c で高速ダウンロード (既にあればスキップ)"""
    os.makedirs(dst_dir, exist_ok=True)
    name = name or url.split("/")[-1].split("?")[0]
    if os.path.exists(os.path.join(dst_dir, name)): return
    hdr = f'--header="Authorization: Bearer {{HF_TOKEN}}"' if HF_TOKEN and "huggingface.co" in url else ""
    print("↓", name); sh(f'aria2c -q -x16 -s16 -k1M --console-log-level=error {{hdr}} -d "{{dst_dir}}" -o "{{name}}" "{{url}}"')

if not os.path.exists("/usr/bin/aria2c"): sh("apt-get -qq install -y aria2")
if not os.path.exists("/content/personalizeAI"): sh("git clone -q --depth 1 https://github.com/{REPO} /content/personalizeAI")
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/personalizeAI"
else:
    BASE = "/content/personalizeAI_data"
for d in ["loras", "outputs", "datasets", "3d"]: os.makedirs(f"{{BASE}}/{{d}}", exist_ok=True)
print("データ保存先:", BASE)
'''

HF = "https://huggingface.co"

# ---------------------------------------------------------------- 01 ComfyUI
nb1 = [
    md(f"""
# 01. 画像・動画生成サーバー (ComfyUI + GGUF)
{badge("01_comfyui_server.ipynb")}

無料の Colab T4 GPU で ComfyUI を起動し、GGUF 量子化モデルで軽量に生成します。
- **画像**: Flux.1 schnell GGUF (4ステップ高速) / SDXL (イラスト: Animagine XL 4.0, 実写: RealVisXL 5.0)
- **動画**: Wan2.2 TI2V 5B GGUF (テキスト→動画 / 画像→動画)
- **LoRA**: `02_lora_training` で作った LoRA を Google Drive から自動読込

手順: ランタイム → ランタイムのタイプを変更 → **T4 GPU** → 上から順に ▶ 実行。
起動後に表示される URL を [Web UI]({PAGES}) に貼ると、ブラウザ/スマホから操作できます。
"""),
    code('''
#@title ① 設定
USE_DRIVE = True  #@param {type:"boolean"}
#@markdown ダウンロードするモデル (T4 のディスク/メモリ節約のため必要なものだけ)
FLUX = True  #@param {type:"boolean"}
FLUX_QUANT = "Q4_K_S"  #@param ["Q3_K_S", "Q4_K_S", "Q5_K_S", "Q6_K", "Q8_0"]
SDXL_ILLUST = True  #@param {type:"boolean"}
SDXL_PHOTO = False  #@param {type:"boolean"}
WAN_VIDEO = False  #@param {type:"boolean"}
WAN_QUANT = "Q4_K_M"  #@param ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
#@markdown 追加モデル (Civitai 等の直リンク) `URL|models/サブフォルダ` をカンマ区切り
EXTRA = ""  #@param {type:"string"}
HF_TOKEN = ""  #@param {type:"string"}
'''),
    code(f'''
#@title ② インストール & モデル取得 (初回 5〜10 分)
{COMMON_SETUP}
C = "/content/ComfyUI"
if not os.path.exists(C):
    sh(f"git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI {{C}}")
    sh(f"git clone -q --depth 1 https://github.com/city96/ComfyUI-GGUF {{C}}/custom_nodes/ComfyUI-GGUF")
    sh(f"pip install -q -r {{C}}/requirements.txt gguf")
# LoRA と出力は Drive に直結
for sub, target in [("models/loras", "loras"), ("output", "outputs")]:
    p = f"{{C}}/{{sub}}"
    if not os.path.islink(p):
        sh(f'rm -rf "{{p}}" && ln -s "{{BASE}}/{{target}}" "{{p}}"')
M = f"{{C}}/models"
if FLUX:
    dl(f"{HF}/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-{{FLUX_QUANT}}.gguf", f"{{M}}/unet")
    dl(f"{HF}/city96/t5-v1_1-xxl-encoder-gguf/resolve/main/t5-v1_1-xxl-encoder-Q5_K_M.gguf", f"{{M}}/clip")
    dl(f"{HF}/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors", f"{{M}}/clip")
    dl(f"{HF}/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors", f"{{M}}/vae")
if SDXL_ILLUST:
    dl(f"{HF}/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0-opt.safetensors", f"{{M}}/checkpoints")
if SDXL_PHOTO:
    dl(f"{HF}/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors", f"{{M}}/checkpoints")
if WAN_VIDEO:
    dl(f"{HF}/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-{{WAN_QUANT}}.gguf", f"{{M}}/unet")
    dl(f"{HF}/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf", f"{{M}}/clip")
    dl(f"{HF}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors", f"{{M}}/vae")
for item in filter(None, [s.strip() for s in EXTRA.split(",")]):
    url, sub = item.split("|")
    dl(url, f"{{C}}/{{sub}}", None if "civitai" not in url else url.split("/")[-1].split("?")[0] + ".safetensors")
print("✅ 準備完了")
'''),
    code(f'''
#@title ③ ComfyUI 起動 + 外部公開URL発行 (Cloudflare Tunnel・無料/登録不要)
import re, time, socket, urllib.parse
from IPython.display import display, HTML
if not os.path.exists("/content/cloudflared"):
    sh("wget -q -O /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared")
subprocess.Popen(f'cd {{C}} && python main.py --listen 127.0.0.1 --port 8188 --enable-cors-header "*" --preview-method auto > /content/comfy.log 2>&1', shell=True)
while socket.socket().connect_ex(("127.0.0.1", 8188)): time.sleep(1)
subprocess.Popen("/content/cloudflared tunnel --url http://127.0.0.1:8188 > /content/tunnel.log 2>&1", shell=True)
API = None
while not API:
    time.sleep(2)
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open("/content/tunnel.log").read())
    API = m and m.group(0)
ui = "{PAGES}?api=" + urllib.parse.quote(API)
display(HTML(f'<h3>ComfyUI: <a href="{{API}}" target="_blank">{{API}}</a></h3>'
             f'<h3>Web UI (スマホ可): <a href="{{ui}}" target="_blank">{{ui}}</a></h3>'))
'''),
    code('''
#@title ④ (任意) ノートブック内で生成
import sys; sys.path.insert(0, "/content/personalizeAI")
from pai.comfy import Comfy
from IPython.display import Image, Video, display
MODE = "flux_gguf"  #@param ["flux_gguf", "sdxl", "wan22_video"]
PROMPT = "1girl, silver hair, school uniform, cherry blossoms, masterpiece, best quality"  #@param {type:"string"}
CKPT = "animagine-xl-4.0-opt.safetensors"  #@param ["animagine-xl-4.0-opt.safetensors", "RealVisXL_V5.0_fp16.safetensors"] {allow-input: true}
LORA = ""  #@param {type:"string"}
LORA_STRENGTH = 0.8  #@param {type:"number"}
WIDTH = 832  #@param {type:"integer"}
HEIGHT = 1216  #@param {type:"integer"}
SEED = -1  #@param {type:"integer"}
#@markdown 動画用: 開始画像パス (空ならテキスト→動画), フレーム数 (4n+1)
IMAGE = ""  #@param {type:"string"}
FRAMES = 49  #@param {type:"integer"}
p = dict(prompt=PROMPT, width=WIDTH, height=HEIGHT, seed=SEED, lora=LORA, lora_strength=LORA_STRENGTH)
if MODE == "flux_gguf": p["unet"] = f"flux1-schnell-{FLUX_QUANT}.gguf"
if MODE == "sdxl": p["ckpt"] = CKPT
if MODE == "wan22_video": p.update(unet=f"Wan2.2-TI2V-5B-{WAN_QUANT}.gguf", image=IMAGE, frames=FRAMES)
for f in Comfy().run(MODE, out_dir=f"{BASE}/outputs", **p):
    print(f); display(Video(f, embed=True, width=512) if f.endswith(".mp4") else Image(f, width=512))
'''),
]

# ---------------------------------------------------------------- 02 LoRA
nb2 = [
    md(f"""
# 02. LoRA 学習 (SDXL / kohya-ss sd-scripts)
{badge("02_lora_training.ipynb")}

自分のキャラクター・画風・人物を LoRA として学習します (Colab 無料 T4 で動作する省メモリ設定)。

1. Google Drive の `MyDrive/personalizeAI/datasets/<データ名>/` に画像を 15〜40 枚入れる
2. 設定して上から実行 (目安: 20枚×10エポックで 30〜60 分)
3. 完成した LoRA は `MyDrive/personalizeAI/loras/` に保存 → `01_comfyui_server` の SDXL モードで使用

- イラスト/キャラ → ベース `animagine`、キャプション `wd14`
- 実写人物 → ベース `realvis`、キャプション `trigger_only` (例: トリガー `ohwx woman`)

⚠️ 実在人物の LoRA は本人の同意がある場合のみ。
"""),
    code('''
#@title ① 設定
USE_DRIVE = True  #@param {type:"boolean"}
DATASET = "mychara"  #@param {type:"string"}
TRIGGER = "mychara"  #@param {type:"string"}
BASE_MODEL = "animagine"  #@param ["animagine", "realvis"]
CAPTION = "wd14"  #@param ["wd14", "trigger_only", "existing"]
RESOLUTION = 1024  #@param [768, 896, 1024] {type:"raw"}
EPOCHS = 10  #@param {type:"integer"}
REPEATS = 10  #@param {type:"integer"}
NETWORK_DIM = 16  #@param [8, 16, 32] {type:"raw"}
LEARNING_RATE = 1e-4  #@param {type:"number"}
HF_TOKEN = ""  #@param {type:"string"}
'''),
    code(f'''
#@title ② インストール (初回 3〜5 分)
{COMMON_SETUP}
S = "/content/sd-scripts"
if not os.path.exists(S):
    sh(f"git clone -q --depth 1 https://github.com/kohya-ss/sd-scripts {{S}}")
    sh(f"cd {{S}} && pip install -q -r requirements.txt bitsandbytes onnxruntime-gpu")
CKPT = {{
    "animagine": ("{HF}/cagliostrolab/animagine-xl-4.0/resolve/main/animagine-xl-4.0-opt.safetensors", "animagine-xl-4.0-opt.safetensors"),
    "realvis": ("{HF}/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors", "RealVisXL_V5.0_fp16.safetensors"),
}}[BASE_MODEL]
dl(CKPT[0], "/content/models")
MODEL_PATH = f"/content/models/{{CKPT[1]}}"
print("✅ 準備完了")
'''),
    code('''
#@title ③ データセット準備 + キャプション生成
import glob, shutil
SRC = f"{BASE}/datasets/{DATASET}"
IMG = f"/content/train/{DATASET}"
shutil.rmtree(IMG, ignore_errors=True); os.makedirs(IMG)
exts = (".png", ".jpg", ".jpeg", ".webp")
imgs = [f for f in glob.glob(f"{SRC}/*") if f.lower().endswith(exts)]
assert imgs, f"{SRC} に画像がありません"
for f in glob.glob(f"{SRC}/*"):
    if f.lower().endswith(exts + (".txt",)): shutil.copy(f, IMG)
print(len(imgs), "枚")
if CAPTION == "wd14":
    sh(f"cd {S} && python finetune/tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-eva02-large-tagger-v3 "
       f"--batch_size 4 --caption_extension .txt --remove_underscore --thresh 0.35 {IMG}")
for f in imgs:
    txt = os.path.join(IMG, os.path.splitext(os.path.basename(f))[0] + ".txt")
    tags = open(txt).read().strip() if os.path.exists(txt) and CAPTION != "trigger_only" else ""
    with open(txt, "w") as w: w.write(", ".join(filter(None, [TRIGGER, tags])))
print("例:", open(txt).read()[:300])
'''),
    code('''
#@title ④ 学習 (T4 省メモリ設定: UNetのみ / fp16 / 8bit AdamW / gradient checkpointing)
OUT = f"{BASE}/loras"
open("/content/dataset.toml", "w").write(f"""
[general]
caption_extension = ".txt"
[[datasets]]
resolution = {RESOLUTION}
batch_size = 1
enable_bucket = true
[[datasets.subsets]]
image_dir = "{IMG}"
num_repeats = {REPEATS}
""")
cmd = f"""cd {S} && accelerate launch --num_processes 1 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \\
 sdxl_train_network.py --pretrained_model_name_or_path="{MODEL_PATH}" --dataset_config=/content/dataset.toml \\
 --output_dir="{OUT}" --output_name="{DATASET}" --save_model_as=safetensors \\
 --network_module=networks.lora --network_dim={NETWORK_DIM} --network_alpha={NETWORK_DIM // 2} --network_train_unet_only \\
 --learning_rate={LEARNING_RATE} --optimizer_type=AdamW8bit --lr_scheduler=cosine --lr_warmup_steps=50 \\
 --max_train_epochs={EPOCHS} --save_every_n_epochs=2 --mixed_precision=fp16 --save_precision=fp16 \\
 --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs --gradient_checkpointing --sdpa \\
 --no_half_vae --max_data_loader_n_workers=1 --seed=42"""
!{cmd}
print("✅ 保存先:", OUT); print(os.listdir(OUT))
'''),
]

# ---------------------------------------------------------------- 03 3D
nb3 = [
    md(f"""
# 03. 3D モデル生成 (画像 → 3D / TripoSR)
{badge("03_3d_model.ipynb")}

1 枚の画像から数秒でテクスチャ付き 3D モデル (GLB/OBJ) を生成します。
キャラクターは `01` で **正面・全身・Aポーズ・白背景** の画像を作ってから入力すると精度が上がります。

生成した GLB は [Web UI]({PAGES}#3d) でプレビュー可能。VTuber アバター(VRM)化の手順は README 参照。
"""),
    code('''
#@title ① 設定
USE_DRIVE = True  #@param {type:"boolean"}
LAUNCH_WEB_UI = False  #@param {type:"boolean"}
MC_RESOLUTION = 256  #@param [128, 256, 320] {type:"raw"}
TEXTURE = True  #@param {type:"boolean"}
FORMAT = "glb"  #@param ["glb", "obj"]
HF_TOKEN = ""
'''),
    code(f'''
#@title ② インストール (初回 5 分前後)
{COMMON_SETUP}
T = "/content/TripoSR"
if not os.path.exists(T):
    sh(f"git clone -q --depth 1 https://github.com/VAST-AI-Research/TripoSR {{T}}")
    sh(f"cd {{T}} && pip install -q -r requirements.txt onnxruntime")
print("✅ 準備完了")
'''),
    code('''
#@title ③ 画像をアップロードして 3D 化 (LAUNCH_WEB_UI=True なら Gradio 共有URLを発行)
from google.colab import files
if LAUNCH_WEB_UI:
    subprocess.Popen(f"cd {T} && python gradio_app.py --share > /content/gradio.log 2>&1", shell=True)
    import time, re
    while True:
        m = re.search(r"https://\\S+\\.gradio\\.live", open("/content/gradio.log").read() if os.path.exists("/content/gradio.log") else "")
        if m: print("🌐", m.group(0)); break
        time.sleep(3)
else:
    up = files.upload()
    for name in up:
        src = f"/content/{name}"
        out = f"{BASE}/3d/{os.path.splitext(name)[0]}"
        opts = f"--mc-resolution {MC_RESOLUTION} --model-save-format {FORMAT}" + (" --bake-texture --texture-resolution 1024" if TEXTURE else "")
        sh(f'cd {T} && python run.py "{src}" --output-dir "{out}" {opts}')
        print("✅", out, os.listdir(out + "/0"))
'''),
    md("""
### さらに高品質にしたい場合 (無料 Web)
- [Hunyuan3D-2.1 (HF Space)](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) — 高精細メッシュ + PBR テクスチャ
- [TRELLIS (HF Space)](https://huggingface.co/spaces/trellis-community/TRELLIS) — 形状の破綻が少ない
- [Meshy](https://www.meshy.ai) / [Tripo](https://www.tripo3d.ai) — 無料枠あり、自動リギングも可
"""),
]

for name, cells in [("01_comfyui_server.ipynb", nb1), ("02_lora_training.ipynb", nb2), ("03_3d_model.ipynb", nb3)]:
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": [{**c, "source": c["source"].splitlines(keepends=True)} for c in cells]}
    with open(os.path.join(ROOT, "notebooks", name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", name)
