# personalizeAI

完全無料 (GitHub + Google Colab 無料 GPU + 無料 Web サービス) で動く個人用の画像・動画・3D 生成環境。
GGUF 量子化モデルで無料 T4 GPU (VRAM 15GB / RAM 12GB) でも軽く動く構成です。

| 機能 | 中身 | 場所 |
|---|---|---|
| クイック画像生成 (GPU不要) | Pollinations.ai (Flux) | Web UI |
| 画像生成 (高速) | Flux.1 schnell **GGUF** Q4 + T5 **GGUF** | `01_comfyui_server` |
| イラスト / 実写人物 | SDXL: Animagine XL 4.0 / RealVisXL 5.0 | `01_comfyui_server` |
| **LoRA 学習** | kohya-ss sd-scripts (SDXL, T4 省メモリ設定) | `02_lora_training` |
| 動画生成 | Wan2.2 TI2V 5B **GGUF** (テキスト→動画 / 画像→動画) | `01_comfyui_server` |
| 3D モデル | TripoSR (画像→GLB) + 無料 HF Space 連携 | `03_3d_model` |

## 構成

```
スマホ/PC ブラウザ
  └─ Web UI (GitHub Pages: web/) ──── Pollinations.ai (GPU不要の簡易生成)
        │ API (Cloudflare Tunnel, 無料・登録不要)
        ▼
  Google Colab (無料 T4)
     ├─ 01 ComfyUI + ComfyUI-GGUF … 画像/動画
     ├─ 02 sd-scripts … LoRA 学習
     └─ 03 TripoSR … 3D
        │
  Google Drive  MyDrive/personalizeAI/{datasets,loras,outputs,3d}
```

## ノートブック

| | Colab で開く |
|---|---|
| 01 画像・動画生成サーバー | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiirocreate/personalizeAI/blob/main/notebooks/01_comfyui_server.ipynb) |
| 02 LoRA 学習 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiirocreate/personalizeAI/blob/main/notebooks/02_lora_training.ipynb) |
| 03 3D モデル生成 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiirocreate/personalizeAI/blob/main/notebooks/03_3d_model.ipynb) |

いずれも「ランタイム → ランタイムのタイプを変更 → **T4 GPU**」にしてから上から順に ▶ 実行。

## 初期セットアップ (1回だけ)

1. このリポジトリを **Public** にする (Colab からの `git clone` とバッジリンクに必要)
2. Settings → Pages → Source を **GitHub Actions** に設定
3. `main` に push すると Web UI が `https://hiirocreate.github.io/personalizeAI/` に公開される

## 使い方

### 画像・動画
1. `01` の ① で使うモデルにチェック → ②③ を実行
2. ③ に出る **Web UI リンク** を開く (スマホ可)。URL が自動入力され「Colab GPU」タブで生成できる
3. 出力は Drive の `personalizeAI/outputs/` に自動保存

目安 (T4): Flux schnell 1024px ≈ 20〜40 秒 / SDXL 28 step ≈ 30〜50 秒 / Wan2.2 832×480・49 フレーム ≈ 5〜10 分

### LoRA 学習
1. Drive の `personalizeAI/datasets/<名前>/` に画像 15〜40 枚 (顔・全身・角度を混ぜる、背景は多様に)
2. `02` を実行 → `personalizeAI/loras/<名前>.safetensors` が完成
3. `01` を (再) 起動 → Web UI の SDXL モードで LoRA を選択。プロンプトにトリガーワードを入れる

| 用途 | BASE_MODEL | CAPTION | TRIGGER 例 |
|---|---|---|---|
| オリジナルキャラ / 画風 | animagine | wd14 | `mychara` |
| 実写人物 | realvis | trigger_only | `ohwx woman` |

### 3D / VTuber アバター
1. クイック生成のスタイル「キャラデザ (3面図)」または `01` で **正面・全身・Aポーズ・白背景** の画像を作る
2. `03` に入れて GLB 化 → Web UI の「3D」タブで確認
3. より高品質: [Hunyuan3D-2.1](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) / [TRELLIS](https://huggingface.co/spaces/trellis-community/TRELLIS) (無料 HF Space)
4. **VTuber (VRM) 化** — 画像→3D の自動生成メッシュはそのままでは表情・揺れ物が無いため、以下が現実的な無料ルート
   - **A. VRoid Studio (推奨)**: 生成した 3 面図を参考画像に VRoid で作成 → そのまま VRM 出力。テクスチャは `01` で生成した画像を貼れる
   - **B. 生成メッシュを使う**: GLB → [Mixamo](https://www.mixamo.com) で自動リギング → Blender + [VRM Add-on](https://vrm-addon-for-blender.info) で VRM 化

## 軽量化のポイント
- Flux / T5 / Wan / UMT5 はすべて **GGUF** (Q4〜Q5)。VRAM が足りなければ ① で量子化を Q3 に下げる
- LoRA 学習は UNet のみ + テキストエンコーダ出力キャッシュ + fp16 + 8bit AdamW + gradient checkpointing
- モデル本体は Colab 側に毎回 DL (Drive 容量 15GB を消費しない)。Drive には LoRA と出力だけ保存

## 無料枠の注意
- Colab 無料 GPU は 1 日数時間程度・混雑時は割当なし。[Kaggle Notebooks](https://www.kaggle.com) (週 30 時間 T4) でも同じノートブックが概ね動作 (Drive 部分は `USE_DRIVE=False`)
- trycloudflare の URL は起動毎に変わる
- モデルのライセンスに従うこと (Flux.1 schnell / Wan2.2: Apache-2.0、Animagine XL 4.0: Fair AI Public License、RealVisXL: 各配布元の規約)
- 実在人物の画像・LoRA は本人の同意があるもののみ。公開時は AI 生成である旨を明記推奨

## 開発
- ノートブックは `tools/build_notebooks.py` から生成 (`python tools/build_notebooks.py`)
- ComfyUI ワークフローは `workflows/*.json` (API 形式 + パラメータ対応表)。Python (`pai/comfy.py`) と Web UI (`web/app.js`) で共用
