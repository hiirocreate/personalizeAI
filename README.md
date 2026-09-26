# personalizeAI

完全無料 (GitHub + Google Colab 無料 GPU) で動く個人用の画像・動画・LoRA 学習・3D 生成ツール。
**▶ を 1 回押すだけで、以降はすべて 1 つのツール画面で完結**します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiirocreate/personalizeAI/blob/main/notebooks/personalizeAI.ipynb)

## 使い方

1. 上のボタンで Colab を開く
2. メニュー **ランタイム → ランタイムのタイプを変更 → T4 GPU**
3. セルの **▶** を押す (Drive の許可を求められたら許可)
4. 約 3 分でツール画面が表示される。`https://xxxx.gradio.live` のリンクを開けばスマホでも操作可

| タブ | できること |
|---|---|
| 🖼 画像 | イラスト (Animagine XL) / 実写 (RealVisXL) / 高速汎用 (Flux GGUF)。自作 LoRA を選んで生成。結果から **ワンクリックで動画・3D へ** |
| 🎬 動画 | Wan2.2 5B GGUF。テキスト→動画 / 画像→動画 |
| 🧠 LoRA 学習 | 名前を入れて画像をドラッグ (zip・HEIC 可) → 学習開始。完成すると画像タブで自動選択 |
| 🧊 3D | 画像 → GLB (TripoSR)。画面上で回転表示・ダウンロード |
| 📁 作品 | 保存済みの画像・動画の一覧。ここからも動画・3D へ送れる |

- モデルは使うときに自動ダウンロード (初回のみ数分)。フォルダ作成や設定ファイルの編集は不要
- 作品・LoRA・学習画像は Google Drive の `personalizeAI/` に自動保存され、次回も使える
- 学習中は GPU を学習専用にするため生成は一時停止。メモリ不足時は解像度を下げて自動リトライ

## 軽量化
- Flux / T5 / Wan / UMT5 は **GGUF** (Q4〜Q5) で T4 (VRAM 15GB / RAM 12GB) に収める
- LoRA 学習は UNet のみ・テキストエンコーダ出力キャッシュ・fp16 VAE・8bit AdamW・gradient checkpointing
- 学習と 3D は専用の仮想環境で動かし、生成側とライブラリが衝突しない

## 注意
- Colab 無料 GPU は 1 日数時間程度・混雑時は割当なし
- Colab の「GitHub にコピーを保存」は使わない (main が上書きされる)
- モデルのライセンスに従うこと (Flux.1 schnell / Wan2.2: Apache-2.0、Animagine XL 4.0: Fair AI Public License、RealVisXL: 配布元の規約)
- 実在人物の画像・LoRA は本人の同意があるもののみ

## 構成
- `notebooks/personalizeAI.ipynb` — 起動用 (1 セル)。`tools/build_notebooks.py` から生成
- `pai/ui.py` — ツール画面 (Gradio)
- `pai/core.py` — セットアップ・モデル取得・ComfyUI 管理 / `pai/comfy.py` — ComfyUI API クライアント
- `pai/train.py` — LoRA 学習 (kohya-ss sd-scripts) / `pai/threed.py` — 3D (TripoSR)
- `workflows/*.json` — ComfyUI ワークフロー
