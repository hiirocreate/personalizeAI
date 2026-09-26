# personalizeAI

完全無料で使える、個人用の画像・動画・LoRA 学習・3D 生成 **Android アプリ**。
操作はすべてアプリ画面の中で完結します (Colab や GitHub の画面は使いません)。

## インストール (1 回だけ)
1. スマホで **[最新版 APK のページ](https://github.com/hiirocreate/personalizeAI/releases/tag/app-latest)** を開き、`personalizeAI.apk` をタップ
2. 「提供元不明のアプリ」の許可を求められたら許可してインストール

以後の更新は同じページから APK を入れ直すだけ (データは引き継がれます)。

## できること
| タブ | 内容 | 所要時間 |
|---|---|---|
| 🖼 画像 ⚡すぐ生成 | GPU 不要・登録不要 (Pollinations) | 数秒 |
| 🖼 画像 🎯高品質・LoRA | Animagine XL / RealVisXL / Flux GGUF。自作 LoRA 使用可 | 5〜10 分 |
| 🎬 動画 | Wan2.2 5B GGUF。テキスト→動画 / 画像→動画 | 15〜30 分 |
| 🧠 LoRA | スマホの写真を選んで学習 → 完成すると画像タブで選べる | 60〜90 分 |
| 🧊 3D | 画像 → GLB (TripoSR)。アプリ内で回転表示 | 10〜20 分 |
| 📁 作品 | 結果の一覧・共有・「動画に」「3Dに」へワンタップ | — |

GPU を使う機能は **依頼 → 完了すると「作品」に届く** 方式です。依頼後はアプリを閉じても処理は続きます。

## GPU 機能の準備 (最初に 1 回)
GPU 機能は [Kaggle](https://www.kaggle.com) の無料 GPU (週 30 時間) を **Kaggle 公式 API** で使います。
アプリの ⚙️ に手順とリンクがあります。
1. Kaggle に無料登録
2. Kaggle の設定ページで電話番号認証 (GPU とインターネット接続に必要)
3. 同じページの API → **Create New Token** で出たトークンと、ユーザー名をアプリの ⚙️ に入力

## なぜこの方式か
- Google Colab の無料枠は「ノートブックを使わず Web UI から生成する使い方」を禁止しており、実行中に強制終了されます。Kaggle も Web UI 接続を禁止しています。
- そのため、GPU をリアルタイムに遠隔操作する方式ではなく、**Kaggle 公式 API でバッチ処理 (ジョブ) を投入し、結果を受け取る** 方式にしています。
- 各ジョブは使用者本人の Kaggle アカウントで非公開ノートブックとして実行され、終了後に自動で片付けます (LoRA 学習の結果は LoRA の保管場所として残します)。

## 軽量化
- Flux / T5 / Wan / UMT5 は **GGUF** (Q4〜Q5) で T4 GPU に収める
- LoRA 学習は UNet のみ・テキストエンコーダ出力キャッシュ・fp16 VAE・8bit AdamW・gradient checkpointing。メモリ不足時は解像度を下げて自動リトライ

## 注意
- モデルのライセンスに従うこと (Flux.1 schnell / Wan2.2: Apache-2.0、Animagine XL 4.0: Fair AI Public License、RealVisXL: 配布元の規約)
- 実在人物の画像・LoRA は本人の同意があるもののみ
- Kaggle の API トークンは端末内にのみ保存されます

## 構成
- `mobile/` — Android アプリ (Capacitor)。`main` への push で GitHub Actions が APK をビルドし、上記ページに公開
  - `src/main.js` 画面とジョブ管理 / `src/kaggle.js` Kaggle 公式 API クライアント
- `pai/job.py` — Kaggle 上で 1 件のジョブを実行する入口
- `pai/core.py` (セットアップ・モデル取得・ComfyUI) / `pai/comfy.py` (ComfyUI API) / `pai/train.py` (LoRA 学習) / `pai/threed.py` (3D)
- `workflows/*.json` — ComfyUI ワークフロー
