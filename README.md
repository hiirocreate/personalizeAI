# personalizeAI

無料で使える、個人用の画像・動画・3D・LoRA 学習 **Android アプリ**。操作はアプリ画面の中で完結します。
用途ごとに、無料で・規約の範囲内で使えるサービスを使い分けています。

| 機能 | 使うサービス | 無料枠 | 時間 |
|---|---|---|---|
| 🖼 ⚡ すぐ生成 | Pollinations | 登録不要 | 数秒 |
| 🖼 🎯 高品質画像 (FLUX) | Hugging Face Spaces | 1 日 2 分 (トークンありで 5 分) の GPU | 10〜20 秒 |
| 🎬 動画 (Wan2.2) | Hugging Face Spaces | 同上 | 1〜3 分 |
| 🧊 3D (TRELLIS / Hunyuan3D) | Hugging Face Spaces | 同上 | 1〜2 分 |
| 🧠 LoRA 学習 | Modal (自分の GPU バックエンド) | 月 $30 分 (カード登録が必要) | 60〜100 分 (約 $0.6〜1/回) |
| 🖼 🧠 自作 LoRA で画像生成 | Modal | 同上 | 1〜3 分 |

- Hugging Face は公開 Space を公式クライアント経由で呼び出します (Space は API 利用が想定された公開アプリ)。混雑時は別の Space に自動で切り替えます。
- Modal はアプリのバックエンド用のサーバーレス GPU サービスで、今回の使い方はそのまま想定された用途です。GPU の利用にはカード登録が必要です。Modal の設定で使用額の上限を設定しておくと、無料枠を超えた請求を防げます。

## インストール
スマホで **[最新版 APK](https://github.com/hiirocreate/personalizeAI/releases/tag/app-latest)** を開き `personalizeAI.apk` をタップ (「提供元不明のアプリ」を許可)。

## 設定 (アプリの ⚙️)
- **Hugging Face トークン** (任意): [トークン作成](https://huggingface.co/settings/tokens) で Read トークンを作って貼る
- **Modal** (LoRA 機能を使う場合): 下の「Modal の準備」を 1 回行い、ワークスペース名と API キーを入れる

## Modal の準備 (最初に 1 回だけ)
1. [modal.com](https://modal.com) に無料登録 (GitHub アカウントでログイン可)
2. Modal の Settings → **API Tokens** → New Token で表示される `token id` と `token secret` を控える
3. このリポジトリの Settings → Secrets and variables → Actions → **New repository secret** で 3 つ登録
   - `MODAL_TOKEN_ID` : 2 の token id
   - `MODAL_TOKEN_SECRET` : 2 の token secret
   - `PAI_API_KEY` : 自分で決めた長めの合言葉 (アプリとバックエンドの間の鍵)
4. Actions → **Deploy GPU backend (Modal)** → Run workflow
5. アプリの ⚙️ に Modal のワークスペース名 (Modal の画面左上 / URL の `modal.com/apps/〇〇` の 〇〇) と `PAI_API_KEY` を入れて「保存して接続テスト」

## 注意
- Hugging Face の無料枠は 1 日数分なので、動画・3D は 1 日数回が目安です。枠を使い切ると翌日まで使えません
- モデルのライセンスに従うこと (FLUX.1 schnell / Wan2.2: Apache-2.0、Animagine XL 4.0: Fair AI Public License、RealVisXL: 配布元の規約、TRELLIS / Hunyuan3D: 各ライセンス)
- 実在人物の画像・LoRA は本人の同意があるもののみ
- トークン・API キーは端末内にのみ保存されます

## 構成
- `mobile/` — Android アプリ (Capacitor)。`main` への push で GitHub Actions が APK をビルドして上記ページに公開
  - `src/services.js` 各サービスの呼び出し / `src/main.js` 画面・ジョブ管理
- `modal_app/pai_modal.py` — Modal の GPU バックエンド (LoRA 学習: kohya-ss sd-scripts / 画像: diffusers SDXL + LoRA)。`modal_app/` 変更時に自動デプロイ
