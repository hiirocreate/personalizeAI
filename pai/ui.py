"""1 画面で完結するツール UI (Gradio)。画像 → 動画 / 3D、LoRA 学習 → 即生成 までタブ内で連携。"""
import os
import random

import gradio as gr

from . import core, threed, train
from .comfy import Comfy

IMAGE_MODELS = {
    "Animagine XL 4.0 (イラスト・LoRA可)": ("sdxl", "animagine"),
    "RealVisXL 5.0 (実写・LoRA可)": ("sdxl", "realvis"),
    "Flux schnell GGUF (高速・汎用)": ("flux_gguf", "flux"),
}
BASE_TO_MODEL = {"animagine": "Animagine XL 4.0 (イラスト・LoRA可)", "realvis": "RealVisXL 5.0 (実写・LoRA可)"}
SIZES = {"縦 832×1216": (832, 1216), "正方形 1024×1024": (1024, 1024), "横 1216×832": (1216, 832)}
NEG = "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry, watermark"
NO_LORA = "なし"
BUSY = "⚠️ LoRA 学習中は生成できません (学習タブで完了を待つか中止してください)"


def _comfy(keys):
    for k in keys:
        if not core.ready(k):
            yield f"⏬ モデルをダウンロード中 (初回のみ・約 {core.SIZES_GB[k]:.0f}GB)…"
        core.ensure(k)
    core.start_comfy()


def pick(ps, e: gr.SelectData):
    return ps[e.index] if ps and e.index < len(ps) else None


def lora_choices():
    return [NO_LORA] + core.list_loras()


# ------------------------------------------------------------------ handlers
def gen_image(prompt, model, lora, strength, size, neg, steps, seed, count):
    if train.running:
        yield BUSY, gr.skip(), gr.skip(), gr.skip()
        return
    wf, key = IMAGE_MODELS[model]
    for msg in _comfy([key]):
        yield msg, gr.skip(), gr.skip(), gr.skip()
    w, h = SIZES[size]
    paths = []
    for i in range(int(count)):
        yield f"🎨 生成中 {i + 1}/{int(count)}…", gr.skip(), gr.skip(), gr.skip()
        s = int(seed) + i if seed >= 0 else random.randint(0, 2**31 - 1)
        p = dict(prompt=prompt, width=w, height=h, seed=s)
        if wf == "sdxl":
            p.update(ckpt=os.path.basename(core.model_path(key)), negative=neg or NEG,
                     steps=int(steps) or 28)
            if lora != NO_LORA:
                p.update(lora=lora, lora_strength=strength)
        else:
            p["steps"] = int(steps) or 4
        paths += Comfy().run(wf, out_dir=f"{core.DATA}/outputs", **p)
    yield f"✅ 完了 ({len(paths)} 枚)", paths, paths, paths[0]


def gen_video(prompt, image, orient, seconds, steps, seed):
    if train.running:
        yield BUSY, gr.skip()
        return
    for msg in _comfy(["wan"]):
        yield msg, gr.skip()
    core.free_vram()
    w, h = (832, 480) if orient.startswith("横") else (480, 832)
    frames = int(seconds * 24) // 4 * 4 + 1
    yield f"🎬 生成中… ({frames} フレーム、T4 で 5〜15 分)", gr.skip()
    p = dict(prompt=prompt, width=w, height=h, frames=frames, steps=int(steps),
             seed=int(seed) if seed >= 0 else random.randint(0, 2**31 - 1), image=image or "")
    files = Comfy().run("wan22_video", out_dir=f"{core.DATA}/outputs", **p)
    core.free_vram()
    yield "✅ 完了", next((f for f in files if f.endswith(".mp4")), files[0])


def gen_3d(image, texture):
    if train.running:
        yield BUSY, gr.skip(), gr.skip()
        return
    if not image:
        yield "画像を入れてください", gr.skip(), gr.skip()
        return
    if not threed.installed():
        yield "⏳ 3D 環境をセットアップ中 (初回のみ 5 分前後)…", gr.skip(), gr.skip()
    yield "🧊 3D 化中 (30 秒〜1 分)…", gr.skip(), gr.skip()
    glb = threed.run(image, texture)
    yield "✅ 完了 (右上のボタンでダウンロード)", glb, glb


def add_images(name, files):
    name = (name or "").strip()
    if not name:
        return "LoRA 名を先に入力してください", gr.skip()
    n = train.import_images(name, files)
    return f"📁 {name}: {n} 枚 (15〜40 枚推奨)", gr.update(choices=train.list_datasets(), value=name)


def run_train(name, trigger, base_label, caption, resolution, epochs, dim):
    name = (name or "").strip()
    no = (gr.skip(),) * 3
    if not name:
        yield "LoRA 名を入力してください", "", *no
        return
    base = next(k for k, v in train.BASES.items() if v == base_label)
    g = train.run(name, (trigger or name).strip(), base, "tags" if caption.startswith("自動") else "trigger",
                  resolution, int(epochs), int(dim))
    try:
        while True:
            msg, log = next(g)
            yield msg, log, *no
    except StopIteration as e:
        lora = e.value
    except Exception as e:  # noqa: BLE001 — UI に表示
        yield f"❌ {e}", "", *no
        return
    yield (f"✅ 完成: **{lora}**  → 画像タブで選択済み。プロンプトに `{trigger or name}` を入れて生成",
           "", gr.update(choices=lora_choices(), value=lora), BASE_TO_MODEL[base], gr.Tabs(selected="img"))


# ------------------------------------------------------------------ layout
CSS = "footer{display:none!important} .status{min-height:1.6em}"


def build():
    with gr.Blocks(title="personalizeAI", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("## 🎨 personalizeAI — 画像・動画・LoRA・3D")
        paths = gr.State([])
        picked = gr.State(None)
        with gr.Tabs() as tabs:
            # ---------------- 画像
            with gr.Tab("🖼 画像", id="img"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=300):
                        i_prompt = gr.Textbox(label="プロンプト (英語推奨)", lines=3,
                                              value="1girl, silver hair, school uniform, cherry blossoms, masterpiece, best quality")
                        i_model = gr.Dropdown(list(IMAGE_MODELS), value=list(IMAGE_MODELS)[0], label="モデル")
                        with gr.Row():
                            i_lora = gr.Dropdown(lora_choices(), value=NO_LORA, label="LoRA", scale=2)
                            i_str = gr.Slider(0, 1.5, 0.8, step=0.05, label="LoRA 強度", scale=1)
                        i_size = gr.Radio(list(SIZES), value="縦 832×1216", label="サイズ")
                        i_count = gr.Slider(1, 4, 1, step=1, label="枚数")
                        with gr.Accordion("詳細設定", open=False):
                            i_neg = gr.Textbox(NEG, label="ネガティブ (SDXL)", lines=2)
                            with gr.Row():
                                i_steps = gr.Number(0, label="ステップ (0=自動)", precision=0)
                                i_seed = gr.Number(-1, label="シード (-1=ランダム)", precision=0)
                        i_btn = gr.Button("生成", variant="primary", size="lg")
                        i_status = gr.Markdown(elem_classes="status")
                    with gr.Column(scale=3, min_width=300):
                        i_gal = gr.Gallery(label="結果 (クリックで選択)", columns=2, height=560, preview=True)
                        with gr.Row():
                            i_to_vid = gr.Button("🎬 選択画像を動画に")
                            i_to_3d = gr.Button("🧊 選択画像を3Dに")
            # ---------------- 動画
            with gr.Tab("🎬 動画", id="vid"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=300):
                        v_prompt = gr.Textbox(label="動きの説明 (英語推奨)", lines=3,
                                              value="the girl smiles and waves her hand, petals falling, gentle camera push-in")
                        v_img = gr.Image(label="開始画像 (空ならテキストだけで生成)", type="filepath", height=260)
                        v_orient = gr.Radio(["横 832×480", "縦 480×832"], value="縦 480×832", label="向き")
                        v_sec = gr.Slider(1, 5, 2, step=0.5, label="長さ (秒)")
                        with gr.Accordion("詳細設定", open=False):
                            with gr.Row():
                                v_steps = gr.Number(20, label="ステップ", precision=0)
                                v_seed = gr.Number(-1, label="シード", precision=0)
                        v_btn = gr.Button("動画を生成", variant="primary", size="lg")
                        v_status = gr.Markdown(elem_classes="status")
                    with gr.Column(scale=3, min_width=300):
                        v_out = gr.Video(label="結果", height=560, autoplay=True, loop=True)
            # ---------------- LoRA 学習
            with gr.Tab("🧠 LoRA 学習", id="lora"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=300):
                        l_name = gr.Dropdown(train.list_datasets(), label="LoRA 名 (英数字・新規は入力)",
                                             allow_custom_value=True)
                        l_files = gr.File(label="学習画像 15〜40 枚 (複数選択 / zip / HEIC 可)",
                                          file_count="multiple", height=160)
                        l_count = gr.Markdown(elem_classes="status")
                        l_base = gr.Radio(list(train.BASES.values()), value=list(train.BASES.values())[0], label="種類")
                        with gr.Accordion("詳細設定", open=False):
                            l_trigger = gr.Textbox(label="トリガーワード (空=LoRA名)")
                            l_caption = gr.Radio(["自動タグ付け (イラスト向け)", "トリガーのみ (実写人物向け)"],
                                                 value="自動タグ付け (イラスト向け)", label="キャプション")
                            with gr.Row():
                                l_res = gr.Radio(["768", "1024"], value="1024", label="解像度")
                                l_epochs = gr.Slider(4, 20, 10, step=1, label="エポック")
                                l_dim = gr.Radio([8, 16, 32], value=16, label="Dim")
                        with gr.Row():
                            l_btn = gr.Button("学習開始", variant="primary", size="lg", scale=3)
                            l_stop = gr.Button("中止", variant="stop", scale=1)
                        l_status = gr.Markdown(elem_classes="status")
                    with gr.Column(scale=3, min_width=300):
                        l_log = gr.Textbox(label="ログ", lines=18, max_lines=18, autoscroll=True)
                        gr.Markdown("完成後は自動で画像タブに移り、作った LoRA が選択された状態になります。"
                                    "実在人物は本人の同意がある場合のみ学習してください。")
            # ---------------- 3D
            with gr.Tab("🧊 3D", id="3d"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=300):
                        t_img = gr.Image(label="元画像 (正面・全身・白背景がおすすめ)", type="filepath", height=320)
                        t_tex = gr.Checkbox(True, label="テクスチャを焼き込む")
                        t_btn = gr.Button("3D 化", variant="primary", size="lg")
                        t_status = gr.Markdown(elem_classes="status")
                        gr.Markdown("より高品質: [Hunyuan3D-2.1](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1) / "
                                    "[TRELLIS](https://huggingface.co/spaces/trellis-community/TRELLIS)")
                    with gr.Column(scale=3, min_width=300):
                        t_view = gr.Model3D(label="結果 (ドラッグで回転)", height=480)
                        t_file = gr.File(label="GLB ダウンロード")
            # ---------------- 作品
            with gr.Tab("📁 作品", id="gal") as tab_gal:
                g_gal = gr.Gallery(label="保存済み (Drive/personalizeAI/outputs)", columns=4, height=600)
                with gr.Row():
                    g_to_vid = gr.Button("🎬 選択を動画に")
                    g_to_3d = gr.Button("🧊 選択を3Dに")
                    g_refresh = gr.Button("🔄 更新")
                g_paths = gr.State([])

        # ---------------- events
        i_model.change(lambda m: gr.update(interactive=IMAGE_MODELS[m][0] == "sdxl"), i_model, i_lora)
        i_btn.click(gen_image, [i_prompt, i_model, i_lora, i_str, i_size, i_neg, i_steps, i_seed, i_count],
                    [i_status, i_gal, paths, picked])
        i_gal.select(pick, paths, picked)
        send_vid = lambda p: (p, gr.Tabs(selected="vid"))  # noqa: E731
        send_3d = lambda p: (p, gr.Tabs(selected="3d"))  # noqa: E731
        i_to_vid.click(send_vid, picked, [v_img, tabs])
        i_to_3d.click(send_3d, picked, [t_img, tabs])

        v_btn.click(gen_video, [v_prompt, v_img, v_orient, v_sec, v_steps, v_seed], [v_status, v_out])

        l_files.upload(add_images, [l_name, l_files], [l_count, l_name])
        l_name.change(lambda n: f"📁 {n}: 保存済みの画像を使用" if n in train.list_datasets() else "", l_name, l_count)
        l_base.change(lambda b: "トリガーのみ (実写人物向け)" if "実写" in b else "自動タグ付け (イラスト向け)",
                      l_base, l_caption)
        l_btn.click(run_train, [l_name, l_trigger, l_base, l_caption, l_res, l_epochs, l_dim],
                    [l_status, l_log, i_lora, i_model, tabs])
        l_stop.click(train.stop)

        t_btn.click(gen_3d, [t_img, t_tex], [t_status, t_view, t_file])

        def refresh():
            ps = core.list_outputs()[:60]
            return ps, ps
        tab_gal.select(refresh, None, [g_gal, g_paths])
        g_refresh.click(refresh, None, [g_gal, g_paths])
        g_gal.select(pick, g_paths, picked)
        g_to_vid.click(send_vid, picked, [v_img, tabs])
        g_to_3d.click(send_3d, picked, [t_img, tabs])
    return demo


def launch():
    """Colab から呼ぶ入口: セットアップ → 既定モデルを裏で先読み → 共有 URL で起動。"""
    import subprocess
    if subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode:
        raise SystemExit("GPU がありません: メニュー「ランタイム → ランタイムのタイプを変更 → T4 GPU」にしてから再実行")
    core.install()
    core.prefetch("animagine")
    core.start_comfy()
    build().queue().launch(share=True, allowed_paths=[core.DATA, core.WORK], debug=True, show_error=True)
