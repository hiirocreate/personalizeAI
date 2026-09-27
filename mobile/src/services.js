// 生成サービスの窓口。用途ごとに無料で使えるサービスを使い分ける。
//  - Pollinations : GPU 不要の即時画像
//  - Hugging Face : 公開 Space (ZeroGPU) を公式クライアントで利用 → 画像 / 動画 / 3D
//  - Modal        : 自分の GPU バックエンド (月 $30 の無料枠) → LoRA 学習・LoRA 画像
import { Client, handle_file } from "@gradio/client";

export const settings = {
  get() { try { return JSON.parse(localStorage.getItem("services") || "{}"); } catch { return {}; } },
  set(v) { localStorage.setItem("services", JSON.stringify({ ...settings.get(), ...v })); },
};

// ---------------------------------------------------------------- Pollinations
export function pollinationsUrl(prompt, w, h, seed) {
  return `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=${w}&height=${h}&seed=${seed}&model=flux&nologo=true`;
}

// ---------------------------------------------------------------- Hugging Face Spaces
const SPACES = {
  image: ["black-forest-labs/FLUX.1-schnell"],
  video: ["KingNish/wan2-2-fast", "zerogpu-aoti/wan2-2-fp8da-aoti-faster"],
  threeD: ["trellis-community/TRELLIS", "tencent/Hunyuan3D-2.1"],
};

function friendly(e) {
  const m = String(e?.message || e);
  if (/quota|exceeded/i.test(m)) {
    return "Hugging Face の本日の無料 GPU 枠 (1 日 数分) を使い切りました。"
      + (settings.get().hfToken ? "明日また使えます。" : "⚙️ で Hugging Face トークンを入れると枠が増えます。");
  }
  return m;
}

async function runSpace(space, endpoint, payload, onStatus) {
  const token = settings.get().hfToken;
  const client = await Client.connect(space, { ...(token ? { token } : {}), record_history: false });
  for await (const ev of client.submit(endpoint, payload)) {
    if (ev.type === "status") {
      if (ev.stage === "error") throw new Error(ev.message || "Space でエラー");
      if (ev.stage === "pending") onStatus?.(ev.position != null ? `順番待ち ${ev.position + 1} 番目…` : "準備中…");
      if (ev.stage === "generating") onStatus?.("生成中…");
    }
    if (ev.type === "data") return ev.data;
  }
  throw new Error("結果が返りませんでした");
}

/** 候補の Space を順に試す (混雑・停止中に備える) */
async function tryAll(list, fn, onStatus) {
  let last;
  for (const space of list) {
    try { return await fn(space); } catch (e) {
      last = e;
      if (/quota|exceeded/i.test(String(e?.message))) break; // 枠切れは他でも同じ
      onStatus?.(`${space} が混雑中のため別のサービスで再試行…`);
    }
  }
  throw new Error(friendly(last));
}

const fileUrl = (f) => (typeof f === "string" ? f : f?.url || f?.video?.url || f?.path);

export async function hfImage({ prompt, width, height }, onStatus) {
  return tryAll(SPACES.image, async (space) => {
    const d = await runSpace(space, "/infer",
      { prompt, seed: 0, randomize_seed: true, width, height, num_inference_steps: 4 }, onStatus);
    return fileUrl(d[0]);
  }, onStatus);
}

/** image: Blob または null (null ならテキストのみ。対応していない Space は飛ばす) */
export async function hfVideo({ prompt, image, seconds, portrait }, onStatus) {
  const [w, h] = portrait ? [480, 832] : [832, 480];
  const list = image ? SPACES.video : SPACES.video.slice(0, 1);
  return tryAll(list, async (space) => {
    const input = image ? handle_file(image) : null;
    const payload = space.startsWith("KingNish")
      ? { input_image: input, prompt, height: h, width: w, duration_seconds: seconds, steps: 4, randomize_seed: true }
      : { input_image: input, prompt, steps: 6, duration_seconds: Math.min(seconds, 5), randomize_seed: true };
    const d = await runSpace(space, "/generate_video", payload, onStatus);
    return fileUrl(d[0]);
  }, onStatus);
}

export async function hf3D({ image }, onStatus) {
  return tryAll(SPACES.threeD, async (space) => {
    if (space.includes("TRELLIS")) {
      const d = await runSpace(space, "/generate_and_extract_glb", {
        image: handle_file(image), multiimages: [], seed: 0, ss_guidance_strength: 7.5, ss_sampling_steps: 12,
        slat_guidance_strength: 3, slat_sampling_steps: 12, multiimage_algo: "stochastic",
        mesh_simplify: 0.95, texture_size: 1024,
      }, onStatus);
      return fileUrl(d[1]) || fileUrl(d[2]);
    }
    const d = await runSpace(space, "/shape_generation", [handle_file(image), null, null, null, null,
      30, 5.0, 1234, 256, true, 8000, true], onStatus);
    return fileUrl(d[0]);
  }, onStatus);
}

// ---------------------------------------------------------------- Modal (自分の GPU バックエンド)
function modalBase() {
  const { modal } = settings.get();
  if (!modal) throw new Error("⚙️ で Modal のワークスペース名 (または URL) を入力してください");
  return modal.startsWith("http") ? modal.replace(/\/$/, "") : `https://${modal}--personalizeai-api.modal.run`;
}

async function modalCall(path, body) {
  const { modalKey } = settings.get();
  const r = await fetch(modalBase() + path, {
    method: body ? "POST" : "GET",
    headers: { Authorization: `Bearer ${modalKey || ""}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  if (r.status === 401) throw new Error("Modal の API キーが違います (⚙️ で確認)");
  if (r.status === 404) throw new Error("Modal のバックエンドが見つかりません。ワークスペース名とデプロイ状況を確認してください");
  if (!r.ok) throw new Error(`Modal ${r.status}: ${text.slice(0, 300)}`);
  return JSON.parse(text);
}

export const modal = {
  health: () => modalCall("/api/health"),
  loras: () => modalCall("/api/loras"),
  train: (body) => modalCall("/api/train", body),
  image: (body) => modalCall("/api/image", body),
  result: (id) => modalCall(`/api/result/${encodeURIComponent(id)}`),
};
