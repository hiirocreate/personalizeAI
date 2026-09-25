const $ = (id) => document.getElementById(id);
const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
                set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
const rand = () => Math.floor(Math.random() * 2 ** 31);

// ---- tabs
function showTab(t) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  document.querySelectorAll("section").forEach((s) => s.classList.toggle("on", s.id === t));
}
$("tabs").onclick = (e) => { if (e.target.dataset.t) { showTab(e.target.dataset.t); location.hash = e.target.dataset.t; } };

// ---- quick (Pollinations)
$("q-go").onclick = () => {
  const style = $("q-style").value, p = $("q-prompt").value.trim();
  const prompt = style ? `${p}, ${style}` : p;
  const n = Math.min(4, Math.max(1, +$("q-n").value || 1));
  const out = $("q-out"); out.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const seed = +$("q-seed").value >= 0 ? +$("q-seed").value + i : rand();
    const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=${$("q-w").value}&height=${$("q-h").value}&seed=${seed}&model=flux&nologo=true`;
    out.insertAdjacentHTML("beforeend", `<div><img loading="lazy" src="${url}" alt=""><a href="${url}" target="_blank">seed ${seed} を開く</a></div>`);
  }
};

// ---- ComfyUI
const DEFAULTS = {
  flux_gguf: { width: 1024, height: 1024, steps: 4 },
  sdxl: { width: 832, height: 1216, steps: 28 },
  wan22_video: { width: 832, height: 480, steps: 20 },
};
const templates = {};
let models = { unet: [], ckpt: [], lora: [] };
const api = () => $("api").value.trim().replace(/\/$/, "");

function status(msg, err) { const s = $("status"); s.textContent = msg; s.classList.toggle("err", !!err); }

function fillSelect(el, items, empty) {
  el.innerHTML = (empty ? `<option value="">${empty}</option>` : "") + items.map((m) => `<option>${m}</option>`).join("");
}

function applyMode() {
  const m = $("mode").value;
  document.querySelectorAll("[data-m]").forEach((el) => el.classList.toggle("hide", !el.dataset.m.split(" ").includes(m)));
  Object.entries(DEFAULTS[m]).forEach(([k, v]) => ($(k).value = v));
  const key = m === "flux_gguf" ? "flux" : "wan";
  fillSelect($("unet"), models.unet.filter((u) => u.toLowerCase().includes(key)));
}
$("mode").onchange = applyMode;

async function connect() {
  const s = $("conn-status");
  s.classList.remove("err");
  if (!api()) return;
  store.set("api", api());
  try {
    const info = async (node, field) => {
      const r = await fetch(`${api()}/object_info/${node}`);
      const j = await r.json();
      return j[node]?.input.required[field][0] || [];
    };
    const [unet, ckpt, lora] = await Promise.all([
      info("UnetLoaderGGUF", "unet_name"), info("CheckpointLoaderSimple", "ckpt_name"), info("LoraLoader", "lora_name")]);
    models = { unet, ckpt, lora };
    fillSelect($("ckpt"), ckpt);
    fillSelect($("lora"), lora, "(なし)");
    applyMode();
    s.textContent = `✅ 接続OK — GGUF ${unet.length} / ckpt ${ckpt.length} / LoRA ${lora.length}`;
  } catch (e) {
    s.textContent = "❌ 接続失敗: Colab のセル③が実行中か、URL が正しいか確認 (" + e.message + ")";
    s.classList.add("err");
  }
}
$("connect").onclick = connect;

// optional パラメータが空ならノードを外して配線をつなぎ直す (pai/comfy.py の build と同じ)
function build(tpl, params) {
  const wf = structuredClone(tpl.workflow);
  for (const [k, [node, field]] of Object.entries(tpl.params))
    if (params[k] !== undefined && params[k] !== "") wf[node].inputs[field] = params[k];
  for (const [k, opt] of Object.entries(tpl.optional || {})) {
    if (params[k] !== undefined && params[k] !== "") continue;
    const removed = wf[opt.node]; delete wf[opt.node];
    for (const n of Object.values(wf))
      for (const [f, v] of Object.entries(n.inputs))
        if (Array.isArray(v) && v[0] === opt.node) {
          const src = opt.passthrough[String(v[1])];
          if (src) n.inputs[f] = removed.inputs[src]; else delete n.inputs[f];
        }
  }
  return wf;
}

async function upload(file) {
  const fd = new FormData(); fd.append("image", file); fd.append("overwrite", "true");
  const r = await fetch(`${api()}/upload/image`, { method: "POST", body: fd });
  return (await r.json()).name;
}

$("go").onclick = async () => {
  const m = $("mode").value, go = $("go");
  if (!api()) return status("ComfyUI URL を入力してください", true);
  go.disabled = true;
  try {
    templates[m] ||= await (await fetch(`workflows/${m}.json`)).json();
    const num = (id) => +$(id).value;
    const p = { prompt: $("prompt").value, width: num("width"), height: num("height"), steps: num("steps"),
                seed: num("seed") >= 0 ? num("seed") : rand() };
    if (m !== "sdxl") p.unet = $("unet").value;
    if (m === "sdxl") p.ckpt = $("ckpt").value;
    if (m !== "flux_gguf" && $("negative").value.trim()) p.negative = $("negative").value;
    if (m !== "wan22_video") { p.lora = $("lora").value; p.lora_strength = num("lora_strength"); }
    if (m === "wan22_video") {
      p.frames = num("frames");
      if ($("image").files[0]) { status("画像アップロード中…"); p.image = await upload($("image").files[0]); }
    }
    const r = await fetch(`${api()}/prompt`, { method: "POST", headers: { "Content-Type": "application/json" },
                                               body: JSON.stringify({ prompt: build(templates[m], p) }) });
    const j = await r.json();
    if (!j.prompt_id) throw new Error(JSON.stringify(j.node_errors || j.error).slice(0, 400));
    const t0 = Date.now();
    for (;;) {
      await new Promise((res) => setTimeout(res, 1500));
      status(`生成中… ${Math.round((Date.now() - t0) / 1000)} 秒 (seed ${p.seed})`);
      const h = (await (await fetch(`${api()}/history/${j.prompt_id}`)).json())[j.prompt_id];
      if (!h) continue;
      if (h.status?.status_str === "error") throw new Error(JSON.stringify(h.status.messages).slice(0, 400));
      const files = Object.values(h.outputs).flatMap((o) => Object.values(o).flat()).filter((f) => f?.filename);
      files.forEach(show);
      status(`✅ 完了 ${Math.round((Date.now() - t0) / 1000)} 秒 (seed ${p.seed})`);
      break;
    }
  } catch (e) { status("❌ " + e.message, true); }
  go.disabled = false;
};

function show(f) {
  const url = `${api()}/view?` + new URLSearchParams({ filename: f.filename, subfolder: f.subfolder || "", type: f.type || "output" });
  const media = /\.(mp4|webm)$/i.test(f.filename)
    ? `<video src="${url}" controls loop autoplay muted playsinline></video>` : `<img src="${url}" alt="">`;
  $("out").insertAdjacentHTML("afterbegin", `<div>${media}<a href="${url}" download="${f.filename}" target="_blank">${f.filename}</a></div>`);
}

// ---- 3D viewer
$("glb").onchange = (e) => { const f = e.target.files[0]; if (f) $("viewer").src = URL.createObjectURL(f); };

// ---- init
const qs = new URLSearchParams(location.search);
$("api").value = qs.get("api") || store.get("api") || "";
applyMode();
if (qs.get("api")) { showTab("gpu"); connect(); }
else if (location.hash) showTab(location.hash.slice(1));
