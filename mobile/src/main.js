import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";
import { Share } from "@capacitor/share";
import JSZip from "jszip";
import * as kaggle from "./kaggle.js";

const $ = (id) => document.getElementById(id);
const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
const save = (k, v) => localStorage.setItem(k, JSON.stringify(v));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const src = (uri) => (uri.startsWith("http") ? uri : Capacitor.convertFileSrc(uri));
const rand = () => Math.floor(Math.random() * 2 ** 31);

let jobs = load("jobs", []);       // {slug, kind, title, status, created, error, dataset, lora}
let works = load("works", []);     // {uri, name, type, prompt, created}
let loras = load("loras", []);     // {name, file, kernel, base}
const KIND = { image: "画像", video: "動画", "3d": "3D", train: "LoRA 学習" };
const ETA = { image: 10, video: 25, "3d": 15, train: 80 };

function status(id, msg, err) { const s = $(id); s.textContent = msg; s.classList.toggle("err", !!err); }
function seg(id) { return $(id).querySelector("button.on").dataset.v; }
document.querySelectorAll(".seg").forEach((el) =>
  el.addEventListener("click", (e) => {
    if (!e.target.dataset.v) return;
    el.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === e.target));
    el.dispatchEvent(new Event("change"));
  }));

// ---------------------------------------------------------------- tabs
function showTab(t) {
  document.querySelectorAll("nav#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  document.querySelectorAll("main section").forEach((s) => s.classList.toggle("on", s.id === t));
  window.scrollTo(0, 0);
}
$("tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) showTab(b.dataset.t); });

// ---------------------------------------------------------------- 画像を base64 (縮小 JPEG) に
async function toJpegB64(blobOrUrl, max = 1024) {
  const url = typeof blobOrUrl === "string" ? blobOrUrl : URL.createObjectURL(blobOrUrl);
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });
  const k = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(img.naturalWidth * k); c.height = Math.round(img.naturalHeight * k);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.9).split(",")[1];
}

// ---------------------------------------------------------------- 設定
function openSettings() {
  const c = kaggle.creds.get();
  $("k-user").value = c.user || ""; $("k-key").value = c.key || "";
  $("settings").showModal();
}
$("open-settings").onclick = openSettings;
$("k-close").onclick = () => $("settings").close();
$("k-save").onclick = async () => {
  kaggle.creds.set({ user: $("k-user").value.trim(), key: $("k-key").value.trim() });
  status("k-status", "接続テスト中…");
  try {
    const q = await kaggle.quota();
    status("k-status", `✅ 接続OK — 今週の GPU 残り ${(q.total - q.used).toFixed(1)} / ${q.total.toFixed(0)} 時間`);
  } catch (e) { status("k-status", "❌ " + e.message, true); }
};
function needCreds(statusId) {
  const c = kaggle.creds.get();
  if (c.user && c.key) return false;
  status(statusId, "先に ⚙️ 設定で Kaggle を登録してください", true);
  openSettings();
  return true;
}

// ---------------------------------------------------------------- ジョブ
async function submit(kind, title, job, opts = {}, extra = {}) {
  const slug = await kaggle.pushJob(kind === "3d" ? "3d" : kind, job, opts);
  jobs.unshift({ slug, kind, title, status: "QUEUED", created: Date.now(), ...extra });
  save("jobs", jobs); renderJobs();
  return slug;
}

async function finish(j) {
  const { files, log } = await kaggle.outputs(j.slug);
  const resFile = files.find((f) => f.fileName === "result.json");
  let result = { ok: false, message: "結果が取得できませんでした" };
  if (resFile) result = JSON.parse(await (await fetch(resFile.url)).text());
  if (!result.ok) {
    j.status = "ERROR"; j.error = result.message || log.slice(-300);
  } else if (j.kind === "train") {
    const file = result.files[0];
    loras = loras.filter((l) => l.name !== j.lora.name);
    loras.unshift({ name: j.lora.name, file, kernel: j.slug, base: j.lora.base });
    save("loras", loras); renderLoras();
    j.status = "DONE";
  } else {
    for (const name of result.files) {
      const f = files.find((x) => x.fileName === name || x.fileName.endsWith("/" + name));
      if (!f) continue;
      const uri = await kaggle.download(f.url, `${Date.now()}_${name}`);
      const type = /\.mp4$/i.test(name) ? "video" : /\.glb$/i.test(name) ? "glb" : "image";
      works.unshift({ uri, name, type, prompt: j.title, created: Date.now() });
    }
    save("works", works); renderWorks();
    j.status = "DONE";
    kaggle.deleteKernel(j.slug); // LoRA 以外のジョブは片付ける
  }
  if (j.dataset) kaggle.deleteDataset(j.dataset);
}

let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    for (const j of jobs.filter((x) => ["QUEUED", "RUNNING"].includes(x.status))) {
      try {
        const s = await kaggle.status(j.slug);
        if (s.status === "COMPLETE") await finish(j);
        else if (s.status === "ERROR" || s.status.startsWith("CANCEL")) {
          j.status = "ERROR"; j.error = s.failure || "Kaggle 側でエラー";
          try { j.error = (await kaggle.outputs(j.slug)).log.slice(-300) || j.error; } catch { /* ignore */ }
        } else j.status = s.status;
      } catch (e) { j.note = e.message; }
    }
    save("jobs", jobs); renderJobs();
  } finally { polling = false; }
}
setInterval(poll, 20000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });

function renderJobs() {
  const active = jobs.filter((j) => ["QUEUED", "RUNNING"].includes(j.status)).length;
  $("jobs-badge").textContent = active ? `⏳ ${active} 件処理中` : "";
  if (!jobs.length) { $("job-list").innerHTML = '<p class="hint">依頼はまだありません</p>'; return; }
  $("job-list").innerHTML = jobs.slice(0, 15).map((j) => {
    const min = Math.round((Date.now() - j.created) / 60000);
    const [label, cls] = { QUEUED: ["順番待ち", ""], RUNNING: ["実行中", ""], DONE: ["完了", "ok"], ERROR: ["失敗", "err"] }[j.status] || [j.status, ""];
    const sub = j.status === "ERROR" ? `<small class="err">${esc(j.error)}</small>`
      : ["QUEUED", "RUNNING"].includes(j.status) ? `<small>${min} 分経過 / 目安 ${ETA[j.kind]} 分</small>`
      : `<small>${new Date(j.created).toLocaleString()}</small>`;
    return `<div class="job"><div class="t"><b>${KIND[j.kind]}: ${esc(j.title)}</b>${sub}</div><span class="pill ${cls}">${label}</span></div>`;
  }).join("");
}

// ---------------------------------------------------------------- 作品
function renderWorks() {
  $("works-grid").innerHTML = works.map((w, i) => {
    const media = w.type === "video" ? `<video src="${src(w.uri)}" muted playsinline preload="metadata"></video>`
      : w.type === "glb" ? '<div class="glb-thumb">🧊</div>' : `<img src="${src(w.uri)}" loading="lazy" alt="">`;
    return `<div class="item" data-i="${i}">${media}<span class="tag">${{ image: "画像", video: "動画", glb: "3D" }[w.type]}</span></div>`;
  }).join("");
}
let current = null;
function openViewer(w) {
  current = w;
  const s = src(w.uri);
  $("viewer-body").innerHTML = w.type === "video" ? `<video src="${s}" controls autoplay loop playsinline></video>`
    : w.type === "glb" ? `<model-viewer src="${s}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>`
    : `<img src="${s}" alt="">`;
  $("viewer-caption").textContent = w.prompt || "";
  $("v-vid").disabled = $("v-3d").disabled = w.type !== "image";
  $("viewer").classList.add("on");
}
const closeViewer = () => { $("viewer").classList.remove("on"); $("viewer-body").innerHTML = ""; };
$("works-grid").addEventListener("click", (e) => { const it = e.target.closest(".item"); if (it) openViewer(works[+it.dataset.i]); });
$("img-out").addEventListener("click", (e) => { const it = e.target.closest(".item"); if (it && it.dataset.w) openViewer(JSON.parse(it.dataset.w)); });
$("v-close").onclick = closeViewer;
$("v-share").onclick = async () => {
  try { await Share.share({ files: [current.uri] }); } catch { /* キャンセル */ }
};
$("v-del").onclick = async () => {
  if (!confirm("削除しますか？")) return;
  works = works.filter((w) => w.uri !== current.uri); save("works", works);
  if (!current.uri.startsWith("http")) Filesystem.deleteFile({ path: current.uri }).catch(() => {});
  closeViewer(); renderWorks();
};
let vidImage = null, tdImage = null;
function setPicked(kind, url) {
  const [prev] = kind === "vid" ? ["vid-preview"] : ["td-preview"];
  $(prev).src = url; $(prev).classList.remove("hide");
  if (kind === "vid") vidImage = url; else tdImage = url;
}
$("v-vid").onclick = () => { setPicked("vid", src(current.uri)); closeViewer(); showTab("t-vid"); };
$("v-3d").onclick = () => { setPicked("td", src(current.uri)); closeViewer(); showTab("t-3d"); };
$("vid-file").onchange = (e) => e.target.files[0] && setPicked("vid", URL.createObjectURL(e.target.files[0]));
$("td-file").onchange = (e) => e.target.files[0] && setPicked("td", URL.createObjectURL(e.target.files[0]));

// ---------------------------------------------------------------- 画像
const imgModeHint = {
  quick: "GPU 不要・数秒で表示 (Pollinations)。ラフ案やお試しに。",
  gpu: "Kaggle の無料 GPU で生成 (目安 5〜10 分)。自作 LoRA を使えます。完了すると「作品」に届きます。",
};
function applyImgMode() {
  const m = seg("img-mode");
  $("img-quick-opts").classList.toggle("hide", m !== "quick");
  $("img-gpu-opts").classList.toggle("hide", m !== "gpu");
  $("img-mode-hint").textContent = imgModeHint[m];
  $("img-go").textContent = m === "quick" ? "生成" : "生成を依頼";
}
$("img-mode").addEventListener("change", applyImgMode);
function renderLoraSelect() {
  const base = $("img-model").value;
  const list = loras.filter((l) => l.base === base);
  $("img-lora").innerHTML = '<option value="">なし</option>' + list.map((l) => `<option value="${esc(l.name)}">${esc(l.name)}</option>`).join("");
  $("img-lora").disabled = base === "flux";
}
$("img-model").onchange = renderLoraSelect;

$("img-go").onclick = async () => {
  const prompt = $("img-prompt").value.trim();
  const [w, h] = $("img-size").value.split("x").map(Number);
  const n = +$("img-count").value;
  if (seg("img-mode") === "quick") {
    const style = $("img-style").value;
    const full = style ? `${prompt}, ${style}` : prompt;
    status("img-status", "生成中… (数秒〜数十秒)");
    const items = [];
    for (let i = 0; i < n; i++) {
      const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(full)}?width=${w}&height=${h}&seed=${rand()}&model=flux&nologo=true`;
      items.push(url);
    }
    $("img-out").innerHTML = items.map((u) => `<div class="item"><img src="${u}" alt=""></div>`).join("");
    // 端末に保存して作品へ
    let okCount = 0;
    await Promise.all(items.map(async (u, i) => {
      try {
        const uri = await kaggle.download(u, `${Date.now()}_${i}.jpg`);
        works.unshift({ uri, name: `quick_${i}.jpg`, type: "image", prompt: full, created: Date.now() });
        okCount++;
      } catch { /* 表示のみ */ }
    }));
    save("works", works); renderWorks();
    status("img-status", `✅ 完了 — ${okCount} 枚を「作品」に保存`);
    return;
  }
  if (needCreds("img-status")) return;
  const lora = loras.find((l) => l.name === $("img-lora").value && l.base === $("img-model").value);
  const params = { model: $("img-model").value, prompt, negative: $("img-neg").value.trim(), size: [w, h], count: n, seed: -1 };
  const opts = {};
  if (lora) { params.lora = lora.file; params.lora_strength = +$("img-lora-str").value; opts.kernels = [`${kaggle.userName()}/${lora.kernel}`]; }
  $("img-go").disabled = true;
  try {
    await submit("image", prompt.slice(0, 40), { type: "image", params }, opts);
    status("img-status", "✅ 依頼しました。完了すると「作品」に届きます (アプリを閉じても処理は続きます)");
  } catch (e) { status("img-status", "❌ " + e.message, true); }
  $("img-go").disabled = false;
};

// ---------------------------------------------------------------- 動画
$("vid-go").onclick = async () => {
  if (needCreds("vid-status")) return;
  $("vid-go").disabled = true;
  try {
    const files = {};
    const params = { prompt: $("vid-prompt").value.trim(), orient: $("vid-orient").value, seconds: +$("vid-sec").value, seed: -1 };
    if (vidImage) { files["start.jpg"] = await toJpegB64(vidImage, 832); params.image = "/tmp/pai_in/start.jpg"; }
    await submit("video", params.prompt.slice(0, 40), { type: "video", params }, { files });
    status("vid-status", "✅ 依頼しました。完了すると「作品」に届きます");
  } catch (e) { status("vid-status", "❌ " + e.message, true); }
  $("vid-go").disabled = false;
};

// ---------------------------------------------------------------- 3D
$("td-go").onclick = async () => {
  if (!tdImage) return status("td-status", "画像を選んでください", true);
  if (needCreds("td-status")) return;
  $("td-go").disabled = true;
  try {
    const files = { "input.jpg": await toJpegB64(tdImage, 1024) };
    await submit("3d", "画像から3D", { type: "3d", params: { image: "/tmp/pai_in/input.jpg", texture: $("td-tex").checked } }, { files });
    status("td-status", "✅ 依頼しました。完了すると「作品」に届きます");
  } catch (e) { status("td-status", "❌ " + e.message, true); }
  $("td-go").disabled = false;
};

// ---------------------------------------------------------------- LoRA
function renderLoras() {
  const training = jobs.filter((j) => j.kind === "train" && ["QUEUED", "RUNNING"].includes(j.status));
  const rows = [
    ...training.map((j) => `<div class="lora"><span>${esc(j.lora.name)}</span><span class="pill">学習中</span></div>`),
    ...loras.map((l) => `<div class="lora"><span>${esc(l.name)} <small class="hint">${l.base === "realvis" ? "実写" : "イラスト"}</small></span><span class="pill ok">使用可</span></div>`),
  ];
  $("lora-list").innerHTML = rows.join("") || '<p class="hint">まだありません</p>';
  renderLoraSelect();
}
$("lora-files").onchange = (e) => {
  const fs = [...e.target.files];
  $("lora-thumbs").innerHTML = fs.slice(0, 40).map((f) => `<img src="${URL.createObjectURL(f)}" alt="">`).join("")
    + `<p class="hint" style="width:100%">${fs.length} 枚選択</p>`;
};
$("lora-go").onclick = async () => {
  const name = $("lora-name").value.trim();
  const files = [...$("lora-files").files];
  if (!/^[A-Za-z0-9_-]{2,40}$/.test(name)) return status("lora-status", "名前は英数字 (- _ 可) で入力してください", true);
  if (files.length < 5) return status("lora-status", "画像を 5 枚以上選んでください (15〜40 枚推奨)", true);
  if (needCreds("lora-status")) return;
  $("lora-go").disabled = true;
  try {
    status("lora-status", "画像をまとめています…");
    const zip = new JSZip();
    for (const [i, f] of files.entries()) {
      try { zip.file(`${i}.jpg`, await toJpegB64(f, 1536), { base64: true }); } // 端末で縮小して転送量を削減
      catch { zip.file(`${i}_${f.name}`, f); }                                   // HEIC 等はそのまま (サーバー側で変換)
    }
    const b64 = await zip.generateAsync({ type: "base64" });
    const path = `upload_${Date.now()}.zip`;
    await Filesystem.writeFile({ directory: Directory.Cache, path, data: b64 });
    const { uri } = await Filesystem.getUri({ directory: Directory.Cache, path });
    const size = Math.floor((b64.length * 3) / 4) - (b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0);
    const dataset = await kaggle.uploadDataset(uri, size, (m) => status("lora-status", m));
    Filesystem.deleteFile({ directory: Directory.Cache, path }).catch(() => {});
    const base = seg("lora-base");
    const params = { name, trigger: name, base, caption: base === "realvis" ? "trigger" : "tags", resolution: 1024, epochs: 10, dim: 16 };
    await submit("train", name, { type: "train", params }, { datasets: [`${kaggle.userName()}/${dataset}`] },
      { dataset, lora: { name, base } });
    renderLoras();
    status("lora-status", "✅ 学習を依頼しました (目安 60〜90 分)。完成すると画像タブの LoRA 欄に出ます");
  } catch (e) { status("lora-status", "❌ " + e.message, true); }
  $("lora-go").disabled = false;
};

// ---------------------------------------------------------------- init
applyImgMode();
renderLoras();
renderJobs();
renderWorks();
poll();
if (!kaggle.creds.get().user) status("img-status", "「⚡ すぐ生成」はそのまま使えます。GPU 機能は ⚙️ から Kaggle を登録");
