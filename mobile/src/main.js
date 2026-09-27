import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";
import { FileTransfer } from "@capacitor/file-transfer";
import { Share } from "@capacitor/share";
import { settings, pollinationsUrl, hfImage, hfVideo, hf3D, modal } from "./services.js";

const $ = (id) => document.getElementById(id);
const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
const save = (k, v) => localStorage.setItem(k, JSON.stringify(v));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const src = (uri) => (uri.startsWith("http") ? uri : Capacitor.convertFileSrc(uri));
const rand = () => Math.floor(Math.random() * 2 ** 31);

let jobs = load("jobs2", []);      // Modal の非同期ジョブ {id, kind, title, status, created, error, lora}
let works = load("works", []);     // {uri, name, type, prompt, created}
let loras = load("loras2", []);    // Modal 上の LoRA {name, base}
const KIND = { image: "LoRA 画像", train: "LoRA 学習" };
const ETA = { image: 3, train: 80 };

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

// ---------------------------------------------------------------- 画像ユーティリティ
async function fileB64(file) {
  return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1]); r.onerror = rej; r.readAsDataURL(file); });
}
async function toJpeg(blobOrUrl, max = 1024, quality = 0.9) {
  const url = typeof blobOrUrl === "string" ? blobOrUrl : URL.createObjectURL(blobOrUrl);
  const img = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });
  const k = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(img.naturalWidth * k); c.height = Math.round(img.naturalHeight * k);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, "image/jpeg", quality));
}
const blobB64 = (b) => fileB64(b);

// ---------------------------------------------------------------- 作品の保存
async function saveUrl(url, name, type, prompt) {
  const path = `works/${Date.now()}_${name}`;
  await Filesystem.mkdir({ directory: Directory.Data, path: "works", recursive: true }).catch(() => {});
  const { uri } = await Filesystem.getUri({ directory: Directory.Data, path });
  await FileTransfer.downloadFile({ url, path: uri });
  addWork({ uri, name, type, prompt });
}
async function saveB64(b64, name, type, prompt) {
  const { uri } = await Filesystem.writeFile({ directory: Directory.Data, path: `works/${Date.now()}_${name}`, data: b64, recursive: true });
  addWork({ uri, name, type, prompt });
}
function addWork(w) { works.unshift({ ...w, created: Date.now() }); save("works", works); renderWorks(); }

// ---------------------------------------------------------------- 設定
function openSettings() {
  const s = settings.get();
  $("s-hf").value = s.hfToken || ""; $("s-modal").value = s.modal || ""; $("s-modal-key").value = s.modalKey || "";
  $("settings").showModal();
}
$("open-settings").onclick = openSettings;
$("s-close").onclick = () => $("settings").close();
$("s-save").onclick = async () => {
  const hf = $("s-hf").value.trim();
  if (hf && !hf.startsWith("hf_")) return status("s-status", "Hugging Face トークンは hf_ で始まります", true);
  settings.set({ hfToken: hf, modal: $("s-modal").value.trim(), modalKey: $("s-modal-key").value.trim() });
  if (!settings.get().modal) return status("s-status", "✅ 保存しました (Modal 未設定: LoRA 機能以外は使えます)");
  status("s-status", "Modal に接続テスト中… (初回は 10〜30 秒)");
  try { await modal.health(); await refreshLoras(); status("s-status", "✅ 保存しました。Modal 接続OK"); }
  catch (e) { status("s-status", "❌ " + e.message, true); }
};
function needModal(statusId) {
  const s = settings.get();
  if (s.modal && s.modalKey) return false;
  status(statusId, "LoRA 機能には ⚙️ で Modal の設定が必要です", true);
  openSettings();
  return true;
}

// ---------------------------------------------------------------- Modal ジョブ (非同期)
const ACTIVE = ["QUEUED", "RUNNING"];
async function finish(j, result) {
  if (j.kind === "train") {
    await refreshLoras();
  } else {
    for (const [i, b64] of (result.images || []).entries()) await saveB64(b64, `lora_${i}.png`, "image", j.prompt || j.title);
  }
  j.status = "DONE";
}
let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  try {
    for (const j of jobs.filter((x) => ACTIVE.includes(x.status))) {
      try {
        const r = await modal.result(j.id);
        if (r.status === "done") await finish(j, r.result);
        else if (r.status === "error") { j.status = "ERROR"; j.error = r.message; }
        else j.status = "RUNNING";
        j.note = r.progress || "";
      } catch (e) { j.note = e.message; }
    }
    save("jobs2", jobs); renderJobs(); renderLoras();
  } finally { polling = false; }
}
setInterval(poll, 15000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });

function addJob(id, kind, title, extra = {}) {
  jobs.unshift({ id, kind, title, status: "QUEUED", created: Date.now(), ...extra });
  jobs = jobs.slice(0, 50);
  save("jobs2", jobs); renderJobs(); renderLoras();
}
function renderJobs() {
  const active = jobs.filter((j) => ACTIVE.includes(j.status)).length;
  $("jobs-badge").textContent = `⏳ ${active}件処理中`;
  $("jobs-badge").classList.toggle("hide", !active);
  if (!jobs.length) { $("job-list").innerHTML = '<p class="hint">依頼はまだありません</p>'; return; }
  $("job-list").innerHTML = jobs.map((j, i) => {
    const min = Math.round((Date.now() - j.created) / 60000);
    const [label, cls] = { QUEUED: ["順番待ち", ""], RUNNING: ["実行中", ""], DONE: ["完了", "ok"], ERROR: ["失敗", "err"] }[j.status] || [j.status, ""];
    const when = ACTIVE.includes(j.status) ? `${min} 分経過 / 目安 ${ETA[j.kind]} 分${j.note ? " — " + esc(j.note) : ""}`
      : new Date(j.created).toLocaleString();
    const err = j.status === "ERROR"
      ? `<details><summary class="err">エラー内容を表示</summary><pre>${esc(j.error)}</pre>`
        + `<button class="mini" data-act="copy-err" data-i="${i}">エラーをコピー</button></details>` : "";
    const tools = j.kind === "image"
      ? `<div><button class="mini" data-act="copy" data-i="${i}">プロンプトをコピー</button>`
        + `<button class="mini" data-act="reuse" data-i="${i}">このプロンプトで作る</button></div>` : "";
    return `<div class="job"><div class="t"><b>${KIND[j.kind]}</b><div class="ptext">${esc(j.prompt || j.title)}</div>`
      + `<small>${when}</small>${err}${tools}</div><span class="pill ${cls}">${label}</span></div>`;
  }).join("");
}
async function copyText(t) {
  try { await navigator.clipboard.writeText(t); toast("コピーしました"); } catch { toast("コピーできませんでした"); }
}
function toast(msg) {
  const el = Object.assign(document.createElement("div"), { textContent: msg });
  el.style.cssText = "position:fixed;left:50%;bottom:96px;transform:translateX(-50%);background:#000c;color:#fff;"
    + "padding:8px 14px;border-radius:99px;font-size:13px;z-index:50";
  document.body.append(el); setTimeout(() => el.remove(), 1500);
}
function reusePrompt(text) { $("img-prompt").value = text; closeViewer(); showTab("t-img"); }
$("job-list").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]"); if (!b) return;
  const j = jobs[+b.dataset.i];
  if (b.dataset.act === "copy-err") copyText(j.error || "");
  if (b.dataset.act === "copy") copyText(j.prompt || j.title);
  if (b.dataset.act === "reuse") reusePrompt(j.prompt || j.title);
});
$("jobs-badge").onclick = () => showTab("t-log");

// ---------------------------------------------------------------- 作品
function renderWorks() {
  $("works-grid").innerHTML = works.map((w, i) => {
    const media = w.type === "video" ? `<video src="${src(w.uri)}" muted playsinline preload="metadata"></video>`
      : w.type === "glb" ? '<div class="glb-thumb">🧊</div>' : `<img src="${src(w.uri)}" loading="lazy" alt="">`;
    return `<div class="item" data-i="${i}">${media}<span class="tag">${{ image: "画像", video: "動画", glb: "3D" }[w.type]}</span></div>`;
  }).join("") || '<p class="hint">まだ作品はありません</p>';
}
let current = null, curIndex = 0;
function openViewer(i) {
  if (i < 0 || i >= works.length) return;
  curIndex = i; current = works[i];
  const w = current, s = src(w.uri);
  $("viewer-body").innerHTML = w.type === "video" ? `<video src="${s}" controls autoplay loop playsinline></video>`
    : w.type === "glb" ? `<model-viewer src="${s}" camera-controls auto-rotate shadow-intensity="1"></model-viewer>`
    : `<img src="${s}" alt="" draggable="false">`;
  $("viewer-caption").textContent = w.prompt || "(プロンプトなし)";
  $("viewer-caption").classList.remove("full");
  $("v-more").textContent = "全文";
  $("v-count").textContent = `${i + 1} / ${works.length}`;
  $("v-prev").disabled = i === 0; $("v-next").disabled = i === works.length - 1;
  $("v-vid").disabled = $("v-3d").disabled = w.type !== "image";
  $("viewer").classList.add("on");
}
function closeViewer() { $("viewer").classList.remove("on"); $("viewer-body").innerHTML = ""; }
$("works-grid").addEventListener("click", (e) => { const it = e.target.closest(".item"); if (it?.dataset.i) openViewer(+it.dataset.i); });
$("v-close").onclick = closeViewer;
$("v-prev").onclick = () => openViewer(curIndex - 1);
$("v-next").onclick = () => openViewer(curIndex + 1);
document.addEventListener("keydown", (e) => {
  if (!$("viewer").classList.contains("on")) return;
  if (e.key === "ArrowLeft") openViewer(curIndex - 1);
  if (e.key === "ArrowRight") openViewer(curIndex + 1);
  if (e.key === "Escape") closeViewer();
});
// 左右スワイプで前後の作品へ (3D は回転操作と競合するので対象外)
let touch = null;
$("viewer-body").addEventListener("touchstart", (e) => {
  touch = current?.type === "glb" ? null : { x: e.touches[0].clientX, y: e.touches[0].clientY };
}, { passive: true });
$("viewer-body").addEventListener("touchend", (e) => {
  if (!touch) return;
  const dx = e.changedTouches[0].clientX - touch.x, dy = e.changedTouches[0].clientY - touch.y;
  touch = null;
  if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) openViewer(curIndex + (dx < 0 ? 1 : -1));
});
$("v-more").onclick = () => {
  const full = $("viewer-caption").classList.toggle("full");
  $("v-more").textContent = full ? "たたむ" : "全文";
};
$("viewer-caption").onclick = () => { if (!$("viewer-caption").classList.contains("full")) $("v-more").click(); };
$("v-copy").onclick = () => copyText(current?.prompt || "");
$("v-reuse").onclick = () => reusePrompt(current?.prompt || "");
$("v-share").onclick = async () => { try { await Share.share({ files: [current.uri] }); } catch { /* キャンセル */ } };
$("v-del").onclick = async () => {
  if (!confirm("削除しますか？")) return;
  works = works.filter((w) => w.uri !== current.uri); save("works", works);
  Filesystem.deleteFile({ path: current.uri }).catch(() => {});
  renderWorks();
  if (works.length) openViewer(Math.min(curIndex, works.length - 1)); else closeViewer();
};
const picked = { vid: null, td: null }; // Blob
async function setPicked(kind, blobOrUrl) {
  const blob = typeof blobOrUrl === "string" ? await (await fetch(blobOrUrl)).blob() : blobOrUrl;
  picked[kind] = blob;
  const prev = $(`${kind}-preview`);
  prev.src = URL.createObjectURL(blob); prev.classList.remove("hide");
}
$("v-vid").onclick = () => { setPicked("vid", src(current.uri)); closeViewer(); showTab("t-vid"); };
$("v-3d").onclick = () => { setPicked("td", src(current.uri)); closeViewer(); showTab("t-3d"); };
$("vid-file").onchange = (e) => e.target.files[0] && setPicked("vid", e.target.files[0]);
$("td-file").onchange = (e) => e.target.files[0] && setPicked("td", e.target.files[0]);

// ---------------------------------------------------------------- 画像
const imgModeHint = {
  quick: "⚡ 登録不要・数秒 (Pollinations)。お試しやラフ案に。",
  hq: "🎯 高品質 (Hugging Face・FLUX)。1 日数十枚程度の無料枠。",
  lora: "🧠 自作 LoRA で生成 (Modal・月 $30 の無料枠)。1〜3 分。",
};
function applyImgMode() {
  const m = seg("img-mode");
  $("img-quick-opts").classList.toggle("hide", m !== "quick");
  $("img-lora-opts").classList.toggle("hide", m !== "lora");
  $("img-mode-hint").textContent = imgModeHint[m];
}
$("img-mode").addEventListener("change", applyImgMode);

function showResults(items) {
  $("img-out").innerHTML = items.map((u) => `<div class="item"><img src="${u}" alt=""></div>`).join("");
}
$("img-go").onclick = async () => {
  const prompt = $("img-prompt").value.trim();
  const [w, h] = $("img-size").value.split("x").map(Number);
  const n = +$("img-count").value;
  const m = seg("img-mode");
  $("img-go").disabled = true;
  try {
    if (m === "quick") {
      const style = $("img-style").value;
      const full = style ? `${prompt}, ${style}` : prompt;
      const urls = Array.from({ length: n }, () => pollinationsUrl(full, w, h, rand()));
      showResults(urls);
      status("img-status", "生成中… (数秒〜数十秒)");
      const ok = await Promise.allSettled(urls.map((u, i) => saveUrl(u, `quick_${i}.jpg`, "image", full)));
      status("img-status", `✅ ${ok.filter((r) => r.status === "fulfilled").length} 枚を「作品」に保存`);
    } else if (m === "hq") {
      const urls = [];
      for (let i = 0; i < n; i++) {
        const url = await hfImage({ prompt, width: w, height: h }, (s) => status("img-status", `${i + 1}/${n}: ${s}`));
        urls.push(url); showResults(urls);
        await saveUrl(url, `hq_${i}.webp`, "image", prompt);
      }
      status("img-status", `✅ ${urls.length} 枚を「作品」に保存`);
    } else {
      if (needModal("img-status")) return;
      const lora = $("img-lora").value;
      // LoRA はトリガーワード (= LoRA 名) がプロンプトに無いと効きにくいので自動で先頭に付ける
      const fullPrompt = lora && !prompt.toLowerCase().includes(lora.toLowerCase()) ? `${lora}, ${prompt}` : prompt;
      const base = lora ? loras.find((l) => l.name === lora)?.base : $("img-base").value;
      const { call_id } = await modal.image({
        prompt: fullPrompt, model: base, lora, lora_strength: +$("img-lora-str").value, negative: $("img-neg").value.trim(),
        width: w, height: h, count: n, seed: -1,
      });
      addJob(call_id, "image", fullPrompt.slice(0, 40), { prompt: fullPrompt });
      status("img-status", "✅ 依頼しました。1〜3 分で「作品」に届きます");
      setTimeout(poll, 30000);
    }
  } catch (e) { status("img-status", "❌ " + e.message, true); }
  finally { $("img-go").disabled = false; }
};

// ---------------------------------------------------------------- 動画 (Hugging Face)
$("vid-go").onclick = async () => {
  $("vid-go").disabled = true;
  try {
    const image = picked.vid ? await toJpeg(picked.vid, 1024) : null;
    const prompt = $("vid-prompt").value.trim();
    const url = await hfVideo({ prompt, image, seconds: +$("vid-sec").value, portrait: $("vid-orient").value === "portrait" },
      (s) => status("vid-status", s + " (画面を開いたままお待ちください)"));
    status("vid-status", "保存中…");
    await saveUrl(url, "video.mp4", "video", prompt);
    status("vid-status", "✅ 完成。「作品」に保存しました");
    openViewer(0);
  } catch (e) { status("vid-status", "❌ " + e.message, true); }
  finally { $("vid-go").disabled = false; }
};

// ---------------------------------------------------------------- 3D (Hugging Face)
$("td-go").onclick = async () => {
  if (!picked.td) return status("td-status", "画像を選んでください", true);
  $("td-go").disabled = true;
  try {
    const url = await hf3D({ image: await toJpeg(picked.td, 1024) }, (s) => status("td-status", s + " (画面を開いたままお待ちください)"));
    status("td-status", "保存中…");
    await saveUrl(url, "model.glb", "glb", "画像から3D");
    status("td-status", "✅ 完成。「作品」に保存しました");
    openViewer(0);
  } catch (e) { status("td-status", "❌ " + e.message, true); }
  finally { $("td-go").disabled = false; }
};

// ---------------------------------------------------------------- LoRA (Modal)
async function refreshLoras() {
  try { loras = await modal.loras(); save("loras2", loras); } catch { /* オフライン時は前回の一覧 */ }
  renderLoras();
}
function renderLoras() {
  const training = jobs.filter((j) => j.kind === "train" && ACTIVE.includes(j.status));
  const rows = [
    ...training.map((j) => `<div class="lora"><span>${esc(j.lora.name)}</span><span class="pill">学習中</span></div>`),
    ...loras.map((l) => `<div class="lora"><span>${esc(l.name)} <small class="hint">${l.base === "realvis" ? "実写" : "イラスト"}</small></span><span class="pill ok">使用可</span></div>`),
  ];
  $("lora-list").innerHTML = rows.join("") || '<p class="hint">まだありません</p>';
  const cur = $("img-lora").value;
  $("img-lora").innerHTML = '<option value="">なし (ベースモデルのみ)</option>'
    + loras.map((l) => `<option value="${esc(l.name)}">${esc(l.name)}</option>`).join("");
  $("img-lora").value = loras.some((l) => l.name === cur) ? cur : "";
  $("img-base-row").classList.toggle("hide", !!$("img-lora").value);
}
$("img-lora").onchange = renderLoras;
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
  if (needModal("lora-status")) return;
  $("lora-go").disabled = true;
  try {
    status("lora-status", "画像を準備中…");
    const images = [];
    for (const [i, f] of files.entries()) {
      try { images.push({ name: `${i}.jpg`, b64: await blobB64(await toJpeg(f, 1536, 0.9)) }); }
      catch { images.push({ name: `${i}_${f.name}`, b64: await fileB64(f) }); } // HEIC 等はサーバー側で変換
    }
    status("lora-status", "送信中…");
    const base = seg("lora-base");
    const cls = $("lora-class").value.trim();
    const { call_id } = await modal.train({ name, base, caption: base === "realvis" ? "trigger" : "tags", cls, images });
    addJob(call_id, "train", name, { lora: { name, base } });
    status("lora-status", "✅ 学習を開始しました (目安 30〜60 分)。完成すると画像タブの LoRA 欄に出ます");
  } catch (e) { status("lora-status", "❌ " + e.message, true); }
  finally { $("lora-go").disabled = false; }
};

// ---------------------------------------------------------------- init
applyImgMode();
renderLoras();
renderJobs();
renderWorks();
if (settings.get().modal) { refreshLoras(); poll(); }
