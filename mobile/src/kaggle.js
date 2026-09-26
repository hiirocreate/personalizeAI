// Kaggle 公式 API クライアント (公式ドキュメントの www.kaggle.com/api/v1 を優先し、RPC 形式にもフォールバック)
import { Filesystem, Directory } from "@capacitor/filesystem";
import { FileTransfer } from "@capacitor/file-transfer";

const API = "https://api.kaggle.com/v1";
const WWW = "https://www.kaggle.com/api/v1";
// RPC 名 → REST (kagglesdk の各リクエスト型に定義された endpoint/method)
const ROUTES = {
  "kernels.KernelsApiService/ListKernels": ["GET", "/kernels/list"],
  "kernels.KernelsApiService/GetAcceleratorQuotaStatistics": ["GET", "/kernels/quota"],
  "kernels.KernelsApiService/SaveKernel": ["POST", "/kernels/push"],
  "kernels.KernelsApiService/GetKernelSessionStatus": ["GET", "/kernels/status"],
  "kernels.KernelsApiService/ListKernelSessionOutput": ["GET", "/kernels/output"],
  "kernels.KernelsApiService/DeleteKernel": ["POST", "/kernels/delete/{userName}/{kernelSlug}"],
  "blobs.BlobApiService/StartBlobUpload": ["POST", "/blobs/upload"],
  "datasets.DatasetApiService/CreateDataset": ["POST", "/datasets/create/new"],
  "datasets.DatasetApiService/GetDatasetStatus": ["GET", "/datasets/status/{ownerSlug}/{datasetSlug}"],
  "datasets.DatasetApiService/DeleteDataset": ["POST", "/dataset/{ownerSlug}/{datasetSlug}/delete"],
  "security.OAuthService/IntrospectToken": ["POST", "/oauth2/introspect"],
};

function restRequest(service, method, body) {
  const route = ROUTES[`${service}/${method}`];
  if (!route) return null;
  const params = { ...body };
  const path = route[1].replace(/\{(\w+)\}/g, (_, k) => { const v = params[k]; delete params[k]; return encodeURIComponent(v ?? ""); });
  if (route[0] === "GET") {
    const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== ""));
    return { url: `${WWW}${path}${q.toString() ? "?" + q : ""}`, init: { method: "GET" } };
  }
  return { url: `${WWW}${path}`, init: { method: "POST", body: JSON.stringify(params) } };
}

function parseJson(text) {
  if (typeof text !== "string") return text;
  if (!text) return {};
  const d = JSON.parse(text); // HTML 等なら例外
  return typeof d === "string" ? { status: d } : d;
}
const REPO = "https://github.com/hiirocreate/personalizeAI";

export const creds = {
  get() { try { return JSON.parse(localStorage.getItem("kaggle") || "{}"); } catch { return {}; } },
  set(v) { localStorage.setItem("kaggle", JSON.stringify(v)); },
};

function parseKey(raw) {
  // kaggle.json の中身をそのまま貼った場合にも対応
  try { const j = JSON.parse(raw); if (j.key) return { key: j.key, user: j.username }; } catch { /* 文字列 */ }
  return { key: raw.replace(/\s+/g, "") }; // コピー時に混ざる改行・空白を除去
}

function header(mode) {
  const { user, key } = creds.get();
  // 新形式 API トークンは Bearer、旧形式 (kaggle.json の 32 桁 key) は Basic
  return mode === "basic" ? `Basic ${btoa(`${user}:${key}`)}` : `Bearer ${key}`;
}

async function call(service, method, body = {}) {
  const c = creds.get();
  if (!c.key) throw new Error("⚙️ 設定で Kaggle の API トークンを入力してください");
  const first = c.mode || (/^[0-9a-f]{32}$/i.test(c.key) ? "basic" : "bearer");
  const tried = [];
  for (const mode of [first, first === "basic" ? "bearer" : "basic"]) {
    if (mode === "basic" && !c.user) { tried.push("basic: ユーザー名なしのため省略"); continue; }
    const headers = { "Content-Type": "application/json", "User-Agent": "kaggle-api/v1.7.0", Authorization: header(mode) };
    const rest = restRequest(service, method, body);
    const attempts = [
      ...(rest ? [["www", rest.url, rest.init]] : []),
      ["rpc", `${API}/${service}/${method}`, { method: "POST", body: JSON.stringify(body) }],
    ];
    let unauthorized = false;
    for (const [label, url, init] of attempts) {
      const r = await fetch(url, { ...init, headers });
      const text = await r.text();
      let data;
      try { data = parseJson(text); } catch { tried.push(`${mode}/${label}: ${r.status} (JSON 以外)`); continue; }
      const denied = String(text).match(/Permission .(\w[\w.]*). was denied/);
      if (r.status === 403 && denied) {
        throw new Error(`Kaggle のトークンに「${denied[1]}」の権限がありません。Kaggle の設定ページ → API でトークンを作り直す際に、`
          + "ノートブック (kernels) の作成・実行・閲覧・削除の権限を付けてください。");
      }
      if (r.status === 401 || r.status === 403) { tried.push(`${mode}/${label}: ${r.status} ${String(text).slice(0, 100)}`); unauthorized = true; continue; }
      if (r.status === 404 && label === "www") { tried.push(`${mode}/www: 404`); continue; }
      if (!r.ok) throw new Error(`Kaggle API ${r.status}: ${String(text).slice(0, 300)}`);
      if (c.mode !== mode) creds.set({ ...creds.get(), mode });
      if (data.error) throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data.error));
      return data;
    }
    if (!unauthorized) break;
  }
  // 送信したヘッダーが途中で落ちていないかを、エコーサービスで確認 (トークン本体は送らない)
  let echo = "";
  try {
    const r = await fetch("https://httpbin.org/anything", {
      method: "POST",
      headers: { "Content-Type": "application/json", "User-Agent": "kaggle-api/v1.7.0", Authorization: "Bearer KGAT_test" },
      body: "{}",
    });
    const t = await r.text();
    const h = (typeof t === "string" ? JSON.parse(t) : t).headers || {};
    echo = `\necho: Authorization=${h.Authorization || h.authorization || "(届いていない)"} UA=${(h["User-Agent"] || "").slice(0, 40)}`;
  } catch (e) { echo = `\necho: ${e.message}`; }
  throw new Error("認証に失敗しました。Kaggle の設定ページ → API → Create New Token で作ったトークンを貼り直してください"
    + (c.user ? "" : " (旧形式のキーの場合はユーザー名も必要)")
    + `\n[診断] トークン長 ${c.key.length} 文字 / 先頭 ${c.key.slice(0, 5)}… / ユーザー名 ${c.user || "(未入力)"}\n` + tried.join("\n") + echo);
}

/** トークンの有効性を Kaggle に問い合わせる (認証ヘッダー不要の公式エンドポイント) */
async function introspect(token) {
  let last = {};
  for (const url of [`${WWW}/oauth2/introspect`, `${API}/security.OAuthService/IntrospectToken`]) {
    try {
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
      last = { status: r.status, ...parseJson(await r.text()) };
      if ("active" in last) return last;
    } catch (e) { last = { error: e.message }; }
  }
  return last;
}

/** 入力されたユーザー名/トークンを保存し、トークンから正しいユーザー名を確認する */
export async function login(userInput, keyInput) {
  const p = parseKey(keyInput);
  creds.set({ user: (p.user || userInput || "").trim(), key: p.key });
  if (!/^[0-9a-f]{32}$/i.test(p.key)) { // 新形式トークン
    const info = await introspect(p.key);
    if (!info.active) {
      throw new Error("このトークンは Kaggle 側で「無効」と判定されました (コピーの欠け・失効・別アカウントの可能性)。"
        + "Kaggle の設定ページ → API で新しいトークンを作り、表示された直後にコピーして貼ってください。"
        + `\n[診断] 長さ ${p.key.length} / 先頭 ${p.key.slice(0, 5)}… / 応答 ${info.status} ${JSON.stringify(info).slice(0, 150)}`);
    }
    creds.set({ ...creds.get(), user: info.username || creds.get().user, mode: "bearer" });
  }
  if (!creds.get().user) throw new Error("ユーザー名を入力してください");
  return creds.get().user;
}

const K = "kernels.KernelsApiService";
const secs = (d) => parseFloat(String(d || "0").replace("s", "")) || 0;

/** 接続テスト: 自分のノートブック一覧 (公式 CLI の `kaggle kernels list -m` と同じ API) を取得 */
export async function ping() {
  await call(K, "ListKernels", { user: creds.get().user, pageSize: 1, page: 1 });
}

/** GPU 残り時間 (API トークンでは取得できない場合があるため失敗時は null) */
export async function quota() {
  try {
    const r = await call(K, "GetAcceleratorQuotaStatistics");
    const q = r.gpuQuota || {};
    return { used: secs(q.timeUsed) / 3600, total: secs(q.totalTimeAllowed) / 3600 };
  } catch { return null; }
}

export function script(job, files = {}) {
  return `import base64, json, os, subprocess, sys
JOB = json.loads(${JSON.stringify(JSON.stringify(job))})
FILES = ${JSON.stringify(files)}
os.environ.update(PAI_WORK="/tmp/pai", PAI_DATA="/tmp/pai_data", PAI_OUT="/kaggle/working")
os.makedirs("/tmp/pai_in", exist_ok=True)
for name, b64 in FILES.items():
    open(f"/tmp/pai_in/{name}", "wb").write(base64.b64decode(b64))
subprocess.run("git clone -q --depth 1 ${REPO} /tmp/pai_repo", shell=True, check=True)
sys.path.insert(0, "/tmp/pai_repo")
from pai import job
job.main(JOB)
`;
}

async function saveKernel(slug, text, { gpu = true, kernels = [] } = {}) {
  const { user } = creds.get();
  const r = await call(K, "SaveKernel", {
    slug: `${user}/${slug}`,
    newTitle: slug,
    text,
    language: "python",
    kernelType: "script",
    isPrivate: true,
    enableGpu: gpu,
    enableInternet: gpu,
    ...(gpu ? { machineShape: "NvidiaTeslaT4" } : {}),
    datasetDataSources: [],
    kernelDataSources: kernels,
    competitionDataSources: [],
    modelDataSources: [],
    categoryIds: [],
  });
  if (r.invalidKernelSources?.length)
    throw new Error("入力データが見つかりません: " + r.invalidKernelSources.join(", "));
  return slug;
}

/** ジョブ用カーネルを作成して実行開始。slug を返す。kernels は入力にする他カーネル ("user/slug") */
export async function pushJob(kind, job, { files = {}, kernels = [] } = {}) {
  return saveKernel(`pai-${kind}-${Date.now().toString(36)}`, script(job, files), { kernels });
}

/**
 * 画像などを「出力ファイルとして保存するだけ」の CPU カーネルに載せて送る
 * (データセット権限が不要)。後続ジョブはこのカーネルの出力を入力として読む。
 */
export async function pushFiles(files) {
  const text = `import base64, json\nFILES = ${JSON.stringify(files)}\n`
    + `for name, b64 in FILES.items():\n    open(f"/kaggle/working/{name}", "wb").write(base64.b64decode(b64))\n`;
  return saveKernel(`pai-data-${Date.now().toString(36)}${Math.floor(Math.random() * 1e4)}`, text, { gpu: false });
}

/** "QUEUED" | "RUNNING" | "COMPLETE" | "ERROR" | "CANCEL_*" */
export async function status(slug) {
  const { user } = creds.get();
  const r = await call(K, "GetKernelSessionStatus", { userName: user, kernelSlug: slug });
  return { status: String(r.status || "QUEUED").toUpperCase(), failure: r.failureMessage || "" };
}

export async function outputs(slug) {
  const { user } = creds.get();
  const r = await call(K, "ListKernelSessionOutput", { userName: user, kernelSlug: slug, pageSize: 100 });
  return { files: r.files || [], log: r.log || "" };
}

export async function deleteKernel(slug) {
  const { user } = creds.get();
  try { await call(K, "DeleteKernel", { userName: user, kernelSlug: slug }); } catch { /* 片付け失敗は無視 */ }
}

/** 出力ファイルを端末のアプリ領域に保存し URI を返す */
export async function download(url, name) {
  const { uri } = await Filesystem.getUri({ directory: Directory.Data, path: `works/${name}` });
  await Filesystem.mkdir({ directory: Directory.Data, path: "works", recursive: true }).catch(() => {});
  await FileTransfer.downloadFile({ url, path: uri });
  return uri;
}

export const userName = () => creds.get().user;
