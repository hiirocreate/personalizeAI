// Kaggle 公式 API (kagglesdk と同じ RPC エンドポイント) クライアント
import { Filesystem, Directory } from "@capacitor/filesystem";
import { FileTransfer } from "@capacitor/file-transfer";

const API = "https://api.kaggle.com/v1";
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
    const r = await fetch(`${API}/${service}/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "User-Agent": "kaggle-api/v1.7.0", Authorization: header(mode) },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    if (r.status === 401 || r.status === 403) { // もう一方の認証方式で再試行
      tried.push(`${mode}: ${r.status} ${String(text).slice(0, 120)}`);
      continue;
    }
    if (!r.ok) throw new Error(`Kaggle API ${r.status}: ${String(text).slice(0, 300)}`);
    if (c.mode !== mode) creds.set({ ...creds.get(), mode });
    const data = text ? (typeof text === "string" ? JSON.parse(text) : text) : {};
    if (data.error) throw new Error(typeof data.error === "string" ? data.error : JSON.stringify(data.error));
    return data;
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
  const r = await fetch(`${API}/security.OAuthService/IntrospectToken`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }),
  });
  const text = await r.text();
  try { return { status: r.status, ...(typeof text === "string" ? JSON.parse(text) : text) }; }
  catch { return { status: r.status, raw: String(text).slice(0, 120) }; }
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
const D = "datasets.DatasetApiService";
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

/** ジョブ用カーネルを作成して実行開始。slug を返す。 */
export async function pushJob(kind, job, { files = {}, kernels = [], datasets = [] } = {}) {
  const { user } = creds.get();
  const slug = `pai-${kind}-${Date.now().toString(36)}`;
  const r = await call(K, "SaveKernel", {
    slug: `${user}/${slug}`,
    newTitle: slug,
    text: script(job, files),
    language: "python",
    kernelType: "script",
    isPrivate: true,
    enableGpu: true,
    enableInternet: true,
    machineShape: "NvidiaTeslaT4",
    datasetDataSources: datasets,
    kernelDataSources: kernels,
    competitionDataSources: [],
    modelDataSources: [],
    categoryIds: [],
  });
  if (r.invalidKernelSources?.length || r.invalidDatasetSources?.length)
    throw new Error("入力データが見つかりません: " + [...(r.invalidKernelSources || []), ...(r.invalidDatasetSources || [])].join(", "));
  return slug;
}

/** "QUEUED" | "RUNNING" | "COMPLETE" | "ERROR" | "CANCEL_*" */
export async function status(slug) {
  const { user } = creds.get();
  const r = await call(K, "GetKernelSessionStatus", { userName: user, kernelSlug: slug });
  return { status: r.status || "QUEUED", failure: r.failureMessage || "" };
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

/** 端末内のファイル (Filesystem の URI) を非公開データセットとしてアップロード。slug を返す。 */
export async function uploadDataset(fileUri, size, onStep) {
  const { user } = creds.get();
  onStep?.("アップロード準備中…");
  const up = await call("blobs.BlobApiService", "StartBlobUpload", {
    type: "DATASET", name: "images.zip", contentType: "application/zip",
    contentLength: size, lastModifiedEpochSeconds: Math.floor(Date.now() / 1000),
  });
  onStep?.("画像をアップロード中…");
  await FileTransfer.uploadFile({
    url: up.createUrl, path: fileUri, method: "PUT", chunkedMode: false,
    headers: { "Content-Type": "application/zip" },
  });
  const slug = `pai-ds-${Date.now().toString(36)}`;
  const r = await call(D, "CreateDataset", {
    ownerSlug: user, slug, title: slug, licenseName: "CC0-1.0", isPrivate: true, files: [{ token: up.token }],
  });
  if (r.error) throw new Error(r.error);
  onStep?.("Kaggle 側で処理中…");
  for (let i = 0; i < 90; i++) {
    const s = await call(D, "GetDatasetStatus", { ownerSlug: user, datasetSlug: slug });
    if (s.status === "READY") return slug;
    if (s.status === "FAILED") throw new Error("データセットの作成に失敗しました");
    await new Promise((res) => setTimeout(res, 5000));
  }
  throw new Error("データセットの準備がタイムアウトしました");
}

export async function deleteDataset(slug) {
  const { user } = creds.get();
  try { await call(D, "DeleteDataset", { ownerSlug: user, datasetSlug: slug }); } catch { /* 無視 */ }
}

/** 出力ファイルを端末のアプリ領域に保存し URI を返す */
export async function download(url, name) {
  const { uri } = await Filesystem.getUri({ directory: Directory.Data, path: `works/${name}` });
  await Filesystem.mkdir({ directory: Directory.Data, path: "works", recursive: true }).catch(() => {});
  await FileTransfer.downloadFile({ url, path: uri });
  return uri;
}

export const userName = () => creds.get().user;
