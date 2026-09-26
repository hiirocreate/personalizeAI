// Kaggle 公式 API (kagglesdk と同じ RPC エンドポイント) クライアント
import { Filesystem, Directory } from "@capacitor/filesystem";
import { FileTransfer } from "@capacitor/file-transfer";

const API = "https://api.kaggle.com/v1";
const REPO = "https://github.com/hiirocreate/personalizeAI";

export const creds = {
  get() { try { return JSON.parse(localStorage.getItem("kaggle") || "{}"); } catch { return {}; } },
  set(v) { localStorage.setItem("kaggle", JSON.stringify(v)); },
};

function authHeader() {
  const { user, key } = creds.get();
  if (!user || !key) throw new Error("設定で Kaggle のユーザー名と API トークンを入力してください");
  // 新形式トークン (KGAT_...) は Bearer、旧形式 API キーは Basic
  return key.startsWith("KGAT_") ? `Bearer ${key}` : `Basic ${btoa(`${user}:${key}`)}`;
}

async function call(service, method, body = {}) {
  const r = await fetch(`${API}/${service}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: authHeader() },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`Kaggle API ${r.status}: ${text.slice(0, 300)}`);
  const data = text ? (typeof text === "string" ? JSON.parse(text) : text) : {};
  if (data.error) throw new Error(data.error);
  return data;
}

const K = "kernels.KernelsApiService";
const D = "datasets.DatasetApiService";
const secs = (d) => parseFloat(String(d || "0").replace("s", "")) || 0;

export async function quota() {
  const r = await call(K, "GetAcceleratorQuotaStatistics");
  const q = r.gpuQuota || {};
  return { used: secs(q.timeUsed) / 3600, total: secs(q.totalTimeAllowed) / 3600 };
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
