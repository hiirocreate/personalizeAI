"""ComfyUI API クライアント。workflows/*.json テンプレートにパラメータを流し込んで実行する。"""
import copy
import json
import os
import random
import time
import urllib.parse
import urllib.request
import uuid

WORKFLOW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows")


def load_template(name):
    with open(os.path.join(WORKFLOW_DIR, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def build(template, **params):
    """テンプレートに値を適用。optional のパラメータが空ならノードを外して配線をつなぎ直す。"""
    wf = copy.deepcopy(template["workflow"])
    for key, (node, field) in template["params"].items():
        if params.get(key) not in (None, ""):
            wf[node]["inputs"][field] = params[key]
    for key, opt in template.get("optional", {}).items():
        if params.get(key) not in (None, ""):
            continue
        removed = wf.pop(opt["node"])
        for n in wf.values():
            for field, val in list(n["inputs"].items()):
                if isinstance(val, list) and val[0] == opt["node"]:
                    src = opt["passthrough"].get(str(val[1]))
                    if src:
                        n["inputs"][field] = removed["inputs"][src]
                    else:
                        del n["inputs"][field]
    return wf


class Comfy:
    def __init__(self, url="http://127.0.0.1:8188"):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex

    def _get(self, path):
        with urllib.request.urlopen(self.url + path) as r:
            return r.read()

    def queue(self, workflow):
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        req = urllib.request.Request(self.url + "/prompt", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.load(r)["prompt_id"]

    def upload_image(self, path):
        boundary = uuid.uuid4().hex
        name = os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(self.url + "/upload/image", body,
                                     {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req) as r:
            return json.load(r)["name"]

    def wait(self, prompt_id, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = json.loads(self._get(f"/history/{prompt_id}"))
            if prompt_id in hist:
                entry = hist[prompt_id]
                if entry.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(json.dumps(entry["status"], ensure_ascii=False)[:2000])
                files = []
                for out in entry["outputs"].values():
                    for items in out.values():
                        if isinstance(items, list):
                            files += [i for i in items if isinstance(i, dict) and "filename" in i]
                return files
            time.sleep(1)
        raise TimeoutError(prompt_id)

    def download(self, file, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        q = urllib.parse.urlencode({k: file.get(k, "") for k in ("filename", "subfolder", "type")})
        dst = os.path.join(out_dir, file["filename"])
        with open(dst, "wb") as f:
            f.write(self._get(f"/view?{q}"))
        return dst

    def run(self, name, out_dir="outputs", **params):
        tpl = load_template(name)
        if params.get("seed") in (None, "", -1):
            params["seed"] = random.randint(0, 2**31 - 1)
        if params.get("image") and os.path.exists(params["image"]):
            params["image"] = self.upload_image(params["image"])
        files = self.wait(self.queue(build(tpl, **params)))
        return [self.download(f, out_dir) for f in files]
