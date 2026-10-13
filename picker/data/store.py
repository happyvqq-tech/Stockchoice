"""Parquet 儲存層：本機資料夾（LocalStore）或 Supabase Storage 私人 bucket（SupabaseStore）。
鍵（key）長得像 "bars/2330.parquet"。所有資料不進 git。研究用途，非投資建議。"""
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)

    def _p(self, key):
        return self.root / key

    def exists(self, key):
        return self._p(key).exists()

    def read(self, key):
        return pd.read_parquet(self._p(key))

    def write(self, key, df):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)

    def list(self, prefix):
        base = self._p(prefix)
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)).replace(os.sep, "/") for p in base.rglob("*") if p.is_file())


class SupabaseStore:
    """用 Storage REST API，不依賴 supabase-py。URL 結尾若有 /rest/v1 會自動去掉。"""

    def __init__(self, url, secret_key, bucket="picker-data"):
        p = urllib.parse.urlparse(url.strip())
        self.base = f"{p.scheme}://{p.netloc}/storage/v1"
        self.bucket = bucket
        self.h = {"apikey": secret_key, "Authorization": f"Bearer {secret_key}"}

    def _req(self, method, path, data=None, headers=None, timeout=120):
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={**self.h, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def _obj(self, key):
        return f"/object/{self.bucket}/{urllib.parse.quote(key)}"

    def exists(self, key):
        code, _ = self._req("HEAD", self._obj(key))
        return code == 200

    def read(self, key):
        code, body = self._req("GET", self._obj(key))
        if code != 200:
            raise FileNotFoundError(f"{key}: HTTP {code}")
        return pd.read_parquet(io.BytesIO(body))

    def write(self, key, df):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        code, body = self._req("POST", self._obj(key), buf.getvalue(),
                               {"Content-Type": "application/octet-stream", "x-upsert": "true"})
        if code not in (200, 201):
            raise RuntimeError(f"upload {key} failed: HTTP {code} {body[:200]!r}")

    def list(self, prefix):
        out, offset = [], 0
        while True:
            payload = json.dumps({"prefix": prefix.rstrip("/"), "limit": 1000, "offset": offset}).encode()
            code, body = self._req("POST", f"/object/list/{self.bucket}", payload, {"Content-Type": "application/json"})
            if code != 200:
                raise RuntimeError(f"list {prefix} failed: HTTP {code} {body[:200]!r}")
            rows = json.loads(body)
            out += [f"{prefix.rstrip('/')}/{r['name']}" for r in rows if r.get("id")]
            if len(rows) < 1000:
                return sorted(out)
            offset += 1000


def get_store(cfg=None):
    """有設定 Supabase 環境變數就用雲端，否則用本機資料夾。"""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SECRET_KEY")
    if url and key:
        return SupabaseStore(url, key, (cfg or {}).get("storage", {}).get("bucket", "picker-data"))
    return LocalStore((cfg or {}).get("paths", {}).get("data_store", "data_store"))
