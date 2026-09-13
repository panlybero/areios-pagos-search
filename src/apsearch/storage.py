"""Blob storage for the raw-HTML archive.

Keeping every fetched page means parser bugs can be fixed and replayed without
ever touching the origin again -- the single most effective politeness measure
available. Local disk is the default; GCS is used when the process runs
somewhere with an ephemeral filesystem (Cloud Run jobs).

``google-cloud-storage`` is imported lazily so it stays an optional dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from apsearch.config import settings


class BlobStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalBlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by prefix; a flat dir with ~100k entries is miserable to work with.
        return self.root / key[:2] / key[2:4] / key

    def get(self, key: str) -> str | None:
        p = self._path(key)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def put(self, key: str, value: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(p)  # atomic; concurrent writers can't tear a file

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class GCSBlobStore:
    def __init__(self, bucket: str, prefix: str = ""):
        from google.cloud import storage  # lazy: optional dependency

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self.prefix = prefix.strip("/")

    def _blob(self, key: str):
        name = f"{self.prefix}/{key[:2]}/{key[2:4]}/{key}" if self.prefix else key
        return self._bucket.blob(name)

    def get(self, key: str) -> str | None:
        blob = self._blob(key)
        if not blob.exists():
            return None
        return blob.download_as_text(encoding="utf-8")

    def put(self, key: str, value: str) -> None:
        self._blob(key).upload_from_string(value, content_type="text/html; charset=utf-8")

    def exists(self, key: str) -> bool:
        return self._blob(key).exists()


class NullBlobStore:
    def get(self, key: str) -> str | None:
        return None

    def put(self, key: str, value: str) -> None:
        return None

    def exists(self, key: str) -> bool:
        return False


def get_blob_store() -> BlobStore:
    if not settings.cache_enabled:
        return NullBlobStore()
    backend = settings.cache_backend.lower()
    if backend == "gcs":
        if not settings.cache_bucket:
            raise ValueError("APSEARCH_CACHE_BACKEND=gcs requires APSEARCH_CACHE_BUCKET")
        return GCSBlobStore(settings.cache_bucket, settings.cache_prefix)
    if backend == "local":
        return LocalBlobStore(settings.cache_dir)
    if backend in ("none", "null", "off"):
        return NullBlobStore()
    raise ValueError(f"unknown cache backend: {settings.cache_backend!r}")
