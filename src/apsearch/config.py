"""Central configuration.

Every value is overridable from the environment (``APSEARCH_*``), so the same
code runs unchanged on a laptop, a VM, or a CI runner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def user_config_path() -> Path:
    return Path.home() / ".apsearch" / "config.json"


def load_user_config() -> dict:
    p = user_config_path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_user_config(updates: dict) -> None:
    p = user_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_user_config()
    cfg.update(updates)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_env_files() -> tuple[Path, ...]:
    """Find .env files regardless of the caller's current working directory.

    When spawned via SSH or outside the repository root (e.g. from an agent or
    OpenWork whose cwd is $HOME), a relative ".env" would fail to find the
    configuration and fall back to defaults. We look in:
      1. Repo root (.env next to pyproject.toml)
      2. ~/.config/apsearch/.env
      3. Current working directory .env (highest precedence)
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / ".env",
        Path.home() / ".config" / "apsearch" / ".env",
        Path.cwd() / ".env",
    ]
    unique: list[Path] = []
    for c in candidates:
        if c.is_file() and c.resolve() not in [u.resolve() for u in unique]:
            unique.append(c)
    return tuple(unique) if unique else (Path(".env"),)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_resolve_env_files(), env_prefix="APSEARCH_", extra="ignore"
    )

    env: str = "local"  # local | cloud
    log_level: str = "INFO"
    #: Emit structured JSON logs (what Cloud Logging wants). Auto-on when
    #: K_SERVICE / CLOUD_RUN_JOB is present.
    log_json: bool | None = None

    # ---------------------------------------------------------------- database
    #: Single-file SQLite database. sqlite-vec provides vector search, FTS5 the
    #: lexical index: no daemon, no ports, no separate setup.
    sqlite_path: str = "data/areios_pagos.db"

    @property
    def sqlite_file(self) -> Path:
        p = Path(self.sqlite_path)
        if p.is_absolute():
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        # If running inside or next to the repo directory, use repo's data/
        repo_data = Path(__file__).resolve().parents[2] / self.sqlite_path
        if repo_data.parent.is_dir():
            repo_data.parent.mkdir(parents=True, exist_ok=True)
            return repo_data
        # Otherwise use standard user data directory
        user_dir = Path.home() / ".apsearch"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / p.name

    # ----------------------------------------------------------------- crawler
    base_url: str = "https://www.areiospagos.gr"
    user_agent: str = Field(
        default="apsearch/0.1 (+https://github.com/; open-source Areios Pagos case-law index)"
    )
    #: Strongly recommended: lets the site's admins reach you instead of
    #: silently blocking. Appended to the User-Agent when set.
    contact_email: str = ""

    #: Seconds between consecutive requests (~0.22s => ~4.5 req/s).
    crawl_delay: float = 0.22
    crawl_concurrency: int = 3
    request_timeout: float = 90.0
    max_retries: int = 4
    #: Optional hard ceiling on requests per run, as a runaway-job safety net.
    max_requests_per_run: int = 0  # 0 = unlimited

    # ------------------------------------------------------------ cache / blob
    #: "local" or "gcs". GCS keeps the raw HTML archive durable across the
    #: ephemeral filesystems of Cloud Run jobs.
    cache_backend: str = "local"
    cache_dir: str = "data/http_cache"
    cache_bucket: str = ""
    cache_prefix: str = "http_cache"
    cache_enabled: bool = True

    model_cache_dir: str = "data/models"

    # --------------------------------------------------------------- embedding
    #: Which backend produces vectors: "gemini" (hosted, fast, best quality on
    #: Greek) or "fastembed" (local ONNX, fully open source, no API key).
    #: The rest of the stack -- SQLite, sqlite-vec, FTS5, fusion -- is open
    #: source either way, so this is a swappable component, not a lock-in.
    #:
    #: Defaults to "gemini"/768 because that is what the standalone packaged
    #: app (PyInstaller build, no .env file, no bundled ONNX runtime) actually
    #: ships and what the pre-seeded database's chunk_vec table was built with.
    embed_backend: str = "gemini"
    embed_dim: int = 768
    embed_batch_size: int = 32

    # -- local ONNX backend --
    embed_model: str = "intfloat/multilingual-e5-small"
    #: e5 models are asymmetric and need these prefixes; "" for other models.
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "
    embed_threads: int = 0  # 0 => onnxruntime default

    # -- Gemini backend --
    #: NEVER commit this. Locally: .env (gitignored). On GCP: Secret Manager
    #: mounted as an env var on the Cloud Run service/job.
    gemini_api_key: str = Field(
        default_factory=lambda: (
            load_user_config().get("gemini_api_key")
            or os.environ.get("APSEARCH_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
    )
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_concurrency: int = 2
    gemini_rpm: int = 0          # 0 = no client-side cap; raise if you hit 429s
    gemini_timeout: float = 120.0
    gemini_max_retries: int = 8

    # ---------------------------------------------------------------- chunking
    chunk_target_chars: int = 1400
    chunk_overlap_chars: int = 200

    # ------------------------------------------------------------------ search
    default_limit: int = 20
    rrf_k: int = 60
    candidate_pool: int = 200

    # ------------------------------------------------------------ query cache
    query_cache_enabled: bool = True
    #: Max number of cached query vectors to retain.
    #: 20,000 queries ≈ 65 MB. An LRU prune runs whenever this cap is reached.
    query_cache_max_entries: int = 20000
    #: Auto-expire cached queries not accessed within this many days.
    query_cache_ttl_days: int = 60
    #: Maximum query character length allowed (prevents prompt-stuffing abuse).
    max_query_chars: int = 600

    # -------------------------------------------------------------------- http
    #: Cloud Run injects PORT; honour it.
    port: int = Field(default_factory=lambda: int(os.environ.get("PORT", "8080")))
    host: str = "0.0.0.0"
    #: Optional shared-secret for the public API/MCP endpoints.
    api_key: str = ""

    @property
    def is_cloud_run(self) -> bool:
        return bool(os.environ.get("K_SERVICE") or os.environ.get("CLOUD_RUN_JOB"))

    @property
    def use_json_logs(self) -> bool:
        if self.log_json is not None:
            return self.log_json
        return self.is_cloud_run


settings = Settings()
