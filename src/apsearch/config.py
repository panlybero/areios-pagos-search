"""Central configuration.

Strictly 12-factor: every value is overridable from the environment, so the
same image runs unchanged on a laptop, a VM, or Cloud Run. Nothing here reaches
for a GCP SDK -- the cloud-specific bits are isolated behind
``APSEARCH_CACHE_BACKEND`` and the Cloud SQL socket handling below.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    pg_host: str = "localhost"
    pg_port: int = 5433
    pg_user: str = "apsearch"
    pg_password: str = "apsearch"
    pg_database: str = "apsearch"
    pg_sslmode: str = ""
    #: Cloud SQL instance connection name, e.g. "proj:europe-west3:apsearch".
    #: When set, connect over the /cloudsql unix socket that Cloud Run mounts,
    #: which needs no proxy sidecar and no IP allow-listing.
    cloudsql_instance: str = ""
    #: Escape hatch: a fully-formed DSN wins over everything above.
    database_url: str = ""

    @property
    def dsn(self) -> str:
        if self.database_url:
            return self.database_url
        user = quote(self.pg_user, safe="")
        pwd = quote(self.pg_password, safe="")
        if self.cloudsql_instance:
            socket_dir = os.environ.get("CLOUD_SQL_SOCKET_DIR", "/cloudsql")
            host = quote(f"{socket_dir}/{self.cloudsql_instance}", safe="")
            return f"postgresql://{user}:{pwd}@/{self.pg_database}?host={host}"
        dsn = f"postgresql://{user}:{pwd}@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        if self.pg_sslmode:
            dsn += f"?sslmode={self.pg_sslmode}"
        return dsn

    pool_min_size: int = 1
    pool_max_size: int = 8

    # ----------------------------------------------------------------- crawler
    base_url: str = "https://www.areiospagos.gr"
    user_agent: str = Field(
        default="apsearch/0.1 (+https://github.com/; open-source Areios Pagos case-law index)"
    )
    #: Strongly recommended: lets the site's admins reach you instead of
    #: silently blocking. Appended to the User-Agent when set.
    contact_email: str = ""

    #: Seconds between consecutive requests. The origin is one IIS box run by a
    #: public institution; 2s => 0.5 req/s is deliberately slow.
    crawl_delay: float = 2.0
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
    #: The rest of the stack -- Postgres, pgvector, FTS, fusion -- is open
    #: source either way, so this is a swappable component, not a lock-in.
    embed_backend: str = "fastembed"
    embed_dim: int = 384
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
    gemini_api_key: str = ""
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_concurrency: int = 4
    gemini_rpm: int = 0          # 0 = no client-side cap; raise if you hit 429s
    gemini_timeout: float = 120.0
    gemini_max_retries: int = 5

    # ---------------------------------------------------------------- chunking
    chunk_target_chars: int = 1400
    chunk_overlap_chars: int = 200

    # ------------------------------------------------------------------ search
    default_limit: int = 20
    rrf_k: int = 60
    candidate_pool: int = 200

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
