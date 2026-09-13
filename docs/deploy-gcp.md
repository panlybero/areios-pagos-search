# Deploying on GCP

Not deployed yet. This records the intended shape so the code stays compatible
with it — every choice below is already reflected in the configuration layer.

## Component mapping

| Local                      | GCP                                                    |
| -------------------------- | ------------------------------------------------------ |
| `docker compose` Postgres  | Cloud SQL for PostgreSQL 16+ (`pgvector` extension)     |
| `apsearch serve`           | Cloud Run **service** (autoscaling, scale-to-zero)      |
| `apsearch mcp --transport http` | Cloud Run **service**                             |
| `apsearch crawl poll`      | Cloud Run **job** + Cloud Scheduler (daily)             |
| `apsearch index build`     | Cloud Run **job** (after each poll)                     |
| `data/http_cache`          | GCS bucket (`APSEARCH_CACHE_BACKEND=gcs`)               |
| `.env`                     | Secret Manager, mounted as env vars                     |

## Database

Cloud SQL is reachable over a unix socket that Cloud Run mounts for you, so no
proxy sidecar and no IP allow-listing:

```
APSEARCH_CLOUDSQL_INSTANCE=PROJECT:REGION:INSTANCE
APSEARCH_PG_USER=apsearch
APSEARCH_PG_DATABASE=apsearch
APSEARCH_PG_PASSWORD=<from Secret Manager>
```

`Settings.dsn` detects `cloudsql_instance` and builds
`postgresql://user:pw@/db?host=/cloudsql/INSTANCE` automatically.

Enable the extensions once, as the `apsearch` user:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

Then `apsearch db migrate` as a one-off job.

> **Constraint that shaped the design:** Cloud SQL gives no filesystem access,
> so a custom Greek stopword file cannot be installed. Stopword handling is
> therefore done query-side in `apsearch/search/query.py`, which works on any
> managed Postgres. Don't "fix" this by adding a `.stop` file — it won't deploy.

## Secrets

The Gemini API key must never be in the image or the repo.

```bash
gcloud secrets create apsearch-gemini-key --data-file=-   # paste key, Ctrl-D

gcloud run services update apsearch-api \
  --set-secrets=APSEARCH_GEMINI_API_KEY=apsearch-gemini-key:latest
```

Grant the runtime service account `roles/secretmanager.secretAccessor`, plus
`roles/cloudsql.client` and (if using the GCS cache) `roles/storage.objectAdmin`
on the bucket only.

## Scheduled crawl

The daily poll is deliberately small: it re-enumerates the current and previous
year (~10 requests) and fetches only decisions it has not seen.

```bash
gcloud run jobs create apsearch-poll \
  --image=REGION-docker.pkg.dev/PROJECT/apsearch/apsearch:latest \
  --command=apsearch --args=crawl,poll \
  --set-cloudsql-instances=PROJECT:REGION:INSTANCE \
  --max-retries=1 --task-timeout=3600s

gcloud scheduler jobs create http apsearch-poll-daily \
  --schedule="30 4 * * *" --time-zone="Europe/Athens" \
  --uri="https://REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/PROJECT/jobs/apsearch-poll:run" \
  --oauth-service-account-email=SA@PROJECT.iam.gserviceaccount.com
```

Run `apsearch index build` as a second job after the poll.

Set `APSEARCH_MAX_REQUESTS_PER_RUN` (e.g. `2000`) on scheduled jobs. A bug in
partition-splitting logic could otherwise turn a daily poll into a crawl of the
entire site; the budget makes that failure loud and bounded instead.

## Sizing

- **Cloud SQL**: the corpus is ~66k decisions / ~1.3M chunks. At 768 dims that
  is ~4 GB of vectors plus ~2 GB of text and FTS indexes. Start at 2 vCPU / 8 GB
  with 50 GB SSD. HNSW index build wants `maintenance_work_mem` raised.
- **Cloud Run API**: 1 vCPU / 512 MB is plenty — it only issues SQL and one
  embedding call per query. Set `--min-instances=1` if p99 latency matters, as
  cold starts dominate otherwise.
- **Index job**: 2 vCPU / 2 GB. Throughput is bounded by the Gemini API, not CPU.

## Backfill cost

Embedding the full corpus is a one-time ~700M tokens. Check current
`gemini-embedding` pricing before running it; at $0.15/1M that is roughly $100.
Reduce it by lowering `APSEARCH_CHUNK_OVERLAP_CHARS` or by indexing only the
headnote plus the `ΣΚΕΦΘΗΚΕ` reasoning section.

The local backend (`APSEARCH_EMBED_BACKEND=fastembed`) costs nothing but runs at
~4 chunks/s per vCPU, so the same backfill is a multi-day job. On an AVX-512
machine (N2/C3) the int8 model is several times faster — build the image with
`--build-arg EXTRAS=api,mcp,embed` and set
`APSEARCH_EMBED_MODEL=intfloat/multilingual-e5-small-int8`.
