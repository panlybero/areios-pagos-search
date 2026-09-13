# Single image for every role: API service, MCP server, and crawl/index jobs.
# Keeping one artifact means the thing you tested is the thing that runs.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APSEARCH_ENV=cloud

WORKDIR /app

# --- dependencies (cached independently of source) --------------------------
FROM base AS deps
COPY pyproject.toml README.md ./
COPY src/apsearch/__init__.py src/apsearch/__init__.py
# `api` and `mcp` extras are always installed; `embed` (onnxruntime, ~200MB) is
# only needed for the local embedding backend, so it is opt-in via build arg.
ARG EXTRAS="api,mcp"
RUN pip install --no-cache-dir ".[${EXTRAS}]"

# --- runtime ----------------------------------------------------------------
FROM deps AS runtime
COPY src/ src/
RUN pip install --no-cache-dir --no-deps -e .

# Cloud Run requires a non-root user for the second-generation execution env.
RUN useradd --create-home --uid 1000 apsearch \
    && mkdir -p /app/data && chown -R apsearch:apsearch /app
USER apsearch

# Cloud Run injects PORT; the config layer reads it.
ENV PORT=8080
EXPOSE 8080

# Default role is the REST API. Override `command` for the other roles:
#   Cloud Run Job (daily poll):  ["apsearch","crawl","poll"]
#   Cloud Run Job (indexing):    ["apsearch","index","build"]
#   MCP over HTTP:               ["apsearch","mcp","--transport","http"]
ENTRYPOINT ["apsearch"]
CMD ["serve"]
