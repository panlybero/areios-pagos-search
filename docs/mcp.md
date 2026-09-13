# MCP server

`apsearch mcp` exposes the corpus to agents over the Model Context Protocol.
Verified against protocol revision `2025-11-25` with a real stdio client
handshake (see `tests/test_integration.py::TestMCPTools`).

## Tools

| Tool | Purpose |
| --- | --- |
| `search_decisions` | Hybrid/keyword/semantic search with year, category, chamber and theme filters. Returns metadata + matching passages. |
| `get_decision` | Full text by citation (`144/2015`) or opaque `cd`. |
| `list_themes` | The court's own controlled vocabulary (~713 subject headings) with live decision counts. |
| `corpus_stats` | Coverage and freshness of the index. |

## Local (stdio)

Claude Desktop — `claude_desktop_config.json`:

```jsonc
{
  "mcpServers": {
    "areios-pagos": {
      "command": "/abs/path/to/.venv/bin/apsearch",
      "args": ["mcp"],
      "env": {
        "APSEARCH_PG_HOST": "localhost",
        "APSEARCH_PG_PORT": "5433",
        "APSEARCH_PG_PASSWORD": "apsearch",
        "APSEARCH_EMBED_BACKEND": "gemini",
        "APSEARCH_EMBED_DIM": "768",
        "APSEARCH_GEMINI_API_KEY": "..."
      }
    }
  }
}
```

OpenCode — `opencode.json`:

```jsonc
{
  "mcp": {
    "areios-pagos": {
      "type": "local",
      "command": ["/abs/path/to/.venv/bin/apsearch", "mcp"],
      "environment": { "APSEARCH_PG_PORT": "5433" }
    }
  }
}
```

Use an absolute path to the executable: MCP clients do not run a login shell,
so `PATH` and `.bashrc` exports are not available. For the same reason, secrets
have to come through `env` (or a `.env` file next to the working directory)
rather than from the ambient environment.

**stdout is the protocol channel.** Anything printed to stdout corrupts the
JSON-RPC stream, which is why `apsearch.logging` writes to stderr only. Keep it
that way.

## Remote (streamable HTTP)

```bash
apsearch mcp --transport http --port 8080
```

This is the Cloud Run shape. Put it behind authentication before exposing it —
either Cloud Run IAM, or set `APSEARCH_API_KEY` and terminate at a proxy. The
MCP server itself does no authentication.

## Design notes

**Search returns passages, not documents.** A single decision runs to ~37k
characters; ten of them would exhaust most context windows. `search_decisions`
returns metadata plus the best-matching passages, and the agent calls
`get_decision` for the few that matter. `get_decision` also takes `max_chars`
and reports `body_truncated` so the agent knows when it is seeing a fragment.

**The controlled vocabulary is exposed deliberately.** Agents are poor at
guessing a domain's classification scheme. `list_themes` lets a model discover
that "Αδικοπραξία" and "Αδικοπραξία (αστική ευθύνη) Δημοσίου" are distinct
headings, then filter on them — which beats inventing Greek legal terminology.
Counts are computed live, because a stale ranking column would actively
mislead.

**Errors are returned, not raised.** A bad citation comes back as
`{"error": "..."}` with guidance on the expected format rather than a protocol
error, so the model can correct itself in the next turn instead of seeing an
opaque tool failure.

**The server instructions tell the model to query in Greek.** The corpus and
the lexical index are Greek; an English query only reaches the semantic
retriever and recall drops accordingly.
