"""Entrypoint for standalone packaged application.

* Double-clicking the app launches the Web UI and opens the browser.
* Running with arguments (e.g. `apsearch mcp` or `apsearch search ...`)
  dispatches to the CLI / MCP server commands.
"""

from __future__ import annotations

import sys


def main() -> None:
    from apsearch.cli import app

    # IMPORTANT: never call a Typer/Click command function directly (e.g.
    # `cli_launch()`) -- its parameter defaults are `typer.Option(...)`
    # sentinel objects, not the actual resolved values. Those are only
    # substituted with real values when Click's own parser drives the call.
    # Route everything through `app()` so that resolution always happens,
    # including the implicit "double-click with no arguments" case.
    if len(sys.argv) == 1:
        sys.argv.append("launch")
    app()


if __name__ == "__main__":
    main()
