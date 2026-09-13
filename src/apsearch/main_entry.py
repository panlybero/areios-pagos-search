"""Entrypoint for standalone packaged application.

* Double-clicking the app launches the Web UI and opens the browser.
* Running with arguments (e.g. `apsearch mcp` or `apsearch search ...`)
  dispatches to the CLI / MCP server commands.
"""

from __future__ import annotations

import sys


def main() -> None:
    from apsearch.cli import app, cli_launch

    if len(sys.argv) == 1:
        cli_launch()
    else:
        app()


if __name__ == "__main__":
    main()
