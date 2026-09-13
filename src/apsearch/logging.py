"""Logging setup.

Plain text locally; Cloud Logging structured JSON when running on Cloud Run
(``severity``/``message`` are the field names the agent understands, and
``logging.googleapis.com/trace`` links entries to a request trace).
"""

from __future__ import annotations

import json
import logging
import os
import sys

from apsearch.config import settings

_CONFIGURED = False


class CloudLoggingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            entry["stack_trace"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            entry[key] = value
        if project := os.environ.get("GOOGLE_CLOUD_PROJECT"):
            if trace := getattr(record, "trace_id", None):
                entry["logging.googleapis.com/trace"] = (
                    f"projects/{project}/traces/{trace}"
                )
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    if settings.use_json_logs:
        handler.setFormatter(CloudLoggingFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or settings.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
