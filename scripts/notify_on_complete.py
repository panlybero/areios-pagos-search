"""Monitors the crawl job and sends email notifications on release and completion."""

import json
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

PROGRESS_FILE = Path("data/crawl_progress.json")
EMAIL = "panliberopoulos@gmail.com"
APP_PWD = "xndl lvvn ubsq iuzr"
TAG = "v0.3.0"
RELEASE_URL = f"https://github.com/panlybero/areios-pagos-search/releases/tag/{TAG}"


def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
            s.login(EMAIL, APP_PWD)
            s.send_message(msg)
        print("Notification sent to", EMAIL, ":", subject)
    except Exception as exc:
        print(f"Failed to send email notification: {exc}")


def main() -> None:
    print("Notification watcher started. Monitoring data/crawl_progress.json...")
    notified_release = False

    while True:
        time.sleep(15)
        if not PROGRESS_FILE.is_file():
            continue

        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            continue

        status = data.get("status")
        fetched = data.get("fetched", 0)
        chunks = data.get("chunks", 0)

        # 1. Release v0.3.0 published (All decisions downloaded!)
        if (status in ("published_v0.3.0", "embedding", "completed")) and not notified_release:
            subject = f"🚀 [Areios Pagos] Release v0.3.0 Published! All {fetched:,} Modern Decisions Ready"
            body = (
                f"Great news! All modern decisions from 2018 through 2026 have been downloaded and packaged!\n\n"
                f"• Total decisions available: {fetched:,}\n"
                f"• Passage chunks currently embedded: {chunks:,}\n\n"
                f"Release v0.3.0 is live on GitHub and ready for your father / users:\n"
                f"{RELEASE_URL}\n\n"
                f"On macOS: download the zip, extract, and double-click launch.command!\n"
                f"(Background vector embedding is continuing in the background)."
            )
            send_email(subject, body)
            notified_release = True

        # 2. Final completion (100% of all vector embeddings finished!)
        if status == "completed":
            subject = f"✅ [Areios Pagos] 100% Vector Embeddings Complete ({chunks:,} chunks)"
            body = (
                f"All passage chunks across the entire modern corpus have finished embedding!\n\n"
                f"• Decisions: {fetched:,}\n"
                f"• Embedded Chunks: {chunks:,}\n\n"
                f"The database has full hybrid + semantic coverage across 100% of cases."
            )
            send_email(subject, body)
            break


if __name__ == "__main__":
    main()
