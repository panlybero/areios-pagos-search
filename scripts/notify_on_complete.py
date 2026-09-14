"""Monitors the background crawl job and sends an email notification when complete."""

import json
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path

PROGRESS_FILE = Path("data/crawl_progress.json")
EMAIL = "[REDACTED]"
APP_PWD = "[REDACTED]"
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
    except Exception as exc:
        print(f"Failed to send email notification: {exc}")


def main() -> None:
    print("Notification watcher started. Monitoring data/crawl_progress.json...")
    while True:
        time.sleep(30)
        if not PROGRESS_FILE.is_file():
            continue

        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            continue

        status = data.get("status")
        if status == "completed":
            fetched = data.get("fetched", 0)
            chunks = data.get("chunks", 0)
            subject = f"✅ [Areios Pagos] Modern Corpus Crawl Complete ({fetched:,} decisions)"
            body = (
                f"The complete 2018–2026 Areios Pagos case law crawl has finished!\n\n"
                f"• Decisions crawled & indexed: {fetched:,}\n"
                f"• Total passage chunks: {chunks:,}\n\n"
                f"The seed database and macOS universal package have been published to GitHub:\n"
                f"{RELEASE_URL}\n\n"
                f"You can now send the release package to your father / users."
            )
            send_email(subject, body)
            print("Completion notification sent to", EMAIL)
            break


if __name__ == "__main__":
    main()
