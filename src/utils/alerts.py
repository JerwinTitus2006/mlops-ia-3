from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from src.utils.settings import get_settings

logger = logging.getLogger(__name__)


def send_alert(message: str) -> None:
    """Simulate an email alert by writing a warning log and alert file."""
    logger.warning("ALERT: %s", message)
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    alert_log = settings.models_dir / "alerts.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    with alert_log.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} | ALERT | {message}\n")
