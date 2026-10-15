"""Wall-clock formatting for UI and logs (Indian Standard Time)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def now_hms() -> str:
    """Current time as HH:MM:SS in IST."""
    return datetime.now(IST).strftime("%H:%M:%S")


def now_datetime_ist() -> str:
    """Current date and time in IST."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
