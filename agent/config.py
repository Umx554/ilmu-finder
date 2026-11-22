"""Load config.yaml and apply environment-variable overrides."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "output"

EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
BM_MONTHS = ["Januari", "Februari", "Mac", "April", "Mei", "Jun", "Julai",
             "Ogos", "September", "Oktober", "November", "Disember"]


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


def load_config(path: Path | None = None) -> dict:
    path = path or Path(os.getenv("CONFIG_FILE", ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    llm = cfg.setdefault("llm", {})
    llm["base_url"] = os.getenv("LLM_BASE_URL") or llm.get("base_url")
    llm["model"] = os.getenv("LLM_MODEL") or llm.get("model")
    fb = os.getenv("LLM_FALLBACK_MODEL") or llm.get("fallback_model") or ""
    llm["fallback_model"] = "" if str(fb).lower() == "none" else fb      # "none" turns the fallback off
    llm["delay_seconds"] = _env_float("LLM_DELAY", llm.get("delay_seconds", 8))
    llm["max_pages"] = _env_int("MAX_PAGES", llm.get("max_pages", 30))
    llm["max_chars_per_page"] = _env_int("MAX_CHARS_PER_PAGE", llm.get("max_chars_per_page", 6000))
    llm.setdefault("temperature", 0)

    s = cfg.setdefault("search", {})
    s["max_queries"] = _env_int("MAX_QUERIES", s.get("max_queries", 45))

    cfg.setdefault("delivery", {}).setdefault("telegram_cards", True)
    cfg.setdefault("direct_sources", [])
    cfg.setdefault("speaker_watchlist", [])
    cfg["direct_sources"] = cfg["direct_sources"] or []
    cfg["speaker_watchlist"] = cfg["speaker_watchlist"] or []
    cfg.setdefault("timezone", "Asia/Kuala_Lumpur")
    cfg.setdefault("window_days", 30)
    cfg.setdefault("this_week_days", 7)
    cfg.setdefault("cache_days", 35)

    cfg["secrets"] = {
        "tavily": os.getenv("TAVILY_API_KEY", ""),
        "llm": os.getenv("LLM_API_KEY", ""),
        "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        # optional e-mail (e.g. Gmail with an app password)
        "smtp_user": os.getenv("SMTP_USER", ""),
        "smtp_password": os.getenv("SMTP_PASSWORD", ""),
        "smtp_host": os.getenv("SMTP_HOST", ""),
        "smtp_port": os.getenv("SMTP_PORT", ""),
        "email_to": os.getenv("EMAIL_TO", ""),
    }
    return cfg


def today_local(cfg: dict) -> date:
    override = os.getenv("TODAY")  # for testing, e.g. TODAY=2026-11-23
    if override:
        return date.fromisoformat(override)
    return datetime.now(ZoneInfo(cfg["timezone"])).date()


def window(cfg: dict) -> tuple[date, date]:
    start = today_local(cfg)
    return start, start + timedelta(days=cfg["window_days"])


def window_months(cfg: dict) -> tuple[list[int], list[int]]:
    """Month numbers and years covered by the window, in order."""
    start, end = window(cfg)
    months, years = [], []
    d = start.replace(day=1)
    while d <= end:
        months.append(d.month)
        if d.year not in years:
            years.append(d.year)
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months, years
