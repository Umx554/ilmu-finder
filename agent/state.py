"""Small JSON files in state/ that the workflow commits back to the repo (memory between runs)."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from .config import STATE_DIR

log = logging.getLogger("agent.state")
FILES = ("extract_cache", "seen", "query_stats", "source_stats")


def load_state() -> dict:
    out = {}
    for name in FILES:
        p = STATE_DIR / f"{name}.json"
        try:
            out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, json.JSONDecodeError):
            log.warning("state file %s unreadable; starting fresh", p.name)
            out[name] = {}
    return out


def prune(state: dict, today: date, cache_days: int) -> None:
    cutoff = (today - timedelta(days=cache_days)).isoformat()
    state["extract_cache"] = {k: v for k, v in state["extract_cache"].items() if v.get("ts", "") >= cutoff}
    state["seen"] = {k: v for k, v in state["seen"].items() if v.get("end", "") >= today.isoformat()}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    for name in FILES:
        p = STATE_DIR / f"{name}.json"
        p.write_text(json.dumps(state.get(name, {}), ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
