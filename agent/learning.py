"""Self-improvement between runs, with zero AI tokens.

Two small memories, both plain JSON-serialisable dicts kept in state/:

1. Query yield (state/query_stats.json)
   Which search queries actually led to validated events. Queries that keep finding
   nothing are "rested" and only retried now and then, so the Tavily budget goes to
   queries that work, and new queries from config.yaml always get a fair try.

2. Source discovery (state/source_stats.json)
   Which websites and Telegram channels the events came from. A site or channel that
   keeps producing events, but is not in `direct_sources` yet, is suggested in the
   digest footer, so the user can add it and stop paying search credits to find it.

Everything here is deterministic and offline: no network, no AI, no randomness.
"""
from __future__ import annotations

import logging
import re
import zlib
from datetime import date, timedelta
from urllib.parse import urlsplit

from .config import BM_MONTHS, EN_MONTHS

log = logging.getLogger("agent.learning")

# A query is "resting" after this many runs in a row without a single event ...
REST_AFTER_ZERO_RUNS = 3
# ... and is then retried only once every this many ISO weeks.
REST_RETRY_WEEKS = 4

MAX_SUGGESTIONS = 3
MIN_EVENTS_TO_SUGGEST = 2
SOURCE_FORGET_DAYS = 120
_MAX_REMEMBERED_IDS = 60  # per source; only used to avoid counting the same event twice

# Sites that are not useful as a direct source: search engines, social networks
# we can't read, link shorteners, and big ticketing sites already covered by queries.
_NOT_A_SOURCE = {
    "google.com", "forms.gle", "docs.google.com", "goo.gl", "bit.ly", "tinyurl.com",
    "linktr.ee", "wa.me", "api.whatsapp.com", "chat.whatsapp.com",
    "facebook.com", "m.facebook.com", "fb.com", "fb.me", "instagram.com",
    "x.com", "twitter.com", "tiktok.com", "youtube.com", "youtu.be",
    "peatix.com", "eventbrite.com", "eventbrite.com.my",
}

# Month names and years are stripped from a query to get its stable "key".
# build_queries() fills {months}/{bulan}/{year} differently every month, so without
# this, "ceramah Shah Alam November Disember 2026" and "... Disember Januari 2027"
# would look like two unrelated queries and the agent would forget everything monthly.
_MONTH_WORDS = {m.lower() for m in EN_MONTHS + BM_MONTHS}
_YEAR_RE = re.compile(r"^(19|20)\d\d$")


# ---------------------------------------------------------------- helpers
def query_key(query: str) -> str:
    """Stable identity of a query: lowercase, without month names and years."""
    words = [w for w in query.lower().split()
             if w not in _MONTH_WORDS and not _YEAR_RE.match(w)]
    return " ".join(words)


def _stable_hash(text: str) -> int:
    # Python's hash() changes between processes; crc32 is the same on every run.
    return zlib.crc32(text.encode("utf-8"))


def _norm_url(url: str) -> str:
    """Light URL normalisation, enough to match a page URL with an event source URL."""
    p = urlsplit((url or "").strip())
    host = p.netloc.lower().removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}" + (f"?{p.query}" if p.query else "")


def _page_queries(page: dict) -> list[str]:
    """The queries that found a page. Accepts 'query' (str/None) and/or 'queries' (list)."""
    qs = list(page.get("queries") or [])
    if page.get("query"):
        qs.append(page["query"])
    return qs


# ---------------------------------------------------------------- 1. query yield
def record_run(query_stats: dict, queries_run: list[str], pages: list[dict],
               events: list[dict], today: date) -> None:
    """Update per-query yield stats in place after a run.

    queries_run: the queries that were actually searched successfully this run.
    pages:       fetched pages; search pages carry 'query' (the query that found them).
    events:      final validated events; each has 'sources' (list of page URLs).

    Per query (keyed by query_key):
      runs, total_events, last_yield_date, consecutive_zero_runs, last_run
    """
    if not queries_run:
        return
    if not events:
        # Zero events overall almost always means an upstream problem (AI quota used up,
        # search outage, pipeline error), not that every query is bad. Don't punish them.
        log.info("[learning] no events this run; query stats left unchanged")
        return

    # Which normalised page URLs each query found.
    found_by: dict[str, set[str]] = {}
    for p in pages:
        for q in _page_queries(p):
            found_by.setdefault(query_key(q), set()).add(_norm_url(p.get("url", "")))

    event_sources = [{_norm_url(u) for u in (ev.get("sources") or []) if u} for ev in events]

    # Pages that passed the filter but were NOT read by the AI (page cap, quota, time budget) tell us
    # nothing about the query that found them; don't count that query as "found nothing".
    unread: set[str] = set()
    for p in pages:
        if p.get("ai_read") is False:
            for q in _page_queries(p):
                unread.add(query_key(q))

    today_iso = today.isoformat()
    for key in dict.fromkeys(query_key(q) for q in queries_run):  # unique, order kept
        st = query_stats.setdefault(key, {"runs": 0, "total_events": 0, "last_yield_date": None,
                                          "consecutive_zero_runs": 0, "last_run": None})
        if st.get("last_run") == today_iso:
            continue  # a second (manual) run on the same day must not double-count
        urls = found_by.get(key, set())
        n = sum(1 for srcs in event_sources if srcs & urls)
        if n == 0 and key in unread:
            continue  # no evidence either way this week
        st["runs"] = st.get("runs", 0) + 1
        st["total_events"] = st.get("total_events", 0) + n
        st["last_run"] = today_iso
        if n:
            st["last_yield_date"] = today_iso
            st["consecutive_zero_runs"] = 0
        else:
            st["consecutive_zero_runs"] = st.get("consecutive_zero_runs", 0) + 1


def is_resting(stats: dict | None) -> bool:
    return bool(stats) and stats.get("consecutive_zero_runs", 0) >= REST_AFTER_ZERO_RUNS


def retry_due(query: str, today: date) -> bool:
    """True in one ISO week out of REST_RETRY_WEEKS, offset per query so resting
    queries are spread over the weeks instead of all being retried together."""
    week = today.isocalendar().week
    return (week + _stable_hash(query_key(query))) % REST_RETRY_WEEKS == 0


def plan_queries(all_queries: list[str], query_stats: dict, max_queries: int,
                 today: date) -> list[str]:
    """Pick and order this week's queries.

    Order: never-tried queries (config order), then proven queries by events-per-run
    (best first; ties keep config order), then resting queries whose retry week it is.
    Resting queries outside their retry week are skipped. At most max_queries.
    Also prunes stats of queries no longer generated by config (mutates query_stats).
    """
    keys = {query_key(q) for q in all_queries}
    for k in [k for k in query_stats if k not in keys]:
        del query_stats[k]

    new, active, due = [], [], []
    for i, q in enumerate(dict.fromkeys(all_queries)):
        st = query_stats.get(query_key(q))
        if not st or not st.get("runs"):
            new.append(q)
        elif is_resting(st):
            if retry_due(q, today):
                due.append(q)
        else:
            rate = st.get("total_events", 0) / st["runs"]
            active.append((-rate, i, q))
    active.sort()
    plan = new + [q for _, _, q in active] + due
    if len(plan) > max_queries > 0:
        # More queries than slots: reserve ~10% for resting queries due a retry and for the
        # least-recently-run active ones, so no query is starved forever below the cut.
        reserve = max(1, max_queries // 10)
        stale = sorted(active, key=lambda t: query_stats[query_key(t[2])].get("last_run") or "")[:reserve]
        plan = list(dict.fromkeys(due[:reserve] + [q for *_, q in stale] + plan))
    skipped = len(all_queries) - len(plan)
    if skipped:
        log.info("[learning] %d resting queries skipped this week", skipped)
    return plan[:max(0, max_queries)]


# ---------------------------------------------------------------- 2. source discovery
def source_id(url: str) -> tuple[str, str] | None:
    """(id, suggestion URL) for a page URL, or None if it can't be a direct source.

    https://t.me/somechannel/123   -> ('t.me/somechannel', 'https://t.me/s/somechannel')
    https://t.me/s/somechannel     -> same
    https://www.example.org/a/b    -> ('example.org', 'https://example.org/')
    """
    p = urlsplit((url or "").strip())
    host = p.netloc.lower().split(":")[0].removeprefix("www.")
    if not host:
        return None
    if host in ("t.me", "telegram.me"):
        parts = [x for x in p.path.split("/") if x]
        if parts and parts[0] == "s":
            parts = parts[1:]
        if not parts or parts[0].startswith("+") or parts[0] in ("joinchat", "addstickers", "share"):
            return None  # private invite links etc. can't be read
        ch = parts[0].lower()
        return f"t.me/{ch}", f"https://t.me/s/{ch}"
    return host, f"https://{host}/"


def _event_key(ev: dict) -> str:
    return str(ev.get("id") or f"{ev.get('title', '')}|{ev.get('start', '')}")


def suggest_sources(events: list[dict], direct_sources: list[str], source_stats: dict,
                    today: date) -> list[str]:
    """Record where this run's events came from, and suggest up to 3 new direct sources.

    source_stats (mutated in place), per source id:
      events (distinct events ever counted), first_seen, last_seen, ids (recent event ids)
    The same event shows up in several weekly digests (30-day window); it is counted once.
    """
    today_iso = today.isoformat()
    for ev in events:
        ek = _event_key(ev)
        ids_this_event = {}
        for u in ev.get("sources") or []:
            sid = source_id(u)
            if sid:
                ids_this_event[sid[0]] = True
        for sid in ids_this_event:
            st = source_stats.setdefault(sid, {"events": 0, "first_seen": today_iso,
                                               "last_seen": today_iso, "ids": []})
            st["last_seen"] = today_iso
            if ek not in st["ids"]:
                st["ids"] = (st["ids"] + [ek])[-_MAX_REMEMBERED_IDS:]
                st["events"] = st.get("events", 0) + 1

    cutoff = (today - timedelta(days=SOURCE_FORGET_DAYS)).isoformat()
    for k in [k for k, v in source_stats.items() if v.get("last_seen", "") < cutoff]:
        del source_stats[k]

    # A direct source covers its whole domain, or its whole Telegram channel.
    covered = {sid[0] for sid in map(source_id, direct_sources or []) if sid}
    candidates = []
    for sid, st in source_stats.items():
        if sid in covered or sid in _NOT_A_SOURCE or st.get("events", 0) < MIN_EVENTS_TO_SUGGEST:
            continue
        if not sid.startswith("t.me/") and any(sid.endswith("." + d) for d in _NOT_A_SOURCE):
            continue  # e.g. business.facebook.com
        candidates.append((-st["events"], _neg_date(st.get("last_seen", "")), sid))
    candidates.sort()
    out = []
    for _, _, sid in candidates[:MAX_SUGGESTIONS]:
        out.append(f"https://t.me/s/{sid[5:]}" if sid.startswith("t.me/") else f"https://{sid}/")
    return out


def _neg_date(iso: str) -> int:
    # Sort helper: more recent last_seen first.
    try:
        return -date.fromisoformat(iso).toordinal()
    except ValueError:
        return 0


def suggestions_text(suggestions: list[str]) -> str:
    """One footer line for the digest ('' when there is nothing to suggest)."""
    if not suggestions:
        return ""
    return ("💡 Sources that keep finding events – add them to direct_sources in config.yaml "
            "to read them every week for free: " + " · ".join(suggestions))
