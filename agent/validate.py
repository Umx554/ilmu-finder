"""Stage 4 - Structurer + checker: validate fields, keep the window, verify against source text, de-duplicate."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import date, time
from difflib import SequenceMatcher

log = logging.getLogger("agent.validate")
CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def parse_date(v) -> date | None:
    if not v or not isinstance(v, str):
        return None
    try:
        return date.fromisoformat(v.strip()[:10])
    except ValueError:
        return None


def parse_time(v) -> time | None:
    if not v or not isinstance(v, str):
        return None
    m = re.match(r"^\s*(\d{1,2})[:.](\d{2})", v)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return time(h, mi) if h < 24 and mi < 60 else None


def clean_str(v, limit: int = 200) -> str | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    if s.lower() in ("", "null", "none", "n/a", "na", "-", "tba", "unknown"):
        return None
    return s[:limit]


def in_text(needle: str, haystack_norm: str) -> bool:
    n = norm(needle)
    return bool(n) and f" {n} " in f" {haystack_norm} "


BM_DAYS = ["isnin", "selasa", "rabu", "khamis", "jumaat", "sabtu", "ahad"]
HONORIFICS = {
    "ustaz", "ustazah", "ust", "dr", "prof", "madya", "sheikh", "syeikh", "shaykh", "sh", "datuk", "dato",
    "datin", "tan", "sri", "seri", "haji", "hj", "hajah", "hjh", "habib", "syed", "sayyid", "tuan", "puan",
    "guru", "al", "bin", "binti", "bt", "ybhg", "yb", "maulana", "hafiz", "mufti", "imam", "tuan guru",
}
TOKEN_RE = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[^\s)\]>\"'<]*)?")


def url_key(u: str) -> str:
    """Compare links fairly: ignore scheme, 'www.', host case and a trailing slash/full stop.
    The path stays case-sensitive (bit.ly/AbC and bit.ly/abc are different links)."""
    u = re.sub(r"^https?://", "", u.strip(), flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I).rstrip("/.,;")
    host, _, rest = u.partition("/")
    return host.lower() + ("/" + rest if rest else "")


def page_urls(text: str) -> set[str]:
    return {url_key(t) for t in TOKEN_RE.findall(text)}


def ground(ev: dict, source_text: str) -> dict:
    """Cross-check AI output against the page text. Unverifiable details are flagged or removed."""
    src = norm(source_text)
    flags = []

    evidence = clean_str(ev.get("date_evidence"), 300)
    day = str(ev["start"].day)
    ev_l = (evidence or "").lower()
    weekday_named = (ev["start"].strftime("%A").lower()[:3] in ev_l) or (BM_DAYS[ev["start"].weekday()] in ev_l)
    if not evidence or not in_text(evidence, src):
        flags.append("date not found verbatim in source")
    elif not re.search(rf"(?<!\d)0?{day}(?!\d)", evidence) and not weekday_named:   # handles 28hb, 28th, 28/11
        flags.append("date quote does not contain the day")

    words = set(src.split())
    verified, unverified = [], []
    for sp in ev.get("speakers") or []:
        sp = clean_str(sp, 80)
        if not sp:
            continue
        tokens = [t for t in norm(sp).split() if len(t) > 2 and t not in HONORIFICS]
        if tokens and all(t in words for t in tokens):      # whole words: "ali" must not match "halal"
            verified.append(sp)
        else:
            unverified.append(sp)
    if unverified:
        flags.append("some speakers unverified")
    ev["speakers"] = verified
    ev["speakers_unverified"] = unverified

    price = ev.get("price_text")
    if price:
        nums = re.findall(r"\d+", price.replace(",", ""))
        page_nums = set(re.findall(r"\d+", source_text.replace(",", "")))   # whole numbers: "20" is not in "2026"
        if nums and not all(n in page_nums for n in nums):
            flags.append("price not found in source; removed")
            ev["price_text"] = None

    reg = ev.get("registration_url")
    if reg:   # links are the easiest thing for an AI to invent: the exact link must be on the page
        if url_key(reg) not in page_urls(source_text):
            flags.append("registration link not found in source; removed")
            ev["registration_url"] = None

    if ev.get("venue") and not in_text(ev["venue"], src):
        vwords = [w for w in norm(ev["venue"]).split() if len(w) > 3]
        if vwords and sum(w in words for w in vwords) / len(vwords) < 0.5:
            flags.append("venue not found in source")

    ev["flags"] = flags
    if flags and ev["confidence"] == "high":
        ev["confidence"] = "medium"
    if any("date" in f for f in flags):
        ev["confidence"] = "low"
    return ev


def as_bool(v):
    """AI output sometimes says "false" as a string."""
    if isinstance(v, bool):
        return v
    return {"true": True, "yes": True, "false": False, "no": False}.get(str(v).strip().lower()) if v is not None else None


def speakers_list(v) -> list[str]:
    if isinstance(v, str):                      # "Ustaz A dan Dr B" instead of a list
        v = re.split(r"\s*(?:,|;|&|\bdan\b|\band\b)\s*", v)
    return [s for s in (v or []) if isinstance(s, str) and s.strip()][:8]


def source_id(url: str | None) -> str:
    """Site or Telegram channel a URL belongs to (two posts from one channel are not independent)."""
    m = re.match(r"https?://(?:www\.)?([^/]+)(/s)?/?([^/?#]*)", url or "")
    if not m:
        return url or ""
    host = m.group(1).lower()
    return f"{host}/{m.group(3).lower()}" if host == "t.me" else host


def coerce(raw: dict, today: date, end: date) -> tuple[dict | None, str]:
    if as_bool(raw.get("is_islamic")) is False:
        return None, "not Islamic"
    title = clean_str(raw.get("title"), 160)
    if not title:
        return None, "no title"
    start = parse_date(raw.get("start_date"))
    if not start:
        return None, "no valid date"
    finish = parse_date(raw.get("end_date")) or start
    if finish < start or (finish - start).days > 14:
        finish = start
    if finish < today:
        return None, "past"
    if start > end:
        return None, "beyond window"
    conf = str(raw.get("confidence") or "low").lower()
    ev = {
        "title": title,
        "event_type": clean_str(raw.get("event_type"), 40),
        "start": start,
        "end": finish,
        "start_time": parse_time(raw.get("start_time")),
        "end_time": parse_time(raw.get("end_time")),
        "time_note": clean_str(raw.get("time_note"), 60),
        "venue": clean_str(raw.get("venue"), 120),
        "address": clean_str(raw.get("address"), 200),
        "city": clean_str(raw.get("city"), 60),
        "state": clean_str(raw.get("state"), 40),
        "ai_in_kv": as_bool(raw.get("in_klang_valley")),
        "is_online": as_bool(raw.get("is_online")) is True,
        "price_text": clean_str(raw.get("price_text"), 80),
        "is_free": as_bool(raw.get("is_free")),
        "speakers": speakers_list(raw.get("speakers")),
        "organizer": clean_str(raw.get("organizer"), 100),
        "topic": clean_str(raw.get("topic"), 160),
        "registration_url": clean_str(raw.get("registration_url"), 2000),   # never cut a link
        "date_evidence": raw.get("date_evidence"),
        "confidence": conf if conf in CONF_RANK else "low",
        "sources": [raw.get("source_url")],
    }
    if ev["registration_url"] and not ev["registration_url"].startswith("http"):
        ev["registration_url"] = None
    return ground(ev, raw.get("source_text", "")), "ok"


def same_event(a: dict, b: dict) -> bool:
    if a["start"] != b["start"]:
        return False
    va, vb = norm(a.get("venue")), norm(b.get("venue"))
    wa, wb = set(va.split()), set(vb.split())
    if va and vb and SequenceMatcher(None, va, vb).ratio() < 0.5 and not (wa <= wb or wb <= wa):
        return False                            # ("IKIM" vs "Institut Kefahaman Islam Malaysia (IKIM)" = same)                            # same generic title ("Kuliah Maghrib") at different places
    ta, tb = norm(a["title"]), norm(b["title"])
    r = SequenceMatcher(None, ta, tb).ratio()
    if r >= 0.8 or (ta and tb and (ta in tb or tb in ta)):
        return True
    if a.get("venue") and b.get("venue") and SequenceMatcher(None, norm(a["venue"]), norm(b["venue"])).ratio() > 0.8:
        return r >= 0.5 or (a.get("start_time") and a.get("start_time") == b.get("start_time"))
    return False


def merge(a: dict, b: dict) -> dict:
    best, other = (a, b) if CONF_RANK[a["confidence"]] >= CONF_RANK[b["confidence"]] else (b, a)
    out = dict(best)
    for k, v in other.items():
        if out.get(k) in (None, "", []) and v not in (None, "", []):
            out[k] = v
    out["speakers"] = list(dict.fromkeys(best["speakers"] + other["speakers"]))
    out["sources"] = list(dict.fromkeys(best["sources"] + other["sources"]))
    independent = {source_id(u) for u in out["sources"]}
    if len(independent) >= 2:         # corroborated by a different site/channel: raise one level
        if out["confidence"] == "medium":
            out["confidence"] = "high"
        elif out["confidence"] == "low" and not any("date" in f for f in best["flags"]):
            out["confidence"] = "medium"
    return out


def fingerprint(ev: dict) -> str:
    return hashlib.sha1(f"{ev['start'].isoformat()}|{norm(ev['title'])[:60]}".encode()).hexdigest()[:12]


def validate(raw_events: list[dict], today: date, end: date, stats: dict) -> list[dict]:
    kept, reasons = [], {}
    for raw in raw_events:
        try:
            ev, why = coerce(raw, today, end)
        except Exception as e:                  # one odd AI answer must not stop the run
            log.warning("skipping malformed event: %s", e)
            ev, why = None, "malformed"
        if ev is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        for i, existing in enumerate(kept):
            if same_event(existing, ev):
                kept[i] = merge(existing, ev)
                reasons["duplicate merged"] = reasons.get("duplicate merged", 0) + 1
                break
        else:
            kept.append(ev)
    for ev in kept:
        ev["id"] = fingerprint(ev)
    stats.update(events_valid=len(kept), validate_drops=reasons)
    log.info("[validate] %d events kept; drops: %s", len(kept), reasons)
    return kept
