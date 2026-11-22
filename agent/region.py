"""Klang Valley check by text only (no maps, no API calls, no AI tokens)."""
from __future__ import annotations

import logging
import re

log = logging.getLogger("agent.region")

KV_PLACES = [
    "kuala lumpur", "kl", "wilayah persekutuan", "selangor", "putrajaya", "shah alam", "petaling jaya",
    "subang", "klang", "port klang", "bangi", "kajang", "cyberjaya", "gombak", "cheras", "ampang",
    "puchong", "seri kembangan", "serdang", "bukit jalil", "setapak", "wangsa maju", "kepong",
    "damansara", "bangsar", "rawang", "selayang", "semenyih", "sepang", "dengkil", "batu caves",
    "sungai buloh", "titiwangsa", "sentul", "setiawangsa", "bukit bintang", "brickfields", "jalan duta", "chow kit",
    "mont kiara", "sri petaling", "salak selatan", "kota kemuning", "klia", "ukm", "upm",
    "uia", "iium", "universiti malaya", "uitm shah alam", "masjid negara", "masjid wilayah", "masjid putra",
    "masjid besar kuala lumpur", "masjid sultan salahuddin", "bukit damansara", "ttdi", "setia alam",
    "bandar utama", "kota damansara", "cyberjaya",
]
WEAK_KV = ["seksyen", "sunway", "bandar sunway", "usj", "taman tun"]
OTHER_STATES = [
    "penang", "pulau pinang", "johor", "kedah", "kelantan", "terengganu", "pahang", "perak", "perlis",
    "melaka", "malacca", "negeri sembilan", "seremban", "nilai", "sabah", "sarawak", "ipoh", "kuantan",
    "kota bharu", "alor setar", "singapore", "singapura", "jakarta", "brunei",
]


ONLINE_RE = re.compile(r"\b(zoom|online|dalam talian|youtube|facebook live|fb live|google meet|meet|"
                       r"webinar|teams|livestream|live stream|siaran langsung)\b")
STREET_RE = re.compile(r"\b(?:jalan|jln|lebuh|lorong|lrg|persiaran)\s+[\w'-]+")   # "Jalan Pahang" is in KL


def _has(blob: str, words: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", blob) for w in words)


def online_only(ev: dict) -> bool:
    """True for purely online events. 'Masjid X & YouTube Live' is a physical (hybrid) event."""
    if not ev.get("is_online"):
        return False
    venue = (ev.get("venue") or "").lower()
    rest = re.sub(r"[^a-z ]+", " ", ONLINE_RE.sub(" ", venue))
    rest = " ".join(w for w in rest.split() if w not in {"via", "and", "dan", "live", "platform", "the", "di"})
    return not rest                       # nothing left but platform names

def location_blob(ev: dict) -> str:
    return " ".join(filter(None, [ev.get("venue"), ev.get("address"), ev.get("city"), ev.get("state")])).lower()


def in_klang_valley(ev: dict) -> tuple[bool, str]:
    """Returns (keep, reason)."""
    blob = location_blob(ev)
    if online_only(ev):
        return True, "online"
    # City/state fields are decisive: "Seksyen 2, Kota Bharu, Kelantan" is not Shah Alam.
    city_state = STREET_RE.sub(" ", " ".join(filter(None, [ev.get("city"), ev.get("state")])).lower())
    if _has(city_state, OTHER_STATES) and not _has(city_state, KV_PLACES) and ev.get("ai_in_kv") is not True:
        return False, "other state"
    if _has(blob, KV_PLACES):
        return True, "Klang Valley place named"
    if _has(STREET_RE.sub(" ", blob), OTHER_STATES) and ev.get("ai_in_kv") is not True:
        return False, "other state"
    if _has(blob, WEAK_KV):              # common in KV but also elsewhere: only after the other-state check
        return True, "Klang Valley place named (weak)"
    if ev.get("ai_in_kv") is True:
        return True, "AI says Klang Valley"
    if ev.get("ai_in_kv") is False:
        return False, "AI says outside"
    if not blob:
        return True, "no location stated"      # kept, shown as "venue not stated"
    return False, "unknown place"


def filter_region(events: list[dict], stats: dict) -> list[dict]:
    kept, reasons = [], {}
    for ev in events:
        ok, why = in_klang_valley(ev)
        if ok:
            kept.append(ev)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    stats.update(events_in_region=len(kept), region_drops=reasons)
    log.info("[region] %d kept; drops: %s", len(kept), reasons)
    return kept
