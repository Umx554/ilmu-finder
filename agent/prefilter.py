"""Stage 2 - Filter: cheap keyword scoring, plus trimming long pages to the useful parts."""
from __future__ import annotations

import logging
import re

from .config import EN_MONTHS, BM_MONTHS, window_months

log = logging.getLogger("agent.prefilter")

EVENT_WORDS = [
    "seminar", "conference", "summit", "talk", "lecture", "forum", "symposium", "workshop",
    "convention", "ceramah", "kuliah", "tazkirah", "daurah", "dauroh", "multaqa", "wacana",
    "konvensyen", "persidangan", "bengkel", "kursus", "majlis ilmu", "himpunan", "ijtimak",
    "halaqah", "simposium", "kolokium", "program", "event", "register", "daftar", "tiket", "ticket",
]
ISLAMIC_WORDS = [
    "islam", "muslim", "masjid", "mosque", "surau", "ustaz", "ustazah", "ustadh", "sheikh", "syeikh",
    "shaykh", "imam", "mufti", "quran", "qur'an", "al-quran", "sunnah", "hadith", "hadis", "fiqh",
    "feqah", "tafsir", "sirah", "seerah", "aqidah", "akidah", "tasawwuf", "dakwah", "dawah", "da'wah",
    "maulid", "maulidur", "hijrah", "ramadan", "ramadhan", "syariah", "shariah", "zakat", "waqf",
    "wakaf", "halal", "jakim", "jawi", "ikim", "mais", "jais", "maiwp", "sahabat", "rasulullah",
]
NOISE_WORDS = ["obituary", "recipe", "resepi", "job vacancy", "jawatan kosong", "for sale"]
_MON = "|".join(sorted({m[:3].lower() for m in EN_MONTHS + BM_MONTHS}))
# "28/11", "28-11-2026", "28hb Nov", "28 November", "28th Dec"
DATE_RE = re.compile(rf"(?<!\d)\d{{1,2}}\s*(?:[-/.]\s*\d{{1,2}}(?:\s*[-/.]\s*\d{{2,4}})?(?![\d.])|"
                     rf"(?:hb|th|st|nd|rd)?\s*(?:{_MON})[a-z]*\b)", re.I)
COPYRIGHT_RE = re.compile(r"(?:©|\(c\)|copyright|hak cipta)[^\n]{0,60}", re.I)
_WORD_RE: dict[str, re.Pattern] = {}
FOCUS_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in EVENT_WORDS + ISLAMIC_WORDS + [
    "rm", "pm", "am", "venue", "lokasi", "tempat", "tarikh", "masa", "date", "time", "speaker",
    "penceramah", "bersama", "pagi", "petang", "malam", "maghrib", "isyak", "asar", "zohor"]) + r")\b", re.I)
YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _count(text: str, words: list[str]) -> int:
    """Words must START at a word boundary: 'islam' matches 'islamic', 'sahabat' does not match 'persahabatan'."""
    n = 0
    for w in words:
        rx = _WORD_RE.get(w) or _WORD_RE.setdefault(w, re.compile(rf"\b{re.escape(w)}"))
        n += bool(rx.search(text))
    return n


def month_hit(text: str, months: list[int]) -> bool:
    names = set()
    for m in months:
        for full in (EN_MONTHS[m - 1].lower(), BM_MONTHS[m - 1].lower()):
            names.update({full, full[:3]})
        if m == 9:
            names.add("sept")
    names.discard("may")             # English "may" is usually a verb; accept it only next to a number
    if any(re.search(rf"\b{re.escape(n)}\b", text) for n in names):
        return True
    if 5 in months and re.search(r"\d\s*(?:th|st|nd|rd)?\s*may\b|\bmay\s+\d", text):
        return True
    return any(re.search(rf"(?<!\d)\d{{1,2}}\s*[-/.]\s*0?{m}(?!\d)", text) for m in months)   # 28/11


def score_page(page: dict, cfg: dict) -> tuple[int, str]:
    """Return (score, reason). Score <= 0 means drop."""
    text = ((page.get("title") or "") + "\n" + (page.get("text") or "")).lower()
    if len(text) < 80:
        return 0, "too short"
    ev, isl = _count(text, EVENT_WORDS), _count(text, ISLAMIC_WORDS)
    if ev == 0:
        return 0, "no event words"
    if isl == 0:
        return 0, "no Islamic words"
    if _count(text, NOISE_WORDS) >= 2:
        return 0, "noise"

    months, years = window_months(cfg)
    dated = month_hit(text, months)
    found_years = {int(y) for y in YEAR_RE.findall(COPYRIGHT_RE.sub(" ", text))}   # ignore "© 2025 Masjid X"
    if found_years and not (found_years & set(years)) and page.get("source") == "search":
        return 0, f"only old/other years {sorted(found_years)[-3:]}"
    if not dated and page.get("source") == "search":
        return 0, "no date in window months"

    score = min(ev, 5) + min(isl, 5) + (3 if dated else 0)
    score += 2 if set(years) & found_years else 0
    score += 2 if page.get("source") in ("telegram", "direct") else 0
    score += 1 if re.search(r"\brm\s?\d|percuma|free\b", text) else 0
    return score, "ok"


URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
REG_HINTS = ("forms.gle", "docs.google.com/forms", "peatix", "eventbrite", "bit.ly", "wa.me", "linktr.ee",
             "tinyurl", "s.id/", "register", "registration", "daftar", "pendaftaran", "tiket", "ticket",
             "signup", "sign-up", "rsvp", "booking", "typeform", "jotform", "lu.ma", "eventzilla")


def registration_links(text: str, limit: int = 8) -> list[str]:
    out = []
    for u in URL_RE.findall(text):
        u = u.rstrip(".,;")
        if any(h in u.lower() for h in REG_HINTS) and u not in out:
            out.append(u)
    return out[:limit]


def focus_text(text: str, max_chars: int) -> str:
    """Keep the parts of a long page most likely to describe events, plus any registration links."""
    links = [u for u in registration_links(text) if len(u) <= 500]
    suffix = ""
    for u in links:                          # links may use at most a third of the budget
        line = ("\n\nLINKS FOUND:\n" if not suffix else "\n") + u
        if len(suffix) + len(line) > max_chars // 3:
            break
        suffix += line
    return _trim(text, max_chars - len(suffix)) + suffix


def _trim(text: str, max_chars: int) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    keep = set()
    for i, line in enumerate(lines):
        if DATE_RE.search(line) or FOCUS_RE.search(line):
            keep.update({i - 1, i, i + 1})
    picked, total = [], 0
    for i in sorted(k for k in keep if 0 <= k < len(lines)):
        if total + len(lines[i]) + 1 > max_chars:
            break
        picked.append(lines[i])
        total += len(lines[i]) + 1
    out = "\n".join(picked)
    return out if out else text[:max_chars]


def prefilter(pages: list[dict], cfg: dict, stats: dict) -> list[dict]:
    kept, reasons = [], {}
    for p in pages:
        try:
            s, why = score_page(p, cfg)
        except Exception as e:               # one odd page must not stop the run
            log.warning("prefilter skipped %s: %s", p.get("url"), e)
            s, why = 0, "error"
        if s > 0:
            p["score"] = s
            kept.append(p)
        else:
            reasons[why.split(" [")[0]] = reasons.get(why.split(" [")[0], 0) + 1
    kept.sort(key=lambda p: p["score"], reverse=True)
    stats.update(pages_prefiltered=len(kept), prefilter_drops=reasons)
    log.info("[prefilter] kept %d of %d; drops: %s", len(kept), len(pages), reasons)
    return kept
