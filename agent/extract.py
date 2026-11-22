"""Stage 3 - Details extractor: one AI call per page returns ALL events on it as JSON."""
from __future__ import annotations

import hashlib
import logging
from datetime import date

from .llm import LLMClient, QuotaExhausted, ServiceError
from .prefilter import focus_text

log = logging.getLogger("agent.extract")

SYSTEM = """You extract Islamic events from web page text for an events digest in the Klang Valley, Malaysia.
Return ONLY a JSON object: {"events": [ ... ]}. If the page lists no qualifying event, return {"events": []}.

Rules:
- A page may list several events. Return one object per distinct event (max 10).
- Only include events with an Islamic theme (talks, ceramah, kuliah, seminars, conferences, summits, daurah,
  forums, workshops on Islamic topics, Islamic finance, Quran, etc.).
- Use ONLY information written in the text. Never guess speakers, prices or venues. Use null when absent.
- Dates: output ISO format YYYY-MM-DD. If the year is missing, infer the nearest FUTURE date relative to TODAY.
  Pages may be in Malay: Mac=March, Mei=May, Jun=June, Julai=July, Ogos=August, Disember=December,
  Isnin=Mon, Selasa=Tue, Rabu=Wed, Khamis=Thu, Jumaat=Fri, Sabtu=Sat, Ahad=Sun.
- Times: 24-hour HH:MM. "8.30 malam" = 20:30, "lepas Maghrib"/"selepas Isyak" -> null time, put the phrase in time_note.
- date_evidence: copy the EXACT words from the text that state the date (short quote, verbatim).
- Recurring weekly classes: only include if a specific date inside the window is stated.
- confidence: "high" if date, venue and title are all explicit; "medium" if one is unclear; "low" otherwise.
- registration_url: the link to register/book/buy tickets (Google Forms, Peatix, Eventbrite, bit.ly, wa.me,
  the organiser's form...). Copy it EXACTLY as it appears in the text (see "LINKS FOUND"). Never build or guess a URL.
- in_klang_valley: true if the venue is in Kuala Lumpur, Putrajaya or Selangor's Klang Valley districts,
  false if it is elsewhere, null if you cannot tell.

Each event object has these keys:
title, is_islamic (bool), event_type, start_date, end_date, start_time, end_time, time_note,
venue, address, city, state, in_klang_valley (bool or null), is_online (bool), price_text, is_free (bool or null), speakers (list of names),
organizer, topic (max 20 words), registration_url, language, date_evidence, confidence"""


def page_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def build_user_prompt(page: dict, text: str, today: date, end: date) -> str:
    return (f"TODAY: {today.isoformat()} ({today.strftime('%A')})\n"
            f"WINDOW: {today.isoformat()} to {end.isoformat()}\n"
            f"PAGE URL: {page['url']}\nPAGE TITLE: {page.get('title', '')}\n"
            f"PUBLISHED: {page.get('published') or 'unknown'}\n\n"
            f"PAGE TEXT:\n{text}")


def extract(pages: list[dict], cfg: dict, cache: dict, today: date, end: date,
            stats: dict, warnings: list[str]) -> list[dict]:
    """Returns raw event dicts, each tagged with source_url and source_text."""
    llm_cfg = cfg["llm"]
    events: list[dict] = []
    hits = fresh = 0
    client = LLMClient(cfg) if cfg["secrets"]["llm"] else None
    ai_open = client is not None              # becomes False when the quota/time budget runs out
    if client is None:
        warnings.append("No LLM_API_KEY: AI extraction skipped (cached results only).")

    for page in pages:
        text = focus_text(page["text"], llm_cfg["max_chars_per_page"])
        h = page_hash(text)
        cached = cache.get(page["key"])
        page["ai_read"] = False
        if cached and cached.get("hash") == h:
            found = cached["events"]
            hits += 1
            page["ai_read"] = True
        else:
            if not ai_open or fresh >= llm_cfg["max_pages"]:
                continue                     # keep looping: later pages may still be in the cache
            try:
                result = client.chat_json(SYSTEM, build_user_prompt(page, text, today, end))
            except QuotaExhausted as e:
                warnings.append(f"AI limit reached ({e}); some pages were not read this week.")
                ai_open = False
                continue
            except ServiceError as e:   # provider problem (rate limit, network, 5xx): not the page's fault
                log.warning("AI service problem on %s: %s (will retry next run)", page["url"], e)
                continue
            except Exception as e:  # content problem (cut off, bad JSON, rejected request)
                log.warning("extraction failed for %s: %s", page["url"], e)
                fails = (cached or {}).get("fails", 0) + 1 if (cached or {}).get("content_hash") == h else 1
                # Retry next week, but after 2 failures on identical content stop paying for it.
                cache[page["key"]] = {"hash": h if fails >= 2 else "", "ts": today.isoformat(),
                                      "events": [], "fails": fails, "content_hash": h}
                continue
            fresh += 1
            page["ai_read"] = True
            found = result.get("events") if isinstance(result, dict) else None
            found = [e for e in (found or []) if isinstance(e, dict)][:10]
            cache[page["key"]] = {"hash": h, "ts": today.isoformat(), "events": found}
        for ev in found:
            ev = dict(ev)
            ev["source_url"] = page["url"]
            ev["source_text"] = text
            events.append(ev)

    stats.update(pages_ai_read=fresh, pages_cache_hits=hits, events_raw=len(events),
                 llm_calls=client.calls if client else 0, llm_tokens=client.tokens if client else 0)
    log.info("[extract] %d pages read by AI, %d from cache, %d raw events", fresh, hits, len(events))
    return events
