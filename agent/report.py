"""Stage 5 - Reporter: build the overview, one card per event (with buttons), and the .ics calendar file."""
from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote_plus, urlencode
from zoneinfo import ZoneInfo

from .region import online_only

log = logging.getLogger("agent.report")
TG_LIMIT = 4000   # Telegram hard limit is 4096 characters


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def fd(d: date) -> str:
    """'Sat 28 Nov' (portable: %-d is not supported on Windows)."""
    return f"{d:%a} {d.day} {d:%b}"


def ft(t) -> str:
    return f"{t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'}"


def fmt_when(ev: dict) -> str:
    s, e = ev["start"], ev["end"]
    day = fd(s) if s == e else f"{fd(s)} – {fd(e)}"
    if ev.get("start_time"):
        t = ft(ev["start_time"])
        if ev.get("end_time"):
            t += f" – {ft(ev['end_time'])}"
        return f"{day}, {t}"
    return f"{day}, {ev['time_note']}" if ev.get("time_note") else day


def fmt_price(ev: dict) -> str | None:
    if ev.get("price_text"):
        return ev["price_text"]
    if ev.get("is_free") is True:
        return "Free"
    return None


def place_of(ev: dict) -> str:
    return ", ".join(filter(None, [ev.get("venue"), ev.get("city")]))


def watch_hit(ev: dict, watchlist: list[str]) -> bool:
    names = " ".join(ev["speakers"] + ev.get("speakers_unverified", [])).lower()
    return any(w.lower() in names for w in watchlist)


def sort_key(ev: dict):
    return (ev["start"], ev.get("start_time") or datetime.min.time(), ev["title"])


def event_times_utc(ev: dict, tz: ZoneInfo):
    """(start, end) as aware UTC datetimes, or None for all-day events."""
    if not ev.get("start_time"):
        return None
    start = datetime.combine(ev["start"], ev["start_time"], tz)
    end = datetime.combine(ev["end"], ev["end_time"], tz) if ev.get("end_time") else start + timedelta(hours=2)
    if end <= start:
        end = start + timedelta(hours=2)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def google_calendar_url(ev: dict, tz: ZoneInfo) -> str:
    """'Add to Google Calendar' link: works with no account setup or server."""
    times = event_times_utc(ev, tz)
    if times:
        dates = f"{times[0]:%Y%m%dT%H%M%SZ}/{times[1]:%Y%m%dT%H%M%SZ}"
    else:
        dates = f"{ev['start']:%Y%m%d}/{ev['end'] + timedelta(days=1):%Y%m%d}"
    details = "\n".join(filter(None, [
        ev.get("topic"), "Speakers: " + ", ".join(ev["speakers"]) if ev["speakers"] else None,
        "Register: " + ev["registration_url"] if ev.get("registration_url") else None,
        "Source: " + ev["sources"][0] if ev["sources"][0] else None]))
    q = {"action": "TEMPLATE", "text": ev["title"][:120], "dates": dates, "details": details[:300]}   # keep URL short
    if place_of(ev):
        q["location"] = place_of(ev)
    return "https://calendar.google.com/calendar/render?" + urlencode(q)


def buttons(ev: dict, tz: ZoneInfo) -> list[list[dict]]:
    """Telegram inline URL buttons (open links; need no bot server)."""
    rows = []
    if ev.get("registration_url"):
        rows.append([{"text": "📝 Register", "url": ev["registration_url"]}])
    second = []
    if ev["sources"][0]:
        second.append({"text": "🔗 Details" if ev.get("registration_url") else "🔗 Details / how to join",
                       "url": ev["sources"][0]})
    second.append({"text": "📅 Add to calendar", "url": google_calendar_url(ev, tz)})
    rows.append(second)
    if place_of(ev):
        rows.append([{"text": "🗺 Open in Maps",
                      "url": "https://www.google.com/maps/search/?api=1&query=" + quote_plus(place_of(ev))}])
    return rows


def is_online_only(ev: dict) -> bool:
    return online_only(ev)


def section_of(ev: dict, cfg: dict, week_end: date) -> str:
    if watch_hit(ev, cfg["speaker_watchlist"]):
        return "⭐ Watchlist speakers"
    if is_online_only(ev):
        return "💻 Online"
    return "📌 This week" if ev["start"] < week_end else "🗓 Coming up"


def card_text(ev: dict, n: int, cfg: dict, seen: dict) -> str:
    tags = ("🆕 " if ev["id"] not in seen else "") + ("⭐ " if watch_hit(ev, cfg["speaker_watchlist"]) else "")
    lines = [f"<b>#{n} {tags}{esc(ev['title'])}</b>", f"📅 {esc(fmt_when(ev))}"]
    if is_online_only(ev):
        lines.append("💻 Online" + (f" ({esc(ev['venue'])})" if ev.get("venue") else ""))
    else:
        lines.append(f"📍 {esc(place_of(ev))}" if place_of(ev) else "📍 Venue not stated – check details")
    price = fmt_price(ev)
    lines.append(f"💰 {esc(price)}" if price else "💰 Price not stated")
    if ev["speakers"]:
        lines.append("🎤 " + esc(", ".join(ev["speakers"])))
    if ev.get("speakers_unverified"):
        lines.append("🎤? " + esc(", ".join(ev["speakers_unverified"])) + " <i>(unverified)</i>")
    if ev.get("organizer"):
        lines.append("🏛 " + esc(ev["organizer"]))
    if ev.get("topic"):
        lines.append("📝 " + esc(ev["topic"]))
    if not ev.get("registration_url"):
        lines.append("<i>No registration link found – open Details.</i>")
    if ev["confidence"] == "low":
        lines.append("⚠️ <i>Low confidence – check the source before going</i>")
    return "\n".join(lines)


def card_html_links(ev: dict, tz: ZoneInfo) -> str:
    """Same buttons as Telegram, as links (for the e-mail version)."""
    return " · ".join(f'<a href="{esc(b["url"])}">{esc(b["text"])}</a>' for row in buttons(ev, tz) for b in row)


def build_digest(events: list[dict], cfg: dict, seen: dict, today: date, stats: dict,
                 warnings: list[str]) -> dict:
    """Returns {'header': str, 'cards': [(text, buttons, links_html)], 'footer': str}."""
    tz = ZoneInfo(cfg["timezone"])
    if not cfg.get("include_online", True):
        events = [e for e in events if not is_online_only(e)]
    week_end = today + timedelta(days=cfg["this_week_days"])
    order = {"⭐ Watchlist speakers": 0, "📌 This week": 1, "🗓 Coming up": 2, "💻 Online": 3}
    events = sorted(events, key=lambda e: (order[section_of(e, cfg, week_end)],) + sort_key(e))

    head = [f"<b>🕌 Islamic events – {fd(today)} {today.year}</b>",
            f"Klang Valley, next {cfg['window_days']} days · {len(events)} events"
            + (f" · {sum(e['id'] not in seen for e in events)} new" if events else "")]
    cards, section = [], None
    for n, ev in enumerate(events, 1):
        cards.append((card_text(ev, n, cfg, seen), buttons(ev, tz), card_html_links(ev, tz)))
        label = section_of(ev, cfg, week_end)
        if label != section:
            head.append(f"\n<b>{label}</b>")
            section = label
        reg = " 📝" if ev.get("registration_url") else ""
        head.append(f"{n}. {esc(fd(ev['start']))} · {esc(ev['title'][:70])}{reg}")
    if not events:
        head.append("\nNo events found this time. The agent ran; see the stats below.")
    else:
        head.append("\n<i>📝 = registration link available. Tap the buttons on each card below.</i>")

    s = stats
    foot = (f"<i>Run stats: {s.get('queries', 0)} searches · {s.get('pages_unique', 0)} pages · "
            f"{s.get('pages_prefiltered', 0)} relevant · {s.get('pages_ai_read', 0)} read by AI "
            f"({s.get('pages_cache_hits', 0)} cached) · {s.get('llm_tokens', 0):,} AI tokens. "
            f"🆕 = new since last week. Details are AI-extracted: confirm at the source.</i>")
    if warnings:
        foot += "\n⚠️ " + esc(" | ".join(dict.fromkeys(warnings)))
    return {"header": "\n".join(head), "cards": cards, "footer": foot}


def split_text(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split a long message on line breaks so each part fits Telegram's limit."""
    parts, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + 1 + len(line) > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return [p[:limit] for p in parts]


def to_plain(text: str) -> str:
    """Telegram-HTML -> plain text. Strip tags BEFORE decoding entities, so '&lt;Maulid&gt;' survives."""
    text = re.sub(r'<a href="([^"]+)">([^<]+)</a>', r"\2 (\1)", text)
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def to_text(digest: dict) -> str:
    """Plain-text version (e-mail fallback part)."""
    out = [to_plain(digest["header"])]
    for text, rows, _ in digest["cards"]:
        out.append(to_plain(text) + "\n" + "\n".join(f"{b['text']}: {b['url']}" for row in rows for b in row))
    out.append(to_plain(digest["footer"]))
    return "\n\n".join(out)


def to_markdown(digest: dict) -> str:
    out = [digest["header"]]
    for text, rows, _ in digest["cards"]:
        out.append(text + "\n" + " | ".join(f"[{b['text']}]({b['url']})" for row in rows for b in row))
    out.append(digest["footer"])
    md = "\n\n".join(out)
    md = re.sub(r"</?b>", "**", md)
    md = re.sub(r"</?i>", "_", md)
    return html.unescape(md)


def ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def fold(line: str) -> str:
    out, b = [], line.encode()
    while len(b) > 74:
        cut = 74
        while (b[cut] & 0xC0) == 0x80:     # don't split a UTF-8 character
            cut -= 1
        out.append(b[:cut].decode())
        b = b[cut:]
    out.append(b.decode())
    return "\r\n ".join(out)


def build_ics(events: list[dict], cfg: dict) -> str:
    tz = ZoneInfo(cfg["timezone"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//events-agent//KV Islamic events//EN",
             "CALSCALE:GREGORIAN", "X-WR-CALNAME:Klang Valley Islamic events"]
    for ev in sorted(events, key=sort_key):
        lines += ["BEGIN:VEVENT", f"UID:{ev['id']}@events-agent", f"DTSTAMP:{stamp}"]
        times = event_times_utc(ev, tz)
        if times:
            lines += [f"DTSTART:{times[0]:%Y%m%dT%H%M%SZ}", f"DTEND:{times[1]:%Y%m%dT%H%M%SZ}"]
        else:
            lines += [f"DTSTART;VALUE=DATE:{ev['start']:%Y%m%d}",
                      f"DTEND;VALUE=DATE:{ev['end'] + timedelta(days=1):%Y%m%d}"]
        desc = "\n".join(filter(None, [
            ev.get("topic"), "Speakers: " + ", ".join(ev["speakers"]) if ev["speakers"] else None,
            "Price: " + fmt_price(ev) if fmt_price(ev) else None,
            "Register: " + ev["registration_url"] if ev.get("registration_url") else None,
            ev["sources"][0], "AI-extracted, please verify at the source."]))
        lines += [fold("SUMMARY:" + ics_escape(ev["title"])), fold("DESCRIPTION:" + ics_escape(desc))]
        if place_of(ev):
            lines.append(fold("LOCATION:" + ics_escape(place_of(ev))))
        url = ev.get("registration_url") or ev["sources"][0]
        if url:
            lines.append(fold("URL:" + url))
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"
