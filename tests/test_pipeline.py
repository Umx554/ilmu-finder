"""Offline tests: no API keys or network needed."""
import json
from datetime import date, time
from pathlib import Path

import pytest

from agent import deliver, main as agent_main, region, report
from agent.config import load_config, window, window_months
from agent.prefilter import focus_text, prefilter, score_page
from agent.search import build_queries, normalize_url, parse_telegram_channel
from agent.validate import coerce, ground, merge, same_event, validate

HERE = Path(__file__).parent
TODAY, END = date(2026, 11, 23), date(2026, 12, 23)


@pytest.fixture
def cfg():
    return load_config()


def raw_event(**kw):
    base = {
        "title": "Ceramah Perdana Maulidur Rasul", "is_islamic": True, "start_date": "2026-11-28",
        "start_time": "08:30", "end_time": "12:30", "venue": "Masjid Contoh", "city": "Shah Alam",
        "price_text": "RM20", "speakers": ["Ustaz Ahmad Contoh", "Ustaz Invented Person"],
        "date_evidence": "Sabtu, 28 November 2026", "confidence": "high",
        "registration_url": "https://example.org/daftar",
        "source_url": "https://example.org/a", "source_text": json.load(open(HERE / "sample_pages.json"))[0]["text"],
    }
    base.update(kw)
    return base


def test_window_and_queries(cfg):
    assert window(cfg) == (TODAY, END)
    assert window_months(cfg) == ([11, 12], [2026])
    qs = build_queries(cfg)
    assert any("Disember" in q for q in qs)
    assert len(qs) <= cfg["search"]["max_queries"]
    assert len(set(q.lower() for q in qs)) == len(qs)


def test_normalize_url():
    assert normalize_url("https://www.Example.org/x/?utm_source=fb&id=3#top") == "https://example.org/x?id=3"


def test_prefilter_drops_junk_and_old(cfg):
    pages = json.load(open(HERE / "sample_pages.json"))
    for p in pages:
        p["source"] = "search"
    s0, _ = score_page(pages[0], cfg)
    s1, why1 = score_page(pages[1], cfg)
    s2, why2 = score_page(pages[2], cfg)
    assert s0 > 0
    assert s1 == 0 and "year" in why1
    assert s2 == 0
    assert len(prefilter(pages, cfg, {})) == 1


def test_focus_text_keeps_event_lines():
    filler = "\n".join(f"Unrelated paragraph number {i} about nothing." for i in range(400))
    text = filler + "\nCeramah di Masjid Contoh pada 28 November 2026, 8.30 malam\n" + filler
    out = focus_text(text, 2000)
    assert len(out) <= 2000 and "Masjid Contoh" in out


def test_coerce_and_grounding():
    ev, why = coerce(raw_event(), TODAY, END)
    assert why == "ok"
    assert ev["start_time"] == time(8, 30)
    assert ev["speakers"] == ["Ustaz Ahmad Contoh"]           # real speaker kept
    assert ev["speakers_unverified"] == ["Ustaz Invented Person"]   # hallucinated speaker flagged
    assert ev["price_text"] == "RM20"
    assert ev["confidence"] == "medium"                       # downgraded because of the flag


def test_hallucinated_price_and_date_caught():
    ev, _ = coerce(raw_event(price_text="RM150", date_evidence="Ahad 29 November"), TODAY, END)
    assert ev["price_text"] is None
    assert ev["confidence"] == "low"


@pytest.mark.parametrize("kw,reason", [
    ({"start_date": "2026-11-01"}, "past"),
    ({"start_date": "2027-02-01"}, "beyond window"),
    ({"start_date": "next week"}, "no valid date"),
    ({"is_islamic": False}, "not Islamic"),
    ({"title": None}, "no title"),
])
def test_coerce_rejections(kw, reason):
    assert coerce(raw_event(**kw), TODAY, END) == (None, reason)


def test_multi_day_event_in_progress_is_kept():
    ev, why = coerce(raw_event(start_date="2026-11-21", end_date="2026-11-24"), TODAY, END)
    assert why == "ok"


def test_dedup_and_merge():
    a = raw_event(source_url="https://a.org", price_text=None)
    b = raw_event(title="Ceramah Perdana Maulidur Rasul 2026", source_url="https://b.org")
    out = validate([a, b], TODAY, END, {})
    assert len(out) == 1
    assert out[0]["price_text"] == "RM20"
    assert set(out[0]["sources"]) == {"https://a.org", "https://b.org"}
    assert out[0]["confidence"] == "high"


def test_different_events_same_day_not_merged():
    a, _ = coerce(raw_event(), TODAY, END)
    b, _ = coerce(raw_event(title="Kuliah Tafsir Mingguan", venue="Surau Lain"), TODAY, END)
    assert not same_event(a, b)


def test_telegram_channel_parser():
    html = """<div class="tgme_channel_info_header_title">Masjid Contoh</div>
    <div class="tgme_widget_message"><div class="tgme_widget_message_text">Kuliah Maghrib 30 Nov</div>
    <a class="tgme_widget_message_date" href="https://t.me/contoh/55"><time datetime="2099-01-01T10:00:00+00:00"></time></a></div>
    <div class="tgme_widget_message"><div class="tgme_widget_message_text">Old post</div>
    <a class="tgme_widget_message_date" href="https://t.me/contoh/1"><time datetime="2001-01-01T10:00:00+00:00"></time></a></div>"""
    posts = parse_telegram_channel(html, "https://t.me/s/contoh")
    assert len(posts) == 1 and posts[0]["url"] == "https://t.me/contoh/55"


def test_region_filter_by_text():
    def ev(**kw):
        base = {"is_online": False, "venue": None, "city": None, "address": None, "state": None, "ai_in_kv": None}
        base.update(kw)
        return base
    assert region.in_klang_valley(ev(venue="Dewan Besar", city="Shah Alam"))[0]
    assert region.in_klang_valley(ev(venue="Masjid Negara"))[0]
    assert not region.in_klang_valley(ev(venue="Masjid Kapitan Keling", city="George Town", state="Pulau Pinang"))[0]
    assert not region.in_klang_valley(ev(venue="Dewan Ipoh"))[0]
    assert region.in_klang_valley(ev(venue="Masjid Al-Hidayah", ai_in_kv=True))[0]
    assert not region.in_klang_valley(ev(venue="Masjid Al-Hidayah", ai_in_kv=False))[0]
    assert region.in_klang_valley(ev(is_online=True))[0]
    assert not region.in_klang_valley(ev(venue="Klinik Kluang"))[0]        # 'kl' must be a whole word


def test_registration_link_grounded():
    ev, _ = coerce(raw_event(), TODAY, END)
    assert ev["registration_url"] == "https://example.org/daftar"
    fake, _ = coerce(raw_event(registration_url="https://forms.gle/INVENTED"), TODAY, END)
    assert fake["registration_url"] is None


def test_registration_links_survive_trimming():
    filler = "\n".join(f"Unrelated paragraph number {i} about nothing." for i in range(400))
    text = filler + "\nCeramah 28 November\n" + filler + "\nDaftar di sini: https://forms.gle/abc123\n" + filler
    out = focus_text(text, 1500)
    assert "https://forms.gle/abc123" in out and len(out) <= 1500


def test_cards_buttons_and_calendar_link(cfg):
    ev, _ = coerce(raw_event(), TODAY, END)
    ev["id"] = "abc"
    digest = report.build_digest([ev], cfg, {}, TODAY, {}, [])
    text, rows, links = digest["cards"][0]
    labels = [b["text"] for row in rows for b in row]
    assert labels[0] == "📝 Register" and rows[0][0]["url"] == "https://example.org/daftar"
    assert "📅 Add to calendar" in labels
    gcal = [b["url"] for row in rows for b in row if "calendar.google.com" in b["url"]][0]
    assert "dates=20261128T003000Z%2F20261128T043000Z" in gcal          # 08:30–12:30 MYT
    assert "#1 🆕" in text and "📌 This week" in digest["header"] and "📝" in digest["header"]
    assert "example.org/daftar" in deliver.email_html(digest)


def test_split_respects_telegram_limit():
    msgs = report.split_text("\n".join(["x" * 1500] * 7))
    assert all(len(m) <= report.TG_LIMIT for m in msgs) and len(msgs) >= 3


def test_ics_is_valid(cfg):
    ev, _ = coerce(raw_event(title="Ceramah, dengan; koma"), TODAY, END)
    ev["id"] = "abc"
    ics = report.build_ics([ev], cfg)
    assert ics.startswith("BEGIN:VCALENDAR") and "DTSTART:20261128T003000Z" in ics   # 08:30 MYT = 00:30 UTC
    assert r"Ceramah\, dengan\; koma" in ics
    assert all(len(l.encode()) <= 75 for l in ics.split("\r\n"))


def test_end_to_end_offline(cfg, monkeypatch, tmp_path, capsys):
    """Full pipeline with the AI and map services mocked."""
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    monkeypatch.setenv("LLM_DELAY", "0")
    monkeypatch.setattr(agent_main, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("agent.state.STATE_DIR", tmp_path / "state")

    def fake_chat(self, system, user, max_tokens=1500):
        ev = {k: v for k, v in raw_event().items() if k not in ("source_url", "source_text")}
        return {"events": [ev]}

    monkeypatch.setattr("agent.llm.LLMClient.chat_json", fake_chat)

    class Args:
        no_send, pages_file = False, str(HERE / "sample_pages.json")   # no keys set -> printed, state saved

    assert agent_main.run(Args()) == 0
    out = capsys.readouterr().out
    assert "Ceramah Perdana Maulidur Rasul" in out and "RM20" in out
    assert "[📝 Register](https://example.org/daftar)" in out
    assert "🆕 Ceramah" in out
    events = json.loads((tmp_path / "out" / "events.json").read_text())
    assert len(events) == 1
    seen = json.loads((tmp_path / "state" / "seen.json").read_text())
    assert events[0]["id"] in seen

    # Second run: cached extraction is reused (no AI call) and the event is no longer "new".
    monkeypatch.setattr("agent.llm.LLMClient.chat_json", lambda *a, **k: pytest.fail("AI should not be called"))
    assert agent_main.run(Args()) == 0
    assert "🆕 Ceramah" not in capsys.readouterr().out
