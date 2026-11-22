"""A realistic end-to-end run of main.run() against simulated Tavily, Groq, Telegram and web pages.

Fixture pages (tests/fixtures/): a Malay mosque poster page, a public Telegram channel (t.me/s/...), a
Peatix-style conference page with three events and RM prices, a Penang forum (must be dropped), a 2024 page
(must never reach the AI) and an online webinar. The fake LLM answers like a real model would, including one
hallucinated speaker and one invented registration link, a 429 per-minute limit, ```json fences and prose.
"""
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import yaml

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
sys.path.insert(0, str(FIX))

from _fakehttp import CLEAR_ENV, FakeServer, HardTimeout, Reply, groq_error, openai_reply  # noqa: E402

from agent import deliver as deliver_mod  # noqa: E402
from agent import llm as llm_mod  # noqa: E402
from agent import main as agent_main  # noqa: E402
from agent import region, search  # noqa: E402
from agent.config import ROOT, load_config  # noqa: E402
from agent.search import build_queries  # noqa: E402
from agent.validate import coerce  # noqa: E402

TODAY, END = date(2026, 11, 23), date(2026, 12, 23)
TOKEN, CHAT = "555000111:AAH-realistic_test", "424242"

PEATIX_URL = "https://peatix.com/event/4471203"
PENANG_URL = "https://www.berita-semasa.my/agama/2026/11/forum-perdana-generasi-muda-al-quran"
OLD_URL = "https://www.example-univ.edu.my/news/seminar-fiqh-muamalat-2024"
WEBINAR_URL = "https://darulilmu.org.my/webinar/kewangan-islam-keluarga"
TG_KULIAH, TG_DAURAH = "https://t.me/alhidayahbbb/1241", "https://t.me/alhidayahbbb/1247"
MOSQUE_URL = "https://masjidalfalah-usj9.example.my/program-ilmu-disember-2026"
CHANNEL_URL = "https://t.me/s/alhidayahbbb"
INVENTED_LINK = "https://forms.gle/AlFalahCeramah2026"
HALLUCINATED_SPEAKER = "Ustazah Nur Aisyah Rahman"


def ev(**kw):
    """One event object the way llama-3.3-70b tends to return it (every key present, nulls included)."""
    base = dict(title=None, is_islamic=True, event_type="talk", start_date=None, end_date=None, start_time=None,
                end_time=None, time_note=None, venue=None, address=None, city=None, state=None, in_klang_valley=None,
                is_online=False, price_text=None, is_free=None, speakers=[], organizer=None, topic=None,
                registration_url=None, language="ms", date_evidence=None, confidence="high")
    base.update(kw)
    return base


LLM_ANSWERS = {
    MOSQUE_URL: ("fenced", [ev(
        title="Ceramah Perdana: Akhlak Muslim di Era Digital", event_type="ceramah", start_date="2026-12-05",
        start_time="20:30", time_note="selepas solat Isyak", venue="Dewan Solat Utama, Masjid Al-Falah",
        address="USJ 9", city="Subang Jaya", state="Selangor", in_klang_valley=True, price_text="Percuma",
        is_free=True, speakers=["Ustaz Mohd Faizal bin Hamzah", HALLUCINATED_SPEAKER],     # 2nd one is invented
        organizer="Jawatankuasa Pengurusan Masjid Al-Falah", topic="Akhlak Muslim di era digital",
        registration_url=INVENTED_LINK,                                                    # not on the page
        date_evidence="Tarikh: Sabtu, 5 Disember 2026")]),
    TG_KULIAH: ("plain", [ev(
        title="Kuliah Maghrib Istimewa: Tadabbur Surah Al-Kahfi", event_type="kuliah", start_date="2026-11-27",
        time_note="Selepas solat Maghrib", venue="Dewan Solat Utama, Masjid Al-Hidayah", city="Bandar Baru Bangi",
        state="Selangor", in_klang_valley=True, price_text="Percuma", is_free=True,
        speakers=["Ustaz Abdul Rahman Idris"], organizer="Masjid Al-Hidayah Bandar Baru Bangi",
        topic="Tadabbur Surah Al-Kahfi", date_evidence="Jumaat, 27 November 2026")]),
    TG_DAURAH: ("plain", [ev(
        title="Daurah Ilmu Muslimah: Fiqh Solat untuk Wanita", event_type="daurah", start_date="2026-12-13",
        start_time="09:00", end_time="13:00", venue="Dewan Seminar Aras 1, Masjid Al-Hidayah",
        city="Bandar Baru Bangi", state="Selangor", in_klang_valley=True, price_text="RM30", is_free=False,
        speakers=["Ustazah Siti Khadijah Mohd Noor"], topic="Fiqh solat khusus untuk wanita",
        registration_url="https://forms.gle/Xy7kLmN2pQ8vR3wE9", date_evidence="Ahad, 13 Disember 2026")]),
    PEATIX_URL: ("plain", [
        ev(title="Kuala Lumpur Islamic Finance & Halal Economy Summit 2026", event_type="summit", language="en",
           start_date="2026-12-09", end_date="2026-12-10", start_time="09:00", end_time="17:00",
           venue="Kuala Lumpur Convention Centre", address="Jalan Pinang, 50088 Kuala Lumpur", city="Kuala Lumpur",
           in_klang_valley=True, price_text="RM150 – RM250", is_free=False,
           speakers=["Dr. Aisha Rahman Malik", "Encik Hafiz Zulkifli", "Prof. Dr. Ahmad Kamil Yusof"],
           organizer="Malaysian Islamic Finance Professionals Association",
           topic="Sukuk, takaful, zakat, waqf and the global halal economy",
           registration_url="https://klifhes2026.peatix.com/",
           date_evidence="Wed, 9 Dec 2026 – Thu, 10 Dec 2026"),
        ev(title="Pre-Summit Workshop: Zakat & Waqf Fintech", event_type="workshop", language="en",
           start_date="2026-12-08", start_time="14:00", end_time="17:30",
           venue="Seminar Room 3, Kuala Lumpur Convention Centre", city="Kuala Lumpur", in_klang_valley=True,
           price_text="RM 99", is_free=False, registration_url="https://klifhes2026.peatix.com/",
           date_evidence="Tue, 8 Dec 2026, 2:00 PM – 5:30 PM", confidence="medium"),
        ev(title="Halal Economy Networking Dinner", event_type="dinner", language="en", start_date="2026-12-09",
           start_time="20:00", venue="Grand Ballroom, Kuala Lumpur Convention Centre", city="Kuala Lumpur",
           in_klang_valley=True, price_text="RM 120 per pax", is_free=False,
           date_evidence="Wed, 9 Dec 2026, 8:00 PM", confidence="medium"),
    ]),
    PENANG_URL: ("plain", [ev(
        title="Forum Perdana Hal Ehwal Islam: Generasi Muda dan Al-Quran", event_type="forum",
        start_date="2026-12-12", start_time="21:00", venue="Masjid Negeri Pulau Pinang", address="Jalan Air Itam",
        city="George Town", state="Pulau Pinang", in_klang_valley=False, is_free=True,
        speakers=["Ustaz Kamal Ariffin", "Ustaz Syed Hakim Al-Idrus"],
        date_evidence="Sabtu, 12 Disember 2026")]),
    WEBINAR_URL: ("prose", [ev(
        title="Webinar: Pengurusan Kewangan Islam Untuk Keluarga", event_type="webinar", start_date="2026-11-26",
        start_time="21:00", end_time="22:30", is_online=True, price_text="Percuma", is_free=True,
        speakers=["Ustazah Dr. Nurul Huda Kamaruddin"], organizer="Pertubuhan Darul Ilmu Malaysia",
        topic="Bajet keluarga, hutang, zakat pendapatan dan wasiat",
        registration_url="https://bit.ly/WebinarKewanganIslam26", date_evidence="Khamis, 26 November 2026")]),
}


def tavily_item(url, title, text, raw=True, published=None):
    return {"url": url, "title": title, "content": text[:180], "raw_content": text if raw else None,
            "score": 0.8, "published_date": published}


def tavily_pages(queries):
    """Results for the first three configured queries; every other query finds nothing."""
    peatix = (FIX / "peatix_summit.txt").read_text(encoding="utf-8")
    return {
        queries[0]: [tavily_item(PEATIX_URL, "KLIFHES 2026 | Peatix", peatix, published="2026-10-28"),
            tavily_item(PENANG_URL, "Forum Perdana di Pulau Pinang", (FIX / "penang_forum.txt").read_text("utf-8"))],
        queries[1]: [tavily_item("https://www.peatix.com/event/4471203/?utm_source=google&utm_medium=organic",
                        "KLIFHES 2026", peatix, raw=False),               # same page, snippet only -> dedup
            tavily_item(OLD_URL, "Seminar Fiqh Muamalat 2024", (FIX / "old_2024_seminar.txt").read_text("utf-8"),
                        published="2024-11-02"),
            tavily_item(WEBINAR_URL, "Webinar Kewangan Islam", (FIX / "webinar.txt").read_text("utf-8"))],
        queries[2]: [],
    }


class World:
    """All fake services behind one local HTTP server."""

    def __init__(self, server: FakeServer, queries: list[str]):
        self.server, self.queries = server, queries
        self.results = tavily_pages(queries)
        self.llm_prompts: list[str] = []
        self.llm_429_sent = False
        self.mosque = (FIX / "mosque_ceramah.html").read_bytes()
        self.channel = (FIX / "telegram_channel.html").read_bytes()
        server.handler = self.handle

    def handle(self, req):
        if req.path == "/search":
            q = req.json()["query"]
            return Reply(200, {"query": q, "results": self.results.get(q, []), "response_time": 0.9})
        if req.path == "/masjidalfalah-usj9.example.my/program-ilmu-disember-2026":
            return Reply(200, self.mosque)
        if req.path == "/t.me/s/alhidayahbbb":
            return Reply(200, self.channel)
        if req.path == "/openai/v1/chat/completions":
            return self.llm(req)
        if req.path.startswith(f"/bot{TOKEN}/"):
            return Reply(200, {"ok": True, "result": {"message_id": len(self.server.requests)}})
        return Reply(404, "not found")

    def llm(self, req):
        if not self.llm_429_sent:                        # first call hits the per-minute limit once
            self.llm_429_sent = True
            return groq_error(429, "Rate limit reached for model `llama-3.3-70b-versatile` on tokens per minute "
                                   "(TPM): Limit 12000, Used 11500, Requested 1700. Please try again in 1.2s.",
                              code="rate_limit_exceeded", etype="tokens", headers={"retry-after": "2"})
        body = req.json()
        user = body["messages"][1]["content"]
        self.llm_prompts.append(user)
        page_url = re.search(r"^PAGE URL: (.+)$", user, re.M).group(1)
        style, events = next(v for k, v in LLM_ANSWERS.items() if page_url.endswith(k))
        js = json.dumps({"events": events}, ensure_ascii=False)
        content = {"fenced": f"```json\n{js}\n```", "plain": js,
                   "prose": f"Here are the events found on the page:\n{js}\nAll dates are in the window."}[style]
        return openai_reply(content, model=body["model"], tokens=(len(user) // 4, 350))

    def tg_messages(self):
        return [r.form() for r in self.server.requests if r.path.endswith("/sendMessage")]

    def count(self, path):
        return sum(r.path == path for r in self.server.requests)


@pytest.fixture
def world(monkeypatch, tmp_path):
    for k in CLEAR_ENV:
        monkeypatch.delenv(k, raising=False)
    server = FakeServer().start()
    cfg_data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg_data["direct_sources"] = [MOSQUE_URL, CHANNEL_URL]
    cfg_data["speaker_watchlist"] = ["Abdul Rahman Idris"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")
    env = {"CONFIG_FILE": str(tmp_path / "config.yaml"), "TAVILY_API_KEY": "tvly-dev-fake", "MAX_QUERIES": "40",
           "LLM_API_KEY": "gsk_fake", "LLM_BASE_URL": server.url + "/openai/v1", "LLM_DELAY": "0",
           "TELEGRAM_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": CHAT,
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    monkeypatch.setattr(search, "TAVILY_URL", server.url + "/search")
    real_get = requests.Session.get        # direct sources: https://host/path -> fake server /host/path
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: real_get(
        self, server.url + "/" + url.split("://", 1)[1], **kw))
    real_post = requests.post
    monkeypatch.setattr(requests, "post", lambda url, **kw: real_post(
        server.url + url[len("https://api.telegram.org"):] if url.startswith("https://api.telegram.org/") else url, **kw))
    sleeps: list[float] = []
    fake_time = SimpleNamespace(sleep=sleeps.append, monotonic=time.monotonic)
    monkeypatch.setattr(llm_mod, "time", fake_time)
    monkeypatch.setattr(deliver_mod, "time", fake_time)
    monkeypatch.setattr(agent_main, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("agent.state.STATE_DIR", tmp_path / "state")

    w = World(server, build_queries(load_config(), cap=False))
    w.sleeps, w.out = sleeps, tmp_path / "out"
    with HardTimeout(30):
        yield w
    server.stop()
    assert not server.errors, server.errors[0]


class Args:
    no_send, pages_file = False, None


def card_by_title(msgs, title):
    hits = [m for m in msgs if "reply_markup" in m and title in m["text"]]
    assert len(hits) == 1, (title, [m["text"][:60] for m in msgs])
    return hits[0]


def button_labels(msg):
    return [b["text"] for row in json.loads(msg["reply_markup"])["inline_keyboard"] for b in row]


def test_full_run_realistic_digest_and_cached_second_run(world, caplog):
    caplog.set_level("INFO")
    assert agent_main.run(Args()) == 0

    # ---------------------------------------------------------------- what reached the AI
    prompts = "\n".join(world.llm_prompts)
    assert len(world.llm_prompts) == 6                          # mosque, 2 telegram posts, peatix, penang, webinar
    assert world.count("/openai/v1/chat/completions") == 7      # + one 429 retry
    assert 2.0 in world.sleeps
    assert "Seminar Fiqh Muamalat" not in prompts                # 2024 page dropped by the prefilter
    assert "OldOgos2026" not in prompts and "17 Ogos 2026" not in prompts   # old Telegram post never read
    daurah_prompt = next(p for p in world.llm_prompts if TG_DAURAH in p)
    assert "LINKS FOUND:\nhttps://forms.gle/Xy7kLmN2pQ8vR3wE9" in daurah_prompt
    mosque_prompt = next(p for p in world.llm_prompts if "masjidalfalah" in p)
    assert "TODAY: 2026-11-23 (Monday)" in mosque_prompt and "Tarikh: Sabtu, 5 Disember 2026" in mosque_prompt
    assert INVENTED_LINK not in mosque_prompt                    # so the AI really did invent it
    assert world.count("/search") == 34

    # ---------------------------------------------------------------- Telegram messages
    msgs = world.tg_messages()
    assert len(msgs) == 1 + 7 + 1 and world.count(f"/bot{TOKEN}/sendDocument") == 1
    header, cards, footer = msgs[0], msgs[1:-1], msgs[-1]
    assert all(m["parse_mode"] == "HTML" for m in msgs)          # nothing had to fall back to plain text

    expected_header = "\n".join([
        "<b>🕌 Islamic events – Mon 23 Nov 2026</b>",
        "Klang Valley, next 30 days · 7 events · 7 new",
        "", "<b>⭐ Watchlist speakers</b>",
        "1. Fri 27 Nov · Kuliah Maghrib Istimewa: Tadabbur Surah Al-Kahfi",
        "", "<b>🗓 Coming up</b>",                                  # no other event starts before 30 Nov
        "2. Sat 5 Dec · Ceramah Perdana: Akhlak Muslim di Era Digital",
        "3. Tue 8 Dec · Pre-Summit Workshop: Zakat &amp; Waqf Fintech 📝",
        "4. Wed 9 Dec · Kuala Lumpur Islamic Finance &amp; Halal Economy Summit 2026 📝",
        "5. Wed 9 Dec · Halal Economy Networking Dinner",
        "6. Sun 13 Dec · Daurah Ilmu Muslimah: Fiqh Solat untuk Wanita 📝",
        "", "<b>💻 Online</b>",
        "7. Thu 26 Nov · Webinar: Pengurusan Kewangan Islam Untuk Keluarga 📝",
        "", "<i>📝 = registration link available. Tap the buttons on each card below.</i>"])
    assert header["text"] == expected_header
    assert "Pulau Pinang" not in json.dumps(msgs, ensure_ascii=False)   # Penang event dropped
    assert "Fiqh Muamalat" not in json.dumps(msgs, ensure_ascii=False)

    # cards are numbered in the same order as the overview
    nums = [re.match(r"<b>#(\d+) ", c["text"]).group(1) for c in cards]
    assert nums == [str(i) for i in range(1, 8)]
    assert all("🆕" in c["text"] for c in cards)
    assert cards[0]["text"].startswith("<b>#1 🆕 ⭐ Kuliah Maghrib Istimewa")

    # Register buttons exactly where the page really had a registration link
    with_register = {re.sub(r"<b>#\d+ (🆕 )?(⭐ )?", "", c["text"].split("</b>")[0])
                     for c in cards if "📝 Register" in button_labels(c)}
    assert with_register == {"Webinar: Pengurusan Kewangan Islam Untuk Keluarga",
                             "Pre-Summit Workshop: Zakat &amp; Waqf Fintech",
                             "Kuala Lumpur Islamic Finance &amp; Halal Economy Summit 2026",
                             "Daurah Ilmu Muslimah: Fiqh Solat untuk Wanita"}

    ceramah = card_by_title(cards, "Ceramah Perdana")
    kb = json.loads(ceramah["reply_markup"])["inline_keyboard"]
    assert INVENTED_LINK not in ceramah["reply_markup"] and INVENTED_LINK not in ceramah["text"]
    assert "📝 Register" not in button_labels(ceramah)
    assert "<i>No registration link found – open Details.</i>" in ceramah["text"]
    assert kb[0][0] == {"text": "🔗 Details / how to join", "url": MOSQUE_URL}
    assert "📅 Sat 5 Dec, 8:30 PM" in ceramah["text"] and "💰 Percuma" in ceramah["text"]
    assert "📍 Dewan Solat Utama, Masjid Al-Falah, Subang Jaya" in ceramah["text"]
    assert "🎤 Ustaz Mohd Faizal bin Hamzah\n" in ceramah["text"]
    assert f"🎤? {HALLUCINATED_SPEAKER} <i>(unverified)</i>" in ceramah["text"]
    assert "⚠️" not in ceramah["text"]                          # flagged, but date was verified -> not low

    daurah = card_by_title(cards, "Daurah Ilmu Muslimah")
    assert json.loads(daurah["reply_markup"])["inline_keyboard"][0] == [
        {"text": "📝 Register", "url": "https://forms.gle/Xy7kLmN2pQ8vR3wE9"}]
    assert "💰 RM30" in daurah["text"] and "📅 Sun 13 Dec, 9:00 AM – 1:00 PM" in daurah["text"]
    assert "🔗 Details" in button_labels(daurah)

    kuliah = cards[0]
    assert "📅 Fri 27 Nov, Selepas solat Maghrib" in kuliah["text"]
    assert "🗺 Open in Maps" in button_labels(kuliah)

    webinar = card_by_title(cards, "Webinar: Pengurusan")
    assert "💻 Online" in webinar["text"] and "🗺 Open in Maps" not in button_labels(webinar)
    assert "📅 Thu 26 Nov, 9:00 PM – 10:30 PM" in webinar["text"]

    summit = card_by_title(cards, "Halal Economy Summit")
    assert "📅 Wed 9 Dec – Thu 10 Dec, 9:00 AM – 5:00 PM" in summit["text"]
    assert "💰 RM150 – RM250" in summit["text"]
    assert "🎤 Dr. Aisha Rahman Malik, Encik Hafiz Zulkifli, Prof. Dr. Ahmad Kamil Yusof" in summit["text"]
    gcal = next(b["url"] for row in json.loads(summit["reply_markup"])["inline_keyboard"] for b in row
                if b["text"] == "📅 Add to calendar")
    assert "dates=20261209T010000Z%2F20261210T090000Z" in gcal

    dinner = card_by_title(cards, "Networking Dinner")
    assert "💰 RM 120 per pax" in dinner["text"] and "📝 Register" not in button_labels(dinner)

    assert "Run stats: 34 searches · 7 pages · 6 relevant · 6 read by AI (0 cached)" in footer["text"]
    assert "⚠️" not in footer["text"]
    assert "💡" not in footer["text"]          # the Telegram channel already is a direct source: no suggestion

    # ---------------------------------------------------------------- files written
    events = json.loads((world.out / "events.json").read_text(encoding="utf-8"))
    assert len(events) == 7
    by_title = {e["title"]: e for e in events}
    c = by_title["Ceramah Perdana: Akhlak Muslim di Era Digital"]
    assert c["registration_url"] is None and c["speakers_unverified"] == [HALLUCINATED_SPEAKER]
    assert "registration link not found in source; removed" in c["flags"] and c["confidence"] == "medium"
    assert by_title["Kuala Lumpur Islamic Finance & Halal Economy Summit 2026"]["flags"] == []
    stats = json.loads((world.out / "stats.json").read_text(encoding="utf-8"))["stats"]
    assert stats["pages_raw"] == 8 and stats["pages_unique"] == 7
    assert stats["prefilter_drops"] == {"only old/other years": 1}
    assert stats["region_drops"] == {"other state": 1}
    assert stats["llm_calls"] == 6 and stats["llm_tokens"] > 0
    assert stats["delivered_via"] == ["telegram"] and stats["source_suggestions"] == []
    ics = (world.out / "events.ics").read_text(encoding="utf-8")
    assert ics.count("BEGIN:VEVENT") == 7 and INVENTED_LINK not in ics
    assert "**#1 🆕 ⭐ Kuliah Maghrib Istimewa" in (world.out / "digest.md").read_text(encoding="utf-8")

    # ---------------------------------------------------------------- second run: all from cache
    world.server.requests.clear()
    world.llm_prompts.clear()
    assert agent_main.run(Args()) == 0
    assert world.count("/openai/v1/chat/completions") == 0
    assert world.count("/search") == 34
    msgs2 = world.tg_messages()
    assert len(msgs2) == 9
    assert not any("🆕" in m["text"] for m in msgs2[:-1])          # footer legend still explains 🆕
    assert "Klang Valley, next 30 days · 7 events · 0 new" in msgs2[0]["text"]
    assert msgs2[0]["text"].replace(" · 0 new", " · 7 new") == expected_header
    assert msgs2[1]["text"].startswith("<b>#1 ⭐ Kuliah Maghrib Istimewa")
    assert "0 read by AI (6 cached) · 0 AI tokens" in msgs2[-1]["text"]


def test_no_send_dry_run_does_not_mark_events_seen(world, capsys):
    class DryRun:
        no_send, pages_file = True, None
    assert agent_main.run(DryRun()) == 0
    out = capsys.readouterr().out
    assert "**#1 🆕 ⭐ Kuliah Maghrib Istimewa" in out and world.tg_messages() == []
    assert agent_main.run(DryRun()) == 0
    assert "**#1 🆕 ⭐ Kuliah Maghrib Istimewa" in capsys.readouterr().out   # still new: dry runs don't touch "seen"
    assert world.count("/openai/v1/chat/completions") == 7      # but the extraction cache was used


def test_llm_outage_still_delivers_partial_digest(world):
    """The AI endpoint is down: the run must still send a (0-event) digest with stats and a warning."""
    handle = world.handle
    world.server.handler = lambda req: (Reply(503, {"error": {"message": "Service Unavailable"}})
                                        if req.path.startswith("/openai") else handle(req))
    assert agent_main.run(Args()) == 0
    msgs = world.tg_messages()
    assert len(msgs) == 2                                         # header + footer, no cards
    assert "0 events" in msgs[0]["text"] and "No events found this time" in msgs[0]["text"]
    assert world.count(f"/bot{TOKEN}/sendDocument") == 0


# ---------------------------------------------------------------- regressions found on realistic input (now fixed)
def _raw(answer_key, idx=0, **over):
    """A raw LLM event for one of the fixture pages, as extract() would hand it to validate()."""
    texts = {WEBINAR_URL: (FIX / "webinar.txt").read_text("utf-8"),
             PEATIX_URL: (FIX / "peatix_summit.txt").read_text("utf-8"),
             TG_KULIAH: ("📢\nKULIAH MAGHRIB ISTIMEWA\n🗓\nJumaat, 27 November 2026\n🕖 Selepas solat Maghrib\n📍 Dewan "
                         "Solat Utama, Masjid Al-Hidayah, Bandar Baru Bangi\n🎙 Ustaz Abdul Rahman Idris\nTajuk:\n"
                         "Tadabbur Surah Al-Kahfi\nTerbuka kepada semua. Percuma. Jemput hadir dan sebarkan 🤲")}
    raw = dict(LLM_ANSWERS[answer_key][1][idx])
    raw.update(over, source_url=answer_key, source_text=texts[answer_key])
    return raw


def test_online_webinar_with_zoom_venue_is_kept():
    e, why = coerce(_raw(WEBINAR_URL, venue="Zoom", city=None), TODAY, END)
    assert why == "ok"
    assert region.in_klang_valley(e)[0]


def test_online_events_have_their_own_section(monkeypatch):
    for k in CLEAR_ENV:
        monkeypatch.delenv(k, raising=False)
    from agent.report import build_digest
    e, _ = coerce(_raw(WEBINAR_URL), TODAY, END)
    e["id"] = "w"
    header = build_digest([e], load_config(), {}, TODAY, {}, [])["header"]
    assert re.search(r"<b>[^<]*Online[^<]*</b>", header)


def test_hallucinated_price_on_free_event_is_removed():
    e, _ = coerce(_raw(TG_KULIAH, price_text="RM20", is_free=False), TODAY, END)
    assert e["price_text"] is None


def test_truncated_registration_link_is_removed():
    e, _ = coerce(_raw(PEATIX_URL, registration_url="https://peatix.com"), TODAY, END)
    assert e["registration_url"] is None


def test_miscased_short_link_is_removed():
    raw = _raw(WEBINAR_URL, registration_url="https://bit.ly/webinarkewanganislam26")
    e, _ = coerce(raw, TODAY, END)
    assert e["registration_url"] is None
