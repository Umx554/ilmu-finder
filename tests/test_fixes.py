"""Focused regression tests for the fixes made after the first HTTP/realistic test round.

One small test (or parametrised group) per behaviour change in agent/, plus an end-to-end run without
--pages-file that exercises the learning loop (query tagging, state/query_stats.json, source suggestions).
"""
import json
import sys
import time
from datetime import date, time as dtime
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
from agent import extract as extract_mod  # noqa: E402
from agent import learning  # noqa: E402
from agent import llm as llm_mod  # noqa: E402
from agent import main as agent_main  # noqa: E402
from agent import region, report, search  # noqa: E402
from agent.config import ROOT, load_config  # noqa: E402
from agent.llm import BadResponse, LLMClient, QuotaExhausted, ServiceError, parse_json  # noqa: E402
from agent.prefilter import focus_text, month_hit, prefilter, score_page  # noqa: E402
from agent.validate import coerce, ground, validate  # noqa: E402

TODAY, END = date(2026, 11, 23), date(2026, 12, 23)
MAIN, FALLBACK = "openai/gpt-oss-120b", "openai/gpt-oss-20b"
EVENTS = {"events": [{"title": "Kuliah", "start_date": "2026-11-27"}]}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for k in CLEAR_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with HardTimeout(20):
        yield


@pytest.fixture
def server():
    s = FakeServer().start()
    yield s
    s.stop()
    assert not s.errors, s.errors[0]


@pytest.fixture
def sleeps(monkeypatch):
    rec: list[float] = []
    fake_time = SimpleNamespace(sleep=rec.append, monotonic=time.monotonic)
    monkeypatch.setattr(llm_mod, "time", fake_time)
    monkeypatch.setattr(deliver_mod, "time", fake_time)
    return rec


def make_cfg(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return load_config()


@pytest.fixture
def llm_cfg(server, monkeypatch):
    return make_cfg(monkeypatch, LLM_BASE_URL=server.url, LLM_API_KEY="k", LLM_DELAY="0")


def script(server, *replies):
    queue = list(replies)

    def handler(req):
        if not queue:
            raise AssertionError("unexpected extra request")
        return queue.pop(0)
    server.handler = handler


# ================================================================= llm.py
def test_llm_max_output_tokens_from_config(server, llm_cfg, sleeps):
    llm_cfg["llm"]["max_output_tokens"] = 2500
    script(server, openai_reply("{}"), openai_reply("{}"))
    c = LLMClient(llm_cfg)
    c.chat_json("s", "u")
    c.chat_json("s", "u", max_tokens=700)
    assert [r.json()["max_tokens"] for r in server.requests] == [2500, 700]


def test_llm_cut_off_reply_raises_bad_response(server, llm_cfg, sleeps):
    rep = json.loads(openai_reply('{"events": [{"title": "A"', tokens=(3000, 4000)).body)
    rep["choices"][0]["finish_reason"] = "length"
    script(server, Reply(200, rep))
    c = LLMClient(llm_cfg)
    with pytest.raises(BadResponse):
        c.chat_json("s", "u")
    assert c.tokens == 7000 and len(server.requests) == 1     # tokens are still counted


@pytest.mark.parametrize("content,expected", [
    ('{"events": []}', {"events": []}),
    ('```json\n{"events": [{"title": "A"}]}\n```', {"events": [{"title": "A"}]}),
    ('```\n{"events": []}\n```', {"events": []}),
    ('Sure:\n{"events": [{"title": "A"}]}\nDone.', {"events": [{"title": "A"}]}),
    ('[{"title": "A"}, {"title": "B"}]', {"events": [{"title": "A"}, {"title": "B"}]}),
    ('"just a string"', None),
    ("not json at all", None),
    ("", None),
])
def test_parse_json_variants(content, expected):
    assert parse_json(content) == expected


@pytest.mark.parametrize("content,n", [
    ('Here are the events:\n[{"title": "A"}]', 1),
    ('Here are the events:\n[{"title": "A"}, {"title": "B"}]\nHope this helps.', 2),
])
def test_parse_json_prose_wrapped_list(content, n):
    got = parse_json(content)
    assert isinstance(got, dict) and len(got.get("events", [])) == n


def test_llm_network_error_is_retried(server, llm_cfg, sleeps, monkeypatch):
    real_post = requests.Session.post
    calls = []

    def flaky(self, url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ConnectionError("connection reset by peer")
        return real_post(self, url, **kw)
    monkeypatch.setattr(requests.Session, "post", flaky)
    script(server, openai_reply(json.dumps(EVENTS)))
    assert LLMClient(llm_cfg).chat_json("s", "u") == EVENTS
    assert len(calls) == 2 and sleeps == [10]


def test_llm_network_down_is_bounded(monkeypatch, sleeps):
    cfg = make_cfg(monkeypatch, LLM_BASE_URL="http://127.0.0.1:1/v1", LLM_API_KEY="k", LLM_DELAY="0")
    with pytest.raises(ServiceError):
        LLMClient(cfg).chat_json("s", "u")
    assert sleeps == [10] * 8


def test_llm_time_budget_turns_long_wait_into_quota_exhausted(server, llm_cfg, sleeps):
    llm_cfg["llm"]["time_budget_minutes"] = 1
    script(server, groq_error(429, "Rate limit reached on tokens per minute (TPM). Please try again in 2m.",
                              headers={"retry-after": "120"}))
    with pytest.raises(QuotaExhausted):
        LLMClient(llm_cfg).chat_json("s", "u")
    assert sleeps == [] and len(server.requests) == 1


@pytest.mark.parametrize("message,daily", [
    ("Rate limit reached for model `m` on requests per day (RPD): Limit 1000, Used 1000.", True),
    ("Rate limit reached for model `m` on tokens per day (TPD): Limit 100000.", True),
    ("Limit (TPD) reached", True),
    ("Rate limit reached on tokens per minute (TPM). Upgrade to Dev Tier today at console.groq.com", False),
    ("Rate limit reached on requests per minute (RPM): Limit 30. Please try again in 2s.", False),
])
def test_llm_daily_limit_detection(server, llm_cfg, sleeps, message, daily):
    script(server, groq_error(429, message, code="rate_limit_exceeded", headers={"retry-after": "1"}),
           openai_reply("{}"))
    LLMClient(llm_cfg).chat_json("s", "u")
    assert [r.json()["model"] for r in server.requests] == ([MAIN, FALLBACK] if daily else [MAIN, MAIN])


def test_llm_400_without_model_gone_code_does_not_fall_back(server, llm_cfg, sleeps):
    script(server, groq_error(400, "Failed to generate JSON for model `x`.", code="json_validate_failed"))
    c = LLMClient(llm_cfg)
    with pytest.raises(requests.HTTPError):
        c.chat_json("s", "u")
    assert c.models == [MAIN, FALLBACK]


# ================================================================= extract.py
def _pages(n):
    return [{"url": f"https://example.my/p{i}", "key": f"https://example.my/p{i}", "title": "t",
             "text": f"Ceramah nombor {i} di masjid pada 30 November 2026"} for i in range(n)]


def test_extract_skips_bad_response_pages_without_caching(monkeypatch):
    cfg = make_cfg(monkeypatch, LLM_API_KEY="k", LLM_DELAY="0")
    answers = [BadResponse("cut off"), {"events": [{"title": "B"}]}]

    def fake(self, system, user, max_tokens=None):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(LLMClient, "chat_json", fake)
    cache, stats = {}, {}
    out = extract_mod.extract(_pages(2), cfg, cache, TODAY, END, stats, [])
    assert [e["title"] for e in out] == ["B"]
    assert cache["https://example.my/p1"]["events"] == [{"title": "B"}]
    failed = cache["https://example.my/p0"]
    assert failed["hash"] == "" and failed["fails"] == 1       # not a cache hit: retried next week
    assert stats["pages_ai_read"] == 1


def test_extract_gives_up_on_page_after_two_failures(monkeypatch):
    """A page whose reply is always cut off stops costing tokens after 2 tries (same content)."""
    cfg = make_cfg(monkeypatch, LLM_API_KEY="k", LLM_DELAY="0")
    calls = []

    def fake(self, system, user, max_tokens=None):
        calls.append(1)
        raise BadResponse("cut off")
    monkeypatch.setattr(LLMClient, "chat_json", fake)
    cache = {}
    for _ in range(3):                                          # three weekly runs, page unchanged
        extract_mod.extract(_pages(1), cfg, cache, TODAY, END, {}, [])
    assert len(calls) == 2
    assert cache["https://example.my/p0"]["fails"] == 2


def test_extract_after_quota_exhausted_still_uses_cache(monkeypatch):
    cfg = make_cfg(monkeypatch, LLM_API_KEY="k", LLM_DELAY="0")
    pages = _pages(3)
    cache = {}
    calls = []

    def fake(self, system, user, max_tokens=None):
        calls.append(user)
        raise QuotaExhausted("both models")
    monkeypatch.setattr(LLMClient, "chat_json", fake)
    text2 = focus_text(pages[2]["text"], cfg["llm"]["max_chars_per_page"])
    cache[pages[2]["key"]] = {"hash": extract_mod.page_hash(text2), "ts": "2026-11-20",
                              "events": [{"title": "Cached event"}]}
    warnings, stats = [], {}
    out = extract_mod.extract(pages, cfg, cache, TODAY, END, stats, warnings)
    assert len(calls) == 1                                       # AI not called again after the quota error
    assert [e["title"] for e in out] == ["Cached event"]         # ... but the later cached page is still used
    assert stats["pages_cache_hits"] == 1
    assert sum(w.startswith("AI limit reached") for w in warnings) == 1


# ================================================================= validate.py
SRC = ("Kuliah Perdana bersama Sheikh Dr. Haji Ahmad Fauzi. Tarikh: Sabtu 28hb Nov 2026, 28/11. "
       "Yuran RM1,200 untuk korporat. Produk halal dijual. Daftar: http://www.Example.my/Daftar/ "
       "atau forms.gle/AbC123. Tempat: Masjid Al-Ikhlas, Kajang.")


def mk(**kw):
    base = {"title": "Kuliah Perdana", "is_islamic": True, "start_date": "2026-11-28", "venue": "Masjid Al-Ikhlas",
            "city": "Kajang", "date_evidence": "28hb Nov 2026", "confidence": "high", "speakers": [],
            "source_url": "https://example.my/a", "source_text": SRC}
    base.update(kw)
    return base


@pytest.mark.parametrize("evidence", ["28hb Nov 2026", "Sabtu 28hb Nov", "28/11", "Sabtu"])
def test_date_evidence_formats_accepted(evidence):
    ev, _ = coerce(mk(date_evidence=evidence), TODAY, END)
    assert not any("date" in f for f in ev["flags"]), (evidence, ev["flags"])


def test_date_evidence_wrong_day_still_flagged():
    ev, _ = coerce(mk(start_date="2026-11-29", date_evidence="28hb Nov 2026"), TODAY, END)
    assert "date quote does not contain the day" in ev["flags"] and ev["confidence"] == "low"


def test_speakers_whole_words_and_honorifics():
    ev, _ = coerce(mk(speakers=["Ustaz Ali", "Sheikh Dr. Haji Ahmad Fauzi", "Prof. Madya Dato' Ahmad Fauzi"]),
                   TODAY, END)
    assert ev["speakers_unverified"] == ["Ustaz Ali"]            # "ali" is inside "halal", not a word on the page
    assert len(ev["speakers"]) == 2


def test_speakers_given_as_string_are_split():
    ev, _ = coerce(mk(speakers="Sheikh Dr. Haji Ahmad Fauzi dan Ustaz Ali"), TODAY, END)
    assert ev["speakers"] == ["Sheikh Dr. Haji Ahmad Fauzi"] and ev["speakers_unverified"] == ["Ustaz Ali"]


@pytest.mark.parametrize("price,kept", [("RM1,200", True), ("RM1200", True), ("RM200", False), ("RM20", False),
                                        ("Percuma", True)])
def test_price_whole_number_matching(price, kept):
    ev, _ = coerce(mk(price_text=price), TODAY, END)
    assert (ev["price_text"] == price) is kept


@pytest.mark.parametrize("url,kept", [
    ("https://example.my/Daftar", True),          # scheme, www, trailing slash ignored
    ("http://EXAMPLE.my/Daftar/", True),          # host case-insensitive
    ("https://example.my/daftar", False),         # path case-sensitive
    ("https://forms.gle/AbC123", True),           # bare token on the page (no scheme)
    ("https://forms.gle/AbC12", False),           # truncated
    ("https://forms.gle", False),                 # domain only
    ("https://example.my", False),
])
def test_registration_link_exact_match(url, kept):
    ev, _ = coerce(mk(registration_url=url), TODAY, END)
    assert (ev["registration_url"] == url) is kept


@pytest.mark.parametrize("field,value,check", [
    ("is_islamic", "false", lambda r: r == (None, "not Islamic")),
    ("is_online", "true", lambda r: r[0]["is_online"] is True),
    ("is_online", "false", lambda r: r[0]["is_online"] is False),
    ("in_klang_valley", "yes", lambda r: r[0]["ai_in_kv"] is True),
    ("is_free", "no", lambda r: r[0]["is_free"] is False),
])
def test_string_booleans(field, value, check):
    assert check(coerce(mk(**{field: value}), TODAY, END))


def test_same_day_different_venues_not_merged():
    a = mk(title="Kuliah Maghrib", venue="Masjid Al-Ikhlas", source_url="https://a.my/1")
    b = mk(title="Kuliah Maghrib", venue="Surau Ar-Rahman Seksyen 7", source_url="https://b.my/1",
           source_text="Kuliah Maghrib 28hb Nov 2026 di Surau Ar-Rahman Seksyen 7")
    assert len(validate([a, b], TODAY, END, {})) == 2


@pytest.mark.parametrize("urls,raised", [
    (("https://t.me/alhidayahbbb/1241", "https://t.me/s/alhidayahbbb/1250"), False),   # same channel
    (("https://www.example.my/a", "https://example.my/b"), False),                     # same site
    (("https://t.me/alhidayahbbb/1241", "https://t.me/masjidkajang/77"), True),         # two channels
    (("https://example.my/a", "https://peatix.com/event/1"), True),
])
def test_corroboration_needs_independent_sources(urls, raised):
    evs = [mk(confidence="medium", source_url=u) for u in urls]
    out, = validate(evs, TODAY, END, {})
    assert len(out["sources"]) == 2
    assert out["confidence"] == ("high" if raised else "medium")


def test_malformed_event_skipped():
    stats = {}
    out = validate([mk(source_text=["not", "a", "string"]), mk(title="Good one")], TODAY, END, stats)
    assert [e["title"] for e in out] == ["Good one"]
    assert stats["validate_drops"] == {"malformed": 1}


# ================================================================= region.py
def loc(**kw):
    base = {"is_online": False, "venue": None, "address": None, "city": None, "state": None, "ai_in_kv": None}
    base.update(kw)
    return base


@pytest.mark.parametrize("ev,keep,why", [
    (loc(is_online=True, venue="Zoom"), True, "online"),
    (loc(is_online=True, venue="Online (Google Meet)"), True, "online"),
    (loc(is_online=True, venue="Siaran langsung YouTube"), True, "online"),
    (loc(venue="Wisma MAIWP, Jalan Pahang", ai_in_kv=True), True, "AI says Klang Valley"),
    (loc(venue="Dewan Orang Ramai, Jalan Ipoh"), False, "unknown place"),          # not "other state"
    (loc(venue="Masjid Al-Hidayah, Taman Melawati", state="Melaka?", ai_in_kv=True), True, "AI says Klang Valley"),
    (loc(venue="Rumah Um Salamah"), False, "unknown place"),                        # 'um' alone is not UM
    (loc(venue="Dewan Tunku Canselor, Universiti Malaya"), True, "Klang Valley place named"),
    (loc(venue="Masjid Negeri", city="Ipoh", state="Perak"), False, "other state"),
])
def test_region_rules(ev, keep, why):
    assert region.in_klang_valley(ev) == (keep, why)


@pytest.mark.parametrize("ev", [
    loc(venue="Sunway Carnival Mall", city="Seberang Jaya", state="Pulau Pinang", ai_in_kv=False),
    loc(venue="Dewan Seksyen 2", city="Kota Bharu", state="Kelantan", ai_in_kv=False),
    loc(venue="Surau Taman Tun Aminah", city="Johor Bahru", state="Johor", ai_in_kv=False),
])
def test_region_other_state_with_kv_lookalike_word_dropped(ev):
    assert region.in_klang_valley(ev)[0] is False


# ================================================================= prefilter.py
def _page(text, title="Program", source="search"):
    return {"url": "https://x.my", "title": title, "text": text, "source": source}


def test_prefilter_keywords_match_word_starts(monkeypatch):
    cfg = make_cfg(monkeypatch)
    text = "Program persahabatan komuniti pada 5 Disember 2026 di dewan orang ramai, terbuka kepada semua penduduk."
    assert score_page(_page(text), cfg) == (0, "no Islamic words")
    assert score_page(_page(text.replace("persahabatan", "sahabat Rasulullah")), cfg)[0] > 0


def test_prefilter_day_month_without_year_counts_as_date(monkeypatch):
    cfg = make_cfg(monkeypatch)
    text = "Ceramah di masjid kariah kami pada 28/11 selepas Maghrib. Semua jemaah dijemput hadir beramai-ramai."
    assert score_page(_page(text), cfg)[0] > 0
    assert score_page(_page(text.replace("28/11", "28/10")), cfg) == (0, "no date in window months")


def test_prefilter_ignores_copyright_years(monkeypatch):
    cfg = make_cfg(monkeypatch)
    text = "Ceramah Perdana di Masjid Al-Ikhlas pada 5 Disember. Daftar sekarang.\n© 2025 Masjid Al-Ikhlas. Hak cipta."
    assert score_page(_page(text), cfg)[0] > 0
    assert score_page(_page(text.replace("© 2025", "Kemaskini 2025")), cfg)[1].startswith("only old/other years")


def test_month_hit_may_and_sept():
    assert not month_hit("you may register for the ceramah", [5])
    assert month_hit("ceramah on 12 may at the masjid", [5]) and month_hit("may 12", [5])
    assert month_hit("kuliah 10 sept", [9])


def test_focus_text_links_use_at_most_a_third_of_budget():
    links = "\n".join(f"https://forms.gle/{'x' * 40}{i}" for i in range(60))
    out = focus_text("Ceramah 5 Disember 2026 di masjid\n" + links, 1500)
    assert len(out) <= 1500
    assert len(out.split("LINKS FOUND:")[1]) <= 500 and "Ceramah 5 Disember" in out


def test_prefilter_null_title_and_bad_page_do_not_crash(monkeypatch):
    cfg = make_cfg(monkeypatch)
    good = _page("Ceramah Perdana di Masjid Al-Ikhlas pada 5 Disember 2026. Daftar sekarang, percuma.", title=None)
    bad = {"url": "https://bad.my", "title": "x", "text": 12345, "source": "search"}
    stats = {}
    assert prefilter([bad, good], cfg, stats) == [good]
    assert stats["prefilter_drops"] == {"error": 1}


# ================================================================= search.py
TG_HTML = """<div class="tgme_channel_info_header_title"><span>Masjid Kajang</span></div>
<div class="tgme_widget_message"><div class="tgme_widget_message_text">Kuliah 5 Disember 2026</div>
<a class="tgme_widget_message_date" href="https://t.me/mk/9"><time datetime="2026-11-01T10:00:00+00:00"></time></a></div>"""


def test_parse_telegram_channel_uses_given_today():
    assert len(search.parse_telegram_channel(TG_HTML, "https://t.me/s/mk", today=date(2026, 11, 23))) == 1
    assert search.parse_telegram_channel(TG_HTML, "https://t.me/s/mk", today=date(2027, 2, 1)) == []


def test_fetch_all_passes_today_override_to_telegram(server, monkeypatch):
    server.handler = lambda req: Reply(200, TG_HTML)
    cfg = make_cfg(monkeypatch, TODAY="2027-02-01")
    cfg["direct_sources"] = [server.url + "/t.me/s/mk"]
    assert search.fetch_all(cfg, {}, []) == []
    cfg = make_cfg(monkeypatch, TODAY="2026-11-23")
    cfg["direct_sources"] = [server.url + "/t.me/s/mk"]
    posts = search.fetch_all(cfg, {}, [])
    assert len(posts) == 1 and posts[0]["query"] is None


def test_fetch_all_tags_pages_with_queries(server, monkeypatch):
    monkeypatch.setattr(search, "TAVILY_URL", server.url + "/search")
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="k", MAX_QUERIES="3")
    qs = search.build_queries(cfg)

    def handler(req):
        q = req.json()["query"]
        shared = {"url": "https://example.my/shared?utm_source=x", "title": "s", "content": "shared " + q}
        own = {"url": f"https://example.my/{qs.index(q)}", "title": "o", "content": "own"}
        return Reply(200, {"results": [shared, own] if q != qs[2] else []})
    server.handler = handler
    stats = {}
    pages = {p["key"]: p for p in search.fetch_all(cfg, stats, [])}
    shared = pages["https://example.my/shared"]
    assert shared["queries"] == sorted(qs[:2]) and shared["query"] in qs[:2]
    assert pages["https://example.my/0"]["queries"] == [qs[0]] and pages["https://example.my/0"]["query"] == qs[0]
    assert stats["queries_ok"] == qs and stats["queries_all"] == len(search.build_queries(cfg, cap=False)) == 34
    assert stats["queries"] == 3


def test_fetch_all_skips_resting_queries(server, monkeypatch):
    monkeypatch.setattr(search, "TAVILY_URL", server.url + "/search")
    server.handler = lambda req: Reply(200, {"results": []})
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="k")
    all_q = search.build_queries(cfg, cap=False)
    resting = next(q for q in all_q if not learning.retry_due(q, TODAY))
    qstats = {learning.query_key(resting): {"runs": 5, "total_events": 0, "consecutive_zero_runs": 5}}
    stats = {}
    search.fetch_all(cfg, stats, [], qstats)
    asked = [r.json()["query"] for r in server.requests]
    assert resting not in asked and len(asked) == 33 and stats["queries_all"] == 34


# ================================================================= report.py
def _ev(**kw):
    raw = mk(**kw)
    ev, why = coerce(raw, TODAY, END)
    assert why == "ok", why
    ev["id"] = kw.get("title", "x")
    return ev


def test_report_online_section_is_last(monkeypatch):
    cfg = make_cfg(monkeypatch)
    cfg["speaker_watchlist"] = ["Ahmad Fauzi"]
    evs = [_ev(title="Webinar Zoom", is_online=True, venue="Zoom", start_date="2026-11-24", date_evidence="Selasa"),
           _ev(title="Later talk", start_date="2026-12-12", date_evidence="12"),
           _ev(title="Soon talk", start_date="2026-11-28"),
           _ev(title="Watch talk", start_date="2026-12-20", date_evidence="20",
               speakers=["Sheikh Dr. Haji Ahmad Fauzi"])]
    header = report.build_digest(evs, cfg, {}, TODAY, {}, [])["header"]
    sections = [l for l in header.split("\n") if l.startswith("<b>") and "Islamic events" not in l]
    assert sections == ["<b>⭐ Watchlist speakers</b>", "<b>📌 This week</b>", "<b>🗓 Coming up</b>",
                        "<b>💻 Online</b>"]
    assert header.index("4. Tue 24 Nov · Webinar Zoom") > header.index("<b>💻 Online</b>")


def test_report_include_online_false_hides_online(monkeypatch):
    cfg = make_cfg(monkeypatch)
    cfg["include_online"] = False
    d = report.build_digest([_ev(title="Webinar", is_online=True, venue=None)], cfg, {}, TODAY, {}, [])
    assert d["cards"] == [] and "0 events" in d["header"]


def test_google_calendar_details_capped(monkeypatch):
    from urllib.parse import parse_qs, urlsplit
    from zoneinfo import ZoneInfo
    e = _ev(topic="x" * 160, speakers=["Sheikh Dr. Haji Ahmad Fauzi"] * 8)
    e["sources"] = ["https://example.my/" + "p" * 400]
    q = parse_qs(urlsplit(report.google_calendar_url(e, ZoneInfo("Asia/Kuala_Lumpur"))).query)
    assert len(q["details"][0]) <= 300


def test_to_plain_strips_tags_before_unescaping():
    assert report.to_plain("<b>Ceramah &lt;Maulid&gt; &amp; Selawat</b> <i>x</i>") == "Ceramah <Maulid> & Selawat x"


def test_to_text_lists_buttons_as_links(monkeypatch):
    cfg = make_cfg(monkeypatch)
    e = _ev(registration_url="https://forms.gle/AbC123")
    txt = report.to_text(report.build_digest([e], cfg, {}, TODAY, {}, []))
    assert "📝 Register: https://forms.gle/AbC123" in txt and "<b>" not in txt and "**" not in txt


# ================================================================= deliver.py
def test_send_telegram_raises_when_header_rejected(server, monkeypatch, sleeps):
    real_post = requests.post
    monkeypatch.setattr(requests, "post", lambda url, **kw: real_post(
        server.url + url[len("https://api.telegram.org"):], **kw))
    server.handler = lambda req: Reply(403, {"ok": False, "error_code": 403,
                                             "description": "Forbidden: bot was blocked by the user"})
    cfg = make_cfg(monkeypatch, TELEGRAM_TOKEN="1:x", TELEGRAM_CHAT_ID="2")
    with pytest.raises(RuntimeError, match="TELEGRAM_CHAT_ID"):
        deliver_mod.send_telegram({"header": "h", "cards": [("c", [], "")] * 5, "footer": "f"}, "ICS", cfg)
    assert len(server.requests) == 2                  # header + plain retry; cards/footer/.ics never attempted


# ================================================================= main.py + learning, end to end
CH = "kajangdakwah"
POST_A, POST_B = f"https://t.me/{CH}/88", f"https://t.me/{CH}/95"
SITE = "https://www.masjidjamekkajang.example.my/aktiviti/kuliah-dhuha"


def tg_post_text(day, title):
    return (f"📢 {title}\n🗓 {day} Disember 2026\n🕖 8.30 malam\n📍 Masjid Jamek Kajang\n"
            f"🎙 Ustaz Kamarul Zaman\nTerbuka kepada muslimin dan muslimat. Percuma.")


LLM_BY_URL = {
    POST_A: {"title": "Kuliah Tafsir Al-Mulk", "start_date": "2026-12-03", "start_time": "20:30",
             "venue": "Masjid Jamek Kajang", "city": "Kajang", "is_islamic": True, "speakers": ["Ustaz Kamarul Zaman"],
             "date_evidence": "3 Disember 2026", "confidence": "high", "is_free": True},
    POST_B: {"title": "Ceramah Isra Mikraj", "start_date": "2026-12-10", "start_time": "20:30",
             "venue": "Masjid Jamek Kajang", "city": "Kajang", "is_islamic": True, "speakers": ["Ustaz Kamarul Zaman"],
             "date_evidence": "10 Disember 2026", "confidence": "high", "is_free": True},
    SITE: {"title": "Kuliah Dhuha Muslimat", "start_date": "2026-12-06", "start_time": "10:00",
           "venue": "Masjid Jamek Kajang", "city": "Kajang", "is_islamic": True,
           "date_evidence": "Ahad, 6 Disember 2026", "confidence": "high"},
}


@pytest.fixture
def learn_world(monkeypatch, tmp_path):
    server = FakeServer().start()
    cfg_data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg_data["direct_sources"] = []
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg_data, allow_unicode=True), encoding="utf-8")
    for k, v in {"CONFIG_FILE": str(tmp_path / "config.yaml"), "TAVILY_API_KEY": "tvly-x", "LLM_API_KEY": "gsk_x",
                 "LLM_BASE_URL": server.url + "/v1", "LLM_DELAY": "0", "MAX_QUERIES": "40"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(search, "TAVILY_URL", server.url + "/search")
    monkeypatch.setattr(agent_main, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("agent.state.STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(llm_mod, "time", SimpleNamespace(sleep=lambda s: None, monotonic=time.monotonic))
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no Telegram keys: nothing may be sent"))

    w = SimpleNamespace(server=server, tavily={}, llm_urls=[], seen_pages=[], state=tmp_path / "state",
                        out=tmp_path / "out")

    def handler(req):
        if req.path == "/search":
            return Reply(200, {"results": w.tavily.get(req.json()["query"], [])})
        if req.path == "/v1/chat/completions":
            user = req.json()["messages"][1]["content"]
            url = next(u for u in LLM_BY_URL if f"PAGE URL: {u}\n" in user)
            w.llm_urls.append(url)
            return openai_reply(json.dumps({"events": [LLM_BY_URL[url]]}))
        return Reply(404, "")
    server.handler = handler

    real_prefilter = agent_main.prefilter

    def spy(pages, cfg, stats):
        w.seen_pages.append([dict(p) for p in pages])
        return real_prefilter(pages, cfg, stats)
    monkeypatch.setattr(agent_main, "prefilter", spy)
    with HardTimeout(30):
        yield w
    server.stop()
    assert not server.errors, server.errors[0]


class Args:
    no_send, pages_file = False, None


def test_learning_end_to_end_query_tags_stats_and_source_suggestion(learn_world, monkeypatch, capsys):
    w = learn_world
    qs = search.build_queries(load_config(), cap=False)
    item = lambda url, text: {"url": url, "title": "t", "content": text[:100], "raw_content": text}  # noqa: E731
    site_text = ("Kuliah Dhuha Muslimat\nTarikh: Ahad, 6 Disember 2026\nMasa: 10.00 pagi\n"
                 "Tempat: Masjid Jamek Kajang\nSemua muslimat dijemput hadir. Program percuma.")

    # ---- run 1 (Mon 23 Nov): one post from the channel + one mosque website
    monkeypatch.setenv("TODAY", "2026-11-23")
    w.tavily = {qs[0]: [item(POST_A, tg_post_text(3, "KULIAH TAFSIR AL-MULK"))], qs[1]: [item(SITE, site_text)],
                qs[2]: [item(SITE, site_text)]}
    assert agent_main.run(Args()) == 0
    out1 = capsys.readouterr().out
    pages = {p["url"]: p for p in w.seen_pages[0]}
    assert pages[POST_A]["query"] == qs[0] and pages[POST_A]["queries"] == [qs[0]]
    assert pages[SITE]["queries"] == sorted([qs[1], qs[2]])          # found by two queries: both get credit
    assert sorted(w.llm_urls) == sorted([POST_A, SITE])
    assert "💡" not in out1                                          # only 1 event from the channel so far

    qstats = json.loads((w.state / "query_stats.json").read_text(encoding="utf-8"))
    k = learning.query_key
    assert len(qstats) == 34
    assert qstats[k(qs[0])]["total_events"] == 1 and qstats[k(qs[0])]["last_yield_date"] == "2026-11-23"
    assert qstats[k(qs[1])]["total_events"] == 1 and qstats[k(qs[2])]["total_events"] == 1
    assert qstats[k(qs[3])] == {"runs": 1, "total_events": 0, "last_yield_date": None,
                                "consecutive_zero_runs": 1, "last_run": "2026-11-23"}
    src1 = json.loads((w.state / "source_stats.json").read_text(encoding="utf-8"))
    assert src1[f"t.me/{CH}"]["events"] == 1
    assert json.loads((w.out / "stats.json").read_text())["stats"]["source_suggestions"] == []

    # ---- run 2 (Mon 30 Nov): a second, new post from the same channel
    monkeypatch.setenv("TODAY", "2026-11-30")
    w.llm_urls.clear()
    w.tavily[qs[0]] = [item(POST_A, tg_post_text(3, "KULIAH TAFSIR AL-MULK")),
                       item(POST_B, tg_post_text(10, "CERAMAH ISRA MIKRAJ"))]
    assert agent_main.run(Args()) == 0
    out2 = capsys.readouterr().out
    assert w.llm_urls == [POST_B]                                    # the rest came from the cache
    assert f"https://t.me/s/{CH}" in out2 and "💡 Sources that keep finding events" in out2
    stats2 = json.loads((w.out / "stats.json").read_text())["stats"]
    assert stats2["source_suggestions"] == [f"https://t.me/s/{CH}"]
    qstats = json.loads((w.state / "query_stats.json").read_text(encoding="utf-8"))
    assert qstats[k(qs[0])]["runs"] == 2 and qstats[k(qs[0])]["total_events"] == 3
    assert qstats[k(qs[3])]["consecutive_zero_runs"] == 2
    src2 = json.loads((w.state / "source_stats.json").read_text(encoding="utf-8"))
    assert src2[f"t.me/{CH}"]["events"] == 2                         # POST_A's event is counted once
    seen = json.loads((w.state / "seen.json").read_text())
    assert {e["id"] for e in json.loads((w.out / "events.json").read_text())} <= set(seen)


def test_pages_file_run_skips_learning(learn_world, monkeypatch, capsys):
    monkeypatch.setattr(LLMClient, "chat_json", lambda self, s, u, max_tokens=None: {"events": [
        {"title": "Ceramah Perdana", "is_islamic": True, "start_date": "2026-11-28", "venue": "Masjid Contoh",
         "city": "Shah Alam", "date_evidence": "Sabtu, 28 November 2026", "confidence": "high"}]})

    class PF:
        no_send, pages_file = False, str(HERE / "sample_pages.json")
    assert agent_main.run(PF()) == 0
    w = learn_world
    assert json.loads((w.state / "query_stats.json").read_text()) == {}
    assert json.loads((w.state / "source_stats.json").read_text()) == {}
    assert "source_suggestions" not in json.loads((w.out / "stats.json").read_text())["stats"]
    assert w.server.requests == []                                   # no Tavily calls either


def test_state_saved_even_if_report_crashes(learn_world, monkeypatch):
    w = learn_world
    qs = search.build_queries(load_config(), cap=False)
    w.tavily = {qs[0]: [{"url": POST_A, "title": "t", "raw_content": tg_post_text(3, "KULIAH TAFSIR AL-MULK")}]}

    def boom(*a, **k):
        raise RuntimeError("report bug")
    monkeypatch.setattr(agent_main, "build_digest", boom)
    with pytest.raises(RuntimeError):
        agent_main.run(Args())
    cache = json.loads((w.state / "extract_cache.json").read_text(encoding="utf-8"))
    assert list(cache) == ["https://t.me/kajangdakwah/88"]           # the paid-for AI result was kept


@pytest.mark.parametrize("venue,expected", [
    (None, True), ("Zoom", True), ("Online via YouTube Live", True), ("Google Meet", True),
    ("Masjid Negeri Pulau Pinang & YouTube Live", False), ("Dewan Besar UIA dan Zoom", False),
])
def test_online_only_vs_hybrid(venue, expected):
    from agent.region import online_only
    assert online_only({"is_online": True, "venue": venue}) is expected


def test_hybrid_event_in_penang_is_dropped():
    from agent.region import in_klang_valley
    ev = {"is_online": True, "venue": "Masjid Negeri Pulau Pinang & YouTube Live", "city": "George Town",
          "address": None, "state": "Pulau Pinang", "ai_in_kv": False}
    assert in_klang_valley(ev) == (False, "other state")
