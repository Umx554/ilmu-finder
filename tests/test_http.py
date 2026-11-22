"""HTTP-level tests: the agent's real `requests` code talks to scripted fake services on 127.0.0.1.

Covers agent/search.py (Tavily + direct pages + Telegram channels), agent/llm.py (OpenAI-compatible client:
rate limits, model fallback, JSON repair) and agent/deliver.py (Telegram Bot API + SMTP e-mail).
No real network access is used; every test runs under a hard SIGALRM timeout.
"""
import json
import smtplib
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
sys.path.insert(0, str(FIX))

from _fakehttp import CLEAR_ENV, FakeServer, HardTimeout, Reply, groq_error, openai_reply  # noqa: E402

from agent import deliver as deliver_mod  # noqa: E402
from agent import llm as llm_mod  # noqa: E402
from agent import report, search  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.extract import extract  # noqa: E402
from agent.llm import BadResponse, LLMClient, QuotaExhausted, ServiceError  # noqa: E402
from agent.prefilter import focus_text, prefilter  # noqa: E402
from agent.validate import coerce  # noqa: E402

TODAY, END = date(2026, 11, 23), date(2026, 12, 23)
MAIN, FALLBACK = "openai/gpt-oss-120b", "openai/gpt-oss-20b"
TOKEN, CHAT = "123456789:AAFake-Token_x", "987654321"


# ---------------------------------------------------------------- shared fixtures
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
    """Replace time.sleep inside agent.llm / agent.deliver with a recorder (keeps tests fast)."""
    rec: list[float] = []
    fake_time = SimpleNamespace(sleep=rec.append, monotonic=time.monotonic)
    monkeypatch.setattr(llm_mod, "time", fake_time)
    monkeypatch.setattr(deliver_mod, "time", fake_time)
    return rec


def make_cfg(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return load_config()


class FixedNow(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 23, 1, 0, tzinfo=timezone.utc).astimezone(tz) if tz else datetime(2026, 11, 23, 9, 0)


# ================================================================= agent/search.py
TAVILY_OK = {
    "query": "q", "follow_up_questions": None, "answer": None, "images": [], "response_time": 1.07,
    "results": [
        {"url": "https://peatix.com/event/4471203", "title": "KLIFHES 2026 | Peatix",
         "content": "Short snippet about the summit.", "raw_content": "FULL PAGE TEXT of the summit, 9 Dec 2026",
         "score": 0.91, "published_date": "2026-11-02"},
        {"url": "https://www.example-news.my/berita/ceramah", "title": "Ceramah di Shah Alam",
         "content": "Only the snippet is available.", "raw_content": None, "score": 0.55},
    ],
}


@pytest.fixture
def tavily(server, monkeypatch):
    monkeypatch.setattr(search, "TAVILY_URL", server.url + "/search")
    return server


def test_tavily_search_request_body_auth_and_parsing(tavily, monkeypatch):
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="tvly-dev-TESTKEY")
    tavily.handler = lambda req: Reply(200, TAVILY_OK)
    pages = search.tavily_search("ceramah Shah Alam Disember 2026", cfg, requests.Session())

    req, = tavily.requests
    assert req.method == "POST" and req.path == "/search"
    assert req.headers["authorization"] == "Bearer tvly-dev-TESTKEY"
    assert req.headers["content-type"] == "application/json"
    assert req.json() == {"query": "ceramah Shah Alam Disember 2026", "search_depth": "basic", "max_results": 8,
                          "include_raw_content": True, "time_range": "year"}
    assert [p["text"] for p in pages] == ["FULL PAGE TEXT of the summit, 9 Dec 2026", "Only the snippet is available."]
    assert pages[0]["published"] == "2026-11-02" and pages[1]["published"] is None
    assert {p["source"] for p in pages} == {"search"}


def test_tavily_time_range_omitted_when_null(tavily, monkeypatch):
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="k")
    cfg["search"]["time_range"] = None
    tavily.handler = lambda req: Reply(200, {"results": []})
    assert search.tavily_search("q", cfg, requests.Session()) == []
    assert "time_range" not in tavily.requests[0].json()


@pytest.mark.parametrize("code,body", [
    (401, {"detail": {"error": "Unauthorized: missing or invalid API key."}}),
    (403, {"detail": {"error": "Forbidden"}}),
    (429, {"detail": {"error": "Your request has been blocked due to excessive requests."}}),
    (432, {"detail": {"error": "This request exceeds your plan's set usage limit. Please upgrade your plan."}}),
    (433, {"detail": {"error": "This request exceeds the pay-as-you-go limit."}}),
])
def test_fetch_all_stops_early_on_auth_or_quota_errors(tavily, monkeypatch, caplog, code, body):
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="tvly-x", MAX_QUERIES="5")
    tavily.handler = lambda req: Reply(code, body)
    stats, warnings = {}, []
    pages = search.fetch_all(cfg, stats, warnings)
    assert pages == []
    assert len(tavily.requests) == 1                      # did not burn the other 4 queries
    assert f"Search stopped early (HTTP {code}); check Tavily key or credits." in warnings
    assert stats["queries"] == 5 and stats["queries_failed"] == 1
    assert f"search failed ({code})" in caplog.text


def test_fetch_all_continues_after_a_single_500(tavily, monkeypatch):
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="tvly-x", MAX_QUERIES="5")

    def handler(req):
        n = len(tavily.requests)
        if n == 1:
            return Reply(500, {"detail": {"error": "Internal Server Error"}})
        return Reply(200, {"results": [{"url": f"https://example.my/e{n}", "title": f"E{n}",
                                        "content": f"page {n} text", "raw_content": None}]})
    tavily.handler = handler
    stats, warnings = {}, []
    pages = search.fetch_all(cfg, stats, warnings)
    assert len(tavily.requests) == 5
    assert sorted(p["url"] for p in pages) == [f"https://example.my/e{n}" for n in range(2, 6)]
    assert stats["queries_failed"] == 1 and stats["pages_unique"] == 4
    assert not any("stopped early" in w for w in warnings)


def test_fetch_all_survives_connection_errors(monkeypatch):
    monkeypatch.setattr(search, "TAVILY_URL", "http://127.0.0.1:1/search")     # nothing listens there
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="tvly-x", MAX_QUERIES="3")
    stats, warnings = {}, []
    assert search.fetch_all(cfg, stats, warnings) == []
    assert stats["queries_failed"] == 3 and not any("stopped early" in w for w in warnings)


@pytest.mark.parametrize("order", ["long_first", "long_last"])
def test_fetch_all_dedup_by_normalised_url_keeps_longest_text(tavily, monkeypatch, order):
    cfg = make_cfg(monkeypatch, TAVILY_API_KEY="tvly-x", MAX_QUERIES="1")
    results = [
        {"url": "https://www.Peatix.com/event/4471203?utm_source=google&utm_medium=cpc", "title": "a",
         "content": "short", "raw_content": None},
        {"url": "https://peatix.com/event/4471203?fbclid=IwAR0abc", "title": "b",
         "content": "medium length text", "raw_content": None},
        {"url": "https://peatix.com/event/4471203/", "title": "c", "content": "x",
         "raw_content": "the longest full raw content of the page " * 5},
        {"url": "https://example.my/empty", "title": "empty", "content": "", "raw_content": None},
        {"url": "", "title": "no url", "content": "orphan text"},
    ]
    if order == "long_first":
        results.insert(0, results.pop(2))
    tavily.handler = lambda req: Reply(200, {"results": results})
    stats = {}
    pages = search.fetch_all(cfg, stats, [])
    assert len(pages) == 1
    assert pages[0]["title"] == "c" and pages[0]["key"] == "https://peatix.com/event/4471203"
    assert stats["pages_raw"] == 5 and stats["pages_unique"] == 1


def test_direct_html_source_keeps_link_targets(server, monkeypatch):
    html = (FIX / "mosque_ceramah.html").read_bytes()
    server.handler = lambda req: Reply(200, html) if req.path == "/program-ilmu" else Reply(404, "not found")
    cfg = make_cfg(monkeypatch)
    url = server.url + "/program-ilmu"
    cfg["direct_sources"] = [url]
    warnings = []
    pages = search.fetch_all(cfg, {}, warnings)

    assert "No TAVILY_API_KEY: web search skipped, only direct sources used." in warnings
    p, = pages
    assert server.requests[0].headers["user-agent"].startswith("Mozilla/5.0 (compatible; events-agent")
    assert p["source"] == "direct" and p["url"] == url
    assert p["title"] == "Program Ilmu Disember 2026 | Masjid Al-Falah USJ 9"
    t = p["text"]
    assert "Tarikh: Sabtu, 5 Disember 2026" in t and "Masa: 8.30 malam" in t and "Yuran: Percuma" in t
    assert "012-345 6789\n (https://wa.me/60123456789)" in t        # href kept right after the link text
    assert "(https://t.me/masjidalfalahusj9)" in t
    assert "dataLayer" not in t and "Derma" not in t and "Hak cipta" not in t   # script / nav / footer stripped
    # the registration-ish link survives trimming for the LLM
    assert "LINKS FOUND:\nhttps://wa.me/60123456789" in focus_text(t, 6000)


def test_direct_source_failure_warns_and_continues(server, monkeypatch):
    html = (FIX / "mosque_ceramah.html").read_bytes()
    server.handler = lambda req: Reply(200, html) if req.path == "/ok" else Reply(404, "<h1>404</h1>")
    cfg = make_cfg(monkeypatch)
    cfg["direct_sources"] = [server.url + "/gone", server.url + "/ok"]
    warnings = []
    pages = search.fetch_all(cfg, {}, warnings)
    assert [p["url"] for p in pages] == [server.url + "/ok"]
    assert f"Direct source failed: {server.url}/gone" in warnings


def test_telegram_channel_page_from_fixture(server, monkeypatch):
    html = (FIX / "telegram_channel.html").read_bytes()
    server.handler = lambda req: Reply(200, html)
    monkeypatch.setattr(search, "datetime", FixedNow)          # pin "now" to the TODAY of the test suite
    cfg = make_cfg(monkeypatch)
    url = server.url + "/t.me/s/alhidayahbbb"
    cfg["direct_sources"] = [url]
    stats = {}
    posts = search.fetch_all(cfg, stats, [])

    # Aug post is older than 45 days; the photo-only post has no text.
    assert [p["url"] for p in posts] == ["https://t.me/alhidayahbbb/1241", "https://t.me/alhidayahbbb/1247"]
    assert all(p["title"] == "Masjid Al-Hidayah Bandar Baru Bangi (Telegram)" for p in posts)
    assert all(p["source"] == "telegram" for p in posts)
    assert posts[0]["published"] == "2026-11-18T10:12:45+00:00"
    kuliah, daurah = posts[0]["text"], posts[1]["text"]
    assert "KULIAH MAGHRIB ISTIMEWA" in kuliah and "Jumaat, 27 November 2026" in kuliah
    assert "Ustaz Abdul Rahman Idris" in kuliah and "#kuliahmaghrib" in kuliah
    assert "?q=%23" not in kuliah                                 # relative hashtag links not appended
    assert "Ahad, 13 Disember 2026" in daurah and "RM30" in daurah and "Sumbangan" in daurah
    assert daurah.rstrip().endswith("https://forms.gle/Xy7kLmN2pQ8vR3wE9\nhttps://wa.me/60198765432")
    assert not any("OldOgos2026" in p["text"] for p in posts)
    assert stats["pages_unique"] == 2

    kept = prefilter(posts, cfg, {})          # telegram posts bypass the search-only year/month rules
    assert len(kept) == 2
    assert "LINKS FOUND:\nhttps://forms.gle/Xy7kLmN2pQ8vR3wE9" in focus_text(daurah, 6000)


# ================================================================= agent/llm.py
@pytest.fixture
def llm_cfg(server, monkeypatch):
    return make_cfg(monkeypatch, LLM_BASE_URL=server.url + "/openai/v1/", LLM_API_KEY="gsk_test", LLM_DELAY="0")


def script(server, *replies):
    """Serve the given replies in order; fail loudly if the client asks for more."""
    queue = list(replies)

    def handler(req):
        if not queue:
            raise AssertionError(f"unexpected extra request #{len(server.requests)}")
        return queue.pop(0)
    server.handler = handler


def models_used(server):
    return [r.json()["model"] for r in server.requests]


EVENTS = {"events": [{"title": "Kuliah Maghrib", "start_date": "2026-11-27"}]}
TPM_429 = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01abc` service tier "
           "`on_demand` on tokens per minute (TPM): Limit 12000, Used 11021, Requested 1689. Please try again in "
           "3.55s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing")
TPD_429 = ("Rate limit reached for model `{m}` in organization `org_01abc` service tier `on_demand` on tokens per "
           "day (TPD): Limit 100000, Used 99412, Requested 1650. Please try again in 14m22.56s. Need more tokens? "
           "Upgrade to Dev Tier today at https://console.groq.com/settings/billing")


def test_llm_success_request_shape_and_token_count(server, llm_cfg, sleeps):
    script(server, openai_reply(json.dumps(EVENTS), tokens=(1000, 250)),
           openai_reply('{"events": []}', tokens=(900, 10)))
    c = LLMClient(llm_cfg)
    assert c.chat_json("SYSTEM PROMPT", "PAGE TEXT") == EVENTS
    assert c.calls == 1 and c.tokens == 1250
    req = server.requests[0]
    assert req.path == "/openai/v1/chat/completions"           # trailing slash in base URL handled
    assert req.headers["authorization"] == "Bearer gsk_test"
    assert req.json() == {"model": MAIN, "temperature": 0, "max_tokens": 3000, "reasoning_effort": "low",
                          "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": "SYSTEM PROMPT"},
                                       {"role": "user", "content": "PAGE TEXT"}]}
    assert c.chat_json("s", "u") == {"events": []}
    assert c.calls == 2 and c.tokens == 2160 and sleeps == []


def test_llm_waits_between_calls(server, monkeypatch, sleeps):
    cfg = make_cfg(monkeypatch, LLM_BASE_URL=server.url, LLM_API_KEY="k", LLM_DELAY="8")
    script(server, openai_reply("{}"), openai_reply("{}"))
    c = LLMClient(cfg)
    c.chat_json("s", "u")
    c.chat_json("s", "u")
    assert len(sleeps) == 1 and 7 < sleeps[0] <= 8              # per-minute limits: pause before the 2nd call


def test_llm_429_with_retry_after_then_success(server, llm_cfg, sleeps):
    script(server, groq_error(429, TPM_429, code="rate_limit_exceeded", etype="tokens", headers={"retry-after": "4"}),
           openai_reply(json.dumps(EVENTS)))
    c = LLMClient(llm_cfg)
    assert c.chat_json("s", "u") == EVENTS
    assert models_used(server) == [MAIN, MAIN]                  # per-minute limit: same model, just wait
    assert sleeps == [4.0]
    assert c.calls == 1 and c.models == [MAIN, FALLBACK]


def test_llm_persistent_429_is_bounded(server, llm_cfg, sleeps):
    server.handler = lambda req: groq_error(429, TPM_429, headers={"retry-after": "600"})
    c = LLMClient(llm_cfg)
    with pytest.raises(ServiceError):
        c.chat_json("s", "u")
    assert len(server.requests) == 6                            # gives up after 6 attempts, no endless loop
    assert sleeps and all(s == 120 for s in sleeps)             # retry-after is capped at 120 s


def test_llm_does_not_sleep_after_final_attempt(server, llm_cfg, sleeps):
    server.handler = lambda req: groq_error(429, TPM_429, headers={"retry-after": "600"})
    with pytest.raises(ServiceError):
        LLMClient(llm_cfg).chat_json("s", "u")
    assert len(sleeps) == len(server.requests) - 1


def test_llm_tpd_429_switches_to_fallback_model(server, llm_cfg, sleeps, caplog):
    script(server, groq_error(429, TPD_429.format(m=MAIN), code="rate_limit_exceeded", etype="tokens",
                              headers={"retry-after": "862"}),
           openai_reply(json.dumps(EVENTS), model=FALLBACK),
           openai_reply('{"events": []}', model=FALLBACK))
    c = LLMClient(llm_cfg)
    assert c.chat_json("s", "u") == EVENTS
    assert models_used(server) == [MAIN, FALLBACK]
    assert sleeps == []                                          # no point waiting 14 minutes: switch at once
    assert c.model == FALLBACK
    assert f"daily quota used up on {MAIN}; switching to {FALLBACK}" in caplog.text
    c.chat_json("s", "u")                                        # later pages stay on the fallback model
    assert models_used(server) == [MAIN, FALLBACK, FALLBACK]


def test_llm_both_models_exhausted_raises_quota_exhausted(server, llm_cfg, sleeps):
    server.handler = lambda req: groq_error(429, TPD_429.format(m=req.json()["model"]), code="rate_limit_exceeded")
    c = LLMClient(llm_cfg)
    with pytest.raises(QuotaExhausted):
        c.chat_json("s", "u")
    assert models_used(server) == [MAIN, FALLBACK]
    with pytest.raises(QuotaExhausted):                          # no further HTTP once both are dead
        c.chat_json("s", "u")
    assert len(server.requests) == 2


def test_extract_stops_calling_ai_when_quota_exhausted(server, llm_cfg, sleeps):
    server.handler = lambda req: groq_error(429, TPD_429.format(m=req.json()["model"]))
    pages = [{"url": f"https://example.my/p{i}", "key": f"https://example.my/p{i}", "title": "t",
              "text": f"Ceramah {i} di masjid 30 November 2026"} for i in range(3)]
    warnings, cache, stats = [], {}, {}
    assert extract(pages, llm_cfg, cache, TODAY, END, stats, warnings) == []
    assert any(w.startswith("AI limit reached") for w in warnings)
    assert len(server.requests) == 2 and cache == {} and stats["llm_calls"] == 0


@pytest.mark.parametrize("status,message,code", [
    (400, f"The model `{MAIN}` has been decommissioned and is no longer supported. Please refer to "
          "https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.",
     "model_decommissioned"),
    (404, f"The model `{MAIN}` does not exist or you do not have access to it.", "model_not_found"),
])
def test_llm_retired_model_falls_back(server, llm_cfg, sleeps, status, message, code):
    script(server, groq_error(status, message, code=code), openai_reply(json.dumps(EVENTS), model=FALLBACK))
    c = LLMClient(llm_cfg)
    assert c.chat_json("s", "u") == EVENTS
    assert models_used(server) == [MAIN, FALLBACK]


def test_llm_retired_model_without_fallback_is_quota_exhausted(server, monkeypatch, sleeps):
    cfg = make_cfg(monkeypatch, LLM_BASE_URL=server.url, LLM_API_KEY="k", LLM_DELAY="0", LLM_FALLBACK_MODEL="none")
    script(server, groq_error(400, f"The model `{MAIN}` has been decommissioned.", code="model_decommissioned"))
    with pytest.raises(QuotaExhausted):
        LLMClient(cfg).chat_json("s", "u")
    assert len(server.requests) == 1


def test_llm_context_length_error_does_not_drop_main_model(server, llm_cfg, sleeps):
    ctx = groq_error(400, "This model's maximum context length is 8192 tokens. However, your messages resulted in "
                          "9120 tokens. Please reduce the length of the messages.", code="context_length_exceeded")
    server.handler = lambda req: ctx if len(server.requests) == 1 else openai_reply(json.dumps(EVENTS))
    c = LLMClient(llm_cfg)
    with pytest.raises(requests.HTTPError):
        c.chat_json("s", "a very long page")
    assert len(server.requests) == 1
    c.chat_json("s", "next, normal page")
    assert models_used(server) == [MAIN, MAIN]


def test_llm_server_error_is_retried(server, llm_cfg, sleeps):
    script(server, groq_error(503, "Service Unavailable", etype="internal_server_error"),
           openai_reply(json.dumps(EVENTS)))
    assert LLMClient(llm_cfg).chat_json("s", "u") == EVENTS
    assert len(server.requests) == 2 and sleeps == [15]


def test_llm_persistent_server_error_is_bounded(server, llm_cfg, sleeps):
    server.handler = lambda req: groq_error(502, "Bad Gateway", etype="internal_server_error")
    with pytest.raises(ServiceError):
        LLMClient(llm_cfg).chat_json("s", "u")
    assert len(server.requests) == 8


def test_llm_invalid_json_once_then_valid(server, llm_cfg, sleeps):
    script(server, openai_reply('{"events": [{"title": "Kuliah Maghrib", "start_date": "2026-11-', tokens=(800, 40)),
           openai_reply(json.dumps(EVENTS), tokens=(800, 60)))
    c = LLMClient(llm_cfg)
    assert c.chat_json("s", "u") == EVENTS
    assert len(server.requests) == 2 and c.calls == 2 and c.tokens == 1700


def test_llm_invalid_json_twice_raises_bad_response(server, llm_cfg, sleeps):
    script(server, openai_reply("I could not find any events, sorry."), openai_reply("{'events': [}"))
    with pytest.raises(BadResponse):
        LLMClient(llm_cfg).chat_json("s", "u")
    assert len(server.requests) == 2


@pytest.mark.parametrize("content", [
    "```json\n" + json.dumps(EVENTS, indent=2) + "\n```",
    "Here is the JSON you asked for:\n\n" + json.dumps(EVENTS) + "\n\nLet me know if you need anything else!",
    "```\n" + json.dumps(EVENTS) + "\n```\nNote: the venue was not stated {so I used null}.",
])
def test_llm_json_wrapped_in_prose_or_fences(server, llm_cfg, sleeps, content):
    script(server, openai_reply(content), openai_reply(json.dumps(EVENTS)))
    got = LLMClient(llm_cfg).chat_json("s", "u")
    assert got == EVENTS
    assert len(server.requests) in (1, 2)       # repaired on the spot, or at worst re-asked once


def _null_content_reply():
    body = json.loads(openai_reply("x").body)
    body["choices"][0]["message"]["content"] = None
    del body["usage"]
    return Reply(200, body)


def test_llm_null_content_counts_as_unparseable(server, llm_cfg, sleeps):
    script(server, _null_content_reply(), openai_reply(json.dumps(EVENTS)))
    c = LLMClient(llm_cfg)
    assert c.chat_json("s", "u") == EVENTS and len(server.requests) == 2
    script(server, _null_content_reply(), _null_content_reply())
    with pytest.raises(BadResponse):
        c.chat_json("s", "u")


# ================================================================= agent/deliver.py
def raw_event(**kw):
    src = ("Ceramah Perdana. Tarikh: Sabtu, 28 November 2026. Masa: 8.30 malam. Tempat: Masjid Contoh, Shah Alam. "
           "Penceramah: Ustaz Ahmad Contoh. Yuran: RM20. Daftar: https://forms.gle/AbCdEf123 & lain-lain <info>")
    base = {"title": "Ceramah Perdana <Maulid> & Selawat", "is_islamic": True, "start_date": "2026-11-28",
            "start_time": "20:30", "venue": "Masjid Contoh", "city": "Shah Alam", "price_text": "RM20",
            "speakers": ["Ustaz Ahmad Contoh"], "date_evidence": "Sabtu, 28 November 2026", "confidence": "high",
            "registration_url": "https://forms.gle/AbCdEf123", "source_url": "https://example.my/ceramah",
            "source_text": src}
    base.update(kw)
    return base


@pytest.fixture
def digest(monkeypatch):
    cfg = make_cfg(monkeypatch)
    evs = []
    for i, raw in enumerate([raw_event(),
                             raw_event(title="Kuliah Tafsir Al-Kahfi", start_date="2026-12-05",
                                       registration_url=None, source_url="https://example.my/kuliah")]):
        ev, why = coerce(raw, TODAY, END)
        assert why == "ok"
        ev["id"] = f"id{i}"
        evs.append(ev)
    stats = {"queries": 3, "pages_unique": 5}
    return report.build_digest(evs, cfg, {}, TODAY, stats, []), report.build_ics(evs, cfg)


@pytest.fixture
def tg(server, monkeypatch):
    """Route requests.post('https://api.telegram.org/...') to the fake server (real HTTP, other host)."""
    real_post = requests.post
    sent_urls = []

    def post(url, **kw):
        assert url.startswith("https://api.telegram.org/bot"), url
        sent_urls.append(url)
        return real_post(server.url + url[len("https://api.telegram.org"):], **kw)
    monkeypatch.setattr(requests, "post", post)
    server.handler = lambda req: Reply(200, {"ok": True, "result": {"message_id": len(server.requests)}})
    server.sent_urls = sent_urls
    return server


def tg_cfg(monkeypatch, **extra):
    return make_cfg(monkeypatch, TELEGRAM_TOKEN=TOKEN, TELEGRAM_CHAT_ID=CHAT, **extra)


def test_telegram_sends_header_cards_footer_and_ics(tg, digest, monkeypatch, sleeps):
    d, ics = digest
    cfg = tg_cfg(monkeypatch)
    assert deliver_mod.send_telegram(d, ics, cfg) is True
    assert tg.paths() == [f"/bot{TOKEN}/sendMessage"] * 4 + [f"/bot{TOKEN}/sendDocument"]
    assert tg.sent_urls[0] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    forms = [r.form() for r in tg.requests[:4]]
    assert all(f["chat_id"] == CHAT and f["parse_mode"] == "HTML" and f["disable_web_page_preview"] == "true"
               for f in forms)
    assert forms[0]["text"] == d["header"] and "reply_markup" not in forms[0]
    assert forms[3]["text"] == d["footer"] and "reply_markup" not in forms[3]
    for f, (text, rows, _) in zip(forms[1:3], d["cards"]):
        assert f["text"] == text
        assert json.loads(f["reply_markup"]) == {"inline_keyboard": rows}
    kb1 = json.loads(forms[1]["reply_markup"])["inline_keyboard"]
    assert kb1[0] == [{"text": "📝 Register", "url": "https://forms.gle/AbCdEf123"}]
    kb2 = json.loads(forms[2]["reply_markup"])["inline_keyboard"]
    assert all(b["text"] != "📝 Register" for row in kb2 for b in row)
    assert "&lt;Maulid&gt; &amp; Selawat" in forms[1]["text"]          # HTML-escaped for parse_mode=HTML

    parts = tg.requests[4].multipart()
    assert parts["chat_id"].value == CHAT.encode()
    assert parts["caption"].value.decode() == "All events as a calendar file (optional)"
    doc = parts["document"]
    assert doc.filename == "klang-valley-events.ics" and doc.content_type == "text/calendar"
    assert doc.value.startswith(b"BEGIN:VCALENDAR") and doc.value.count(b"BEGIN:VEVENT") == 2
    assert sleeps.count(1.1) == 4                                       # paced at ~1 msg/s


def test_telegram_cards_disabled_and_no_ics(tg, digest, monkeypatch, sleeps):
    d, _ = digest
    cfg = tg_cfg(monkeypatch)
    cfg["delivery"]["telegram_cards"] = False
    assert deliver_mod.send_telegram(d, None, cfg)
    assert tg.paths() == [f"/bot{TOKEN}/sendMessage"] * 2


def test_telegram_long_header_is_split(tg, monkeypatch, sleeps):
    header = "\n".join(f"{n}. Sat 28 Nov · " + "Ceramah panjang " * 12 for n in range(1, 60))
    d = {"header": header, "cards": [], "footer": "footer"}
    deliver_mod.send_telegram(d, None, tg_cfg(monkeypatch))
    texts = [r.form()["text"] for r in tg.requests]
    assert len(texts) >= 3 and all(len(t) <= report.TG_LIMIT for t in texts)
    assert "\n".join(texts[:-1]) == header and texts[-1] == "footer"


def test_telegram_html_parse_error_resends_as_plain_text(tg, monkeypatch, sleeps):
    def handler(req):
        if "parse_mode" in req.form():
            return Reply(400, {"ok": False, "error_code": 400,
                               "description": "Bad Request: can't parse entities: Unsupported start tag \"br\" "
                                              "at byte offset 18"})
        return Reply(200, {"ok": True, "result": {"message_id": 7}})
    tg.handler = handler
    rows = [[{"text": "🔗 Details", "url": "https://example.my/x"}]]
    deliver_mod._send_msg(tg_cfg(monkeypatch), "<b>Tom &amp; Jerry</b> <i>&lt;kuliah&gt;</i>", rows)
    first, second = (r.form() for r in tg.requests)
    assert len(tg.requests) == 2
    assert first["parse_mode"] == "HTML"
    assert "parse_mode" not in second and second["text"] == "Tom & Jerry <kuliah>"
    assert json.loads(second["reply_markup"]) == {"inline_keyboard": rows}   # not a button problem: keep them


def test_telegram_flood_control_429_waits_and_retries(tg, monkeypatch, sleeps):
    def handler(req):
        if len(tg.requests) == 1:
            return Reply(429, {"ok": False, "error_code": 429, "description": "Too Many Requests: retry after 7",
                               "parameters": {"retry_after": 7}})
        return Reply(200, {"ok": True, "result": {"message_id": 2}})
    tg.handler = handler
    deliver_mod._send_msg(tg_cfg(monkeypatch), "<b>hello</b>")
    assert len(tg.requests) == 2
    assert tg.requests[0].form() == tg.requests[1].form()         # identical retry, still HTML
    assert sleeps[0] == 7


def test_telegram_flood_wait_is_capped(tg, monkeypatch, sleeps):
    tg.handler = lambda req: (Reply(429, {"ok": False, "error_code": 429, "parameters": {"retry_after": 3600}})
                              if len(tg.requests) == 1 else Reply(200, {"ok": True}))
    deliver_mod._send_msg(tg_cfg(monkeypatch), "x")
    assert sleeps[0] == 60


def test_telegram_rejected_buttons_become_links_in_text(tg, monkeypatch, sleeps):
    tg.handler = lambda req: (Reply(400, {"ok": False, "error_code": 400,
                                          "description": "Bad Request: BUTTON_URL_INVALID"})
                              if len(tg.requests) == 1 else Reply(200, {"ok": True}))
    rows = [[{"text": "📝 Register", "url": "https://forms.gle/x"}],
            [{"text": "📅 Add to calendar", "url": "https://calendar.google.com/calendar/render?x=1"}]]
    assert deliver_mod._send_msg(tg_cfg(monkeypatch), "<b>Card</b>", rows) is True
    assert len(tg.requests) == 2
    retry = tg.requests[1].form()
    assert "reply_markup" not in retry and "parse_mode" not in retry
    assert retry["text"] == "Card\n📝 Register: https://forms.gle/x"       # calendar link left out


def test_telegram_invalid_button_urls_dropped_before_sending(tg, monkeypatch, sleeps):
    rows = [[{"text": "📝 Register", "url": "https://bad url with spaces"}],
            [{"text": "🔗 Details", "url": "javascript:alert(1)"}, {"text": "📅", "url": "https://ok.example/x"}],
            [{"text": "x", "url": None}]]
    deliver_mod._send_msg(tg_cfg(monkeypatch), "<b>Card</b>", rows)
    assert len(tg.requests) == 1
    assert json.loads(tg.requests[0].form()["reply_markup"]) == {
        "inline_keyboard": [[{"text": "📅", "url": "https://ok.example/x"}]]}


def test_telegram_flood_429_with_non_json_body(tg, monkeypatch, sleeps):
    tg.handler = lambda req: (Reply(429, "<html>Too Many Requests</html>") if len(tg.requests) == 1
                              else Reply(200, {"ok": True}))
    assert deliver_mod._send_msg(tg_cfg(monkeypatch), "x") is True
    assert len(tg.requests) == 2 and sleeps[0] == 5


def test_telegram_total_failure_is_reported(tg, digest, monkeypatch, sleeps, capsys):
    tg.handler = lambda req: Reply(400, {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"})
    d, ics = digest
    warnings = []
    done = deliver_mod.deliver(d, ics, tg_cfg(monkeypatch), "subject", warnings)
    assert "telegram" not in done
    assert any("telegram" in w for w in warnings)
    assert len(tg.requests) == 2              # header + its plain-text retry; no 30 failing cards
    assert "=" * 60 in capsys.readouterr().out                 # nothing delivered -> printed as fallback


def test_send_alert_is_plain_and_swallows_network_errors(tg, monkeypatch, sleeps):
    cfg = tg_cfg(monkeypatch)
    deliver_mod.send_alert("⚠️ crashed <Traceback>" + "x" * 5000, cfg)
    f = tg.requests[0].form()
    assert "parse_mode" not in f and len(f["text"]) == 4000 and f["text"].startswith("⚠️ crashed <Traceback>")
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")))
    deliver_mod.send_alert("again", cfg)                         # must not raise


class FakeSMTP:
    instances: list["FakeSMTP"] = []
    fail_login: Exception | None = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.logins, self.sent, self.closed = [], [], False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def login(self, user, password):
        if FakeSMTP.fail_login:
            raise FakeSMTP.fail_login
        self.logins.append((user, password))

    def send_message(self, msg):
        msg.as_bytes()                  # must serialise
        self.sent.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances, FakeSMTP.fail_login = [], None
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def mail_env(**extra):
    return dict(SMTP_USER="me@gmail.com", SMTP_PASSWORD="abcd efgh ijkl mnop", EMAIL_TO="you@example.com", **extra)


def test_email_builds_multipart_with_html_and_ics(smtp, digest, monkeypatch):
    d, ics = digest
    cfg = make_cfg(monkeypatch, **mail_env())
    subject = "Islamic events in the Klang Valley – Mon 23 Nov 2026 (2 events)"
    assert deliver_mod.send_email(d, ics, cfg, subject) is True
    s, = smtp.instances
    assert (s.host, s.port, s.timeout) == ("smtp.gmail.com", 465, 60)
    assert s.logins == [("me@gmail.com", "abcd efgh ijkl mnop")] and s.closed
    msg, = s.sent
    assert msg["Subject"] == subject and msg["From"] == "me@gmail.com" and msg["To"] == "you@example.com"
    assert msg.get_content_type() == "multipart/mixed"
    types = [p.get_content_type() for p in msg.walk()]
    assert types == ["multipart/mixed", "multipart/alternative", "text/plain", "text/html", "text/calendar"]
    plain = next(p for p in msg.walk() if p.get_content_type() == "text/plain").get_content()
    html = next(p for p in msg.walk() if p.get_content_type() == "text/html").get_content()
    assert "Ceramah Perdana" in plain and "& Selawat" in plain and "<b>" not in plain and "&amp;" not in plain
    assert '<a href="https://forms.gle/AbCdEf123">📝 Register</a>' in html
    assert "&lt;Maulid&gt; &amp; Selawat" in html
    att = next(p for p in msg.walk() if p.get_content_type() == "text/calendar")
    assert att.get_filename() == "klang-valley-events.ics"
    assert att.get_content_disposition() == "attachment"
    assert att.get_payload(decode=True).startswith(b"BEGIN:VCALENDAR")


def test_email_plain_part_keeps_angle_brackets(smtp, digest, monkeypatch):
    d, ics = digest
    deliver_mod.send_email(d, ics, make_cfg(monkeypatch, **mail_env()), "s")
    plain = next(p for p in smtp.instances[0].sent[0].walk() if p.get_content_type() == "text/plain").get_content()
    assert "Ceramah Perdana <Maulid> & Selawat" in plain
    assert "📝 Register: https://forms.gle/AbCdEf123" in plain and "**" not in plain


def test_email_custom_server_and_no_ics(smtp, digest, monkeypatch):
    d, _ = digest
    cfg = make_cfg(monkeypatch, **mail_env(SMTP_HOST="smtp.example.com", SMTP_PORT="2465"))
    deliver_mod.send_email(d, None, cfg, "s")
    s, = smtp.instances
    assert (s.host, s.port) == ("smtp.example.com", 2465)
    assert s.sent[0].get_content_type() == "multipart/alternative"


def test_email_not_configured_returns_false(smtp, digest, monkeypatch):
    d, ics = digest
    cfg = make_cfg(monkeypatch, SMTP_USER="me@gmail.com")       # password and recipient missing
    assert deliver_mod.send_email(d, ics, cfg, "s") is False and smtp.instances == []


def test_telegram_down_does_not_stop_email(smtp, digest, monkeypatch, sleeps):
    real_post = requests.post
    monkeypatch.setattr(requests, "post", lambda url, **kw: real_post(
        "http://127.0.0.1:1" + url[len("https://api.telegram.org"):], **kw))     # connection refused
    d, ics = digest
    warnings = []
    done = deliver_mod.deliver(d, ics, tg_cfg(monkeypatch, **mail_env()), "s", warnings)
    assert done == ["email"]
    assert "telegram delivery failed" in warnings
    assert len(smtp.instances[0].sent) == 1


def test_email_auth_failure_does_not_stop_telegram(tg, smtp, digest, monkeypatch, sleeps, capsys):
    smtp.fail_login = smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted.")
    d, ics = digest
    warnings = []
    done = deliver_mod.deliver(d, ics, tg_cfg(monkeypatch, **mail_env()), "s", warnings)
    assert done == ["telegram"]
    assert warnings == ["email delivery failed"]
    assert len(tg.requests) == 5
    assert "=" * 60 not in capsys.readouterr().out             # something was delivered: no console fallback


def test_no_keys_prints_digest(digest, monkeypatch, capsys, smtp):
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no HTTP expected"))
    d, ics = digest
    warnings = []
    assert deliver_mod.deliver(d, ics, make_cfg(monkeypatch), "s", warnings) == []
    out = capsys.readouterr().out
    assert "=" * 60 in out and "**🕌 Islamic events – Mon 23 Nov 2026**" in out
    assert "[📝 Register](https://forms.gle/AbCdEf123)" in out
    assert warnings == [] and smtp.instances == []
