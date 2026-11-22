"""Regression tests for the second code-review round."""
import json
from datetime import date

import pytest

from agent import extract as extract_mod
from agent.config import load_config
from agent.learning import plan_queries, record_run
from agent.llm import LLMClient, QuotaExhausted, ServiceError
from agent.region import in_klang_valley
from agent.validate import coerce

TODAY, END = date(2026, 11, 23), date(2026, 12, 23)


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self._body)
        self.ok = status < 400

    def json(self):
        return self._body

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(response=self)


def ok_reply(content='{"events": []}'):
    return FakeResp(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                          "usage": {"total_tokens": 10}})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_DELAY", "0")
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    c = LLMClient(load_config())
    c.sent = []

    def use(*replies):
        q = list(replies)

        def post(url, json=None, **kw):
            c.sent.append(json)
            return q.pop(0)
        c.session.post = post
    c.use = use
    return c


def test_bad_key_stops_ai_immediately(client):
    client.use(FakeResp(401, {"error": {"message": "Invalid API Key"}}))
    with pytest.raises(QuotaExhausted, match="key rejected"):
        client.chat_json("s", "u")
    assert len(client.sent) == 1


def test_413_retries_with_smaller_output(client):
    client.use(FakeResp(413, {"error": {"message": "Request too large"}}), ok_reply())
    assert client.chat_json("s", "u") == {"events": []}
    assert [b["max_tokens"] for b in client.sent] == [3000, 1500]


def test_service_errors_are_not_cached(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    cfg = load_config()

    def boom(self, system, user, max_tokens=None):
        raise ServiceError("still rate limited")
    monkeypatch.setattr(LLMClient, "chat_json", boom)
    page = {"url": "https://x.my/a", "key": "https://x.my/a", "title": "t", "text": "Ceramah 28 Nov di masjid"}
    cache = {}
    for _ in range(3):
        extract_mod.extract([dict(page)], cfg, cache, TODAY, END, {}, [])
    assert cache == {}                       # a provider outage never freezes a page as "empty"


def test_unread_pages_do_not_punish_their_query():
    stats = {}
    pages = [{"url": "https://a.my/1", "query": "q0", "ai_read": True},
             {"url": "https://b.my/1", "query": "q1", "ai_read": False},   # filtered in, never read (page cap)
             {"url": "https://c.my/1", "query": "q2"}]                     # dropped by prefilter: real evidence
    events = [{"sources": ["https://a.my/1"], "id": "e1"}]
    for day in (1, 8, 15):
        record_run(stats, ["q0", "q1", "q2"], pages, events, date(2026, 11, day))
    assert "q1" not in stats or stats["q1"]["consecutive_zero_runs"] == 0
    assert stats["q2"]["consecutive_zero_runs"] == 3
    assert stats["q0"]["total_events"] == 3


def test_no_query_is_starved_when_over_the_cap():
    stats = {f"q{i}": {"runs": 2, "total_events": 10 - i, "consecutive_zero_runs": 0,
                       "last_run": f"2026-11-{10 + i:02d}", "last_yield_date": "2026-11-10"} for i in range(10)}
    plan = plan_queries([f"q{i}" for i in range(10)], stats, 5, TODAY)
    assert len(plan) == 5 and "q0" in plan     # best and least-recently-run query keeps a slot


@pytest.mark.parametrize("ev,keep", [
    ({"address": "Seksyen 2, 25200 Kuantan, Pahang"}, False),
    ({"venue": "Masjid Sunway City Ipoh"}, False),
    ({"venue": "Dewan Seksyen 7", "city": "Shah Alam"}, True),
    ({"venue": "Sunway Pyramid Convention Centre"}, True),
])
def test_weak_klang_valley_words(ev, keep):
    base = {"is_online": False, "venue": None, "city": None, "address": None, "state": None, "ai_in_kv": None}
    assert in_klang_valley({**base, **ev})[0] is keep


def test_abbreviated_venue_still_dedupes():
    from agent.validate import validate
    text = "Seminar Fiqh Muamalat 5 Disember 2026 di Institut Kefahaman Islam Malaysia (IKIM)"
    raw = lambda venue, url: {"title": "Seminar Fiqh Muamalat", "is_islamic": True, "start_date": "2026-12-05",
                              "venue": venue, "date_evidence": "5 Disember 2026", "confidence": "high",
                              "source_url": url, "source_text": text}
    out = validate([raw("IKIM", "https://a.my"), raw("Institut Kefahaman Islam Malaysia (IKIM)", "https://b.my")],
                   TODAY, END, {})
    assert len(out) == 1


def test_dry_run_does_not_teach_learner_and_seen_needs_delivery(monkeypatch, tmp_path):
    from agent import main as agent_main
    monkeypatch.setattr(agent_main, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("agent.state.STATE_DIR", tmp_path / "state")
    called = []
    monkeypatch.setattr(agent_main, "record_run", lambda *a: called.append(1))
    monkeypatch.setattr(agent_main, "fetch_all", lambda *a, **k: [])

    class Args:
        no_send, pages_file = True, None
    assert agent_main.run(Args()) == 0
    assert called == []


def test_job_fails_when_no_channel_delivers(monkeypatch, tmp_path):
    from agent import main as agent_main
    monkeypatch.setenv("TELEGRAM_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setattr(agent_main, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("agent.state.STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(agent_main, "fetch_all", lambda *a, **k: [])
    monkeypatch.setattr(agent_main, "deliver", lambda *a, **k: [])      # every channel failed

    class Args:
        no_send, pages_file = False, None
    assert agent_main.run(Args()) == 1                                    # GitHub marks the run as failed
    assert (tmp_path / "state" / "seen.json").exists()                    # state still saved


def test_search_time_budget(monkeypatch):
    from agent import search
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    cfg = load_config()
    cfg["search"]["time_budget_minutes"] = 0
    monkeypatch.setattr(search, "tavily_search", lambda *a: pytest.fail("should not search"))
    warnings = []
    search.fetch_all(cfg, {}, warnings, {})
    assert any("time budget" in w for w in warnings)


def test_fallback_none_disables(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "none")
    assert load_config()["llm"]["fallback_model"] == ""
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "")                        # unset GitHub variable -> config.yaml
    assert load_config()["llm"]["fallback_model"] == "openai/gpt-oss-20b"


def test_unsupported_option_is_dropped_and_retried(client):
    client.use(FakeResp(400, {"error": {"message": "`response_format` json_object is not supported with this model",
                                        "type": "invalid_request_error"}}), ok_reply())
    assert client.chat_json("s", "u") == {"events": []}
    assert "response_format" in client.sent[0] and "response_format" not in client.sent[1]
    assert client.sent[1]["reasoning_effort"] == "low"


def test_retired_models_give_a_clear_message(client):
    gone = {"error": {"message": "The model has been decommissioned", "code": "model_decommissioned"}}
    client.use(FakeResp(400, gone), FakeResp(400, gone))
    with pytest.raises(QuotaExhausted) as e:
        client.chat_json("s", "u")
    assert "retired" in str(e.value) and "llm.model" in str(e.value) and "daily quota" not in str(e.value)


def test_shipped_config_uses_current_groq_models():
    llm = load_config()["llm"]
    assert not any("llama-3" in m for m in (llm["model"], llm["fallback_model"]))   # retired Aug 2026


def test_unclear_location_is_kept_and_marked():
    """Live run 22 Nov 2026: 'Daurah Anak Soleh' at 'Masjid Al-Husna' (no town) was dropped."""
    from agent import report
    from agent.region import filter_region
    base = {"is_online": False, "city": None, "address": None, "state": None, "ai_in_kv": None}
    unclear = {**base, "venue": "Masjid Al-Husna"}
    jakarta = {**base, "venue": "JCC", "city": "Jakarta", "ai_in_kv": False}
    stats = {}
    kept = filter_region([unclear, jakarta], stats)
    assert kept == [unclear] and unclear["location_unclear"] is True
    assert stats["region_drops"] == {"kept, location unclear": 1, "other state": 1}
    ev, _ = coerce({"title": "Daurah Anak Soleh 2026", "is_islamic": True, "start_date": "2026-12-09",
                    "venue": "Masjid Al-Husna", "date_evidence": "9 Disember 2026", "confidence": "medium",
                    "source_url": "https://x.my/p", "source_text": "Daurah Anak Soleh 2026, 9 Disember 2026, Masjid Al-Husna"},
                   TODAY, END)
    ev.update(id="x", location_unclear=True)
    text = report.card_text(ev, 1, load_config(), {})
    assert "Masjid Al-Husna – <i>location unclear" in text
