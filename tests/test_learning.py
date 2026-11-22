"""Offline tests for agent/learning.py (query yield memory + source discovery)."""
import json
from datetime import date, timedelta

from agent import learning
from agent.learning import (plan_queries, query_key, record_run, retry_due, source_id,
                            suggest_sources, suggestions_text)

TODAY = date(2026, 11, 23)


def page(url, query=None):
    return {"url": url, "query": query, "text": "x"}


def ev(eid, *sources):
    return {"id": eid, "title": eid, "sources": list(sources)}


# ---------------------------------------------------------------- query_key
def test_query_key_strips_months_and_years():
    a = query_key("ceramah OR kuliah Shah Alam November Disember 2026")
    b = query_key("Ceramah OR kuliah Shah Alam Disember Januari 2026 2027")
    assert a == b == "ceramah or kuliah shah alam"
    assert query_key("Islamic summit Malaysia 2026 Kuala Lumpur") == "islamic summit malaysia kuala lumpur"


# ---------------------------------------------------------------- record_run
def test_record_run_counts_events_per_query():
    qs = {}
    pages = [page("https://www.a.org/x/", "q1"), page("https://b.org/y", "q1"),
             page("https://c.org/z", "q2"), page("https://t.me/s/chan", None)]
    events = [ev("e1", "https://a.org/x"), ev("e2", "https://b.org/y", "https://c.org/z"),
              ev("e3", "https://t.me/s/chan")]
    record_run(qs, ["q1", "q2", "q3"], pages, events, TODAY)
    assert qs["q1"] == {"runs": 1, "total_events": 2, "last_yield_date": "2026-11-23",
                        "consecutive_zero_runs": 0, "last_run": "2026-11-23"}
    assert qs["q2"]["total_events"] == 1
    assert qs["q3"]["total_events"] == 0 and qs["q3"]["consecutive_zero_runs"] == 1
    assert qs["q3"]["last_yield_date"] is None
    json.dumps(qs)  # must be JSON-serialisable


def test_record_run_accepts_queries_list_and_month_variants():
    qs = {}
    p = {"url": "https://a.org/x", "queries": ["talk KL November 2026", "other"], "query": None}
    record_run(qs, ["talk KL November 2026", "other"], [p], [ev("e1", "https://a.org/x")], TODAY)
    assert qs["talk kl"]["total_events"] == 1 and qs["other"]["total_events"] == 1


def test_record_run_zero_streak_and_reset():
    qs = {}
    for i in range(3):
        record_run(qs, ["q"], [page("https://a.org", "q")], [ev("e", "https://other.org")],
                   TODAY + timedelta(days=7 * i))
    assert qs["q"]["consecutive_zero_runs"] == 3 and qs["q"]["runs"] == 3
    record_run(qs, ["q"], [page("https://a.org", "q")], [ev("e", "https://a.org")], TODAY + timedelta(days=21))
    assert qs["q"]["consecutive_zero_runs"] == 0 and qs["q"]["total_events"] == 1
    assert qs["q"]["last_yield_date"] == (TODAY + timedelta(days=21)).isoformat()


def test_record_run_skips_runs_without_any_events():
    qs = {"q": {"runs": 2, "total_events": 3, "last_yield_date": "2026-11-01",
                "consecutive_zero_runs": 0, "last_run": "2026-11-16"}}
    before = json.dumps(qs, sort_keys=True)
    record_run(qs, ["q"], [page("https://a.org", "q")], [], TODAY)
    record_run(qs, [], [], [ev("e", "https://a.org")], TODAY)
    assert json.dumps(qs, sort_keys=True) == before


def test_record_run_same_day_not_double_counted():
    qs = {}
    args = (["q"], [page("https://a.org", "q")], [ev("e", "https://a.org")], TODAY)
    record_run(qs, *args)
    record_run(qs, *args)
    assert qs["q"]["runs"] == 1 and qs["q"]["total_events"] == 1


# ---------------------------------------------------------------- plan_queries
def st(runs, events, zeros=0):
    return {"runs": runs, "total_events": events, "last_yield_date": None,
            "consecutive_zero_runs": zeros, "last_run": "2026-11-16"}


def test_plan_orders_new_then_by_yield():
    qs = {"low": st(4, 1), "high": st(2, 6), "mid": st(2, 2)}
    plan = plan_queries(["low", "high", "brand new", "mid", "new2"], qs, 10, TODAY)
    assert plan == ["brand new", "new2", "high", "mid", "low"]


def test_plan_respects_max_queries():
    qs = {"a": st(1, 5)}
    # Over the cap, a slot is reserved for the least-recently-run proven query (no starvation).
    assert plan_queries(["a", "b", "c"], qs, 2, TODAY) == ["a", "b"]
    assert plan_queries(["a", "b", "c"], qs, 0, TODAY) == []


def test_plan_prunes_stats_for_removed_queries():
    qs = {"gone": st(3, 0), "kept": st(1, 1), "talk kl": st(1, 1)}
    plan_queries(["kept", "talk KL Disember 2026"], qs, 10, TODAY)
    assert set(qs) == {"kept", "talk kl"}


def test_resting_query_retried_once_every_four_weeks():
    qs = {"rest": st(5, 0, zeros=3), "good": st(1, 1)}
    weeks_run = []
    for w in range(12):
        day = TODAY + timedelta(weeks=w)
        plan = plan_queries(["rest", "good"], qs, 10, day)
        assert "good" in plan
        if "rest" in plan:
            weeks_run.append(w)
            assert plan[-1] == "rest"  # retries come after proven queries
    assert len(weeks_run) == 3 and weeks_run[1] - weeks_run[0] == 4


def test_resting_queries_are_spread_across_weeks():
    queries = [f"query number {i}" for i in range(40)]
    offsets = set()
    for q in queries:
        weeks = [w for w in range(4) if retry_due(q, TODAY + timedelta(weeks=w))]
        assert len(weeks) == 1  # exactly one retry week per 4-week cycle
        offsets.add(weeks[0])
    assert len(offsets) == 4  # not all on the same week


def test_retry_due_is_deterministic():
    assert retry_due("x y z", TODAY) == retry_due("X Y z November 2026", TODAY)


def test_resting_with_two_zero_runs_still_active():
    qs = {"q": st(2, 0, zeros=2)}
    for w in range(4):
        assert plan_queries(["q"], qs, 5, TODAY + timedelta(weeks=w)) == ["q"]


# ---------------------------------------------------------------- source discovery
def test_source_id_forms():
    assert source_id("https://t.me/SomeChannel/123") == ("t.me/somechannel", "https://t.me/s/somechannel")
    assert source_id("https://t.me/s/somechannel/55") == ("t.me/somechannel", "https://t.me/s/somechannel")
    assert source_id("https://www.Masjid.org.my/events/a?b=1") == ("masjid.org.my", "https://masjid.org.my/")
    assert source_id("https://t.me/+AbCdEf") is None
    assert source_id("") is None


def test_suggest_sources_after_two_distinct_events():
    ss = {}
    assert suggest_sources([ev("e1", "https://t.me/chan/1")], [], ss, TODAY) == []
    # the same event again next week does not count twice
    assert suggest_sources([ev("e1", "https://t.me/chan/1")], [], ss, TODAY + timedelta(days=7)) == []
    out = suggest_sources([ev("e2", "https://t.me/chan/2")], [], ss, TODAY + timedelta(days=14))
    assert out == ["https://t.me/s/chan"]
    assert ss["t.me/chan"]["events"] == 2
    json.dumps(ss)


def test_suggest_sources_skips_existing_direct_sources_and_aggregators():
    events = [ev("e1", "https://t.me/chan/1", "https://www.site.org/a", "https://peatix.com/event/1",
                 "https://www.facebook.com/x"),
              ev("e2", "https://t.me/chan/2", "https://site.org/b", "https://peatix.com/event/2",
                 "https://www.facebook.com/y")]
    ss = {}
    assert suggest_sources(events, ["https://t.me/s/Chan", "https://www.site.org/events/"], ss, TODAY) == []
    assert sorted(suggest_sources([], [], ss, TODAY)) == ["https://site.org/", "https://t.me/s/chan"]


def test_suggest_sources_max_three_ranked_by_count():
    ss = {}
    events = []
    for i, dom in enumerate(["a.org", "b.org", "c.org", "d.org"]):
        for j in range(2 + i):
            events.append(ev(f"{dom}-{j}", f"https://{dom}/p{j}"))
    out = suggest_sources(events, [], ss, TODAY)
    assert out == ["https://d.org/", "https://c.org/", "https://b.org/"]


def test_one_event_with_two_pages_from_same_site_counts_once():
    ss = {}
    suggest_sources([ev("e1", "https://a.org/1", "https://a.org/2")], [], ss, TODAY)
    assert ss["a.org"]["events"] == 1


def test_suggest_sources_prunes_old_entries():
    ss = {"old.org": {"events": 9, "first_seen": "2026-01-01", "last_seen": "2026-07-01", "ids": []},
          "fresh.org": {"events": 2, "first_seen": "2026-10-01", "last_seen": "2026-10-01", "ids": []}}
    out = suggest_sources([], [], ss, TODAY)
    assert "old.org" not in ss and out == ["https://fresh.org/"]


def test_remembered_ids_are_capped():
    ss = {}
    suggest_sources([ev(f"e{i}", "https://a.org/x") for i in range(100)], [], ss, TODAY)
    assert ss["a.org"]["events"] == 100
    assert len(ss["a.org"]["ids"]) == learning._MAX_REMEMBERED_IDS


def test_suggestions_text():
    assert suggestions_text([]) == ""
    t = suggestions_text(["https://t.me/s/chan", "https://a.org/"])
    assert "direct_sources" in t and "https://t.me/s/chan · https://a.org/" in t
