"""Offline tests for agent/doctor.py: every network call is mocked."""
import smtplib

import pytest
import requests

from agent import doctor
from agent.config import load_config
from agent.doctor import FAIL, OK, SKIP, WARN

TG_TOKEN = "123456789:AAFakeTokenValueXYZ9876"
TAVILY_KEY = "tvly-secretkey-abcd1234"
LLM_KEY = "gsk_supersecretllmkey5678"


class Resp:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code, self._json, self.text = status, json_data, text
        self.ok = 200 <= status < 300

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeNet:
    """Routes requests.get/post by URL substring; records every call."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        for part, resp in self.routes.items():
            if part in url:
                if isinstance(resp, Exception):
                    raise resp
                return resp(url, kw) if callable(resp) else resp
        raise AssertionError(f"unexpected request: {url}")


@pytest.fixture
def cfg():
    c = load_config()
    c["direct_sources"] = []
    c["secrets"].update(tavily=TAVILY_KEY, llm=LLM_KEY, telegram_token=TG_TOKEN, telegram_chat_id="555",
                        smtp_user="", smtp_password="", email_to="", smtp_host="", smtp_port="")
    return c


GOOD = {
    "api.tavily.com": Resp(200, {"results": []}),
    "/models": Resp(200, {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "llama-3.1-8b-instant"}]}),
    "/getMe": Resp(200, {"ok": True, "result": {"username": "my_events_bot"}}),
    "/sendChatAction": Resp(200, {"ok": True}),
    "/sendMessage": Resp(200, {"ok": True}),
}


def net(monkeypatch, **override):
    routes = dict(GOOD)
    for k, v in override.items():
        routes[k.replace("__", "/")] = v
    fake = FakeNet(routes)
    monkeypatch.setattr(doctor.requests, "post", fake)
    monkeypatch.setattr(doctor.requests, "get", fake)
    return fake


def by_title(results, title):
    return [r for r in results if r.title == title]


def statuses(results):
    return {r.status for r in results}


# ---------------------------------------------------------------- helpers
def test_mask_and_scrub():
    assert doctor.mask("abcdefghijkl1234") == "…1234"
    assert doctor.mask("short") == "…****"
    assert doctor.mask("") == "(not set)"
    s = doctor.scrub(f"https://api.telegram.org/bot{TG_TOKEN}/getMe failed", [TG_TOKEN, ""])
    assert TG_TOKEN not in s and "…9876" in s


# ---------------------------------------------------------------- config / budget
def test_config_budget_ok(cfg):
    res = doctor.check_config(cfg)
    budget = by_title(res, "Search budget")[0]
    assert budget.status == OK and "credits/month" in budget.detail


def test_config_budget_warns_when_advanced(cfg):
    cfg["search"]["depth"] = "advanced"
    # the default ~34 queries stay well under budget even at 2 credits each, so force a big count
    cfg["search"]["max_queries"] = 200
    cfg["places"] = [f"Place {i}" for i in range(60)]
    budget = by_title(doctor.check_config(cfg), "Search budget")[0]
    assert budget.status == WARN and budget.fix


def test_config_bad_depth_and_window(cfg):
    cfg["search"]["depth"] = "deep"
    cfg["window_days"] = 0
    res = doctor.check_config(cfg)
    assert by_title(res, "Window")[0].status == FAIL
    assert by_title(res, "Search depth")[0].status == FAIL


def test_config_bad_placeholder(cfg):
    cfg["general_queries"] = ["talk {town}"]
    assert by_title(doctor.check_config(cfg), "Search queries")[0].status == FAIL


# ---------------------------------------------------------------- full run, all good
def test_all_good_exit_zero(cfg, monkeypatch, capsys):
    fake = net(monkeypatch)
    monkeypatch.setattr(doctor, "load_config", lambda: cfg)
    monkeypatch.setattr(doctor, "load_dotenv", lambda: None)
    assert doctor.main([]) == 0
    out = capsys.readouterr().out
    assert "@my_events_bot" in out and FAIL not in out.split("\n0 problem")[0]
    for secret in (TG_TOKEN, TAVILY_KEY, LLM_KEY):
        assert secret not in out
    # Tavily test is tiny; no message is sent without --send-test
    tav = [kw for url, kw in fake.calls if "tavily" in url][0]
    assert tav["json"]["max_results"] == 1 and tav["timeout"] <= 15
    assert not any("sendMessage" in url for url, _ in fake.calls)
    action = [kw for url, kw in fake.calls if "sendChatAction" in url][0]
    assert action["data"] == {"chat_id": "555", "action": "typing"}


def test_send_test_sends_one_message(cfg, monkeypatch):
    fake = net(monkeypatch)
    res = doctor.check_telegram(cfg, offline=False, send_test=True)
    assert by_title(res, "Telegram test message")[0].status == OK
    assert sum("sendMessage" in url for url, _ in fake.calls) == 1


def test_offline_makes_no_requests(cfg, monkeypatch):
    fake = net(monkeypatch)
    cfg["direct_sources"] = ["https://t.me/s/chan"]
    res = doctor.run_checks(cfg, offline=True)
    assert fake.calls == [] and FAIL not in statuses(res)


# ---------------------------------------------------------------- Tavily
@pytest.mark.parametrize("code,status,word", [(401, FAIL, "rejected"), (432, FAIL, "credits"),
                                              (433, FAIL, "pay-as-you-go"), (429, WARN, "429"),
                                              (500, WARN, "500")])
def test_tavily_errors(cfg, monkeypatch, code, status, word):
    net(monkeypatch, **{"api.tavily.com": Resp(code)})
    r = doctor.check_tavily(cfg, offline=False)[0]
    assert r.status == status and word in r.detail and r.fix
    assert TAVILY_KEY not in r.detail


def test_tavily_missing_key(cfg):
    cfg["secrets"]["tavily"] = ""
    assert doctor.check_tavily(cfg, False)[0].status == FAIL
    cfg["direct_sources"] = ["https://t.me/s/chan"]
    assert doctor.check_tavily(cfg, False)[0].status == WARN


def test_tavily_connection_error(cfg, monkeypatch):
    net(monkeypatch, **{"api.tavily.com": requests.ConnectionError("boom")})
    assert doctor.check_tavily(cfg, False)[0].status == WARN


# ---------------------------------------------------------------- LLM
def test_llm_missing_key(cfg):
    cfg["secrets"]["llm"] = ""
    assert doctor.check_llm(cfg, False)[0].status == FAIL


def test_llm_rejected_key(cfg, monkeypatch):
    net(monkeypatch, __models=Resp(401))
    assert doctor.check_llm(cfg, False)[0].status == FAIL


def test_llm_model_missing_warns(cfg, monkeypatch):
    net(monkeypatch, __models=Resp(200, {"data": [{"id": "llama-3.3-70b-versatile"}]}))
    res = doctor.check_llm(cfg, False)
    assert by_title(res, "AI model")[0].status == OK
    fb = by_title(res, "AI fallback model")[0]
    assert fb.status == WARN and "retired" in fb.detail


def test_llm_gemini_style_ids(cfg, monkeypatch):
    cfg["llm"].update(model="gemini-2.5-flash", fallback_model="")
    net(monkeypatch, __models=Resp(200, {"data": [{"id": "models/gemini-2.5-flash"}]}))
    res = doctor.check_llm(cfg, False)
    assert by_title(res, "AI model")[0].status == OK and not by_title(res, "AI fallback model")


def test_llm_auth_header_and_url(cfg, monkeypatch):
    fake = net(monkeypatch)
    doctor.check_llm(cfg, False)
    url, kw = fake.calls[0]
    assert url == "https://api.groq.com/openai/v1/models"
    assert kw["headers"]["Authorization"] == f"Bearer {LLM_KEY}" and kw["timeout"] <= 15


# ---------------------------------------------------------------- Telegram
def test_telegram_bad_token(cfg, monkeypatch):
    net(monkeypatch, __getMe=Resp(401, {"ok": False, "description": "Unauthorized"}))
    r = doctor.check_telegram(cfg, False, False)
    assert r[0].status == FAIL and TG_TOKEN not in r[0].detail


def test_telegram_chat_not_found_gives_getupdates_tip(cfg, monkeypatch):
    net(monkeypatch, __sendChatAction=Resp(400, {"ok": False, "description": "Bad Request: chat not found"}))
    r = by_title(doctor.check_telegram(cfg, False, False), "Telegram chat")[0]
    assert r.status == FAIL and "getUpdates" in r.fix and "chat not found" in r.detail


def test_telegram_blocked(cfg, monkeypatch):
    net(monkeypatch, __sendChatAction=Resp(403, {"ok": False,
                                                  "description": "Forbidden: bot was blocked by the user"}))
    r = by_title(doctor.check_telegram(cfg, False, False), "Telegram chat")[0]
    assert r.status == FAIL and "unblock" in r.fix


def test_telegram_missing_chat_id(cfg, monkeypatch):
    net(monkeypatch)
    cfg["secrets"]["telegram_chat_id"] = ""
    r = by_title(doctor.check_telegram(cfg, False, False), "Telegram chat")[0]
    assert r.status == FAIL and "getUpdates" in r.fix


def test_no_delivery_at_all_fails(cfg):
    cfg["secrets"].update(telegram_token="", telegram_chat_id="")
    assert doctor.check_telegram(cfg, False, False)[0].status == FAIL
    cfg["secrets"].update(smtp_user="me@gmail.com", smtp_password="pw", email_to="me@gmail.com")
    assert doctor.check_telegram(cfg, False, False)[0].status == SKIP


def test_connection_error_with_token_in_message_is_scrubbed(cfg, monkeypatch, capsys):
    net(monkeypatch, __getMe=requests.ConnectionError(f"Max retries: https://api.telegram.org/bot{TG_TOKEN}/getMe"))
    monkeypatch.setattr(doctor, "load_config", lambda: cfg)
    monkeypatch.setattr(doctor, "load_dotenv", lambda: None)
    doctor.main([])
    out = capsys.readouterr().out
    assert TG_TOKEN not in out and "…9876" in out


# ---------------------------------------------------------------- e-mail
class FakeSMTP:
    instances = []
    fail_with = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout, self.sent = host, port, timeout, False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, pw):
        if FakeSMTP.fail_with:
            raise FakeSMTP.fail_with

    def send_message(self, *a, **k):
        self.sent = True


@pytest.fixture
def smtp(monkeypatch, cfg):
    FakeSMTP.instances, FakeSMTP.fail_with = [], None
    monkeypatch.setattr(doctor.smtplib, "SMTP_SSL", FakeSMTP)
    cfg["secrets"].update(smtp_user="me@gmail.com", smtp_password="app-pass-word-wxyz", email_to="me@gmail.com")
    return FakeSMTP


def test_email_skipped_when_not_set(cfg):
    assert doctor.check_email(cfg, False)[0].status == SKIP


def test_email_partial(cfg):
    cfg["secrets"]["smtp_user"] = "me@gmail.com"
    r = doctor.check_email(cfg, False)[0]
    assert r.status == WARN and "SMTP_PASSWORD" in r.detail


def test_email_login_ok_sends_nothing(cfg, smtp):
    r = doctor.check_email(cfg, False)[0]
    assert r.status == OK
    inst = smtp.instances[0]
    assert (inst.host, inst.port) == ("smtp.gmail.com", 465) and inst.timeout <= 15 and not inst.sent


def test_email_auth_failure(cfg, smtp):
    smtp.fail_with = smtplib.SMTPAuthenticationError(535, b"bad")
    r = doctor.check_email(cfg, False)[0]
    assert r.status == FAIL and "App Password" in r.fix


def test_email_connect_failure_and_bad_port(cfg, smtp):
    smtp.fail_with = OSError("timed out")
    assert doctor.check_email(cfg, False)[0].status == FAIL
    cfg["secrets"]["smtp_port"] = "abc"
    assert doctor.check_email(cfg, False)[0].status == FAIL


def test_email_password_never_printed(cfg, smtp, monkeypatch, capsys):
    net(monkeypatch)
    monkeypatch.setattr(doctor, "load_config", lambda: cfg)
    monkeypatch.setattr(doctor, "load_dotenv", lambda: None)
    doctor.main([])
    out = capsys.readouterr().out
    assert "app-pass-word-wxyz" not in out and "me@gmail.com" in out


# ---------------------------------------------------------------- direct sources
def test_sources(cfg, monkeypatch):
    tg_page = Resp(200, text='<div class="tgme_widget_message">hi</div>')
    net(monkeypatch, **{"t.me/s/good": tg_page, "t.me/s/empty": Resp(200, text="<html></html>"),
                        "ok.org": Resp(200, text="<html>x</html>"), "gone.org": Resp(404),
                        "down.org": requests.Timeout("slow")})
    cfg["direct_sources"] = ["https://t.me/s/good", "https://t.me/badform", "https://t.me/s/empty",
                             "https://ok.org/", "https://gone.org/x", "https://down.org", "www.nohttp.org"]
    res = doctor.check_sources(cfg, False)
    st = [r.status for r in res]
    assert st == [OK, WARN, WARN, OK, WARN, WARN, WARN]
    assert "https://t.me/s/badform" in res[1].fix


def test_sources_none_listed(cfg):
    assert doctor.check_sources(cfg, False)[0].status == SKIP


# ---------------------------------------------------------------- exit codes / robustness
def test_exit_one_on_failure(cfg, monkeypatch, capsys):
    net(monkeypatch, __getMe=Resp(401, {"ok": False}))
    monkeypatch.setattr(doctor, "load_config", lambda: cfg)
    monkeypatch.setattr(doctor, "load_dotenv", lambda: None)
    assert doctor.main([]) == 1
    assert "1 problem" in capsys.readouterr().out


def test_broken_config_exit_one(monkeypatch, capsys):
    def boom():
        raise ValueError("mapping values are not allowed here")
    monkeypatch.setattr(doctor, "load_config", boom)
    monkeypatch.setattr(doctor, "load_dotenv", lambda: None)
    assert doctor.main([]) == 1
    assert "config.yaml" in capsys.readouterr().out


def test_crashing_check_does_not_stop_others(cfg, monkeypatch):
    net(monkeypatch)
    monkeypatch.setattr(doctor, "check_llm", lambda c, o: 1 / 0)
    res = doctor.run_checks(cfg)
    assert by_title(res, "Check crashed") and by_title(res, "Telegram bot")
