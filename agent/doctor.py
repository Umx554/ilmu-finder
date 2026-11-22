"""Pre-flight checker for first-time setup.

Run:  python -m agent.doctor              (check config, keys and sources)
      python -m agent.doctor --send-test  (also send one short test message to Telegram)
      python -m agent.doctor --offline    (config checks only, no network calls)

Each check prints ✅ (fine), ⚠️ (works, but look at it), ❌ (must fix) or ➖ (optional,
not set up), with a one-line fix. Exit code is 1 if anything is ❌, else 0.
The live Tavily check uses 1 search credit. Secret values are never printed.
"""
from __future__ import annotations

import argparse
import smtplib
import sys
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from .config import load_config

OK, WARN, FAIL, SKIP = "✅", "⚠️", "❌", "➖"
TIMEOUT = 10
TAVILY_URL = "https://api.tavily.com/search"
TG_API = "https://api.telegram.org/bot{token}/{method}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; events-agent/1.0)"}
TAVILY_FREE_CREDITS = 1000
TAVILY_WARN_CREDITS = 800
RUNS_PER_MONTH = 4.35  # weekly schedule


@dataclass
class Result:
    status: str
    title: str
    detail: str = ""
    fix: str = ""


def mask(secret: str) -> str:
    """'abcdef123456' -> '…3456'. Never show more than the last 4 characters."""
    if not secret:
        return "(not set)"
    return "…" + secret[-4:] if len(secret) > 8 else "…" + "*" * 4


def scrub(text: str, secrets: list[str]) -> str:
    """Remove secret values from any text (error messages can contain URLs with the bot token)."""
    for s in sorted((s for s in secrets if s and len(s) >= 4), key=len, reverse=True):
        text = text.replace(s, mask(s))
    return text


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:160]}"


# ---------------------------------------------------------------- config + budget
def check_config(cfg: dict) -> list[Result]:
    from .search import build_queries  # imported here so a broken search.py doesn't hide config errors

    out = [Result(OK, "config.yaml", "loaded")]
    wd = cfg.get("window_days")
    if not isinstance(wd, int) or wd <= 0:
        out.append(Result(FAIL, "Window", f"window_days = {wd!r}",
                          "Set window_days in config.yaml to a whole number of days, e.g. 30."))
    else:
        out.append(Result(OK if wd <= 90 else WARN, "Window", f"next {wd} days",
                          "" if wd <= 90 else "Over 90 days finds many stale pages; 30 is a good value."))

    s = cfg.get("search", {})
    depth = s.get("depth", "basic")
    if depth not in ("basic", "advanced"):
        out.append(Result(FAIL, "Search depth", f"depth = {depth!r}",
                          "Set search.depth in config.yaml to basic or advanced."))
        return out
    try:
        n = len(build_queries(cfg))
    except (KeyError, IndexError, ValueError) as e:
        out.append(Result(FAIL, "Search queries", _err(e),
                          "A query template has an unknown {placeholder}; use only {place} {months} {bulan} {year}."))
        return out
    per_month = n * RUNS_PER_MONTH * (2 if depth == "advanced" else 1)
    detail = (f"{n} queries/run ({depth}) ≈ {per_month:.0f} Tavily credits/month "
              f"of {TAVILY_FREE_CREDITS:,} free")
    if n == 0:
        out.append(Result(WARN, "Search budget", detail,
                          "No queries: add places or query templates, or rely on direct_sources only."))
    elif per_month > TAVILY_WARN_CREDITS:
        out.append(Result(WARN, "Search budget", detail,
                          "Too close to the free limit: lower search.max_queries, remove places, or use depth basic."))
    else:
        out.append(Result(OK, "Search budget", detail))
    return out


# ---------------------------------------------------------------- Tavily
def check_tavily(cfg: dict, offline: bool) -> list[Result]:
    key = cfg["secrets"].get("tavily", "")
    if not key:
        if cfg.get("direct_sources"):
            return [Result(WARN, "Tavily key", "TAVILY_API_KEY not set: only direct_sources will be read",
                           "Get a free key at tavily.com and put it in .env as TAVILY_API_KEY.")]
        return [Result(FAIL, "Tavily key", "TAVILY_API_KEY not set and no direct_sources: nothing to search",
                       "Get a free key at tavily.com and put it in .env as TAVILY_API_KEY.")]
    if offline:
        return [Result(OK, "Tavily key", f"set ({mask(key)}), live test skipped")]
    try:
        r = requests.post(TAVILY_URL, timeout=TIMEOUT, headers={"Authorization": f"Bearer {key}"},
                          json={"query": "masjid Kuala Lumpur ceramah", "max_results": 1,
                                "search_depth": "basic"})
    except requests.RequestException as e:
        return [Result(WARN, "Tavily search", f"could not connect ({_err(e)})",
                       "Check your internet connection and try again.")]
    code = r.status_code
    if code == 200:
        return [Result(OK, "Tavily search", f"key {mask(key)} works (used 1 credit)")]
    if code in (401, 403):
        return [Result(FAIL, "Tavily search", f"HTTP {code}: key {mask(key)} rejected",
                       "Copy the key again from app.tavily.com; it should start with tvly-.")]
    if code == 432:
        return [Result(FAIL, "Tavily search", "HTTP 432: this month's plan credits are used up",
                       "Wait for the monthly reset or lower search.max_queries.")]
    if code == 433:
        return [Result(FAIL, "Tavily search", "HTTP 433: pay-as-you-go spending limit reached",
                       "Raise the limit in the Tavily dashboard, or wait for the reset.")]
    if code == 429:
        return [Result(WARN, "Tavily search", "HTTP 429: too many requests right now",
                       "Wait a minute and run the doctor again.")]
    return [Result(WARN, "Tavily search", f"unexpected HTTP {code}",
                   "Tavily may be having problems; try again later.")]


# ---------------------------------------------------------------- LLM
def _model_listed(model: str, ids: set[str]) -> bool:
    # Gemini's OpenAI-compatible endpoint lists ids as 'models/<name>'.
    return model in ids or f"models/{model}" in ids or model.removeprefix("models/") in ids


def check_llm(cfg: dict, offline: bool) -> list[Result]:
    c, key = cfg.get("llm", {}), cfg["secrets"].get("llm", "")
    if not key:
        return [Result(FAIL, "AI key", "LLM_API_KEY not set: no events can be extracted",
                       "Create a free key at console.groq.com → API Keys and put it in .env as LLM_API_KEY.")]
    base = (c.get("base_url") or "").rstrip("/")
    if not base:
        return [Result(FAIL, "AI endpoint", "llm.base_url is empty",
                       "Set llm.base_url in config.yaml, e.g. https://api.groq.com/openai/v1.")]
    if offline:
        return [Result(OK, "AI key", f"set ({mask(key)}), live test skipped")]
    try:
        r = requests.get(f"{base}/models", timeout=TIMEOUT, headers={"Authorization": f"Bearer {key}"})
    except requests.RequestException as e:
        return [Result(WARN, "AI endpoint", f"could not connect to {base} ({_err(e)})",
                       "Check llm.base_url / LLM_BASE_URL and your internet connection.")]
    if r.status_code in (401, 403):
        return [Result(FAIL, "AI key", f"HTTP {r.status_code}: key {mask(key)} rejected by {base}",
                       "Copy the key again, and make sure it belongs to the provider in llm.base_url.")]
    if r.status_code != 200:
        return [Result(WARN, "AI endpoint", f"GET /models returned HTTP {r.status_code}",
                       "The provider may be down, or llm.base_url is wrong; it should end in /v1 or similar.")]
    try:
        ids = {m.get("id", "") for m in r.json().get("data", []) if isinstance(m, dict)}
    except ValueError:
        return [Result(WARN, "AI endpoint", "model list was not JSON",
                       "Check that llm.base_url points to an OpenAI-compatible API.")]
    out = [Result(OK, "AI key", f"{mask(key)} accepted by {base}")]
    for label, model in (("AI model", c.get("model")), ("AI fallback model", c.get("fallback_model"))):
        if not model:
            if label == "AI model":
                out.append(Result(FAIL, label, "llm.model is empty",
                                  "Set llm.model in config.yaml or LLM_MODEL in .env."))
            continue
        if _model_listed(model, ids):
            out.append(Result(OK, label, model))
        else:
            out.append(Result(WARN, label, f"{model} is not in the provider's model list (maybe retired)",
                              f"Pick a current model from {base}/models and set it in config.yaml or .env."))
    return out


# ---------------------------------------------------------------- Telegram
def _tg(token: str, method: str, **data) -> requests.Response:
    return requests.post(TG_API.format(token=token, method=method), data=data, timeout=TIMEOUT)


def _tg_desc(r: requests.Response) -> str:
    try:
        return str(r.json().get("description", ""))
    except ValueError:
        return r.text[:120]


def email_configured(cfg: dict) -> bool:
    s = cfg["secrets"]
    return bool(s.get("smtp_user") and s.get("smtp_password") and s.get("email_to"))


def check_telegram(cfg: dict, offline: bool, send_test: bool) -> list[Result]:
    s = cfg["secrets"]
    token, chat = s.get("telegram_token", ""), str(s.get("telegram_chat_id", "") or "")
    if not token and not chat:
        if email_configured(cfg):
            return [Result(SKIP, "Telegram", "not set up (e-mail will be used)")]
        return [Result(FAIL, "Delivery", "neither Telegram nor e-mail is set up: the digest will only be printed",
                       "Create a bot with @BotFather and set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env.")]
    if not token:
        return [Result(FAIL, "Telegram token", "TELEGRAM_CHAT_ID is set but TELEGRAM_TOKEN is not",
                       "Message @BotFather, send /newbot and put the token in .env as TELEGRAM_TOKEN.")]
    if offline:
        out = [Result(OK, "Telegram token", f"set ({mask(token)}), live test skipped")]
        if not chat:
            out.append(Result(FAIL, "Telegram chat", "TELEGRAM_CHAT_ID not set", _chat_tip()))
        return out
    try:
        r = _tg(token, "getMe")
    except requests.RequestException as e:
        return [Result(WARN, "Telegram", f"could not connect ({_err(e)})",
                       "Check your internet connection and try again.")]
    if not r.ok:
        return [Result(FAIL, "Telegram token", f"HTTP {r.status_code}: token {mask(token)} rejected",
                       "Copy the token again from @BotFather (it looks like 123456:ABC...).")]
    bot = (r.json().get("result") or {}).get("username", "?")
    out = [Result(OK, "Telegram bot", f"@{bot}")]
    if not chat:
        out.append(Result(FAIL, "Telegram chat", "TELEGRAM_CHAT_ID not set", _chat_tip()))
        return out
    try:
        # 'typing' shows for a few seconds and leaves no message in the chat.
        r = _tg(token, "sendChatAction", chat_id=chat, action="typing")
    except requests.RequestException as e:
        out.append(Result(WARN, "Telegram chat", f"could not connect ({_err(e)})", "Try again later."))
        return out
    if not r.ok:
        desc = _tg_desc(r)
        if "blocked" in desc.lower():
            fix = f"Open @{bot} in Telegram, unblock it and press Start."
        else:
            fix = _chat_tip()
        out.append(Result(FAIL, "Telegram chat", f"chat {chat} not reachable ({desc or r.status_code})", fix))
        return out
    out.append(Result(OK, "Telegram chat", f"bot can reach chat {chat}"))
    if send_test:
        try:
            r = _tg(token, "sendMessage", chat_id=chat,
                    text="✅ Test message from the events agent doctor. Setup looks good!")
            out.append(Result(OK, "Telegram test message", "sent; check your chat")
                       if r.ok else Result(FAIL, "Telegram test message", f"failed: {_tg_desc(r)}", _chat_tip()))
        except requests.RequestException as e:
            out.append(Result(WARN, "Telegram test message", f"could not connect ({_err(e)})", "Try again later."))
    return out


def _chat_tip() -> str:
    return ("Send your bot any message, then open https://api.telegram.org/bot<TOKEN>/getUpdates "
            "and copy the number after \"chat\":{\"id\": into TELEGRAM_CHAT_ID.")


# ---------------------------------------------------------------- e-mail
def check_email(cfg: dict, offline: bool) -> list[Result]:
    s = cfg["secrets"]
    have = {k: bool(s.get(k)) for k in ("smtp_user", "smtp_password", "email_to")}
    if not any(have.values()):
        return [Result(SKIP, "E-mail", "not set up (optional)")]
    if not all(have.values()):
        missing = ", ".join(k.upper() for k, v in have.items() if not v)
        return [Result(WARN, "E-mail", f"partly set up; missing {missing}",
                       "Set SMTP_USER, SMTP_PASSWORD and EMAIL_TO together, or remove all three.")]
    host = s.get("smtp_host") or "smtp.gmail.com"
    try:
        port = int(s.get("smtp_port") or 465)
    except ValueError:
        return [Result(FAIL, "E-mail", f"SMTP_PORT {s.get('smtp_port')!r} is not a number",
                       "Set SMTP_PORT to 465 (or leave it empty).")]
    if offline:
        return [Result(OK, "E-mail", f"set for {s['email_to']} via {host}, live test skipped")]
    try:
        with smtplib.SMTP_SSL(host, port, timeout=TIMEOUT) as smtp:
            smtp.login(s["smtp_user"], s["smtp_password"])  # login only; nothing is sent
    except smtplib.SMTPAuthenticationError:
        return [Result(FAIL, "E-mail login", f"{host} rejected the login for {s['smtp_user']}",
                       "For Gmail use an App Password (Google Account → Security → 2-Step Verification → App passwords).")]
    except (smtplib.SMTPException, OSError) as e:
        return [Result(FAIL, "E-mail", f"could not connect to {host}:{port} ({_err(e)})",
                       "Check SMTP_HOST and SMTP_PORT (Gmail: smtp.gmail.com, 465).")]
    return [Result(OK, "E-mail login", f"{s['smtp_user']} → {s['email_to']} via {host}")]


# ---------------------------------------------------------------- direct sources
def check_sources(cfg: dict, offline: bool) -> list[Result]:
    srcs = cfg.get("direct_sources") or []
    if not srcs:
        return [Result(SKIP, "Direct sources", "none listed (optional)",
                       "Adding Telegram channels of mosques you follow (https://t.me/s/<channel>) improves coverage.")]
    out = []
    for url in srcs:
        url = str(url).strip()
        if not url.startswith(("http://", "https://")):
            out.append(Result(WARN, "Source", url, "Write the full address, starting with https://."))
            continue
        is_tg = "://t.me/" in url or "://telegram.me/" in url
        if is_tg and "/s/" not in url:
            channel = url.split("/")[3] if len(url.split("/")) > 3 else "<channel>"
            out.append(Result(WARN, "Source", f"{url}: posts can't be read from this form",
                              f"Use https://t.me/s/{channel} instead."))
            continue
        if offline:
            out.append(Result(OK, "Source", f"{url} (not fetched)"))
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        except requests.RequestException as e:
            out.append(Result(WARN, "Source", f"{url}: could not fetch ({_err(e)})",
                              "Check the address in a browser; remove it if the site is gone."))
            continue
        if r.status_code != 200:
            out.append(Result(WARN, "Source", f"{url}: HTTP {r.status_code}",
                              "Check the address in a browser; remove it if the page is gone."))
        elif is_tg and "tgme_widget_message" not in r.text:
            out.append(Result(WARN, "Source", f"{url}: no public posts visible",
                              "The channel may be private or have its web preview off; pick a public channel."))
        else:
            out.append(Result(OK, "Source", url))
    return out


# ---------------------------------------------------------------- runner
def run_checks(cfg: dict, offline: bool = False, send_test: bool = False) -> list[Result]:
    results: list[Result] = []
    for fn in (lambda: check_config(cfg), lambda: check_tavily(cfg, offline),
               lambda: check_llm(cfg, offline), lambda: check_telegram(cfg, offline, send_test),
               lambda: check_email(cfg, offline), lambda: check_sources(cfg, offline)):
        try:
            results.extend(fn())
        except Exception as e:  # one broken check must not hide the others
            results.append(Result(WARN, "Check crashed", _err(e), "Please report this; other checks still ran."))
    return results


SECRET_KEYS = ("tavily", "llm", "telegram_token", "smtp_password")


def secret_values(cfg: dict) -> list[str]:
    """The values that must never be printed (addresses and chat ids are fine to show)."""
    return [str(cfg.get("secrets", {}).get(k) or "") for k in SECRET_KEYS]


def render(results: list[Result], secrets: list[str]) -> str:
    lines = []
    for r in results:
        line = f"{r.status} {r.title}" + (f": {r.detail}" if r.detail else "")
        if r.fix and r.status != OK:
            line += f"\n     → {r.fix}"
        lines.append(scrub(line, secrets))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent.doctor", description="Check your events-agent setup.")
    ap.add_argument("--send-test", action="store_true", help="send one short test message to Telegram")
    ap.add_argument("--offline", action="store_true", help="skip all network checks")
    args = ap.parse_args(argv)
    try:  # Windows consoles may not support emoji; never crash on printing
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    load_dotenv()
    print("Events agent doctor\n")
    try:
        cfg = load_config()
    except Exception as e:
        print(f"{FAIL} config.yaml: could not be read ({_err(e)})\n"
              "     → Check the indentation near the line shown; use spaces, not tabs.")
        return 1
    results = run_checks(cfg, offline=args.offline, send_test=args.send_test)
    print(render(results, secret_values(cfg)))
    fails = sum(r.status == FAIL for r in results)
    warns = sum(r.status == WARN for r in results)
    print(f"\n{fails} problem(s), {warns} warning(s). "
          + ("Fix the ❌ items, then run this again." if fails else "Ready to run: python -m agent.main --no-send"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
