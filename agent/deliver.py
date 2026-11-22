"""Delivery: Telegram (default) and/or e-mail. Whichever has its keys set is used."""
from __future__ import annotations

import json
import logging
import smtplib
import time
from email.message import EmailMessage

import requests

from .report import TG_LIMIT, split_text, to_markdown, to_plain, to_text

log = logging.getLogger("agent.deliver")


# ---------------------------------------------------------------- Telegram
def _tg(cfg: dict, method: str, **kw) -> requests.Response:
    token = cfg["secrets"]["telegram_token"]
    url = f"https://api.telegram.org/bot{token}/{method}"
    r = requests.post(url, timeout=60, **kw)
    if r.status_code == 429:                       # flood control: wait as told, then retry once
        try:
            wait = (r.json().get("parameters") or {}).get("retry_after", 5)
        except ValueError:
            wait = 5
        time.sleep(min(float(wait), 60))
        r = requests.post(url, timeout=60, **kw)
    return r


def _ok_url(u: str) -> bool:
    return isinstance(u, str) and u.startswith(("http://", "https://")) and " " not in u and len(u) <= 2000


def _send_msg(cfg: dict, text: str, rows=None) -> bool:
    """Send one message. Returns True if Telegram accepted it (possibly after a plain-text fallback)."""
    rows = [[b for b in row if _ok_url(b.get("url"))] for row in (rows or [])]
    rows = [row for row in rows if row]
    data = {"chat_id": cfg["secrets"]["telegram_chat_id"], "text": text, "parse_mode": "HTML",
            "disable_web_page_preview": "true"}
    if rows:
        data["reply_markup"] = json.dumps({"inline_keyboard": rows})
    r = _tg(cfg, "sendMessage", data=data)
    ok = r.ok
    if not ok:
        log.warning("Telegram send failed (%s): %s", r.status_code, r.text[:200])
        data.pop("parse_mode")
        plain = to_plain(text)
        if rows and "button" in r.text.lower():     # a button Telegram didn't like: put the links in the text
            data.pop("reply_markup")
            plain += "\n" + "\n".join(f'{b["text"]}: {b["url"]}' for row in rows for b in row
                                       if "calendar.google" not in b["url"])
        data["text"] = plain[:4096]
        r2 = _tg(cfg, "sendMessage", data=data)
        ok = r2.ok
        if not ok:
            log.error("Telegram fallback failed too (%s): %s", r2.status_code, r2.text[:200])
    time.sleep(1.1)                                 # stay under ~1 message/second per chat
    return ok


def send_telegram(digest: dict, ics: str | None, cfg: dict) -> bool:
    s = cfg["secrets"]
    if not (s["telegram_token"] and s["telegram_chat_id"]):
        return False
    header_ok = all([_send_msg(cfg, part) for part in split_text(digest["header"])])
    if not header_ok:   # wrong chat id / blocked bot: don't spam 30 failing cards, report the failure
        raise RuntimeError("Telegram rejected the digest (check TELEGRAM_CHAT_ID; run python -m agent.doctor)")
    failed = 0
    if cfg.get("delivery", {}).get("telegram_cards", True):
        for text, rows, _ in digest["cards"]:
            failed += not _send_msg(cfg, text[:TG_LIMIT], rows)
    _send_msg(cfg, digest["footer"])
    if ics:
        _tg(cfg, "sendDocument", data={"chat_id": s["telegram_chat_id"],
                                       "caption": "All events as a calendar file (optional)"},
            files={"document": ("klang-valley-events.ics", ics.encode(), "text/calendar")})
    log.info("[send] Telegram: header + %d cards (%d failed)", len(digest["cards"]), failed)
    return True


# ---------------------------------------------------------------- E-mail
def email_html(digest: dict) -> str:
    def br(t):
        return t.replace("\n", "<br>")
    cards = "".join(
        f'<div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0">'
        f'{br(text)}<br><br>{links}</div>' for text, _, links in digest["cards"])
    return (f'<div style="font-family:Arial,sans-serif;font-size:14px;max-width:640px">'
            f'{br(digest["header"])}{cards}<p style="color:#666">{br(digest["footer"])}</p></div>')


def send_email(digest: dict, ics: str | None, cfg: dict, subject: str) -> bool:
    s = cfg["secrets"]
    if not (s.get("smtp_user") and s.get("smtp_password") and s.get("email_to")):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, s["smtp_user"], s["email_to"]
    msg.set_content(to_text(digest))
    msg.add_alternative(email_html(digest), subtype="html")
    if ics:
        msg.add_attachment(ics.encode(), maintype="text", subtype="calendar", filename="klang-valley-events.ics")
    with smtplib.SMTP_SSL(s.get("smtp_host") or "smtp.gmail.com", int(s.get("smtp_port") or 465), timeout=60) as smtp:
        smtp.login(s["smtp_user"], s["smtp_password"])
        smtp.send_message(msg)
    log.info("[send] e-mail sent to %s", s["email_to"])
    return True


def deliver(digest: dict, ics: str | None, cfg: dict, subject: str, warnings: list[str]) -> list[str]:
    """Send through every configured channel. Returns the channels that worked."""
    done = []
    for name, fn in (("telegram", lambda: send_telegram(digest, ics, cfg)),
                     ("email", lambda: send_email(digest, ics, cfg, subject))):
        try:
            if fn():
                done.append(name)
        except Exception as e:      # one channel failing must not stop the other
            log.error("%s delivery failed: %s", name, e)
            warnings.append(f"{name} delivery failed")
    if not done:
        print("\n" + "=" * 60 + "\n" + to_markdown(digest) + "\n" + "=" * 60)
        log.info("[send] no delivery keys set; printed digest instead")
    return done


def send_alert(text: str, cfg: dict) -> None:
    s = cfg["secrets"]
    if s["telegram_token"] and s["telegram_chat_id"]:
        try:
            _tg(cfg, "sendMessage", data={"chat_id": s["telegram_chat_id"], "text": text[:4000]})
        except requests.RequestException:
            pass
