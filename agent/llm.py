"""Minimal client for any OpenAI-compatible chat API (Groq, OpenRouter, Gemini's OpenAI endpoint, ...)."""
from __future__ import annotations

import json
import logging
import re
import time

import requests

log = logging.getLogger("agent.llm")

MODEL_GONE_CODES = {"model_not_found", "model_decommissioned"}
DAILY_LIMIT_RE = re.compile(r"\((?:tpd|rpd)\)|per day", re.I)


class QuotaExhausted(Exception):
    """Daily quota for every configured model is used up (or the run's time budget is spent)."""


class ServiceError(Exception):
    """The provider failed (rate limit, network, server error). Not the page's fault: never cached."""


class BadResponse(Exception):
    """The model replied, but not with usable JSON (e.g. cut off). The page is skipped and NOT cached."""


def _error(r: requests.Response) -> dict:
    try:
        return (r.json() or {}).get("error") or {}
    except ValueError:
        return {}


def _retry_after(r: requests.Response, default: float = 20) -> float:
    try:
        return float(r.headers.get("retry-after") or default)
    except ValueError:              # an HTTP-date instead of seconds
        return default


def parse_json(content: str) -> dict | None:
    """Accept bare JSON, ```json fenced JSON, or JSON with prose around it."""
    content = (content or "").strip()
    for candidate in (content, re.sub(r"^```(?:json)?\s*|\s*```$", "", content)):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict) and "events" not in v and "title" in v:
                return {"events": [v]}
            return v if isinstance(v, dict) else {"events": v} if isinstance(v, list) else None
        except json.JSONDecodeError:
            pass
    for open_c, close_c in (("{", "}"), ("[", "]")):          # JSON object or list inside prose
        start, end = content.find(open_c), content.rfind(close_c)
        if start != -1 and end > start:
            try:
                v = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(v, list):
                return {"events": v}
            if isinstance(v, dict):
                return v if "events" in v or "title" not in v else {"events": [v]}   # a lone event object
    return None


class LLMClient:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.base_url = c["base_url"].rstrip("/")
        self.models = [m for m in (c["model"], c.get("fallback_model")) if m]
        self.delay = float(c["delay_seconds"])
        self.temperature = c.get("temperature", 0)
        self.max_tokens = int(c.get("max_output_tokens", 4000))
        # Optional extras; dropped automatically if a provider/model rejects them.
        self.extras = {"response_format": {"type": "json_object"}}
        if c.get("reasoning_effort"):
            self.extras["reasoning_effort"] = c["reasoning_effort"]   # gpt-oss / reasoning models on Groq
        self.dropped: list[str] = []
        self.key = cfg["secrets"]["llm"]
        self.session = requests.Session()
        self._last_call = 0.0
        # Whole-run budget so a stuck rate limit can't hit the GitHub job timeout before the digest is sent.
        self.deadline = time.monotonic() + float(c.get("time_budget_minutes", 15)) * 60
        self.calls = 0
        self.tokens = 0

    @property
    def model(self) -> str:
        if not self.models:
            raise QuotaExhausted("no models left")
        return self.models[0]

    def _sleep(self, seconds: float):
        if time.monotonic() + seconds > self.deadline:
            raise QuotaExhausted("time budget for AI calls used up")
        time.sleep(seconds)

    def _wait(self):
        gap = time.monotonic() - self._last_call
        if gap < self.delay:
            self._sleep(self.delay - gap)
        self._last_call = time.monotonic()

    def _drop_model(self, why: str):
        dead = self.models.pop(0)
        self.dropped.append(f"{dead}: {why}")
        log.warning("%s on %s; switching to %s", why, dead, self.models[0] if self.models else "none")
        if not self.models:
            raise QuotaExhausted("no AI model available – " + "; ".join(self.dropped))

    def chat_json(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        """Send one request and return parsed JSON. Handles 429s, 5xx, model fallback."""
        attempts, bad_json = 0, 0
        while True:
            attempts += 1
            if attempts > 8:
                raise ServiceError("too many attempts")
            self._wait()
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                **self.extras,
            }
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", json=body, timeout=90,
                                      headers={"Authorization": f"Bearer {self.key}"})
            except requests.RequestException as e:
                log.info("network error (%s), retrying", e)
                self._sleep(10)
                continue

            if r.status_code == 429:
                if DAILY_LIMIT_RE.search(_error(r).get("message", "") or r.text):
                    self._drop_model("daily quota used up")
                    continue
                if attempts >= 6:
                    raise ServiceError("still rate limited after 6 attempts")
                wait = min(_retry_after(r), 120)
                log.info("rate limited, sleeping %.0fs", wait)
                self._sleep(wait)
                continue
            if r.status_code in (401, 403):     # bad key: stop now instead of failing every page
                raise QuotaExhausted(f"AI key rejected (HTTP {r.status_code}); check LLM_API_KEY")
            if r.status_code == 413 and body["max_tokens"] > 1500:
                max_tokens = 1500               # request too large for the per-minute limit: ask for less output
                log.info("413 request too large; retrying with max_tokens=1500")
                continue
            if r.status_code in (400, 404) and _error(r).get("code") in MODEL_GONE_CODES:
                self._drop_model("model retired or not found – change llm.model in config.yaml "
                                 "(see console.groq.com/docs/models)")
                continue
            if r.status_code == 400:
                msg = (_error(r).get("message") or "").lower()
                rejected = [k for k in self.extras if k in msg]
                if rejected:                    # e.g. a model without JSON mode: retry without that option
                    for k in rejected:
                        log.info("model rejected option %s; retrying without it", k)
                        self.extras.pop(k)
                    continue
            if r.status_code >= 500:
                log.info("server error %s, retrying", r.status_code)
                self._sleep(15)
                continue
            r.raise_for_status()

            data = r.json()
            self.calls += 1
            self.tokens += (data.get("usage") or {}).get("total_tokens", 0)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise BadResponse("reply was cut off (max tokens)")
            parsed = parse_json(choice["message"].get("content") or "")
            if parsed is not None:
                return parsed
            bad_json += 1
            if bad_json >= 2:
                raise BadResponse("unparseable JSON twice")
