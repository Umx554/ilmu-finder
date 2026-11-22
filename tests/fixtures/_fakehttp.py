"""Tiny scripted HTTP server on 127.0.0.1 used to stand in for Tavily, Groq, Telegram and web pages.

Tests set ``server.handler = fn(req) -> Reply`` and inspect ``server.requests`` afterwards.
Everything goes over a real socket, so the agent's real ``requests`` code paths are exercised.
"""
from __future__ import annotations

import json
import signal
import threading
import traceback
from email import message_from_bytes
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit


class Reply:
    def __init__(self, status: int = 200, body: bytes | str | dict | list = b"", headers: dict | None = None,
                 content_type: str | None = None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
            content_type = content_type or "application/json"
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.status, self.body = status, body
        self.headers = dict(headers or {})
        self.headers.setdefault("Content-Type", content_type or "text/html; charset=utf-8")


class Request:
    def __init__(self, method: str, path: str, headers: dict, body: bytes):
        self.method, self.raw_path, self.headers, self.body = method, path, headers, body
        self.path = urlsplit(path).path

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def form(self) -> dict:
        """application/x-www-form-urlencoded -> {key: value}."""
        return {k: v[0] for k, v in parse_qs(self.body.decode("utf-8"), keep_blank_values=True).items()}

    def multipart(self) -> dict:
        """multipart/form-data -> {name: SimpleNamespace(value, filename, content_type)}."""
        msg = message_from_bytes(b"Content-Type: " + self.headers["content-type"].encode() + b"\r\n\r\n" + self.body,
                                 policy=email_policy)
        out = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            out[name] = SimpleNamespace(value=part.get_payload(decode=True), filename=part.get_filename(),
                                        content_type=part.get_content_type())
        return out


class FakeServer:
    def __init__(self):
        self.requests: list[Request] = []
        self.errors: list[str] = []
        self.handler = lambda req: Reply(404, "no handler")
        fake = self

        class H(BaseHTTPRequestHandler):
            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = Request(self.command, self.path, {k.lower(): v for k, v in self.headers.items()},
                              self.rfile.read(n) if n else b"")
                fake.requests.append(req)
                try:
                    rep = fake.handler(req)
                except Exception:                       # surface handler bugs in the test, not as a hang
                    fake.errors.append(traceback.format_exc())
                    rep = Reply(599, "fake handler crashed")
                self.send_response(rep.status)
                for k, v in rep.headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(rep.body)))
                self.end_headers()
                self.wfile.write(rep.body)

            do_GET = do_POST = _handle

            def log_message(self, *a):
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._srv.daemon_threads = True
        self._t = threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def start(self) -> "FakeServer":
        self._t.start()
        return self

    def stop(self):
        self._srv.shutdown()
        self._srv.server_close()

    def paths(self) -> list[str]:
        return [r.path for r in self.requests]


def openai_reply(content: str, model: str = "llama-3.3-70b-versatile", tokens: tuple[int, int] = (1200, 300)) -> Reply:
    """A chat.completions response in the OpenAI/Groq wire format."""
    return Reply(200, {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1795000000, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1], "total_tokens": sum(tokens)},
        "x_groq": {"id": "req_test"},
    })


def groq_error(status: int, message: str, code: str | None = None, etype: str = "invalid_request_error",
               headers: dict | None = None) -> Reply:
    err = {"message": message, "type": etype}
    if code:
        err["code"] = code
    return Reply(status, {"error": err}, headers=headers)


class HardTimeout:
    """SIGALRM guard so that no test can hang (pytest-timeout is not installed)."""

    def __init__(self, seconds: int = 20):
        self.seconds = seconds

    def __enter__(self):
        def boom(*_):
            raise TimeoutError(f"test exceeded {self.seconds}s (possible retry loop / hang)")
        self._old = signal.signal(signal.SIGALRM, boom)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


CLEAR_ENV = ("TAVILY_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_FALLBACK_MODEL", "LLM_DELAY",
             "MAX_PAGES", "MAX_CHARS_PER_PAGE", "MAX_QUERIES", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "SMTP_USER",
             "SMTP_PASSWORD", "SMTP_HOST", "SMTP_PORT", "EMAIL_TO", "CONFIG_FILE")
