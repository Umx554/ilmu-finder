"""Stage 1 - Fetcher: search queries (Tavily) + direct sources (incl. Telegram channels)."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests
from bs4 import BeautifulSoup

from .config import EN_MONTHS, BM_MONTHS, today_local, window_months
from .learning import plan_queries

log = logging.getLogger("agent.fetch")
TAVILY_URL = "https://api.tavily.com/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; events-agent/1.0)"}
TRACKING = {"fbclid", "gclid", "igshid", "si", "ref", "mibextid"}


def normalize_url(url: str) -> str:
    p = urlsplit(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_") and k.lower() not in TRACKING]
    return urlunsplit((p.scheme.lower(), p.netloc.lower().removeprefix("www."), p.path.rstrip("/"), urlencode(q), ""))


def build_queries(cfg: dict, cap: bool = True) -> list[str]:
    months, years = window_months(cfg)
    fill = {
        "months": " ".join(EN_MONTHS[m - 1] for m in months),
        "bulan": " ".join(BM_MONTHS[m - 1] for m in months),
        "year": " ".join(str(y) for y in years),
    }
    queries = [t.format(**fill) for t in cfg.get("general_queries", [])]
    for place in cfg.get("places", []):
        for t in cfg.get("place_queries", []):
            queries.append(t.format(place=place, **fill))
    seen, out = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    limit = cfg["search"]["max_queries"]
    if cap and len(out) > limit:
        log.warning("%d queries generated, capped at %d", len(out), limit)
    return out[:limit] if cap else out


def tavily_search(query: str, cfg: dict, session: requests.Session) -> list[dict]:
    s = cfg["search"]
    body = {
        "query": query,
        "search_depth": s.get("depth", "basic"),
        "max_results": s.get("max_results", 8),
        "include_raw_content": True,
    }
    if s.get("time_range"):
        body["time_range"] = s["time_range"]
    r = session.post(TAVILY_URL, json=body, timeout=60,
                     headers={"Authorization": f"Bearer {cfg['secrets']['tavily']}"})
    r.raise_for_status()
    pages = []
    for item in r.json().get("results", []):
        text = item.get("raw_content") or item.get("content") or ""
        pages.append({
            "url": item.get("url") or "",
            "title": item.get("title") or "",
            "text": text,
            "published": item.get("published_date"),
            "source": "search",
            "query": query,
        })
    return pages


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    for a in soup.find_all("a", href=True):           # keep link targets so registration URLs survive
        if a["href"].startswith("http"):
            a.append(f" ({a['href']})")
    text = soup.get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return title, text.strip()


def parse_telegram_channel(html: str, base_url: str, max_age_days: int = 45, today=None) -> list[dict]:
    """Each post on a public t.me/s/<channel> page becomes its own 'page'."""
    soup = BeautifulSoup(html, "html.parser")
    now = datetime.combine(today, datetime.min.time(), timezone.utc) if today else datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    channel = soup.select_one(".tgme_channel_info_header_title")
    channel_name = channel.get_text(strip=True) if channel else base_url
    posts = []
    for msg in soup.select(".tgme_widget_message"):
        body = msg.select_one(".tgme_widget_message_text")
        if not body:
            continue
        link = msg.select_one("a.tgme_widget_message_date")
        t = msg.select_one("time[datetime]")
        published = t["datetime"] if t else None
        if published:
            try:
                if datetime.fromisoformat(published) < cutoff:
                    continue
            except ValueError:
                pass
        posts.append({
            "url": link["href"] if link and link.get("href") else base_url,
            "title": f"{channel_name} (Telegram)",
            "text": body.get_text("\n", strip=True) + "\n" + "\n".join(
                a["href"] for a in body.find_all("a", href=True) if a["href"].startswith("http")),
            "published": published,
            "source": "telegram",
            "query": None,
        })
    return posts


def fetch_direct(url: str, session: requests.Session, today=None) -> list[dict]:
    r = session.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    if "t.me/s/" in url:
        return parse_telegram_channel(r.text, url, today=today)
    title, text = html_to_text(r.text)
    return [{"url": url, "title": title, "text": text, "published": None, "source": "direct", "query": None}]


def fetch_all(cfg: dict, stats: dict, warnings: list[str], query_stats: dict | None = None) -> list[dict]:
    session = requests.Session()
    pages: list[dict] = []
    today = today_local(cfg)
    queries, all_q = [], []
    if cfg["secrets"]["tavily"]:
        all_q = build_queries(cfg, cap=False)
        limit = cfg["search"]["max_queries"]
        # Learned plan: proven queries first, rest queries that keep finding nothing (see learning.py).
        queries = plan_queries(all_q, query_stats, limit, today) if query_stats is not None else all_q[:limit]
    ok_queries = []
    fetch_deadline = time.monotonic() + float(cfg["search"].get("time_budget_minutes", 8)) * 60
    if not cfg["secrets"]["tavily"]:
        warnings.append("No TAVILY_API_KEY: web search skipped, only direct sources used.")
    failed = 0
    for q in queries:
        if time.monotonic() > fetch_deadline:
            warnings.append("Search time budget used up; some queries skipped this week.")
            break
        try:
            pages.extend(tavily_search(q, cfg, session))
            ok_queries.append(q)          # only successful searches count towards "this query finds nothing"
        except requests.HTTPError as e:
            failed += 1
            code = e.response.status_code if e.response is not None else "?"
            log.warning("search failed (%s): %s", code, q)
            if code in (401, 403, 429, 432, 433):   # auth / quota problems: stop wasting time
                warnings.append(f"Search stopped early (HTTP {code}); check Tavily key or credits.")
                break
        except requests.RequestException as e:
            failed += 1
            log.warning("search error: %s (%s)", q, e)
    for url in cfg["direct_sources"]:
        try:
            pages.extend(fetch_direct(url, session, today))
        except requests.RequestException as e:
            log.warning("direct source failed: %s (%s)", url, e)
            warnings.append(f"Direct source failed: {url}")

    unique: dict[str, dict] = {}
    for p in pages:
        if not p.get("url") or not p.get("text"):
            continue
        key = normalize_url(p["url"])
        found_by = set(unique[key].get("queries", [])) if key in unique else set()
        if p.get("query"):
            found_by.add(p["query"])
        if key not in unique or len(p["text"]) > len(unique[key]["text"]):
            p["key"] = key
            unique[key] = p
        unique[key]["queries"] = sorted(found_by)   # every query that found this page gets credit
    stats.update(queries=len(queries), queries_all=len(all_q), queries_failed=failed, queries_ok=ok_queries,
                 pages_raw=len(pages), pages_unique=len(unique))
    log.info("[fetch] %d queries, %d results, %d unique pages", len(queries), len(pages), len(unique))
    return list(unique.values())
