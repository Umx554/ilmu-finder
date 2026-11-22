"""Pipeline: fetch -> prefilter -> extract -> validate -> Klang Valley filter -> report -> deliver.

Run:  python -m agent.main            (normal run)
      python -m agent.main --no-send  (print digest, don't send to Telegram)
      python -m agent.main --pages-file tests/sample_pages.json   (skip web search; use saved pages)
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import traceback

from dotenv import load_dotenv

from .config import OUTPUT_DIR, load_config, window
from .extract import extract
from .learning import record_run, suggest_sources, suggestions_text
from .deliver import deliver, send_alert
from .prefilter import prefilter
from .region import filter_region
from .report import build_digest, build_ics, fd, to_markdown
from .search import fetch_all, normalize_url
from .state import load_state, prune, save_state
from .validate import validate

log = logging.getLogger("agent")


def serialisable(ev: dict) -> dict:
    out = {k: v for k, v in ev.items()}
    for k in ("start", "end", "start_time", "end_time"):
        if out.get(k) is not None:
            out[k] = out[k].isoformat()
    return out


def run(args) -> int:
    cfg = load_config()
    state = load_state()
    try:
        return _run(args, cfg, state)
    finally:
        save_state(state)     # keep the AI cache even if delivery or printing crashes


def _run(args, cfg: dict, state: dict) -> int:
    today, end = window(cfg)
    stats: dict = {"today": today.isoformat()}
    warnings: list[str] = []
    prune(state, today, cfg["cache_days"])

    delivery_failed = False
    # Each stage is wrapped so that a failure later on still delivers what was found earlier.
    events: list[dict] = []
    fetched: list[dict] = []
    pipeline_ok = True
    try:
        if args.pages_file:
            pages = json.load(open(args.pages_file, encoding="utf-8"))
            for p in pages:
                p.setdefault("source", "direct")
                p["key"] = normalize_url(p["url"])
            stats.update(queries=0, pages_unique=len(pages))
        else:
            pages = fetch_all(cfg, stats, warnings, state["query_stats"])
        fetched = pages
        pages = prefilter(pages, cfg, stats)
        raw = extract(pages, cfg, state["extract_cache"], today, end, stats, warnings)
        events = validate(raw, today, end, stats)
        events = filter_region(events, stats)
    except Exception as e:
        pipeline_ok = False
        log.error("pipeline failed: %s", e, exc_info=True)
        warnings.append(f"Pipeline error ({type(e).__name__}); digest may be incomplete.")

    # Learning (no AI): which queries found events, which sites/channels keep producing them.
    suggestions = []
    if pipeline_ok and not args.pages_file and not args.no_send:   # dry runs don't teach the Learner
        try:
            record_run(state["query_stats"], stats.get("queries_ok", []), fetched, events, today)
            suggestions = suggest_sources(events, cfg["direct_sources"], state["source_stats"], today)
            stats["source_suggestions"] = suggestions
        except Exception as e:
            log.warning("learning step failed: %s", e)

    digest = build_digest(events, cfg, state["seen"], today, stats, warnings)
    tip = suggestions_text(suggestions)
    if tip:
        digest["footer"] += "\n" + tip
    ics = build_ics(events, cfg) if events else None
    subject = f"Islamic events in the Klang Valley – {fd(today)} {today.year} ({len(events)} events)"

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "digest.md").write_text(to_markdown(digest), encoding="utf-8")
    (OUTPUT_DIR / "events.json").write_text(json.dumps([serialisable(e) for e in events], indent=1,
                                                       ensure_ascii=False, default=str), encoding="utf-8")
    if ics:
        (OUTPUT_DIR / "events.ics").write_text(ics, encoding="utf-8")

    if args.no_send:
        print(to_markdown(digest))
    else:
        stats["delivered_via"] = deliver(digest, ics, cfg, subject, warnings)
        sec = cfg["secrets"]
        configured = (sec["telegram_token"] and sec["telegram_chat_id"]) or (sec["smtp_user"] and sec["email_to"])
        if configured and not stats["delivered_via"]:
            delivery_failed = True     # fail the GitHub job so GitHub e-mails you a failure notice

    sec = cfg["secrets"]
    has_channel = (sec["telegram_token"] and sec["telegram_chat_id"]) or (sec["smtp_user"] and sec["email_to"])
    # Delivered = a channel worked, or no channel is set up and the digest was printed (local use).
    delivered = not args.no_send and (bool(stats.get("delivered_via")) or not has_channel)
    for ev in events if delivered else []:   # only events you actually received count as "seen"
        state["seen"].setdefault(ev["id"], {"first_seen": today.isoformat(), "end": ev["end"].isoformat()})
    (OUTPUT_DIR / "stats.json").write_text(json.dumps({"stats": stats, "warnings": warnings}, indent=1,
                                                      default=str), encoding="utf-8")
    log.info("done: %s", json.dumps(stats, default=str))
    if delivery_failed:
        log.error("The digest could not be delivered by any channel. Run: python -m agent.doctor")
        return 1
    return 0


def _on_sigterm(*_):
    # GitHub sends SIGTERM at the job timeout; exiting normally lets run()'s finally: save the state.
    sys.exit(1)


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    for stream in (sys.stdout, sys.stderr):      # emoji-safe output on Windows consoles / redirected output
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-send", action="store_true", help="print the digest instead of sending it")
    ap.add_argument("--pages-file", help="JSON list of {url,title,text} to use instead of web search")
    args = ap.parse_args()
    try:
        sys.exit(run(args))
    except Exception:
        tb = traceback.format_exc()
        log.error(tb)
        try:
            send_alert("⚠️ Events agent crashed:\n" + tb[-1500:], load_config())
        finally:
            sys.exit(1)


if __name__ == "__main__":
    main()
