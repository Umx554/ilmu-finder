# Ilmu Finder: Klang Valley Islamic events agent

A free, scheduled agent. **Every Friday at about 10 PM (Malaysia time)** it searches the web and your chosen Telegram channels for Islamic events in the Klang Valley over the next 30 days. It checks what the AI extracted against the source pages, then sends you:

1. **An overview message**: a numbered list of every event, in sections: "⭐ Watchlist speakers" (if you set any), "📌 This week", "🗓 Coming up" and "💻 Online". 📝 marks events with a registration link.
2. **One card per event**, with tap-to-open buttons: **📝 Register**, **🔗 Details** (called "Details / how to join" when there is no registration link), **📅 Add to calendar** (Google Calendar), **🗺 Open in Maps**.
3. **A stats line** and an optional `.ics` file with all events.

You go through the cards, register for the ones you want, and ignore the rest.

### Where the events come from, and where you receive them

| | What | Cost |
|---|---|---|
| **Event sources** | ① **Tavily web search**: about 34 English and Malay queries per run (mosque sites, IKIM, Peatix, Eventbrite, news, organiser pages). ② **Pages you list in `direct_sources`**, especially **public Telegram channels** of mosques, organisations and speakers. | Free |
| **Delivery** | **Telegram** (default): a bot sends the messages to your own chat. **E-mail** (optional): the same digest as an HTML e-mail with links and the `.ics` attached. Set either, or both. | Free |

**Note on the "Register" button.** It appears only when the page contains an actual registration link (Google Forms, Peatix, Eventbrite, bit.ly, WhatsApp, etc.). The AI is not allowed to make one up: the code checks that the link appears on the page and removes it if not. Many mosque talks are walk-in and have no form. For those, the card says "No registration link found – open Details".

**Why there is no "I'll attend / Skip" button.** Buttons that record your choice need a server running all the time to receive the taps, which this free setup doesn't have. The free equivalent is **📅 Add to calendar**: tapping it on an event you want saves it to your Google Calendar.

---

## 1. What changed from v1, and why

v1 was a sound design. These are the weaknesses I found, with the fix for each.

| # | v1 weakness | Why it matters | v2 fix |
|---|---|---|---|
| 1 | **Token budget.** 40 pages × about 3k tokens = **about 120k tokens/run**. | The most-cited Groq free-tier limit for `llama-3.3-70b-versatile` is about **100k tokens/day** (unconfirmed, see §10). The weekly run would hit 429 errors partway through. | Pages are trimmed to their event-relevant lines (`max_chars_per_page: 6000`, about 1.5k tokens). The default is 30 pages. The run switches to `llama-3.1-8b-instant` automatically when the daily quota runs out. |
| 2 | **One event per page.** | Listing pages (mosque schedules, Peatix, IKIM) hold many events, and v1 would keep only one. | The AI returns a list of events per page (up to 10). |
| 3 | **Trusting the AI.** The "checker agent" was only an idea. | LLMs invent speakers and prices, and get dates wrong. | **Code-based grounding check.** The date quote must appear word for word in the page. Every speaker name must appear in the text, or it is shown as "unverified". Price digits must appear, or the price is removed. Failing checks lower the confidence level. This is cheaper and more reliable than a second AI call. |
| 4 | **Re-reading the same pages every week.** | Wastes most of the AI quota. | **Extraction cache** (`state/extract_cache.json`), keyed by URL and a content hash. An unchanged page costs 0 tokens. |
| 5 | **"Seen events" would skip repeats.** | You want the full 30-day view every week, so hiding last week's events is wrong. | Repeats stay in the digest. Events new since the last digest get a 🆕 marker. |
| 6 | **Old pages.** Search returns 2023 posters. | Wasted AI calls and stale events. | Tavily `time_range: year`. The prefilter needs a date in the window's months (e.g. "Nov" or "28/11") and drops search pages that mention only other years. The validator drops past dates. |
| 7 | **Instagram/Facebook posters are invisible.** | This is the largest source of missed events. | **Public Telegram channels** (`https://t.me/s/<channel>`) are read directly at no cost, and each post becomes its own "page". Many Malaysian mosques and organisations post there. Instagram and Facebook are still not covered (see §7). |
| 8 | **Fail-safe was only an idea.** | One error meant no digest and no warning. | Each stage is wrapped, so a partial digest is still sent. A crash sends you a Telegram alert. Empty weeks still send "0 events" with stats, so you know the job ran. |
| 9 | **Telegram's 4096-character limit.** | A long digest fails to send. | Messages are split between events. If Telegram rejects the formatting, the digest is resent as plain text. |
| 10 | **The "60-day inactivity" note.** | GitHub's docs say this applies to **public** repos. | The workflow commits `state/` after each run that changed it, which counts as activity anyway. |
| 11 | **No tests ("syntax-checked only").** | You find bugs on the first live run. | 246 offline tests (at the time of writing), including fake Tavily, Groq and Telegram servers and a full simulated run on realistic Malay and English pages ([sample output](docs/sample-digest.md)). A `tests.yml` CI workflow runs them on every push that changes the code, tests, `config.yaml` or requirements. |
| 12 | **Locked to Groq.** | Free tiers change often. | Works with any OpenAI-compatible API through `llm.base_url` and `llm.model` in `config.yaml` (or `LLM_BASE_URL` and `LLM_MODEL` in `.env` on your computer): Groq, OpenRouter, or Gemini's OpenAI-compatible endpoint. |

**On the Gemini claim in v1.** The statement that "search grounding is not available on the free tier for Gemini 3, 2.5 Flash closed to new users" came from forum reports. I have not verified it, so treat it as unconfirmed. It doesn't matter for this design, which avoids depending on any single provider's search feature.

**v2.1 changes (your requests):**
- The schedule is now **Friday ~10 PM MYT** (`45 13 * * 5`: starts 9:45 PM, the digest arrives around 10 PM).
- **Geolocation removed** (no Nominatim or OSRM). This does **not** reduce AI tokens, because that stage never used the AI. It makes runs simpler and faster. Klang Valley filtering is now done by text: the venue or city must name a Klang Valley place, or the AI must judge that the venue is in the Klang Valley. Venues naming another state are dropped.
- **Event cards with Register, Details, Add-to-calendar and Maps buttons.** Registration links are checked against the page.
- **Optional e-mail delivery.**

**Also added:** an `.ics` file, a speaker watchlist (⭐ shown first), `output/` files kept 30 days as Actions artifacts, and a "Run workflow" button with a print-only option.

---

## 2. How it works: a team of seven specialist agents (plus a Doctor)

The system works as a pipeline of specialist agents, each with one job. **Only one of them, the Reader, uses an LLM.** The others are deterministic code agents. This is deliberate: code is free, fast and can't make things up, so the AI is used only where language understanding is needed. The Verifier then checks the Reader's work.

| Agent | Job | File | Uses AI? |
|---|---|---|---|
| 🔎 **Scout** | Plans this week's searches from what it has learned, runs them in English and Malay, and reads your Telegram channels | `search.py` | No |
| 🧹 **Filter** | Drops junk and stale pages and trims long pages to the useful lines and registration links (saves tokens) | `prefilter.py` | No |
| 📖 **Reader** | Reads each new page and extracts every event as structured JSON. Cached, so an unchanged page is never read twice | `extract.py`, `llm.py` | **Yes** |
| ✅ **Verifier** | Checks the Reader's output against the page: date quote, speakers, price and registration link. Removes or flags anything it can't confirm, and merges duplicates | `validate.py` | No |
| 📍 **Locator** | Keeps Klang Valley events and drops other states (place names + the Reader's judgment) | `region.py` | No |
| 📰 **Editor + Courier** | Builds the overview and event cards with Register, Details, Calendar and Maps buttons, then delivers by Telegram and/or e-mail | `report.py`, `deliver.py` | No |
| 🧠 **Learner** | Remembers which queries found events, rests queries that keep finding nothing, and suggests new sources | `learning.py` | No |
| 🩺 **Doctor** (run by you) | Pre-flight check of keys, model names, Telegram chat and sources | `doctor.py` | No |

`main.py` is the head agent. It runs them in order, makes sure a failure in one doesn't stop the digest, and saves memory (`state/`) after every run.

```
Tavily search (≤45 queries, EN + BM) ─┐
Direct sources / Telegram channels ───┴─► 1 Fetch ─► 2 Prefilter ─► 3 AI extract ─► 4 Validate + ground + dedupe
                                           (code)     (code, free)    (cached)        (code: checks AI vs source)
                                                                                              │
                     Telegram cards / e-mail + .ics ◄─ 6 Report ◄─ 5 Klang Valley text filter
                                                         (code)       (code, no API calls)
```


## 3. Project files

```
ilmu-finder/
├── agent/
│   ├── __init__.py
│   ├── config.py        # loads config.yaml + env overrides, date window
│   ├── search.py        # Tavily + direct URLs + Telegram channel parser
│   ├── prefilter.py     # keyword scoring, stale-page filter, text trimming
│   ├── llm.py           # OpenAI-compatible client: rate limits, model fallback
│   ├── extract.py       # prompt + cache
│   ├── validate.py      # field checks, grounding, dedupe/merge
│   ├── region.py        # Klang Valley check by place names (no maps)
│   ├── report.py        # overview, event cards + buttons, Google Calendar links, .ics
│   ├── deliver.py       # Telegram and/or e-mail
│   ├── state.py         # memory between runs
│   ├── learning.py      # Learner: query yield + source suggestions (no AI)
│   ├── doctor.py        # Doctor: pre-flight setup checker
│   └── main.py          # head agent: pipeline + fail-safes + CLI
├── docs/sample-digest.md  # what a Friday digest looks like
├── tests/               # offline tests (pytest) incl. fake Tavily/Groq/Telegram servers
├── state/               # committed by the workflow: AI cache, seen events, query_stats, source_stats
├── output/              # created by each run (digest.md, events.json, events.ics, stats.json); not committed
├── .github/workflows/
│   ├── digest.yml       # every Friday ~10 PM MYT
│   └── tests.yml        # runs the tests when code, tests or config change
├── config.yaml          # all settings you're meant to edit
├── requirements.txt / requirements-dev.txt
├── README.md
├── .env.example
└── .gitignore           # keeps .env and output/ out of git
```

---

## 4. Setup, step by step

### Step 1: Free accounts and keys
1. **GitHub**: github.com.
2. **Telegram bot**: message **@BotFather**, send `/newbot` and copy the token. Send your bot any message, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `"chat":{"id": ...}`.
3. **Tavily**: tavily.com, then copy the API key (free plan: 1,000 credits/month).
4. **Groq**: console.groq.com, then API Keys. Check your real limits at console.groq.com/settings/limits.

### Step 2: Run it on your computer
You need Python 3.10 or newer (GitHub uses 3.12). On Windows, tick "Add python.exe to PATH" when installing it.
```bash
cd ilmu-finder
python -m venv .venv
```
On macOS, type `python3` instead of `python` for this one command. Then activate the environment:
- Windows (Command Prompt): `.venv\Scripts\activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`. If it says running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
- macOS/Linux: `source .venv/bin/activate`

Once it is active (you see `(.venv)` at the start of the line), `python` and `pip` work on every system.
```bash
pip install -r requirements-dev.txt
```
```bash
pytest -q
```
All tests should pass before you add any keys.

Copy `.env.example` to `.env` (`copy .env.example .env` on Windows, `cp .env.example .env` on macOS) and fill in the four keys. **Then run the Doctor**, which checks every key, the AI model names, your Telegram chat and your sources, and tells you how to fix anything wrong:
```bash
python -m agent.doctor
```
`--send-test` also sends you one test Telegram message. `--offline` checks only the config and whether the keys are filled in, and uses no credits. Without `--offline`, the Tavily check uses 1 search credit.

Then try a small run:
```bash
MAX_QUERIES=3 MAX_PAGES=3 python -m agent.main --no-send
```
On Windows (PowerShell), run `$env:MAX_QUERIES=3; $env:MAX_PAGES=3; python -m agent.main --no-send` instead, then close and reopen PowerShell (and activate `.venv` again) before the real send, so the limits are cleared. (Or put `MAX_QUERIES=3` and `MAX_PAGES=3` in `.env`, and remove them again afterwards.)

Then do a real send. You should receive the overview, the cards and the stats line:
```bash
python -m agent.main
```

### Step 3: Put it on GitHub
1. Create a **private** repo, e.g. `ilmu-finder`.
2. Upload everything **except** `.env` (`.gitignore` already excludes it). Make sure the `.github/workflows/` folder is uploaded.
3. Command-line option:
```bash
git init && git add . && git commit -m "Events agent v2"
```
```bash
git branch -M main && git remote add origin https://github.com/<you>/ilmu-finder.git && git push -u origin main
```

### Step 4: Secrets and variables
Go to the repo, then **Settings**, then **Secrets and variables**, then **Actions**.
- **Secrets**: `TAVILY_API_KEY`, `LLM_API_KEY`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`
- **E-mail (optional)**: `SMTP_USER` (your Gmail address), `SMTP_PASSWORD` (a Gmail **App Password**, not your normal password: Google Account, then Security, then 2-Step Verification, then App passwords), `EMAIL_TO` (where to send). Leave these out for Telegram only. Leave out the Telegram secrets for e-mail only. The workflow sends e-mail through Gmail (`smtp.gmail.com`).

### Step 5: First run
Open **Actions**, choose **Ilmu Finder weekly digest**, then **Run workflow**. Tick "Print only" for a dry run. Open the run log and the `digest-N` artifact to check the results, then run it again without the tick. After that it runs **every Friday** on its own.

To change the day, edit the cron line in `.github/workflows/digest.yml`. The times are UTC, which is Malaysia time minus 8 hours.
- Friday ~10 PM (current): `45 13 * * 5`. It starts 9:45 PM so the digest arrives about 10 PM.
- Friday 8 AM: `0 0 * * 5`
- Saturday 8 AM: `0 0 * * 6`
- Thursday 9 PM, so you can plan the weekend the night before: `0 13 * * 4`

### Step 6: Tune it (most of the value is here)
- Add the **Telegram channels** of mosques, organisations and speakers you follow to `direct_sources`. This is the best single improvement to coverage.
- Add favourite speakers to `speaker_watchlist`.
- Edit `places` and the query templates. Each place costs 2 search credits per run.

---

## 5. Settings (`config.yaml`, env var in brackets)

The env vars are handy for testing on your computer (in `.env`). For the weekly run, either change `config.yaml`, or add a repository **Variable** with the same name (Settings → Secrets and variables → Actions → **Variables**): `LLM_MODEL`, `LLM_FALLBACK_MODEL` (`none` turns it off), `LLM_BASE_URL`, `LLM_DELAY`, `MAX_PAGES`, `MAX_QUERIES`, `SMTP_HOST`, `SMTP_PORT`. An unset variable means "use config.yaml".

| Setting | Default | Notes |
|---|---|---|
| `window_days` | 30 | |
| `this_week_days` | 7 | Events starting within this many days go in "This week" |
| `include_online` | true | Online events get their own section |
| `search.max_queries` (`MAX_QUERIES`) | 45 | Credit guard. The default places and templates make 34 queries |
| `search.depth` | basic | `advanced` costs 2 credits per query |
| `search.max_results` | 8 | Results per query |
| `search.time_range` | year | `null` to disable |
| `direct_sources` | [] | URLs and `t.me/s/...` channels |
| `speaker_watchlist` | [] | |
| `llm.model` (`LLM_MODEL`) | llama-3.3-70b-versatile | |
| `llm.fallback_model` (`LLM_FALLBACK_MODEL`) | llama-3.1-8b-instant | Used when the main model's daily quota runs out |
| `llm.base_url` (`LLM_BASE_URL`) | Groq | Any OpenAI-compatible endpoint |
| `llm.delay_seconds` (`LLM_DELAY`) | 8 | Raise it if you get many 429s |
| `llm.max_pages` (`MAX_PAGES`) | 30 | New pages read by the AI per run |
| `llm.max_chars_per_page` (`MAX_CHARS_PER_PAGE`) | 6000 | The main lever on token use |
| `llm.max_output_tokens` | 4000 | Room for up to 10 events per page |
| `llm.time_budget_minutes` | 15 | Stops AI calls after this, so the digest always gets sent |
| `search.time_budget_minutes` | 8 | Stops searching after this (search + AI stay under the 30-min job limit) |
| `delivery.telegram_cards` | true | `false` = overview list only (fewer messages) |
| `cache_days` | 35 | Cached AI results older than this are forgotten |

## 5b. Self-improving search (the Learner, free)

- Each run records which search queries led to verified events. Month names and years are ignored, so learning survives month changes.
- **New queries are always tried first. Proven queries run next.** A query with 3 empty runs in a row **rests** and is retried about once every 4 weeks, with retries spread across weeks. This saves search credits.
- Runs with zero events overall (usually an outage) are not counted.
- Sites or Telegram channels that produced **2 or more different events** and aren't in your `direct_sources` appear in the digest footer as **"💡 Sources that keep finding events"**. Add them to `direct_sources` and they'll be read every week without using search credits.
- To reset learning, delete `state/query_stats.json` or `state/source_stats.json`.

## 6. Budget check (verify the current limits yourself)

| Resource | Usage per weekly run | Free allowance (as reported Nov 2026) | Headroom |
|---|---|---|---|
| Tavily | 34 basic searches by default (cap 45) = 34 credits (≈148/month, at most ≈195) | 1,000 credits/month | About 5× |
| Groq tokens | ≤30 pages × ~2.5k = ≤75k, much less once the cache fills | ~100k TPD (70B), with an automatic fallback to the 8B model | OK |
| GitHub Actions | ~5–10 min (private repo) | 2,000 min/month on private repos | Large |
| Telegram / Gmail | About 30 messages or 1 e-mail per week | Far below limits | OK |

## 7. Honest limitations

- **Instagram and Facebook posters stay largely invisible.** Scraping them breaks their terms of service and is fragile, so this project doesn't do it. Use Telegram channels, official websites and Peatix/Eventbrite instead.
- **Text inside poster images is not read.** See the roadmap for vision/OCR.
- **The grounding check is strict text matching.** It can wrongly flag correct details, for example when the page writes "28hb Nov" and the AI quotes it differently. Flags are warnings, not deletions, except for prices and registration links.
- **Free tiers change.** The model names and limits here are correct as reported in November 2026. If a model is retired, change `llm.model` in `config.yaml` (the Doctor warns you when a model is no longer listed).
- **GitHub cron is "best effort".** Runs can start minutes to about an hour late at busy times.
- **Region filtering is by name, so it is approximate.** A venue named only "Masjid Al-Hidayah" depends on the AI's judgment, and some events from Seremban or Nilai may slip in or out. Add place names to `KV_PLACES` or `OTHER_STATES` in `agent/region.py` to tune it.
- **One message per event.** A busy month means about 30 Telegram messages each Friday. Set `delivery.telegram_cards: false` for the overview only.
- **The tests are offline.** The live API calls (Tavily, Groq, Telegram, Gmail) follow their documented interfaces, but I couldn't run them from my build environment. Expect small fixes on the first real run. The logs show which stage dropped what (`[fetch]`, `[prefilter]`, `[extract]`, `[validate]`, `[region]`, `[send]`).

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| 0 events | Read the log drop counts at each stage. If `prefilter` drops everything, your queries are too narrow. If `validate` shows mostly "past", search is returning old pages. |
| `Search stopped early (HTTP 432)` or `(HTTP 433)` | Tavily credits (432) or your pay-as-you-go limit (433) are used up for the month. `HTTP 401`/`403` means the key is wrong: run the Doctor |
| Many `rate limited, sleeping …` lines in the log | Raise `llm.delay_seconds` or lower `llm.max_pages` and `llm.max_chars_per_page` |
| `daily quota hit on …; switching to …` in the log, or `AI limit reached (<model name>)` in the digest | Normal on the first run. The cache makes later runs cheaper. |
| Events outside the Klang Valley | Add the town to `OTHER_STATES` in `agent/region.py` |
| Good events dropped as "unknown place" | Add the area to `KV_PLACES` in `agent/region.py` |
| GitHub e-mails you "Run failed", or no Telegram messages arrive. The log shows `Telegram send failed (400): … chat not found` / `The digest could not be delivered by any channel` | `TELEGRAM_CHAT_ID` is wrong, or you never sent your bot a message. If the log says the bot was blocked, unblock it and press Start. Run `python -m agent.doctor` |
| No Register button | The page had no registration link (common for walk-in talks), or the AI's link failed the check |
| E-mail fails with an auth error | Use a Gmail App Password, not your normal password |
| First-run problems | Run `python -m agent.doctor`. It names the broken key or setting and the fix. |
| A query never runs any more | It is resting after 3 empty runs. It is retried about every 4 weeks, or delete `state/query_stats.json`. |
| `AI limit reached (AI key rejected (HTTP 401) …)` | `LLM_API_KEY` is wrong or revoked. Create a new key at console.groq.com, update the secret, and run the Doctor. |
| `AI limit reached (time budget for AI calls used up)` | Rate limits were slow this week. Raise `llm.time_budget_minutes` (keep it under the job's 30-minute timeout) or lower `llm.max_pages`. |
| State not saved | The repo must allow workflow write access: Settings, Actions, General, Workflow permissions, "Read and write" |

## 9. Roadmap (in order of value for effort)

1. **Curate sources**: Telegram channels plus `site:` queries for reliable organisers. No new code needed.
2. **Poster OCR**: send images from Telegram posts to a free vision-capable model. Many announcements exist only as images.
3. **"Attend / Skip" buttons**: these need an always-on listener, e.g. a free Cloudflare Worker that receives the taps and writes them to the repo. Until then, use 📅 Add to calendar.
4. **Mid-week watchlist alert**: a second, light daily workflow that only checks direct sources for watchlisted speakers.
5. **Shared Google Calendar**: subscribe to a published `.ics` file (GitHub Pages) instead of importing it every week.
6. **Other regions**: copy `config.yaml` per region and add a workflow matrix.
7. **LangGraph/CrewAI**: worthwhile only as a learning or portfolio exercise. This pipeline is linear and deterministic, so an agent framework adds complexity without improving accuracy.

## 10. Sources checked for this guide
- Tavily credits and parameters: https://docs.tavily.com/documentation/api-credits · https://docs.tavily.com/documentation/api-reference/endpoint/search
- Groq rate limits (official, verify there): https://console.groq.com/docs/rate-limits. Third-party summaries disagree (12k vs 6k TPM).
- GitHub scheduled-workflow inactivity rule: https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows
- Telegram Bot API (inline URL buttons, message limits): https://core.telegram.org/bots/api
