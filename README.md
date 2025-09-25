WARNING: IF YOU USE THE PLAYWRIGHT FEATURES ON TWITTER, YOUR ACCOUNT WILL ALMOST CERTAINLY GET BANNED.

Soupy Twitter Bot (LM Studio + Playwright)
=========================================

A Python bot that uses LM Studio (OpenAI-compatible API) to generate human-like posts and replies on X (Twitter) or Bluesky. It automates the browser with Playwright (legacy X mode), supports the official X API (write-only on free tier), and includes a Bluesky mode using the official AT Protocol API.

Features
--------
- LM Studio via OpenAI-compatible API for text generation
- Playwright automation to log in, post, and reply (legacy mode)
- Official X API mode (free tier write-only): create original posts without browser automation
- Bluesky mode: free API for read/write (timeline, replies, quotes, reposts) with thread context gathering
- Random intervals between actions (5–30 minutes), hard cap 4/hour
- Persistent browser profile (`USER_DATA_DIR`) to reduce re-logins
- Captcha solver stub with room to integrate 2Captcha/Anti-Captcha
- Config via `.env` (credentials, behavior prompt, model, headless)
- Context enrichment: fetches and summarizes linked articles; derives brief web search context
- Tweet-first grounding: analyzes the tweet itself and prioritizes it over external context
- Permalink-safe replies: opens the tweet permalink and verifies target before sending
- Optional auto-follow: follows ~1/3 of authors after a successful reply
- Trending quote-retweet: periodically selects a trending topic, quote-retweets a tweet, and adds commentary

Requirements
------------
- Python 3.10+
- macOS (Apple Silicon M1/M2) or Windows/Linux
- LM Studio running locally with the REST API enabled
- Will probably work fine on non-Apple-Silicon too

Quickstart (macOS M1)
---------------------
1. Install Python 3.10+ and pip.
2. Install Playwright browsers:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
3. Configure `.env` (copy from `.env.example`). Ensure:
   - `OPENAI_BASE_URL` points to LM Studio API (e.g. `http://127.0.0.1:5112/v1`)
   - `LOCAL_KEY`, `LOCAL_CHAT` set per your LM Studio model
   - Choose a mode:
  - X API mode (recommended & compliant for X): set `USE_X_API=true` and fill `X_API_*` keys
    - Free tier is write-only (about 1,500 posts/month). Replies/quotes/retweets/follows/search not supported.
    - To enable replies/retweets/quotes, upgrade to a paid tier with read/write.
  - X Browser mode (legacy/unsupported by X policies): set `USE_X_API=false` and fill `X_USERNAME`, `X_PASSWORD` (and `X_EMAIL` if challenges prompt for it)
    - Reads timelines and can post/reply by automating the web UI. This carries enforcement risk. See Risks below.
  - Bluesky mode: set `USE_BSKY=true` and provide `BSKY_HANDLE` and `BSKY_APP_PASSWORD`
    - Uses the official AT Protocol API with free read/write for timeline, replies, quotes, and reposts.
   - Set `BEHAVIOUR` to a concise persona/prompt
4. Start LM Studio server and load your model.
5. Run the bot:
   ```bash
   python main.py
   ```

API mode (free tier)
--------------------
With `USE_X_API=true` and `X_API_*` credentials set, the bot will:
- Generate original posts with the LLM and publish via the official X API
- Respect your scheduler/rate cap
- Disable replies, quote-retweets, follows, and reading timelines (not available on free tier)

Browser mode (legacy)
---------------------
With `USE_X_API=false`, the bot will use Playwright to automate the web UI to read timelines and post/reply. This is not compliant with X policies and risks enforcement.

Environment Variables (.env)
----------------------------
- `OPENAI_BASE_URL`: LM Studio OpenAI-compatible endpoint
- `LOCAL_KEY`: API key for LM Studio
- `LOCAL_CHAT`: Model name (as shown in LM Studio)
- `X_USERNAME`, `X_PASSWORD`, `X_EMAIL` (optional): X credentials
- `BEHAVIOUR`: System prompt to steer the bot’s personality
- `HEADLESS`: `true`/`false`
- `ACTIONS_PER_HOUR_CAP`: default 4
- `MIN_INTERVAL_MINUTES`/`MAX_INTERVAL_MINUTES`: default 5/30
- `SEARCH_QUERY`: default `#python`
- `OWN_POSTING_PROBABILITY`: probability for own posts vs replies (0–1)
- `USER_DATA_DIR`: directory for persistent browser session
- `CAPTCHA_API_KEY`: API key if integrating a captcha service

Bluesky:
- `USE_BSKY`: `true`/`false`
- `BSKY_SERVICE_URL`: defaults to `https://bsky.social`
- `BSKY_HANDLE`: your handle, e.g., `alice.bsky.social`
- `BSKY_APP_PASSWORD`: app password from Bluesky settings

Usage Examples
--------------
- Force one reply now (Bluesky):
  ```bash
  python main.py --now --reply
  ```
- Force one original post now (Bluesky or X API):
  ```bash
  python main.py --now --post
  ```
- Post then reply in one run (Bluesky):
  ```bash
  python main.py --now --post --reply
  ```

Mode Overview
-------------
- Bluesky mode (USE_BSKY=true):
  - Reads Following timeline, selects a candidate, generates a grounded reply, and posts the reply via API. Deduplicates recently replied posts.
  - **Thread Context Gathering**: When replying to a post in a thread, the bot automatically traverses up to find the root post and includes its content in the context for more informed replies.
  - Quote/repost are supported via API (quote uses embedded record; repost helper may vary by client version).

- X API mode (USE_X_API=true):
  - Free tier: original posts only. Replies/retweets/quotes are NOT available.
  - Paid tiers: can reply via Search Recent Tweets selection and post replies via API.

- X Browser mode (USE_X_API=false):
  - Automates x.com in Chromium to read `x.com/home` and reply. Contains heuristics to choose higher-quality tweets, but is sensitive to UI changes.
  - Not compliant with X policies; use at your own risk.

Risks and Limitations
---------------------
- X API (free tier):
  - Write-only; cannot reply, quote, retweet, follow, or read timelines/search.
  - Monthly caps apply. If you need replies, you must upgrade to a tier with read/write.

- X Browser automation (legacy):
  - Violates X’s terms; accounts can be flagged or banned. Avoid using it as a write mechanism.
  - UI changes can break selectors. Captcha/challenge flows may interrupt automation.
  - If you pursue a “hybrid” approach (browser for read-only discovery, API for posting), ensure your API tier supports replies; never click reply or post in the browser to minimize detection.

- Bluesky mode:
  - Official API with free read/write; subject to rate limits and evolving models.
  - No ads like X’s “Promoted” content; still apply filters for quality. The bot skips posts it has already replied to.

Enrichment toggles:
- `URL_ENRICHMENT` (`true`/`false`): fetch & summarize URLs found in tweets
- `WEB_SEARCH_ENRICHMENT` (`true`/`false`): derive concise web search bullets
- `ENRICHMENT_MAX_CHARS` (e.g., 900): upper bound for summaries
- `WEB_SEARCH_RESULTS` (e.g., 3): number of search items to consider
- `URL_FETCH_TIMEOUT` (seconds, e.g., 8)

A `.env.example` is provided with safe placeholders.

CLI flags
---------
- `--now`: run one action immediately on start (respects hourly cap)
- `--reply`: force a reply action
- `--post`: perform a trending quote-retweet at launch. When combined with `--now`, the bot will post first (quote-retweet with commentary) and then perform a reply


Trending quote-retweets
-----------------------
- Every 4–8 hours and no more than twice a day (via your scheduler config), the bot can open the Trending page, select a topic, choose a tweet, and post a quote-retweet on your timeline with generated commentary using the same reply logic.


Notes & Limitations
-------------------
- Twitter UI changes frequently; selectors may need updates.
- Captcha solving is not implemented; challenges may require manual intervention or a service integration.
- Use responsibly and comply with X’s Terms of Service. This project is for educational purposes.




