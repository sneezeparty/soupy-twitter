# Soupy Bluesky Bot

An LLM‑powered Bluesky autoposter and auto‑replier. It runs on your schedule, reads your Bluesky feeds, and automatically:
- Replies to posts with tight, tone‑matched, context‑aware messages
- Posts original updates (e.g., daily news summaries with a link preview)

Under the hood, it uses a local LLM (LM Studio/OpenAI‑compatible) plus thread context, URL summaries, and lightweight web search to stay grounded and relevant.

## What this bot actually does

Soupy runs in the background and looks at your Bluesky feeds during your chosen hours. When it sees a good target, it reads the post, peeks at the surrounding thread, and, if there’s a link in the post, opens the page and skims it. Then it writes a short reply that sounds natural and fits the tone of the conversation. It stays on topic and keeps things tight so your replies don’t get cut off.

It also makes its own posts a few times per day within your operating hours. It picks a timely article, pulls out the important bits, and writes a brief take with a single link so people can check the source. Posts are spaced out so they don’t bunch up.

Replies mostly go to the original post (about seventy percent of the time), and the rest go to a first‑level reply so you still show up in active conversations. The bot mixes up who it responds to, avoids replying to the same author over and over, and won’t double‑reply to the same thread.

When a post includes a link, Soupy uses that link first. If there isn’t one, it may do a quick web search to stay grounded—but it won’t drift into unrelated topics. It deliberately avoids off‑topic or speculative sources unless the post itself is about that subject.

Everything runs locally using a model you point it at through LM Studio (or any OpenAI‑compatible API), so you’re in control.

## Setup and running

First install the dependencies and create a configuration file:
```bash
pip install -r requirements.txt
cp .env.example .env
```
Open `.env` and fill in your Bluesky handle and app password, the LM Studio URL, and your operating hours. This file is ignored by git and won’t be checked in.

Start the bot with:
```bash
python main.py
```
If you want it to post right now, run:
```bash
python main.py --postnow
```
If you want a reply immediately, run:
```bash
python main.py --now --reply
```

## Tuning the behavior

If replies feel too neutral or too sharp, change the `BEHAVIOUR` line in `.env`. Keep it short and specific for best results. You can change operating hours with `HOURS_START` and `HOURS_END`. To control original posts per day and spacing, adjust `DAILY_POSTS_PER_DAY`, `DAILY_POST_MIN_INTERVAL_HOURS`, and `DAILY_POST_MAX_INTERVAL_HOURS`.

## Safety and privacy

Your credentials live in `.env`, which is not tracked by git. The bot talks to Bluesky using the official API and respects your schedule and rate limits. If something looks off, check `soupy.log` to see what it decided and why.