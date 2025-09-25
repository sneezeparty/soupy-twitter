from __future__ import annotations

import random
import time
from typing import Optional
import argparse

from loguru import logger
from playwright.sync_api import sync_playwright

from bot.config import AppConfig
from bot.llm_client import LLMClient
from bot.scheduler import IntervalScheduler, RateLimiter, run_loop
from bot.twitter import TwitterBot
from bot.x_api import XApiBot
from bot.bsky import BskyBot
from bot.context_enricher import ContextEnricher


def choose_action(own_posting_probability: float) -> str:
    return "own" if random.random() < own_posting_probability else "reply"


def main() -> None:
    parser = argparse.ArgumentParser(description="Soupy Twitter Bot")
    parser.add_argument("--now", action="store_true", help="Run one action immediately on start (respects hourly cap)")
    parser.add_argument("--reply", action="store_true", help="Force a reply action (overrides config/auto)")
    parser.add_argument("--post", action="store_true", help="Force a trending quote-retweet action at launch; if combined with --now, do post first then reply")
    args = parser.parse_args()
    config = AppConfig.from_env()
    config.validate()

    logger.add("soupy.log", rotation="1 MB", retention=5)
    logger.info("Starting Soupy Twitter Bot")

    llm = LLMClient(config)
    rate_limiter = RateLimiter(config.actions_per_hour_cap)
    scheduler = IntervalScheduler(config.min_interval_minutes, config.max_interval_minutes)

    # Choose implementation based on mode and feature availability
    if config.use_bsky:
        bot = BskyBot(config, llm)
        bot.start()
        enricher = ContextEnricher(config, llm)

        try:
            def bsky_action() -> None:
                now_local = time.localtime()
                hour = now_local.tm_hour
                start_h = config.operating_start_hour % 24
                end_h = config.operating_end_hour % 24
                within_hours = start_h <= hour <= end_h if start_h <= end_h else (hour >= start_h or hour <= end_h)
                if not within_hours:
                    logger.info("Outside operating hours ({}-{}); skipping action", start_h, end_h)
                    return
                # Determine action similar to X flow; default to reply
                if args.reply and args.post:
                    action = "post_then_reply"
                elif args.reply:
                    action = "reply"
                elif args.post:
                    action = "post"
                elif force_reply_first_run:
                    action = "reply"
                else:
                    action = "reply"
                # Honor reply_only
                if config.reply_only and action != "reply":
                    action = "reply"

                def do_reply_once() -> bool:
                    post_text = bot.prepare_reply_from_timeline()
                    if not post_text:
                        logger.info("Bsky: no timeline target found for reply")
                        return False
                    
                    # Get current post context
                    ctx = bot.get_current_post_context() or {}
                    
                    # Get thread context including root post
                    current_uri = ctx.get("uri")
                    if current_uri:
                        thread_context = bot.get_thread_context(current_uri)
                        if thread_context:
                            # Merge thread context with current context
                            ctx = {**ctx, **thread_context}
                            logger.info("Bsky: enhanced context with thread information")
                        else:
                            logger.info("Bsky: could not get thread context, using current post only")
                    
                    inferred_topic = llm.infer_topic(post_text)
                    if inferred_topic:
                        ctx = {**ctx, "topic": inferred_topic}
                    context_str = enricher.build_context(post_text, ctx)
                    # Prefer Bluesky-specific reply generator for longer, link-friendly replies
                    reply_text = llm.generate_bsky_reply(post_text, context_str if context_str else None)
                    if bot.send_prepared_reply(reply_text):
                        logger.info("Bsky mode: reply succeeded")
                        return True
                    logger.info("Bsky mode: reply failed")
                    return False

                if action in ("reply", "post_then_reply"):
                    if not do_reply_once() and action == "post_then_reply":
                        logger.info("Bsky: falling back to own post after failed reply")
                    elif action == "reply":
                        return
                if action in ("post", "post_then_reply"):
                    topic_hint = None
                    # Use Bluesky-specific own post generator to allow longer posts with one URL
                    context_for_post = None
                    post_text = llm.generate_bsky_post(topic_hint, context_for_post)
                    if not post_text:
                        logger.warning("Bsky mode: no post generated")
                        return
                    ok = bot.create_post(post_text)
                    logger.info("Bsky mode: post {}", "succeeded" if ok else "failed")

            # Force first-run reply if --now is used without explicit flags
            force_reply_first_run = bool(args.now and not args.reply and not args.post)
            run_loop(bsky_action, rate_limiter, scheduler, run_immediately=(args.now or args.post))
        finally:
            bot.stop()
        return
    elif config.use_x_api:
        bot = XApiBot(config, llm)
        bot.start()
        enricher = ContextEnricher(config, llm)

        # Run API-mode loop (original posts only)
        try:
            def api_mode_action() -> None:
                now_local = time.localtime()
                hour = now_local.tm_hour
                start_h = config.operating_start_hour % 24
                end_h = config.operating_end_hour % 24
                within_hours = start_h <= hour <= end_h if start_h <= end_h else (hour >= start_h or hour <= end_h)
                if not within_hours:
                    logger.info("Outside operating hours ({}-{}); skipping action", start_h, end_h)
                    return
                # If API tier allows reads, attempt a reply using API search
                can_read = bool(config.x_api_bearer_token)
                if can_read and not config.reply_only:
                    tweet_text = bot.search_and_prepare_reply(config.search_query)
                    if tweet_text:
                        ctx = bot.get_current_tweet_context() or {}
                        # Use tweet text, enrich, generate reply
                        inferred_topic = llm.infer_topic(tweet_text)
                        if inferred_topic:
                            ctx = {**ctx, "topic": inferred_topic}
                        context_str = ContextEnricher(config, llm).build_context(tweet_text, ctx)
                        reply_text = llm.generate_reply(tweet_text, context_str if context_str else None)
                        if bot.send_prepared_reply(reply_text):
                            logger.info("API mode: reply succeeded")
                            return
                        else:
                            logger.info("API mode: reply failed; falling back to own post")
                # Otherwise, or on failure, generate an original post and publish via API
                topic_hint = None
                post_text = llm.generate_own_post(topic_hint)
                if not post_text:
                    logger.warning("API mode: no post generated")
                    return
                ok = bot.create_post(post_text)
                logger.info("API mode: post {}", "succeeded" if ok else "failed")

            run_loop(api_mode_action, rate_limiter, scheduler, run_immediately=(args.now or args.post))
        finally:
            bot.stop()
        return
    else:
        with sync_playwright() as p:
            bot = TwitterBot(config, p)
            bot.start()
            enricher = ContextEnricher(config, llm)
        

        # Trending schedule state (2/day, 4–8h apart)
        trending_last_24h: list[float] = []
        next_trending_at: Optional[float] = None

        def schedule_next_trending(now_s: float) -> None:
            nonlocal next_trending_at
            delay = random.randint(4 * 3600, 8 * 3600)
            next_trending_at = now_s + delay

        def trending_due(now_s: float) -> bool:
            # prune
            nonlocal trending_last_24h
            cutoff = now_s - 24 * 3600
            trending_last_24h = [t for t in trending_last_24h if t >= cutoff]
            if len(trending_last_24h) >= 2:
                return False
            return next_trending_at is not None and now_s >= next_trending_at

        # initialize trending schedule
        schedule_next_trending(time.time())

        def do_action() -> None:
            nonlocal force_reply_first_run
            # Enforce hours of operation
            now_local = time.localtime()
            hour = now_local.tm_hour
            start_h = config.operating_start_hour % 24
            end_h = config.operating_end_hour % 24
            within_hours = start_h <= hour <= end_h if start_h <= end_h else (hour >= start_h or hour <= end_h)
            if not within_hours:
                logger.info("Outside operating hours ({}-{}); skipping action", start_h, end_h)
                return
            if args.reply and args.post:
                # If both set, do both: post then reply
                action = "post_then_reply"
            elif args.reply:
                action = "reply"
            elif args.post:
                action = "post"
            elif force_reply_first_run:
                action = "reply"
                force_reply_first_run = False
            else:
                # Default cadence: reply only
                action = "reply"
            # Opportunistic trending post if due (only when enabled)
            now_s = time.time()
            trending_by_schedule = config.trending_quotes_enabled and trending_due(now_s)
            trending_by_flag = config.trending_quotes_enabled and action in ("post", "post_then_reply")
            if (not config.use_x_api) and (trending_by_flag or (trending_by_schedule and action not in ("post", "post_then_reply"))):
                # Prepare a quote-retweet from home selection (politics preferred)
                qt_text = bot.prepare_quote_retweet_from_home_selection()
                if qt_text:
                    ctx = bot.get_current_tweet_context() or {}
                    if ctx.get("text"):
                        qt_text = ctx["text"]
                    inferred_topic = llm.infer_topic(qt_text)
                    if inferred_topic:
                        ctx = {**ctx, "topic": inferred_topic}
                    context_str = enricher.build_context(qt_text, ctx)
                    commentary = llm.generate_reply(qt_text, context_str if context_str else None)
                    if bot.send_prepared_quote(commentary):
                        # Follow ~1/3 of authors after quote-retweeting as well
                        if random.random() < (1.0/3.0):
                            ok = bot.follow_author_from_context()
                            logger.info("Follow attempt after quote: {}", "success" if ok else "skipped/failed")
                        trending_last_24h.append(now_s)
                        schedule_next_trending(now_s)
                else:
                    logger.warning("Trending quote-retweet preparation failed; skipping post")
                # If this was a scheduled trending post (not a --post_then_reply), continue to reply instead of returning,
                # so replies are not starved when trending quotes are enabled.
                # Previously this returned early, which could result in only quotes being posted.
                if trending_by_schedule and not trending_by_flag and action != "reply":
                    action = "reply"
                # If explicitly only post, return; if post_then_reply, fall through to reply
                if action == "post":
                    return
            if (not config.use_x_api) and (action == "reply" or action == "post_then_reply"):
                # Contextual reply from home timeline: choose a tweet first, then generate the reply to its text
                tweet_text = bot.prepare_reply_from_home_timeline()
                if not tweet_text:
                    logger.warning("No tweet text available for contextual reply; skipping")
                    return
                ctx = bot.get_current_tweet_context() or {}
                # Use canonical text from permalink context if available
                if ctx.get("text"):
                    tweet_text = ctx["text"]
                logger.info("Context: original tweet => {}", tweet_text)
                logger.info("Context: base ctx => {}", ctx)
                # Infer a concrete topic for grounding
                inferred_topic = llm.infer_topic(tweet_text)
                if inferred_topic:
                    ctx = {**ctx, "topic": inferred_topic}
                context_str = enricher.build_context(tweet_text, ctx)
                reply_text = llm.generate_reply(tweet_text, context_str if context_str else None)
                if bot.send_prepared_reply(reply_text):
                    # Follow ~1/3 of authors after replying
                    if random.random() < (1.0/3.0):
                        ok = bot.follow_author_from_context()
                        logger.info("Follow attempt after reply: {}", "success" if ok else "skipped/failed")

        try:
            # In API mode (free tier), restrict to own posts only
            if config.use_x_api:
                def api_mode_action() -> None:
                    now_local = time.localtime()
                    hour = now_local.tm_hour
                    start_h = config.operating_start_hour % 24
                    end_h = config.operating_end_hour % 24
                    within_hours = start_h <= hour <= end_h if start_h <= end_h else (hour >= start_h or hour <= end_h)
                    if not within_hours:
                        logger.info("Outside operating hours ({}-{}); skipping action", start_h, end_h)
                        return
                    # Generate an original post and publish via API
                    topic_hint = None
                    post_text = llm.generate_own_post(topic_hint)
                    if not post_text:
                        logger.warning("API mode: no post generated")
                        return
                    ok = bot.create_post(post_text)
                    logger.info("API mode: post {}", "succeeded" if ok else "failed")

                # Force run if requested and then proceed on scheduler
                run_loop(api_mode_action, rate_limiter, scheduler, run_immediately=(args.now or args.post))
            else:
                # Force first-run reply if --now is used without explicit --post/--reply.
                # If --post is used with --now, do post first then reply.
                force_reply_first_run = bool(args.now and not args.reply and not args.post)
                # Trigger an immediate run when --now OR --post is specified
                run_loop(do_action, rate_limiter, scheduler, run_immediately=(args.now or args.post))
        finally:
            bot.stop()


if __name__ == "__main__":
    main()


