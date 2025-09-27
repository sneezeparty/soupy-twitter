from __future__ import annotations

import random
import re
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
    parser.add_argument("--postnow", action="store_true", help="Force the daily news search/post at launch (Bluesky)")
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

        # Helper: daily news post via websearch + trending/discover
        def _do_daily_news_post(bot: BskyBot, llm: LLMClient) -> None:
            try:
                logger.info("Daily post: starting selection via discover/trending + websearch")
                # Pivot: choose a popular post from feed as the starting point
                popular = bot.select_popular_post_text(use_discover=True, limit=80)
                if not popular:
                    popular = bot.select_popular_post_text(use_discover=False, limit=80)
                if not popular:
                    logger.info("Daily post: no popular post found; aborting")
                    return
                seed_text = (popular.get("text") or "").strip()
                logger.info("Daily post: seed post => {}", seed_text[:180])

                # Extract URLs from seed text for additional context
                from bot.utils import extract_urls
                seed_urls = extract_urls(seed_text)
                url_context = ""
                if seed_urls:
                    logger.info("Daily post: found {} URL(s) in seed post: {}", len(seed_urls), ", ".join(seed_urls[:3]))
                    # Fetch content from URLs to enrich context
                    try:
                        url_contents = []
                        for url in seed_urls[:2]:  # Limit to first 2 URLs to avoid timeout
                            try:
                                from bot.utils import fetch_and_extract_readable_text
                                title, text = fetch_and_extract_readable_text(url, timeout=6)
                                if title and text:
                                    # Create a brief summary of the URL content
                                    summary = f"{title}: {text[:300]}"
                                    url_contents.append(summary)
                                    logger.debug("Daily post: fetched URL context from {}", url)
                            except Exception as exc:
                                logger.warning("Daily post: failed to fetch URL {}: {}", url, exc)
                                continue
                        if url_contents:
                            url_context = " | ".join(url_contents)
                            logger.info("Daily post: URL context ready ({} chars)", len(url_context))
                    except Exception as exc:
                        logger.warning("Daily post: URL context extraction failed: {}", exc)

                # Use websearch to get articles; prefer first successful from reputable news
                import asyncio as _asyncio
                # Lazy import to avoid discord dependency at module import time
                from bot.websearch import SearchCog  # type: ignore
                cog = SearchCog(bot=None)  # type: ignore
                try:
                    async def _search_pick() -> tuple[str, str, list[str]]:
                        NEWS_PREF = {
                            "apnews.com","reuters.com","bbc.com","nytimes.com","washingtonpost.com","wsj.com","bloomberg.com",
                            "ft.com","theguardian.com","npr.org","axios.com","politico.com","aljazeera.com","cnn.com","cbsnews.com",
                            "abcnews.go.com","nbcnews.com","pbs.org","latimes.com","economist.com"
                        }
                        AVOID = {"stackoverflow.com","github.com","stackexchange.com","reddit.com","medium.com","quora.com"}
                        # Use LLM to analyze the seed post and determine the best topic and search queries
                        logger.info("Daily post: analyzing seed post with LLM to determine topic and search queries")
                        
                        # Prepare context for LLM analysis
                        analysis_context = f"Seed post: {seed_text}"
                        if url_context:
                            analysis_context += f"\n\nURL context from seed post: {url_context}"
                        
                        # Use LLM to determine the main topic and generate search queries
                        topic_analysis = llm.analyze_topic_and_generate_queries(analysis_context)
                        
                        # Parse the LLM response to extract topic and queries
                        lines = topic_analysis.strip().split('\n')
                        topic = "politics news"  # Default fallback
                        
                        # Extract topic from first line
                        for line in lines:
                            if line.strip().startswith('TOPIC:'):
                                topic = line.strip().replace('TOPIC:', '').strip()
                                break
                        
                        # Extract search queries from the analysis
                        queries = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('TOPIC:') and not line.startswith('#'):
                                # Clean up the query
                                query = line.lstrip('- ').strip()
                                if query and len(query) > 3:
                                    queries.append(query)
                        
                        # Fallback if LLM didn't generate enough queries
                        if len(queries) < 3:
                            logger.warning("Daily post: LLM generated only {} queries, adding fallbacks", len(queries))
                            fallback_queries = ["politics news", "current events", "breaking news"]
                            for fq in fallback_queries:
                                if fq not in queries:
                                    queries.append(fq)
                        
                        logger.info("Daily post: LLM determined topic: '{}'", topic)
                        logger.info("Daily post: LLM generated {} search queries", len(queries))
                        
                        # Prepend ddgs top headlines as extra queries if the current queries are weak
                        if len(queries) < 3:
                            try:
                                news_titles = await cog.fetch_top_news_queries(max_items=6)
                                for t in news_titles:
                                    if t and t not in queries:
                                        queries.append(t)
                                logger.info("Daily post: augmented queries with top news ({} added)", len(news_titles))
                            except Exception:
                                pass

                        # Validate and refine queries: require news hits
                        for q in queries[:12]:
                            # Enhance query with URL context if available
                            enhanced_query = q
                            if url_context:
                                # Combine the original query with URL context for better results
                                enhanced_query = f"{q} {url_context[:200]}"
                                logger.debug("Daily post: enhanced query with URL context: {}", enhanced_query[:100])
                            
                            results = await cog.perform_text_search(enhanced_query, max_results=8)
                            logger.info("Daily post: {} result(s) for query '{}'", len(results), q)
                            selected = await cog.select_articles(results)
                            logger.info("Daily post: {} selected article(s) for query '{}'", len(selected), q)
                            if not selected:
                                continue
                            # Prefer reputable news domains first
                            ranked = []
                            for it in selected:
                                href = it.get('href','')
                                domain = href.split('/')[2] if '://' in href else ''
                                is_news = any(d in domain for d in NEWS_PREF)
                                is_avoid = any(d in domain for d in AVOID)
                                rank = (0 if is_news else 10) + (50 if is_avoid else 0)
                                ranked.append((rank, it))
                            ranked.sort(key=lambda x: x[0])
                            filtered = [it for (_, it) in ranked][:5]
                            # Fetch contents and pick best article (not just longest)
                            tasks = [cog.fetch_article_content(item.get('href', '')) for item in filtered]
                            contents = await _asyncio.gather(*tasks, return_exceptions=True)
                            
                            # Score articles by multiple factors, not just length
                            article_scores = []
                            snippets: list[str] = []
                            
                            for idx, (item, content) in enumerate(zip(filtered, contents)):
                                if isinstance(content, str) and content.strip():
                                    text = content.strip().replace('\n', ' ')
                                    title = item.get('title') or ''
                                    url = item.get('href', '')
                                    
                                    # Create snippet
                                    snippet = f"{title}: {text[:400]}"
                                    snippets.append(snippet)
                                    
                                    # Score article by multiple factors
                                    score = 0
                                    
                                    # Factor 1: Content length (but not the only factor)
                                    content_len = len(text)
                                    score += min(content_len / 1000.0, 5.0)  # Cap at 5 points
                                    
                                    # Factor 2: Title quality (not too short, not too long)
                                    title_len = len(title)
                                    if 20 <= title_len <= 100:
                                        score += 2.0
                                    elif 10 <= title_len <= 150:
                                        score += 1.0
                                    
                                    # Factor 3: Domain reputation bonus
                                    domain = url.split('/')[2] if '://' in url else ''
                                    if any(d in domain for d in NEWS_PREF):
                                        score += 3.0
                                    
                                    # Factor 4: Avoid clickbait patterns
                                    title_lower = title.lower()
                                    clickbait_patterns = ['breaking', 'shocking', 'you won\'t believe', 'this will blow your mind']
                                    if not any(pattern in title_lower for pattern in clickbait_patterns):
                                        score += 1.0
                                    
                                    # Factor 5: Content quality indicators
                                    quality_words = ['analysis', 'report', 'study', 'research', 'data', 'policy', 'legislation']
                                    if any(word in text.lower() for word in quality_words):
                                        score += 2.0
                                    
                                    article_scores.append((score, idx, item, content_len))
                            
                            if article_scores:
                                # Sort by score and pick the best
                                article_scores.sort(key=lambda x: x[0], reverse=True)
                                best_score, best_idx, best_item, best_len = article_scores[0]
                                
                                url = best_item.get('href', '')
                                topic = best_item.get('title', q) or q
                                logger.info("Daily post: picked '{}' -> {} (score: {:.2f}, content {} chars)", 
                                          topic, url, best_score, best_len)
                                return (topic, url, snippets)
                        return ("news", "", [])

                    topic, url, snippets = _asyncio.run(_search_pick())
                finally:
                    try:
                        _asyncio.run(cog.cog_unload())
                    except Exception:
                        pass

                if not url:
                    logger.info("Daily post: no suitable URL found")
                    return

                max_chars = min(300, max(1, int(config.bsky_post_max_chars)))
                logger.info("Daily post: drafting with max_chars={} for topic '{}'", max_chars, topic)
                
                # Include URL context in snippets if available
                enhanced_snippets = snippets.copy()
                if url_context:
                    enhanced_snippets.insert(0, f"Seed post context: {url_context[:400]}")
                    logger.debug("Daily post: enhanced snippets with URL context")
                
                post_text = llm.draft_brief_link_opinion_post(topic=topic, url=url, snippets=enhanced_snippets, max_chars=max_chars)
                if not post_text:
                    logger.info("Daily post: LLM returned empty text")
                    return
                logger.info("Daily post: preview => {}", (post_text[:220] + ("…" if len(post_text) > 220 else "")))
                ok = bot.create_post(post_text)
                logger.info("Daily post: {}", "posted" if ok else "failed to post")
            except Exception as exc:
                logger.warning("Daily post: error {}", exc)

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
                # Daily Bluesky news post around configured hour
                try:
                    if config.daily_post_enabled:
                        target_hour = int(getattr(config, "daily_post_hour", 14)) % 24
                        window_min = int(getattr(config, "daily_post_window_minutes", 45))
                        # post once per day; track last post day in closure
                        nonlocal last_daily_post_day
                        today_day = time.strftime("%Y-%m-%d", now_local)
                        minutes_from_target = abs((hour * 60 + now_local.tm_min) - (target_hour * 60))
                        within_window = minutes_from_target <= max(1, window_min)
                        if within_window and last_daily_post_day != today_day:
                            _do_daily_news_post(bot, llm)
                            last_daily_post_day = today_day
                except Exception as exc:
                    logger.warning("Daily post attempt failed: {}", exc)

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
                    # Helper: pretty-print a compact thread view for humans
                    def _format_thread_view(ctx: dict, post_text: str) -> str:
                        def _line(label: str, author: str, text: str, max_len: int = 220) -> str:
                            a = author or ""
                            t = (text or "").replace("\n", " ")[:max_len]
                            return f"- {label}: {a} {t}".rstrip()
                        parts: list[str] = []
                        root = (ctx or {}).get("root_post") or {}
                        if root:
                            parts.append(_line("Root", root.get("author") or "", root.get("text") or ""))
                        # Ancestors (nearest first)
                        ancestors = (ctx or {}).get("ancestors") or []
                        for idx, anc in enumerate(ancestors[:3], start=1):
                            parts.append(_line(f"Ancestor{idx}", anc.get("author") or "", anc.get("text") or ""))
                        # Current
                        parts.append(_line("Current", (ctx or {}).get("author") or "", post_text))
                        # Siblings (vibe)
                        sibs = (ctx or {}).get("sibling_replies") or []
                        for idx, s in enumerate(sibs[:3], start=1):
                            parts.append(_line(f"Sibling{idx}", s.get("author") or "", s.get("text") or ""))
                        # Children (immediate replies)
                        childs = (ctx or {}).get("child_replies") or []
                        for idx, c in enumerate(childs[:2], start=1):
                            parts.append(_line(f"Child{idx}", c.get("author") or "", c.get("text") or ""))
                        return "\n".join(parts)

                    # 50/50 choose discover vs following
                    use_discover = (random.random() < 0.5)
                    post_text = bot.prepare_reply_from_discover() if use_discover else bot.prepare_reply_from_timeline()
                    logger.info("===== STEP 1: Source feed =====\n- chosen: {}", "discover" if use_discover else "following")
                    if not post_text:
                        # Fallback: try the other feed once before giving up
                        alt_use_discover = not use_discover
                        logger.info("Bsky: no target found, falling back to {} feed", "discover" if alt_use_discover else "following")
                        post_text = bot.prepare_reply_from_discover() if alt_use_discover else bot.prepare_reply_from_timeline()
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
                            logger.info("===== STEP 2: Thread gathered =====\n- enhanced with root/ancestors/siblings/children")
                            try:
                                logger.info("===== STEP 3: Thread view =====\n{}", _format_thread_view(ctx, post_text))
                            except Exception:
                                pass
                        else:
                            logger.info("Bsky: could not get thread context, using current post only")
                    
                    inferred_topic = llm.infer_topic(post_text)
                    if inferred_topic:
                        ctx = {**ctx, "topic": inferred_topic}
                    context_str = enricher.build_context(post_text, ctx)
                    logger.info("===== STEP 4: Context to LLM =====\nPost: {}\nContext: {}", post_text, context_str or "-")
                    # Generate multiple candidates and select the best
                    logger.info("===== STEP 5: Generate candidates =====")
                    cands = llm.generate_bsky_reply_candidates(post_text, context_str if context_str else None, num_candidates=4)
                    logger.info("===== STEP 6: Selection =====\n- {} candidate(s) generated", len(cands))
                    draft, scored = llm.select_best_bsky_reply_with_scores(post_text, context_str if context_str else None, cands)
                    # Human-readable scoreboard
                    try:
                        lines = ["===== STEP 7: Candidate scoreboard ====="]
                        for rank, (score, text) in enumerate(scored[:5], start=1):
                            lines.append(f"{rank}. score={score} len={len(text)} :: {text[:160]}")
                        logger.info("{}", "\n".join(lines))
                    except Exception:
                        pass
                    logger.info("===== STEP 7b: Draft selected =====\n({} chars) {}", len(draft), draft)
                    reply_text = llm.refine_bsky_reply(post_text, context_str if context_str else None, draft)
                    # Enforce final-length safety net before posting
                    if len(reply_text) > config.bsky_reply_max_chars:
                        reply_text = reply_text[: config.bsky_reply_max_chars - 1] + "…"
                    logger.info("===== STEP 8: Final reply =====\n({} chars) {}", len(reply_text), reply_text)
                    if bot.send_prepared_reply(reply_text):
                        logger.info("Bsky mode: reply succeeded")
                        # Opportunistically like a few nearby replies with human-like delays
                        try:
                            bot.like_some_thread_replies(ctx, max_likes=5)
                        except Exception:
                            pass
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
            # Track last daily news post day
            last_daily_post_day: Optional[str] = None
            # If requested, force the daily news search/post immediately at launch (regardless of hours)
            if args.postnow:
                try:
                    _do_daily_news_post(bot, llm)
                    last_daily_post_day = time.strftime("%Y-%m-%d", time.localtime())
                except Exception as exc:
                    logger.warning("--postnow failed: {}", exc)
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


