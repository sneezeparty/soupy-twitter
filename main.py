from __future__ import annotations

import random
import re
import time
from typing import Optional
import argparse

from loguru import logger

from bot.config import AppConfig
from bot.llm_client import LLMClient
from bot.scheduler import IntervalScheduler, RateLimiter, run_loop
from bot.bsky import BskyBot
from bot.context_enricher import ContextEnricher


def main() -> None:
    parser = argparse.ArgumentParser(description="Soupy Bluesky Bot")
    parser.add_argument("--now", action="store_true", help="Run one action immediately on start (respects hourly cap)")
    parser.add_argument("--reply", action="store_true", help="Force a reply action (overrides config/auto)")
    parser.add_argument("--post", action="store_true", help="Force a trending quote-retweet action at launch; if combined with --now, do post first then reply")
    parser.add_argument("--postnow", action="store_true", help="Force the daily news search/post at launch (Bluesky)")
    args = parser.parse_args()
    config = AppConfig.from_env()
    config.validate()

    logger.add("soupy.log", rotation="1 MB", retention=5)
    logger.info("Starting Soupy Bluesky Bot")

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
                # Guard: if seed has too little public-affairs content, switch to a better seed
                try:
                    if len(seed_text.split()) < 4:
                        alt = bot.select_popular_post_text(use_discover=True, limit=120)
                        if alt and (alt.get("text") or "").strip() and len((alt.get("text") or "").split()) >= 4:
                            seed_text = (alt.get("text") or "").strip()
                except Exception:
                    pass
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

                # Pull trending terms from Bluesky feed (heuristic) and via LLM extraction to bias toward timely topics
                try:
                    trending_terms = bot.get_trending_terms(limit=8) or []
                except Exception:
                    trending_terms = []
                try:
                    # Sample texts from discover/timeline and ask LLM to extract concise, timely topics
                    sample_texts = bot.get_discover_texts(limit=40) or []
                    llm_terms = llm.extract_trending_terms(sample_texts, max_terms=8) or []
                except Exception:
                    llm_terms = []
                merged_terms = []
                seen_tt = set()
                for term in (trending_terms + llm_terms):
                    low = (term or "").strip().lower()
                    if low and low not in seen_tt:
                        seen_tt.add(low)
                        merged_terms.append(term)
                if merged_terms:
                    logger.info("Daily post: trending terms => {}", ", ".join(merged_terms[:8]))

                # Use websearch to get articles; prefer first successful from reputable news
                import asyncio as _asyncio
                # Lazy import to avoid discord dependency at module import time
                from bot.websearch import SearchCog  # type: ignore
                cog = SearchCog(bot=None)  # type: ignore
                try:
                    async def _search_pick() -> tuple[str, str, list[str]]:
                        AVOID = {
                            "stackoverflow.com","github.com","stackexchange.com","medium.com","quora.com",
                            "forums.redflagdeals.com","dealnews.com","slickdeals.net","retailmenot.com",
                            "amazon.com","ebay.com","walmart.com","target.com","bestbuy.com",
                            "tripadvisor.com","yelp.com","zillow.com","realtor.com",
                            "indeed.com","linkedin.com","glassdoor.com","monster.com",
                            "webmd.com","healthline.com","mayoclinic.org","medlineplus.gov"
                        }
                        # Build LLM-driven queries in two passes: rewrite and clean
                        logger.info("Daily post: deriving LLM queries (rewrite + clean)")
                        topic = llm.infer_topic(seed_text) or "politics news"
                        try:
                            if not llm.is_public_affairs_topic(topic):
                                # If not public affairs, try infer from merged trending terms
                                tt_hint = ", ".join((merged_terms or [])[:4])
                                alt = llm.infer_topic(tt_hint) if tt_hint else None
                                if alt and llm.is_public_affairs_topic(alt):
                                    topic = alt
                        except Exception:
                            pass
                        initial_queries = llm.rewrite_news_queries(seed_text=seed_text, article_snippet=url_context)
                        # Mix in trending terms, then clean (and avoid outlet/brand terms via LLM where possible)
                        mixed_queries: list[str] = []
                        seen_mix = set()
                        for q in (initial_queries + (merged_terms or [])[:8]):
                            q2 = (q or "").strip()
                            if not q2:
                                continue
                            if q2.lower() in seen_mix:
                                continue
                            # Optional: drop short brand-like tokens (LLM check; best-effort)
                            try:
                                # fast heuristic: avoid very short tokens that look like outlet names
                                low = q2.lower()
                                looks_brand = (len(low.split()) <= 3 and any(tok.isalpha() and len(tok) <= 10 for tok in low.split()))
                                if looks_brand and llm.infer_topic(q2) and False:  # keep minimal calls; placeholder no-op
                                    pass
                            except Exception:
                                pass
                            seen_mix.add(q2.lower())
                            mixed_queries.append(q2)
                        queries = llm.clean_news_queries(mixed_queries, max_queries=8) or mixed_queries[:8]
                        # Ensure minimum viable queries
                        if len(queries) < 3:
                            logger.warning("Daily post: few queries after clean ({}); adding generic fallbacks", len(queries))
                            for fq in ["US congress bill", "labor strike negotiation", "state court ruling", "climate policy decision"] + (merged_terms[:3] if merged_terms else []):
                                if len(queries) >= 5:
                                    break
                                if fq not in queries:
                                    queries.append(fq)
                        logger.info("Daily post: topic='{}' | {} queries prepared", topic, len(queries))
                        
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

                        # Execute searches for the queries and select best article by LLM relevance + heuristics
                        # Strictly ignore any accidental instruction-like queries
                        def _looks_like_instruction(q: str) -> bool:
                            ql = q.lower().strip()
                            if any(ql.startswith(p) for p in ["here is", "topic:", "queries:", "format:"]):
                                return True
                            if any(w in ql for w in ["one per line", "max queries", "filtered list"]):
                                return True
                            return False

                        tried = 0
                        best_fallback: Optional[tuple[float, dict, int, list[str]]] = None  # (score, item, content_len, snippets)
                        THRESHOLD = 6.5
                        for q in queries:
                            if _looks_like_instruction(q):
                                continue
                            tried += 1
                            if tried > 12:
                                break
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
                                # No results; continue to next query
                                selected = []
                            if not selected:
                                continue
                            # Rank results, penalizing avoid-list domains only (no whitelist)
                            ranked = []
                            for it in selected:
                                href = it.get('href','')
                                domain = href.split('/')[2] if '://' in href else ''
                                is_avoid = any(d in domain for d in AVOID)
                                rank = (0) + (50 if is_avoid else 0)
                                ranked.append((rank, it))
                            ranked.sort(key=lambda x: x[0])
                            # Take top results after penalizing avoid domains (no whitelist filtering)
                            filtered = [it for (_, it) in ranked][:5]
                            # Fetch contents and pick best article with LLM relevance score (+ recency, - press release)
                            tasks = [cog.fetch_article_content(item.get('href', '')) for item in filtered]
                            contents = await _asyncio.gather(*tasks, return_exceptions=True)
                            
                            # Score articles by multiple factors, including LLM relevance
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
                                    
                                    # Factor 3: (removed) Domain reputation bonus — no whitelist bias
                                    
                                    # Factor 4: Avoid clickbait patterns
                                    title_lower = title.lower()
                                    clickbait_patterns = ['breaking', 'shocking', 'you won\'t believe', 'this will blow your mind']
                                    if not any(pattern in title_lower for pattern in clickbait_patterns):
                                        score += 1.0
                                    
                                    # Factor 5: Content quality indicators
                                    quality_words = ['analysis', 'report', 'study', 'research', 'data', 'policy', 'legislation']
                                    if any(word in text.lower() for word in quality_words):
                                        score += 2.0
                                    
                                    # Factor 6: LLM relevance to seed subject
                                    try:
                                        rel = llm.score_news_article_relevance(seed_subject=seed_text, article_title=title, article_excerpt=text[:800])
                                    except Exception:
                                        rel = 0.5
                                    score += rel * 5.0  # Softer relevance weight
                                    if rel < 0.55:
                                        score -= 4.0  # Softer penalty
                                        logger.warning("Daily post: low LLM relevance ({:.2f}) for '{}'", rel, title[:50])
                                    # Factor 6a: Recency boost via LLM heuristic
                                    try:
                                        rec = llm.score_article_recency(title=title, article_excerpt=text[:800])
                                    except Exception:
                                        rec = 0.5
                                    score += rec * 3.0
                                    # Factor 6b: Must be a specific article page (not section hub)
                                    try:
                                        is_article = llm.is_specific_article_page(url=url, title=title, article_excerpt=text[:800])
                                    except Exception:
                                        is_article = True
                                    if not is_article:
                                        score -= 12.0
                                        logger.info("Daily post: non-article page skipped '{}'", title[:60])
                                    # Factor 6b2: Penalize instructional/reference pages
                                    try:
                                        is_instr = llm.is_instructional_page(url=url, title=title, article_excerpt=text[:800])
                                    except Exception:
                                        is_instr = False
                                    if is_instr:
                                        score -= 10.0
                                        logger.info("Daily post: instructional/reference page penalized '{}'", title[:60])
                                    # Factor 6c: Penalize press releases via LLM classifier
                                    try:
                                        is_pr = llm.is_press_release(url=url, title=title, article_excerpt=text[:800])
                                    except Exception:
                                        is_pr = False
                                    if is_pr:
                                        score -= 6.0
                                    
                                    # Factor 7: Avoid retail/commercial content
                                    commercial_indicators = ['price', 'buy', 'sale', 'discount', 'deal', 'shop', 'store', 'retail', 'tire', 'inventory']
                                    if any(indicator in text.lower() for indicator in commercial_indicators):
                                        score -= 5.0
                                        logger.warning("Daily post: commercial content detected in '{}'", title[:50])

                                    # Factor 8: Trending/political alignment boost
                                    try:
                                        tt_low = [t.lower() for t in (trending_terms or [])]
                                        hay = f"{title} {text}".lower()
                                        if any(t in hay for t in tt_low):
                                            score += 3.0
                                    except Exception:
                                        pass
                                    
                                    article_scores.append((score, idx, item, content_len))
                            
                            if article_scores:
                                # Sort by score and pick the best
                                article_scores.sort(key=lambda x: x[0], reverse=True)
                                # Require a strong score threshold to avoid off-topic picks; keep best as fallback
                                best_score, best_idx, best_item, best_len = article_scores[0]
                                # Track best fallback across queries
                                try:
                                    if (best_fallback is None) or (best_score > best_fallback[0]):
                                        best_fallback = (best_score, best_item, best_len, snippets.copy())
                                except Exception:
                                    pass
                                if best_score >= THRESHOLD:
                                    url = best_item.get('href', '')
                                    topic = best_item.get('title', q) or q
                                    logger.info("Daily post: picked '{}' -> {} (score: {:.2f}, content {} chars)", 
                                              topic, url, best_score, best_len)
                                    # Multi-source enrichment: fetch 1–2 additional sources and condense
                                    try:
                                        enrich_query = (topic or q)[:140]
                                        extra_results = await cog.perform_text_search(enrich_query, max_results=8)
                                        selected_extra = await cog.select_articles(extra_results)
                                        # Keep up to 3 extras not duplicating the lead domain
                                        lead_domain = url.split('/')[2] if '://' in url else ''
                                        extras = []
                                        for it in selected_extra:
                                            href = it.get('href','')
                                            dom = href.split('/')[2] if '://' in href else ''
                                            if href and dom != lead_domain and href != url:
                                                extras.append(it)
                                            if len(extras) >= 3:
                                                break
                                        if extras:
                                            texts = await _asyncio.gather(*[cog.fetch_article_content(it.get('href','')) for it in extras], return_exceptions=True)
                                            extra_snips: list[str] = []
                                            for it, txt in zip(extras, texts):
                                                if isinstance(txt, str) and txt.strip():
                                                    _san = txt.strip().replace("\n", " ")[:400]
                                                    extra_snips.append(f"{it.get('title','')}: {_san}")
                                            if extra_snips:
                                                try:
                                                    condensed = llm.condense_search_snippets(query=enrich_query, snippets=extra_snips, max_chars=500, entities=None)
                                                    if condensed:
                                                        # Split condensed bullets into individual snippets
                                                        for line in (condensed.split('\n')):
                                                            ln = line.strip().lstrip('-').strip()
                                                            if ln:
                                                                snippets.append(ln)
                                                        logger.info("Daily post: enriched snippets with multi-source context ({} added)", len(condensed.split('\n')))
                                                except Exception:
                                                    # If condense fails, append raw extra snippets (trimmed)
                                                    snippets.extend(extra_snips[:2])
                                    except Exception:
                                        pass
                                    return (topic, url, snippets)
                        # If strict pass failed, try fallback to best available news article
                        if best_fallback is not None:
                            fb_score, fb_item, fb_len, fb_snips = best_fallback
                            url = fb_item.get('href', '')
                            topic = fb_item.get('title', 'news') or 'news'
                            logger.info("Daily post: fallback pick '{}' -> {} (score: {:.2f}, content {} chars)", topic, url, fb_score, fb_len)
                            # Multi-source enrichment on fallback as well
                            try:
                                enrich_query = (topic or q)[:140]
                                extra_results = await cog.perform_text_search(enrich_query, max_results=8)
                                selected_extra = await cog.select_articles(extra_results)
                                lead_domain = url.split('/')[2] if '://' in url else ''
                                extras = []
                                for it in selected_extra:
                                    href = it.get('href','')
                                    dom = href.split('/')[2] if '://' in href else ''
                                    if href and dom != lead_domain and href != url:
                                        extras.append(it)
                                    if len(extras) >= 3:
                                        break
                                if extras:
                                    texts = await cog.fetch_article_content(extras[0].get('href','')) if extras else None
                                    extra_snips: list[str] = []
                                    for it in extras[:2]:
                                        txt = await cog.fetch_article_content(it.get('href',''))
                                        if isinstance(txt, str) and txt.strip():
                                            _san = txt.strip().replace("\n", " ")[:400]
                                            extra_snips.append(f"{it.get('title','')}: {_san}")
                                    if extra_snips:
                                        try:
                                            condensed = llm.condense_search_snippets(query=enrich_query, snippets=extra_snips, max_chars=500, entities=None)
                                            if condensed:
                                                for line in (condensed.split('\n')):
                                                    ln = line.strip().lstrip('-').strip()
                                                    if ln:
                                                        fb_snips.append(ln)
                                                logger.info("Daily post: enriched fallback snippets with multi-source context")
                                        except Exception:
                                            fb_snips.extend(extra_snips[:2])
                            except Exception:
                                pass
                            return (topic, url, fb_snips)
                        return ("news", "", [])

                    def _check_topic_relevance(text: str, title: str, query: str) -> float:
                        """Check how relevant the content is to the search query."""
                        # Extract key terms from query
                        query_terms = [word.lower() for word in query.split() if len(word) > 3]
                        
                        # Combine text and title for analysis
                        combined_text = f"{title} {text}".lower()
                        
                        # Count how many query terms appear in the content
                        matches = sum(1 for term in query_terms if term in combined_text)
                        
                        # Calculate relevance score (0.0 to 1.0)
                        if not query_terms:
                            return 0.5  # Default if no query terms
                        
                        relevance = matches / len(query_terms)
                        return min(relevance, 1.0)

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
                logger.info("Daily post: drafting 4 candidates for topic '{}'", topic)
                base_snippets = snippets.copy()
                if url_context:
                    base_snippets.insert(0, f"Seed post context: {url_context[:400]}")
                candidates = llm.generate_bsky_post_candidates(topic=topic, url=url, snippets=base_snippets, num_candidates=4, max_chars=max_chars)
                best, scored = llm.select_best_bsky_post_with_scores(topic, base_snippets, candidates)
                try:
                    lines = ["Daily post: candidate scoreboard:"]
                    for rank, (score, text) in enumerate(scored[:5], start=1):
                        lines.append(f"{rank}. score={score} len={len(text)} :: {text[:160]}")
                    logger.info("{}", "\n".join(lines))
                except Exception:
                    pass
                if not best:
                    logger.info("Daily post: no viable candidate")
                    return
                # Consider generating a short thread (2-3 posts) when topic/snippets are rich
                try:
                    make_thread = True if len(base_snippets) >= 3 else False
                    thread_posts: list[str] = []
                    total = 0
                    if make_thread:
                        base = llm.generate_bsky_thread(topic=topic, snippets=base_snippets, segments=3, max_chars=max_chars)
                        total = len(base)
                        if total >= 2:
                            # Build labelled replies from base[1:]
                            replies: list[str] = []
                            for idx, p in enumerate(base[1:], start=2):
                                label = f"({idx}/{total}) "
                                room = max_chars - len(label)
                                if len(p) > room:
                                    p = p[: max(1, room - 1)] + "…"
                                replies.append(label + p)
                            thread_posts = replies
                        else:
                            thread_posts = []
                except Exception:
                    thread_posts = []
                # Behaviour-guided refine (tone/wording) before appending URL
                try:
                    final_text = llm.refine_bsky_post(topic=topic, snippets=base_snippets, draft_post=best, max_chars=max_chars)
                except Exception:
                    final_text = best
                # Ensure the selected URL is present as a TinyURL and is the only URL
                try:
                    if url:
                        try:
                            from bot.utils import shorten_url_tinyurl
                            short_url = shorten_url_tinyurl(url)
                        except Exception:
                            short_url = url
                        # Strip any existing URLs from the candidate to avoid conflicts or truncated links
                        try:
                            import re as _re
                            final_text = _re.sub(r"https?://\S+", "", final_text).strip()
                        except Exception:
                            pass
                        # Respect character limit when appending the URL
                        effective_max = max_chars
                        room_for_text = max(1, effective_max - (len(short_url) + 1))
                        if len(final_text) > room_for_text:
                            # Trim with ellipsis if needed
                            final_text = final_text[: max(1, room_for_text - 1)] + "…"
                        final_text = (final_text.rstrip() + " " + short_url).strip()
                        logger.info("Daily post: appended TinyURL and removed other URLs")
                except Exception:
                    pass
                if thread_posts:
                    # Label the head as (1/n) and then add labelled replies, preserving full URL
                    try:
                        # Determine total from first reply label e.g., (2/3)
                        import re as _re
                        m = _re.match(r"^\((\d+)/(\d+)\)\s+", thread_posts[0])
                        total = int(m.group(2)) if m else (1 + len(thread_posts))
                    except Exception:
                        total = 1 + len(thread_posts)
                    head_label = f"(1/{total}) "
                    # Preserve the full URL in the head; trim the non-URL portion to make room for label and URL
                    try:
                        import re as _re2
                        urls = _re2.findall(r"https?://\S+", final_text)
                        url_keep = urls[-1] if urls else None
                        if url_keep:
                            # Remove URLs from content
                            content_only = _re2.sub(r"https?://\S+", "", final_text).strip()
                            room_for_content = max_chars - len(head_label) - (len(url_keep) + 1)
                            if room_for_content < 1:
                                room_for_content = 1
                            if len(content_only) > room_for_content:
                                content_only = content_only[: max(1, room_for_content - 1)] + "…"
                            head_text = (content_only.rstrip() + " " + url_keep).strip()
                        else:
                            # No URL; just trim for label space
                            room_for_content = max_chars - len(head_label)
                            head_text = final_text if len(final_text) <= room_for_content else (final_text[: max(1, room_for_content - 1)] + "…")
                    except Exception:
                        # Fallback to simple trim if anything goes wrong
                        room_for_content = max_chars - len(head_label)
                        head_text = final_text if len(final_text) <= room_for_content else (final_text[: max(1, room_for_content - 1)] + "…")
                    labelled_head = head_label + head_text
                    logger.info("Daily post: final (thread head) => {}", (labelled_head[:220] + ("…" if len(labelled_head) > 220 else "")))
                    thread = [labelled_head] + thread_posts[:2]
                    ok = bot.create_thread(thread)
                    logger.info("Daily post: {}", "posted" if ok else "failed to post")
                else:
                    logger.info("Daily post: final => {}", (final_text[:220] + ("…" if len(final_text) > 220 else "")))
                    ok = bot.create_post(final_text)
                    logger.info("Daily post: {}", "posted" if ok else "failed to post")
            except Exception as exc:
                logger.warning("Daily post: error {}", exc)

        try:
            # Daily post schedule state (N/day with min/max spacing)
            daily_posts_last_24h: list[float] = []
            next_daily_post_at: Optional[float] = None

            def _prune_daily_posts(now_s: float) -> None:
                nonlocal daily_posts_last_24h
                cutoff = now_s - 24 * 3600
                daily_posts_last_24h = [t for t in daily_posts_last_24h if t >= cutoff]

            def _count_today_posts(now_local: time.struct_time) -> int:
                day_str = time.strftime("%Y-%m-%d", now_local)
                return sum(1 for ts in daily_posts_last_24h if time.strftime("%Y-%m-%d", time.localtime(ts)) == day_str)

            def schedule_next_daily_post(now_s: float) -> None:
                nonlocal next_daily_post_at
                min_h = max(1, int(getattr(config, "daily_post_min_interval_hours", 4)))
                max_h = max(min_h, int(getattr(config, "daily_post_max_interval_hours", 8)))
                delay = random.randint(min_h * 3600, max_h * 3600)
                next_daily_post_at = now_s + delay

            # Initialize schedule at startup
            schedule_next_daily_post(time.time())

            def bsky_action() -> None:
                now_local = time.localtime()
                hour = now_local.tm_hour
                start_h = config.operating_start_hour % 24
                end_h = config.operating_end_hour % 24
                within_hours = start_h <= hour <= end_h if start_h <= end_h else (hour >= start_h or hour <= end_h)
                if not within_hours:
                    logger.info("Outside operating hours ({}-{}); skipping action", start_h, end_h)
                    return
                # Multi-post daily schedule within operating hours
                try:
                    if config.daily_post_enabled:
                        now_s = time.time()
                        _prune_daily_posts(now_s)
                        per_day = max(1, int(getattr(config, "daily_posts_per_day", 2)))
                        if next_daily_post_at is not None and now_s >= next_daily_post_at:
                            if _count_today_posts(now_local) < per_day:
                                _do_daily_news_post(bot, llm)
                                daily_posts_last_24h.append(now_s)
                            else:
                                logger.info("Daily posts quota reached for today ({}).", per_day)
                            schedule_next_daily_post(now_s)
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
                    # Generate without explicit anchor to avoid meta leakage
                    cands = llm.generate_bsky_reply_candidates(post_text, context_str if context_str else None, num_candidates=4, anchor_phrase=None)
                    logger.info("===== STEP 6: Selection =====\n- {} candidate(s) generated", len(cands))
                    draft, scored = llm.select_best_bsky_reply_with_scores(post_text, context_str if context_str else None, cands, anchor_phrase=None)
                    # Human-readable scoreboard
                    try:
                        lines = ["===== STEP 7: Candidate scoreboard ====="]
                        for rank, (score, text) in enumerate(scored[:5], start=1):
                            lines.append(f"{rank}. score={score} len={len(text)} :: {text[:160]}")
                        logger.info("{}", "\n".join(lines))
                    except Exception:
                        pass
                    logger.info("===== STEP 7b: Draft selected =====\n({} chars) {}", len(draft), draft)
                    reply_text = llm.refine_bsky_reply(post_text, context_str if context_str else None, draft, anchor_phrase=None)
                    # Alignment validation gate: ensure the reply truly aligns; else fall back to next candidate or a brief ack
                    try:
                        align = llm.validate_reply_alignment(post_text, context_str if context_str else None, reply_text, anchor_phrase=anchor)
                    except Exception:
                        align = 0.6
                    # Hard reject if second-person slipped in
                    def _has_second_person(t: str) -> bool:
                        tl = (t or "").lower()
                        return (" you " in tl) or (" your " in tl) or ("you're" in tl) or (" you." in tl) or (" your." in tl)
                    if _has_second_person(reply_text):
                        logger.info("Reply alignment: rejected for using second-person pronouns; trying next candidate")
                        align = 0.0
                    if align < 0.6 and scored:
                        # Try the next best candidate once
                        try:
                            alt = scored[1][1]
                            alt_refined = llm.refine_bsky_reply(post_text, context_str if context_str else None, alt, anchor_phrase=None)
                            if _has_second_person(alt_refined):
                                raise Exception("alt uses second-person")
                            alt_align = llm.validate_reply_alignment(post_text, context_str if context_str else None, alt_refined, anchor_phrase=anchor)
                            if alt_align > align:
                                reply_text = alt_refined
                                align = alt_align
                                logger.info("Reply alignment: switched to next candidate (score {:.2f})", align)
                        except Exception:
                            pass
                    if align < 0.5:
                        # Fallback to extremely brief acknowledgment to avoid off-topic reply
                        try:
                            ack = llm.generate_brief_ack(post_text, context_str if context_str else None)
                            if ack:
                                reply_text = ack
                                logger.info("Reply alignment: fell back to brief ack due to low alignment ({:.2f})", align)
                        except Exception:
                            pass
                    # Enforce final-length safety net before posting
                    if len(reply_text) > config.bsky_reply_max_chars:
                        reply_text = reply_text[: config.bsky_reply_max_chars - 1] + "…"
                    # Final coherence refinement pass
                    try:
                        max_reply_len = max(200, int(getattr(config, "bsky_reply_max_chars", 300)))
                        reply_text = llm.coherence_refine_reply(post_text, context_str if context_str else None, reply_text, max_chars=max_reply_len)
                    except Exception:
                        pass
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
            # If requested, force the daily news search/post immediately at launch (regardless of hours)
            if args.postnow:
                try:
                    _do_daily_news_post(bot, llm)
                    # Record and reschedule
                    now_s = time.time()
                    daily_posts_last_24h.append(now_s)
                    schedule_next_daily_post(now_s)
                except Exception as exc:
                    logger.warning("--postnow failed: {}", exc)
            run_loop(bsky_action, rate_limiter, scheduler, run_immediately=(args.now or args.post))
        finally:
            bot.stop()
        return
    elif False:
        bot = None  # X API removed

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
        # Browser-based Twitter automation removed; Bluesky only
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


