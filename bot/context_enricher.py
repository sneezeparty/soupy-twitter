from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import asyncio

from loguru import logger

from .config import AppConfig
from .llm_client import LLMClient
from .utils import extract_urls, fetch_and_extract_readable_text
import requests
from bs4 import BeautifulSoup
import re


class ContextEnricher:
    """Fetches URLs and performs lightweight web-search-like enrichment to provide
    concise context bullets for reply generation.
    """

    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm

    def build_context(self, tweet_text: str, base_context: Optional[Dict]) -> str:
        details: List[str] = []
        if not base_context:
            base_context = {}

        # Include existing details (never include author/handle in LLM context)
        if base_context.get("link"):
            details.append(f"link: {base_context['link']}")
        if base_context.get("hashtags"):
            details.append("hashtags: " + ", ".join(base_context["hashtags"]))
        if base_context.get("mentions"):
            details.append("mentions: " + ", ".join(base_context["mentions"]))

        # Include a hint if the original post already contains a URL
        try:
            urls_in_text = extract_urls(tweet_text)
            if urls_in_text:
                details.append("original_has_url: yes")
        except Exception:
            pass

        # Include thread context if available (for Bluesky)
        if base_context.get("root_post"):
            root_post = base_context["root_post"]
            root_text = root_post.get("text", "")
            thread_depth = base_context.get("thread_depth", 1)

            if root_text and root_text != tweet_text:  # Only include if different from current post
                details.append(f"root_post: {root_text[:200]}")
                if thread_depth > 1:
                    details.append(f"thread_depth: {thread_depth} levels")

        # Add ancestors (nearest first)
        if base_context.get("ancestors"):
            try:
                ancestors = base_context["ancestors"][:3]
                for idx, anc in enumerate(ancestors, start=1):
                    a_text = (anc.get("text") or "")[:160]
                    if a_text:
                        details.append(f"ancestor{idx}: {a_text}")
            except Exception:
                pass

        # Add a sample of sibling replies to capture thread vibe
        if base_context.get("sibling_replies"):
            try:
                sibs = base_context["sibling_replies"][:4]
                quotes: List[str] = []
                for s in sibs:
                    s_text = (s.get("text") or "").strip()
                    if s_text:
                        quotes.append(s_text[:140])
                if quotes:
                    details.append("thread_vibe: " + " | ".join(quotes[:3]))
            except Exception:
                pass

        # Optionally include a hint of immediate child replies
        if base_context.get("child_replies"):
            try:
                crs = base_context["child_replies"][:2]
                for c in crs:
                    c_text = (c.get("text") or "").strip()
                    if c_text:
                        details.append(f"child: {c_text[:140]}")
            except Exception:
                pass

        # Include quoted post context if available (for Bluesky quotes)
        if base_context.get("quoted_post"):
            qp = base_context["quoted_post"]
            qp_text = qp.get("text") or ""
            if qp_text:
                details.append(f"quoted_post: {qp_text[:200]}")

        # URL enrichment
        url_summary_added = False
        if self._config.url_enrichment:
            urls = list(base_context.get("urls") or [])
            if not urls:
                urls = extract_urls(tweet_text)
            if urls:
                logger.info("Enrichment: found {} URL(s): {}", len(urls), ", ".join(urls[:3]))
            url_bullets = self._summarize_urls(urls)
            if url_bullets:
                details.append(f"url_summary: {url_bullets}")
                logger.info("Enrichment: URL summary ready ({} chars)", len(url_bullets))
                url_summary_added = True
            else:
                logger.info("Enrichment: no usable URL summary")

        # Tweet-first analysis (always include)
        try:
            tweet_bullets = self._llm.analyze_tweet(tweet_text, max_chars=min(400, self._config.enrichment_max_chars))
            if tweet_bullets:
                details.append(f"tweet: {tweet_bullets}")
                logger.info("Enrichment: tweet analysis ready ({} chars)", len(tweet_bullets))
        except Exception as exc:
            logger.warning("Tweet analysis failed: {}", exc)

        # Web search enrichment (secondary). If we already summarized a URL, skip adding generic web search to avoid drift.
        if self._config.web_search_enrichment and not url_summary_added:
            logger.info("Enrichment: deriving brief search context from tweet text")
            logger.info("Enrichment: tweet excerpt => {}", tweet_text.strip().replace("\n", " ")[:200])
            search_result = self._search_context(tweet_text, base_context)
            if search_result:
                search_bullets, used_urls, query_used = search_result
                logger.info("Enrichment: search query => {}", query_used)
                if used_urls:
                    logger.info("Enrichment: search URLs used ({}): {}", len(used_urls), ", ".join(used_urls))
                    # Surface sources directly so downstream generators can include a link
                    details.append("sources: " + ", ".join(used_urls[:2]))
                if search_bullets:
                    details.append(f"search: {search_bullets}")
                    logger.info("Enrichment: search context ready ({} chars)", len(search_bullets))
                    logger.info("Enrichment: search summary => {}", search_bullets)
                else:
                    logger.info("Enrichment: no search context produced")
            else:
                logger.info("Enrichment: no search context produced")

        context_str = " | ".join(details)
        return context_str[:1500]

    def _summarize_urls(self, urls: List[str]) -> Optional[str]:
        if not urls:
            return None
        bullets: List[str] = []
        max_chars = self._config.enrichment_max_chars
        fetch_limit = max(0, int(self._config.url_summary_fetch_limit)) or 2
        for u in urls[:fetch_limit]:  # limit fetches
            try:
                title, text = fetch_and_extract_readable_text(u, timeout=self._config.url_fetch_timeout)
                if not text:
                    continue
                summary = self._llm.summarize_web_page(title, text, max_chars=max_chars)
                if summary:
                    # Keep only first 2 bullets or first ~220 chars
                    bullets.append(summary[:220])
            except Exception as exc:
                logger.warning("URL enrichment failed for {}: {}", u, exc)
        if not bullets:
            return None
        return " ".join(bullets[:2])

    def _search_context(self, tweet_text: str, base_context: Optional[Dict]) -> Optional[Tuple[str, List[str], str]]:
        # Build a focused query from topic/hashtags/text
        query, entities = self._build_search_query(tweet_text, base_context)
        try:
            # First try rich websearch integration if available
            rich_snippets: Optional[Tuple[List[str], List[str]]] = None
            try:
                rich_snippets = self._websearch_rich_snippets(query, max_results=max(5, self._config.web_search_results))
            except Exception:
                rich_snippets = None
            if rich_snippets and rich_snippets[0]:
                snippets, used_urls = rich_snippets
            else:
                # Fallback to lightweight DDG HTML search
                results = self._perform_duckduckgo_search(query, self._config.web_search_results)
                used_results: List[Tuple[str, str, str]]
                if entities:
                    entity_set = {e.lower() for e in entities}
                    filtered: List[Tuple[str, str, str]] = []
                    for (title, url, snippet) in results:
                        hay = f"{title} {snippet}".lower()
                        if any(ent in hay for ent in entity_set):
                            filtered.append((title, url, snippet))
                    used_results = filtered if filtered else results
                else:
                    used_results = results
                used_urls = [u for (_, u, _) in used_results]
                snippets = [s for (_, _, s) in used_results if s]
            # Fallback: include tweet text snippet if search yielded nothing
            if not snippets and tweet_text:
                snippets.append(tweet_text[:240])
            if not snippets:
                return None
            max_chars = min(600, self._config.enrichment_max_chars)
            bullets = self._llm.condense_search_snippets(query, snippets, max_chars=max_chars, entities=entities or None)
            return (bullets, used_urls, query)
        except Exception as exc:
            logger.warning("Search enrichment failed: {}", exc)
            return None

    def _websearch_rich_snippets(self, query: str, max_results: int = 5) -> Optional[Tuple[List[str], List[str]]]:
        """Use the robust websearch integration (if available) to fetch article content and produce snippets.

        Returns (snippets, used_urls) or None if unavailable or failed.
        """
        try:
            from .websearch import SearchCog  # type: ignore
        except Exception:
            return None

        async def run() -> Tuple[List[str], List[str]]:
            cog = SearchCog(bot=None)  # type: ignore
            try:
                results = await cog.perform_text_search(query, max_results=max_results)
                selected = await cog.select_articles(results)
                # Fetch contents concurrently (limit to top N)
                tasks = [cog.fetch_article_content(item.get('href', '')) for item in selected[:max_results]]
                contents = await asyncio.gather(*tasks, return_exceptions=True)
                snippets: List[str] = []
                urls: List[str] = []
                for item, content in zip(selected[:max_results], contents):
                    url = item.get('href', '')
                    if isinstance(content, str) and content.strip():
                        # Build a compact snippet from title and first ~400 chars
                        title = (item.get('title') or '').strip()
                        text = content.strip().replace('\n', ' ')
                        snippet = (f"{title}: {text[:400]}") if title else text[:400]
                        snippets.append(snippet)
                        if url:
                            urls.append(url)
                return (snippets, urls)
            finally:
                try:
                    await cog.cog_unload()
                except Exception:
                    pass

        try:
            return asyncio.run(run())
        except RuntimeError:
            # If an event loop is already running, create a new one in a thread
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(run())
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        except Exception:
            return None

    def _perform_duckduckgo_search(self, query: str, limit: int) -> List[Tuple[str, str, str]]:
        """Return list of (title, url, snippet) using DuckDuckGo HTML endpoint."""
        headers = {
            "User-Agent": self._config.web_search_user_agent,
        }
        params = {"q": query, "kl": "us-en"}
        r = requests.get(self._config.web_search_engine_url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        items: List[Tuple[str, str, str]] = []
        # Typical structure: div.result with a.result__a and div.result__snippet
        for res in soup.select("div.result"):
            a = res.select_one("a.result__a") or res.find("a", href=True)
            if not a:
                continue
            href = a.get("href")
            if not href or not href.startswith("http"):
                continue
            title = a.get_text(" ", strip=True)
            snippet_el = res.select_one("div.result__snippet") or res.find("span", class_="result__snippet")
            snippet = (snippet_el.get_text(" ", strip=True) if snippet_el else "")
            items.append((title, href, snippet))
            if len(items) >= max(1, limit):
                break
        return items

    def _build_search_query(self, tweet_text: str, base_context: Optional[Dict]) -> Tuple[str, List[str]]:
        """Create a compact, relevant query anchored to the tweet.

        Rules:
        - Use inferred topic only if it overlaps with tweet tokens
        - Add concrete terms (currency amounts, orgs like CNN, key noun phrases)
        - Append up to two hashtags (without '#')
        """
        raw = tweet_text or ""
        # Clean: drop URLs/mentions
        clean = re.sub(r"https?://\S+", " ", raw)
        clean = re.sub(r"@[A-Za-z0-9_]{1,15}", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        # Candidate concrete terms
        terms: List[str] = []
        # Money amounts and counts
        for m in re.findall(r"\$?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b(?:\s*(?:million|billion|k))?", clean, flags=re.I):
            s = m.strip()
            if s and s not in terms:
                terms.append(s)
        # Common phrases seen in economy tweets
        for phrase in [
            "american dream",
            "median lifetime earnings",
            "bachelor's degree",
            "cost of living",
            "owning a home",
            "raising a family",
            "retiring with dignity",
        ]:
            if phrase in clean.lower() and phrase not in terms:
                terms.append(phrase)
        # Organizations / outlets and politics acronyms
        for org in [
            "CNN", "ABC", "CBS", "NBC", "FOX", "NYTimes", "WSJ", "Bloomberg",
            "DNC", "RNC", "MAGA", "GOP", "Democrats", "Republicans", "FactPost", "Fact Post"
        ]:
            if re.search(rf"\b{re.escape(org)}\b", clean, flags=re.I) and org not in terms:
                terms.append(org)
        # Proper noun sequences (simple heuristic)
        for pn in re.findall(r"\b(?:[A-Z][a-z]+\s){1,3}[A-Z][a-z]+\b", raw):
            if pn not in terms and pn.lower() not in {t.lower() for t in terms}:
                terms.append(pn)
        # Do not add author handle as a search term to avoid misattributing handles/domains as entities

        # Consider hashtags
        hashtags = (base_context or {}).get("hashtags") or []
        hashtag_terms = [h.lstrip('#') for h in hashtags][:2]

        # Topic guard: include only if it shares tokens with tweet
        topic = (base_context or {}).get("topic")
        accepted_topic: Optional[str] = None
        if topic:
            tweet_tokens = {w.lower() for w in re.findall(r"[a-zA-Z']+", clean)}
            topic_tokens = [w.lower() for w in re.findall(r"[a-zA-Z']+", topic)]
            overlap = [w for w in topic_tokens if w in tweet_tokens]
            if len(overlap) >= max(1, int(0.4 * len(set(topic_tokens)))):
                accepted_topic = topic
                logger.info("Enrichment: accepted topic '{}' (overlap: {})", topic, ", ".join(overlap))
            else:
                logger.info("Enrichment: discarded topic '{}' due to low overlap", topic)

        base_part = accepted_topic or " ".join(clean.split()[:12])
        chosen_terms = (terms[:6] + hashtag_terms)[:8]
        if chosen_terms:
            logger.info("Enrichment: chosen terms => {}", ", ".join(chosen_terms))
        query = " ".join([base_part] + chosen_terms).strip()
        return (query[:160], chosen_terms)


