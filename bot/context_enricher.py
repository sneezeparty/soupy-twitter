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
            # Use thread context if available, otherwise fall back to tweet text
            search_source = self._get_search_context_source(base_context, tweet_text)
            logger.info("Enrichment: deriving brief search context from {}", search_source["source"])
            logger.info("Enrichment: search excerpt => {}", search_source["text"].strip().replace("\n", " ")[:200])
            # Use LLM decision instead of brittle keyword heuristics
            txt_low = search_source["text"].lower()
            try:
                do_search = self._llm.should_enrich_with_web_search(tweet_text, (" ".join(details) if details else None))
            except Exception:
                do_search = True
            if do_search:
                search_result = self._search_context(search_source["text"], base_context)
            else:
                logger.info("Enrichment: LLM decided to skip web search for this post")
                search_result = None
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
                # Prefer reputable news domains to bias timeliness; fall back only if empty
                NEWS_PREF = {
                    "apnews.com","reuters.com","bbc.com","nytimes.com","washingtonpost.com","wsj.com","bloomberg.com",
                    "ft.com","theguardian.com","npr.org","axios.com","politico.com","aljazeera.com","cnn.com","cbsnews.com",
                    "abcnews.go.com","nbcnews.com","pbs.org","latimes.com","economist.com","thehill.com","vox.com",
                    "apnews.com","reuters.com","france24.com","dw.com","japantimes.co.jp","scmp.com","straitstimes.com",
                    "abc.net.au","cbc.ca","rte.ie","euronews.com","independent.co.uk","theatlantic.com","newyorker.com",
                    "time.com","newsweek.com","usnews.com","foreignpolicy.com","foreignaffairs.com","defenseone.com",
                    "scientificamerican.com","nature.com","science.org","newscientist.com","technologyreview.com",
                    "wired.com","arstechnica.com","techcrunch.com","theverge.com","engadget.com","zdnet.com",
                    "marketwatch.com","cnbc.com","forbes.com","fortune.com","businessinsider.com","hbr.org",
                    "nationalgeographic.com","smithsonianmag.com","sciencenews.org","sciencedaily.com","phys.org",
                    "theconversation.com","project-syndicate.org","worldpoliticsreview.com","devex.com","undark.org"
                }
                def _domain(u: str) -> str:
                    try:
                        return u.split('/')[2] if '://' in u else ''
                    except Exception:
                        return ''
                news_only = [(t, u, s) for (t, u, s) in used_results if any(d in _domain(u) for d in NEWS_PREF)]
                final = news_only if news_only else used_results
                used_urls = [u for (_, u, _) in final]
                snippets = [s for (_, _, s) in final if s]
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
        # Proper noun sequences (simple heuristic) - but avoid common character names
        common_character_names = {
            "humpty dumpty", "little red riding hood", "goldilocks", "jack and jill",
            "hansel and gretel", "snow white", "cinderella", "sleeping beauty",
            "peter pan", "alice in wonderland", "winnie the pooh", "mickey mouse",
            "donald duck", "goofy", "bugs bunny", "daffy duck", "tom and jerry"
        }
        
        for pn in re.findall(r"\b(?:[A-Z][a-z]+\s){1,3}[A-Z][a-z]+\b", raw):
            if pn not in terms and pn.lower() not in {t.lower() for t in terms}:
                # Skip if it's a common character name and we're in a thread about that character
                if base_context:
                    root_post = base_context.get("root_post", "")
                    # Handle both string and dict formats for root_post
                    if isinstance(root_post, dict):
                        root_post_text = root_post.get("text", "")
                    else:
                        root_post_text = str(root_post)
                    
                    if root_post_text.lower().find(pn.lower()) != -1:
                        if pn.lower() in common_character_names:
                            logger.info("Enrichment: skipping character name '{}' to avoid confusion", pn)
                            continue
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
            # Remove stopwords for a content-based overlap
            STOP = {
                "the","a","an","and","or","but","of","to","for","in","on","with","by","as","at","from","that","this","these","those","is","are","be","was","were","it","its","it's","you","your","we","our","they","their"
            }
            topic_content = [w for w in topic_tokens if w not in STOP]
            overlap = [w for w in topic_content if w in tweet_tokens]
            # Require at least two content-word overlaps or >=40% content overlap (min 2)
            enough_overlap = (len(overlap) >= 2) or (len(topic_content) > 0 and len(set(overlap)) >= max(2, int(0.4 * len(set(topic_content)))))
            # Disallow speculative-science topics unless present in tweet tokens
            SCI = {"multiverse","quantum","string","higgs","dark","cosmology","entropy"}
            sci_in_topic = any(w in SCI for w in topic_content)
            sci_in_tweet = any(w in tweet_tokens for w in SCI)
            if enough_overlap and (not sci_in_topic or sci_in_tweet):
                accepted_topic = topic
                logger.info("Enrichment: accepted topic '{}' (overlap: {})", topic, ", ".join(overlap))
            else:
                logger.info("Enrichment: discarded topic '{}' due to low overlap or mismatch", topic)

        # Prioritize thread context for base part if we're in a thread
        base_part = accepted_topic
        if not base_part and base_context:
            # If we're in a thread, try to extract key terms from root post
            root_post = base_context.get("root_post", "")
            # Handle both string and dict formats for root_post
            if isinstance(root_post, dict):
                root_post_text = root_post.get("text", "")
            else:
                root_post_text = str(root_post)
            
            if root_post_text and len(root_post_text.strip()) > 20:
                # Extract key terms from root post that might be more relevant
                root_clean = re.sub(r"[^\w\s]", " ", root_post_text.lower())
                root_terms = [w for w in root_clean.split() if len(w) > 3 and w not in {"that", "this", "with", "from", "they", "have", "been", "were", "said", "will", "would", "could", "should"}]
                if root_terms:
                    base_part = " ".join(root_terms[:8])
                    logger.info("Enrichment: using root post terms for base: {}", base_part)
        
        # Fall back to tweet text if no better context found
        if not base_part:
            base_part = " ".join(clean.split()[:12])
        chosen_terms = (terms[:6] + hashtag_terms)[:8]
        if chosen_terms:
            logger.info("Enrichment: chosen terms => {}", ", ".join(chosen_terms))
        # Add soft news/recency bias to query
        try:
            from datetime import datetime
            month_token = datetime.utcnow().strftime("%B %Y")
        except Exception:
            month_token = "news"
        query = " ".join([base_part, "news", month_token] + chosen_terms).strip()
        return (query[:160], chosen_terms)

    def _get_search_context_source(self, base_context: Optional[Dict], tweet_text: str) -> Dict[str, str]:
        """Determine the best source for search context - prioritize thread context over individual tweet."""
        # Check if we have thread context that indicates the main topic
        if base_context:
            # Look for root post or thread context that gives us the main topic
            root_post = base_context.get("root_post", "")
            thread_depth = base_context.get("thread_depth", 0)
            
            # Handle both string and dict formats for root_post
            if isinstance(root_post, dict):
                root_post_text = root_post.get("text", "")
            else:
                root_post_text = str(root_post)
            
            # If we're in a thread (depth > 1) and have a root post, use that as primary context
            if thread_depth > 1 and root_post_text and len(root_post_text.strip()) > 20:
                # Combine root post with current tweet for better context
                combined_context = f"{root_post_text} {tweet_text}".strip()
                return {"source": "thread context", "text": combined_context}
            
            # Check for ancestor context that might be more informative
            for i in range(1, 4):  # Check ancestor1, ancestor2, ancestor3
                ancestor_key = f"ancestor{i}"
                ancestor_text = base_context.get(ancestor_key, "")
                if ancestor_text and len(ancestor_text.strip()) > 20:
                    # Use ancestor if it seems more informative than current tweet
                    if len(ancestor_text) > len(tweet_text) * 1.5:
                        return {"source": f"ancestor{i} context", "text": ancestor_text}
        
        # Fall back to tweet text
        return {"source": "tweet text", "text": tweet_text}


