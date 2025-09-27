"""
Search functionality for Soupy Bot
Provides DuckDuckGo search capabilities with rate limiting and result processing
"""
from loguru import logger
from collections import defaultdict
import time
import os
import asyncio
from typing import Optional, List, Dict
# ddgs dependency removed; using HTML scraping fallback
from openai import OpenAI
import httpx
import aiohttp
import trafilatura
from bs4 import BeautifulSoup
import requests
import json
import re

# Using loguru logger (consistent with rest of project)

# Initialize OpenAI client
_http_client = httpx.Client(http2=False)
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    http_client=_http_client,
)

async def async_chat_completion(*args, **kwargs):
    """Wraps the OpenAI chat completion in an async context"""
    return await asyncio.to_thread(client.chat.completions.create, *args, **kwargs)

class SearchCog:
    def __init__(self, bot=None):
        self.bot = bot
        self.search_rate_limits = defaultdict(list)
        self.MAX_SEARCHES_PER_MINUTE = 10
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> None:
        if self.session is None or getattr(self.session, "closed", False):
            self.session = aiohttp.ClientSession()

    async def cog_unload(self):
        """Cleanup when cog is unloaded"""
        if getattr(self, 'session', None):
            try:
                await self.session.close()  # type: ignore
            except Exception:
                pass

    async def fetch_top_news_queries(self, region: str = "us-en", max_items: int = 10) -> List[str]:
        """Return a list of top news headlines (queries) using ddgs.news as a fallback source.

        Uses a thread to avoid blocking the event loop.
        """
        try:
            def _news() -> List[str]:
                from ddgs import DDGS  # type: ignore
                titles: List[str] = []
                with DDGS() as ddg:
                    try:
                        for it in ddg.news(keywords="", region=region, safesearch="off", timelimit="d"):
                            t = (it.get("title") or "").strip()
                            if t:
                                titles.append(t)
                            if len(titles) >= max_items:
                                break
                    except Exception:
                        return titles
                return titles
            return await asyncio.wait_for(asyncio.to_thread(_news), timeout=10)
        except Exception:
            return []

    async def is_rate_limited(self, user_id: int) -> bool:
        """Check if user has exceeded rate limits"""
        current_time = time.time()
        search_times = self.search_rate_limits.get(user_id, [])
        
        # Clean up old timestamps
        search_times = [t for t in search_times if current_time - t < 60]
        self.search_rate_limits[user_id] = search_times
        
        if len(search_times) >= self.MAX_SEARCHES_PER_MINUTE:
            return True
        
        self.search_rate_limits[user_id].append(current_time)
        return False

    async def fetch_article_content(self, url: str) -> Optional[str]:
        """Fetch and extract the main content of an article with improved error handling"""
        try:
            await self._ensure_session()
            assert self.session is not None
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Try trafilatura first
                    content = trafilatura.extract(html)
                    if content:
                        return content.strip()
                    
                    # Fallback to BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                        element.decompose()
                    
                    # Get main content
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|article|post'))
                    if main_content:
                        return main_content.get_text(strip=True, separator=' ')
                    
                    # Last resort: get body text
                    body = soup.find('body')
                    if body:
                        return body.get_text(strip=True, separator=' ')
                    
                    return None
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {e}")
            return None

    async def perform_text_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """DuckDuckGo-only search. Prefer ddgs library; fallback to DDG HTML/lite.

        Returns list of dicts: {title, body, href, source}.
        """
        try:
            start_ts = time.time()
            # 1) Prefer ddgs library (handles tokens/backends)
            try:
                def _ddgs_search() -> List[Dict]:
                    from ddgs import DDGS  # type: ignore
                    results: List[Dict] = []
                    with DDGS() as ddg:
                        attempts = [
                            {"backend": "api"},
                            {"backend": "html"},
                            {"backend": "lite"},
                            {},
                        ]
                        for opts in attempts:
                            try:
                                arr = list(ddg.text(query=query, max_results=max_results, **opts))
                            except Exception:
                                continue
                            if not arr:
                                continue
                            results = []
                            for it in arr:
                                href = (it.get("href") or it.get("url") or "").strip()
                                if not href.startswith("http"):
                                    continue
                                title = (it.get("title") or href).strip()
                                body = (it.get("body") or it.get("snippet") or "").strip()
                                domain = re.sub(r"^www\\.", "", (re.sub(r"https?://", "", href).split("/")[0] if href else "")).strip()
                                results.append({"title": title, "body": body, "href": href, "source": domain})
                            if results:
                                return results
                    return results
                ddg_results = await asyncio.wait_for(asyncio.to_thread(_ddgs_search), timeout=18)
                if ddg_results:
                    logger.info("ddgs returned {} results in {}s", len(ddg_results), round(time.time() - start_ts, 2))
                    return ddg_results
            except Exception as e:
                logger.warning("ddgs failed/unavailable: {}", e)

            # 2) Fallback to DDG HTML endpoints
            headers = {
                "User-Agent": os.getenv(
                    "WEB_SEARCH_USER_AGENT",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",
            }
            params = {"q": query, "kl": "us-en"}
            base_urls = [
                os.getenv("WEB_SEARCH_ENGINE_URL") or "https://duckduckgo.com/html/",
                "https://html.duckduckgo.com/html/",
                "https://lite.duckduckgo.com/lite/",
            ]
            results: List[Dict] = []
            for base_url in base_urls:
                try:
                    r = requests.get(base_url, params=params, headers=headers, timeout=12)
                    if r.status_code != 200 or not r.text:
                        logger.warning("DDG HTML {} status {}", base_url, r.status_code)
                        continue
                    soup = BeautifulSoup(r.text, "lxml")
                    # html layout
                    for res in soup.select("div.result"):
                        a = res.select_one("a.result__a") or res.find("a", href=True)
                        if not a:
                            continue
                        href = a.get("href")
                        if not href or not href.startswith("http"):
                            continue
                        title = a.get_text(" ", strip=True) or href
                        snippet_el = res.select_one("div.result__snippet") or res.find("span", class_="result__snippet")
                        snippet = (snippet_el.get_text(" ", strip=True) if snippet_el else "")
                        domain = re.sub(r"^www\\.", "", (re.sub(r"https?://", "", href).split("/")[0] if href else "")).strip()
                        results.append({"title": title, "body": snippet, "href": href, "source": domain})
                        if len(results) >= max_results:
                            break
                    # lite layout fallback
                    if not results:
                        for a in soup.select("a"):
                            href = a.get("href")
                            if not href or not href.startswith("http"):
                                continue
                            title = a.get_text(" ", strip=True)
                            if not title:
                                continue
                            domain = re.sub(r"^www\\.", "", (re.sub(r"https?://", "", href).split("/")[0] if href else "")).strip()
                            results.append({"title": title or href, "body": "", "href": href, "source": domain})
                            if len(results) >= max_results:
                                break
                    if results:
                        logger.info("DDG HTML {} returned {} results in {}s", base_url, len(results), round(time.time() - start_ts, 2))
                        return results
                except Exception as e:
                    logger.warning("DDG HTML {} error {}", base_url, e)
            return []
        except Exception as e:
            logger.error("Search error: {}", e)
            return []

    # Provider-specific helpers removed; DDG-only implementation above.

    async def select_articles(self, search_results: List[Dict]) -> List[Dict]:
        """Select the 5 most relevant articles using improved selection criteria"""
        try:
            if len(search_results) <= 5:
                return search_results

            # Format articles for evaluation
            # Keep mapping from formatted index -> original index
            formatted_results: List[Dict] = []
            index_mapping: List[int] = []
            for idx, result in enumerate(search_results):
                # Skip results without required fields
                if not all(key in result for key in ['title', 'body', 'href']):
                    continue
                    
                formatted_results.append({
                    'title': result['title'],
                    'preview': result.get('body', '')[:500],  # Limit preview length
                    'url': result['href']
                })
                index_mapping.append(idx)

            if not formatted_results:
                return search_results[:5]

            # Create selection prompt
            prompt = (
                "Select the 5 most informative and relevant articles from these search results. "
                "Consider:\n"
                "1. Relevance to the topic\n"
                "2. Information quality and depth\n"
                "3. Source credibility\n"
                "4. Content uniqueness\n\n"
                "Respond ONLY with the numbers (0-based) of the 5 best articles, separated by spaces.\n\n"
            )
            
            for i, result in enumerate(formatted_results):
                prompt += f"[{i}] {result['title']}\n"
                prompt += f"URL: {result['url']}\n"
                prompt += f"Preview: {result['preview']}\n\n"

            response = await async_chat_completion(
                model=os.getenv("LOCAL_CHAT"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that selects the most relevant and informative articles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            # Parse indices and validate
            try:
                indices = [int(idx) for idx in response.choices[0].message.content.strip().split()]
                # Map model-selected indices (based on formatted_results) back to original indices
                valid_formatted = [i for i in indices if 0 <= i < len(formatted_results)]
                mapped_indices = [index_mapping[i] for i in valid_formatted]
                if len(mapped_indices) >= 5:
                    return [search_results[i] for i in mapped_indices[:5]]
            except Exception:
                logger.warning("Failed to parse article selection response")
                
            return search_results[:5]

        except Exception as e:
            logger.error(f"Error selecting articles: {e}")
            return search_results[:5]

    async def generate_final_response(self, query: str, articles: List[Dict]) -> str:
        """Generate a conversational response with proper citations"""
        try:
            if not articles:
                return "❌ No articles found to analyze."

            # Process articles and extract content
            processed_articles = []
            total_tokens = 0
            MAX_TOKENS_PER_ARTICLE = 1000  # Limit tokens per article

            for article in articles:
                content = await self.fetch_article_content(article.get('href', ''))
                if not content:
                    continue

                # Truncate content to manage token count
                content = content[:MAX_TOKENS_PER_ARTICLE]
                processed_articles.append({
                    'title': article.get('title', 'Untitled'),
                    'source': article.get('source', 'Unknown Source'),
                    'url': article['href'],
                    'content': content
                })

            if not processed_articles:
                return "❌ Could not extract content from any articles."

            # Create system message with Soupy's personality and strong citation requirements
            system_message = (
                f"{os.getenv('BEHAVIOUR_SEARCH', '')}\n\n"
                "CRITICAL INSTRUCTIONS FOR RESPONSE GENERATION:\n"
                "1. You MUST include citations for every piece of information you provide\n"
                "2. Format ALL citations as [Source Name](URL) using Discord markdown\n"
                "3. Citations MUST be naturally integrated into your response\n"
                "4. EVERY paragraph or major point MUST have at least one citation\n"
                "5. Be conversational and engaging while maintaining accuracy\n"
                "6. Keep responses concise but informative\n"
                "7. Use Soupy's sarcastic and witty personality\n"
                "8. Organize the response in a clear, readable format\n"
                "9. DO NOT generate a response without citations\n"
                "10. If you reference multiple sources for a point, cite them all\n"
            )

            # Format content for LLM with emphasis on citation requirements
            content_prompt = (
                f"Search Query: {query}\n\n"
                "IMPORTANT: Your response MUST include citations from the following articles. "
                "Every significant piece of information MUST be backed by at least one citation. "
                "Here are the articles to analyze and cite:\n\n"
            )

            for i, article in enumerate(processed_articles, 1):
                content_prompt += (
                    f"Article {i}:\n"
                    f"Title: {article['title']}\n"
                    f"Source: {article['source']}\n"
                    f"URL: {article['url']}\n"
                    f"Content: {article['content']}\n\n"
                )

            content_prompt += (
                "REMINDER: Format your response as a natural conversation, but ensure EVERY "
                "significant point has a citation in [Source Name](URL) format. DO NOT skip citations.\n\n"
            )

            # Generate response with chunked content if needed
            try:
                response = await async_chat_completion(
                    model=os.getenv("LOCAL_CHAT"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": content_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                return response.choices[0].message.content.strip()

            except Exception as e:
                if "context length" in str(e).lower():
                    # Fallback to shorter content if context length is exceeded
                    logger.warning("Context length exceeded, falling back to shorter content")
                    shortened_articles = []
                    for article in processed_articles:
                        shortened_articles.append({
                            'title': article['title'],
                            'source': article['source'],
                            'url': article['url'],
                            'content': article['content'][:300]  # Use shorter excerpts
                        })

                    content_prompt = (
                        f"Search Query: {query}\n\n"
                        "IMPORTANT: Your response MUST include citations from these articles. "
                        "Every significant piece of information MUST be backed by at least one citation. "
                        "Here are brief excerpts from the articles to analyze and cite:\n\n"
                    )

                    for i, article in enumerate(shortened_articles, 1):
                        content_prompt += (
                            f"Article {i}:\n"
                            f"Title: {article['title']}\n"
                            f"Source: {article['source']}\n"
                            f"URL: {article['url']}\n"
                            f"Excerpt: {article['content']}\n\n"
                        )

                    content_prompt += (
                        "REMINDER: Format your response as a natural conversation, but ensure EVERY "
                        "significant point has a citation in [Source Name](URL) format. DO NOT skip citations.\n\n"
                    )

                    response = await async_chat_completion(
                        model=os.getenv("LOCAL_CHAT"),
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": content_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    
                    return response.choices[0].message.content.strip()
                else:
                    raise

        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "❌ An error occurred while generating the response."

    # Note: Discord slash-command integration removed. This class is now a standalone search utility.
