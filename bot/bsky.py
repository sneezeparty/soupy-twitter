from __future__ import annotations

from typing import Optional, Dict, Any, List
import re
import math
from datetime import datetime, timezone
import random
import time
import os
import json

from loguru import logger
from atproto import Client as BskyClient, models as bsky_models

from .config import AppConfig


class BskyBot:
    """Minimal Bluesky bot with read timeline, search, post, reply, quote, and repost.

    This mirrors the X API facade shape to ease main loop integration.
    """

    def __init__(self, config: AppConfig, llm_client) -> None:
        self._config = config
        self._client = BskyClient(config.bsky_service_url)
        self._llm = llm_client
        self._current_post_context: Optional[Dict[str, Any]] = None
        self._reply_target: Optional[Dict[str, str]] = None  # {uri, cid}
        # Recent dedupe store (URIs) persisted under user_data_dir
        self._replied_uris: set[str] = set()
        self._replied_order: List[str] = []  # maintain recency for pruning
        self._replied_log_path = os.path.join(self._config.user_data_dir, "bsky_replied.json")
        # In-memory cooldown for authors to diversify targets within a run
        # Recent authors cooldown persisted to disk with timestamps
        self._recent_authors_path = os.path.join(self._config.user_data_dir, "bsky_recent_authors.json")
        self._recent_authors: Dict[str, float] = {}
        # Enforce at least 24h no-repeat per author
        configured_cooldown_s = float(self._config.bsky_author_cooldown_minutes) * 60.0
        self._author_cooldown_s = max(24 * 3600.0, configured_cooldown_s)
        self._candidate_pool_size = max(4, int(self._config.bsky_candidate_pool_size))

    def start(self) -> None:
        try:
            os.makedirs(self._config.user_data_dir, exist_ok=True)
        except Exception:
            pass
        self._load_replied_log()
        self._load_recent_authors()
        self._client.login(self._config.bsky_handle or "", self._config.bsky_app_password or "")
        logger.info("BskyBot logged in as {}", self._config.bsky_handle)

    def stop(self) -> None:
        logger.info("BskyBot stopped")

    # ---------------- Persistence for dedupe -----------------
    def _load_replied_log(self) -> None:
        try:
            if os.path.exists(self._replied_log_path):
                with open(self._replied_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # Keep only strings
                    self._replied_order = [u for u in data if isinstance(u, str)]
                    self._replied_uris = set(self._replied_order)
        except Exception:
            self._replied_uris = set()
            self._replied_order = []

    def _save_replied_log(self) -> None:
        try:
            with open(self._replied_log_path, "w", encoding="utf-8") as f:
                json.dump(self._replied_order, f)
        except Exception:
            pass

    def get_trending_terms(self, limit: int = 5) -> List[str]:
        """Extract top trending terms from Popular/Explore feed with political biasing.

        Combines hashtag frequency and simple proper-noun phrases, filters stopwords,
        and boosts terms that co-occur with political vocabulary.
        """
        try:
            params_pop = bsky_models.AppBskyFeedGetFeed.Params(
                feed="at://did:plc:public/app.bsky.feed.generator/popular", limit=60
            )
            try:
                # Try a popular generator
                pop = self._client.app.bsky.feed.get_feed(params_pop)
                pitems: List[Any] = getattr(pop, "feed", []) or []
            except Exception:
                pitems = []
            # Fallback: timeline as explore proxy
            if not pitems:
                params_tl = bsky_models.AppBskyFeedGetTimeline.Params(limit=60)
                tl = self._client.app.bsky.feed.get_timeline(params_tl)
                pitems = getattr(tl, "feed", []) or []
            texts: List[str] = []
            for it in pitems[:60]:
                rec = getattr(getattr(it, 'post', None), 'record', None)
                t = getattr(rec, 'text', '') if rec else ''
                if t:
                    texts.append(t)
            if not texts:
                return []

            STOP = {
                "from","to","subject","time","help","all","yours","friday","monday","tuesday","wednesday","thursday","saturday","sunday",
                "today","tomorrow","yesterday","breaking","update","thread","read","link","please","thanks","amp","http","https"
            }
            POL = {
                "election","vote","voting","ballot","congress","senate","house","supreme court","scotus","union","strike","labor",
                "worker","workers","minimum wage","medicare","medicaid","social security","healthcare","abortion","immigration","border",
                "tax","billionaires","wealth","inequality","climate","gaza","ukraine","palestine","israel","police","protest","student debt"
            }

            freq: Dict[str, int] = {}
            pol_boost: Dict[str, int] = {}

            for t in texts:
                tl = t.lower()
                # hashtags
                for m in re.findall(r"#[A-Za-z][A-Za-z0-9_]{2,}", t):
                    k = m.lstrip('#').lower()
                    if k in STOP or len(k) < 3:
                        continue
                    freq[k] = freq.get(k, 0) + 1
                    if any(p in tl for p in POL):
                        pol_boost[k] = pol_boost.get(k, 0) + 1
                # Proper noun phrases up to 4 words
                for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", t):
                    k = m.strip()
                    if not k or len(k) < 3:
                        continue
                    if k.lower() in STOP:
                        continue
                    freq[k] = freq.get(k, 0) + 1
                    if any(p in tl for p in POL):
                        pol_boost[k] = pol_boost.get(k, 0) + 1

            if not freq:
                return []
            # Score = freq + 2*political boost; minor length penalty to prefer concise topics
            scored = sorted(
                freq.items(), key=lambda kv: (kv[1] + 2*pol_boost.get(kv[0], 0) - 0.01*len(kv[0])), reverse=True
            )
            # Deduplicate case-insensitively, keep original form as key string
            seen_lower: set[str] = set()
            top: List[str] = []
            for term, _ in scored:
                low = term.lower()
                if low in seen_lower:
                    continue
                seen_lower.add(low)
                top.append(term)
                if len(top) >= max(1, limit):
                    break
            return top
        except Exception:
            return []

    def get_explore_trending_topics(self, limit: int = 10) -> List[str]:
        """Attempt to fetch Explore/Trending topic labels via unspecced endpoints.

        Falls back to heuristic terms if the endpoint is unavailable.
        """
        try:
            topics: List[str] = []
            # Try common unspecced method names defensively
            try:
                uns = getattr(self._client.app.bsky, "unspecced", None)
                if uns:
                    # e.g., getTrendingTopics or get_trending_topics
                    for meth in ("get_trending_topics", "getTrendingTopics", "getTrendingTopicsSkeleton"):
                        fn = getattr(uns, meth, None)
                        if fn:
                            try:
                                params_cls = getattr(bsky_models, "AppBskyUnspeccedGetTrendingTopics".replace(".", ""), None)
                            except Exception:
                                params_cls = None
                            try:
                                # Call without params if signature unknown
                                resp = fn() if params_cls is None else fn(params_cls.Params(limit=max(1, limit)))
                            except Exception:
                                resp = fn()
                            # Try to read fields generically
                            data = []
                            try:
                                data = getattr(resp, "topics", []) or getattr(resp, "items", []) or getattr(resp, "feed", [])
                            except Exception:
                                data = []
                            for it in data:
                                name = (
                                    getattr(it, "label", None)
                                    or getattr(it, "title", None)
                                    or getattr(it, "displayName", None)
                                    or getattr(it, "name", None)
                                )
                                if isinstance(name, str) and name.strip():
                                    topics.append(name.strip())
                                if len(topics) >= limit:
                                    break
                            if topics:
                                break
            except Exception:
                topics = []
            if topics:
                # De-duplicate while preserving order and drop trivial tokens
                seen: set[str] = set()
                cleaned: List[str] = []
                STOP = {"the", "and", "or", "a", "an", "of", "for", "to", "in", "on", "more"}
                for t in topics:
                    low = t.lower().strip()
                    if low in seen or low in STOP or len(low) < 3:
                        continue
                    seen.add(low)
                    cleaned.append(t)
                return cleaned[: max(1, limit)]
            # Fallback to heuristic
            return self.get_trending_terms(limit=limit)
        except Exception:
            return []

    def select_popular_post_text(self, use_discover: bool = True, limit: int = 60) -> Optional[Dict[str, Any]]:
        """Pick a high-engagement, recent post from Discover/Popular or Following timeline.

        Returns a dict: {text, uri, cid, author, created_ts, like_count, repost_count, reply_count}
        """
        try:
            items: List[Any] = []
            if use_discover:
                try:
                    params = bsky_models.AppBskyFeedGetFeed.Params(
                        feed="at://did:plc:public/app.bsky.feed.generator/popular", limit=max(10, limit)
                    )
                    feed = self._client.app.bsky.feed.get_feed(params)
                    items = getattr(feed, "feed", []) or []
                except Exception:
                    items = []
            if not items:
                params = bsky_models.AppBskyFeedGetTimeline.Params(limit=max(10, limit))
                feed = self._client.app.bsky.feed.get_timeline(params)
                items = getattr(feed, "feed", []) or []
            if not items:
                return None

            def _metric(post_like: Any) -> tuple[float, Dict[str, Any]]:
                post = getattr(post_like, "post", post_like)
                record = getattr(post, "record", None)
                text = getattr(record, "text", "") if record else ""
                uri = getattr(post, "uri", None)
                cid = getattr(post, "cid", None)
                author_obj = getattr(post, "author", None)
                author = getattr(author_obj, "handle", None)
                # Engagement metrics (defensive attribute access)
                lc = getattr(post, "like_count", None) or getattr(post, "likeCount", None) or 0
                rc = getattr(post, "repost_count", None) or getattr(post, "repostCount", None) or 0
                rpc = getattr(post, "reply_count", None) or getattr(post, "replyCount", None) or 0
                # Recency
                created_raw = getattr(record, "created_at", None) or getattr(record, "createdAt", None)
                created_ts = None
                try:
                    if isinstance(created_raw, str) and created_raw:
                        iso = created_raw.replace("Z", "+00:00")
                        created_ts = datetime.fromisoformat(iso).astimezone(timezone.utc).timestamp()
                except Exception:
                    created_ts = None
                now_ts = time.time()
                age_h = 24.0
                if isinstance(created_ts, (int, float)) and created_ts > 0:
                    age_h = max(0.0, (now_ts - created_ts) / 3600.0)
                # Score: engagement with recency decay; prefer posts with text
                engagement = float(lc) * 3.0 + float(rc) * 4.0 + float(rpc) * 2.0
                recency = 1.0 if age_h <= 0.1 else math.exp(-age_h / 12.0)
                text_bonus = 0.2 if text and len(text) >= max(1, int(getattr(self._config, "bsky_min_text_len", 10))) else 0.0
                score = (engagement + 1.0) * recency + text_bonus
                data = {
                    "text": text,
                    "uri": uri,
                    "cid": cid,
                    "author": author,
                    "created_ts": created_ts,
                    "like_count": int(lc or 0),
                    "repost_count": int(rc or 0),
                    "reply_count": int(rpc or 0),
                }
                return (score, data)

            scored: List[tuple[float, Dict[str, Any]]] = []
            for it in items:
                try:
                    scored.append(_metric(it))
                except Exception:
                    continue
            if not scored:
                return None
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # More diverse selection strategy to avoid "samey" posts
            my_handle_plain = (self._config.bsky_handle or "").lstrip("@").lower()
            
            # Strategy 1: Diversify by author (avoid same authors)
            author_groups = {}
            for score, post_data in scored:
                author = (post_data.get("author") or "").lstrip("@").lower()
                if author != my_handle_plain and (post_data.get("text") or "").strip():
                    if author not in author_groups:
                        author_groups[author] = []
                    author_groups[author].append((score, post_data))
            
            # Strategy 2: Diversify by content type (look for different topics)
            content_diversity = []
            seen_topics = set()
            
            # Sort authors by their best post score
            sorted_authors = sorted(author_groups.items(), key=lambda x: max(s[0] for s in x[1]), reverse=True)
            
            for author, posts in sorted_authors[:10]:  # Consider top 10 authors
                for score, post_data in posts[:2]:  # Max 2 posts per author
                    text = post_data.get("text", "").lower()
                    
                    # Extract topic keywords for diversity
                    topic_keywords = []
                    for word in text.split():
                        if len(word) > 4 and word.isalpha():
                            topic_keywords.append(word)
                    
                    # Check if this adds diversity
                    topic_signature = " ".join(sorted(topic_keywords[:5]))
                    if topic_signature not in seen_topics or len(content_diversity) < 5:
                        seen_topics.add(topic_signature)
                        content_diversity.append((score, post_data))
                        
                        if len(content_diversity) >= 12:  # Limit pool size
                            break
                if len(content_diversity) >= 12:
                    break
            
            # Strategy 3: Weighted random selection favoring diversity
            if content_diversity:
                import random
                # Weight by score but add diversity bonus
                weights = []
                for score, post_data in content_diversity:
                    # Base weight from engagement score
                    base_weight = max(0.1, score)
                    # Add diversity bonus (lower scores get slight boost)
                    diversity_bonus = 1.0 + (1.0 / (score + 1.0))
                    weights.append(base_weight * diversity_bonus)
                
                # Weighted random selection
                total_weight = sum(weights)
                if total_weight > 0:
                    r = random.random() * total_weight
                    acc = 0.0
                    for i, weight in enumerate(weights):
                        acc += weight
                        if r <= acc:
                            best = content_diversity[i][1]
                            logger.debug("Bsky: selected diverse post from {} candidates (score: {:.2f})", 
                                       len(content_diversity), content_diversity[i][0])
                            return best
            
            # Fallback: if diversity strategy fails, use original method
            top_count = min(8, max(3, len(scored) // 4))
            top_posts = scored[:top_count]
            valid_candidates = []
            for _, post_data in top_posts:
                author = (post_data.get("author") or "").lstrip("@").lower()
                if author != my_handle_plain and (post_data.get("text") or "").strip():
                    valid_candidates.append(post_data)
            
            if valid_candidates:
                import random
                best = random.choice(valid_candidates)
                logger.debug("Bsky: fallback selection from {} top candidates", len(valid_candidates))
                return best
            
            return None
        except Exception:
            return None

    def get_discover_texts(self, limit: int = 30) -> List[str]:
        """Return up to `limit` post texts from Popular/Discover (fallback to timeline)."""
        try:
            try:
                params = bsky_models.AppBskyFeedGetFeed.Params(
                    feed="at://did:plc:public/app.bsky.feed.generator/popular", limit=max(10, limit)
                )
                feed = self._client.app.bsky.feed.get_feed(params)
                items = getattr(feed, "feed", []) or []
            except Exception:
                params = bsky_models.AppBskyFeedGetTimeline.Params(limit=max(10, limit))
                feed = self._client.app.bsky.feed.get_timeline(params)
                items = getattr(feed, "feed", []) or []
            texts: List[str] = []
            for it in items:
                post = getattr(it, "post", None)
                record = getattr(post, "record", None)
                text = getattr(record, "text", "") if record else ""
                if text:
                    texts.append(text)
                if len(texts) >= max(1, limit):
                    break
            return texts
        except Exception:
            return []

    def _load_recent_authors(self) -> None:
        try:
            if os.path.exists(self._recent_authors_path):
                with open(self._recent_authors_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._recent_authors = {k: float(v) for k, v in data.items() if isinstance(k, str)}
        except Exception:
            self._recent_authors = {}

    def _save_recent_authors(self) -> None:
        try:
            with open(self._recent_authors_path, "w", encoding="utf-8") as f:
                json.dump(self._recent_authors, f)
        except Exception:
            pass

    def _record_replied(self, uri: Optional[str]) -> None:
        if not uri:
            return
        if uri in self._replied_uris:
            # Move to end to refresh recency
            try:
                self._replied_order.remove(uri)
            except Exception:
                pass
        else:
            self._replied_uris.add(uri)
        self._replied_order.append(uri)
        # Cap size to avoid unbounded growth
        cap = max(100, int(getattr(self._config, "bsky_replied_log_max", 1000)))
        if len(self._replied_order) > cap:
            # Drop oldest until within cap
            drop = len(self._replied_order) - cap
            for _ in range(drop):
                old = self._replied_order.pop(0)
                if old in self._replied_uris and old not in self._replied_order:
                    self._replied_uris.remove(old)
        self._save_replied_log()

    def _create_url_facets(self, text: str) -> Optional[List]:
        """Create facets for URLs in the text to make them clickable.
        
        Returns a list of facet objects or None if no URLs found.
        """
        try:
            import re
            # Find all URLs in the text
            url_pattern = re.compile(r'https?://[^\s]+')
            urls = url_pattern.findall(text)
            
            if not urls:
                return None
            
            facets = []
            for url in urls:
                # Find the start and end positions of the URL in the text
                start = text.find(url)
                if start != -1:
                    end = start + len(url)
                    # Create a facet for this URL
                    facet = bsky_models.AppBskyRichtextFacet.Main(
                        index=bsky_models.AppBskyRichtextFacet.ByteSlice(
                            byteStart=start,
                            byteEnd=end
                        ),
                        features=[
                            bsky_models.AppBskyRichtextFacet.Link(
                                uri=url
                            )
                        ]
                    )
                    facets.append(facet)
            
            logger.debug("Bsky: created {} URL facets", len(facets))
            return facets if facets else None
            
        except Exception as exc:
            logger.warning("Bsky: failed to create URL facets: {}", exc)
            return None

    def _create_external_embed(self, text: str) -> Optional[Any]:
        """Create an external embed for URLs in the text to generate rich link previews.
        
        Returns an AppBskyEmbedExternal object or None if no URLs found.
        """
        try:
            import re
            # Find the first URL in the text
            url_pattern = re.compile(r'https?://[^\s]+')
            urls = url_pattern.findall(text)
            
            if not urls:
                return None
            
            # Use the first URL found
            url = urls[0]
            logger.debug("Bsky: creating external embed for URL: {}", url)
            
            # Fetch URL metadata with improved error handling
            try:
                title, description, image_url = self._fetch_url_metadata(url)
            except Exception as exc:
                logger.warning("Bsky: metadata fetch failed for {}, using fallback: {}", url, exc)
                # Use fallback metadata
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    domain = parsed.netloc or url
                    title = f"Link from {domain}"
                    description = ""
                    image_url = None
                except Exception:
                    title = "Link"
                    description = ""
                    image_url = None
            
            # Create the external embed object
            external_data = {
                "uri": url,
                "title": title or "Link",
                "description": description or ""
            }
            
            # Add image thumbnail if available
            if image_url:
                try:
                    thumb_blob = self._upload_image_blob(image_url)
                    if thumb_blob:
                        external_data["thumb"] = thumb_blob
                        logger.info("Bsky: added image thumbnail to external embed for {}", url)
                    else:
                        logger.warning("Bsky: failed to upload image blob for {}", image_url)
                except Exception as exc:
                    logger.warning("Bsky: failed to add image thumbnail: {}", exc)
            
            # Create external embed
            external_embed = bsky_models.AppBskyEmbedExternal.Main(
                external=bsky_models.AppBskyEmbedExternal.External(**external_data)
            )
            
            logger.info("Bsky: created external embed with title: '{}' (has_thumb: {})", 
                       title, "thumb" in external_data)
            return external_embed
            
        except Exception as exc:
            logger.warning("Bsky: failed to create external embed: {}", exc)
            return None

    def _fetch_url_metadata(self, url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Fetch Open Graph metadata from a URL for embed metadata.
        
        Returns (title, description, image_url)
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin, urlparse
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
            
            # Fetch the HTML
            resp = requests.get(url, headers=headers, timeout=8, allow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Parse Open Graph meta tags
            title = None
            description = None
            image_url = None
            
            # Try Open Graph tags first
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                title = og_title["content"].strip()
            
            og_description = soup.find("meta", property="og:description")
            if og_description and og_description.get("content"):
                description = og_description["content"].strip()
            
            og_image = soup.find("meta", property="og:image")
            if og_image and og_image.get("content"):
                image_url = og_image["content"].strip()
                # Convert relative URLs to absolute
                if not image_url.startswith(('http://', 'https://')):
                    image_url = urljoin(url, image_url)
            
            # Fallback to standard meta tags if Open Graph not available
            if not title:
                title_tag = soup.find("title")
                if title_tag and title_tag.string:
                    title = title_tag.string.strip()
            
            if not description:
                desc_tag = soup.find("meta", attrs={"name": "description"})
                if desc_tag and desc_tag.get("content"):
                    description = desc_tag["content"].strip()
            
            # Clean up the title
            if title and len(title) > 100:
                title = title[:97] + "..."
            
            # Clean up the description
            if description and len(description) > 200:
                description = description[:197] + "..."
            
            logger.debug("Bsky: fetched metadata for {}: title='{}', description='{}', image='{}'", 
                        url, title, description, image_url)
            
            return title, description, image_url
            
        except Exception as exc:
            # Log the error but don't let it crash the posting process
            logger.warning("Bsky: failed to fetch URL metadata for {}: {}", url, exc)
            # Return a fallback title based on the URL
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain = parsed.netloc or url
                fallback_title = f"Link from {domain}"
                return fallback_title, None, None
            except Exception:
                return "Link", None, None

    def _upload_image_blob(self, image_url: str) -> Optional[Any]:
        """Upload an image from URL as a blob for embedding.
        
        Returns a blob object or None if upload fails.
        """
        try:
            import requests
            
            # Fetch the image
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
            
            resp = requests.get(image_url, headers=headers, timeout=10)
            resp.raise_for_status()
            
            # Check content type
            content_type = resp.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning("Bsky: URL does not appear to be an image: {}", image_url)
                return None
            
            # Check file size (limit to 1MB as per BlueSky docs)
            if len(resp.content) > 1000000:
                logger.warning("Bsky: image too large ({} bytes), skipping", len(resp.content))
                return None
            
            # Upload as blob using the correct API call from BlueSky docs
            blob_resp = self._client.com.atproto.repo.upload_blob(
                data=resp.content
            )
            
            logger.debug("Bsky: uploaded image blob for {} (size: {} bytes)", image_url, len(resp.content))
            return blob_resp.blob
            
        except Exception as exc:
            logger.warning("Bsky: failed to upload image blob for {}: {}", image_url, exc)
            return None

    # Posting
    def create_post(self, text: str) -> bool:
        try:
            # Enforce Bluesky max graphemes (hard limit is 300)
            max_len = min(300, max(1, int(getattr(self._config, "bsky_post_max_chars", 300))))
            safe_text = text if len(text) <= max_len else (text[: max_len - 1] + "…")
            
            # Check if the post contains a URL and create external embed
            embed = self._create_external_embed(safe_text)
            
            # Create URL facets to make URLs clickable in the text
            facets = self._create_url_facets(safe_text)
            
            if embed:
                # Use external embed for rich link preview
                if facets:
                    self._client.send_post(text=safe_text, embed=embed, facets=facets)
                    logger.info("Bsky: posted with external embed and URL facets ({} chars)", len(text))
                else:
                    self._client.send_post(text=safe_text, embed=embed)
                    logger.info("Bsky: posted with external embed ({} chars)", len(text))
            else:
                # Regular post without URLs
                if facets:
                    self._client.send_post(text=safe_text, facets=facets)
                    logger.info("Bsky: posted with URL facets ({} chars)", len(text))
                else:
                    self._client.send_post(text=safe_text)
                    logger.info("Bsky: posted ({} chars)", len(text))
            return True
        except Exception as exc:
            logger.error("Bsky: failed to post: {}", exc)
            return False

    # Read timeline and prepare a reply target
    def prepare_reply_from_timeline(self) -> Optional[str]:
        """Fetch Following timeline, pick a candidate, and store reply target (uri,cid).

        Returns the selected post text, or None if unavailable.
        """
        try:
            params = bsky_models.AppBskyFeedGetTimeline.Params(limit=50)
            feed = self._client.app.bsky.feed.get_timeline(params)
            items: List[Any] = getattr(feed, "feed", []) or []
            if not items:
                logger.warning("Bsky: timeline empty")
                return None
            # Build candidate list, then choose randomly to diversify
            candidates: List[Dict[str, Any]] = []
            relaxed_candidates: List[Dict[str, Any]] = []
            now_s = time.time()
            # prune expired author cooldowns
            self._recent_authors = {a: t for a, t in self._recent_authors.items() if (now_s - t) < self._author_cooldown_s}
            my_handle_plain = (self._config.bsky_handle or "").lstrip("@")
            for it in items:
                post = getattr(it, "post", None)
                record = getattr(post, "record", None)
                text = getattr(record, "text", "") if record else ""
                if text and len(text.strip()) >= max(1, int(getattr(self._config, "bsky_min_text_len", 10))):
                    uri = getattr(post, "uri", None)
                    cid = getattr(post, "cid", None)
                    author_obj = getattr(post, "author", None)
                    author = getattr(author_obj, "handle", None)
                    author_did = getattr(author_obj, "did", None)
                    viewer = getattr(author_obj, "viewer", None)
                    is_following = bool(getattr(viewer, "following", None)) if viewer else False
                    # Skip posts authored by ourselves (never reply to own posts)
                    if author and my_handle_plain and author.lower() == my_handle_plain.lower():
                        logger.info("Bsky: skipping own post (author matched) uri={}", uri)
                        continue
                    # Skip if we've already replied to this post
                    already_replied_this = bool(uri and uri in self._replied_uris)
                    if not uri or not cid:
                        continue
                    # Skip sports-related posts
                    if self._is_sports_text(text):
                        logger.info("Bsky: skipping sports post uri={}", uri)
                        continue
                    # Skip recently replied-to authors (cooldown)
                    recent_author = bool(author and author in self._recent_authors)
                    # If this is a reply, try to find the root and skip if we've replied to that root recently
                    try:
                        thread_ctx = self.get_thread_context(uri)
                        root_uri = None
                        if thread_ctx and thread_ctx.get("root_post", {}).get("uri"):
                            root_uri = thread_ctx["root_post"]["uri"]
                            root_already_replied = bool(root_uri in self._replied_uris)
                            # Also skip threads where the root author is ourselves (never reply to own thread)
                            root_author = (thread_ctx.get("root_post", {}) or {}).get("author") or ""
                            if my_handle_plain and root_author.lstrip("@").lower() == my_handle_plain.lower():
                                logger.info("Bsky: skipping thread (root authored by self) root_uri={}", root_uri)
                                continue
                    except Exception:
                        thread_ctx = None
                        root_uri = None
                    # Parse createdAt to epoch seconds for recency weighting
                    created_at_raw = getattr(record, "created_at", None) or getattr(record, "createdAt", None)
                    created_ts: Optional[float] = None
                    try:
                        if isinstance(created_at_raw, str) and created_at_raw:
                            iso = created_at_raw.replace("Z", "+00:00")
                            created_ts = datetime.fromisoformat(iso).astimezone(timezone.utc).timestamp()
                    except Exception:
                        created_ts = None

                    ctx: Dict[str, Any] = {
                        "text": text,
                        "author": f"@{author}" if author else None,
                        "uri": uri,
                        "cid": cid,
                        "author_did": author_did,
                        "is_following": is_following,
                    }
                    # Try to extract quoted post (if this is a quote)
                    try:
                        quoted = self._extract_quoted_from_post_view(post)
                        if quoted:
                            ctx["quoted_post"] = quoted
                    except Exception:
                        pass
                    # Try to pull an external link if present in embed
                    try:
                        external_url = self._extract_external_url_from_post_view(post)
                        if external_url:
                            ctx["urls"] = [external_url]
                            ctx["original_has_url"] = True
                        else:
                            # Also mark if text itself has a URL-like substring
                            if "http://" in text or "https://" in text:
                                ctx["original_has_url"] = True
                    except Exception:
                        # Basic heuristic from text
                        if "http://" in text or "https://" in text:
                            ctx["original_has_url"] = True
                    # Include thread, quoted, and external link context if present
                    if thread_ctx and thread_ctx.get("root_post"):
                        ctx.update({
                            "root_post": thread_ctx["root_post"],
                            "thread_depth": thread_ctx.get("thread_depth", 1),
                            "root_child_replies": thread_ctx.get("root_child_replies", []),
                        })
                    try:
                        quoted = self._extract_quoted_from_post_view(post)
                        if quoted:
                            ctx["quoted_post"] = quoted
                    except Exception:
                        pass
                    try:
                        external_url = self._extract_external_url_from_post_view(post)
                        if external_url:
                            ctx["urls"] = [external_url]
                            ctx["original_has_url"] = True
                        else:
                            if "http://" in text or "https://" in text:
                                ctx["original_has_url"] = True
                    except Exception:
                        if "http://" in text or "https://" in text:
                            ctx["original_has_url"] = True
                    item_obj = {
                        "uri": uri,
                        "cid": cid,
                        "author": author,
                        "root_uri": root_uri,
                        "ctx": ctx,
                        "text": text,
                        "created_ts": created_ts,
                        "is_following": is_following,
                    }
                    # Strict pool: only if not recently replied and not root duplicate
                    if not recent_author and not root_already_replied and not already_replied_this:
                        candidates.append(item_obj)
                    # Relaxed pool: allow recent/root duplicates to avoid empty selections
                    relaxed_candidates.append(item_obj)
            if not candidates:
                logger.warning("Bsky: no suitable post found in timeline; trying relaxed selection")
                if not relaxed_candidates:
                    return None
                # Prefer relaxed candidates that we have not replied to (by uri or root) and whose author isn't in recent cooldown
                try:
                    filtered_relaxed: List[Dict[str, Any]] = []
                    for it in relaxed_candidates:
                        u = it.get("uri")
                        ru = it.get("root_uri")
                        a = (it.get("author") or "").strip()
                        if (u and u in self._replied_uris) or (ru and ru in self._replied_uris):
                            continue
                        if a and a in self._recent_authors:
                            continue
                        filtered_relaxed.append(it)
                    pool = filtered_relaxed if filtered_relaxed else relaxed_candidates
                    # Shuffle to add variability across runs before trimming the pool
                    import random as _rand
                    pool = pool.copy()
                    _rand.shuffle(pool)
                    pick_from = pool[: min(self._candidate_pool_size, len(pool))]
                except Exception:
                    pick_from = relaxed_candidates[: min(self._candidate_pool_size, len(relaxed_candidates))]
            else:
                # Randomize among first N candidates to diversify
                pick_from = candidates[: min(self._candidate_pool_size, len(candidates))]
            # If we already follow the author, bias to more recent posts but keep older possible
            try:
                followed_present = any(bool(c.get("is_following")) for c in pick_from)
            except Exception:
                followed_present = False
            if followed_present:
                now_ts = time.time()
                half_life_hours = 6.0  # bias toward last ~6h for followed authors
                weights: List[float] = []
                for c in pick_from:
                    ts = c.get("created_ts")
                    if isinstance(ts, (int, float)) and ts > 0:
                        age_h = max(0.0, (now_ts - ts) / 3600.0)
                    else:
                        age_h = 48.0  # treat unknown as old
                    w = math.exp(-age_h / half_life_hours)
                    weights.append(w)
                # Normalize and sample
                total = sum(weights) or 1.0
                r = random.random() * total
                acc = 0.0
                chosen = pick_from[-1]
                for idx, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen = pick_from[idx]
                        break
                logger.info("Bsky: recency-weighted pick among followed authors (half-life {}h)", half_life_hours)
            else:
                # Shuffle before random choice to add cross-run variability
                try:
                    import random as _rand
                    pool = pick_from.copy()
                    _rand.shuffle(pool)
                    chosen = random.choice(pool)
                except Exception:
                    chosen = random.choice(pick_from)
            uri = chosen["uri"]
            cid = chosen["cid"]
            author = chosen.get("author")
            root_uri = chosen.get("root_uri")
            ctx = chosen["ctx"]
            self._current_post_context = ctx
            # Build reply_ref with 70/30 targeting (root vs first-level reply)
            reply_ref = None
            try:
                r = random.random()
                root_info = (ctx or {}).get("root_post") or {}
                root_uri_eff = root_uri or root_info.get("uri")
                root_cid = root_info.get("cid")
                if root_uri_eff and root_cid:
                    if r < 0.7:
                        # Reply to the root post (top-level)
                        reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                            root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                            parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                        )
                        logger.info("Bsky: targeting root post for reply (70%) uri={}", root_uri_eff)
                    else:
                        # Reply to a first-level reply to the root
                        first_level = (ctx or {}).get("root_child_replies") or []
                        my_handle_plain = (self._config.bsky_handle or "").lstrip("@").lower()
                        filtered = []
                        for it in first_level:
                            a = (it.get("author") or "").lstrip("@").lower()
                            if a and my_handle_plain and a == my_handle_plain:
                                continue
                            if it.get("uri") and it.get("cid"):
                                filtered.append(it)
                        target = random.choice(filtered) if filtered else None
                        if target:
                            t_uri = target.get("uri")
                            t_cid = target.get("cid")
                            reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                                root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                                parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=t_uri, cid=t_cid),
                            )
                            logger.info("Bsky: targeting first-level reply for reply (30%) parent_uri={} root_uri={}", t_uri, root_uri_eff)
                        else:
                            reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                                root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                                parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                            )
                            logger.info("Bsky: no first-level replies found; falling back to root")
            except Exception:
                reply_ref = None

            self._reply_target = {
                "uri": uri,
                "cid": cid,
                "root_uri": root_uri,
                "author_handle": author,
                "author_did": chosen.get("author_did"),
                "reply_ref": reply_ref or bsky_models.AppBskyFeedPost.ReplyRef(
                    root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                    parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                ),
            }
            logger.info("Bsky: prepared reply target uri={} cid={} text='{}'", uri, cid, str(chosen["text"])[:120])
            # Context flags/logs (compact)
            has_url = bool(ctx.get("original_has_url"))
            urls = ", ".join((ctx.get("urls") or [])[:2]) or "-"
            qp = "yes" if ctx.get("quoted_post") else "no"
            td = ctx.get("thread_depth") or 1
            logger.info("Bsky: context flags: original_has_url={} urls=[{}] quoted_post={} thread_depth={}", has_url, urls, qp, td)
            return chosen["text"]
            logger.warning("Bsky: no suitable post found in timeline")
            return None
        except Exception as exc:
            logger.error("Bsky: failed to prepare reply from timeline: {}", exc)
            return None

    def prepare_reply_from_discover(self) -> Optional[str]:
        """Fetch a Discover-like feed (e.g., `get_feed` or `get_timeline` algorithmic) and pick a candidate.

        Falls back to timeline if the Discover endpoint is unavailable.
        """
        try:
            # 1/3 chance: try explore trending topics (top 5) and sample items from one of them
            try:
                if random.random() < (1.0 / 3.0):
                    # Some clients expose app.bsky.feed.get_trending_topics (hypothetical). If not, use popular generator.
                    # We approximate by using the popular generator feed as a proxy and extracting top hashtags/phrases.
                    params_pop = bsky_models.AppBskyFeedGetFeed.Params(feed="at://did:plc:public/app.bsky.feed.generator/popular", limit=50)
                    pop = self._client.app.bsky.feed.get_feed(params_pop)
                    pitems: List[Any] = getattr(pop, "feed", []) or []
                    # Extract up to 5 trending-like terms (hashtags or frequent capitalized phrases)
                    texts: List[str] = []
                    for it in pitems[:50]:
                        rec = getattr(getattr(it, 'post', None), 'record', None)
                        t = getattr(rec, 'text', '') if rec else ''
                        if t:
                            texts.append(t)
                    # Naive trending extraction
                    terms: Dict[str, int] = {}
                    for t in texts:
                        for m in re.findall(r"#[A-Za-z][A-Za-z0-9_]{2,}", t):
                            terms[m.lower()] = terms.get(m.lower(), 0) + 1
                    # Fallback: frequent Title Case words
                    if not terms:
                        for t in texts:
                            for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", t):
                                terms[m] = terms.get(m, 0) + 1
                    top5 = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    if top5:
                        chosen_term = random.choice(top5)[0]
                        logger.info("Bsky: explore trending proxy chose term='{}' from top5", chosen_term)
                        # Filter feed items that contain the chosen term
                        filtered: List[Any] = []
                        for it in pitems:
                            rec = getattr(getattr(it, 'post', None), 'record', None)
                            t = getattr(rec, 'text', '') if rec else ''
                            if t and chosen_term and chosen_term.lower() in t.lower():
                                filtered.append(it)
                        if filtered:
                            items = filtered
                        else:
                            items = pitems
                    else:
                        items = pitems
                else:
                    items = None  # continue below
            except Exception:
                items = None
            try:
                # Attempt to use Discover/Popular feed if available in atproto client
                params = bsky_models.AppBskyFeedGetFeed.Params(feed="at://did:plc:public/app.bsky.feed.generator/popular", limit=50)
                feed = self._client.app.bsky.feed.get_feed(params)
                items = items if items is not None else (getattr(feed, "feed", []) or [])
            except Exception:
                # Fallback: use timeline (algorithmic) as a proxy for Discover
                params = bsky_models.AppBskyFeedGetTimeline.Params(limit=50)
                feed = self._client.app.bsky.feed.get_timeline(params)
                items = items if items is not None else (getattr(feed, "feed", []) or [])
            if not items:
                logger.warning("Bsky: discover feed empty")
                return None
            # Reuse same candidate construction as timeline
            candidates: List[Dict[str, Any]] = []
            now_s = time.time()
            self._recent_authors = {a: t for a, t in self._recent_authors.items() if (now_s - t) < self._author_cooldown_s}
            my_handle_plain = (self._config.bsky_handle or "").lstrip("@")
            for it in items:
                post = getattr(it, "post", None)
                record = getattr(post, "record", None)
                text = getattr(record, "text", "") if record else ""
                if text and len(text.strip()) >= max(1, int(getattr(self._config, "bsky_min_text_len", 10))):
                    uri = getattr(post, "uri", None)
                    cid = getattr(post, "cid", None)
                    author_obj = getattr(post, "author", None)
                    author = getattr(author_obj, "handle", None)
                    author_did = getattr(author_obj, "did", None)
                    viewer = getattr(author_obj, "viewer", None)
                    is_following = bool(getattr(viewer, "following", None)) if viewer else False
                    # Skip posts authored by ourselves
                    if author and my_handle_plain and author.lower() == my_handle_plain.lower():
                        logger.info("Bsky: skipping own post (author matched) uri={}", uri)
                        continue
                    if uri and uri in self._replied_uris:
                        continue
                    if not uri or not cid:
                        continue
                    if author and author in self._recent_authors:
                        continue
                    # Thread/root dedupe
                    try:
                        thread_ctx = self.get_thread_context(uri)
                        root_uri = None
                        if thread_ctx and thread_ctx.get("root_post", {}).get("uri"):
                            root_uri = thread_ctx["root_post"]["uri"]
                            if root_uri in self._replied_uris:
                                continue
                            # Skip if the root author is ourselves
                            root_author = (thread_ctx.get("root_post", {}) or {}).get("author") or ""
                            if my_handle_plain and root_author.lstrip("@").lower() == my_handle_plain.lower():
                                logger.info("Bsky: skipping thread (root authored by self) root_uri={}", root_uri)
                                continue
                    except Exception:
                        thread_ctx = None
                        root_uri = None
                    ctx: Dict[str, Any] = {
                        "text": text,
                        "author": f"@{author}" if author else None,
                        "uri": uri,
                        "cid": cid,
                        "author_did": author_did,
                        "is_following": is_following,
                    }
                    try:
                        quoted = self._extract_quoted_from_post_view(post)
                        if quoted:
                            ctx["quoted_post"] = quoted
                    except Exception:
                        pass
                    try:
                        external_url = self._extract_external_url_from_post_view(post)
                        if external_url:
                            ctx["urls"] = [external_url]
                            ctx["original_has_url"] = True
                        else:
                            if "http://" in text or "https://" in text:
                                ctx["original_has_url"] = True
                    except Exception:
                        if "http://" in text or "https://" in text:
                            ctx["original_has_url"] = True
                    if thread_ctx and thread_ctx.get("root_post"):
                        ctx.update({
                            "root_post": thread_ctx["root_post"],
                            "thread_depth": thread_ctx.get("thread_depth", 1),
                            "root_child_replies": thread_ctx.get("root_child_replies", []),
                        })
                    candidates.append({
                        "uri": uri,
                        "cid": cid,
                        "author": author,
                        "root_uri": root_uri,
                        "ctx": ctx,
                        "text": text,
                        "author_did": author_did,
                        "is_following": is_following,
                    })
            if not candidates:
                logger.warning("Bsky: no suitable post found in discover")
                return None
            # Shuffle candidate pool for variability before trimming
            try:
                import random as _rand
                pool = candidates.copy()
                _rand.shuffle(pool)
                pick_from = pool[: min(self._candidate_pool_size, len(pool))]
            except Exception:
                pick_from = candidates[: min(self._candidate_pool_size, len(candidates))]
            chosen = random.choice(pick_from)
            uri = chosen["uri"]
            cid = chosen["cid"]
            author = chosen.get("author")
            root_uri = chosen.get("root_uri")
            ctx = chosen["ctx"]
            self._current_post_context = ctx
            # Build reply_ref with 70/30 targeting (root vs first-level reply)
            reply_ref = None
            try:
                r = random.random()
                root_info = (ctx or {}).get("root_post") or {}
                root_uri_eff = root_uri or root_info.get("uri")
                root_cid = root_info.get("cid")
                if root_uri_eff and root_cid:
                    if r < 0.7:
                        reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                            root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                            parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                        )
                        logger.info("Bsky: targeting root post for reply (70%) uri={}", root_uri_eff)
                    else:
                        first_level = (ctx or {}).get("root_child_replies") or []
                        my_handle_plain = (self._config.bsky_handle or "").lstrip("@").lower()
                        filtered = []
                        for it in first_level:
                            a = (it.get("author") or "").lstrip("@").lower()
                            if a and my_handle_plain and a == my_handle_plain:
                                continue
                            if it.get("uri") and it.get("cid"):
                                filtered.append(it)
                        target = random.choice(filtered) if filtered else None
                        if target:
                            t_uri = target.get("uri")
                            t_cid = target.get("cid")
                            reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                                root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                                parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=t_uri, cid=t_cid),
                            )
                            logger.info("Bsky: targeting first-level reply for reply (30%) parent_uri={} root_uri={}", t_uri, root_uri_eff)
                        else:
                            reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                                root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                                parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=root_uri_eff, cid=root_cid),
                            )
                            logger.info("Bsky: no first-level replies found; falling back to root")
            except Exception:
                reply_ref = None

            self._reply_target = {
                "uri": uri,
                "cid": cid,
                "root_uri": root_uri,
                "author_handle": author,
                "author_did": chosen.get("author_did"),
                "reply_ref": reply_ref or bsky_models.AppBskyFeedPost.ReplyRef(
                    root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                    parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                ),
            }
            logger.info("Bsky: prepared discover reply target uri={} cid={} text='{}'", uri, cid, str(chosen["text"])[:120])
            has_url = bool(ctx.get("original_has_url"))
            urls = ", ".join((ctx.get("urls") or [])[:2]) or "-"
            qp = "yes" if ctx.get("quoted_post") else "no"
            td = ctx.get("thread_depth") or 1
            logger.info("Bsky: discover context flags: original_has_url={} urls=[{}] quoted_post={} thread_depth={}", has_url, urls, qp, td)
            return chosen["text"]
        except Exception as exc:
            logger.error("Bsky: failed to prepare reply from discover: {}", exc)
            return None

    def get_current_post_context(self) -> Optional[dict]:
        return self._current_post_context

    def get_thread_context(self, post_uri: str) -> Optional[Dict[str, Any]]:
        """Get expanded thread context for a given post URI.
        
        Returns a dictionary with:
        - root_post: dict with text, author, uri
        - current_post: the post we're replying to
        - thread_depth: depth from root to current
        - ancestors: up to 3 ancestor posts above current (nearest first)
        - sibling_replies: up to 4 sibling replies to the same parent
        - child_replies: up to 2 immediate replies to the current post
        """
        try:
            # Get the thread using the AT Protocol API
            params = bsky_models.AppBskyFeedGetPostThread.Params(uri=post_uri)
            thread_response = self._client.app.bsky.feed.get_post_thread(params)
            
            if not thread_response or not hasattr(thread_response, 'thread'):
                logger.warning("Bsky: failed to get thread for URI: {}", post_uri)
                return None
            
            thread = thread_response.thread
            
            # Find the root post by traversing up the thread
            root_post = self._find_root_post(thread)
            if not root_post:
                logger.warning("Bsky: could not find root post in thread")
                return None
            
            # Extract root post information (handle both ThreadViewPost and PostView shapes)
            root_post_like = getattr(root_post, 'post', root_post)
            root_record = getattr(root_post_like, 'record', None)
            root_text = getattr(root_record, 'text', '') if root_record else ''
            root_author = getattr(getattr(root_post_like, 'author', None), 'handle', None)
            root_uri = getattr(root_post_like, 'uri', None)
            root_cid = getattr(root_post_like, 'cid', None)
            
            # Count thread depth
            thread_depth = self._count_thread_depth(thread)

            # Expand context: ancestors, siblings, and children
            ancestors = self._collect_ancestors(thread, limit=3)
            sibling_replies = self._collect_sibling_replies(thread, limit=4)
            child_replies = self._collect_child_replies(thread, limit=2)
            # Also collect first-level replies to the root post (for targeting policy)
            try:
                root_child_replies = self._collect_child_replies(root_post, limit=10)
            except Exception:
                root_child_replies = []

            context = {
                'root_post': {
                    'text': root_text,
                    'author': f"@{root_author}" if root_author else None,
                    'uri': root_uri,
                    'cid': root_cid,
                },
                'current_post': self._current_post_context,
                'thread_depth': thread_depth,
                'ancestors': ancestors,
                'sibling_replies': sibling_replies,
                'child_replies': child_replies,
                'root_child_replies': root_child_replies,
            }
            
            # Verbose summary (compact)
            anc_count = len(ancestors)
            sib_count = len(sibling_replies)
            child_count = len(child_replies)
            anc_sample = (ancestors[0].get('text') or '')[:100] if anc_count else ''
            sib_sample = (sibling_replies[0].get('text') or '')[:100] if sib_count else ''
            logger.info(
                "Bsky: thread ctx: depth={} root='{}' | ancestors={} first='{}' | siblings={} first='{}' | children={}",
                thread_depth,
                root_text[:100],
                anc_count,
                anc_sample,
                sib_count,
                sib_sample,
                child_count,
            )
            return context
            
        except Exception as exc:
            logger.error("Bsky: failed to get thread context: {}", exc)
            return None

    def _find_root_post(self, thread_node) -> Optional[Any]:
        """Recursively find the root post by traversing up the thread."""
        try:
            # If this node has a parent, recurse up
            if hasattr(thread_node, 'parent') and thread_node.parent:
                return self._find_root_post(thread_node.parent)
            
            # If this node has a reply field with parent, check the parent
            if hasattr(thread_node, 'reply') and thread_node.reply:
                reply_ref = thread_node.reply
                if hasattr(reply_ref, 'parent') and reply_ref.parent:
                    # Get the parent post
                    parent_uri = getattr(reply_ref.parent, 'uri', None)
                    if parent_uri:
                        try:
                            params = bsky_models.AppBskyFeedGetPostThread.Params(uri=parent_uri)
                            parent_response = self._client.app.bsky.feed.get_post_thread(params)
                            if parent_response and hasattr(parent_response, 'thread'):
                                return self._find_root_post(parent_response.thread)
                        except Exception:
                            pass
            
            # This is the root post
            return thread_node
            
        except Exception as exc:
            logger.warning("Bsky: error finding root post: {}", exc)
            return thread_node  # Return current node as fallback

    def _count_thread_depth(self, thread_node) -> int:
        """Count the depth of the thread by traversing up."""
        try:
            depth = 0
            current = thread_node
            
            # Traverse up the thread
            while current:
                depth += 1
                
                # Check if there's a parent in the reply field
                if hasattr(current, 'reply') and current.reply:
                    reply_ref = current.reply
                    if hasattr(reply_ref, 'parent') and reply_ref.parent:
                        # We need to get the parent post to continue traversal
                        parent_uri = getattr(reply_ref.parent, 'uri', None)
                        if parent_uri:
                            try:
                                params = bsky_models.AppBskyFeedGetPostThread.Params(uri=parent_uri)
                                parent_response = self._client.app.bsky.feed.get_post_thread(params)
                                if parent_response and hasattr(parent_response, 'thread'):
                                    current = parent_response.thread
                                    continue
                            except Exception:
                                break
                
                # Check if there's a direct parent field
                if hasattr(current, 'parent') and current.parent:
                    current = current.parent
                    continue
                
                break
            
            return depth
            
        except Exception as exc:
            logger.warning("Bsky: error counting thread depth: {}", exc)
            return 1  # Return 1 as fallback

    def _post_view_to_dict(self, node_like: Any) -> Dict[str, Any]:
        """Extract minimal fields from a ThreadViewPost or PostView into a plain dict.

        Supports both shapes by checking for a nested 'post'.
        """
        try:
            post = getattr(node_like, 'post', node_like)
            record = getattr(post, 'record', None)
            text = getattr(record, 'text', '') if record else ''
            author = getattr(getattr(post, 'author', None), 'handle', None)
            uri = getattr(post, 'uri', None)
            cid = getattr(post, 'cid', None)
            return {
                'text': text or '',
                'author': f"@{author}" if author else None,
                'uri': uri,
                'cid': cid,
            }
        except Exception:
            return {'text': '', 'author': None, 'uri': None, 'cid': None}

    def _collect_ancestors(self, thread_node: Any, limit: int = 3) -> List[Dict[str, Any]]:
        """Walk up the parent chain and collect up to `limit` ancestor posts (nearest first)."""
        collected: List[Dict[str, Any]] = []
        try:
            current = thread_node
            while len(collected) < max(0, limit):
                parent_node = getattr(current, 'parent', None)
                if not parent_node and hasattr(current, 'reply') and current.reply and hasattr(current.reply, 'parent'):
                    # Fetch parent via URI if only a reference is available
                    parent_uri = getattr(current.reply.parent, 'uri', None)
                    if parent_uri:
                        try:
                            params = bsky_models.AppBskyFeedGetPostThread.Params(uri=parent_uri)
                            parent_response = self._client.app.bsky.feed.get_post_thread(params)
                            parent_node = getattr(parent_response, 'thread', None)
                        except Exception:
                            parent_node = None
                if not parent_node:
                    break
                try:
                    collected.append(self._post_view_to_dict(parent_node))
                except Exception:
                    pass
                current = parent_node
        except Exception:
            pass
        return collected

    def _collect_sibling_replies(self, thread_node: Any, limit: int = 4) -> List[Dict[str, Any]]:
        """Collect up to `limit` sibling replies to the same parent as the current node."""
        results: List[Dict[str, Any]] = []
        try:
            parent_node = getattr(thread_node, 'parent', None)
            if not parent_node and hasattr(thread_node, 'reply') and thread_node.reply and hasattr(thread_node.reply, 'parent'):
                parent_uri = getattr(thread_node.reply.parent, 'uri', None)
                if parent_uri:
                    try:
                        params = bsky_models.AppBskyFeedGetPostThread.Params(uri=parent_uri)
                        parent_response = self._client.app.bsky.feed.get_post_thread(params)
                        parent_node = getattr(parent_response, 'thread', None)
                    except Exception:
                        parent_node = None
            if not parent_node:
                return results
            replies = getattr(parent_node, 'replies', None) or []
            # Exclude current node by URI
            cur_post = getattr(thread_node, 'post', thread_node)
            cur_uri = getattr(cur_post, 'uri', None)
            for r in replies:
                try:
                    post_like = getattr(r, 'post', r)
                    r_uri = getattr(post_like, 'uri', None)
                    if cur_uri and r_uri and r_uri == cur_uri:
                        continue
                    d = self._post_view_to_dict(post_like)
                    if d.get('text') or d.get('uri'):
                        results.append(d)
                except Exception:
                    continue
            return results[: max(0, limit)]
        except Exception:
            return results

    def _collect_child_replies(self, thread_node: Any, limit: int = 2) -> List[Dict[str, Any]]:
        """Collect a small sample of immediate child replies to the current node."""
        results: List[Dict[str, Any]] = []
        try:
            replies = getattr(thread_node, 'replies', None) or []
            for r in replies[: max(0, limit)]:
                try:
                    post_like = getattr(r, 'post', r)
                    results.append(self._post_view_to_dict(post_like))
                except Exception:
                    continue
        except Exception:
            return results
        return results

    def like_some_thread_replies(self, thread_ctx: Optional[Dict[str, Any]], max_likes: int = 5) -> None:
        """Randomly like up to `max_likes` replies from the visible thread context with human-like delays.

        Picks from sibling_replies and child_replies. Waits 5–10 seconds between likes.
        Safe to call when no thread context is present.
        """
        try:
            if not thread_ctx:
                return
            candidates: List[Dict[str, Any]] = []
            for key in ("sibling_replies", "child_replies"):
                items = thread_ctx.get(key) or []
                for it in items:
                    uri = it.get('uri')
                    if uri:
                        candidates.append(it)
            if not candidates:
                return
            # Deduplicate by uri
            seen: set[str] = set()
            uniq: List[Dict[str, Any]] = []
            for it in candidates:
                u = it.get('uri')
                if u and u not in seen:
                    seen.add(u)
                    uniq.append(it)
            if not uniq:
                return
            # Choose random count up to max_likes and available size
            import random as _random
            import time as _time
            like_count = _random.randint(0, min(max_likes, len(uniq)))
            if like_count <= 0:
                return
            _random.shuffle(uniq)
            to_like = uniq[:like_count]
            logger.info("Bsky: will like {} thread reply(ies)", like_count)
            for idx, it in enumerate(to_like, start=1):
                uri = it.get('uri')
                cid = it.get('cid')
                author = it.get('author') or ''
                try:
                    # Try like by uri; fall back to (uri, cid)
                    try:
                        self._client.like(uri)  # some client versions accept uri only
                    except Exception:
                        if cid:
                            self._client.like(uri, cid)
                        else:
                            raise
                    logger.info("Bsky: liked reply {}/{} uri={} author={}", idx, like_count, uri, author)
                except Exception as exc:
                    logger.warning("Bsky: failed to like uri={} : {}", uri, exc)
                # Human-like pause 5–10s between likes
                if idx < like_count:
                    pause = _random.uniform(5.0, 10.0)
                    logger.info("Bsky: waiting {:.1f}s before next like", pause)
                    _time.sleep(pause)
        except Exception:
            # Swallow errors to avoid impacting main flow
            return

    def send_prepared_reply(self, text: str) -> bool:
        if not self._reply_target:
            logger.warning("Bsky: send_prepared_reply called without target")
            return False
        try:
            reply_ref = self._reply_target.get("reply_ref")
            if not reply_ref:
                uri = self._reply_target.get("uri")
                cid = self._reply_target.get("cid")
                reply_ref = bsky_models.AppBskyFeedPost.ReplyRef(
                    root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                    parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                )
            # Check if the reply contains URLs and create external embed
            embed = self._create_external_embed(text)
            
            # Create URL facets to make URLs clickable in the text
            facets = self._create_url_facets(text)
            
            if embed:
                # Use external embed for rich link preview in replies
                if facets:
                    self._client.send_post(text=text, reply_to=reply_ref, embed=embed, facets=facets)
                    logger.info("Bsky: replied with external embed and URL facets ({} chars)", len(text))
                else:
                    self._client.send_post(text=text, reply_to=reply_ref, embed=embed)
                    logger.info("Bsky: replied with external embed ({} chars)", len(text))
            else:
                # Regular reply without URLs
                if facets:
                    self._client.send_post(text=text, reply_to=reply_ref, facets=facets)
                    logger.info("Bsky: replied with URL facets ({} chars)", len(text))
                else:
                    self._client.send_post(text=text, reply_to=reply_ref)
                    logger.info("Bsky: replied ({} chars)", len(text))
            try:
                # Record current post URI and root URI to avoid thread repeats
                self._record_replied(self._reply_target.get("uri"))
                root_uri = self._reply_target.get("root_uri")
                if root_uri:
                    self._record_replied(root_uri)
                # Record author cooldown
                author = self._reply_target.get("author_handle")
                if author:
                    self._recent_authors[author] = time.time()
                    # Persist
                    self._save_recent_authors()
                # Opportunistic follow with 50% chance if not already following
                try:
                    if random.random() < 0.5:
                        author_did = self._reply_target.get("author_did")
                        already_following = False
                        try:
                            # If we have viewer info in context, prefer it
                            already_following = bool((self._current_post_context or {}).get("is_following"))
                        except Exception:
                            pass
                        if author_did and not already_following:
                            try:
                                self._client.follow(author_did)
                                logger.info("Bsky: followed author did={} handle={}", author_did, author)
                            except Exception as exc:
                                logger.warning("Bsky: failed to follow did={}: {}", author_did, exc)
                except Exception:
                    pass
            except Exception:
                pass
            return True
        except Exception as exc:
            logger.error("Bsky: failed to send reply: {}", exc)
            return False

    def quote_post(self, text: str, target_uri: str) -> bool:
        try:
            self._client.send_post(text=text, embed=self._client.get_post_embed(target_uri))
            logger.info("Bsky: quote-posted ({} chars)", len(text))
            return True
        except Exception as exc:
            logger.error("Bsky: failed to quote post: {}", exc)
            return False

    def repost(self, target_uri: str) -> bool:
        try:
            self._client.like(target_uri)  # Placeholder; repost is a separate record, but some clients expose helper
            logger.info("Bsky: (placeholder) liked target {}; replace with repost helper if available", target_uri)
            return True
        except Exception as exc:
            logger.error("Bsky: failed to repost: {}", exc)
            return False

    # ---------------- Helpers -----------------
    def _extract_quoted_from_post_view(self, post_view: Any) -> Optional[Dict[str, Any]]:
        """If the post is a quote-post, return a dict with text, author, uri."""
        try:
            embed = getattr(post_view, "embed", None)
            if not embed:
                return None
            # AppBskyEmbedRecord.View
            rec = getattr(embed, "record", None)
            if rec:
                rec_uri = getattr(rec, "uri", None)
                rec_author = getattr(getattr(rec, "author", None), "handle", None)
                rec_value = getattr(rec, "value", None)
                rec_text = getattr(rec_value, "text", "") if rec_value else ""
                if rec_text or rec_uri:
                    return {
                        "text": rec_text or None,
                        "author": f"@{rec_author}" if rec_author else None,
                        "uri": rec_uri,
                    }
            # AppBskyEmbedRecordWithMedia.View: has .record as well
            recwm = getattr(embed, "record", None)
            if recwm:
                # Already handled above, but in case of different shape
                rec_uri = getattr(recwm, "uri", None)
                rec_author = getattr(getattr(recwm, "author", None), "handle", None)
                rec_value = getattr(recwm, "value", None)
                rec_text = getattr(rec_value, "text", "") if rec_value else ""
                if rec_text or rec_uri:
                    return {
                        "text": rec_text or None,
                        "author": f"@{rec_author}" if rec_author else None,
                        "uri": rec_uri,
                    }
        except Exception:
            return None
        return None

    def _extract_external_url_from_post_view(self, post_view: Any) -> Optional[str]:
        """If the post has an external link embed, return its URL."""
        try:
            embed = getattr(post_view, "embed", None)
            if not embed:
                return None
            external = getattr(embed, "external", None)
            if not external:
                return None
            url = getattr(external, "uri", None) or getattr(external, "url", None)
            return url
        except Exception:
            return None

    def _is_sports_text(self, text: str) -> bool:
        """Check if text is sports-related."""
        t = text.lower()
        sports_terms = [
            # leagues
            "nfl","nba","mlb","nhl","ncaa","premier league","la liga","serie a","bundesliga","uefa","fifa",
            # events
            "super bowl","world series","stanley cup","finals","playoffs","draft","combine","opening day",
            # teams/positions/common words
            "sixers","76ers","embiid","joel embiid","iverson","maxey","edgecombe","mccain",
            "yankees","mets","dodgers","giants","cardinals","braves","astros","phillies","padres","cubs","red sox",
            "lakers","clippers","warriors","kings","suns","spurs","rockets","mavericks","mavs","nuggets","jazz","thunder","grizzlies","trail blazers","blazers","pelicans","hornets","hawks","heat","magic","wizards","raptors","celtics","knicks","nets","bulls","bucks","pacers","pistons","cavaliers","cavs","timberwolves","wolves",
            "patriots","cowboys","eagles","giants","jets","bills","vikings","packers","bears","steelers","ravens","chiefs","broncos","49ers","dolphins","lions",
            "lebron","james","curry","steph","jokic","doncic","tatum","kyrie","harden",
            "quarterback","qb","wide receiver","running back","coach","ot","halftime",
            "homerun","home run","inning","pitcher","strikeout","walk-off",
            "goal","hat trick","offside","penalty","power play","overtime","shootout",
            # generic
            "game tonight","big game","season opener","trade deadline","free agency","playoff spot",
        ]
        strong_hits = sum(1 for k in sports_terms if k in t)
        return strong_hits >= 2 or any(k in t for k in ["super bowl","world series","nba finals","stanley cup"])


