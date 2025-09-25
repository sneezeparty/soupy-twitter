from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple

from loguru import logger
import tweepy

from .config import AppConfig


class XApiClient:
    """Thin wrapper around Tweepy for posting with the official X API.

    Note: Free tier allows write-only (create tweets), with strict monthly caps.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        # v1.1 auth (kept for potential media in future)
        self._auth_v1 = tweepy.OAuth1UserHandler(
            consumer_key=config.x_api_consumer_key or "",
            consumer_secret=config.x_api_consumer_secret or "",
            access_token=config.x_api_access_token or "",
            access_token_secret=config.x_api_access_token_secret or "",
        )
        self._api_v1 = tweepy.API(self._auth_v1)

        # v2 client for create_tweet (recommended in free/basic tiers)
        self._client_v2 = tweepy.Client(
            consumer_key=config.x_api_consumer_key or None,
            consumer_secret=config.x_api_consumer_secret or None,
            access_token=config.x_api_access_token or None,
            access_token_secret=config.x_api_access_token_secret or None,
            bearer_token=config.x_api_bearer_token or None,
            wait_on_rate_limit=True,
        )

    def create_tweet(self, text: str) -> Optional[str]:
        """Create a tweet and return its id (or None on failure)."""
        # Prefer v2 create_tweet; falls back to v1.1 update_status if necessary
        try:
            resp = self._client_v2.create_tweet(text=text)
            tweet_id = None
            try:
                tweet_id = getattr(resp, "data", {}).get("id")  # tweepy.Client returns Response with data
            except Exception:
                pass
            if tweet_id:
                logger.info("X API v2: posted tweet id={}", tweet_id)
                return str(tweet_id)
        except Exception as exc:
            logger.error("X API v2 create_tweet failed: {}", exc)
            # Attempt v1.1 as a fallback (may still be blocked at access level)
            try:
                status = self._api_v1.update_status(status=text)
                logger.info("X API v1.1: posted tweet id={}", getattr(status, "id_str", None))
                return getattr(status, "id_str", None)
            except Exception as exc2:
                logger.error("X API v1.1 update_status failed: {}", exc2)
                return None


class XApiBot:
    """Bot facade exposing a subset of TwitterBot methods using the official API.

    Free tier constraints:
    - Can post original tweets (write-only)
    - Replies, retweets, follows, and search APIs are not available on free tier
    """

    def __init__(self, config: AppConfig, llm_client) -> None:
        self._config = config
        self._client = XApiClient(config)
        self._llm = llm_client
        self._current_tweet_context: Optional[Dict[str, Any]] = None
        self._reply_target_tweet_id: Optional[str] = None

    # Lifecycle no-ops to keep main wiring similar
    def start(self) -> None:
        logger.info("XApiBot started (official API mode: {})", self._config.x_api_tier)

    def stop(self) -> None:
        logger.info("XApiBot stopped")

    # Posting
    def create_post(self, text: str) -> bool:
        tweet_id = self._client.create_tweet(text)
        return bool(tweet_id)

    # Unsupported in free tier: provide graceful stubs
    def prepare_reply_from_home_timeline(self) -> Optional[str]:
        logger.warning("Free API tier does not support reading timelines. Skipping reply flow.")
        return None

    def get_current_tweet_context(self) -> Optional[dict]:
        return self._current_tweet_context

    def send_prepared_reply(self, text: str) -> bool:
        if not self._reply_target_tweet_id:
            logger.warning("send_prepared_reply called without a prepared reply target")
            return False
        try:
            resp = self._client._client_v2.create_tweet(text=text, in_reply_to_tweet_id=self._reply_target_tweet_id)
            tweet_id = None
            try:
                tweet_id = getattr(resp, "data", {}).get("id")
            except Exception:
                pass
            if tweet_id:
                logger.info("X API v2: replied with tweet id={}", tweet_id)
                return True
            return False
        except Exception as exc:
            logger.error("X API v2 reply failed: {}", exc)
            return False

    def prepare_quote_retweet_from_home_selection(self) -> Optional[str]:
        logger.warning("Free API tier: quote-retweet not supported.")
        return None

    def send_prepared_quote(self, text: str) -> bool:
        logger.warning("Free API tier: quote-retweet not supported.")
        return False

    def follow_author_from_context(self) -> bool:
        logger.warning("Free API tier: follow actions not supported.")
        return False

    # ---------------- API read + selection (non-free tiers) ----------------
    def _score_tweet(self, text: str, replies: int, retweets: int, likes: int) -> float:
        score = 0.0
        t = (text or "").strip()
        if len(t) < 3:
            return -1.0
        score += 1.0 - min(abs(len(t) - 120) / 120.0, 1.0)
        score += min(replies, 10) / 10.0 * 0.7
        score += min(retweets, 20) / 20.0 * 0.5
        score += min(likes, 100) / 100.0 * 0.5
        tl = t.lower()
        if "http" in tl:
            score -= 0.3
        if tl.count("#") >= 3:
            score -= 0.2
        if "promoted" in tl:
            score -= 1.0
        return score

    def _extract_entities(self, tweet_obj: Any) -> Tuple[List[str], List[str], List[str]]:
        hashtags: List[str] = []
        mentions: List[str] = []
        urls: List[str] = []
        try:
            ents = getattr(tweet_obj, "entities", None) or {}
            for h in (ents.get("hashtags") or []):
                tag = h.get("tag")
                if tag:
                    hashtags.append(f"#{tag}")
            for m in (ents.get("mentions") or []):
                u = m.get("username")
                if u:
                    mentions.append(f"@{u}")
            for u in (ents.get("urls") or []):
                expanded = u.get("expanded_url") or u.get("url")
                if expanded and expanded.startswith("http"):
                    urls.append(expanded)
        except Exception:
            pass
        return (hashtags[:10], mentions[:10], urls[:5])

    def search_and_prepare_reply(self, query: Optional[str] = None) -> Optional[str]:
        """Use API v2 to search recent tweets and select a good reply target.

        Returns the selected tweet's text and stores context; sets reply target id.
        Requires non-free tier with read permission.
        """
        q = query or self._config.search_query
        try:
            resp = self._client._client_v2.search_recent_tweets(
                query=q,
                max_results=50,
                tweet_fields=["public_metrics", "entities", "created_at"],
                expansions=["author_id"],
                user_fields=["username", "name"],
            )
        except Exception as exc:
            logger.error("X API v2 search failed: {}", exc)
            return None

        data = getattr(resp, "data", None) or []
        includes = getattr(resp, "includes", None) or {}
        users = {getattr(u, "id", None): u for u in (includes.get("users") or [])}
        if not data:
            logger.warning("API search: no tweets for query '{}'", q)
            return None

        best = None
        best_score = -1e9
        for tw in data:
            try:
                pm = getattr(tw, "public_metrics", {})
                replies = int(pm.get("reply_count", 0))
                retweets = int(pm.get("retweet_count", 0))
                likes = int(pm.get("like_count", 0))
                text = getattr(tw, "text", "")
                score = self._score_tweet(text, replies, retweets, likes)
                if score > best_score:
                    best = tw
                    best_score = score
            except Exception:
                continue

        if not best:
            logger.warning("API search: no suitable tweet found")
            return None

        author_id = getattr(best, "author_id", None)
        user = users.get(author_id)
        username = getattr(user, "username", None) if user else None
        handle = f"@{username}" if username else None
        link = None
        if username:
            link = f"https://x.com/{username}/status/{getattr(best, 'id', '')}"

        hashtags, mentions, urls = self._extract_entities(best)
        text = getattr(best, "text", "")

        self._current_tweet_context = {
            "text": text,
            "author": handle,
            "link": link,
            "hashtags": hashtags,
            "mentions": mentions,
            "urls": urls,
        }
        self._reply_target_tweet_id = str(getattr(best, "id", "")) if getattr(best, "id", None) else None
        if not self._reply_target_tweet_id:
            logger.warning("API search: selected tweet missing id; cannot reply")
            return None
        logger.info("API search: prepared reply target id={} author={} link={}", self._reply_target_tweet_id, handle, link)
        return text or None


