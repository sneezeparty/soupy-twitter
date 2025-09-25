from __future__ import annotations

from typing import Optional, Dict, Any, List
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
        self._author_cooldown_s = max(60.0, float(self._config.bsky_author_cooldown_minutes) * 60.0)
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

    # Posting
    def create_post(self, text: str) -> bool:
        try:
            self._client.send_post(text=text)
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
            now_s = time.time()
            # prune expired author cooldowns
            self._recent_authors = {a: t for a, t in self._recent_authors.items() if (now_s - t) < self._author_cooldown_s}
            for it in items:
                post = getattr(it, "post", None)
                record = getattr(post, "record", None)
                text = getattr(record, "text", "") if record else ""
                if text and len(text.strip()) >= max(1, int(getattr(self._config, "bsky_min_text_len", 10))):
                    uri = getattr(post, "uri", None)
                    cid = getattr(post, "cid", None)
                    author = getattr(getattr(post, "author", None), "handle", None)
                    # Skip if we've already replied to this post
                    if uri and uri in self._replied_uris:
                        continue
                    if not uri or not cid:
                        continue
                    # Skip recently replied-to authors (cooldown)
                    if author and author in self._recent_authors:
                        continue
                    # If this is a reply, try to find the root and skip if we've replied to that root recently
                    try:
                        thread_ctx = self.get_thread_context(uri)
                        root_uri = None
                        if thread_ctx and thread_ctx.get("root_post", {}).get("uri"):
                            root_uri = thread_ctx["root_post"]["uri"]
                            if root_uri in self._replied_uris:
                                continue
                    except Exception:
                        thread_ctx = None
                        root_uri = None
                    ctx: Dict[str, Any] = {
                        "text": text,
                        "author": f"@{author}" if author else None,
                        "uri": uri,
                        "cid": cid,
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
                    except Exception:
                        pass
                    # Include thread, quoted, and external link context if present
                    if thread_ctx and thread_ctx.get("root_post"):
                        ctx.update({
                            "root_post": thread_ctx["root_post"],
                            "thread_depth": thread_ctx.get("thread_depth", 1),
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
                    except Exception:
                        pass
                    candidates.append({
                        "uri": uri,
                        "cid": cid,
                        "author": author,
                        "root_uri": root_uri,
                        "ctx": ctx,
                        "text": text,
                    })
            if not candidates:
                logger.warning("Bsky: no suitable post found in timeline")
                return None
            # Randomize among first N candidates to diversify
            pick_from = candidates[: min(self._candidate_pool_size, len(candidates))]
            chosen = random.choice(pick_from)
            uri = chosen["uri"]
            cid = chosen["cid"]
            author = chosen.get("author")
            root_uri = chosen.get("root_uri")
            ctx = chosen["ctx"]
            self._current_post_context = ctx
            self._reply_target = {
                "uri": uri,
                "cid": cid,
                "root_uri": root_uri,
                "author_handle": author,
                "reply_ref": bsky_models.AppBskyFeedPost.ReplyRef(
                    root=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                    parent=bsky_models.ComAtprotoRepoStrongRef.Main(uri=uri, cid=cid),
                ),
            }
            logger.info("Bsky: prepared reply target uri={} cid={} text='{}'", uri, cid, str(chosen["text"])[:120])
            return chosen["text"]
            logger.warning("Bsky: no suitable post found in timeline")
            return None
        except Exception as exc:
            logger.error("Bsky: failed to prepare reply from timeline: {}", exc)
            return None

    def get_current_post_context(self) -> Optional[dict]:
        return self._current_post_context

    def get_thread_context(self, post_uri: str) -> Optional[Dict[str, Any]]:
        """Get the full thread context including the root post.
        
        Returns a dictionary with:
        - root_post: The original post at the top of the thread
        - current_post: The post we're replying to
        - thread_depth: Number of levels in the thread
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
            
            # Extract root post information
            root_record = getattr(root_post, 'record', None)
            root_text = getattr(root_record, 'text', '') if root_record else ''
            root_author = getattr(getattr(root_post, 'author', None), 'handle', None)
            root_uri = getattr(root_post, 'uri', None)
            
            # Count thread depth
            thread_depth = self._count_thread_depth(thread)
            
            context = {
                'root_post': {
                    'text': root_text,
                    'author': f"@{root_author}" if root_author else None,
                    'uri': root_uri,
                },
                'current_post': self._current_post_context,
                'thread_depth': thread_depth,
            }
            
            logger.info("Bsky: gathered thread context - root: '{}', depth: {}", 
                       root_text[:100], thread_depth)
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


