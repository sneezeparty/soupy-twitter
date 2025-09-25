from __future__ import annotations

import os
import random
import re
import time
from typing import List, Optional, Tuple

from loguru import logger
from playwright.sync_api import BrowserContext, Page, Playwright, expect

from .config import AppConfig
from .captcha import CaptchaSolver
from .utils import extract_urls


class TwitterBot:
    """Automates basic interactions on X (Twitter) using Playwright."""

    def __init__(self, config: AppConfig, playwright: Playwright) -> None:
        self._config = config
        self._playwright = playwright
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._captcha_solver = CaptchaSolver(config)
        self._reply_prepared: bool = False
        self._current_tweet_context: Optional[dict] = None
        self._last_opened_permalink: Optional[str] = None
        self._quote_prepared: bool = False

    def start(self) -> None:
        os.makedirs(self._config.user_data_dir, exist_ok=True)
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=self._config.user_data_dir,
            headless=self._config.headless,
            channel="chromium",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        self._page = self._context.new_page()
        self._page.set_default_timeout(20000)
        logger.info("Browser context started. Headless={}", self._config.headless)

    def stop(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        finally:
            self._context = None
            self._page = None
        logger.info("Browser context closed.")

    # ---------------------- Login & State ----------------------
    def ensure_logged_in(self) -> None:
        assert self._page is not None
        page = self._page
        page.goto("https://x.com/home", wait_until="domcontentloaded")
        time.sleep(random.uniform(1.0, 2.0))

        if self._is_logged_in():
            logger.info("Already logged in.")
            return

        logger.info("Logging in...")
        page.goto("https://x.com/login", wait_until="domcontentloaded")
        time.sleep(random.uniform(1.0, 2.0))

        try:
            username_field = page.get_by_label("Phone, email, or username").or_(
                page.locator('input[autocomplete="username"]')
            )
            username_field.fill(self._config.x_username)
            page.get_by_role("button", name="Next").click()
        except Exception:
            # Fallback older layout
            page.locator('input[name="text"]').fill(self._config.x_username)
            page.get_by_role("button", name="Next").click()

        time.sleep(random.uniform(1.0, 2.0))

        # Optional email confirmation step
        if self._element_exists('input[name="text"]') and "Enter your email" in page.content():
            if self._config.x_email:
                page.locator('input[name="text"]').fill(self._config.x_email)
                page.get_by_role("button", name="Next").click()
                time.sleep(random.uniform(1.0, 2.0))
            else:
                logger.warning("Email challenge required but X_EMAIL not set.")

        # Password
        try:
            password_field = page.get_by_label("Password").or_(page.locator('input[name="password"]'))
            password_field.fill(self._config.x_password)
            page.get_by_role("button", name="Log in").click()
        except Exception:
            page.locator('input[name="password"]').fill(self._config.x_password)
            page.get_by_role("button", name="Log in").click()

        time.sleep(random.uniform(2.0, 3.0))

        # Possible captcha or challenge
        if self._maybe_handle_captcha():
            logger.info("Captcha handled.")

        if not self._is_logged_in():
            raise RuntimeError("Login failed; verify credentials or challenges.")

        logger.info("Logged in successfully.")

    def _is_logged_in(self) -> bool:
        assert self._page is not None
        page = self._page
        try:
            return page.locator('[data-testid="SideNav_NewTweet_Button"]').first.is_visible()
        except Exception:
            return False

    def _element_exists(self, selector: str) -> bool:
        assert self._page is not None
        return self._page.locator(selector).count() > 0

    def _get_composer(self):
        """Return a narrowed locator for the visible tweet composer textbox."""
        assert self._page is not None
        page = self._page
        candidates = [
            page.get_by_test_id("primaryColumn").get_by_test_id("tweetTextarea_0"),
            page.get_by_role("textbox", name="Post text"),
            page.locator('[data-testid="tweetTextarea_0"]'),
            page.get_by_role("textbox", name=re.compile("post text|tweet your reply", re.I)),
        ]
        for loc in candidates:
            try:
                if loc.count() > 0:
                    return loc.first
            except Exception:
                continue
        # last resort
        return page.locator('[data-testid="tweetTextarea_0"]').first

    def _fill_composer(self, text: str) -> bool:
        """Fill the composer using execCommand/keyboard to trigger Draft.js updates."""
        assert self._page is not None
        page = self._page
        try:
            # Focus editor via Playwright or JS as fallback
            try:
                self._get_composer().focus()
            except Exception:
                page.evaluate("(sel)=>{const el=document.querySelector(sel); if(el) el.focus();}",
                              '[data-testid="tweetTextarea_0"]')

            time.sleep(random.uniform(0.1, 0.2))
            escaped = text.replace("'", "\\'").replace('"', '\\"')
            js = f"""
            (function() {{
                const el = document.querySelector('[data-testid="tweetTextarea_0"]');
                if (!el) return false;
                el.focus();
                try {{ document.execCommand('selectAll', false, null); }} catch(e) {{}}
                try {{ document.execCommand('insertText', false, '{escaped}'); }} catch(e) {{ el.textContent = '{escaped}'; }}
                el.dispatchEvent(new InputEvent('input', {{bubbles:true}}));
                el.dispatchEvent(new Event('change', {{bubbles:true}}));
                return true;
            }})()
            """
            ok = page.evaluate(js)
            if not ok:
                logger.warning("execCommand insertText failed; falling back to typing")
                # Type as a fallback
                for combo in ("Control+A", "Meta+A"):
                    try:
                        page.keyboard.press(combo)
                        time.sleep(0.05)
                    except Exception:
                        pass
                page.keyboard.type(text, delay=random.randint(10, 30))
            # Nudge events
            try:
                page.keyboard.press('Space')
                page.keyboard.press('Backspace')
            except Exception:
                pass
            time.sleep(random.uniform(0.5, 1.0))
            logger.info("Text filled via execCommand/keyboard: {}", text[:50] + "..." if len(text) > 50 else text)
            return True
        except Exception as exc:
            logger.exception("Failed to fill composer via typing: {}", exc)
            return False

    def _send_via_hotkey(self) -> bool:
        """Attempt to send with keyboard shortcut (Cmd/Ctrl + Enter)."""
        assert self._page is not None
        page = self._page
        for combo in ("Meta+Enter", "Control+Enter"):
            try:
                page.keyboard.press(combo)
                time.sleep(random.uniform(0.6, 1.0))
                logger.info("Tried send via hotkey: {}", combo)
                return True
            except Exception:
                continue
        return False

    def _get_current_reply_target(self) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort extract of the author handle and link of the tweet currently being replied to.

        Returns (author_handle, link) or (None, None) if unknown.
        """
        assert self._page is not None
        page = self._page
        try:
            info = page.evaluate(
                """
                (function() {
                  const editor = document.querySelector('[data-testid="tweetTextarea_0"]');
                  if (!editor) return null;
                  let node = editor;
                  // Find nearest dialog scope if present
                  while (node && !(node.getAttribute && node.getAttribute('role') === 'dialog')) {
                    node = node.parentElement;
                  }
                  const scope = node || document;
                  const linkEl = scope.querySelector('a[href*="/status/"]');
                  const userEl = scope.querySelector('div[data-testid="User-Name"]');
                  const authorText = userEl ? userEl.textContent : null;
                  const href = linkEl ? linkEl.getAttribute('href') : null;
                  const loc = window.location && window.location.href ? window.location.href : null;
                  return { href, authorText, loc };
                })()
                """
            )
            if not info:
                return (None, None)
            href = info.get("href") or info.get("loc")
            link = None
            if href:
                link = href if href.startswith("http") else f"https://x.com{href}"
            author_text = info.get("authorText") or ""
            handle = None
            try:
                m = re.search(r"@[A-Za-z0-9_]{1,15}", author_text)
                if m:
                    handle = m.group(0)
            except Exception:
                handle = None
            return (handle, link)
        except Exception:
            return (None, None)

    def _click_tweet_button(self) -> bool:
        """Click the tweet button using JavaScript to bypass overlays."""
        assert self._page is not None
        page = self._page
        try:
            # First, let's see what buttons are available with detailed info
            debug_script = """
            (function() {
                const buttons = document.querySelectorAll('[data-testid*="tweetButton"]');
                const details = [];
                for (let i = 0; i < buttons.length; i++) {
                    const btn = buttons[i];
                    details.push({
                        testid: btn.getAttribute('data-testid'),
                        disabled: btn.disabled,
                        visible: btn.offsetParent !== null,
                        ariaLabel: btn.getAttribute('aria-label'),
                        className: btn.className,
                        textContent: btn.textContent?.trim()
                    });
                }
                return details;
            })()
            """
            
            button_details = page.evaluate(debug_script)
            logger.info("Found {} tweet buttons:", len(button_details))
            for i, detail in enumerate(button_details):
                logger.info("  Button {}: testid='{}', disabled={}, visible={}, aria='{}', text='{}'", 
                           i, detail['testid'], detail['disabled'], detail['visible'], 
                           detail['ariaLabel'], detail['textContent'])
            
            # Try multiple button selectors
            button_selectors = [
                '[data-testid="tweetButtonInline"]',
                '[data-testid="tweetButton"]',
                'button[data-testid*="tweetButton"]',
                'button[aria-label*="Post"]',
                'button[aria-label*="Tweet"]'
            ]
            
            for selector in button_selectors:
                script = f"""
                (function() {{
                    const button = document.querySelector('{selector}');
                    if (button && !button.disabled && button.offsetParent !== null) {{
                        button.click();
                        return true;
                    }}
                    return false;
                }})()
                """
                
                result = page.evaluate(script)
                if result:
                    logger.info("Tweet button clicked via JavaScript using selector: {}", selector)
                    return True
            
            logger.warning("No clickable tweet button found with any selector")
            return False
        except Exception as exc:
            logger.exception("Failed to click tweet button via JavaScript: {}", exc)
            return False

    def _click_reply_button(self) -> bool:
        """Click the reply send button scoped to the active reply dialog/composer.

        Scopes to the nearest reply dialog to avoid sending a standalone post or a quote post.
        """
        assert self._page is not None
        page = self._page
        try:
            script = """
            (function() {
                const composer = document.querySelector('[data-testid="tweetTextarea_0"]');
                if (!composer) return { ok:false, reason:'no-composer' };
                const scope = composer.closest('[role="dialog"]') || composer.closest('form') || composer.parentElement || document;
                const candidates = Array.from(scope.querySelectorAll('[data-testid="tweetButtonInline"], [data-testid="tweetButton"], button[data-testid*="tweetButton"], button[aria-label], div[role="button"][data-testid*="tweetButton"]'))
                    .filter(btn => btn && !btn.disabled && btn.offsetParent !== null);
                if (candidates.length === 0) return { ok:false, reason:'no-buttons' };
                const scored = candidates.map(btn => {
                    const text = (btn.textContent || '').trim().toLowerCase();
                    const aria = (btn.getAttribute('aria-label') || '').trim().toLowerCase();
                    let score = 0;
                    if (text.includes('reply') || aria.includes('reply')) score += 10;
                    if (text.includes('quote') || aria.includes('quote')) score -= 5;
                    return { btn, score };
                }).sort((a,b) => b.score - a.score);
                const target = (scored[0] || {}).btn || candidates[0];
                if (!target) return { ok:false, reason:'no-target' };
                target.click();
                return { ok:true };
            })()
            """
            result = page.evaluate(script)
            if isinstance(result, dict) and result.get("ok"):
                logger.info("Reply button clicked in reply dialog")
                return True
            logger.warning("Failed to click reply button (reason={})", (result or {}).get("reason"))
            return False
        except Exception as exc:
            logger.exception("Failed to click reply button via JavaScript: {}", exc)
            return False

    def _open_reply_on_permalink(self, link: str, expected_author: Optional[str] = None) -> bool:
        """Navigate to a tweet permalink and open the reply composer on that page.

        If expected_author is provided, logs whether the root tweet author matches.
        """
        assert self._page is not None
        page = self._page
        try:
            page.goto(link, wait_until="domcontentloaded")
            time.sleep(random.uniform(0.8, 1.4))
            root = page.locator('article[data-testid="tweet"]').first
            if expected_author:
                try:
                    root_author = self._extract_author_handle(root)
                    logger.info("Permalink: root author detected={} expected={}", root_author, expected_author)
                except Exception:
                    logger.info("Permalink: could not detect root author for verification")
            # Update current context from the root tweet on the permalink page
            try:
                root_text = self._extract_tweet_text(root)
                root_author = self._extract_author_handle(root)
                root_urls = self._extract_embedded_urls(root)
                self._current_tweet_context = {
                    "text": root_text,
                    "author": root_author,
                    "link": link,
                    "hashtags": self._extract_hashtags(root_text),
                    "mentions": self._extract_mentions(root_text),
                    "urls": root_urls,
                }
                logger.info("Permalink: updated context text => {}", root_text[:140] + ("..." if len(root_text) > 140 else ""))
            except Exception:
                pass
            reply_button = root.locator('[data-testid="reply"]').first
            if reply_button.count() == 0:
                logger.warning("Permalink: reply button not found on {}", link)
                return False
            reply_button.click()
            time.sleep(random.uniform(0.5, 1.0))
            composer = page.locator('[data-testid="tweetTextarea_0"]').first
            ok = composer.count() > 0
            if ok:
                self._reply_prepared = True
                self._quote_prepared = False
                logger.info("Permalink: opened reply composer on {}", link)
                self._last_opened_permalink = link
            return ok
        except Exception as exc:
            logger.exception("Permalink: failed to open reply on {}: {}", link, exc)
            return False

    def _open_quote_on_permalink(self, link: str, expected_author: Optional[str] = None) -> bool:
        """Navigate to a tweet permalink and open the Quote Post composer."""
        assert self._page is not None
        page = self._page
        try:
            page.goto(link, wait_until="domcontentloaded")
            time.sleep(random.uniform(0.8, 1.4))
            root = page.locator('article[data-testid="tweet"]').first
            if expected_author:
                try:
                    root_author = self._extract_author_handle(root)
                    logger.info("Permalink: root author detected={} expected={}", root_author, expected_author)
                except Exception:
                    logger.info("Permalink: could not detect root author for verification")
            # Update context
            try:
                root_text = self._extract_tweet_text(root)
                root_author = self._extract_author_handle(root)
                root_urls = self._extract_embedded_urls(root)
                self._current_tweet_context = {
                    "text": root_text,
                    "author": root_author,
                    "link": link,
                    "hashtags": self._extract_hashtags(root_text),
                    "mentions": self._extract_mentions(root_text),
                    "urls": root_urls,
                }
                logger.info("Permalink: updated context text => {}", root_text[:140] + ("..." if len(root_text) > 140 else ""))
            except Exception:
                pass

            # Open retweet menu then choose Quote
            rt_btn = root.locator('[data-testid="retweet"]').first
            if rt_btn.count() == 0:
                logger.warning("Permalink: retweet button not found on {}", link)
                return False
            rt_btn.click()
            time.sleep(random.uniform(0.4, 0.8))
            # Try various selectors for Quote
            candidates = [
                page.get_by_role("menuitem", name=re.compile(r"^Quote", re.I)),
                page.locator('[data-testid="retweetConfirm"]:has-text("Quote")'),
                page.locator('div[role="menuitem"]:has-text("Quote")'),
                page.locator('span:has-text("Quote post")'),
            ]
            clicked = False
            for loc in candidates:
                try:
                    if loc.count() == 0:
                        continue
                    loc.first.click()
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                logger.warning("Permalink: could not click Quote menu item")
                return False
            time.sleep(random.uniform(0.5, 1.0))
            composer = page.locator('[data-testid="tweetTextarea_0"]').first
            ok = composer.count() > 0
            if ok:
                self._quote_prepared = True
                logger.info("Permalink: opened quote composer on {}", link)
                self._last_opened_permalink = link
            return ok
        except Exception as exc:
            logger.exception("Permalink: failed to open quote on {}: {}", link, exc)
            return False

    def _extract_tweet_text(self, tweet_locator) -> str:
        """Extract visible tweet text from a tweet article locator."""
        try:
            text_nodes = tweet_locator.locator('div[data-testid="tweetText"]').all_inner_texts()
            if text_nodes:
                # Join parts preserving reasonable spacing
                return " ".join([t.strip() for t in text_nodes if t and t.strip()])
        except Exception:
            pass
        try:
            # Fallback: inner_text of the article (may include UI labels)
            raw = tweet_locator.inner_text(timeout=2000)
            return raw.strip()
        except Exception:
            return ""

    def _extract_author_handle(self, tweet_locator) -> Optional[str]:
        try:
            name_block = tweet_locator.locator('div[data-testid="User-Name"]').first
            text = name_block.inner_text(timeout=1500)
            m = re.search(r"@[A-Za-z0-9_]{1,15}", text)
            if m:
                return m.group(0)
        except Exception:
            return None
        return None

    def _extract_tweet_link(self, tweet_locator) -> Optional[str]:
        """Extract the permalink of the main tweet (prefer timestamp link)."""
        try:
            # Prefer the timestamp anchor inside the tweet (closest 'a' parent of <time>)
            time_node = tweet_locator.locator('a[href*="/status/"] time').first
            if time_node.count() > 0:
                href = time_node.evaluate("el => el.closest('a')?.getAttribute('href')")
                if href:
                    return href if href.startswith("http") else f"https://x.com{href}"
        except Exception:
            pass
        try:
            # Fallback: any status link in this article
            a = tweet_locator.locator('a[href*="/status/"]').first
            href = a.get_attribute("href")
            if href:
                return href if href.startswith("http") else f"https://x.com{href}"
        except Exception:
            pass
        return None

    def _extract_embedded_urls(self, tweet_locator) -> List[str]:
        """Extract URLs that are present in tweet text or card attachments."""
        urls: List[str] = []
        try:
            # From text
            text = self._extract_tweet_text(tweet_locator)
            urls.extend(extract_urls(text))
        except Exception:
            pass
        try:
            # From link cards
            anchors = tweet_locator.locator('a[role="link"]').all()
            for a in anchors:
                try:
                    href = a.get_attribute("href")
                    if href and href.startswith("http"):
                        if href not in urls:
                            urls.append(href)
                except Exception:
                    continue
        except Exception:
            pass
        # Cap
        return urls[:5]

    def _extract_hashtags(self, text: str) -> List[str]:
        tags = re.findall(r"#[A-Za-z0-9_]+", text)
        # Deduplicate while preserving order
        seen = set()
        result: List[str] = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                result.append(t)
        return result[:10]

    def _extract_mentions(self, text: str) -> List[str]:
        ats = re.findall(r"@[A-Za-z0-9_]{1,15}", text)
        seen = set()
        result: List[str] = []
        for a in ats:
            if a.lower() not in seen:
                seen.add(a.lower())
                result.append(a)
        return result[:10]

    def _maybe_handle_captcha(self) -> bool:
        assert self._page is not None
        page = self._page
        content = page.content()
        if "captcha" not in content.lower():
            return False
        try:
            solved = self._captcha_solver.solve_in_page(page)
            return solved
        except Exception as exc:
            logger.warning("Captcha solving failed or not implemented: {}", exc)
            return False

    # ---------------------- Posting & Replying ----------------------
    def create_post(self, text: str) -> bool:
        assert self._page is not None
        page = self._page
        self.ensure_logged_in()

        try:
            page.goto("https://x.com/home", wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))
            page.locator('[data-testid="SideNav_NewTweet_Button"]').first.click()
            time.sleep(random.uniform(0.5, 1.0))
            
            if not self._fill_composer(text):
                return False
                
            # Wait a moment for the button to become enabled
            time.sleep(random.uniform(1.0, 2.0))
            
            # Try hotkey first to bypass overlays; fallback to JS click
            if self._send_via_hotkey() or self._click_tweet_button():
                logger.info("Posted a tweet ({} chars)", len(text))
                time.sleep(random.uniform(1.0, 2.0))
                return True
            else:
                logger.warning("Failed to click tweet button")
                return False
        except Exception as exc:
            logger.exception("Failed to post tweet: {}", exc)
            return False

    def reply_to_random_search_result(self, text: str, query: Optional[str] = None) -> bool:
        assert self._page is not None
        page = self._page
        self.ensure_logged_in()
        q = query or self._config.search_query
        try:
            search_url = f"https://x.com/search?q={q}&src=typed_query&f=live"
            page.goto(search_url, wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))

            articles = page.locator('article[data-testid="tweet"]')
            count = min(articles.count(), 20)
            if count == 0:
                logger.warning("No tweets found for query: {}", q)
                return False

            idx = random.randrange(count)
            tweet = articles.nth(idx)
            tweet.scroll_into_view_if_needed()
            time.sleep(random.uniform(0.5, 1.0))

            reply_button = tweet.locator('[data-testid="reply"]')
            reply_button.first.click()
            time.sleep(random.uniform(0.5, 1.0))

            composer = page.locator('[data-testid="tweetTextarea_0"]')
            composer.fill(text)
            time.sleep(random.uniform(0.3, 0.8))

            send = page.locator('[data-testid="tweetButtonInline"], [data-testid="tweetButton"]')
            send.first.click()
            logger.info("Replied to a tweet with {} chars (query: {})", len(text), q)
            time.sleep(random.uniform(1.0, 1.5))
            return True
        except Exception as exc:
            logger.exception("Failed to reply: {}", exc)
            return False

    # ---------------------- Contextual Reply Flow ----------------------
    def _parse_action_count(self, button_locator) -> int:
        try:
            btn = button_locator.first
            label = btn.get_attribute("aria-label")
            if label:
                m = re.search(r"(\d[\d,\.]*)", label)
                if m:
                    return int(m.group(1).replace(",", "").replace(".", ""))
            try:
                txt = btn.inner_text(timeout=800)
                if txt:
                    m2 = re.search(r"(\d[\d,\.]*)", txt)
                    if m2:
                        return int(m2.group(1).replace(",", "").replace(".", ""))
            except Exception:
                pass
        except Exception:
            pass
        return 0

    def _score_tweet(self, text: str, replies: int, retweets: int, likes: int) -> float:
        score = 0.0
        t = text.strip()
        if len(t) < 3:
            return -1.0
        score += 1.0 - min(abs(len(t) - 120) / 120.0, 1.0)  # prefer medium length
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

    def _choose_best_tweet(self, articles, sample_size: int = 10) -> Optional[Tuple[object, str]]:
        total = min(articles.count(), 100)
        if total == 0:
            return None
        k = min(sample_size, total)
        indices = random.sample(range(total), k)
        best: Optional[Tuple[object, str]] = None
        best_score = -1e9
        for idx in indices:
            tweet = articles.nth(idx)
            try:
                tweet.scroll_into_view_if_needed()
                time.sleep(random.uniform(0.2, 0.5))
            except Exception:
                continue
            text = self._extract_tweet_text(tweet)
            if not text or len(text.strip()) < 3:
                continue
            replies = self._parse_action_count(tweet.locator('[data-testid="reply"]'))
            retweets = self._parse_action_count(tweet.locator('[data-testid="retweet"]'))
            likes = self._parse_action_count(tweet.locator('[data-testid="like"]'))
            score = self._score_tweet(text, replies, retweets, likes)
            if score > best_score:
                best = (tweet, text)
                best_score = score
        return best

    def _score_tweet_political(self, text: str, replies: int, retweets: int, likes: int) -> float:
        base = self._score_tweet(text, replies, retweets, likes)
        t = text.lower()
        # Simple political keyword boost
        keywords = [
            # parties, elections, branches
            "trump","biden","democrat","democrats","republican","republicans","gop","dnc","rnc","liberal","conservative",
            "election","primary","caucus","runoff","midterms","ballot","voter","turnout","gerrymander","redistricting",
            "congress","senate","house","speaker","majority","minority","whip","filibuster","committee","oversight",
            # courts and law
            "supreme court","scotus","appeals court","district court","doj","fbi","indictment","impeachment","subpoena",
            "special counsel","grand jury","trial","sentencing","plea","gag order","hatch act","ethics","fec",
            # policy topics
            "policy","bill","budget","deficit","debt ceiling","appropriations","continuing resolution","shutdown","reconciliation",
            "economy","inflation","taxes","jobs","minimum wage","healthcare","medicare","medicaid","social security",
            "guns","gun control","nra","second amendment","immigration","border","asylum","daca","dreamers","title 42",
            "climate","energy","environment","epa","paris agreement","green new deal","esg","crt","dei",
            "abortion","roe","roe v. wade","dobbs","obergefell","affirmative action",
            # foreign policy and conflicts
            "ukraine","russia","putin","nato","china","taiwan","iran","north korea","israel","gaza","hamas","hezbollah",
            "yemen","saudi","eu","un","united nations","brics",
            # notable politicians/figures
            "kamala","harris","pence","desantis","newsom","schumer","mcconnell","pelosi","aoc","ocasio-cortez","greene",
            "mtg","boebert","gaetz","fetterman","jd vance","rfk","kennedy","obama","clinton","hillary","bill clinton",
            # media and outlets
            "kimmel","abc","cbs","nbc","cnn","fox","msnbc","nyt","nytimes","washington post","wapo","reuters","ap",
            "politico","axios","the hill","realclearpolitics","fivethirtyeight","gallup",
            # legal/docs buzzwords
            "classified","declassified","top secret","mar-a-lago","wilmington","documents","search warrant",
            # misc civic
            "state of the union","sotu","executive order","veto","veto override","cbo","omb","federal reserve","interest rates",
            "media","press"
        ]
        hits = sum(1 for k in keywords if k in t)
        base += min(hits, 5) * 0.4
        return base

    def _is_sports_text(self, text: str) -> bool:
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

    def prepare_reply_to_random_search_result(self, query: Optional[str] = None) -> Optional[str]:
        """Open a random tweet from search in reply mode and return its text.

        Returns None if no tweet found or on error. On success, leaves the reply
        composer open and sets internal state to allow send_prepared_reply.
        """
        assert self._page is not None
        page = self._page
        self.ensure_logged_in()
        q = query or self._config.search_query
        try:
            search_url = f"https://x.com/search?q={q}&src=typed_query&f=live"
            page.goto(search_url, wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))

            # Wait for some tweets to load, scrolling if necessary
            articles = page.locator('article[data-testid="tweet"]')
            for _ in range(6):
                if articles.count() >= 5:
                    break
                page.mouse.wheel(0, random.randint(1200, 2200))
                time.sleep(random.uniform(0.6, 1.0))

            count = min(articles.count(), 40)
            if count == 0:
                logger.warning("No tweets found for query: {}", q)
                # Fallback to home timeline
                return self.prepare_reply_from_home_timeline()

            # Evaluate up to 10 candidates and pick the best
            # Prefer political tweets when selecting for trending quote-retweet
            total = min(articles.count(), 100)
            if total == 0:
                return None
            k = min(10, total)
            indices = random.sample(range(total), k)
            best: Optional[Tuple[object, str]] = None
            best_score = -1e9
            for idx in indices:
                tweet = articles.nth(idx)
                try:
                    tweet.scroll_into_view_if_needed()
                    time.sleep(random.uniform(0.2, 0.5))
                except Exception:
                    continue
                text = self._extract_tweet_text(tweet)
                if not text or len(text.strip()) < 3:
                    continue
                replies = self._parse_action_count(tweet.locator('[data-testid="reply"]'))
                retweets = self._parse_action_count(tweet.locator('[data-testid="retweet"]'))
                likes = self._parse_action_count(tweet.locator('[data-testid="like"]'))
                score = self._score_tweet_political(text, replies, retweets, likes)
                if score > best_score:
                    best = (tweet, text)
                    best_score = score
            chosen = best
            if chosen is not None:
                tweet, tweet_text = chosen
                link = self._extract_tweet_link(tweet)
                self._current_tweet_context = {
                    "text": tweet_text,
                    "author": self._extract_author_handle(tweet),
                    "link": link,
                    "hashtags": self._extract_hashtags(tweet_text),
                    "mentions": self._extract_mentions(tweet_text),
                    "urls": self._extract_embedded_urls(tweet),
                }
                if link and self._open_reply_on_permalink(link, expected_author=self._extract_author_handle(tweet)):
                    return tweet_text
                # Fallback to inline if permalink failed
                reply_button = tweet.locator('[data-testid="reply"]').first
                if reply_button.count() > 0:
                    reply_button.click()
                    time.sleep(random.uniform(0.5, 1.0))
                    composer = page.locator('[data-testid="tweetTextarea_0"]').first
                    if composer.count() > 0:
                        self._reply_prepared = True
                        self._quote_prepared = False
                        logger.info("Fallback: opened inline reply from search results")
                        return tweet_text

            logger.warning("Could not find a suitable tweet in search results; falling back to home timeline")
            return self.prepare_reply_from_home_timeline()
        except Exception as exc:
            logger.exception("Failed to prepare reply: {}", exc)
            self._reply_prepared = False
            return None

    def prepare_quote_retweet_from_trending(self) -> Optional[str]:
        """Open a trending topic and prepare a quote-retweet on a selected tweet; return its text."""
        assert self._page is not None
        page = self._page
        self.ensure_logged_in()
        try:
            # Open trending tab
            page.goto("https://x.com/explore/tabs/trending", wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))

            # Prefer News/Politics tab if available
            try:
                news_tab = page.get_by_role("tab", name=re.compile(r"news|politics", re.I))
                if news_tab.count() > 0:
                    news_tab.first.click()
                    time.sleep(random.uniform(0.8, 1.2))
                    logger.info("Trending: switched to News/Politics tab")
            except Exception:
                pass

            # Collect tweets from the trending feed
            articles = page.locator('article[data-testid="tweet"]')
            for _ in range(6):
                if articles.count() >= 8:
                    break
                page.mouse.wheel(0, random.randint(1200, 2000))
                time.sleep(random.uniform(0.6, 1.0))

            count = min(articles.count(), 40)
            if count == 0:
                # Fallback: click into a specific trending topic link to load its timeline
                logger.warning("No tweets found under trending; attempting to open a trending topic")
                topic_links = page.locator('a[href*="/search?q="]')
                # Heuristic: prefer links that look like topics (exclude media links)
                tcount = min(topic_links.count(), 20)
                if tcount > 0:
                    try:
                        chosen_idx = None
                        # Choose first non-sports topic if possible
                        for i in range(tcount):
                            cand = topic_links.nth(i)
                            txt = ""
                            try:
                                txt = cand.inner_text(timeout=800) or ""
                            except Exception:
                                txt = ""
                            href = cand.get_attribute("href") or ""
                            hay = f"{txt} {href}"
                            if not self._is_sports_text(hay):
                                chosen_idx = i
                                break
                        if chosen_idx is None:
                            chosen_idx = 0
                        link = topic_links.nth(chosen_idx)
                        href = link.get_attribute("href")
                        logger.info("Trending: opening topic link {}", href if href else "(unknown)")
                        link.click()
                        time.sleep(random.uniform(1.0, 1.6))
                        # Now gather tweets from the topic timeline
                        articles = page.locator('article[data-testid="tweet"]')
                        for _ in range(6):
                            if articles.count() >= 8:
                                break
                            page.mouse.wheel(0, random.randint(1200, 2000))
                            time.sleep(random.uniform(0.6, 1.0))
                        count = min(articles.count(), 40)
                    except Exception as exc:
                        logger.warning("Trending: failed to open topic link: {}", exc)
                if count == 0:
                    logger.warning("Trending: still no tweets after opening a topic")
                    return None

            # Choose best non-sports political tweet from the page
            total = min(articles.count(), 100)
            if total == 0:
                return None
            k = min(10, total)
            indices = random.sample(range(total), k)
            best: Optional[Tuple[object, str]] = None
            best_score = -1e9
            for idx in indices:
                tweet = articles.nth(idx)
                try:
                    tweet.scroll_into_view_if_needed()
                    time.sleep(random.uniform(0.2, 0.5))
                except Exception:
                    continue
                text = self._extract_tweet_text(tweet)
                if not text or len(text.strip()) < 3:
                    continue
                if self._is_sports_text(text):
                    continue
                replies = self._parse_action_count(tweet.locator('[data-testid="reply"]'))
                retweets = self._parse_action_count(tweet.locator('[data-testid="retweet"]'))
                likes = self._parse_action_count(tweet.locator('[data-testid="like"]'))
                score = self._score_tweet_political(text, replies, retweets, likes)
                if score > best_score:
                    best = (tweet, text)
                    best_score = score
            chosen = best
            if chosen is not None:
                tweet, tweet_text = chosen
                link = self._extract_tweet_link(tweet)
                if link and self._open_quote_on_permalink(link, expected_author=self._extract_author_handle(tweet)):
                    return tweet_text
                # Fallback: try inline retweet menu
                rt_btn = tweet.locator('[data-testid="retweet"]').first
                if rt_btn.count() > 0:
                    try:
                        rt_btn.click()
                        time.sleep(random.uniform(0.4, 0.8))
                        page.get_by_role("menuitem", name=re.compile(r"^Quote", re.I)).first.click()
                        time.sleep(random.uniform(0.5, 1.0))
                        self._quote_prepared = True
                        self._current_tweet_context = {
                            "text": tweet_text,
                            "author": self._extract_author_handle(tweet),
                            "link": link,
                            "hashtags": self._extract_hashtags(tweet_text),
                            "mentions": self._extract_mentions(tweet_text),
                            "urls": self._extract_embedded_urls(tweet),
                        }
                        logger.info("Prepared inline quote on trending feed")
                        return tweet_text
                    except Exception:
                        pass

            logger.warning("Failed to prepare quote-retweet from trending")
            self._quote_prepared = False
            return None
        except Exception as exc:
            logger.exception("Error preparing quote-retweet from trending: {}", exc)
            self._quote_prepared = False
            return None

    def send_prepared_quote(self, text: str) -> bool:
        """Send a quote-retweet using an already opened quote composer."""
        assert self._page is not None
        if not self._quote_prepared:
            logger.warning("send_prepared_quote called without a prepared quote state")
            return False
        page = self._page
        try:
            if not self._fill_composer(text):
                return False
            time.sleep(random.uniform(1.0, 2.0))
            if self._send_via_hotkey() or self._click_tweet_button():
                logger.info("Quote-retweeted with {} chars", len(text))
                time.sleep(random.uniform(1.0, 1.5))
                return True
            else:
                logger.warning("Failed to click tweet (quote) button")
                return False
        except Exception as exc:
            logger.exception("Failed to send prepared quote: {}", exc)
            return False
        finally:
            self._quote_prepared = False

    def prepare_quote_retweet_from_home_selection(self) -> Optional[str]:
        """Choose a tweet from the home timeline (like reply flow), but open Quote instead of Reply."""
        assert self._page is not None
        page = self._page
        self.ensure_logged_in()
        try:
            page.goto("https://x.com/home", wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))

            articles = page.locator('article[data-testid="tweet"]')
            for _ in range(8):
                if articles.count() >= 8:
                    break
                page.mouse.wheel(0, random.randint(1400, 2400))
                time.sleep(random.uniform(0.6, 1.0))

            total = min(articles.count(), 50)
            if total == 0:
                logger.warning("No tweets found on home timeline for quote-retweet")
                return None

            # Prefer political content; skip sports
            k = min(10, total)
            indices = random.sample(range(total), k)
            best: Optional[Tuple[object, str]] = None
            best_score = -1e9
            for idx in indices:
                tweet = articles.nth(idx)
                try:
                    tweet.scroll_into_view_if_needed()
                    time.sleep(random.uniform(0.2, 0.5))
                except Exception:
                    continue
                text = self._extract_tweet_text(tweet)
                if not text or len(text.strip()) < 3:
                    continue
                if self._is_sports_text(text):
                    continue
                replies = self._parse_action_count(tweet.locator('[data-testid="reply"]'))
                retweets = self._parse_action_count(tweet.locator('[data-testid="retweet"]'))
                likes = self._parse_action_count(tweet.locator('[data-testid="like"]'))
                score = self._score_tweet_political(text, replies, retweets, likes)
                if score > best_score:
                    best = (tweet, text)
                    best_score = score

            if best is None:
                logger.warning("No suitable home timeline tweet found for quote-retweet")
                return None

            tweet, tweet_text = best
            link = self._extract_tweet_link(tweet)
            if link and self._open_quote_on_permalink(link, expected_author=self._extract_author_handle(tweet)):
                return tweet_text

            # Fallback: inline quote
            try:
                rt_btn = tweet.locator('[data-testid="retweet"]').first
                if rt_btn.count() > 0:
                    rt_btn.click()
                    time.sleep(random.uniform(0.4, 0.8))
                    page.get_by_role("menuitem", name=re.compile(r"^Quote", re.I)).first.click()
                    time.sleep(random.uniform(0.5, 1.0))
                    self._quote_prepared = True
                    self._current_tweet_context = {
                        "text": tweet_text,
                        "author": self._extract_author_handle(tweet),
                        "link": link,
                        "hashtags": self._extract_hashtags(tweet_text),
                        "mentions": self._extract_mentions(tweet_text),
                        "urls": self._extract_embedded_urls(tweet),
                    }
                    logger.info("Prepared inline quote from home timeline")
                    return tweet_text
            except Exception:
                pass

            logger.warning("Failed to prepare quote-retweet from home timeline selection")
            return None
        except Exception as exc:
            logger.exception("Error in home selection quote-retweet: {}", exc)
            return None

    def send_prepared_reply(self, text: str) -> bool:
        """Send a reply using an already opened composer from prepare_reply..."""
        assert self._page is not None
        if not self._reply_prepared:
            logger.warning("send_prepared_reply called without a prepared reply state")
            return False
        page = self._page
        try:
            # Ensure we are replying on the expected permalink when available
            expected_link = (self._current_tweet_context or {}).get("link")
            expected_author = (self._current_tweet_context or {}).get("author")
            cur_author, cur_link = self._get_current_reply_target()
            if (cur_link and cur_link.endswith("/compose/post")) and self._last_opened_permalink:
                # Some layouts hide the original link; trust our last opened permalink
                logger.info("Reply target (current): author={} link={} (using last opened permalink {})", cur_author, cur_link, self._last_opened_permalink)
                cur_link = self._last_opened_permalink
            else:
                logger.info("Reply target (current): author={} link={}", cur_author, cur_link)
            if expected_link:
                if (not cur_link) or (expected_link.rstrip('/') != cur_link.rstrip('/')):
                    logger.warning("Reply target mismatch or unknown; forcing navigate to expected permalink {}", expected_link)
                    if not self._open_reply_on_permalink(expected_link, expected_author=expected_author):
                        logger.warning("Could not open reply on expected permalink; proceeding with current composer")
                # Log target after (re)open
                cur_author, cur_link = self._get_current_reply_target()
                if (cur_link and cur_link.endswith("/compose/post")) and self._last_opened_permalink:
                    logger.info("Reply target (final): author={} link={} (using last opened permalink {})", cur_author, cur_link, self._last_opened_permalink)
                    cur_link = self._last_opened_permalink
                else:
                    logger.info("Reply target (final): author={} link={}", cur_author, cur_link)
                if (not cur_link) or (expected_link.rstrip('/') != cur_link.rstrip('/')):
                    logger.error("Reply target still does not match expected permalink. Aborting send to avoid replying under wrong tweet.")
                    return False

            if not self._fill_composer(text):
                return False
                
            # Wait a moment for the button to become enabled
            time.sleep(random.uniform(1.0, 2.0))
            
            # Try hotkey first; fall back to reply-scoped click
            if self._send_via_hotkey() or self._click_reply_button():
                logger.info("Replied (contextual) with {} chars (query: {})", len(text), self._config.search_query)
                time.sleep(random.uniform(1.0, 1.5))
                return True
            else:
                logger.warning("Failed to click reply button")
                return False
        except Exception as exc:
            logger.exception("Failed to send prepared reply: {}", exc)
            return False
        finally:
            self._reply_prepared = False

    def prepare_reply_from_home_timeline(self) -> Optional[str]:
        """Open a random tweet from the home timeline in reply mode and return its text.

        Scrolls to load more if needed and filters for tweets with visible text and a reply button.
        """
        assert self._page is not None
        page = self._page
        self.ensure_logged_in()
        try:
            page.goto("https://x.com/home", wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))

            articles = page.locator('article[data-testid="tweet"]')
            # Scroll to load a reasonable number of tweets
            for _ in range(8):
                if articles.count() >= 8:
                    break
                page.mouse.wheel(0, random.randint(1400, 2400))
                time.sleep(random.uniform(0.6, 1.0))

            count = min(articles.count(), 50)
            if count == 0:
                logger.warning("No tweets found on home timeline")
                return None

            # Choose the best among 10 candidates on the home timeline
            chosen = self._choose_best_tweet(articles, sample_size=10)
            if chosen is not None:
                tweet, tweet_text = chosen
                link = self._extract_tweet_link(tweet)
                self._current_tweet_context = {
                    "text": tweet_text,
                    "author": self._extract_author_handle(tweet),
                    "link": link,
                    "hashtags": self._extract_hashtags(tweet_text),
                    "mentions": self._extract_mentions(tweet_text),
                    "urls": self._extract_embedded_urls(tweet),
                }
                if link and self._open_reply_on_permalink(link, expected_author=self._extract_author_handle(tweet)):
                    logger.info("Prepared reply via permalink from home timeline")
                    return tweet_text
                # Fallback to inline if permalink unavailable
                reply_button = tweet.locator('[data-testid="reply"]').first
                if reply_button.count() > 0:
                    reply_button.click()
                    time.sleep(random.uniform(0.5, 1.0))
                    composer = page.locator('[data-testid="tweetTextarea_0"]').first
                    if composer.count() > 0:
                        self._reply_prepared = True
                        self._quote_prepared = False
                        logger.info("Prepared inline reply on home timeline (no permalink)")
                        return tweet_text

            logger.warning("Failed to prepare reply from home timeline candidates")
            return None
        except Exception as exc:
            logger.exception("Error preparing reply from home timeline: {}", exc)
            self._reply_prepared = False
            return None

    def get_current_tweet_context(self) -> Optional[dict]:
        return self._current_tweet_context

    def follow_author_from_context(self) -> bool:
        """Visit the author's profile from the saved context and click Follow if available."""
        assert self._page is not None
        page = self._page
        if not self._current_tweet_context or not self._current_tweet_context.get("author"):
            logger.warning("No author in current tweet context; cannot follow")
            return False
        handle = self._current_tweet_context["author"]
        username = handle.lstrip("@").strip()
        try:
            profile_url = f"https://x.com/{username}"
            page.goto(profile_url, wait_until="domcontentloaded")
            time.sleep(random.uniform(1.0, 2.0))

            # Try multiple selectors for the Follow button
            candidates = [
                page.get_by_role("button", name=re.compile(r"^Follow$", re.I)),
                page.get_by_role("button", name=re.compile(r"^Follow .*", re.I)),
                page.locator('button[aria-label*="Follow" i]'),
                page.locator('[data-testid="placementTracking"]:has-text("Follow")'),
                page.locator('[data-testid*="follow"][role="button"]'),
            ]

            for loc in candidates:
                try:
                    if loc.count() == 0:
                        continue
                    btn = loc.first
                    # Skip if already following
                    try:
                        txt = btn.inner_text(timeout=1000).strip()
                        if re.search(r"^Following|^Requested", txt, re.I):
                            logger.info("Already following {}", handle)
                            return True
                    except Exception:
                        pass
                    btn.click()
                    time.sleep(random.uniform(0.8, 1.4))
                    logger.info("Followed author {}", handle)
                    return True
                except Exception:
                    continue

            logger.warning("Follow button not found for {}", handle)
            return False
        except Exception as exc:
            logger.exception("Failed to follow author {}: {}", handle, exc)
            return False


