import random
import time

from loguru import logger
import re
from typing import List
import html
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs


def human_delay(min_s: float = 0.2, max_s: float = 0.6) -> None:
    delay = random.uniform(min_s, max_s)
    time.sleep(delay)
    logger.debug("Human delay {:.2f}s", delay)


def extract_urls(text: str) -> List[str]:
    """Extract likely URLs from a text, including x.com expanded links.

    Returns up to 5 unique URLs preserving order.
    """
    if not text:
        return []
    # Basic URL regex; capture http/https and bare domains with paths
    url_pattern = re.compile(r"(https?://[\w\-\.]+(?:\:[0-9]+)?(?:/[\w\-\./%\?&=#]*)?)", re.I)
    urls = []
    seen = set()
    for m in url_pattern.finditer(text):
        u = m.group(1)
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
            if len(urls) >= 5:
                break
    return urls


def fetch_and_extract_readable_text(url: str, timeout: int = 8) -> tuple[str, str]:
    """Fetch a URL and return (title, main_text) best-effort.

    - Strips scripts/styles
    - Prefers <article> content if present; else longest <p>-dense block
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    # Attempt to resolve common redirect wrappers (e.g., t.co, news.google)
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        final_url = resp.url
        logger.info("Fetch: {} -> {} (status {})", url, final_url, resp.status_code)
        # Handle google news 'url=' param
        parsed = urlparse(final_url)
        qs = parse_qs(parsed.query)
        if "url" in qs and qs["url"] and qs["url"][0].startswith("http"):
            redirect_url = qs["url"][0]
            logger.info("Fetch: following embedded url= to {}", redirect_url)
            resp = requests.get(redirect_url, headers=headers, timeout=timeout, allow_redirects=True)
    except Exception:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        logger.info("Fetch (retry): {} (status {})", url, resp.status_code)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        # Non-HTML; return short preview
        text = resp.text[:2000]
        logger.info("Fetch: non-HTML content-type '{}', {} chars", content_type, len(text))
        return (url, text)
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else url)
    # Prefer <article>
    best_node = soup.find("article")
    if not best_node:
        # Heuristic: choose div with most <p> text length
        candidates = soup.find_all(["main", "section", "div"])
        best_score = -1
        for node in candidates:
            ps = node.find_all("p")
            text_len = sum(len(p.get_text(" ", strip=True)) for p in ps)
            if text_len > best_score:
                best_score = text_len
                best_node = node
    text = best_node.get_text(" ", strip=True) if best_node else soup.get_text(" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = html.unescape(text)
    excerpt = text[:8000]
    logger.info("Fetch: parsed '{}' ({} chars extracted)", title, len(excerpt))
    return (title, excerpt)


def shorten_url_tinyurl(long_url: str, timeout: int = 6) -> str:
    """Shorten a URL using TinyURL API. Returns original on failure.

    API: https://tinyurl.com/api-create.php?url=<long>
    """
    if not long_url or not long_url.strip():
        logger.warning("TinyURL: empty URL provided")
        return long_url
    
    try:
        api = "https://tinyurl.com/api-create.php"
        resp = requests.get(api, params={"url": long_url.strip()}, timeout=timeout)
        if resp.status_code == 200:
            short = resp.text.strip()
            # Validate the response is actually a URL
            if short.startswith("http") and len(short) > 10 and "." in short:
                logger.info("TinyURL: shortened {} -> {}", long_url, short)
                return short
            else:
                logger.warning("TinyURL: invalid response format: '{}'", short)
        else:
            logger.warning("TinyURL: failed status {} for {}", resp.status_code, long_url)
    except Exception as exc:
        logger.warning("TinyURL: error for {} => {}", long_url, exc)
    
    logger.info("TinyURL: falling back to original URL: {}", long_url)
    return long_url
