from typing import Optional
import unicodedata
import re

from loguru import logger
from openai import OpenAI
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import AppConfig
from .utils import shorten_url_tinyurl


class LLMClient:
    """Client for LM Studio via OpenAI-compatible API."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        # Use explicit httpx client to avoid environment-driven proxy issues
        self._http_client = httpx.Client(http2=False)
        self._client = OpenAI(
            base_url=config.openai_base_url,
            api_key=config.local_api_key,
            http_client=self._http_client,
        )
        self._model = config.local_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_reply(self, tweet_text: str, conversation_context: Optional[str] = None) -> str:
        """Generate a short, natural reply to a tweet.

        Ensures <= 280 characters and avoids links unless necessary.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You are posting a reply. Be concise, natural, and human-like.\n"
            "Constraints: Max 280 characters. No quotation marks of any kind. Do not mention or refer to any @handles or usernames.\n"
            "No hashtags unless relevant. Avoid emojis unless fitting.\n"
            "Priority: The tweet itself is the primary source. Use additional context only if it directly supports the tweet. Stay on topic.\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        user = (
            "Write one short reply to the tweet below. Be relevant: reference a specific detail from the tweet in your reply.\n"
            f"Tweet: {tweet_text}\n"
            + (f"Additional context (optional, secondary to the tweet): {conversation_context}\n" if conversation_context else "")
            + "Reply:"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_reply_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        # Enforce: remove any wrapping quotes
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        # Enforce: remove any quotation characters that remain (straight or smart quotes/backticks)
        content = content.replace('"', "").replace("'", "")
        content = content.replace("“", "").replace("”", "").replace("‘", "").replace("’", "").replace("`", "")
        # Enforce: remove any @mentions (@handle patterns)
        content = re.sub(r"@[A-Za-z0-9_\.\-]+", "", content)
        # Collapse extra whitespace produced by removals
        content = re.sub(r"\s+", " ", content).strip()
        trimmed = (content[:279] + "…") if len(content) > 280 else content
        logger.debug("LLM reply generated (len={}): {}", len(trimmed), trimmed)
        return trimmed

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_bsky_reply(self, post_text: str, conversation_context: Optional[str] = None, anchor_phrase: Optional[str] = None) -> str:
        """Generate a Bluesky reply that can be slightly longer and optionally include a URL.

        Target: up to ~360 chars. If a useful source URL is present in context (e.g., 'sources:'),
        you may include exactly one non-tracking URL at the end.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You are posting a Bluesky reply. Be concise and decisive; if the post is opinionated/critical, reply with a crisp, pointed line.\n"
            "Constraints: Prefer one short sentence (two max). No surrounding quotes. No @handles. Avoid hedging (no 'maybe', 'might', 'seems', 'perhaps').\n"
            "Do not address the author or readers; avoid second-person pronouns entirely (no 'you', 'your').\n"
            "Avoid restating the original link if the post already includes one.\n"
            "Stay strictly grounded in the post and provided context; do not introduce unrelated references.\n"
            "Absolutely do not invent numbers, dollar amounts, or statistics that are not explicitly present in the Post or Context.\n"
            "CRITICAL: Read the context carefully. Distinguish between who is speaking vs who is being discussed. "
            "If context mentions multiple people, understand their roles (interviewer vs subject, narrator vs protagonist).\n"
            "If the post is an image or very short with little text, do not speculate about identity, gender, or backstory. React briefly to what is visible or stay neutral.\n"
            "Never infer gender, identity, or intent unless explicitly stated in the post/context.\n"
            "Style: crisp, coherent sentences; engaging but not hostile. Witty is fine; avoid ad hominem. A slightly cryptic tone is okay if it clearly references the post.\n"
            "DO NOT include any meta commentary (e.g., 'anchor word', 'in this case', 'the author is discussing', 'non sequitur').\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        user_parts = [
            "Write one Bluesky reply to the post below. Reference a concrete detail.",
            f"Post: {post_text}",
        ]
        if conversation_context:
            user_parts.append(
                "Context: "
                + conversation_context
                + "\nGuidance: Align with 'root_post', 'ancestors', or 'thread_vibe' if present."
                + " If 'url_summary:' is present, rely on it and avoid adding unrelated web-search claims."
                + " Match tone: if it's critical, be concise and pointed; if it's light, keep it light."
                + " CRITICAL: Before replying, identify WHO is the main subject of the story vs WHO is the narrator/interviewer. "
                + "Don't confuse the person telling the story with the person the story is about."
                + " Anchor the reply to at least one concrete word or short phrase that appears in the Post or Context (excluding stopwords)."
                + " If a named person/show/topic appears in Post or Context (e.g., 'Ezra Klein' or some other popular figure or topic), make the reply about that subject rather than generic commentary."
                + " Offer one concrete insight (why it matters, implication, or trade-off) using post or url summary details."
                + " Only refer to entities present in the post or context. If uncertain, ask a concise, relevant question."
            )
        # Do not include explicit anchor instructions; keep generation purely post/context-grounded
        user_parts.append("Reply:")
        user = "\n".join(user_parts)

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_reply_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        # Remove any @mentions
        content = re.sub(r"@[A-Za-z0-9_\.\-]+", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        # Trim to configured max
        max_chars = max(200, int(self._config.bsky_reply_max_chars))
        trimmed = (content[: max_chars - 1] + "…") if len(content) > max_chars else content
        # Sanitize to avoid weird symbols
        trimmed = self._strip_meta(self.sanitize_for_bsky(trimmed, max_chars))
        logger.debug("LLM bsky reply generated (len={}): {}", len(trimmed), trimmed)
        return trimmed

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_bsky_reply_candidates(self, post_text: str, conversation_context: Optional[str] = None, num_candidates: int = 3, anchor_phrase: Optional[str] = None) -> list[str]:
        """Produce multiple Bluesky reply candidates and return them as a list."""
        candidates: list[str] = []
        for _ in range(max(1, num_candidates)):
            try:
                c = self.generate_bsky_reply(post_text, conversation_context, anchor_phrase)
                candidates.append(c)
            except Exception as exc:
                logger.warning("LLM: candidate generation failed: {}", exc)
        # Always add a very brief neutral/supportive acknowledgment candidate as a fallback option
        try:
            brief = self.generate_brief_ack(post_text, conversation_context)
            if brief:
                candidates.append(brief)
        except Exception as exc:
            logger.debug("LLM: brief ack generation failed: {}", exc)
        # De-duplicate while preserving order
        seen = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    def select_best_bsky_reply_with_scores(self, post_text: str, conversation_context: Optional[str], candidates: list[str], anchor_phrase: Optional[str] = None) -> tuple[str, list[tuple[int, str]]]:
        """Return (best_reply, scored_candidates) where scored are sorted desc by score."""
        if not candidates:
            return ("", [])
        scored: list[tuple[int, str]] = []
        original_has_url = False
        if conversation_context and "original_has_url: yes" in conversation_context:
            original_has_url = True
        # Build a lightweight token set from post + context for grounding
        import re as _re
        basis = (post_text or "") + " " + (conversation_context or "")
        basis_tokens = {w for w in _re.findall(r"[A-Za-z0-9']+", basis.lower()) if len(w) > 2}
        # Build bigram set for stricter overlap
        def bigrams(text: str) -> set[str]:
            toks = [w for w in _re.findall(r"[A-Za-z0-9']+", text.lower()) if len(w) > 2]
            return {f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)} if len(toks) > 1 else set()
        basis_bigrams = bigrams(basis)
        # Named-entity-like proper tokens (simple heuristic: capitalized words from original post_text)
        proper_tokens = {w for w in _re.findall(r"\b[A-Z][a-zA-Z]+\b", post_text or "")}
        # Anchor tokens/bigrams
        anchor_tokens: set[str] = set()
        anchor_bigrams: set[str] = set()
        if anchor_phrase:
            anchor_tokens = {w for w in _re.findall(r"[A-Za-z0-9']+", anchor_phrase.lower()) if len(w) > 2}
            anchor_bigrams = bigrams(anchor_phrase)
        # Extract context-specific numbers (e.g., percentages like 81.7, 0.8)
        context_numbers = set(_re.findall(r"\b\d+(?:\.\d+)?%?\b", basis))
        for idx, c in enumerate(candidates, start=1):
            score = 0
            lc = c.lower()
            # Penalize if basically restating link/title
            if original_has_url and ("http://" in lc or "https://" in lc):
                score -= 2
            if "on.ft.com" in lc or "ft.com" in lc:
                score -= 1
            # Penalize handle/domain mentions and awkward domain-as-name artifacts (we avoid calling out users/domains)
            if _re.search(r"\b@[A-Za-z0-9_\.\-]+\b", c):
                score -= 2
            try:
                has_bare_com = (".com" in c) and _re.search(r"\b\w+\.com\b", c) and not _re.search(r"https?://\S*\.com", c)
            except Exception:
                has_bare_com = False
            if has_bare_com:
                score -= 2
            # Grounding: reward overlap with basis tokens; penalize very low overlap
            cand_tokens = {w for w in _re.findall(r"[A-Za-z0-9']+", lc) if len(w) > 2}
            overlap = len(cand_tokens & basis_tokens)
            if overlap >= 6:
                score += 2
            elif overlap >= 3:
                score += 1
            else:
                score -= 4  # strong penalty for weak grounding
            # Reward bigram overlap (stronger lexical anchoring)
            cand_bi = bigrams(c)
            bi_overlap = len(cand_bi & basis_bigrams)
            if bi_overlap >= 2:
                score += 3
            elif bi_overlap == 1:
                score += 1
            # Extra reward for anchor overlap; penalize if none
            if anchor_tokens:
                atok = len(cand_tokens & anchor_tokens)
                abi = len(cand_bi & anchor_bigrams)
                if abi >= 1 or atok >= 2:
                    score += 3
                else:
                    score -= 2
            # Require at least one concrete word (>=3 letters) from post/context
            base_sig = {w for w in basis_tokens if len(w) >= 3}
            if len(cand_tokens & base_sig) == 0:
                score -= 3
            # Penalize introducing new proper nouns not present in the original post text
            new_propers = {w for w in _re.findall(r"\b[A-Z][a-zA-Z]+\b", c) if w not in proper_tokens}
            if new_propers:
                score -= min(3, len(new_propers))
            # Penalize meta/didactic framings or analysis-language that isn't a direct reaction
            if any(phrase in lc for phrase in [
                "in this case", "the author is discussing", "this is key", "anchor word", "non-sequitur", "non sequit", "consider that"
            ]):
                score -= 5
            # Reward question or concrete angle (but avoid confrontational framing words)
            if "?" in c:
                score += 2
            if any(k in lc for k in ["because", "so that", "implies", "means", "drivers", "downstream"]):
                score += 2
            # Penalize overtly adversarial or accusatory framings when not warranted by context
            if any(term in lc for term in ["should focus on", "the silence from", "is deafening", "not just", "stop", "trivializing"]):
                score -= 1
            # Reward specificity only if number appears in the context; penalize currency amounts not in context
            has_digit = any(ch.isdigit() for ch in c)
            uses_ctx_number = any(num in c for num in context_numbers)
            if has_digit and uses_ctx_number:
                score += 1
            # Penalize invented currency amounts (e.g., $10 billion) if not present in context
            import re as _recur
            currency_mentions = _recur.findall(r"\$\s?\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|k|m|bn)?", c.lower())
            if currency_mentions and not uses_ctx_number:
                score -= 6
            # Prefer brevity and decisiveness
            # Penalize hedging phrases
            if any(h in lc for h in ["maybe", "might", "seems", "perhaps", "could be", "i think", "i feel"]):
                score -= 2
            # Penalize speculation about gender/identity without evidence
            if any(term in lc for term in [
                "female", "woman", "women", "man", "male", "he's probably", "she's probably",
                "likely a woman", "likely a man", "gender"
            ]):
                score -= 3
            # Prefer brevity: reward 80-220 the most; above max gets penalized
            if 80 <= len(c) <= 220:
                score += 2
            elif 221 <= len(c) <= 360:
                score += 0
            if len(c) > max(200, int(self._config.bsky_reply_max_chars)):
                score -= 3
            # Penalize second-person pronouns (avoid addressing author/readers)
            if any(tok in lc for tok in [" you ", " your ", "you're", "you are", "your.", "you."]):
                score -= 4
            # Small reward if it reuses a distinctive bigram from the current post (root/current text)
            basis_bi = bigrams(basis)
            if len(cand_bi & basis_bi) >= 1:
                score += 1
            # If ambiguity (low overlap) add bonus for very short replies to prefer concise acknowledgments
            if len(cand_tokens & basis_tokens) < 3:
                if len(c) <= 80:
                    score += 1
                if len(c) <= 40:
                    score += 1
                if any(w in lc for w in ["awesome", "agreed", "well said", "nice", "makes sense", "love this", "true"]):
                    score += 1
            scored.append((score, c))
            logger.debug("LLM: candidate#{} score={} len={} preview='{}'", idx, score, len(c), c[:140])
        scored.sort(key=lambda t: t[0], reverse=True)
        # LLM validation pass: always re-rank top-3 by alignment score
        best = scored[0][1]
        try:
            contenders = [txt for (_, txt) in scored[:3]]
            vals: list[tuple[float, str]] = []
            for txt in contenders:
                vals.append((self.score_reply_candidate(post_text, conversation_context, txt), txt))
            vals.sort(key=lambda t: t[0], reverse=True)
            best = vals[0][1]
        except Exception:
            pass
        logger.debug("LLM: selected best candidate score={} len={} preview='{}'", scored[0][0], len(best), best[:180])
        return (best, scored)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_brief_ack(self, post_text: str, conversation_context: Optional[str] = None) -> str:
        """Generate a very brief, neutral/supportive acknowledgment (3–10 words).

        Use when context is ambiguous or a light touch is better.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You acknowledge posts with extremely brief, human replies when appropriate.\n"
            "Constraints: 3–10 words, no @handles, no quotes, neutral or supportive tone. Lower-case is fine.\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        user = (
            "Write one extremely brief acknowledgment to the post below.\n"
            f"Post: {post_text}\n"
            + (f"Context: {conversation_context}\n" if conversation_context else "")
            + "Reply:"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.5, self._config.llm_reply_temperature),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        content = re.sub(r"@[A-Za-z0-9_\.\-]+", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        # Enforce word count
        words = content.split()
        if len(words) > 10:
            content = " ".join(words[:10])
        return content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_bsky_post_candidates(self, topic: str, url: str, snippets: list[str], num_candidates: int = 4, max_chars: int = 300) -> list[str]:
        variants = [
            "variant: focus on one concrete detail",
            "variant: emphasize why it matters",
            "variant: highlight trade-off or implication",
            "variant: connect to current policy or public impact",
        ]
        out: list[str] = []
        for i in range(max(1, num_candidates)):
            hint = variants[i % len(variants)]
            joined_snips = "\n- ".join([s[:220] for s in snippets[:5]]) if snippets else ""
            system = (
                f"{self._config.behaviour_prompt}\n"
                "Write a concise, timely Bluesky update with a clear stance.\n"
                "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
            )
            user_parts = [
                f"Topic: {topic}",
                hint,
                "Use 1-2 sentences to summarize what's important and timely,",
                "then add one crisp opinion (no hashtags unless essential).",
                "Stance: subtly critique power structures, wealth inequality, and corporate influence when relevant.",
                "Be skeptical of authority and capitalism's excesses. Support workers and question the system.",
                "If not obviously political, find the governance, labor, or inequality angle.",
                "Avoid explicit labels like 'socialist' or 'marxist' - just show the perspective naturally.",
                "Format: Commentary first, then a space and the URL as the final token.",
                "Do not include any usernames or quotes. No hashtags unless essential.",
                f"Snippets (for grounding):\n- {joined_snips}" if joined_snips else "",
                f"Include exactly this URL at the end: {url}",
                f"Keep under {max_chars} characters total.",
                "Post:",
            ]
            user = "\n".join([p for p in user_parts if p])
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=self._config.llm_post_temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = (resp.choices[0].message.content or "").strip()
            if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                content = content[1:-1].strip()
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > max_chars:
                content = content[: max_chars - 1] + "…"
            # Sanitize to avoid emojis and weird Unicode
            content = self.sanitize_for_bsky(content, max_chars)
            out.append(content)
        uniq: list[str] = []
        seen = set()
        for c in out:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def select_best_bsky_post_with_scores(self, topic: str, snippets: list[str], candidates: list[str]) -> tuple[str, list[tuple[int, str]]]:
        if not candidates:
            return ("", [])
        import re as _re
        basis_text = (topic or "") + " " + " ".join(snippets or [])
        basis_tokens = {w for w in _re.findall(r"[A-Za-z0-9']+", basis_text.lower()) if len(w) > 2}
        # Extract proper nouns/numbers from snippets for stronger grounding
        proper_tokens = {w.lower() for w in _re.findall(r"\b[A-Z][a-zA-Z]+\b", " ".join(snippets or []))}
        number_tokens = set(_re.findall(r"\b\d+(?:\.\d+)?%?\b", basis_text))
        scored: list[tuple[int, str]] = []
        for idx, c in enumerate(candidates, start=1):
            s = 0
            lc = c.lower()
            # Penalize URL-heavy candidates (after stripping URLs, require meaningful text)
            try:
                c_no_urls = _re.sub(r"https?://\S+", "", c).strip()
            except Exception:
                c_no_urls = c
            if len(c_no_urls) < 40:
                s -= 5
            # Penalize if starts with a bare domain-like token
            if _re.match(r"^\s*\w+\.(?:com|org|net|gov|co)(?:\b|/)", lc):
                s -= 3
            # Penalize summary/meta prefixes
            if any(lc.startswith(p) for p in [
                "summary:", "key issue", "post:", "post 1:", "tl;dr:", "recap:", "update:"]):
                s -= 4
            cand_tokens = {w for w in _re.findall(r"[A-Za-z0-9']+", lc) if len(w) > 2}
            overlap = len(cand_tokens & basis_tokens)
            if overlap >= 8:
                s += 3
            elif overlap >= 4:
                s += 2
            elif overlap >= 2:
                s += 1
            else:
                s -= 3
            # Reward concrete proper-noun overlap and numbers
            if proper_tokens and len({w for w in cand_tokens if w in proper_tokens}) >= 1:
                s += 2
            if any(num in c for num in number_tokens):
                s += 1
            if any(lc.startswith(pfx) for pfx in ["breaking:", "update:", "news:", "alert:", "urgent:"]):
                s -= 3
            if lc.count('#') > 2:
                s -= 1
            if 200 <= len(c) <= 320:
                s += 2
            elif len(c) <= 500:
                s += 1
            # Penalize generic ideological phrasing
            if any(phrase in lc for phrase in [
                "the system", "the system's", "wealthy interests", "corporate elite", "corporate elites", "the elites"
            ]):
                s -= 3
            scored.append((s, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        best = scored[0][1]
        try:
            if len(scored) >= 2 and (scored[0][0] - scored[1][0]) <= 1:
                vals: list[tuple[float, str]] = []
                grounding_basis = topic + "\n" + ("\n".join(snippets[:3]) if snippets else "")
                for txt in [c for (_, c) in scored[:3]]:
                    vals.append((self.score_reply_candidate(grounding_basis, None, txt), txt))
                vals.sort(key=lambda t: t[0], reverse=True)
                best = vals[0][1]
        except Exception:
            pass
        return (best, scored)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def is_press_release(self, url: str, title: str, article_excerpt: str) -> bool:
        """Classify if a page is a press release (party, gov, corporate, or campaign comms)."""
        system = (
            "Decide if this page is a press release: issued by a political party, government office, corporation, or campaign. "
            "Signs: official site sections like 'press', 'media center', 'newsroom', institutional voice, calls to action. "
            "Answer yes or no only."
        )
        sample = article_excerpt[:1000]
        user = (
            f"URL: {url}\nTitle: {title}\nExcerpt:\n{sample}\n\nAnswer yes or no only."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def score_article_recency(self, title: str, article_excerpt: str) -> float:
        """Heuristic LLM score (0.0–1.0) of recency/currentness based on textual cues in title/excerpt."""
        system = (
            "Estimate recency (0.0-1.0) from text only (no browsing). 1.0=about events unfolding now/today/this week; 0.0=stale. "
            "Look for time expressions, dates, 'today/this week', ongoing coverage language vs evergreen content. Return ONLY a number."
        )
        sample = article_excerpt[:1000]
        user = (
            f"Title: {title}\nExcerpt:\n{sample}\n\nScore:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.1, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "0").strip()
        try:
            val = float(content.split()[0].strip())
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.5

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def is_instructional_page(self, url: str, title: str, article_excerpt: str) -> bool:
        """Classify if a page is instructional/reference (how-to, guide, definition, format, encyclopedia)."""
        system = (
            "Say yes if the page is instructional/reference: how-to, guide, tutorial, format/definition/encyclopedia/reference. "
            "Say no if it is a news report or analysis about current events. Answer yes or no only."
        )
        sample = article_excerpt[:1000]
        user = (
            f"URL: {url}\nTitle: {title}\nExcerpt:\n{sample}\n\nAnswer yes or no only."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def is_public_affairs_topic(self, text: str) -> bool:
        """Classify whether the topic is public affairs (politics, policy, labor, courts, economy, social issues)."""
        system = (
            "Say yes if the topic is about public affairs (government, politics, policy, labor, courts, economy, social issues, climate). "
            "Say no if it is celebrity culture, consumer tips, generic definitions, product how-tos, or unrelated entertainment. Answer yes or no only."
        )
        user = "Topic or text:" + "\n" + (text or "") + "\nAnswer yes or no only."
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_bsky_thread(self, topic: str, snippets: list[str], segments: int = 3, max_chars: int = 300) -> list[str]:
        """Generate a small thread (2-3 posts) on the topic using provided multi-source snippets.

        Returns list of posts (without URLs). We'll append a link to the first post later.
        """
        segments = max(2, min(segments, 5))
        joined_snips = "\n- ".join([s[:240] for s in (snippets or [])[:8]]) if snippets else ""
        system = (
            f"{self._config.behaviour_prompt}\n"
            "Compose a short Bluesky thread (2-3 posts) with crisp, grounded commentary.\n"
            "Rules: No @handles, no hashtags unless essential. Avoid hedging. Each post <= max chars.\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        user = (
            f"Topic: {topic}\n"
            f"Grounding snippets:\n- {joined_snips}\n\n"
            f"Write {segments} posts. Make each post self-contained but connected; escalate ideas across posts.\n"
            f"Keep each under {max_chars} characters.\n"
            "Format: Post 1:, Post 2:, Post 3: (omit extras if fewer)."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.7, self._config.llm_post_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "").strip()
        # Parse lines into segments
        parts: list[str] = []
        try:
            import re as _re
            for line in content.splitlines():
                m = _re.match(r"^\s*Post\s*(\d+)\s*:\s*(.+)$", line.strip(), flags=_re.IGNORECASE)
                if m:
                    parts.append(m.group(2).strip())
            if not parts:
                # Fallback: split by blank lines
                raw = [p.strip() for p in content.split("\n\n") if p.strip()]
                parts = raw[:segments]
        except Exception:
            parts = [content]
        # Strip any residual labels like "post 1:", "1)" etc.
        _label_re = __import__('re').compile(r"^(?:post\s*\d+\s*:\s*|\d+\)\s*|\d+\.\s*)", __import__('re').IGNORECASE)
        cleaned_parts: list[str] = []
        for p in parts:
            try:
                p2 = _label_re.sub("", p).strip()
            except Exception:
                p2 = p.strip()
            if p2:
                cleaned_parts.append(p2)
        if cleaned_parts:
            parts = cleaned_parts
        # Sanitize each segment
        cleaned: list[str] = []
        for seg in parts[:segments]:
            txt = self.sanitize_for_bsky(seg, max_chars)
            # Strip any URLs here; caller may add one on the first post
            import re as _re2
            txt = _re2.sub(r"https?://\S+", "", txt).strip()
            if txt:
                cleaned.append(txt)
        return cleaned[:segments]

    def select_best_bsky_reply(self, post_text: str, conversation_context: Optional[str], candidates: list[str]) -> str:
        best, _ = self.select_best_bsky_reply_with_scores(post_text, conversation_context, candidates)
        return best

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def refine_bsky_reply(self, post_text: str, conversation_context: Optional[str], draft_reply: str, anchor_phrase: Optional[str] = None) -> str:
        """Polish a selected draft reply to improve coherence and human tone while keeping the edge.

        Rules:
        - Preserve meaning; no new claims beyond post/context
        - Avoid awkward artifacts (e.g., treating domains like names)
        - Keep <= configured max chars, sentence case, crisp phrasing
        - Do not add @handles; do not add links unless allowed by context 'sources:'
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You refine Bluesky replies for clarity, tone-match, and brevity. Keep it friendly; avoid hostility.\n"
            "Constraints: 1–2 short sentences, stay under the configured max (default 300–360). No @handles, no quotes.\n"
            "Grounding: Do not introduce new facts; stay within the post and provided context. The refined reply MUST reuse at least one distinctive phrase or bigram from the Post or Context (excluding stopwords), must not add new named entities, and MUST NOT use second-person pronouns (no 'you', 'your').\n"
            "CRITICAL: Ensure you understand who is the subject of the story vs who is telling it. "
            "Don't confuse the narrator/interviewer with the main subject being discussed.\n"
            "Avoid artifacts like using '.com' as a person's name unless it is a URL; prefer the person's name or 'the senator'.\n"
            "Style option: a touch of mystery is acceptable—imply the point rather than spelling it out—while staying anchored to the post/context.\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        user_parts = [
            "Post:", post_text,
            "Context:", conversation_context or "-",
            "Draft:", draft_reply,
            "Refine the Draft (keep tone, 1 short sentence preferred; 2 max):",
        ]
        if anchor_phrase:
            user_parts.extend(["Anchor:", anchor_phrase, "Keep focus on this anchor; reuse 2–5 consecutive words from it."])
        user = "\n".join(user_parts)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_reply_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        content = re.sub(r"@[A-Za-z0-9_\.\-]+", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        max_chars = max(200, int(self._config.bsky_reply_max_chars))
        refined = (content[: max_chars - 1] + "…") if len(content) > max_chars else content
        refined = self._strip_meta(self.sanitize_for_bsky(refined, max_chars))
        logger.debug("LLM: refined reply (len={}): {}", len(refined), refined)
        return refined

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def validate_reply_alignment(self, post_text: str, conversation_context: Optional[str], reply_text: str, anchor_phrase: Optional[str] = None) -> float:
        """LLM gate: return 0.0–1.0 for alignment; must be grounded with lexical overlap, no new entities."""
        system = (
            "Score how well the reply aligns with the post and context. 1.0 = strongly aligned, 0.0 = off-topic.\n"
            "Rules: Require reuse of at least one distinctive phrase or bigram from the post/context. Penalize new named entities not in post/context."
        )
        user = (
            "Post:" + "\n" + (post_text or "") + "\n\n"
            + ("Context:\n" + conversation_context + "\n\n" if conversation_context else "")
            + ("Anchor:\n" + anchor_phrase + "\n\n" if anchor_phrase else "")
            + "Reply:\n" + (reply_text or "") + "\n\nScore 0.0-1.0 only:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.1, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "0").strip()
        try:
            val = float(content.split()[0].strip())
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.5

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def coherence_refine_reply(self, post_text: str, conversation_context: Optional[str], reply_text: str, max_chars: int) -> str:
        """Final pass to ensure the reply fits thematically/contextually with the post and context.

        Rules: preserve meaning, remove meta, avoid invented numbers, avoid '-', keep behaviour and tone.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You are a careful editor. Ensure the reply matches the post and context.\n"
            "Rules: No invented numbers, dollar amounts, or entities not present in Post/Context. No meta ('in this case', 'author is discussing', etc.).\n"
            "Do not use the '-' hyphen character; prefer commas or periods. Keep 1–2 short sentences. No @handles, no quotes."
        )
        user = (
            "Post:\n" + (post_text or "") + "\n\n"
            + ("Context:\n" + (conversation_context or "") + "\n\n" if conversation_context else "")
            + "Reply (to fix if needed):\n" + (reply_text or "") + f"\n\nRewrite the reply if needed to be coherent and grounded. Keep under {max_chars} chars."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.4, self._config.llm_reply_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "").strip()
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        # Remove any @mentions and collapse whitespace
        content = re.sub(r"@[A-Za-z0-9_\.\-]+", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        # Enforce max
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        return self._strip_meta(self.sanitize_for_bsky(content, max_chars))

    # -------- Classification / Control Helpers (LLM-driven, keyword-free) --------

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def should_enrich_with_web_search(self, post_text: str, conversation_context: Optional[str]) -> bool:
        """Ask the LLM whether generic web search would add value.

        Returns True if enrichment is likely helpful; False if the post is low-info (image/meme/insult-only)
        or the context suggests search is unnecessary/off-topic.
        """
        system = (
            "You are a careful decision-maker. Decide if a quick web search would help produce a grounded, useful reply.\n"
            "Say 'yes' only if the post invites factual context (news event, cited claim, policy, data).\n"
            "Say 'no' if the post is an image-only/meme/insult-only, or purely expressive with no factual angle."
        )
        user = (
            "Post:" + "\n" + post_text + "\n" +
            ("Context:\n" + conversation_context + "\n" if conversation_context else "") +
            "Answer yes or no only."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_search_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def is_sports_text(self, text: str) -> bool:
        """Use the LLM to decide if text is primarily sports-related (to skip if desired)."""
        system = "Classify if the text is about sports (teams, games, players, league news). Respond yes or no only."
        user = "Text:\n" + text + "\nAnswer yes or no only."
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def extract_trending_terms(self, texts: list[str], max_terms: int = 5) -> list[str]:
        """Ask the LLM to extract concise, timely topics/entities from a set of posts.

        Returns up to max_terms short phrases (1–4 words), no punctuation.
        """
        if not texts:
            return []
        sample = "\n".join((t[:240] for t in texts[:50] if t))
        system = (
            "Extract the top concise topics/entities that are timely news subjects (prefer politics/public affairs).\n"
            "Return each on a new line, 1–4 words, no punctuation, no stopwords, no generic words."
        )
        user = "Posts:\n" + sample + f"\nMax terms: {max_terms}\nList only the terms."
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.4, self._config.llm_search_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        terms = [ln.strip("- •\t ") for ln in raw.splitlines() if ln.strip()]
        # Simple cleanup
        cleaned: list[str] = []
        seen = set()
        for t in terms:
            t2 = t.strip().strip(",.;:")
            if 2 <= len(t2) <= 32 and t2.lower() not in seen:
                cleaned.append(t2)
                seen.add(t2.lower())
            if len(cleaned) >= max_terms:
                break
        return cleaned

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def score_reply_candidate(self, post_text: str, conversation_context: Optional[str], candidate: str) -> float:
        """Return a numeric score (0.0–1.0) for relevance, groundedness, and tone-match.

        1.0 is best. Avoid rewarding speculation or identity/gender inference.
        """
        system = (
            "Score the reply candidate for (a) relevance to post, (b) groundedness in provided context, (c) tone-match.\n"
            "Penalize speculation or identity/gender inferences not present. Return ONLY a number 0.0 to 1.0."
        )
        user = (
            "Post:" + "\n" + post_text + "\n" +
            ("Context:\n" + conversation_context + "\n" if conversation_context else "") +
            "Candidate:\n" + candidate + "\nScore:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.1, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "0").strip()
        try:
            val = float(content.split()[0].strip())
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.5

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_bsky_post(self, topic_hint: Optional[str] = None, context: Optional[str] = None) -> str:
        """Generate an original Bluesky post that is easier to ground and may include one URL.

        Target length: 320–500 chars. If 'sources:' or URL enrichment provides a good URL, include exactly one.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "Compose a natural Bluesky post.\n"
            "Constraints: Prefer 320–500 characters. Avoid spam. One relevant URL max; place it at the end.\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        user_parts = ["Write one Bluesky post"]
        if topic_hint:
            user_parts.append(f"about: {topic_hint}")
        if context:
            user_parts.append(f"Context: {context}")
        user = "\n".join(user_parts)

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_post_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        content = re.sub(r"\s+", " ", content).strip()
        max_chars = max(1, min(300, int(self._config.bsky_post_max_chars)))
        if len(content) > max_chars + 20:
            content = content[: max_chars - 1] + "…"
        logger.debug("LLM bsky post generated (len={}): {}", len(content), content)
        return content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def analyze_tweet(self, tweet_text: str, max_chars: int = 400) -> str:
        """Produce a compact analysis of the tweet: key claims, entities, stance.

        Output: 2-4 bullets grounded ONLY in the tweet content.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You analyze tweets to extract key points. Keep it grounded strictly in the tweet text.\n"
            "CRITICAL: Distinguish between who is speaking/acting vs who is being discussed. "
            "If a post mentions multiple people, clearly identify the subject vs the narrator/interviewer."
        )
        user = (
            "Analyze the tweet. List 2-4 concise bullets that capture:\n"
            "- main claim(s) or point(s)\n- named entities (people/orgs) and their roles\n- tone/stance if evident\n"
            "- WHO is the subject being discussed vs WHO is doing the discussing\n"
            f"Tweet: {tweet_text}\n"
            f"Keep under {max_chars} characters total.\n"
            "Bullets:"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_analyze_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        logger.info("LLM: tweet analysis ({} chars)", len(content))
        return content[:max_chars]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def infer_topic(self, tweet_text: str) -> str:
        """Infer the concrete topic of a tweet in 2-6 words.

        Output should be a terse noun phrase like "apple earnings call" or "python packaging".
        No quotes, lowercase preferred.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You are an expert tweet topic tagger."
        )
        user = (
            "Infer the topic of the tweet below in 2-6 words (no quotes).\n"
            f"Tweet: {tweet_text}\n"
            "Topic:"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_analyze_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        topic = response.choices[0].message.content.strip()
        if (topic.startswith('"') and topic.endswith('"')) or (topic.startswith("'") and topic.endswith("'")):
            topic = topic[1:-1].strip()
        # Keep it short
        return topic[:60]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_own_post(self, topic_hint: Optional[str] = None) -> str:
        """Generate an original post (generic)."""
        system = (
            f"{self._config.behaviour_prompt}\n"
            "Compose a natural, human-sounding X post.\n"
            "Constraints: Max 280 characters, avoid spammy tone, no hashtags unless relevant."
        )
        user = "Write one post" + (f" about: {topic_hint}" if topic_hint else "")
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_post_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        if (content.startswith("\"") and content.endswith("\"")) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        trimmed = (content[:279] + "…") if len(content) > 280 else content
        logger.debug("LLM own post generated (len={}): {}", len(trimmed), trimmed)
        return trimmed

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def summarize_web_page(self, title: str, text: str, max_chars: int = 900) -> str:
        """Summarize a web page's content into crisp bullet points; keep to max_chars.

        Emphasize key claims, numbers, and any actionable takeaways.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You concisely summarize web pages for context used in replies.\n"
            "CRITICAL: When summarizing, clearly distinguish between who is speaking/interviewing vs who is being discussed. "
            "Identify the main subject of the story vs the narrator or interviewer."
        )
        user = (
            "Summarize the following page to help craft a relevant tweet reply. "
            "Use 2-5 short bullets with the most important facts. Avoid fluff.\n"
            "IMPORTANT: Identify WHO is the main subject of the story vs WHO is telling the story.\n"
            f"Title: {title}\n"
            f"Content: {text[: max(0, max_chars * 3)]}\n"
            f"Keep under {max_chars} characters total.\n"
            "Summary:"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_summarize_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        logger.info("LLM: summarized web page ({} chars)", len(content))
        return content[:max_chars]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def is_url_context_useful(self, post_text: str, title: str, content: str) -> bool:
        """Ask the LLM if the URL content provides rich grounding for a concise reply to the post.

        Returns True for articles that add concrete facts, names, numbers, quotes, or clear background relevant to the post.
        Returns False for thin pages (home/login/index), generic summaries, or unrelated content.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "Decide if the article provides rich, directly relevant context for replying to the post.\n"
            "Say 'yes' only if the article meaningfully grounds a short reply (facts, names, numbers, quotes).\n"
            "Say 'no' if it's generic, off-topic, or too thin (home pages, login pages, unrelated how-tos)."
        )
        body = content[:1500]
        user = (
            "Post:" + "\n" + post_text + "\n\n"
            f"Article title: {title}\n"
            f"Article excerpt:\n{body}\n\n"
            "Answer yes or no only."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_search_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def rewrite_news_queries(self, seed_text: str, article_snippet: str = "", max_queries: int = 8) -> list[str]:
        """Ask the LLM to produce focused current-news search queries from seed text.

        Emphasize entities, events, locations, dates, offices/legislation; avoid platform/meta terms.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "Generate focused, current-news search queries.")
        guidance = (
            "Rules:\n"
            "- Prefer names of people, agencies, offices, courts, companies, bills, places, and dates.\n"
            "- Avoid platform/meta terms (bluesky, bsky, twitter, x.com, social media drama) unless they are the actual news story.\n"
            "- Make each query specific enough to retrieve a single article, not a section hub (politics/world/etc).\n"
            "- Include a distinctive anchor (person, place, bill number, or month/year), or quotes/fragments OK.\n"
            "- No quotation marks, no leading dashes, no 'topic:' prefixes, no hashtags. One query per line."
        )
        user = (
            "Seed:\n" + (seed_text or "") + "\n\n"
            + ("Article snippet:\n" + article_snippet[:800] + "\n\n" if article_snippet else "")
            + guidance + f"\nMax queries: {max_queries}\nQueries:\n"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.4, self._config.llm_search_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        lines = [ln.strip("- •\t ") for ln in raw.splitlines() if ln.strip()]
        cleaned: list[str] = []
        seen = set()
        bad_markers = [
            "here is", "here are", "filtered list", "filtered queries", "one per line", "queries:", "query:",
            "topic:", "format:", "guidelines:", "max queries", "max:", "each targeting", "targeting a single specific article",
        ]
        for ln in lines:
            q = ln.strip().strip('"\'').strip()
            # Strip leading numbering/bullets like "1. ", "(1)", "- ", "• "
            try:
                q = re.sub(r"^\s*(?:\(?\d+[\)\.]\s*|[-•*]\s*)", "", q)
            except Exception:
                pass
            ql = q.lower()
            if any(m in ql for m in bad_markers):
                continue
            if any(w in ql.split() for w in ["query", "queries", "guidelines", "format", "answer", "score"]):
                continue
            if ql in {"news", "politics", "current events", "breaking news"}:
                continue
            if 2 <= len(q.split()) <= 12 and 4 <= len(q) <= 100 and ql not in seen:
                cleaned.append(q)
                seen.add(ql)
            if len(cleaned) >= max_queries:
                break
        return cleaned

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def clean_news_queries(self, queries: list[str], max_queries: int = 8) -> list[str]:
        """Ask the LLM to drop/replace platform/meta queries and return only topical, specific news queries."""
        system = (
            "Filter queries to topical news and make them target a single specific article. "
            "Remove platform/meta (bluesky, bsky, twitter, social media) unless it's the news itself. "
            "Avoid generic section queries (like 'nytimes politics'). Return one query per line."
        )
        joined = "\n".join(queries[:20])
        user = "Queries:\n" + joined + f"\nMax: {max_queries}\nFiltered list (one per line):"
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_search_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        lines = [ln.strip("- •\t ") for ln in raw.splitlines() if ln.strip()]
        out: list[str] = []
        seen = set()
        bad_markers = [
            "here is", "here are", "filtered list", "filtered queries", "one per line", "queries:", "query:",
            "topic:", "format:", "guidelines:", "max queries", "max:", "each targeting", "targeting a single specific article",
            "list:",
        ]
        for ln in lines:
            q = ln.strip().strip('"\'').strip()
            try:
                q = re.sub(r"^\s*(?:\(?\d+[\)\.]\s*|[-•*]\s*)", "", q)
            except Exception:
                pass
            ql = q.lower()
            if any(m in ql for m in bad_markers):
                continue
            if any(w in ql.split() for w in ["query", "queries", "guidelines", "format", "answer", "score"]):
                continue
            if ql in {"news", "politics", "current events", "breaking news"}:
                continue
            if 2 <= len(q.split()) <= 12 and 4 <= len(q) <= 100 and ql not in seen:
                out.append(q)
                seen.add(ql)
            if len(out) >= max_queries:
                break
        return out

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def score_news_article_relevance(self, seed_subject: str, article_title: str, article_excerpt: str) -> float:
        """Score how relevant an article is to the seed subject (0.0–1.0)."""
        system = (
            "Score relevance 0.0 to 1.0 of the article to the subject. Return ONLY a number. "
            "Prefer specific, single reporting (headline+story) over section/topic hubs. Penalize platform/meta drama unless seed is that."
        )
        user = (
            "Subject:\n" + (seed_subject or "") + "\n\n"
            + f"Title: {article_title}\n"
            + f"Excerpt:\n{article_excerpt[:1000]}\n"
            + "Score:"
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.1, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "0").strip()
        try:
            val = float(content.split()[0].strip())
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.5

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def is_specific_article_page(self, url: str, title: str, article_excerpt: str) -> bool:
        """Classify if the page is a single, specific news article (not a section/topic/home hub).

        Returns True for a story with a clear headline and body; False for section indexes, live hubs, multi-article landing pages, or paywall stubs without story content.
        """
        system = (
            "Decide if this page is a single, specific news article with a headline and story body. "
            "Say 'yes' only if it reads like one story (not a section index, category page, live blog hub, or topic landing). "
            "If it's a section like '/politics', 'world', or an index, say 'no'. Answer yes or no only."
        )
        sample = article_excerpt[:1200]
        user = (
            f"URL: {url}\nTitle: {title}\nExcerpt:\n{sample}\n\nAnswer yes or no only."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.2, self._config.llm_analyze_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def condense_search_snippets(self, query: str, snippets: list[str], max_chars: int = 500, entities: Optional[list[str]] = None) -> str:
        """Condense search snippets into exactly 3 concise factual bullets.

        Emphasize facts relevant to the provided entities (if any). Avoid dictionary-style definitions.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You distill quick web search snippets for contextual grounding."
        )
        joined = "\n- ".join(snippets[:6])
        parts = [
            f"Query: {query}\n",
            "From these brief snippets, extract exactly 3 distinct, factual bullets. ",
            "Ground your bullets in the named entities below and the query; avoid generic word definitions. ",
            "Prefer concrete, attributable facts over opinions. Avoid redundancy.\n",
        ]
        if entities:
            parts.append(f"Entities: {', '.join(entities)}\n")
        parts.extend([
            f"Snippets:\n- {joined}\n",
            f"Keep under {max_chars} characters.\n",
            "List exactly 3 bullets:",
        ])
        user = "".join(parts)
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_search_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        logger.info("LLM: condensed search snippets ({} chars)", len(content))
        return content[:max_chars]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def analyze_topic_and_generate_queries(self, context: str) -> str:
        """Analyze a seed post and URL context to determine the main topic and generate relevant search queries.
        
        Returns a formatted response with the topic on the first line and search queries on subsequent lines.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You analyze social media posts to determine their main topic and generate relevant search queries for news articles."
        )
        
        user = (
            "Analyze the following post and any URL context to determine:\n"
            "1. The main topic/subject (one concise phrase)\n"
            "2. 5-8 relevant search queries for finding related news articles\n\n"
            "Guidelines:\n"
            "- Focus on the core political/social/economic issues\n"
            "- Generate queries that would find current news articles\n"
            "- Include specific names, organizations, or events mentioned\n"
            "- Make queries specific enough to find relevant content\n"
            "- Avoid generic terms like 'news' or 'politics'\n\n"
            f"Context:\n{context}\n\n"
            "Format your response as:\n"
            "TOPIC: [main topic]\n"
            "- [search query 1]\n"
            "- [search query 2]\n"
            "- [search query 3]\n"
            "- [search query 4]\n"
            "- [search query 5]\n"
        )
        
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_analyze_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        
        content = response.choices[0].message.content.strip()
        logger.debug("LLM: topic analysis generated ({} chars)", len(content))
        return content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def draft_brief_link_opinion_post(self, topic: str, url: str, snippets: list[str], max_chars: int) -> str:
        """Produce a concise Bluesky post: one URL at end, 1-2 sentence summary, and a clear brief opinion.

        Tone: subtly left-leaning/pro-worker when applicable (no slogans), grounded in news. Keeps
        personality via behaviour_prompt. Avoids @mentions and quotes. One URL only (provided).
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "Write a concise, timely Bluesky update with a clear stance."
        )
        joined_snips = "\n- ".join([s[:240] for s in snippets[:5]]) if snippets else ""
        user_parts = [
            f"Topic: {topic}",
            "Use 1-2 sentences to summarize what's important and timely,",
            "then add one crisp opinion (no hashtags unless essential).",
            "Stance: subtly critique power structures, wealth inequality, and corporate influence when relevant.",
            "Be skeptical of authority and capitalism's excesses. Support workers and question the system.",
            "If not obviously political, find the governance, labor, or inequality angle.",
            "Avoid explicit labels like 'socialist' or 'marxist' - just show the perspective naturally.",
            "Format: Commentary first, then a space and the URL as the final token.",
            "Do not include any usernames or quotes. No hashtags unless essential.",
            "CRITICAL: Never start with prefixes like 'important:', 'BREAKING:', 'UPDATE:', 'NEWS:', 'ALERT:', 'URGENT:', 'LATEST:', 'DEVELOPING:', 'JUST IN:', 'EXCLUSIVE:', or any other news-style prefixes. Write naturally as a regular post without any introductory labels.",
            f"Snippets (for grounding):\n- {joined_snips}" if joined_snips else "",
            f"Include exactly this URL at the end: {url}",
            f"Keep under {max_chars} characters total.",
            "Post:",
        ]
        user = "\n".join([p for p in user_parts if p])
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._config.llm_post_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content.strip()
        # Sanitize
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        # Remove @mentions
        content = re.sub(r"@[A-Za-z0-9_\.\-]+", "", content)
        # Remove any existing URLs from content (more comprehensive regex)
        content = re.sub(r"https?://[^\s]+", "", content).strip()
        # Remove prefixes like "important:", "BREAKING:", "UPDATE:", etc. (only at the very beginning)
        content = re.sub(r"^(important|breaking|update|news|alert|urgent|latest|developing|just in|exclusive):\s*", "", content, flags=re.IGNORECASE)
        content = content.strip()
        # Shorten URL first to know its actual length
        short_url = shorten_url_tinyurl(url)
        logger.debug("LLM: original URL: {} -> shortened: {}", url, short_url)
        
        # Calculate space needed for URL (including space before it)
        url_space_needed = len(short_url) + 1 if short_url else 0  # +1 for space before URL
        
        # Apply character limit to content BEFORE adding URL
        effective_max = min(300, max_chars)
        content_max = effective_max - url_space_needed
        
        logger.debug("LLM: URL space needed: {}, content max: {}, current content length: {}", 
                    url_space_needed, content_max, len(content))
        
        if len(content) > content_max:
            # Try multiple truncation strategies for better readability
            truncate_at = content_max - 1
            if truncate_at > 0:
                # Strategy 1: Look for sentence endings (. ! ?) - be more lenient
                sentence_endings = ['.', '!', '?']
                best_sentence_end = -1
                for ending in sentence_endings:
                    last_ending = content.rfind(ending, 0, truncate_at)
                    if last_ending > content_max * 0.4:  # More lenient threshold
                        best_sentence_end = max(best_sentence_end, last_ending)
                
                if best_sentence_end > content_max * 0.4:
                    truncate_at = best_sentence_end + 1  # Include the punctuation
                    content = content[:truncate_at]
                    logger.debug("LLM: truncated at sentence boundary at position {}", truncate_at)
                else:
                    # Strategy 2: Look for comma or semicolon - natural pause
                    pause_chars = [',', ';', ':']
                    best_pause = -1
                    for pause in pause_chars:
                        last_pause = content.rfind(pause, 0, truncate_at)
                        if last_pause > content_max * 0.5:
                            best_pause = max(best_pause, last_pause)
                    
                    if best_pause > content_max * 0.5:
                        truncate_at = best_pause + 1
                        content = content[:truncate_at] + "…"
                        logger.debug("LLM: truncated at pause character at position {}", truncate_at)
                    else:
                        # Strategy 3: Word boundary - be more lenient
                        last_space = content.rfind(' ', 0, truncate_at)
                        if last_space > content_max * 0.6:  # More lenient threshold
                            truncate_at = last_space
                            content = content[:truncate_at] + "…"
                            logger.debug("LLM: truncated at word boundary at position {}", truncate_at)
                        else:
                            # Strategy 4: Last resort - character limit
                            content = content[:truncate_at] + "…"
                            logger.debug("LLM: truncated at character limit at position {}", truncate_at)
            else:
                content = "…"
            logger.debug("LLM: final truncated content length: {} chars", len(content))
        
        # Now add the URL
        if short_url:
            # Ensure URL is on its own line/space for proper link recognition
            content = content.rstrip()  # Remove trailing whitespace
            if not content.endswith(" "):
                content += " "  # Add space before URL
            content += short_url
            logger.debug("LLM: final post with URL: {}", content)
        # Sanitize and enforce max
        content = self.sanitize_for_bsky(content, max_chars)
        return content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def refine_bsky_post(self, topic: str, snippets: list[str], draft_post: str, max_chars: int = 300) -> str:
        """Polish the selected own-timeline post using BEHAVIOUR to guide final wording.

        Keeps 1–2 short sentences, no hashtags or @handles, no URL (we append later).
        Focuses on clarity, specificity, and tone per behaviour_prompt.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You refine a Bluesky post for clarity, specificity, and tone.\n"
            "Do not use the '-' hyphen character anywhere; prefer commas or periods instead."
        )
        joined_snips = "\n- ".join([s[:240] for s in (snippets or [])[:5]]) if snippets else ""
        user_parts = [
            f"Topic: {topic}",
            (f"Snippets for grounding:\n- {joined_snips}" if joined_snips else ""),
            "Draft post (no URL included):",
            draft_post,
            "Refine the Draft: keep 1–2 short sentences, crisp and concrete; no hashtags, no @handles, no URL.",
            f"Keep under {max_chars} characters.",
            "Post:",
        ]
        user = "\n".join([p for p in user_parts if p])
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=min(0.6, self._config.llm_post_temperature),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = (resp.choices[0].message.content or "").strip()
        # Strip quotes and links
        if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
            content = content[1:-1].strip()
        import re as _re
        content = _re.sub(r"https?://\S+", "", content).strip()
        # Sanitize and enforce length
        content = self.sanitize_for_bsky(content, max_chars)
        return content

    def sanitize_for_bsky(self, text: str, max_chars: int) -> str:
        """Normalize quotes, remove emojis/non-ASCII, collapse spaces, and enforce max length.

        Uses Unicode NFKD then ASCII encode/drop to strip unusual symbols and emojis.
        """
        if not text:
            return ""
        # Normalize and strip non-ASCII
        try:
            norm = unicodedata.normalize("NFKD", text)
            ascii_text = norm.encode("ascii", "ignore").decode("ascii")
        except Exception:
            ascii_text = text
        # Replace fancy quotes if any leaked through
        ascii_text = ascii_text.replace('"', '"').replace("'", "'")
        # Remove stray backticks
        ascii_text = ascii_text.replace("`", "")
        # Collapse whitespace
        import re as _re
        ascii_text = _re.sub(r"\s+", " ", ascii_text).strip()
        # Trim if needed
        if len(ascii_text) > max_chars:
            ascii_text = ascii_text[: max_chars - 3].rstrip() + "..."
        return ascii_text

    def _strip_meta(self, text: str) -> str:
        """Remove parenthetical meta notes and prompt/instruction mentions."""
        try:
            t = re.sub(r"\(\s*(?:note|editor's note|as requested|as instructed)[^\)]*\)", "", text, flags=re.IGNORECASE)
            t = re.sub(r"\b(prompt|instruction|as requested|as instructed)\b.*$", "", t, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", t).strip()
        except Exception:
            return text


