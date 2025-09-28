from typing import Optional
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
            "Priority: The tweet itself is the primary source. Use additional context only if it directly supports the tweet. Stay on topic."
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
    def generate_bsky_reply(self, post_text: str, conversation_context: Optional[str] = None) -> str:
        """Generate a Bluesky reply that can be slightly longer and optionally include a URL.

        Target: up to ~360 chars. If a useful source URL is present in context (e.g., 'sources:'),
        you may include exactly one non-tracking URL at the end.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You are posting a Bluesky reply. Be concise and decisive; if the post is opinionated/critical, reply with a crisp, pointed line.\n"
            "Constraints: Prefer one short sentence (two max). No surrounding quotes. No @handles. Avoid hedging (no 'maybe', 'might', 'seems', 'perhaps').\n"
            "Avoid restating the original link if the post already includes one.\n"
            "Stay strictly grounded in the post and provided context; do not introduce unrelated references.\n"
            "CRITICAL: Read the context carefully. Distinguish between who is speaking vs who is being discussed. "
            "If context mentions multiple people, understand their roles (interviewer vs subject, narrator vs protagonist).\n"
            "Style: crisp, coherent sentences; engaging but not hostile. Witty is fine; avoid ad hominem.\n"
            "If 'sources:' contains a truly useful URL, include at most one plain URL at the end."
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
                + " Offer one concrete insight (why it matters, implication, or trade-off) using post or url summary details."
                + " Only refer to entities present in the post or context. If uncertain, ask a concise, relevant question."
            )
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
        logger.debug("LLM bsky reply generated (len={}): {}", len(trimmed), trimmed)
        return trimmed

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate_bsky_reply_candidates(self, post_text: str, conversation_context: Optional[str] = None, num_candidates: int = 3) -> list[str]:
        """Produce multiple Bluesky reply candidates and return them as a list."""
        candidates: list[str] = []
        for _ in range(max(1, num_candidates)):
            try:
                c = self.generate_bsky_reply(post_text, conversation_context)
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

    def select_best_bsky_reply_with_scores(self, post_text: str, conversation_context: Optional[str], candidates: list[str]) -> tuple[str, list[tuple[int, str]]]:
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
                score -= 2
            # Penalize ideological/meta framings that drift from the post (e.g., 'the system', 'wealthy interests', 'corporate elites')
            if any(phrase in lc for phrase in [
                "the system", "wealthy interests", "corporate elite", "corporate elites", "the elites", "culture war",
                "mainstream media", "agenda", "propaganda"
            ]):
                score -= 2
            # Reward question or concrete angle (but avoid confrontational framing words)
            if "?" in c:
                score += 2
            if any(k in lc for k in ["because", "so that", "implies", "means", "drivers", "downstream"]):
                score += 2
            # Penalize overtly adversarial or accusatory framings when not warranted by context
            if any(term in lc for term in ["should focus on", "the silence from", "is deafening", "not just", "stop", "trivializing"]):
                score -= 1
            # Reward specificity (numbers)
            if any(ch.isdigit() for ch in c):
                score += 1
            # Bonus if it uses a number seen in context (e.g., 81.7, 0.8)
            if any(num in c for num in context_numbers):
                score += 1
            # Prefer brevity and decisiveness
            # Penalize hedging phrases
            if any(h in lc for h in ["maybe", "might", "seems", "perhaps", "could be", "i think", "i feel"]):
                score -= 2
            # Prefer brevity: reward 80-220 the most; above max gets penalized
            if 80 <= len(c) <= 220:
                score += 2
            elif 221 <= len(c) <= 360:
                score += 0
            if len(c) > max(200, int(self._config.bsky_reply_max_chars)):
                score -= 3
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
        best = scored[0][1]
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
            "Constraints: 3–10 words, no @handles, no quotes, neutral or supportive tone. Lower-case is fine."
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

    def select_best_bsky_reply(self, post_text: str, conversation_context: Optional[str], candidates: list[str]) -> str:
        best, _ = self.select_best_bsky_reply_with_scores(post_text, conversation_context, candidates)
        return best

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def refine_bsky_reply(self, post_text: str, conversation_context: Optional[str], draft_reply: str) -> str:
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
            "Grounding: Do not introduce new facts; stay within the post and provided context.\n"
            "CRITICAL: Ensure you understand who is the subject of the story vs who is telling it. "
            "Don't confuse the narrator/interviewer with the main subject being discussed.\n"
            "Avoid artifacts like using '.com' as a person's name unless it is a URL; prefer the person's name or 'the senator'."
        )
        user_parts = [
            "Post:", post_text,
            "Context:", conversation_context or "-",
            "Draft:", draft_reply,
            "Refine the Draft (keep tone, 1 short sentence preferred; 2 max):",
        ]
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
        logger.debug("LLM: refined reply (len={}): {}", len(refined), refined)
        return refined

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
            "Constraints: Prefer 320–500 characters. Avoid spam. One relevant URL max; place it at the end."
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
        return content


