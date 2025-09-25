from typing import Optional
import re

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import AppConfig


class LLMClient:
    """Client for LM Studio via OpenAI-compatible API."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = OpenAI(base_url=config.openai_base_url, api_key=config.local_api_key)
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
            "You are posting a Bluesky reply. Write naturally and be helpful.\n"
            "Constraints: Prefer <= 360 characters. No surrounding quotes. Do not mention @handles.\n"
            "If 'sources:' contains a relevant URL, you may include one plain URL at the end."
        )
        user_parts = [
            "Write one Bluesky reply to the post below. Reference a concrete detail.",
            f"Post: {post_text}",
        ]
        if conversation_context:
            user_parts.append(f"Context: {conversation_context}")
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
        max_chars = max(320, int(self._config.bsky_post_max_chars))
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
            "You analyze tweets to extract key points. Keep it grounded strictly in the tweet text."
        )
        user = (
            "Analyze the tweet. List 2-4 concise bullets that capture:\n"
            "- main claim(s) or point(s)\n- named entities (people/orgs)\n- tone/stance if evident.\n"
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
        """Generate an original post for X (Twitter)."""
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
            "You concisely summarize web pages for context used in replies."
        )
        user = (
            "Summarize the following page to help craft a relevant tweet reply. "
            "Use 2-5 short bullets with the most important facts. Avoid fluff.\n"
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
        """Condense search snippets into 2-3 bullets with diverse angles; keep concise.

        Emphasize facts relevant to the provided entities (if any). Avoid dictionary-style definitions.
        """
        system = (
            f"{self._config.behaviour_prompt}\n"
            "You distill quick web search snippets for contextual grounding."
        )
        joined = "\n- ".join(snippets[:6])
        parts = [
            f"Query: {query}\n",
            "From these brief snippets, extract 2-3 distinct, factual bullets. ",
            "Ground your bullets in the named entities below and the query; avoid generic word definitions. ",
            "Prefer concrete, attributable facts over opinions. Avoid redundancy.\n",
        ]
        if entities:
            parts.append(f"Entities: {', '.join(entities)}\n")
        parts.extend([
            f"Snippets:\n- {joined}\n",
            f"Keep under {max_chars} characters.\n",
            "Bullets:",
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


