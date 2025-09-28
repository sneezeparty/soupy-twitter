import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AppConfig:
    """Holds configuration loaded from environment variables.

    All fields have sensible defaults to simplify local development.
    """

    # LM Studio (OpenAI-compatible) API
    openai_base_url: str
    local_api_key: str
    local_model: str

    # Removed Twitter/X integration

    # Behavior / system prompt
    behaviour_prompt: str

    # Captcha solver
    captcha_api_key: Optional[str]

    # Automation options
    headless: bool
    actions_per_hour_cap: int
    min_interval_minutes: int
    max_interval_minutes: int
    search_query: str
    own_posting_probability: float
    user_data_dir: str
    reply_only: bool

    # Context enrichment
    url_enrichment: bool
    web_search_enrichment: bool
    enrichment_max_chars: int
    web_search_results: int
    url_fetch_timeout: int
    url_summary_fetch_limit: int
    web_search_engine_url: str
    web_search_user_agent: str
    trending_quotes_enabled: bool
    operating_start_hour: int
    operating_end_hour: int

    # Bluesky / AT Protocol
    use_bsky: bool
    bsky_service_url: str
    bsky_handle: Optional[str]
    bsky_app_password: Optional[str]
    # Bluesky selection controls
    bsky_author_cooldown_minutes: int
    bsky_candidate_pool_size: int
    bsky_replied_log_max: int
    bsky_min_text_len: int
    bsky_reply_max_chars: int
    bsky_post_max_chars: int

    # Daily Bluesky news post
    daily_post_enabled: bool
    daily_post_hour: int  # 0-23 local time
    daily_post_window_minutes: int  # fire within +/- window
    daily_posts_per_day: int  # number of posts per day
    daily_post_min_interval_hours: int  # minimum hours between posts
    daily_post_max_interval_hours: int  # maximum hours between posts
    daily_post_first_hour: int  # hour for first post of the day

    # LLM/temperature controls
    llm_reply_temperature: float
    llm_post_temperature: float
    llm_analyze_temperature: float
    llm_summarize_temperature: float
    llm_search_temperature: float

    @staticmethod
    def _get_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y"}

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv(override=False)

        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:5112/v1"),
            local_api_key=os.getenv("LOCAL_KEY", "lm-studio"),
            local_model=os.getenv("LOCAL_CHAT", "Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF"),
            
            behaviour_prompt=os.getenv("BEHAVIOUR", "You are a helpful, friendly user."),
            captcha_api_key=os.getenv("CAPTCHA_API_KEY"),
            headless=cls._get_bool(os.getenv("HEADLESS", "false"), default=False),
            actions_per_hour_cap=int(os.getenv("ACTIONS_PER_HOUR_CAP", "4")),
            min_interval_minutes=int(os.getenv("MIN_INTERVAL_MINUTES", "5")),
            max_interval_minutes=int(os.getenv("MAX_INTERVAL_MINUTES", "30")),
            search_query=os.getenv("SEARCH_QUERY", "#python"),
            own_posting_probability=float(os.getenv("OWN_POSTING_PROBABILITY", "0.3")),
            user_data_dir=os.getenv("USER_DATA_DIR", os.path.join(os.getcwd(), "user_data")),
            reply_only=cls._get_bool(os.getenv("REPLY_ONLY"), default=False),
            # Enrichment toggles and limits
            url_enrichment=cls._get_bool(os.getenv("URL_ENRICHMENT", "true"), default=True),
            web_search_enrichment=cls._get_bool(os.getenv("WEB_SEARCH_ENRICHMENT", "true"), default=True),
            enrichment_max_chars=int(os.getenv("ENRICHMENT_MAX_CHARS", "900")),
            web_search_results=int(os.getenv("WEB_SEARCH_RESULTS", "3")),
            url_fetch_timeout=int(os.getenv("URL_FETCH_TIMEOUT", "8")),
            url_summary_fetch_limit=int(os.getenv("URL_SUMMARY_FETCH_LIMIT", "2")),
            web_search_engine_url=os.getenv("WEB_SEARCH_ENGINE_URL", "https://duckduckgo.com/html/"),
            web_search_user_agent=os.getenv("WEB_SEARCH_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            trending_quotes_enabled=cls._get_bool(os.getenv("TRENDING_QUOTES_ENABLED", "true"), default=True),
            operating_start_hour=int(os.getenv("HOURS_START", "5")),
            operating_end_hour=int(os.getenv("HOURS_END", "23")),
            # Bluesky / AT Protocol
            use_bsky=cls._get_bool(os.getenv("USE_BSKY", "false"), default=False),
            bsky_service_url=os.getenv("BSKY_SERVICE_URL", "https://bsky.social"),
            bsky_handle=os.getenv("BSKY_HANDLE"),
            bsky_app_password=os.getenv("BSKY_APP_PASSWORD"),
            # Bluesky selection controls
            bsky_author_cooldown_minutes=int(os.getenv("BSKY_AUTHOR_COOLDOWN_MINUTES", "120")),
            bsky_candidate_pool_size=int(os.getenv("BSKY_CANDIDATE_POOL_SIZE", "16")),
            bsky_replied_log_max=int(os.getenv("BSKY_REPLIED_LOG_MAX", "1000")),
            bsky_min_text_len=int(os.getenv("BSKY_MIN_TEXT_LEN", "10")),
            bsky_reply_max_chars=int(os.getenv("BSKY_REPLY_MAX_CHARS", "360")),
            bsky_post_max_chars=int(os.getenv("BSKY_POST_MAX_CHARS", "300")),
            # Daily Bluesky news post
            daily_post_enabled=cls._get_bool(os.getenv("DAILY_POST_ENABLED", "true"), default=True),
            daily_post_hour=int(os.getenv("DAILY_POST_HOUR", "14")),
            daily_post_window_minutes=int(os.getenv("DAILY_POST_WINDOW_MINUTES", "45")),
            daily_posts_per_day=int(os.getenv("DAILY_POSTS_PER_DAY", "2")),
            daily_post_min_interval_hours=int(os.getenv("DAILY_POST_MIN_INTERVAL_HOURS", "4")),
            daily_post_max_interval_hours=int(os.getenv("DAILY_POST_MAX_INTERVAL_HOURS", "8")),
            daily_post_first_hour=int(os.getenv("DAILY_POST_FIRST_HOUR", "8")),
            # LLM temps
            llm_reply_temperature=float(os.getenv("LLM_REPLY_TEMPERATURE", "0.7")),
            llm_post_temperature=float(os.getenv("LLM_POST_TEMPERATURE", "0.85")),
            llm_analyze_temperature=float(os.getenv("LLM_ANALYZE_TEMPERATURE", "0.2")),
            llm_summarize_temperature=float(os.getenv("LLM_SUMMARIZE_TEMPERATURE", "0.3")),
            llm_search_temperature=float(os.getenv("LLM_SEARCH_TEMPERATURE", "0.3")),
        )

    def validate(self) -> None:
        # Credentials validation depends on mode
        if self.use_bsky:
            # Bluesky requires handle and app password
            if not self.bsky_handle or not self.bsky_app_password:
                raise ValueError("Missing Bluesky credentials: set BSKY_HANDLE and BSKY_APP_PASSWORD (create an app password in Bluesky settings).")
        
        if self.min_interval_minutes < 1 or self.max_interval_minutes < self.min_interval_minutes:
            raise ValueError("Invalid interval minutes. Ensure 1 <= MIN <= MAX.")
        if not (0 <= self.own_posting_probability <= 1):
            raise ValueError("OWN_POSTING_PROBABILITY must be between 0 and 1.")


