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

    # Twitter/X credentials
    x_username: str
    x_password: str
    x_email: Optional[str]

    # X API mode (official API)
    use_x_api: bool
    x_api_tier: str  # e.g., "free", "basic", "pro", "enterprise"
    x_api_consumer_key: Optional[str]
    x_api_consumer_secret: Optional[str]
    x_api_access_token: Optional[str]
    x_api_access_token_secret: Optional[str]
    x_api_bearer_token: Optional[str]

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
            x_username=os.getenv("X_USERNAME", ""),
            x_password=os.getenv("X_PASSWORD", ""),
            x_email=os.getenv("X_EMAIL"),
            use_x_api=cls._get_bool(os.getenv("USE_X_API", "true"), default=True),
            x_api_tier=os.getenv("X_API_TIER", "free"),
            x_api_consumer_key=os.getenv("X_API_CONSUMER_KEY"),
            x_api_consumer_secret=os.getenv("X_API_CONSUMER_SECRET"),
            x_api_access_token=os.getenv("X_API_ACCESS_TOKEN"),
            x_api_access_token_secret=os.getenv("X_API_ACCESS_TOKEN_SECRET"),
            x_api_bearer_token=os.getenv("X_API_BEARER_TOKEN"),
            behaviour_prompt=os.getenv("BEHAVIOUR", "You are a helpful, friendly Twitter user."),
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
            bsky_post_max_chars=int(os.getenv("BSKY_POST_MAX_CHARS", "500")),
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
        elif self.use_x_api:
            missing = [
                k for k, v in {
                    "X_API_CONSUMER_KEY": self.x_api_consumer_key,
                    "X_API_CONSUMER_SECRET": self.x_api_consumer_secret,
                    "X_API_ACCESS_TOKEN": self.x_api_access_token,
                    "X_API_ACCESS_TOKEN_SECRET": self.x_api_access_token_secret,
                }.items() if not v
            ]
            if missing:
                raise ValueError(
                    "Missing X API credentials: " + ", ".join(missing) + 
                    ". Populate these in .env or set USE_X_API=false to use browser automation."
                )
        else:
            if not self.x_username or not self.x_password:
                raise ValueError("X_USERNAME and X_PASSWORD must be set in environment.")
        if self.min_interval_minutes < 1 or self.max_interval_minutes < self.min_interval_minutes:
            raise ValueError("Invalid interval minutes. Ensure 1 <= MIN <= MAX.")
        if not (0 <= self.own_posting_probability <= 1):
            raise ValueError("OWN_POSTING_PROBABILITY must be between 0 and 1.")


