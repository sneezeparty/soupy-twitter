from __future__ import annotations

from typing import Optional

from loguru import logger
from playwright.sync_api import Page

from .config import AppConfig


class CaptchaSolver:
    """Stub for captcha solving.

    Integrate with services like 2Captcha, Anti-Captcha, or hCaptcha/Turnstile solvers.
    Currently returns False to skip automated solving.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def solve_in_page(self, page: Page) -> bool:
        if not self._config.captcha_api_key:
            logger.warning("CAPTCHA_API_KEY not set; skipping captcha solving.")
            return False
        # TODO: Implement service-specific solving here using sitekey + page URL
        logger.info("Captcha solver not implemented yet.")
        return False


