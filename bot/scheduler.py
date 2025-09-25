import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque

from loguru import logger


@dataclass
class RateLimiter:
    """Tracks action timestamps to enforce max actions per rolling hour."""

    max_actions_per_hour: int
    _timestamps: Deque[float]

    def __init__(self, max_actions_per_hour: int) -> None:
        self.max_actions_per_hour = max_actions_per_hour
        self._timestamps = deque()

    def _prune(self) -> None:
        cutoff = time.time() - 3600
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def allow_now(self) -> bool:
        self._prune()
        return len(self._timestamps) < self.max_actions_per_hour

    def record(self) -> None:
        self._timestamps.append(time.time())

    def seconds_until_next_allowed(self) -> float:
        self._prune()
        if len(self._timestamps) < self.max_actions_per_hour:
            return 0.0
        oldest = self._timestamps[0]
        return max(0.0, oldest + 3600 - time.time())


class IntervalScheduler:
    """Generates randomized delays between actions within configured bounds."""

    def __init__(self, min_minutes: int, max_minutes: int) -> None:
        self._min_s = max(60, int(min_minutes * 60))
        self._max_s = max(self._min_s, int(max_minutes * 60))

    def next_delay_seconds(self) -> int:
        delay = random.randint(self._min_s, self._max_s)
        logger.info("Next action in {} minutes ({} seconds)", round(delay / 60, 2), delay)
        return delay


def run_loop(
    do_action: Callable[[], None],
    rate_limiter: RateLimiter,
    scheduler: IntervalScheduler,
    run_immediately: bool = False,
) -> None:
    """Runs an infinite loop that executes do_action under interval and rate limits.

    If run_immediately is True, attempts one action right away (respecting the rate cap),
    then continues with the normal interval loop.
    """
    if run_immediately:
        if not rate_limiter.allow_now():
            extra = rate_limiter.seconds_until_next_allowed()
            logger.warning(
                "Hit hourly cap. Skipping immediate run; next allowed in {} minutes.", round(extra / 60, 2)
            )
        else:
            try:
                do_action()
                rate_limiter.record()
            except Exception as exc:
                logger.exception("Immediate action failed: {}", exc)

    while True:
        wait_for = scheduler.next_delay_seconds()
        time.sleep(wait_for)

        if not rate_limiter.allow_now():
            extra = rate_limiter.seconds_until_next_allowed()
            logger.warning(
                "Hit hourly cap. Waiting an extra {} minutes before next action.", round(extra / 60, 2)
            )
            time.sleep(extra)

        try:
            do_action()
            rate_limiter.record()
        except Exception as exc:
            logger.exception("Action failed: {}", exc)


