"""LLM API client with retry logic and error handling."""

import time
import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
import litellm
from ..config import LLMConfig
from .prompts import build_prompt
from .parsing import parse_dag_response

logger = logging.getLogger(__name__)


class RateLimiter:
    """Reserve provider calls across all threads in one normal-mode run."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.limit = requests_per_minute
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.limit:
                    self.calls.append(now)
                    return
                delay = 60 - (now - self.calls[0])
            time.sleep(delay)


class LLMClient:
    """Wrapper for LiteLLM API calls with retry logic."""

    def __init__(self, config: LLMConfig, rate_limiter: RateLimiter | None = None):
        if not config.api_key:
            raise ValueError("API key is required for normal-mode LLM requests")
        self.config = config
        self.rate_limiter = rate_limiter or RateLimiter(config.requests_per_minute)

        # Build full model name with provider prefix for LiteLLM
        if '/' not in config.model:
            self.model_name = f"{config.provider}/{config.model}"
        else:
            self.model_name = config.model

    def _rate_limit(self):
        self.rate_limiter.acquire()

    def _parse_json_response(self, response_text: str) -> Optional[List[Dict]]:
        """Extract and parse JSON from LLM response.

        Handles cases where LLM wraps JSON in markdown code blocks.
        """
        try:
            return parse_dag_response(response_text)
        except ValueError as e:
            logger.error("Invalid DAG response: %s", e)
            return None

    def analyze_reasoning_chain(
        self,
        problem: str,
        steps: List[Dict]
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """Analyze a reasoning chain and return dependency DAG.

        Args:
            problem: Problem statement
            steps: List of reasoning steps

        Returns:
            Tuple of (parsed_dag, error_message)
            - parsed_dag: List of dependency objects if successful
            - error_message: Error description if failed
        """
        system_prompt, user_prompt = build_prompt(problem, steps)

        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()

                response = litellm.completion(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )

                response_text = response.choices[0].message.content
                try:
                    return parse_dag_response(response_text, steps), None
                except ValueError as exc:
                    return None, str(exc)

            except litellm.exceptions.RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt+1}): {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))

            except litellm.exceptions.APIError as e:
                logger.error(f"API error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    return None, f"API error after {self.config.max_retries} attempts: {str(e)}"

            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt+1}): {e}")
                return None, f"Unexpected error: {str(e)}"

        return None, f"Failed after {self.config.max_retries} attempts"
