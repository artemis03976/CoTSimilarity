"""LLM API client with retry logic and error handling."""

import time
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import httpx
from ..config import LLMConfig
from .prompts import build_prompt
from .parsing import parse_dag_response
from .protocols import request_body, response_content

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
    """Native OpenAI-compatible and Anthropic Messages calls with shared retries."""

    def __init__(self, config: LLMConfig, rate_limiter: RateLimiter | None = None):
        if not config.api_key:
            raise ValueError("API key is required for normal-mode LLM requests")
        self.config = config
        self.rate_limiter = rate_limiter or RateLimiter(config.requests_per_minute)

        if not config.base_url:
            raise ValueError(f"A base URL is required for provider {config.provider}")
        self._raw_response_records: List[Dict] = []

    def _request(self, body: dict) -> dict:
        base = self.config.base_url.rstrip("/")
        if self.config.provider == "anthropic":
            if not base.endswith("/v1"):
                base += "/v1"
            url = base + "/messages"
            headers = {"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"}
        else:
            url = base + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
        response = httpx.post(url, headers=headers, json=body, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()

    def _rate_limit(self):
        self.rate_limiter.acquire()

    def take_raw_response_records(self) -> List[Dict]:
        """Return and clear raw records collected by the last API calls.

        Normal mode calls this after each outer retry so every provider attempt
        is written exactly once, while the public analysis return value stays
        compatible with the batch and legacy callers.
        """
        records = self._raw_response_records
        self._raw_response_records = []
        return records

    def _record_raw_response(
        self,
        attempt: int,
        response=None,
        *,
        parse_error: Optional[str] = None,
        provider_error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Keep only JSON-serializable response content and safe error text."""
        if response is not None and not isinstance(response, (str, int, float, bool, list, dict)):
            response = str(response)
        api_key = self.config.api_key or ""

        def redact(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            text = str(value)
            return text.replace(api_key, "[REDACTED]") if api_key else text

        if isinstance(response, str):
            response = redact(response)

        self._raw_response_records.append({
            "api_attempt": attempt,
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "parse_error": redact(parse_error),
            "provider_error": redact(provider_error),
            **(metadata or {}),
        })

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
        body = request_body(self.config, system_prompt, user_prompt)

        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()

                response = self._request(body)
                response_text, metadata = None, {}
                try:
                    response_text, metadata = response_content(response)
                    dag = parse_dag_response(response_text, steps)
                    self._record_raw_response(attempt, response_text, metadata=metadata)
                    return dag, None
                except ValueError as exc:
                    self._record_raw_response(attempt, response_text or response, parse_error=str(exc), metadata=metadata)
                    return None, str(exc)

            except (httpx.HTTPError, ValueError) as e:
                detail = e.response.text[:2000] if isinstance(e, httpx.HTTPStatusError) else str(e)
                error = f"{e}: {detail}".replace(self.config.api_key, "[REDACTED]")
                self._record_raw_response(attempt, provider_error=error)
                logger.error("API error (attempt %s): %s", attempt + 1, error)
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return None, f"API error after {self.config.max_retries} attempts: {error}"

            except Exception as e:
                error = str(e).replace(self.config.api_key, "[REDACTED]")
                self._record_raw_response(attempt, provider_error=error)
                logger.error("Unexpected error (attempt %s): %s", attempt + 1, error)
                return None, f"Unexpected error: {error}"

        return None, f"Failed after {self.config.max_retries} attempts"
