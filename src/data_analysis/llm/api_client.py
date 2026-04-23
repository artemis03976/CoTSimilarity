"""LLM API client with retry logic and error handling."""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
import litellm
from .config import LLMConfig
from .prompt_template import build_prompt

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for LiteLLM API calls with retry logic."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.request_count = 0
        self.last_request_time = time.time()

        # Build full model name with provider prefix for LiteLLM
        if '/' not in config.model:
            self.model_name = f"{config.provider}/{config.model}"
        else:
            self.model_name = config.model

    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < 60:
            if self.request_count >= self.config.requests_per_minute:
                sleep_time = 60 - elapsed
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            self.request_count = 0
            self.last_request_time = current_time

    def _parse_json_response(self, response_text: str) -> Optional[List[Dict]]:
        """Extract and parse JSON from LLM response.

        Handles cases where LLM wraps JSON in markdown code blocks.
        """
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            else:
                logger.error(f"Response is not a list: {type(result)}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nResponse: {text[:200]}")
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

                self.request_count += 1

                response_text = response.choices[0].message.content
                parsed_dag = self._parse_json_response(response_text)

                if parsed_dag is None:
                    return None, "Failed to parse JSON response"

                # Validate response structure
                if len(parsed_dag) != len(steps):
                    return None, f"Expected {len(steps)} steps, got {len(parsed_dag)}"

                # Validate macro_action_tag field
                VALID_TAGS = {"Define", "Recall", "Derive", "Calculate", "Verify", "Conclude"}
                for i, step_dag in enumerate(parsed_dag):
                    if "macro_action_tag" not in step_dag:
                        return None, f"Step {i+1} missing 'macro_action_tag' field"

                    tag = step_dag["macro_action_tag"]
                    if tag not in VALID_TAGS:
                        return None, f"Step {i+1} has invalid tag '{tag}'. Must be one of {VALID_TAGS}"

                return parsed_dag, None

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
