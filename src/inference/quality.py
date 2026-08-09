"""Mechanical quality checks for generated chain-of-thought trajectories."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable

import numpy as np


SPECIAL_TOKEN_STRINGS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<unk>",
    "<s>",
    "</s>",
)


@dataclass(frozen=True)
class CoTValidationConfig:
    """Thresholds for hard, model-independent trajectory validation."""

    require_boxed_answer: bool = True
    min_characters: int = 20
    # Maximum period, in tokens, considered for a *consecutive* loop.  This
    # must not be interpreted as a global frequency threshold: normal math
    # expressions may recur at distant positions in a long solution.
    repeat_ngram_size: int = 64
    max_ngram_repeats: int = 5
    # Short repeats are common in valid arithmetic and LaTeX.  Only reject a
    # periodic decoding loop once its contiguous span is substantial.
    min_repeat_span_tokens: int = 64

    def __post_init__(self) -> None:
        if self.min_characters < 1:
            raise ValueError("min_characters must be positive")
        if self.repeat_ngram_size < 1:
            raise ValueError("repeat_ngram_size must be positive")
        if self.max_ngram_repeats < 2:
            raise ValueError("max_ngram_repeats must be at least 2")
        if self.min_repeat_span_tokens < 1:
            raise ValueError("min_repeat_span_tokens must be positive")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...]


def _has_complete_boxed_answer(text: str) -> bool:
    marker = r"\boxed{"
    search_from = 0
    while True:
        start = text.find(marker, search_from)
        if start < 0:
            return False
        payload_start = start + len(marker)
        depth = 1
        escaped = False
        for offset, char in enumerate(text[payload_start:], start=payload_start):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    if text[payload_start:offset].strip():
                        return True
                    search_from = offset + 1
                    break
        else:
            search_from = start + len(marker)


def _contains_disallowed_control(text: str) -> bool:
    allowed = {"\n", "\r", "\t"}
    return any(
        char not in allowed and unicodedata.category(char) in {"Cc", "Cs"}
        for char in text
    )


def _has_extreme_ngram_repetition(
    token_ids: Iterable[int],
    max_period: int,
    min_repeats: int,
    min_span: int,
) -> bool:
    """Detect a local periodic decoding loop, not distant phrase reuse.

    A loop with period ``p`` repeated ``r`` times has
    ``tokens[i] == tokens[i-p]`` for ``p * (r - 1)`` consecutive positions.
    The total contiguous loop span must also reach ``min_span``; otherwise
    normal math expressions such as repeated multiplication signs or zeros
    in an integer would be rejected.
    Checking periods up to ``max_period`` catches both single-token loops and
    repeated multi-step fragments while allowing the same phrase to recur in
    separate parts of a normal chain of thought.
    """

    tokens = np.asarray(tuple(int(token) for token in token_ids), dtype=np.int64)
    if tokens.size < min_repeats or max_period < 1 or min_span < 1:
        return False
    max_period = min(max_period, int(tokens.size) // min_repeats)
    for period in range(1, max_period + 1):
        required_matches = period * (min_repeats - 1)
        matches = tokens[period:] == tokens[:-period]
        if matches.size < required_matches:
            continue
        # Find contiguous True-run lengths without a Python loop over every
        # token.  This keeps validation cheap for tens of thousands of paths.
        padded = np.empty(matches.size + 2, dtype=np.bool_)
        padded[0] = False
        padded[-1] = False
        padded[1:-1] = matches
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        if transitions.size:
            run_lengths = transitions[1::2] - transitions[::2]
            if np.any(run_lengths >= required_matches):
                # Recover the start/end of each qualifying True run.  The
                # vectorised run-length check above is retained for speed;
                # this small number of candidate runs is cheap to inspect.
                starts = transitions[::2]
                ends = transitions[1::2]
                for start, end in zip(starts, ends):
                    if (
                        end - start >= required_matches
                        and end - start + period >= min_span
                    ):
                        return True
    return False


def validate_cot(
    text: str,
    finish_reason: str | None,
    token_ids: Iterable[int] = (),
    config: CoTValidationConfig | None = None,
) -> ValidationResult:
    """Validate structural quality without using answer correctness as a filter."""

    config = config or CoTValidationConfig()
    reasons: list[str] = []
    stripped = text.strip()
    if finish_reason != "stop":
        reasons.append(f"finish_reason:{finish_reason or 'missing'}")
    if not stripped:
        reasons.append("empty_response")
    elif len(stripped) < config.min_characters:
        reasons.append("response_too_short")
    if "\ufffd" in text:
        reasons.append("replacement_character")
    if _contains_disallowed_control(text):
        reasons.append("control_character")
    if any(token in text for token in SPECIAL_TOKEN_STRINGS):
        reasons.append("special_token_leak")
    if config.require_boxed_answer and not _has_complete_boxed_answer(text):
        reasons.append("missing_or_unbalanced_boxed_answer")
    if _has_extreme_ngram_repetition(
        token_ids,
        config.repeat_ngram_size,
        config.max_ngram_repeats,
        config.min_repeat_span_tokens,
    ):
        reasons.append("extreme_ngram_repetition")
    return ValidationResult(valid=not reasons, reasons=tuple(reasons))
