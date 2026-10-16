"""Lightweight training progress protocol for workers and K-fold parents."""

from __future__ import annotations

import json
from typing import Any


PROGRESS_EVENT_PREFIX = "__DSPR_TRAIN_PROGRESS__"
TRAIN_LOG_PREFIX = "__DSPR_TRAIN_LOG__"


def parse_progress_event(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(PROGRESS_EVENT_PREFIX):
        return None
    try:
        payload = json.loads(stripped[len(PROGRESS_EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
