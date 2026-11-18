"""Lightweight training progress protocol for workers and K-fold parents."""

from __future__ import annotations

import json
from typing import Any


PROGRESS_EVENT_PREFIX = "__DSPR_TRAIN_PROGRESS__"
TRAIN_LOG_PREFIX = "__DSPR_TRAIN_LOG__"


def parse_progress_event(line: str) -> dict[str, Any] | None:
    # A carriage-return progress display on merged stderr/stdout can precede
    # the event on the same line. The marker, rather than column zero, frames it.
    _, marker, payload_text = line.partition(PROGRESS_EVENT_PREFIX)
    if not marker:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
