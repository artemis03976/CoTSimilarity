"""Lightweight progress events shared by inference workers and K-fold parents."""

from __future__ import annotations

import json


PROGRESS_PREFIX = "__INFERENCE_PROGRESS__"


def emit_progress(stage: str, **details) -> None:
    print(PROGRESS_PREFIX + json.dumps({"stage": stage, **details}), flush=True)


def parse_progress(line: str) -> dict | None:
    if not line.startswith(PROGRESS_PREFIX):
        return None
    return json.loads(line[len(PROGRESS_PREFIX):])
