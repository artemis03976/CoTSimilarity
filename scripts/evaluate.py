#!/usr/bin/env python3
"""Unified entrypoint for held-out evaluation and OOF aggregation."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _usage() -> str:
    return (
        "Usage: python scripts/evaluate.py METHOD ACTION [ACTION OPTIONS]\n"
        "\n"
        "METHOD: dspr | lora\n"
        "ACTION: run | aggregate\n"
        "Use 'python scripts/evaluate.py METHOD ACTION --help' for action-specific options."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0
    if len(sys.argv) < 3:
        print(f"Missing evaluation action.\n\n{_usage()}", file=sys.stderr)
        return 2

    method = sys.argv[1].lower()
    action = sys.argv[2].lower()
    if (method, action) == ("dspr", "run"):
        from experiment_pipeline.evaluate_dspr import main as action_main
    elif (method, action) == ("dspr", "aggregate"):
        from experiment_pipeline.aggregate_dspr import main as action_main
    elif (method, action) == ("lora", "run"):
        from experiment_pipeline.evaluate_lora import main as action_main
    elif (method, action) == ("lora", "aggregate"):
        from experiment_pipeline.aggregate_lora import main as action_main
    else:
        print(f"Unsupported evaluation target: {method} {action}\n\n{_usage()}", file=sys.stderr)
        return 2

    sys.argv = [sys.argv[0], *sys.argv[3:]]
    return action_main()


if __name__ == "__main__":
    raise SystemExit(main())
