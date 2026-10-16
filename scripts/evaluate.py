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
        "METHOD: dspr | spt | lora\n"
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
    if method not in {"dspr", "spt", "lora"} or action not in {"run", "aggregate"}:
        print(f"Unsupported evaluation target: {method} {action}\n\n{_usage()}", file=sys.stderr)
        return 2

    if action == "run":
        from experiment_pipeline.evaluate import main as action_main
    else:
        from experiment_pipeline.aggregate import main as action_main

    sys.argv = [sys.argv[0], *sys.argv[3:]]
    return action_main(method)


if __name__ == "__main__":
    raise SystemExit(main())
