#!/usr/bin/env python3
"""Unified entrypoint for K-fold training orchestration."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _usage() -> str:
    return (
        "Usage: python scripts/train_kfold.py METHOD [METHOD OPTIONS]\n"
        "\n"
        "METHOD: dspr (router -> frozen-router prefix) | spt | lora\n"
        "Use 'python scripts/train_kfold.py METHOD --help' for method-specific options."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    method = sys.argv[1].lower()
    if method not in {"dspr", "spt", "lora"}:
        print(f"Unknown K-fold method: {method}\n\n{_usage()}", file=sys.stderr)
        return 2

    from experiment_pipeline.kfold import main as method_main

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    return method_main(method)


if __name__ == "__main__":
    raise SystemExit(main())
