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
        "METHOD: dspr | lora\n"
        "Use 'python scripts/train_kfold.py METHOD --help' for method-specific options."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    method = sys.argv[1].lower()
    if method == "dspr":
        from experiment_pipeline.kfold_dspr import main as method_main
    elif method == "lora":
        from experiment_pipeline.kfold_lora import main as method_main
    else:
        print(f"Unknown K-fold method: {method}\n\n{_usage()}", file=sys.stderr)
        return 2

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    return method_main()


if __name__ == "__main__":
    raise SystemExit(main())
