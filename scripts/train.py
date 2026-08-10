#!/usr/bin/env python3
"""Unified entrypoint for single-run model training."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _usage() -> str:
    return (
        "Usage: python scripts/train.py METHOD [METHOD OPTIONS]\n"
        "\n"
        "METHOD: dspr | spt | lora\n"
        "Use 'python scripts/train.py METHOD --help' for method-specific options."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return 0

    method = sys.argv[1].lower()
    if method == "dspr":
        from experiment_pipeline.train_dspr import main as method_main
    elif method == "spt":
        from experiment_pipeline.train_spt import main as method_main
    elif method == "lora":
        from experiment_pipeline.train_lora import main as method_main
    else:
        print(f"Unknown training method: {method}\n\n{_usage()}", file=sys.stderr)
        return 2

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    result = method_main()
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
