"""Shared experiment defaults, paths, and frozen fold identifiers."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.io import read_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DEFAULT_FOLDS = tuple(range(5))
DEFAULT_DSPR_BUDGET = 337_025
DEFAULT_RANK = 1
DEFAULT_LORA_ALPHA = 2
DEFAULT_CHECKPOINT_ROOTS = {
    "dspr": "checkpoints/qwen_staged_kfold_seed42",
    "spt": "checkpoints/qwen_spt_seed42",
    "lora": "checkpoints/qwen_lora_qv_r1_seed42",
}
DEFAULT_RESULT_ROOTS = {
    method: f"output/{Path(path).name}" for method, path in DEFAULT_CHECKPOINT_ROOTS.items()
}
VARIANTS = ("original", "simple", "hard")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_fold_list(values: list[int] | None) -> list[int]:
    folds = list(DEFAULT_FOLDS) if values is None else values
    if not folds:
        raise ValueError("At least one fold must be selected")
    if len(set(folds)) != len(folds):
        raise ValueError(f"Duplicate fold IDs: {folds}")
    invalid = [fold for fold in folds if fold < 0 or fold >= 5]
    if invalid:
        raise ValueError(f"Fold IDs must be in [0, 4], got {invalid}")
    return folds


def normalise_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Problem ID must be an integer, got {value!r}") from exc


def held_out_id_path(id_root: Path, fold: int) -> Path:
    path = id_root / f"fold_{fold}" / "test_ids.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold test IDs: {path}")
    return path


def expected_ids_for_fold(id_root: Path, fold: int) -> list[int]:
    path = held_out_id_path(id_root, fold)
    payload = read_json(path)
    ids = [normalise_id(value) for value in payload.get("all", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid or duplicate held-out IDs in {path}")
    return sorted(ids)


def output_is_nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))
