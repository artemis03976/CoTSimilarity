"""OOF slice metrics and paired comparisons; no file writes or model dependencies."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Callable, Iterable
from typing import Any

from utils.scoring import require_scored
from .common import VARIANTS


def extract_flags(record: dict[str, Any], variant: str, source: str) -> list[bool]:
    data = record.get(variant)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: problem {record.get('problem_id')} lacks {variant}")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{source}: problem {record.get('problem_id')} {variant} has no samples")
    flags = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(
                f"{source}: problem {record.get('problem_id')} {variant} has invalid correctness"
            )
        flags.append(require_scored(sample, f"{source}: problem {record.get('problem_id')}/{variant}"))
    return flags


def extract_alpha_values(record: dict[str, Any], variant: str, source: str) -> list[float]:
    data = record.get(variant)
    if not isinstance(data, dict) or "alpha" not in data:
        return []
    raw = data["alpha"]
    values = raw if isinstance(raw, list) else [raw]
    result = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: problem {record.get('problem_id')} {variant} has invalid alpha {value!r}"
            ) from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(
                f"{source}: problem {record.get('problem_id')} {variant} alpha outside [0,1]: {numeric}"
            )
        result.append(numeric)
    return result


def metric_for_variant(
    records: dict[int, dict[str, Any]], ids: Iterable[int], variant: str, source: str
) -> dict[str, Any]:
    selected = sorted(set(ids))
    first_correct = 0
    any_correct = 0
    sample_correct = 0
    sample_total = 0
    for pid in selected:
        flags = extract_flags(records[pid], variant, source)
        first_correct += int(flags[0])
        any_correct += int(any(flags))
        sample_correct += sum(flags)
        sample_total += len(flags)
    total = len(selected)
    return {
        "total": total,
        "first_correct": first_correct,
        "first_accuracy": first_correct / total if total else None,
        "any_correct": any_correct,
        "any_accuracy": any_correct / total if total else None,
        "sample_correct": sample_correct,
        "sample_total": sample_total,
        "sample_accuracy": sample_correct / sample_total if sample_total else None,
    }


def all_variants_metric(
    records: dict[int, dict[str, Any]], ids: Iterable[int], source: str
) -> dict[str, Any]:
    selected = sorted(set(ids))
    correct = sum(
        all(extract_flags(records[pid], variant, source)[0] for variant in VARIANTS)
        for pid in selected
    )
    total = len(selected)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "definition": "all three variants are correct at first@1 for the same problem",
    }


def alpha_summary(records: dict[int, dict[str, Any]], ids: Iterable[int], variant: str) -> dict[str, Any]:
    values: list[float] = []
    for pid in sorted(set(ids)):
        values.extend(extract_alpha_values(records[pid], variant, "DSPR OOF"))
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty list")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def exact_mcnemar_p(comparator_only: int, method_only: int) -> float:
    discordant = comparator_only + method_only
    if discordant == 0:
        return 1.0
    tail = min(comparator_only, method_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def comparison_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], byteorder="big", signed=False)


def paired_comparison(
    ids: Iterable[int],
    method_outcome: Callable[[int], bool],
    comparator_outcome: Callable[[int], bool],
    bootstrap_samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    selected = sorted(set(ids))
    if not selected:
        raise ValueError(f"Cannot compare an empty slice: {label}")
    method_values = [int(method_outcome(pid)) for pid in selected]
    comparator_values = [int(comparator_outcome(pid)) for pid in selected]
    differences = [new - base for new, base in zip(method_values, comparator_values)]
    delta = statistics.fmean(differences)

    rng = random.Random(comparison_seed(seed, label))
    bootstrapped = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(selected)) for _ in selected]
        bootstrapped.append(statistics.fmean(differences[index] for index in indices))
    bootstrapped.sort()

    comparator_only = sum(base == 1 and new == 0 for new, base in zip(method_values, comparator_values))
    method_only = sum(new == 1 and base == 0 for new, base in zip(method_values, comparator_values))
    both_correct = sum(new == 1 and base == 1 for new, base in zip(method_values, comparator_values))
    both_wrong = len(selected) - comparator_only - method_only - both_correct
    return {
        "total": len(selected),
        "method_correct": sum(method_values),
        "method_accuracy": statistics.fmean(method_values),
        "comparator_correct": sum(comparator_values),
        "comparator_accuracy": statistics.fmean(comparator_values),
        "delta": delta,
        "paired_bootstrap_95_ci": [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "comparator_only_correct": comparator_only,
        "method_only_correct": method_only,
        "mcnemar_exact_p": exact_mcnemar_p(comparator_only, method_only),
    }


def metric_slices(
    all_ids: set[int], eligible: dict[str, set[int]],
) -> dict[str, dict[str, set[int]]]:
    return {
        "full": {variant: all_ids for variant in (*VARIANTS, "all_variants")},
        "eligible": {variant: eligible[variant] for variant in ("simple", "hard")},
        "noneligible": {variant: all_ids - eligible[variant] for variant in ("simple", "hard")},
    }


def build_metrics(
    records: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
    method: str,
) -> dict[str, Any]:
    label = f"{method.upper()} OOF"
    return {
        scope: {
            variant: (
                all_variants_metric(records, ids, label)
                if variant == "all_variants"
                else metric_for_variant(records, ids, variant, label)
            )
            for variant, ids in variants.items()
        }
        for scope, variants in metric_slices(all_ids, eligible).items()
    }


def first_outcome(records: dict[int, dict[str, Any]], pid: int, variant: str) -> bool:
    variants = VARIANTS if variant == "all_variants" else (variant,)
    return all(extract_flags(records[pid], current, "paired comparison")[0] for current in variants)


def comparison_fields(method: str) -> dict[str, str]:
    """Retain existing report column names while sharing all calculations."""
    comparator = "baseline" if method == "dspr" else "comparator"
    return {
        "method_correct": f"{method}_correct",
        "method_accuracy": f"{method}_accuracy",
        "comparator_correct": f"{comparator}_correct",
        "comparator_accuracy": f"{comparator}_accuracy",
        "delta": "delta" if method == "dspr" else f"delta_{method}_minus_comparator",
        "comparator_only_correct": f"{comparator}_only_correct",
        "method_only_correct": f"{method}_only_correct",
    }


def compare_records(
    records: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
    all_ids: set[int],
    eligible: dict[str, set[int]],
    bootstrap_samples: int,
    seed: int,
    comparator_name: str,
    method: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = comparison_fields(method)
    # Preserve historical seed labels so refactoring does not change the CIs.
    prefix = "" if method == "dspr" else f"{comparator_name}/"
    for scope, variants in metric_slices(all_ids, eligible).items():
        result[scope] = {}
        for variant, ids in variants.items():
            raw = paired_comparison(
                ids,
                lambda pid, v=variant: first_outcome(records, pid, v),
                lambda pid, v=variant: first_outcome(comparator, pid, v),
                bootstrap_samples,
                seed,
                f"{prefix}{scope}/{variant}",
            )
            result[scope][variant] = {fields.get(key, key): value for key, value in raw.items()}
    return result
