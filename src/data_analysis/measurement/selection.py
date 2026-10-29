"""Agreement statistics for fixed candidate pools, with no reference annotator."""

from itertools import combinations
from math import sqrt

import numpy as np


def kendall_tau_b(left, right):
    """Tie-aware Kendall correlation; insufficient/constant rankings are undefined."""
    if len(left) != len(right):
        raise ValueError("Rankings must describe the same candidates")
    concordant = discordant = tied_left = tied_right = tied_both = 0
    for i, j in combinations(range(len(left)), 2):
        a, b = left[i] - left[j], right[i] - right[j]
        if a == b == 0:
            tied_both += 1
        elif a == 0:
            tied_left += 1
        elif b == 0:
            tied_right += 1
        elif a * b > 0:
            concordant += 1
        else:
            discordant += 1
    comparable = concordant + discordant
    denominator = sqrt((comparable + tied_left) * (comparable + tied_right))
    return {
        "kendall_tau_b": (concordant - discordant) / denominator if denominator else None,
        "concordant": concordant, "discordant": discordant,
        "tied_left_only": tied_left, "tied_right_only": tied_right,
        "tied_both": tied_both,
    }


def set_agreement(left, right):
    left, right = set(left), set(right)
    intersection, union = len(left & right), len(left | right)
    return {"intersection": intersection, "union": union,
            "size_a": len(left), "size_b": len(right),
            "jaccard": intersection / union if union else None,
            "dice": 2 * intersection / (len(left) + len(right)) if union else None}


def summarize_ratio_clustered(rows, numerator, denominator, *, bootstrap_samples=10000, seed=42):
    """Pooled set overlap CI, resampling whole problems rather than individual paths."""
    ids = sorted({r["problem_id"] for r in rows})
    index = {pid: i for i, pid in enumerate(ids)}
    values = np.zeros((len(ids), 2))
    for row in rows:
        values[index[row["problem_id"]]] += [row[numerator], row[denominator]]
    total = values.sum(axis=0)
    estimates = []
    if ids:
        rng = np.random.default_rng(seed)
        for start in range(0, bootstrap_samples, 250):
            weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)),
                                      size=min(250, bootstrap_samples - start))
            sums = weights @ values
            valid = sums[:, 1] > 0
            estimates.extend((sums[valid, 0] / sums[valid, 1]).tolist())
    return {"numerator": int(total[0]), "denominator": int(total[1]),
            "mean": float(total[0] / total[1]) if total[1] else None,
            "ci_low": float(np.quantile(estimates, .025)) if estimates else None,
            "ci_high": float(np.quantile(estimates, .975)) if estimates else None,
            "n_problems": len(ids), "bootstrap_defined": len(estimates)}
