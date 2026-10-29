"""Offline agreement statistics on fixed, uncompressed reasoning steps.

There is no reference annotator. Empty edge sets are undefined when both are
empty; one-sided empty sets score zero. Confidence intervals resample problems,
retaining all their CoTs and their original CoT-level weights.
"""

from collections import defaultdict

import numpy as np


ROLE_ORDER = ("Define", "Recall", "Derive", "Calculate", "Verify", "Conclude")
EDGE_KINDS = ("all", "internal", "root", "external")


def edge_sets(dag):
    edges = {kind: set() for kind in EDGE_KINDS}
    for node in dag:
        for source in node["depends_on"]:
            edge = (source, node["step_id"])
            kind = "external" if source == "External" else "root" if source == 0 else "internal"
            edges[kind].add(edge)
            edges["all"].add(edge)
    return edges


def edge_agreement(left, right):
    denominator = len(left) + len(right)
    return 2 * len(left & right) / denominator if denominator else None


def compare_dags(left, right):
    tags_a = {n["step_id"]: n["macro_action_tag"] for n in left}
    tags_b = {n["step_id"]: n["macro_action_tag"] for n in right}
    if not tags_a or tags_a.keys() != tags_b.keys():
        raise ValueError("Agreement requires identical, nonempty step IDs")
    edges_a, edges_b = edge_sets(left), edge_sets(right)
    result = {"n_steps": len(tags_a), "role_agreement": sum(
        tags_a[k] == tags_b[k] for k in tags_a) / len(tags_a)}
    for kind in EDGE_KINDS:
        a, b = edges_a[kind], edges_b[kind]
        result.update({f"edge_{kind}_f1": edge_agreement(a, b),
                       f"edge_{kind}_a": len(a), f"edge_{kind}_b": len(b),
                       f"edge_{kind}_intersection": len(a & b),
                       f"edge_{kind}_both_empty": not a and not b,
                       f"edge_{kind}_one_empty": bool(a) != bool(b)})
    return result


def summarize_clustered(rows, metrics, *, bootstrap_samples=10000, seed=42):
    """CoT/pair macro means with percentile 95% problem-cluster bootstrap CIs.

    Missing metrics do not become zero or one. All rows from a sampled problem
    recur together, including rows with undefined metrics. Resampling is shared
    across metric columns. Quantiles describe individual CoTs/pairs, not clusters.
    """
    problem_ids = sorted({r["problem_id"] for r in rows})
    positions = {pid: i for i, pid in enumerate(problem_ids)}
    sums = np.zeros((len(problem_ids), len(metrics)))
    counts = np.zeros_like(sums)
    values = defaultdict(list)
    for row in rows:
        i = positions[row["problem_id"]]
        for j, metric in enumerate(metrics):
            value = row.get(metric)
            if value is not None and np.isfinite(value):
                sums[i, j] += value
                counts[i, j] += 1
                values[metric].append(value)
    estimates = np.full((bootstrap_samples, len(metrics)), np.nan)
    if problem_ids:
        rng = np.random.default_rng(seed)
        # Multinomial weights are exactly a bootstrap sample of problem IDs.
        for start in range(0, bootstrap_samples, 250):
            weights = rng.multinomial(len(problem_ids),
                                      np.full(len(problem_ids), 1 / len(problem_ids)),
                                      size=min(250, bootstrap_samples - start))
            denominator = weights @ counts
            np.divide(weights @ sums, denominator,
                      out=estimates[start:start + len(weights)], where=denominator > 0)
    results = {}
    for j, metric in enumerate(metrics):
        vals = np.asarray(values[metric], dtype=float)
        boot = estimates[:, j]
        boot = boot[np.isfinite(boot)]
        results[metric] = {
            "n": len(vals), "n_problems": int(np.count_nonzero(counts[:, j])),
            "n_undefined": len(rows) - len(vals),
            "mean": float(vals.mean()) if len(vals) else None,
            "ci_low": float(np.quantile(boot, .025)) if len(boot) else None,
            "ci_high": float(np.quantile(boot, .975)) if len(boot) else None,
            "median": float(np.median(vals)) if len(vals) else None,
            "p90": float(np.quantile(vals, .9)) if len(vals) else None,
            "p95": float(np.quantile(vals, .95)) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
            "bootstrap_defined": len(boot),
        }
    return results


def confusion_rows(left, right):
    """Return label pairs, preserving the fixed step correspondence."""
    tags_b = {n["step_id"]: n["macro_action_tag"] for n in right}
    if {n["step_id"] for n in left} != tags_b.keys():
        raise ValueError("Confusion requires identical step IDs")
    return [(n["macro_action_tag"], tags_b[n["step_id"]]) for n in left]
