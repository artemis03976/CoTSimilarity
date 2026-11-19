"""Match learned DSPR routing values to the two perturbations of each held-out group."""

from .common import ground_truth_for, ground_truth_type_for, normalise_problem_id, read_math_records


PARTNER = {"simple": "hard", "hard": "simple"}


def load_native_alphas(records, path, expected_ids=None):
    reference = {int(row["problem_id"]): row for row in read_math_records(path)}
    if expected_ids is not None and set(reference) != set(expected_ids):
        raise ValueError(f"Learned routing IDs differ from the held-out fold: {path}")
    return native_alphas(records, reference)


def native_alphas(records, reference):
    """Return the original, unsorted routing values, indexed by group and variant."""
    result = {}
    for row in records:
        pid = normalise_problem_id(row["problem_id"])
        pair = {}
        for variant in PARTNER:
            source = reference[pid][variant]
            target = row[variant]
            if (source["problem"] != target["problem"]
                    or ground_truth_for(source) != ground_truth_for(target)
                    or ground_truth_type_for(source, variant) != ground_truth_type_for(target, variant)):
                raise ValueError(f"Learned reference differs for {pid}/{variant}")
            alpha = source["alpha"]
            if isinstance(alpha, list) and len(alpha) == 1:
                alpha = alpha[0]
            if type(alpha) not in (int, float) or not 0 <= alpha <= 1:
                raise ValueError(f"Invalid learned alpha for {pid}/{variant}: {alpha!r}")
            pair[variant] = float(alpha)
        result[pid] = pair
    return result


def check_swapped_pair(record, native):
    for variant, partner in PARTNER.items():
        entry = record[variant]
        if entry.get("native_alpha") != native[variant] or entry.get("alpha") != native[partner]:
            raise ValueError(f"Swapped alpha mismatch for {record['problem_id']}/{variant}")
