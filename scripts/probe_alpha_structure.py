"""Measure fixed-alpha OOF trajectories against the same base Original graph."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_analysis.annotation.parsing import validate_dag
from data_analysis.cot_segmenter import process_record
from data_analysis.records import iter_samples
from utils.artifacts import response_hash
from utils.io import read_json, read_jsonl, write_json, write_jsonl

ALPHAS = (0.0, 0.5, 1.0)
RUNS = ("alpha_0", "alpha_0p5", "alpha_1")
VARIANTS = ("simple", "hard")
SEED = 42
N_GROUPS = 30
MODEL = "ep-20261001103240-d9m5m"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def cohort(dataset):
    sources = defaultdict(list)
    for r in read_jsonl(dataset):
        sources[r["source"]].append(int(r["problem_id"]))
    n = sum(map(len, sources.values()))
    quotas = {s: len(ids) * N_GROUPS // n for s, ids in sources.items()}
    by_remainder = sorted(sources, key=lambda s: (-(len(sources[s])*N_GROUPS % n), s))
    for s in by_remainder[:N_GROUPS - sum(quotas.values())]:
        quotas[s] += 1
    rng = random.Random(SEED)
    return {i: s for s in sorted(sources) for i in rng.sample(sorted(sources[s]), quotas[s])}


def subset(path, selected):
    rows = [r for r in read_jsonl(path) if int(r["problem_id"]) in selected]
    records = {int(r["problem_id"]): r for r in rows}
    if set(records) != set(selected) or len(records) != len(rows):
        raise ValueError(f"Incomplete or duplicate selected problem IDs in {path}")
    return records


def prepare(args):
    out = args.output_dir
    selected = cohort(args.dataset)
    fixed = [subset(args.fixed_root / run / "oof_all_records.jsonl", selected) for run in RUNS]
    base = subset(args.base_root / "all_records.jsonl", selected)
    base_dags = args.base_root / "dag_analysis/deepseek/analyzed_records.jsonl"
    originals = subset(base_dags, selected)
    checkpoints = {}
    for pid in sorted(selected):
        fold = fixed[0][pid]["outer_fold"]
        if fold in checkpoints:
            continue
        paths = []
        for alpha, run in zip(ALPHAS, RUNS):
            status = read_json(args.fixed_root / run / f"fold_{fold}/run_status.json")
            if status["status"] != "succeeded" or status["forced_alpha"] != alpha:
                raise ValueError(f"Incorrect fixed-alpha run: {run}/fold_{fold}")
            paths.append(status["checkpoint"])
        if len(set(paths)) != 1:
            raise ValueError(f"Different checkpoints for fold {fold}")
        learned = read_json(args.reference_root / f"fold_{fold}/run_status.json")
        if (learned["status"] != "succeeded" or learned.get("forced_alpha") is not None
                or learned["checkpoint"] != paths[0]):
            raise ValueError(f"Fixed controls do not match the dynamic reference for fold {fold}")
        checkpoints[fold] = paths[0]

    # Only the problem and exact segmented response identify a reusable annotation.
    known = {}
    for path in [base_dags, args.base_root.parent / "multiple_n16/dag_analysis/deepseek/analyzed_records.jsonl"]:
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if int(r["problem_id"]) not in selected:
                continue
            for key, entry, sample in iter_samples([r], VARIANTS):
                if sample.get("dag_metadata", {}).get("model") == MODEL and sample.get("dag_analysis"):
                    known.setdefault((key.problem_id, key.variant, entry["problem"], sample["response"]), sample)

    anchors, records, pending, conditions = [], [], [], []
    for pid, source in sorted(selected.items()):
        orig = originals[pid]["original"]
        anchor, = orig["samples"]
        base_sample, = base[pid]["original"]["samples"]
        if (anchor["response"] != base_sample["response"] or orig["problem"] != base[pid]["original"]["problem"]
                or anchor["dag_metadata"]["model"] != MODEL):
            raise ValueError(f"Original anchor mismatch: {pid}")
        validate_dag(anchor["dag_analysis"], anchor["steps"])
        anchors.append({"problem_id":pid, "original":orig})
        row, todo = {"problem_id":pid}, {"problem_id":pid}
        fold = fixed[0][pid]["outer_fold"]
        for variant in VARIANTS:
            problem = base[pid][variant]["problem"]
            samples, by_text = [], {}
            for index, (alpha, run) in enumerate(zip(ALPHAS, fixed)):
                entry = run[pid][variant]
                item, = entry["samples"]
                if entry["alpha"] != alpha or entry["problem"] != problem or run[pid]["outer_fold"] != fold:
                    raise ValueError(f"Mismatched condition: {pid}/{variant}/{alpha}")
                response = item["response"]
                if response not in by_text:
                    sid = f"{pid}_{variant}_{index}"
                    by_text[response] = sid
                    samples.append(dict(item, sample_id=sid))
                conditions.append(dict(problem_id=pid, source=source, fold=fold, variant=variant,
                    alpha=alpha, sample_id=by_text[response], correct=item["correct"], valid=item["valid"],
                    response_hash=response_hash(response), anchor_hash=response_hash(anchor["response"])))
            row[variant] = {"problem":problem, "samples":samples}
        process_record(row)
        for key, entry, s in iter_samples([row], VARIANTS):
            historic = known.get((pid, key.variant, entry["problem"], s["response"]))
            if historic and historic["steps"] == s["steps"]:
                s["dag_analysis"] = validate_dag(historic["dag_analysis"], s["steps"])
                s["dag_metadata"] = historic["dag_metadata"]
            else:
                dest = todo.setdefault(key.variant, {"problem":entry["problem"], "samples":[]})
                dest["samples"].append({"sample_id":str(key), "steps":s["steps"]})
        records.append(row)
        if len(todo) > 1:
            pending.append(todo)
    write_json(out / "cohort.json", dict(seed=SEED, allocation="proportional largest remainder by source",
        sample=dict(sorted(selected.items())), checkpoints=checkpoints, annotator=MODEL,
        ged_revision=REVISION, ged_timeout_seconds=120, primary="normalized GED on complete triples",
        tie="equal at pipeline precision (4 decimals)"))
    write_jsonl(out / "conditions.jsonl", conditions)
    write_jsonl(out / "anchors.jsonl", anchors)
    write_jsonl(out / "trajectories.jsonl", records)
    write_jsonl(out / "pending.jsonl", pending)
    counts = Counter(selected.values())
    print(f"{len(selected)} groups {dict(counts)}, {len(conditions)} conditions, "
          f"{sum(1 for _ in iter_samples(records, VARIANTS))} distinct trajectories, "
          f"{sum(1 for _ in iter_samples(pending, VARIANTS))} new annotations")


def measure(args):
    out = args.output_dir
    annotations = out / "annotation/analyzed_records.jsonl"
    records = list(read_jsonl(out / "trajectories.jsonl"))
    new = {str(k):s for k, _, s in iter_samples(read_jsonl(annotations), VARIANTS)} if annotations.exists() else {}
    for key, _, s in iter_samples(records, VARIANTS):
        if "dag_analysis" in s:
            continue
        annotated = new.get(str(key))
        if not annotated or not annotated.get("dag_analysis"):
            continue  # The shared GED pipeline records missing graphs explicitly.
        if annotated["steps"] != s["steps"] or annotated["dag_metadata"]["model"] != MODEL:
            raise ValueError(f"Annotation mismatch: {key}")
        s["dag_analysis"] = validate_dag(annotated["dag_analysis"], s["steps"])
        s["dag_metadata"] = annotated["dag_metadata"]
    write_jsonl(out / "variant_graphs.jsonl", records)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_ged_analysis.py"),
        "--output-root", str(out), "--original-records", str(out / "anchors.jsonl"),
        "--variant-records", str(out / "variant_graphs.jsonl"), "--correctness-file", str(out / "trajectories.jsonl"),
        "--all-results-output", str(out / "ged.jsonl"), "--num-samples", "3", "--ged-timeout", "120",
        "--text-similarity-revision", REVISION, "--embedding-device", "cpu", "--workers", str(args.workers),
        "--rebuild-graph-cache"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare","measure","report"])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/canonical_math_paired.jsonl")
    parser.add_argument("--base-root", type=Path, default=ROOT / "output/qwen-2.5/base")
    parser.add_argument("--fixed-root", type=Path, default=ROOT / "output/qwen_fixed_alpha_seed42")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "output/qwen_staged_kfold_seed42")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/qwen_fixed_alpha_seed42/structure_probe")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "measure":
        measure(args)
    else:
        from data_analysis.measurement.alpha_trend import report
        report(args.output_dir)


if __name__ == "__main__":
    main()
