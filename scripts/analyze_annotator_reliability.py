#!/usr/bin/env python3
"""Audit and summarize the four existing base-greedy annotator artifacts.

This command makes no model calls and never recomputes or replaces source GEDs.
It intentionally does not estimate within-variant ranking/selection from n=1.
"""

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dspr-reliability-matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from data_analysis.annotation.parsing import validate_dag
from data_analysis.measurement.reliability import (
    EDGE_KINDS, ROLE_ORDER, compare_dags, confusion_rows, summarize_clustered,
)
from data_analysis.records import iter_samples
from utils.artifacts import file_identity, fingerprint
from utils.io import read_jsonl, write_json, write_jsonl

ANNOTATORS = ("deepseek", "gpt", "haiku", "qwen")
DISPLAY_NAMES = {"deepseek": "DeepSeek", "gpt": "GPT", "haiku": "Haiku", "qwen": "Qwen"}
VARIANTS = ("original", "simple", "hard")
PAIRS = tuple(combinations(ANNOTATORS, 2))
ANNOTATION_METRICS = ("role_agreement",) + tuple(f"edge_{k}_f1" for k in EDGE_KINDS)
GED_METRICS = ("absolute_difference", "signed_difference_b_minus_a", "raw_absolute_difference",
               "normalizer_absolute_difference")


def read_json(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in r.items()})


def samples_by_id(path):
    records = list(read_jsonl(path))
    ids = [r["problem_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate problem IDs in {path}")
    return {str(key): (key, entry, sample) for key, entry, sample in iter_samples(records)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_inputs(base, canonical_path):
    segmented_path = base / "dag_analysis/segmented_records.jsonl"
    segmented = samples_by_id(segmented_path)
    raw = samples_by_id(base / "all_records.jsonl")
    require(segmented.keys() == raw.keys(), "Raw/segmented sample IDs differ")
    canonical = {r["problem_id"]: r for r in read_jsonl(canonical_path)}
    require({key.problem_id for key, _, _ in segmented.values()} == canonical.keys(),
            "Base does not cover canonical problems exactly")
    for sid, (key, entry, sample) in segmented.items():
        require(key.sample_index == 0, "This report is scoped to base greedy n=1")
        require(entry["problem"] == canonical[key.problem_id][key.variant]["problem"],
                f"Canonical question mismatch: {sid}")
        require(sample["response"] == raw[sid][2]["response"], f"Raw CoT mismatch: {sid}")
        require(sample["steps"] and len({s['index'] for s in sample['steps']}) == len(sample['steps']),
                f"Empty or duplicate frozen steps: {sid}")
    files = [segmented_path, base / "all_records.jsonl", canonical_path]
    datasets, valid, configs, provenance, recovered, failures = {}, {}, {}, {}, {}, []
    for annotator in ANNOTATORS:
        directory = base / "dag_analysis" / annotator
        path = directory / "analyzed_records.jsonl"
        files.append(path)
        data = samples_by_id(path)
        require(data.keys() == segmented.keys(), f"Sample IDs differ: {annotator}")
        datasets[annotator], valid[annotator] = data, set()
        config_path = directory / ("batch/batch_requests.config.json" if annotator == "qwen"
                                   else "normal/analysis.config.json")
        files.append(config_path)
        config = read_json(config_path)
        settings = config.get("settings", config)
        configs[annotator] = settings
        recovery = config.get("recovery", {})
        recovered[annotator] = set(recovery.get("recovered_samples", recovery.get("recovered_sample_ids", [])))
        for sid, (_, entry, sample) in data.items():
            original = segmented[sid][2]
            require(entry["problem"] == segmented[sid][1]["problem"], f"Question drift: {annotator}/{sid}")
            for field in ("response", "steps", "correct", "valid"):
                require(sample.get(field) == original.get(field), f"{field} drift: {annotator}/{sid}")
            try:
                validate_dag(sample.get("dag_analysis"), original["steps"])
                for node in sample["dag_analysis"]:
                    require(len(node["depends_on"]) == len(set(node["depends_on"])), "Duplicate dependencies")
                valid[annotator].add(sid)
            except ValueError as exc:
                failures.append({"annotator": annotator, "sample_id": sid,
                                 "problem_id": data[sid][0].problem_id, "variant": data[sid][0].variant,
                                 "status": "invalid_dag" if sample.get("dag_analysis") else "missing_dag",
                                 "error": sample.get("dag_error") or str(exc)})
            metadata = sample.get("dag_metadata", {})
            if metadata.get("recovery") or metadata.get("format_recovery"):
                recovered[annotator].add(sid)
            if annotator == "qwen":
                require(config["samples"][sid] == fingerprint({"problem": entry["problem"],
                        "steps": sample["steps"], "response": sample["response"]}),
                        f"Batch identity mismatch: {sid}")
        require(recovered[annotator] <= data.keys(), "Unknown recovered sample")
        versions = Counter(data[s][2].get("dag_metadata", {}).get("response_model") or "unrecorded"
                           for s in valid[annotator])
        recovery_details = []
        for sid in sorted(recovered[annotator]):
            metadata = data[sid][2].get("dag_metadata", {})
            recovery_details.append({"sample_id": sid, "recovery": metadata.get("recovery"),
                                     "annotation_signature": metadata.get("annotation_signature")})
        provenance[annotator] = {
            "provider": settings.get("provider"), "model": settings.get("model"),
            "parameters": settings.get("parameters"), "prompt_hash": settings.get("prompts"),
            "response_models": dict(versions), "recovered_count": len(recovered[annotator]),
            "recovery_details": recovery_details, "run_recovery_metadata": recovery,
            "final_valid": len(valid[annotator]), "final_invalid": len(data) - len(valid[annotator]),
        }
    require(len({c["prompts"] for c in configs.values()}) == 1, "Prompt hashes differ")
    provenance["prompt_file_matches_manifests"] = (
        file_identity(REPO_ROOT / "src/data_analysis/annotation/prompts.py")["sha256"] == configs["gpt"]["prompts"])
    ged, ged_failures, ged_configs = {}, {}, {}
    expected = {sid for sid, (key, _, _) in segmented.items() if key.variant != "original"}
    for annotator in ANNOTATORS:
        directory = base / "ged" / annotator
        path = directory / "all_ged_results.jsonl"
        cfg_path = directory / "all_ged_results.config.json"
        failure_path = directory / "all_ged_results.failures.jsonl"
        files.extend([path, cfg_path, failure_path])
        config = read_json(cfg_path)
        ged_configs[annotator] = config
        metadata = config["graph_cache_metadata"]
        require(metadata["sources"]["original_records"] is None,
                "Base audit requires anchor from the same annotator dataset")
        require(metadata["sources"]["variant_records"]["sha256"] ==
                file_identity(base / "dag_analysis" / annotator / "analyzed_records.jsonl")["sha256"],
                f"GED uses a different DAG file: {annotator}")
        require(config["inputs"]["correctness"]["sha256"] == file_identity(base / "all_records.jsonl")["sha256"],
                f"GED correctness source changed: {annotator}")
        require(config["failure_details"]["sha256"] == file_identity(failure_path)["sha256"],
                f"Failure ledger changed: {annotator}")
        records, missing = list(read_jsonl(path)), list(read_jsonl(failure_path))
        ged[annotator] = {r["sample_id"]: r for r in records}
        ged_failures[annotator] = {r["sample_id"]: r for r in missing}
        require(len(ged[annotator]) == len(records) and len(ged_failures[annotator]) == len(missing),
                f"Duplicate GED or failure IDs: {annotator}")
        require(not (ged[annotator].keys() & ged_failures[annotator].keys()), "GED/failure overlap")
        require(ged[annotator].keys() | ged_failures[annotator].keys() == expected,
                f"GED outcomes do not account for all frozen pairs: {annotator}")
        require(dict(Counter(r['status'] for r in missing)) == config['failure_counts'],
                f"Failure counts disagree: {annotator}")
        for sid, r in ged[annotator].items():
            key, entry, sample = segmented[sid]
            require(r['problem_id'] == key.problem_id and r['variant'] == key.variant, "GED identity mismatch")
            require(r["status"] == "ok" and not r["timed_out"] and not r["approximate"], "Non-exact GED retained")
            require(r["response"] == sample["response"] and r['problem'] == entry['problem']
                    and r['correct'] == sample['correct'], f"GED input drift: {sid}")
            require(sid in valid[annotator] and f"{key.problem_id}_original_0" in valid[annotator],
                    f"GED without valid same-annotator graphs: {sid}")
            require(r['ged_normalizer'] > 0 and np.isfinite(r['ged']) and r['ged'] >= 0, "Invalid GED")
            require(abs(r['ged_normalized'] - np.clip(r['ged'] / r['ged_normalizer'], 0, 1)) <= .00005001,
                    "GED normalization inconsistent with stored four-decimal precision")
        provenance[annotator]["ged_success"] = len(records)
        provenance[annotator]["ged_failure_counts"] = dict(Counter(r["status"] for r in missing))
    reference = ged_configs[ANNOTATORS[0]]
    for config in ged_configs.values():
        require(config["inputs"]["options"] == reference["inputs"]["options"], "Different GED options")
        require(config["inputs"]["code"] == reference["inputs"]["code"], "Different GED code")
        require(config["graph_cache_metadata"]["embedding"] == reference["graph_cache_metadata"]["embedding"],
                "Different embedding configuration")
    common = set.intersection(*valid.values())
    common_ged = set.intersection(*(set(g) for g in ged.values()))
    sources = {pid: c["source"] for pid, c in canonical.items()}
    manifest = {
        "scope": "Qwen-2.5 base greedy; all frozen inputs; n=1 per problem-variant",
        "input_files": [file_identity(p) for p in files],
        "problems": sorted(canonical), "n_samples": len(segmented),
        "source_problem_counts": dict(Counter(sources.values())),
        "common_annotation_ids": sorted(common), "common_ged_ids": sorted(common_ged),
        "recovered_ids": {a: sorted(ids) for a, ids in recovered.items()},
        "annotator_provenance": provenance,
        "ged_options": reference["inputs"]["options"],
        "embedding": reference["graph_cache_metadata"]["embedding"],
        "ged_code": reference["inputs"]["code"],
        "passed_checks": ["frozen IDs, questions, responses, segmentation, correctness and validity",
                          "exact step alignment and legal roles/dependencies (failures retained)",
                          "same prompt hashes in saved manifests", "Qwen batch input fingerprints",
                          "GED DAG/correctness/failure file hashes", "same-annotator anchor source",
                          "GED options, embedding revision and code hashes match across annotators",
                          "all expected GED comparisons accounted for exactly once",
                          "stored normalized GED agrees with raw GED and normalizer"],
        "limitations": ["No multi-candidate cross-annotator data: Kendall tau-b, top-k and eligibility flips not estimable.",
                        "Base greedy coverage does not describe the n=16 supervision candidate pool.",
                        "The experiment measures cross-annotator consistency under fixed CoTs and segmentation.",
                        "Saved configurations include recovery and output-budget differences.",
                        "Legacy DeepSeek per-response model version is unrecorded for 1631 outputs; current migrated manifest is not a full historical request log."],
        "case_rule": "Retrospective descriptive cases: within each variant, smallest and largest four-annotator normalized GED range on the common-success set; tie-break by numeric problem ID. No correctness or role/edge-agreement filter. These are extremes, not representative or pre-registered cases.",
    }
    return segmented, datasets, valid, ged, ged_failures, sources, common, common_ged, recovered, failures, manifest


def analyze(args):
    (segmented, datasets, valid, ged, ged_failures, sources, common, common_ged,
     recovered, failures, manifest) = audit_inputs(args.base_root, args.canonical)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    recovered_union = set.union(*recovered.values())
    annotation_rows, confusion = [], []
    for a, b in PAIRS:
        pair_rows = []
        counts = Counter()
        weights = Counter()
        for sid in sorted(common):
            key = segmented[sid][0]
            dag_a, dag_b = datasets[a][sid][2]["dag_analysis"], datasets[b][sid][2]["dag_analysis"]
            row = {"annotator_a": a, "annotator_b": b, "sample_id": sid,
                   "problem_id": key.problem_id, "variant": key.variant, "source": sources[key.problem_id],
                   "any_recovery": sid in recovered_union, **compare_dags(dag_a, dag_b)}
            pair_rows.append(row)
            for tags in confusion_rows(dag_a, dag_b):
                counts[tags] += 1
                weights[tags] += 1 / len(dag_a)
        annotation_rows.extend(pair_rows)
        for left in ROLE_ORDER:
            for right in ROLE_ORDER:
                confusion.append({"annotator_a": a, "annotator_b": b, "role_a": left, "role_b": right,
                                  "step_count": counts[left, right],
                                  "cot_macro_joint_fraction": weights[left, right] / len(common)})
    ged_rows = []
    for a, b in PAIRS:
        for sid in sorted(common_ged):
            key = segmented[sid][0]
            x, y = ged[a][sid], ged[b][sid]
            ged_rows.append({
                "annotator_a": a, "annotator_b": b, "sample_id": sid, "problem_id": key.problem_id,
                "variant": key.variant, "source": sources[key.problem_id],
                "any_recovery": sid in recovered_union or f"{key.problem_id}_original_0" in recovered_union,
                "normalized_a": x["ged_normalized"], "normalized_b": y["ged_normalized"],
                "absolute_difference": abs(x["ged_normalized"] - y["ged_normalized"]),
                "signed_difference_b_minus_a": y["ged_normalized"] - x["ged_normalized"],
                "raw_a": x["ged"], "raw_b": y["ged"],
                "raw_absolute_difference": abs(x["ged"] - y["ged"]),
                "normalizer_a": x["ged_normalizer"], "normalizer_b": y["ged_normalizer"],
                "normalizer_absolute_difference": abs(x["ged_normalizer"] - y["ged_normalizer"]),
            })
    summary = []
    for stage, rows, metrics in (("annotation", annotation_rows, ANNOTATION_METRICS),
                                 ("ged", ged_rows, GED_METRICS)):
        for a, b in PAIRS:
            pair = [r for r in rows if (r["annotator_a"], r["annotator_b"]) == (a, b)]
            groups = [("all", "all", pair),
                      ("no_recovery", "all", [r for r in pair if not r["any_recovery"]])]
            groups += [(v, "all", [r for r in pair if r["variant"] == v])
                       for v in (VARIANTS if stage == "annotation" else VARIANTS[1:])]
            groups += [("all", source, [r for r in pair if r["source"] == source])
                       for source in sorted(set(sources.values()))]
            for variant, source, subset in groups:
                stats = summarize_clustered(subset, metrics, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
                for metric, estimate in stats.items():
                    row = {"stage": stage, "annotator_a": a, "annotator_b": b,
                           "variant": variant, "source": source, "metric": metric, **estimate}
                    if metric.startswith("edge_"):
                        stem = metric.removesuffix("_f1")
                        row.update({"both_empty": sum(r[f"{stem}_both_empty"] for r in subset),
                                    "one_empty": sum(r[f"{stem}_one_empty"] for r in subset)})
                    summary.append(row)
    coverage, status_rows = [], []
    for annotator in ANNOTATORS:
        for variant in VARIANTS:
            ids = {sid for sid, (k, _, _) in segmented.items() if k.variant == variant}
            good = ids & valid[annotator]
            coverage.append({"stage": "annotation", "annotator": annotator, "variant": variant,
                             "expected": len(ids), "succeeded": len(good), "failed": len(ids - good),
                             "coverage": len(good) / len(ids), "common_success": len(ids & common),
                             "recovered": len(ids & recovered[annotator])})
            if variant == "original":
                continue
            good = ids & ged[annotator].keys()
            fail = [ged_failures[annotator][sid] for sid in sorted(ids - good)]
            coverage.append({"stage": "ged", "annotator": annotator, "variant": variant,
                             "expected": len(ids), "succeeded": len(good), "failed": len(ids - good),
                             "coverage": len(good) / len(ids), "common_success": len(ids & common_ged),
                             **dict(Counter(r["status"] for r in fail))})
        for sid in sorted(s for s, (k, _, _) in segmented.items() if k.variant != "original"):
            key = segmented[sid][0]
            result = ged[annotator].get(sid, ged_failures[annotator].get(sid))
            status_rows.append({"annotator": annotator, "sample_id": sid, "problem_id": key.problem_id,
                                "variant": key.variant, "source": sources[key.problem_id],
                                "status": result["status"], "common_success": sid in common_ged})
    # Success-state disagreement retains all inputs, including failures.
    success_agreement = []
    for a, b in PAIRS:
        for variant in VARIANTS[1:]:
            counts = Counter((sid in ged[a], sid in ged[b]) for sid, (k, _, _) in segmented.items()
                             if k.variant == variant)
            success_agreement.append({"annotator_a": a, "annotator_b": b, "variant": variant,
                                      "both_success": counts[True, True], "a_only": counts[True, False],
                                      "b_only": counts[False, True], "neither": counts[False, False]})
    case_candidates = []
    for sid in common_ged:
        key = segmented[sid][0]
        scores = {a: ged[a][sid]["ged_normalized"] for a in ANNOTATORS}
        case_candidates.append({"sample_id": sid, "problem_id": key.problem_id, "variant": key.variant,
                                "ged_range": max(scores.values()) - min(scores.values()), "scores": scores})
    cases = []
    for variant in VARIANTS[1:]:
        subset = [c for c in case_candidates if c["variant"] == variant]
        for kind, direction in (("smallest_spread", 1), ("largest_spread", -1)):
            selected = min(subset, key=lambda c: (direction * c["ged_range"], c["problem_id"]))
            cases.append({"selection": kind, **selected})
    manifest.update({
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "unit": "problem_id",
                      "interval": "95% percentile", "point_estimate": "mean of per-CoT (or per-GED-pair) scores"},
        "empty_edges": "both empty: undefined, excluded with count; one empty: zero",
        "normalization_precision": "Use saved four-decimal ged_normalized; raw GED and normalizers retained.",
        "cases": cases, "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "analysis_code": [file_identity(Path(__file__))] + [
            file_identity(REPO_ROOT / "src/data_analysis" / name) for name in (
                "measurement/reliability.py", "annotation/parsing.py", "schemas.py",
                "graph/builder.py", "graph/compression.py", "records.py")],
        "unavailable_metrics": {"within_variant_kendall_tau_b": "n=1, not estimable",
                                "top_k_overlap": "n=1; all-selected overlap would be mechanical",
                                "eligibility_flips": "n=1 cannot meet frozen min=3; all-ineligible is not stability evidence"},
    })
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summary)
    csv_write(output / "summary.csv", summary)
    csv_write(output / "annotation_per_cot.csv", annotation_rows)
    csv_write(output / "role_confusions.csv", confusion)
    csv_write(output / "ged_per_pair.csv", ged_rows)
    csv_write(output / "coverage.csv", coverage)
    csv_write(output / "ged_success_disagreement.csv", success_agreement)
    csv_write(output / "ged_status_all_inputs.csv", status_rows)
    write_jsonl(output / "annotation_failures.jsonl", failures)
    write_jsonl(output / "ged_failures.jsonl", [dict(r, annotator=a) for a in ANNOTATORS
                                               for r in ged_failures[a].values()])
    render_figures(output, summary, ged_rows, confusion)
    write_cases(output, cases, segmented, datasets, ged)
    write_tables(output, summary)
    print(json.dumps({"output": str(output), "common_annotation": len(common),
                      "common_ged": len(common_ged), "cases": cases}, ensure_ascii=False, indent=2))


def render_figures(output, summary, ged_rows, confusion):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    labels = [DISPLAY_NAMES[a] for a in ANNOTATORS]
    for ax, metric, title in zip(axes[:2], ("role_agreement", "edge_all_f1"),
                                 ("Role agreement (CoT macro)", "Directed edge F1 (CoT macro)")):
        matrix = np.eye(4)
        for row in summary:
            if row["metric"] == metric and row["variant"] == "all" and row["source"] == "all":
                i, j = ANNOTATORS.index(row["annotator_a"]), ANNOTATORS.index(row["annotator_b"])
                matrix[i, j] = matrix[j, i] = row["mean"]
        ax.imshow(matrix, vmin=.4, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, "—" if i == j else f"{matrix[i,j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i,j] > .78 else "black")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels, title=title)
    for a, b in PAIRS:
        values = np.sort([r["absolute_difference"] for r in ged_rows
                          if (r["annotator_a"], r["annotator_b"]) == (a, b)])
        axes[2].step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                     label=f"{DISPLAY_NAMES[a]}–{DISPLAY_NAMES[b]}")
    axes[2].set(xlabel="Absolute normalized GED difference", ylabel="Empirical CDF",
                title="GED values: four-way common success", ylim=(0, 1.01))
    axes[2].legend(fontsize=8, loc="lower right")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"agreement_overview.{suffix}", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax, (a, b) in zip(axes.flat, PAIRS):
        matrix = np.zeros((6, 6))
        for r in confusion:
            if (r["annotator_a"], r["annotator_b"]) == (a, b):
                matrix[ROLE_ORDER.index(r["role_a"]), ROLE_ORDER.index(r["role_b"])] = r["cot_macro_joint_fraction"]
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=.3)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{100*matrix[i,j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i,j] > .17 else "black")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=ROLE_ORDER, yticklabels=ROLE_ORDER,
               xlabel=DISPLAY_NAMES[b], ylabel=DISPLAY_NAMES[a], title="CoT-weighted joint proportions (%)")
        ax.tick_params(axis="x", rotation=45)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"role_confusions.{suffix}", dpi=200)
    plt.close(fig)


def write_tables(output, summary):
    manifest = read_json(output / "manifest.json")
    annotation_n = len(manifest["common_annotation_ids"])
    simple_n = sum("_simple_" in sid for sid in manifest["common_ged_ids"])
    hard_n = sum("_hard_" in sid for sid in manifest["common_ged_ids"])
    def estimate(a, b, metric, variant="all"):
        return next(r for r in summary if r["annotator_a"] == a and r["annotator_b"] == b
                    and r["metric"] == metric and r["variant"] == variant and r["source"] == "all")

    def cell(r, scale=1, digits=3):
        return (f"{r['mean'] * scale:.{digits}f} "
                f"[{r['ci_low'] * scale:.{digits}f}, {r['ci_high'] * scale:.{digits}f}]")

    lines = ["# Four-annotator base-greedy agreement", "",
             "All intervals are 95% problem-cluster percentile bootstrap intervals (10,000 draws by default). "
             "Check manifest.json for the exact settings. Main comparisons share the four-way success set.", "",
             "## A. Uncompressed annotations", "",
             "| Pair | Role agreement (%) | All-edge F1 | Internal F1 | Root F1 | External F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for a, b in PAIRS:
        cells = [cell(estimate(a, b, "role_agreement"), 100, 2)]
        cells += [cell(estimate(a, b, f"edge_{k}_f1")) for k in EDGE_KINDS]
        lines.append(f"| {DISPLAY_NAMES[a]}–{DISPLAY_NAMES[b]} | " + " | ".join(cells) + " |")
    lines += ["", "Both-empty edge subsets are undefined and excluded from that metric only; "
              "see summary.csv for metric-specific N, both-empty and one-empty counts.", "",
              "## B. Normalized GED numerical sensitivity", "",
              "| Pair | Variant | N | Mean absolute difference [95% CI] | Median | P90 | P95 | Max |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for a, b in PAIRS:
        for variant in VARIANTS[1:]:
            r = estimate(a, b, "absolute_difference", variant)
            lines.append(f"| {DISPLAY_NAMES[a]}–{DISPLAY_NAMES[b]} | {variant} | {r['n']} | {cell(r)} | "
                         + " | ".join(f"{r[k]:.4f}" for k in ("median", "p90", "p95", "max")) + " |")
    lines += ["", "## Annotation breakdown by variant", "",
              "| Pair | Variant | N | Role agreement (%) | All-edge F1 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for a, b in PAIRS:
        for variant in VARIANTS:
            role = estimate(a, b, "role_agreement", variant)
            edge = estimate(a, b, "edge_all_f1", variant)
            lines.append(f"| {DISPLAY_NAMES[a]}–{DISPLAY_NAMES[b]} | {variant} | {role['n']} | "
                         f"{cell(role, 100, 2)} | {cell(edge)} |")
    (output / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    latex = [r"% Generated base-greedy fragment; not a claim of n=16 supervision stability.",
             r"\begin{table*}[t]", r"\centering\small",
             r"\begin{tabular}{lcc}", r"\hline",
             r"\multicolumn{3}{c}{A. Uncompressed annotation agreement ($N=" + str(annotation_n) + r"$ CoTs)} \\",
             r"Annotator pair & Role agreement (\%) & Directed edge F1 \\", r"\hline"]
    for a, b in PAIRS:
        latex.append(f"{DISPLAY_NAMES[a]}--{DISPLAY_NAMES[b]} & "
                     + cell(estimate(a, b, "role_agreement"), 100, 2) + " & "
                     + cell(estimate(a, b, "edge_all_f1")) + r" \\")
    latex += [r"\hline", r"\multicolumn{3}{c}{B. Mean absolute normalized GED difference} \\",
              f"Annotator pair & Simple ($N={simple_n}$) & Hard ($N={hard_n}$)" + r" \\", r"\hline"]
    for a, b in PAIRS:
        latex.append(f"{DISPLAY_NAMES[a]}--{DISPLAY_NAMES[b]} & "
                     + cell(estimate(a, b, "absolute_difference", "simple")) + " & "
                     + cell(estimate(a, b, "absolute_difference", "hard")) + r" \\")
    latex += [r"\hline", r"\end{tabular}",
              r"\caption{Cross-annotator agreement on frozen Qwen-2.5 base-greedy outputs. "
              r"Brackets give 95\% problem-cluster bootstrap intervals. "
              f"Annotation coverage is ${annotation_n}/{manifest['n_samples']}$; "
              f"common GED coverage is ${simple_n + hard_n}/{2 * len(manifest['problems'])}$. "
              r"Each GED uses the same annotator for its anchor and variant graph. "
              r"One response per problem--variant cannot establish within-variant ranking, top-$k$ selection, or eligibility stability.}",
              r"\label{tab:base-annotator-reliability}", r"\end{table*}"]
    (output / "section_4_1_base_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_cases(output, cases, segmented, datasets, ged):
    from data_analysis.graph.builder import build_digraph_with_tags
    from data_analysis.graph.compression import compress_dag_combined
    from data_analysis.measurement.reliability import edge_sets
    lines = ["# Base greedy annotator sensitivity: reproducible case records", "",
             "Retrospective descriptive extremes, selected by the rule in manifest.json. "
             "All annotators are compared symmetrically at the fixed step IDs. "
             "With one candidate per variant, ranking and selection effects are not estimable.", ""]
    records = []
    for case in cases:
        sid = case["sample_id"]
        anchor = f"{case['problem_id']}_original_0"
        record = {**case, "anchor_id": anchor, "inputs": {}, "compressed_sizes": {}}
        lines += [f"## {sid}: {case['selection']}", "", f"Normalized GED range: {case['ged_range']:.4f}", "",
                  "| Annotator | GED | Normalizer | Normalized GED |", "| --- | ---: | ---: | ---: |"]
        for a in ANNOTATORS:
            r = ged[a][sid]
            lines.append(f"| {a} | {r['ged']:.6f} | {r['ged_normalizer']} | {r['ged_normalized']:.4f} |")
            record["compressed_sizes"][a] = {}
            for item in (anchor, sid):
                graph, _ = compress_dag_combined(build_digraph_with_tags(datasets[a][item][2]["dag_analysis"]))
                record["compressed_sizes"][a][item] = {"nodes": graph.number_of_nodes(),
                                                      "edges": graph.number_of_edges()}
            denominator = sum(sum(sizes.values()) for sizes in record["compressed_sizes"][a].values())
            require(denominator == r["ged_normalizer"], f"Case compression no longer matches stored GED: {a}/{sid}")
        lines += ["", "| Annotator | Compressed anchor (V, E) | Compressed variant (V, E) |",
                  "| --- | --- | --- |"]
        for a in ANNOTATORS:
            sizes = record["compressed_sizes"][a]
            lines.append(f"| {a} | ({sizes[anchor]['nodes']}, {sizes[anchor]['edges']}) | "
                         f"({sizes[sid]['nodes']}, {sizes[sid]['edges']}) |")
        for item in (anchor, sid):
            _, entry, sample = segmented[item]
            record["inputs"][item] = {"problem": entry["problem"], "response": sample["response"],
                                      "correct": sample["correct"], "steps": sample["steps"],
                                      "dags": {a: datasets[a][item][2]["dag_analysis"] for a in ANNOTATORS}}
            lines += ["", f"### {item}", "", entry["problem"], "",
                      f"Saved answer correctness: {sample['correct']}", "",
                      "| Step | Literal CoT span | DeepSeek | GPT | Haiku | Qwen |",
                      "| --- | --- | --- | --- | --- | --- |"]
            maps = {a: {n["step_id"]: n for n in datasets[a][item][2]["dag_analysis"]} for a in ANNOTATORS}
            for step in sample["steps"]:
                text = step["text"].replace("|", "\\|").replace("\n", "<br>")
                cells = [f"{maps[a][step['index']]['macro_action_tag']}; "
                         f"{maps[a][step['index']]['depends_on']}" for a in ANNOTATORS]
                lines.append(f"| {step['index']} | {text} | " + " | ".join(cells) + " |")
        records.append(record)
    (output / "cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "cases.json", records)
    # Source -> target matrices retain the exact uncompressed node identities.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    for record in records:
        fig, axes = plt.subplots(2, 4, figsize=(13, 8), layout="constrained")
        for row, (sid, item) in enumerate(record["inputs"].items()):
            steps = [s["index"] for s in item["steps"]]
            columns = [0, "External"] + steps
            edges = {a: edge_sets(item["dags"][a])["all"] for a in ANNOTATORS}
            unanimous = set.intersection(*edges.values())
            for col, a in enumerate(ANNOTATORS):
                matrix = np.zeros((len(steps), len(columns)))
                for source, target in edges[a]:
                    matrix[steps.index(target), columns.index(source)] = 1 if (source, target) in unanimous else 2
                ax = axes[row, col]
                ax.imshow(matrix, cmap=ListedColormap(["#fafafa", "#225b88", "#dc843b"]), vmin=0, vmax=2,
                          aspect="auto", interpolation="nearest")
                tick_cols = [0, 1] + list(range(2, len(columns), max(1, (len(columns) + 9) // 10)))
                ax.set(xticks=tick_cols, xticklabels=["E" if columns[i] == "External" else columns[i] for i in tick_cols],
                       yticks=list(range(0, len(steps), max(1, len(steps) // 10))),
                       yticklabels=steps[::max(1, len(steps) // 10)],
                       title=f"{DISPLAY_NAMES[a]} · {sid}", xlabel="Source step (0=root, E=External)", ylabel="Target step")
                ax.tick_params(labelsize=8)
        scores = ", ".join(f"{a}: {v:.4f}" for a, v in record['scores'].items())
        fig.suptitle(f"{record['sample_id']} · {record['selection']}\nBlue: all four agree; orange: present, not unanimous\nNormalized GED — {scores}", fontsize=12)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"case_{record['sample_id']}.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=REPO_ROOT / "output/qwen-2.5/base")
    parser.add_argument("--canonical", type=Path, default=REPO_ROOT / "data/canonical_math_paired.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output/qwen-2.5/reliability_audit/base_four_annotators")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    analyze(args)


if __name__ == "__main__":
    main()
