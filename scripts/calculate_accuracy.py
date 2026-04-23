#!/usr/bin/env python3
"""统一计算 all_records.jsonl 指标，并支持 baseline 与 DCPR 对比。"""

import argparse
import json
import sys
from pathlib import Path

VARIANTS = ["original", "simple", "hard"]


def _extract_correct_flags(question_data):
    """从 question_data 中提取每个 sample 的 correct 标记列表。"""
    if "samples" in question_data and question_data["samples"]:
        return [bool(sample.get("correct", False)) for sample in question_data["samples"]]
    if "correct" in question_data:
        return [bool(question_data["correct"])]
    return []


def calculate_metrics(jsonl_file):
    """计算每种题型的 first@1 / any@k / avg_sample_acc。"""
    stats = {
        variant: {
            "total": 0,
            "first_correct": 0,
            "any_correct": 0,
            "sample_correct": 0,
            "sample_total": 0,
        }
        for variant in VARIANTS
    }

    record_total = 0
    record_all_pass = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_total += 1

            all_pass_this_record = True
            for variant in VARIANTS:
                question_data = record.get(variant)
                if not question_data:
                    continue
                flags = _extract_correct_flags(question_data)
                if not flags:
                    continue

                stats[variant]["total"] += 1
                stats[variant]["sample_total"] += len(flags)
                stats[variant]["sample_correct"] += sum(flags)
                stats[variant]["first_correct"] += int(flags[0])
                stats[variant]["any_correct"] += int(any(flags))
                all_pass_this_record = all_pass_this_record and any(flags)

            if all_pass_this_record:
                record_all_pass += 1

    metrics = {}
    for variant in VARIANTS:
        s = stats[variant]
        total = s["total"]
        sample_total = s["sample_total"]
        metrics[variant] = {
            "total": total,
            "first_acc": (s["first_correct"] / total) if total else 0.0,
            "any_acc": (s["any_correct"] / total) if total else 0.0,
            "sample_acc": (s["sample_correct"] / sample_total) if sample_total else 0.0,
        }

    metrics["overall"] = {
        "records": record_total,
        "all_variant_any_acc": (record_all_pass / record_total) if record_total else 0.0,
    }
    return metrics


def _pct(x):
    return f"{x * 100:.2f}%"


def print_metrics(title, metrics):
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(f"{'Variant':10} {'Count':>8} {'First@1':>10} {'Any@k':>10} {'AvgSample':>12}")
    for variant in VARIANTS:
        m = metrics[variant]
        print(
            f"{variant:10} {m['total']:8d} {_pct(m['first_acc']):>10} "
            f"{_pct(m['any_acc']):>10} {_pct(m['sample_acc']):>12}"
        )
    overall = metrics["overall"]
    print(
        f"\nOverall(all variants pass@k on each record): "
        f"{_pct(overall['all_variant_any_acc'])} ({overall['records']} records)"
    )


def print_delta(base_metrics, new_metrics):
    print(f"\n{'-' * 72}")
    print("Delta (New - Base)")
    print(f"{'-' * 72}")
    print(f"{'Variant':10} {'ΔFirst@1':>12} {'ΔAny@k':>12} {'ΔAvgSample':>14}")
    for variant in VARIANTS:
        b = base_metrics[variant]
        n = new_metrics[variant]
        d_first = n["first_acc"] - b["first_acc"]
        d_any = n["any_acc"] - b["any_acc"]
        d_sample = n["sample_acc"] - b["sample_acc"]
        print(
            f"{variant:10} {d_first * 100:>+11.2f}% {d_any * 100:>+11.2f}% "
            f"{d_sample * 100:>+13.2f}%"
        )
    d_overall = new_metrics["overall"]["all_variant_any_acc"] - base_metrics["overall"]["all_variant_any_acc"]
    print(f"\nΔOverall(all variants pass@k): {d_overall * 100:+.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一实验指标统计脚本")
    parser.add_argument("file", type=str, help="待评估 all_records*.jsonl")
    parser.add_argument("--compare-file", type=str, default=None, help="对比文件（如 baseline all_records*.jsonl）")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"错误: 文件不存在 - {args.file}")
        sys.exit(1)

    new_metrics = calculate_metrics(args.file)
    print_metrics(f"Metrics - {Path(args.file).name}", new_metrics)

    if args.compare_file:
        if not Path(args.compare_file).exists():
            print(f"错误: 对比文件不存在 - {args.compare_file}")
            sys.exit(1)
        base_metrics = calculate_metrics(args.compare_file)
        print_metrics(f"Baseline - {Path(args.compare_file).name}", base_metrics)
        print_delta(base_metrics, new_metrics)
