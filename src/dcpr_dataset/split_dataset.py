#!/usr/bin/env python3
"""Split DCPR dataset into train/val/test by problem_id."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def dump_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def variant_stats(items):
    stats = defaultdict(int)
    for item in items:
        stats[item.get("variant_type", "unknown")] += 1
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="按 problem_id 划分 DCPR 训练/验证/测试集")
    parser.add_argument("--input", required=True, help="输入 JSONL（如 data/<model>/dcpr_dataset.jsonl）")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-output", type=str, default=None, help="训练集输出路径")
    parser.add_argument("--val-output", type=str, default=None, help="验证集输出路径")
    parser.add_argument("--test-output", type=str, default=None, help="测试集输出路径")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {ratio_sum}")

    input_path = Path(args.input)
    all_items = load_jsonl(input_path)
    if not all_items:
        raise ValueError(f"输入文件为空：{input_path}")

    by_problem = defaultdict(list)
    for item in all_items:
        pid = item.get("problem_id")
        if pid is None:
            raise ValueError("检测到缺失 problem_id 的样本，无法按题划分")
        by_problem[pid].append(item)

    problem_ids = list(by_problem.keys())
    random.Random(args.seed).shuffle(problem_ids)

    num_problems = len(problem_ids)
    n_train = int(num_problems * args.train_ratio)
    n_val = int(num_problems * args.val_ratio)
    n_test = num_problems - n_train - n_val

    train_ids = set(problem_ids[:n_train])
    val_ids = set(problem_ids[n_train:n_train + n_val])
    test_ids = set(problem_ids[n_train + n_val:])

    train_items = []
    val_items = []
    test_items = []
    for pid, items in by_problem.items():
        if pid in train_ids:
            train_items.extend(items)
        elif pid in val_ids:
            val_items.extend(items)
        else:
            test_items.extend(items)

    out_dir = input_path.parent
    train_output = Path(args.train_output) if args.train_output else out_dir / "dcpr_train.jsonl"
    val_output = Path(args.val_output) if args.val_output else out_dir / "dcpr_val.jsonl"
    test_output = Path(args.test_output) if args.test_output else out_dir / "dcpr_test.jsonl"

    dump_jsonl(train_output, train_items)
    dump_jsonl(val_output, val_items)
    dump_jsonl(test_output, test_items)

    print("=== Dataset Split Summary ===")
    print(f"Input: {input_path}")
    print(f"Unique problems: {num_problems}")
    print(f"Train/Val/Test problems: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"Train samples: {len(train_items)}, variants: {variant_stats(train_items)}")
    print(f"Val samples: {len(val_items)}, variants: {variant_stats(val_items)}")
    print(f"Test samples: {len(test_items)}, variants: {variant_stats(test_items)}")
    print(f"Train output: {train_output}")
    print(f"Val output: {val_output}")
    print(f"Test output: {test_output}")


if __name__ == "__main__":
    main()
