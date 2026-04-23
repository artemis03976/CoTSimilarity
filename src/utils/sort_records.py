"""Sort analyzed records by problem_id for easier comparison with response visualization."""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_records(input_path: str) -> List[Dict]:
    """Load records from JSONL file."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_problem_id_from_record(record: Dict) -> str:
    """Extract problem_id from record, handling both batch and processed formats.

    Args:
        record: Record dictionary

    Returns:
        Problem ID string
    """
    # Check if it's batch format (has custom_id)
    if "custom_id" in record:
        custom_id = record["custom_id"]
        # Format: "{problem_id}_{variant}"
        parts = custom_id.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
        return custom_id

    # Check if it's processed format (has problem_id)
    if "problem_id" in record:
        return record["problem_id"]

    # Fallback
    return "999999"


def extract_numeric_id(problem_id: str) -> int:
    """Extract numeric part from problem_id for sorting.

    Args:
        problem_id: Problem ID string (e.g., "123", "456")

    Returns:
        Integer value for sorting
    """
    try:
        # Try to convert directly to int
        return int(problem_id)
    except ValueError:
        # If it contains non-numeric characters, extract digits
        import re
        match = re.search(r'\d+', str(problem_id))
        if match:
            return int(match.group())
        # Fallback: return a large number to put at end
        return 999999


def sort_records(records: List[Dict]) -> List[Dict]:
    """Sort records by problem_id numerically.

    Args:
        records: List of record dictionaries

    Returns:
        Sorted list of records
    """
    return sorted(records, key=lambda r: extract_numeric_id(extract_problem_id_from_record(r)))


def save_records(records: List[Dict], output_path: str):
    """Save records to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="对 DAG 分析结果按 problem_id 排序"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/deepseek/dag_analysis/analyzed_records.jsonl",
        help="输入的分析结果文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/deepseek/dag_analysis/analyzed_records_sorted.jsonl",
        help="输出的排序后文件"
    )
    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return

    # Load records
    print(f"正在加载数据: {args.input}")
    records = load_records(args.input)
    print(f"已加载 {len(records)} 条记录")

    if not records:
        print("警告: 没有找到任何记录")
        return

    # Show sample of original order
    print("\n原始顺序 (前5条):")
    for i, record in enumerate(records[:5]):
        pid = extract_problem_id_from_record(record)
        print(f"  {i+1}. Problem ID: {pid}")

    # Sort records
    print("\n正在排序...")
    sorted_records = sort_records(records)

    # Show sample of sorted order
    print("\n排序后顺序 (前5条):")
    for i, record in enumerate(sorted_records[:5]):
        pid = extract_problem_id_from_record(record)
        print(f"  {i+1}. Problem ID: {pid}")

    # Save sorted records
    print(f"\n正在保存到: {args.output}")
    save_records(sorted_records, args.output)

    print(f"\n完成! 已保存 {len(sorted_records)} 条排序后的记录")
    print(f"输出文件: {Path(args.output).absolute()}")

    # Show statistics
    problem_ids = [extract_problem_id_from_record(r) for r in sorted_records]
    unique_ids = sorted(set(problem_ids), key=extract_numeric_id)
    print(f"\nProblem ID 统计:")
    print(f"  总记录数: {len(sorted_records)}")
    print(f"  唯一 Problem ID 数: {len(unique_ids)}")
    print(f"  最小 ID: {unique_ids[0]}")
    print(f"  最大 ID: {unique_ids[-1]}")


if __name__ == "__main__":
    main()
