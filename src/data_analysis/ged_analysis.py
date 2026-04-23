"""Analyze GED between original and variant (simple/hard) problem responses."""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import sys

try:
    from data_analysis.dag_similarity import (
        compute_ged_similarity,
        extract_dag_from_batch_response
    )
    from data_analysis.dag_compressor import compress_dag_combined, build_digraph_with_tags
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_analysis.dag_similarity import (
        compute_ged_similarity,
        extract_dag_from_batch_response
    )
    from data_analysis.dag_compressor import compress_dag_combined, build_digraph_with_tags


def load_batch_records(path: str) -> List[Dict]:
    """Load batch format JSONL records."""
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_correctness_data(path: str) -> Dict[int, Dict]:
    """Load correctness labels from all_records_50.jsonl."""
    correctness = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                problem_id = data['problem_id']
                correctness[problem_id] = {
                    'original': data['original'],
                    'simple': data['simple'],
                    'hard': data['hard']
                }
    return correctness


def get_original_graph(records: List[Dict], problem_id: int):
    """Find original problem graph.

    Supports both legacy custom_id (`<pid>_original`) and current batch style
    custom_id (`<pid>_original_0`).
    """
    target_ids = {f"{problem_id}_original_0", f"{problem_id}_original"}
    for record in records:
        custom_id = record.get('custom_id', '')
        if custom_id in target_ids:
            dag = extract_dag_from_batch_response(record)
            if dag:
                return build_digraph_with_tags(dag)
            else:
                print(f"Warning: Found {custom_id} but failed to extract DAG")
                return None
    print(f"Warning: original graph not found for problem {problem_id} in {len(records)} records")
    return None


def get_variant_samples(records: List[Dict], problem_id: int, variant: str, num_samples: int) -> List[Dict]:
    """Get variant samples from dag_analysis_50."""
    samples = []
    for record in records:
        custom_id = record.get('custom_id', '')
        if custom_id.startswith(f"{problem_id}_{variant}_"):
            sample_num = int(custom_id.split('_')[-1])
            dag = extract_dag_from_batch_response(record)
            if dag:
                samples.append({'sample_num': sample_num, 'dag': dag, 'custom_id': custom_id})
    samples.sort(key=lambda x: x['sample_num'])
    return samples[:num_samples]


def analyze_problem(
    problem_id: int,
    num_samples: int,
    original_records: Optional[List[Dict]],
    variant_records: List[Dict],
    correctness_data: Dict
) -> List[Dict]:
    """Analyze GED for one problem's variants."""

    original_graph = None
    if original_records:
        original_graph = get_original_graph(original_records, problem_id)
    if original_graph is None:
        # New default path: use the same analyzed file that already includes original_0.
        original_graph = get_original_graph(variant_records, problem_id)

    if not original_graph:
        print(f"No original graph found for problem {problem_id}")
        return []

    try:
        original_graph, _ = compress_dag_combined(original_graph)
    except Exception as exc:
        print(f"Warning: Compression failed for {problem_id}_original, using uncompressed graph: {exc}")

    problem_data = correctness_data.get(problem_id, {})
    variant_samples = {}
    for variant in ['simple', 'hard']:
        variant_samples[variant] = get_variant_samples(variant_records, problem_id, variant, num_samples)

    total_samples = sum(len(samples) for samples in variant_samples.values())
    processed_samples = 0
    results = []
    for variant in ['simple', 'hard']:
        samples = variant_samples[variant]
        variant_info = problem_data.get(variant, {})
        correct_labels = variant_info.get('samples', [])

        for sample_idx, sample in enumerate(samples, start=1):
            processed_samples += 1
            print(
                f"[{processed_samples}/{total_samples}] "
                f"Computing GED for {sample['custom_id']} "
                f"({variant} {sample_idx}/{len(samples)})"
            )
            try:
                sample_graph = build_digraph_with_tags(sample['dag'])
                sample_graph_compressed, _ = compress_dag_combined(sample_graph)
            except Exception as exc:
                print(f"Warning: Skip {sample['custom_id']} due to graph build/compression error: {exc}")
                continue

            try:
                ged_result = compute_ged_similarity(original_graph, sample_graph_compressed)
            except Exception as exc:
                print(f"Warning: Skip {sample['custom_id']} due to GED computation error: {exc}")
                continue

            if ged_result.get('ged') is None and not ged_result.get('timed_out', False):
                print(f"Warning: Skip {sample['custom_id']} because GED result is invalid")
                continue

            correct = correct_labels[sample['sample_num']]['correct'] if sample['sample_num'] < len(correct_labels) else None
            response = correct_labels[sample['sample_num']]['response'] if sample['sample_num'] < len(correct_labels) else ""

            results.append({
                'problem_id': problem_id,
                'sample_id': sample['custom_id'],
                'variant': variant,
                'problem': variant_info.get('problem', ''),
                'response': response,
                'correct': correct,
                'ged': ged_result['ged'],
                'similarity_normalized': ged_result['similarity_normalized'],
                'timed_out': ged_result['timed_out']
            })

    return results


def write_results_to_csv(results: List[Dict], problem_id: int, output_dir: str):
    """Write results and summary statistics to CSV."""
    output_path = Path(output_dir) / f"{problem_id}_ged_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['problem_id', 'sample_id', 'variant', 'problem', 'response', 'correct', 'ged', 'similarity_normalized', 'timed_out'])
        writer.writeheader()
        writer.writerows(results)

        # Calculate and write summary
        f.write('\n# Summary Statistics\n')
        for variant in ['simple', 'hard']:
            variant_results = [r for r in results if r['variant'] == variant]
            correct_geds = [r['ged'] for r in variant_results if r['correct'] and r['ged'] is not None]
            incorrect_geds = [r['ged'] for r in variant_results if not r['correct'] and r['ged'] is not None]

            f.write(f"# {variant.capitalize()}\n")
            if correct_geds:
                f.write(f"# Correct avg GED,{sum(correct_geds)/len(correct_geds):.4f}\n")
            if incorrect_geds:
                f.write(f"# Incorrect avg GED,{sum(incorrect_geds)/len(incorrect_geds):.4f}\n")

    print(f"Results saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Analyze GED between original and variant responses')
    parser.add_argument('--output-root', type=str, default=None,
                        help='模型输出根目录（包含 all_records*.jsonl 与 dag_analysis* 目录）')
    parser.add_argument('--original-records', type=str, default=None,
                        help='[可选] original DAG 分析结果 JSONL 路径；未提供时从 variant-records 中读取 original_0')
    parser.add_argument('--variant-records', type=str, default=None,
                        help='simple/hard DAG 分析结果 JSONL 路径（默认: <output-root>/dag_analysis_50/analyzed_records.jsonl）')
    parser.add_argument('--correctness-file', type=str, default=None,
                        help='回答正确性 JSONL 路径（默认: <output-root>/all_records_50.jsonl）')
    parser.add_argument('--all-results-output', type=str, default=None,
                        help='GED 汇总 JSONL 输出路径（默认: <output-root>/all_ged_results.jsonl）')
    parser.add_argument('--problem-id', type=int, help='Specific problem ID to analyze (default: all 279)')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples per variant (max 50)')
    parser.add_argument('--save-csv', action='store_true', help='Save individual CSV files per problem')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root) if args.output_root else None

    if args.variant_records:
        variant_records_path = Path(args.variant_records)
    else:
        if output_root is None:
            raise ValueError("需要提供 --output-root 或 --variant-records")
        variant_records_path = output_root / "dag_analysis_50" / "analyzed_records.jsonl"

    if args.correctness_file:
        correctness_path = Path(args.correctness_file)
    else:
        if output_root is None:
            raise ValueError("需要提供 --output-root 或 --correctness-file")
        correctness_path = output_root / "all_records_50.jsonl"

    if output_root is None:
        output_root = correctness_path.parent

    all_results_path = Path(args.all_results_output) if args.all_results_output else output_root / "all_ged_results.jsonl"

    print("Loading data...")
    variant_records = load_batch_records(str(variant_records_path))
    if args.original_records:
        original_records = load_batch_records(str(Path(args.original_records)))
        print(f"Loaded optional original records from {args.original_records}")
    else:
        original_records = None
        print("No --original-records provided, using original_0 from variant records as GED baseline.")
    correctness_data = load_correctness_data(str(correctness_path))

    if args.problem_id:
        problem_ids = [args.problem_id]
        print(f"Analyzing problem {args.problem_id}...")
    else:
        problem_ids = sorted(correctness_data.keys())
        print(f"Analyzing all {len(problem_ids)} problems...")

    all_results = []
    for idx, problem_id in enumerate(problem_ids, 1):
        print(f"\n=== Problem {problem_id} ({idx}/{len(problem_ids)}) ===")
        try:
            results = analyze_problem(problem_id, args.num_samples, original_records, variant_records, correctness_data)
        except Exception as exc:
            print(f"Warning: Skip problem {problem_id} due to unexpected error: {exc}")
            continue
        all_results.extend(results)

        if args.save_csv and results:
            write_results_to_csv(results, problem_id, output_dir=str(output_root / "similarity_check"))

    if all_results:
        # Save all results to JSONL for post-processing
        with open(all_results_path, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\nAll GED results saved to {all_results_path}")
        print(f"Total samples processed: {len(all_results)}")
        print(f"\nTo generate DCPR dataset, run:")
        print(f"  python src/dcpr_training/data_filter.py --input {all_results_path}")
    else:
        print("No results generated")


if __name__ == '__main__':
    main()
