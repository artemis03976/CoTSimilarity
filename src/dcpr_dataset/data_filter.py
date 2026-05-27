"""
GED-based trajectory filtering for DCPR dataset construction.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_ged_results(input_path: str) -> List[Dict]:
    """Load GED analysis results from JSONL."""
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def filter_trajectories_by_ged(
    all_results: List[Dict],
    output_path: str,
    top_k: int = 5,
    min_ged_variance: float = 1.0
) -> None:
    """
    Filter trajectories using GED to create D_exploit and D_explore datasets.

    Args:
        all_results: All GED analysis results
        output_path: Path to output filtered dataset
        top_k: Number of top samples to select per problem (default: 5)
        min_ged_variance: Minimum GED variance required (default: 1.0)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group by problem_id and variant
    problems = {}
    for r in all_results:
        key = (r['problem_id'], r['variant'])
        if key not in problems:
            problems[key] = []
        problems[key].append(r)

    dcpr_data = []
    stats = {'total_problems': 0, 'skipped_no_correct': 0, 'skipped_low_variance': 0, 'selected': 0}

    for (problem_id, variant), samples in sorted(problems.items()):
        stats['total_problems'] += 1

        # Step 1: Filter correct answers only
        correct_samples = [s for s in samples if s['correct'] and s['ged'] is not None and not s['timed_out']]

        if not correct_samples:
            stats['skipped_no_correct'] += 1
            print(f"  Problem {problem_id} {variant}: No correct samples, skipped")
            continue

        # Step 2: Check GED variance
        geds = [s['ged'] for s in correct_samples]
        ged_variance = max(geds) - min(geds)

        if ged_variance < min_ged_variance:
            stats['skipped_low_variance'] += 1
            print(f"  Problem {problem_id} {variant}: Low GED variance ({ged_variance:.2f}), skipped")
            continue

        # Step 3: Relative ranking - select top-k
        if variant == 'simple':
            sorted_samples = sorted(correct_samples, key=lambda x: x['ged'])
        else:
            sorted_samples = sorted(correct_samples, key=lambda x: x['ged'], reverse=True)

        selected = sorted_samples[:top_k]

        for sample in selected:
            dcpr_data.append({
                'problem_id': problem_id,
                'problem': sample['problem'],
                'response': sample['response'],
                'variant_type': variant,
                'target_alpha': 0.0 if variant == 'simple' else 1.0,
                'ged_score': sample['ged'],
                'sample_id': sample['sample_id']
            })
            stats['selected'] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dcpr_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n=== DCPR Dataset Statistics ===")
    print(f"Total problem variants: {stats['total_problems']}")
    print(f"Skipped (no correct): {stats['skipped_no_correct']}")
    print(f"Skipped (low variance): {stats['skipped_low_variance']}")
    print(f"Selected samples: {stats['selected']}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Filter GED results to create DCPR training dataset')
    parser.add_argument('--input', required=True, help='Input JSONL file with GED results')
    parser.add_argument('--output', help='Output JSONL file (default: data/dcpr_dataset.jsonl)')
    parser.add_argument('--top-k', type=int, default=5, help='Top-k samples per problem (default: 5)')
    parser.add_argument('--min-variance', type=float, default=1.0, help='Min GED variance (default: 1.0)')
    args = parser.parse_args()

    output_path = args.output or 'data/dcpr_dataset.jsonl'

    print(f"Loading GED results from {args.input}...")
    all_results = load_ged_results(args.input)
    print(f"Loaded {len(all_results)} samples")

    print(f"\nFiltering with top_k={args.top_k}, min_variance={args.min_variance}...")
    filter_trajectories_by_ged(all_results, output_path, args.top_k, args.min_variance)


if __name__ == '__main__':
    main()

