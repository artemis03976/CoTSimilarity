"""
GED-based trajectory filtering for DSPR dataset construction.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


VARIANTS = ("simple", "hard")
DEFAULT_TOP_K = 5
DEFAULT_MIN_GED_RANGE = 3.0


def load_ged_results(input_path: str) -> List[Dict]:
    """Load GED analysis results from JSONL."""
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def valid_correct_samples(
    samples: List[Dict],
    score_field: str = 'ged',
) -> List[Dict]:
    """Return correct, non-timeout samples with a usable selection score."""

    return [
        sample
        for sample in samples
        if sample.get('correct')
        and sample.get(score_field) is not None
        and not sample.get('timed_out', False)
    ]


def build_eligibility_payload(
    all_results: List[Dict],
    eligible: Dict[str, List[int]],
    top_k: int,
    min_ged_range: float,
    model_name: str | None = None,
    source_file: str | None = None,
    score_field: str = 'ged',
    min_correct_samples: int = 1,
) -> Dict:
    """Build the eligibility manifest consumed by k-fold construction."""

    simple = set(eligible['simple'])
    hard = set(eligible['hard'])
    union = simple | hard
    intersection = simple & hard
    payload = {
        'ged_range_threshold': min_ged_range,
        'score_field': score_field,
        'min_correct_samples': min_correct_samples,
        'top_k': top_k,
        'comparison': (
            f'max(correct non-timeout {score_field}) - '
            f'min(correct non-timeout {score_field}) >= threshold'
        ),
        'total_problem_ids': len({result['problem_id'] for result in all_results}),
        'counts': {
            'simple_eligible': len(simple),
            'hard_eligible': len(hard),
            'eligible_union': len(union),
            'eligible_intersection': len(intersection),
            'simple_only': len(simple - hard),
            'hard_only': len(hard - simple),
        },
        'problem_ids': {
            'simple': sorted(simple),
            'hard': sorted(hard),
            'union': sorted(union),
            'intersection': sorted(intersection),
            'simple_only': sorted(simple - hard),
            'hard_only': sorted(hard - simple),
        },
    }
    if model_name:
        payload['model'] = model_name
    if source_file:
        payload['source_files'] = [source_file]
    return payload


def filter_trajectories_by_ged(
    all_results: List[Dict],
    output_path: str,
    top_k: int = DEFAULT_TOP_K,
    min_ged_range: float = DEFAULT_MIN_GED_RANGE,
    eligibility_output_path: str | None = None,
    model_name: str | None = None,
    source_file: str | None = None,
    score_field: str = 'ged',
    min_correct_samples: int = 1,
) -> Dict:
    """
    Filter trajectories using GED to create D_reuse and D_adapt datasets.

    Args:
        all_results: All GED analysis results
        output_path: Path to output filtered dataset
        top_k: Number of top samples to select per problem-variant (default: 5)
        min_ged_range: Minimum within-group GED range (default: 3.0)
        score_field: Field used for range computation and ranking. The legacy
            default is ``ged``; use ``ged_normalized`` for scale-free curation.
        min_correct_samples: Minimum number of correct, non-timeout samples
            required before applying the range test.
        eligibility_output_path: Optional k-fold eligibility manifest output
    """
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if min_ged_range < 0:
        raise ValueError('min_ged_range must be non-negative')
    if not score_field:
        raise ValueError('score_field must be non-empty')
    if min_correct_samples < 1:
        raise ValueError('min_correct_samples must be positive')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group by problem_id and variant
    problems = {}
    for r in all_results:
        key = (r['problem_id'], r['variant'])
        if key not in problems:
            problems[key] = []
        problems[key].append(r)

    dspr_data = []
    stats = {
        'total_problem_variants': 0,
        'skipped_no_correct': 0,
        'skipped_insufficient_correct': 0,
        'skipped_low_ged_range': 0,
        'selected': 0,
    }
    eligible = {variant: [] for variant in VARIANTS}

    for (problem_id, variant), samples in sorted(problems.items()):
        if variant not in VARIANTS:
            continue
        stats['total_problem_variants'] += 1

        # Step 1: keep correct, non-timeout trajectories with a GED value.
        correct_samples = valid_correct_samples(samples, score_field=score_field)

        if not correct_samples:
            stats['skipped_no_correct'] += 1
            print(f"  Problem {problem_id} {variant}: No correct samples, skipped")
            continue

        if len(correct_samples) < min_correct_samples:
            stats['skipped_insufficient_correct'] += 1
            print(
                f"  Problem {problem_id} {variant}: Only "
                f"{len(correct_samples)} correct samples "
                f"(< {min_correct_samples}), skipped"
            )
            continue

        # Step 2: require the shared within-group GED range threshold.
        scores = [s[score_field] for s in correct_samples]
        ged_range = max(scores) - min(scores)

        if ged_range < min_ged_range:
            stats['skipped_low_ged_range'] += 1
            print(
                f"  Problem {problem_id} {variant}: Low GED range "
                f"({ged_range:.2f} < {min_ged_range:.2f}), skipped"
            )
            continue

        eligible[variant].append(problem_id)

        # Step 3: Relative ranking - select top-k
        if variant == 'simple':
            sorted_samples = sorted(
                correct_samples,
                key=lambda x: (x[score_field], str(x.get('sample_id', ''))),
            )
        else:
            sorted_samples = sorted(
                correct_samples,
                key=lambda x: (-x[score_field], str(x.get('sample_id', ''))),
            )

        selected = sorted_samples[:top_k]

        for sample in selected:
            dspr_data.append({
                'problem_id': problem_id,
                'problem': sample['problem'],
                'response': sample['response'],
                'variant_type': variant,
                'target_alpha': 0.0 if variant == 'simple' else 1.0,
                'ged_score': sample['ged'],
                'selection_score': sample[score_field],
                'selection_score_field': score_field,
                'ged_normalized': sample.get('ged_normalized'),
                'similarity_normalized': sample.get('similarity_normalized'),
                'sample_id': sample['sample_id']
            })
            stats['selected'] += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dspr_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    eligibility_payload = build_eligibility_payload(
        all_results,
        eligible,
        top_k,
        min_ged_range,
        score_field=score_field,
        min_correct_samples=min_correct_samples,
        model_name=model_name,
        source_file=source_file,
    )
    if eligibility_output_path:
        eligibility_path = Path(eligibility_output_path)
        eligibility_path.parent.mkdir(parents=True, exist_ok=True)
        eligibility_path.write_text(
            json.dumps(eligibility_payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    print(f"\n=== DSPR Dataset Statistics ===")
    print(f"Total problem variants: {stats['total_problem_variants']}")
    print(f"Skipped (no correct): {stats['skipped_no_correct']}")
    print(
        f"Skipped (fewer than {min_correct_samples} correct): "
        f"{stats['skipped_insufficient_correct']}"
    )
    print(
        f"Skipped ({score_field} range < {min_ged_range:g}): "
        f"{stats['skipped_low_ged_range']}"
    )
    print(f"Selected samples: {stats['selected']}")
    print(f"Output: {output_path}")
    if eligibility_output_path:
        print(f"Eligibility: {eligibility_output_path}")

    return {
        'stats': stats,
        'eligibility': eligibility_payload,
        'selected': dspr_data,
    }


def main():
    parser = argparse.ArgumentParser(description='Filter GED results to create DSPR training dataset')
    parser.add_argument('--input', required=True, help='Input JSONL file with GED results')
    parser.add_argument('--output', help='Output JSONL file (default: data/dspr_dataset.jsonl)')
    parser.add_argument(
        '--top-k',
        type=int,
        default=DEFAULT_TOP_K,
        help=f'Top-k samples per problem-variant (default: {DEFAULT_TOP_K})',
    )
    parser.add_argument(
        '--min-ged-range',
        '--min-variance',
        dest='min_ged_range',
        type=float,
        default=DEFAULT_MIN_GED_RANGE,
        help=(
            'Minimum range of correct, non-timeout GEDs; --min-variance is a '
            f'legacy alias (default: {DEFAULT_MIN_GED_RANGE:g})'
        ),
    )
    parser.add_argument(
        '--score-field',
        choices=('ged', 'ged_normalized'),
        default='ged',
        help=(
            'Field used for within-problem range and ranking. Use '
            'ged_normalized for scale-free curation (default: ged).'
        ),
    )
    parser.add_argument(
        '--min-correct-samples',
        type=int,
        default=1,
        help=(
            'Minimum correct, non-timeout trajectories per problem-variant '
            'before the range test (default: 1).'
        ),
    )
    parser.add_argument(
        '--eligibility-output',
        help='Optional eligibility JSON used by k-fold construction',
    )
    parser.add_argument('--model-name', help='Optional model label stored in eligibility JSON')
    args = parser.parse_args()

    output_path = args.output or 'data/dspr_dataset.jsonl'

    print(f"Loading GED results from {args.input}...")
    all_results = load_ged_results(args.input)
    print(f"Loaded {len(all_results)} samples")

    print(
        f"\nFiltering with top_k={args.top_k}, "
        f"score_field={args.score_field}, "
        f"min_range={args.min_ged_range}, "
        f"min_correct_samples={args.min_correct_samples}..."
    )
    filter_trajectories_by_ged(
        all_results,
        output_path,
        top_k=args.top_k,
        min_ged_range=args.min_ged_range,
        score_field=args.score_field,
        min_correct_samples=args.min_correct_samples,
        eligibility_output_path=args.eligibility_output,
        model_name=args.model_name,
        source_file=args.input,
    )


if __name__ == '__main__':
    main()
