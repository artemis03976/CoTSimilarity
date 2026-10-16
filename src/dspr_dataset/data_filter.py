"""
Shared pre-DAG screening and GED trajectory selection for DSPR datasets.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


VARIANTS = ("simple", "hard")
DEFAULT_TOP_K = 5
DEFAULT_MIN_GED_RANGE = 3.0


@dataclass(frozen=True)
class EligibilityStrategy:
    """Keep correct candidates and require a minimum count.

    This candidate qualification is independent of GED spread and ranking.
    GED selection requires a score and excludes GED timeouts by default.
    Pre-DAG screening explicitly disables these two unavailable conditions.
    The default pipeline marks a variant eligible only after the range check
    also passes; top-k selection never changes that eligibility label.
    """

    min_correct_samples: int = 1
    score_field: str | None = 'ged'
    exclude_timeouts: bool = True

    def __post_init__(self):
        if self.min_correct_samples < 1:
            raise ValueError('min_correct_samples must be positive')
        if self.score_field is not None and not self.score_field:
            raise ValueError('score_field must be non-empty')

    def candidates(self, samples: List[Dict]) -> List[Dict]:
        return [
            sample for sample in samples
            if sample.get('correct')
            and (self.score_field is None or sample.get(self.score_field) is not None)
            and (not self.exclude_timeouts or not sample.get('timed_out', False))
        ]

    def accepts(self, candidates: List[Dict]) -> bool:
        return self.accepts_count(len(candidates))

    def accepts_count(self, count: int) -> bool:
        return count >= self.min_correct_samples


@dataclass(frozen=True)
class CheapScreenStrategy:
    """Require a valid original anchor and enough valid correct paths on both variants.

    This pre-DAG screen neither requires original correctness nor applies GED
    range/ranking. Records are summarized without modifying their contents.
    """

    min_correct_samples: int = 3

    def __post_init__(self):
        if self.min_correct_samples < 3:
            raise ValueError('--min-correct must be at least 3')

    @property
    def eligibility(self) -> EligibilityStrategy:
        return EligibilityStrategy(
            self.min_correct_samples, score_field=None, exclude_timeouts=False,
        )

    def sample_stats(self, samples: List[Dict]) -> Dict:
        """Summarize saved quality/correctness labels without answer evaluation."""
        valid = [sample for sample in samples if bool(sample.get('valid'))]
        correct = self.eligibility.candidates(valid)
        unknown = [sample for sample in valid if sample.get('answer_correct') is None]
        responses = {' '.join(str(sample.get('response') or '').split()) for sample in valid}
        return {
            'sample_count': len(samples),
            'valid_count': len(valid),
            'correct_count': len(correct),
            'incorrect_count': len(valid) - len(correct) - len(unknown),
            'answer_unknown_count': len(unknown),
            'unique_valid_response_count': len(responses),
        }

    def classify(self, original_valid: bool, simple_count: int, hard_count: int) -> str:
        if not original_valid:
            return 'invalid_original'
        simple_ok = self.eligibility.accepts_count(simple_count)
        hard_ok = self.eligibility.accepts_count(hard_count)
        if simple_ok and hard_ok:
            return 'pair_candidate'
        if simple_ok:
            return 'simple_only'
        if hard_ok:
            return 'hard_only'
        if simple_count or hard_count:
            return 'insufficient_correct'
        return 'no_correct'

    def evaluate(self, record: Dict) -> Dict:
        stats = {
            variant: self.sample_stats(record.get(variant, {}).get('samples', []))
            for variant in ('original', *VARIANTS)
        }
        original_valid = stats['original']['valid_count'] > 0
        original_samples = record.get('original', {}).get('samples', [])
        original_correct = bool(original_samples and original_samples[0].get('correct'))
        category = self.classify(
            original_valid, stats['simple']['correct_count'], stats['hard']['correct_count'],
        )
        return {
            **stats,
            'original_valid': original_valid,
            'original_correct': original_valid and original_correct,
            'category': category,
            'pre_eligible': category == 'pair_candidate',
        }


@dataclass(frozen=True)
class GEDRangeStrategy:
    """Require max(score) - min(score) on the qualified candidates."""

    min_ged_range: float = DEFAULT_MIN_GED_RANGE
    score_field: str = 'ged'

    def __post_init__(self):
        if self.min_ged_range < 0:
            raise ValueError('min_ged_range must be non-negative')
        if not self.score_field:
            raise ValueError('score_field must be non-empty')

    def accepts(self, candidates: List[Dict]) -> bool:
        if not candidates:
            return False
        scores = [sample[self.score_field] for sample in candidates]
        return max(scores) - min(scores) >= self.min_ged_range


@dataclass(frozen=True)
class TopKStrategy:
    """Select lowest-GED simple or highest-GED hard trajectories (default: 5)."""

    top_k: int = DEFAULT_TOP_K
    score_field: str = 'ged'

    def __post_init__(self):
        if self.top_k < 1:
            raise ValueError('top_k must be positive')
        if not self.score_field:
            raise ValueError('score_field must be non-empty')

    def select(self, candidates: List[Dict], variant: str) -> List[Dict]:
        if variant not in VARIANTS:
            raise ValueError(f'Unsupported variant: {variant}')
        direction = 1 if variant == 'simple' else -1
        return sorted(
            candidates,
            key=lambda sample: (
                direction * sample[self.score_field], str(sample.get('sample_id', '')),
            ),
        )[:self.top_k]


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

    return EligibilityStrategy(score_field=score_field).candidates(samples)


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


def select_trajectories_by_ged(
    all_results: List[Dict],
    top_k: int = DEFAULT_TOP_K,
    min_ged_range: float = DEFAULT_MIN_GED_RANGE,
    model_name: str | None = None,
    source_file: str | None = None,
    score_field: str = 'ged',
    min_correct_samples: int = 1,
    verbose: bool = False,
) -> Dict:
    """Compose candidate eligibility, GED min-max screening, then top-k selection.

    This is the shared, in-memory implementation for dataset export and k-fold
    construction. No files are written here. ``ged`` defaults retain the legacy
    protocol; normalized GED runs specify their score field and thresholds.
    """
    eligibility_strategy = EligibilityStrategy(min_correct_samples, score_field)
    range_strategy = GEDRangeStrategy(min_ged_range, score_field)
    top_k_strategy = TopKStrategy(top_k, score_field)

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
        correct_samples = eligibility_strategy.candidates(samples)

        if not correct_samples:
            stats['skipped_no_correct'] += 1
            if verbose:
                print(f"  Problem {problem_id} {variant}: No correct samples, skipped")
            continue

        if not eligibility_strategy.accepts(correct_samples):
            stats['skipped_insufficient_correct'] += 1
            if verbose:
                print(
                    f"  Problem {problem_id} {variant}: Only "
                    f"{len(correct_samples)} correct samples "
                    f"(< {min_correct_samples}), skipped"
                )
            continue

        # Step 2: require the shared within-group GED range threshold.
        if not range_strategy.accepts(correct_samples):
            stats['skipped_low_ged_range'] += 1
            if verbose:
                print(f"  Problem {problem_id} {variant}: Low GED range, skipped")
            continue

        eligible[variant].append(problem_id)

        # Step 3: Relative ranking - select top-k
        selected = top_k_strategy.select(correct_samples, variant)

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
    return {
        'stats': stats,
        'eligibility': eligibility_payload,
        'selected': dspr_data,
    }


def load_filtered_trajectories(
    eligibility_path: str | Path,
    ged_results_path: str | Path | None = None,
) -> Dict:
    """Apply a saved protocol to its GED source; never expand a mask to raw samples.

    The saved IDs are checked against recomputed eligibility, rather than used
    as a second filter. Callers receive the same selected records as exports.
    """
    eligibility_path = Path(eligibility_path)
    payload = json.loads(eligibility_path.read_text(encoding='utf-8'))
    required = ('score_field', 'ged_range_threshold', 'min_correct_samples', 'top_k')
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError(f'Eligibility manifest must specify the full filtering protocol: {eligibility_path}')
    if ged_results_path is None:
        ged_results_path = payload.get('source', {}).get('ged_results')
        if not ged_results_path:
            sources = payload.get('source_files', [])
            if len(sources) == 1:
                ged_results_path = sources[0]
    if not ged_results_path:
        raise ValueError('GED source is missing; pass --ged-results or record it in the eligibility manifest')
    source_path = Path(ged_results_path)
    if not source_path.is_absolute() and not source_path.is_file():
        for base in (eligibility_path.parent, Path(__file__).resolve().parents[2]):
            candidate = base / source_path
            if candidate.is_file():
                source_path = candidate
                break
    result = select_trajectories_by_ged(
        load_ged_results(str(source_path)),
        top_k=payload['top_k'],
        min_ged_range=payload['ged_range_threshold'],
        score_field=payload['score_field'],
        min_correct_samples=payload['min_correct_samples'],
        model_name=payload.get('model'),
        source_file=str(source_path),
    )
    saved_ids = payload.get('problem_ids', {})
    for variant in VARIANTS:
        if not isinstance(saved_ids.get(variant), list):
            raise ValueError(f'Eligibility IDs for {variant} must be a list: {eligibility_path}')
        expected = {int(pid) for pid in saved_ids[variant]}
        actual = set(result['eligibility']['problem_ids'][variant])
        if expected != actual:
            raise ValueError(
                f'Eligibility manifest disagrees with GED filtering for {variant}: '
                f'manifest-only IDs {sorted(expected - actual)}, '
                f'GED-only IDs {sorted(actual - expected)}'
            )
    result['source_file'] = str(source_path)
    return result


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
    """Write the shared selector's trajectories and optional eligibility manifest."""
    result = select_trajectories_by_ged(
        all_results, top_k=top_k, min_ged_range=min_ged_range,
        model_name=model_name, source_file=source_file, score_field=score_field,
        min_correct_samples=min_correct_samples, verbose=True,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in result['selected']:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    if eligibility_output_path:
        eligibility_path = Path(eligibility_output_path)
        eligibility_path.parent.mkdir(parents=True, exist_ok=True)
        eligibility_path.write_text(
            json.dumps(result['eligibility'], ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    stats = result['stats']
    print("\n=== DSPR Dataset Statistics ===")
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

    return result
