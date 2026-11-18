"""Endpoint preference weights for trajectories already assigned to a fold."""

from collections import defaultdict
from itertools import groupby
import math

from utils.io import read_jsonl


def resolve_prefix_sampling(args):
    weight = args.ged_preference_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('--ged-preference-weight must be finite and nonnegative')
    if args.prefix_trajectory_sampling is None:
        args.prefix_trajectory_sampling = 'random_one' if weight else 'flat'
    if weight and args.prefix_trajectory_sampling != 'random_one':
        raise ValueError('GED endpoint preference requires random_one prefix sampling')


def load_ged_scores(path):
    scores = {}
    for row in read_jsonl(path):
        score = row.get('ged_normalized')
        if (score is None or row.get('status', 'ok') != 'ok' or row.get('timed_out')
                or row.get('approximate')):
            continue
        key = row['sample_id']
        if key in scores:
            raise ValueError(f'Duplicate GED sample: {key}')
        scores[key] = row
    return scores


def assign_ged_weights(rows, scores=None):
    """Rank within this split only. Missing scores retain the ordinary LM target."""
    groups = defaultdict(list)
    for row in rows:
        row['ged_weight'] = 0.0
        variant = row['variant_type']
        if variant not in ('simple', 'hard') or row['target_alpha'] != int(variant == 'hard'):
            raise ValueError(f"Unexpected endpoint label for {row['sample_id']}")
        if scores is None:
            score = row.get('ged_normalized')
        else:
            match = scores.get(row['sample_id'])
            if match is None:
                continue
            if (int(match['problem_id']) != int(row['problem_id']) or match['variant'] != variant
                    or match['problem'] != row['problem'] or match['response'] != row['response']):
                raise ValueError(f"GED trajectory differs from fold data: {row['sample_id']}")
            score = match['ged_normalized']
        if score is None:
            continue
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid normalized GED for {row['sample_id']}: {score}")
        groups[int(row['problem_id']), variant, row['problem']].append((score, row))

    for (_, variant, _), group in groups.items():
        ordered = sorted(group, key=lambda x: x[0])
        preceding = 0
        for _, equal in groupby(ordered, key=lambda x: x[0]):
            tied = list(equal)
            rank = preceding + (len(tied) + 1) / 2
            r = rank / (len(group) + 1)
            for _, row in tied:
                row['ged_weight'] = r if variant == 'hard' else 1 - r
            preceding += len(tied)
    return sum(map(len, groups.values()))
