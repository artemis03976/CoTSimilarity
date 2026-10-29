#!/usr/bin/env python3
"""Audit n16 GEDs and compare selection using the shared DSPR filtering strategies.

Offline only: writes diagnostic artifacts, never training masks or source results.
"""

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dspr-selection-matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from data_analysis.annotation.parsing import validate_dag
from data_analysis.measurement.reliability import summarize_clustered
from data_analysis.measurement.selection import kendall_tau_b, set_agreement, summarize_ratio_clustered
from data_analysis.records import iter_samples
from dspr_dataset.data_filter import (
    CheapScreenStrategy, EligibilityStrategy, GEDRangeStrategy,
    select_trajectories_by_ged,
)
from utils.artifacts import file_identity, fingerprint
from utils.io import read_jsonl, write_json, write_jsonl

ANNOTATORS = ("deepseek", "gpt", "haiku", "qwen")
NAMES = dict(zip(ANNOTATORS, ("DeepSeek", "GPT", "Haiku", "Qwen")))
PAIRS = tuple(combinations(ANNOTATORS, 2))
VARIANTS = ("simple", "hard")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_write(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def samples(path):
    records = list(read_jsonl(path))
    require(len({r['problem_id'] for r in records}) == len(records), f"Duplicate problems: {path}")
    return records, {str(k): (k, e, s) for k, e, s in iter_samples(records)}


def audit(args):
    root = args.data_root
    frozen_path = root / "dag_analysis/segmented_records.jsonl"
    pool_path = root / "pre_eligible/pre_eligible_set.jsonl"
    records, frozen = samples(frozen_path)
    pool_records, pool = samples(pool_path)
    require(frozen.keys() == pool.keys(), "Screened/frozen input IDs differ")
    screen = CheapScreenStrategy()
    require(all(screen.evaluate(r)['pre_eligible'] for r in pool_records), "Input fails cheap screen")
    for sid, (_, e, s) in frozen.items():
        require(e['problem'] == pool[sid][1]['problem'], f"Question mismatch: {sid}")
        require(all(s.get(k) == pool[sid][2].get(k) for k in ('response', 'correct', 'valid')),
                f"Frozen input mismatch: {sid}")
    expected = {sid for sid, (key, _, _) in frozen.items() if key.variant in VARIANTS}
    qualified = {sid for sid in expected if frozen[sid][2].get('valid') and frozen[sid][2].get('correct')}
    protocol_manifest = read_json(args.eligibility_config)
    protocol = {key: protocol_manifest[key] for key in
                ('score_field', 'ged_range_threshold', 'min_correct_samples')}
    # The user confirmed the current experiment retains all eligible candidates.
    # The old manifest's top_k=5 is historical, not the current selection rule.
    protocol['top_k'] = None
    files = [frozen_path, pool_path, args.eligibility_config, args.canonical]
    canonical = {r['problem_id']: r for r in read_jsonl(args.canonical)}
    source = {r['problem_id']: canonical[r['problem_id']]['source'] for r in records}
    dags, ged, failures, configs, provenance = {}, {}, {}, {}, {}
    coverage = []
    correctness_identities = {}
    for a in ANNOTATORS:
        path = root / 'dag_analysis' / a / 'analyzed_records.jsonl'
        _, data = samples(path); dags[a] = data
        require(data.keys() == frozen.keys(), f"Annotation coverage differs: {a}")
        acpath = path.parent / ('batch/batch_requests.config.json' if a == 'qwen' else 'normal/analysis.config.json')
        ac = read_json(acpath); settings = ac.get('settings', ac)
        recovered = set()
        for sid, (_, e, s) in data.items():
            fs = frozen[sid][2]
            require(e['problem'] == frozen[sid][1]['problem'], f"Question drift: {a}/{sid}")
            require(all(s.get(k) == fs.get(k) for k in ('response', 'steps', 'correct', 'valid')),
                    f"Input drift: {a}/{sid}")
            dag = validate_dag(s.get('dag_analysis'), fs['steps'])
            require([n['step_id'] for n in dag] == [step['index'] for step in fs['steps']], 'Node order drift')
            require(all(len(n['depends_on']) == len(set(n['depends_on'])) for n in dag), 'Repeated dependencies')
            meta = s.get('dag_metadata', {})
            if meta.get('recovery') or meta.get('format_recovery'):
                recovered.add(sid)
            if a == 'qwen':
                require(ac['samples'][sid] == fingerprint({'problem': e['problem'], 'steps': fs['steps'],
                                                          'response': fs['response']}), 'Batch input drift')
        gd = root / 'ged' / a
        rp, fp, cp = [gd / f'all_ged_results{x}' for x in ('.jsonl', '.failures.jsonl', '.config.json')]
        configs[a] = config = read_json(cp)
        rr, ff = list(read_jsonl(rp)), list(read_jsonl(fp))
        ged[a] = {r['sample_id']: r for r in rr}; failures[a] = {r['sample_id']: r for r in ff}
        require(len(ged[a]) == len(rr) and len(failures[a]) == len(ff), 'Duplicate GED outcomes')
        require(not (ged[a].keys() & failures[a].keys()), 'Success/failure overlap')
        require(ged[a].keys() | failures[a].keys() == expected, f"Unaccounted GED pairs: {a}")
        md = config['graph_cache_metadata']
        require(md['sources']['original_records'] is None, 'Anchor must come from same n16 annotator')
        require(md['sources']['variant_records']['sha256'] == file_identity(path)['sha256'], f"Stale DAG GED: {a}")
        # A historical correctness export may differ from the local file.
        # Never infer why or conceal that mismatch: verify the
        # score-relevant fields of EVERY outcome below and record the limitation.
        correctness_identities[a] = {
            'recorded': config['inputs']['correctness'], 'local': file_identity(pool_path),
            'byte_identical': config['inputs']['correctness']['sha256'] == file_identity(pool_path)['sha256'],
        }
        require(config['failure_details']['sha256'] == file_identity(fp)['sha256'], 'Failure hash mismatch')
        require(config['failure_counts'] == dict(Counter(r['status'] for r in ff)), 'Failure counts mismatch')
        for sid, row in ged[a].items():
            key, entry, sample = frozen[sid]
            require(row['problem_id'] == key.problem_id and row['variant'] == key.variant, 'GED identity mismatch')
            require(row['status'] == 'ok' and not row['timed_out'] and not row['approximate'], 'Inexact GED retained')
            require(row['response'] == sample['response'] and row['problem'] == entry['problem']
                    and row['correct'] == sample['correct'], f"GED input drift: {a}/{sid}")
            require(not row['correct'] or sample['valid'], 'Invalid correct candidate needs explicit protocol handling')
            require(np.isfinite(row['ged']) and row['ged'] >= 0 and row['ged_normalizer'] > 0, 'Invalid distance')
            require(abs(row['ged_normalized'] - np.clip(row['ged'] / row['ged_normalizer'], 0, 1)) <= .00005001,
                    'Normalized score mismatch')
        for sid, row in failures[a].items():
            key, _, sample = frozen[sid]
            require(row['problem_id'] == key.problem_id and row['variant'] == key.variant
                    and row['correct'] == sample['correct'] and row['valid'] == sample['valid'], 'Failure identity drift')
        recovery = ac.get('recovery', {})
        for name in ('recovered_samples', 'sample_ids'):
            recovered.update(recovery.get(name, []))
        provenance[a] = {'provider': settings.get('provider'), 'model': settings.get('model'),
                         'prompt_hash': settings.get('prompts', settings.get('prompt')),
                         'settings': settings, 'recovery': recovery, 'recovered_ids': sorted(recovered),
                         'response_models': dict(Counter(s.get('dag_metadata', {}).get('response_model') or 'unrecorded'
                                                         for _, _, s in data.values()))}
        for v in ('all', *VARIANTS):
            ids = {sid for sid in expected if v == 'all' or frozen[sid][0].variant == v}
            q = ids & qualified; successes = ids & ged[a].keys()
            coverage.append({'annotator': a, 'variant': v, 'total': len(ids), 'success': len(successes),
                             'coverage': len(successes) / len(ids), 'qualified_total': len(q),
                             'qualified_success': len(q & successes), 'qualified_coverage': len(q & successes) / len(q),
                             'failure_counts': dict(Counter(failures[a][s]['status'] for s in ids - successes))})
        files.extend([path, acpath, rp, fp, cp])
    reference = configs[ANNOTATORS[0]]
    require(len({c['inputs']['correctness']['sha256'] for c in configs.values()}) == 1,
            'Annotators used different correctness inputs')
    for a, config in configs.items():
        require(config['inputs']['options'] == reference['inputs']['options'], 'GED parameters differ')
        require(config['inputs']['code'] == reference['inputs']['code'], 'GED code differs')
        require(config['graph_cache_metadata']['embedding'] == reference['graph_cache_metadata']['embedding'], 'Encoder differs')
        require(config['graph_cache_metadata'].get('sources') is not None, 'Missing graph provenance')
    require(len({p['prompt_hash'] for p in provenance.values()}) == 1, 'Base annotation prompts differ')
    common_all = set.intersection(*(set(g) for g in ged.values()))
    common = common_all & qualified
    manifest = {'scope': 'All cheap-screened n16 groups; fixed valid-correct candidates; no reference annotator',
                'input_files': [file_identity(p) for p in files], 'problem_ids': sorted(source),
                'source_counts': dict(Counter(source.values())), 'sample_count': len(frozen),
                'expected_ged_pairs': len(expected), 'qualified_candidates': len(qualified),
                'qualified_candidate_ids': sorted(qualified), 'common_all_ged_ids': sorted(common_all),
                'common_qualified_ids': sorted(common), 'selection_protocol': protocol,
                'ged_options': reference['inputs']['options'], 'ged_code': reference['inputs']['code'],
                'embedding': reference['graph_cache_metadata']['embedding'], 'annotator_provenance': provenance,
                'correctness_file_identity': correctness_identities,
                'correctness_verification': 'Every successful GED question, response and correct label matches the frozen pool; every failure correct/valid label matches; frozen inputs match all SHA-verified annotation files.',
                'historical_eligibility_source': protocol_manifest.get('source'),
                'historical_manifest_top_k': protocol_manifest.get('top_k'),
                'selection_protocol_authority': 'User-confirmed current experiment: no top-k truncation; count/range parameters retained.',
                'historical_eligibility_ids_reused': False,
                'bootstrap': {'samples': args.bootstrap_samples, 'seed': args.seed, 'unit': 'problem_id'},
                'tie_policy': 'Saved four-decimal normalized GED; Kendall tau-b preserves ties. Ranking does not select candidates.',
                'case_rule': 'Retrospective minimum/maximum mean pairwise final-set Jaccard per variant on nonempty four-way unions; both-empty pairs undefined; numeric problem_id breaks ties.',
                'limitations': ['Common-success analyses are conditional on computational coverage.',
                                'The recorded historical correctness-file hash differs from the local export; scoring-relevant fields were verified record by record, but byte identity is not claimed.',
                                'Historical screening parameters are reused, but 120-second GEDs define a new diagnostic, not a reconstruction of completed kfold training.',
                                'Annotation recovery budgets and feedback differ; historical DeepSeek response versions are mostly unrecorded.',
                                'The experiment measures cross-annotator consistency and screening stability for fixed CoTs and segmentation.']}
    return frozen, dags, ged, failures, source, qualified, common, coverage, manifest



def analyze(args):
    frozen, dags, ged, failures, sources, qualified, common, coverage, manifest = audit(args)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    protocol = manifest['selection_protocol']; score = protocol['score_field']
    eligibility = EligibilityStrategy(protocol['min_correct_samples'], score)
    range_check = GEDRangeStrategy(protocol['ged_range_threshold'], score)
    group_ids = [(pid, v) for pid in sorted(sources) for v in VARIANTS]
    qualified_groups = {g: [] for g in group_ids}
    for sid in sorted(qualified):
        key = frozen[sid][0]; qualified_groups[key.problem_id, key.variant].append(sid)
    common_groups = {g: [sid for sid in ids if sid in common] for g, ids in qualified_groups.items()}
    decisions, selected, available, fixed_selected = [], {}, {}, {}
    for a in ANNOTATORS:
        result = select_trajectories_by_ged(list(ged[a].values()), top_k=None,
                    min_ged_range=protocol['ged_range_threshold'], score_field=score,
                    min_correct_samples=protocol['min_correct_samples'])
        selected[a] = {g: set() for g in group_ids}
        for row in result['selected']:
            selected[a][row['problem_id'], row['variant_type']].add(row['sample_id'])
        write_json(out / f'{a}_diagnostic_eligibility.json', result['eligibility'])
        write_jsonl(out / f'{a}_selected_ids.jsonl', [{key: row[key] for key in
                    ('problem_id', 'variant_type', 'sample_id', 'selection_score', 'ged_score')}
                                                   for row in result['selected']])
        available[a], fixed_selected[a] = {}, {}
        for pid, v in group_ids:
            g = pid, v
            candidates = eligibility.candidates([ged[a][sid] for sid in qualified_groups[g] if sid in ged[a]])
            shared = [ged[a][sid] for sid in common_groups[g]]
            available[a][g] = {r['sample_id'] for r in candidates}
            count_ok = eligibility.accepts(candidates); range_ok = range_check.accepts(candidates)
            is_eligible = count_ok and range_ok
            common_eligible = eligibility.accepts(shared) and range_check.accepts(shared)
            fixed_selected[a][g] = set(common_groups[g]) if common_eligible else set()
            require(selected[a][g] == (available[a][g] if is_eligible else set()), 'No-cutoff shared selector mismatch')
            scores = [r[score] for r in candidates]
            shared_scores = [r[score] for r in shared]
            decisions.append({'annotator': a, 'problem_id': pid, 'variant': v,
                              'candidate_count_before_ged': len(qualified_groups[g]),
                              'candidate_count': len(candidates), 'common_candidate_count': len(shared),
                              'ged_min': min(scores) if scores else None, 'ged_max': max(scores) if scores else None,
                              'ged_range': max(scores) - min(scores) if scores else None,
                              'common_ged_range': max(shared_scores) - min(shared_scores) if shared_scores else None,
                              'eligible': is_eligible, 'common_eligible': common_eligible,
                              'reason': 'eligible' if is_eligible else 'insufficient_successful_correct' if not count_ok else 'low_ged_range',
                              'selected_ids': sorted(selected[a][g]), 'available_ids': sorted(available[a][g]),
                              'common_selected_ids': sorted(fixed_selected[a][g])})
    decision_index = {(r['annotator'], r['problem_id'], r['variant']): r for r in decisions}
    ranking, actual, numeric = [], [], []
    for a, b in PAIRS:
        for pid, v in group_ids:
            g = pid, v; ids = common_groups[g]
            scores_a, scores_b = ([ged[x][sid][score] for sid in ids] for x in (a, b))
            ranking.append({'annotator_a': a, 'annotator_b': b, 'problem_id': pid, 'variant': v,
                            'n_candidates': len(ids), **kendall_tau_b(scores_a, scores_b)})
            da, db = (decision_index[x, pid, v] for x in (a, b))
            row = {'annotator_a': a, 'annotator_b': b, 'problem_id': pid, 'variant': v,
                   'eligible_a': da['eligible'], 'eligible_b': db['eligible'],
                   'eligibility_flip': float(da['eligible'] != db['eligible']),
                   'common_eligibility_flip': float(da['common_eligible'] != db['common_eligible']),
                   'reason_a': da['reason'], 'reason_b': db['reason']}
            for scope, left, right in (
                ('available', available[a][g], available[b][g]),
                ('fixed_pool', fixed_selected[a][g], fixed_selected[b][g]),
                ('actual', selected[a][g], selected[b][g]),
                ('eligible_groups', {pid} if da['eligible'] else set(), {pid} if db['eligible'] else set()),
            ):
                row.update({f'{scope}_{key}': value for key, value in set_agreement(left, right).items()})
            actual.append(row)
        for sid in sorted(common):
            left, right = ged[a][sid], ged[b][sid]
            numeric.append({'annotator_a': a, 'annotator_b': b, 'sample_id': sid,
                            'problem_id': left['problem_id'], 'variant': left['variant'],
                            'ged_a': left['ged'], 'ged_b': right['ged'],
                            'normalizer_a': left['ged_normalizer'], 'normalizer_b': right['ged_normalizer'],
                            'score_a': left[score], 'score_b': right[score],
                            'absolute_difference': abs(left[score] - right[score]),
                            'raw_absolute_difference': abs(left['ged'] - right['ged']),
                            'normalizer_absolute_difference': abs(left['ged_normalizer'] - right['ged_normalizer'])})
    summary, pooled = [], []
    boot = dict(bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    for a, b in PAIRS:
        for v in VARIANTS:
            base = {'annotator_a': a, 'annotator_b': b, 'variant': v}
            select = lambda rows: [r for r in rows if all(r[key] == value for key, value in base.items())]
            act = select(actual)
            for stage, rows, metrics in (
                ('ranking_auxiliary', select(ranking), ['kendall_tau_b']),
                ('eligibility', act, ['eligibility_flip', 'common_eligibility_flip']),
                ('numeric_auxiliary', select(numeric), ['absolute_difference', 'raw_absolute_difference', 'normalizer_absolute_difference']),
            ):
                summary.extend(dict(base, stage=stage, metric=metric, **stats) for metric, stats in
                               summarize_clustered(rows, metrics, **boot).items())
            for scope in ('available', 'fixed_pool', 'actual', 'eligible_groups'):
                pooled.append(dict(base, scope=scope,
                    **summarize_ratio_clustered(act, f'{scope}_intersection', f'{scope}_union', **boot),
                    both_empty=sum(r[f'{scope}_union'] == 0 for r in act)))
    groups = []
    for pid, v in group_ids:
        g = pid, v
        overlaps = [r['actual_jaccard'] for r in actual if r['problem_id'] == pid and r['variant'] == v
                    and r['actual_jaccard'] is not None]
        sets = [selected[a][g] for a in ANNOTATORS]
        groups.append({'problem_id': pid, 'variant': v, 'n_common_candidates': len(common_groups[g]),
                       'n_qualified_candidates': len(qualified_groups[g]),
                       'mean_pairwise_actual_jaccard': float(np.mean(overlaps)) if overlaps else None,
                       'defined_pairs': len(overlaps), 'intersection': len(set.intersection(*sets)),
                       'union': len(set.union(*sets)),
                       'eligible_annotators': [a for a in ANNOTATORS if decision_index[a, pid, v]['eligible']]})
    manifest['four_way_selection'] = {
        v: {'intersection': sum(g['intersection'] for g in groups if g['variant'] == v),
            'union': sum(g['union'] for g in groups if g['variant'] == v),
            'all_eligible_problems': sum(len(g['eligible_annotators']) == 4 for g in groups if g['variant'] == v)}
        for v in VARIANTS}
    manifest['analysis_code'] = [file_identity(path) for path in
        (Path(__file__), ROOT / 'src/data_analysis/measurement/selection.py',
         ROOT / 'src/data_analysis/measurement/reliability.py', ROOT / 'src/dspr_dataset/data_filter.py')]
    manifest['source_inputs_unchanged'] = all(file_identity(f['path']) == f for f in manifest['input_files'])
    require(manifest['source_inputs_unchanged'], 'Input changed during analysis')
    write_json(out / 'manifest.json', manifest)
    for name, rows in [('coverage', coverage), ('summary', summary), ('ranking_by_group', ranking),
                       ('selection_comparison_by_group', actual), ('selection_decisions', decisions),
                       ('ged_numeric_pairs', numeric), ('pooled_jaccard', pooled), ('four_way_groups', groups)]:
        csv_write(out / f'{name}.csv', rows)
        if name in ('summary', 'pooled_jaccard', 'coverage'):
            write_json(out / f'{name}.json', rows)
    write_jsonl(out / 'ged_failures.jsonl', [dict(row, annotator=a) for a in ANNOTATORS for row in failures[a].values()])
    write_cases(out, groups, dags, ged, failures, decision_index)
    plot_results(out, pooled)
    write_report(args, manifest, coverage, summary, pooled, decisions)
    print(json.dumps({'output': str(out), 'report': str(args.report), 'top_k': None,
                      'selected_counts': {a: sum(len(s) for s in selected[a].values()) for a in ANNOTATORS},
                      'inputs_unchanged': True}, ensure_ascii=False))


def write_cases(out, groups, dags, ged, failures, decisions):
    cases = []; text = ['# No-cutoff selection cases', '',
        'Retrospective extremes of mean pairwise final-set Jaccard on nonempty unions; both-empty pairs undefined.', '']
    for v in VARIANTS:
        candidates = [g for g in groups if g['variant'] == v and g['union']]
        for label, direction in [('lowest_agreement', 1), ('highest_agreement', -1)]:
            if not candidates:
                continue
            g = min(candidates, key=lambda x: (direction * x['mean_pairwise_actual_jaccard'], x['problem_id']))
            pid = g['problem_id']
            ids = sorted(sid for sid, (key, _, sample) in dags['deepseek'].items()
                         if key.problem_id == pid and key.variant == v and sample.get('correct') and sample.get('valid'))
            cases.append({'rule': label, **g, 'samples': {sid: {'problem': dags['deepseek'][sid][1]['problem'],
                'response': dags['deepseek'][sid][2]['response'], 'steps': dags['deepseek'][sid][2]['steps'],
                'annotations': {a: dags[a][sid][2]['dag_analysis'] for a in ANNOTATORS},
                'ged': {a: ged[a].get(sid) for a in ANNOTATORS},
                'failures': {a: failures[a].get(sid) for a in ANNOTATORS}}
                for sid in [f'{pid}_original_0', *ids]}})
            text += [f'## {pid} / {v}: {label}', '',
                     f"Mean pairwise Jaccard: {g['mean_pairwise_actual_jaccard']:.3f}; defined pairs: {g['defined_pairs']}/6.", '',
                     dags['deepseek'][ids[0]][1]['problem'], '',
                     '| Annotator | Available correct | GED range | Eligible | Retained IDs |',
                     '| --- | ---: | ---: | --- | --- |']
            for a in ANNOTATORS:
                d = decisions[a, pid, v]; spread = f"{d['ged_range']:.4f}" if d['ged_range'] is not None else 'NA'
                text.append(f"| {NAMES[a]} | {d['candidate_count']} | {spread} | {d['eligible']} | {', '.join(d['selected_ids'])} |")
            text.append('')
    write_jsonl(out / 'case_inputs.jsonl', cases)
    (out / 'cases.md').write_text('\n'.join(text), encoding='utf-8')


def plot_results(out, pooled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for directory in (Path('/mnt/c/Windows/Fonts'), Path('/usr/share/fonts/truetype/msttcorefonts')):
        if directory.is_dir():
            for path in directory.iterdir():
                if path.suffix.lower() == '.ttf' and 'times' in path.name.lower():
                    font_manager.fontManager.addfont(str(path))
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    pools = {(s['annotator_a'], s['annotator_b'], s['variant'], s['scope']): s for s in pooled}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), sharey=True)
    for j, (title, scope) in enumerate([('Eligible problem sets', 'eligible_groups'),
                                       ('Retained paths: common pool', 'fixed_pool'),
                                       ('Retained paths: actual pools', 'actual')]):
        ax = axes[j]
        for v, color, offset in [('simple', '#246b94', -.13), ('hard', '#bd6b2d', .13)]:
            for i, (a, b) in enumerate(PAIRS):
                s = pools[a, b, v, scope]
                ax.errorbar(s['mean'], i + offset, xerr=[[s['mean'] - s['ci_low']], [s['ci_high'] - s['mean']]],
                            fmt='o', color=color, capsize=2, label=v.capitalize() if i == 0 else None)
        ax.set_title(title); ax.set_xlim(0, 1); ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Jaccard with 95% CI')
    axes[0].set_yticks(range(len(PAIRS)), [f'{NAMES[a]}–{NAMES[b]}' for a, b in PAIRS]); axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', frameon=False)
    fig.text(.5, .025, 'No top-k cutoff: keep every correct, successfully scored candidate in eligible groups.\n'
             'Eligibility: at least 3 candidates and normalized GED range >= 0.10. Problem-cluster bootstrap.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'selection_consistency.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def estimate(row, percent=False):
    if row['mean'] is None:
        return 'NA'
    scale, decimals = (100, 1) if percent else (1, 3)
    return f"{row['mean'] * scale:.{decimals}f} [{row['ci_low'] * scale:.{decimals}f}, {row['ci_high'] * scale:.{decimals}f}]"


def write_report(args, manifest, coverage, summary, pooled, decisions):
    idx = {(s['annotator_a'], s['annotator_b'], s['variant'], s['stage'], s['metric']): s for s in summary}
    pools = {(s['annotator_a'], s['annotator_b'], s['variant'], s['scope']): s for s in pooled}
    output_link = os.path.relpath(args.output_dir, args.report.parent)
    p = manifest['selection_protocol']; n = len(manifest['problem_ids'])
    def span(v, scope):
        values = [pools[a, b, v, scope]['mean'] for a, b in PAIRS]
        return f'{min(values):.3f}–{max(values):.3f}'
    lines = ['# 4.1B：跨 annotator 筛选一致性（不使用 top-5）', '',
             '分析日期：2026-10-28。Qwen-2.5 生成 CoT；四个 annotator 为 DeepSeek、GPT、Haiku、Qwen。', '',
             '**按用户确认的当前实验口径：不对候选轨迹做 top-k 截断。** 上一版把旧 manifest 的 `top_k=5` 带入诊断，不能用其结果代表当前实验。',
             f"去掉截断后，最终保留轨迹的两两 Jaccard 为 Simple **{span('simple', 'actual')}**、Hard **{span('hard', 'actual')}**。",
             '当前筛选保留了较大比例的相同轨迹，Hard 仍有更多差异；这与 GED 排序是否稳定是不同问题。', '',
             '## 1. 固定协议与核验', '',
             f"使用 cheap screen 后全部 {n} 个题组、{manifest['sample_count']} 条固定 CoT，比较 {manifest['expected_ged_pairs']} 对 Simple/Hard–Original 图。",
             f"有效且正确候选为 {manifest['qualified_candidates']} 条；Original 只要求有效，不要求答案正确。来源分布：`{manifest['source_counts']}`。", '',
             f"筛选规则：保留正确且 GED 成功、非 timeout 的候选；每题每侧至少 {p['min_correct_samples']} 条；`{p['score_field']}` 的 max−min ≥ {p['ged_range_threshold']:.2f}；通过后保留该侧全部合格候选。",
             'Simple/Hard 独立判定 eligibility。既不从 Simple 取最低五条，也不从 Hard 取最高五条；样本排序不再决定进入监督集合的身份。',
             '使用 `data_filter.select_trajectories_by_ged(top_k=None)`；候选资格和极差仍调用唯一共享实现。旧 manifest 仅提供 score/count/range 参数；`top_k=5` 被用户确认的不截断口径覆盖，旧训练文件不改写。', '',
             f"GED 设置一致：{manifest['ged_options']['ged_timeout']:g} 秒硬超时、32 节点/64 边、role/text 权重各 1，anchor 与候选来自同一个 annotator。",
             'Encoder 为 all-MiniLM-L6-v2，revision `'+manifest['embedding']['resolved_revision']+'`，mean pooling、L2 normalization、256-token 上限。',
             f"置信区间为按 problem_id 聚类 bootstrap 的 percentile 95% CI，{args.bootstrap_samples:,} 次、seed={args.seed}；不把同题多轨迹当独立样本。", '',
             '逐条核对了题目、CoT、分割、正确性/有效性标签、DAG 合法性、DAG 文件哈希、失败账本、GED 代码与 encoder 配置。',
             '**溯源限制：** 四套 GED 记录的 correctness 文件哈希相同（`ee16eb4f…`），与本地现存文件（`e4aa2fa4…`）不同；全部成功结果的题干/CoT/correct 与冻结输入一致，失败记录标签也一致，但不宣称历史 correctness 文件字节一致。'
             '标注恢复预算与反馈有所不同，DeepSeek 历史响应版本多数未记录，详情在 manifest。', '',
             '## 2. 计算覆盖率', '',
             '| Annotator | 成功图对 / 5323 | 正确候选成功 / 4162 | 正确候选覆盖率 | 失败原因 |',
             '| --- | ---: | ---: | ---: | --- |']
    for c in coverage:
        if c['variant'] == 'all':
            lines.append(f"| {NAMES[c['annotator']]} | {c['success']} | {c['qualified_success']} | {c['qualified_coverage']:.2%} | {c['failure_counts']} |")
    lines += ['', f"四家共同成功的正确候选为 {len(manifest['common_qualified_ids'])}/{manifest['qualified_candidates']}（{len(manifest['common_qualified_ids'])/manifest['qualified_candidates']:.2%}）。",
              'GPT `179_hard_14`、Haiku `264_simple_8` 为 ged=null 的普通计算错误；保留在失败账本中，不重新计算或忽略其覆盖损失。', '',
              '## 3. 实际保留的题组与全部轨迹', '',
              '| Annotator | Simple eligible | Hard eligible | 题组 union | 题组 intersection | Simple 轨迹 | Hard 轨迹 | 总轨迹 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in ANNOTATORS:
        dd = [d for d in decisions if d['annotator'] == a]
        es = {v: {d['problem_id'] for d in dd if d['variant'] == v and d['eligible']} for v in VARIANTS}
        counts = {v: sum(len(d['selected_ids']) for d in dd if d['variant'] == v) for v in VARIANTS}
        lines.append(f"| {NAMES[a]} | {len(es['simple'])} | {len(es['hard'])} | {len(es['simple']|es['hard'])} | {len(es['simple']&es['hard'])} | {counts['simple']} | {counts['hard']} | {sum(counts.values())} |")
    lines += ['', '**Eligibility 与上一版完全相同。** Top-k 原本发生在 eligibility 判定之后，因此去掉截断只改变保留的轨迹集合，不改变哪些题组通过 count/range 条件。', '',
              '| Pair | Variant | Eligible 题组 Jaccard [CI] | Eligibility 翻转 /167 | 最终轨迹 Jaccard [CI] |',
              '| --- | --- | --- | ---: | --- |']
    for a, b in PAIRS:
        for v in VARIANTS:
            flip = idx[a, b, v, 'eligibility', 'eligibility_flip']
            lines.append(f"| {NAMES[a]}–{NAMES[b]} | {v} | {estimate(pools[a,b,v,'eligible_groups'])} | {round(flip['mean']*n)} | {estimate(pools[a,b,v,'actual'])} |")
    lines += ['', 'Jaccard=交集大小/并集大小。轨迹统计使用 sample ID，先汇集该侧全部题组再计算交并比；CI 以题组重采样并重算比值。双方均为空的组不人为贡献 1。',
              '题组计数相近不代表 ID 完全相同。Simple eligibility 翻转 10–21/167（6.0%–12.6%），Hard 翻转 21–38/167（12.6%–22.8%）。', '',
              '| Variant | 最终轨迹四方交集 | 最终轨迹四方并集 | 四方 Jaccard | 四家均 eligible 的题组 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for v, g in manifest['four_way_selection'].items():
        lines.append(f"| {v} | {g['intersection']} | {g['union']} | {g['intersection']/g['union']:.3f} | {g['all_eligible_problems']} |")
    lines += ['', '## 4. 区分计算覆盖与极差阈值', '',
              '列出筛选前可用正确候选的重叠；再把输入固定为四家共同成功候选，分别应用相同的 count/range 筛选；最后使用各家实际可用候选池。',
              '**共同池上双方均通过筛选时，全部候选都被保留，轨迹重叠必然为 100%。这是不截断策略的机械结果，不是 GED 数值或排序一致的证据。** 下表共同池指标保留 eligibility 不一致导致的集合差异。', '',
              '| Pair | Variant | 可用正确候选 Jaccard | 固定共同池筛选后 Jaccard [CI] | 实际候选池筛选后 Jaccard [CI] |',
              '| --- | --- | ---: | --- | --- |']
    for a, b in PAIRS:
        for v in VARIANTS:
            lines.append(f"| {NAMES[a]}–{NAMES[b]} | {v} | {pools[a,b,v,'available']['mean']:.3f} | {estimate(pools[a,b,v,'fixed_pool'])} | {estimate(pools[a,b,v,'actual'])} |")
    lines += ['', '共同池重算极差也会改变阈值判断；不同列之间的差值不能直接解释为超时的因果贡献。所有判断和失败原因均保存在逐题明细。', '',
              '## 5. 辅助诊断：GED 数值与排序', '',
              '当前实验不按排名截断，因此这些是辅助测量，不用它们替代实际筛选集合的一致性。使用同题同侧的共同候选，τ-b 保留平局；不足两条或全平局时不可定义。', '',
              '| Pair | Variant | τ-b 有效题组数 | Kendall τ-b [CI] | Mean absolute Δ normalized GED [CI] |',
              '| --- | --- | ---: | --- | --- |']
    for a, b in PAIRS:
        for v in VARIANTS:
            rank = idx[a,b,v,'ranking_auxiliary','kendall_tau_b']; numeric = idx[a,b,v,'numeric_auxiliary','absolute_difference']
            lines.append(f"| {NAMES[a]}–{NAMES[b]} | {v} | {rank['n']} | {estimate(rank)} | {estimate(numeric)} |")
    lines += ['', '排序统计不再使用 n>5 条件；Simple 有效 160 组、Hard 145 组，不可定义分别 7/22 组。与前版 n>5 表的分母不同。数值差异以候选等权，CI 按题组聚类。', '',
              '## 6. 正文解释与交付物', '',
              '**可支持的结论：** 当前不截断的筛选规则在不同 annotator 下保留了较大比例的相同轨迹；Simple 集合更一致，Hard 的题组资格与覆盖损失更敏感。',
              '**需撤回的套用：** 前版 top-5 最终集合 Jaccard 约 0.33–0.40 是额外截断策略的结果，不能据此否定当前实验的筛选稳定性。当前 Jaccard 上升符合保留全部合格候选的行为，不表示 GED 的排序变得更一致。',
              '结论限定于固定 CoT、分割及当前筛选规则下的跨 annotator 稳定性。部分图/题组仍有超时与阈值分歧；本次未执行重训或额外 GED 计算。', '',
              f'![不截断筛选一致性]({output_link}/selection_consistency.png)', '',
              f'[矢量 PDF]({output_link}/selection_consistency.pdf) · [完整统计]({output_link}/summary.csv) · [集合统计及 CI]({output_link}/pooled_jaccard.csv)',
              f'[逐题决策]({output_link}/selection_decisions.csv) · [案例]({output_link}/cases.md) · [输入与配置 manifest]({output_link}/manifest.json)', '',
              '案例按每侧实际最终集合平均两两 Jaccard 的最低/最高值取例，要求四方并集非空，双方空集合的配对不可定义；属于事后描述性案例，完整 CoT/DAG 在 case_inputs.jsonl 中。', '',
              '```bash', 'python scripts/analyze_selection_consistency.py', '```', '',
              '新的独立产物位于 `reliability_audit/n16_selection_no_topk/`；旧 top-5 诊断保留在 `n16_selection/` 供溯源。既有数据、eligibility、DAG、GED 和五折结果均未改写。']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'output/qwen-2.5/multiple_n16')
    parser.add_argument('--eligibility-config', type=Path, default=ROOT / 'output/qwen-2.5/eligible_problem.json')
    parser.add_argument('--canonical', type=Path, default=ROOT / 'data/canonical_math_paired.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/qwen-2.5/reliability_audit/n16_selection_no_topk')
    parser.add_argument('--report', type=Path, help='Per-run report path (default: OUTPUT_DIR/report.md)')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.report is None:
        args.report = args.output_dir / 'report.md'
    if args.bootstrap_samples < 1:
        parser.error('--bootstrap-samples must be positive')
    analyze(args)


if __name__ == '__main__':
    main()
