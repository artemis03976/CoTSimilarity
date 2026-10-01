"""Compare each simple/hard trajectory with the original reference graph."""

from typing import Dict, List, Optional

import networkx as nx

from utils.artifacts import response_hash
from data_analysis.graph.compression import compress_dag_combined
from data_analysis.graph.builder import build_digraph_with_tags
from data_analysis.metrics.ged import compute_ged_similarity
from data_analysis.records import iter_samples


def get_original_graph(records: Dict, problem_id: int):
    """Find original problem graph.

    Supports both legacy custom_id (`<pid>_original`) and current batch style
    custom_id (`<pid>_original_0`).
    """
    target_ids = (f"{problem_id}_original_0", f"{problem_id}_original")
    for custom_id in target_ids:
        record = records.get(custom_id)
        if isinstance(record, nx.DiGraph):
            return record
        if record:
            return build_digraph_with_tags(record)
    print(f"Warning: original graph not found for problem {problem_id} in {len(records)} DAG records")
    return None


def get_variant_samples(
    records: Dict,
    problem_id: int,
    variant: str,
    num_samples: int,
) -> List[Dict]:
    """Get DAG-analyzed samples for one variant from dag_analysis_50 output."""
    samples = []
    for custom_id, record in records.items():
        if custom_id.startswith(f"{problem_id}_{variant}_"):
            try:
                sample_num = int(custom_id.rsplit('_', 1)[-1])
            except ValueError:
                continue
            sample = {'sample_num': sample_num, 'custom_id': custom_id}
            if isinstance(record, nx.DiGraph):
                sample['graph'] = record
            else:
                sample['dag'] = record
            samples.append(sample)
    samples.sort(key=lambda x: x['sample_num'])
    return samples[:num_samples]


def analyze_problem(
    problem_id: int,
    num_samples: int,
    original_records: Optional[Dict],
    variant_records: Dict,
    correctness_data: Dict,
    *,
    lambda_role: float = 1.0,
    lambda_text: float = 1.0,
    ged_timeout: float = 10.0,
    max_ged_nodes: Optional[int] = 32,
    max_ged_edges: Optional[int] = 64,
    hard_timeout: bool = True,
    skip_timeouts: bool = True,
    stats: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """Analyze GED for one problem's simple/hard variants.

    The original graph is compressed once and reused as the reference. Each
    variant sample graph is also compressed before GED so comparisons focus on
    macro reasoning structure rather than repeated same-tag micro steps.
    """

    if stats is None:
        stats = {}

    original_graph = None
    if original_records:
        original_graph = get_original_graph(original_records, problem_id)
    if original_graph is None:
        # Default path for the current pipeline: use original_0 from the same
        # analyzed file that also contains simple/hard samples.
        original_graph = get_original_graph(variant_records, problem_id)

    if not original_graph:
        stats["missing_original"] = stats.get("missing_original", 0) + 1
        print(f"No original graph found for problem {problem_id}")
        return []

    if not original_graph.graph.get("ged_prepared", False):
        try:
            original_graph, _ = compress_dag_combined(original_graph)
        except Exception as exc:
            stats["original_compression_error"] = stats.get("original_compression_error", 0) + 1
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
        correct_labels = {
            str(key): sample
            for key, _, sample in iter_samples([{'problem_id': problem_id, variant: variant_info}], [variant])
        }

        for sample_idx, sample in enumerate(samples, start=1):
            processed_samples += 1
            print(
                f"[{processed_samples}/{total_samples}] "
                f"Computing GED for {sample['custom_id']} "
                f"({variant} {sample_idx}/{len(samples)})"
            )
            try:
                sample_graph_compressed = sample.get('graph')
                if sample_graph_compressed is None:
                    sample_graph = build_digraph_with_tags(sample['dag'])
                    sample_graph_compressed, _ = compress_dag_combined(sample_graph)
            except Exception as exc:
                stats["graph_error"] = stats.get("graph_error", 0) + 1
                print(f"Warning: Skip {sample['custom_id']} due to graph build/compression error: {exc}")
                continue

            try:
                ged_result = compute_ged_similarity(
                    original_graph,
                    sample_graph_compressed,
                    timeout=ged_timeout,
                    lambda_role=lambda_role,
                    lambda_text=lambda_text,
                    max_nodes=max_ged_nodes,
                    max_edges=max_ged_edges,
                    hard_timeout=hard_timeout,
                )
            except Exception as exc:
                stats["error"] = stats.get("error", 0) + 1
                print(f"Warning: Skip {sample['custom_id']} due to GED computation error: {exc}")
                continue

            status = ged_result.get("status", "ok")
            if status != "ok":
                stats[status] = stats.get(status, 0) + 1
                if status == "timeout" and not skip_timeouts and ged_result.get("ged") is not None:
                    pass
                else:
                    print(f"Warning: Skip {sample['custom_id']} because GED status is {status}")
                    continue

            if ged_result.get('ged') is None:
                stats["invalid"] = stats.get("invalid", 0) + 1
                print(f"Warning: Skip {sample['custom_id']} because GED result is invalid")
                continue

            label = correct_labels.get(sample['custom_id'])
            if label is None:
                stats['sample_mismatch'] = stats.get('sample_mismatch', 0) + 1
                print(f"Warning: Missing correctness sample {sample['custom_id']}")
                continue
            response = label.get('response', '')
            expected_hash = sample_graph_compressed.graph.get('response_hash')
            if expected_hash and expected_hash != response_hash(response):
                stats['sample_mismatch'] = stats.get('sample_mismatch', 0) + 1
                print(f"Warning: Response changed for sample {sample['custom_id']}")
                continue
            correct = label.get('correct')

            results.append({
                'problem_id': problem_id,
                'sample_id': sample['custom_id'],
                'variant': variant,
                'problem': variant_info.get('problem', ''),
                'response': response,
                'correct': correct,
                'ged': ged_result['ged'],
                'ged_normalized': ged_result['ged_normalized'],
                'ged_normalizer': ged_result['ged_normalizer'],
                'similarity_normalized': ged_result['similarity_normalized'],
                'timed_out': ged_result['timed_out'],
                'status': status,
                'approximate': ged_result.get('approximate', False),
                'elapsed_seconds': ged_result.get('elapsed_seconds'),
                'ged_max_nodes': ged_result.get('max_nodes'),
                'ged_max_edges': ged_result.get('max_edges'),
            })
            stats['kept'] = stats.get('kept', 0) + 1

    return results
