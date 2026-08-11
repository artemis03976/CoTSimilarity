"""Analyze GED between baseline-original and variant responses.

This is the bridge between DAG analysis and DSPR dataset construction. For each
problem, it compares simple/hard response DAGs against an original-problem
reference DAG, joins answer correctness labels from all_records*.jsonl, and
writes one flat GED record per sampled response.
"""

import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import sys

import networkx as nx

try:
    from data_analysis.dag_similarity import (
        compute_ged_similarity,
        extract_dag_from_batch_response
    )
    from data_analysis.dag_compressor import compress_dag_combined, build_digraph_with_tags
    from data_analysis.graph_cache import build_graph_cache, load_graph_cache
    from data_analysis.text_similarity import (
        DEFAULT_TEXT_SIMILARITY_MODEL,
        TextSimilarityEncoder,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_analysis.dag_similarity import (
        compute_ged_similarity,
        extract_dag_from_batch_response
    )
    from data_analysis.dag_compressor import compress_dag_combined, build_digraph_with_tags
    from data_analysis.graph_cache import build_graph_cache, load_graph_cache
    from data_analysis.text_similarity import (
        DEFAULT_TEXT_SIMILARITY_MODEL,
        TextSimilarityEncoder,
    )


VARIANTS = ("original", "simple", "hard")


def _attach_original_step_text(dag: List[Dict], steps: List[Dict]) -> List[Dict]:
    """Associate DAG nodes with their original segmented CoT spans."""
    step_texts = {
        str(step.get("index")): step.get("text", "")
        for step in steps
        if step.get("index") is not None
    }
    enriched = []
    for entry in dag:
        node = dict(entry)
        original_text = step_texts.get(str(entry.get("step_id")), "")
        # Legacy merged files may not preserve segmentation. In that case the
        # annotator-written analysis is the only available textual span.
        node["text"] = original_text or entry.get("text") or entry.get("analysis", "")
        enriched.append(node)
    return enriched


def load_dag_records(path: str) -> Dict[str, List[Dict]]:
    """Load DAG annotations from either supported on-disk representation.

    Historical experiments stored the provider's batch responses directly,
    with one top-level ``custom_id`` per trajectory.  The current
    ``merge-batch`` pipeline stores DAG annotations inside each problem's
    variant/sample hierarchy.  Normalize both representations to the same
    ``custom_id -> dag_analysis`` lookup used by GED computation.
    """

    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    dag_records: Dict[str, List[Dict]] = {}
    for record in records:
        custom_id = record.get('custom_id')
        if custom_id is not None:
            dag = extract_dag_from_batch_response(record)
            if dag:
                dag_records[str(custom_id)] = dag
            continue

        problem_id = record.get('problem_id', record.get('id'))
        if problem_id is None:
            continue
        for variant in VARIANTS:
            samples = record.get(variant, {}).get('samples', [])
            for sample_num, sample in enumerate(samples):
                dag = sample.get('dag_analysis')
                if isinstance(dag, list) and dag:
                    dag_records[f"{problem_id}_{variant}_{sample_num}"] = (
                        _attach_original_step_text(dag, sample.get("steps", []))
                    )

    return dag_records


def load_correctness_data(path: str) -> Dict[int, Dict]:
    """Load correctness labels and raw responses from all_records_50.jsonl.

    The DAG analysis files only contain dependency annotations. We keep this
    separate lookup so the final GED records can include both structure metrics
    and whether the sampled answer was correct.
    """
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


def get_original_graph(records: Dict, problem_id: int):
    """Find original problem graph.

    Supports both legacy custom_id (`<pid>_original`) and current batch style
    custom_id (`<pid>_original_0`).
    """
    target_ids = {f"{problem_id}_original_0", f"{problem_id}_original"}
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
) -> List[Dict]:
    """Analyze GED for one problem's simple/hard variants.

    The original graph is compressed once and reused as the reference. Each
    variant sample graph is also compressed before GED so comparisons focus on
    macro reasoning structure rather than repeated same-tag micro steps.
    """

    original_graph = None
    if original_records:
        original_graph = get_original_graph(original_records, problem_id)
    if original_graph is None:
        # Default path for the current pipeline: use original_0 from the same
        # analyzed file that also contains simple/hard samples.
        original_graph = get_original_graph(variant_records, problem_id)

    if not original_graph:
        print(f"No original graph found for problem {problem_id}")
        return []

    if not original_graph.graph.get("ged_prepared", False):
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
                sample_graph_compressed = sample.get('graph')
                if sample_graph_compressed is None:
                    sample_graph = build_digraph_with_tags(sample['dag'])
                    sample_graph_compressed, _ = compress_dag_combined(sample_graph)
            except Exception as exc:
                print(f"Warning: Skip {sample['custom_id']} due to graph build/compression error: {exc}")
                continue

            try:
                ged_result = compute_ged_similarity(
                    original_graph,
                    sample_graph_compressed,
                    lambda_role=lambda_role,
                    lambda_text=lambda_text,
                )
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
                'ged_normalized': ged_result['ged_normalized'],
                'ged_normalizer': ged_result['ged_normalizer'],
                'similarity_normalized': ged_result['similarity_normalized'],
                'timed_out': ged_result['timed_out']
            })

    return results


def write_results_to_csv(results: List[Dict], problem_id: int, output_dir: str):
    """Write results and summary statistics to CSV."""
    output_path = Path(output_dir) / f"{problem_id}_ged_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'problem_id', 'sample_id', 'variant', 'problem', 'response',
                'correct', 'ged', 'ged_normalized', 'ged_normalizer',
                'similarity_normalized', 'timed_out',
            ],
        )
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
                        help='Model output root containing all_records*.jsonl and dag_analysis* directories')
    parser.add_argument('--original-records', type=str, default=None,
                        help='Optional original-problem DAG analysis JSONL path; highest priority')
    parser.add_argument('--baseline-output-root', type=str, default=None,
                        help='Optional baseline model output root, e.g. output/qwen-2.5, used to load original DAGs as GED references')
    parser.add_argument('--baseline-original-records', type=str, default=None,
                        help='Optional baseline original DAG analysis JSONL path; overrides --baseline-output-root inference')
    parser.add_argument('--variant-records', type=str, default=None,
                        help=(
                            'Merged DAG analysis JSONL (legacy raw batch responses are also '
                            'accepted); default: <output-root>/dag_analysis_50/analyzed_records.jsonl'
                        ))
    parser.add_argument('--correctness-file', type=str, default=None,
                        help='Answer correctness JSONL path; default: <output-root>/all_records_50.jsonl')
    parser.add_argument('--all-results-output', type=str, default=None,
                        help='GED summary JSONL output path; default: <output-root>/all_ged_results.jsonl')
    parser.add_argument('--problem-id', type=int, help='Specific problem ID to analyze (default: all 279)')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples per variant (max 50)')
    parser.add_argument('--save-csv', action='store_true', help='Save individual CSV files per problem')
    parser.add_argument(
        '--graph-cache',
        type=str,
        default=None,
        help=(
            'Compressed graph + text-embedding cache; default: '
            '<output-root>/ged_graph_cache.pt (or a problem-specific cache)'
        ),
    )
    parser.add_argument(
        '--rebuild-graph-cache',
        action='store_true',
        help='Ignore and overwrite an existing graph cache',
    )
    parser.add_argument(
        '--prepare-graph-cache-only',
        action='store_true',
        help='Build/load the prepared graph cache and exit before GED',
    )
    parser.add_argument(
        '--text-similarity-model',
        type=str,
        default=DEFAULT_TEXT_SIMILARITY_MODEL,
        help='Hugging Face encoder used for compressed-node text similarity',
    )
    parser.add_argument(
        '--text-similarity-revision',
        type=str,
        default=None,
        help='Optional pinned Hugging Face model revision',
    )
    parser.add_argument(
        '--embedding-device',
        type=str,
        default='auto',
        help='Embedding device, e.g. auto, cuda, cuda:0, or cpu',
    )
    parser.add_argument('--embedding-batch-size', type=int, default=64)
    parser.add_argument('--embedding-max-length', type=int, default=256)
    parser.add_argument('--lambda-role', type=float, default=1.0)
    parser.add_argument('--lambda-text', type=float, default=1.0)
    args = parser.parse_args()

    if args.lambda_role < 0 or args.lambda_text < 0:
        parser.error('--lambda-role and --lambda-text must be non-negative')

    output_root = Path(args.output_root) if args.output_root else None

    if args.correctness_file:
        correctness_path = Path(args.correctness_file)
    else:
        if output_root is None:
            raise ValueError("Provide --output-root or --correctness-file")
        correctness_path = output_root / "all_records_50.jsonl"

    if output_root is None:
        output_root = correctness_path.parent

    if args.variant_records:
        variant_records_path = Path(args.variant_records)
    else:
        variant_records_path = output_root / "dag_analysis_50" / "analyzed_records.jsonl"

    all_results_path = Path(args.all_results_output) if args.all_results_output else output_root / "all_ged_results.jsonl"

    # Resolve baseline-original source for GED reference graph.
    baseline_output_root = Path(args.baseline_output_root) if args.baseline_output_root else None

    baseline_original_records_path = None
    if args.baseline_original_records:
        baseline_original_records_path = Path(args.baseline_original_records)
    elif baseline_output_root is not None:
        candidate_a = baseline_output_root / "dag_analysis" / "analyzed_records.jsonl"
        candidate_b = baseline_output_root / "dag_analysis_50" / "analyzed_records.jsonl"
        if candidate_a.exists():
            baseline_original_records_path = candidate_a
        elif candidate_b.exists():
            baseline_original_records_path = candidate_b

    explicit_original_records_path = None
    if args.original_records:
        explicit_original_records_path = Path(args.original_records)
    elif baseline_original_records_path:
        explicit_original_records_path = baseline_original_records_path

    correctness_data = load_correctness_data(str(correctness_path))

    if args.graph_cache:
        graph_cache_path = Path(args.graph_cache)
    elif args.problem_id is not None:
        graph_cache_path = output_root / f"ged_graph_cache_problem_{args.problem_id}.pt"
    else:
        graph_cache_path = output_root / "ged_graph_cache.pt"

    if graph_cache_path.is_file() and not args.rebuild_graph_cache:
        print(f"Loading prepared compressed graphs from: {graph_cache_path}")
        variant_records, original_records, cache_metadata = load_graph_cache(
            graph_cache_path
        )
        print(
            f"Loaded {len(variant_records)} trajectory graphs and "
            f"{len(original_records)} explicit reference graphs without "
            "initializing the embedding model."
        )
        embedding_metadata = cache_metadata.get("embedding", {})
        print(
            "Cached text similarity: "
            f"{embedding_metadata.get('model_name')} @ "
            f"{embedding_metadata.get('resolved_revision') or embedding_metadata.get('requested_revision') or 'default revision'}"
        )
    else:
        print("Loading DAG annotations for graph-cache preparation...")
        variant_dag_records = load_dag_records(str(variant_records_path))
        print(
            f"Loaded {len(variant_dag_records)} trajectory DAGs from "
            f"{variant_records_path}"
        )
        if explicit_original_records_path is not None:
            original_dag_records = load_dag_records(
                str(explicit_original_records_path)
            )
            print(
                f"Loaded {len(original_dag_records)} optional reference DAGs "
                f"from {explicit_original_records_path}"
            )
        else:
            original_dag_records = None
            print(
                "No explicit baseline original records provided; fallback to "
                "original_0 from variant records as GED baseline."
            )

        print(
            f"Loading text-similarity encoder {args.text_similarity_model} "
            f"on {args.embedding_device}..."
        )
        encoder = TextSimilarityEncoder(
            args.text_similarity_model,
            revision=args.text_similarity_revision,
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
        )
        source_metadata = {
            "variant_records": str(variant_records_path.resolve()),
            "original_records": (
                str(explicit_original_records_path.resolve())
                if explicit_original_records_path is not None
                else None
            ),
        }
        selected_problem_ids = [args.problem_id] if args.problem_id is not None else None
        variant_records, original_records, cache_metadata = build_graph_cache(
            variant_dag_records,
            original_dag_records,
            graph_cache_path,
            encoder,
            problem_ids=selected_problem_ids,
            source_metadata=source_metadata,
        )
        del encoder
        del variant_dag_records
        del original_dag_records

    cached_problem_ids = cache_metadata.get("problem_ids")
    if args.problem_id is None and cached_problem_ids is not None:
        raise ValueError(
            "The selected graph cache only covers specific problem IDs. "
            "Use the matching --problem-id or rebuild the full cache."
        )
    if (
        args.problem_id is not None
        and cached_problem_ids is not None
        and args.problem_id not in cached_problem_ids
    ):
        raise ValueError(
            f"Graph cache does not contain requested problem {args.problem_id}"
        )

    if args.prepare_graph_cache_only:
        print("Graph-cache preparation complete; skipping GED as requested.")
        return

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
            results = analyze_problem(
                problem_id,
                args.num_samples,
                original_records,
                variant_records,
                correctness_data,
                lambda_role=args.lambda_role,
                lambda_text=args.lambda_text,
            )
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
        results_config_path = all_results_path.with_suffix('.config.json')
        with open(results_config_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'graph_cache': str(graph_cache_path.resolve()),
                    'lambda_role': args.lambda_role,
                    'lambda_text': args.lambda_text,
                    'graph_cache_metadata': cache_metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nAll GED results saved to {all_results_path}")
        print(f"GED run configuration saved to {results_config_path}")
        print(f"Total samples processed: {len(all_results)}")
        print(f"\nTo generate DSPR dataset, run:")
        print(f"  python src/dspr_dataset/data_filter.py --input {all_results_path}")
    else:
        print("No results generated")


if __name__ == '__main__':
    main()
