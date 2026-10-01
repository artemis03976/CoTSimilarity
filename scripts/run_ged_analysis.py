#!/usr/bin/env python3
"""CLI and execution for the unchanged DAG -> compression/embedding -> GED chain."""

import argparse
import multiprocessing as mp
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "src" / "data_analysis"
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.artifacts import file_identity, fingerprint  # noqa: E402
from utils.io import write_json, write_jsonl  # noqa: E402
from data_analysis.measurement.checkpoints import load_problem_checkpoint, problem_status, write_problem_checkpoint  # noqa: E402
from data_analysis.measurement.analysis import analyze_problem  # noqa: E402
from data_analysis.measurement.export import write_results_to_csv  # noqa: E402
from data_analysis.graph.cache import build_graph_cache, load_graph_cache  # noqa: E402
from data_analysis.records import load_correctness_data, load_dag_records  # noqa: E402
from data_analysis.metrics.text import DEFAULT_TEXT_SIMILARITY_MODEL, TextSimilarityEncoder  # noqa: E402


_WORKER_STATE = {}


def _init_worker(original_records, variant_records, correctness_data, options):
    global _WORKER_STATE
    _WORKER_STATE = {
        "original_records": original_records,
        "variant_records": variant_records,
        "correctness_data": correctness_data,
        "options": options,
    }


def _run_problem(problem_id):
    stats = {}
    try:
        results = analyze_problem(
            problem_id,
            original_records=_WORKER_STATE["original_records"],
            variant_records=_WORKER_STATE["variant_records"],
            correctness_data=_WORKER_STATE["correctness_data"],
            stats=stats,
            **_WORKER_STATE["options"],
        )
    except Exception as exc:
        stats["problem_error"] = 1
        print(f"Warning: Problem {problem_id} failed: {exc}")
        results = []
    return problem_id, results, stats


def merge_stats(target, source):
    for key, count in source.items():
        target[key] = target.get(key, 0) + count


def parse_args(argv=None):
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
    parser.add_argument(
        '--ged-timeout', type=float, default=10.0,
        help='Maximum wall-clock seconds for one GED pair (default: 10)',
    )
    parser.add_argument(
        '--max-ged-nodes', type=int, default=32,
        help='Skip a pair when either graph exceeds this node count; 0 disables',
    )
    parser.add_argument(
        '--max-ged-edges', type=int, default=64,
        help='Skip a pair when either graph exceeds this edge count; 0 disables',
    )
    parser.add_argument(
        '--soft-timeout', action='store_true',
        help='Use NetworkX cooperative timeout instead of POSIX hard wall-clock cutoff',
    )
    parser.add_argument(
        '--keep-timeouts', action='store_true',
        help='Keep timeout GED values when NetworkX returned a best-so-far value',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel problem workers on POSIX (default: 1)',
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Per-problem checkpoint directory; defaults beside the output JSONL',
    )
    parser.add_argument(
        '--no-checkpoint', action='store_true',
        help='Disable per-problem checkpoint files for this run',
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Reuse completed problem checkpoints and process only pending problems',
    )
    args = parser.parse_args(argv)

    if args.lambda_role < 0 or args.lambda_text < 0:
        parser.error('--lambda-role and --lambda-text must be non-negative')
    if not all(math.isfinite(value) for value in (args.lambda_role, args.lambda_text, args.ged_timeout)):
        parser.error('GED weights and timeout must be finite')
    if args.ged_timeout <= 0:
        parser.error('--ged-timeout must be positive')
    if args.max_ged_nodes < 0 or args.max_ged_edges < 0:
        parser.error('--max-ged-nodes and --max-ged-edges must be non-negative')
    if args.workers <= 0:
        parser.error('--workers must be positive')
    if args.num_samples <= 0:
        parser.error('--num-samples must be positive')
    if args.no_checkpoint and args.resume:
        parser.error('--resume cannot be combined with --no-checkpoint')
    # A zero limit is a convenient CLI spelling for an unbounded limit.
    args.max_ged_nodes = args.max_ged_nodes or None
    args.max_ged_edges = args.max_ged_edges or None

    return args


def resolve_paths(args):
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

    return output_root, correctness_path, variant_records_path, all_results_path, explicit_original_records_path


def prepare_graphs(args, output_root, variant_records_path, explicit_original_records_path):
    source_metadata = {
        "variant_records": file_identity(variant_records_path),
        "original_records": file_identity(explicit_original_records_path) if explicit_original_records_path else None,
    }
    preparation = {
        "sources": source_metadata,
        "embedding": {
            "model": args.text_similarity_model,
            "revision": args.text_similarity_revision,
            "max_length": args.embedding_max_length,
        },
        "code": {
            name: file_identity(ANALYSIS_ROOT / name)["sha256"]
            for name in ("graph/compression.py", "graph/builder.py", "annotation/parsing.py", "records.py", "metrics/text.py")
        },
    }
    signature = fingerprint(preparation)
    if args.graph_cache:
        graph_cache_path = Path(args.graph_cache)
    elif args.problem_id is not None:
        graph_cache_path = output_root / f"ged_graph_cache_problem_{args.problem_id}.pt"
    else:
        graph_cache_path = output_root / "ged_graph_cache.pt"

    if graph_cache_path.is_file() and not args.rebuild_graph_cache:
        print(f"Loading prepared compressed graphs from: {graph_cache_path}")
        variant_records, original_records, cache_metadata = load_graph_cache(
            graph_cache_path, expected_signature=signature
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
        selected_problem_ids = [args.problem_id] if args.problem_id is not None else None
        variant_records, original_records, cache_metadata = build_graph_cache(
            variant_dag_records,
            original_dag_records,
            graph_cache_path,
            encoder,
            problem_ids=selected_problem_ids,
            source_metadata=source_metadata,
            signature=signature,
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

    return variant_records, original_records, cache_metadata, graph_cache_path


def main(argv=None):
    args = parse_args(argv)
    output_root, correctness_path, variant_path, results_path, original_path = resolve_paths(args)
    correctness_data = load_correctness_data(str(correctness_path))
    variants, originals, cache_metadata, cache_path = prepare_graphs(
        args, output_root, variant_path, original_path
    )
    if args.prepare_graph_cache_only:
        print("Graph-cache preparation complete; skipping GED as requested.")
        return

    problem_ids = [args.problem_id] if args.problem_id is not None else sorted(correctness_data)
    options = {
        "num_samples": args.num_samples,
        "lambda_role": args.lambda_role,
        "lambda_text": args.lambda_text,
        "ged_timeout": args.ged_timeout,
        "max_ged_nodes": args.max_ged_nodes,
        "max_ged_edges": args.max_ged_edges,
        "hard_timeout": not args.soft_timeout,
        "skip_timeouts": not args.keep_timeouts,
    }
    identity = {
        "graph_cache": file_identity(cache_path),
        "correctness": file_identity(correctness_path),
        "options": options,
        "code": {
            name: file_identity(ANALYSIS_ROOT / name)["sha256"]
            for name in ("measurement/analysis.py", "metrics/ged.py")
        },
    }
    signature = fingerprint(identity)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else results_path.parent / f"{results_path.stem}_checkpoints"
    results = []
    completed_ids = set()
    failed_ids = set()
    run_stats = {}
    if not args.no_checkpoint and args.resume:
        for problem_id in problem_ids:
            checkpoint = load_problem_checkpoint(checkpoint_dir, problem_id, signature)
            if checkpoint is not None:
                rows, stats = checkpoint
                results.extend(rows)
                merge_stats(run_stats, stats)
                completed_ids.add(problem_id)

    pending = [pid for pid in problem_ids if pid not in completed_ids]
    print(f"Analyzing {len(pending)} pending problems; {len(completed_ids)} restored.")
    workers = args.workers if pending else 1
    if workers > 1 and "fork" not in mp.get_all_start_methods():
        print("Warning: fork is unavailable; falling back to serial GED analysis.")
        workers = 1

    def consume(problem_id, rows, stats):
        results.extend(rows)
        merge_stats(run_stats, stats)
        if problem_status(rows, stats) == "failed":
            failed_ids.add(problem_id)
        else:
            completed_ids.add(problem_id)
        if not args.no_checkpoint:
            write_problem_checkpoint(checkpoint_dir, problem_id, rows, stats, signature)
        if args.save_csv and rows:
            write_results_to_csv(rows, problem_id, str(output_root / "similarity_check"))
        print(f"Processed problem {problem_id} ({len(completed_ids) + len(failed_ids)}/{len(problem_ids)}); kept {len(rows)} rows")

    state = (originals, variants, correctness_data, options)
    try:
        if workers == 1:
            _init_worker(*state)
            for problem_id in pending:
                consume(*_run_problem(problem_id))
        else:
            with mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=state) as pool:
                for problem_id, rows, stats in pool.imap_unordered(_run_problem, pending):
                    consume(problem_id, rows, stats)
    finally:
        _WORKER_STATE.clear()

    results.sort(key=lambda item: (item.get("problem_id", 0), item.get("variant", ""), item.get("sample_id", "")))
    write_jsonl(results_path, results)
    write_json(results_path.with_suffix(".config.json"), {
        "signature": signature,
        "inputs": identity,
        "graph_cache": str(cache_path.resolve()),
        **{key: value for key, value in options.items() if key != "num_samples"},
        "num_samples": args.num_samples,
        "workers": args.workers,
        "effective_workers": workers,
        "checkpoint_dir": str(checkpoint_dir.resolve()) if not args.no_checkpoint else None,
        "checkpoint_enabled": not args.no_checkpoint,
        "resumed": args.resume,
        "problem_ids": problem_ids,
        "completed_problem_ids": sorted(completed_ids),
        "failed_problem_ids": sorted(failed_ids),
        "run_stats": run_stats,
        "graph_cache_metadata": cache_metadata,
    })
    print(f"All GED results saved to {results_path}; kept {len(results)} rows")
    print(f"GED run statistics: {run_stats}")


if __name__ == "__main__":
    main()
