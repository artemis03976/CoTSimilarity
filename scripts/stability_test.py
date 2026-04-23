"""Test the stability of DAG analysis by running multiple iterations on the same sample.

This script:
1. Randomly samples records from segmented_records.jsonl
2. Runs DAG analysis N times on each sample (with different random seeds)
3. Computes pairwise similarity between all iterations using dag_similarity
4. Reports average similarity to assess LLM output stability
"""

import json
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict
import statistics

# Import from existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_analysis.llm.config import LLMConfig
from data_analysis.llm.api_client import LLMClient
from data_analysis.dag_similarity import build_digraph, compute_ged_similarity, compute_feature_similarity

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "output/qwen/segmented_records.jsonl"
DEFAULT_OUTPUT = "output/dag_analysis/stability_test"


def load_segmented_records(input_path: str, limit: int = None) -> List[Dict]:
    """Load segmented records from JSONL."""
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def analyze_single_variant(
    problem: str,
    steps: List[Dict],
    client: LLMClient,
    variant_name: str = "test"
) -> Dict:
    """Run DAG analysis on a single variant."""
    try:
        dag_analysis, error = client.analyze_reasoning_chain(problem, steps)
        if error:
            logger.error(f"Analysis failed for {variant_name}: {error}")
            return {
                "variant": variant_name,
                "problem": problem,
                "steps": steps,
                "dag_analysis": None,
                "success": False,
                "error": error,
            }
        return {
            "variant": variant_name,
            "problem": problem,
            "steps": steps,
            "dag_analysis": dag_analysis,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Analysis failed for {variant_name}: {e}")
        return {
            "variant": variant_name,
            "problem": problem,
            "steps": steps,
            "dag_analysis": None,
            "success": False,
            "error": str(e),
        }


def compute_pairwise_similarities(
    iterations: List[Dict],
    timeout: float = 15.0,
    skip_ged: bool = False,
) -> Dict:
    """Compute pairwise similarities between all iterations.

    Returns:
        {
            "ged_similarities": [...],
            "feature_similarities": [...],
            "ged_values": [...],
            "failed_pairs": int,
        }
    """
    n = len(iterations)
    ged_sims = []
    feat_sims = []
    ged_vals = []
    failed = 0

    for i in range(n):
        for j in range(i + 1, n):
            dag1 = iterations[i].get("dag_analysis")
            dag2 = iterations[j].get("dag_analysis")

            if not dag1 or not dag2:
                failed += 1
                continue

            try:
                G1 = build_digraph(dag1, exclude_external=False)
                G2 = build_digraph(dag2, exclude_external=False)

                # Always compute feature similarity (fast)
                feat_sim = compute_feature_similarity(G1, G2)
                feat_sims.append(feat_sim)

                # Optionally compute GED
                if not skip_ged:
                    result = compute_ged_similarity(G1, G2, timeout=timeout)
                    if result["ged"] is not None:
                        ged_vals.append(result["ged"])
                        if result["similarity_normalized"] is not None:
                            ged_sims.append(result["similarity_normalized"])
            except Exception as e:
                logger.warning(f"Similarity computation failed for pair ({i},{j}): {e}")
                failed += 1

    return {
        "ged_similarities": ged_sims,
        "feature_similarities": feat_sims,
        "ged_values": ged_vals,
        "failed_pairs": failed,
        "total_pairs": n * (n - 1) // 2,
    }


def run_stability_test(
    record: Dict,
    num_iterations: int,
    client: LLMClient,
    variant_key: str = "original",
) -> Dict:
    """Run stability test on a single record with multiple iterations."""
    variant = record.get(variant_key, {})
    problem = variant.get("problem", "")
    steps = variant.get("steps", [])

    if not problem or not steps:
        logger.error(f"Record {record.get('problem_id')} missing {variant_key} data")
        return None

    logger.info(f"Running {num_iterations} iterations on problem_id={record['problem_id']}, variant={variant_key}")

    iterations = []
    for i in range(num_iterations):
        logger.info(f"  Iteration {i+1}/{num_iterations}")
        result = analyze_single_variant(
            problem, steps, client, variant_name=f"iter_{i+1}"
        )
        iterations.append(result)

    return {
        "problem_id": record["problem_id"],
        "type": record.get("type", ""),
        "level": record.get("level", ""),
        "variant": variant_key,
        "num_iterations": num_iterations,
        "iterations": iterations,
        "success_count": sum(1 for it in iterations if it.get("success")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="测试 DAG 分析的稳定性：在高随机性条件下（T=1.0, p=0.95）对同一样本重复分析多次并计算相似度"
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="输入的分段记录 JSONL 文件")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="输出目录")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="随机抽取的样本数量")
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="每个样本重复分析的次数")
    parser.add_argument("--variant", type=str, default="original",
                        choices=["original", "simple", "hard"],
                        help="要测试的变体")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="GED 计算超时时间（秒）")
    parser.add_argument("--skip-ged", action="store_true",
                        help="跳过 GED 计算，仅使用特征相似度")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    parser.add_argument("--provider", type=str, default="deepseek",
                        help="LLM provider")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="LLM model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="LLM temperature (default: 1.0 for high randomness)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="LLM top_p (default: 0.95 for sampling)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    random.seed(args.seed)

    logger.info(f"Loading records from {args.input}")
    all_records = load_segmented_records(args.input)
    logger.info(f"Loaded {len(all_records)} records")

    if args.num_samples > len(all_records):
        logger.warning(f"Requested {args.num_samples} samples but only {len(all_records)} available")
        args.num_samples = len(all_records)

    sampled_records = random.sample(all_records, args.num_samples)
    logger.info(f"Sampled {len(sampled_records)} records: {[r['problem_id'] for r in sampled_records]}")

    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    logger.info(f"LLM Config: temperature={args.temperature}, top_p={args.top_p}")
    client = LLMClient(llm_config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run stability tests
    all_results = []
    for idx, record in enumerate(sampled_records):
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {idx+1}/{len(sampled_records)}: problem_id={record['problem_id']}")
        logger.info(f"{'='*60}")

        test_result = run_stability_test(
            record, args.num_iterations, client, args.variant
        )

        if test_result:
            # Save iterations
            iterations_file = output_dir / f"problem_{record['problem_id']}_iterations.jsonl"
            with open(iterations_file, "w", encoding="utf-8") as f:
                for it in test_result["iterations"]:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

            # Compute similarities
            logger.info(f"Computing pairwise similarities...")
            similarities = compute_pairwise_similarities(
                test_result["iterations"],
                timeout=args.timeout,
                skip_ged=args.skip_ged,
            )

            test_result["similarities"] = similarities
            all_results.append(test_result)

            # Print summary for this sample
            print(f"\n--- Problem {record['problem_id']} Summary ---")
            print(f"  Success rate: {test_result['success_count']}/{args.num_iterations}")
            print(f"  Pairwise comparisons: {similarities['total_pairs']} total, {similarities['failed_pairs']} failed")

            if similarities["feature_similarities"]:
                feat_sims = similarities["feature_similarities"]
                print(f"  Feature similarity (cosine):")
                print(f"    mean={statistics.mean(feat_sims):.4f}  "
                      f"median={statistics.median(feat_sims):.4f}  "
                      f"stdev={statistics.stdev(feat_sims):.4f}" if len(feat_sims) > 1 else f"    value={feat_sims[0]:.4f}")

            if similarities["ged_similarities"]:
                ged_sims = similarities["ged_similarities"]
                print(f"  GED similarity (normalized):")
                print(f"    mean={statistics.mean(ged_sims):.4f}  "
                      f"median={statistics.median(ged_sims):.4f}  "
                      f"stdev={statistics.stdev(ged_sims):.4f}" if len(ged_sims) > 1 else f"    value={ged_sims[0]:.4f}")

            if similarities["ged_values"]:
                geds = similarities["ged_values"]
                print(f"  GED values:")
                print(f"    mean={statistics.mean(geds):.2f}  "
                      f"median={statistics.median(geds):.2f}  "
                      f"min={min(geds):.0f}  max={max(geds):.0f}")

    # Save summary
    summary_file = output_dir / "stability_summary.json"
    summary = {
        "config": {
            "num_samples": args.num_samples,
            "num_iterations": args.num_iterations,
            "variant": args.variant,
            "seed": args.seed,
            "skip_ged": args.skip_ged,
            "llm_temperature": args.temperature,
            "llm_top_p": args.top_p,
            "llm_provider": args.provider,
            "llm_model": args.model,
        },
        "results": [
            {
                "problem_id": r["problem_id"],
                "type": r["type"],
                "level": r["level"],
                "success_count": r["success_count"],
                "similarities": r["similarities"],
            }
            for r in all_results
        ],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\nResults saved to {output_dir}")

    # Print overall summary
    print(f"\n{'='*60}")
    print("Overall Stability Summary")
    print(f"{'='*60}")

    all_feat_sims = []
    all_ged_sims = []
    all_ged_vals = []

    for r in all_results:
        all_feat_sims.extend(r["similarities"]["feature_similarities"])
        all_ged_sims.extend(r["similarities"]["ged_similarities"])
        all_ged_vals.extend(r["similarities"]["ged_values"])

    print(f"\nAcross {len(all_results)} samples, {sum(r['success_count'] for r in all_results)} total iterations:")

    if all_feat_sims:
        print(f"\nFeature similarity (cosine):")
        print(f"  mean={statistics.mean(all_feat_sims):.4f}  "
              f"median={statistics.median(all_feat_sims):.4f}  "
              f"stdev={statistics.stdev(all_feat_sims):.4f}" if len(all_feat_sims) > 1 else f"  value={all_feat_sims[0]:.4f}")
        print(f"  min={min(all_feat_sims):.4f}  max={max(all_feat_sims):.4f}")

    if all_ged_sims:
        print(f"\nGED similarity (normalized):")
        print(f"  mean={statistics.mean(all_ged_sims):.4f}  "
              f"median={statistics.median(all_ged_sims):.4f}  "
              f"stdev={statistics.stdev(all_ged_sims):.4f}" if len(all_ged_sims) > 1 else f"  value={all_ged_sims[0]:.4f}")
        print(f"  min={min(all_ged_sims):.4f}  max={max(all_ged_sims):.4f}")

    if all_ged_vals:
        print(f"\nGED values:")
        print(f"  mean={statistics.mean(all_ged_vals):.2f}  "
              f"median={statistics.median(all_ged_vals):.2f}")
        print(f"  min={min(all_ged_vals):.0f}  max={max(all_ged_vals):.0f}")

    print(f"\nInterpretation:")
    if all_feat_sims:
        avg_feat = statistics.mean(all_feat_sims)
        if avg_feat > 0.95:
            print(f"  [HIGH] High stability (avg feature sim={avg_feat:.4f}): LLM output is very consistent")
        elif avg_feat > 0.85:
            print(f"  [MODERATE] Moderate stability (avg feature sim={avg_feat:.4f}): Some variation in structure")
        else:
            print(f"  [LOW] Low stability (avg feature sim={avg_feat:.4f}): Significant variation between runs")


if __name__ == "__main__":
    main()
