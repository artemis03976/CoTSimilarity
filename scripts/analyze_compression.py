"""Analyze compression effectiveness from similarity results."""

import json
import argparse
import statistics
from collections import Counter


def analyze_compression_effectiveness(similarity_results_path: str):
    """Analyze compression statistics and impact on similarity."""

    compression_ratios = []
    absorption_counts = []
    similarity_changes = []

    with open(similarity_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)

            # Analyze each comparison pair
            for key in result.keys():
                if key.endswith('_vs_simple') or key.endswith('_vs_hard'):
                    data = result[key]

                    if 'compression_stats' not in data:
                        continue

                    # Collect compression statistics
                    for variant, stats in data['compression_stats'].items():
                        if stats and 'compression_ratio' in stats:
                            compression_ratios.append(stats['compression_ratio'])
                            absorption_counts.append(len(stats['absorptions']))

                    # Compare similarity before and after compression
                    if 'original' in data and 'compressed' in data:
                        orig_sim = data['original'].get('feature_similarity')
                        comp_sim = data['compressed'].get('feature_similarity')

                        if orig_sim is not None and comp_sim is not None:
                            similarity_changes.append(comp_sim - orig_sim)

    # Print statistics
    print("=" * 60)
    print("Compression Effectiveness Analysis")
    print("=" * 60)

    if compression_ratios:
        print(f"\n[Compression Ratios] (lower = more compression):")
        print(f"   Mean:   {statistics.mean(compression_ratios):.3f}")
        print(f"   Median: {statistics.median(compression_ratios):.3f}")
        print(f"   Min:    {min(compression_ratios):.3f}")
        print(f"   Max:    {max(compression_ratios):.3f}")
        if len(compression_ratios) > 1:
            print(f"   StdDev: {statistics.stdev(compression_ratios):.3f}")

        # Calculate average node reduction
        avg_reduction = (1 - statistics.mean(compression_ratios)) * 100
        print(f"\n   Average node reduction: {avg_reduction:.1f}%")

    if absorption_counts:
        print(f"\n[Node Absorptions]:")
        print(f"   Mean:   {statistics.mean(absorption_counts):.1f}")
        print(f"   Median: {statistics.median(absorption_counts):.1f}")
        print(f"   Min:    {min(absorption_counts)}")
        print(f"   Max:    {max(absorption_counts)}")
        print(f"   Total:  {sum(absorption_counts)}")

    if similarity_changes:
        print(f"\n[Similarity Change] (compressed - original):")
        print(f"   Mean:   {statistics.mean(similarity_changes):+.4f}")
        print(f"   Median: {statistics.median(similarity_changes):+.4f}")

        improved = sum(1 for x in similarity_changes if x > 0)
        degraded = sum(1 for x in similarity_changes if x < 0)
        unchanged = sum(1 for x in similarity_changes if x == 0)

        print(f"\n   Improved:  {improved} ({improved/len(similarity_changes)*100:.1f}%)")
        print(f"   Degraded:  {degraded} ({degraded/len(similarity_changes)*100:.1f}%)")
        print(f"   Unchanged: {unchanged} ({unchanged/len(similarity_changes)*100:.1f}%)")

    print("\n" + "=" * 60)


def analyze_tag_distribution(analyzed_records_path: str):
    """Analyze macro-action tag distribution and correlations."""

    tag_counts = Counter()
    tag_dep_correlation = {
        "Define": {"0": 0, "External": 0, "steps": 0},
        "Recall": {"0": 0, "External": 0, "steps": 0},
        "Derive": {"0": 0, "External": 0, "steps": 0},
        "Calculate": {"0": 0, "External": 0, "steps": 0},
        "Verify": {"0": 0, "External": 0, "steps": 0},
        "Conclude": {"0": 0, "External": 0, "steps": 0},
    }

    with open(analyzed_records_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)

            for variant in ["original", "simple", "hard"]:
                if variant not in record:
                    continue

                dag = record[variant].get("dag_analysis")
                if not dag:
                    continue

                for entry in dag:
                    tag = entry.get("macro_action_tag")
                    if not tag:
                        continue

                    deps = entry["depends_on"]

                    tag_counts[tag] += 1

                    # Categorize dependency
                    if deps == [0]:
                        dep_type = "0"
                    elif "External" in deps:
                        dep_type = "External"
                    else:
                        dep_type = "steps"

                    if tag in tag_dep_correlation:
                        tag_dep_correlation[tag][dep_type] += 1

    # Print statistics
    print("=" * 60)
    print("Macro-Action Tag Analysis")
    print("=" * 60)

    print("\nTag Distribution:")
    total = sum(tag_counts.values())
    for tag, count in tag_counts.most_common():
        print(f"   {tag:10s}: {count:5d} ({count/total*100:5.1f}%)")

    print("\nTag-Dependency Correlation:")
    for tag in ["Define", "Recall", "Derive", "Calculate", "Verify", "Conclude"]:
        deps = tag_dep_correlation[tag]
        total_tag = sum(deps.values())
        if total_tag > 0:
            print(f"\n   {tag}:")
            print(f"      [0]:       {deps['0']:5d} ({deps['0']/total_tag*100:5.1f}%)")
            print(f"      [External]: {deps['External']:5d} ({deps['External']/total_tag*100:5.1f}%)")
            print(f"      [steps]:    {deps['steps']:5d} ({deps['steps']/total_tag*100:5.1f}%)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze compression effectiveness and tag distribution"
    )
    parser.add_argument("--similarity", type=str,
                        help="Path to similarity results JSONL file")
    parser.add_argument("--analyzed", type=str,
                        help="Path to analyzed records JSONL file")
    args = parser.parse_args()

    if args.similarity:
        analyze_compression_effectiveness(args.similarity)

    if args.analyzed:
        analyze_tag_distribution(args.analyzed)

    if not args.similarity and not args.analyzed:
        print("Please provide --similarity and/or --analyzed path")


if __name__ == "__main__":
    main()
