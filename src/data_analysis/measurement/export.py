"""Export GED measurements without changing the existing result schema."""

import csv
from pathlib import Path
from typing import Dict, List

from utils.io import atomic_text


def write_results_to_csv(results: List[Dict], problem_id: int, output_dir: str):
    """Write results and summary statistics to CSV."""
    output_path = Path(output_dir) / f"{problem_id}_ged_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with atomic_text(output_path) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'problem_id', 'sample_id', 'variant', 'problem', 'response',
                'correct', 'ged', 'ged_normalized', 'ged_normalizer',
                'similarity_normalized', 'timed_out', 'status', 'approximate',
                'elapsed_seconds', 'ged_max_nodes', 'ged_max_edges',
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
