"""Paired DeepSeekMath prompt-placement control on MATH-Perturb Originals."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_pipeline.aggregate import load_id_map
from experiment_pipeline.metrics import paired_comparison
from inference.common import ground_truth_for, ground_truth_type_for, read_math_records
from inference.greedy import GreedyModelConfig
from inference.runner import run_greedy_evaluation
from inference.vllm_greedy import VLLMGreedyGenerator
from utils.config import load_experiment
from utils.io import read_json, read_records, write_json
from utils.prompts import MATH_SYSTEM_PROMPT
from utils.scoring import require_scored


def official_prompt(tokenizer, problem, instruction=MATH_SYSTEM_PROMPT):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n{instruction}"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(records, output_dir, config, batch_size):
    if (output_dir / "all_records.jsonl").exists() or (output_dir / "generation_qc.json").exists():
        raise FileExistsError(f"Predictions already exist in {output_dir}. Select a fresh --output-dir.")
    problems = [r["original"]["problem"] for r in records]
    generator = VLLMGreedyGenerator(config, problems=problems, prompt_builder=official_prompt)
    try:
        run_greedy_evaluation(
            generator, records, output_dir, variants=("original",), batch_size=batch_size,
            run_metadata={
                "model_name": config.model_name, "backend": config.backend, "seed": config.seed,
                "dtype": config.dtype, "max_new_tokens": config.max_new_tokens,
                "source": "math_perturb", "instruction_placement": "user_suffix",
                "instruction": MATH_SYSTEM_PROMPT if config.system_prompt is None else config.system_prompt,
            },
        )
    finally:
        generator.close()
    print(f"Copy {output_dir} to the scoring machine, then run this script with 'score'.")


def score_and_compare(records, output_dir, baseline_dir):
    qc = read_json(output_dir / "generation_qc.json")
    if qc.get("variants") != ["original"] or qc["run"].get("instruction_placement") != "user_suffix":
        raise ValueError(f"{output_dir} is not the Original prompt-control run")
    baseline_qc = read_json(baseline_dir / "generation_qc.json")
    for key in ("model_name", "backend", "dtype", "seed", "sampling", "max_model_len", "batch_size"):
        if qc["run"][key] != baseline_qc["run"][key]:
            raise ValueError(f"Greedy settings differ between the paired runs: {key}")
    if qc["validation"] != baseline_qc["validation"]:
        raise ValueError("Generation quality settings differ between the paired runs")
    subprocess.run([sys.executable, str(ROOT / "scripts/score_answers.py"), str(output_dir)], check=True)
    ids = {r["problem_id"] for r in records}
    outcomes = []
    for directory in (baseline_dir, output_dir):
        path = directory / "all_records.jsonl"
        rows = load_id_map(read_records(path), path)
        if directory == output_dir and rows.keys() != ids:
            raise ValueError(f"{path}: control IDs differ from the canonical MATH-Perturb selection")
        correct = {}
        for record in records:
            pid = record["problem_id"]
            original = rows[pid]["original"]
            canonical = record["original"]
            if (original["problem"] != canonical["problem"]
                    or original["ground_truth"] != ground_truth_for(canonical)
                    or original["ground_truth_type"] != ground_truth_type_for(canonical, "original")
                    or len(original["samples"]) != 1):
                raise ValueError(f"{path}: Original question, reference or greedy sample count differs for ID {pid}")
            correct[pid] = require_scored(original["samples"][0], f"{path}: Original {pid}")
        outcomes.append(correct)

    baseline, control = outcomes
    result = paired_comparison(ids, control.__getitem__, baseline.__getitem__, 10000, 42, "original_prompt")
    result.update(
        comparator="system_prefix", method="user_suffix",
        gained_ids=sorted(pid for pid in ids if control[pid] and not baseline[pid]),
        lost_ids=sorted(pid for pid in ids if baseline[pid] and not control[pid]),
    )
    write_json(output_dir / "paired_comparison.json", result)
    for label, key in (("System prompt", "comparator"), ("Official user prompt", "method")):
        print(f"{label}: {result[key + '_correct']}/{len(ids)} ({result[key + '_accuracy']:.2%})")
    print(f"Paired difference: {100 * result['delta']:+.2f} pp; "
          f"gained {len(result['gained_ids'])}, lost {len(result['lost_ids'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "score"))
    parser.add_argument("--output-dir", type=Path, help="Directory for the user-prompt control")
    args = parser.parse_args()
    settings = load_experiment(ROOT / "configs/infer_deepseek.yaml", "generation")
    options = {**settings["common"], **settings["greedy"]}
    dataset = ROOT / options.pop("problem_dataset")
    baseline_dir = ROOT / options.pop("output_dir")
    output_dir = args.output_dir or baseline_dir.with_name("original_user_prompt")
    batch_size = options.pop("batch_size")
    records = [r for r in read_math_records(dataset) if r["source"] == "math_perturb"]
    print(f"MATH-Perturb Originals: {len(records)} | {options['model_name']} | {output_dir}", flush=True)
    if args.action == "generate":
        generate(records, output_dir, GreedyModelConfig(**options), batch_size)
    else:
        score_and_compare(records, output_dir, baseline_dir)


if __name__ == "__main__":
    main()
