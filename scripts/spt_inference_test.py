"""
Inference evaluation script for StaticPromptModel baseline.

Mirrors dcpr_inference_test.py but uses StaticPromptModel instead of DCPRModel.
"""

import json
import os
import argparse
import sys
from pathlib import Path
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.evaluate import answer_check
from dcpr.config import MATH_SYSTEM_PROMPT
from spt import SPTConfig, StaticPromptModel

DATA_PATH = "data/math_paired.jsonl"


def load_data(path, problem_id=None):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if problem_id is not None:
        data = [d for d in data if d["problem_id"] == problem_id]
    return data


def build_prompt(tokenizer, problem):
    messages = [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_one_answer(model, problem, max_new_tokens=2048, temperature=0.0, top_p=1.0):
    tokenizer = model.tokenizer
    prompt = build_prompt(tokenizer, problem)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.config.device)
    attention_mask = torch.ones_like(input_ids)
    do_sample = temperature is not None and temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    new_token_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    return tokenizer.decode(new_token_ids[0], skip_special_tokens=True).strip()


def generate_answer(model, problem, temperature=0.0, top_p=1.0, n=1, max_new_tokens=2048):
    responses = []
    for _ in range(n):
        responses.append(
            generate_one_answer(model, problem, max_new_tokens, temperature, top_p)
        )
    return responses


def check_answer(problem, response, ground_truth, dataset_type):
    try:
        return answer_check(problem, response, ground_truth, dataset_type)
    except Exception as e:
        print(f"[WARN] answer_check exception: {e}")
        return False


def run_eval(model, data, output_dir, temperature=0.0, top_p=1.0, n=1, max_new_tokens=2048):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "all_records.jsonl")

    total = len(data)
    correct_count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        progress_bar = tqdm(data, total=total, desc="Evaluating")
        for idx, item in enumerate(progress_bar):
            pid = item["problem_id"]

            results = {}
            all_correct = True

            for label, key in [("original", "original"), ("simple", "simple"), ("hard", "hard")]:
                problem = item[key]["problem"]
                ground_truth = item[key].get("solution") or item[key].get("answer")
                dataset_type = "original" if key == "original" else "perturb"

                responses = generate_answer(model, problem, temperature, top_p, n, max_new_tokens)

                sample_results = []
                for resp in responses:
                    correct = check_answer(problem, resp, str(ground_truth), dataset_type)
                    sample_results.append({"response": resp, "correct": correct})

                results[label] = {
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "samples": sample_results,
                }
                if not any(s["correct"] for s in sample_results):
                    all_correct = False

            status = "PASS" if all_correct else "FAIL"
            progress_bar.set_postfix(problem_id=pid, status=status)
            if all_correct:
                correct_count += 1

            record = {
                "problem_id": pid,
                "type": item["type"],
                "level": item["level"],
                **results,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    error_count = total - correct_count
    print(f"\nDone: {total} groups, passed {correct_count}, failed {error_count}, saved to {out_path}")


def load_model(args):
    config = SPTConfig(
        model_name=args.model_name,
        prefix_length=args.prefix_length,
        max_seq_length=args.max_seq_length,
        device=args.device,
        gradient_checkpointing=False,
    )

    model = StaticPromptModel(config)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.soft_prompt.data.copy_(ckpt["soft_prompt"])
    model.to(args.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained baseline checkpoint path")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="output_baseline")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prefix_length", type=int, default=50)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    print("Loading StaticPromptModel baseline...")
    model = load_model(args)

    data = load_data(args.data_path, args.id)
    if not data:
        print(f"No data found (problem_id={args.id})")
        return
    if args.id is None and args.num is not None:
        import random
        random.shuffle(data)
        data = data[:args.num]

    run_eval(model, data, args.output_dir, args.temperature, args.top_p, args.n, args.max_new_tokens)


if __name__ == "__main__":
    main()
