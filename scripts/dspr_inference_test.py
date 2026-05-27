import json
import os
import argparse
import sys
from pathlib import Path
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.evaluate import answer_check
from dspr.config import DSPRConfig, MATH_SYSTEM_PROMPT
from dspr.model import DSPRModel
from utils.visualization import plot_alpha_density

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


def generate_one_answer(model, problem, max_new_tokens=2048, temperature=0.0, top_p=1.0, forced_alpha=None):
    tokenizer = model.tokenizer
    prompt = build_prompt(tokenizer, problem)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.config.device)
    attention_mask = torch.ones_like(input_ids)
    prompt_input_ids = input_ids.clone()
    prompt_attention_mask = attention_mask.clone()
    do_sample = temperature is not None and temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    new_token_ids, alpha = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        forced_alpha=forced_alpha,
        **generation_kwargs,
    )
    alpha_val = alpha.item()
    return tokenizer.decode(new_token_ids[0], skip_special_tokens=True).strip(), alpha_val


def generate_answer(model, problem, temperature=0.0, top_p=1.0, n=1, max_new_tokens=2048, forced_alpha=None):
    responses = []
    alphas = []
    for _ in range(n):
        resp, alpha_val = generate_one_answer(
            model,
            problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            forced_alpha=forced_alpha,
        )
        responses.append(resp)
        alphas.append(alpha_val)
    return responses, alphas


def check_answer(problem, response, ground_truth, dataset_type):
    try:
        return answer_check(problem, response, ground_truth, dataset_type)
    except Exception as e:
        print(f"[WARN] answer_check failed: {e}")
        return False


def save_alpha_cache(alpha_dict, alpha_cache_path):
    os.makedirs(os.path.dirname(alpha_cache_path) or ".", exist_ok=True)
    with open(alpha_cache_path, "w", encoding="utf-8") as f:
        json.dump(alpha_dict, f, ensure_ascii=False, indent=2)
    print(f"Alpha cache saved to {alpha_cache_path}")


def load_alpha_cache(alpha_cache_path):
    with open(alpha_cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: [float(v) for v in vals] for k, vals in data.items()}


def run_eval(model, data, output_dir, alpha_cache_path, temperature=0.0, top_p=1.0, n=1, max_new_tokens=2048):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "all_records.jsonl")

    total = len(data)
    correct_count = 0
    all_alphas = {"Original": [], "Simple": [], "Hard": []}
    label_key_map = [("original", "original", "Original"), ("simple", "simple", "Simple"), ("hard", "hard", "Hard")]

    with open(out_path, "w", encoding="utf-8") as fout:
        progress_bar = tqdm(data, total=total, desc="Evaluating")
        for idx, item in enumerate(progress_bar):
            pid = item["problem_id"]

            results = {}
            all_correct = True

            for label, key, display in label_key_map:
                problem = item[key]["problem"]
                ground_truth = item[key].get("solution") or item[key].get("answer")
                dataset_type = "original" if key == "original" else "perturb"

                responses, alphas = generate_answer(model, problem, temperature, top_p, n, max_new_tokens, forced_alpha=None)
                all_alphas[display].extend(alphas)

                sample_results = []
                for resp in responses:
                    correct = check_answer(problem, resp, str(ground_truth), dataset_type)
                    sample_results.append({"response": resp, "correct": correct})

                results[label] = {
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "alpha": alphas[0] if len(alphas) == 1 else alphas,
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
    print(f"\nEvaluation complete: {total} groups, {correct_count} fully passed, {error_count} failed, saved to {out_path}")

    save_alpha_cache(all_alphas, alpha_cache_path)
    plot_alpha_density(all_alphas, output_dir)


def run_alpha_probe(model, data, output_dir, forced_alpha, temperature=0.0, top_p=1.0, n=1, max_new_tokens=2048):
    """Run inference on all data with a manually specified alpha, skipping accuracy evaluation.

    Each record contains problems and raw responses. GED computation is left to a separate pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"alpha{forced_alpha:.2f}".replace(".", "p")
    out_path = os.path.join(output_dir, f"probe_{tag}.jsonl")

    label_key_map = [("original", "original"), ("simple", "simple"), ("hard", "hard")]

    with open(out_path, "w", encoding="utf-8") as fout:
        progress_bar = tqdm(data, total=len(data), desc=f"Alpha probe α={forced_alpha:.2f}")
        for item in progress_bar:
            pid = item["problem_id"]
            progress_bar.set_postfix(problem_id=pid)

            results = {}
            for label, key in label_key_map:
                problem = item[key]["problem"]
                ground_truth = item[key].get("solution") or item[key].get("answer")

                responses, alphas = generate_answer(
                    model, problem, temperature, top_p, n, max_new_tokens, forced_alpha=forced_alpha
                )
                alpha_val = alphas[0] if len(alphas) == 1 else alphas
                results[label] = {
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "alpha": alpha_val,
                    "samples": [{"response": r} for r in responses],
                }

            record = {
                "problem_id": pid,
                "type": item["type"],
                "level": item["level"],
                "forced_alpha": forced_alpha,
                **results,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nAlpha probe complete: {len(data)} records, forced_alpha={forced_alpha:.2f}, saved to {out_path}")


def load_model(args):
    config = DSPRConfig(
        model_name=args.model_name,
        context_layer_idx=args.context_layer_idx,
        prefix_length=args.prefix_length,
        router_intermediate_dim=args.router_intermediate_dim,
        router_dropout=args.router_dropout,
        max_seq_length=args.max_seq_length,
        device=args.device,
        gradient_checkpointing=False,
        checkpoint_dir=os.path.dirname(args.checkpoint) or "checkpoints",
    )

    model = DSPRModel(config)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.dual_prefix.load_state_dict(ckpt["dual_prefix_state_dict"])
    model.router.load_state_dict(ckpt["router_state_dict"])
    model.to(args.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained DSPR checkpoint")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct", help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Evaluation data path")
    parser.add_argument("--id", type=int, default=None, help="Specific problem_id")
    parser.add_argument("--num", type=int, default=None, help="Number of records to test when --id is not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 means greedy decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling")
    parser.add_argument("--n", type=int, default=1, help="Samples per problem")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum generated tokens")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--context_layer_idx", type=int, default=15, help="Context encoding layer")
    parser.add_argument("--prefix_length", type=int, default=50, help="Prefix length")
    parser.add_argument("--router_intermediate_dim", type=int, default=256, help="Router hidden dimension")
    parser.add_argument("--router_dropout", type=float, default=0.05, help="router dropout")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument(
        "--alpha_cache_path",
        type=str,
        default=None,
        help="Alpha cache file path (default: output_dir/alpha_cache.json)",
    )
    parser.add_argument(
        "--plot_only_from_alpha_cache",
        action="store_true",
        help="Only redraw the density plot from the alpha cache without loading the model or rerunning inference",
    )
    parser.add_argument(
        "--forced_alpha",
        type=float,
        default=None,
        help="Manually set alpha in [0, 1], bypass the dynamic router, and enable alpha probe mode",
    )
    args = parser.parse_args()

    alpha_cache_path = args.alpha_cache_path or os.path.join(args.output_dir, "alpha_cache.json")

    if args.plot_only_from_alpha_cache:
        if not os.path.exists(alpha_cache_path):
            print(f"Alpha cache not found: {alpha_cache_path}")
            return
        all_alphas = load_alpha_cache(alpha_cache_path)
        plot_alpha_density(all_alphas, args.output_dir)
        return

    print("Loading DSPR model...")
    model = load_model(args)

    data = load_data(args.data_path, args.id)
    if not data:
        print(f"No data found (problem_id={args.id})")
        return
    if args.id is None and args.num is not None:
        import random
        random.shuffle(data)
        data = data[:args.num]

    if args.forced_alpha is not None:
        if not (0.0 <= args.forced_alpha <= 1.0):
            print(f"[ERROR] --forced_alpha must be in [0, 1], got: {args.forced_alpha}")
            return
        print(f"Alpha probe mode: forced_alpha={args.forced_alpha:.2f}, records={len(data)}")
        run_alpha_probe(
            model,
            data,
            args.output_dir,
            args.forced_alpha,
            args.temperature,
            args.top_p,
            args.n,
            args.max_new_tokens,
        )
    else:
        run_eval(
            model,
            data,
            args.output_dir,
            alpha_cache_path,
            args.temperature,
            args.top_p,
            args.n,
            args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
