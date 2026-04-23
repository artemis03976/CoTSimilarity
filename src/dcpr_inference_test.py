import json
import os
import argparse
import torch
from tqdm import tqdm

from utils.evaluate import answer_check
from dcpr.config import DCPRConfig, MATH_SYSTEM_PROMPT
from dcpr.model import DCPRModel

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

    new_token_ids, _ = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        **generation_kwargs,
    )
    return tokenizer.decode(new_token_ids[0], skip_special_tokens=True).strip()


def generate_answer(model, problem, temperature=0.0, top_p=1.0, n=1, max_new_tokens=2048):
    responses = []
    for _ in range(n):
        responses.append(
            generate_one_answer(
                model,
                problem,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )
    return responses


def check_answer(problem, response, ground_truth, dataset_type):
    try:
        return answer_check(problem, response, ground_truth, dataset_type)
    except Exception as e:
        print(f"[WARN] answer_check 异常: {e}")
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
    print(f"\n测试完成: {total} 组, 全部通过 {correct_count} 组, 存在错误 {error_count} 组, 已保存至 {out_path}")


def load_model(args):
    config = DCPRConfig(
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

    model = DCPRModel(config)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.dual_prefix.load_state_dict(ckpt["dual_prefix_state_dict"])
    model.router.load_state_dict(ckpt["router_state_dict"])
    model.to(args.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 DCPR checkpoint 路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct", help="底座模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="测试数据路径")
    parser.add_argument("--id", type=int, default=None, help="指定 problem_id")
    parser.add_argument("--num", type=int, default=None, help="测试条数（未指定 id 时生效）")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度，0 表示 greedy decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling")
    parser.add_argument("--n", type=int, default=1, help="每个问题的采样数")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成 token 数")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--context_layer_idx", type=int, default=15, help="上下文编码层")
    parser.add_argument("--prefix_length", type=int, default=50, help="prefix 长度")
    parser.add_argument("--router_intermediate_dim", type=int, default=256, help="router 中间层维度")
    parser.add_argument("--router_dropout", type=float, default=0.05, help="router dropout")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
    args = parser.parse_args()

    print("正在加载 DCPR 模型...")
    model = load_model(args)

    data = load_data(args.data_path, args.id)
    if not data:
        print(f"未找到数据 (problem_id={args.id})")
        return
    if args.id is None and args.num is not None:
        import random
        random.shuffle(data)
        data = data[:args.num]

    run_eval(model, data, args.output_dir, args.temperature, args.top_p, args.n, args.max_new_tokens)


if __name__ == "__main__":
    main()
