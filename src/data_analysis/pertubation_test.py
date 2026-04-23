import json
import os
import argparse
from vllm import LLM, SamplingParams
from utils.evaluate import answer_check

DATA_PATH = "data/math_paired.jsonl"

def load_data(path, problem_id=None):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    if problem_id is not None:
        data = [d for d in data if d["problem_id"] == problem_id]
    return data

def generate_answer(llm, problem, temperature=1.0, top_p=1.0, n=1, max_tokens=2048):
    """Generate answer using vLLM with configurable sampling."""
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n,
    )
    outputs = llm.chat(messages, sampling_params=sampling_params)
    return [output.text for output in outputs[0].outputs]

def check_answer(problem, response, ground_truth, dataset_type):
    """调用 utils/evaluate.py 中的 answer_check 验证答案正确性"""
    try:
        return answer_check(problem, response, ground_truth, dataset_type)
    except Exception as e:
        print(f"[WARN] answer_check 异常: {e}")
        return False

def run_eval(
    llm,
    data,
    output_path,
    temperature=1.0,
    top_p=1.0,
    n=50,
    original_temperature=0.0,
    original_top_p=1.0,
    n_original=1,
    max_tokens=2048,
):
    """测试指定数据，保存所有记录（含模型回答与正确性判断）"""
    output_path_obj = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path_obj), exist_ok=True)
    out_path = output_path_obj

    total = len(data)
    correct_count = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(data):
            pid = item["problem_id"]
            print(f"[{idx+1}/{total}] Problem ID: {pid}", end="  ")

            results = {}
            all_correct = True

            for label, key in [("original", "original"), ("simple", "simple"), ("hard", "hard")]:
                problem = item[key]["problem"]
                ground_truth = item[key].get("solution") or item[key].get("answer")
                dataset_type = "original" if key == "original" else "perturb"

                if key == "original":
                    variant_temperature = original_temperature
                    variant_top_p = original_top_p
                    variant_n = n_original
                else:
                    variant_temperature = temperature
                    variant_top_p = top_p
                    variant_n = n

                responses = generate_answer(
                    llm,
                    problem,
                    temperature=variant_temperature,
                    top_p=variant_top_p,
                    n=variant_n,
                    max_tokens=max_tokens,
                )

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
            print(status)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct", help="模型名称或路径")
    parser.add_argument("--output_path", type=str, default="output/all_records_50.jsonl", help="输出文件路径")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="输入数据路径")
    parser.add_argument("--id", type=int, default=None, help="指定 problem_id")
    parser.add_argument("--num", type=int, default=None, help="测试条数（未指定 id 时生效）")
    parser.add_argument("--temperature", type=float, default=1.0, help="simple/hard 采样温度")
    parser.add_argument("--top_p", type=float, default=1.0, help="simple/hard nucleus sampling")
    parser.add_argument("--n", type=int, default=50, help="simple/hard 每个问题的采样数")
    parser.add_argument("--original_temperature", type=float, default=0.0, help="original 采样温度")
    parser.add_argument("--original_top_p", type=float, default=1.0, help="original nucleus sampling")
    parser.add_argument("--n_original", type=int, default=1, help="original 每个问题采样数")
    parser.add_argument("--max_tokens", type=int, default=2048, help="单次生成最大 token 数")
    args = parser.parse_args()

    print("正在加载模型...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
    )

    data = load_data(args.data_path, args.id)
    if not data:
        print(f"未找到数据 (problem_id={args.id})")
        return
    if args.id is None and args.num is not None:
        import random
        random.shuffle(data)
        data = data[:args.num]

    run_eval(
        llm,
        data,
        output_path=args.output_path,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        original_temperature=args.original_temperature,
        original_top_p=args.original_top_p,
        n_original=args.n_original,
        max_tokens=args.max_tokens,
    )

if __name__ == "__main__":
    main()
