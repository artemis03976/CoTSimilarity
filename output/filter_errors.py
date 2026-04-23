import json
import argparse

INPUT_PATH = "output/qwen/error_records.jsonl"

FILTERS = {
    "original_pass_simple_fail": {
        "output": "output/qwen/original_pass_simple_fail.jsonl",
        "desc": "original PASS, simple FAIL",
        "fn": lambda item: item["original"]["correct"] and not item["simple"]["correct"],
    },
    "original_pass_simple_fail_hard_pass": {
        "output": "output/qwen/original_pass_simple_fail_hard_pass.jsonl",
        "desc": "original PASS, simple FAIL, hard PASS",
        "fn": lambda item: item["original"]["correct"] and not item["simple"]["correct"] and item["hard"]["correct"],
    },
    "easy_pass_hard_fail": {
        "output": "output/qwen/easy_pass_hard_fail.jsonl",
        "desc": "(original OR simple) PASS, hard FAIL",
        "fn": lambda item: (item["original"]["correct"] or item["simple"]["correct"]) and not item["hard"]["correct"],
    },
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", type=str, default="original_pass_simple_fail",
                        choices=FILTERS.keys(), help="过滤条件")
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    cfg = FILTERS[args.filter]
    filtered = [item for item in data if cfg["fn"](item)]

    with open(cfg["output"], "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"共筛选出 {len(filtered)} 条 ({cfg['desc']})，已保存至 {cfg['output']}")

if __name__ == "__main__":
    main()
