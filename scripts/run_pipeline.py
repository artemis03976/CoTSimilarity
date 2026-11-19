#!/usr/bin/env python3
"""Run configured K-fold training and generation; print offline scoring commands."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dspr", "spt", "lora"))
    parser.add_argument("--config")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--stage", choices=("all", "train", "eval"), default="all")
    stages.add_argument("--train-only", dest="stage", action="store_const", const="train")
    stages.add_argument("--eval-only", dest="stage", action="store_const", const="eval")
    parser.add_argument("--gpus")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--seed", type=int, help="Override the training seed")
    for name in ("model-name", "fold-root", "checkpoint-root", "result-root", "problem-dataset"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--router-checkpoint-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    if args.method != "dspr" and args.router_checkpoint_root:
        parser.error("Saved routers apply only to DSPR")
    config_path = Path(args.config or ROOT / "configs" / f"{args.method}.yaml").resolve()
    config = load_experiment(config_path, args.method)
    common = config.get("common", {})
    shared = ["--config", str(config_path)]
    for name in ("gpus", "folds", "model_name"):
        value = getattr(args, name)
        if value is not None:
            shared += ["--" + name.replace("_", "-")]
            shared += [str(x) for x in value] if isinstance(value, list) else [str(value)]
    for flag in ("dry_run", "allow_existing"):
        if getattr(args, flag):
            shared.append("--" + flag.replace("_", "-"))
    paths = {name: getattr(args, name) or common[name] for name in
             ("fold_root", "checkpoint_root", "result_root", "problem_dataset")}
    train = [sys.executable, "scripts/train_kfold.py", args.method, *shared,
             "--fold-root", paths["fold_root"], "--output-root", paths["checkpoint_root"]]
    if args.seed is not None:
        train += ["--seed", str(args.seed)]
    if args.method == "dspr":
        train += ["--problem-dataset", paths["problem_dataset"]]
        if args.router_checkpoint_root:
            train += ["--router-checkpoint-root", args.router_checkpoint_root]
    infer = [sys.executable, "scripts/evaluate.py", args.method, "run", *shared,
             "--checkpoint-root", paths["checkpoint_root"], "--output-root", paths["result_root"],
             "--test-root", paths["fold_root"], "--id-root", paths["fold_root"]]
    if args.stage != "eval":
        subprocess.run(train, cwd=ROOT, check=True)
        if args.dry_run:
            if args.stage == "all":
                print("After training: " + shlex.join([x for x in infer if x != "--dry-run"]))
            return 0
    if args.stage == "train":
        return 0
    subprocess.run(infer, cwd=ROOT, check=True)
    if args.dry_run:
        return 0
    folds = args.folds or common.get("folds", list(range(5)))
    fold_args = ["--folds", *map(str, folds)]
    print("On the scoring machine:")
    print(shlex.join(["python", "scripts/score_answers.py", paths["result_root"], *fold_args]))
    print(shlex.join(["python", "scripts/evaluate.py", args.method, "aggregate", "--config", str(config_path),
                      "--result-root", paths["result_root"], "--id-root", paths["fold_root"],
                      "--raw-data", paths["problem_dataset"], *fold_args]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
