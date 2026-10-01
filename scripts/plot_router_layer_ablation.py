#!/usr/bin/env python3
"""Plot router training curves and summarize a context-layer sweep.

Expected input layout::

    <root>/layer_<idx>/router/training_history.jsonl

The script is intended for headless experiment machines.  It writes a CSV
summary and two PNG figures: metric curves over training and the best/final
metric values as a function of layer index.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


LAYER_RE = re.compile(r"^layer[_-]?(\d+)$")


def read_history(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            logs = item.get("logs", {})
            if not isinstance(logs, dict):
                continue
            rows.append(
                {
                    "step": float(item.get("step", logs.get("step", 0))),
                    "epoch": float(item.get("epoch", logs.get("epoch", 0.0))),
                    **logs,
                }
            )
    return rows


def numeric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def numeric_any(row: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = numeric(row, key)
        if value is not None:
            return value
    return None


def layer_runs(root: Path) -> list[tuple[int, Path, list[dict]]]:
    runs = []
    for child in sorted(root.iterdir() if root.is_dir() else []):
        if not child.is_dir():
            continue
        match = LAYER_RE.match(child.name)
        if match is None:
            continue
        history_path = child / "router" / "training_history.jsonl"
        if not history_path.is_file():
            print(f"warning: missing history, skipping {child}")
            continue
        history = read_history(history_path)
        if history:
            runs.append((int(match.group(1)), child, history))
    return sorted(runs, key=lambda item: item[0])


def summarize(layer: int, history: list[dict]) -> dict:
    eval_rows = [row for row in history if numeric(row, "eval_router_auc") is not None]
    train_rows = [row for row in history if numeric(row, "loss") is not None]

    def best(key: str | tuple[str, ...], rows: list[dict], maximize: bool):
        keys = (key,) if isinstance(key, str) else key
        values = [(numeric_any(row, keys), row) for row in rows]
        values = [(value, row) for value, row in values if value is not None]
        if not values:
            return None, None
        value, row = (max if maximize else min)(values, key=lambda item: item[0])
        return value, row.get("epoch")

    best_auc, best_auc_epoch = best("eval_router_auc", eval_rows, True)
    best_loss, best_loss_epoch = best(("eval_router_loss", "eval_loss"), eval_rows, False)
    final_eval = eval_rows[-1] if eval_rows else {}
    final_train = train_rows[-1] if train_rows else {}
    return {
        "layer": layer,
        "history_points": len(history),
        "eval_points": len(eval_rows),
        "best_eval_auc": best_auc,
        "best_eval_auc_epoch": best_auc_epoch,
        "best_eval_loss": best_loss,
        "best_eval_loss_epoch": best_loss_epoch,
        "final_eval_auc": numeric(final_eval, "eval_router_auc"),
        "final_eval_loss": numeric_any(final_eval, ("eval_router_loss", "eval_loss")),
        "final_eval_router_loss": numeric(final_eval, "eval_router_loss"),
        "final_eval_balanced_accuracy": numeric(final_eval, "eval_router_balanced_accuracy"),
        "final_eval_alpha_gap": numeric(final_eval, "eval_alpha_gap"),
        "final_eval_simple_mean": numeric(final_eval, "eval_alpha_simple_mean"),
        "final_eval_hard_mean": numeric(final_eval, "eval_alpha_hard_mean"),
        "final_train_loss": numeric(final_train, "loss"),
        "final_train_router_loss": numeric(final_train, "router_loss"),
    }


def plot_curves(runs: list[tuple[int, Path, list[dict]]], output: Path) -> None:
    layers = [layer for layer, _, _ in runs]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(min(layers), max(layers) if max(layers) > min(layers) else min(layers) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()
    panels = [
        (("router_loss", "loss"), "Training router loss"),
        (("eval_router_loss", "eval_loss"), "Validation router loss"),
        (("eval_router_auc",), "Validation router AUC"),
        (("eval_alpha_gap",), "Validation alpha gap"),
    ]
    for axis, (keys, title) in zip(axes, panels):
        for layer, _, history in runs:
            points = [(numeric(row, "epoch"), numeric_any(row, keys)) for row in history]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, color=cmap(norm(layer)), linewidth=1.2, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(np.asarray(layers))
    fig.colorbar(sm, ax=axes.tolist(), label="context layer index")
    fig.suptitle("Router layer ablation: training curves")
    fig.savefig(output / "router_layer_curves.png", dpi=180)
    plt.close(fig)


def plot_summary(summary: list[dict], output: Path) -> None:
    layers = np.asarray([row["layer"] for row in summary])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    plots = [
        ("best_eval_auc", "Best validation AUC", "AUC"),
        ("best_eval_loss", "Best validation loss", "BCE loss"),
        ("final_eval_alpha_gap", "Final validation alpha gap", "hard - simple alpha"),
        ("final_eval_balanced_accuracy", "Final balanced accuracy", "balanced accuracy"),
    ]
    for axis, (key, title, ylabel) in zip(axes.ravel(), plots):
        y = np.asarray([
            np.nan if row.get(key) is None else row[key]
            for row in summary
        ], dtype=float)
        axis.plot(layers, y, marker="o", linewidth=1.5)
        if np.isfinite(y).any():
            index = int(np.nanargmax(y) if key != "best_eval_loss" else np.nanargmin(y))
            axis.scatter([layers[index]], [y[index]], color="tab:red", zorder=3)
            axis.annotate(
                f"layer {layers[index]}",
                (layers[index], y[index]),
                xytext=(5, 5),
                textcoords="offset points",
            )
        axis.set_title(title)
        axis.set_xlabel("context layer index")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    fig.suptitle("Router layer ablation: summary metrics")
    fig.savefig(output / "router_layer_summary.png", dpi=180)
    plt.close(fig)


def write_summary(summary: list[dict], path: Path) -> None:
    fields = list(summary[0].keys()) if summary else ["layer"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Layer-ablation output root")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figures/CSV (default: <root>/plots)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output_dir) if args.output_dir else root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    runs = layer_runs(root)
    if not runs:
        raise SystemExit(f"No layer histories found under {root}")

    summary = [summarize(layer, history) for layer, _, history in runs]
    write_summary(summary, output / "router_layer_summary.csv")
    (output / "router_layer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_curves(runs, output)
    plot_summary(summary, output)

    best = max(
        (row for row in summary if row["best_eval_auc"] is not None),
        key=lambda row: row["best_eval_auc"],
        default=None,
    )
    print(f"Processed {len(runs)} layer runs")
    print(f"Summary: {output / 'router_layer_summary.csv'}")
    print(f"Curves:  {output / 'router_layer_curves.png'}")
    print(f"Summary plot: {output / 'router_layer_summary.png'}")
    if best:
        print(
            f"Best validation AUC: layer={best['layer']} "
            f"auc={best['best_eval_auc']:.4f} "
            f"epoch={best['best_eval_auc_epoch']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
