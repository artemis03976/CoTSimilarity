"""Paired three-point structural probe; the problem group is the bootstrap unit."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from utils.artifacts import response_hash
from utils.io import atomic_text, read_json, read_jsonl, write_json


def direction(d):
    return "up" if d > 0 else "down" if d < 0 else "tie"


def summarize(rows, field="ged_normalized"):
    complete = [r for r in rows if all(x["status"] == "ok" for x in r)]
    result = {"planned": len(rows), "complete": len(complete)}
    if not complete:
        return result
    vals = np.array([[x[field] for x in r] for r in complete])
    steps = np.diff(vals, axis=1)
    delta = vals[:, 2] - vals[:, 0]
    by_group = defaultdict(list)
    for r, d in zip(complete, delta):
        by_group[r[0]["problem_id"]].append(d)
    groups = list(by_group.values())
    sizes = np.array([len(g) for g in groups])
    sums = np.array([sum(g) for g in groups])
    rng = np.random.default_rng(42)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    bootstrap = sums[draws].sum(1) / sizes[draws].sum(1)
    result.update(groups=len(groups), mean=vals.mean(axis=0).tolist(),
        endpoint_mean=float(delta.mean()), endpoint_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
        endpoint=dict(Counter(direction(d) for d in delta)),
        step_01=dict(Counter(direction(d) for d in steps[:,0])),
        step_12=dict(Counter(direction(d) for d in steps[:,1])),
        strict_rise=int((steps > 0).all(1).sum()),
        nondecreasing_with_rise=int(((steps >= 0).all(1) & (delta > 0)).sum()),
        flat=int((steps == 0).all(1).sum()),
        nonincreasing_with_drop=int(((steps <= 0).all(1) & (delta < 0)).sum()),
        peak=int(((steps[:,0] > 0) & (steps[:,1] < 0)).sum()),
        valley=int(((steps[:,0] < 0) & (steps[:,1] > 0)).sum()))
    return result


def report(out: Path):
    good = {r["sample_id"]:r for r in read_jsonl(out/"ged.jsonl")}
    failures = {r["sample_id"]:r for r in read_jsonl(out/"ged.failures.jsonl")}
    annotations = {s["sample_id"] for r in read_jsonl(out/"variant_graphs.jsonl")
                   for v in ("simple","hard") for s in r[v]["samples"] if s.get("dag_analysis")}
    by_question = defaultdict(list)
    counts = Counter()
    for x in read_jsonl(out/"conditions.jsonl"):
        sid = x["sample_id"]
        if sid in good:
            g = good[sid]
            if g["status"] != "ok" or g["timed_out"] or response_hash(g["response"]) != x["response_hash"]:
                raise ValueError(f"Unmatched GED: {sid}")
            x.update(status="ok", ged_normalized=g["ged_normalized"], ged=g["ged"])
        else:
            x.update(status=failures[sid]["status"], ged_normalized=None, ged=None)
        counts[x["status"]] += 1
        by_question[x["problem_id"],x["variant"]].append(x)
    rows = []
    for conditions in by_question.values():
        triple = sorted(conditions, key=lambda x:x["alpha"])
        if [x["alpha"] for x in triple] != [0,.5,1] or len({x["anchor_hash"] for x in triple}) != 1:
            raise ValueError("Incomplete design or unmatched Original anchor")
        rows.append(triple)
    with atomic_text(out/"paired_ged.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0][0]))
        writer.writeheader()
        writer.writerows(x for r in rows for x in r)
    scopes = {"all":rows, "simple":[r for r in rows if r[0]["variant"]=="simple"],
              "hard":[r for r in rows if r[0]["variant"]=="hard"],
              "endpoints_correct":[r for r in rows if r[0]["correct"] and r[2]["correct"]]}
    info = {k:summarize(v) for k,v in scopes.items()}
    endpoints = [r[2]["ged_normalized"]-r[0]["ged_normalized"] for r in rows
                 if r[0]["status"]==r[2]["status"]=="ok"]
    total_distinct = len({x["sample_id"] for r in rows for x in r})
    all_identical = sum(len({x["response_hash"] for x in r}) == 1 for r in rows)
    summary = dict(normalized=info, raw_ged=summarize(rows,"ged"), condition_status=dict(counts),
        distinct_trajectories=total_distinct, annotated=len(annotations),
        three_identical_responses=all_identical,
        endpoint_available=dict(total=len(endpoints), directions=dict(Counter(map(direction,endpoints)))))
    write_json(out/"trend.json", summary)
    lines = ["# 原版 DSPR 固定 α 结构实验", "",
        f"计划 {len(rows)} 个题目变体，每个比较 α=0、0.5、1 与同一 Base Original greedy 图的距离。",
        f"截至本次汇总，{len(annotations)}/{total_distinct} 条不同轨迹有合法标注。"
        f"{sum(counts.values())} 个条件中有 {counts['ok']} 个取得 GED。{all_identical} 个题目变体的三档回答文本完全相同。", "",
        "主分析在同一变体的三档 GED 均成功时进入。表中“递增”允许一段持平且端点上升；"
        "“平”表示三档相等。归一化 GED 按计算流程保存的四位小数比较。", "",
        "| 范围 | 完整 / 计划 | 均值 0 / 0.5 / 1 | 端点升 / 平 / 降 | 递增（其中严格递增） | 三档平 | 递减 | 中间峰 / 谷 | 平均端点差 [95% CI] |",
        "|---|---:|---|---|---:|---:|---:|---|---|"]
    labels = {"all":"全部","simple":"Simple","hard":"Hard","endpoints_correct":"两个端点均答对"}
    for k,s in info.items():
        n=f"{s['complete']} / {s['planned']}"
        if not s["complete"]:
            lines.append(f"| {labels[k]} | {n} | — | — | — | — | — | — | — |")
            continue
        means=" / ".join(f"{x:.4f}" for x in s["mean"])
        dirs=" / ".join(str(s["endpoint"].get(a,0)) for a in ("up","tie","down"))
        uncertainty = ""
        if len(annotations) == total_distinct and s["groups"] > 1:
            ci=", ".join(f"{x:+.4f}" for x in s["endpoint_ci95"])
            uncertainty=f" [{ci}]"
        lines.append(f"| {labels[k]} | {n} | {means} | {dirs} | {s['nondecreasing_with_rise']} ({s['strict_rise']}) | {s['flat']} | {s['nonincreasing_with_drop']} | {s['peak']} / {s['valley']} | {s['endpoint_mean']:+.4f}{uncertainty} |")
    lines += ["", "样本按 canonical 题源比例、最大余数分配和 seed 42 抽取，每题保留 Simple 和 Hard。"
        "相同题干与回答共用同一张变体图。未按正确性、生成有效性或测得的 GED 替换样本。"
        "区间以原题组为单位进行 10,000 次配对 bootstrap，合并行按变体取平均。"
        "归一化分母为两图节点和边的总数。原始 GED 敏感性结果见 trend.json。", "",
        f"每图对硬超时 {read_json(out/'cohort.json')['ged_timeout_seconds']} 秒，节点上限 32，边上限 64。"
        f"条件覆盖明细：{dict(counts)}。"]
    if info["all"]["complete"]:
        primary = info["all"]
        rising = primary["nondecreasing_with_rise"]
        missing = len(rows) - primary["complete"]
        lines += ["", f"在 {primary['complete']} 个完整变体中，"
            f"{primary['complete'] - primary['flat']} 个在至少一档改变了到 Original 的距离。"
            f"允许相邻持平时，三档递增的有 {rising} 个。"
            f"即使将 {missing} 个不完整变体全部计入递增，全队列也至多有 "
            f"{rising + missing}/{len(rows)} 个满足这一趋势。"]
        raw = summary["raw_ged"]
        lo, hi = raw["endpoint_ci95"]
        endpoint_dirs = summary["endpoint_available"]["directions"]
        lines += ["", "### 端点与答案正确性", "",
            f"放宽为仅要求两个端点可用时，共有 {len(endpoints)}/{len(rows)} 个变体可以比较。"
            f"从 α=0 到 α=1，{endpoint_dirs.get('up', 0)} 个上升，"
            f"{endpoint_dirs.get('tie', 0)} 个持平，{endpoint_dirs.get('down', 0)} 个下降。", "",
            f"在相同的完整三档配对中，未归一化 GED 的平均端点差（α=1 减 α=0）为 "
            f"{raw['endpoint_mean']:+.3f}，题组 bootstrap 95% 区间为 [{lo:+.3f}, {hi:+.3f}]。",
            "", "下表沿用已有 OOF 答案评分，分母包含抽中的全部变体，包括 GED 超时的回答。", "",
            "| 变体 | α=0 答对 | α=0.5 答对 | α=1 答对 | 0→1 错转对 | 0→1 对转错 |",
            "|---|---:|---:|---:|---:|---:|"]
        for variant in ("simple", "hard"):
            triples = scopes[variant]
            correct = [sum(bool(t[i]["correct"]) for t in triples) for i in range(3)]
            gained = sum(not t[0]["correct"] and t[2]["correct"] for t in triples)
            lost = sum(t[0]["correct"] and not t[2]["correct"] for t in triples)
            cells = " | ".join(f"{n}/{len(triples)}" for n in correct)
            lines.append(f"| {labels[variant]} | {cells} | {gained} | {lost} |")

        for pid in sorted({t[0]["problem_id"] for t in rows}):
            pair = [sorted(by_question[pid, v], key=lambda x: x["alpha"]) for v in ("simple", "hard")]
            if not all(x["status"] == "ok" and x["correct"] for triple in pair for x in triple):
                continue
            distances = np.array([[x["ged_normalized"] for x in triple] for triple in pair])
            changes = np.diff(distances, axis=1)
            if not (np.ptp(distances, axis=1) > 0).all():
                continue
            if not ((changes >= 0).all(axis=1).any() and (changes <= 0).all(axis=1).any()):
                continue
            lines += ["", f"### 同题正反向实例 · 第 {pid} 题", "",
                "取两种变体单调方向相反、六个回答均正确且 GED 完整的最小题号。"
                "两行共用同一张 Base Original 图。", "",
                "| 变体 | α=0 | α=0.5 | α=1 |", "|---|---:|---:|---:|"]
            for variant, values in zip(("Simple", "Hard"), distances):
                cells = " | ".join(f"{value:.4f}" for value in values)
                lines.append(f"| {variant} | {cells} |")
            break
    if len(annotations) < total_distinct:
        lines += ["", "**标注覆盖不完整，以上为本地已取得的子集结果，不能据此推断整个队列的趋势。**"]
    with atomic_text(out/"report.md") as stream:
        stream.write("\n".join(lines)+"\n")
    print(lines[2],lines[3], "Complete triples:", info["all"]["complete"])
