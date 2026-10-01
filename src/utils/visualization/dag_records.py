"""Shared adapters for sample-based records and historical batch reports."""

import copy

from data_analysis.graph.compression import compress_dag_combined
from data_analysis.graph.builder import build_digraph_with_tags
from data_analysis.annotation.parsing import extract_dag_from_batch_response, validate_dag
from data_analysis.records import iter_samples
from utils.io import read_jsonl
from data_analysis.schemas import SampleKey, VARIANTS


def load_analyzed_records(input_path):
    rows = {}
    for record in read_jsonl(input_path):
        if record.get("custom_id"):
            key = SampleKey.parse(record["custom_id"])
            dag = extract_dag_from_batch_response(record)
            if dag is None:
                continue
            entries = [(key, {"problem": f"Problem {key.problem_id} ({key.variant})"}, {"dag_analysis": dag})]
            metadata = {}
        else:
            metadata = {name: value for name, value in record.items() if name not in VARIANTS}
            entries = list(iter_samples([record]))
            # Historical single-response files stored annotations on the variant.
            for variant in VARIANTS:
                entry = record.get(variant) or {}
                if "samples" not in entry and entry.get("dag_analysis"):
                    validate_dag(entry["dag_analysis"], entry.get("steps"))
                    entries.append((SampleKey(int(record["problem_id"]), variant), entry, entry))
        for key, entry, sample in entries:
            if sample.get("dag_analysis") is not None:
                validate_dag(sample["dag_analysis"], sample.get("steps"))
            row = rows.setdefault((key.problem_id, key.sample_index), {
                **metadata, "problem_id": key.problem_id, "sample_index": key.sample_index,
            })
            if key.variant in row:
                raise ValueError(f"Duplicate visualization sample: {key}")
            row[key.variant] = {
                **{name: value for name, value in entry.items() if name != "samples"},
                **copy.deepcopy(sample), "sample_id": str(key),
            }
    return list(rows.values())


def load_segmented_records(input_path, problem_ids):
    wanted = {str(pid) for pid in problem_ids}
    return {str(row["problem_id"]): row for row in load_analyzed_records(input_path)
            if str(row["problem_id"]) in wanted and row["sample_index"] == 0}


def load_dag_analysis(input_path, problem_ids):
    return {
        pid: {variant: entry["dag_analysis"] for variant in VARIANTS
              if (entry := row.get(variant, {})).get("dag_analysis")}
        for pid, row in load_segmented_records(input_path, problem_ids).items()
    }


def compress_dag_analysis(dag_analysis, exclude_external=False):
    if not dag_analysis:
        return dag_analysis
    graph = build_digraph_with_tags(dag_analysis, exclude_external=exclude_external)
    graph, _ = compress_dag_combined(graph, merge_metadata=True)
    compressed = []
    for node in sorted(n for n in graph if isinstance(n, int) and n > 0):
        attrs = graph.nodes[node]
        merged_ids = sorted([node] + attrs.get("absorbed_nodes", []))
        compressed.append({
            "step_id": node, "merged_ids": merged_ids,
            "merged_label": ",".join(map(str, merged_ids)),
            "macro_action_tag": attrs.get("macro_action_tag") or "N/A",
            "depends_on": list(graph.predecessors(node)),
            "analysis": attrs.get("analysis", ""),
        })
    return compressed
