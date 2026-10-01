"""One parser and validator for normal, batch, and stored DAG annotations."""

import json
import logging

from ..schemas import DAGNode, Step, VALID_TAGS


logger = logging.getLogger(__name__)


def validate_dag(dag, steps: list[Step] | None = None) -> list[DAGNode]:
    if not isinstance(dag, list) or not dag:
        raise ValueError("DAG must be a non-empty JSON array")
    ids = []
    for node in dag:
        if not isinstance(node, dict):
            raise ValueError("Each DAG node must be an object")
        step_id = node.get("step_id")
        if type(step_id) is not int or step_id <= 0:
            raise ValueError(f"Invalid step_id: {step_id}")
        ids.append(step_id)
        tag = node.get("macro_action_tag")
        if not isinstance(tag, str) or tag not in VALID_TAGS:
            raise ValueError(f"Step {step_id} has an invalid macro_action_tag")
        if not isinstance(node.get("depends_on"), list):
            raise ValueError(f"Step {step_id} must have a depends_on array")
    if steps is not None and any(not isinstance(step, dict) or type(step.get("index")) is not int for step in steps):
        raise ValueError("Reasoning steps must have integer indices")
    expected = [step["index"] for step in steps] if steps is not None else list(range(1, len(dag) + 1))
    if len(set(ids)) != len(ids) or sorted(ids) != sorted(expected):
        raise ValueError("DAG step IDs must match the reasoning steps exactly")
    for node in dag:
        for dependency in node["depends_on"]:
            if dependency == "External":
                continue
            if type(dependency) is not int or dependency < 0:
                raise ValueError(f"Invalid dependency: {dependency}")
            if dependency != 0 and (dependency not in ids or dependency >= node["step_id"]):
                raise ValueError(f"Step {node['step_id']} must depend only on earlier steps")
    return dag


def parse_dag_response(text: str, steps: list[Step] | None = None) -> list[DAGNode]:
    if not isinstance(text, str):
        raise ValueError("DAG response content must be text")
    text = text.strip()
    if text.startswith("```"):
        opening, separator, text = text.partition("\n")
        if not separator or opening.strip() not in {"```", "```json"} or not text.rstrip().endswith("```"):
            raise ValueError("Invalid fenced DAG response")
        text = text.rstrip()[:-3].strip()
    return validate_dag(json.loads(text), steps)


def batch_response_content(record: dict) -> str:
    if record.get("error"):
        raise ValueError(f"Provider error: {record['error']}")
    response = record.get("response") or {}
    if not isinstance(response, dict):
        raise ValueError("Invalid batch response envelope")
    status = response.get("status_code", 200)
    if status != 200:
        raise ValueError(f"Provider HTTP status: {status}")
    try:
        return response["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Missing batch response content") from exc


def extract_dag_from_batch_response(record: dict):
    try:
        return parse_dag_response(batch_response_content(record))
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid DAG for %s: %s", record.get("custom_id"), exc)
        return None
