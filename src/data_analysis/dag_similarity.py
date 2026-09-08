"""Compute structural similarity between CoT DAGs.

The main metric is Graph Edit Distance (GED). A lower GED means two reasoning
graphs are structurally closer. Insertions/deletions use unit costs, while node
substitution combines semantic-role mismatch and compressed-node text cosine.
"""

import json
import logging
import time
import re
import signal
import threading
from functools import partial
from pathlib import Path
from typing import List, Dict, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def extract_dag_from_batch_response(response_data: Dict) -> Optional[List[Dict]]:
    """Extract DAG analysis from a provider batch-response object.

    LLM batch outputs often wrap the actual JSON in markdown fences or may be
    truncated. This helper performs conservative cleanup before parsing.
    """
    try:
        content = response_data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None

        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        content = content.strip()

        if not content:
            return None

        try:
            dag_analysis = json.loads(content)
            if isinstance(dag_analysis, list):
                return dag_analysis
        except json.JSONDecodeError:
            try:
                if content.startswith('['):
                    last_complete_idx = content.rfind('}')
                    if last_complete_idx != -1:
                        fixed_content = content[:last_complete_idx + 1] + ']'
                        dag_analysis = json.loads(fixed_content)
                        if isinstance(dag_analysis, list):
                            logger.warning("Fixed truncated JSON response")
                            return dag_analysis
            except:
                pass
            logger.warning(f"Failed to parse DAG analysis from batch response")
            return None
        return None
    except Exception as e:
        logger.error(f"Error extracting DAG: {e}")
        return None


def build_digraph(dag_analysis: List[Dict], exclude_external: bool = False) -> nx.DiGraph:
    """Build a NetworkX DiGraph from step dependency annotations.

    Each reasoning step becomes a node. Dependencies point into the dependent
    step, so an edge A -> B means B depends on A.
    """
    G = nx.DiGraph()
    for step in dag_analysis:
        node_id = step["id"]
        node_type = step.get("type", "Unknown")
        if exclude_external and node_id == "External":
            continue
        G.add_node(node_id, type=node_type)
        for dep in step.get("dependencies", []):
            if exclude_external and dep == "External":
                continue
            G.add_edge(dep, node_id)
    return G


def compute_dag_depth(G: nx.DiGraph) -> int:
    """Compute the depth (longest path) of a DAG."""
    if len(G) == 0:
        return 0
    try:
        return nx.dag_longest_path_length(G)
    except:
        return 0


def compute_dag_max_width(G: nx.DiGraph) -> int:
    """Compute the maximum width (max nodes at any level) of a DAG."""
    if len(G) == 0:
        return 0
    levels = {}
    for node in nx.topological_sort(G):
        pred_levels = [levels.get(p, 0) for p in G.predecessors(node)]
        levels[node] = max(pred_levels, default=0) + 1
    from collections import Counter
    level_counts = Counter(levels.values())
    return max(level_counts.values()) if level_counts else 0


# Cost functions for GED
def node_subst_cost(
    attrs1: dict,
    attrs2: dict,
    *,
    lambda_role: float = 1.0,
    lambda_text: float = 1.0,
) -> float:
    """Role-plus-text node substitution cost from the paper's Eq. 6.

    ``build_digraph_with_tags`` stores the coarse node kind (problem, step, or
    external) in ``type`` and the IFD semantic role in ``macro_action_tag``.
    Prepared graph caches additionally store a normalized sentence embedding
    for the original CoT text associated with each compressed step node.

    Legacy or hand-built graphs without embeddings retain role-only behavior;
    the main GED pipeline always prepares embeddings before calling this cost.
    """

    kind1 = attrs1.get("type")
    kind2 = attrs2.get("type")
    if kind1 != kind2:
        return 1.0
    if kind1 == "step":
        role_cost = lambda_role * float(
            attrs1.get("macro_action_tag") != attrs2.get("macro_action_tag")
        )
        embedding1 = attrs1.get("text_embedding")
        embedding2 = attrs2.get("text_embedding")
        if embedding1 is None or embedding2 is None or lambda_text == 0:
            return role_cost

        cosine = float(
            np.dot(
                np.asarray(embedding1, dtype=np.float32),
                np.asarray(embedding2, dtype=np.float32),
            )
        )
        text_similarity = max(0.0, min(1.0, cosine))
        return role_cost + lambda_text * (1.0 - text_similarity)
    return 0.0

def node_del_cost(attrs: dict) -> float:
    return 1.0

def node_ins_cost(attrs: dict) -> float:
    return 1.0

def edge_subst_cost(attrs1: dict, attrs2: dict) -> float:
    return 0.0

def edge_del_cost(attrs: dict) -> float:
    return 1.0

def edge_ins_cost(attrs: dict) -> float:
    return 1.0


def _greedy_upper_bound(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    *,
    lambda_role: float,
    lambda_text: float,
) -> float:
    """Build a cheap valid edit-path upper bound for NetworkX pruning.

    Nodes are paired by coarse type and then by macro-action role.  The
    resulting one-to-one mapping is not intended to be optimal; it only gives
    the branch-and-bound search a substantially tighter bound than deleting
    and reinserting every node and edge.
    """
    mapping = {}
    if 0 in G1 and 0 in G2:
        mapping[0] = 0

    def groups(graph: nx.DiGraph):
        grouped = {}
        for node, attrs in graph.nodes(data=True):
            if node == 0:
                continue
            key = (attrs.get("type"), attrs.get("macro_action_tag"))
            grouped.setdefault(key, []).append(node)
        for nodes in grouped.values():
            nodes.sort(key=lambda value: str(value))
        return grouped

    groups1 = groups(G1)
    groups2 = groups(G2)
    used2 = set(mapping.values())

    # First match nodes with identical type/role, then pair remaining nodes of
    # the same coarse type so role substitution remains cheaper than deletion
    # plus insertion.
    for key in sorted(set(groups1) | set(groups2), key=str):
        left = groups1.get(key, [])
        right = [node for node in groups2.get(key, []) if node not in used2]
        for node1, node2 in zip(left, right):
            mapping[node1] = node2
            used2.add(node2)

    by_type1 = {}
    by_type2 = {}
    for node, attrs in G1.nodes(data=True):
        if node not in mapping:
            by_type1.setdefault(attrs.get("type"), []).append(node)
    for node, attrs in G2.nodes(data=True):
        if node not in used2:
            by_type2.setdefault(attrs.get("type"), []).append(node)
    for values in by_type1.values():
        values.sort(key=lambda value: str(value))
    for values in by_type2.values():
        values.sort(key=lambda value: str(value))
    for node_type in sorted(set(by_type1) | set(by_type2), key=str):
        for node1, node2 in zip(by_type1.get(node_type, []), by_type2.get(node_type, [])):
            mapping[node1] = node2
            used2.add(node2)

    cost = 0.0
    for node1, attrs1 in G1.nodes(data=True):
        node2 = mapping.get(node1)
        if node2 is None:
            cost += node_del_cost(attrs1)
        else:
            cost += node_subst_cost(
                attrs1,
                G2.nodes[node2],
                lambda_role=lambda_role,
                lambda_text=lambda_text,
            )
    cost += sum(node_ins_cost(attrs) for node, attrs in G2.nodes(data=True) if node not in used2)

    # Mapped edges that exist are retained; all other source edges are deleted
    # and unmatched target edges inserted.
    source_mapped_edges = {
        (mapping[source], mapping[target])
        for source, target in G1.edges
        if source in mapping and target in mapping
    }
    retained = source_mapped_edges & set(G2.edges)
    cost += G1.number_of_edges() - len(retained)
    cost += G2.number_of_edges() - len(retained)
    return max(0.0, cost)


def _run_ged_with_timeout(callable_, timeout: float, hard_timeout: bool) -> tuple[Optional[float], bool, float]:
    """Run a GED call and report elapsed time and timeout status.

    On POSIX main processes, ``setitimer`` provides a real wall-clock cutoff,
    including cases where NetworkX has not yielded its first candidate yet.
    Other platforms fall back to NetworkX's cooperative timeout plus elapsed
    time measurement.
    """
    start = time.perf_counter()
    timed_out = False
    value = None

    use_alarm = (
        hard_timeout
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    previous_handler = None

    try:
        if use_alarm:
            def _alarm_handler(_signum, _frame):
                raise TimeoutError("GED wall-clock timeout")

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        value = callable_()
    except TimeoutError:
        timed_out = True
        value = None
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    elapsed = time.perf_counter() - start
    if elapsed >= timeout:
        timed_out = True
    return value, timed_out, elapsed


def compute_ged_similarity(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    timeout: float = 10.0,
    *,
    lambda_role: float = 1.0,
    lambda_text: float = 1.0,
    max_nodes: Optional[int] = 32,
    max_edges: Optional[int] = 64,
    hard_timeout: bool = True,
) -> Dict:
    """Compute GED and normalized similarity between two DAGs.

    Exact GED can be expensive. A root match, a greedy upper bound, and graph
    size limits reduce pathological searches. The timeout is a hard wall-clock
    cutoff on POSIX main processes and a cooperative cutoff elsewhere.

    With unit insertion/deletion costs, deleting every node/edge in ``G1`` and
    inserting every node/edge in ``G2`` is a valid upper bound on the edit
    distance regardless of the weighted substitution cost. We normalize by
    ``|V1| + |V2| + |E1| + |E2|``.  The previous ``max(|V|) + max(|E|)``
    denominator was not an upper bound when both graphs contained different
    edges and could produce misleading negative similarities.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    graph_max_nodes = max(len(G1), len(G2))
    graph_max_edges = max(G1.number_of_edges(), G2.number_of_edges())
    node_edit_budget = len(G1) + len(G2)
    edge_edit_budget = G1.number_of_edges() + G2.number_of_edges()
    normalizer = node_edit_budget + edge_edit_budget

    if (max_nodes is not None and graph_max_nodes > max_nodes) or (
        max_edges is not None and graph_max_edges > max_edges
    ):
        return {
            "ged": None,
            "timed_out": False,
            "status": "skipped_size",
            "approximate": False,
            "elapsed_seconds": 0.0,
            "ged_normalizer": normalizer,
            "node_edit_budget": node_edit_budget,
            "edge_edit_budget": edge_edit_budget,
            "max_nodes": graph_max_nodes,
            "max_edges": graph_max_edges,
            "upper_bound": None,
            "ged_normalized": None,
            "similarity_normalized": None,
            "similarity_inverse": None,
        }

    timed_out = False
    ged = None

    cost_args = dict(
        node_subst_cost=partial(
            node_subst_cost,
            lambda_role=lambda_role,
            lambda_text=lambda_text,
        ),
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
    )

    upper_bound = min(
        float(normalizer),
        _greedy_upper_bound(
            G1,
            G2,
            lambda_role=lambda_role,
            lambda_text=lambda_text,
        ),
    )
    roots = (0, 0) if 0 in G1 and 0 in G2 else None

    computation_start = time.perf_counter()
    try:
        ged, timed_out, elapsed = _run_ged_with_timeout(
            lambda: nx.graph_edit_distance(
                G1,
                G2,
                roots=roots,
                upper_bound=upper_bound,
                timeout=timeout,
                **cost_args,
            ),
            timeout,
            hard_timeout,
        )
    except Exception as e:
        logger.warning(f"GED computation error: {e}")
        elapsed = time.perf_counter() - computation_start

    if timed_out:
        status = "timeout"
    elif ged is None:
        status = "error"
    else:
        status = "ok"

    result = {
        "ged": ged,
        "timed_out": timed_out,
        "status": status,
        "approximate": bool(timed_out and ged is not None),
        "elapsed_seconds": round(elapsed, 4),
        "ged_normalizer": normalizer,
        "node_edit_budget": node_edit_budget,
        "edge_edit_budget": edge_edit_budget,
        "max_nodes": graph_max_nodes,
        "max_edges": graph_max_edges,
        "upper_bound": upper_bound,
    }
    if ged is not None and normalizer > 0:
        ged_normalized = max(0.0, min(1.0, ged / normalizer))
        result["ged_normalized"] = round(ged_normalized, 4)
        result["similarity_normalized"] = round(1 - ged_normalized, 4)
        result["similarity_inverse"] = round(1 / (1 + ged), 4)
    else:
        result["ged_normalized"] = None
        result["similarity_normalized"] = None
        result["similarity_inverse"] = None

    return result
