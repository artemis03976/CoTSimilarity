"""Compute structural similarity between CoT DAGs.

The main metric is Graph Edit Distance (GED). A lower GED means two reasoning
graphs are structurally closer. Insertions/deletions use unit costs, while node
substitution combines semantic-role mismatch and compressed-node text cosine.
"""

import json
import logging
import time
import re
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


def compute_ged_similarity(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    timeout: float = 30.0,
    *,
    lambda_role: float = 1.0,
    lambda_text: float = 1.0,
) -> Dict:
    """Compute GED and normalized similarity between two DAGs.

    Exact GED can be expensive. Small graphs use optimize_graph_edit_distance;
    larger graphs use NetworkX's timeout-aware graph_edit_distance path.

    With unit insertion/deletion costs, deleting every node/edge in ``G1`` and
    inserting every node/edge in ``G2`` is a valid upper bound on the edit
    distance regardless of the weighted substitution cost. We normalize by
    ``|V1| + |V2| + |E1| + |E2|``.  The previous ``max(|V|) + max(|E|)``
    denominator was not an upper bound when both graphs contained different
    edges and could produce misleading negative similarities.
    """
    max_nodes = max(len(G1), len(G2))
    node_edit_budget = len(G1) + len(G2)
    edge_edit_budget = G1.number_of_edges() + G2.number_of_edges()
    normalizer = node_edit_budget + edge_edit_budget

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

    try:
        if max_nodes <= 12:
            start = time.time()
            for v in nx.optimize_graph_edit_distance(G1, G2, **cost_args):
                ged = v
                if time.time() - start > timeout:
                    timed_out = True
                    break
        else:
            ged = nx.graph_edit_distance(G1, G2, timeout=timeout, **cost_args)
            if ged is None:
                timed_out = True
    except Exception as e:
        logger.warning(f"GED computation error: {e}")

    result = {
        "ged": ged,
        "timed_out": timed_out,
        "ged_normalizer": normalizer,
        "node_edit_budget": node_edit_budget,
        "edge_edit_budget": edge_edit_budget,
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
