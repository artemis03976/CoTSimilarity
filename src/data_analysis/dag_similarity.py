"""Compute DAG similarity between CoT variants using Graph Edit Distance."""

import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Optional

import networkx as nx

logger = logging.getLogger(__name__)


def extract_dag_from_batch_response(response_data: Dict) -> Optional[List[Dict]]:
    """Extract DAG analysis from batch inference response format."""
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
    """Build a directed graph from DAG analysis."""
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
def node_subst_cost(attrs1: dict, attrs2: dict) -> float:
    return 0.0 if attrs1.get("type") == attrs2.get("type") else 1.0

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


def compute_ged_similarity(G1: nx.DiGraph, G2: nx.DiGraph, timeout: float = 30.0) -> Dict:
    """Compute GED and normalized similarity between two DAGs."""
    max_nodes = max(len(G1), len(G2))
    max_edges = max(G1.number_of_edges(), G2.number_of_edges())
    normalizer = max_nodes + max_edges

    timed_out = False
    ged = None

    cost_args = dict(
        node_subst_cost=node_subst_cost,
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

    result = {"ged": ged, "timed_out": timed_out}
    if ged is not None and normalizer > 0:
        result["similarity_normalized"] = round(1 - ged / normalizer, 4)
        result["similarity_inverse"] = round(1 / (1 + ged), 4)
    else:
        result["similarity_normalized"] = None
        result["similarity_inverse"] = None

    return result
