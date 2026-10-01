"""Build the existing problem/step/external graph representation."""

from typing import Dict, List
import networkx as nx

from ..annotation.parsing import validate_dag


def build_digraph_with_tags(dag_analysis: List[Dict],
                            exclude_external: bool = False) -> nx.DiGraph:
    """Build DiGraph from dag_analysis with macro_action_tag attributes.

    Args:
        dag_analysis: List of dependency objects with macro_action_tag
        exclude_external: Whether to exclude External nodes

    Returns:
        DiGraph with 'type' and 'macro_action_tag' node attributes
    """
    validate_dag(dag_analysis)
    G = nx.DiGraph()
    if dag_analysis[0].get("response_hash"):
        G.graph["response_hash"] = dag_analysis[0]["response_hash"]
    # Node 0 is the problem statement. Step dependencies on 0 mean the step
    # directly uses information from the original problem.
    G.add_node(0, type="problem", macro_action_tag=None)

    for entry in dag_analysis:
        step_id = entry["step_id"]
        tag = entry.get("macro_action_tag")
        analysis = entry.get("analysis", "")
        # ``text`` is the original segmented CoT span. Legacy DAG records do
        # not contain it, so their annotator-written analysis remains a useful
        # fallback for text-aware GED.
        text = entry.get("text") or analysis

        if step_id not in G:
            G.add_node(
                step_id,
                type="step",
                macro_action_tag=tag,
                analysis=analysis,
                text=text,
            )
        else:
            # A malformed/out-of-order annotation may have introduced this
            # step as a dependency placeholder before its own entry appeared.
            G.nodes[step_id].update(
                type="step",
                macro_action_tag=tag,
                analysis=analysis,
                text=text,
            )

        for dep in entry["depends_on"]:
            if dep == "External":
                if not exclude_external:
                    if "External" not in G:
                        G.add_node("External", type="external", macro_action_tag=None)
                    G.add_edge("External", step_id)
            else:
                if dep not in G:
                    node_type = "problem" if dep == 0 else "step"
                    G.add_node(dep, type=node_type, macro_action_tag=None)
                G.add_edge(dep, step_id)

    return G
