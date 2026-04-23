"""DAG compression module using connected-component contraction."""

import networkx as nx
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


def _has_alternate_path(G: nx.DiGraph, u, v) -> bool:
    """Check if there exists a path from u to v after removing the direct edge u→v.

    Uses BFS on the graph with the direct edge temporarily removed.
    If a path exists, contracting u and v would create a cycle.

    Args:
        G: DiGraph to check
        u: Source node
        v: Target node

    Returns:
        True if an alternate path from u to v exists (contraction unsafe)
    """
    # BFS from u, skipping the direct edge u→v
    visited = set()
    queue = deque()

    # Seed with u's successors except v (via direct edge)
    for succ in G.successors(u):
        if succ != v:
            visited.add(succ)
            queue.append(succ)

    while queue:
        node = queue.popleft()
        if node == v:
            return True
        for succ in G.successors(node):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    return False


def can_contract(G: nx.DiGraph, u, v) -> bool:
    """Check if node v can be contracted into node u.

    Conditions:
    1. Edge u→v exists
    2. u and v have the same macro_action_tag (both non-None)
    3. Anti-cycle: no alternate path from u to v exists in the graph

    Args:
        G: DiGraph with 'macro_action_tag' node attribute
        u: Source node
        v: Target node

    Returns:
        True if v can be safely contracted into u
    """
    if not G.has_edge(u, v):
        return False

    tag_u = G.nodes[u].get('macro_action_tag')
    tag_v = G.nodes[v].get('macro_action_tag')

    if tag_u is None or tag_v is None:
        return False

    if tag_u != tag_v:
        return False

    # Anti-cycle check: if an alternate path u~>v exists, contraction would create a cycle
    if _has_alternate_path(G, u, v):
        return False

    return True


def contract_node(G: nx.DiGraph, u, v,
                  merge_metadata: bool = True) -> None:
    """Contract node v into node u.

    Actions:
    1. Redirect all incoming edges of v (from predecessors != u) to u
    2. Redirect all outgoing edges of v to u
    3. Optionally merge metadata (analysis text, absorbed_nodes)
    4. Remove self-loops on u
    5. Remove node v

    Args:
        G: DiGraph to modify in-place
        u: Node to contract into (survives)
        v: Node to be contracted (removed)
        merge_metadata: Whether to merge 'analysis' and other metadata
    """
    # Redirect incoming edges of v to u
    for pred in list(G.predecessors(v)):
        if pred != u:
            G.add_edge(pred, u)

    # Redirect outgoing edges of v to u
    for succ in list(G.successors(v)):
        if succ != u:
            G.add_edge(u, succ)

    # Merge metadata
    if merge_metadata:
        analysis_u = G.nodes[u].get('analysis', '')
        analysis_v = G.nodes[v].get('analysis', '')
        G.nodes[u]['analysis'] = f"{analysis_u} | {analysis_v}"

        absorbed = list(G.nodes[u].get('absorbed_nodes', []))
        absorbed.append(v)
        absorbed.extend(G.nodes[v].get('absorbed_nodes', []))
        G.nodes[u]['absorbed_nodes'] = absorbed

    # Remove node v (removes all its edges)
    G.remove_node(v)

    # Remove self-loops on u
    if G.has_edge(u, u):
        G.remove_edge(u, u)


def can_fold_parallel(G: nx.DiGraph, u, v) -> bool:
    """Check if nodes u and v can be folded (parallel same-layer same-tag nodes).

    Conditions:
    1. u and v have the same macro_action_tag (both non-None)
    2. u and v have identical parent sets (predecessors)
    3. u and v have identical child sets (successors)
    4. u and v are not connected by an edge

    Args:
        G: DiGraph with 'macro_action_tag' node attribute
        u: First node
        v: Second node

    Returns:
        True if u and v can be safely folded together
    """
    # Check same tag
    tag_u = G.nodes[u].get('macro_action_tag')
    tag_v = G.nodes[v].get('macro_action_tag')

    if tag_u is None or tag_v is None:
        return False

    if tag_u != tag_v:
        return False

    # Check no direct edge between them
    if G.has_edge(u, v) or G.has_edge(v, u):
        return False

    # Check identical parent sets
    parents_u = set(G.predecessors(u))
    parents_v = set(G.predecessors(v))
    if parents_u != parents_v:
        return False

    # Check identical child sets
    children_u = set(G.successors(u))
    children_v = set(G.successors(v))
    if children_u != children_v:
        return False

    return True


def fold_parallel_node(G: nx.DiGraph, u, v,
                       merge_metadata: bool = True) -> None:
    """Fold node v into node u (parallel nodes with same parents/children).

    Since u and v have identical edges, we just need to:
    1. Merge metadata
    2. Remove node v (edges are already duplicated on u)

    Args:
        G: DiGraph to modify in-place
        u: Node to fold into (survives)
        v: Node to be folded (removed)
        merge_metadata: Whether to merge 'analysis' and other metadata
    """
    # Merge metadata
    if merge_metadata:
        analysis_u = G.nodes[u].get('analysis', '')
        analysis_v = G.nodes[v].get('analysis', '')
        G.nodes[u]['analysis'] = f"{analysis_u} | {analysis_v}"

        absorbed = list(G.nodes[u].get('absorbed_nodes', []))
        absorbed.append(v)
        absorbed.extend(G.nodes[v].get('absorbed_nodes', []))
        G.nodes[u]['absorbed_nodes'] = absorbed

    # Remove node v (all its edges are duplicates of u's edges)
    G.remove_node(v)


def compress_parallel_layers(G: nx.DiGraph,
                             max_iterations: int = 100,
                             merge_metadata: bool = True) -> Tuple[nx.DiGraph, Dict]:
    """Compress DAG by folding parallel same-layer same-tag nodes.

    Finds pairs of nodes with:
    - Same macro_action_tag
    - Identical parent sets
    - Identical child sets
    - No direct edge between them

    Args:
        G: Original DiGraph with 'macro_action_tag' node attributes
        max_iterations: Maximum number of folding passes
        merge_metadata: Whether to merge node metadata during folding

    Returns:
        Tuple of (compressed_graph, compression_stats)
    """
    G_compressed = G.copy()

    stats = {
        'original_nodes': len(G),
        'original_edges': G.number_of_edges(),
        'folds': [],
        'iterations': 0
    }

    for iteration in range(max_iterations):
        folded_this_iteration = False

        # Get all step nodes (exclude Problem and External)
        step_nodes = [n for n in G_compressed.nodes()
                     if isinstance(n, int) and n > 0]

        # Try all pairs
        for i, u in enumerate(step_nodes):
            if u not in G_compressed:
                continue
            for v in step_nodes[i+1:]:
                if v not in G_compressed:
                    continue
                if can_fold_parallel(G_compressed, u, v):
                    logger.info(f"Folding parallel node {v} into {u}")
                    fold_parallel_node(G_compressed, u, v, merge_metadata)
                    stats['folds'].append((u, v))
                    folded_this_iteration = True
                    break
            if folded_this_iteration:
                break

        stats['iterations'] = iteration + 1

        if not folded_this_iteration:
            break

    stats['compressed_nodes'] = len(G_compressed)
    stats['compressed_edges'] = G_compressed.number_of_edges()
    stats['compression_ratio'] = (
        stats['compressed_nodes'] / stats['original_nodes']
        if stats['original_nodes'] > 0 else 1.0
    )

    return G_compressed, stats


def compress_dag(G: nx.DiGraph,
                 max_iterations: int = 100,
                 merge_metadata: bool = True) -> Tuple[nx.DiGraph, Dict]:
    """Compress DAG using connected-component contraction.

    Iteratively traverses edges in topological order. For each edge (u, v),
    if Tag(u)==Tag(v) and no alternate path exists, contract v into u.
    Restart iteration after each contraction since graph structure changes.

    Args:
        G: Original DiGraph with 'macro_action_tag' node attributes
        max_iterations: Maximum number of contraction passes
        merge_metadata: Whether to merge node metadata during contraction

    Returns:
        Tuple of (compressed_graph, compression_stats)
    """
    G_compressed = G.copy()

    stats = {
        'original_nodes': len(G),
        'original_edges': G.number_of_edges(),
        'absorptions': [],
        'iterations': 0,
        'skipped': False,
        'skip_reason': None
    }

    if not nx.is_directed_acyclic_graph(G_compressed):
        logger.warning('Graph is not a DAG, skipping contraction compression')
        stats['skipped'] = True
        stats['skip_reason'] = 'graph_not_dag'
        stats['compressed_nodes'] = len(G_compressed)
        stats['compressed_edges'] = G_compressed.number_of_edges()
        stats['compression_ratio'] = (
            stats['compressed_nodes'] / stats['original_nodes']
            if stats['original_nodes'] > 0 else 1.0
        )
        return G_compressed, stats

    for iteration in range(max_iterations):
        contracted_this_iteration = False

        try:
            topo_order = list(nx.topological_sort(G_compressed))
        except nx.NetworkXException as exc:
            logger.warning(f'Topological sort failed, skipping further contraction: {exc}')
            stats['skipped'] = True
            stats['skip_reason'] = 'topological_sort_failed'
            break

        # Iterate edges in topological order
        for u in topo_order:
            if u not in G_compressed:
                continue
            for v in list(G_compressed.successors(u)):
                if v not in G_compressed:
                    continue
                if can_contract(G_compressed, u, v):
                    logger.info(f"Contracting node {v} into {u}")
                    contract_node(G_compressed, u, v, merge_metadata)
                    stats['absorptions'].append((u, v))
                    contracted_this_iteration = True
                    break  # Restart after each contraction
            if contracted_this_iteration:
                break

        stats['iterations'] = iteration + 1

        if not contracted_this_iteration:
            break

    stats['compressed_nodes'] = len(G_compressed)
    stats['compressed_edges'] = G_compressed.number_of_edges()
    stats['compression_ratio'] = (
        stats['compressed_nodes'] / stats['original_nodes']
        if stats['original_nodes'] > 0 else 1.0
    )

    return G_compressed, stats


def compress_dag_combined(G: nx.DiGraph,
                          max_iterations: int = 100,
                          merge_metadata: bool = True,
                          use_contraction: bool = True,
                          use_parallel_fold: bool = True) -> Tuple[nx.DiGraph, Dict]:
    """Apply multiple compression strategies in sequence.

    Strategies:
    1. Connected-component contraction (vertical compression)
    2. Parallel same-layer folding (horizontal compression)

    Args:
        G: Original DiGraph with 'macro_action_tag' node attributes
        max_iterations: Maximum iterations per strategy
        merge_metadata: Whether to merge node metadata
        use_contraction: Whether to apply contraction strategy
        use_parallel_fold: Whether to apply parallel folding strategy

    Returns:
        Tuple of (compressed_graph, combined_stats)
    """
    G_result = G.copy()
    combined_stats = {
        'original_nodes': len(G),
        'original_edges': G.number_of_edges(),
        'strategies_applied': [],
        'strategy_errors': {},
    }

    # Strategy 1: Contraction
    if use_contraction:
        try:
            G_result, stats_contract = compress_dag(G_result, max_iterations, merge_metadata)
            combined_stats['contraction'] = stats_contract
            combined_stats['strategies_applied'].append('contraction')
        except Exception as exc:
            logger.warning(f'Contraction strategy failed, skipping: {exc}')
            combined_stats['strategy_errors']['contraction'] = str(exc)

    # Strategy 2: Parallel folding
    if use_parallel_fold:
        try:
            G_result, stats_fold = compress_parallel_layers(G_result, max_iterations, merge_metadata)
            combined_stats['parallel_fold'] = stats_fold
            combined_stats['strategies_applied'].append('parallel_fold')
        except Exception as exc:
            logger.warning(f'Parallel fold strategy failed, skipping: {exc}')
            combined_stats['strategy_errors']['parallel_fold'] = str(exc)

    combined_stats['final_nodes'] = len(G_result)
    combined_stats['final_edges'] = G_result.number_of_edges()
    combined_stats['total_compression_ratio'] = (
        combined_stats['final_nodes'] / combined_stats['original_nodes']
        if combined_stats['original_nodes'] > 0 else 1.0
    )

    return G_result, combined_stats


def build_digraph_with_tags(dag_analysis: List[Dict],
                            exclude_external: bool = False) -> nx.DiGraph:
    """Build DiGraph from dag_analysis with macro_action_tag attributes.

    Args:
        dag_analysis: List of dependency objects with macro_action_tag
        exclude_external: Whether to exclude External nodes

    Returns:
        DiGraph with 'type' and 'macro_action_tag' node attributes
    """
    G = nx.DiGraph()
    G.add_node(0, type="problem", macro_action_tag=None)

    for entry in dag_analysis:
        step_id = entry["step_id"]
        tag = entry.get("macro_action_tag")
        analysis = entry.get("analysis", "")

        if step_id not in G:
            G.add_node(step_id, type="step", macro_action_tag=tag, analysis=analysis)

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
