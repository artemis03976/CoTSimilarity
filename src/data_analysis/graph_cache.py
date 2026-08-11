"""Persistent cache for compressed DAGs and their node text embeddings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import torch

from .dag_compressor import build_digraph_with_tags, compress_dag_combined


logger = logging.getLogger(__name__)

CACHE_FORMAT = "structured_cot_ged_graph_cache"
CACHE_SCHEMA_VERSION = 1
EMBEDDING_INDEX_ATTRIBUTE = "text_embedding_index"

GraphRecords = Dict[str, nx.DiGraph]


def _problem_id_from_custom_id(custom_id: str) -> Optional[int]:
    try:
        return int(custom_id.split("_", 1)[0])
    except (TypeError, ValueError):
        return None


def _select_dag_records(
    dag_records: Mapping[str, List[dict]],
    problem_ids: Optional[Set[int]],
    *,
    originals_only: bool = False,
) -> Dict[str, List[dict]]:
    selected = {}
    for custom_id, dag in dag_records.items():
        if problem_ids is not None:
            problem_id = _problem_id_from_custom_id(custom_id)
            if problem_id not in problem_ids:
                continue
        if originals_only and "_original" not in custom_id:
            continue
        selected[custom_id] = dag
    return selected


def prepare_compressed_graphs(
    dag_records: Mapping[str, List[dict]],
) -> GraphRecords:
    """Build and compress every annotated trajectory graph once."""
    graphs: GraphRecords = {}
    for custom_id, dag in dag_records.items():
        try:
            graph = build_digraph_with_tags(dag)
        except Exception as exc:
            logger.warning("Skip %s because graph construction failed: %s", custom_id, exc)
            continue

        try:
            graph, compression_stats = compress_dag_combined(graph)
        except Exception as exc:
            logger.warning(
                "Compression failed for %s; caching the uncompressed graph: %s",
                custom_id,
                exc,
            )
            compression_stats = {
                "skipped": True,
                "skip_reason": str(exc),
            }

        graph.graph["ged_prepared"] = True
        graph.graph["compression_stats"] = compression_stats
        graphs[custom_id] = graph
    return graphs


def _iter_graphs(*graph_groups: Mapping[str, nx.DiGraph]) -> Iterable[nx.DiGraph]:
    for group in graph_groups:
        yield from group.values()


def _collect_unique_step_texts(
    *graph_groups: Mapping[str, nx.DiGraph],
) -> Tuple[List[str], Dict[str, int]]:
    texts: List[str] = []
    text_to_index: Dict[str, int] = {}
    for graph in _iter_graphs(*graph_groups):
        for _, attrs in graph.nodes(data=True):
            if attrs.get("type") != "step":
                continue
            text = str(attrs.get("text", "")).strip()
            if text and text not in text_to_index:
                text_to_index[text] = len(texts)
                texts.append(text)
    return texts, text_to_index


def _attach_embeddings(
    graphs: Mapping[str, nx.DiGraph],
    embeddings: torch.Tensor,
    text_to_index: Mapping[str, int],
) -> None:
    embedding_array = embeddings.numpy()
    for graph in graphs.values():
        for _, attrs in graph.nodes(data=True):
            if attrs.get("type") != "step":
                continue
            text = str(attrs.get("text", "")).strip()
            embedding_index = text_to_index.get(text)
            if embedding_index is None:
                continue
            attrs[EMBEDDING_INDEX_ATTRIBUTE] = embedding_index
            attrs["text_embedding"] = embedding_array[embedding_index]


def _serialize_graph(graph: nx.DiGraph) -> dict:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        serialized_attrs = {
            name: value
            for name, value in attrs.items()
            if name != "text_embedding"
        }
        nodes.append({"id": node_id, "attributes": serialized_attrs})

    edges = [
        {"source": source, "target": target, "attributes": dict(attrs)}
        for source, target, attrs in graph.edges(data=True)
    ]
    return {
        "graph_attributes": dict(graph.graph),
        "nodes": nodes,
        "edges": edges,
    }


def _serialize_graphs(graphs: Mapping[str, nx.DiGraph]) -> Dict[str, dict]:
    return {
        custom_id: _serialize_graph(graph)
        for custom_id, graph in graphs.items()
    }


def _deserialize_graph(record: Mapping[str, object], embedding_array: np.ndarray) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.graph.update(dict(record.get("graph_attributes", {})))

    for node_record in record.get("nodes", []):
        attrs = dict(node_record.get("attributes", {}))
        embedding_index = attrs.get(EMBEDDING_INDEX_ATTRIBUTE)
        if embedding_index is not None:
            attrs["text_embedding"] = embedding_array[int(embedding_index)]
        graph.add_node(node_record["id"], **attrs)

    for edge_record in record.get("edges", []):
        graph.add_edge(
            edge_record["source"],
            edge_record["target"],
            **dict(edge_record.get("attributes", {})),
        )
    return graph


def _deserialize_graphs(
    records: Mapping[str, Mapping[str, object]],
    embedding_array: np.ndarray,
) -> GraphRecords:
    return {
        custom_id: _deserialize_graph(record, embedding_array)
        for custom_id, record in records.items()
    }


def build_graph_cache(
    variant_dag_records: Mapping[str, List[dict]],
    original_dag_records: Optional[Mapping[str, List[dict]]],
    cache_path: Path,
    encoder,
    *,
    problem_ids: Optional[Sequence[int]] = None,
    source_metadata: Optional[Mapping[str, object]] = None,
) -> Tuple[GraphRecords, GraphRecords, dict]:
    """Compress graphs, batch-encode node text, and write one reusable cache."""
    selected_problem_ids = set(problem_ids) if problem_ids is not None else None
    selected_variants = _select_dag_records(
        variant_dag_records,
        selected_problem_ids,
    )
    selected_originals = _select_dag_records(
        original_dag_records or {},
        selected_problem_ids,
        originals_only=True,
    )

    print(f"Preparing {len(selected_variants)} variant/reference trajectory graphs...")
    variant_graphs = prepare_compressed_graphs(selected_variants)
    print(f"Preparing {len(selected_originals)} explicit original-reference graphs...")
    original_graphs = prepare_compressed_graphs(selected_originals)

    texts, text_to_index = _collect_unique_step_texts(
        variant_graphs,
        original_graphs,
    )
    print(f"Encoding {len(texts)} unique compressed-node texts...")
    embeddings = encoder.encode(texts).to(dtype=torch.float32, device="cpu").contiguous()

    _attach_embeddings(variant_graphs, embeddings, text_to_index)
    _attach_embeddings(original_graphs, embeddings, text_to_index)

    metadata = {
        "format": CACHE_FORMAT,
        "schema_version": CACHE_SCHEMA_VERSION,
        "embedding": dict(encoder.cache_metadata),
        "unique_texts": len(texts),
        "variant_graphs": len(variant_graphs),
        "original_graphs": len(original_graphs),
        "problem_ids": sorted(selected_problem_ids) if selected_problem_ids else None,
        "sources": dict(source_metadata or {}),
    }
    payload = {
        "metadata": metadata,
        "embeddings": embeddings,
        "variant_graphs": _serialize_graphs(variant_graphs),
        "original_graphs": _serialize_graphs(original_graphs),
    }

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(cache_path)
    print(f"Compressed graph and text-embedding cache saved to: {cache_path}")
    return variant_graphs, original_graphs, metadata


def load_graph_cache(cache_path: Path) -> Tuple[GraphRecords, GraphRecords, dict]:
    """Load prepared graphs without initializing the embedding model."""
    cache_path = Path(cache_path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    metadata = dict(payload["metadata"])
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported graph cache format: {metadata.get('format')}")
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported graph cache schema version: "
            f"{metadata.get('schema_version')}"
        )

    embeddings = payload["embeddings"].to(dtype=torch.float32, device="cpu").contiguous()
    embedding_array = embeddings.numpy()
    variant_graphs = _deserialize_graphs(payload["variant_graphs"], embedding_array)
    original_graphs = _deserialize_graphs(payload["original_graphs"], embedding_array)
    return variant_graphs, original_graphs, metadata
