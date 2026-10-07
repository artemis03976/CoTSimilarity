"""Offline context-vector cache for prompt-level router training."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm

from .dataset import RouterPromptDataset, router_prompt_key


def _metadata(model, max_length: int) -> dict:
    return {
        'model_name': model.config.model_name,
        'context_layer_idx': model.context_encoder.layer_idx,
        'max_seq_length': int(max_length),
    }


def _read_cache(path: Path, metadata: dict, dataset: RouterPromptDataset):
    payload = torch.load(path, map_location='cpu')
    if payload.get('metadata') != metadata:
        raise ValueError('cache metadata does not match the current model/configuration')
    keys = [tuple(key) for key in payload['keys']]
    expected = [router_prompt_key(item) for item in dataset.data]
    if set(keys) != set(expected):
        raise ValueError('cache prompts do not match the problem dataset')
    contexts = payload['contexts']
    if contexts.ndim != 2 or len(contexts) != len(keys):
        raise ValueError('cache has an invalid context tensor')
    return {key: contexts[index] for index, key in enumerate(keys)}


def load_or_build_router_context_cache(
    model,
    problem_dataset,
    tokenizer,
    max_length: int,
    cache_path,
    device: str,
    batch_size: int,
):
    """Return ``prompt_key -> context`` and persist it for later runs."""
    dataset = RouterPromptDataset(problem_dataset, tokenizer, max_length)
    metadata = _metadata(model, max_length)
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            contexts = _read_cache(cache_path, metadata, dataset)
            print(f'[router] loaded context cache: {cache_path} ({len(contexts)} prompts)', flush=True)
            return contexts
        except (KeyError, OSError, RuntimeError, ValueError):
            print(f'[router] ignoring incompatible context cache: {cache_path}', flush=True)

    model.to(device)
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=default_data_collator,
    )
    vectors = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Caching router contexts'):
            prompt_ids = batch['prompt_input_ids'].to(device)
            prompt_mask = batch['prompt_attention_mask'].to(device)
            vectors.append(
                model.context_encoder(prompt_ids, prompt_mask).detach().cpu()
            )
    contexts = torch.cat(vectors, dim=0)
    payload = {
        'metadata': metadata,
        'keys': [router_prompt_key(item) for item in dataset.data],
        'contexts': contexts,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # A PID-specific temporary file allows several fold processes to populate
    # the same shared cache without clobbering each other's rename step.
    temporary_path = cache_path.with_name(f'{cache_path.name}.{os.getpid()}.tmp')
    torch.save(payload, temporary_path)
    temporary_path.replace(cache_path)
    print(f'[router] wrote context cache: {cache_path} ({len(dataset)} prompts)', flush=True)
    return {
        router_prompt_key(item): contexts[index]
        for index, item in enumerate(dataset.data)
    }
