#!/usr/bin/env python3
"""Staged DSPR training through `python scripts/train.py dspr`.

Stage 1 trains the prompt-only router on one example per problem-variant.
Stage 2 freezes that router and trains the dual prefix on CoT trajectories.
Prefix training can use predicted mixtures or binary labels; validation and
default inference always use the predicted continuous mixture.
Either stage can also run separately.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback, default_data_collator

from dspr.config import DSPRConfig
from dspr.model import DSPRModel
from dspr_training.dataset import RouterPromptDataset
from dspr_training.router_cache import load_or_build_router_context_cache
from dspr_training.router_trainer import RouterPromptTrainer, compute_router_metrics
from dspr_training.trainer import DSPRTrainer, compute_dspr_metrics
from utils.config import parse_config_args, write_resolved_config
from dspr_dataset.ged_weights import assign_ged_weights, load_ged_scores, resolve_prefix_sampling
from utils.datasets import TrajectoryDataset, ProblemResampledDataset
from utils.io import read_json, read_jsonl
from utils.training import (
    KeepFrozenModulesEvalCallback, ProgressEventCallback, TrainingHistoryCallback, set_seed,
    training_arguments, update_steps_per_epoch,
)


def make_training_args(
    *,
    output_dir: Path,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    warmup_ratio: float | None,
    warmup_steps: int,
    max_steps: int,
    eval_strategy: str,
    eval_steps: int | None,
    seed: int,
    device: str,
    logging_steps: int,
    save_total_limit: int,
    label_names: list[str],
    emit_progress_events: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        warmup_steps=warmup_steps if warmup_ratio is None else 0,
        warmup_ratio=0.0 if warmup_ratio is None else warmup_ratio,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_strategy=eval_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=eval_strategy != 'no',
        dataloader_pin_memory=True,
        disable_tqdm=emit_progress_events,
        fp16=(device.startswith('cuda') and torch.cuda.is_available()),
        seed=seed,
        label_names=label_names,
    )
    if eval_strategy != 'no' and eval_steps is not None:
        kwargs['eval_steps'] = eval_steps
        kwargs['save_steps'] = eval_steps
    return training_arguments(eval_strategy=eval_strategy, **kwargs)


def add_early_stopping(callbacks: list[TrainerCallback], patience: int, threshold: float) -> None:
    if patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )


def load_router_state(model: DSPRModel, checkpoint: Path, device: str) -> None:
    if checkpoint.is_dir():
        checkpoint = checkpoint / 'router_trainable.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Router checkpoint not found: {checkpoint}')
    payload = torch.load(checkpoint, map_location=device)
    model.router.load_state_dict(payload['router_state_dict'])


def build_config(args: argparse.Namespace) -> DSPRConfig:
    names = {field.name for field in fields(DSPRConfig)}
    values = {key: value for key, value in vars(args).items() if key in names}
    return DSPRConfig(**values)


def read_id_manifest(path: Path) -> set[int]:
    return {int(value) for value in read_json(path)['all']}


def read_source_map(path: Path) -> dict[int, str]:
    source_map = {}
    for item in read_jsonl(path):
        problem_id = int(item['problem_id'])
        source = str(item.get('source', 'unknown'))
        if problem_id in source_map and source_map[problem_id] != source:
            raise ValueError(f'Conflicting source for problem_id={problem_id} in {path}')
        source_map[problem_id] = source
    return source_map


def parse_source_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    sources = {item.strip() for item in value.split(',') if item.strip()}
    if not sources:
        raise ValueError('Source filter must contain at least one non-empty source name')
    return sources


def resolve_data_paths(args: argparse.Namespace) -> dict[str, Path | set[int] | None]:
    """Resolve canonical router source and fold-specific prefix inputs."""
    problem_dataset = Path(args.problem_dataset)
    if not problem_dataset.is_file():
        raise FileNotFoundError(f'Problem dataset not found: {problem_dataset}')

    prefix_train = Path(args.prefix_train_data_path) if args.prefix_train_data_path else None
    prefix_val = Path(args.prefix_val_data_path) if args.prefix_val_data_path else None
    train_ids = Path(args.router_train_ids) if args.router_train_ids else None
    val_ids = Path(args.router_val_ids) if args.router_val_ids else None
    eval_ids = Path(args.router_eval_ids) if args.router_eval_ids else None

    if args.fold is not None:
        fold_dir = Path(args.kfold_root) / f'fold_{args.fold}'
        if not fold_dir.is_dir():
            raise FileNotFoundError(f'K-fold directory not found: {fold_dir}')
        prefix_train = prefix_train or (fold_dir / 'train.jsonl')
        prefix_val = prefix_val or (fold_dir / 'val.jsonl')
        train_ids = train_ids or (fold_dir / 'train_ids.json')
        val_ids = val_ids or (fold_dir / 'val_ids.json')

    if args.run_stage in ('all', 'prefix'):
        if args.ged_preference_weight and args.ged_results and not Path(args.ged_results).is_file():
            raise FileNotFoundError(f'Prefix GED file not found: {args.ged_results}')
        if prefix_train is None or not prefix_train.is_file():
            raise FileNotFoundError(
                'Prefix training data is required. Set --fold/--kfold-root or '
                '--prefix-train-data-path.'
            )
        if prefix_val is not None and not prefix_val.is_file():
            raise FileNotFoundError(f'Prefix validation data not found: {prefix_val}')

    if args.run_stage in ('all', 'router'):
        if train_ids is None:
            raise ValueError(
                'Router training IDs are required. Set --fold/--kfold-root or '
                '--router-train-ids.'
            )
        router_train_ids = read_id_manifest(train_ids)
        router_val_ids = read_id_manifest(val_ids) if val_ids is not None else None
        if router_val_ids is not None and router_train_ids & router_val_ids:
            raise ValueError('Router train/validation ID manifests overlap')
        router_eval_ids = read_id_manifest(eval_ids) if eval_ids is not None else None
        if router_eval_ids is not None and router_train_ids & router_eval_ids:
            raise ValueError('Router train/evaluation ID manifests overlap')
        if router_eval_ids is not None and router_val_ids is not None and router_val_ids & router_eval_ids:
            raise ValueError('Router validation/evaluation ID manifests overlap')

        train_sources = parse_source_filter(args.router_train_sources)
        if train_sources is not None:
            source_map = read_source_map(problem_dataset)
            unknown_ids = router_train_ids - set(source_map)
            if unknown_ids:
                raise ValueError(
                    f'Router train IDs are absent from canonical source: {sorted(unknown_ids)[:10]}'
                )
            router_train_ids = {
                problem_id for problem_id in router_train_ids
                if source_map[problem_id] in train_sources
            }
            if not router_train_ids:
                raise ValueError(
                    f'Router source filter selected no training IDs: {sorted(train_sources)}'
                )
    else:
        router_train_ids = None
        router_val_ids = None
        router_eval_ids = None

    return {
        'problem_dataset': problem_dataset,
        'prefix_train': prefix_train,
        'prefix_val': prefix_val,
        'router_train_ids': router_train_ids,
        'router_val_ids': router_val_ids,
        'router_eval_ids': router_eval_ids,
    }


def run_router_stage(
    model: DSPRModel,
    args: argparse.Namespace,
    config: DSPRConfig,
    problem_dataset: Path,
    train_ids: set[int],
    val_ids: set[int] | None,
    eval_ids: set[int] | None = None,
) -> Path:
    stage_dir = Path(args.output_path) / 'router'
    train_dataset = RouterPromptDataset(
        problem_dataset,
        model.tokenizer,
        config.max_seq_length,
        problem_ids=train_ids,
    )
    val_dataset = None
    if val_ids is not None:
        val_dataset = RouterPromptDataset(
            problem_dataset,
            model.tokenizer,
            config.max_seq_length,
            problem_ids=val_ids,
        )
    eval_dataset = None
    if eval_ids is not None:
        eval_dataset = RouterPromptDataset(
            problem_dataset,
            model.tokenizer,
            config.max_seq_length,
            problem_ids=eval_ids,
        )
    cache_path = (
        Path(args.router_context_cache_path)
        if args.router_context_cache_path
        else Path(args.output_path) / 'router_context_cache.pt'
    )
    contexts = load_or_build_router_context_cache(
        model,
        problem_dataset,
        model.tokenizer,
        config.max_seq_length,
        cache_path,
        args.device,
        args.batch_size,
    )
    train_dataset.set_contexts(contexts)
    if val_dataset is not None:
        val_dataset.set_contexts(contexts)
    if eval_dataset is not None:
        eval_dataset.set_contexts(contexts)
    callbacks: list[TrainerCallback] = [
        TrainingHistoryCallback(write_train_log=True),
        KeepFrozenModulesEvalCallback('frozen_llm'),
    ]
    if args.emit_progress_events:
        callbacks.append(ProgressEventCallback(stage='router'))
    if val_dataset is not None:
        add_early_stopping(callbacks, args.router_early_stopping_patience, args.early_stopping_threshold)

    model.dual_prefix.requires_grad_(False)
    model.router.requires_grad_(True)
    training_args = make_training_args(
        output_dir=stage_dir,
        learning_rate=args.router_learning_rate,
        weight_decay=args.router_weight_decay,
        num_epochs=args.router_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=-1,
        eval_strategy='epoch' if val_dataset is not None else 'no',
        eval_steps=None,
        seed=config.seed,
        device=config.device,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        label_names=['target_alpha'],
        emit_progress_events=args.emit_progress_events,
    )
    trainer = RouterPromptTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_router_metrics if (val_dataset is not None or eval_dataset is not None) else None,
        target_smoothing=args.router_target_smoothing,
        positive_weight=args.router_positive_weight,
        callbacks=callbacks,
    )
    print(
        f'[router] examples={len(train_dataset)} classes={train_dataset.class_counts} '
        f'lr={args.router_learning_rate:g} weight_decay={args.router_weight_decay:g} '
        f'cached_contexts={cache_path}',
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(stage_dir))
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix='router_eval',
        )
        (stage_dir / 'router_eval_metrics.json').write_text(
            json.dumps(eval_metrics, ensure_ascii=False, indent=2, default=str) + '\n',
            encoding='utf-8',
        )
        print(f'[router] evaluation metrics: {stage_dir / "router_eval_metrics.json"}', flush=True)
    print(f'[router] checkpoint: {stage_dir / "router_trainable.pt"}', flush=True)
    return stage_dir / 'router_trainable.pt'


def build_prefix_datasets(
    model: DSPRModel,
    args: argparse.Namespace,
    config: DSPRConfig,
    train_path: Path,
    val_path: Path | None,
):
    options = dict(tokenizer=model.tokenizer, max_length=config.max_seq_length, include_router_inputs=True)
    resampled = None
    if args.prefix_trajectory_sampling == 'random_one':
        resampled = ProblemResampledDataset(train_path, seed=config.seed, **options)
        train_dataset = resampled
    else:
        train_dataset = TrajectoryDataset(train_path, **options)
    flat_steps = update_steps_per_epoch(
        len(train_dataset.data), config.batch_size, config.gradient_accumulation_steps
    )
    val_dataset = None
    if val_path is not None:
        val_dataset = TrajectoryDataset(val_path, **options)
    if args.ged_preference_weight:
        scores = load_ged_scores(args.ged_results) if args.ged_results else None
        for split, dataset in (('train', train_dataset), ('val', val_dataset)):
            if dataset is not None:
                matched = assign_ged_weights(dataset.data, scores)
                print(f'[prefix] {split} GED coverage: {matched}/{len(dataset.data)} trajectories', flush=True)
                if split == 'train' and not matched:
                    raise ValueError('No usable training GED. Supply --ged-results or fold rows with ged_normalized.')
    return train_dataset, val_dataset, resampled, flat_steps


def run_prefix_stage(
    model: DSPRModel,
    args: argparse.Namespace,
    config: DSPRConfig,
    train_path: Path,
    val_path: Path | None,
    stage_name: str = 'prefix',
) -> Path:
    train_dataset, val_dataset, resampled, flat_steps = build_prefix_datasets(
        model, args, config, train_path, val_path
    )
    stage_dir = Path(args.output_path) / stage_name
    callbacks: list[TrainerCallback] = [
        TrainingHistoryCallback(write_train_log=True, resampled_dataset=resampled),
        KeepFrozenModulesEvalCallback('router', 'frozen_llm'),
    ]
    if args.emit_progress_events:
        callbacks.append(ProgressEventCallback(stage=stage_name))
    add_early_stopping(callbacks, args.prefix_early_stopping_patience, args.early_stopping_threshold)

    model.router.requires_grad_(False)
    model.dual_prefix.requires_grad_(True)
    if resampled is not None:
        max_steps = flat_steps * math.ceil(args.prefix_epochs)
        eval_strategy = 'steps' if val_dataset is not None else 'no'
        eval_steps = flat_steps if val_dataset is not None else None
    else:
        max_steps = -1
        eval_strategy = 'epoch' if val_dataset is not None else 'no'
        eval_steps = None
    training_args = make_training_args(
        output_dir=stage_dir,
        learning_rate=args.prefix_learning_rate,
        weight_decay=args.prefix_weight_decay,
        num_epochs=args.prefix_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        seed=config.seed,
        device=config.device,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        label_names=['labels', 'target_alpha'],
        emit_progress_events=args.emit_progress_events,
    )
    trainer = DSPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_dspr_metrics if val_dataset is not None else None,
        ged_preference_weight=args.ged_preference_weight,
        callbacks=callbacks,
    )
    print(
        f'[{stage_name}] trajectories={len(train_dataset)} sampling={args.prefix_trajectory_sampling} '
        f'ged_preference_weight={args.ged_preference_weight:g} '
        f'lr={args.prefix_learning_rate:g} weight_decay={args.prefix_weight_decay:g}',
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(stage_dir))
    print(f'[{stage_name}] checkpoint: {stage_dir / "dspr_trainable.pt"}', flush=True)
    return stage_dir / 'dspr_trainable.pt'


def parse_args() -> argparse.Namespace:
    defaults = DSPRConfig()
    parser = argparse.ArgumentParser(description='Train DSPR in router -> prefix stages.')
    parser.add_argument('--run-stage', choices=('all', 'router', 'prefix'), default='all')
    parser.add_argument('--model-name', '--model_name', default=defaults.model_name)
    parser.add_argument('--context-layer-idx', '--context_layer_idx', type=int, default=defaults.context_layer_idx)
    parser.add_argument('--prefix-length', '--prefix_length', type=int, default=defaults.prefix_length)
    parser.add_argument('--router-intermediate-dim', '--router_intermediate_dim', type=int, default=defaults.router_intermediate_dim)
    parser.add_argument('--router-dropout', '--router_dropout', type=float, default=defaults.router_dropout)
    parser.add_argument(
        '--problem-dataset', '--problem_dataset',
        default='data/canonical_math_paired.jsonl',
        help='Canonical source containing original/simple/hard problem variants.',
    )
    parser.add_argument(
        '--kfold-root', '--kfold_root',
        default='output/qwen-2.5/kfold',
        help='Prepared K-fold root; each fold contains train/val JSONL and ID manifests.',
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help='Use fold_N/train.jsonl and fold_N/val.jsonl for prefix training and ID manifests for router training.',
    )
    parser.add_argument(
        '--router-train-ids', '--router_train_ids', default=None,
        help='Optional JSON ID manifest with an all list. IDs select examples from --problem-dataset; prompt text is never read from this file.',
    )
    parser.add_argument(
        '--router-val-ids', '--router_val_ids', default=None,
        help='Optional validation JSON ID manifest with an all list. IDs select examples from --problem-dataset.',
    )
    parser.add_argument(
        '--router-eval-ids', '--router_eval_ids', default=None,
        help='Optional held-out JSON ID manifest with an all list. Metrics are written to router/router_eval_metrics.json.',
    )
    parser.add_argument(
        '--router-train-sources', '--router_train_sources', default=None,
        help='Optional comma-separated source filter for router training IDs (for example math_perturb).',
    )
    parser.add_argument(
        '--router-context-cache-path', '--router_context_cache_path', default=None,
        help='Context cache path; defaults to <output-path>/router_context_cache.pt.',
    )
    parser.add_argument(
        '--prefix-train-data-path', '--prefix_train_data_path', default=None,
        help='Optional refined trajectory train JSONL. With --fold, defaults to fold_N/train.jsonl.',
    )
    parser.add_argument(
        '--prefix-val-data-path', '--prefix_val_data_path', default=None,
        help='Optional refined trajectory validation JSONL. With --fold, defaults to fold_N/val.jsonl.',
    )
    parser.add_argument('--router-checkpoint', '--router_checkpoint', default=None)
    parser.add_argument('--output-path', '--output_path', default=None)
    parser.add_argument('--ged-preference-weight', type=float, default=0.0)
    parser.add_argument('--ged-results', help='Normalized trajectory GED JSONL for folds without embedded scores')
    parser.add_argument('--router-learning-rate', '--router_learning_rate', type=float, default=1e-5)
    parser.add_argument('--prefix-learning-rate', '--prefix_learning_rate', type=float, default=4e-5)
    parser.add_argument('--router-weight-decay', '--router_weight_decay', type=float, default=1e-3)
    parser.add_argument('--prefix-weight-decay', '--prefix_weight_decay', type=float, default=0.0)
    parser.add_argument('--router-target-smoothing', '--router_target_smoothing', type=float, default=0.05)
    parser.add_argument('--router-positive-weight', '--router_positive_weight', type=float, default=None)
    parser.add_argument('--router-epochs', '--router_epochs', type=int, default=20)
    parser.add_argument('--prefix-epochs', '--prefix_epochs', type=int, default=15)
    parser.add_argument('--batch-size', '--batch_size', type=int, default=defaults.batch_size)
    parser.add_argument('--gradient-accumulation-steps', '--gradient_accumulation_steps', type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument('--max-grad-norm', '--max_grad_norm', type=float, default=defaults.max_grad_norm)
    parser.add_argument('--warmup-steps', '--warmup_steps', type=int, default=defaults.warmup_steps)
    parser.add_argument('--warmup-ratio', '--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--max-seq-length', '--max_seq_length', type=int, default=defaults.max_seq_length)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--device', default=defaults.device)
    parser.add_argument('--logging-steps', '--logging_steps', type=int, default=10)
    parser.add_argument('--emit-progress-events', '--emit_progress_events', action='store_true')
    parser.add_argument('--save-total-limit', '--save_total_limit', type=int, default=3)
    parser.add_argument('--early-stopping-threshold', '--early_stopping_threshold', type=float, default=0.0)
    parser.add_argument('--router-early-stopping-patience', '--router_early_stopping_patience', type=int, default=3)
    parser.add_argument('--prefix-early-stopping-patience', '--prefix_early_stopping_patience', type=int, default=3)
    parser.add_argument('--prefix-trajectory-sampling', '--prefix_trajectory_sampling', choices=('flat', 'random_one'))
    parser.add_argument('--gradient-checkpointing', '--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=defaults.gradient_checkpointing)
    return parse_config_args(parser, method='dspr', section='training', aliases={
        'fold_root': 'kfold_root', 'checkpoint_root': 'output_path',
    })


def main() -> int:
    args = parse_args()
    resolve_prefix_sampling(args)
    if args.output_path is None:
        args.output_path = 'checkpoints/staged_dspr_ged' if args.ged_preference_weight else 'checkpoints/staged_dspr'
    if args.router_learning_rate <= 0 or args.prefix_learning_rate <= 0:
        raise ValueError('All learning rates must be positive')
    if args.router_weight_decay < 0 or args.prefix_weight_decay < 0:
        raise ValueError('Weight decay must be non-negative')
    if args.warmup_ratio is not None and not 0 <= args.warmup_ratio < 1:
        raise ValueError('warmup_ratio must be in [0, 1)')
    if args.run_stage == 'prefix' and not args.router_checkpoint:
        raise ValueError('--router-checkpoint is required when --run-stage prefix')

    paths = resolve_data_paths(args)

    config = build_config(args)
    config.device = args.device
    set_seed(args.seed)
    model = DSPRModel(config)
    root = Path(args.output_path)
    root.mkdir(parents=True, exist_ok=True)
    write_resolved_config(root / 'training_config.yaml', args)

    router_checkpoint = Path(args.router_checkpoint) if args.router_checkpoint else None
    if args.run_stage in ('all', 'router'):
        router_checkpoint = run_router_stage(
            model,
            args,
            config,
            paths['problem_dataset'],
            paths['router_train_ids'],
            paths['router_val_ids'],
            paths['router_eval_ids'],
        )
    elif router_checkpoint is not None:
        load_router_state(model, router_checkpoint, args.device)

    if args.run_stage in ('all', 'prefix'):
        run_prefix_stage(
            model,
            args,
            config,
            paths['prefix_train'],
            paths['prefix_val'],
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
