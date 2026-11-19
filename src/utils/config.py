"""YAML experiment settings shared by training, inference, and aggregation CLIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .io import atomic_text

SECTIONS = {"common", "training", "inference", "aggregation"}
COMMON_KEYS = {
    "model_name", "seed", "folds", "gpus", "problem_dataset", "fold_root",
    "checkpoint_root", "result_root", "prefix_length", "context_layer_idx",
    "router_intermediate_dim", "router_dropout", "max_seq_length",
}
GENERATION_COMMON_KEYS = {
    "model_name", "problem_dataset", "seed", "dtype", "max_model_len",
    "gpu_memory_utilization", "system_prompt", "instruction_placement",
}


def load_experiment(path, method=None):
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or payload.get("method") not in {"dspr", "spt", "lora", "generation"}:
        raise ValueError(f"Unknown experiment method in {path}")
    if method is not None and payload["method"] != method:
        raise ValueError(f"Config method {payload['method']} differs from CLI method {method}")
    sections, common_keys = SECTIONS, COMMON_KEYS
    if payload["method"] == "generation":
        sections, common_keys = {"common", "greedy", "sampling"}, GENERATION_COMMON_KEYS
    unknown_sections = set(payload) - sections - {"method"}
    if unknown_sections:
        raise ValueError(f"Unknown config sections in {path}: {sorted(unknown_sections)}")
    for section in sections:
        if not isinstance(payload.get(section, {}), dict):
            raise ValueError(f"{path}: {section} must be a mapping")
    unknown = set(payload.get("common", {})) - common_keys
    if unknown:
        raise ValueError(f"Unknown common settings: {sorted(unknown)}")
    return payload


def parse_config_args(parser, argv=None, *, method, section, aliases=None):
    """Apply YAML defaults before CLI overrides, validating through CLI actions.

    Common settings apply to each relevant entrypoint. Section-specific settings
    must be recognized, so misspelled training/inference options cannot disappear.
    Paths are relative to the repository working directory, as in the CLI.
    """
    parser.add_argument("--config", type=str, help="YAML experiment; explicit CLI options override it")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    config_path = probe.parse_known_args(argv)[0].config
    if config_path:
        config = load_experiment(config_path, method)
        actions = {action.dest: action for action in parser._actions}
        aliases = aliases or {}
        defaults = {}
        for source in ("common", section):
            for key, value in config.get(source, {}).items():
                destinations = aliases.get(key, (key,))
                if isinstance(destinations, str):
                    destinations = (destinations,)
                for dest in destinations:
                    if dest not in actions:
                        if source == "common":
                            continue
                        parser.error(f"Unknown {section} option in {config_path}: {key}")
                    action = actions[dest]
                    if value is None:
                        defaults[dest] = None
                        continue
                    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                           argparse.BooleanOptionalAction)):
                        if not isinstance(value, bool):
                            parser.error(f"{key} must be a YAML boolean")
                        converted = value
                    else:
                        multiple = action.nargs in ("+", "*") or isinstance(action.nargs, int)
                        if multiple and not isinstance(value, list):
                            parser.error(f"{key} must be a YAML list")
                        values = value if multiple else [value]
                        try:
                            converted_values = [action.type(v) if action.type else v for v in values]
                        except (TypeError, ValueError) as exc:
                            parser.error(f"Invalid {key}: {exc}")
                        if action.choices and any(v not in action.choices for v in converted_values):
                            parser.error(f"{key} must be one of {list(action.choices)}")
                        converted = converted_values if multiple else converted_values[0]
                    defaults[dest] = converted
                    action.required = False
        explicit = {
            parser._option_string_actions[token.split("=", 1)[0]].dest
            for token in (sys.argv[1:] if argv is None else argv)
            if token.split("=", 1)[0] in parser._option_string_actions
        }
        for group in parser._mutually_exclusive_groups:
            names = {action.dest for action in group._group_actions}
            if names & explicit:
                for name in names - explicit:
                    defaults.pop(name, None)
            elif sum(defaults.get(name) is not None for name in names) > 1:
                parser.error(f"Mutually exclusive YAML settings: {sorted(names)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def write_resolved_config(path, args):
    """Save the effective settings without requiring the original YAML later."""
    values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with atomic_text(path) as stream:
        yaml.safe_dump(values, stream, allow_unicode=True, sort_keys=False)
