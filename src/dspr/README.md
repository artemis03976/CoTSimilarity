# DSPR model and staged training

`model.py` combines a frozen causal LM, prompt-context readout, a scalar router,
and two trainable soft-prefix tables. Training runs in two stages:

1. Train the router on canonical Simple/Hard prompts with targets 0/1.
2. Freeze the router and train the prefixes on selected CoT trajectories using
   language-model loss. Router BCE in this stage is a diagnostic only.

`prefix_routing: router` mixes prefixes using the frozen router during training.
`prefix_routing: label` selects the corresponding prefix using each example's
binary target. Validation and default inference use the continuous router mixture
in both modes.

The router dataset lives in `src/dspr_training/dataset.py`; all methods share
trajectory encoding in `src/utils/datasets.py`. Both training stages share the
optimizer construction in `src/utils/training.py`, with learning rate and weight
decay supplied separately for each stage.

Run the full pipeline with a YAML configuration:

```bash
python scripts/run_pipeline.py dspr --config /path/to/dspr.yaml --gpus 0,1
```

See the [root README](../../README.md) for configuration, single-stage commands,
offline scoring, and aggregation. The checkpoint weight fields remain
`router_state_dict` and `dual_prefix_state_dict`.
