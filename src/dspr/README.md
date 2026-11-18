# DSPR model and staged training

`model.py` combines a frozen causal LM, prompt-context readout, a scalar router,
and two trainable soft-prefix tables. Training runs in two stages:

1. Train the router on canonical Simple/Hard prompts with targets 0/1.
2. Freeze the router and train the prefixes on selected CoT trajectories using
   language-model loss. Router BCE in this stage is a diagnostic only.

Both presets use the continuous router mixture for LM training and for validation
checkpoint selection. In `configs/dspr_ged.yaml` each sampling epoch draws one
trajectory per prompt. For the same response let `s0` and `s1` be its mean
answer-token log probabilities under the two fixed prefixes. The auxiliary loss
is `weight * softplus(-gap)` where `gap = s0 - s1` for Simple and `s1 - s0` for
Hard. With ascending within-prompt normalized GED rank `k` (average for ties)
among `n` scored trajectories, the weight is `1-k/(n+1)` for Simple and
`k/(n+1)` for Hard. Missing scores give zero weight. The trainer averages the
weighted losses over all sampled examples and scales them by `ged_preference_weight`.

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
