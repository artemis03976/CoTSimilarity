# DSPR Model Implementation

This directory contains the implementation of the Dynamic Structural Prefix Routing (DSPR) model.

## Structure

```
src/
├── dspr/                      # Core model components
│   ├── model.py              # Main DSPRModel
│   ├── router.py             # UCB Router MLP
│   ├── context_encoder.py    # Context extraction
│   ├── dual_prefix.py        # Dual structural prefixes
│   └── config.py             # Configuration
│
├── dspr_training/            # Training pipeline
    ├── dataset.py            # Dataset loader
    ├── trainer.py            # Training loop
│   └── loss.py               # Joint loss function
│
└── dspr_dataset/             # Dataset construction utilities
    ├── data_filter.py        # GED-based trajectory filtering
    └── split_dataset.py      # Train/validation/test split by problem_id
```

## Training

```bash
python scripts/train_dspr.py
```

## Dataset Format

Expected JSONL format:
```json
{
  "problem_id": 1,
  "problem": "Problem text...",
  "response": "Response text...",
  "variant_type": "simple",
  "target_alpha": 0.0,
  "ged_score": 2.5
}
```

- `variant_type`: "simple" or "hard"
- `target_alpha`: 0.0 for simple (exploit), 1.0 for hard (explore)
- `ged_score`: Graph edit distance used to select training samples

## Key Features

1. **Single-input architecture**: No need for original problem at inference
2. **Lightweight**: Only ~1.2M trainable parameters
3. **Joint loss**: L_total = L_LLM_Gen + λ * L_Router
4. **Memory efficient**: Fits in 24GB GPU with mixed precision
