# DCPR Model Implementation

This directory contains the implementation of the Dynamic Cognitive Prefix Routing (DCPR) model.

## Structure

```
src/
├── dcpr/                      # Core model components
│   ├── model.py              # Main DCPRModel
│   ├── router.py             # UCB Router MLP
│   ├── context_encoder.py    # Context extraction
│   ├── dual_prefix.py        # Dual cognitive prefixes
│   └── config.py             # Configuration
│
└── dcpr_training/            # Training pipeline
    ├── dataset.py            # Dataset loader
    ├── trainer.py            # Training loop
    ├── loss.py               # Joint loss function
    └── data_filter.py        # GED-based filtering (interface only)
```

## Model Architecture

- **Frozen LLM**: Qwen2.5-Math-7B-Instruct (7B parameters, frozen)
- **Trainable Components**: ~1.2M parameters
  - P_exploit: (20, 4096) - Conservative reasoning prefix
  - P_explore: (20, 4096) - Divergent reasoning prefix
  - UCB Router: MLP (4096 → 256 → 1) - Predicts exploration weight α

## Training

```bash
python scripts/train_dcpr.py
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
- `ged_score`: Graph edit distance (placeholder for now)

## Key Features

1. **Single-input architecture**: No need for original problem at inference
2. **Lightweight**: Only ~1.2M trainable parameters
3. **Joint loss**: L_total = L_LLM_Gen + λ * L_Router
4. **Memory efficient**: Fits in 24GB GPU with mixed precision
