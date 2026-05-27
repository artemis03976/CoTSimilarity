from dataclasses import dataclass


@dataclass
class SPTConfig:
    """Configuration for the Static Prompt Tuning baseline."""

    # Model architecture
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    prefix_length: int = 50

    # Training
    learning_rate: float = 4e-5
    batch_size: int = 4
    num_epochs: int = 15
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Data
    train_data_path: str = "data/qwen/dspr_train.jsonl"
    val_data_path: str = "data/qwen/dspr_val.jsonl"
    checkpoint_dir: str = "checkpoints"
    max_seq_length: int = 2048

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "cuda"
    gradient_checkpointing: bool = True
