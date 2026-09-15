from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
@dataclass
class TrainingConfig:
    epochs: int = 150
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    input_shape: Tuple[int, int, int] = (448, 448, 3)
    dataset_dir: Optional[str] = None
    output_dir: str = "runs/train"
    seed: int = 42
