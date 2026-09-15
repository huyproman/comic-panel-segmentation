from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

@dataclass
class DatasetConfig:
    images_dir: Union[str, Path]
    masks_dir: Union[str, Path]
    target_height: int = 448
    target_width: int = 448
    color_order: str = "RGB"
    normalize_image: bool = True
    batch_size: int = 8
    shuffle: bool = True
    seed: int = 42

@dataclass
class DataConfig:
    dataset_dir: Optional[Union[str, Path]] = None
    images_dir: Optional[Union[str, Path]] = None
    masks_dir: Optional[Union[str, Path]] = None
    target_height: int = 448
    target_width: int = 448
    batch_size: int = 8
    shuffle: bool = True
    seed: int = 42
