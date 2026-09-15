from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf

from comics_panel_extraction.data.config import DataConfig, DatasetConfig

class ComicDatasetLoader:
    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        images_dir: Optional[Union[str, Path]] = None,
        masks_dir: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = (448, 448),
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.target_size = target_size
        self.shuffle = shuffle
        self.seed = seed

        if dataset_dir is not None:
            base = Path(dataset_dir)
            self.images_dir = base / "images"
            self.masks_dir = base / "masks"
        else:
            self.images_dir = Path(images_dir) if images_dir else None
            self.masks_dir = Path(masks_dir) if masks_dir else None

        self.pairs: List[Tuple[Path, Path]] = []
        if self.images_dir and self.masks_dir and self.images_dir.exists() and self.masks_dir.exists():
            img_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
            for ip in img_files:
                mp = self.masks_dir / ip.name
                if mp.exists():
                    self.pairs.append((ip, mp))

    def __len__(self) -> int:
        return len(self.pairs)

    def load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img_p, mask_p = self.pairs[idx]
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image at {img_p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = im_resized.astype(np.float32) / 255.0

        mask_raw = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise ValueError(f"Failed to read mask at {mask_p}")
        mask_resized = cv2.resize(mask_raw, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_tensor = (mask_resized > 127).astype(np.float32)[..., np.newaxis]

        return img_tensor, mask_tensor

    def generator(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.pairs))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        for idx in indices:
            yield self.load_sample(int(idx))

def build_tf_dataset(
    dataset_dir: Optional[Union[str, Path]] = None,
    images_dir: Optional[Union[str, Path]] = None,
    masks_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    target_size: Tuple[int, int] = (448, 448),
    shuffle: bool = True,
    seed: int = 42,
    repeat: bool = True,
) -> tf.data.Dataset:
    loader = ComicDatasetLoader(
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        target_size=target_size,
        shuffle=shuffle,
        seed=seed,
    )

    if len(loader) == 0:
        raise ValueError(f"No valid image/mask pairs found in dataset: {dataset_dir or images_dir}")

    output_signature = (
        tf.TensorSpec(shape=(target_size[1], target_size[0], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(target_size[1], target_size[0], 1), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(loader.generator, output_signature=output_signature)
    if repeat:
        ds = ds.repeat()
    if batch_size > 0:
        ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
