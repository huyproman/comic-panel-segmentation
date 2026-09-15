from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
import keras

from comics_panel_extraction.model.unetpp import build_unetpp
from comics_panel_extraction.training.config import TrainingConfig
from comics_panel_extraction.training.loss import RegionBCEDiceLoss
from comics_panel_extraction.training.optimizer import WarmupCosineDecaySchedule

class Trainer:
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        steps_per_epoch: Optional[int] = None,
    ):
        self.config = config or TrainingConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_unetpp(input_shape=self.config.input_shape)
        self.loss_fn = RegionBCEDiceLoss()

        total_steps = max(1, (steps_per_epoch or 100) * self.config.epochs)
        warmup_steps = max(1, (steps_per_epoch or 100) * self.config.warmup_epochs)
        self.lr_schedule = WarmupCosineDecaySchedule(
            base_learning_rate=self.config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
        )
        self.optimizer = keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=self.config.weight_decay,
        )

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> float:
        with tf.device('/CPU:0'):
            with tf.GradientTape() as tape:
                preds = self.model(x, training=True)
                loss = self.loss_fn(y, preds)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss.numpy())

    def eval_step(self, x: tf.Tensor, y: tf.Tensor) -> float:
        with tf.device('/CPU:0'):
            preds = self.model(x, training=False)
        loss = self.loss_fn(y, preds)
        return float(loss.numpy())

    def train_dataset(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        steps_per_epoch: int = 100,
        val_steps: int = 10,
    ) -> Dict[str, Any]:
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(self.config.epochs):
            epoch_losses = []
            for step, (x, y) in enumerate(train_ds):
                if step >= steps_per_epoch:
                    break
                loss_val = self.train_step(x, y)
                epoch_losses.append(loss_val)

            mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["train_loss"].append(mean_train_loss)

            current_step = (epoch + 1) * steps_per_epoch
            current_lr = float(self.lr_schedule(current_step).numpy())
            history["lr"].append(current_lr)

            if val_ds is not None:
                val_losses = []
                for v_step, (vx, vy) in enumerate(val_ds):
                    if v_step >= val_steps:
                        break
                    v_loss = self.eval_step(vx, vy)
                    val_losses.append(v_loss)
                mean_val_loss = float(np.mean(val_losses)) if val_losses else mean_train_loss
                history["val_loss"].append(mean_val_loss)
            else:
                mean_val_loss = mean_train_loss

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                self.model.save(str(self.output_dir / "best_model.keras"))

        self.model.save(str(self.output_dir / "latest_model.keras"))
        return {
            "best_loss": best_val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
        }
