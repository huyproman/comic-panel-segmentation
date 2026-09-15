import math
import tensorflow as tf
import keras
class WarmupCosineDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        base_learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        alpha: float = 0.0,
        name: str = "WarmupCosineDecay",
    ):
        super().__init__()
        self.base_learning_rate = float(base_learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.alpha = float(alpha)
        self.name = name
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(self.warmup_steps, tf.float32)
        decay_steps_f = tf.cast(self.decay_steps, tf.float32)
        def warmup_lr():
            return self.base_learning_rate * (step / tf.maximum(warmup_steps_f, 1.0))
        def cosine_decay_lr():
            completed_fraction = (step - warmup_steps_f) / tf.maximum(decay_steps_f - warmup_steps_f, 1.0)
            completed_fraction = tf.clip_by_value(completed_fraction, 0.0, 1.0)
            cosine_decayed = 0.5 * (1.0 + tf.cos(math.pi * completed_fraction))
            decayed = (1.0 - self.alpha) * cosine_decayed + self.alpha
            return self.base_learning_rate * decayed
        return tf.where(step < warmup_steps_f, warmup_lr(), cosine_decay_lr())
    def get_config(self):
        return {
            "base_learning_rate": self.base_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
        }
def build_optimizer(learning_rate: float = 1e-4, weight_decay: float = 1e-4):
    return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
