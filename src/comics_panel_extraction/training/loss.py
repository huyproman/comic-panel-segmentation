from typing import Tuple
import tensorflow as tf
import keras
@keras.saving.register_keras_serializable(package="comics_panel_extraction")
class RegionBCEDiceLoss(keras.losses.Loss):
    def __init__(self, eps: float = 1e-7, name: str = "region_bce_dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        p = tf.clip_by_value(y_pred, self.eps, 1.0 - self.eps)
        bce = - (y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p))
        bce_loss = tf.reduce_mean(bce)
        intersection = tf.reduce_sum(y_true * p, axis=[1, 2, 3])
        cardinality = tf.reduce_sum(y_true + p, axis=[1, 2, 3])
        dice_coeff = (2.0 * intersection + 1.0) / (cardinality + 1.0)
        dice_loss = tf.reduce_mean(1.0 - dice_coeff)
        return bce_loss + dice_loss
    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config
def compute_sample_ious(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.50) -> tf.Tensor:
    y_true_bin = tf.cast(y_true > 0.5, tf.float32)
    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)
    intersection = tf.reduce_sum(y_true_bin * y_pred_bin, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true_bin + y_pred_bin - (y_true_bin * y_pred_bin), axis=[1, 2, 3])
    sample_ious = tf.where(union > 0.0, intersection / union, tf.ones_like(union))
    return sample_ious
def compute_batch_miou(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.50) -> tf.Tensor:
    return tf.reduce_mean(compute_sample_ious(y_true, y_pred, threshold=threshold))
