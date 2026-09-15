from typing import Tuple
import keras
from keras import layers, models
def conv_block(x: keras.KerasTensor, filters: int, name: str) -> keras.KerasTensor:
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x
def build_unetpp(
    input_shape: Tuple[int, int, int] = (448, 448, 3),
    nb_filter=(32, 64, 128, 256, 512)
) -> keras.Model:
    inputs = layers.Input(shape=input_shape, name="input_image")
    x_00 = conv_block(inputs, nb_filter[0], "node_00")
    p0 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool_0")(x_00)
    x_10 = conv_block(p0, nb_filter[1], "node_10")
    p1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool_1")(x_10)
    up_10_to_01 = layers.Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), padding="same", name="up_10_01"
    )(x_10)
    cat_01 = layers.Concatenate(axis=-1, name="cat_01")([x_00, up_10_to_01])
    x_01 = conv_block(cat_01, nb_filter[0], "node_01")
    x_20 = conv_block(p1, nb_filter[2], "node_20")
    p2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool_2")(x_20)
    up_20_to_11 = layers.Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), padding="same", name="up_20_11"
    )(x_20)
    cat_11 = layers.Concatenate(axis=-1, name="cat_11")([x_10, up_20_to_11])
    x_11 = conv_block(cat_11, nb_filter[1], "node_11")
    up_11_to_02 = layers.Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), padding="same", name="up_11_02"
    )(x_11)
    cat_02 = layers.Concatenate(axis=-1, name="cat_02")([x_00, x_01, up_11_to_02])
    x_02 = conv_block(cat_02, nb_filter[0], "node_02")
    x_30 = conv_block(p2, nb_filter[3], "node_30")
    p3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool_3")(x_30)
    up_30_to_21 = layers.Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), padding="same", name="up_30_21"
    )(x_30)
    cat_21 = layers.Concatenate(axis=-1, name="cat_21")([x_20, up_30_to_21])
    x_21 = conv_block(cat_21, nb_filter[2], "node_21")
    up_21_to_12 = layers.Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), padding="same", name="up_21_12"
    )(x_21)
    cat_12 = layers.Concatenate(axis=-1, name="cat_12")([x_10, x_11, up_21_to_12])
    x_12 = conv_block(cat_12, nb_filter[1], "node_12")
    up_12_to_03 = layers.Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), padding="same", name="up_12_03"
    )(x_12)
    cat_03 = layers.Concatenate(axis=-1, name="cat_03")([x_00, x_01, x_02, up_12_to_03])
    x_03 = conv_block(cat_03, nb_filter[0], "node_03")
    x_40 = conv_block(p3, nb_filter[4], "node_40")
    up_40_to_31 = layers.Conv2DTranspose(
        nb_filter[3], (2, 2), strides=(2, 2), padding="same", name="up_40_31"
    )(x_40)
    cat_31 = layers.Concatenate(axis=-1, name="cat_31")([x_30, up_40_to_31])
    x_31 = conv_block(cat_31, nb_filter[3], "node_31")
    up_31_to_22 = layers.Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), padding="same", name="up_31_22"
    )(x_31)
    cat_22 = layers.Concatenate(axis=-1, name="cat_22")([x_20, x_21, up_31_to_22])
    x_22 = conv_block(cat_22, nb_filter[2], "node_22")
    up_22_to_13 = layers.Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), padding="same", name="up_22_13"
    )(x_22)
    cat_13 = layers.Concatenate(axis=-1, name="cat_13")([x_10, x_11, x_12, up_22_to_13])
    x_13 = conv_block(cat_13, nb_filter[1], "node_13")
    up_13_to_04 = layers.Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), padding="same", name="up_13_04"
    )(x_13)
    cat_04 = layers.Concatenate(axis=-1, name="cat_04")([x_00, x_01, x_02, x_03, up_13_to_04])
    x_04 = conv_block(cat_04, nb_filter[0], "node_04")
    outputs = layers.Conv2D(
        1,
        (1, 1),
        activation="sigmoid",
        padding="same",
        name="segmentation_output"
    )(x_04)
    model = models.Model(inputs=inputs, outputs=outputs, name="Manuscript_UNetPlusPlus")
    return model
