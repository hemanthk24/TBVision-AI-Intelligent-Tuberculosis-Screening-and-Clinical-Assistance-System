import tensorflow as tf
from tensorflow.keras import layers
from src.config.setting import MODEL_INPUT_SHAPE, IMG_SIZE
from tensorflow.keras.applications import ResNet50
from src.config.setting import TB_MODEL_WEIGHTS_PATH

_model = None

def build_model():
    
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=MODEL_INPUT_SHAPE   
    )
    
    inputs = tf.keras.Input(shape=MODEL_INPUT_SHAPE)
    
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def cv_model():
    global _model
    
    if _model is None:
        _model = build_model()
        _model.load_weights(TB_MODEL_WEIGHTS_PATH)
    return _model

    

