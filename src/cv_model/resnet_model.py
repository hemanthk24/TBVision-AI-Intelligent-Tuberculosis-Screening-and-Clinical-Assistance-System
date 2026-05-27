import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

def build_model():
    
    input_shape = (224,224,3)
    
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=input_shape    
    )
    
    inputs = tf.keras.Input(shape=input_shape)
    
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    return model


    

    

