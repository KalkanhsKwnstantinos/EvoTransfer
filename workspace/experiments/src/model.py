import tensorflow as tf

kernel_size=3
dense_units=128
dropout_rate=0.5

def build_model(filters, learning_rate):

    inputs = tf.keras.Input(shape=(28, 28, 1))
    
    x = tf.keras.layers.Conv2D(filters[0], kernel_size, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    filters.pop(0)
    for i in filters:
        x = tf.keras.layers.Conv2D(i, kernel_size, activation="relu", padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model