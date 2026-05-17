import math, tensorflow as tf

def build_model(ind, shape, num_classes, min_feature_size=4):
    #ind[] = [lr, batch, dropout, kernel, conv_filters, dense_units]
    inputs = tf.keras.Input(shape=shape)
    
    filters = ind[4].copy()
    
    x = tf.keras.layers.Conv2D(filters.pop(0), ind[3], activation="relu", padding="same")(inputs)

    distribution = get_beautiful_distribution(len(filters), shape[0])
    
    for block_size in distribution:
        for _ in range(block_size):
            x = tf.keras.layers.Conv2D(filters.pop(0), ind[3], activation="relu", padding="same")(x)
            
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for dense_size in ind[5]:
        x = tf.keras.layers.Dense(dense_size, activation="relu")(x)
    
    x = tf.keras.layers.Dropout(ind[2])(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ind[0]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_beautiful_distribution(total_layers, input_shape, min_feature_size=4):

    if total_layers==0:
        return [0]

    # Calculate max pooling operations safely
    max_pools = int(math.log2(input_shape / min_feature_size))
    
    # Ensure at least 1 block, and don't exceed max_pools
    num_blocks = min(max_pools, max(1, int(math.log2(total_layers))))
    
    base_layers = total_layers // num_blocks
    remainder = total_layers % num_blocks
    distribution = [base_layers] * num_blocks
    
    # Distribute remainder to the later blocks
    for i in range(remainder):
        distribution[-(i + 1)] += 1
        
    return distribution