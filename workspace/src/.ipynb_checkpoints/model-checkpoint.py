from keras.layers import Input,Dense,Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

def build_model(filter_1, filter_2, filter_3, batch_size, learning_rate):
    model = Sequential([
        Input((28,28, 1)), 

        Conv2D(filter_1, (3,3), activation="relu", name="layer1"),
        MaxPooling2D((2,2)),

        Conv2D(filter_2, (3,3), activation="relu", name="layer2"),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(filter_3, activation="relu", name="layer3"),
        Dense(10, activation="softmax"),
    ])

    model.compile(optimizer=Adam(learning_rate= learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
