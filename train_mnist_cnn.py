# save this as train_mnist_cnn.py and run once
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28,28,1), input_shape=(28,28)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2,2)),
                        tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(100, activation='relu'),
                                tf.keras.layers.Dense(10, activation='softmax')
                                ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
model.save("mnist_cnn.h5")