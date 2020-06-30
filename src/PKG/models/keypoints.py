import tensorflow as tf 
import numpy as np 
import cv2


net = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        input_shape=(None, None, 1), 
        filters=32, kernel_size=(5, 5),
        strides=1, padding="same",
    ),
    tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
    tf.keras.layers.Conv2D(
        input_shape=(None, None, 1), 
        filters=32, kernel_size=(5, 5),
        strides=1, padding="same",
    ),
    tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
    tf.keras.layers.Conv2D(
        input_shape=(None, None, 1), 
        filters=32, kernel_size=(5, 5),
        strides=1, padding="same",
    ),
    tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation="softmax")
])


print(net.summary())
batch = np.load("data/batch.npy", allow_pickle=True)
batch1 = np.ones((64, 100, 100, 1))
batch2 = np.ones((64, 48, 48, 1))
batch3 = np.ones((1, 64, 64, 1))
for data in batch:
print("batch1: ", net.predict(batch).shape)
# print("batch2: ", net.predict(batch2).shape)
# print("batch3: ", net.predict(batch3).shape)