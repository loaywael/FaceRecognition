from PKG.utils import ConvBlock
import tensorflow as tf 
import numpy as np 
import cv2


class MyFaceNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyFaceNet, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 3, padding="same", activation="elu")
        self.bn1 = layers.BatchNormalization()
        # X = layers.Conv2D(32, 3, padding="same", activation="elu")
        # X = layers.BatchNormalization()
        # X = layers.Dropout(0.2)
        # ============= Stage 2 =============
        self.block1 = ConvBlock([32, 32, 256], 3, 1, "a", "1", block_type="conv")
        # X = ConvBlock(X, [64, 64, 256], 3, 1, "2", "2")
        # X = layers.Dropout(0.2)
        # # ============= Stage 3 =============
        # X = ConvBlock(X, [128, 128, 512], 3, 2, "3", "3", block_type="conv")
        # X = ConvBlock(X, [128, 128, 512], 3, 1, "3", "4")
        # X = layers.Dropout(0.2)
        # # ============= Stage 4 =============
        # X = layers.GlobalAvgPool2D()
        # X = layers.Dropout(0.4)


    def call(self, X):
        return self.block1(X)

    