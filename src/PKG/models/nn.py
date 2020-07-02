from PKG.utils import ResnetBlock
import tensorflow as tf 
import numpy as np 
import cv2


class ResnetV2(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyFaceNet, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation("elu")
        # ============= Stage 2 =============
        self.block1 = ConvBlock([32, 32, 256], 3, 1, "a", "1", block_type="conv")
        self.block2 = ConvBlock(X, [64, 64, 256], 3, 1, "2", "2")
        self.dropout1 = layers.Dropout(0.2)
        # ============= Stage 3 =============
        self.block3 = ConvBlock(X, [128, 128, 512], 3, 2, "3", "3", block_type="conv")
        self.block4 = ConvBlock(X, [128, 128, 512], 3, 1, "3", "4")
        self.dropout2 = layers.Dropout(0.2)
        # # ============= Stage 4 =============
        self.gap = layers.GlobalAvgPool2D()
        self.dropout3 = layers.Dropout(0.4)
        

    def call(self, X, training=False):
        Y = self.activation1(self.bn1(self.conv1(X)))
        Y = self.block1(Y)
        Y = self.block2(Y)
        Y = self.dropout1(Y, training=training)
        Y = self.block3(Y)
        Y = self.block4(Y)
        Y = self.dropout2(Y, training=training)
        Y = self.gap(Y)
        Y = self.dropout3(Y, training=training)

    