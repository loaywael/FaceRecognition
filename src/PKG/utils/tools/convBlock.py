import tensorflow as tf 
import numpy as np


class ConvBlock(tf.keras.models.Model):
    def __init__(self, filters, k_size, strides, stage, block, block_type="identity", **kwargs):
        f1, f2, f3 = filters
        self.block_type = block_type
        super(ConvBlock, self).__init__(**kwargs)
        blockName = "STAGE_" + str(stage) + "_"+self.block_type.upper()+"_BLOCK_" + str(block)
        # ============= sub-stage 1 =============
        self.conv1 = ConvBlock.convLayer(f1, k=1, s=strides, weightsDecay=1e-4, name=blockName+"_conv_a")
        self.bn1 = tf.keras.layers.BatchNormalization(name=blockName+"_bn_a")
        self.activation1 = tf.keras.layers.Activation("elu", name=blockName+"_activ_a")
        # ============= sub-stage 2 =============
        self.conv2 = ConvBlock.convLayer(f2, k=k_size, s=1, p="same", weightsDecay=1e-4, name=blockName+"_conv_b")
        self.bn2 = tf.keras.layers.BatchNormalization(name=blockName+"_bn_b")
        self.activation2 = tf.keras.layers.Activation("elu", name=blockName+"_activ_b")
        # ============= sub-stage 3 =============
        self.conv3 = ConvBlock.convLayer(f3, k=1, s=1, weightsDecay=1e-4, name=blockName+"_conv_c")
        self.bn3 = tf.keras.layers.BatchNormalization(name=blockName+"_bn_c")
        # ============= sub-stage 4 =============
        if self.block_type == "conv":
            self.convShortcut = ConvBlock.convLayer(
                f3, k=1, s=strides, weightsDecay=1e-4, 
                name=blockName+"_shortcut_conv", bias=False
            )
            self.bnShortcut = tf.keras.layers.BatchNormalization(name=blockName+"_shortcut_bn")
        self.add = tf.keras.layers.Add(name=blockName+"_merged_shortcut")
        self.activation4 = tf.keras.layers.Activation("elu", name=blockName+"_activ_c")
    

    @staticmethod
    def convLayer(f, k, s=1, p="valid", weightsDecay=None, bias=True, name=""):
        regularizer = None
        if weightsDecay:
            regularizer = tf.keras.regularizers.l2(weightsDecay)
        conv_layer = tf.keras.layers.Conv2D(
            filters=f, 
            kernel_size=k,
            strides=s,
            padding=p,
            # kernel_initializer="he_uniform",
            kernel_regularizer=regularizer,
            use_bias=bias,
            name=name
        )
        return conv_layer

    def call(self, X, training=False):
        Xin = X
        X = self.conv1(X)
        X = self.bn1(X, training=training)
        X = self.activation1(X)
        # ---------------------
        X = self.conv2(X)
        X = self.bn2(X, training=training)
        X = self.activation2(X)
        # ---------------------
        X = self.conv3(X)
        X = self.bn3(X, training=training)
        # ---------------------
        if self.block_type == "conv":
            Xin = self.convShortcut(Xin)
            Xin = self.bnShortcut(Xin, training=training)
        Y = self.add([X, Xin])
        Y = self.activation4(Y)
        return Y


if __name__ == "__main__":
    X = tf.random.uniform((64, 96, 96, 1))
    conv_block = ConvBlock([32, 32, 256], 3, 1, "a", "1", block_type="conv")
    Y = conv_block(X)
    print(conv_block.summary(line_length=125))
    print(Y.shape)

