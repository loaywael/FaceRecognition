import tensorflow as tf 
import numpy as np 
import cv2


class MyFaceNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyFaceNet, self).__init__(**kwargs)
        self.conv1 = MyFaceNet.convLayer(32, 3, 1, name="conv1") 
        X = layers.Conv2D(32, 3, padding="same", activation="elu")(Xin)
        X = layers.BatchNormalization()(X)
        X = layers.Dropout(0.2)(X)
        # ============= Stage 2 =============
        X = ConvBlock(X, [64, 64, 256], 3, 1, "2", "1")
        X = IdentityBlock(X, [64, 64, 256], 3, 1, "2", "2")
        X = layers.Dropout(0.2)(X)
        # ============= Stage 3 =============
        X = ConvBlock(X, [128, 128, 512], 3, 2, "3", "3")
        X = IdentityBlock(X, [128, 128, 512], 3, 1, "3", "4")
        X = layers.Dropout(0.2)(X)
        # ============= Stage 4 =============
        X = layers.GlobalAvgPool2D()(X)
        X = layers.Dropout(0.4)(X)


    @staticmethod
    def convLayer(f, k, s=1, p="valid", weightsDecay=None, bias=True, name=""):
        regularizer = None
        if weightsDecay:
            regularizer = l2(weightsDecay)
        conv_layer = layers.Conv2D(
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

    @staticmethod
    def ConvBlock(inputs, filters, k_size, strides, stage, block):
        f1, f2, f3 = filters
        blockName = "STAGE_" + str(stage) + "_BLOCK_" + str(block)
        X = convLayer(inputs, f1, k=1, s=strides, weightsDecay=1e-4, name=blockName+"_conv_a")
        X = layers.BatchNormalization(name=blockName+"_bn_a")(X)
        X = layers.Activation("elu", name=blockName+"_activ_a")(X)

        X = convLayer(X, f2, k=k_size, s=1, p="same", weightsDecay=1e-4, name=blockName+"_conv_b")
        X = layers.BatchNormalization(name=blockName+"_bn_b")(X)
        X = layers.Activation("elu", name=blockName+"_activ_b")(X)

        X = convLayer(X, f3, k=1, s=1, weightsDecay=1e-4, name=blockName+"_conv_c")
        X = layers.BatchNormalization(name=blockName+"_bn_c")(X)

        inputs = convLayer(inputs, f3, k=1, s=strides, weightsDecay=1e-4, name=blockName+"_shortcut_conv", bias=False)
        inputs = layers.BatchNormalization(name=blockName+"shortcut_bn")(inputs)
        X = layers.Add(name=blockName+"_merged_shortcut")([inputs, X])
        X = layers.Activation("elu", name=blockName+"_activ_c")(X)
        return X

    # @staticmethod
    # def IdentityBlock(inputs, filters, k_size, strides, stage, block):
    #     f1, f2, f3 = filters
    #     blockName = "STAGE_" + str(stage) + "_BLOCK_" + str(block)
    #     X = layers.BatchNormalization(name=blockName + "_bn_a1")(inputs)
    #     X = convLayer(X, f1, k=1, s=1, weightsDecay=1e-4, name=blockName + "_conv_a")
    #     X = layers.BatchNormalization(name=blockName + "_bn_a2")(inputs)
    #     X = layers.Activation("elu", name=blockName + "_activ_a")(X)

    #     X = layers.BatchNormalization(name=blockName + "_bn_b1")(X)
    #     X = convLayer(X, f2, k=k_size, s=strides, p="same", weightsDecay=1e-4, name=blockName + "_conv_b")
    #     X = layers.BatchNormalization(name=blockName + "_bn_b2")(X)
    #     X = layers.Activation("elu", name=blockName + "_activ_b")(X)

    #     X = layers.BatchNormalization(name=blockName + "_bn_c1")(X)
    #     X = convLayer(X, f3, k=1, s=1, weightsDecay=1e-4, name=blockName + "_conv_c")
    #     X = layers.BatchNormalization(name=blockName + "_bn_c2")(X)

    #     X = layers.Add(name=blockName+"_merged_shortcut")([inputs, X])
    #     X = layers.Activation("elu", name=blockName+"_activ_c")(X)
    #     return X
