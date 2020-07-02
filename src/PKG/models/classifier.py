from PKG.models import ResnetV2
import tensorflow as tf



class Classifier(tf.keras.models.Model):
    def __init__(self, nclasses, weights=None):
        super(Classifier, self).__init__()
        self.nn = ResnetV2()
        self.classifier = tf.keras.layers.Dense(nclasses)
        self.activation = tf.keras.layers.Activation("softmax")
        self.dropout = tf.keras.layers.Dropout(0.3)
    
    def call(self, X, training=False):
        Y = self.nn(X)
        Y = self.classifier(Y)
        Y = self.activation(Y)
        Y = self.dropout(Y, training=training)
        return Y