from PKG.models import ResnetV2
import tensorflow as tf



class Regressor(tf.keras.models.Model):
    def __init__(self, nmarks, weights=None):
        super(Regressor, self).__init__()
        self.nn = ResnetV2()
        self.classifier = tf.keras.layers.Dense(nmarks)
        self.activation = tf.keras.layers.Activation("elu")
        self.dropout = tf.keras.layers.Dropout(0.3)
    
    def call(self, X, training=False):
        Y = self.nn(X)
        Y = self.classifier(Y)
        Y = self.activation(Y)
        Y = self.dropout(Y, training=training)
        return Y


if __name__ == "__main__":
    net = Regressor(68)
    batch = tf.random.uniform((64, 96, 96, 1))
    print(net(batch).shape)