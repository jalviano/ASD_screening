# autoencoder.py


import gc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from utils import save_history


class Autoencoder(object):

    def __init__(self, num_classes,
                 dropout=(0.6, 0.8),
                 learning_rate=(1e-4, 1e-4, 1e-4),
                 momentum=0.1,
                 noise=(0.7, 0.9),
                 batch_size=(72, 72, 72),
                 num_epochs=(10, 10, 10)):
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.noise = noise
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _add_noise(self, x, n_layer):
        if self.noise[n_layer] > 0.0:
            p = (self.noise[n_layer], 1 - self.noise[n_layer])
            x_noisy = np.multiply(x, np.random.choice((0, 1), size=x.shape, p=p))
            return x_noisy
        return x

    def _init_autoencoder(self, X, n_layer, units, input_dim):
        x_noisy = self._add_noise(X, n_layer)
        initializer = 'glorot_uniform'
        optimizer = tf.train.AdamOptimizer(self.learning_rate[n_layer])
        layers = [
            tf.layers.Dense(units, activation=tf.nn.tanh, kernel_initializer=initializer, input_dim=input_dim),
            tf.layers.Dense(input_dim, activation=None, kernel_initializer=initializer),
        ]
        autoencoder = tf.keras.Sequential(layers)
        autoencoder.compile(optimizer=optimizer, loss=tf.keras.metrics.mean_squared_error, metrics=['accuracy'])
        autoencoder.fit(x_noisy, X, epochs=self.num_epochs[n_layer], batch_size=self.batch_size[n_layer],
                        validation_data=(x_noisy, X))
        if n_layer == 0:
            encoder_layer = autoencoder.layers[0]
            input_img = tf.keras.Input(shape=(input_dim,))
            encoder = tf.keras.Model(input_img, encoder_layer(input_img))
            trn_output = encoder.predict(x_noisy)
        else:
            trn_output = autoencoder.predict(x_noisy)
        weights = autoencoder.layers[0].get_weights()
        return trn_output, weights

    def _init_neural_net(self, trn_x, trn_y, val_x, val_y, h1, h2, W1, W2):
        trn_y = tf.keras.utils.to_categorical(trn_y, num_classes=self.num_classes)
        val_y = tf.keras.utils.to_categorical(val_y, num_classes=self.num_classes)
        layers = [
            tf.layers.Dense(h1, activation=tf.nn.tanh, input_dim=trn_x.shape[1]),
            tf.layers.Dropout(self.dropout[0]),
            tf.layers.Dense(h2, activation=tf.nn.tanh),
            tf.layers.Dropout(self.dropout[1]),
            tf.layers.Dense(self.num_classes, activation=tf.nn.softmax),
        ]
        nn = tf.keras.Sequential(layers)
        optimizer = tf.train.MomentumOptimizer(self.learning_rate[-1], self.momentum, use_nesterov=True)
        nn.compile(optimizer=optimizer, loss=tf.keras.metrics.mean_squared_error, metrics=['accuracy'])
        nn.layers[0].set_weights(W1)
        nn.layers[2].set_weights(W2)
        history = nn.fit(trn_x, trn_y, epochs=self.num_epochs[-1], batch_size=self.batch_size[-1],
                         validation_data=(val_x, val_y))
        nn.save('../../results/heinsfeld_autoencoder.h5')
        return history

    def train(self, X, y, h1=1000, h2=600):
        trn_acc = []
        val_acc = []
        trn_loss = []
        val_loss = []
        ae1, W1 = self._init_autoencoder(X, 0, h1, X.shape[1])
        ae2, W2 = self._init_autoencoder(ae1, 1, h2, h1)
        for trn_i, val_i in StratifiedKFold(n_splits=10, shuffle=True).split(X, y):
            trn_x, trn_y = X[trn_i], y[trn_i]
            val_x, val_y = X[val_i], y[val_i]
            history = self._init_neural_net(trn_x, trn_y, val_x, val_y, h1, h2, W1, W2)
            trn_acc.append(history.history['acc'])
            val_acc.append(history.history['val_acc'])
            trn_loss.append(history.history['loss'])
            val_loss.append(history.history['val_loss'])
            del history
            gc.collect()
        save_history(trn_acc, val_acc, trn_loss, val_loss, 'heinsfeld_autoencoder_training')

    def predict(self, trn_x, trn_y, tst_x, tst_y, h1=1000, h2=600):
        ae1, W1 = self._init_autoencoder(trn_x, 0, h1, trn_x.shape[1])
        ae2, W2 = self._init_autoencoder(ae1, 1, h2, h1)
        history = self._init_neural_net(trn_x, trn_y, tst_x, tst_y, h1, h2, W1, W2)
        save_history(history.history['acc'], history.history['val_acc'], history.history['loss'],
                     history.history['val_loss'], 'heinsfeld_autoencoder_test')
