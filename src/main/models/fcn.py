# fcn.py


import tensorflow as tf


class FmriFCN(object):

    def __init__(self, input_shape,
                 num_classes,
                 dropout=0.5,
                 learning_rate=1e-4,
                 epsilon=1e-8,
                 regularization=1e-3,
                 batch_size=64,
                 num_epochs=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.regularization = regularization
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _init_model(self):
        initializer = 'glorot_uniform'
        regularizer = tf.keras.regularizers.l2(self.regularization)
        layers = [
            tf.layers.Flatten(input_shape=self.input_shape),
            tf.layers.Dense(units=256, kernel_initializer=initializer, kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('tanh'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=128, kernel_initializer=initializer, kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('tanh'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=64, kernel_initializer=initializer, kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('tanh'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=self.num_classes, activation=tf.nn.softmax),
        ]
        model = tf.keras.Sequential(layers)
        return model

    def _init_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon)
        return optimizer

    def train(self, trn_x, trn_y, val_x, val_y):
        trn_y = tf.keras.utils.to_categorical(trn_y, num_classes=self.num_classes)
        val_y = tf.keras.utils.to_categorical(val_y, num_classes=self.num_classes)
        model = self._init_model()
        optimizer = self._init_optimizer()
        model.compile(optimizer=optimizer, loss=tf.losses.mean_squared_error, metrics=['accuracy'])
        h = model.fit(trn_x, trn_y, epochs=self.num_epochs, batch_size=self.batch_size, validation_data=(val_x, val_y))
        model.save('../../results/fmri_fcn.h5')
        del model
        tf.keras.backend.clear_session()
        return h


class StructFCN(object):

    def __init__(self, input_shape,
                 num_classes,
                 dropout=0.5,
                 learning_rate=1e-4,
                 epsilon=1e-8,
                 regularization=1e-3,
                 batch_size=64,
                 num_epochs=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.regularization = regularization
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def _init_model(self):
        initializer = tf.keras.initializers.he_normal(seed=1)
        regularizer = tf.keras.regularizers.l2(self.regularization)
        layers = [
            tf.layers.Flatten(input_shape=self.input_shape),
            tf.layers.Dense(units=64, kernel_initializer=initializer, kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=64, kernel_initializer=initializer, kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=self.num_classes, activation=tf.nn.softmax),
        ]
        model = tf.keras.Sequential(layers)
        return model

    def _init_optimizer(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon)
        return optimizer

    def train(self, trn_x, trn_y, val_x, val_y):
        trn_y = tf.keras.utils.to_categorical(trn_y, num_classes=self.num_classes)
        val_y = tf.keras.utils.to_categorical(val_y, num_classes=self.num_classes)
        model = self._init_model()
        optimizer = self._init_optimizer()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        h = model.fit(trn_x, trn_y, epochs=self.num_epochs, batch_size=self.batch_size, validation_data=(val_x, val_y))
        model.save('../../results/struct_fcn.h5')
        del model
        tf.keras.backend.clear_session()
        return h
