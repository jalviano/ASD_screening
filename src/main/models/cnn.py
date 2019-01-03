# cnn.py


import tensorflow as tf


class CNN3D(object):

    def __init__(self, input_shape,
                 num_classes,
                 dropout=0.5,
                 learning_rate=1e-4,
                 epsilon=1e-8,
                 regularization=1e-3,
                 batch_size=72,
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
            tf.layers.Conv3D(filters=16, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=initializer,
                             kernel_regularizer=regularizer, input_shape=self.input_shape),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.MaxPooling3D(pool_size=2, strides=2, padding='same'),
            tf.layers.Conv3D(filters=32, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=initializer,
                             kernel_regularizer=regularizer),
            tf.layers.MaxPooling3D(pool_size=2, strides=2, padding='same'),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Conv3D(filters=32, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=initializer,
                             kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.MaxPooling3D(pool_size=2, strides=2, padding='same'),
            tf.layers.Flatten(),
            tf.layers.Dense(units=1024, kernel_initializer=initializer, kernel_regularizer=regularizer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=1024, kernel_initializer=initializer, kernel_regularizer=regularizer),
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
        model.save('../../results/3d_cnn.h5')
        del model
        tf.keras.backend.clear_session()
        return h


class CNN2D(object):

    def __init__(self, input_shape,
                 num_classes,
                 dropout=0.5,
                 learning_rate=1e-4,
                 epsilon=1e-8,
                 regularization=1e-3,
                 batch_size=72,
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
        layers = [
            tf.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=initializer,
                             input_shape=self.input_shape),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_initializer=initializer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer=initializer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer=initializer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.layers.Flatten(),
            tf.layers.Dense(units=1024, kernel_initializer=initializer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=1024, kernel_initializer=initializer),
            tf.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.layers.Dropout(self.dropout),
            tf.layers.Dense(units=2, activation=tf.nn.softmax),
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
        model.save('../../results/2d_cnn.h5')
        del model
        tf.keras.backend.clear_session()
        return h
