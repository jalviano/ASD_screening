# gen_paper_data.py


from sklearn.model_selection import StratifiedKFold

from utils.utils import *
from models.autoencoder import Autoencoder
from models.cnn import CNN3D, CNN2D
from models.fcn import FmriFCN, StructFCN


# ======================================================================================================================
# Functional MRI Analysis
# ======================================================================================================================

def train_heinsfeld_autoencoder():
    X, y = read_data('rois_cc200')
    clf = Autoencoder(num_classes=2,
                      dropout=(0.6, 0.8),
                      learning_rate=(0.0001, 0.0001, 0.0005),
                      momentum=0.9,
                      noise=(0.2, 0.3),
                      batch_size=(100, 10, 10),
                      num_epochs=(700, 2000, 100))
    clf.train(X, y)


def predict_heinsfeld_autoencoder():
    trn_x, trn_y = read_data('rois_cc200')
    tst_x, tst_y = read_data('rois_cc200', training=False)
    clf = Autoencoder(num_classes=2,
                      dropout=(0.6, 0.8),
                      learning_rate=(0.0001, 0.0001, 0.0005),
                      momentum=0.9,
                      noise=(0.2, 0.3),
                      batch_size=(100, 10, 10),
                      num_epochs=(700, 2000, 100))
    clf.predict(trn_x, trn_y, tst_x, tst_y)


def train_fmri_fcn():
    X, y = read_data('rois_cc200')
    trn_acc = []
    val_acc = []
    trn_loss = []
    val_loss = []
    for trn_i, val_i in StratifiedKFold(n_splits=10, shuffle=True).split(X, y):
        trn_x, trn_y = X[trn_i], y[trn_i]
        val_x, val_y = X[val_i], y[val_i]
        clf = FmriFCN(input_shape=trn_x.shape[1:],
                      num_classes=2,
                      dropout=0.25,
                      learning_rate=5e-6,
                      regularization=0.0,
                      batch_size=32,
                      num_epochs=25)
        history = clf.train(trn_x, trn_y, val_x, val_y)
        trn_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        trn_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        del history
        gc.collect()
    save_history(trn_acc, val_acc, trn_loss, val_loss, 'fmri_fcn_training-reg')


def predict_fmri_fcn():
    trn_x, trn_y = read_data('rois_cc200')
    tst_x, tst_y = read_data('rois_cc200', training=False)
    clf = FmriFCN(input_shape=trn_x.shape[1:],
                  num_classes=2,
                  dropout=0.25,
                  learning_rate=5e-6,
                  regularization=0.0,
                  batch_size=32,
                  num_epochs=25)
    history = clf.train(trn_x, trn_y, tst_x, tst_y)
    save_history(history.history['acc'], history.history['val_acc'], history.history['loss'],
                 history.history['val_loss'], 'fmri_fcn_test')


# ======================================================================================================================
# Structural MRI Analysis
# ======================================================================================================================

def train_3d_cnn():
    X, y = read_data('anat_thickness')
    trn_acc = []
    val_acc = []
    trn_loss = []
    val_loss = []
    for trn_i, val_i in StratifiedKFold(n_splits=5, shuffle=True).split(X, y):
        trn_x, trn_y = X[trn_i], y[trn_i]
        val_x, val_y = X[val_i], y[val_i]
        clf = CNN3D(input_shape=trn_x.shape[1:],
                    num_classes=2,
                    dropout=0.5,
                    learning_rate=0.00003,
                    regularization=0.0001,
                    batch_size=32,
                    num_epochs=35)
        history = clf.train(trn_x, trn_y, val_x, val_y)
        trn_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        trn_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        del history
        gc.collect()
    save_history(trn_acc, val_acc, trn_loss, val_loss, '3d_cnn_training')


def predict_3d_cnn():
    trn_x, trn_y = read_data('anat_thickness')
    tst_x, tst_y = read_data('anat_thickness', training=False)
    clf = CNN3D(input_shape=trn_x.shape[1:],
                num_classes=2,
                dropout=0.5,
                learning_rate=0.00003,
                regularization=0.0001,
                batch_size=32,
                num_epochs=35)
    history = clf.train(trn_x, trn_y, tst_x, tst_y)
    save_history(history.history['acc'], history.history['val_acc'], history.history['loss'],
                 history.history['val_loss'], '3d_cnn_test')


def train_2d_cnn():
    X, y = read_data('anat_thickness')
    N, H, W, C, _ = X.shape
    X = X.reshape(N, H, W, C)
    trn_acc = []
    val_acc = []
    trn_loss = []
    val_loss = []
    for trn_i, val_i in StratifiedKFold(n_splits=5, shuffle=True).split(X, y):
        trn_x, trn_y = X[trn_i], y[trn_i]
        val_x, val_y = X[val_i], y[val_i]
        clf = CNN2D(input_shape=trn_x.shape[1:],
                    num_classes=2,
                    dropout=0.75,
                    learning_rate=0.0001,
                    regularization=0.0,
                    batch_size=32,
                    num_epochs=15)
        history = clf.train(trn_x, trn_y, val_x, val_y)
        trn_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        trn_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        del history
        gc.collect()
    save_history(trn_acc, val_acc, trn_loss, val_loss, '2d_cnn_training')


def predict_2d_cnn():
    trn_x, trn_y = read_data('anat_thickness')
    tst_x, tst_y = read_data('anat_thickness', training=False)
    _, H, W, C, _ = trn_x.shape
    trn_x = trn_x.reshape(trn_x.shape[0], H, W, C)
    tst_x = tst_x.reshape(tst_x.shape[0], H, W, C)
    clf = CNN2D(input_shape=trn_x.shape[1:],
                num_classes=2,
                dropout=0.75,
                learning_rate=0.0001,
                regularization=0.0,
                batch_size=32,
                num_epochs=15)
    history = clf.train(trn_x, trn_y, tst_x, tst_y)
    save_history(history.history['acc'], history.history['val_acc'], history.history['loss'],
                 history.history['val_loss'], '3d_cnn_test')


def train_struct_fcn():
    X, y = read_data('roi_thickness')
    trn_acc = []
    val_acc = []
    trn_loss = []
    val_loss = []
    for trn_i, val_i in StratifiedKFold(n_splits=5, shuffle=True).split(X, y):
        trn_x, trn_y = X[trn_i], y[trn_i]
        val_x, val_y = X[val_i], y[val_i]
        clf = StructFCN(input_shape=trn_x.shape[1:],
                        num_classes=2,
                        dropout=1.0,
                        learning_rate=0.0007,
                        regularization=0.0,
                        batch_size=32,
                        num_epochs=150)
        history = clf.train(trn_x, trn_y, val_x, val_y)
        trn_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        trn_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        del history
        gc.collect()
    save_history(trn_acc, val_acc, trn_loss, val_loss, 'struct_fcn_training')


def predict_struct_fcn():
    trn_x, trn_y = read_data('roi_thickness')
    tst_x, tst_y = read_data('roi_thickness', training=False)
    clf = StructFCN(input_shape=trn_x.shape[1:],
                    num_classes=2,
                    dropout=1.0,
                    learning_rate=0.0007,
                    regularization=0.0,
                    batch_size=32,
                    num_epochs=150)
    history = clf.train(trn_x, trn_y, tst_x, tst_y)
    save_history(history.history['acc'], history.history['val_acc'], history.history['loss'],
                 history.history['val_loss'], 'struct_fcn_test')


# ======================================================================================================================
# Pipeline
# ======================================================================================================================

if __name__ == '__main__':
    split = True
    autoencoder = True
    fmri_fcn = True
    cnn_2d = True
    cnn_3d = True
    struct_fcn = True
    # Data loading
    if split:
        print('Generating training and test split...')
        get_train_val_test()
        print('Calculating dataset statistics...')
        get_dataset_stats()
    # Functional MRI analysis
    if autoencoder:
        print('Training autoencoder...')
        train_heinsfeld_autoencoder()
        print('Predicting autoencoder...')
        predict_heinsfeld_autoencoder()
    if fmri_fcn:
        print('Training fMRI FCN...')
        train_fmri_fcn()
        print('Predicting fMRI FCN...')
        predict_fmri_fcn()
    # Structural MRI Analysis
    if cnn_2d:
        print('Training 2D CNN...')
        train_2d_cnn()
        print('Predicting 2D CNN...')
        predict_2d_cnn()
    if cnn_3d:
        print('Training 3D CNN...')
        train_3d_cnn()
        print('Predicting 3D CNN...')
        predict_3d_cnn()
    if struct_fcn:
        print('Training structural FCN...')
        train_struct_fcn()
        print('Predicting structural FCN...')
        predict_struct_fcn()
