# gen_paper_figs.py


import csv
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = '../../data'


# ======================================================================================================================
# Functional MRI Figures
# ======================================================================================================================

def get_heinsfeld_autoencoder_results():
    _get_training_history('heinsfeld_autoencoder')
    _get_test_history('heinsfeld_autoencoder')


def get_fmri_fcn_results():
    _get_training_history('fmri_fcn')
    _get_test_history('fmri_fcn')


# ======================================================================================================================
# Structural MRI Figures
# ======================================================================================================================

def get_3d_cnn_results():
    _get_training_history('3d_cnn')
    _get_test_history('3d_cnn')


def plot_3d_cnn_results():
    _plot_results('3d_cnn', '3D CNN')


def get_2d_cnn_results():
    _get_training_history('2d_cnn')
    _get_test_history('2d_cnn')


def plot_2d_cnn_results():
    _plot_results('2d_cnn', '2D CNN')


def get_struct_fcn_results():
    _get_training_history('struct_fcn')
    _get_test_history('struct_fcn')


def plot_struct_fcn_results():
    _plot_results('struct_fcn', 'FCN')


def plot_struct_rois():
    atlas = nib.load('{}/oasis_roi_atlas.nii.gz'.format(DATA_DIR)).get_data().astype(np.float32)
    cmap = plt.cm.hsv
    cmap.set_bad(color='gray')
    atlas = _map_roi_atlas('Caltech_0051456', atlas)
    img = atlas[:, :, 64]
    img = np.rot90(img)
    img = np.ma.masked_where(img <= 0, img)
    plt.figure(figsize=(6.75, 9.75))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('roi_thickness-horizontal.pdf')
    plt.clf()
    img = atlas[:, 78, :]
    img = np.rot90(img)
    img = np.ma.masked_where(img <= 0, img)
    plt.figure(figsize=(6.75, 8))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('roi_thickness-coronal.pdf')
    plt.clf()
    img = atlas[54, :, :]
    img = np.flip(np.rot90(img), axis=1)
    img = np.ma.masked_where(img <= 0, img)
    plt.figure(figsize=(9.75, 8))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('roi_thickness-sagittal.pdf')
    plt.clf()


# ======================================================================================================================
# Utilities
# ======================================================================================================================

def _get_training_history(model, ver=None):
    file = '{}_training'.format(model) if ver is None else '{}_training-{}'.format(model, ver)
    fin = open('{}/{}.csv'.format(DATA_DIR, file), 'r')
    folds = {}
    fin.readline()
    trn_acc = []
    val_acc = []
    trn_loss = []
    val_loss = []
    for i, l in enumerate(fin.readlines()):
        f = l.split('],')
        trn_a = [float(j) for j in f[0].split('[')[1].split(']')[0].split('\n')[0].split(', ')]
        val_a = [float(j) for j in f[1].split('[')[1].split(']')[0].split('\n')[0].split(', ')]
        trn_l = [float(j) for j in f[2].split('[')[1].split(']')[0].split('\n')[0].split(', ')]
        val_l = [float(j) for j in f[3].split('[')[1].split(']')[0].split('\n')[0].split(', ')]
        folds[i] = {
            'trn_acc': trn_a,
            'val_acc': val_a,
            'trn_loss': trn_l,
            'val_loss': val_l,
        }
        trn_acc.append(trn_a)
        val_acc.append(val_a)
        trn_loss.append(trn_l)
        val_loss.append(val_l)
    trn_acc_std = np.std(trn_acc, axis=0)
    trn_acc = np.mean(trn_acc, axis=0)
    val_acc_std = np.std(val_acc, axis=0)
    val_acc = np.mean(val_acc, axis=0)
    trn_loss_std = np.std(trn_loss, axis=0)
    trn_loss = np.mean(trn_loss, axis=0)
    val_loss_std = np.std(val_loss, axis=0)
    val_loss = np.mean(val_loss, axis=0)
    print('trn_acc: {:.3f}+/-{:.3f}, val_acc: {:.3f}+/-{:.3f}, trn_loss: {:.3f}+/-{:.3f}, val_loss: {:.3f}+/-{:.3f}'
          .format(trn_acc[-1], trn_acc_std[-1], val_acc[-1], val_acc_std[-1], trn_loss[-1], trn_loss_std[-1],
                  val_loss[-1], val_loss_std[-1]))
    return trn_acc, val_acc, trn_loss, val_loss


def _get_test_history(model, ver=None):
    file = '{}_test'.format(model) if ver is None else '{}_test-{}'.format(model, ver)
    fin = open('{}/{}.csv'.format(DATA_DIR, file), 'r')
    trn_acc = []
    tst_acc = []
    trn_loss = []
    tst_loss = []
    if model != 'heinsfeld_autoencoder':
        for l in csv.DictReader(fin):
            trn_acc.append(float(l['trn_acc']))
            tst_acc.append(float(l['val_acc']))
            trn_loss.append(float(l['trn_loss']))
            tst_loss.append(float(l['val_loss']))
    else:
        for l in fin.readlines():
            if not l.startswith('Epoch'):
                l = l.split(' - ')
                trn_acc.append(float(l[3].split(': ')[1]))
                tst_acc.append(float(l[5].split(': ')[1]))
                trn_loss.append(float(l[2].split(': ')[1]))
                tst_loss.append(float(l[4].split(': ')[1]))
    print('trn_acc: {:.3f}, tst_acc: {:.3f}, trn_loss: {:.3f}, tst_loss: {:.3f}'
          .format(trn_acc[-1], tst_acc[-1], trn_loss[-1], tst_loss[-1]))


def _map_roi_atlas(subject, atlas):
    df = pd.read_csv('{}/roi_thickness/{}_roi_thickness.txt'.format(DATA_DIR, subject), sep="\t", header=0)
    df = df.iloc[:, 2:]
    means = df.as_matrix().T.tolist()
    labels = [int(i.split('_')[-1]) for i in df.keys().tolist()]
    rois = {labels[i]: means[i][0] for i in range(len(labels))}
    for roi, mean in rois.items():
        atlas[atlas == roi] = mean
    return atlas


def _plot_results(model, title):
    trn_acc, val_acc, trn_loss, val_loss = _get_training_history(model)
    num_epochs = len(trn_acc)
    plt.plot(np.arange(1, num_epochs + 1), trn_acc, color='r', alpha=0.75, label='Training accuracy')
    plt.plot(np.arange(1, num_epochs + 1), val_acc, color='b', alpha=0.75, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('{} training and validation accuracies'.format(title))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../../results/{}-accuracy.pdf'.format(model))
    plt.clf()
    plt.plot(np.arange(1, num_epochs + 1), trn_loss, color='r', alpha=0.75, label='Training loss')
    plt.plot(np.arange(1, num_epochs + 1), val_loss, color='b', alpha=0.75, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('training and validation losses'.format(title))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('../../results/{}-loss.pdf'.format(model))
    plt.clf()


if __name__ == '__main__':
    get_heinsfeld_autoencoder_results()
    get_fmri_fcn_results()
    get_3d_cnn_results()
    plot_3d_cnn_results()
    get_2d_cnn_results()
    plot_2d_cnn_results()
    get_struct_fcn_results()
    plot_struct_fcn_results()
    plot_struct_rois()

