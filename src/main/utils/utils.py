# utils.py


import csv
import gc
import os
import scipy
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


DATA_DIR = '../../data'


def get_train_val_test():
    n_test = 100
    trn_data_f = open('{}/train_data.txt'.format(DATA_DIR), 'w+')
    trn_labels_f = open('{}/train_labels.txt'.format(DATA_DIR), 'w+')
    tst_data_f = open('{}/test_data.txt'.format(DATA_DIR), 'w+')
    tst_labels_f = open('{}/test_labels.txt'.format(DATA_DIR), 'w+')
    data = []
    labels = []
    subjects = [subject.split('_func')[0] for subject in os.listdir('{}/func_mean'.format(DATA_DIR))]
    for line in csv.DictReader(open('{}/phenotypic.csv'.format(DATA_DIR), 'r')):
        subject = line['FILE_ID']
        dx_group = 1 - (int(line['DX_GROUP']) - 1)
        if subject in subjects:
            data.append(subject)
            labels.append(dx_group)
    data = np.array(data)
    labels = np.array(labels)
    for trn_i, val_i in StratifiedShuffleSplit(n_splits=1, test_size=n_test).split(data, labels):
        trn_x, trn_y = data[trn_i].tolist(), labels[trn_i].tolist()
        tst_x, tst_y = data[val_i].tolist(), labels[val_i].tolist()
        for i in range(len(trn_x)):
            trn_data_f.write('{}\n'.format(trn_x[i]))
            trn_labels_f.write('{}\n'.format(trn_y[i]))
        for i in range(len(tst_x)):
            tst_data_f.write('{}\n'.format(tst_x[i]))
            tst_labels_f.write('{}\n'.format(tst_y[i]))


def read_data(der, training=True):
    data = 'train_data' if training else 'test_data'
    labels = 'train_labels' if training else 'test_labels'
    subs = np.loadtxt('{}/{}.txt'.format(DATA_DIR, data), delimiter=',', dtype=str)
    if der == 'rois_cc200':
        X = np.array([load_roi(sub) for sub in subs])
    elif der == 'roi_thickness':
        X = np.array([load_roi_thickness(sub) for sub in subs])
    elif der == 'anat_thickness':
        try:
            X = np.load('{}/{}_anat_thickness.npy'.format(DATA_DIR, data))
            y = np.load('{}/{}_anat_thickness.npy'.format(DATA_DIR, labels))
        except IOError:
            save_npys(training)
            load_npys(training)
            X = np.load('{}/{}_anat_thickness.npy'.format(DATA_DIR, data))
            y = np.load('{}/{}_anat_thickness.npy'.format(DATA_DIR, labels))
        return X, y
    else:
        X = np.array([nib.load('{}/{}/{}_{}.nii.gz'.format(DATA_DIR, der, sub, der)).get_data() for sub in subs])
    y = np.loadtxt('{}/{}.txt'.format(DATA_DIR, labels), delimiter=',', dtype=int)
    if 'roi' not in der:
        X = reshape_data(X)
        X = scale(X)
    return X, y


def scale(x):
    x = np.clip(x, 0, 255)
    mn = x.min(axis=(0, 1, 2, 3), keepdims=True)
    mx = x.max(axis=(0, 1, 2, 3), keepdims=True)
    x = (x - mn) / (mx - mn)
    return x


def reshape_data(x):
    N, D, H, W = x.shape
    x = x.reshape(N, D, H, W, 1)
    return x


def load_roi_thickness(subject):
    df = pd.read_csv('{}/roi_thickness/{}_roi_thickness.txt'.format(DATA_DIR, subject), sep="\t", header=0)
    df = df.iloc[:, 2:]
    means = df.as_matrix().T.tolist()
    return means


def map_roi_atlas(subject, atlas):
    df = pd.read_csv('{}/roi_thickness/{}_roi_thickness.txt'.format(DATA_DIR, subject), sep="\t", header=0)
    df = df.iloc[:, 2:]
    means = df.as_matrix().T.tolist()
    labels = [int(i.split('_')[-1]) for i in df.keys().tolist()]
    rois = {labels[i]: means[i][0] for i in range(len(labels))}
    for roi, mean in rois.items():
        atlas[atlas == roi] = mean
    return atlas


def load_roi(subject):
    df = pd.read_csv('{}/rois_cc200/{}_rois_cc200.1D'.format(DATA_DIR, subject), sep='\t', header=0)
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    rois = ['#' + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]
    functional = np.nan_to_num(df[rois].as_matrix().T).tolist()
    functional = preprocessing.scale(functional, axis=1)
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(functional))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = np.ma.masked_where(mask == 1, mask)
        functional = np.ma.masked_where(m, corr).compressed()
    functional = functional.astype(np.float32)
    return functional


def save_npys(training=True, batch=30):
    data = 'train_data' if training else 'test_data'
    labels = 'train_labels' if training else 'test_labels'
    subs = np.loadtxt('{}/{}.txt'.format(DATA_DIR, data), delimiter=',', dtype=str)
    y = np.loadtxt('{}/{}.txt'.format(DATA_DIR, labels), delimiter=',', dtype=int)
    X = []
    for i, sub in enumerate(subs):
        j = i + 1
        if j % 10 == 0:
            print('{}/{}'.format(j, len(subs)))
        full = nib.load('{}/anat_thickness/{}_anat_thickness.nii.gz'.format(DATA_DIR, sub)).get_data()
        small = scipy.ndimage.interpolation.zoom(full, 0.25)
        X.append(small)
        if j % batch == 0 or j == len(subs):
            name = int(j / batch) + 1 if j == len(subs) else int(j / batch)
            print('-------', j, name)
            X = np.array(X)
            X = reshape_data(X)
            np.save('{}/npys/{}-{}.npy'.format(DATA_DIR, data, name), X)
            X = []
            gc.collect()
    np.save('{}/{}_anat_thickness.npy'.format(DATA_DIR, labels), y)


def load_npys(training=True, batch=30):
    size = 784 if training else 100
    data = 'train_data' if training else 'test_data'
    X = None
    for i in range(int(np.floor(size / batch)) + 1):
        j = i + 1
        file = '{}-{}.npy'.format(data, j)
        if X is not None:
            f = np.load('{}/npys/{}'.format(DATA_DIR, file))
            print(j, f.shape)
            X = np.concatenate((X, f))
        else:
            X = np.load('{}/npys/{}'.format(DATA_DIR, file))
            print(j, X.shape)
    print(X.shape)
    X = scale(X)
    np.save('{}/{}_anat_thickness.npy'.format(DATA_DIR, data), X)


def save_history(trn_acc, val_acc, trn_loss, val_loss, file):
    with open('{}/{}.csv'.format(DATA_DIR, file), 'w+') as out:
        out.write('trn_acc,val_acc,trn_loss,val_loss\n')
        for i in range(len(trn_acc)):
            out.write('{},{},{},{}\n'.format(trn_acc[i], val_acc[i], trn_loss[i], val_loss[i]))
    out.close()


def get_dataset_stats():
    trn_ages = []
    trn_male = 0
    trn_female = 0
    trn_asd = 0
    trn_tdc = 0
    tst_ages = []
    tst_male = 0
    tst_female = 0
    tst_asd = 0
    tst_tdc = 0
    all_ages = []
    all_male = 0
    all_female = 0
    all_asd = 0
    all_tdc = 0
    trn_subjects = [sub.split('\n')[0] for sub in open('{}/train_data.txt'.format(DATA_DIR), 'r').readlines()]
    tst_subjects = [sub.split('\n')[0] for sub in open('{}/test_data.txt'.format(DATA_DIR), 'r').readlines()]
    for line in csv.DictReader(open('{}/phenotypic.csv'.format(DATA_DIR), 'r')):
        if line['FILE_ID'] in trn_subjects:
            trn_ages.append(float(line['AGE_AT_SCAN']))
            all_ages.append(float(line['AGE_AT_SCAN']))
            if int(line['SEX']) == 1:
                trn_male += 1
                all_male += 1
            else:
                trn_female += 1
                all_female += 1
        elif line['FILE_ID'] in tst_subjects:
            tst_ages.append(float(line['AGE_AT_SCAN']))
            all_ages.append(float(line['AGE_AT_SCAN']))
            if int(line['SEX']) == 1:
                tst_male += 1
                all_male += 1
            else:
                tst_female += 1
                all_female += 1
    for label in open('{}/train_labels.txt'.format(DATA_DIR), 'r').readlines():
        if int(label.split('\n')[0]) == 0:
            trn_tdc += 1
            all_tdc += 1
        else:
            trn_asd += 1
            all_asd += 1
    for label in open('{}/test_labels.txt'.format(DATA_DIR), 'r').readlines():
        if int(label.split('\n')[0]) == 0:
            tst_tdc += 1
            all_tdc += 1
        else:
            tst_asd += 1
            all_asd += 1
    print('Dataset: training, Samples: {}, ASD: {}, TDC: {}, Age Avg (SD): {} ({}), Age Range: {}-{}, Male: {}, '
          'Female: {}'.format(trn_asd + trn_tdc, trn_asd, trn_tdc, np.mean(trn_ages), np.std(trn_ages), min(trn_ages),
                              max(trn_ages), trn_male, trn_female))
    print('Dataset: test, Samples: {}, ASD: {}, TDC: {}, Age Avg (SD): {} ({}), Age Range: {}-{}, Male: {}, '
          'Female: {}'.format(tst_asd + tst_tdc, tst_asd, tst_tdc, np.mean(tst_ages), np.std(tst_ages), min(tst_ages),
                              max(tst_ages), tst_male, tst_female))
    print('Dataset: all, Samples: {}, ASD: {}, TDC: {}, Age Avg (SD): {} ({}), Age Range: {}-{}, Male: {}, '
          'Female: {}'.format(all_asd + all_tdc, all_asd, all_tdc, np.mean(all_ages), np.std(all_ages), min(all_ages),
                              max(all_ages), all_male, all_female))
