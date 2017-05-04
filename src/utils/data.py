import os
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils

def get_filepath(dataset):
    dirpath = os.path.join(dataset['io']['path'], dataset['io']['directory'])
    train_filename = dataset['io']['train filename'] + dataset['io']['filetype']
    test_filename = dataset['io']['test filename'] + dataset['io']['filetype']
    train_filepath = os.path.join(dirpath, train_filename)
    test_filepath = os.path.join(dirpath, test_filename)
    return(train_filepath, test_filepath)

def format_data(data, shape):
    (h, w, c) = shape
    X = data['x']
    X = X.reshape(X.shape[0], c, w, h)
    X = np.transpose(X, (0, 2, 3, 1)) ## ?
    data['x'] = X.astype('float32') / 255
    return None

def split_data(dataset, data):
    (w,h,c) = dataset['image']['shape']
    n_classes = dataset['num classes']

    n_valid = dataset['num train'] // dataset['validation split']
    n_valid = int(n_valid)
    n_train = dataset['num train'] - n_valid
    cutoff = n_valid // n_classes

    dataset['num train'] = n_train
    dataset['num valid'] = n_valid

    train = {}
    train['x'] = np.zeros((n_train, h, w, c))
    train['y'] = np.zeros((n_train, 1))

    valid = {}
    valid['x'] = np.zeros((n_valid, h, w, c))
    valid['y'] = np.zeros((n_valid, 1))

    label = 0
    val_idx = 0
    remove_list = []
    for idx in range(n_train + n_valid):
        if val_idx >= cutoff:
            label = val_idx // cutoff
        if label >= n_classes:
            break
        if data['y'][idx] == label:
            valid['x'][val_idx] = data['x'][idx]
            valid['y'][val_idx] = data['y'][idx]
            remove_list.append(idx)
            val_idx += 1
    print(len(remove_list))

    train['x'] = np.delete(data['x'], remove_list, 0)
    train['y'] = np.delete(data['y'], remove_list, 0)

    train['y'] = np_utils.to_categorical(train['y'])
    valid['y'] = np_utils.to_categorical(valid['y'])

    print(train['x'].shape)
    print(train['y'].shape)

    return (train, valid)

def load_data(dataset):
    """Load cifar-3 dataset,

    Args:
        dataset: dictionary defining
            - `io`: a dictionary defining:
                - `path`: the relative path to the data directory w.r.t. abs path of `utils.py`
                - `train filename`: filename of train dataset
                - `test filename`: filename of test dataset
                - `directory`: the directory the data is stored in
                - `filetype`: the file format of both the training and testing datasets
            - `image`: a dictionary defining:
                - `shape`: (height, width, channels)
                - `unrolled`: height*width*channels
            - `num classes`: number of classifications
            - `num train`: number of training examples (later modified by validation split)
            - 'num valid': number of validation examples (originally 0)
            - `num test`: number of test examples (immutable)
            - `validation split`: integer; `num valid` = `num train` // `validation split`

    Returns:   tuple of train, valid, and test data
        train: dictionary of training data and labels
            - `x`: shape = (num_train, unrolled)
            - `y`: shape = (num_train, unrolled)
        valid: dictionary of validation data and labels
            - `x`: shape = (num_valid, unrolled)
            - `y`: shape = (num_valid, unrolled)
        test: dictionary of data and labels
            - `x`: shape = (num_test, unrolled)
            - `y`: defaults to `None`; can be set by user to `shape = (num_test, 1)`
            - 'y_pred': defaults to `None`; set by model to `shape = (num_test, unrolled)`
    """
    n_train = dataset['num train'] # 12000
    n_test = dataset['num test'] # 3000
    shape = dataset['image']['shape']
    unrolled = dataset['image']['unrolled']

    (train_filepath, test_filepath) = get_filepath(dataset)

    train_data = {'x': {}, 'y': {}}
    test_data = {'x': {}, 'y': None, 'y_pred': None}
    if dataset['io']['filetype'] == '.mat':
        train_mat = loadmat(train_filepath)
        test_mat = loadmat(test_filepath)

        train_data['x'] = train_mat.get('data')
        train_data['y'] = train_mat.get('labels')
        test_data['x'] = test_mat.get('data')
    elif dataset['io']['filetype'] == '.bin':
        with np.memmap(train_filepath, dtype='uint8', mode='c', shape=(n_train, unrolled+1)) as mm:
            train_data['x'] = mm[np.repeat(np.arange(n_train), unrolled), np.tile(np.arange(1,unrolled+1), n_train)]
            train_data['y'] = mm[np.arange(n_train), np.repeat(0, n_train)]

        with np.memmap(test_filepath, dtype='uint8', mode='c', shape=(n_test, unrolled)) as mm:
            test_data['x'] = np.reshape(mm, dataset['image']['shape'])
    else:
        raise ValueError, "unsupported filetype: %s \n" %(dataset['io']['filetype'])

    format_data(train_data, shape)
    format_data(test_data, shape)
    (train, valid) = split_data(dataset, train_data)
    test = test_data
    return (train, valid, test)
