import os

utils_path = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.normpath(os.path.join(utils_path, '..', 'data'))

def get_dataset(filetype='.mat'):
    """
    """
    dataset = {}
    dataset['image'] = {}
    dataset['image']['shape'] = (32, 32, 3)
    dataset['image']['unrolled'] = 32*32*3

    dataset['num classes'] = 3
    dataset['num train'] = 12000
    dataset['num test'] = 3000
    dataset['validation split'] = 8 # 12000 / 8 = 1500

    dataset['io'] = {}
    dataset['io']['path'] = dataset_path
    dataset['io']['train filename'] = 'data_batch'
    dataset['io']['test filename'] = 'test_data'

    if filetype == '.mat':
        dataset['io']['directory'] = 'mat'
        dataset['io']['filetype'] = '.mat'

    elif filetype == '.bin':
        dataset['io']['directory'] = 'bin'
        dataset['io']['filetype'] = '.bin'
    else:
        raise ValueError, "unsupported filetype: %s \n" %(filetype)

    return dataset
