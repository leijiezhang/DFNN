import torch
import scipy.io as sio
from dataset import Dataset


def load_data(dataset_str):
    dir_dataset = f"./datasets/{dataset_str}.pt"
    load_data = torch.load(dir_dataset)
    dataset_name = load_data['name']
    x = load_data['X']
    y = load_data['Y']
    task = load_data['task']
    dataset = Dataset(dataset_name, x, y, task)
    dataset.normalize(-1, 1)
    return dataset


def dataset_parse(dataset_name):
    """
    todo: parse dataset from .mat to .pt
    :param dataset_name:
    :return:
    """
    dir_dataset = f"./datasets/{dataset_name}.mat"
    load_data = sio.loadmat(dir_dataset)
    dataset_name = load_data['name'][0]
    x_orig = load_data['X']
    y_orig = load_data['Y']
    x = torch.tensor(x_orig).double()
    y_tmp = []
    for i in torch.arange(len(y_orig)):
        y_tmp.append(float(y_orig[i]))
    y = torch.tensor(y_tmp).double()
    task = load_data['task'][0]
    data_save = dict()
    data_save['task'] = task
    data_save['name'] = dataset_name
    data_save['X'] = x
    if task == 'C':
        y_min = torch.min(y)
        y_gap = y_min - 0
        y = y - y_gap
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['Y_r'] = y
        data_save['Y'] = y_c
    else:
        data_save['Y'] = y.unsqueeze(1)
    dir_dataset = f"./datasets/{dataset_name}.pt"
    torch.save(data_save, dir_dataset)
