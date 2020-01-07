import torch
import scipy.io as sio
from dataset import Dataset
import logging
import os
import time


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


class Logger(object):
    def __init__(self, clevel=logging.DEBUG, Flevel=logging.DEBUG):
        # create dictionary
        file_name = f"./log/log_{time.strftime('%M_%S ',time.localtime(time.time()))}.log"
        if not os.path.exists(file_name):
            folder_name = './log'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            open(file_name, 'a')
        self.logger = logging.getLogger(file_name)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # set CMD dairy
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(clevel)
        # set log file
        fh = logging.FileHandler(file_name, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(Flevel)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)
