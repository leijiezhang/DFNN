from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_rules_ao
from utils import load_data
from math_utils import mapminmax
import torch
from dataset import Dataset
import os
import scipy.io as io
import matplotlib.pyplot as plt


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('loss_config')
n_rule_list = torch.arange(2, 21, 2)
param_config.n_rules_list = n_rule_list

n_list_rule = n_rule_list.shape[0]
loss_mtrx = torch.zeros(4, n_list_rule)
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file, param_config.dataset_name)
    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)

    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)
    param_config.patition_strategy.set_current_folds(0)
    train_data, test_data = dataset.get_run_set()
    # if the dataset is like a eeg data, which has trails hold sample blocks
    if dataset.X.shape.__len__() == 3:
        # reform training dataset
        y = torch.empty(0, 1).double()
        x = torch.empty(0, dataset.X.shape[2]).double()
        for ii in torch.arange(train_data.Y.shape[0]):
            x = torch.cat((x, train_data.X[ii]), 0)
            size_smpl_ii = train_data.X[ii].shape[0]
            y_tmp = train_data.Y[ii].repeat(size_smpl_ii, 1)
            y = torch.cat((y, y_tmp), 0)
        train_data = Dataset(train_data.name, x, y, train_data.task)

        # reform test dataset
        y = torch.empty(0, 1).double()
        x = torch.empty(0, dataset.X.shape[2]).double()
        for ii in torch.arange(test_data.Y.shape[0]):
            x = torch.cat((x, test_data.X[ii]), 0)
            size_smpl_ii = test_data.X[ii].shape[0]
            y_tmp = test_data.Y[ii].repeat(size_smpl_ii, 1)
            y = torch.cat((y, y_tmp), 0)
        test_data = Dataset(test_data.name, x, y, test_data.task)

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()
        dataset.Y = mapminmax(dataset.Y)

    loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
        dfnn_rules_ao(param_config, train_data, test_data)

    loss_mtrx[0, :] = loss_c_train
    loss_mtrx[1, :] = loss_c_test
    loss_mtrx[2, :] = loss_d_train
    loss_mtrx[3, :] = loss_d_test

    plt.figure(12)
    plt.subplot(221)
    plt.plot(torch.arange(n_list_rule), loss_mtrx[0, :], 'r--')
    plt.subplot(222)
    plt.plot(torch.arange(n_list_rule), loss_mtrx[1, :], 'r--')
    plt.subplot(223)
    plt.plot(torch.arange(n_list_rule), loss_mtrx[2, :], 'r--')
    plt.subplot(224)
    plt.plot(torch.arange(n_list_rule), loss_mtrx[3, :], 'r--')

    plt.show()

    save_dict = dict()
    save_dict["loss_mtrx"] = loss_mtrx

    data_save_dir = f"./results/{param_config.dataset_name}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}/{dataset_file}_loss_rule.pt"
    torch.save(save_dict, data_save_file)

    save_dict = dict()
    save_dict["loss_c_train"] = loss_c_train.numpy()
    save_dict["loss_c_test"] = loss_c_test.numpy()
    save_dict["loss_d_train"] = loss_d_train.numpy()
    save_dict["loss_d_test"] = loss_d_test.numpy()
    data_save_file = f"{data_save_dir}/{dataset_file}_loss_rule.mat"
    io.savemat(data_save_file, save_dict)
