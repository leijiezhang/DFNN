from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_kfolds
from utils import load_data
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('eegdual_config')
# param_config.dataset_list = ['eegDual_subj_all']

acc_c_train_arr = torch.empty(0, param_config.n_kfolds).double()
acc_c_test_arr = torch.empty(0, param_config.n_kfolds).double()
acc_d_train_arr = torch.empty(0, param_config.n_kfolds).double()
acc_d_test_arr = torch.empty(0, param_config.n_kfolds).double()
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file, param_config.dataset_name)
    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    # acc_c_train, acc_c_test, acc_d_train, acc_d_test, acc_curve = \
    #     dfnn_ite_rules_mu(15, param_config, dataset)

    # acc_c_train, acc_c_test = svm_local(param_config, train_data, test_data)
    # acc_d_train = 0.0
    # acc_d_test = 0.0

    # acc_c_train, acc_c_test, acc_d_train, acc_d_test = \
    #     neuron_run(param_config, dataset)

    acc_c_train, acc_c_test, acc_d_train, acc_d_test = \
        dfnn_kfolds(param_config, dataset)

    # test_acc, train_losses = mlp_run(param_config, dataset)

    acc_c_train_arr = torch.cat((acc_c_train_arr, acc_c_train.unsqueeze(0).double()), 0)
    acc_c_test_arr = torch.cat((acc_c_test_arr, acc_c_test.unsqueeze(0).double()), 0)
    acc_d_train_arr = torch.cat((acc_d_train_arr, acc_d_train.unsqueeze(0).double()), 0)
    acc_d_test_arr = torch.cat((acc_d_test_arr, acc_d_test.unsqueeze(0).double()), 0)

acc_c_train = acc_c_train_arr.mean()
acc_c_test = acc_c_test_arr.mean()
acc_d_train = acc_d_train_arr.mean()
acc_d_test = acc_d_test_arr.mean()
acc_c_train_std = acc_c_train_arr.std()
acc_c_test_std = acc_c_test_arr.std()
acc_d_train_std = acc_d_train_arr.std()
acc_d_test_std = acc_d_test_arr.std()

param_config.log.info(f"mAp of training data on centralized method: {round(float(acc_c_train), 4)}/{round(float(acc_c_train_std), 4)}")
param_config.log.info(f"mAp of test data on centralized method: {round(float(acc_c_test), 4)}/{round(float(acc_c_test_std), 4)}")
param_config.log.info(f"mAp of training data on distributed method: {round(float(acc_d_train), 4)}/{round(float(acc_d_train_std), 4)}")
param_config.log.info(f"mAp of test data on distributed method: {round(float(acc_d_test), 4)}/{round(float(acc_d_test_std), 4)}")
