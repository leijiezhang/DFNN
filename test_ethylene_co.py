from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_rules_ao
from utils import load_data
from math_utils import mapminmax
import torch
import os
import scipy.io as io


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('ethylene_co_config')
n_rule_list = torch.arange(1, 51, 5)
param_config.n_rules_list = n_rule_list
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file, param_config.dataset_name)
    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)

    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)
    param_config.patition_strategy.set_current_folds(0)
    train_data, test_data = dataset.get_run_set()
    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()
        dataset.Y = mapminmax(dataset.Y)

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve = \
    #     dfnn_ite_rules_mu(15, param_config, dataset)

    # loss_c_train, loss_c_test = svm_kfolds(param_config, dataset)
    # loss_d_train = torch.tensor([0.0, 0.0])
    # loss_d_test = torch.tensor([0.0, 0.0])

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
    #     neuron_run(param_config, dataset)

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
    #     fuzzy_net_run(param_config, dataset)

    loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
        dfnn_rules_ao(param_config, train_data, test_data)

    # test_acc, train_losses = mlp_run(param_config, dataset)

    # loss_c_train_mean = loss_c_train.mean()
    # loss_c_test_mean = loss_c_test.mean()
    # loss_d_train_mean = loss_d_train.mean()
    # loss_d_test_mean = loss_d_test.mean()
    #
    # loss_c_train_std = loss_c_train.std()
    # loss_c_test_std = loss_c_test.std()
    # loss_d_train_std = loss_d_train.std()
    # loss_d_test_std = loss_d_test.std()
    #
    # param_config.log.info(f"mAp of training data on centralized method: {round(float(loss_c_train_mean), 4)}"
    #                       f"/{round(float(loss_c_train_std), 4)}")
    # param_config.log.info(f"mAp of test data on centralized method: {round(float(loss_c_test_mean), 4)}"
    #                       f"/{round(float(loss_c_test_std), 4)}")
    # param_config.log.info(f"mAp of training data on distributed method"
    #                       f": {round(float(loss_d_train_mean), 4)}/{round(float(loss_d_train_std), 4)}")
    # param_config.log.info(f"mAp of test data on distributed method: {round(float(loss_d_test_mean), 4)}"
    #                       f"/{round(float(loss_d_test_std), 4)}")
    save_dict = dict()
    save_dict["loss_c_train"] = loss_c_train
    save_dict["loss_c_test"] = loss_c_test
    save_dict["loss_d_train"] = loss_d_train
    save_dict["loss_d_test"] = loss_d_test

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