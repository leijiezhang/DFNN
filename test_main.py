from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import fuzzy_net_run, svm_local, mlp_run, neuron_run, dfnn_kfolds
from dataset import Result
from utils import load_data, Logger
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('normal_config')

for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file)
    train_data, test_data = dataset.get_run_set()

    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)
    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve = \
    #     dfnn_ite_rules_mu(15, param_config, dataset)

    # loss_train_tsr, loss_test_tsr = svm_local(param_config, dataset)

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
    #     neuron_run(param_config, dataset)

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
    #     fuzzy_net_run(param_config, dataset)

    loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
        dfnn_kfolds(param_config, dataset)

    # test_acc, train_losses = mlp_run(param_config, dataset)

    loss_c_train_mean = loss_c_train.mean()
    loss_c_test_mean = loss_c_test.mean()
    loss_d_train_mean = loss_d_train.mean()
    loss_d_test_mean = loss_d_test.mean()

    results_reg = Result(param_config.para_mu_list)
    results_reg.loss_c_train = loss_c_train
    results_reg.loss_c_train_mean = loss_c_train_mean
    results_reg.loss_c_test = loss_c_test
    results_reg.loss_c_test_mean = loss_c_test_mean
    results_reg.loss_d_train = loss_d_train
    results_reg.loss_d_train_mean = loss_d_train_mean
    results_reg.loss_d_test = loss_d_test
    results_reg.loss_d_test_mean = loss_d_test_mean

    data_save_dir = "./results/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}{dataset_file}_fc_test.pt"
    # torch.save(1, data_save_file)
