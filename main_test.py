from param_config import ParamConfig
from partition import KFoldPartition
from loss_utils import MapLoss, RMSELoss
from dfnn_run import dfnn_ite_rules_mu, dfnn_ite_rules, dfnn_method
from h_utils import HNormal
from fnn_solver import FnnSolveCls
from loss_utils import LossComputeNormal
from rules import RuleKmeans
from utils import load_data
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
# dataset_list = ['housing', 'HRSS_anomalous_optimized', 'HRSS_anomalous_standard',
#                 'kc_house', 'motor_temperature', 'quake', 'skills',
#                 'strength', 'telemonitoring', 'yacht']
param_config.dataset_list = ['eegDual_sub1']
# para_mu_list = torch.linspace(-4, 4, 9)
para_mu_list = -2 * torch.zeros(1)
# para_mu_list = torch.linspace(-3, -1, 3)
param_config.para_mu_list = torch.pow(10, para_mu_list).double()
param_config.para_mu_current = param_config.para_mu_list[0]
param_config.h_computer = HNormal()
param_config.fnn_solver = FnnSolveCls()
param_config.loss_compute = LossComputeNormal()
param_config.rules = RuleKmeans()
# generate partitions of dataset
param_config.patition_strategy = KFoldPartition(param_config.kfolds)


for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.dataset_list[int(i)]
    # load dataset
    dataset = load_data(dataset_file)

    dataset.generate_n_partitions(param_config.runs, param_config.patition_strategy)
    print(dataset.name)
    loss_fun = None
    if dataset.task == 'C':
        param_config.loss_fun = MapLoss()
    else:
        param_config.loss_fun = RMSELoss()

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve, best_idx, best_mu = \
    #     dfnn_ite_rules_mu(15, param_config, dataset)
    param_config.n_agents = 5
    loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve_list = \
        dfnn_method(13, param_config, dataset)

    data_save = dict()
    data_save['loss_c_train'] = loss_c_train
    data_save['loss_c_test'] = loss_c_test
    data_save['loss_d_train'] = loss_d_train
    data_save['loss_d_test'] = loss_d_test
    # data_save['loss_curve_list'] = loss_curve
    # data_save['best_idx'] = best_idx
    # data_save['best_mu'] = best_mu
    data_save_dir = "./results/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}{dataset_file}_k_best1.pt"
    torch.save(data_save, data_save_file)
