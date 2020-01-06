from param_config import ParamConfig
from partition import KFoldPartition
from loss_utils import MapLoss, RMSELoss
from dfnn_run import dfnn_ite_rules_mu
from h_utils import HNormal
from fnn_solver import FnnSolveReg
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
para_mu_list = torch.linspace(-3, -1, 3)
param_config.para_mu_list = torch.pow(10, para_mu_list).double()
param_config.h_computer = HNormal()
param_config.fnn_solver = FnnSolveReg()
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
    loss_g, loss_d, loss_curve, best_idx, best_mu = dfnn_ite_rules_mu(3, param_config, dataset)
    data_save = dict()
    data_save['loss_g_tsr'] = loss_g
    data_save['loss_curve_list'] = loss_curve
    data_save['loss_d_tsr'] = loss_d
    data_save['best_idx'] = best_idx
    data_save['best_mu'] = best_mu
    data_save_dir = "./results/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}{dataset_file}_k.pt"
    torch.save(data_save, data_save_file)
