from param_config import ParamConfig
from partition import KFoldPartition
from loss_utils import MapLoss, RMSELoss
from dfnn_run import dfnn_ite_rules_mu
from h_utils import HNormal
from fnn_solver import FnnSolveCls
from loss_utils import LossComputeNormal
from rules import RuleKmeans
from utils import load_data
from dataset import Result
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.dataset_list = ['eegDual_sub1', 'HRSS_anomalous_optimized', 'HRSS_anomalous_standard']
para_mu_list = torch.linspace(-4, 4, 9)
# para_mu_list = torch.ones(1)
# para_mu_list = torch.linspace(-3, -1, 3)
param_config.para_mu_list = torch.pow(10, para_mu_list).double()
param_config.h_computer = HNormal()
param_config.fnn_solver = FnnSolveCls()
param_config.loss_compute = LossComputeNormal()
param_config.rules = RuleKmeans()
param_config.n_agents = 5
# generate partitions of dataset
param_config.patition_strategy = KFoldPartition(param_config.kfolds)


param_config.log.info('==========start training!=========')

for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.dataset_list[int(i)]
    # load dataset
    dataset = load_data(dataset_file)
    param_config.log.info(f"dataset: {dataset.name} is loaded for {dataset.task}.")
    dataset.generate_n_partitions(param_config.runs, param_config.patition_strategy)
    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = MapLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve = \
        dfnn_ite_rules_mu(15, param_config, dataset)

    loss_c_train_mean = loss_c_train.mean(2)
    loss_c_test_mean = loss_c_test.mean(2)
    loss_d_train_mean = loss_d_train.mean(2)
    loss_d_test_mean = loss_d_test.mean(2)

    results = Result(param_config.para_mu_list)
    results.loss_c_train = loss_c_train
    results.loss_c_train_mean = loss_c_train_mean
    results.loss_c_test = loss_c_test
    results.loss_c_test_mean = loss_c_test_mean
    results.loss_d_train = loss_d_train
    results.loss_d_train_mean = loss_d_train_mean
    results.loss_d_test = loss_d_test
    results.loss_d_test_mean = loss_d_test_mean

    results.loss_curve = loss_curve

    data_save_dir = "./results/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}{dataset_file}_k_cls.pt"
    torch.save(results, data_save_file)
