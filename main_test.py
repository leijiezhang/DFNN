from param_config import ParamConfig
from partition import KFoldPartition
from loss_utils import MapLoss, RMSELoss
from dfnn_run import dfnn_ite_rules_mu, dfnn_ite_rules, dfnn_method
from dataset import Result
from h_utils import HNormal
from fnn_solver import FnnSolveReg, FnnSolveCls
from loss_utils import LossComputeNormal
from rules import RuleKmeans, RuleFuzzyCmeans
from utils import load_data, Logger
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()

# param_config.dataset_list = ['CASP']
param_config.dataset_list = ['eegDual_sub1']

# para_mu_list = torch.linspace(-4, 4, 9)
para_mu_list = -2 * torch.zeros(1)
# para_mu_list = torch.linspace(-3, -1, 3)

param_config.para_mu_list = torch.pow(10, para_mu_list).double()
param_config.para_mu_current = param_config.para_mu_list[0]
param_config.h_computer = HNormal()
param_config.loss_compute = LossComputeNormal()
param_config.rules = RuleKmeans()
# param_config.rules = RuleFuzzyCmeans()
# generate partitions of dataset
# param_config.fnn_solver = FnnSolveCls()
param_config.patition_strategy = KFoldPartition(param_config.kfolds)


for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.dataset_list[int(i)]
    # load dataset
    dataset = load_data(dataset_file)

    dataset.generate_n_partitions(param_config.runs, param_config.patition_strategy)
    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = MapLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    # loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve, best_idx, best_mu = \
    #     dfnn_ite_rules_mu(15, param_config, dataset)
    param_config.n_agents = 5
    param_config.fnn_solver = FnnSolveCls()
    loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve_list = \
        dfnn_method(5, param_config, dataset)

    loss_c_train_mean = loss_c_train.mean()
    loss_c_test_mean = loss_c_test.mean()
    loss_d_train_mean = loss_d_train.mean()
    loss_d_test_mean = loss_d_test.mean()

    results_cls = Result(param_config.para_mu_list)
    results_cls.loss_c_train = loss_c_train
    results_cls.loss_c_train_mean = loss_c_train_mean
    results_cls.loss_c_test = loss_c_test
    results_cls.loss_c_test_mean = loss_c_test_mean
    results_cls.loss_d_train = loss_d_train
    results_cls.loss_d_train_mean = loss_d_train_mean
    results_cls.loss_d_test = loss_d_test
    results_cls.loss_d_test_mean = loss_d_test_mean

    results_cls.loss_curve = loss_curve_list

    param_config.fnn_solver = FnnSolveReg()
    loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_curve_list = \
        dfnn_method(13, param_config, dataset)

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

    results_reg.loss_curve = loss_curve_list

    data_save_dir = "./results/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}{dataset_file}_k_best1.pt"
    torch.save(1, data_save_file)
