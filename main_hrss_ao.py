from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_ite_rules_para_kfold_ao
from utils import load_data
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('hrss_config_ao')

para_mu_list = torch.linspace(-4, 4, 9)
param_config.para_mu_list = torch.pow(10, para_mu_list).double()
param_config.para_mu1_list = torch.pow(10, para_mu_list).double()

n_rule_list = torch.arange(1, 26, 1)
param_config.n_rules_list = n_rule_list

acc_c_train_arr = []
acc_c_test_arr = []
acc_d_train_arr = []
acc_d_test_arr = []

acc_c_train_list = []
acc_c_test_list = []
acc_d_train_list = []
acc_d_test_list = []

dataset_file = param_config.get_cur_dataset(0)

# load dataset
dataset = load_data(dataset_file)
dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)

dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)
param_config.log.debug(f"=====starting on {dataset.name}=======")
loss_fun = None
if dataset.task == 'C':
    param_config.log.war(f"=====Mission: Classification=======")
    param_config.loss_fun = LikelyLoss()
else:
    param_config.log.war(f"=====Mission: Regression=======")
    param_config.loss_fun = RMSELoss()

loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr = \
    dfnn_ite_rules_para_kfold_ao(param_config, dataset)

loss_c_train_mean_mtrx = loss_c_train_tsr.mean(4)
loss_c_test_mean_mtrx = loss_c_test_tsr.mean(4)
loss_d_train_mean_mtrx = loss_d_train_tsr.mean(4)
loss_d_test_mean_mtrx = loss_d_test_tsr.mean(4)

best_d_test = loss_d_test_mean_mtrx.max()
pos_best = torch.where(loss_d_test_mean_mtrx == best_d_test)

acc_c_train_best = loss_c_train_mean_mtrx[pos_best[0][0], pos_best[1][0], pos_best[2][0]]
acc_c_test_best = loss_c_test_mean_mtrx[pos_best[0][0], pos_best[1][0], pos_best[2][0]]
acc_d_train_best = loss_d_train_mean_mtrx[pos_best[0][0], pos_best[1][0, pos_best[2][0]]]
acc_d_test_best = loss_d_test_mean_mtrx[pos_best[0][0], pos_best[1][0], pos_best[2][0]]


param_config.log.info(
    f"mAp of training data on centralized method: "
    f"{round(float(acc_c_train_best), 4)}")
param_config.log.info(
    f"mAp of test data on centralized method: "
    f"{round(float(acc_c_test_best), 4)}")
param_config.log.info(
    f"mAp of training data on distributed method: "
    f"{round(float(acc_d_train_best), 4)}")
param_config.log.info(
    f"mAp of test data on distributed method:"
    f" {round(float(acc_d_test_best), 4)}")

dave_dict = dict()
dave_dict["loss_c_train_tsr"] = loss_c_train_tsr
dave_dict["loss_c_test_tsr"] = loss_c_test_tsr
dave_dict["loss_d_train_tsr"] = loss_d_train_tsr
dave_dict["loss_d_test_tsr"] = loss_d_test_tsr

data_save_dir = f"./results/hrss/"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/h_dfnn_ao.pt"
torch.save(dave_dict, data_save_file)
