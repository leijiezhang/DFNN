from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_ite_rules_mu_kfolds
from utils import load_data, Logger
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('hrss_config')

para_mu_list = torch.linspace(-10, 10, 21)
param_config.para_mu_list = torch.pow(2, para_mu_list).double()

acc_c_train_arr = []
acc_c_test_arr = []
acc_d_train_arr = []
acc_d_test_arr = []

acc_c_train_list = []
acc_c_test_list = []
acc_d_train_list = []
acc_d_test_list = []
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
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

    loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr = \
        dfnn_ite_rules_mu_kfolds(25, param_config, dataset)

    acc_c_train_list.append(loss_c_train_mu_tsr)
    acc_c_test_list.append(loss_c_test_mu_tsr)
    acc_d_train_list.append(loss_d_train_mu_tsr)
    acc_d_test_list.append(loss_d_test_mu_tsr)

    loss_c_train_mean_mtrx = loss_c_train_mu_tsr.mean(2)
    loss_c_test_mean_mtrx = loss_c_test_mu_tsr.mean(2)
    loss_d_train_mean_mtrx = loss_d_train_mu_tsr.mean(2)
    loss_d_test_mean_mtrx = loss_d_test_mu_tsr.mean(2)

    best_d_test = loss_d_test_mean_mtrx.max()
    pos_best = torch.where(loss_d_test_mean_mtrx == best_d_test)

    acc_c_train_best = loss_c_train_mean_mtrx[pos_best[0][0], pos_best[1][0]]
    acc_c_test_best = loss_c_test_mean_mtrx[pos_best[0][0], pos_best[1][0]]
    acc_d_train_best = loss_d_train_mean_mtrx[pos_best[0][0], pos_best[1][0]]
    acc_d_test_best = loss_d_test_mean_mtrx[pos_best[0][0], pos_best[1][0]]

acc_c_train = torch.tensor(acc_c_train_arr).mean()
acc_c_test = torch.tensor(acc_c_test_arr).mean()
acc_d_train = torch.tensor(acc_d_train_arr).mean()
acc_d_test = torch.tensor(acc_d_test_arr).mean()
acc_c_train_std = torch.tensor(acc_c_train_arr).std()
acc_c_test_std = torch.tensor(acc_c_test_arr).std()
acc_d_train_std = torch.tensor(acc_d_train_arr).std()
acc_d_test_std = torch.tensor(acc_d_test_arr).std()

param_config.log.info(
    f"mAp of training data on centralized method: "
    f"{round(float(acc_c_train), 4)}/{round(float(acc_c_train_std), 4)}")
param_config.log.info(
    f"mAp of test data on centralized method: "
    f"{round(float(acc_c_test), 4)}/{round(float(acc_c_test_std), 4)}")
param_config.log.info(
    f"mAp of training data on distributed method: "
    f"{round(float(acc_d_train), 4)}/{round(float(acc_d_train_std),  4)}")
param_config.log.info(
    f"mAp of test data on distributed method:"
    f" {round(float(acc_d_test), 4)}/{round(float(acc_d_test_std), 4)}")

dave_dict = dict()
dave_dict["acc_c_train_list"] = acc_c_train_list
dave_dict["acc_c_test_list"] = acc_c_test_list
dave_dict["acc_d_train_list"] = acc_d_train_list
dave_dict["acc_d_test_list"] = acc_d_test_list

data_save_dir = f"./results/hrss/"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/dfnn.pt"
torch.save(dave_dict, data_save_file)
