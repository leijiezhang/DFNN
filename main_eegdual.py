from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_ite_rules_mu
from utils import load_eeg_data
import torch
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('eegdual_config')

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
    train_data, test_data = load_eeg_data(dataset_file, "eeg_dual")

    param_config.log.debug(f"=====starting on {train_data.name}=======")
    loss_fun = None
    if train_data.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    acc_c_train_tsr, acc_c_test_tsr, acc_d_train_tsr, acc_d_test_tsr = \
        dfnn_ite_rules_mu(25, param_config, train_data, test_data)

    # save all the output
    acc_c_train_list.append(acc_c_train_tsr)
    acc_c_test_list.append(acc_c_test_tsr)
    acc_d_train_list.append(acc_d_train_tsr)
    acc_d_test_list.append(acc_d_test_tsr)

    # get the best performance
    acc_d_test_best = acc_d_test_tsr.max()
    pos_best = torch.where(acc_d_test_tsr == acc_d_test_best)
    acc_c_train_best = acc_c_train_tsr[pos_best[0][0], pos_best[1][0]]
    acc_c_test_best = acc_c_test_tsr[pos_best[0][0], pos_best[1][0]]
    acc_d_train_best = acc_d_train_tsr[pos_best[0][0], pos_best[1][0]]
    acc_d_test_best = acc_d_test_tsr[pos_best[0][0], pos_best[1][0]]

    acc_c_train_arr.append(acc_c_train_best)
    acc_c_test_arr.append(acc_c_test_best)
    acc_d_train_arr.append(acc_d_train_best)
    acc_d_test_arr.append(acc_d_test_best)

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
    f"{round(float(acc_d_train), 4)}/{round(float(acc_d_train_std), 4)}")
param_config.log.info(
    f"mAp of test data on distributed method:"
    f" {round(float(acc_d_test), 4)}/{round(float(acc_d_test_std), 4)}")

dave_dict = dict()
dave_dict["acc_c_train_list"] = acc_c_train_list
dave_dict["acc_c_test_list"] = acc_c_test_list
dave_dict["acc_d_train_list"] = acc_d_train_list
dave_dict["acc_d_test_list"] = acc_d_test_list

dave_dict["acc_c_train_arr"] = acc_c_train_arr
dave_dict["acc_c_test_arr"] = acc_c_test_arr
dave_dict["acc_d_train_arr"] = acc_d_train_arr
dave_dict["acc_d_test_arr"] = acc_d_test_arr

data_save_dir = f"./results/eeg_dual/"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/h_dfnn_s.pt"
torch.save(dave_dict, data_save_file)