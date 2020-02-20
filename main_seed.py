from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import dfnn_rules_para_ao
from utils import load_data
import torch
import os


# Dataset configuration
# init the parameters
n_channel_list = [4, 6, 9, 12]
param_config = ParamConfig()
param_config.config_parse('seed_config')

para_mu_list = torch.arange(-10, 1, 1).double()
param_config.para_mu_list = torch.pow(2, para_mu_list).double()
param_config.para_mu1_list = torch.pow(2, para_mu_list).double()

n_rule_list = torch.arange(4, 7, 1)
param_config.n_rules_list = n_rule_list

dataset_list_12 = ['seed_c12_e1_subj1', 'seed_c12_e1_subj2', 'seed_c12_e1_subj3', 'seed_c12_e1_subj4',
                   'seed_c12_e1_subj5', 'seed_c12_e1_subj6', 'seed_c12_e1_subj7', 'seed_c12_e1_subj8',
                   'seed_c12_e1_subj9', 'seed_c12_e1_subj10', 'seed_c12_e1_subj11', 'seed_c12_e1_subj12',
                   'seed_c12_e1_subj13', 'seed_c12_e1_subj14', 'seed_c12_e1_subj15', 'seed_c12_e2_subj1',
                   'seed_c12_e2_subj2', 'seed_c12_e2_subj3', 'seed_c12_e2_subj4', 'seed_c12_e2_subj5',
                   'seed_c12_e2_subj6', 'seed_c12_e2_subj7', 'seed_c12_e2_subj8', 'seed_c12_e2_subj9',
                   'seed_c12_e2_subj10',  'seed_c12_e2_subj11', 'seed_c12_e2_subj12', 'seed_c12_e2_subj13',
                   'seed_c12_e2_subj14', 'seed_c12_e2_subj15', 'seed_c12_e3_subj1', 'seed_c12_e3_subj2',
                   'seed_c12_e3_subj3', 'seed_c12_e3_subj4', 'seed_c12_e3_subj5',  'seed_c12_e3_subj6',
                   'seed_c12_e3_subj7', 'seed_c12_e3_subj8', 'seed_c12_e3_subj9', 'seed_c12_e3_subj10',
                   'seed_c12_e3_subj11', 'seed_c12_e3_subj12', 'seed_c12_e3_subj13', 'seed_c12_e3_subj14',
                   'seed_c12_e3_subj15']
dataset_list_9 = ['seed_c9_e1_subj1', 'seed_c9_e1_subj2', 'seed_c9_e1_subj3', 'seed_c9_e1_subj4',
                  'seed_c9_e1_subj5', 'seed_c9_e1_subj6', 'seed_c9_e1_subj7', 'seed_c9_e1_subj8',
                  'seed_c9_e1_subj9', 'seed_c9_e1_subj10', 'seed_c9_e1_subj11', 'seed_c9_e1_subj12',
                  'seed_c9_e1_subj13', 'seed_c9_e1_subj14', 'seed_c9_e1_subj15', 'seed_c9_e2_subj1',
                  'seed_c9_e2_subj2', 'seed_c9_e2_subj3', 'seed_c9_e2_subj4', 'seed_c9_e2_subj5',
                  'seed_c9_e2_subj6', 'seed_c9_e2_subj7', 'seed_c9_e2_subj8', 'seed_c9_e2_subj9',
                  'seed_c9_e2_subj10',  'seed_c9_e2_subj11', 'seed_c9_e2_subj12', 'seed_c9_e2_subj13',
                  'seed_c9_e2_subj14', 'seed_c9_e2_subj15', 'seed_c9_e3_subj1', 'seed_c9_e3_subj2',
                  'seed_c9_e3_subj3', 'seed_c9_e3_subj4', 'seed_c9_e3_subj5',  'seed_c9_e3_subj6',
                  'seed_c9_e3_subj7', 'seed_c9_e3_subj8', 'seed_c9_e3_subj9', 'seed_c9_e3_subj10',
                  'seed_c9_e3_subj11', 'seed_c9_e3_subj12', 'seed_c9_e3_subj13', 'seed_c9_e3_subj14',
                  'seed_c9_e3_subj15']
dataset_list_6 = ['seed_c6_e1_subj1', 'seed_c6_e1_subj2', 'seed_c6_e1_subj3', 'seed_c6_e1_subj4',
                  'seed_c6_e1_subj5', 'seed_c6_e1_subj6', 'seed_c6_e1_subj7', 'seed_c6_e1_subj8',
                  'seed_c6_e1_subj9', 'seed_c6_e1_subj10', 'seed_c6_e1_subj11', 'seed_c6_e1_subj12',
                  'seed_c6_e1_subj13', 'seed_c6_e1_subj14', 'seed_c6_e1_subj15', 'seed_c6_e2_subj1',
                  'seed_c6_e2_subj2', 'seed_c6_e2_subj3', 'seed_c6_e2_subj4', 'seed_c6_e2_subj5',
                  'seed_c6_e2_subj6', 'seed_c6_e2_subj7', 'seed_c6_e2_subj8', 'seed_c6_e2_subj9',
                  'seed_c6_e2_subj10',  'seed_c6_e2_subj11', 'seed_c6_e2_subj12', 'seed_c6_e2_subj13',
                  'seed_c6_e2_subj14', 'seed_c6_e2_subj15', 'seed_c6_e3_subj1', 'seed_c6_e3_subj2',
                  'seed_c6_e3_subj3', 'seed_c6_e3_subj4', 'seed_c6_e3_subj5',  'seed_c6_e3_subj6',
                  'seed_c6_e3_subj7', 'seed_c6_e3_subj8', 'seed_c6_e3_subj9', 'seed_c6_e3_subj10',
                  'seed_c6_e3_subj11', 'seed_c6_e3_subj12', 'seed_c6_e3_subj13', 'seed_c6_e3_subj14',
                  'seed_c6_e3_subj15']
dataset_list_4 = ['seed_c4_e1_subj1', 'seed_c4_e1_subj2', 'seed_c4_e1_subj3', 'seed_c4_e1_subj4',
                  'seed_c4_e1_subj5', 'seed_c4_e1_subj6', 'seed_c4_e1_subj7', 'seed_c4_e1_subj8',
                  'seed_c4_e1_subj9', 'seed_c4_e1_subj10', 'seed_c4_e1_subj11', 'seed_c4_e1_subj12',
                  'seed_c4_e1_subj13', 'seed_c4_e1_subj14', 'seed_c4_e1_subj15', 'seed_c4_e2_subj1',
                  'seed_c4_e2_subj2', 'seed_c4_e2_subj3', 'seed_c4_e2_subj4', 'seed_c4_e2_subj5',
                  'seed_c4_e2_subj6', 'seed_c4_e2_subj7', 'seed_c4_e2_subj8', 'seed_c4_e2_subj9',
                  'seed_c4_e2_subj10',  'seed_c4_e2_subj11', 'seed_c4_e2_subj12', 'seed_c4_e2_subj13',
                  'seed_c4_e2_subj14', 'seed_c4_e2_subj15', 'seed_c4_e3_subj1', 'seed_c4_e3_subj2',
                  'seed_c4_e3_subj3', 'seed_c4_e3_subj4', 'seed_c4_e3_subj5',  'seed_c4_e3_subj6',
                  'seed_c4_e3_subj7', 'seed_c4_e3_subj8', 'seed_c4_e3_subj9', 'seed_c4_e3_subj10',
                  'seed_c4_e3_subj11', 'seed_c4_e3_subj12', 'seed_c4_e3_subj13', 'seed_c4_e3_subj14',
                  'seed_c4_e3_subj15']
dataset_list = [dataset_list_4, dataset_list_6, dataset_list_9, dataset_list_12]

for jj in torch.arange(len(n_channel_list)):
    n_channel = n_channel_list[int(jj)]
    param_config.dataset_list = dataset_list[int(jj)]

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
        train_data, test_data = load_data(dataset_file, f"seed/channel{n_channel}")

        param_config.log.debug(f"=====starting on {train_data.name}=======")
        loss_fun = None
        if train_data.task == 'C':
            param_config.log.war(f"=====Mission: Classification=======")
            param_config.loss_fun = LikelyLoss()
        else:
            param_config.log.war(f"=====Mission: Regression=======")
            param_config.loss_fun = RMSELoss()

        acc_c_train_tsr, acc_c_test_tsr, acc_d_train_tsr, acc_d_test_tsr = \
            dfnn_rules_para_ao(param_config, train_data, test_data)

        # save all the output
        acc_c_train_list.append(acc_c_train_tsr)
        acc_c_test_list.append(acc_c_test_tsr)
        acc_d_train_list.append(acc_d_train_tsr)
        acc_d_test_list.append(acc_d_test_tsr)

        # get the best performance
        acc_c_test_best = acc_c_test_tsr.max()
        best_c_mask = torch.eq(acc_c_test_tsr, acc_c_test_best)
        acc_c_train_best = acc_c_train_tsr[best_c_mask].max()

        acc_d_test_best = acc_d_test_tsr.max()
        best_d_mask = torch.eq(acc_d_test_tsr, acc_d_test_best)
        acc_d_train_best = acc_d_train_tsr[best_d_mask].max()

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

    save_dict = dict()
    save_dict["acc_c_train_list"] = acc_c_train_list
    save_dict["acc_c_test_list"] = acc_c_test_list
    save_dict["acc_d_train_list"] = acc_d_train_list
    save_dict["acc_d_test_list"] = acc_d_test_list

    save_dict["acc_c_train_arr"] = acc_c_train_arr
    save_dict["acc_c_test_arr"] = acc_c_test_arr
    save_dict["acc_d_train_arr"] = acc_d_train_arr
    save_dict["acc_d_test_arr"] = acc_d_test_arr

    save_dict["acc_c_train"] = acc_c_train
    save_dict["acc_c_test"] = acc_c_test
    save_dict["acc_d_train"] = acc_d_train
    save_dict["acc_d_test"] = acc_d_test

    save_dict["acc_c_train_std"] = acc_c_train_std
    save_dict["acc_c_test_std"] = acc_c_test_std
    save_dict["acc_d_train_std"] = acc_d_train_std
    save_dict["acc_d_test_std"] = acc_d_test_std

    data_save_dir = f"./results/seed_ao"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}/{param_config.model_name}_{n_channel}_{param_config.n_rules}.pt"
    torch.save(save_dict, data_save_file)
