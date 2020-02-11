from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import fuzzy_net_run, mlp_run, neuron_run, dfnn_kfolds
from utils import load_data, load_eeg_data
import torch


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('seed_config')

acc_c_train_arr = []
acc_c_test_arr = []
acc_d_train_arr = []
acc_d_test_arr = []
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    train_data, test_data = load_eeg_data(dataset_file, "seed/channel12")

    param_config.log.debug(f"=====starting on {train_data.name}=======")
    loss_fun = None
    if train_data.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    # acc_c_train, acc_c_test, acc_d_train, acc_d_test, acc_curve = \
    #     dfnn_ite_rules_mu(15, param_config, dataset)

    # acc_c_train, acc_c_test = svm_local(param_config, train_data, test_data)
    # acc_d_train = 0.0
    # acc_d_test = 0.0

    # acc_c_train, acc_c_test, acc_d_train, acc_d_test = \
    #     neuron_run(param_config, dataset)

    acc_c_train, acc_c_test, acc_d_train, acc_d_test = \
        fuzzy_net_run(param_config, train_data, test_data)

    # test_acc, train_losses = mlp_run(param_config, dataset)

    acc_c_train_arr.append(acc_c_train)
    acc_c_test_arr.append(acc_c_test)
    acc_d_train_arr.append(acc_d_train)
    acc_d_test_arr.append(acc_d_test)

acc_c_train = torch.tensor(acc_c_train_arr).mean()
acc_c_test = torch.tensor(acc_c_test_arr).mean()
acc_d_train = torch.tensor(acc_d_train_arr).mean()
acc_d_test = torch.tensor(acc_d_test_arr).mean()
acc_c_train_std = torch.tensor(acc_c_train_arr).std()
acc_c_test_std = torch.tensor(acc_c_test_arr).std()
acc_d_train_std = torch.tensor(acc_d_train_arr).std()
acc_d_test_std = torch.tensor(acc_d_test_arr).std()

param_config.log.info(f"mAp of training data on centralized method: {round(float(acc_c_train), 4)}/{round(float(acc_c_train_std), 4)}")
param_config.log.info(f"mAp of test data on centralized method: {round(float(acc_c_test), 4)}/{round(float(acc_c_test_std), 4)}")
param_config.log.info(f"mAp of training data on distributed method: {round(float(acc_d_train), 4)}/{round(float(acc_d_train_std), 4)}")
param_config.log.info(f"mAp of test data on distributed method: {round(float(acc_d_test), 4)}/{round(float(acc_d_test_std), 4)}")
