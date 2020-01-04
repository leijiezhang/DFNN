from param_config import ParamConfig
from partition import KFoldPartition
# from dfnn_k import dfnn_k_method
from dfnn_k import dfnn_k_ite_rules
from loss_function import MapLoss, RMSELoss
import torch
import os


# -------------------------------------------------------------------------
# --- PARAMS_SELECTION - ---------------------------------------------------
# --- Define the parameters for the simulation - ---------------------------
# -------------------------------------------------------------------------

# dataset_file = 'CASP'
dataset_list = ['abalone', 'airfoil', 'bikesharing',
                'blog', 'CASP', 'CCPP', 'eegDual_sub1',
                'housing', 'HRSS_anomalous_optimized', 'HRSS_anomalous_standard',
                'kc_house', 'motor_temperature', 'quake', 'skills',
                'strength', 'telemonitoring', 'yacht']

for i in torch.arange(len(dataset_list)):
    dataset_file = dataset_list[int(i)]
    # init the parameters and load dataset
    param_setting = ParamConfig()

    dataset = param_setting.load_data(dataset_file)

    # generate partitions of dataset
    patition_strategy = KFoldPartition(param_setting.kfolds)
    dataset.generate_n_partitions(param_setting.runs, patition_strategy)
    loss_fun = []
    if dataset.task == 'C':
        loss_fun = MapLoss()
    else:
        loss_fun = RMSELoss()

    loss_list, loss_dlist, loss_admm_list = dfnn_k_ite_rules(15, param_setting,
                                                             patition_strategy, dataset, loss_fun)
    data_save = dict()
    data_save['loss_list'] = loss_list
    data_save['loss_admm_list'] = loss_admm_list
    data_save['loss_dlist'] = loss_dlist
    data_save_dir = "./results/"
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    data_save_file = f"{data_save_dir}{dataset_file}_k.pt"
    torch.save(data_save, data_save_file)


# ==================below are the original codes ====================
# # init the parameters and load dataset
# param_setting = ParamConfig()
#
# dataset = param_setting.load_data(dataset_file)
#
# # generate partitions of dataset
# patition_strategy = KFoldPartition(param_setting.kfolds)
# dataset.generate_n_partitions(param_setting.runs, patition_strategy)
#
# loss_list, loss_dlist, loss_admm_list = dfnn_k_method(5,
#                                                       param_setting, patition_strategy,
#                                                       dataset, RMSELoss())

# loss_list, loss_dlist, loss_admm_list = dfnn_k_ite_rules(15, param_setting, patition_strategy, dataset)

# data_save = dict()
# data_save['loss_list'] = loss_list
# data_save['loss_admm_list'] = loss_admm_list
# data_save['loss_dlist'] = loss_dlist
# data_save_dir = "./results/"
# if not os.path.exists(data_save_dir):
#     os.makedirs(data_save_dir)
# data_save_file = f"{data_save_dir}{dataset_file}_k.pt"
# torch.save(data_save, data_save_file)

