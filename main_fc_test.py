from param_config import ParamConfig
from partition import KFoldPartition
from dfnn_fc import dfnn_fc_method
from dfnn_fc import dfnn_fc_ite_rules
from loss_function import MapLoss, RMSELoss
import torch
import os


# -------------------------------------------------------------------------
# --- PARAMS_SELECTION - ---------------------------------------------------
# --- Define the parameters for the simulation - ---------------------------
# -------------------------------------------------------------------------

# ==================below are the original codes ====================
# init the parameters and load dataset
dataset_file = 'CASP'
param_setting = ParamConfig()

dataset = param_setting.load_data(dataset_file)

# generate partitions of dataset
patition_strategy = KFoldPartition(param_setting.kfolds)
dataset.generate_n_partitions(param_setting.runs, patition_strategy)

loss_list, loss_dlist, loss_admm_list = dfnn_fc_method(5, param_setting,
                                                       patition_strategy,
                                                       dataset, RMSELoss(),
                                                       param_setting.para_mu)

loss_list, loss_dlist, loss_admm_list = dfnn_fc_ite_rules(10,
                                                          param_setting,
                                                          patition_strategy,
                                                          dataset, MapLoss(),
                                                          param_setting.para_mu)

data_save = dict()
data_save['loss_list'] = loss_list
data_save['loss_admm_list'] = loss_admm_list
data_save['loss_dlist'] = loss_dlist
data_save_dir = "./results/"
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}{dataset_file}.pt"
torch.save(data_save, data_save_file)
