from param_config import ParamConfig
from partition import KFoldPartition
from dfnn_fc import dfnn_fc_method, dfnn_fc_ite_rules
import torch
import os


# -------------------------------------------------------------------------
# --- PARAMS_SELECTION - ---------------------------------------------------
# --- Define the parameters for the simulation - ---------------------------
# -------------------------------------------------------------------------

# Dataset configuration
# dataset_file = 'airfoil.mat'
#  # Dataset to load
# dataset_file = 'CCPP.mat'
dataset_file = 'CASP'
# dataset_file = 'HRSS_anomalous_optimized'
# dataset_file = 'HRSS_anomalous_standard'
# dataset_file = 'eegDual_sub1_format'
# dataset_file = 'kc_house'
# dataset_file = 'motor_temperature'
# dataset_file = 'motor_temperature_2' # delete Torque
# dataset_file = 'motor_temperature_3' # Wilhelm's setting
# dataset_file = 'motor_temperature_4' # My setting

# init the parameters and load dataset
param_setting = ParamConfig()

dataset = param_setting.load_data(dataset_file)

# generate partitions of dataset
patition_strategy = KFoldPartition(param_setting.kfolds)
dataset.generate_n_partitions(param_setting.runs, patition_strategy)

loss_list, loss_dlist, loss_admm_list = dfnn_fc_method(10, param_setting, patition_strategy, dataset)

# loss_list, loss_dlist, loss_admm_list = dfnn_fc_ite_rules(10, param_setting, patition_strategy, dataset)

data_save = dict()
data_save['loss_list'] = loss_list
data_save['loss_admm_list'] = loss_admm_list
data_save['loss_dlist'] = loss_dlist
data_save_dir = "./results/"
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}{dataset_file}.pt"
torch.save(data_save, data_save_file)

