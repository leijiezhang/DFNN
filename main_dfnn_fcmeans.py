from param_config import ParamConfig
from partition import KFoldPartition
from dataset import Dataset
from loss_function import LossFunc, RMSELoss
from kmeans_tools import KmeansUtils
from utils import compute_h_fc, compute_loss_fc
from fnn_tools import FnnKmeansTools
from rules import RuleFuzzyCmeans
import torch


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

loss_list = []
loss_dlist = []
loss_admm_list = []
for k in torch.arange(param_setting.kfolds):
    patition_strategy.set_current_folds(k)
    train_data, test_data = dataset.get_fold_data()
    train_data.generate_distribute(KFoldPartition(param_setting.n_agents))
    d_train_data = train_data.distribute_dataset()

    def fnn_main(train_data: Dataset, test_data: Dataset, n_rules, mu, loss_function: LossFunc):
        rules_train = RuleFuzzyCmeans()
        rules_train.fit(train_data.X, n_rules)
        h_train = compute_h_fc(train_data.X, rules_train)
        # run FNN solver for given rule number
        fnn_tools = FnnKmeansTools(mu)
        w_optimal = fnn_tools.fnn_solve_r(h_train, train_data.Y)
        rules_train.consequent_list = w_optimal

        # compute loss
        loss = compute_loss_fc(test_data, rules_train, loss_function)
        return loss, rules_train


    # trainning global method
    loss, rules = fnn_main(train_data, test_data, param_setting.n_rules,
                           param_setting.para_mu, RMSELoss())
    loss_list.append(loss)

    # train distributed fnn
    kmeans_utils = KmeansUtils()
    center_optimal, errors = kmeans_utils.kmeans_admm(d_train_data, param_setting.n_rules,
                                                      param_setting.n_agents, RuleFuzzyCmeans())
    loss_admm_list.append(errors)
    d_rules = RuleFuzzyCmeans()

    fnn_tools = FnnKmeansTools(param_setting.para_mu)
    n_fea = train_data.X.shape[1]
    h_all_agent = []
    w_all_agent = torch.empty((0, param_setting.n_rules, n_fea + 1)).double()

    for i in torch.arange(param_setting.n_agents):
        d_rules.update_rules(d_train_data[i].X, center_optimal)
        h_per_agent = compute_h_fc(d_train_data[i].X, d_rules)
        h_all_agent.append(h_per_agent)

        w_optimal_per_agent = fnn_tools.fnn_solve_r(h_per_agent, d_train_data[i].Y)
        w_all_agent = torch.cat((w_all_agent, w_optimal_per_agent.unsqueeze(0)), 0)

    w_optimal_all_agent, z, errors = fnn_tools.fnn_admm(d_train_data,
                                                        param_setting,
                                                        w_all_agent, h_all_agent)
    # calculate loss
    d_rules.consequent_list = z
    cfnn_loss = compute_loss_fc(test_data, d_rules, RMSELoss())
    loss_dlist.append(cfnn_loss)

data_save = dict()
data_save['loss_list'] = loss_list
data_save['loss_admm_list'] = loss_admm_list
data_save['loss_dlist'] = loss_dlist
data_save_dir = f"./results/{dataset_file}.pt"
torch.save(data_save, data_save_dir)

