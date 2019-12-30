from param_config import ParamConfig
from partition import KFoldPartition
from dataset import Dataset
from loss_function import LossFunc, RMSELoss
from kmeans_tools import KmeansUtils
from utils import compute_h
from fnn_tools import FnnKmeansTools
from rules import RuleKmeans
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

loss_list = []
loss_dlist = []
loss_admm_list = []
for k in torch.arange(param_setting.kfolds):
    patition_strategy.set_current_folds(k)
    dataset.generate_n_partitions(param_setting.runs, patition_strategy)
    train_data, text_data = dataset.get_fold_data()
    train_data.generate_distribute(KFoldPartition(param_setting.n_agents))
    d_train_data = train_data.distribute_dataset()


    def fnn_main(data: Dataset, n_rules, mu, loss_function: LossFunc):
        rules = RuleKmeans()
        rules.fit(data.X, n_rules)
        h = compute_h(data.X, rules)
        # run FNN solver for given rule number
        fnn_tools = FnnKmeansTools(mu)
        w_optimal, y_hat = fnn_tools.fnn_solve_r(h, data.Y)
        loss = loss_function(data.Y, y_hat)

        return w_optimal, y_hat, loss, rules, h


    # trainning global method
    w_optimal, y_hat, loss, rules, h = \
        fnn_main(train_data, param_setting.n_rules, param_setting.para_mu, RMSELoss())

    loss_list.append(loss)
    # train distributed fnn
    kmeans_utils = KmeansUtils()
    center_optimal, errors = kmeans_utils.kmeans_admm(d_train_data, param_setting.n_rules,
                                                      param_setting.n_agents, RuleKmeans())
    loss_admm_list.append(errors)
    d_rules = RuleKmeans()
    d_rules.update_rules(train_data.X, center_optimal)

    fnn_tools = FnnKmeansTools(param_setting.para_mu)
    n_fea = train_data.X.shape[1]
    h_all_agent = []
    w_all_agent = torch.empty((0, param_setting.n_rules, n_fea + 1)).double()
    y_hat_all_agent = []
    for i in torch.arange(param_setting.n_agents):
        h_per_agent = compute_h(d_train_data[i].X, d_rules)
        h_all_agent.append(h_per_agent)

        w_optimal_per_agent, y_hat_per_agent = fnn_tools.fnn_solve_r(h_per_agent, d_train_data[i].Y)
        w_all_agent = torch.cat((w_all_agent, w_optimal_per_agent.unsqueeze(0)), 0)
        y_hat_all_agent.append(y_hat_per_agent)

    w_optimal_all_agent, z, errors = fnn_tools.fnn_admm(d_train_data,
                                                        param_setting,
                                                        w_all_agent, h_all_agent)

    d_rules.consequent_list = w_optimal_all_agent

    cfnn_loss = fnn_tools.fnn_loss(train_data, d_rules, z, RMSELoss())
    loss_dlist.append(cfnn_loss)

data_save = dict()
data_save['loss_list'] = loss_list
data_save['loss_admm_list'] = loss_admm_list
data_save['loss_dlist'] = loss_dlist
data_save_dir = f"./results/{dataset_file}.pt"
torch.save(data_save, data_save_dir)

