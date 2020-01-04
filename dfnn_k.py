from param_config import ParamConfig
from partition import KFoldPartition
from dataset import Dataset
from loss_function import LossFunc
from kmeans_tools import KmeansUtils
from utils import compute_h, compute_loss_k
from fnn_tools import FnnKmeansTools
from rules import RuleKmeans
import torch


def dfnn_k_method(n_rules, param_setting: ParamConfig,
                   patition_strategy: KFoldPartition,
                   dataset: Dataset, loss_func: LossFunc):
    """
    todo: this is the method for distribute fuzzy neural network using normal kmeans
    :param param_setting:
    :param patition_strategy:
    :param dataset:
    :param n_rules:
    :param loss_func:
    :return:
    """
    loss_list = []
    loss_dlist = []
    loss_admm_list = []
    param_setting.n_rules = n_rules
    for k in torch.arange(param_setting.kfolds):
        patition_strategy.set_current_folds(k)
        train_data, test_data = dataset.get_fold_data()
        train_data.generate_distribute(KFoldPartition(param_setting.n_agents))
        d_train_data = train_data.distribute_dataset()

        def fnn_main(train_data: Dataset, test_data: Dataset, n_rules, mu, loss_function: LossFunc):
            rules_train = RuleKmeans()
            rules_train.fit(train_data.X, n_rules)
            h_train = compute_h(train_data.X, rules_train)
            # run FNN solver for given rule number
            fnn_tools = FnnKmeansTools(mu)
            w_optimal = fnn_tools.fnn_solve_r(h_train, train_data.Y.double())
            rules_train.consequent_list = w_optimal

            # compute loss
            loss = compute_loss_k(test_data, rules_train, loss_function)
            return loss, rules_train

        # trainning global method
        loss, rules = fnn_main(train_data, test_data, param_setting.n_rules,
                               param_setting.para_mu, loss_func)
        loss_list.append(loss)

        # train distributed fnn
        kmeans_utils = KmeansUtils()
        center_optimal, errors = kmeans_utils.kmeans_admm(d_train_data, param_setting.n_rules,
                                                          param_setting.n_agents, RuleKmeans())
        loss_admm_list.append(errors)
        d_rules = RuleKmeans()

        fnn_tools = FnnKmeansTools(param_setting.para_mu)
        n_fea = train_data.X.shape[1]
        h_all_agent = []
        # the shape of w set is n_agents *  n_output * n_rules * len_w
        w_all_agent = torch.empty((0, train_data.Y.shape[1],
                                   param_setting.n_rules, n_fea + 1)).double()

        for i in torch.arange(param_setting.n_agents):
            d_rules.update_rules(d_train_data[i].X, center_optimal)
            h_per_agent = compute_h(d_train_data[i].X, d_rules)
            h_all_agent.append(h_per_agent)

            w_optimal_per_agent = fnn_tools.fnn_solve_r(h_per_agent, d_train_data[i].Y.double())
            w_all_agent = torch.cat((w_all_agent, w_optimal_per_agent.unsqueeze(0)), 0)

        w_optimal_all_agent, z, errors = fnn_tools.fnn_admm(d_train_data,
                                                            param_setting,
                                                            w_all_agent, h_all_agent)
        # calculate loss
        d_rules.consequent_list = z
        cfnn_loss = compute_loss_k(test_data, d_rules, loss_func)
        loss_dlist.append(cfnn_loss)

    loss_list = torch.tensor(loss_list)
    loss_dlist = torch.tensor(loss_dlist)
    return loss_list, loss_dlist, loss_admm_list


def dfnn_k_ite_rules(max_rules, param_setting: ParamConfig, patition_strategy: KFoldPartition, dataset: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy neural network iterately
    :param max_rules:
    :param param_setting:
    :param patition_strategy:
    :param dataset:
    :return:
    """
    loss_list = torch.empty(0, dataset.n_fea + 1).double()
    loss_dlist = torch.empty(0, dataset.n_fea + 1).double()
    loss_admm_list = []

    for i in torch.arange(max_rules):
        n_rule = int(i + 1)
        loss_list_temp, loss_dlist_temp, loss_admm_list_temp = \
            dfnn_k_method(n_rule, param_setting, patition_strategy, dataset)
        loss_list = torch.cat((loss_list, loss_list_temp.unsqueeze(0)), 0)
        loss_dlist = torch.cat((loss_dlist, loss_dlist_temp.unsqueeze(0)), 0)
        loss_admm_list.append(loss_admm_list_temp)

    return loss_list, loss_dlist, loss_admm_list
