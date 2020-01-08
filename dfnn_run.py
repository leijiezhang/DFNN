from param_config import ParamConfig
from partition import KFoldPartition
from dataset import Dataset
from kmeans_tools import KmeansUtils
from loss_utils import LossFunc, LossComputeBase
from rules import RuleBase
from h_utils import HBase
from fnn_utils import fnn_admm
from fnn_solver import FnnSolveBase
import torch


def fnn_main_c(train_data: Dataset, test_data: Dataset,
               n_rules, mu, loss_function: LossFunc,
               h_computer: HBase, fnn_solver: FnnSolveBase,
               loss_compute: LossComputeBase, rules: RuleBase):
    """
    todo: fnn method in centralized way
    :param train_data:
    :param test_data:
    :param n_rules:
    :param mu:
    :param loss_function:
    :param h_computer:
    :param fnn_solver:
    :param loss_compute:
    :param rules:
    :return:
    """
    rules_train = rules
    rules_train.fit(train_data.X, n_rules)
    h_train = h_computer.comute_h(train_data.X, rules_train)
    # run FNN solver for given rule number
    fnn_solver.h = h_train
    fnn_solver.y = train_data.Y.double()
    fnn_solver.para_mu = mu
    w_optimal = fnn_solver.solve()
    rules_train.consequent_list = w_optimal

    # compute train loss
    loss_compute.data = train_data
    loss_compute.rules = rules_train
    loss_compute.loss_function = loss_function
    loss_compute.h_util = h_computer
    train_loss = loss_compute.comute_loss()

    # compute test loss
    loss_compute.data = test_data
    loss_compute.rules = rules_train
    loss_compute.loss_function = loss_function
    loss_compute.h_util = h_computer
    test_loss = loss_compute.comute_loss()
    return train_loss, test_loss, rules_train


def dfnn_method(n_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy neural network
    :param param_config:
    :param dataset:
    :param n_rules:
    :return:
    """
    loss_c_train_tsr = []
    loss_c_test_tsr = []
    loss_d_train_tsr = []
    loss_d_test_tsr = []
    loss_curve_list = []
    param_config.n_rules = n_rules
    for k in torch.arange(param_config.kfolds):
        param_config.log.info(f"start traning at {k + 1}-fold!")

        param_config.patition_strategy.set_current_folds(k)
        train_data, test_data = dataset.get_fold_data()
        train_data.generate_distribute(KFoldPartition(param_config.n_agents))
        d_train_data = train_data.distribute_dataset()

        # trainning global method
        train_loss, test_loss, rules = fnn_main_c(train_data, test_data, param_config.n_rules,
                                                  param_config.para_mu_current, param_config.loss_fun,
                                                  param_config.h_computer, param_config.fnn_solver,
                                                  param_config.loss_compute, param_config.rules)
        loss_c_train_tsr.append(train_loss)
        loss_c_test_tsr.append(test_loss)
        param_config.log.info(f"loss of training data on centralized method: {train_loss}")
        param_config.log.info(f"loss of test data on centralized method: {test_loss}")

        # train distributed fnn
        kmeans_utils = KmeansUtils()
        center_optimal, errors = kmeans_utils.kmeans_admm(param_config.para_rho, d_train_data, param_config.n_rules,
                                                          param_config.n_agents, rules)
        loss_curve_list.append(errors)
        d_rules = rules

        n_fea = train_data.X.shape[1]
        h_all_agent = []
        # the shape of w set is n_agents *  n_output * n_rules * len_w
        w_all_agent = torch.empty((0, train_data.Y.shape[1],
                                   param_config.n_rules, n_fea + 1)).double()

        for i in torch.arange(param_config.n_agents):
            d_rules.update_rules(d_train_data[i].X, center_optimal)
            h_per_agent = param_config.h_computer.comute_h(d_train_data[i].X, d_rules)
            h_all_agent.append(h_per_agent)

            param_config.fnn_solver.h = h_per_agent
            param_config.fnn_solver.y = d_train_data[i].Y.double()
            param_config.fnn_solver.para_mu = param_config.para_mu_current
            w_optimal_per_agent = param_config.fnn_solver.solve()
            w_all_agent = torch.cat((w_all_agent, w_optimal_per_agent.unsqueeze(0)), 0)

        w_optimal_all_agent, z, errors = fnn_admm(d_train_data, param_config,
                                                  w_all_agent, h_all_agent)

        # calculate train loss
        d_rules.consequent_list = z
        param_config.loss_compute.data = train_data
        param_config.loss_compute.rules = d_rules
        param_config.loss_compute.loss_function = param_config.loss_fun
        param_config.loss_compute.h_util = param_config.h_computer
        cfnn_train_loss = param_config.loss_compute.comute_loss()
        param_config.log.info(f"loss of training data on distributed method: {cfnn_train_loss}")

        # calculate test loss
        d_rules.consequent_list = z
        param_config.loss_compute.data = test_data
        param_config.loss_compute.rules = d_rules
        param_config.loss_compute.loss_function = param_config.loss_fun
        param_config.loss_compute.h_util = param_config.h_computer
        cfnn_test_loss = param_config.loss_compute.comute_loss()
        param_config.log.info(f"loss of test data on distributed method: {cfnn_test_loss}")

        loss_d_train_tsr.append(cfnn_train_loss)
        loss_d_test_tsr.append(cfnn_test_loss)

    loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
    loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
    loss_d_train_tsr = torch.tensor(loss_d_train_tsr)
    loss_d_test_tsr = torch.tensor(loss_d_test_tsr)
    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr, loss_curve_list


def dfnn_ite_rules(max_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy neural network iterately
    :param max_rules:
    :param param_config:
    :param dataset:
    :return:
    """
    loss_c_train_tsr = torch.empty(0, param_config.kfolds).double()
    loss_c_test_tsr = torch.empty(0, param_config.kfolds).double()
    loss_d_train_tsr = torch.empty(0, param_config.kfolds).double()
    loss_d_test_tsr = torch.empty(0, param_config.kfolds).double()
    # n_max_rule * k_fold * len_curve
    loss_curve_list = []

    for i in torch.arange(max_rules):
        n_rules = int(i + 1)
        param_config.log.info(f"running at rule number: {n_rules}")

        loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_admm_list = \
            dfnn_method(n_rules, param_config, dataset)
        loss_c_train_tsr = torch.cat((loss_c_train_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_tsr = torch.cat((loss_c_test_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_tsr = torch.cat((loss_d_train_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_tsr = torch.cat((loss_d_test_tsr, loss_d_test.unsqueeze(0).double()), 0)
        loss_curve_list.append(loss_admm_list)

    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr, loss_curve_list


def dfnn_ite_rules_mu(max_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: consider all parameters in para_mu_list into algorithm
    :param max_rules:
    :param param_config:
    :param dataset:
    :return:
    """
    loss_c_train_mu_tsr = torch.empty(0, max_rules, param_config.kfolds).double()
    loss_c_test_mu_tsr = torch.empty(0, max_rules, param_config.kfolds).double()
    loss_d_train_mu_tsr = torch.empty(0, max_rules, param_config.kfolds).double()
    loss_d_test_mu_tsr = torch.empty(0, max_rules, param_config.kfolds).double()
    # n_para_list * n_max_rule * k_fold * len_curve
    loss_curve_list = []

    for i in torch.arange(param_config.para_mu_list.shape[0]):
        param_config.para_mu_current = param_config.para_mu_list[i]
        param_config.log.info(f"running param mu: {param_config.para_mu_current}")

        loss_c_train, loss_c_test, loss_d_train, loss_d_test, loss_admm_list = \
            dfnn_ite_rules(max_rules, param_config, dataset)
        loss_c_train_mu_tsr = torch.cat((loss_c_train_mu_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_mu_tsr = torch.cat((loss_c_test_mu_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_mu_tsr = torch.cat((loss_d_train_mu_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_mu_tsr = torch.cat((loss_d_test_mu_tsr, loss_d_test.unsqueeze(0).double()), 0)
        loss_curve_list.append(loss_admm_list)

    return loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr, loss_curve_list
