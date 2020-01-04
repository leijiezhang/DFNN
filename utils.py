import torch
import skfuzzy
from rules import RuleBase, RuleFuzzyCmeans
from dataset import Dataset
from loss_function import LossFunc
import scipy.io as sio


def compute_h(x, rules: RuleBase):
    n_smpl = x.shape[0]
    n_fea = x.shape[1]
    n_rules = rules.n_rules
    mf_set = torch.zeros(n_rules, n_smpl, n_fea)
    for i in torch.arange(n_rules):
        for j in torch.arange(n_fea):
            mf = skfuzzy.membership.gaussmf(x[:, j], rules.center_list[i][j], rules.widths_list[i][j])
            mf_set[i, :, j] = mf

    w = torch.prod(mf_set, 2)
    w_hat = w / torch.sum(w, 0).repeat(n_rules, 1)
    w_hat[torch.isnan(w_hat)] = 1/n_rules

    h = torch.empty(0, n_smpl, n_fea + 1).double()
    for i in torch.arange(n_rules):
        w_hat_per_rule = w_hat[i, :].unsqueeze(1).repeat(1, n_fea + 1)
        x_extra = torch.cat((torch.ones(n_smpl, 1).double(), x), 1)
        h_per_rule = torch.mul(w_hat_per_rule, x_extra).unsqueeze(0)
        h = torch.cat((h, h_per_rule), 0)
    return h


def compute_h_fc(x, rules: RuleFuzzyCmeans):
    """
    todo: using fuzzy cmeans to get h
    :param x:
    :param rules:
    :return:
    """
    n_smpl = x.shape[0]
    n_fea = x.shape[1]
    n_rules = rules.n_rules

    w_hat = rules.data_partition.t()
    w_hat[torch.isnan(w_hat)] = 1/n_rules

    h = torch.empty(0, n_smpl, n_fea + 1).double()
    for i in torch.arange(n_rules):
        w_hat_per_rule = w_hat[i, :].unsqueeze(1).repeat(1, n_fea + 1)
        x_extra = torch.cat((torch.ones(n_smpl, 1).double(), x), 1)
        h_per_rule = torch.mul(w_hat_per_rule, x_extra).unsqueeze(0)
        h = torch.cat((h, h_per_rule), 0)
    return h


def compute_loss_fc(test_data: Dataset, rules_test: RuleBase, loss_function: LossFunc):
    """
    """
    # update rules on test data
    rules_test.update_rules(test_data.X, rules_test.center_list)
    h_test = compute_h_fc(test_data.X, rules_test)
    n_rule = h_test.shape[0]
    n_smpl = h_test.shape[1]
    n_fea = h_test.shape[2]
    h_cal = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

    # calculate Y hat
    y_hat = h_cal.mm(rules_test.consequent_list.reshape(test_data.Y.shape[1], n_rule * n_fea).t())
    loss = loss_function.forward(test_data.Y, y_hat)
    return loss


def compute_loss_k(test_data: Dataset, rules_test: RuleBase, loss_function: LossFunc):
    """
    """
    # update rules on test data
    rules_test.update_rules(test_data.X, rules_test.center_list)
    h_test = compute_h(test_data.X, rules_test)
    n_rule = h_test.shape[0]
    n_smpl = h_test.shape[1]
    n_fea = h_test.shape[2]
    h_cal = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

    # calculate Y hat
    y_hat = h_cal.mm(rules_test.consequent_list.reshape(test_data.Y.shape[1], n_rule * n_fea).t())
    loss = loss_function.forward(test_data.Y, y_hat)
    return loss


def dataset_parse(dataset_name):
    """
    todo: parse dataset from .mat to .pt
    :param dataset_name:
    :return:
    """
    dir_dataset = f"./datasets/{dataset_name}.mat"
    load_data = sio.loadmat(dir_dataset)
    dataset_name = load_data['name'][0]
    x_orig = load_data['X']
    y_orig = load_data['Y']
    x = torch.tensor(x_orig).double()
    y_tmp = []
    for i in torch.arange(len(y_orig)):
        y_tmp.append(float(y_orig[i]))
    y = torch.tensor(y_tmp).double()
    task = load_data['task'][0]
    data_save = dict()
    data_save['task'] = task
    data_save['name'] = dataset_name
    data_save['X'] = x
    if task == 'C':
        y_unique = torch.unique(y)
        y_c = torch.zeros(y.shape[0], y_unique.shape[0])
        for i in torch.arange(y_c.shape[1]):
            y_idx = torch.where(y == y_unique[i])
            y_c[y_idx[0], i] = 1
        data_save['Y_r'] = y
        data_save['Y'] = y_c
    else:
        data_save['Y'] = y.unsqueeze(1)
    dir_dataset = f"./datasets/{dataset_name}.pt"
    torch.save(data_save, dir_dataset)
