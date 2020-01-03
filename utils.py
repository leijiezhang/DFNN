import torch
import skfuzzy
from rules import RuleBase, RuleFuzzyCmeans
from dataset import Dataset
from loss_function import LossFunc


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


def compute_loss(test_data: Dataset, rules_test: RuleBase, loss_function: LossFunc):
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
    y_hat = h_cal.mm(rules_test.consequent_list.reshape(n_rule * n_fea, -1))
    loss = loss_function(test_data.Y, y_hat)
    return loss
