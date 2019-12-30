import torch
from param_config import ParamConfig
from utils import compute_h
from loss_function import LossFunc
from dataset import Dataset
from rules import RuleBase
from typing import List


class FnnKmeansTools(object):
    def __init__(self, para_mu):
        self.para_mu = para_mu

    def fnn_solve_r(self, h: torch.Tensor, y):
        """
        todo: fnn solver for regression problem
        :param self:
        :param h: Hessian matrix
        :param y: label data
        :return:
        """
        n_rule = h.shape[0]
        n_smpl = h.shape[1]
        n_fea = h.shape[2]
        h_cal = h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension
        w_comb_optimal = torch.inverse(h_cal.t().mm(h_cal) +
                                      self.para_mu * torch.eye(n_rule * n_fea)).mm(h_cal.t().mm(y))
        w_optimal = w_comb_optimal.reshape(n_rule, n_fea)

        # calculate Y hat
        y_hat = h_cal.mm(w_comb_optimal)
        return w_optimal, y_hat

    def fnn_solve_c(self, h: torch.Tensor, y):
        """
        todo: fnn solver for classification problem
        :param self:
        :param h: Hessian matrix
        :param y: label data
        :return:
        """
        n_rule = h.shape[0]
        n_smpl = h.shape[1]
        n_fea = h.shape[2]
        h_cal = h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

        s = torch.ones(n_smpl)
        z = torch.ones(n_smpl)
        w_temp = torch.zeros(n_rule * n_fea)

        sh_cal = 0.001  # initiate the threshold
        w_loss = 1      # initiate the loss of W
        w_loss_list = []
        w_optimal = None
        while w_loss > sh_cal:
            w_old = w_temp.clone()

            w_temp = torch.inverse(h_cal.t().mm(torch.diag(s)).mm(h_cal) + self.para_mu *
                                   torch.eye(n_rule * n_fea)).mm(h_cal.t().mm(torch.diag(s)).mm(z))
            mu_cal = torch.sigmoid(h_cal.mm(w_temp))
            s = torch.mul(mu_cal, (torch.ones(n_smpl) - mu_cal))
            z = h_cal.mm(w_temp) + (y - mu_cal) / s

            w_loss = torch.norm(w_temp, w_old)
            w_loss_list.append(w_loss)

            w_optimal = w_temp

        # calculate Y hat
        y_hat = h_cal.mm(w_optimal)
        return w_optimal, y_hat, w_loss_list

    def fnn_centralized(self, x, y, max_n_rules, loss_functions: LossFunc):
        # try all possible rule methods
        rules_set = []
        h_set = []
        y_hat_set = []
        w_optimal_tmp_set = []
        loss_set = torch.zeros(max_n_rules)
        for i in torch.arange(max_n_rules):
            ruls_tmp = RuleBase()
            ruls_tmp.fit(x, i)
            rules_set.append(ruls_tmp)
            h_tmp = compute_h(x, ruls_tmp)
            h_set.append(h_tmp)

            # run FNN solver for all each rule number
            w_optimal_tmp_tmp, y_hat_tmp = self.fnn_solve_r(h_tmp, y)
            w_optimal_tmp_set.append(w_optimal_tmp_tmp)
            y_hat_set.append(y_hat_tmp)
            loss_set[i] = loss_functions.forward(y_hat_tmp, y)

        min_loss = loss_set.min(0)
        min_idx = int(min_loss[1])
        n_rules = min_idx
        rules = rules_set[min_idx]
        y_hat = y_hat_set[min_idx]
        h_best = h_set[min_idx]
        w_optimal_tmp = w_optimal_tmp_set[min_idx]
        # w is a column vector
        for i in torch.arange(n_rules):
            rules[i].consequent = w_optimal_tmp[i, :]

        return y_hat, min_loss, loss_set, rules, n_rules, h_best, w_optimal_tmp

    @staticmethod
    def fnn_admm(d_train_data: List[Dataset], param_setting: ParamConfig, w, h):
        # parameters initialize
        rho = 1
        max_steps = 300
        admm_reltol = 0.001
        admm_abstol = 0.001

        param_mu = param_setting.para_mu
        n_node = param_setting.n_agents
        n_rules = param_setting.n_rules
        n_fea = d_train_data[1].X.shape[1]

        len_w = n_rules * (n_fea + 1)

        errors = torch.zeros(max_steps, n_node)
        
        z = torch.zeros(len_w)
        lagrange_mul = torch.zeros(n_node, len_w)

        # precompute the matrices
        h_inv = torch.zeros(n_node, len_w, len_w)
        h_y = torch.zeros(n_node, len_w)

        for i in torch.arange(n_node):
            h_tmp = h[i]
            n_smpl = h_tmp.shape[1]
            h_cal = h_tmp.permute((1, 0, 2))  # N * n_rules * (d + 1)
            h_cal = h_cal.reshape(n_smpl, len_w)
            h_inv[i, :, :] = torch.inverse(torch.eye(len_w) * rho + h_cal.t().mm(h_cal))
            h_y[i, :] = h_cal.t().mm(d_train_data[i].Y).t()

        for i in torch.arange(max_steps):
            for j in torch.arange(n_node):
                w_cal = w.reshape(n_node, len_w)
                w_cal[j, :] = h_inv[j, :, :].double().mm((h_y[j, :].double() +
                                                          rho * z - lagrange_mul[j, :]).unsqueeze(1)).squeeze()

            # store the old z while update it
            z_old = z.clone()
            z = (rho * torch.sum(w_cal, 0) + torch.sum(lagrange_mul, 0)) / (param_mu + rho * n_node)

            # compute the update for the lagrangian multipliers
            for j in torch.arange(n_node):
                lagrange_mul[j, :] = lagrange_mul[j, :] + rho * (w_cal[j, :] - z)

            # check stopping criterion
            z_norm = rho * (z - z_old)
            lagrange_mul_norm = torch.zeros(n_node)

            primal_criterion = torch.zeros(n_node)

            for j in torch.arange(n_node):
                w_error = w_cal[j, :] - z
                errors[i, j] = torch.norm(w_error)
                if errors[i, j] < torch.sqrt(torch.tensor(n_node).float()) * admm_abstol + \
                        admm_reltol * torch.max(torch.norm(w[j, :], 2), torch.norm(z, 2)):
                    primal_criterion[j] = 1
                lagrange_mul_norm[j] = torch.norm(lagrange_mul[j, :], 2)

            if torch.norm(z_norm) < torch.sqrt(torch.tensor(n_node).float()) * admm_abstol + admm_reltol * \
                    lagrange_mul_norm.max() and primal_criterion.max() == 1:
                break
        # w_cal = w_cal.reshape(n_node, n_rules, (n_fea + 1))
        return w_cal, z, errors

    @staticmethod
    def fnn_loss(data: Dataset, rules: RuleBase, w, loss_function: LossFunc):
        h = compute_h(data.X, rules)
        h = h.permute(1, 0, 2)
        h = h.reshape(h.shape[0], -1)
        y_hat = h.mm(w.unsqueeze(1))
        loss = loss_function(data.Y, y_hat)
        return loss
