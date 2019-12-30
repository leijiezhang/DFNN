import torch
from rules import RuleBase
from dataset import Dataset
from typing import List


class KmeansUtils(object):
    def __init__(self):
        useless = 0

    def kmeans_admm(self, data: List[Dataset], n_rules, n_agents, rules: RuleBase):
        # rules = self.rules_kmeans(data.X, n_rules)
        # calculate H matrix and w for each node
        # parameters initialize
        rho = 1
        max_steps = 300
        admm_reltol = 0.001
        admm_abstol = 0.001

        errors = torch.zeros(max_steps, 1)
        # initiate lagrange multiplier
        lagrange_mul = torch.zeros(n_agents, n_rules, data[1].X.shape[1])
        # initiate global term centrio mu
        rules.fit(data[1].X, n_rules)
        center_global = rules.center_list
        # initiate global center set
        center_agent_set = torch.zeros((n_agents, center_global.shape[0], center_global.shape[1]))

        for i in torch.arange(max_steps):

            for j in torch.arange(n_agents):
                # assign clusters for each node based on the global centoids
                # ye's method
                # dist_x = center_global.centerl(center_global) / 2 - center_global.mm(data[j].X.t())
                # labels = torch.min(dist_x, 0)[1]

                labels = rules.x_center_idx
                labels_matrix = torch.zeros(n_rules, data[j].X.shape[0])
                for k in torch.arange(data[j].X.shape[0]):
                    labels_matrix[labels[k].long(), k] = 1
                labels_matrix = torch.nn.functional.normalize(labels_matrix, 2, 1)

                center_agent_set[j, :, :] = labels_matrix.double().mm(data[j].X)

            # store the old global centrio and update
            center_global_old = center_global.clone()

            # for each cluster
            for j in torch.arange(n_rules):
                center_global[j, :] = (rho * center_agent_set[:, j, :].sum(0) +\
                                   lagrange_mul[:, j, :].sum(0)) / (rho * n_agents)

            # compute the update for the Lagrange Mltipliers
            for j in torch.arange(n_rules):
                for k in torch.arange(n_agents):
                    lagrange_mul[k, j, :] = lagrange_mul[k, j, :] +\
                        rho * (center_agent_set[k, j, :] - center_global[j, :])
            # check stoping criterion
            stop_crtn = - rho * (center_global - center_global_old)

            errors[i] = torch.norm(stop_crtn)

            if errors[i] < torch.sqrt(torch.tensor(n_agents).double()) * admm_abstol:
                break
        center_optimal = center_global

        return center_optimal, errors



