import abc
import torch


class FnnSolveBase(object):
    def __init__(self):
        self.para_mu = None
        self.h: torch.Tensor = None
        self.y: torch.Tensor = None

    @abc.abstractmethod
    def solve(self):
        w_optimal = []
        return w_optimal


class FnnSolveReg(FnnSolveBase):
    def __init__(self):
        super(FnnSolveReg, self).__init__()

    def solve(self):
        """
        todo: fnn solver for regression problem
        """
        n_rule = self.h.shape[0]
        n_smpl = self.h.shape[1]
        n_fea = self.h.shape[2]
        h_cal = self.h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension
        w_comb_optimal = torch.inverse(h_cal.t().mm(h_cal) +
                                       self.para_mu * torch.eye(n_rule * n_fea)).mm(h_cal.t().mm(self.y))
        w_comb_optimal = w_comb_optimal.permute((1, 0))
        w_optimal = w_comb_optimal.reshape(self.y.shape[1], n_rule, n_fea)

        return w_optimal


class FnnSolveCls(FnnSolveBase):
    def __init__(self):
        super(FnnSolveCls, self).__init__()

    def solve(self):
        """
        todo: fnn solver for classification problem
        """
        n_rule = self.h.shape[0]
        n_smpl = self.h.shape[1]
        n_fea = self.h.shape[2]
        h_cal = self.h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

        s = torch.ones(n_smpl)
        z = torch.ones(n_smpl)
        w_temp = torch.zeros(n_rule * n_fea)

        sh_cal = 0.001  # initiate the threshold
        w_loss = 1  # initiate the loss of W
        w_loss_list = []
        w_optimal = None
        while w_loss > sh_cal:
            w_old = w_temp.clone()

            w_temp = torch.inverse(h_cal.t().mm(torch.diag(s)).mm(h_cal) + self.para_mu *
                                   torch.eye(n_rule * n_fea)).mm(h_cal.t().mm(torch.diag(s)).mm(z))
            mu_cal = torch.sigmoid(h_cal.mm(w_temp))
            s = torch.mul(mu_cal, (torch.ones(n_smpl) - mu_cal))
            z = h_cal.mm(w_temp) + (self.y - mu_cal) / s

            w_loss = torch.norm(w_temp, w_old)
            w_loss_list.append(w_loss)

            w_optimal = w_temp

        return w_optimal
