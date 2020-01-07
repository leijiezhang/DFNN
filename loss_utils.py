import torch.nn as nn
import torch
from rules import RuleBase
from dataset import Dataset
from h_utils import HBase
import abc


class LossFunc(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    @abc.abstractmethod
    def forward(self, yhat, y):
        loss = []
        return loss


class RMSELoss(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        loss = torch.sqrt(torch.pow((yhat - y), 2).sum() / (y.shape[0] * y.var()))
        return loss


class MapLoss(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        yhat_idx = torch.argmax(yhat, 1)
        y_idx = torch.argmax(y, 1)
        acc_num = torch.where(yhat_idx == y_idx)[0].shape[0]
        loss = 1 - (acc_num / y_idx.shape[0])
        return loss


class LikelyLoss(LossFunc):
    """
    todo: used for calculate the loss of classification task with one output node
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y, yhat):
        yhat = torch.round(yhat)
        acc_num = torch.where(yhat == y)[0].shape[0]
        loss = 1 - (acc_num / y.shape[0])
        return loss


class LossComputeBase(object):
    def __init__(self):
        self.data: Dataset = None
        self.rules: RuleBase = None
        self.loss_function: LossFunc = None
        self.h_util: HBase = None

    @abc.abstractmethod
    def comute_loss(self):
        loss = []
        return loss


class LossComputeNormal(LossComputeBase):
    def __init__(self):
        super(LossComputeNormal, self).__init__()

    def comute_loss(self):
        # update rules on test data
        self.rules.update_rules(self.data.X, self.rules.center_list)
        h = self.h_util.comute_h(self.data.X, self.rules)
        n_rule = h.shape[0]
        n_smpl = h.shape[1]
        n_fea = h.shape[2]
        h_cal = h.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

        # calculate Y hat
        y_hat = h_cal.mm(self.rules.consequent_list.reshape(self.data.Y.shape[1],
                                                            n_rule * n_fea).t())
        loss = self.loss_function.forward(self.data.Y, y_hat)
        return loss


class LossComputeFuzzy(LossComputeBase):
    def __init__(self):
        super(LossComputeFuzzy, self).__init__()

    def comute_loss(self):
        """
        """
        # update rules on test data
        self.rules.update_rules(self.data.X, self.rules.center_list)
        h_test = self.h_util.comute_h(self.data.X, self.rules)
        n_rule = h_test.shape[0]
        n_smpl = h_test.shape[1]
        n_fea = h_test.shape[2]
        h_cal = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
        h_cal = h_cal.reshape(n_smpl, n_rule * n_fea)  # squess the last dimension

        # calculate Y hat
        y_hat = h_cal.mm(self.rules.consequent_list.reshape(self.data.Y.shape[1],
                                                                 n_rule * n_fea).t())
        loss = self.loss_function.forward(self.data.Y, y_hat)
        return loss


class FnnLossBase(object):
    def __init__(self):
        self.data: Dataset = None
        self.rules: RuleBase = None
        self.w: torch.Tensor = None
        self.loss_function: LossFunc = None
        self.h_compute: HBase = None

    @abc.abstractmethod
    def comute_loss(self):
        loss = []
        return loss


class FnnLossKmeans(FnnLossBase):
    def __init__(self):
        super(FnnLossKmeans, self).__init__()

    @abc.abstractmethod
    def comute_loss(self):
        h = self.h_compute.comute_h(self.data.X, self.rules)
        h = h.permute(1, 0, 2)
        h = h.reshape(h.shape[0], -1)
        y_hat = h.mm(self.w.unsqueeze(1))
        loss = self.loss_function(self.data.Y, y_hat)
        return loss


class FnnLossFuzzy(FnnLossBase):
    def __init__(self):
        super(FnnLossFuzzy, self).__init__()

    @abc.abstractmethod
    def comute_loss(self):
        h = self.h_compute.comute_h(self.data.X, self.rules)
        h = h.permute(1, 0, 2)
        h = h.reshape(h.shape[0], -1)
        y_hat = h.mm(self.w.unsqueeze(1))
        loss = self.loss_function(self.data.Y, y_hat)
        return loss
