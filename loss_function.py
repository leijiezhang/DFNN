import torch.nn as nn
import abc
import torch


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

    def forward(self, yhat, y):
        loss = torch.sqrt(torch.norm(yhat - y) / (y.shape[0] * yhat.var()))
        return loss


class MapLoss(LossFunc):
    def __init__(self, eps=1e-6):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        yhat_idx = torch.argmax(yhat, 1)
        y_idx = torch.argmax(y, 1)
        acc_num = torch.where(yhat_idx == y_idx)[0].shape[0]
        loss = 1 - (acc_num / y_idx.shape[0])
        return loss
