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
        y_unique = torch.unique(y)
        yhat_cal = y.clone()
        for i in torch.arange(yhat.shape[1]):
            y_idx = torch.where(yhat_idx == i)
            yhat_cal[y_idx] = y_unique[i]
        loss = 1 - (torch.sum(torch.where(yhat_cal == y)) / yhat.shape[0])
        return loss
