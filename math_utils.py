import torch


def cal_fc_w(x: torch.Tensor, y: torch.Tensor, para_mu):
    """
    compute W = (Y^T*Y + mu* I) * Y^T*y
    :param x:
    :param y:
    :param para_mu:
    :return:
    """
    n_fea = x.shape[1]
    w = torch.inverse(x.t().mm(x) + para_mu * torch.eye(n_fea).double()).mm(x.t().mm(y))
    return w
