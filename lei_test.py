import torch
from rules import Rules
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


X = torch.randn(150, 6)
Y = torch.randn(50, 6)
a = torch.tensor(cdist(X, Y))
b = torch.argmax(a, 1)

rules = Rules(X, 50)
rules.x_rule_idx
kmeans = KMeans(5).fit(X)
kmeans.set_params
a = torch.arange(24)
b = a.view(2, 3, 4)
c = b.permute(1, 0, 2)