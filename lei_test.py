import torch
from rules import RuleBase
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist



dataset_file = 'CASP_5'
data_save_dir = f"./results/{dataset_file}.pt"
data_5 = torch.load(data_save_dir)
dataset_file = 'CASP_10'
data_save_dir = f"./results/{dataset_file}.pt"
data_10 = torch.load(data_save_dir)
print('lei')