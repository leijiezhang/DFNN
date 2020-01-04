import torch
from utils import dataset_parse

dataset_list = ['blog', 'housing', 'strength', 'yacht']
for i in torch.arange(len(dataset_list)):
    dataset_name = dataset_list[i]

    dataset_parse(dataset_name)

print('lei')