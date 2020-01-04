import torch
from utils import dataset_parse

dataset_list = ['abalone', 'airfoil', 'bikesharing',
                             'blog', 'bodyfat', 'CASP', 'CCPP', 'eegDual_sub1_format',
                             'housing', 'HRSS_anomalous_optimized', 'HRSS_anomalous_standard',
                             'kc_house', 'motor_temperature', 'quake', 'skills',
                             'strength', 'telemonitoring', 'yacht']
for i in torch.arange(len(dataset_list)):
    dataset_name = dataset_list[i]

    dataset_parse(dataset_name)

print('lei')