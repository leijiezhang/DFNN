import torch
from utils import dataset_parse
from dataset import Result

dataset_file = 'eegDual_sub1'
# dataset_file = 'CASP'
data_save_dir = "./results/"
data_save_file = f"{data_save_dir}{dataset_file}_k.pt"
data_load = torch.load(data_save_file)



torch.save(results, data_save_file)
loss_g_mean = data_load['loss_g_tsr'].mean(2)
loss_g_mean_best = loss_g_mean[data_load['best_idx'], :]

loss_d_mean = data_load['loss_d_tsr'].mean(2)
loss_d_mean_best = loss_d_mean[data_load['best_idx'], :]

loss_curv_mean_best = data_load['loss_curve_list'][data_load['best_idx']][1]

print('lei')