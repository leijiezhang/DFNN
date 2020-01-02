import torch


dataset_file = 'CASP'
data_save_dir = f"./results/{dataset_file}.pt"
data = torch.load(data_save_dir)
print('lei')