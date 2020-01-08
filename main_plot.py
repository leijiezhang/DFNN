import torch
from dataset import Result
import matplotlib.pyplot as plt


dataset_file = 'eegDual_sub1'
# dataset_file = 'CASP'
data_save_dir = "./results/"
data_save_dir = f"{data_save_dir}{dataset_file}_k.pt"
results: Result = torch.load(data_save_dir)
loss_c_train = results.loss_c_train_best
loss_d_train = results.loss_d_train_best

loss_c_test = results.loss_c_test_best
loss_d_test = results.loss_d_test_best

x = torch.arange(loss_c_test.shape[0])
loss_min_c_test = loss_c_test.min() * torch.ones(loss_c_test.shape[0])
loss_min_d_test = loss_d_test.min() * torch.ones(loss_c_test.shape[0])

fig, ax = plt.subplots()
ax.plot(x, loss_c_train, 'g-', label='loss on centralized training data')
ax.plot(x, loss_d_train, 'g:', label='loss on distributed training data')
ax.plot(x, loss_c_test, 'r-', label='loss on centralized test data')
ax.plot(x, loss_min_c_test, 'b-')
ax.plot(x, loss_d_test, 'r:', label='loss on distributed test data')
ax.plot(x, loss_min_d_test, 'b:')
ax.set(ylabel='loss', xlabel='number of rules', title=dataset_file)
ax.legend()
plt.show()
print('lei')