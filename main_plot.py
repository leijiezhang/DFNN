import torch
from dataset import Result
import matplotlib.pyplot as plt
import torch.distributions as tdist


# dataset_file = 'eegDual_sub1'
# # dataset_file = 'CASP'
# data_save_dir = "./results/"
# data_save_dir = f"{data_save_dir}{dataset_file}_k_sigmoid.pt"
# results: Result = torch.load(data_save_dir)
# results.get_best_idx(2)
# loss_c_train = results.loss_c_train_best
# loss_d_train = results.loss_d_train_best
# acc_c_train = 1 - loss_c_train
# acc_d_train = 1 - loss_d_train
#
# loss_c_test = results.loss_c_test_best
# loss_d_test = results.loss_d_test_best
# acc_c_test = 1 - loss_c_test
# acc_d_test = 1 - loss_d_test
#
# x = torch.arange(loss_c_test.shape[0])
# loss_min_c_test = loss_c_test.min() * torch.ones(loss_c_test.shape[0])
# loss_min_d_test = loss_d_test.min() * torch.ones(loss_c_test.shape[0])
# acc_min_c_test = 1 - loss_min_c_test
# acc_min_d_test = 1 - loss_min_d_test
#
# fig, ax = plt.subplots()
# # ax.plot(x, loss_c_train, 'g-', label='loss on centralized training data')
# # ax.plot(x, loss_d_train, 'g:', label='loss on distributed training data')
# # ax.plot(x, loss_c_test, 'r-', label='loss on centralized test data')
# # ax.plot(x, loss_min_c_test, 'b-')
# # ax.plot(x, loss_d_test, 'r:', label='loss on distributed test data')
# # ax.plot(x, loss_min_d_test, 'b:')
# # ax.set(ylabel='loss', xlabel='number of rules', title=dataset_file)
# ax.plot(x, acc_c_train, 'g-', label='accuracy on centralized training data')
# ax.plot(x, acc_d_train, 'g:', label='accuracy on distributed training data')
# ax.plot(x, acc_c_test, 'r-', label='accuracy on centralized test data')
# ax.plot(x, acc_min_c_test, 'b-')
# ax.plot(x, acc_d_test, 'r:', label='accuracy on distributed test data')
# ax.plot(x, acc_min_d_test, 'b:')
# ax.set(ylabel='Acc', xlabel='number of rules', title=dataset_file)
# ax.legend()
# plt.show()

dataset_file = 'rules'
data_save_dir = "./results/"
data_save_dir = f"{data_save_dir}{dataset_file}.pt"
rules = torch.load(data_save_dir)
center_tmp = rules.center_list[0, 0]
width_tmp = rules.widths_list[0, 0]
y = tdist.Normal(center_tmp, width_tmp)
p1 = plt.figure()
x = torch.arange(center_tmp - width_tmp, center_tmp + width_tmp, 0.01)
p1.add_subplot(2, 1, 1)
plt.plot(y)
plt.legend(['lei'])

print('lei')