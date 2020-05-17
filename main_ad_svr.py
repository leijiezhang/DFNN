from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from sklearn.svm import SVR
from utils import load_data
from math_utils import mapminmax
from dataset import Dataset
import torch
import time
import scipy.io as io
import os
from sklearn.metrics import mean_squared_error


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('ad_config')
para_list = torch.arange(-4, 5, 1).double()
para_list = torch.pow(10, para_list).double()

loss_rbf_list_train = []
loss_rbf_list_test = []
loss_lin_list_train = []
loss_lin_list_test = []

for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file, param_config.dataset_name)
    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)

    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)
    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()
        dataset.Y = mapminmax(dataset.Y)

    loss_rbf_list_train_item = []
    loss_rbf_list_test_item = []
    loss_lin_list_train_item = []
    loss_lin_list_test_item = []

    # ========svr========
    for C in para_list:
        for gamma in para_list:
            
            loss_train_rbf_tsr = []
            loss_test_rbf_tsr = []
            loss_train_lin_tsr = []
            loss_test_lin_tsr = []
            for k in torch.arange(param_config.n_kfolds):
                param_config.patition_strategy.set_current_folds(k)
                train_data, test_data = dataset.get_run_set()
                # if the dataset is like a eeg data, which has trails hold sample blocks
                if dataset.X.shape.__len__() == 3:
                    # reform training dataset
                    y = torch.empty(0, 1).double()
                    x = torch.empty(0, dataset.X.shape[2]).double()
                    for ii in torch.arange(train_data.Y.shape[0]):
                        x = torch.cat((x, train_data.X[ii]), 0)
                        size_smpl_ii = train_data.X[ii].shape[0]
                        y_tmp = train_data.Y[ii].repeat(size_smpl_ii, 1)
                        y = torch.cat((y, y_tmp), 0)
                    train_data = Dataset(train_data.name, x, y, train_data.task)

                    # reform test dataset
                    y = torch.empty(0, 1).double()
                    x = torch.empty(0, dataset.X.shape[2]).double()
                    for ii in torch.arange(test_data.Y.shape[0]):
                        x = torch.cat((x, test_data.X[ii]), 0)
                        size_smpl_ii = test_data.X[ii].shape[0]
                        y_tmp = test_data.Y[ii].repeat(size_smpl_ii, 1)
                        y = torch.cat((y, y_tmp), 0)
                    test_data = Dataset(test_data.name, x, y, test_data.task)

                # normalize the dataset
                n_train_smpl = train_data.X.shape[0]
                x_all = torch.cat((train_data.X, test_data.X), 0)
                x_all_norm = mapminmax(x=x_all)
                train_data.X = x_all_norm[0:n_train_smpl, :]
                test_data.X = x_all_norm[n_train_smpl::, :]

                param_config.log.info(f"start traning at {param_config.patition_strategy.current_fold + 1}-fold!")

                # ========rbf========
                param_config.log.info(f"svm rbf running param gamma: {param_config.para_mu_current}")
                param_config.log.info(f"svm rbf running param c: {param_config.para_mu1_current}")
                svr = SVR(kernel='rbf', gamma=gamma, C=C)
                # 记录训练时间
                t0 = time.time()
                # 训练
                svr.fit(train_data.X.numpy(), train_data.Y.numpy().ravel())
                svr_fit = time.time() - t0
                train_label_hat = svr.predict(train_data.X.numpy())
                test_label_hat = svr.predict(test_data.X.numpy())
                train_acc = mean_squared_error(train_data.Y.squeeze(), torch.tensor(train_label_hat))
                test_acc = mean_squared_error(test_data.Y.squeeze(), torch.tensor(test_label_hat))
                if test_data.task == 'C':
                    param_config.log.info(f"Accuracy of training data using SVM: {train_acc}")
                    param_config.log.info(f"Accuracy of test data using SVM: {test_acc}")
                else:
                    param_config.log.info(f"loss of training data using SVM: {train_acc}")
                    param_config.log.info(f"loss of test data using SVM: {test_acc}")
                loss_train_rbf_tsr.append(train_acc)
                loss_test_rbf_tsr.append(test_acc)

                if C == para_list[0]:
                    # ========lin========
                    param_config.log.info(f"svm lin running param gamma: {param_config.para_mu_current}")
                    param_config.log.info(f"svm lin running param c: {param_config.para_mu1_current}")
                    svr = SVR(kernel='linear', C=C)
                    # 记录训练时间
                    t0 = time.time()
                    # 训练
                    svr.fit(train_data.X.numpy(), train_data.Y.numpy().ravel())
                    svr_fit = time.time() - t0
                    train_label_hat = svr.predict(train_data.X.numpy())
                    test_label_hat = svr.predict(test_data.X.numpy())
                    train_acc = mean_squared_error(train_data.Y.squeeze(), torch.tensor(train_label_hat))
                    test_acc = mean_squared_error(test_data.Y.squeeze(), torch.tensor(test_label_hat))
                    if test_data.task == 'C':
                        param_config.log.info(f"Accuracy of training data using SVM: {train_acc}")
                        param_config.log.info(f"Accuracy of test data using SVM: {test_acc}")
                    else:
                        param_config.log.info(f"loss of training data using SVM: {train_acc}")
                        param_config.log.info(f"loss of test data using SVM: {test_acc}")
                    loss_train_lin_tsr.append(train_acc)
                    loss_test_lin_tsr.append(test_acc)
                
        loss_rbf_list_train_item.append(loss_train_rbf_tsr)
        loss_rbf_list_test_item.append(loss_test_rbf_tsr)
            
        loss_lin_list_train_item.append(loss_train_lin_tsr)
        loss_lin_list_test_item.append(loss_test_lin_tsr)

loss_rbf_list_test = torch.tensor(loss_rbf_list_train_item).numpy()
loss_rbf_list_train = torch.tensor(loss_rbf_list_test_item).numpy()
loss_lin_list_test = torch.tensor(loss_lin_list_train_item).numpy()
loss_lin_list_train = torch.tensor(loss_lin_list_test_item).numpy()
save_dict = dict()
save_dict["loss_rbf_list_test"] = loss_rbf_list_test
save_dict["loss_rbf_list_train"] = loss_rbf_list_train
save_dict["loss_lin_list_test"] = loss_lin_list_test
save_dict["loss_lin_list_train"] = loss_lin_list_train
data_save_dir = f"./results/{param_config.dataset_name}"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/lasso.mat"
io.savemat(data_save_file, save_dict)
