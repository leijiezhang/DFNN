from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from utils import load_data
from math_utils import mapminmax
import torch
import scipy.io as io
import os


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('ad_config')
para_list = torch.arange(-4, 5, 1).double()
para_list = torch.pow(10, para_list).double()

loss_list_train = []
loss_list_test = []
for i in torch.arange(len(param_config.dataset_list)):
    dataset_file = param_config.get_cur_dataset(int(i))
    # load dataset
    dataset = load_data(dataset_file, param_config.dataset_name)
    dataset.generate_n_partitions(param_config.n_run, param_config.patition_strategy)
    param_config.patition_strategy.set_current_folds(0)
    train_data, test_data = dataset.get_run_set()

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

    # normalize the dataset
    n_train_smpl = train_data.X.shape[0]
    x_all = torch.cat((train_data.X, test_data.X), 0)
    x_all_norm = mapminmax(x=x_all)
    train_data.X = x_all_norm[0:n_train_smpl, :]
    test_data.X = x_all_norm[n_train_smpl::, :]

    loss_list_train_item = []
    loss_list_test_item = []
    # ========Lasso回归========
    for para in para_list:
        model = Lasso(alpha=para)  # 调节alpha可以实现对拟合的程度
        # model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
        # model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
        model.fit(train_data.X, train_data.Y)  # 线性回归建模
        print('Lasso回归:\n', model.coef_)
        # print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
        # 使用模型预测
        predicted = model.predict(train_data.X)
        loss_tmp = mean_squared_error(train_data.Y, torch.tensor(predicted))
        loss_list_train_item.append(loss_tmp)
        print(f"Lasso regression tain: {loss_tmp}")

        predicted = model.predict(test_data.X)
        loss_tmp = mean_squared_error(test_data.Y, torch.tensor(predicted))
        loss_list_test_item.append(loss_tmp)
        print(f"Lasso regression test: {loss_tmp}")

    loss_list_test.append(loss_list_test_item)
    loss_list_train.append(loss_list_train_item)
loss_list_test = torch.tensor(loss_list_test).numpy()
loss_list_train = torch.tensor(loss_list_train).numpy()
save_dict = dict()
save_dict["loss_list_test"] = loss_list_test
save_dict["loss_list_train"] = loss_list_train
data_save_dir = f"./results/{param_config.dataset_name}"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/lasso.mat"
io.savemat(data_save_file, save_dict)
