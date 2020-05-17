from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from dfnn_run import hdfnn_run
from sklearn.metrics import mean_squared_error
from utils import load_data
from math_utils import mapminmax
from sklearn.svm import SVR
import torch


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('ad_config')

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

    # ========Lasso回归========
    model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
    # model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
    # model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
    model.fit(train_data.X, train_data.Y)  # 线性回归建模
    # print('系数矩阵:\n', model.coef_)
    # print('线性回归模型:\n', model)
    # print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
    # 使用模型预测
    predicted = model.predict(train_data.X)
    loss_tmp = mean_squared_error(train_data.Y, torch.tensor(predicted))
    param_config.log.info(f"Lasso regression tain: {loss_tmp}")

    predicted = model.predict(test_data.X)
    loss_tmp = mean_squared_error(test_data.Y, torch.tensor(predicted))
    param_config.log.info(f"Lasso regression test: {loss_tmp}")

    lr = LinearRegression()
    lr.fit(train_data.X, train_data.Y)  # 训练
    # print("预测的决定系数R平方:", lr.score(train_data.X, train_data.Y))
    # print("线性回归的估计系数:", lr.coef_)  # 打印线性回归的估计系数
    # print("线性模型的独立项:", lr.intercept_)  # 打印线性模型的独立项

    y_predict = lr.predict(train_data.X)  # 预测

    loss_tmp = mean_squared_error(train_data.Y, torch.tensor(y_predict))
    param_config.log.info(f"linear regression train: {loss_tmp}")

    y_predict = lr.predict(test_data.X)  # 预测

    loss_tmp = mean_squared_error(test_data.Y, torch.tensor(y_predict))
    param_config.log.info(f"linear regression test: {loss_tmp}")

    # ========岭回归========
    model = Ridge(alpha=0.5)
    # model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
    model.fit(train_data.X, train_data.Y)  # 线性回归建模
    # print('系数矩阵:\n', model.coef_)
    # print('线性回归模型:\n', model)
    # print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
    # 使用模型预测
    predicted = model.predict(train_data.X)
    loss_tmp = mean_squared_error(train_data.Y, torch.tensor(predicted))
    param_config.log.info(f"Ridge regression train: {loss_tmp}")
    predicted = model.predict(test_data.X)
    loss_tmp = mean_squared_error(test_data.Y, torch.tensor(predicted))
    param_config.log.info(f"Ridge regression test: {loss_tmp}")
    # ========polynomial回归========
    polynomial = PolynomialFeatures(degree=3)  # 二次多项式
    x_transformed = polynomial.fit_transform(train_data.X)  # x每个数据对应的多项式系数

    poly_linear_model = LinearRegression()  # 创建回归器
    poly_linear_model.fit(x_transformed, train_data.Y)  # 训练数据

    yy = poly_linear_model.predict(x_transformed)
    loss_tmp = mean_squared_error(train_data.Y, torch.tensor(yy))
    param_config.log.info(f"polynomial regression train: {loss_tmp}")

    xx_transformed = polynomial.fit_transform(test_data.X)  # 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
    yy = poly_linear_model.predict(xx_transformed)
    loss_tmp = mean_squared_error(test_data.Y, torch.tensor(yy))
    param_config.log.info(f"polynomial regression test: {loss_tmp}")

    # ===============rbf svr===========
    param_config.log.info(f"svm rbf running param gamma: {param_config.para_mu_current}")
    param_config.log.info(f"svm rbf running param c: {param_config.para_mu1_current}")
    svr = SVR(kernel='rbf', gamma=0.01, C=0.01)
    # 训练
    svr.fit(train_data.X.numpy(), train_data.Y.numpy().ravel())
    train_label_hat = svr.predict(train_data.X.numpy())
    test_label_hat = svr.predict(test_data.X.numpy())
    train_acc = mean_squared_error(train_data.Y.squeeze(), torch.tensor(train_label_hat))
    test_acc = mean_squared_error(test_data.Y.squeeze(), torch.tensor(test_label_hat))
    param_config.log.info(f"loss of training data using rbf SVR: {train_acc}")
    param_config.log.info(f"loss of test data using rbf SVR: {test_acc}")

    # ===============rbf svr===========
    param_config.log.info(f"svm lin running param c: {param_config.para_mu1_current}")
    svr = SVR(kernel='linear', C=0.01)
    # 训练
    svr.fit(train_data.X.numpy(), train_data.Y.numpy().ravel())
    train_label_hat = svr.predict(train_data.X.numpy())
    test_label_hat = svr.predict(test_data.X.numpy())
    train_acc = mean_squared_error(train_data.Y.squeeze(), torch.tensor(train_label_hat))
    test_acc = mean_squared_error(test_data.Y.squeeze(), torch.tensor(test_label_hat))
    param_config.log.info(f"loss of training data using lin SVR: {train_acc}")
    param_config.log.info(f"loss of test data using lin SVR: {test_acc}")

    # ===============hdfnn===========
    train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss = \
        hdfnn_run(param_config, train_data, test_data)
