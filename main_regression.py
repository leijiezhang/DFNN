from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from dfnn_run import dfnn_kfolds
from sklearn.metrics import mean_squared_error
from utils import load_data
from math_utils import mapminmax
import torch


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('ethylene_co_config')

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
    print('系数矩阵:\n', model.coef_)
    print('线性回归模型:\n', model)
    # print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
    # 使用模型预测
    predicted = model.predict(test_data.X)
    loss_tmp = mean_squared_error(test_data.Y, torch.tensor(predicted))
    print(f"Lasso regression: {loss_tmp}")

    # lr = LinearRegression()
    # lr.fit(train_data.X, train_data.Y)  # 训练
    # print("预测的决定系数R平方:", lr.score(train_data.X, train_data.Y))
    # print("线性回归的估计系数:", lr.coef_)  # 打印线性回归的估计系数
    # print("线性模型的独立项:", lr.intercept_)  # 打印线性模型的独立项
    #
    # y_predict = lr.predict(test_data.X)  # 预测
    #
    # loss_tmp = mean_squared_error(test_data.Y, torch.tensor(y_predict))
    # print(f"linear regression: {loss_tmp}")

    # # ========岭回归========
    # model = Ridge(alpha=0.5)
    # model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
    # model.fit(train_data.X, train_data.Y)  # 线性回归建模
    # print('系数矩阵:\n', model.coef_)
    # print('线性回归模型:\n', model)
    # # print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
    # # 使用模型预测
    # predicted = model.predict(test_data.X)
    # loss_tmp = mean_squared_error(test_data.Y, torch.tensor(predicted))
    # print(f"Ridge regression: {loss_tmp}")

    # polynomial = PolynomialFeatures(degree=3)  # 二次多项式
    # x_transformed = polynomial.fit_transform(test_data.X)  # x每个数据对应的多项式系数
    #
    # poly_linear_model = LinearRegression()  # 创建回归器
    # poly_linear_model.fit(x_transformed, test_data.Y)  # 训练数据
    #
    # xx_transformed = polynomial.fit_transform(test_data.X)  # 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
    # yy = poly_linear_model.predict(xx_transformed)
    print(f"polynomial regression: {yy}")
