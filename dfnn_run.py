from param_config import ParamConfig
from dataset import Dataset, DatasetNN
from torch.utils.data import DataLoader
from neuron import NeuronC, NeuronD, NeuronDC, NeuronDN
# from svmutil import *
import torch.nn as nn
from loss_utils import LossFunc
import torch
from model import MLP


def neuron_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    # get training model
    net = param_config.net

    # trainning global method
    neuron_c = NeuronC(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)

    neuron_c.forward(data=train_data, para_mu=param_config.para_mu_current,
                     n_rules=param_config.n_rules)
    y_hat_train = neuron_c.predict(train_data)
    train_loss_c = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = neuron_c.predict(test_data)
    test_loss_c = param_config.loss_fun.forward(test_data.Y, y_hat_test)

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on centralized method: {train_loss_c}")
        param_config.log.info(f"Accuracy of test data on centralized method: {test_loss_c}")
    else:
        param_config.log.info(f"loss of training data on centralized method: {train_loss_c}")
        param_config.log.info(f"loss of test data on centralized method: {test_loss_c}")

    # train distributed fnn
    neuron_d = NeuronD(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)
    neuron_d.forward(data=train_data, para_mu=param_config.para_mu_current,
                     para_rho=param_config.para_rho, n_agents=param_config.n_agents,
                     n_rules=param_config.n_rules)

    y_hat_train = neuron_d.predict(train_data)
    cfnn_train_loss = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = neuron_d.predict(test_data)
    cfnn_test_loss = param_config.loss_fun.forward(test_data.Y, y_hat_test)

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"Accuracy of test data on distributed method: {cfnn_test_loss}")
    else:
        param_config.log.info(f"loss of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"loss of test data on distributed method: {cfnn_test_loss}")

    return train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss


def fuzzy_net_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    # get training model
    net = param_config.net

    # trainning global method
    neuron_c = NeuronC(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)

    fuzy_tree_c = type(net)(neuron_c)
    fuzy_tree_c.forward(data=train_data, para_mu=param_config.para_mu_current,
                        para_rho=param_config.para_rho, n_agents=param_config.n_agents,
                        n_rules=param_config.n_rules, seperator=param_config.fea_seperator)
    y_hat_train = fuzy_tree_c.predict(train_data, param_config.fea_seperator)
    train_loss_c = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = fuzy_tree_c.predict(test_data, param_config.fea_seperator)
    test_loss_c = param_config.loss_fun.forward(test_data.Y, y_hat_test)
    fuzy_tree_c.clear()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on centralized method: {train_loss_c}")
        param_config.log.info(f"Accuracy of test data on centralized method: {test_loss_c}")
    else:
        param_config.log.info(f"loss of training data on centralized method: {train_loss_c}")
        param_config.log.info(f"loss of test data on centralized method: {test_loss_c}")

    # train distributed fnn
    neuron_d = NeuronD(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)
    fuzy_tree_d = type(net)(neuron_d)
    fuzy_tree_d.forward(data=train_data, para_mu=param_config.para_mu_current,
                        para_rho=param_config.para_rho, n_agents=param_config.n_agents,
                        n_rules=param_config.n_rules, seperator=param_config.fea_seperator)

    y_hat_train = fuzy_tree_d.predict(train_data, param_config.fea_seperator)
    cfnn_train_loss = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = fuzy_tree_d.predict(test_data, param_config.fea_seperator)
    cfnn_test_loss = param_config.loss_fun.forward(test_data.Y, y_hat_test)
    fuzy_tree_d.clear()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"Accuracy of test data on distributed method: {cfnn_test_loss}")
    else:
        param_config.log.info(f"loss of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"loss of test data on distributed method: {cfnn_test_loss}")

    return train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss


def fuzzy_dnn_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    # get training model
    net = param_config.net

    # train distributed fnn
    neuron_dn = NeuronDN(param_config.rules, param_config.h_computer,
                         param_config.fnn_solver)
    fuzy_tree_dn = type(net)(neuron_dn)
    fuzy_tree_dn.forward(data=train_data, para_mu=param_config.para_mu_current,
                         para_rho=param_config.para_rho, n_agents=param_config.n_agents,
                         n_rules=param_config.n_rules, seperator=param_config.fea_seperator)

    cfnn_train_loss = fuzy_tree_dn.predict(train_data, param_config.fea_seperator)
    cfnn_test_loss = fuzy_tree_dn.predict(test_data, param_config.fea_seperator)
    fuzy_tree_dn.clear()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"Accuracy of test data on distributed method: {cfnn_test_loss}")
    else:
        param_config.log.info(f"loss of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"loss of test data on distributed method: {cfnn_test_loss}")

    return cfnn_train_loss, cfnn_test_loss


def mlp_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """

    train_dataset = DatasetNN(x=train_data.X, y=train_data.Y)
    valid_dataset = DatasetNN(x=test_data.X, y=test_data.Y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    model: nn.Module = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    valid_acc_list = []
    epochs = 15

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()

        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs.double(), labels.long().squeeze(1))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                outputs = model(images)
                loss = loss_fn(outputs, labels.long().squeeze(1))

                valid_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels.squeeze().long()).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        valid_acc_list.append(accuracy)
        print(f"epoch : {epoch + 1}, train loss : {train_losses[-1]}, "
              f"valid loss : {valid_losses[-1]}, valid acc : {accuracy}%")

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_losses}")
        param_config.log.info(f"Accuracy of test data using SVM: {valid_losses}")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_losses}")
        param_config.log.info(f"loss of test data using SVM: {valid_losses}")
    test_map = torch.tensor(valid_acc_list).mean()
    return test_map, train_losses


# def svm_local(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
#     """
#     todo: this is the method for distribute fuzzy Neuron network
#     :param param_config:
#     :param train_data: training dataset
#     :param test_data: test dataset
#     :return:
#     """
#     # classifier = svm.SVC()
#     train_label = train_data.Y.squeeze().numpy()
#     test_label = test_data.Y.squeeze().numpy()
#     x = train_data.X.numpy()
#
#     print("training the one-class SVM")
#     prob_train = svm_problem(train_label, x)
#
#     param = svm_parameter('-s 0 -t 2 -d 4 -c 10')
#
#     model = svm_train(prob_train, param)
#
#     print("predicting the test data")
#
#     # classifier.fit(x, train_label)
#     # # trainning global method
#     # train_label_hat = classifier.predict(x)
#     test_x = test_data.X.numpy()
#     # test_label_hat = classifier.predict(test_x)
#
#     print("predicting")
#
#     test_label_hat, _, _ = svm_predict(test_label, test_x, model, '-b 0')
#     train_label_hat, _, _ = svm_predict(train_label, x, model, '-b 0')
#     loss_fun: LossFunc = param_config.loss_fun
#
#     train_acc = loss_fun.forward(train_data.Y.squeeze(), torch.tensor(train_label_hat))
#     test_acc = loss_fun.forward(test_data.Y.squeeze(), torch.tensor(test_label_hat))
#     if test_data.task == 'C':
#         param_config.log.info(f"Accuracy of training data using SVM: {train_acc}")
#         param_config.log.info(f"Accuracy of test data using SVM: {test_acc}")
#     else:
#         param_config.log.info(f"loss of training data using SVM: {train_acc}")
#         param_config.log.info(f"loss of test data using SVM: {test_acc}")
#
#     return train_acc, test_acc
#
#
# def svm_kfolds(param_config: ParamConfig, dataset: Dataset):
#     """
#     todo: this is the method for distribute fuzzy Neuron network
#     :param param_config:
#     :param dataset:  dataset
#     :return:
#     """
#     loss_c_train_tsr = []
#     loss_c_test_tsr = []
#     for k in torch.arange(param_config.n_kfolds):
#         param_config.patition_strategy.set_current_folds(k)
#         train_data, test_data = dataset.get_run_set()
#         param_config.log.info(f"start traning at {param_config.patition_strategy.current_fold + 1}-fold!")
#         train_loss_c, test_loss_c = svm_local(param_config, train_data, test_data)
#
#         loss_c_train_tsr.append(train_loss_c)
#         loss_c_test_tsr.append(test_loss_c)
#
#     loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
#     loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
#     return loss_c_train_tsr, loss_c_test_tsr


def dfnn_kfolds(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset: dataset
    :return:
    """
    loss_c_train_tsr = []
    loss_c_test_tsr = []
    loss_d_train_tsr = []
    loss_d_test_tsr = []
    for k in torch.arange(param_config.n_kfolds):
        param_config.patition_strategy.set_current_folds(k)
        train_data, test_data = dataset.get_run_set()
        param_config.log.info(f"start traning at {param_config.patition_strategy.current_fold + 1}-fold!")
        train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss = \
            fuzzy_net_run(param_config, train_data, test_data)

        loss_c_train_tsr.append(train_loss_c)
        loss_c_test_tsr.append(test_loss_c)

        loss_d_train_tsr.append(cfnn_train_loss)
        loss_d_test_tsr.append(cfnn_test_loss)

    loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
    loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
    loss_d_train_tsr = torch.tensor(loss_d_train_tsr)
    loss_d_test_tsr = torch.tensor(loss_d_test_tsr)
    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr


def eeg_dfnn_kfolds(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for trial based eeg signal dataset using kfolds validate
    :param param_config:
    :param dataset: dataset
    :return:
    """
    loss_c_train_tsr = []
    loss_c_test_tsr = []
    loss_d_train_tsr = []
    loss_d_test_tsr = []

    n_fea = 24
    for k in torch.arange(param_config.n_kfolds):
        param_config.patition_strategy.set_current_folds(k)
        train_data, test_data = dataset.get_run_set()

        train_x = train_data.X
        train_x = train_x.view(train_x.shape[0], n_fea, -1).t()
        n_trial = train_x.shape[2]
        train_x = train_x.permute(0, 2, 1)
        train_x = train_x.view(-1, n_fea)

        train_y = train_data.Y
        train_y = train_y.repeat(1, n_trial)
        train_y = train_y.view(-1)
        train_data.X = train_x
        train_data.Y = train_y

        test_x = test_data.X
        test_x = test_x.view(test_x.shape[0], n_fea, -1).t()
        n_trial = test_x.shape[2]
        test_x = test_x.permute(0, 2, 1)
        test_x = test_x.view(-1, n_fea)

        test_y = test_data.Y
        test_y = test_y.repeat(1, n_trial)
        test_y = test_y.view(-1)
        test_data.X = test_x
        test_data.Y = test_y

        param_config.log.info(f"start traning at {param_config.patition_strategy.current_fold + 1}-fold!")
        train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss = \
            fuzzy_net_run(param_config, train_data, test_data)

        loss_c_train_tsr.append(train_loss_c)
        loss_c_test_tsr.append(test_loss_c)

        loss_d_train_tsr.append(cfnn_train_loss)
        loss_d_test_tsr.append(cfnn_test_loss)

    loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
    loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
    loss_d_train_tsr = torch.tensor(loss_d_train_tsr)
    loss_d_test_tsr = torch.tensor(loss_d_test_tsr)
    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr


def dfnn_ite_rules_kfolds(max_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy Neuron network iterately
    :param max_rules:
    :param param_config:
    :param dataset: dataset
    :return:
    """
    loss_c_train_tsr = torch.empty(0, param_config.n_kfolds).double()
    loss_c_test_tsr = torch.empty(0, param_config.n_kfolds).double()
    loss_d_train_tsr = torch.empty(0, param_config.n_kfolds).double()
    loss_d_test_tsr = torch.empty(0, param_config.n_kfolds).double()

    for i in torch.arange(max_rules):
        n_rules = int(i + 1)
        param_config.log.info(f"running at rule number: {n_rules}")
        param_config.n_rules = n_rules

        loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
            dfnn_kfolds(param_config, dataset)

        loss_c_train_tsr = torch.cat((loss_c_train_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_tsr = torch.cat((loss_c_test_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_tsr = torch.cat((loss_d_train_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_tsr = torch.cat((loss_d_test_tsr, loss_d_test.unsqueeze(0).double()), 0)

    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr


def dfnn_ite_rules_mu_kfolds(max_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: consider all parameters in para_mu_list into algorithm
    :param max_rules:
    :param param_config:
    :param dataset: training dataset
    :return:
    """
    loss_c_train_mu_tsr = torch.empty(0, max_rules, param_config.n_kfolds).double()
    loss_c_test_mu_tsr = torch.empty(0, max_rules, param_config.n_kfolds).double()
    loss_d_train_mu_tsr = torch.empty(0, max_rules, param_config.n_kfolds).double()
    loss_d_test_mu_tsr = torch.empty(0, max_rules, param_config.n_kfolds).double()

    for i in torch.arange(param_config.para_mu_list.shape[0]):
        param_config.para_mu_current = param_config.para_mu_list[i]
        param_config.log.info(f"running param mu: {param_config.para_mu_current}")

        loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
            dfnn_ite_rules_kfolds(max_rules, param_config, dataset)
        loss_c_train_mu_tsr = torch.cat((loss_c_train_mu_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_mu_tsr = torch.cat((loss_c_test_mu_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_mu_tsr = torch.cat((loss_d_train_mu_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_mu_tsr = torch.cat((loss_d_test_mu_tsr, loss_d_test.unsqueeze(0).double()), 0)

    return loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr


def dfnn_ite_rules(max_rules, param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy Neuron network iterately
    :param max_rules:
    :param param_config:
    :param train_data: dataset
    :param test_data: dataset
    :return:
    """
    loss_c_train_list = []
    loss_c_test_list = []
    loss_d_train_list = []
    loss_d_test_list = []

    for i in torch.arange(max_rules):
        n_rules = int(i + 1)
        param_config.log.info(f"running at rule number: {n_rules}")
        param_config.n_rules = n_rules

        loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
            fuzzy_net_run(param_config, train_data, test_data)

        loss_c_train_list.append(loss_c_train)
        loss_c_test_list.append(loss_c_test)
        loss_d_train_list.append(loss_d_train)
        loss_d_test_list.append(loss_d_test)

    return loss_c_train_list, loss_c_test_list, loss_d_train_list, loss_d_test_list


def dfnn_ite_rules_mu(max_rules, param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: consider all parameters in para_mu_list into algorithm
    :param max_rules:
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    loss_c_train_mu_list = []
    loss_c_test_mu_list = []
    loss_d_train_mu_list = []
    loss_d_test_mu_list = []

    for i in torch.arange(param_config.para_mu_list.shape[0]):
        param_config.para_mu_current = param_config.para_mu_list[i]
        param_config.log.info(f"running param mu: {param_config.para_mu_current}")

        loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
            dfnn_ite_rules(max_rules, param_config, train_data, test_data)
        loss_c_train_mu_list.append(loss_c_train)
        loss_c_test_mu_list.append(loss_c_test)
        loss_d_train_mu_list.append(loss_d_train)
        loss_d_test_mu_list.append(loss_d_test)
    loss_c_train_mu_tsr = torch.tensor(loss_c_train_mu_list)
    loss_c_test_mu_tsr = torch.tensor(loss_c_test_mu_list)
    loss_d_train_mu_tsr = torch.tensor(loss_d_train_mu_list)
    loss_d_test_mu_tsr = torch.tensor(loss_d_test_mu_list)

    return loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr


def hdfnn_ao_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for hierarchical distribute fuzzy Neuron network using alternative optimizing
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    # get training model
    net = param_config.net

    # trainning global method
    neuron_c = NeuronC(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)

    fuzy_tree_c = type(net)(neuron_c)
    fuzy_tree_c.forward(data=train_data, para_mu=param_config.para_mu_current,
                        para_mu1=param_config.para_mu1_current,
                        para_rho=param_config.para_rho, n_agents=param_config.n_agents,
                        n_rules=param_config.n_rules, seperator=param_config.fea_seperator)
    y_hat_train = fuzy_tree_c.predict(train_data, param_config.fea_seperator)
    train_loss_c = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = fuzy_tree_c.predict(test_data, param_config.fea_seperator)
    test_loss_c = param_config.loss_fun.forward(test_data.Y, y_hat_test)
    fuzy_tree_c.clear()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on centralized method: {train_loss_c}")
        param_config.log.info(f"Accuracy of test data on centralized method: {test_loss_c}")
    else:
        param_config.log.info(f"loss of training data on centralized method: {train_loss_c}")
        param_config.log.info(f"loss of test data on centralized method: {test_loss_c}")

    # train distributed fnn
    neuron_d = NeuronD(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)
    fuzy_tree_d = type(net)(neuron_d)
    fuzy_tree_d.forward(data=train_data, para_mu=param_config.para_mu_current,
                        para_mu1=param_config.para_mu1_current,
                        para_rho=param_config.para_rho, n_agents=param_config.n_agents,
                        n_rules=param_config.n_rules, seperator=param_config.fea_seperator)

    y_hat_train = fuzy_tree_d.predict(train_data, param_config.fea_seperator)
    cfnn_train_loss = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = fuzy_tree_d.predict(test_data, param_config.fea_seperator)
    cfnn_test_loss = param_config.loss_fun.forward(test_data.Y, y_hat_test)
    fuzy_tree_d.clear()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"Accuracy of test data on distributed method: {cfnn_test_loss}")
    else:
        param_config.log.info(f"loss of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"loss of test data on distributed method: {cfnn_test_loss}")

    return train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss


def dfnn_kfolds_ao(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset: dataset
    :return:
    """
    loss_c_train_tsr = []
    loss_c_test_tsr = []
    loss_d_train_tsr = []
    loss_d_test_tsr = []
    for k in torch.arange(param_config.n_kfolds):
        param_config.patition_strategy.set_current_folds(k)
        train_data, test_data = dataset.get_run_set()
        param_config.log.info(f"start traning at {param_config.patition_strategy.current_fold + 1}-fold!")
        train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss = \
            hdfnn_ao_run(param_config, train_data, test_data)

        loss_c_train_tsr.append(train_loss_c)
        loss_c_test_tsr.append(test_loss_c)

        loss_d_train_tsr.append(cfnn_train_loss)
        loss_d_test_tsr.append(cfnn_test_loss)

    loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
    loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
    loss_d_train_tsr = torch.tensor(loss_d_train_tsr)
    loss_d_test_tsr = torch.tensor(loss_d_test_tsr)

    if dataset.task == 'C':
        param_config.log.war(f"Mean Accuracy of training data on centralized method:"
                             f" {loss_c_train_tsr.mean()}")
        param_config.log.war(f"Mean Accuracy  of test data on centralized method: "
                             f"{loss_c_test_tsr.mean()}")
        param_config.log.war(f"Mean Accuracy  of training data on distributed method:"
                             f" {loss_d_train_tsr.mean()}")
        param_config.log.war(f"Mean Accuracy  of test data on distributed method: "
                             f"{loss_d_test_tsr.mean()}")
    else:
        param_config.log.war(f"loss of training data on centralized method: "
                             f"{loss_c_train_tsr.mean()}")
        param_config.log.war(f"loss of test data on centralized method: "
                             f"{loss_c_test_tsr.mean()}")
        param_config.log.war(f"loss of training data on distributed method: "
                             f"{loss_d_train_tsr.mean()}")
        param_config.log.war(f"loss of test data on distributed method: "
                             f"{loss_d_test_tsr.mean()}")
    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr


def dfnn_ite_para_ao(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: consider all parameters in para_mu_list into algorithm
    :param param_config:
    :param train_data: dataset
    :param test_data: dataset
    :return:
    """
    n_mu_list = param_config.para_mu_list.shape[0]
    n_mu1_list = param_config.para_mu1_list.shape[0]

    loss_c_train_mu_tsr = torch.zeros(n_mu_list, n_mu1_list).double()
    loss_c_test_mu_tsr = torch.zeros(n_mu_list, n_mu1_list).double()
    loss_d_train_mu_tsr = torch.zeros(n_mu_list, n_mu1_list).double()
    loss_d_test_mu_tsr = torch.zeros(n_mu_list, n_mu1_list).double()

    for i in torch.arange(n_mu_list):
        for j in torch.arange(n_mu1_list):
            param_config.para_mu_current = param_config.para_mu_list[i]
            param_config.para_mu1_current = param_config.para_mu1_list[j]
            param_config.log.info(f"running param mu: {param_config.para_mu_current}")
            param_config.log.info(f"running param mu1: {param_config.para_mu1_current}")

            loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
                hdfnn_ao_run(param_config, train_data, test_data)
            loss_c_train_mu_tsr[i, j, :] = loss_c_train.double()
            loss_c_test_mu_tsr[i, j, :] = loss_c_test.double()
            loss_d_train_mu_tsr[i, j, :] = loss_d_train.double()
            loss_d_test_mu_tsr[i, j, :] = loss_d_test.double()

    return loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr


def dfnn_ite_para_kfolds_ao(param_config: ParamConfig, dataset: Dataset):
    """
    todo: consider all parameters in para_mu_list into algorithm
    :param param_config:
    :param dataset: training dataset
    :return:
    """
    n_mu_list = param_config.para_mu_list.shape[0]
    n_mu1_list = param_config.para_mu1_list.shape[0]

    loss_c_train_mu_tsr = torch.zeros(n_mu_list, n_mu1_list, param_config.n_kfolds).double()
    loss_c_test_mu_tsr = torch.zeros(n_mu_list, n_mu1_list, param_config.n_kfolds).double()
    loss_d_train_mu_tsr = torch.zeros(n_mu_list, n_mu1_list, param_config.n_kfolds).double()
    loss_d_test_mu_tsr = torch.zeros(n_mu_list, n_mu1_list, param_config.n_kfolds).double()

    for i in torch.arange(n_mu_list):
        for j in torch.arange(n_mu1_list):
            param_config.para_mu_current = param_config.para_mu_list[i]
            param_config.para_mu1_current = param_config.para_mu1_list[j]
            param_config.log.info(f"running param mu: {param_config.para_mu_current}")
            param_config.log.info(f"running param mu1: {param_config.para_mu1_current}")

            loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
                dfnn_kfolds_ao(param_config, dataset)
            loss_c_train_mu_tsr[i, j, :] = loss_c_train.squeeze().double()
            loss_c_test_mu_tsr[i, j, :] = loss_c_test.squeeze().double()
            loss_d_train_mu_tsr[i, j, :] = loss_d_train.squeeze().double()
            loss_d_test_mu_tsr[i, j, :] = loss_d_test.squeeze().double()

    return loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr


def dfnn_ite_rules_para_ao(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy Neuron network iterately
    :param param_config:
    :param train_data: dataset
    :param test_data: dataset
    :return:
    """
    n_mu_list = param_config.para_mu_list.shape[0]
    n_mu1_list = param_config.para_mu1_list.shape[0]

    loss_c_train_tsr = torch.empty(0, n_mu_list, n_mu1_list).double().double()
    loss_c_test_tsr = torch.empty(0, n_mu_list, n_mu1_list).double().double()
    loss_d_train_tsr = torch.empty(0, n_mu_list, n_mu1_list).double().double()
    loss_d_test_tsr = torch.empty(0, n_mu_list, n_mu1_list).double().double()

    n_rule_list = param_config.n_rules_list
    for i in torch.arange(n_rule_list.shape[0]):
        n_rules = n_rule_list[int(i)]
        param_config.log.info(f"running at rule number: {n_rules}")
        param_config.n_rules = n_rules

        loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
            dfnn_ite_para_ao(param_config, train_data, test_data)

        loss_c_train_tsr = torch.cat((loss_c_train_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_tsr = torch.cat((loss_c_test_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_tsr = torch.cat((loss_d_train_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_tsr = torch.cat((loss_d_test_tsr, loss_d_test.unsqueeze(0).double()), 0)

    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr


def dfnn_ite_rules_para_kfold_ao(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy Neuron network iterately
    :param param_config:
    :param dataset: dataset
    :return:
    """
    n_mu_list = param_config.para_mu_list.shape[0]
    n_mu1_list = param_config.para_mu1_list.shape[0]

    loss_c_train_tsr = torch.empty(0, n_mu_list, n_mu1_list, param_config.n_kfolds).double().double()
    loss_c_test_tsr = torch.empty(0, n_mu_list, n_mu1_list, param_config.n_kfolds).double().double()
    loss_d_train_tsr = torch.empty(0, n_mu_list, n_mu1_list, param_config.n_kfolds).double().double()
    loss_d_test_tsr = torch.empty(0, n_mu_list, n_mu1_list, param_config.n_kfolds).double().double()

    n_rule_list = param_config.n_rules_list
    for i in torch.arange(n_rule_list.shape[0]):
        n_rules = n_rule_list[int(i)]
        param_config.log.info(f"running at rule number: {n_rules}")
        param_config.n_rules = n_rules

        loss_c_train, loss_c_test, loss_d_train, loss_d_test = \
            dfnn_ite_para_kfolds_ao(param_config, dataset)

        loss_c_train_tsr = torch.cat((loss_c_train_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_tsr = torch.cat((loss_c_test_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_tsr = torch.cat((loss_d_train_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_tsr = torch.cat((loss_d_test_tsr, loss_d_test.unsqueeze(0).double()), 0)

    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr
