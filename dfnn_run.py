from param_config import ParamConfig
from dataset import Dataset, DatasetNN
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from neuron import NeuronC, NeuronD
from sklearn import svm
import torch.nn as nn
import torch
from model import MLP


def fuzzy_net_run(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset:
    :return:
    """
    train_data, test_data = dataset.get_run_set()
    # get training model
    net = param_config.net
    # updata feature seperator
    param_config.update_seperator(dataset.name)

    # trainning global method
    neuron_c = NeuronC(param_config.rules, param_config.h_computer,
                       param_config.fnn_solver)

    fuzy_tree_c = type(net)(neuron_c)
    fuzy_tree_c.forward(data=train_data, para_mu=param_config.para_mu_current,
                        n_rules=param_config.n_rules,
                        seperator=param_config.fea_seperator)
    y_hat_train = fuzy_tree_c.predict(train_data, param_config.fea_seperator)
    train_loss_c = param_config.loss_fun.forward(train_data.Y, y_hat_train)
    y_hat_test = fuzy_tree_c.predict(test_data, param_config.fea_seperator)
    test_loss_c = param_config.loss_fun.forward(test_data.Y, y_hat_test)
    fuzy_tree_c.clear()

    if dataset.task == 'C':
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

    if dataset.task == 'C':
        param_config.log.info(f"Accuracy of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"Accuracy of test data on distributed method: {cfnn_test_loss}")
    else:
        param_config.log.info(f"loss of training data on distributed method: {cfnn_train_loss}")
        param_config.log.info(f"loss of test data on distributed method: {cfnn_test_loss}")

    return train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss


def mlp_run(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset:
    :return:
    """
    train_data, test_data = dataset.get_run_set()

    train_dataset = DatasetNN(x=train_data.X, y=train_data.Y)
    valid_dataset = DatasetNN(x=test_data.X, y=test_data.Y)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

    if dataset.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_losses}")
        param_config.log.info(f"Accuracy of test data using SVM: {valid_losses}")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_losses}")
        param_config.log.info(f"loss of test data using SVM: {valid_losses}")
    test_map = torch.tensor(valid_acc_list).mean()
    return test_map, train_losses


def svm_local(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset:
    :return:
    """
    loss_train_tsr = []
    loss_test_tsr = []
    for k in torch.arange(param_config.n_kfolds):
        param_config.log.info(f"start traning at {k + 1}-fold!")

        param_config.patition_strategy.set_current_folds(k)
        train_data, test_data = dataset.get_run_set()
        train_label = train_data.Y.squeeze()
        test_label = test_data.Y.squeeze()
        x = train_data.X
        classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
        classifier.fit(x, train_label)
        # trainning global method
        train_loss = classifier.score(train_data.X, train_label)
        test_loss = classifier.score(test_data.X, test_label)
        loss_train_tsr.append(train_loss)
        loss_test_tsr.append(test_loss)
        if dataset.task == 'C':
            param_config.log.info(f"Accuracy of training data using SVM: {train_loss}")
            param_config.log.info(f"Accuracy of test data using SVM: {test_loss}")
        else:
            param_config.log.info(f"loss of training data using SVM: {train_loss}")
            param_config.log.info(f"loss of test data using SVM: {test_loss}")

    loss_train_tsr = torch.tensor(loss_train_tsr)
    loss_test_tsr = torch.tensor(loss_test_tsr)
    return loss_train_tsr.mean(), loss_test_tsr.mean()


def dfnn_kfolds(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset:
    :return:
    """
    loss_c_train_tsr = []
    loss_c_test_tsr = []
    loss_d_train_tsr = []
    loss_d_test_tsr = []
    for k in torch.arange(param_config.n_kfolds):
        param_config.patition_strategy.set_current_folds(k)
        param_config.log.info(f"start traning at {param_config.patition_strategy.current_fold + 1}-fold!")
        train_loss_c, test_loss_c, cfnn_train_loss, cfnn_test_loss = fuzzy_net_run(param_config, dataset)

        loss_c_train_tsr.append(train_loss_c)
        loss_c_test_tsr.append(test_loss_c)

        loss_d_train_tsr.append(cfnn_train_loss)
        loss_d_test_tsr.append(cfnn_test_loss)

    loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
    loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
    loss_d_train_tsr = torch.tensor(loss_d_train_tsr)
    loss_d_test_tsr = torch.tensor(loss_d_test_tsr)
    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr


def dfnn_ite_rules(max_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: this method is to calculate different rule numbers on distribute fuzzy Neuron network iterately
    :param max_rules:
    :param param_config:
    :param dataset:
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


def dfnn_ite_rules_mu(max_rules, param_config: ParamConfig, dataset: Dataset):
    """
    todo: consider all parameters in para_mu_list into algorithm
    :param max_rules:
    :param param_config:
    :param dataset:
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
            dfnn_ite_rules(max_rules, param_config, dataset)
        loss_c_train_mu_tsr = torch.cat((loss_c_train_mu_tsr, loss_c_train.unsqueeze(0).double()), 0)
        loss_c_test_mu_tsr = torch.cat((loss_c_test_mu_tsr, loss_c_test.unsqueeze(0).double()), 0)
        loss_d_train_mu_tsr = torch.cat((loss_d_train_mu_tsr, loss_d_train.unsqueeze(0).double()), 0)
        loss_d_test_mu_tsr = torch.cat((loss_d_test_mu_tsr, loss_d_test.unsqueeze(0).double()), 0)

    return loss_c_train_mu_tsr, loss_c_test_mu_tsr, loss_d_train_mu_tsr, loss_d_test_mu_tsr
