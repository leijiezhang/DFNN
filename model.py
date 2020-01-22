from neuron import Neuron
from dataset import Dataset
from seperator import FeaSeperator
from typing import List
import torch
import torch.nn as nn
from math_utils import cal_fc_w
import abc


class NetBase(object):
    def __init__(self, neuron_seed: Neuron):
        self.__neuron_seed: Neuron = neuron_seed

    def get_neuron_seed(self):
        return self.__neuron_seed

    def set_neuron_seed(self, neuron_seed: Neuron):
        self.__neuron_seed = neuron_seed

    @abc.abstractmethod
    def forward(self, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        pass

    @abc.abstractmethod
    def clear(self):
        pass


class TreeNet(NetBase):
    def __init__(self, neuron_seed: Neuron):
        super(TreeNet, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator = kwargs['seperator'].get_seperator()

        data_tmp = data
        neuron_seed = self.get_neuron_seed()

        rules_tree: List[List[type(neuron_seed)]] = []
        for j in torch.arange(len(seperator)):
            sub_seperator = seperator[int(j)]
            if len(sub_seperator):
                sub_dataset_list = data_tmp.get_subset_fea(sub_seperator)
                # set level j
                rules_sub: List[type(neuron_seed)] = []
                y_sub = torch.empty(data.Y.shape[0], 0).double()
                for i in torch.arange(len(sub_seperator)):
                    neuron_seed.clear()
                    neuron_c = neuron_seed.clone()
                    sub_dataset_i = sub_dataset_list[i]
                    kwargs['data'] = sub_dataset_i
                    neuron_c.forward(**kwargs)
                    y_i = neuron_c.predict(sub_dataset_i)
                    rules_sub.append(neuron_c)
                    y_sub = torch.cat((y_sub, y_i), 1)

                # set outputs of this level as dataset
                data_tmp = Dataset(f"{data}_level{int(j + 1)}", y_sub, data.Y, data.task)
                rules_tree.append(rules_sub)
            else:
                # set bottom level
                kwargs['data'] = data_tmp
                neuron_seed.clear()
                neuron = neuron_seed.clone()
                neuron.forward(**kwargs)

                # set bottom level neuron
                rules_btm: List[type(neuron_seed)] = [neuron]
                rules_tree.append(rules_btm)

        self.set_neuron_tree(rules_tree)

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator = seperator_p.get_seperator()
        data_tmp = data

        for i in torch.arange(len(neuron_tree) - 1):
            output_sub = torch.empty(data.Y.shape[0], 0).double()
            # get level i
            rules_sub: List[type(neuron_seed)] = neuron_tree[int(i)]
            sub_dataset_list = data_tmp.get_subset_fea(seperator[int(i)])
            for j in torch.arange(len(rules_sub)):
                neuron = rules_sub[int(j)]
                sub_dataset_j = sub_dataset_list[j]
                output_j = neuron.predict(sub_dataset_j)
                output_sub = torch.cat((output_sub, output_j), 1)

            # set outputs of this level as dataset
            data_tmp = Dataset(f"{data}_level{int(i + 1)}", output_sub, data.Y, data.task)

        # get bottom neuron
        neuron_btm = neuron_tree[len(neuron_tree) - 1][0]
        y_hat = neuron_btm.predict(data_tmp)
        return y_hat

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree


class TreeFNNet(NetBase):
    """
    the bottome layer is a fc layer instead of a neuron
    """
    def __init__(self, neuron_seed: Neuron):
        super(TreeFNNet, self).__init__(neuron_seed)
        self.__neuron_tree: List[List[type(neuron_seed)]] = []
        self.__fc_w: torch.Tensor = []

    def forward(self, **kwargs):
        data: Dataset = kwargs['data']
        if 'seperator' not in kwargs:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator = kwargs['seperator'].get_seperator()

        data_tmp = data
        neuron_seed = self.get_neuron_seed()

        rules_tree: List[List[type(neuron_seed)]] = []
        for j in torch.arange(len(seperator)):
            sub_seperator = seperator[int(j)]
            if len(sub_seperator):
                sub_dataset_list = data_tmp.get_subset_fea(sub_seperator)
                # set level j
                rules_sub: List[type(neuron_seed)] = []
                y_sub = torch.empty(data.Y.shape[0], 0).double()
                for i in torch.arange(len(sub_seperator)):
                    neuron_seed.clear()
                    neuron_c = neuron_seed.clone()
                    sub_dataset_i = sub_dataset_list[i]
                    kwargs['data'] = sub_dataset_i
                    neuron_c.forward(**kwargs)
                    y_i = neuron_c.predict(sub_dataset_i)
                    rules_sub.append(neuron_c)
                    y_sub = torch.cat((y_sub, y_i), 1)

                # set outputs of this level as dataset
                data_tmp = Dataset(f"{data}_level{int(j + 1)}", y_sub, data.Y, data.task)
                rules_tree.append(rules_sub)
            else:
                # set bottom level fc
                para_mu = kwargs['para_mu']
                w = cal_fc_w(data_tmp.X, data_tmp.Y, para_mu)
                self.set_fc_w(w)

        self.set_neuron_tree(rules_tree)

    def clear(self):
        neuron_seed = self.get_neuron_seed()
        self.set_neuron_seed(type(neuron_seed)(neuron_seed.get_h_computer(),
                                               neuron_seed.get_h_computer(),
                                               neuron_seed.get_fnn_solver()))
        neuron_tree: List[List[type(neuron_seed)]] = []
        self.set_neuron_tree(neuron_tree)
        self.set_fc_w(torch.empty(0))

    def predict(self, data: Dataset, seperator_p: FeaSeperator = None):
        neuron_tree = self.get_neuron_tree()
        neuron_seed = self.get_neuron_seed()
        if seperator_p is None:
            seperator = FeaSeperator(data.name).get_seperator()
        else:
            seperator = seperator_p.get_seperator()
        data_tmp = data

        for i in torch.arange(len(neuron_tree)):
            output_sub = torch.empty(data.Y.shape[0], 0).double()
            # get level i
            rules_sub: List[type(neuron_seed)] = neuron_tree[int(i)]
            sub_dataset_list = data_tmp.get_subset_fea(seperator[int(i)])
            for j in torch.arange(len(rules_sub)):
                neuron = rules_sub[int(j)]
                sub_dataset_j = sub_dataset_list[j]
                output_j = neuron.predict(sub_dataset_j)
                output_sub = torch.cat((output_sub, output_j), 1)

            # set outputs of this level as dataset
            data_tmp = Dataset(f"{data}_level{int(i + 1)}", output_sub, data.Y, data.task)

        # get bottom fc layer
        w = self.get_fc_w()
        y_hat = data_tmp.X.mm(w)
        return y_hat

    def get_neuron_tree(self):
        return self.__neuron_tree

    def set_neuron_tree(self, neuron_tree: List[List[type(Neuron)]]):
        self.__neuron_tree = neuron_tree

    def set_fc_w(self, fc_w: torch.Tensor):
        self.__fc_w = fc_w

    def get_fc_w(self):
        return self.__fc_w


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(18, 18),
            nn.Linear(18, 9),
            nn.Linear(9, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1).float()
        x = self.layers(x)
        return x
