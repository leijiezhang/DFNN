import torch
from partition import PartitionStrategy


class Dataset(object):
    """
        we suppose the data structure is X: N x D (N is the number of data samples and D is the data sample dimention)
        and the label set Y as: N x 1
    """
    def __init__(self, name, x, y, task):
        """
        init the Dataset class
        :param name: the name of data set
        :param x: features for train
        :param y: labels for regression or classification
        :param task: R for regression C for classification
        """
        self.name = name
        self.X = x
        self.Y = y
        self.task = task
        self.shuffle = True
        self.random_state = 1
        self.partitions = []
        self.distry_partitions: PartitionStrategy = []
        self.current_partition = 0
        self.n_agents = 1  # use only if get distribute dataset, this denotes the number of distribute
        self.n_fea = self.X.shape[1]

    @staticmethod
    def mapminmax(x: torch.Tensor, l_range=-1, u_range=1):
        xmax = torch.max(x, 0)[0].unsqueeze(0)
        xmin = torch.min(x, 0)[0].unsqueeze(0)
        xmin = xmin.repeat(x.shape[0], 1)
        xmax = xmax.repeat(x.shape[0], 1)

        if (xmax == xmin).any():
            raise ValueError("some rows have no variation")
        x_proj = ((u_range - l_range) * (x - xmin) / (xmax - xmin)) + l_range

        return x_proj

    def normalize(self, l_range: torch.int, u_range: int, flag=None):
        self.X = self.mapminmax(self.X, l_range, u_range)
        if flag is not None:
            d = torch.sum(self.X, 1)
            d = d.repeat(1, self.X.shape[1])
            self.X = self.X / d

    def generate_n_partitions(self, n, partition_strategy):

        """
        Generate N partitions of the dataset using a given
         partitioning strategy. In semi-supervised mode, two
         strategies must be provided.
        :param n: running times of the dataset, how many times do the experiments operate
        :param partition_strategy:
        :return:
        """

        self.partitions = []
        self.current_partition = 0

        for i in torch.arange(n):
            current_y = self.Y

            partition_strategy.partition(current_y, self.random_state, self.shuffle)
            self.partitions.append(partition_strategy)

    def generate_single_partitions(self, partition_strategy):
        self.generate_n_partitions(1, partition_strategy)

    def get_folds(self):
        return self.partitions[0].get_num_folds()

    def set_current_partition(self, cur_part):
        self.current_partition = cur_part

    def get_fold_data(self, n_fold=None):
        x = self.X
        y = self.Y
        partition_strategy = self.partitions[self.current_partition]
        if n_fold is not None:
            partition_strategy.set_current_folds(n_fold)
        train_idx = partition_strategy.get_train_indexes()
        text_idx = partition_strategy.get_test_indexes()
        train_name = f"{self.name}_train"
        train_data = Dataset(train_name, x[train_idx[0], :], y[train_idx[0], :], self.task)
        test_name = f"{self.name}_test"
        text_data = Dataset(test_name, x[text_idx[0], :], y[text_idx[0], :], self.task)

        return train_data, text_data

    def generate_distribute(self, d_partition_strategy):
        self.n_agents = d_partition_strategy.get_num_folds()
        self.distry_partitions = d_partition_strategy
        self.distry_partitions.partition(self.Y, self.random_state, self.shuffle)

    def distribute_dataset(self):
        d_train_data = []
        x = self.X
        y = self.Y
        disp_partition = self.distry_partitions
        for i in torch.arange(self.n_agents):
            disp_partition.set_current_folds(i)
            disp_part_idx = disp_partition.get_test_indexes()
            disp_part_dataset = Dataset(f"{self.name}_distry", x[disp_part_idx[0], :],
                                        y[disp_part_idx[0], :], self.task)
            d_train_data.append(disp_part_dataset)
        return d_train_data


class Result(object):
    """
    todo: save results
    """
    def __init__(self, param_mu_list):
        self.loss_c_train = torch.empty(1)
        self.loss_c_train_mean = torch.empty(1)
        self.loss_c_test = torch.empty(1)
        self.loss_c_test_mean = torch.empty(1)
        self.loss_d_train = torch.empty(1)
        self.loss_d_train_mean = torch.empty(1)
        self.loss_d_test = torch.empty(1)
        self.loss_d_test_mean = torch.empty(1)

        self.loss_curve = []
        self.best_idx = 0
        self.param_mu_list = param_mu_list
        self.best_mu = 0.1

        self.loss_c_train_best = torch.empty(1)
        self.loss_c_test_best = torch.empty(1)
        self.loss_d_train_best = torch.empty(1)
        self.loss_d_test_best = torch.empty(1)

    def get_best_idx(self, best_idx):
        self.best_idx = best_idx
        self.best_mu = self.param_mu_list[best_idx]

        self.loss_c_train_best = self.loss_c_train[best_idx, :]
        self.loss_c_test_best = self.loss_c_test[best_idx, :]
        self.loss_d_train_best = self.loss_d_train[best_idx, :]
        self.loss_d_test_best = self.loss_d_test[best_idx, :]
