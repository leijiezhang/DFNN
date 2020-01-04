from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import torch
from skfuzzy.cluster import cmeans, cmeans_predict
import abc


class RuleBase(object):
    """
    this is the base class of rules which is generated by cluster methods
    """
    def __init__(self):
        """
        n_rules: the number of rules, namely the number of cluster centers
        center_list: the list of cluster centers
        consequent_list: list of consequent layer
        x_center_idx: the labels of which center traning data belongs to
        """
        self.n_rules = 0
        self.center_list = None
        self.consequent_list = None
        self.x_center_idx = None

    @abc.abstractmethod
    def fit(self, x, n_rules=5):
        """
        todo initiate the rule class, actually, rules are initially constructed by
         a combination of cluster center, std and labels
        :param x: the data where rules are generated
        :param n_rules: number of the rules, namely the number of cluster centers
        """
        self.n_rules = n_rules

    @abc.abstractmethod
    def update_rules(self, x, center):
        """
        todo: update rule object according to the given cluster center list
        :param x: the data where rules are generated
        :param center: the given cluster center list
        :return: None
        """
        pass


class RuleKmeans(RuleBase):
    def __init__(self):
        """
        widths_list: standard deviation of each clusters
        """
        super(RuleKmeans, self).__init__()
        self.widths_list = None

    def fit(self, x, n_rules=5):
        """
        todo initiate the rule class, actually, rules are initially constructed by
         a combination of cluster center, std and labels
        :param x: the data where rules are generated
        :param n_rules: number of the rules, namely the number of cluster centers
        """
        kmeans = KMeans(n_rules).fit(x)
        self.n_rules = n_rules
        self.center_list = torch.tensor(kmeans.cluster_centers_)
        self.consequent_list = None
        self.x_center_idx = torch.tensor(kmeans.labels_)
        self.widths_list = self.get_widths_list(x)

    def get_widths_list(self, x):
        """
        todo  get standard deviation of each clusters
        :param x: the data where rules are generated
        :return std: standard deviation of each clusters
        """
        # get the std of data x
        std = torch.empty((0, x.shape[1])).double()
        for i in range(self.n_rules):
            mask = self.x_center_idx == i
            cluster_samples = x[mask]
            std_tmp = torch.std(cluster_samples, 0).unsqueeze(0)
            std = torch.cat((std, std_tmp), 0)
        return std

    def update_rules(self, x, center):
        """
        todo: update rule object according to the given cluster center list
        :param x: the data where rules are generated
        :param center: the given cluster center list
        :return: None
        """
        self.center_list = center
        self.n_rules = center.shape[0]
        x_dist = torch.tensor(cdist(x, center))
        center_idx = torch.argmin(x_dist, 1)
        self.x_center_idx = center_idx
        self.widths_list = self.get_widths_list(x)


class RuleFuzzyKmeans(RuleKmeans):
    """
    todo: use fuzzy c means to generate rules and update with normal k means
    this class may be deleted in the future cause this class is awkward in theory
    """
    def __init__(self):
        """
        center_list: the list of cluster centers
        x_center_idx: the labels of which center traning data belongs to
        consequent_list: list of consequent layer
        widths_list: standard deviation of each clusters
        """
        super(RuleFuzzyKmeans, self).__init__()
        self.data_dist = None
        self.data_partition = None

    def fit(self, x, n_rules=5):
        """
        todo initiate the rule class, actually, rules are initially constructed by
         a combination of cluster center, std and labels
        :param x: the data where rules are generated
        :param n_rules: number of the rules, namely the number of cluster centers
        """
        center_list, data_partition, _, data_dist, _, _, _ = \
            cmeans(x.t(), n_rules, 2, error=0.005, maxiter=1000)
        self.n_rules = n_rules
        self.center_list = torch.tensor(center_list)
        self.data_dist = torch.tensor(data_dist).t()
        self.data_partition = torch.tensor(data_partition).t()
        self.consequent_list = None
        self.x_center_idx = torch.argmax(self.data_partition, 1)
        self.widths_list = self.get_widths_list(x)


class RuleFuzzyCmeans(RuleKmeans):
    """
    todo: use fuzzy c means to generate rules
    """
    def __init__(self):
        """
        center_list: the list of cluster centers
        x_center_idx: the labels of which center traning data belongs to
        consequent_list: list of consequent layer
        widths_list: standard deviation of each clusters
        """
        super(RuleFuzzyCmeans, self).__init__()
        self.data_partition = None

    def fit(self, x, n_rules=5):
        """
        todo initiate the rule class, actually, rules are initially constructed by
         a combination of cluster center, std and labels
        :param x: the data where rules are generated
        :param n_rules: number of the rules, namely the number of cluster centers
        """
        center_list, data_partition, _, _, _, _, _ = \
            cmeans(x.t(), n_rules, 2, error=0.005, maxiter=1000)
        self.n_rules = n_rules
        self.center_list = torch.tensor(center_list)
        self.data_partition = torch.tensor(data_partition).t()
        self.x_center_idx = torch.argmax(self.data_partition, 1)
        self.consequent_list = None

    def update_rules(self, x, center):
        """
        todo: update rule object according to the given cluster center list
        :param x: the data where rules are generated
        :param center: the given cluster center list
        :return: None
        """
        self.center_list = center
        self.n_rules = center.shape[0]
        data_partition, _, _, _, _, _ = \
            cmeans_predict(x.t(), center, 2, error=0.005, maxiter=1000)
        self.data_partition = torch.tensor(data_partition).t()
        self.x_center_idx = torch.argmax(self.data_partition, 1)
