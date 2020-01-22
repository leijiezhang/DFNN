import torch


class FeaSeperator(object):
    """
    in order to divide datasets into several subsets of original samples
    Forms: double level list: List[List[Tensor]]--> first level list denotes the level of fuzzy tree
           and the second level list stands for the real seperators of features of dataset
    """
    def __init__(self, data_name):
        self.data_name = data_name
        self.__seperator = [[]]

    def set_seperator_by_slice_window(self, window_size, step=1, n_level=2):
        """set seperator using slice window"""
        seperator = []
        if self.data_name.find('HRSS') != -1:
            n_fea = 18
            seperator = slide_window(n_fea, window_size, step, n_level)

        self.__seperator = seperator

    def set_seperator_by_random_pick(self, window_size, n_repeat=2, n_level=2):
        """set seperator using slice window"""
        seperator = []
        if self.data_name.find('HRSS') != -1:
            n_fea = 18
            seperator = random_pick(n_fea, window_size, n_repeat, n_level)

        self.__seperator = seperator

    def set_seperator_by_no_seperate(self):
        """do not seperate features"""
        seperator = [[]]
        self.__seperator = seperator

    def get_seperator(self):
        return self.__seperator


def slide_window(n_fea, window_size, step=1, n_level=2):
    """
    slide window to get the index of feature seperators
    :param n_fea:
    :param window_size:
    :param step:
    :param n_level:
    :return:
    """
    n_fea_tmp = n_fea
    level_idx = 1
    seperator = []

    while True:
        idx_sub = 0
        seperator_sub = []
        if not window_size < n_fea_tmp or not level_idx < n_level:
            seperator.append([])
            break
        while idx_sub + window_size <= n_fea_tmp:
            fea_idx = torch.linspace(idx_sub, idx_sub + window_size - 1, window_size)
            fea_idx_real = fea_idx % n_fea_tmp
            fea_idx_real = fea_idx_real.long()
            seperator_sub.append(fea_idx_real)
            idx_sub = idx_sub + step
        n_fea_tmp = len(seperator_sub)
        seperator.append(seperator_sub)
        level_idx = level_idx + 1
    return seperator


def random_pick(n_fea, window_size, n_repeat=3, n_level=2):
    """
    randomly picking to get the index of feature seperators
    :param n_fea:
    :param window_size:
    :param n_repeat: the times of repeat selecting
    :param n_level:
    :return:
    """
    n_fea_tmp = n_fea
    step = window_size
    level_idx = 1
    seperator = []

    while True:
        seperator_sub = []
        if not window_size < n_fea_tmp or not level_idx < n_level:
            seperator.append([])
            break

        for i in torch.arange(n_repeat):
            idx_sub = 0
            # disoder the sequnce of features
            fea_seq = torch.randperm(n_fea_tmp)
            while idx_sub + window_size < n_fea_tmp:
                fea_idx = torch.linspace(idx_sub, idx_sub + window_size - 1, window_size).long()
                fea_idx_real = fea_seq[fea_idx] % n_fea_tmp
                fea_idx_real = fea_idx_real.long()
                seperator_sub.append(fea_idx_real)
                idx_sub = idx_sub + step

        n_fea_tmp = len(seperator_sub)
        seperator.append(seperator_sub)
        level_idx = level_idx + 1
    return seperator
