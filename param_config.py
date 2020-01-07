from h_utils import HBase
from fnn_solver import FnnSolveReg
from loss_utils import LossComputeBase
from rules import RuleBase
from partition import KFoldPartition
from loss_utils import LossFunc
from utils import Logger
import torch


class ParamConfig(object):
    def __init__(self, runs=1, kfolds=10, n_agents=25, nrules=10, nrules_s2=6):
        self.runs = runs  # Number of simulations
        self.kfolds = kfolds  # Number of folds

        # Network configuration
        # agents = [5:5: 25] # Nodes in the network(can be a vector for testing multiple sizes simultaneously)
        self.n_agents = n_agents
        self.connectivity = 0.25  # Connectivity in the networks(must be between 0 and 1)

        self.n_rules = nrules  # number of rules in stage 1
        self.n_rules_s2 = nrules_s2  # number of rules in stage 2
        self.dataset_list_full = ['abalone', 'airfoil', 'bikesharing',
                                  'blog', 'bodyfat', 'CASP', 'CCPP', 'eegDual_sub1',
                                  'housing', 'HRSS_anomalous_optimized', 'HRSS_anomalous_standard',
                                  'kc_house', 'motor_temperature', 'quake', 'skills',
                                  'statlib_calhousing', 'strength', 'telemonitoring', 'yacht']
        self.dataset_list = ['CASP']

        # set mu
        self.para_mu_current = 0
        para_mu_list = torch.linspace(-4, 4, 9)
        # para_mu_list = torch.linspace(-3, -1, 3)
        self.para_mu_list = torch.pow(10, para_mu_list).double()

        # initiate tools
        self.h_computer = HBase()
        self.fnn_solver = FnnSolveReg()
        self.loss_compute = LossComputeBase()
        self.loss_fun = LossFunc()
        self.rules = RuleBase()
        self.patition_strategy = KFoldPartition(self.kfolds)
        self.log = Logger()








