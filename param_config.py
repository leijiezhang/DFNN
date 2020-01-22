from h_utils import HBase, HNormal
from fnn_solver import FnnSolveBase, FnnSolveReg, FnnSolveCls
from rules import RuleBase, RuleKmeans, RuleFuzzyCmeans
from partition import PartitionStrategy, KFoldPartition
from loss_utils import LossFunc, NRMSELoss, RMSELoss, MSELoss, Map, LikelyLoss
from utils import Logger
from neuron import Neuron
from seperator import FeaSeperator
from model import NetBase, TreeNet, TreeFNNet
import yaml


class ParamConfig(object):
    def __init__(self, n_run=1, n_kfolds=10, n_agents=25, nrules=10):
        self.n_run = n_run  # Number of simulations
        self.n_kfolds = n_kfolds  # Number of folds

        # Network configuration
        self.n_agents = n_agents
        self.n_agents_list = []

        self.connectivity = 0.25  # Connectivity in the networks(must be between 0 and 1)

        self.n_rules = nrules  # number of rules in stage 1
        self.n_rules_list = []

        self.dataset_list_all = []
        self.dataset_list = ['CASP']

        # set mu
        self.para_mu_current = 0
        self.para_mu_list = []
        # set rho
        self.para_rho = 1

        # initiate tools
        self.h_computer: HBase = None
        self.fnn_solver: FnnSolveBase = None
        self.loss_fun: LossFunc = None
        self.rules: RuleBase = None
        self.patition_strategy: PartitionStrategy = None

        # set feature seperator
        self.fea_seperator: FeaSeperator = None
        self.seperator_type = None
        self.window_size = 0
        self.n_level = 0
        # for random pick
        self.n_repeat = 0
        # for slide window
        self.step = 0

        # initiate net
        self.net: NetBase = None
        self.log = None

    def config_parse(self, config_name):
        config_dir = f"./configs/{config_name}.yaml"
        config_file = open(config_dir)
        config_content = yaml.load(config_file, Loader=yaml.FullLoader)
        self.n_run = config_content['n_run']
        self.n_kfolds = config_content['n_kfolds']

        self.n_agents = config_content['n_agents']
        self.n_agents_list = config_content['n_agents_list']

        self.n_rules = config_content['n_rules']
        self.n_rules_list = config_content['n_rules_list']

        self.dataset_list_all = config_content['dataset_list_all']
        self.dataset_list = config_content['dataset_list']

        self.para_mu_current = config_content['mu_current']
        self.para_mu_list = config_content['mu_list']

        self.para_rho = config_content['rho']

        # set h_computer
        if config_content['h_computer'] == 'base':
            self.h_computer = HBase()
        elif config_content['h_computer'] == 'normal':
            self.h_computer = HNormal()

        # set fnn_solver
        if config_content['fnn_solver'] == 'base':
            self.fnn_solver = FnnSolveBase()
        elif config_content['fnn_solver'] == 'normal':
            self.fnn_solver = FnnSolveReg()
        elif config_content['fnn_solver'] == 'sigmoid':
            self.fnn_solver = FnnSolveCls()

        # set loss_fun:loss function
        if config_content['loss_fun'] == 'base':
            self.loss_fun = LossFunc()
        elif config_content['loss_fun'] == 'rmse':
            self.loss_fun = RMSELoss()
        elif config_content['loss_fun'] == 'nrmse':
            self.loss_fun = NRMSELoss()
        elif config_content['loss_fun'] == 'mse':
            self.loss_fun = MSELoss()
        elif config_content['loss_fun'] == 'map':
            self.loss_fun = Map()
        elif config_content['loss_fun'] == 'likely':
            self.loss_fun = LikelyLoss()

        # set rules
        if config_content['rules'] == 'base':
            self.rules = RuleBase()
        elif config_content['rules'] == 'kmeans':
            self.rules = RuleKmeans()
        elif config_content['rules'] == 'fuzzyc':
            self.rules = RuleFuzzyCmeans()

        # set patition_strategy
        if config_content['rules'] == 'kmeans':
            self.patition_strategy = KFoldPartition(self.n_kfolds)

        # set feature seperator
        self.seperator_type = config_content['feature_seperator']
        if config_content['feature_seperator'] == 'slice_window':
            self.window_size = config_content['window_size']
            self.step = config_content['step']
            self.n_level = config_content['n_level']
        elif config_content['feature_seperator'] == 'random_pick':
            self.window_size = config_content['window_size']
            self.n_repeat = config_content['n_repeat_select']
            self.n_level = config_content['n_level']

        # set logger to decide whether write log into files
        if config_content['log_to_file'] == 'false':
            self.log = Logger()
        else:
            self.log = Logger(True)

        # set model
        neuron = Neuron(self.rules, self.h_computer, self.fnn_solver)
        if config_content['model'] == 'base':
            self.net = NetBase(neuron)
        elif config_content['model'] == 'fuzzy_tree_fn':
            self.net = TreeFNNet(neuron)
        elif config_content['model'] == 'fuzzy_tree':
            self.net = TreeNet(neuron)

    def update_seperator(self, data_name):
        # set feature seperator
        self.fea_seperator = FeaSeperator(data_name)
        if self.seperator_type == 'slice_window':
            self.fea_seperator.set_seperator_by_slice_window(self.window_size, self.step,
                                                             self.n_level)
        elif self.seperator_type == 'random_pick':
            self.fea_seperator.set_seperator_by_random_pick(self.window_size, self.n_repeat,
                                                            self.n_level)
        elif self.seperator_type == 'no_seperate':
            self.fea_seperator.set_seperator_by_no_seperate()
