from dataset import Dataset
import scipy.io as sio
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
        self.dataset_list = ['abalone', 'airfoil', 'bikesharing',
                             'blog', 'bodyfat', 'CASP', 'CCPP', 'eegDual_sub1_format',
                             'housing', 'HRSS_anomalous_optimized', 'HRSS_anomalous_standard',
                             'kc_house', 'motor_temperature', 'quake', 'skills',
                             'statlib_calhousing', 'strength', 'telemonitoring', 'yacht']
        self.para_mu = 0

    def load_data(self, dataset_str):
        dir_dataset = f"./datasets/{dataset_str}.pt"
        load_data = torch.load(dir_dataset)
        dataset_name = load_data['name']
        x = load_data['X']
        y = load_data['Y']
        task = load_data['task']
        dataset = None
        if dataset_name == 'airfoil':

            dataset = Dataset(dataset_name, x[1:1500, :], y[1:1500, :], task)  # Load and preprocess dataset Data
            # dataset = Dataset(dataset_name, x(1:1500, 1: 4), task, y(1: 1500,:)) # Load and preprocess dataset Data
            # dataset = Dataset(dataset_name, x(1:1500, 5), task, y(1: 1500,:)) # Load and preprocess dataset Data
            dataset.normalize(-1, 1)
            self.para_mu = 0.1
            #  # ==PCA===
            #  [coeff, score, latent]=pca(dataset.x)
            #  la# tent_check = 100 * cumsum(latent). / sum(latent)
            #  dataset.x = score(:, 1: 4)

        elif dataset_name == 'abalone':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'bikesharing':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'blog':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'housing':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'CCPP':
            dataset = Dataset(dataset_name, x[1:9500, :], y[1:9500, :], task)
            #  dataset = Dataset(dataset_name, x[1:9500, 1: 3], task, y[1: 9500,:])
            dataset.normalize(-1, 1)
            self.para_mu = 0.01
            #  #  ==PCA===
            #  [coeff, score, latent]=pca(dataset.x)
            #  la# tent_check = 100 * cumsum(latent). / sum(latent)
            #  dataset.x = score(:, 1: 3)

        elif dataset_name == 'CASP':
            dataset = Dataset(dataset_name, x, y, task)
            #  dataset = Dataset(dataset_name, x[1:2000, : ], task, y[1: 2000,:])
            # dataset = Dataset(dataset_name, x[1:2000, 1: 4], task, y[1: 2000,:])
            #  dataset = Dataset(dataset_name, x[1:2000, 5: 9], task, y[1: 2000,:])
            dataset.normalize(-1, 1)
            self.para_mu = 0.001
            # # ==PCA===
            #  [coeff, score, latent]=pca(dataset.x)
            #  la# tent_check = 100 * cumsum(latent). / sum(latent)
            #  dataset.x = score(:, 1: 3)

        elif dataset_name == 'HRSS_anomalous_optimized':
            dataset = Dataset(dataset_name, x[1:6500, :], y[1:6500, :], task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.01

        elif dataset_name == 'HRSS_anomalous_standard':
            dataset = Dataset(dataset_name, x[1:2244, :], y[1:2244, :], task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.01

        elif dataset_name == 'kc_house':
            dataset = Dataset(dataset_name, x[1:21613, :], y[1:21613, :], task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.001

        elif dataset_name == 'motor_temperature':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.001

        elif dataset_name == 'motor_temperature_2':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.001

        elif dataset_name == 'motor_temperature_3':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.001

        elif dataset_name == 'motor_temperature_4':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(-1, 1)
            self.para_mu = 0.001

        elif dataset_name == 'eegDual_sub1':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'quake':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'skills':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'strength':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'telemonitoring':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001

        elif dataset_name == 'yacht':
            dataset = Dataset(dataset_name, x, y, task)
            dataset.normalize(0, 1)
            self.para_mu = 0.001


        return dataset






