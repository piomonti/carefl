import numpy as np
import torch
from torch.utils.data import Dataset


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def gen_synth_causal_dat(nObs=100, causalFunc='square'):
    """
    generate causal data where one variable causes another 
    Inputs:
        - nObs: number of observations
        - causalFunc: specify causal function
    """

    causalFuncDict = {'linear': lambda x, n: 1 * x + n,
                      'hoyer2009': lambda x, n: x + (.5) * x * x * x + (n),
                      'nueralnet_l1': lambda x, n: sigmoid(sigmoid(np.random.normal(loc=1) * x) + n)
                      }

    # scale divided by np.sqrt(2) to ensure std of 1
    N = (np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(nObs, 2)))

    X = np.zeros((nObs, 2))
    X[:, 0] = N[:, 0]
    X[:, 1] = causalFuncDict[causalFunc](X[:, 0], N[:, 1])

    if np.random.uniform() < .5:
        mod_dir = 'y->x'
        X = X[:, [1, 0]]
    else:
        mod_dir = 'x->y'

    return X, mod_dir


# define dataset class

class CustomSyntheticDatasetDensity(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    # print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }


class CustomSyntheticDatasetDensityClasses(Dataset):
    def __init__(self, X, E, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.e = torch.from_numpy(E).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    # print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.e[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }


def to_one_hot(x, m=None):
    """"batch one hot"""
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh
