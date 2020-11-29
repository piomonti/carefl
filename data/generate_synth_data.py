import numpy as np
import torch
from torch.utils.data import Dataset


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def gen_synth_causal_dat(nObs=100, causalFunc='nueralnet_l1', noise_dist='laplace'):
    """
    generate causal data where one variable causes another
    Inputs:
        - nObs: number of observations
        - causalFunc: specify causal function
    """

    causalFuncDict = {'linear': lambda x, n: 1 * x + n,
                      'hoyer2009': lambda x, n: x + (.5) * x * x * x + (n),
                      'nueralnet_l1': lambda x, n: sigmoid(sigmoid(np.random.normal(loc=1) * x) + n),
                      'mnm': lambda x, n: sigmoid(np.random.normal(loc=1) * x) + .5 * x ** 2
                                          + sigmoid(np.random.normal(loc=1) * x) * n
                      }

    # scale divided by np.sqrt(2) to ensure std of 1
    if noise_dist == 'laplace':
        N = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(nObs, 2))
    elif noise_dist == 'gaussian':
        N = np.random.normal(loc=0, scale=1., size=(nObs, 2))
    elif noise_dist == 'cauchy':
        N = np.random.standard_cauchy(size=(nObs, 2))
    elif noise_dist == 'student':
        N = np.random.standard_t(df=5, size=(nObs, 2))
    else:
        raise ValueError(noise_dist)

    X = np.zeros((nObs, 2))
    X[:, 0] = N[:, 0]
    X[:, 1] = causalFuncDict[causalFunc](X[:, 0], N[:, 1])

    if np.random.uniform() < .5:
        mod_dir = 'y->x'
        X = X[:, [1, 0]]
    else:
        mod_dir = 'x->y'

    return X, mod_dir


def intervention_sem(n_obs, dim=4, seed=0, noise_dist='laplace',
                     random=True, shuffle=False, nonlin='poly', multiplicative=False):
    np.random.seed(seed)
    if dim == 4:
        # generate some 4D data according to the following SEM
        #   X_1 = N_1
        #   X_2 = N_2
        # if nonlinear == 'poly'
        #   X_3 = (X_1 + c_0*X_2^3) +/* N_3   -  c_0 random coeff
        #   X_4 = (c_1*X_1^2 - X_2) +/* N_4   -  c_1 random coeff
        # else if nonlinear == 'sigmoid'
        #   X_3 = sigmoid(sigmoid(c_1 * X_1 + c_2 * X_2) + N_3)  -- c_1 and c_2 random
        #   X_4 = sigmoid(sigmoid(c_3 * X_1) + c_4 * X_2^3 + N_4)  -- c_3 and c_4 random
        # causes
        if noise_dist == 'laplace':
            X_1, X_2 = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_obs, 2)).T
        elif noise_dist == 'gaussian':
            X_1, X_2 = np.random.normal(loc=0, scale=1., size=(n_obs, 2)).T
        elif noise_dist == 'cauchy':
            X_1, X_2 = np.random.standard_cauchy(size=(n_obs, 2)).T
        elif noise_dist == 'student':
            X_1, X_2 = np.random.standard_t(df=5, size=(n_obs, 2)).T
        else:
            raise ValueError(noise_dist)
        # effects
        if nonlin == 'poly':
            coeffs = np.random.uniform(.1, .9, 2) if random else [.5, .5]
            X_3 = X_1 + coeffs[0] * (X_2 * X_2 * X_2)
            X_4 = -X_2 + coeffs[1] * (X_1 * X_1)
            if multiplicative:
                X_3 *= np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
                X_4 *= np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
            else:
                X_3 += np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
                X_4 += np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        elif nonlin == 'sigmoid':
            N_3 = np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
            N_4 = np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
            coeffs = np.random.normal(loc=1, size=4) if random else [.5, .5, .5, .5]
            X_3 = sigmoid(sigmoid(coeffs[0] * X_1 + coeffs[1] * X_2) + N_3)
            X_4 = sigmoid(sigmoid(coeffs[2] * X_1) + coeffs[3] * X_2 * X_2 * X_2 + N_4)
        else:
            raise ValueError('Unknown nonlin argument: {}'.format(nonlin))
        X = np.vstack((X_1, X_2, X_3, X_4)).T
        # create the adjacency matrix
        dag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
        if shuffle:
            if np.random.uniform() < .5:
                X = X[:, [2, 3, 0, 1]]
                dag = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        return X, coeffs, dag
    elif dim == 20:
        # generate a 20D SEM:
        #   X_1 ... X_10 = N_1 ... N_10
        #   c_i random
        #   j>10, X_j = f(X_<=10, N_j), where
        #   f randomly drawn from one of the following:
        #       sigmoid(sigmoid(sum_i c_i * X_i) + N_j)
        #       sigmoid(sigmoid(sum_i<6 c_i * X_i) + N_j)
        #       sigmoid(sum_i>5 c_i * sigmoid(X_i)^{i-5} + N_j))
        # effects
        if noise_dist == 'laplace':
            X = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_obs, 10))
        elif noise_dist == 'gaussian':
            X = np.random.normal(loc=0, scale=1., size=(n_obs, 10))
        elif noise_dist == 'cauchy':
            X = np.random.standard_cauchy(size=(n_obs, 10))
        elif noise_dist == 'student':
            X = np.random.standard_t(df=5, size=(n_obs, 10))
        else:
            raise ValueError(noise_dist)
        # causes:
        coeffs = np.random.normal(loc=1, size=(10, 10)) if random else .5 * np.ones((10, 10))
        dag = np.zeros((20, 20))
        Y = np.zeros((n_obs, 10))
        N = np.random.laplace(0, 1 / np.sqrt(2), size=(n_obs, 10))
        for i in range(10):
            dice = np.random.randint(0, 3)
            if dice == 0:
                Y[:, i] = sigmoid(sigmoid(np.sum(coeffs[i] * X, axis=1)) + N[:, i])
                dag[10 + i, :10] = 1
            elif dice == 1:
                Y[:, i] = sigmoid(sigmoid(np.sum(coeffs[i, :5] * X[:, :5], axis=1)) + N[:, i])
                dag[10 + i, :5] = 1
            else:
                Y[:, i] = sigmoid(np.sum(coeffs[i, 5:] * sigmoid(X[:, 5:]) ** np.arange(5), axis=1) + N[:, i])
                dag[10 + i, 5:10] = 1
        data = np.hstack((X, Y))
        if shuffle:
            if np.random.uniform() < .5:
                data = np.hstack((Y, X))
                dag = dag.T
        return data, coeffs, dag
    else:
        raise NotImplementedError('will be implemented soon')


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
