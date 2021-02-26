# ------------------------------------------
# Additive noise models (Hoyer at al 2009)
# ------------------------------------------

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from nflib.nets import MLP1layer, MLP4


def rbf_dot2(p1, p2, deg):
    if p1.ndim == 1:
        p1 = p1[:, np.newaxis]
        p2 = p2[:, np.newaxis]

    size1 = p1.shape
    size2 = p2.shape

    G = np.sum(p1 * p1, axis=1)[:, np.newaxis]
    H = np.sum(p2 * p2, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(p1, p2.T)
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def rbf_dot(X, deg):
    # Set kernel size to median distance between points, if no kernel specified
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def FastHsicTestGamma(X, Y, sig=np.array([-1, -1]), maxpnt=500):
    """This function implements the HSIC independence test using a Gamma approximation
     to the test threshold. Use at most maxpnt points to save time.

    :param X: contains dx columns, m rows. Each row is an i.i.d sample
    :param Y: contains dy columns, m rows. Each row is an i.i.d sample
    :param sig: [0] (resp [1]) is kernel size for x(resp y) (set to median distance if -1)
    :return: test statistic

    """

    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        #       indx = np.r_[0:maxpnt]
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))

    K = rbf_dot(Xm, sig[0])
    L = rbf_dot(Ym, sig[1])

    Kc = np.dot(H, np.dot(K, H))
    Lc = np.dot(H, np.dot(L, H))

    testStat = (1.0 / m) * (Kc.T * Lc).sum()
    if ~np.isfinite(testStat):
        testStat = 0

    return testStat


def normalized_hsic(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    h = FastHsicTestGamma(x, y)

    return h


class ANM:
    """ANM algorithm.

    **Description**: The Additive noise model is one of the most popular
    approaches for pairwise causality. It bases on the fitness of the data to
    the additive noise model on one direction and the rejection of the model
    on the other direction.

    **Data Type**: Continuous

    **Assumptions**: Assuming that :math:`x\\rightarrow y` then we suppose that
    the data follows an additive noise model, i.e. :math:`y=f(x)+E`.
    E being a noise variable and f a deterministic function.
    The causal inference bases itself on the independence
    between x and e.
    It is proven that in such case if the data is generated using an additive noise model, the model would only be able
    to fit in the true causal direction.

    **Multivariate case**: This class also supports interventions in a multivariate case, where we suppose that
    the interventional variable is a root variable (thus equal to a noise variable as per the ANM model)

    .. note::
       Ref : Hoyer, Patrik O and Janzing, Dominik and Mooij, Joris M and Peters, Jonas and Schölkopf, Bernhard,
       "Nonlinear causal discovery with additive noise models", NIPS 2009
       https://papers.nips.cc/paper/3548-nonlinear-causal-discovery-with-additive-noise-models.pdf
    """

    def __init__(self, method='gp', config=None):
        """Init the model."""
        assert method in ['gp', 'linear', 'nn']
        self.method = method
        self.dag = None
        self.parents = None
        self.regs = None
        self.dim = None
        self.roots = None
        self.effects = None
        self.config = config  # only for when method == 'nn'

    def _get_regressor(self):
        if self.method.lower() == 'gp':
            return GaussianProcessRegressor()
        elif self.method.lower() == 'nn':
            torch.manual_seed(self.config.training.seed)
            return NNRegressor(nh=self.config.flow.nh, net_class=self.config.flow.net_class,
                               n_epochs=self.config.training.epochs, lr=self.config.optim.lr,
                               beta1=self.config.optim.beta1)
        else:
            return LinearRegression()

    def predict_proba(self, data):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
            data (np.ndarray): 2-column np.ndarray of variables to classify

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        d = data.shape[1] // 2
        a, b = data[:, :d], data[:, d:]
        a = scale(a).reshape((-1, d))
        b = scale(b).reshape((-1, d))

        p = self.anm_score(b, a) - self.anm_score(a, b)
        causal_dir = 'x->y' if p > 0 else 'y->x'
        return p, causal_dir

    def anm_score(self, x, y):
        """Compute the fitness score of the ANM model in the x->y direction.

        Args:
            x (numpy.ndarray): Variable seen as cause
            y (numpy.ndarray): Variable seen as effect

        Returns:
            float: ANM fit score
        """
        Reg = self._get_regressor()
        reg = Reg.fit(x, y)
        y_predict = reg.predict(x)
        indepscore = normalized_hsic(y_predict - y, x)

        return indepscore

    def fit_to_sem(self, data, dag):
        """
        Fits an ANM model to an SEM described by the dag adjecency matrix
        :param data: samples to fit to the SEM
        :param dag: adjacency matrix describing the DAG associated with the SEM: dag_ij != 0 iif x_j -> x_i in the DAG
        :return: None -- updated the internal fields of the class
        """
        # the dag is an adjacency matrix that describes the sem followed by the data
        parents, roots, sorted_effects = sorted_roots_effects_from_dag(dag)
        regs = {}
        for u, v in parents.items():
            Reg = self._get_regressor()
            reg = Reg.fit(data[:, v], data[:, u])
            regs[u] = reg
        self.dag = dag
        self.parents = parents
        self.roots = roots
        self.effects = sorted_effects
        self.regs = regs
        self.dim = data.shape[1]

    def predict_intervention(self, x0_val, n_samples=100, iidx=0):
        # we suppose we only intervene on roots
        assert iidx in self.roots
        # generate a laplace noise vector which will be transformed into final sample
        x = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=(n_samples, self.dim))
        x_est = np.zeros((1, self.dim))
        # in an ANM's SEM, roots are equal to their latent disturbance
        x[:, iidx] = x_est[:, iidx] = x0_val
        for u in self.effects:
            x[:, u] = self.regs[u].predict(x.copy()[:, self.parents[u]])
            x_est[:, u] = self.regs[u].predict(x_est.copy()[:, self.parents[u]])

        return x.mean(0).reshape((1, self.dim)), x_est


def sorted_roots_effects_from_dag(dag):
    """
    find the roots (causes) and effects in a dag and return them sorted
    :param dag : (np.ndarray) an adjacency matrix that describes a DAG
    :return: (tuple) return parents, roots, sorted_effects where:
        - parents: (dict) a dictionary whose keys are the effects, and values are their parents in the dag
        - roots: (list) the roots of the dag
        - sorted_effects: (list) a sorted list of the keys in parents, where the sorting follows the permutation that
            describes the ordering of the dag
    roots + sorted_effects together contain all variables sorted according to the SEM
    """
    parents = {}
    # for each variable x_i, get its parents in the dag (dag_ij != 0)
    for u, v in zip(*np.where(dag != 0)):
        if u not in parents.keys():
            parents[DictIdx(u)] = []
        parents[u].append(v)
    for u in parents.keys():
        u.set_dict(parents)
    roots = [u for u in np.arange(dag.shape[0]) if u not in parents.keys()]
    sorted_effects = [u.x for u in sorted(parents)]
    parents = {u.x: v for u, v in parents.items()}
    return parents, roots, sorted_effects


class DictIdx:
    """an int that represents a dictionary index, where __lt__ function uses the dictionary values for comparison"""

    def __init__(self, x, dic=None):
        self.x = x
        self.dic = dic

    def __lt__(self, other):
        return self.x in other.dic[other.x] if other.dic else False

    def __repr__(self):
        return str(int(self.x))

    def __eq__(self, other):
        if type(other) in [float, int, np.int64, np.int_]:
            return self.x == other
        elif type(other) == DictIdx:
            return self.x == other.x
        else:
            raise TypeError("Can't compare to {} of type {}".format(other, type(other)))

    def __hash__(self):
        return int(self.x)

    def set_dict(self, dic):
        self.dic = dic


class NNRegressor:
    def __init__(self, nh, net_class='mlp', n_epochs=200, lr=0.001, beta1=0.9):
        self.net_class = net_class
        self.nh = nh
        self.n_epochs = n_epochs
        self.lr = lr
        self.beta1 = beta1
        self.dim = 1
        self._init_net()

    def _init_net(self):
        if self.net_class.lower() == 'mlp':
            self.net = MLP1layer(self.dim, self.dim, self.nh)
        elif self.net_class.lower() == 'mlp4':
            self.net = MLP4(self.dim, self.dim, self.nh)
        else:
            raise ValueError(self.net_class)

    def _update_dim(self, x):
        if x.ndim == 1:
            self.dim = 1
        else:
            self.dim = x.shape[-1]

    def fit(self, x, y):
        self._update_dim(x)
        self._init_net()

        x, y = torch.from_numpy(x.astype(np.float32)).reshape((-1, self.dim)), torch.from_numpy(
            y.astype(np.float32)).reshape((-1, self.dim))
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(self.beta1, 0.999))
        self.net.train()
        for _ in range(self.n_epochs):
            y_p = self.net(x)
            loss = loss_func(y_p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x):
        x = torch.from_numpy(x.astype(np.float32)).reshape((-1, self.dim))
        return self.net(x).detach().cpu().numpy().squeeze()
