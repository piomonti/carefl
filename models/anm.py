# ------------------------------------------
# Additive noise models (Hoyer at al 2009)
# ------------------------------------------
#
#
#
# this code is taken from cdt toolbox as it was failing

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import scale


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

    def __init__(self):
        """Init the model."""
        self.dag = None
        self.parents = None
        self.gps = None
        self.dim = None
        self.roots = None
        self.effects = None

    def predict_proba(self, data):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
            data (np.ndarray): 2-column np.ndarray of variables to classify

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        a, b = data[:, 0], data[:, 1]
        a = scale(a).reshape((-1, 1))
        b = scale(b).reshape((-1, 1))

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
        gp = GaussianProcessRegressor().fit(x, y)
        y_predict = gp.predict(x)
        indepscore = normalized_hsic(y_predict - y, x)

        return indepscore

    def fit_to_sem(self, data, dag):
        # the dag is an adjacency matrix that describes the sem followed by the data
        parents, roots, sorted_effects = sorted_roots_effects_from_dag(dag)
        gps = {}
        for u, v in parents.items():
            gp = GaussianProcessRegressor().fit(data[:, v], data[:, u])
            gps[u] = gp
        self.dag = dag
        self.parents = parents
        self.roots = roots
        self.effects = sorted_effects
        self.gps = gps
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
            x[:, u] = self.gps[u].predict(x.copy()[:, self.parents[u]])
            x_est[:, u] = self.gps[u].predict(x_est.copy()[:, self.parents[u]])

        return x.mean(0).reshape((1, self.dim)), x_est



def sorted_roots_effects_from_dag(dag):
    """
    find the roots (causes) and effects in a dag and return them sorted
    :param dag (np.ndarray): an adjacency matric that describes a DAG
    :return (tuple): return parents, roots, sorted_effects where:
        - parents: a dictionary whose keys are the effects, and values are their parents in the dag
        - roots: the roots of the dag
        - sorted_effects: a sorted list of the keys in parents, where the sorting follows the permutation that describes
            the ordering of the dag
    roots + sorted_effects together contain all variables sorted according to the SEM
    """
    parents = {}
    # for each variable x_i, get its parents in the dag
    for u, v in zip(*np.where(dag != 0)):
        if u not in parents.keys():
            parents[DicIdx(u)] = []
        parents[u].append(v)
    for u in parents.keys():
        u.set_dict(parents)
    roots = [u for u in np.arange(dag.shape[0]) if u not in parents.keys()]
    return parents, roots, sorted(parents)


class DicIdx:
    """an int that represents a dictionary index, where __lt__ function uses the dictionary values for comparison"""
    def __init__(self, x, dic=None):
        self.x = x
        self.dic = dic

    def __lt__(self, other):
        return self.x in other.dic[other.x] if other.dic is not None else False

    def __repr__(self):
        return str(int(self.x))

    def __eq__(self, other):
        if type(other) in [float, int, np.int64, np.int_]:
            return self.x == other
        elif type(other) == DicIdx:
            return self.x == other.x
        else:
            raise TypeError("Can't compare to {} or type {}".format(other, type(other)))

    def __hash__(self):
        return int(self.x)

    def set_dict(self, dic):
        self.dic = dic
