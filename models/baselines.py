### Baseline methods for comparison
#
#


import numpy as np
import scipy.optimize as sopt
import torch
import torch.nn as nn
from scipy.special import expit as sigmoid
from scipy.stats import kurtosis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, scale

from .notears_helper import LocallyConnected, LBFGSBScipy


# ------------------------------------------
# LiNGAM Likelihood Ratios
# ------------------------------------------
#
#
#
# Hyvarinen & Smith, JMLR, 2013


def DiffEntropyHyv1998(x):
    """
    approximate the differential entropy as described in Hyvarinen (1998)
    """

    # define constants
    K1 = 79.047
    K2 = 7.4129
    gamma = .37457

    diffE = -1 * K1 * (np.log(np.cosh(x)).mean() - gamma) ** 2
    diffE -= K2 * (x * np.exp(-0.5 * x * x)).mean() ** 2

    return 1 * diffE


def DiffEntropyKraskov2004(x):
    """
    approximate the differential entropy using equation (4) of Kraskov et al (2004)

    we note there are additive constants which we ignore as we are only interested in likelihood ratios !
    """

    x = scale(x)
    x = np.unique(x)  # computing log of differences, hence need to remove repeated values (other computing log 0)
    N = len(x)

    x = np.sort(x)

    return np.mean(np.log(N * np.diff(x)))


def DiffEntropyWG1999(x, m=1):
    """
    approximate the differential entropy using equation (3) of Wieczorkowski and Grzegorzewski (1999)

    """

    x = scale(x)
    x = np.unique(x)
    N = len(x)

    x = np.sort(x)

    return (1. / (N - m)) * np.sum(np.log(((N + 1.) / m) * (x[m:] - x[:(len(x) - m)])))


def cumulant_hyv13_ratio(x, y):
    """
    determine causal direction based on cumulants as described in
    section 2.6 of Hyvarinen & Smith (2013)
    """
    rho = np.corrcoef(x, y)[0, 1]  # estimate correlation coef

    kurtx = kurtosis(x)

    R = np.sign(kurtx) * rho * (np.mean(np.power(x, 3) * y - x * np.power(y, 3)))

    predictCausalDir = 'x->y' if R > 0 else 'y->x'
    return R, predictCausalDir


def base_entropy_ratio(x, y, entropy='Hyv98', normalize=True):
    """
    determine causal direction based on LR as described in Hyvarinen & Smith (2013)
    """

    # first scale
    x = scale(x)
    y = scale(y)

    # estimate correlation coef
    rho = np.corrcoef(x, y)[0, 1]

    assert entropy in ['Hyv98', 'Kraskov04', 'WG98']

    functionDict = {'Hyv98': DiffEntropyHyv1998,
                    'Kraskov04': DiffEntropyKraskov2004,
                    'WG98': DiffEntropyWG1999}

    entFunc = functionDict[entropy]

    # compute LR of x->y
    resid = y - rho * x
    if normalize:
        resid /= np.std(resid)
    L_xy = -1 * entFunc(x) - entFunc(resid)

    # compute LR of x<-y
    resid = x - rho * y
    if normalize:
        resid /= np.std(resid)
    L_yx = -1 * entFunc(y) - entFunc(resid)

    LR = (L_xy - L_yx)
    predictCausalDir = 'x->y' if LR > 0 else 'y->x'

    return LR, predictCausalDir


# ------------------------------------------
# Additive noise models (Hoyer at al 2009)
# ------------------------------------------
#
#
#
# this code is taken from cdt toolbox as it was failing


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


class ANM():
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

    .. note::
       Ref : Hoyer, Patrik O and Janzing, Dominik and Mooij, Joris M and Peters, Jonas and Schölkopf, Bernhard,
       "Nonlinear causal discovery with additive noise models", NIPS 2009
       https://papers.nips.cc/paper/3548-nonlinear-causal-discovery-with-additive-noise-models.pdf
    """

    def __init__(self):
        """Init the model."""
        super(ANM, self).__init__()

    def predict_proba(self, data, **kwargs):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
            dataset (tuple): Couple of np.ndarray variables to classify

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        a, b = data
        a = scale(a).reshape((-1, 1))
        b = scale(b).reshape((-1, 1))

        return self.anm_score(b, a) - self.anm_score(a, b)

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


# ------------------------------------------
# Regression error causal inference
# ------------------------------------------
#

class RECI():
    """
    This is heavily adapted from the CDT python toobox 

    """

    def __init__(self):
        """Init the model."""
        super(RECI, self).__init__()

    def predict_proba(self, data, form='linear', scale_input=False, d=3):
        """Prediction method for pairwise causal inference using the ANM model.

        Args:
             - data: np array, one column per variable
        - form: functional form, either linear of GP

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """

        x = data[:, 0]
        y = data[:, 1]
        if scale_input:
            x = scale(x).reshape((-1, 1))
            y = scale(y).reshape((-1, 1))
        else:
            # use min max scaler instead - as suggested by Blobaum et al (2018)
            x = MinMaxScaler().fit_transform(x.reshape((-1, 1)))
            y = MinMaxScaler().fit_transform(y.reshape((-1, 1)))

        return self.compute_residual(x, y, form=form, d=d) - self.compute_residual(y, x, form=form, d=d)

    def compute_residual(self, x, y, form='linear', d=3):
        """Compute the fitness score of the ANM model in the x->y direction.

        Args:
            x (numpy.ndarray): Variable seen as cause
            y (numpy.ndarray): Variable seen as effect

        Returns:
            float: ANM fit score
        """

        assert form in ['linear', 'GP', 'poly']

        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))

        if form == 'linear':
            # use linear regression
            res = LinearRegression().fit(x, y)
            residuals = y - res.predict(x)
            return np.median(residuals ** 2)
        elif form == 'poly':
            features = np.hstack([x ** i for i in range(1, d)])
            res = LinearRegression().fit(features, y)
            residuals = y - res.predict(features)
            return np.median(residuals ** 2)
        else:
            # use Gaussian process regssion
            # kernel = 1.0 * RBF() #+ WhiteKernel()
            x = scale(x)
            y = scale(y)
            gp = GaussianProcessRegressor().fit(x, y)
            residuals = y - gp.predict(x)
            return np.mean(residuals ** 2)


# ------------------------------------------
# NOTEARS (linear)
# ------------------------------------------
#
#
# code shamelessly taken from: 
# https://github.com/xunzheng/notears
#


def linear_notears_dir(x, y, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """
    infer the causal direction based on linear implementation of the NOTEARS
    algorithm presented by Zheng et al (2018, NuerIPS).

    Final changes (lines 97 onwards) apply to bivariate measures of causal direciton!
    
    Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        x (numpy.ndarray): Variable seen as cause
        y (numpy.ndarray): Variable seen as effect
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """

    # stack observations together
    X = np.vstack((x, y)).T

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        #     E = slin.expm(W * W)  # (Zheng et al. 2018)
        #     h = np.trace(E) - d
        M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0

    # get measure of causal direction
    R = np.abs(W_est[1, 0]) - np.abs(W_est[0, 1])
    predictCausalDir = 'x->y' if R < 0 else 'y->x'

    return R, predictCausalDir, W_est


# ------------------------------------------
# NOTEARS (nonlinear - implemented with MLP)
# ------------------------------------------
#
#
# code shamelessly taken from: 
# https://github.com/xunzheng/notears
#


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        M = torch.eye(d) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def nonlinear_notears_dir(X):
    """"""
    model = NotearsMLP(dims=[2, 2, 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    return W_est
