# ------------------------------------------
# LiNGAM Likelihood Ratios
# ------------------------------------------
#
#
#
# Hyvarinen & Smith, JMLR, 2013


import numpy as np
from scipy.stats import kurtosis
from sklearn.preprocessing import scale


def DiffEntropyHyv1998(x):
    """
    approximate the differential entropy as described in Hyvarinen (1998)
    """

    # define constants
    K1 = 79.047
    K2 = 7.4129
    gamma = .37457

    diffE = -1 * K1 * (np.log(np.cosh(x)).mean(0) - gamma) ** 2
    diffE -= K2 * (x * np.exp(-0.5 * x * x)).mean(0) ** 2

    return np.sum(1 * diffE)


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


class EntropyLR:
    """
    determine causal direction based on LR as described in Hyvarinen & Smith (2013)
    """

    def __init__(self, entropy='Hyv98', normalize=True):
        assert entropy in ['Hyv98', 'Kraskov04', 'WG98']
        self.entropy = entropy
        self.normalize = normalize

    def predict_proba(self, data):
        d = data.shape[1] // 2
        if d > 1:
            # only Hyv98 entropy supports high D variables for now
            assert self.entropy == 'Hyv98'
        x, y = data[:, :d], data[:, d:]
        # first scale
        x = scale(x)
        y = scale(y)
        # estimate correlation coef
        rho = np.corrcoef(x, y, rowvar=False)[:d, d:]

        functionDict = {'Hyv98': DiffEntropyHyv1998,
                        'Kraskov04': DiffEntropyKraskov2004,
                        'WG98': DiffEntropyWG1999}

        entFunc = functionDict[self.entropy]

        # compute LR of x->y
        resid = y - np.dot(rho, x.T).T
        if self.normalize:
            resid /= np.std(resid, axis=0)
        L_xy = -1 * entFunc(x) - entFunc(resid)

        # compute LR of x<-y
        resid = x - np.dot(rho, y.T).T
        if self.normalize:
            resid /= np.std(resid, axis=0)
        L_yx = -1 * entFunc(y) - entFunc(resid)

        LR = (L_xy - L_yx)
        causal_dir = 'x->y' if LR > 0 else 'y->x'

        return LR, causal_dir
