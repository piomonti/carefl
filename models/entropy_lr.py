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
