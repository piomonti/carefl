# trial counterfactual predictions of flow models
#
#

import os

import numpy as np
# load data generating code:
import pylab as plt
import seaborn as sns

# load flows
from models.affine_flow_cd import BivariateFlowLR


# plt.ion()


def counterfactuals(results_dir=''):
    # we generate the same SEM as in the intervention example

    nObs = 2500
    np.random.seed(0)

    # causes
    X_0 = np.random.laplace(0, 1 / np.sqrt(2), size=nObs)
    X_1 = np.random.laplace(0, 1 / np.sqrt(2), size=nObs)

    # effects
    X_2 = X_0 + 0.5 * (X_1 * X_1 * X_1) + np.random.laplace(0, 1 / np.sqrt(2), size=nObs)
    X_3 = -X_1 + 0.5 * (X_0 * X_0) + np.random.laplace(0, 1 / np.sqrt(2), size=nObs)

    X2var = X_2.std()
    X3var = X_3.std()

    X_2 /= X2var
    X_3 /= X3var

    dat = np.vstack((X_0, X_1, X_2, X_3)).T

    mod = BivariateFlowLR(n_layers=None, n_hidden=None, prior_dist='laplace', epochs=500, opt_method='scheduling')
    mod.fit_to_sem(dat, n_layers=5, n_hidden=10)

    ### now we run some CF trials:
    # generate latent disturbances:
    np.random.seed(1)
    N = np.random.laplace(0, 1, size=(1, 4))

    # generate obs data:

    def genObs(N):
        """
        N are the latents here
        """

        X_0 = N[0, 0]
        X_1 = N[0, 1]
        X_2 = (X_0 + .5 * X_1 * X_1 * X_1 + N[0, 2]) / X2var
        X_3 = (-X_1 + .5 * X_0 * X_0 + N[0, 3]) / X3var

        return np.array([X_0, X_1, X_2, X_3]).reshape((1, 4))

    xObs = genObs(N)

    # get CF prediction for x0=-1
    mod.predict_counterfactual(x_obs=xObs, cf_val=1, iidx=0)

    # compare to true
    N_CF = np.copy(N)
    N_CF[0, 0] = 1
    xCF_true = genObs(N_CF)

    # make a plot
    # this plot shows counterfactual prediction for X_3 give X_0 goes from -3 to 3
    xvals = np.arange(-3, 3, .1)

    N = np.array([2, 1.5, 1.4, -1]).reshape((1, 4))
    xObs = genObs(N)
    # this is the random value of 4D random varibale x we observe. Now we plot the counterfactual given x_0 had been other values

    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 8))
    # sns.diverging_palette(255, 133, l=60, n=7, center="dark") )#  sns.color_palette("coolwarm", 6) )

    # ----

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(r'Counterfactual predictions', fontsize=22)

    xCF_true = [genObs(np.hstack((x, N[0, 1:])).reshape((1, 4)))[0, 3] for x in xvals]
    xCF_pred = [mod.predict_counterfactual(x_obs=xObs, cf_val=x, iidx=0)[0, 3] for x in xvals]

    ax1.plot(xvals, xCF_true, label=r'True $\mathbb{E} [{X_4}_{X_1 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax1.plot(xvals, xCF_pred, label=r'Predicted $\mathbb{E} [{X_4}_{X_1 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax1.legend(loc=1, fontsize=15)
    ax1.set_xlabel(r'Value of counterfactual variable, $X_1=\alpha$', fontsize=18)
    ax1.set_ylabel(r'Predicted value of $X_4$', fontsize=18)

    xCF_true = [genObs(np.hstack((N[0, 0], x, N[0, 2:])).reshape((1, 4)))[0, 2] for x in xvals]
    xCF_pred = [mod.predict_counterfactual(x_obs=xObs, cf_val=x, iidx=1)[0, 2] for x in xvals]

    ax2.plot(xvals, xCF_true, label=r'True $\mathbb{E} [{X_3}_{X_2 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax2.plot(xvals, xCF_pred, label=r'Predicted $\mathbb{E} [{X_3}_{X_2 \leftarrow \alpha} (n) ] $', linewidth=3,
             linestyle='-.')
    ax2.legend(loc='best', fontsize=15)
    ax2.set_xlabel(r'Value of counterfactual variable, $X_2=\alpha$', fontsize=18)
    ax2.set_ylabel(r'Predicted value of $X_3$', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.925)
    plt.savefig(os.path.join(results_dir, 'counterfactuals_4d.pdf'), dpi=300)
