# trial interventional predictions of flow models
#
#

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# load data generating code:
from data.generate_synth_data import gen_synth_causal_dat
# load flows
from models.affine_flow_cd import BivariateFlowLR


def intervention_sem(n_obs, dim=4, seed=0, random=True):
    if dim == 4:
        # generate some 4D data according to the following SEM
        #
        # X_0 = N_0
        # X_1 = N_1
        # X_2 = X_0 + c_0*X_1^3 + N_2   -  c_0 random coeff
        # X_3 = c_1*X_0^2 - X_1 + N_3   -  c_1 random coeff
        np.random.seed(seed)
        # causes
        X_0 = np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        X_1 = np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        # effects
        coeffs = np.random.uniform(.1, .9, 2) if random else [.5, .5]
        X_2 = X_0 + coeffs[0] * (X_1 * X_1 * X_1) + np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        X_3 = -X_1 + coeffs[1] * (X_0 * X_0) + np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        return np.vstack((X_0, X_1, X_2, X_3)).T, coeffs
    else:
        raise NotImplementedError('will be implemented soon')


def intervention4d(results_dir=''):
    results = {"x2": [], "x3": [], "x2e": [], "x3e": []}
    n_obs_list = [25, 100, 250, 500, 1000, 2000, 5000]
    for n_obs in n_obs_list:
        # generate coeffcients for equation (12), and data from that SEM
        dat, coeffs = intervention_sem(n_obs, dim=4, seed=0, random=False)
        # fit to an affine autoregressive flow
        mod = BivariateFlowLR(n_layers=None, n_hidden=None, prior_dist='laplace', epochs=500, opt_method='scheduling')
        mod.fit_to_sem(dat, n_layers=5, n_hidden=10)
        # intervene on x_0 and get a sample of {x | do(x_0=a)} for a in [-3, 3]
        avals = np.arange(-3, 3, .1)
        x_int_sample = []
        x_int_exp = []
        for a in avals:
            res = mod.predict_intervention(a, n_samples=20, d=4, iidx=0)
            x_int_sample.append(res[0].mean(axis=0))
            x_int_exp.append(res[1].mean(axis=0))
        x_int_sample = np.array(x_int_sample)
        x_int_exp = np.array(x_int_exp)
        # compute the MSE between the true E[x_2|x_0=a] to the empirical expectation from the sample
        # we know that the true E[x_2|x_0=a] = a
        mse_x2 = np.mean((x_int_sample[:, 2] - avals) ** 2)
        mse_x2e = np.mean((x_int_exp[:, 2] - avals) ** 2)
        # do the same for x_3; true E[x_3|x_0=a] = c_1*a^2
        mse_x3 = np.mean((x_int_sample[:, 3] - coeffs[1] * avals ** 2) ** 2)
        mse_x3e = np.mean((x_int_exp[:, 3] - coeffs[1] * avals ** 2) ** 2)
        # store results
        results["x2"].append(mse_x2)
        results["x3"].append(mse_x3)
        results["x2e"].append(mse_x2e)
        results["x3e"].append(mse_x3e)

    # plot the MSEs
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 8))
    plt.figure()
    plt.plot(n_obs_list, results["x2"], 'o-', label=r'$X_2$', color=sns.color_palette("muted", 8)[2], linewidth=2,
             alpha=.8)
    plt.plot(n_obs_list, results["x3"], 'o-', label=r'$X_3$', color=sns.color_palette("muted", 8)[4], linewidth=2,
             alpha=.8)
    plt.xlabel(r'Sample size', fontsize=12)
    plt.ylabel(r'MSE', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'intervention_mse_sample.pdf'), dpi=300)
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 8))
    plt.figure()
    plt.plot(n_obs_list, results["x2e"], 'o-', label=r'$X_2$', color=sns.color_palette("muted", 8)[2], linewidth=2,
             alpha=.8)
    plt.plot(n_obs_list, results["x3e"], 'o-', label=r'$X_3$', color=sns.color_palette("muted", 8)[4], linewidth=2,
             alpha=.8)
    plt.xlabel(r'Sample size', fontsize=12)
    plt.ylabel(r'MSE', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'intervention_mse_exp.pdf'), dpi=300)


def intervention(dim=4, results_dir=''):
    if dim == 4:
        # higher D examples
        n_obs = 1500
        dat, coeffs = intervention_sem(n_obs, dim=dim, seed=0, random=False)

        # example plots:
        plt.scatter(dat[:, 1], dat[:, 2])
        plt.scatter(dat[:, 0], dat[:, 3])

        mod = BivariateFlowLR(n_layers=None, n_hidden=None, prior_dist='laplace', epochs=500, opt_method='scheduling')
        mod.fit_to_sem(dat, n_layers=5, n_hidden=10)

        # -----

        # so the distribution of X_2 | do(X_0=x) should be linear
        xvals = np.arange(-3, 3, .05)

        sns.set_style("whitegrid")
        sns.set_palette(sns.color_palette("muted", 8))
        # sns.diverging_palette(255, 133, l=60, n=7, center="dark") )#  sns.color_palette("coolwarm", 6) )

        plt.figure()
        plt.plot(xvals, xvals, label='True', color=sns.color_palette("muted", 8)[2], linewidth=2, alpha=.8)
        plt.plot(xvals, coeffs[1] * xvals * xvals, color=sns.color_palette("muted", 8)[2], linewidth=2, alpha=.8)

        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=10, d=4)[0, 2] for x in xvals]),
                 linewidth=3, linestyle='-.',
                 label=r'Predicted $\mathbb{E} [X_3| do(X_1=\alpha)]$')  # remove 0 indexing in legend
        # and distribution of X_3 | do(X_0=x) should be quadratic
        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=10, d=4)[0, 3] for x in xvals]),
                 linewidth=3, linestyle='-.',
                 label=r'Predicted $\mathbb{E} [X_4| do(X_1=\alpha)]$')  # remove 0 indexing in legend

        plt.legend(fontsize=12)
        plt.xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=12)
        plt.ylabel(r'Predicted value of $X_3$ or $X_4$', fontsize=12)
        plt.title(r'interventional predictions under $do(X_1=\alpha)$', fontsize=15)
        plt.savefig(os.path.join(results_dir, 'intervention_4d_1.pdf'), dpi=300)

        # -----

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9))
        fig.suptitle(r'Interventional predictions under $do(X_1=\alpha)$', fontsize=18)

        ax1.plot(xvals, xvals, label=r'True $\mathbb{E} [X_3| do(X_1=\alpha)]$', linewidth=3, linestyle=':')
        ax1.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=10, d=4)[0, 2] for x in xvals]),
                 linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_3| do(X_1=\alpha)]$',
                 alpha=.8)  # remove 0 indexing in legend

        ax1.set_xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=18)
        ax1.set_ylabel(r'Predicted value of $X_3$', fontsize=18)
        ax1.legend(loc=1, fontsize=16)

        ax2.plot(xvals, coeffs[1] * xvals * xvals, label=r'True $\mathbb{E} [X_4| do(X_1=\alpha)]$', linewidth=3,
                 linestyle=':')
        ax2.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=10, d=4)[0, 3] for x in xvals]),
                 linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_4| do(X_1=\alpha)]$',
                 alpha=.8)  # remove 0 indexing in legend

        ax2.set_xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=18)
        ax2.set_ylabel(r'Predicted value of $X_4$', fontsize=18)
        ax2.legend(loc=1, fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(results_dir, 'intervention_4d_2.pdf'), dpi=300)

        # -----

        plt.figure()
        # and the distribution of X_2 | do(X_1=x) should be quadratic
        plt.plot(xvals,
                 np.array(
                     [mod.predict_intervention(x0_val=x, n_samples=10, d=4, iidx=1)[0, 2] for x in
                      xvals]),
                 linewidth=3, linestyle='-.', label=r'$X_2| do(X_1=x)$')
        # and distribution of X_3 | do(X_0=x) should be quadratic
        plt.plot(xvals,
                 np.array(
                     [mod.predict_intervention(x0_val=x, n_samples=10, d=4, iidx=1)[0, 3] for x in
                      xvals]),
                 linewidth=3, linestyle='-.', label=r'$X_3| do(X_1=x)$')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'intervention_4d_3.pdf'), dpi=300)

        # -----

        plt.figure()
        plt.scatter(dat[:, 0], dat[:, 3])
        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=10, d=4)[0, 3] for x in xvals]),
                 linewidth=3, linestyle='-.', label=r'$X_3| do(X_0=x)$', color='red')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'intervention_4d_4.pdf'), dpi=300)

        # below are checks to ensure flow is autoregressive
        # input_ = np.ones((1,4))
        # print(mod.invert_flow( input_ ))
        # input_[0,0] = 2
        # print(mod.invert_flow( input_ ))
        # input_[0,1] = 2
        # print(mod.invert_flow( input_ ))
        # input_[0,2] = 2
        # print(mod.invert_flow( input_ ))
        # input_[0,3] = 2
        # print(mod.invert_flow( input_ ))

    elif dim == 2:
        # generate toy data:
        np.random.seed(0)
        dat, mod_dir = gen_synth_causal_dat(nObs=2500, causalFunc='hoyer2009')
        if mod_dir == 'y->x':
            dat = dat[:, [1, 0]]
        vars_ = dat.std(axis=0)
        dat[:, 0] /= vars_[0]
        dat[:, 1] /= vars_[1]

        true_mod = lambda x: (x + (.5) * x * x * x) / vars_[1]

        # define and fit flow model:
        mod = BivariateFlowLR(n_layers=None, n_hidden=None, prior_dist='laplace', epochs=500, opt_method='scheduling')
        mod.fit_to_sem(dat, n_layers=5, n_hidden=10)

        # plot data distribution:
        plt.scatter(dat[:, 0], dat[:, 1])
        plt.xlabel('cause')
        plt.ylabel('effect')

        xvals = np.arange(-5, 5, .1)
        plt.plot(xvals, true_mod(xvals), color='red', linewidth=3, linestyle=':')
        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=500)[1] for x in xvals]), color='green',
                 linewidth=3, linestyle='-.')
        plt.savefig(os.path.join(results_dir, 'intervention_2d.pdf'), dpi=300)

        # plt.figure()
        # x0val = 1.5
        # predDist = mod.predictIntervention(x0val=x0val, nSamples=50000)
        # plt.hist(predDist)

        # input_ = np.ones((5, 2))
        # print(mod.flow.backwardPassFlow(mod.invertFlow(input_)) - input_)
        # input_ = np.random.random((5,2))
        # print( np.abs(mod.flow.backwardPassFlow( mod.invertFlow( input_ ) ) - input_ ).max() )
        # print(mod.invertFlow( input_ ))
        # input_[:,1] = np.random.normal(0,1,5)
        # print(mod.invertFlow( input_ ))


    else:
        raise NotImplementedError('value of dim {}'.format(dim))
