# trial interventional predictions of flow models
#
#

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from data.generate_synth_data import gen_synth_causal_dat
from models import ANM, CAReFl


def res_save_name(config, algo):
    if 'carefl' not in algo.lower():
        return 'int_{}{}{}.p'.format(config.data.n_points, 'r' * config.data.random, 'm' * config.data.multiplicative)
    return 'int_{}{}{}_{}_{}_{}_{}.p'.format(config.data.n_points,
                                             'r' * config.data.random,
                                             'm' * config.data.multiplicative,
                                             config.flow.architecture.lower(),
                                             config.flow.net_class.lower(),
                                             config.flow.nl,
                                             config.flow.nh)


def fig_save_name(config):
    return 'int_mse_{}{}{}_{}_{}_{}_{}.pdf'.format('r' * config.data.random,
                                                   'e' * config.data.expected,
                                                   'm' * config.data.multiplicative,
                                                   config.flow.architecture.lower(),
                                                   config.flow.net_class.lower(),
                                                   config.flow.nl,
                                                   config.flow.nh)


def intervention_sem(n_obs, dim=4, seed=0, random=True, multiplicative=False):
    if dim == 4:
        # generate some 4D data according to the following SEM
        #
        # X_1 = N_1
        # X_2 = N_2
        # X_3 = (X_1 + c_0*X_2^3) +/* N_3   -  c_0 random coeff
        # X_4 = (c_1*X_1^2 - X_2) +/* N_4   -  c_1 random coeff
        np.random.seed(seed)
        # causes
        X_1 = np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        X_2 = np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        # effects
        coeffs = np.random.uniform(.1, .9, 2) if random else [.5, .5]
        X_3 = X_1 + coeffs[0] * (X_2 * X_2 * X_2)
        X_4 = -X_2 + coeffs[1] * (X_1 * X_1)
        if multiplicative:
            X_3 *= np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
            X_4 *= np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        else:
            X_3 += np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
            X_4 += np.random.laplace(0, 1 / np.sqrt(2), size=n_obs)
        # create the adjacency matrix
        dag = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
        return np.vstack((X_1, X_2, X_3, X_4)).T, coeffs, dag
    else:
        raise NotImplementedError('will be implemented soon')


def run_interventions(args, config):
    n_obs = config.data.n_points
    model = config.algorithm.lower()
    print("** {} observations **".format(n_obs))
    # generate coeffcients for equation (12), and data from that SEM
    data, coeffs, dag = intervention_sem(n_obs, dim=4, seed=config.data.seed, random=config.data.random,
                                         multiplicative=config.data.multiplicative)
    print("fitting a {} model".format(model))
    # fit to an affine autoregressive flow or ANM with gp/linear functions
    mod = CAReFl(config) if model == 'carefl' else ANM(method=model)
    mod.fit_to_sem(data, dag)
    # intervene on X_1 and get a sample of {x | do(X_1=a)} for a in [-3, 3]
    avals = np.arange(-3, 3, .1)
    x_int_sample = []
    x_int_exp = []
    for a in avals:
        res = mod.predict_intervention(a, n_samples=20, iidx=0)
        x_int_sample.append(res[0].mean(axis=0))
        x_int_exp.append(res[1].mean(axis=0))
    x_int_sample = np.array(x_int_sample)
    x_int_exp = np.array(x_int_exp)
    # compute the MSE between the true E[x_3|x_1=a] to the empirical expectation from the sample
    # we know that the true E[x_3|x_1=a] = a
    mse_x3 = np.mean((x_int_sample[:, 2] - avals) ** 2)
    mse_x3e = np.mean((x_int_exp[:, 2] - avals) ** 2)
    # do the same for x_4; true E[x_4|x_1=a] = c_1*a^2
    mse_x4 = np.mean((x_int_sample[:, 3] - coeffs[1] * avals * avals) ** 2)
    mse_x4e = np.mean((x_int_exp[:, 3] - coeffs[1] * avals * avals) ** 2)
    # store results
    results = {}
    results["x3"] = mse_x3
    results["x4"] = mse_x4
    results["x3e"] = mse_x3e
    results["x4e"] = mse_x4e
    pickle.dump(results, open(os.path.join(args.output, res_save_name(config, model)), 'wb'))


def plot_interventions(args, config):
    # plot the MSEs
    n_obs_list = [250, 500, 750, 1000, 1250, 1500, 2000, 2500]
    models = ['carefl', 'careflns', 'gp', 'linear']
    to_models = lambda s: s.split('/')[0]
    # load results from disk
    variables = ['x3', 'x3e', 'x4', 'x4e']
    results = {mod: {x: [] for x in variables} for mod in models}

    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    int_list = [_flow, _flow_ns, 'gp', 'linear']

    for a in int_list:
        for n in n_obs_list:
            config.data.n_points = n
            res = pickle.load(
                open(os.path.join(args.run, 'interventions', a, res_save_name(config, to_models(a))), 'rb'))
            for x in variables:
                results[to_models(a)][x].append(res[x])
    # produce plot
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("muted", 8))
    label = {'carefl': 'CAReFl', 'careflns': 'CAReFl-NS', 'gp': 'ANM-GP', 'linear': 'ANM-linear'}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for mod in models:
        # plot E[X_3|do(X_1=a)]
        if config.data.expected:
            axs[0].plot(n_obs_list, results[mod]["x3e"], linestyle='-.', marker='o', linewidth=2, alpha=.8)
        else:
            axs[0].plot(n_obs_list, results[mod]["x3"], linestyle='-', marker='o', linewidth=2, alpha=.8)
        # plot E[X_4|do(X_1=a)]
        if config.data.expected:
            axs[1].plot(n_obs_list, results[mod]["x4e"], label='{}'.format(label[mod]), linestyle='-.',
                        marker='o', linewidth=2, alpha=.8)
        else:
            axs[1].plot(n_obs_list, results[mod]["x4"], label='{}'.format(label[mod]), linestyle='-',
                        marker='o', linewidth=2, alpha=.8)
    axs[0].set_title(r'$\mathbb{E}[X_3|do(X_1=a)]$', fontsize=13)
    axs[1].set_title(r'$\mathbb{E}[X_4|do(X_1=a)]$', fontsize=13)
    for ax in axs:
        ax.set_xlabel(r'Sample size', fontsize=10)
        ax.set_ylabel(r'MSE', fontsize=10)
        ax.set_yscale('log')
    fig.legend(  # The labels for each line
        loc="center right",  # Position of legend
        borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm"  # Title for the legend
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(args.run, fig_save_name(config)), dpi=300)


def intervention(args, config, dim=4):
    if dim == 4:
        # higher D examples
        n_obs = 2500
        dat, coeffs, dag = intervention_sem(n_obs, dim=dim, seed=0, random=False)

        # example plots:
        plt.scatter(dat[:, 1], dat[:, 2])
        plt.scatter(dat[:, 0], dat[:, 3])

        mod = CAReFl(config)
        mod.fit_to_sem(dat, dag)

        # -----

        # so the distribution of X_2 | do(X_0=x) should be linear
        xvals = np.arange(-3, 3, .05)

        sns.set_style("whitegrid")
        sns.set_palette(sns.color_palette("muted", 8))
        # sns.diverging_palette(255, 133, l=60, n=7, center="dark") )#  sns.color_palette("coolwarm", 6) )

        plt.figure()
        plt.plot(xvals, xvals, label='True', color=sns.color_palette("muted", 8)[2], linewidth=2, alpha=.8)
        plt.plot(xvals, coeffs[1] * xvals * xvals, color=sns.color_palette("muted", 8)[2], linewidth=2, alpha=.8)

        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=100)[0][0, 2] for x in xvals]),
                 linewidth=3, linestyle='-.',
                 label=r'Predicted $\mathbb{E} [X_3| do(X_1=\alpha)]$')  # remove 0 indexing in legend
        # and distribution of X_3 | do(X_0=x) should be quadratic
        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=100)[0][0, 3] for x in xvals]),
                 linewidth=3, linestyle='-.',
                 label=r'Predicted $\mathbb{E} [X_4| do(X_1=\alpha)]$')  # remove 0 indexing in legend

        plt.legend(fontsize=12)
        plt.xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=12)
        plt.ylabel(r'Predicted value of $X_3$ or $X_4$', fontsize=12)
        plt.title(r'interventional predictions under $do(X_1=\alpha)$', fontsize=15)
        plt.savefig(os.path.join(args.run, 'intervention_4d_1.pdf'), dpi=300)

        # -----

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 9))
        fig.suptitle(r'Interventional predictions under $do(X_1=\alpha)$', fontsize=18)

        ax1.plot(xvals, xvals, label=r'True $\mathbb{E} [X_3| do(X_1=\alpha)]$', linewidth=3, linestyle=':')
        ax1.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=100)[0][0, 2] for x in xvals]),
                 linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_3| do(X_1=\alpha)]$',
                 alpha=.8)  # remove 0 indexing in legend

        ax1.set_xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=18)
        ax1.set_ylabel(r'Predicted value of $X_3$', fontsize=18)
        ax1.legend(loc=1, fontsize=16)

        ax2.plot(xvals, coeffs[1] * xvals * xvals, label=r'True $\mathbb{E} [X_4| do(X_1=\alpha)]$', linewidth=3,
                 linestyle=':')
        ax2.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=100)[0][0, 3] for x in xvals]),
                 linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_4| do(X_1=\alpha)]$',
                 alpha=.8)  # remove 0 indexing in legend

        ax2.set_xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=18)
        ax2.set_ylabel(r'Predicted value of $X_4$', fontsize=18)
        ax2.legend(loc=1, fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(args.run, 'intervention_4d_2.pdf'), dpi=300)

        # -----

        plt.figure()
        # and the distribution of X_2 | do(X_1=x) should be quadratic
        plt.plot(xvals,
                 np.array(
                     [mod.predict_intervention(x0_val=x, n_samples=100, iidx=1)[0][0, 2] for x in
                      xvals]),
                 linewidth=3, linestyle='-.', label=r'$X_3| do(X_1=x)$')
        # and distribution of X_3 | do(X_0=x) should be quadratic
        plt.plot(xvals,
                 np.array(
                     [mod.predict_intervention(x0_val=x, n_samples=100, iidx=1)[0][0, 3] for x in
                      xvals]),
                 linewidth=3, linestyle='-.', label=r'$X_4| do(X_1=x)$')
        plt.legend()
        plt.savefig(os.path.join(args.run, 'intervention_4d_3.pdf'), dpi=300)

        # -----

        plt.figure()
        plt.scatter(dat[:, 0], dat[:, 3])
        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=100)[0][0, 3] for x in xvals]),
                 linewidth=3, linestyle='-.', label=r'$X_4| do(X_0=x)$', color='red')
        plt.legend()
        plt.savefig(os.path.join(args.run, 'intervention_4d_4.pdf'), dpi=300)

        # below are checks to ensure flow is autoregressive
        # input_ = np.ones((1,4))
        # print(mod._forward_flow( input_ ))
        # input_[0,0] = 2
        # print(mod._forward_flow( input_ ))
        # input_[0,1] = 2
        # print(mod._forward_flow( input_ ))
        # input_[0,2] = 2
        # print(mod._forward_flow( input_ ))
        # input_[0,3] = 2
        # print(mod._forward_flow( input_ ))

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
        mod = CAReFl(config)
        mod.fit_to_sem(dat, None)

        # plot data distribution:
        plt.scatter(dat[:, 0], dat[:, 1])
        plt.xlabel('cause')
        plt.ylabel('effect')

        xvals = np.arange(-5, 5, .1)
        plt.plot(xvals, true_mod(xvals), color='red', linewidth=3, linestyle=':')
        plt.plot(xvals, np.array([mod.predict_intervention(x0_val=x, n_samples=500)[1] for x in xvals]), color='green',
                 linewidth=3, linestyle='-.')
        plt.savefig(os.path.join(args.run, 'intervention_2d.pdf'), dpi=300)

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
