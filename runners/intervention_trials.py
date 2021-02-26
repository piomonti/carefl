# trial interventional predictions of flow models
#
#

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from data.generate_synth_data import intervention_sem
from models import ANM, CAREFL


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


def run_interventions(args, config):
    n_obs = config.data.n_points
    model = config.algorithm.lower()
    print("** {} observations **".format(n_obs))
    # generate coeffcients for equation (12), and data from that SEM
    data, coeffs, dag = intervention_sem(n_obs, dim=4, seed=config.data.seed, random=config.data.random,
                                         multiplicative=config.data.multiplicative)
    print("fitting a {} model".format(model))
    # fit to an affine autoregressive flow or ANM with gp/linear functions
    mod = CAREFL(config) if model == 'carefl' else ANM(method=model)
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
    from configs.plotting import color_dict, label_dict, font_dict
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
    # sns.set_palette(sns.color_palette("muted", 8))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for a in models:
        # plot E[X_3|do(X_1=a)]
        if config.data.expected:
            axs[0].plot(n_obs_list, results[a]["x3e"], color=color_dict[a], linestyle='-.',
                        marker='o', linewidth=2, alpha=.8)
        else:
            axs[0].plot(n_obs_list, results[a]["x3"], color=color_dict[a], linestyle='-',
                        marker='o', linewidth=2, alpha=.8)
        # plot E[X_4|do(X_1=a)]
        if config.data.expected:
            axs[1].plot(n_obs_list, results[a]["x4e"], color=color_dict[a], label=label_dict[a], linestyle='-.',
                        marker='o', linewidth=2, alpha=.8)
        else:
            axs[1].plot(n_obs_list, results[a]["x4"], color=color_dict[a], label=label_dict[a], linestyle='-',
                        marker='o', linewidth=2, alpha=.8)
    axs[0].set_title(r'$\mathbb{E}[X_3|do(X_1=a)]$', fontsize=font_dict['title'])
    axs[1].set_title(r'$\mathbb{E}[X_4|do(X_1=a)]$', fontsize=font_dict['title'])
    for ax in axs:
        ax.set_xlabel(r'Sample size', fontsize=font_dict['xlabel'])
        ax.set_ylabel(r'MSE', fontsize=font_dict['ylabel'])
        ax.set_yscale('log')
    fig.legend(  # The labels for each line
        # loc="center right",  # Position of legend
        # borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm",  # Title for the legend
        fontsize=11,
        bbox_to_anchor=(0.75, 0.7),
        framealpha=.7,
    )
    plt.tight_layout()
    # plt.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(args.run, fig_save_name(config)), dpi=300)
