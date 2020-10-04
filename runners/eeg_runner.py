import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from data.eeg import eeg_data
from models import RECI, ANM, EntropyLR, CAReFl, LinearNOTEARS


def res_save_name(config, algo):
    if 'carefl' not in algo.lower():
        return 'eeg_{}.p'.format(config.data.timeseries_idx)
    return 'eeg_{}_{}_{}_{}_{}.p'.format(config.data.timeseries_idx,
                                         config.flow.architecture.lower(),
                                         config.flow.net_class.lower(),
                                         config.flow.nl,
                                         config.flow.nh)


def fig_save_name(config):
    return 'eeg_arrow_{}_{}_{}_{}.pdf'.format(config.flow.architecture.lower(),
                                              config.flow.net_class.lower(),
                                              config.flow.nl,
                                              config.flow.nh)


def run_eeg(args, config):
    n_obs_list = [25, 50, 75, 100, 150, 250, 500]
    n_sims = args.n_sims
    algo = config.algorithm
    results = {n: {'p': [], 'c': [], 'correct': 0} for n in n_obs_list}
    per_correct = 0
    for n_obs in n_obs_list:
        n_valid_sims = args.n_sims
        for sim in range(n_sims):
            config.training.seed = sim
            data, mod_dir = eeg_data(idx=config.data.timeseries_idx, lag=config.data.lag, n_obs=n_obs)
            if algo.lower() == 'lrhyv':
                mod = EntropyLR()
            elif algo.lower() == 'anm':
                mod = ANM()
            elif algo.lower() == 'reci':
                mod = RECI(form='GP', scale_input=True)
            elif algo.lower() == 'notears':
                mod = LinearNOTEARS(lambda1=.01, loss_type='l2', w_threshold=0)
            elif algo.lower() == 'carefl':
                mod = CAReFl(config)
            else:
                raise ValueError('Unknown algorithm')
            p, direction = mod.predict_proba(data=data)
            if not np.isnan(p):
                per_correct += direction == mod_dir
                results[n_obs]['p'].append(p)
                results[n_obs]['c'].append(1. * (direction == mod_dir))
            else:
                n_valid_sims -= 1
        results[n_obs]['correct'] = per_correct / n_valid_sims
    pickle.dump(results, open(os.path.join(args.output, res_save_name(config, algo)), 'wb'))


def plot_eeg(args, config):
    # produce a plot of synthetic results
    label_dict = {'carefl': 'CAReFl',
                  'careflns': 'CAReFl-NS',
                  'lrhyv': 'Linear LR',
                  'reci': 'RECI',
                  'anm': 'ANM',
                  'notears': 'NO-TEARS'}
    # define some parameters
    nvals = [25, 50, 75, 100, 150, 250, 500]
    algos = ['carefl', 'careflns', 'lrhyv', 'notears', 'reci', 'anm']
    to_algos = lambda s: s.split('/')[0]
    res_all = {a: [] for a in algos}

    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    sim_list = [_flow, _flow_ns, 'lrhyv', 'notears', 'reci', 'anm']

    for a in sim_list:
        for n in nvals:
            mean = []
            for s in range(118):
                config.data.timeseries_idx = s
                res = pickle.load(open(os.path.join(args.run, 'eeg', a, res_save_name(config, to_algos(a))), 'rb'))
                mean.append(res[n]['correct'])
            res_all[to_algos(a)].append(np.mean(np.array(mean) >= .5))

    # prepare plot
    sns.set_style("whitegrid")
    sns.set_palette('deep')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for a in algos:
        ax.plot(nvals, res_all[a], marker='o', label=label_dict[a])

    fontsize = 12
    font_xlab = 10
    ax.set_title('Percentage correct on EEG timeseries', fontsize=fontsize)
    ax.set_xlabel('Sample size', fontsize=font_xlab)
    ax.set_ylabel('Proportion correct', fontsize=font_xlab)

    fig.legend(  # The labels for each line
        # loc="center right",  # Position of legend
        borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm"  # Title for the legend
    )
    plt.tight_layout()
    # plt.subplots_adjust(right=0.86)
    plt.savefig(os.path.join(args.run, fig_save_name(config)), dpi=300)
    pass