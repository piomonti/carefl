import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from data.eeg import eeg_data
from models import RECI, ANM, EntropyLR, CAREFL


def res_save_name(config, algo, obs):
    """
    helper function for formatting when saving pickled results
    """
    if 'carefl' not in algo.lower():
        return 'eeg_{}_{}_{}.p'.format(config.data.timeseries_idx, config.data.lag, obs)
    return 'eeg_{}_{}_{}_{}_{}_{}_{}.p'.format(config.data.timeseries_idx,
                                               config.flow.architecture.lower(),
                                               config.flow.net_class.lower(),
                                               config.flow.nl,
                                               config.flow.nh,
                                               config.data.lag,
                                               obs)


def fig_save_name(config, obs):
    """
    helper function for formatting when saving figures
    """
    return 'eeg_arrow_{}_{}_{}_{}_{}_{}.pdf'.format(config.flow.architecture.lower(),
                                                    config.flow.net_class.lower(),
                                                    config.flow.nl,
                                                    config.flow.nh,
                                                    config.data.lag,
                                                    obs)


def run_eeg(args, config):
    """
    Runs a causal algorithm on a *single* EEG timeseries.
    The timeseries is specified by the config.data.timeseries_idx field
    To reproduce the figure in the paper, this function needs to be run multiple
    times with the different combinations of algorithm, timeseries index, etc...
    """
    n_obs_list = [150, 500]
    n_sims = config.n_sims
    algo = config.algorithm
    results = {n: {'p': [], 'c': [], 'correct': 0} for n in n_obs_list}
    for n_obs in n_obs_list:
        n_valid_sims = config.n_sims
        per_correct = 0
        for sim in range(n_sims):
            config.training.seed = sim
            data, mod_dir = eeg_data(idx=config.data.timeseries_idx, lag=config.data.lag, n_obs=n_obs)
            if algo.lower() == 'lrhyv':
                mod = EntropyLR()
            elif algo.lower() == 'anm':
                mod = ANM()
            elif algo.lower() == 'reci':
                mod = RECI(form='GP', scale_input=True)
            elif algo.lower() == 'carefl':
                mod = CAREFL(config)
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
    pickle.dump(results, open(os.path.join(args.output, res_save_name(config, algo, n_obs_list)), 'wb'))


def plot_eeg(args, config):
    """
    plots the figure for the EEG experiment.
    """
    from configs.plotting import color_dict, label_dict, font_dict

    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    sim_list = [_flow, _flow_ns, 'lrhyv', 'reci', 'anm']
    to_algos = lambda s: s.split('/')[0]

    obs_list = [150, 500]

    def _plot_algo(ax, a, n_obs=500, legend=True):
        correct = []
        conf = {}
        for idx in range(118):
            config.data.timeseries_idx = idx
            res = pickle.load(
                open(os.path.join(args.run, 'eeg', a, res_save_name(config, to_algos(a), obs_list)), 'rb'))
            if res[n_obs]['correct'] >= .5:
                correct.append(idx)
            conf[idx] = np.abs(np.mean(res[500]['p']))
        sorted_conf = {k: v for k, v in sorted(conf.items(), key=lambda item: item[1], reverse=True)}
        sorted_keys = sorted_conf.keys()
        sorted_cum_conf = np.cumsum([1 if x in correct else 0 for x in sorted_keys]) / np.arange(1, 119)
        if legend:
            ax.plot(np.linspace(0, 100, 118), sorted_cum_conf, linewidth=2, color=color_dict[to_algos(a)],
                    label=label_dict[to_algos(a)])
        else:
            ax.plot(np.linspace(0, 100, 118), sorted_cum_conf, linewidth=2, color=color_dict[to_algos(a)])

    # prepare plot
    sns.set_style("whitegrid")
    # sns.set_palette('deep')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    for a in sim_list:
        _plot_algo(ax1, a, 150, legend=False)
        _plot_algo(ax2, a, 500)

    ax1.set_title(r'$n=150$', fontsize=font_dict['title'])
    ax2.set_title(r'$n=500$', fontsize=font_dict['title'])
    ax1.set_xlabel('Decision rate', fontsize=font_dict['xlabel'])
    ax1.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax2.set_xlabel('Decision rate', fontsize=font_dict['xlabel'])
    ax2.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])

    fig.legend(  # The labels for each line
        # loc="lower left",  # Position of legend
        # borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm",  # Title for the legend
        fontsize=11,
        bbox_to_anchor=(0.49, 0.5),
        framealpha=.5,
        # frameon=False,
        # prop={'weight':'bold'}
    )
    plt.tight_layout()
    # plt.subplots_adjust(right=0.84)
    plt.savefig(os.path.join(args.run, fig_save_name(config, obs_list)), dpi=300)
