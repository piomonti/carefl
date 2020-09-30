# run simulations
#
#
#
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from data.generate_synth_data import gen_synth_causal_dat
from models import RECI, ANM, EntropyLR, CAReFl, LinearNOTEARS


def res_save_name(config, algo):
    if algo.lower() != 'carefl':
        return 'sim_{}.p'.format(config.data.n_points)
    return 'sim_{}_{}_{}_{}_{}.p'.format(config.data.n_points,
                                         config.flow.architecture.lower(),
                                         config.flow.net_class.lower(),
                                         config.flow.nl,
                                         config.flow.nh)


def fig_save_name(config):
    return 'sim_causal_dir_{}_{}_{}_{}.pdf'.format(config.flow.architecture.lower(),
                                                   config.flow.net_class.lower(),
                                                   config.flow.nl,
                                                   config.flow.nh)


def run_simulations(args, config):
    n_points = config.data.n_points
    n_sims = n_valid_sims = args.n_sims
    algo = config.algorithm
    causal_mechanism = config.data.causal_mech
    reci_form_dict = {'linear': 'poly', 'hoyer2009': 'poly', 'nueralnet_l1': 'GP'}
    results = {'p': [], 'c': [], 'correct': 0}
    per_correct = 0
    for sim in range(n_sims):
        np.random.seed(sim)
        data, mod_dir = gen_synth_causal_dat(nObs=n_points, causalFunc=causal_mechanism)
        if algo.lower() == 'lrhyv':
            mod = EntropyLR()
        elif algo.lower() == 'anm':
            mod = ANM()
        elif algo.lower() == 'reci':
            mod = RECI(form=reci_form_dict[causal_mechanism], scale_input=True)
        elif algo.lower() == 'carefl':
            mod = CAReFl(config)
        elif algo.lower() == 'notears':
            mod = LinearNOTEARS(lambda1=.01, loss_type='l2', w_threshold=0)
        else:
            raise ValueError('Unknown algorithm')
        p, direction = mod.predict_proba(data=data)
        if not np.isnan(p):
            per_correct += direction == mod_dir
            results['p'].append(p)
            results['c'].append(1. * (direction == mod_dir))
        else:
            n_valid_sims -= 1
    results['correct'] = per_correct / n_valid_sims
    pickle.dump(results, open(os.path.join(args.output, res_save_name(config, algo)), 'wb'))


def plot_simulations(args, config):
    # produce a plot of synthetic results
    title_dic = {'nueralnet_l1': "Neural network" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + n_2 \right)$",
                 'linear': "Linear SEM\n" + r"$x_2 = x_1 + n_2 $",
                 'hoyer2009': "Nonlinear SEM\n" + r"$x_2 = x_1 + \frac{1}{2} x_1^3 + n_2 $"}
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
    sim_type = ['linear', 'hoyer2009', 'nueralnet_l1']
    res_all = {s: {a: [] for a in algos} for s in sim_type}

    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    sim_list = [_flow, _flow_ns, 'lrhyv', 'notears', 'reci', 'anm']

    for s in sim_type:
        for a in sim_list:
            for n in nvals:
                config.data.n_points = n
                res = pickle.load(
                    open(os.path.join(args.run, 'simulations', s, a, res_save_name(config, to_algos(a))), 'rb'))
                res_all[s][to_algos(a)].append(res['correct'])
    # prepare plot
    sns.set_style("whitegrid")
    sns.set_palette('deep')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for a in algos:
        ax1.plot(nvals, res_all['linear'][a], marker='o')
        ax2.plot(nvals, res_all['hoyer2009'][a], marker='o')
        ax3.plot(nvals, res_all['nueralnet_l1'][a], marker='o', label=label_dict[a])
    fontsize = 12
    font_xlab = 10
    ax1.set_title(title_dic['linear'], fontsize=fontsize)
    ax2.set_title(title_dic['hoyer2009'], fontsize=fontsize)
    ax3.set_title(title_dic['nueralnet_l1'], fontsize=fontsize)
    ax1.set_xlabel('Sample size', fontsize=font_xlab)
    ax2.set_xlabel('Sample size', fontsize=font_xlab)
    ax3.set_xlabel('Sample size', fontsize=font_xlab)
    ax1.set_ylabel('Proportion correct', fontsize=font_xlab)
    ax2.set_ylabel('Proportion correct', fontsize=font_xlab)
    ax3.set_ylabel('Proportion correct', fontsize=font_xlab)
    fig.legend(  # The labels for each line
        loc="center right",  # Position of legend
        borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm"  # Title for the legend
    )
    plt.tight_layout()
    plt.subplots_adjust(right=0.87)
    plt.savefig(os.path.join(args.run, fig_save_name(config)), dpi=300)
