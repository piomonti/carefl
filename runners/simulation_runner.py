# run simulations
#
#
#
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from data.generate_synth_data import gen_synth_causal_dat, intervention_sem
from models import RECI, ANM, EntropyLR, CAREFL


# make sure matplotlib doesn't use Type 3 fonts
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def res_save_name(config, algo):
    """
    helper function to format names of pickled results
    """
    if algo.lower() == 'anm-nn':
        return 'sim_{}_nn.p'.format(config.data.n_points)
    elif 'carefl' not in algo.lower():
        return 'sim_{}.p'.format(config.data.n_points)
    return 'sim_{}_{}_{}_{}_{}_{}_{}.p'.format(config.data.n_points,
                                               config.flow.architecture.lower(),
                                               config.flow.net_class.lower(),
                                               config.flow.nl,
                                               config.flow.nh,
                                               config.flow.prior_dist,
                                               config.data.noise_dist)


def fig_save_name(config):
    """
    helper function to format figure names
    """
    return 'sim_causal_dir_{}_{}_{}_{}_{}_{}.pdf'.format(config.flow.architecture.lower(),
                                                         config.flow.net_class.lower(),
                                                         config.flow.nl,
                                                         config.flow.nh,
                                                         config.flow.prior_dist,
                                                         config.data.noise_dist)


def run_simulations(args, config):
    """
    Run simulations for a given config
    To reproduce the figures in the paper, this function needs to be run
    on all different combinations of dataset, algorithm, number of data points, etc...
    This is done by running the `main.py` script multiple times. An example is provided in
    the README.
    """
    n_points = config.data.n_points
    n_sims = n_valid_sims = config.n_sims
    algo = config.algorithm
    causal_mechanism = config.data.causal_mech
    reci_form_dict = {'linear': 'poly', 'hoyer2009': 'poly', 'nueralnet_l1': 'GP', 'highdim': 'poly', 'mnm': 'GP',
                      'highdim_sigmoid': 'GP', 'veryhighdim': 'GP'}
    results = {'p': [], 'c': [], 'correct': 0}
    per_correct = 0
    for sim in range(n_sims):
        if causal_mechanism in ['highdim', 'highdim_sigmoid', 'veryhighdim']:
            dim = 20 if causal_mechanism == 'veryhighdim' else 4
            nonlin = 'poly' if causal_mechanism == 'highdim' else 'sigmoid'  # doesn't matter for 20dim
            data, _, dag = intervention_sem(n_obs=n_points, dim=dim, seed=sim, random=config.data.random, shuffle=True,
                                            nonlin=nonlin, noise_dist=config.data.noise_dist)
            mod_dir = 'x->y' if np.all(dag[0] == 0) else 'y->x'
        else:
            np.random.seed(sim)
            data, mod_dir = gen_synth_causal_dat(nObs=n_points, causalFunc=causal_mechanism,
                                                 noise_dist=config.data.noise_dist)
        if algo.lower() == 'lrhyv':
            mod = EntropyLR()
        elif algo.lower() == 'anm':
            mod = ANM(method=config.anm.method, config=config)
            if config.anm.method.lower() == 'nn':
                algo = 'anm-mm'
        elif algo.lower() == 'reci':
            mod = RECI(form=reci_form_dict[causal_mechanism], scale_input=True)
        elif algo.lower() == 'carefl':
            mod = CAREFL(config)
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
    from configs.plotting import color_dict, label_dict, font_dict
    # produce a plot of synthetic results
    title_dic = {'nueralnet_l1': "Post nonlinear SEM" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + z_2 \right)$",
                 'linear': "Linear SEM\n" + r"$x_2 = x_1 + z_2 $",
                 'hoyer2009': "Additive noise SEM\n" + r"$x_2 = x_1 + \frac{1}{2} x_1^3 + z_2 $",
                 'mnm': "Affine noise SEM\n" + r"$x_2 = \sigma(x_1) + \frac{1}{2} x_1^2 + \sigma(x_1)z_2$",
                 'highdim': "4-dimensional SEM - polynomial",
                 'highdim_sigmoid': "4-dimensional SEM - sigmoid",
                 'veryhighdim': "20-dimensional SEM\n" +
                                r"$\mathbf{x}_{11:20} = \mathbf{g}(\mathbf{x}_{1:10}, \mathbf{z}_{11:20})$"}

    # define some parameters
    nvals = [25, 50, 75, 100, 150, 250, 500]
    algos = ['carefl', 'careflns', 'lrhyv', 'reci', 'anm', 'anm-nn']
    # to_algos = lambda s: s.split('/')[0]
    to_algos = lambda s: 'anm' if s == 'anm-nn' else s
    # sim_type = ['linear', 'hoyer2009', 'nueralnet_l1', 'highdim', 'highdim_sigmoid', 'veryhighdim', 'mnm']
    sim_type = ['linear', 'hoyer2009', 'nueralnet_l1', 'veryhighdim', 'mnm']
    res_all = {s: {a: [] for a in algos} for s in sim_type}

    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    algo_path = [_flow, _flow_ns, 'lrhyv', 'reci', 'anm', 'anm']

    for s in sim_type:
        for (a, ap) in zip(algos, algo_path):
            for n in nvals:
                config.data.n_points = n
                res = pickle.load(
                    open(os.path.join(args.run, 'simulations', s, ap, res_save_name(config, a)), 'rb'))
                res_all[s][a].append(res['correct'])
    # prepare plot
    sns.set_style("whitegrid")
    # sns.set_palette('deep')
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for a in algos:
        ax1.plot(nvals, res_all['linear'][a], color=color_dict[a], marker='o')
        ax2.plot(nvals, res_all['hoyer2009'][a], color=color_dict[a], marker='o')
        ax4.plot(nvals, res_all['nueralnet_l1'][a], color=color_dict[a], marker='o')
        # ax4.plot(nvals, res_all['highdim'][a], color=color_dict[a], marker='o')
        ax3.plot(nvals, res_all['mnm'][a], color=color_dict[a], marker='o')
        ax5.plot(nvals, res_all['veryhighdim'][a], color=color_dict[a], marker='o', label=label_dict[a])

    ax1.set_title(title_dic['linear'], fontsize=font_dict['title'])
    ax2.set_title(title_dic['hoyer2009'], fontsize=font_dict['title'])
    ax4.set_title(title_dic['nueralnet_l1'], fontsize=font_dict['title'])
    # ax4.set_title(title_dic['highdim'], fontsize=font_dict['title'])
    ax3.set_title(title_dic['mnm'], fontsize=font_dict['title'])
    ax5.set_title(title_dic['veryhighdim'], fontsize=font_dict['title'])
    ax1.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax2.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax3.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    # ax4.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax5.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax4.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax1.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax2.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax3.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    # ax4.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax5.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax4.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    fig.legend(  # The labels for each line
        # loc="center right",  # Position of legend
        # borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm",  # Title for the legend
        fontsize=11,
        bbox_to_anchor=(0.2, 0.44),
        framealpha=.7,
        ncol=2,
    )
    plt.tight_layout()
    # plt.subplots_adjust(right=0.9)
    plt.savefig(os.path.join(args.run, fig_save_name(config)), dpi=300)


def plot_prior_mismatch(args, config):
    from configs.plotting import label_dict, font_dict
    title_dic = {'nueralnet_l1': "Post nonlinear SEM" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + z_2 \right)$",
                 'linear': "Linear SEM\n" + r"$x_2 = x_1 + z_2 $",
                 'hoyer2009': "Additive noise SEM\n" + r"$x_2 = x_1 + \frac{1}{2} x_1^3 + z_2 $",
                 'mnm': "Affine noise SEM\n" + r"$x_2 = \sigma(x_1) + \frac{1}{2} x_1^2 + \sigma(x_1)z_2$",
                 'highdim': "4-dimensional SEM - polynomial",
                 'highdim_sigmoid': "4-dimensional SEM - sigmoid",
                 'veryhighdim': "20-dimensional SEM\n" +
                                r"$\mathbf{x}_{11:20} = \mathbf{g}(\mathbf{x}_{1:10}, \mathbf{z}_{11:20})$"}

    # define some parameters
    nvals = [25, 50, 75, 100, 150, 250, 500]
    sim_type = ['linear', 'hoyer2009', 'nueralnet_l1', 'veryhighdim', 'mnm']
    noise_dists = ['laplace', 'gaussian', 'student']
    algos = ['carefl', 'careflns']
    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    algo_path = [_flow, _flow_ns]

    res_all = {s: {a: {nd: [] for nd in noise_dists} for a in algos} for s in sim_type}

    for s in sim_type:
        for (a, ap) in zip(algos, algo_path):
            for n in nvals:
                for nd in noise_dists:
                    config.data.noise_dist = nd
                    config.data.n_points = n
                    res = pickle.load(
                        open(os.path.join(args.run, 'simulations', s, ap, res_save_name(config, a)), 'rb'))
                    res_all[s][a][nd].append(res['correct'])
    # prepare plot
    sns.set_style("whitegrid")
    # sns.set_palette('deep')
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for a in algos:
        for nd in noise_dists:
            ax1.plot(nvals, res_all['linear'][a][nd], marker='o')
            ax2.plot(nvals, res_all['hoyer2009'][a][nd], marker='o')
            ax4.plot(nvals, res_all['nueralnet_l1'][a][nd], marker='o')
            ax3.plot(nvals, res_all['mnm'][a][nd], marker='o')
            ax5.plot(nvals, res_all['veryhighdim'][a][nd], marker='o', label="{} ({})".format(label_dict[a], nd))

    ax1.set_title(title_dic['linear'], fontsize=font_dict['title'])
    ax2.set_title(title_dic['hoyer2009'], fontsize=font_dict['title'])
    ax3.set_title(title_dic['mnm'], fontsize=font_dict['title'])
    ax4.set_title(title_dic['nueralnet_l1'], fontsize=font_dict['title'])
    ax5.set_title(title_dic['veryhighdim'], fontsize=font_dict['title'])
    ax1.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax2.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax3.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax4.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax5.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax1.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax2.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax3.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax4.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax5.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    fig.legend(  # The labels for each line
        # loc="center right",  # Position of legend
        # borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm (noise dist.)",  # Title for the legend
        fontsize=11,
        bbox_to_anchor=(0.2, 0.6),
        framealpha=.7,
    )
    plt.tight_layout()
    # plt.subplots_adjust(right=0.9)
    plt.savefig(os.path.join(args.run, 'prior_mismatch_' + fig_save_name(config)), dpi=300)


def plot_width_vs_depth(args, config):
    from configs.plotting import label_dict, font_dict
    title_dic = {'nueralnet_l1': "Post nonlinear SEM" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + z_2 \right)$",
                 'linear': "Linear SEM\n" + r"$x_2 = x_1 + z_2 $",
                 'hoyer2009': "Additive noise SEM\n" + r"$x_2 = x_1 + \frac{1}{2} x_1^3 + z_2 $",
                 'mnm': "Affine noise SEM\n" + r"$x_2 = \sigma(x_1) + \frac{1}{2} x_1^2 + \sigma(x_1)z_2$",
                 'highdim': "4-dimensional SEM - polynomial",
                 'highdim_sigmoid': "4-dimensional SEM - sigmoid",
                 'veryhighdim': "20-dimensional SEM\n" +
                                r"$\mathbf{x}_{11:20} = \mathbf{g}(\mathbf{x}_{1:10}, \mathbf{z}_{11:20})$"}

    # define some parameters
    nvals = [25, 50, 75, 100, 150, 250, 500]
    sim_type = ['linear', 'hoyer2009', 'nueralnet_l1', 'veryhighdim', 'mnm']
    nhs = [1, 2, 4, 8, 16, 32]
    nls = [2, 4, 8]
    algos = ['carefl']
    _flow = os.path.join('carefl', config.flow.architecture.lower())
    _flow_ns = os.path.join('careflns', config.flow.architecture.lower())
    algo_path = [_flow]

    res_all_w = {s: {a: {nh: [] for nh in nhs} for a in algos} for s in sim_type}
    res_all_d = {s: {a: {nl: [] for nl in nls} for a in algos} for s in sim_type}


    for s in sim_type:
        for (a, ap) in zip(algos, algo_path):
            for n in nvals:
                for nh in nhs:
                    config.flow.nl = 2
                    config.flow.nh = nh
                    config.data.n_points = n
                    res = pickle.load(
                        open(os.path.join(args.run, 'simulations', s, ap, res_save_name(config, a)), 'rb'))
                    res_all_w[s][a][nh].append(res['correct'])
                for nl in nls:
                    config.flow.nl = nl
                    config.flow.nh = 1
                    config.data.n_points = n
                    res = pickle.load(
                        open(os.path.join(args.run, 'simulations', s, ap, res_save_name(config, a)), 'rb'))
                    res_all_d[s][a][nl].append(res['correct'])
    # prepare plot
    sns.set_style("whitegrid")
    # sns.set_palette('deep')
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for a in algos:
        for nl in nls:
            ax1.plot(nvals, res_all_d['linear'][a][nl], marker='o')
            ax2.plot(nvals, res_all_d['hoyer2009'][a][nl], marker='o')
            ax4.plot(nvals, res_all_d['nueralnet_l1'][a][nl], marker='o')
            ax3.plot(nvals, res_all_d['mnm'][a][nl], marker='o')
            ax5.plot(nvals, res_all_d['veryhighdim'][a][nl], marker='o', label="nh:{} nl:{}".format(1, nl))

        for nh in nhs:
            ax1.plot(nvals, res_all_w['linear'][a][nh], marker='o')
            ax2.plot(nvals, res_all_w['hoyer2009'][a][nh], marker='o')
            ax4.plot(nvals, res_all_w['nueralnet_l1'][a][nh], marker='o')
            ax3.plot(nvals, res_all_w['mnm'][a][nh], marker='o')
            ax5.plot(nvals, res_all_w['veryhighdim'][a][nh], marker='o', label="nh:{} nl:{}".format(nh, 2))

    ax1.set_title(title_dic['linear'], fontsize=font_dict['title'])
    ax2.set_title(title_dic['hoyer2009'], fontsize=font_dict['title'])
    ax3.set_title(title_dic['mnm'], fontsize=font_dict['title'])
    ax4.set_title(title_dic['nueralnet_l1'], fontsize=font_dict['title'])
    ax5.set_title(title_dic['veryhighdim'], fontsize=font_dict['title'])
    ax1.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax2.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax3.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax4.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax5.set_xlabel('Sample size', fontsize=font_dict['xlabel'])
    ax1.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax2.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax3.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax4.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    ax5.set_ylabel('Proportion correct', fontsize=font_dict['ylabel'])
    fig.legend(  # The labels for each line
        # loc="center right",  # Position of legend
        # borderaxespad=0.2,  # Small spacing around legend box
        title="Algorithm",  # Title for the legend
        fontsize=11,
        bbox_to_anchor=(0.2, 0.54),
        framealpha=.7,
    )
    plt.tight_layout()
    # plt.subplots_adjust(right=0.9)
    plt.savefig(os.path.join(args.run, 'width_vs_depth_' + fig_save_name(config)), dpi=300)
