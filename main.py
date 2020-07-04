### Affine Flow-based causal discovery & inference experiments 
#
#
#

import argparse

from runners.cause_effect_pairs_runner import RunCauseEffectPairs
from runners.simulation_runner import RunSimulations


def parse_input():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='linear',
                        help='Dataset to run synthetic experiments on. Should be either linear, hoyer2009 or nueralnet_l1 or all to run all')
    parser.add_argument('--nSims', type=int, default=25, help='Number of simulations to run')
    parser.add_argument('--resultsDir', type=str, default='results/', help='Path for saving results.')

    parser.add_argument('--plot', action='store_true', help='Should we plot results')
    parser.add_argument('--runCEP', action='store_true', help='Run Cause Effect Pairs experiments')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_input()

    if (args.dataset in ['all', 'linear', 'hoyer2009', 'nueralnet_l1']) & (not args.runCEP):
        print('Running {} synthetic experiments. Will run {} simulations'.format(args.dataset, args.nSims))

        if args.dataset == 'all':
            exp_list = ['linear', 'hoyer2009', 'nueralnet_l1']
        else:
            exp_list = [args.dataset]

        for exp in exp_list:
            nvals = [25, 50, 75, 100, 150, 250, 500]
            results = []
            causal_mechanism = exp
            nsims = args.nSims
            algos = ['FlowCD', 'LRHyv', 'notears', 'RECI', 'ANM']
            print('Mechanism: {}'.format(causal_mechanism))
            for n in nvals:
                print('### {} ###'.format(n))
                results.append(
                    RunSimulations(nSims=nsims, nPoints=n, causal_mechanism=causal_mechanism, algolist=algos))

            # save results
            import pickle

            pickle.dump(results, open(args.resultsDir + causal_mechanism + "_results.p", 'wb'))

    if (args.plot) & (not args.runCEP):
        # produce a plot of synthetic results
        import seaborn as sns
        import pylab as plt
        import pickle
        import numpy as np

        title_dic = {'nueralnet_l1': "Neural network" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + n_2 \right)$",
                     'linear': "Linear SEM\n" + r"$x_2 = x_1 + n_2 $",
                     'hoyer2009': "Nonlinear SEM\n" + r"$x_2 = x_1 + \frac{1}{2} x_1^3 + n_2 $"}

        label_dict = {'FlowCD': 'Affine flow LR',
                      'LRHyv': 'Linear LR',
                      'RECI': 'RECI',
                      'ANM': 'ANM',
                      'notears': 'NO-TEARS'}

        # define some parameters
        nvals = [25, 50, 75, 100, 150, 250, 500]
        algos = ['FlowCD', 'LRHyv', 'notears', 'RECI', 'ANM']
        sim_type = ['linear', 'hoyer2009', 'nueralnet_l1']

        res_all = {s: {a: [] for a in algos} for s in sim_type}

        for s in sim_type:
            results = pickle.load(open(args.resultsDir + s + '_results.p', 'rb'))
            for a in algos:
                for n in range(len(nvals)):
                    res_all[s][a].append(np.mean(results[n][a] == results[n]['true']))

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
        plt.savefig('CausalDiscSims.pdf', dpi=300)

    if args.runCEP:
        print('running cause effect pairs experiments ')
        RunCauseEffectPairs()
