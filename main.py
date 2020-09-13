import sys

import argparse
import numpy as np
import os
import torch
import yaml

from runners.cause_effect_pairs_runner import run_cause_effect_pairs
from runners.counterfactual_trials import counterfactuals
from runners.intervention_trials import run_interventions, plot_interventions
from runners.simulation_runner import run_simulations, plot_simulations


def parse_input():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='simulations.yaml', help='config file to use')
    parser.add_argument('--n-sims', type=int, default=250, help='Number of synthetic simulations to run')
    parser.add_argument('--run', type=str, default='results', help='Path for saving results.')

    parser.add_argument('-s', '--simulation', action='store_true', help='run the CD exp on synthetic data')
    parser.add_argument('-p', '--pairs', action='store_true', help='Run Cause Effect Pairs experiments')
    parser.add_argument('-i', '--intervention', action='store_true', help='run intervention exp on toy example')
    parser.add_argument('-c', '--counterfactual', action='store_true', help='run counterfactual exp on toy example')

    # params to overwrite config file. useful for batch running in slurm
    parser.add_argument('--causal-mech', type=str, default=None,
                        help='Dataset to run synthetic CD experiments on. Should be either linear, '
                             'hoyer2009 or nueralnet_l1 or all to run all')
    parser.add_argument('-a', '--algorithm', type=str, default=None, help='algorithm to run')
    parser.add_argument('--seed', type=int, default=None, help='random seed for flow')
    parser.add_argument('-n', '--n-points', type=int, default=None, help='number of simulated data points')

    return parser.parse_args()


def debug_options(args, config):
    if args.causal_mech is not None:
        config.data.causal_mech = args.causal_mech
    if args.algorithm is not None:
        config.algorithm = args.algorithm
    if args.n_points is not None:
        config.data.n_points = args.n_points
    if args.seed is not None:
        config.training.seed = args.seed


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def make_and_set_dirs(args, config):
    args.algo = config.algorithm.lower()
    if config.algorithm.lower() == 'carefl':
        args.algo = os.path.join(args.algo, config.flow.architecture.lower())
    _flow_alg = os.path.join('carefl', config.flow.architecture.lower())
    args.sim_list = [_flow_alg, 'lrhyv', 'notears', 'reci', 'anm']
    args.int_list = [_flow_alg, 'gp', 'linear']
    os.makedirs(args.run, exist_ok=True)
    args.output = os.path.join(args.run, args.doc, args.algo)
    os.makedirs(args.output, exist_ok=True)


def main():
    # parse command line arguments
    args = parse_input()
    # load config
    with open(os.path.join('configs', args.config), 'r') as f:
        print('loading config file: {}'.format(os.path.join('configs', args.config)))
        config_raw = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config_raw)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # DEBUG OPTIONS:
    debug_options(args, config)
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.simulation:
        args.doc = os.path.join('simulations', config.data.causal_mech)
        make_and_set_dirs(args, config)
        if not args.plot:
            # run algorithm on simulated data
            # and save the results as pickle files which can be used later to plot Fig 1.
            print('Running {} on {} synthetic experiments ({} simulations)'.format(config.architecture,
                                                                                   config.data.causal_mech,
                                                                                   args.n_sims))
            run_simulations(args, config)
        else:
            plot_simulations(args, config)

    if args.pairs:
        args.doc = 'pairs'
        make_and_set_dirs(args, config)
        # Run proposed method on CauseEffectPair dataset
        # Percentage of correct causal direction is printed to standard output,
        # and updated online after each new pair.
        # The values for baseline methods were taken from their respective papers.
        print('running cause effect pairs experiments ')
        run_cause_effect_pairs(args, config)

    if args.intervention:
        args.doc = 'interventions'
        make_and_set_dirs(args, config)
        if not args.plot:
            # Run proposed method to perform interventions on the toy example described in the manuscript
            print('running interventions on toy example')
            # intervention(dim=4, results_dir=args.run)
            run_interventions(args, config)
        else:
            plot_interventions(args, config)

    if args.counterfactual:
        args.doc = 'counterfactuals'
        make_and_set_dirs(args, config)
        # Run proposed method to perform counterfactuals on the toy example described in the manuscript
        print('running counterfactuals on toy example')
        counterfactuals(args, config)


if __name__ == '__main__':
    sys.exit(main())
