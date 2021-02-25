import sys

import argparse
import numpy as np
import os
import torch
import yaml

from runners.cause_effect_pairs_runner import run_cause_effect_pairs
from runners.counterfactual_trials import counterfactuals
from runners.eeg_runner import run_eeg, plot_eeg
from runners.fmri_runner import run_fmri
from runners.intervention_trials import run_interventions, plot_interventions
from runners.simulation_runner import run_simulations, plot_simulations


def parse_input():
    parser = argparse.ArgumentParser(description='Reproduce experiments for Causal Autoregressive Flows')
    parser.add_argument('--run', type=str, default='results', help='Path for saving results.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--plot', action='store_true', help='plot experiments')
    parser.add_argument('-s', '--simulation', action='store_true', help='run the CD exp on synthetic data')
    parser.add_argument('-p', '--pairs', action='store_true', help='Run Cause Effect Pairs experiments')
    parser.add_argument('-i', '--intervention', action='store_true', help='run intervention exp on toy example')
    parser.add_argument('-c', '--counterfactual', action='store_true', help='run counterfactual exp on toy example')
    parser.add_argument('-e', '--eeg', action='store_true', help='run eeg exp')
    parser.add_argument('-f', '--fmri', action='store_true', help='run fmri exp')

    # params to overwrite config file. useful for batch running in slurm
    parser.add_argument('-y', '--config', type=str, default='', help='config file to use')
    parser.add_argument('-m', '--causal-mech', type=str, default='', help='Dataset to run synthetic experiments on.')
    parser.add_argument('-a', '--algorithm', type=str, default='', help='algorithm to run')
    parser.add_argument('-n', dest='n_points', type=int, default=-1,
                        help='number of simulated data points --- also controls timeseris_idx/pair_idx for real data exps')
    parser.add_argument('--noise-dist', type=str, default='', help='noise dist')
    parser.add_argument('--nl', type=int, default=-1, help='number of layer for flow')
    parser.add_argument('--nh', type=int, default=-1, help='number of hidden units for nets')
    parser.add_argument('--n-sims', type=int, default=-1, help='Number of synthetic simulations to run')

    return parser.parse_args()


def debug_options(args, config):
    """
    helper function to overwrite options in config file based on debug flags
    """
    if args.causal_mech != '':
        config.data.causal_mech = args.causal_mech
    if args.algorithm != '':
        config.algorithm = args.algorithm
    if args.n_points != -1:
        config.data.n_points = args.n_points  # for interventions / simulations
        config.data.pair_id = args.n_points  # for pairs
        config.data.timeseries_idx = args.n_points  # for arrow of time on eeg
    if args.noise_dist != '':
        config.data.noise_dist = args.noise_dist
    if args.nl != -1:
        config.flow.nl = args.nl
    if args.nh != -1:
        config.flow.nh = args.nh
    if args.n_sims != -1:
        config.n_sims = args.n_sims


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
    """
    create folders for checkpoints and results
    """
    if config.algorithm.lower() == 'carefl':
        args.algo = os.path.join('carefl' + 'ns' * (1 - config.flow.scale), config.flow.architecture.lower())
    else:
        args.algo = config.algorithm.lower()
    os.makedirs(args.run, exist_ok=True)
    args.output = os.path.join(args.run, args.doc, args.algo)
    os.makedirs(args.output, exist_ok=True)


def read_config(args):
    """
    automatically find the right config file from run flags
    """
    if args.config != '':
        return
    if args.simulation:
        args.config = 'simulations.yaml'
    if args.intervention:
        args.config = 'interventions.yaml'
    if args.pairs:
        args.config = 'pairs.yaml'
    if args.counterfactual:
        args.config = 'counterfactuals.yaml'
    if args.eeg:
        args.config = 'eeg.yaml'
    if args.fmri:
        args.config = 'fmri.yaml'


def main():
    # parse command line arguments
    args = parse_input()
    read_config(args)
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

    # causal discovery
    if args.simulation:
        # run algorithm on simulated data
        # and save the results as pickle files which can be used later to plot Fig 1.
        args.doc = os.path.join('simulations', config.data.causal_mech)
        make_and_set_dirs(args, config)
        if not args.plot:
            print('Running {} on {} synthetic experiments ({} simulations - {} points)'.format(config.algorithm,
                                                                                               config.data.causal_mech,
                                                                                               config.n_sims,
                                                                                               config.data.n_points))
            run_simulations(args, config)
        else:
            plot_simulations(args, config)

    if args.pairs:
        # Run proposed method on CauseEffectPair dataset
        # The values for baseline methods were taken from their respective papers.
        args.doc = 'pairs'
        make_and_set_dirs(args, config)
        print('running cause effect pairs experiments ')
        run_cause_effect_pairs(args, config)

    if args.eeg:
        args.doc = 'eeg'
        make_and_set_dirs(args, config)
        config.training.seed = args.seed
        if not args.plot:
            print('running eeg experiment')
            run_eeg(args, config)
        else:
            plot_eeg(args, config)

    # interventiuons
    if args.intervention:
        # Run proposed method to perform interventions on the toy example described in the manuscript
        args.doc = 'interventions'
        make_and_set_dirs(args, config)
        if not args.plot:
            print('running interventions on toy example')
            run_interventions(args, config)
        else:
            plot_interventions(args, config)

    if args.fmri:
        # Run proposed method to perform counterfactuals on the toy example described in the manuscript
        args.doc = 'fmri'
        make_and_set_dirs(args, config)
        print('running interventions on es-fMRI data')
        run_fmri(args, config)

    # counterfactuals
    if args.counterfactual:
        # Run proposed method to perform counterfactuals on the toy example described in the manuscript
        args.doc = 'counterfactuals'
        make_and_set_dirs(args, config)
        print('running counterfactuals on toy example')
        counterfactuals(args, config)


if __name__ == '__main__':
    sys.exit(main())
