import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

from data.eeg import eeg_data
from models import RECI, ANM, EntropyLR, CAReFl, LinearNOTEARS


def res_save_name(config, algo):
    if 'carefl' not in algo.lower():
        return 'eeg_{}.p'.format(config.data.n_points)
    return 'eeg_{}_{}_{}_{}_{}.p'.format(config.data.n_points,
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
    n_sims = n_valid_sims = args.n_sims
    algo = config.algorithm
    results = {'p': [], 'c': [], 'correct': 0}
    per_correct = 0
    for sim in range(n_sims):
        data, mod_dir = eeg_data(idx=config.data.timeseries_idx, shuffle=True, lag=config.data.lag)
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
            results['p'].append(p)
            results['c'].append(1. * (direction == mod_dir))
        else:
            n_valid_sims -= 1
    results['correct'] = per_correct / n_valid_sims
    pickle.dump(results, open(os.path.join(args.output, res_save_name(config, algo)), 'wb'))
