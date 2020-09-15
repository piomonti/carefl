import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import scale

from models.carefl import CAReFl

PairDataDir = 'data/pairs/'


def run_single_pair(config, pair_id, remove_outliers=False, scale_dat=True, verbose=False):
    """
    run cause effect discovery for given pair id
    """

    # polish format of pair_id
    pair_id = str(pair_id)
    pair_id = '0' * (4 - len(pair_id)) + pair_id
    dat_id = np.loadtxt(PairDataDir + 'pair' + str(pair_id) + '.txt')
    dir_id = open(PairDataDir + 'pair' + str(pair_id) + '_des.txt', 'r').read().lower()

    # determine causal direction (from dir_id file):
    dir_id = dir_id.replace('\n', '')
    dir_id = dir_id.replace(':', '')
    dir_id = dir_id.replace(' ', '')
    if ('x-->y' in dir_id) | ('x->y' in dir_id):
        dir_id = 'x-->y'
    elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
        dir_id = 'y-->x'

    if remove_outliers:
        print('removing outliers')
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        y_pred = clf.fit_predict(dat_id)
        dat_id = dat_id[np.where(y_pred == 1)[0]]

    # scale data:
    if scale_dat:
        dat_id = scale(dat_id)
    #   dat_id = MinMaxScaler().fit_transform( dat_id )

    if verbose:
        print('Running experiments for CE Pair: ' + pair_id + ' with n=' + str(dat_id.shape[0]) + ' samples')
        print('True causal direction: ' + dir_id)
        print('baseline dist: ' + config.flow.prior_dist)

    model = CAReFl(config)
    p = model.flow_lr(dat_id)
    pred_model = model.direction
    return pred_model, dir_id, np.minimum(np.unique(dat_id[:, 0]).shape[0] / float(dat_id.shape[0]),
                                                   np.unique(dat_id[:, 1]).shape[0] / float(dat_id.shape[0]))


def run_cause_effect_pairs(args, config):
    # define some simulation parameters
    skip_pairs = [52, 54, 55]
    # skip these pairs, as indicated by Mooij et al (2016) because the variables
    # are not bivariate (i.e., X and Y are not univariate)
    correct_count = 0
    running_count = 0
    binary_cutoff = .15
    correct_count_nobinary = 0
    running_count_nobinary = 0
    for i in range(1, 108):
        if i in skip_pairs:
            pass
        else:
            pred_model, true_model, cts_ratio = run_single_pair(config, i, remove_outliers=False, scale_dat=True,
                                                                     verbose=True)

            running_count += 1
            if cts_ratio > binary_cutoff:
                running_count_nobinary += 1
            if pred_model.replace('-', '') == true_model.replace('-', ''):
                print('Correct!')
                correct_count += 1
                if cts_ratio > binary_cutoff:
                    correct_count_nobinary += 1
            # TODO: is correctrCount_nobinary important?
            print('running mean: ' + str(float(correct_count) / running_count))
