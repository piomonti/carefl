import numpy as np
import os
import pickle
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import scale

from models.carefl import CAReFl

PairDataDir = 'data/pairs/'


def res_save_name(config):
    return 'pair_{}_{}_{}_{}_{}.p'.format(config.data.pair_id,
                                          config.flow.architecture.lower(),
                                          config.flow.net_class.lower(),
                                          config.flow.nl,
                                          config.flow.nh)


def run_cause_effect_pairs(args, config):
    # skip these pairs, as indicated by Mooij et al (2016) because the variables
    # are not bivariate (i.e., X and Y are not univariate)
    skip_pairs = [52, 54, 55]
    i = config.data.pair_id

    if i in skip_pairs:
        return

    # polish format of pair_id
    pair_id = str(i)
    pair_id = '0' * (4 - len(pair_id)) + pair_id
    dat_id = np.loadtxt(PairDataDir + 'pair' + str(pair_id) + '.txt')[:, :2]
    dir_id = open(PairDataDir + 'pair' + str(pair_id) + '_des.txt', 'r').read().lower()
    # determine causal direction (from dir_id file):
    dir_id = dir_id.replace('\n', '')
    dir_id = dir_id.replace(':', '')
    dir_id = dir_id.replace(' ', '')
    if ('x-->y' in dir_id) | ('x->y' in dir_id):
        dir_id = 'x-->y'
    elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
        dir_id = 'y-->x'
    if config.data.remove_outliers:
        print('removing outliers')
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        y_pred = clf.fit_predict(dat_id)
        dat_id = dat_id[np.where(y_pred == 1)[0]]
    if config.data.scale:
        # scale data:
        dat_id = scale(dat_id)
        # dat_id = MinMaxScaler().fit_transform( dat_id )
    if config.data.verbose:
        print('Running experiments for CE Pair: ' + pair_id + ' with n=' + str(dat_id.shape[0]) + ' samples')
        print('True causal direction: ' + dir_id)
        print('baseline dist: ' + config.flow.prior_dist)

    results = {'p': [], 'c': [], 'correct': 0, 'dir': dir_id, 'sxy': [], 'syx': []}
    per_correct = 0
    n_sims = n_valid_sims = args.n_sims
    for sim in range(n_sims):
        config.training.seed = sim
        model = CAReFl(config)
        p, pred_model, sxy, syx = model.predict_proba(dat_id, return_scores=True)
        if not np.isnan(p):
            per_correct += pred_model == dir_id
            results['p'].append(p)
            results['c'].append(1. * (pred_model == dir_id))
            results['sxy'].append(sxy)
            results['syx'].append(syx)
        else:
            n_valid_sims -= 1
    results['correct'] = per_correct / n_valid_sims
    pickle.dump(results, open(os.path.join(args.output, res_save_name(config)), 'wb'))


def plot_pairs(args, config):
    cs = []
    ps = []
    sxys = []
    syxs = []
    for seed in range(1):
        args.seed = seed
        res = pickle.load(open(os.path.join(args.output, str(config.data.pair_idx), res_save_name(config)), 'rb'))
        cs.append(res['c'])
        ps.append(res['p'])
        sxys.append(res['sxy'] if 'sxy' in res.keys() else 0)
        syxs.append(res['syx'] if 'syx' in res.keys() else 0)
    print("Average correct:", np.mean(cs))
    print("Average p", np.nanmean(ps))
    print("sequence of p's:", ps)


def plot_all_pairs(args, config):
    # TODO: implement a function to plot accuracy on all pairs
    pass
