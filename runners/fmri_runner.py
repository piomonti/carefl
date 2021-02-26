import numpy as np
import os
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale

from models.carefl import CAREFL


def run_fmri(args, config):
    subject_int_ids = {
        292: {'rest': 0, 'stim': 2},
        294: {'rest': 0, 'stim': 1},
        302: {'rest': 0, 'stim': 1},
        303: {'rest': 0, 'stim': 3},
        307: {'rest': 0, 'stim': 1},
        316: {'rest': 0, 'stim': 1},
        320: {'rest': 0, 'stim': 1},
        # 330:{'rest': 0, 'stim': 5},
        352: {'rest': 0, 'stim': 1},
        357: {'rest': 0, 'stim': 1},
        372: {'rest': 0, 'stim': 2},
        376: {'rest': 0, 'stim': 1},
        # 384:{'rest': 0, 'stim': 1},
        # 395:{'rest': 0, 'stim': 1},
        399: {'rest': 0, 'stim': 1},
        400: {'rest': 0, 'stim': 1},
        403: {'rest': 0, 'stim': 1},
        # 413:{'rest': 0, 'stim': 1}

    }

    # define ROIs
    rois = np.array([28, 44])  # just take 2 ROIs for now
    rois = rois[::-1]

    ##3 BEGIN RUNNING EXP
    results = []
    for k in subject_int_ids.keys():
        # load in data
        path = os.path.join('data', 'fmri', 'subject_data')
        files = [f for f in os.listdir(path) if 'sub-' + str(k) + 'run' in f]

        dat = [np.array(pd.read_csv(os.path.join(path, f), header=0)) for f in files]
        scale_dat = [scale(d, with_mean=True) for d in dat]

        # ------ define train/test data
        train_dat = scale_dat[subject_int_ids[k]['rest']][:, rois]
        intervene_dat = scale_dat[subject_int_ids[k]['stim']][:, rois]

        # ------ run models
        mod = CAREFL(config)
        mod.fit_to_sem(train_dat)
        xvals = intervene_dat[:, 0]
        int_pred = np.array([mod.predict_intervention(x, n_samples=500)[1][0, 1] for x in xvals])

        ## compare with GPs and Linear Regression
        lm = Ridge()  # LinearRegression()
        lm.fit(X=train_dat[:, 0].reshape((-1, 1)), y=train_dat[:, 1])
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        kernel = 1.0 * RBF(1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=25, n_restarts_optimizer=10)
        gp.fit(X=train_dat[:, 0].reshape((-1, 1)), y=train_dat[:, 1])

        results_df = pd.DataFrame({'carefl': np.abs((int_pred) - (intervene_dat[:, 1])),
                                   'linear_regression': np.abs(
                                       (lm.predict(xvals.reshape((-1, 1)))) - (intervene_dat[:, 1])),
                                   'gaussian_process': np.abs(
                                       (gp.predict(xvals.reshape((-1, 1)))) - (intervene_dat[:, 1]))})
        results_df.median(axis=0)
        results.append(results_df.median(axis=0))
        print('### Subject ' + str(k))
        print(results_df.median(axis=0))

    print('\n\n\n\n')
    print(pd.DataFrame(results).mean(axis=0))
