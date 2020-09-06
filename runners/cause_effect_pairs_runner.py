import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import scale

# load flows
from models.affine_flow_cd import BivariateFlowLR

PairDataDir = 'data/pairs/'


def run_single_pair(pair_id, Nlayers, Nhidden, priorDist='laplace', TrainSplit=1., epochs=100, optMethod='adam',
                    removeOutliers=False, scaleDat=True, verbose=False):
    """
    run cause effect discovery for given pair id
    """

    # check input
    assert priorDist in ['laplace', 'uniform']

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

    if removeOutliers:
        print('removing outliers')
        clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        y_pred = clf.fit_predict(dat_id)
        dat_id = dat_id[np.where(y_pred == 1)[0],]

    # scale data:
    if scaleDat:
        dat_id = scale(dat_id)
    #   dat_id = MinMaxScaler().fit_transform( dat_id )

    if verbose:
        print('Running experiments for CE Pair: ' + pair_id + ' with n=' + str(dat_id.shape[0]) + ' samples')
        print('True causal direction: ' + dir_id)
        print('baseline dist: ' + priorDist)

    model = BivariateFlowLR(Nhidden, Nlayers, TrainSplit, priorDist, epochs, opt_method=optMethod, verbose=verbose)
    p, _, results = model.fit_flows(dat_id)
    predModel = model.direction
    return results, predModel, dir_id, np.minimum(np.unique(dat_id[:, 0]).shape[0] / float(dat_id.shape[0]),
                                                  np.unique(dat_id[:, 1]).shape[0] / float(dat_id.shape[0]))


def RunCauseEffectPairs():
    # define some simulation parameters
    skipPairs = [52, 54, 55]
    # skip these pairs, as indicated by Mooij et al (2016) because the variables
    # are not bivariate (i.e., X and Y are not univariate)

    SplitPerc = .8
    scaleDat = True
    priorDist = 'laplace'
    epochs = 500 + 250
    optMethod = 'schedule'
    correctCount = 0
    runningCount = 0
    BinaryCutoff = .15
    correctCount_nobinary = 0
    runningCount_nobinary = 0
    LayerList = [1, 3]
    depthList = [5]

    for i in range(1, 108):
        if i in skipPairs:
            pass
        else:
            res, predModel, trueModel, ctsRatio = run_single_pair(i, LayerList, depthList, priorDist=priorDist,
                                                                  TrainSplit=SplitPerc, epochs=epochs,
                                                                  optMethod=optMethod,
                                                                  scaleDat=scaleDat, verbose=True)

            runningCount += 1
            if ctsRatio > BinaryCutoff:
                runningCount_nobinary += 1
            if predModel.replace('-', '') == trueModel.replace('-', ''):
                print('Correct!')
                correctCount += 1
                if ctsRatio > BinaryCutoff:
                    correctCount_nobinary += 1
            # TODO: is correctrCount_nobinary important?
            print('running mean: ' + str(float(correctCount) / runningCount))
