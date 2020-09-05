import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import scale
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform

from models.classConditionalFlow import Flow  # , ClassCondFlow
# load flows
from nflib.flows import AffineConstantFlow, AffineFullFlow
from nflib.nets import MLP1layer

PairDataDir = '../data/pairs/'


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

    # load in the data
    # os.chdir(PairDataDir)
    dat_id = np.loadtxt(PairDataDir + 'pair' + str(pair_id) + '.txt')
    dir_id = open(PairDataDir + 'pair' + str(pair_id) + '_des.txt',
                  'r').read().lower()  # .split('ground truth:')[1].strip() #split('\n')[1]

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
    # dat_id = MinMaxScaler().fit_transform( dat_id )

    if dat_id.shape[1] > 2:
        dat_id = dat_id[:, :2]

    if TrainSplit == 1.:
        testDat_id = np.copy(dat_id)
    else:
        testDat_id = np.copy(dat_id[int(TrainSplit * dat_id.shape[0]):, :])
        dat_id = dat_id[:int(TrainSplit * dat_id.shape[0]), :]

    if verbose:
        print('Running experiments for CE Pair: ' + pair_id + ' with n=' + str(dat_id.shape[0]) + ' samples')
        print('True causal direction: ' + dir_id)
        print('baseline dist: ' + priorDist)

    # define final variables
    Ncomp = 2
    label = np.zeros(dat_id.shape[0])
    nfs_flow = AffineFullFlow  # AffineHalfFlow
    nfs_mlp = MLP1layer
    segment_flow = AffineConstantFlow

    # now start running LR methods
    results = pd.DataFrame({'L': np.repeat(Nlayers, len(Nhidden)),
                            'nh': Nhidden * len(Nlayers),
                            'x->y': [0] * len(Nlayers) * len(Nhidden),
                            'y->x': [0] * len(Nlayers) * len(Nhidden)})

    for l in Nlayers:
        for nh in Nhidden:
            # -------------------------------------------------------------------------------
            #         Conditional Flow Model: X->Y
            # -------------------------------------------------------------------------------
            torch.manual_seed(0)
            if priorDist == 'laplace':
                prior = Laplace(torch.zeros(Ncomp), torch.ones(Ncomp))
                # TransformedDistribution(Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )), # SigmoidTransform().inv)
            else:
                print('.')
                prior = TransformedDistribution(Uniform(torch.zeros(Ncomp), torch.ones(Ncomp)),
                                                SigmoidTransform().inv)  # Logistic distribution

            flows = [nfs_flow(dim=Ncomp, nh=nh, parity=False, net_class=nfs_mlp) for _ in range(l)]
            # cflows = [ [segment_flow(dim=Ncomp) ] ]

            # flow_mod_cond = ClassCondFlow( prior, flows, cflows, device='cpu' )
            flow_mod_cond = Flow(prior, flows, device='cpu')
            flow_mod_cond.load_data(data=dat_id)  # , labels= to_one_hot( label )[0] )

            # now we train this model and store the likelihood:
            loss_cond = flow_mod_cond.train(epochs=epochs, optMethod=optMethod, verbose=False)
            # print(np.nanmean( flow_mod_cond.EvalLL( dat_pca, to_one_hot(label)[0] ) ))

            # -------------------------------------------------------------------------------
            #         Conditional Flow Model: Y->X
            # -------------------------------------------------------------------------------
            torch.manual_seed(0)
            if priorDist == 'laplace':
                prior_rev = Laplace(torch.zeros(Ncomp), torch.ones(Ncomp))
                # TransformedDistribution(Laplace(torch.zeros( Ncomp ),
                # torch.ones( Ncomp )), SigmoidTransform().inv)
                # MultivariateNormal(loc=np.zeros((Ncomp,)), covariance_matrix =
                # np.eye( Ncomp )).inv)  # SigmoidTransform().inv)
            else:
                print('.')
                prior_rev = TransformedDistribution(Uniform(torch.zeros(Ncomp), torch.ones(Ncomp)),
                                                    SigmoidTransform().inv)  # Logistic distribution

            flows_rev = [nfs_flow(dim=Ncomp, nh=nh, parity=False, net_class=nfs_mlp) for _ in range(l)]
            # cflows_rev = [ [ segment_flow(dim=Ncomp) ] ]

            flow_mod_cond_rev = Flow(prior_rev, flows_rev, device='cpu')
            flow_mod_cond_rev.load_data(data=dat_id[:, [1, 0]])  # , labels= to_one_hot( label )[0] )

            # now we train this model and store the likelihood:
            loss_cond_rev = flow_mod_cond_rev.train(epochs=epochs, optMethod=optMethod, verbose=False)
            # print(np.nanmean( flow_mod_cond_rev.EvalLL( dat_pca[:,[1,0]], to_one_hot(label)[0] ) ))

            # store results
            # results.loc[(results.L == l) & (results.nh == nh), 'x->y'] = np.nanmean(
            #     flow_mod_cond.EvalLL(testDat_id, to_one_hot(label[: testDat_id.shape[0]])[0]))
            # results.loc[(results.L == l) & (results.nh == nh), 'y->x'] = np.nanmean(
            #     flow_mod_cond_rev.EvalLL(testDat_id[:, [1, 0]], to_one_hot(label[: testDat_id.shape[0]])[0]))
            results.loc[(results.L == l) & (results.nh == nh), 'x->y'] = np.nanmean(flow_mod_cond.EvalLL(testDat_id))
            results.loc[(results.L == l) & (results.nh == nh), 'y->x'] = np.nanmean(
                flow_mod_cond_rev.EvalLL(testDat_id[:, [1, 0]]))

    print(results)
    # compute the consensus
    p = results['x->y'].max() - results['y->x'].max()  # np.mean( results['x->y'] > results['y->x'] )
    predModel = 'x->y' if p >= 0 else 'y->x'

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

            print('running mean: ' + str(float(correctCount) / runningCount))
