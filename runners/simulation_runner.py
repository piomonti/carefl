### run simulations
#
#
#
import numpy as np
import pandas as pd

# load data generating code:
from data.generate_synth_data import gen_synth_causal_dat
# load flows
from models import RECI, ANM, linear_notears_dir, base_entropy_ratio, BivariateFlowLR


def laplace_noise(points):
    """Init a noise variable."""
    return np.random.laplace(size=(points, 1))


# define function to run simulations
def RunSimulations(nSims=25, nPoints=100, causal_mechanism='linear', algolist=None):
    """
    run simulations. We use the generator from cdt module

    INPUT:
        - nSims: number of simulations to run
        - nPoints: number of observations of bivariate data
        - causal_mechanism: determine causal associations between bivariate data
        - algolist: causal discovery methods to run
    """

    if algolist is None:
        algolist = ['FlowCD', 'notears', 'LRHyv', 'ANM', 'RECI']

    # nSims = 10
    # nPoints = 100
    # algolist = ['LRHyv']
    # causal_mechanism='linear'

    # make results DF:
    results = pd.DataFrame({x: ['NA'] * nSims for x in algolist})

    # generate data:
    # gen = CausalPairGenerator( causal_mechanism=causal_mechanism, noise=laplace_noise, noise_coeff=.2)
    # np.random.seed(0) # doesnt quite make things reproducible ..
    # data, labels = gen.generate( nSims, npoints=nPoints)

    # add true direction to results
    results['true'] = 'NA'  # fill in later #['x->y' if l==1 else 'y->x' for l in labels.label]

    # run experiments
    reci_form_dict = {'linear': 'poly', 'hoyer2009': 'poly', 'nueralnet_l1': 'GP'}

    for sim in range(nSims):
        # print(sim)
        np.random.seed(sim)
        dat, mod_dir = gen_synth_causal_dat(nObs=nPoints, causalFunc=causal_mechanism)
        data = pd.DataFrame({'A': [dat[:, 0]], 'B': [dat[:, 1]]})
        results.loc[sim, 'true'] = mod_dir
        for a in algolist:
            if a == 'LRHyv':
                results.loc[sim, a] = base_entropy_ratio(x=dat[:, 0], y=dat[:, 1])[1]
            if a == 'ANM':
                mod = ANM()
                results.loc[sim, a] = 'x->y' if mod.predict_proba(data=data.iloc[0]) > 0 else 'y->x'
            # if a == 'IGCI':
            #     mod = IGCI()
            #     results.loc[sim, a] = 'x->y' if mod.predict_proba(dataset=data.iloc[0])[0] < 0 else 'y->x'
            if a == 'RECI':
                mod = RECI()
                results.loc[sim, a] = 'x->y' if mod.predict_proba(data=dat, form=reci_form_dict[causal_mechanism],
                                                                  scale_input=True) < 0 else 'y->x'
            if a == 'FlowCD':
                mod = BivariateFlowLR(n_layers=[2], n_hidden=[1], prior_dist='laplace', split=.8, epochs=100,
                                      opt_method='scheduling')
                results.loc[sim, a] = 'x->y' if mod.predict_proba(dat) >= 0 else 'y->x'
                # results.loc[sim, a] = mod.train(dat)[1]
                # mod.train( np.vstack(( data.A.iloc[0], data.B.iloc[0])).T )[1]
            if a == 'notears':
                results.loc[sim, a] = \
                    linear_notears_dir(x=dat[:, 0], y=dat[:, 1], lambda1=.01, loss_type='l2', w_threshold=0)[1]

    # summarize results
    for a in algolist:
        print('Algo: {}\t Correct:{}'.format(a, (results[a] == results['true']).mean()))

    return results
