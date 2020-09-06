### run simulations
#
#
#
import numpy as np
import pandas as pd

from data.generate_synth_data import gen_synth_causal_dat
from models import RECI, ANM, EntropyLR, BivariateFlowLR, LinearNOTEARS


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

    # make results DF:
    results = pd.DataFrame({x: ['NA'] * nSims for x in algolist})
    # add true direction to results
    results['true'] = 'NA'  # fill in later #['x->y' if l==1 else 'y->x' for l in labels.label]

    # run experiments
    reci_form_dict = {'linear': 'poly', 'hoyer2009': 'poly', 'nueralnet_l1': 'GP'}

    for sim in range(nSims):
        # print(sim)
        np.random.seed(sim)
        data, mod_dir = gen_synth_causal_dat(nObs=nPoints, causalFunc=causal_mechanism)
        results.loc[sim, 'true'] = mod_dir
        for a in algolist:
            if a == 'LRHyv':
                mod = EntropyLR()
            elif a == 'ANM':
                mod = ANM()
            elif a == 'RECI':
                mod = RECI(form=reci_form_dict[causal_mechanism], scale_input=True)
            elif a == 'FlowCD':
                mod = BivariateFlowLR(n_layers=[2], n_hidden=[1], split=.8, opt_method='scheduling')
            elif a == 'notears':
                mod = LinearNOTEARS(lambda1=.01, loss_type='l2', w_threshold=0)
            else:
                raise ValueError('Unknown algorithm')

            p, dir = mod.predict_proba(data=data)
            results.loc[sim, a] = dir

    # summarize results
    for a in algolist:
        print('Algo: {}\t Correct:{}'.format(a, (results[a] == results['true']).mean()))

    return results
