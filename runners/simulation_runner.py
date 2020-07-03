### run simulations
#
#
#

import itertools 

import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader

# load flows
from nflib.flows import AffineConstantFlow, MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm
from nflib.spline_flows import NSF_AR, NSF_CL
from CauseEffectPairs.Run_CEP_01 import * 
from models.classConditionalFlow import Flow #, ClassCondFlow
from models.bivariateFlowCD import BivariateFlowCD

# load baselines
#from cdt.causality.pairwise import IGCI
from models.baselines import base_entropy_ratio, linear_notears_dir, ANM, cumulant_hyv13_ratio, RECI, nonlinear_notears_dir, NotearsMLP, notears_nonlinear

# load data generating code:
from data.generate_synth_data import gen_synth_causal_dat
#from cdt.data import CausalPairGenerator
#from cdt.data.causal_mechanisms import uniform_noise

from sklearn.decomposition import FastICA, PCA
import pickle 

def laplace_noise(points):
    """Init a noise variable."""
    return np.random.laplace( size=(points, 1) ) 

# define function to run simulations
def RunSimulations( nSims=25, nPoints=100, causal_mechanism='linear',
    algolist=['FlowCD', 'notears', 'LRHyv', 'IGCI', 'ANM', 'RECI'] ):
    """
    run simulations. We use the generator from cdt module

    INPUT:
    	- nSims: number of simulations to run
    	- nPoints: number of observations of bivariate data
    	- causal_mechanism: determine causal associations between bivariate data

    """

    # nSims = 10
    # nPoints = 100
    # algolist = ['LRHyv']
    # causal_mechanism='linear'

    # make results DF:
    results = pd.DataFrame({x:['NA']*nSims for x in algolist})

    # generate data:
    #gen = CausalPairGenerator( causal_mechanism=causal_mechanism, noise=laplace_noise, noise_coeff=.2) #, noise='uniform' )
    #np.random.seed(0) # doesnt quite make things reproducible ..
    #data, labels = gen.generate( nSims, npoints=nPoints)	

    # add true direction to results
    results['true'] = 'NA' # fill in later #['x->y' if l==1 else 'y->x' for l in labels.label]

    # run experiments
    reci_form_dict = {'linear': 'poly', 'hoyer2009': 'poly', 'nueralnet_l1':'GP'}

    for sim in range( nSims ):
        #print(sim)
        np.random.seed(sim)
        dat, mod_dir = gen_synth_causal_dat( nObs=nPoints, causalFunc=causal_mechanism )
        data = pd.DataFrame({'A': [dat[:,0]], 'B':[dat[:,1]]}) 
        results.loc[ sim, 'true'] = mod_dir
        for a in algolist:
            if a == 'LRHyv':
                results.loc[sim, a] = base_entropy_ratio( x=dat[:,0], y=dat[:,1] )[1]
            if a == 'ANM':
                mod = ANM()
                results.loc[sim, a] =  'x->y' if mod.predict_proba( data=data.iloc[0] ) > 0 else 'y->x'
            if a == 'IGCI':
                mod = IGCI()
                results.loc[sim, a] =  'x->y' if mod.predict_proba( dataset=data.iloc[0] )[0] <0 else 'y->x'
            if a == 'RECI':
                mod = RECI()
                results.loc[sim, a] =  'x->y' if mod.predict_proba( data=dat, form=reci_form_dict[causal_mechanism], scale_input=True ) < 0 else 'y->x'
            if a == 'FlowCD':
                mod = BivariateFlowCD(Nlayers=[2], Nhidden=[1], priorDist='laplace', TrainSplit=.8, epochs=100, optMethod='scheduling' )
                results.loc[sim, a] = mod.train( dat )[1]
                #mod.train( np.vstack(( data.A.iloc[0], data.B.iloc[0])).T )[1]
            if a == 'notears':
                results.loc[sim, a] = linear_notears_dir( x=dat[:,0], y=dat[:,1], lambda1=.01, loss_type='l2', w_threshold=0 )[1]

    # summarize results
    for a in algolist:
        print('Algo: {}\t Correct:{}'.format(a, (results[a]==results['true']).mean()  ))

    return results

if True:
    nvals = [ 25, 50, 75, 100,  150, 250, 500 ]
    results = []
    causal_mechanism = 'nueralnet_l1' # 'linear' #   'hoyer2009' #  'linear' #   'nueralnet_l1' # 
    nsims = 250 
    algos =  ['FlowCD', 'LRHyv', 'notears', 'RECI', 'ANM' ] # ['notears'] # ['RECI', 'notears'] #
    print('Mechanism: {}'.format(causal_mechanism))
    for n in nvals:
        print('### {} ###'.format(n))
        results.append( RunSimulations( nSims=nsims,  nPoints=n, causal_mechanism=causal_mechanism, algolist=algos ) )


    import pickle
    pickle.dump( results, open( 'results/' + causal_mechanism + "_results.p", 'wb') )





