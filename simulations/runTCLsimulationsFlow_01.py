### initial trials on recovering nonlinear mixtures using TCL & conditional flow models
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
from nflib.flows import AffineConstantFlow, MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm, ClassCondNormalizingFlowModel
from nflib.spline_flows import NSF_AR, NSF_CL
from data.generateTCLdata import * 
from data.generate_artificial_tcl_data import *
from data.generateToyData import to_one_hot
from models.classConditionalFlow import Flow, ClassCondFlow
#from models.tcl import * 
from helper.solveHungarian import SolveHungarian

from sklearn.decomposition import FastICA, PCA
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

import pickle 


def RunTCLsimulationFlow( Ncomp, Nsegment, Nlayer, NobsSeg, seed, epochs, saveDir, device='cpu', verbose=False ):
    """
    run one iteration of unmixing using conditional flows

    """
    # # define simulation parameters 
    # Ncomp    = 4
    # Nsegment = 10
    # Nlayer   = 5
    # NobsSeg  = 512 * 2 

    assert Ncomp % 2 == 0 # for given flow architecture we need even dimension

    # generate some TCL data:
    np.random.seed( seed )
    dat     = gen_TCL_data( Nlayer=Nlayer, Nsegment=Nsegment, NsegmentObs=NobsSeg, Ncomp=Ncomp, NonLin='leaky', LinType='uniform',  negSlope=.2, Niter4condThresh=1e4, varyMean=True )
    dat_pca = PCA().fit_transform( dat['obs'] )
    label   = dat['labels']
    source  = dat['source']

    # -------------------------------------------------------------------------------
    #         Conditional Flow Model
    # -------------------------------------------------------------------------------
    # note that Ncomp must be even for this ! 

    prior = TransformedDistribution(Uniform(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)  # MultivariateNormal(loc=np.zeros((Ncomp,)), covariance_matrix = np.eye( Ncomp )).inv )  # SigmoidTransform().inv) 

    nfs_flow = NSF_CL 
    flows = [nfs_flow(dim=Ncomp, K=8, B=3, hidden_dim=16) for _ in range(Nlayer)]
    convs = [Invertible1x1Conv(dim=Ncomp) for _ in flows]
    norms = [ActNorm(dim=Ncomp) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))

    nclasses = Nsegment
    cflows = []
    segment_flow = AffineConstantFlow # NSF_CL
    for c in range(nclasses):
        flows_e = [segment_flow(dim=Ncomp) for _ in range(1)]
        convs_e = [Invertible1x1Conv(dim=Ncomp) for _ in flows_e]
        norms_e = [ActNorm(dim=Ncomp) for _ in flows_e]
        flows_e = list(itertools.chain(*zip(norms_e, convs_e, flows_e)))
        cflows.append( flows_e )

    flow_mod_cond = ClassCondFlow( prior, flows, cflows, device=device )

    flow_mod_cond.load_data( data=dat_pca, labels= to_one_hot( label )[0] )

    #print( flow_mod_cond.flow_share )

    loss_cond = flow_mod_cond.train( epochs = epochs, verbose=verbose )

    z_full = flow_mod_cond.forwardPassFlow( dat_pca, fullPass=False, labels=to_one_hot(label)[0] )
    # plt.figure()
    # plt.scatter( z_full[:,0], z_full[:,1], c=label) #.argmax(axis=1)) # so actually the raw flow does a reasonable job by itself ...

    results = {'dat_pca': dat_pca,
               'source': source,
               'labels': label,
               'recov': z_full,
               'MCC_ica': SolveHungarian( FastICA().fit_transform( z_full), source )[0],
               'MCC_noica': SolveHungarian( z_full, source )[0] }

    print("#########\nResults\nMCC (no ICA): {}\nMCC (w/ ICA): {}".format( np.round(results['MCC_noica'],2), np.round(results['MCC_ica'],2)))

    # save results
    filename = 'CondFlowTCLexp_Ncomp' + str(Ncomp) + '_Nlayer' + str(Nlayer) + '_NSeg' + str(Nsegment) + '_NobsSeg' + str(NobsSeg) + '_Seed' + str(seed) + '.p'

    pickle.dump( results, open( saveDir + filename, 'wb'))


