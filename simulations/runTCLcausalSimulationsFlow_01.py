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
from torch.distributions import MultivariateNormal, Normal, Laplace, Uniform, TransformedDistribution, SigmoidTransform
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
from nflib.flows import AffineConstantFlow, MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm, ClassCondNormalizingFlowModel, AffineHalfFlow
from nflib.spline_flows import NSF_AR, NSF_CL
from data.generateCausalTCLdata_02 import gen_2dcausalTCL_data, gen_toy_causal_dat
from data.generate_artificial_tcl_data import *
from data.generateToyData import to_one_hot
from helper.solveHungarian import SolveHungarian

from models.classConditionalFlow import Flow, ClassCondFlow

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import scale 
import pandas as pd 
import pickle 

def bivariateTCLcausalSimulationFlow( Nsegment, Nlayer, NobsSeg, Nsims, epochs, saveDir, device='cpu', verbose=False ):
    """
    run one iteration of unmixing using conditional flows

    """
    # # define simulation parameters 
    # Ncomp    = 4
    # Nsegment = 10
    # Nlayer   = 5
    # NobsSeg  = 512 * 2 

    Ncomp = 2 # this is for bivariate causal discovery
    LR_xy      = np.zeros( Nsims )
    LR_yx      = np.zeros( Nsims )
    models     = ['NA'] * Nsims
    predmodels = ['NA'] * Nsims

    for i in range( Nsims ):
        np.random.seed( i )
        dat     = gen_2dcausalTCL_data( Nlayer=Nlayer, Nsegment=Nsegment, NsegmentObs=NobsSeg )
        #dat     = gen_causal_TCL_data( Nlayer=Nlayer, Nsegment=Nsegment, NsegmentObs=NobsSeg, Ncomp=Ncomp, NonLin='leaky', LinType='uniform',  negSlope=.2, Niter4condThresh=1e4, varyMean=True )

        if np.random.uniform() < .5: # just removing this because its confusing me !
            model   = 'y->x'  #dat['model']
            order   = [0,1]
        else:
            model   = 'x->y'  #dat['model']
            order   = [1,0]

        dat_pca = ( dat['obs'] )[:, order]
        label   = dat['labels']
        source  = dat['source'][:, order]
        device  = 'cpu'
        models[i] =  model 

        # -------------------------------------------------------------------------------
        #         Conditional Flow Model: X->Y
        # -------------------------------------------------------------------------------
        # note that Ncomp must be even for this ! 
        # Laplace prior:
        prior = Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )) #TransformedDistribution(Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)  
        # uniform prior:
        #prior = TransformedDistribution(Uniform(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)   

        nfs_flow = AffineHalfFlow 
        flows = [nfs_flow(dim=Ncomp, nh=8, parity=False) for _ in range(Nlayer+1)]

        # note: if we remove the convs and norms above, the one input will remain unchanged throughout due to parity=False

        nclasses = Nsegment
        cflows = []
        segment_flow = AffineConstantFlow # NSF_CL
        for c in range(nclasses):
            flows_e = [segment_flow(dim=Ncomp) for _ in range(1)]
            cflows.append( flows_e )

        flow_mod_cond = ClassCondFlow( prior, flows, cflows, device='cpu' )

        flow_mod_cond.load_data( data=dat_pca, labels= to_one_hot( label )[0] )

        # now we train this model and store the likelihood:
        loss_cond = flow_mod_cond.train( epochs = epochs, verbose=False )

        #LR_xy = flow_mod_cond.EvalLL( dat_pca, to_one_hot(label)[0] ).mean()

        LR_xy[i] = np.nanmean( flow_mod_cond.EvalLL( dat_pca, to_one_hot(label)[0] ) )

        # -------------------------------------------------------------------------------
        #         Conditional Flow Model: Y->X
        # -------------------------------------------------------------------------------
        ## now consider the reverse model:
        # use Laplace prior
        prior_rev = Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )) # TransformedDistribution(Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)  # MultivariateNormal(loc=np.zeros((Ncomp,)), covariance_matrix = np.eye( Ncomp )).inv )  # SigmoidTransform().inv) 

        nfs_flow = AffineHalfFlow 
        flows_rev = [nfs_flow(dim=Ncomp, nh=8, parity=False) for _ in range(Nlayer+1)]

        # note: if we remove the convs and norms above, the one input will remain unchanged throughout due to parity=False

        nclasses = Nsegment
        cflows_rev = []
        segment_flow_rev = AffineConstantFlow # NSF_CL
        for c in range(nclasses):
            flows_e_rev = [segment_flow(dim=Ncomp) for _ in range(1)]
            cflows_rev.append( flows_e_rev )

        flow_mod_cond_rev = ClassCondFlow( prior_rev, flows_rev, cflows_rev , device='cpu' )

        flow_mod_cond_rev.load_data( data=dat_pca[:,[1,0]], labels= to_one_hot( label )[0] )

        # now we train this model and store the likelihood:
        loss_cond_rev = flow_mod_cond_rev.train( epochs = epochs, verbose=False )

        LR_yx[i] = np.nanmean( flow_mod_cond_rev.EvalLL( dat_pca[:,[1,0]], to_one_hot(label)[0] ) )

        predmodels[i] =  'x->y' if LR_xy[i] > LR_yx[i] else 'y->x' 

        res = pd.DataFrame({'true':models, 'LR_xy':LR_xy, 'LR_yx': LR_yx, 'pred':predmodels})
        print(res.iloc[:i,:])
        print( np.mean( res['true'][:i]==res['pred'][:i]))

    # save results
    filename = 'CondFlowCausalTCLexp_Nlayer' + str(Nlayer) + '_NSeg' + str(Nsegment) + '_NobsSeg' + str(NobsSeg) + '_Nsim' + str(Nsims) + '.p'

    pickle.dump( res, open( saveDir + filename, 'wb'))













