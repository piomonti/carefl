### initial trials on recovering nonlinear mixtures using TCL & conditional flow models
#
#

import itertools

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
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
from helper.solveHungarian import SolveHungarian
from models.classConditionalFlow import Flow, ClassCondFlow
#from models.tcl import * 

from sklearn.decomposition import FastICA, PCA
#import pdb 
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#pdb.set_trace()

# define simulation parameters 
Ncomp    = 2
Nsegment = 5
Nlayer   = 2
NobsSeg  = 512 * 2 
#device   = 'cuda'

# generate some TCL data:
np.random.seed( 2 )
dat     = gen_TCL_data( Nlayer=Nlayer, Nsegment=Nsegment, NsegmentObs=NobsSeg, Ncomp=Ncomp, NonLin='leaky', LinType='uniform',  negSlope=.2, Niter4condThresh=1e4, varyMean=True )
dat_pca = PCA().fit_transform( dat['obs'] )
label   = dat['labels']
source  = dat['source']

# plt.figure()
# plt.scatter( dat_pca[:,0], dat_pca[:,1], c=dat['labels'])

# plt.figure()
# plt.scatter( dat['source'][:,0], dat['source'][:,1], c=dat['labels'])


# sensor, source, label = generate_artificial_data(num_comp=Ncomp,
#                                         	         num_segment=Nsegment,
#                                                  num_segmentdata=NobsSeg,
#                                                  num_layer=Nlayer,
#                                                  random_seed=1)
# dat_pca = PCA().fit_transform( sensor.T ) #dat['obs'] )


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

print(1)
flow_mod = Flow( prior, flows, device='cuda' )
print(2)
flow_mod.load_data( data=dat_pca )
print(3)
flow_mod.train( epochs = 100, verbose = True )
print(4)


if False:
	nclasses = Nsegment
	cflows = []
	segment_flow = AffineConstantFlow # NSF_CL
	for c in range(nclasses):
		flows_e = [segment_flow(dim=Ncomp) for _ in range(1)]
		convs_e = [Invertible1x1Conv(dim=Ncomp) for _ in flows_e]
		norms_e = [ActNorm(dim=Ncomp) for _ in flows_e]
		flows_e = list(itertools.chain(*zip(norms_e, convs_e, flows_e)))
		cflows.append( flows_e )

	flow_mod_cond = ClassCondFlow( prior, flows, cflows, device='cpu' )
	#flow_mod_cond.to(device)

	flow_mod_cond.load_data( data=dat_pca, labels= to_one_hot( label )[0] )

	loss_cond = flow_mod_cond.train( epochs = 600, verbose=True )

	z_full = flow_mod_cond.forwardPassFlow( dat_pca, fullPass=False, labels=to_one_hot(label)[0] )
	plt.figure()
	plt.scatter( z_full[:,0], z_full[:,1], c=label) #.argmax(axis=1)) # so actually the raw flow does a reasonable job by itself ...

	SolveHungarian( z_full, source )
	SolveHungarian( FastICA().fit_transform( z_full), source )




