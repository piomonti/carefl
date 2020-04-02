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
#from data.generateCausalTCLdata import * 
from data.generate_artificial_tcl_data import *
from data.generateToyData import to_one_hot
from helper.solveHungarian import SolveHungarian

from models.classConditionalFlow import Flow, ClassCondFlow
#from models.tcl import * 

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import scale 
import pandas as pd 
#import pdb 
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#pdb.set_trace()

# define simulation parameters 
Ncomp    = 2
Nsegment = 10
Nlayer   = 5
NobsSeg  = 512 * 2
method   = 'tcl'
#device   = 'cuda'

Nsims = 20

LR_xy      = np.zeros( Nsims )
LR_yx      = np.zeros( Nsims )
models     = ['NA'] * Nsims
predmodels = ['NA'] * Nsims


for i in range( Nsims ):

	# generate some TCL data:

	if method=='tcl':
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
		useConvNorm = False
	else:
		np.random.seed( i ) 
		dat_pca     = ( gen_toy_causal_dat( nObs = NobsSeg, causalFunc = method) ) 
		label       = np.zeros( dat_pca.shape[0] )

		if np.random.uniform() < .5:
			# dont flip
			model = 'x->y'
		else:
			# flip
			model = 'y->x'
			dat_pca = dat_pca[:, [1,0]] 

		models[i]   = model # 'x->y'
		useConvNorm = False 

		# note that we do not scale as this would be equivalent to a transformation on each X and Y which we do ourselves in the flow

	# plt.figure()
	# plt.scatter( dat_pca[:,0], dat_pca[:,1], c=dat['labels'])

	# plt.figure()
	# plt.scatter( dat['source'][:,0], dat['source'][:,1], c=dat['labels'])


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
	if useConvNorm:
		convs = [Invertible1x1Conv(dim=Ncomp) for _ in flows]
		norms = [ActNorm(dim=Ncomp) for _ in flows]
		flows = list(itertools.chain(*zip(norms, convs, flows)))

	# note: if we remove the convs and norms above, the one input will remain unchanged throughout due to parity=False

	nclasses = Nsegment
	cflows = []
	segment_flow = AffineConstantFlow # NSF_CL
	for c in range(nclasses):
		flows_e = [segment_flow(dim=Ncomp) for _ in range(1)]
		if useConvNorm:
			convs_e = [Invertible1x1Conv(dim=Ncomp) for _ in flows_e]
			norms_e = [ActNorm(dim=Ncomp) for _ in flows_e]
			flows_e = list(itertools.chain(*zip(norms_e, convs_e, flows_e)))
		cflows.append( flows_e )

	flow_mod_cond = ClassCondFlow( prior, flows, cflows, device='cpu' )

	flow_mod_cond.load_data( data=dat_pca, labels= to_one_hot( label )[0] )

	# now we train this model and store the likelihood:
	loss_cond = flow_mod_cond.train( epochs = 100, verbose=False )

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
	if useConvNorm:
		convs_rev = [Invertible1x1Conv(dim=Ncomp) for _ in flows]
		norms_rev = [ActNorm(dim=Ncomp) for _ in flows]
		flows_rev = list(itertools.chain(*zip(norms_rev, convs_rev, flows_rev)))

	# note: if we remove the convs and norms above, the one input will remain unchanged throughout due to parity=False

	nclasses = Nsegment
	cflows_rev = []
	segment_flow_rev = AffineConstantFlow # NSF_CL
	for c in range(nclasses):
		flows_e_rev = [segment_flow(dim=Ncomp) for _ in range(1)]
		if useConvNorm:
			convs_e_rev = [Invertible1x1Conv(dim=Ncomp) for _ in flows_e_rev]
			norms_e_rev = [ActNorm(dim=Ncomp) for _ in flows_e_rev]
			flows_e_rev = list(itertools.chain(*zip(norms_e_rev, convs_e_rev, flows_e_rev)))
		cflows_rev.append( flows_e_rev )

	flow_mod_cond_rev = ClassCondFlow( prior_rev, flows_rev, cflows_rev , device='cpu' )

	flow_mod_cond_rev.load_data( data=dat_pca[:,[1,0]], labels= to_one_hot( label )[0] )

	# now we train this model and store the likelihood:
	loss_cond_rev = flow_mod_cond_rev.train( epochs = 100, verbose=False )

	LR_yx[i] = np.nanmean( flow_mod_cond_rev.EvalLL( dat_pca[:,[1,0]], to_one_hot(label)[0] ) )

	predmodels[i] =  'x->y' if LR_xy[i] > LR_yx[i] else 'y->x' 

	a = pd.DataFrame({'true':models, 'LR_xy':LR_xy, 'LR_yx': LR_yx, 'pred':predmodels})
	print(a)
	print( np.mean( a['true'][:i]==a['pred'][:i]))
		

