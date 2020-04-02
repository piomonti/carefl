### file to run all simulations for conditional flow based models
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
#from models.tcl import * 
from helper.solveHungarian import SolveHungarian
from simulations.runTCLsimulationsFlow_01 import RunTCLsimulationFlow
from simulations.runTCLcausalSimulationsFlow_01 import bivariateTCLcausalSimulationFlow
from models.classConditionalFlow import Flow, ClassCondFlow

from sklearn.decomposition import FastICA, PCA
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

import pickle 

# define some simulation parameters
Ncomp     = 4
Nsegment  = 10
Nlayers   = [2 ,3 ,4 ,5]
NobsSeg   = 512 * 2 
Nsim      = 50
epochDict = {1: 100, 2:200, 3:300, 4:600, 5:800} 
device    = 'cpu'
#saveDir  = '/Users/ricardo/Documents/Projects/FlowNonLinearICA/simulations/results/'
#saveDir  = '/nfs/ghome/live/ricardom/FlowNonLinearICA/simulations/results/'

# for causality experiments !
saveDir  = '/home/projects/FlowNonLinearICA/simulations/resultsCausalLR/'


runTCL = False
# for TCL simulations
if runTCL:
	for l in Nlayers:
		for i in range( 1, Nsim+1 ):
			print('Running Sim: ' + str(i))
			RunTCLsimulationFlow( Ncomp=Ncomp, Nsegment=Nsegment, Nlayer=l, NobsSeg=NobsSeg, seed=i, epochs=epochDict[l], saveDir=saveDir, device=device, verbose=True if i==1 else False )


runCausal = True
if runCausal:
	for numSeg in [10,20]:
		for l in Nlayers:
			bivariateTCLcausalSimulationFlow( Nsegment=numSeg, Nlayer=l, NobsSeg=NobsSeg, Nsims=Nsim, epochs=100, saveDir=saveDir, device=device, verbose=False )






