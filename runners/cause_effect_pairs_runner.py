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
from nflib.flows import AffineConstantFlow, MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm
from nflib.spline_flows import NSF_AR, NSF_CL
from CauseEffectPairs.Run_CEP_01 import * 
from models.classConditionalFlow import Flow 
from models.bivariateFlowCD import BivariateFlowCD
from sklearn.decomposition import FastICA, PCA
import pickle 

# define some simulation parameters
skipPairs = [ 52, 54, 55 ] # skip these pairs, as indicated by Mooij et al (2016) because the variables are not bivariate (i.e., X and Y are not univariate)

saveDir  = '/CauseEffectPairs/results/'
SplitPerc = .8 
scaleDat  = True
priorDist = 'laplace' 
epochs    = 500 + 250
optMethod = 'schedule'

correctCount = 0 
runningCount = 0

BinaryCutoff = .15
correctCount_nobinary = 0
runningCount_nobinary = 0

LayerList = [1,3] 
depthList = [5] 

PairDataDir = 'CauseEffectPairs/pairs/'

for i in range(1, 108):
	if i in skipPairs:
		pass
	else:
		res, predModel, trueModel, ctsRatio = runCEPair( i, LayerList, depthList, priorDist=priorDist, TrainSplit=SplitPerc, epochs=epochs, optMethod=optMethod, scaleDat=scaleDat, verbose=True )

		runningCount += 1
		if ctsRatio > BinaryCutoff:
			runningCount_nobinary += 1
		if predModel.replace('-', '') == trueModel.replace('-', ''):
			print('Correct!')
			correctCount += 1 
			if ctsRatio > BinaryCutoff:
				correctCount_nobinary += 1

		# save results
		fileEnd = '.csv'
		if scaleDat:
			fileEnd = '_Scaled' + fileEnd
		else:
			fileEnd = '_Unscaled' + fileEnd

		# add prior dist
		fileEnd = '_' + priorDist + fileEnd
	
		print( os.getcwd() )
		#res.to_csv( saveDir + 'pair' +str(i) + '_split' + str(int(SplitPerc*100)) + '_Flow' + fileEnd)
		# print progress
		print('running mean: ' + str(float(correctCount) / runningCount) )

