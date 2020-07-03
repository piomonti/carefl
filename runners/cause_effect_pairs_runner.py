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

saveDir  = '/Users/ricardo/Documents/Projects/FlowCausalDirection/CauseEffectPairs/results'
SplitPerc = .8 # .9
scaleDat  = True
removeOultiers = False
priorDist = 'laplace' #'laplace'
epochs    = 500 + 250
optMethod = 'schedule'

correctCount = 0 
runningCount = 0

BinaryCutoff = .15
correctCount_nobinary = 0
runningCount_nobinary = 0

LayerList = [1,3] # [1,2,3]
depthList = [5] # [2,4,5,8,10]

PairDataDir = '/Users/ricardo/Documents/Projects/FlowCausalDirection/CauseEffectPairs/pairs/'

runMethod =  'notears' # 'LRHyv' #

if runMethod == 'flowCD':
	for i in range(1, 108):
		if i in skipPairs:
			pass
		else:
			res, predModel, trueModel, ctsRatio = runCEPair( i, LayerList, depthList, priorDist=priorDist, TrainSplit=SplitPerc, epochs=epochs, optMethod=optMethod, scaleDat=scaleDat, verbose=True )

			# check we get the same results with the class
			flowCD = BivariateFlowCD( Nlayers=LayerList, Nhidden=depthList, priorDist=priorDist, TrainSplit=SplitPerc, epochs=epochs, optMethod=optMethod, verbose=False)

			pair_id = str( i )
			pair_id = '0' * (4-len(pair_id)) + pair_id

			# load in the data 
			os.chdir( PairDataDir )
			dat_id = np.loadtxt('pair' + str(pair_id) + '.txt') 
			dir_id = open('pair' + str(pair_id) + '_des.txt', 'r').read().lower()#.split('ground truth:')[1].strip() #split('\n')[1]

			dir_id = dir_id.replace('\n', '')
			dir_id = dir_id.replace(':' , '')
			dir_id = dir_id.replace(' ', '')

			if ('x-->y' in dir_id) | ('x->y' in dir_id):
				dir_id = 'x-->y'
			elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
				dir_id = 'y-->x'

			flowCD.train( scale( dat_id ) )

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

			if removeOultiers:
				fileEnd = '_outlierRemoval' + fileEnd

			# add prior dist
			fileEnd = '_' + priorDist + fileEnd
		
			res.to_csv( saveDir + 'pair' +str(i) + '_split' + str(int(SplitPerc*100)) + '_PureFlow' + fileEnd)
			# print progress
			print('running mean: ' + str(float(correctCount) / runningCount) )
			#print('running mean (binary cutoff): ' + str(float(correctCount_nobinary) / runningCount_nobinary) )
elif runMethod == 'LRHyv' :
	print('running Hyvarinen & Smith LR method (for linear models)')
	from models.baselines import base_entropy_ratio, cumulant_hyv13_ratio

	for i in range(1, 108):
		if i in skipPairs:
			pass
		else:
			pair_id = str( i )
			pair_id = '0' * (4-len(pair_id)) + pair_id

			# load in the data 
			os.chdir( PairDataDir )
			dat_id = np.loadtxt('pair' + str(pair_id) + '.txt')
			dat_id = scale( dat_id ) 
			dir_id = open('pair' + str(pair_id) + '_des.txt', 'r').read().lower()#.split('ground truth:')[1].strip() #split('\n')[1]

			dir_id = dir_id.replace('\n', '')
			dir_id = dir_id.replace(':' , '')
			dir_id = dir_id.replace(' ', '')

			if ('x-->y' in dir_id) | ('x->y' in dir_id):
				dir_id = 'x-->y'
			elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
				dir_id = 'y-->x'

			trueModel = dir_id

			# predict cause & effect
			#predModel = base_entropy_ratio( x=dat_id[:,0], y=dat_id[:,1] )[1]
			predModel = cumulant_hyv13_ratio( x=dat_id[:,0], y=dat_id[:,1] )[1]

			runningCount += 1
			if predModel.replace('-', '') == trueModel.replace('-', ''):
				print('Correct!')
				correctCount += 1 

			# save results
			fileEnd = '.csv'
			if scaleDat:
				fileEnd = '_Scaled' + fileEnd
			else:
				fileEnd = '_Unscaled' + fileEnd

			if removeOultiers:
				fileEnd = '_outlierRemoval' + fileEnd

			# add prior dist
			fileEnd = '_' + priorDist + fileEnd
		
			#res.to_csv( saveDir + 'HyvarinenSmithLR_pair' +str(i) +  fileEnd)
			# print progress
			print('running mean: ' + str(float(correctCount) / runningCount) )
			#print('running mean (binary cutoff): ' + str(float(correctCount_nobinary) / runningCount_nobinary) )
elif runMethod == 'notears':
	print('Running linear NOTEARS algorithm')
	from models.baselines import linear_notears_dir
	for i in range(1, 108):
		if i in skipPairs:
			pass
		else:
			pair_id = str( i )
			pair_id = '0' * (4-len(pair_id)) + pair_id

			# load in the data 
			os.chdir( PairDataDir )
			dat_id = np.loadtxt('pair' + str(pair_id) + '.txt')
			dat_id = scale( dat_id ) 
			dir_id = open('pair' + str(pair_id) + '_des.txt', 'r').read().lower()#.split('ground truth:')[1].strip() #split('\n')[1]

			dir_id = dir_id.replace('\n', '')
			dir_id = dir_id.replace(':' , '')
			dir_id = dir_id.replace(' ', '')

			if ('x-->y' in dir_id) | ('x->y' in dir_id):
				dir_id = 'x-->y'
			elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
				dir_id = 'y-->x'

			trueModel = dir_id

			# predict cause & effect
			predModel = linear_notears_dir( x=dat_id[:,0], y=dat_id[:,1], lambda1=.001, loss_type='l2', w_threshold=0.0  )[1]

			runningCount += 1
			if predModel.replace('-', '') == trueModel.replace('-', ''):
				print('Correct!')
				correctCount += 1 

			# save results
			fileEnd = '.csv'
			if scaleDat:
				fileEnd = '_Scaled' + fileEnd
			else:
				fileEnd = '_Unscaled' + fileEnd

			if removeOultiers:
				fileEnd = '_outlierRemoval' + fileEnd

			# add prior dist
			fileEnd = '_' + priorDist + fileEnd
		
			#res.to_csv( saveDir + 'HyvarinenSmithLR_pair' +str(i) +  fileEnd)
			# print progress
			print('running mean: ' + str(float(correctCount) / runningCount) )

