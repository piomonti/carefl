### trial interventional predictions of flow models
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

# load data generating code:
from data.generate_synth_data import gen_synth_causal_dat
from sklearn.preprocessing import scale 
import pylab as plt; plt.ion()
import seaborn as sns 

# generate toy data:
np.random.seed(0)
dat, mod_dir = gen_synth_causal_dat( nObs=2500, causalFunc='hoyer2009' )
if mod_dir == 'y->x':
	dat = dat[:, [1,0]]
vars_ = dat.std(axis=0)
dat[:,0] /= vars_[0]
dat[:,1] /= vars_[1]

trueMod = lambda x : (x+ (.5)*x*x*x)/vars_[1]

# define and fit flow model:
mod = BivariateFlowCD(Nlayers=[3], Nhidden=[2], priorDist='laplace', epochs=500, optMethod='scheduling' )

mod.FitFlowSEM( dat=dat, Nlayers=5, Nhidden=10 )

# plot data distribution:
plt.scatter( dat[:,0], dat[:,1])
plt.xlabel('cause')
plt.ylabel('effect')

xvals = np.arange(-5,5,.1)
plt.plot( xvals, trueMod( xvals ), color='red', linewidth=3, linestyle=':')
plt.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=500)[1] for x in xvals ]), color='green', linewidth=3, linestyle='-.')

x0val = 1.5
predDist = mod.predictIntervention(x0val=x0val, nSamples=50000)

plt.hist( predDist )

input_ = np.ones((5,2 ))
print( mod.flow.backwardPassFlow( mod.invertFlow( input_ ) ) - input_ )
# input_ = np.random.random((5,2))
# print( np.abs(mod.flow.backwardPassFlow( mod.invertFlow( input_ ) ) - input_ ).max() )
# print(mod.invertFlow( input_ ))
# input_[:,1] = np.random.normal(0,1,5)
# print(mod.invertFlow( input_ ))


#### higher D examples
#
#
# generate some 4D data according to the following SEM
#
# X_0 = N_0
# X_1 = N_1
# X_2 = X_0 + X_1^3 + N_2
# X_3 = X_0^2 - X_1 + N_3 
#

nObs = 2500
np.random.seed(0)

# causes
X_0 = np.random.laplace( 0,1/np.sqrt(2), size=nObs)
X_1 = np.random.laplace( 0,1/np.sqrt(2), size=nObs)

# effects
X_2 =  X_0 + 0.5 * (X_1*X_1*X_1) + np.random.laplace( 0,1/np.sqrt(2), size=nObs)
X_3 = -X_1 + 0.5 * (X_0*X_0) + np.random.laplace( 0,1/np.sqrt(2), size=nObs)

#X_2 /= X_2.std()
#X_3 /= X_3.std()

dat = np.vstack( ( X_0, X_1, X_2, X_3) ).T

# example plots:
plt.scatter( dat[:,1], dat[:,2] )
plt.scatter( dat[:,0], dat[:,3] )

mod = BivariateFlowCD(Nlayers=[3], Nhidden=[2], priorDist='laplace', epochs=500, optMethod='scheduling' )
mod.FitFlowSEM( dat=dat, Nlayers=5, Nhidden=10 )

sns.set_style("whitegrid")
sns.set_palette( sns.color_palette("muted", 8))# sns.diverging_palette(255, 133, l=60, n=7, center="dark") )#  sns.color_palette("coolwarm", 6) )

# so the distribution of X_2 | do(X_0=x) should be linear
xvals = np.arange(-3,3,.05)

plt.plot( xvals, xvals, label='True', color=sns.color_palette("muted", 8)[2], linewidth=2, alpha=.8 )
plt.plot( xvals, .5*xvals*xvals,  color=sns.color_palette("muted", 8)[2], linewidth=2, alpha=.8 )

plt.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4 )[0,2] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_3| do(X_1=\alpha)]$') # remove 0 indexing in legend
# and distribution of X_3 | do(X_0=x) should be quadratic
plt.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4 )[0,3] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_4| do(X_1=\alpha)]$') # remove 0 indexing in legend



plt.legend( fontsize=12)
plt.xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=12)
plt.ylabel(r'Predicted value of $X_3$ or $X_4$', fontsize=12)
plt.title(r'interventional predictions under $do(X_1=\alpha)$', fontsize=15)





#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,9))
fig.suptitle(r'Interventional predictions under $do(X_1=\alpha)$', fontsize=18)

ax1.plot( xvals, xvals, label=r'True $\mathbb{E} [X_3| do(X_1=\alpha)]$', linewidth=3, linestyle=':' )
ax1.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4 )[0,2] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_3| do(X_1=\alpha)]$',
	alpha=.8) # remove 0 indexing in legend

ax1.set_xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=18)
ax1.set_ylabel(r'Predicted value of $X_3$', fontsize=18)
ax1.legend(loc=1, fontsize=16)



ax2.plot( xvals, .5*xvals*xvals, label=r'True $\mathbb{E} [X_4| do(X_1=\alpha)]$', linewidth=3, linestyle=':' )
ax2.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4 )[0,3] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'Predicted $\mathbb{E} [X_4| do(X_1=\alpha)]$',
	alpha=.8) # remove 0 indexing in legend

ax2.set_xlabel(r'Value of interventional variable, $X_1=\alpha$', fontsize=18)
ax2.set_ylabel(r'Predicted value of $X_4$', fontsize=18)
ax2.legend(loc=1, fontsize=16)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
















# and the distribution of X_2 | do(X_1=x) should be quadratic
plt.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4, interventionIndex=1 )[0,2] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'$X_2| do(X_1=x)$')
# and distribution of X_3 | do(X_0=x) should be quadratic
plt.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4, interventionIndex=1 )[0,3] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'$X_3| do(X_1=x)$')
plt.legend()


plt.scatter( dat[:,0], dat[:,3])
plt.plot( xvals, np.array([ mod.predictIntervention(x0val=x, nSamples=10, dataDim=4 )[0,3] for x in xvals ]), 
	linewidth=3, linestyle='-.', label=r'$X_3| do(X_0=x)$', color='red')


# below are checks to ensure flow is autoregressive
# input_ = np.ones((1,4))
# print(mod.invertFlow( input_ ))
# input_[0,0] = 2
# print(mod.invertFlow( input_ ))
# input_[0,1] = 2
# print(mod.invertFlow( input_ ))
# input_[0,2] = 2
# print(mod.invertFlow( input_ ))
# input_[0,3] = 2
# print(mod.invertFlow( input_ ))



