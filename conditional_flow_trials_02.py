### Initial trial at MLE training of flows on toy data #
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
from data.generateToyData import CustomSyntheticDatasetDensity, CustomSyntheticDatasetDensityClasses, gen2DgaussMix4comp, gen2Dspiral, make_pinwheel_data, to_one_hot
from models.classConditionalFlow import Flow, ClassCondFlow

# generate MoG data
np.random.seed(1)
dat = make_pinwheel_data(radial_std=.3, tangential_std=.05, num_classes=3, num_per_class=500, rate=.25)

# run simple clustering to get classes 
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import scale 

dat_scale = scale(dat)

S = KMeans( n_clusters=3 )
lab = S.fit_predict( dat_scale )
lab = to_one_hot( lab )[0]

plt.scatter( dat_scale[:,0], dat_scale[:,1] , c=lab.argmax(axis=1) )


##### CONDITIONAL FLOWS
# define networks for non conditional flow
prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) 

nfs_flow = NSF_CL 
flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(3)]
convs = [Invertible1x1Conv(dim=2) for _ in flows]
norms = [ActNorm(dim=2) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

nclasses = 3
cflows = []
segment_flow = AffineConstantFlow # NSF_CL
for c in range(nclasses):
	flows_e = [segment_flow(dim=2) for _ in range(1)]
	convs_e = [Invertible1x1Conv(dim=2) for _ in flows_e]
	norms_e = [ActNorm(dim=2) for _ in flows_e]
	flows_e = list(itertools.chain(*zip(norms_e, convs_e, flows_e)))
	cflows.append( flows_e )


flow_mod_cond = ClassCondFlow( prior, flows, cflows  )

flow_mod_cond.load_data( data=dat_scale, labels=lab )

# train the model
loss_cond = flow_mod_cond.train( epochs = 250, verbose=True )

# plot the latent representations
z = flow_mod_cond.forwardPassFlow( dat_scale )# [-1].detach().numpy()
plt.figure()
plt.scatter( z[:,0], z[:,1], c=lab.argmax(axis=1)) # so actually the raw flow does a reasonable job by itself ...

z_full = flow_mod_cond.forwardPassFlow( dat_scale, fullPass=True, labels=lab )
plt.figure()
plt.scatter( z_full[:,0], z_full[:,1], c=lab.argmax(axis=1)) # so actually the raw flow does a reasonable job by itself ...


##### NON CONDITIONAL FLOWS
# define networks for non conditional flow
prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) 

nfs_flow = NSF_CL 
flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(4)]
convs = [Invertible1x1Conv(dim=2) for _ in flows]
norms = [ActNorm(dim=2) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

flow_mod = Flow( prior, flows )

# load training data
flow_mod.load_data( data=dat_scale )

# train the model
loss = flow_mod.train( epochs = 250, verbose=True )

# plot the latent representations
z = flow_mod.forwardPassFlow( dat_scale )# [-1].detach().numpy()
plt.figure()
plt.scatter( z[:,0], z[:,1], c=lab.argmax(axis=1)) # so actually the raw flow does a reasonable job by itself ...


# compare losses over training
plt.figure()
plt.plot( loss_cond, label='Conditional Flow')
plt.plot( loss, label='Standard Flow')
plt.legend()



