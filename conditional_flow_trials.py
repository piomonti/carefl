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

# generate MoG data
#dat = gen2Dspiral( n = 5000, radius=3, sigma=1.5 )
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

dset = CustomSyntheticDatasetDensityClasses( dat_scale.astype(np.float32), lab.astype( np.int32 ) ) 
train_loader = DataLoader( dset, shuffle=True, batch_size=128 )

# define Flow model
prior = TransformedDistribution(Uniform(torch.zeros(2), torch.ones(2)), SigmoidTransform().inv) 

# MAF (with MADE net, so we get very fast density estimation)
#flows = [MAF(dim=2, parity=i%2) for i in range(4)]

nfs_flow = NSF_CL 
flows = [nfs_flow(dim=2, K=8, B=3, hidden_dim=16) for _ in range(3)]
convs = [Invertible1x1Conv(dim=2) for _ in flows]
norms = [ActNorm(dim=2) for _ in flows]
flows = list(itertools.chain(*zip(norms, convs, flows)))

# define shared flow component
model_share = NormalizingFlowModel( prior, flows )

# define the class conditional flows as well
nclasses = 3
cflows = []
segment_flow = AffineConstantFlow # NSF_CL
for c in range(nclasses):
	flows_e = [segment_flow(dim=2) for _ in range(1)]
	convs_e = [Invertible1x1Conv(dim=2) for _ in flows_e]
	norms_e = [ActNorm(dim=2) for _ in flows_e]
	flows_e = list(itertools.chain(*zip(norms_e, convs_e, flows_e)))
	cflows.append( flows_e )


# construct the model
model_segments = [ NormalizingFlowModel(prior, c) for c in cflows ]

# define optimizer
print("number of params (shared model): ", sum(p.numel() for p in model_share.parameters()))
print("number of params (segment model): ", sum(p.numel() for p in model_segments[0].parameters()))
params = list( model_share.parameters() ) 
for c in range(nclasses):
	params +=  list( model_segments[c].parameters() )

optimizer = optim.Adam( params , lr=1e-4, weight_decay=1e-5) # todo tune WD


# run optimization
epochs = 250
loss_vals = []

model_share.train()
for c in range(nclasses):
	model_segments[c].train()

# begin training 
for e in range( epochs ):
	loss_val = 0
	for _, dat in enumerate( train_loader ):
		dat, seg = dat 
		#loss = 0

		# forward pass - run through shared network first:
		z_share, prior_logprob, log_det_share = model_share( dat )

		# now pass through class flows, concatenate for each class then multiple by segment one hot encoding
		prior_logprob_final = torch.zeros( (dat.shape[0], nclasses) )
		for c in range( nclasses ):
			z_c, prior_logprob_c, log_det_c = model_segments[c]( z_share[-1] )
			prior_logprob_final[:,c] = prior_logprob_c + log_det_share + log_det_c

		# take only correct classes
		logprob = ( prior_logprob_final * seg ) 
		loss = - torch.sum( logprob )

		#print(loss.item())
		loss_val += loss.item()

		# 
		model_share.zero_grad()
		for c in range(nclasses):
			model_segments[c].zero_grad()

		optimizer.zero_grad()

		# compute gradients
		loss.backward()

		# update parameters
		optimizer.step()

	print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
	loss_vals.append( loss_val )


plt.plot( range(len(loss_vals)), loss_vals)

# invert samples through the shared network:
x_forward, _, _ = model_share( torch.tensor( dat_scale.astype(np.float32) ) )
x_forward = x_forward[-1].detach().numpy()

plt.scatter( x_forward[:,0], x_forward[:,1], c=lab.argmax(axis=1))

# pass forward through the entire architecture:
x_forward_full = np.zeros( dat_scale.shape )
for c in range( nclasses ):
	ii = np.where( lab[:,c] !=0 )[0]

	# pass through shared
	x_share, _, _ = model_share( torch.tensor( dat_scale[ii,:].astype( np.float32 ) ) )
	x_final, _, _ = model_segments[c]( x_share[-1] )

	x_forward_full[ii,:] = x_final[-1].detach().numpy()

plt.scatter( x_forward_full[:,0], x_forward_full[:,1], c=lab.argmax(axis=1))


# compute the density over a grid
xvals = np.arange(-2.25, 2.25, .15)
yvals = np.arange(-2.25, 2.25, .15)

X, Y = np.meshgrid(xvals, yvals)

model_share.eval() 
Z = np.zeros( X.shape )
for i in range( X.shape[0] ):
	for j in range( X.shape[1] ):
		input_t = torch.Tensor([ [X[i,j], Y[i,j]], [1,1] ]) # we ignore second row !
		zs, prior_logprob, log_det = model_share( input_t )

		if True:
			prior_logprob_final = torch.zeros( (2, nclasses) )
			for c in range( nclasses ):
				z_c, prior_logprob_c, log_det_c = model_segments[c]( zs[-1] )
				prior_logprob_final[:,c] = prior_logprob_c + log_det + log_det_c
			# we ignore second row and marginalize over first
			Z[i,j] += prior_logprob_final[0,:].mean().item()
		else:
			Z[i,j] += (  -1* (prior_logprob[0] + log_det[0]).item() )


plt.imshow( Z / Z.sum() )
plt.title('DEEN Paper Fig 1(a)')
