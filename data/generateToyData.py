## Code to generate synthetic data
#
#

import numpy as np 

from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

from sklearn import preprocessing
from sklearn.metrics import pairwise_distances

import numpy.random as npr 

# define dataset class

class CustomSyntheticDatasetDensity(Dataset):
	def __init__(self, X, device='cpu'):
		self.device = device
		self.x = torch.from_numpy(X).to(device)
		self.len = self.x.shape[0]
		self.data_dim = self.x.shape[1]
		print('data loaded on {}'.format(self.x.device))

	def get_dims(self):
		return self.data_dim

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.x[index]

	def get_metadata(self):
		return {
			'n': self.len,
			'data_dim': self.data_dim,
			}

class CustomSyntheticDatasetDensityClasses(Dataset):
	def __init__(self, X, E, device='cpu'):
		self.device = device
		self.x = torch.from_numpy(X).to(device)
		self.e = torch.from_numpy(E).to(device)
		self.len = self.x.shape[0]
		self.data_dim = self.x.shape[1]
		print('data loaded on {}'.format(self.x.device))

	def get_dims(self):
		return self.data_dim

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.x[index], self.e[index]

	def get_metadata(self):
		return {
			'n': self.len,
			'data_dim': self.data_dim,
			}    


def to_one_hot(x, m=None):
	"batch one hot"
	if type(x) is not list:
		x = [x]
	if m is None:
		ml = []
		for xi in x:
			ml += [xi.max() + 1]
		m = max(ml)
	dtp = x[0].dtype
	xoh = []
	for i, xi in enumerate(x):
		xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
		xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
	return xoh


# define ways of generating data
def gen2DgaussMix4comp( n, scale ):
	""" 
	generate n samples from a 2D Gaussian mixture with 4 components 
	"""

	Means = np.array([[0, 1.5],[-1.5,0],[0, -1.5],[1.5,0]])
	# Var   = np.eye(2) 

	samples = np.random.normal( loc = 0, scale = scale, size = (n,2) ) # generate N(0,I) and then shift the means 

	for i in range(n):
		ii = np.random.randint(4)
		samples[i,] += Means[ii,]

	return samples 




# generate spiral data
def gen2Dspiral( n, radius=2, sigma=1):
	""""""


	theta = np.sqrt(np.random.rand(n))*radius*np.pi 

	r_a = 2*theta + np.pi
	data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
	x_a = data_a + np.random.randn(n,2)*sigma

	# apply some scaling
	x_a = preprocessing.scale( x_a )

	return x_a

def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
	rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)

	features = npr.randn(num_classes*num_per_class, 2) * np.array([radial_std, tangential_std])
	features[:,0] += 1.
	labels = np.repeat(np.arange(num_classes), num_per_class)

	angles = rads[labels] + rate * np.exp(features[:,0])
	rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
	rotations = np.reshape(rotations.T, (-1, 2, 2))

	return 10*npr.permutation(np.einsum('ti,tij->tj', features, rotations))



