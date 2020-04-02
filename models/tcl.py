### recover nonlinearly mixed sources via time contrastive learning (TCL)
#
# for further details see Hyvarinen & Morioka (2016)


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

os.chdir('/Users/ricardo/Documents/Projects/FlowNonLinearICA')
from data.generateToyData import CustomSyntheticDatasetDensity, CustomSyntheticDatasetDensityClasses

### define some NN modules (to experiment with later)

class ResnetModule( nn.Module ):
    """
    implement basic module for resnet MLP 

    note that this module keeps the dimensions fixed! will implement a mapping from a 
    vector of dimension input_size to another vector of dimension input_size

    """

    def __init__( self, input_size, activation_function = nn.functional.relu ):
        super( ResnetModule, self).__init__()
        self.activation_function = activation_function
        self.linear_layer = nn.Linear( input_size, input_size )
        self.bn_layer = nn.BatchNorm1d( input_size )

    def forward( self, x ):
        x = self.bn_layer( x )
        linear_act = self.linear_layer( x )
        H_x = self.activation_function( linear_act )
        return torch.add( linear_act , H_x )



class FeedForwardModule( nn.Module ):
    """
    implement basic module for MLP 

    note that this module keeps the dimensions fixed! will implement a mapping from a 
    vector of dimension input_size to another vector of dimension input_size

    """

    def __init__( self, input_size, activation_function = nn.functional.relu ):
        super( FeedForwardModule, self).__init__()
        self.activation_function = activation_function
        self.linear_layer = nn.Linear( input_size, input_size )
        self.bn_layer = nn.BatchNorm1d( input_size )

    def forward( self, x ):
        x = self.bn_layer( x )
        linear_act = self.linear_layer( x )
        H_x = self.activation_function( linear_act )
        return H_x



class TCLnetwork( nn.Module ):
    """
    define deep network architecture to recover nonlinear features via
    contrastive learning !
    """
    def __init__( self, input_size, hidden_size, output_size, n_classes, n_layers, networkModule, activation_function = F.relu ):
        """

        Input:
        - input_size          : dimension of input data (e.g., 784 for MNIST)
        - hidden_size         : size of hidden representations (may need overparameterization to help networks learn)
        - output_size         : size of output (will typically be the same as input size for TCL)
        - n_classes           : number of classes in the contrastive classification task
        - n_layers            : number of hidden layers
        - networkModule       : type of network module to use (either FeedForwardModule or a ResnetModule)
        - activation function : nonlinearity employed 

        """
        super( TCLnetwork, self ).__init__()

        self.activation_function = activation_function
        self.linear1st           = nn.Linear( input_size, hidden_size ) # map from data dim to dimension of hidden units
        self.Layers              = nn.ModuleList( [networkModule( hidden_size, activation_function=self.activation_function ) for _ in range(n_layers) ] )
        self.linearLast          = nn.Linear( hidden_size, output_size ) # map from dimension of hidden units to dimension of output
        self.projectSegments     = nn.Linear( output_size, n_classes)
        self.input_size          = input_size
        self.hidden_size         = hidden_size
        self.output_size         = output_size

    def forward( self, x ):
        """
        forward pass through network
        """
        x = self.linear1st( x )
        for current_layer in self.Layers :
            x = current_layer( x )
        x = torch.abs( self.linearLast( x ) ) # project onto dimension of latent sources
        x = F.softmax( self.projectSegments( x ) )
        #x = F.softmax( self.linearLast( x ) )
        return x

    def unmix( self, x ):
        """
        forward pass through to penulitmate layer (to do unmixing as in the theory of TCL)
        """
        x = self.linear1st( x )
        for current_layer in self.Layers :
            x = current_layer( x )
        return self.linearLast ( x )




class TCL( nn.Module ):
    """
    implementation of TCL as described in Hyvarinen & Morioka (2016)
    """

    def __init__( self, network ):
        super().__init__()
        self.tclnet       = network # should be feed forward network (eg a TCLnetwork object)
        self.train_loader = None 
        self.nObs         = None

    def __repr__( self ):
        print('Network with ' + str(len(self.tclnet.Layers)) + ' layers' )
        print('Dimensions\nInput: ' + str(self.tclnet.input_size) +'\nHidden: ' + str(self.tclnet.hidden_size))
        return ""

    def load_data( self, data, labels ):
        """
        load in data
        """
        dset = CustomSyntheticDatasetDensityClasses( data.astype( np.float32 ), labels.astype( np.float32 ) ) 
        self.train_loader = DataLoader( dset, shuffle=True, batch_size=128 )
        self.nObs = data.shape[0]

    def unmix( self, dat ):
        """
        """
        self.tclnet.eval() 
        dat = torch.tensor( dat.astype( np.float32 ) )
        return self.tclnet.unmix( dat ).detach().numpy() 

    def train( self, epochs = 100, verbose = False ):
        """
        train TCL using contrastive learning
        """

        # define parameters
        params = list( self.tclnet.parameters() )
        # define optimizer
        optimizer = optim.Adam( params , lr=1e-4, weight_decay=1e-5) 

        # begin training
        loss_vals = []
        acc_vals  = []
        self.tclnet.train()
        for e in range( epochs ):
            loss_val = 0
            correct = 0
            for _, (dat, target) in enumerate( self.train_loader ):
                # get predictions
                pred = self.tclnet( dat ) # get predictions for this batch
                loss = F.nll_loss( pred, target.argmax(axis=1) )
                loss_val += loss.item()

                # run optimization
                self.tclnet.zero_grad()
                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

                # also check percentage of correct classification
                correct += pred.argmax(axis=1).eq( target.argmax(axis=1) ).cpu().sum()  # pred.eq(target.data).cpu().sum()

            if verbose:
                print('epoch {}/{} \tloss: {} \tacc: {}'.format(e, epochs, loss_val, correct/float(self.nObs)))
            loss_vals.append( loss_val )
        return loss_vals, correct / float( self.nObs )















