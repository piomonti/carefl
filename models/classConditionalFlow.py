### class conditional flow networks for nonlinearICA 
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
os.chdir('..')
#os.chdir('/nfs/ghome/live/ricardom/FlowNonLinearICA/')
#os.chdir('/Users/ricardo/Documents/Projects/FlowNonLinearICA')
from nflib.flows import AffineConstantFlow, MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm, ClassCondNormalizingFlowModel
from nflib.spline_flows import NSF_AR, NSF_CL
from data.generateToyData import CustomSyntheticDatasetDensity, CustomSyntheticDatasetDensityClasses

class Flow( nn.Module ):
    """
    a class for a standard normalizing flow
    """

    def __init__( self, prior, flows, device='cpu'):
        super().__init__()
        self.device        = device
        self.prior         = prior
        self.flow_share    = NormalizingFlowModel( prior, flows )
        self.train_loader  = None

    def __repr__( self ):
        print("number of params (shared model): ", sum(p.numel() for p in self.flow_share.parameters()))
        return ""

    def load_data( self, data ):
        """
        load in data
        """
        dset = CustomSyntheticDatasetDensity( data.astype( np.float32 ), device=self.device ) 
        self.train_loader = DataLoader( dset, shuffle=True, batch_size=128 )
	

    def train( self, epochs = 100, verbose=False ):
        """
        train the model
        """

        # define parameters
        params = list( self.flow_share.parameters() ) 
        # define optimizer 
        optimizer = optim.Adam( params , lr=1e-4, weight_decay=1e-5) # todo tune WD

        if self.device!='cpu':
            self.flow_share.to( self.device )
            self.flow_share.cuda()

        # begin training
        loss_vals = []
        self.flow_share.train()
        for e in range( epochs ):
            loss_val = 0
            for _, dat in enumerate( self.train_loader ):
                dat.to( self.device )
                dat.cuda()
                # forward pass:
                z_share, prior_logprob, log_det_share = self.flow_share( dat )
                logprob = prior_logprob + log_det_share

                # take only correct classes
                loss = - torch.sum( logprob )
                loss_val += loss.item()

                # 
                self.flow_share.zero_grad()

                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

            if verbose:
                print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
            loss_vals.append( loss_val )
        return loss_vals

    def forwardPassFlow( self, x ):
        """
        pass samples through the flow
        """
        x_forward, _, _ = self.flow_share( torch.tensor( x.astype( np.float32 ) ) )
        return x_forward[-1].cpu().detach().numpy()



class ClassCondFlow( nn.Module ):
    """

    A normalizing flow that also takes classes as inputs 

    This is a special architecture in an attempt to solve nonlinear ICA 
    via maximum likelihood.

    As such, we assume data is generated as a smooth invertible mixture, f, of 
    latent variables s. Further, we assume latent variables follow a piecewise 
    stationary distribution (see Hyvarinen & Morioka, 2016 or Khemakhem et al 2020 for details)

    The flow will be composed of two parts:
     - the first, will seek to invert the nonlinear mixing (ie to 
       compute g = f^{-1})
     - the second to estimate the exponential family parameters associated 
       with each segment (the Lambdas in the above papers)

    This essentially means each segment will have a distinct prior distribution (I think!)


    """

    def __init__( self, prior, flows, classflows, device='cpu'):
        super().__init__()
        print('initializing with: ' + str(device))
        self.device        = device
        self.prior         = prior
        self.flow_share    = NormalizingFlowModel( prior, flows ).to(device)
        self.flow_segments = [ NormalizingFlowModel( prior, nf ).to(device) for nf in classflows ] # classflows should be a list of flows, one per class
        self.nclasses      = len( classflows )

    def __repr__( self ):
        print("number of params (shared model): ", sum(p.numel() for p in self.flow_share.parameters()))
        print("number of params (segment model): ", sum(p.numel() for p in self.flow_segments[0].parameters()))
        return ""

    def load_data( self, data, labels ):
        """
        load in data
        """
        dset = CustomSyntheticDatasetDensityClasses( data.astype( np.float32 ), labels.astype( np.int32 ), device=self.device ) 
        self.train_loader = DataLoader( dset, shuffle=True, batch_size=128 )


    def train( self, epochs=100, verbose=False ):
        """
        train networks 
        """

        # define parameters
        #print(self.flow_share.parameters())
        params = list( self.flow_share.parameters() ) 
        for c in range( self.nclasses ):
            params +=  list( self.flow_segments[c].parameters() )

        # define optimizer 
        optimizer = optim.Adam( params , lr=1e-4, weight_decay=1e-5) # todo tune WD

        if self.device!='cpu':
            #print('here')
            self.flow_share.to( self.device )
            for c in range( self.nclasses ):
                self.flow_segments[c].to( self.device )
            #print('here.')

        # begin training
        loss_vals = []

        self.flow_share.train()
        for c in range( self.nclasses ):
            self.flow_segments[c].train()

        for e in range( epochs ):
            loss_val = 0
            for _, dat in enumerate( self.train_loader ):
                dat, seg = dat
                dat.to(self.device)
                seg.to(self.device) 

                # forward pass - run through shared network first:
                z_share, prior_logprob, log_det_share = self.flow_share( dat )

                # now pass through class flows, concatenate for each class then multiple by segment one hot encoding
                prior_logprob_final = torch.zeros( (dat.shape[0], self.nclasses) )
                for c in range( self.nclasses ):
                    z_c, prior_logprob_c, log_det_c = self.flow_segments[c]( z_share[-1] )
                    prior_logprob_final[:,c] = prior_logprob_c + log_det_share + log_det_c

                # take only correct classes
                logprob = ( prior_logprob_final * seg ) 
                loss = - torch.sum( logprob )
                loss_val += loss.item()

                # 
                self.flow_share.zero_grad()
                for c in range(self.nclasses):
                    self.flow_segments[c].zero_grad()

                optimizer.zero_grad()

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

            if verbose:
                print('epoch {}/{} \tloss: {}'.format(e, epochs, loss_val))
            loss_vals.append( loss_val )
        return loss_vals


    def EvalLL( self, x, lab ):
        """
        lab should be one hot encoded.
        """
        log_probs = np.zeros( x.shape[0] )

        # pass through shared flow
        x_forward, prior_logprob, log_det_share = self.flow_share( torch.tensor( x.astype( np.float32 ) ) )

        # pass through conditional flows
        prior_logprob_final = torch.zeros( (x.shape[0], self.nclasses) )
        for c in range( self.nclasses ):
            z_c, prior_logprob_c, log_det_c = self.flow_segments[c]( x_forward[-1] )
            prior_logprob_final[:,c] = prior_logprob_c + log_det_share + log_det_c

        # take only correct classes
        logprob = ( prior_logprob_final * torch.tensor( lab.astype(np.float32)  ) ).sum(axis=1)
        return logprob.cpu().detach().numpy()


    def forwardPassFlow( self, x, fullPass=False, labels=None ):
        """
        pass samples through the flow
    
        fullPass: we pass through the segment flows as well, otherwise just pass through the shared flow

        """
        if fullPass:
            x_forward = np.zeros( x.shape )
            for c in range( self.nclasses ):
                ii = np.where( labels[:,c] !=0 )[0]

                # pass through shared
                x_share, _, _ = self.flow_share( torch.tensor( x[ii,:].astype( np.float32 ) ) )
                x_final, _, _ = self.flow_segments[c]( x_share[-1] )
                x_forward[ii,:] = x_final[-1].cpu().detach().numpy()

            return x_forward

        else:
            x_forward, _, _ = self.flow_share( torch.tensor( x.astype( np.float32 ) ) )
            return x_forward[-1].cpu().detach().numpy()














