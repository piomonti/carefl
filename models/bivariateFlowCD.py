### Compute bivariate Flow based measures of causal direction
#
#
# code for flows is based on the following library:
# https://github.com/karpathy/pytorch-normalizing-flows
#

import itertools

import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import MultivariateNormal, Normal, Laplace, Uniform, TransformedDistribution, SigmoidTransform
from torch.nn.parameter import Parameter
import pandas as pd 

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader

# load flows
from nflib.flows import AffineConstantFlow, AffineFullFlow, MAF, NormalizingFlowModel, Invertible1x1Conv, ActNorm, AffineHalfFlow, AffineFullFlowGeneral
from nflib.nets import MLP1layer, MLP
from nflib.spline_flows import NSF_AR, NSF_CL
from models.classConditionalFlow import Flow 


class BivariateFlowCD():
    """
    class for bivariate causl
    """

    def __init__(self, Nlayers, Nhidden, priorDist='laplace', TrainSplit=1., epochs=100, optMethod='adam', verbose=False ):
        self.Nlayers    = Nlayers
        self.Nhidden    = Nhidden
        self.priorDist  = priorDist
        self.TrainSplit = TrainSplit
        self.epochs     = epochs 
        self.optMethod  = optMethod
        self.verbose    = verbose 
        self.results    = None 
        self.flow       = None

    def train( self, dat ):
        """
        train flow models to determine causal direction

        """
        # define final variables
        Ncomp        = 2
        label        = np.zeros( dat.shape[0] )
        nfs_flow     = AffineFullFlow 
        nfs_mlp      = MLP1layer

        # ensure its 2D 
        if dat.shape[1] > 2:
            dat = dat[:, :2]


        # split into training and testing data:
        if self.TrainSplit==1.:
            testDat    = np.copy( dat )
        else:
            testDat    = np.copy( dat[ int(self.TrainSplit * dat.shape[0]): ,:])
            dat        = dat[ :int(self.TrainSplit * dat.shape[0]) ,:]

        # now start running LR methods
        results = pd.DataFrame( {'L' : np.repeat( self.Nlayers, len(self.Nhidden)),
                                 'nh': self.Nhidden * len(self.Nlayers),
                                 'x->y': [0] * len(self.Nlayers) * len(self.Nhidden),
                                 'y->x': [0] * len(self.Nlayers) * len(self.Nhidden) })

        for l in self.Nlayers:
            for nh in self.Nhidden:
                # -------------------------------------------------------------------------------
                #         Conditional Flow Model: X->Y
                # -------------------------------------------------------------------------------
                torch.manual_seed( 0 )
                if self.priorDist=='laplace':
                    prior = Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )) #TransformedDistribution(Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)  
                else:
                    prior = TransformedDistribution( Uniform( torch.zeros( Ncomp ), torch.ones( Ncomp ) ), SigmoidTransform().inv ) # Logistic distribution


                flows = [nfs_flow(dim=Ncomp, nh=nh, parity=False, net_class=nfs_mlp) for _ in range(l)]

                flow_mod_cond = Flow( prior, flows, device='cpu' )
                flow_mod_cond.load_data( data=dat ) 

                # now we train this model and store the likelihood:
                loss_cond = flow_mod_cond.train( epochs = self.epochs, optMethod=self.optMethod, verbose=self.verbose )

                # -------------------------------------------------------------------------------
                #         Conditional Flow Model: Y->X
                # -------------------------------------------------------------------------------
                torch.manual_seed( 0 )
                if self.priorDist=='laplace':
                    prior_rev = Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )) # TransformedDistribution(Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)  # MultivariateNormal(loc=np.zeros((Ncomp,)), covariance_matrix = np.eye( Ncomp )).inv )  # SigmoidTransform().inv) 
                else:
                    prior_rev = TransformedDistribution( Uniform( torch.zeros( Ncomp ), torch.ones( Ncomp ) ), SigmoidTransform().inv ) # Logistic distribution


                flows_rev = [nfs_flow(dim=Ncomp, nh=nh, parity=False, net_class=nfs_mlp) for _ in range(l)]

                flow_mod_cond_rev = Flow( prior_rev, flows_rev , device='cpu' )
                flow_mod_cond_rev.load_data( data=dat[:,[1,0]] ) 

                # now we train this model and store the likelihood:
                loss_cond_rev = flow_mod_cond_rev.train( epochs = self.epochs, optMethod=self.optMethod, verbose=self.verbose )
                results.loc[ (results.L==l) & (results.nh==nh), 'x->y' ] = np.nanmean( flow_mod_cond.EvalLL( testDat ) )
                results.loc[ (results.L==l) & (results.nh==nh), 'y->x' ] = np.nanmean( flow_mod_cond_rev.EvalLL( testDat[:,[1,0]] ) )

        # print(results)
        # compute the consensus
        p = results['x->y'].max() - results['y->x'].max() # np.mean( results['x->y'] > results['y->x'] )
        predModel = 'x->y' if p >= 0 else 'y->x'

        return results, predModel

    def FitFlowSEM( self, dat, Nlayers, Nhidden ):
        """
        assuming causal ordering is provided, we fit the associated SEM

        INPUT:
            - dat: np array with 2 columns (for now limit to bivariate). Assume dat[:,0] is cause
            - Nlayers: number of layers for flow architecture
            - Nhidden: number of hidden units per layer in flow architecture
        """
        # define final variables
        Ncomp        = dat.shape[1]
        nfs_mlp      = MLP1layer

        # ensure its 2D 
        if dat.shape[1] > 2:
            print('using higher D implementation')
            nfs_flow     = AffineFullFlowGeneral            
            #dat = dat[:, :2]
        else:
            nfs_flow     = AffineFullFlow # AffineHalfFlow #  

        l  = Nlayers
        nh = Nhidden

        # -------------------------------------------------------------------------------
        #         Train Flow Model: dat[:,0]->dat[:,1]
        # -------------------------------------------------------------------------------
        torch.manual_seed( 0 )
        if self.priorDist=='laplace':
            prior = Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )) #TransformedDistribution(Laplace(torch.zeros( Ncomp ), torch.ones( Ncomp )), SigmoidTransform().inv)  
        else:
            prior = TransformedDistribution( Uniform( torch.zeros( Ncomp ), torch.ones( Ncomp ) ), SigmoidTransform().inv ) # Logistic distribution

        flows = [nfs_flow(dim=Ncomp, nh=nh, parity=False, net_class=nfs_mlp) for _ in range(l)]

        flow_mod_cond = Flow( prior, flows, device='cpu' )
        flow_mod_cond.load_data( data=dat ) 

        # now we train this model and store the likelihood:
        loss_cond = flow_mod_cond.train( epochs = self.epochs, optMethod=self.optMethod, verbose=self.verbose )

        self.flow = flow_mod_cond

        return None 

    def invertFlow( self, dat ):
        return self.flow.forwardPassFlow( dat )

    def predictIntervention( self, x0val, nSamples=100, dataDim=2, interventionIndex=0):
        """
        we predict the value of x1 given an intervention on x0 (the causal variable)

        this proceeds in 3 steps:
         - invert flow to find corresponding entry for z0 at x0=x0val (this is invariant to x1 as x0 is the cause)
         - sample z1 from prior (number of samples is nSamples)
         - pass [z0, z1 samples] through flow to get predictive samples for x1| do(x0=x0val)

        for now we only support 2D and 4D examples, will generalize in future!
        """

        if dataDim==2:
            # first we invert the flow:
            input_ = np.array([ x0val, 0]).reshape((1,2)) # value of x1 here is indifferent
            z0 = self.invertFlow( input_ )[0,0]

            # now generate samples of z1 from prior
            z1 = self.flow.flow_share.prior.sample( (nSamples, ) )[ :,1 ].cpu().detach().numpy()

            # now we pass forward
            latentSpace = np.vstack(( [z0]*nSamples, z1 )).T
            latentExp   = np.array( [z0, 0] ).reshape((1,2))

            # finally pass through the generative model to get a distribution over x1 | do(x0=x0val)
            x1Intervention = self.flow.backwardPassFlow( latentSpace )
            assert np.abs(x0val - x1Intervention[0,0]) < 1e-5
            return x1Intervention[:,1], self.flow.backwardPassFlow( latentExp )[0,1]
        elif dataDim==4:
            print('intervention for high (4) D case')
            # we are in the 4D case. Assume [x0,x1] are causes of [x2,x3]
            # the interventionIndex variable tells us which cause we intervene over (either 0th or 1st entry)

            # first we invert the flow:
            cause_input = np.zeros((1,2))
            cause_input[ 0, interventionIndex] = x0val

            input_    = np.hstack( (cause_input,  np.zeros((1,2)))) # value of other variables here is indifferent
            latentVal = self.invertFlow( input_ )[0, interventionIndex ]

            # prepare latentExpectation (do sampling later)
            latentExp = np.zeros((1,4))
            latentExp[0, interventionIndex] = latentVal

            # now we pass forward
            x1Intervention = self.flow.backwardPassFlow( latentExp )

            return x1Intervention

            # now generate samples of z1 from prior
            #z1 = self.flow.flow_share.prior.sample( (nSamples, ) )[ :,1 ].cpu().detach().numpy()

            # now we pass forward
            #latentSpace = np.vstack(( [z0]*nSamples, z1 )).T
            #latentExp   = np.array( [z0, 0] ).reshape((1,2))

            # finally pass through the generative model to get a distribution over x1 | do(x0=x0val)
            #x1Intervention = self.flow.backwardPassFlow( latentSpace )
            #assert np.abs(x0val - x1Intervention[0,0]) < 1e-5
            #return x1Intervention[:,1], self.flow.backwardPassFlow( latentExp )[0,1]

    def predictCounterfactual( self, xObs, xCFval, interventionIndex=0 ):
        """

        given observation xObs we estimate the counterfactual of setting
        xObs[ interventionIndex ] = xCFval 

        we follow the 3 steps for counterfactuals
         1) abduction - pass-forward through flow to infer latents for xObs
         2) action - pass-forward again for latent associated with xCFval
         3) prediction - backward pass through the flow
        """

        # abduction:
        latentObs = self.invertFlow( xObs )

        # action (get latent variable value under counterfactual)
        xObs_CF = np.copy( xObs )
        xObs_CF[0, interventionIndex] = xCFval 
        latentObs[ 0, interventionIndex ] = self.invertFlow( xObs_CF )[0, interventionIndex]

        # prediction (pass through the flow):
        return self.flow.backwardPassFlow( latentObs )

