### Generate data based on the non-stationary, non-linear ICA model of Hyvarinen & Morioka (2016)
#
# NOTE: we deliberately for the model to have causal structure !
#
#
#

import numpy as np
import pylab as plt
from sklearn.preprocessing import scale

def leaky_ReLU_1d( d, negSlope ):
  """
  
  one dimensional implementation of leaky ReLU
  
  """
  
  if d > 0:
    return d
  else:
    return d * negSlope
  
leaky1d = np.vectorize( leaky_ReLU_1d )

def leaky_ReLU( D, negSlope ):
  """
  
  implementation of leaky ReLU activation function
  
  
  """
  
  assert negSlope > 0 # must be positive 
  return leaky1d( D, negSlope)


def sigmoidAct( x ):
  """
  
  one dimensional application of sigmoid activation funciton
  
  """
  
  return 1./( 1 + np.exp( -1 * x ) )



def generateUniformMat( Ncomp, condT ):
  """
  
  generate a random matrix by sampling each element uniformly at random 
  check condition number versus a condition threshold
  
  """
  
  A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
  for i in range(Ncomp):
    A[:,i] /= np.sqrt( (A[:,i]**2).sum())
  
  while np.linalg.cond(A) > condT:
    # generate a new A matrix!
    A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1 
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
      
  return A



def gen_causal_TCL_data( Nlayer, Nsegment, NsegmentObs, Ncomp, NonLin='leaky', LinType='uniform', negSlope=.2, Niter4condThresh =1e4, varyMean=False):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)

  Enforcing causality here !

  INPUT
      - Nlayer: number of layers in the "mixing-MLP" (the MLP used to mix the data in a non-linear fashion)
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - Ncomp: number of components (i.e., number of random sources & the dimensionality of the data)
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid 
      - LinType: how to generate linear mixing matricies for each layer. Can be one of "uniform"=entries U[0,1] or "ortho"=random orthonormal matrix
      
      - specfically for leaky activation we also have:
	- negSlope: slope for x < 0 in leaky ReLU
	- Niter4condThresh: number of random matricies to generate to ensure well conditioned 

      
  OUTPUT:
      - output is a dictionary with the following values: 
	  - sources: original non-stationary source
	  - obs: mixed sources 
	  - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']
  assert LinType in ['uniform', 'ortho']
  
  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each obsevation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
  dat = scale( dat ) # set to zero mean and unit variance 
  
  # get modulation parameters
  modMat  = np.random.uniform( 0.5 , 3, (Ncomp, Nsegment) )
  meanMat = np.random.uniform( -1.5, 1.5, (Ncomp, Nsegment)) # for time varying means
  
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    if varyMean:
      dat[ segID, :] = np.add( dat[ segID,: ], meanMat[:, seg ] )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg])
    labels[ segID] = seg 
    
  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)
  
  if LinType=='uniform':
    # will generate random uniform matricies and check thier condition number based on following simulations:
    condList = []
    for i in range(int(Niter4condThresh)):
      A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
      for i in range(Ncomp):
        A[:,i] /= np.sqrt( (A[:,i]**2).sum())
      condList.append( np.linalg.cond( A ))
    
    condThresh = np.percentile( condList, 25 ) # only accept those below 25% percentile 
    

  for l in range(Nlayer, 0, -1):
    if l == Nlayer:
      pass # this is the top layer
    else:
      if NonLin=="leaky":
        mixedDat = leaky_ReLU( mixedDat, negSlope ) 
      elif NonLin=="sigmoid":
        mixedDat = sigmoidAct( mixedDat )
	
    # now apply matrix mutliplication:
    if LinType=="uniform":
      A = generateUniformMat( Ncomp, condThresh )
    elif LinType=="ortho":
      A = np.linalg.qr( np.random.uniform( -1,1, (Ncomp, Ncomp) ) )[0] # we take orthonormal decomposition
    A[1,0] = 0
    print(A)
    
    # finally, apply the linear transformation
    mixedDat = np.dot( mixedDat, A) 

    # finally flip the order randomly
    if np.random.uniform() < .5 :
      # we flip
      dat = dat[:, [1,0]]
      mixedDat = mixedDat[:, [1,0]]
      model = 'y->x'
    else:
      model = 'x->y'

  return {'source': dat, 'obs': mixedDat, 'labels': labels, 'model': model}

  


#X = gen_TCL_data( Nlayer=2, Nsegment=5, NsegmentObs=100, Ncomp=5, NonLin='leaky', LinType='uniform',  negSlope=.2, Niter4condThresh=1e4 )
#X = gen_TCL_data( Nlayer=2, Nsegment=5, NsegmentObs=100, Ncomp=5, NonLin='leaky', LinType='ortho',  negSlope=.2, Niter4condThresh=1e4 )
#X = gen_TCL_data( Nlayer=2, Nsegment=5, NsegmentObs=100, Ncomp=5, NonLin='sigmoid', LinType='ortho',  negSlope=.2, Niter4condThresh=1e4 )




