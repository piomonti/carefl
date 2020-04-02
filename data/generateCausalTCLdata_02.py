### Generate data based on the non-stationary, non-linear ICA model of Hyvarinen & Morioka (2016)
#
# data is generated using two distinct non-linearity functions: leaky-ReLU and sigmoid functions 
#
# NOTE: the fundamental difference here is the data has a clear causal structure! 
#

import numpy as np
from sklearn.preprocessing import scale

def gen_toy_causal_dat( nObs=100, causalFunc='square' ):
    """
    generate causal data where one variable causes another 
    Inputs:
        - nObs: number of observations
        - causalFunc: specify causal function
    """

    causalFuncDict = {'linear'    : lambda x, n: .5*x+n,
                      'square'    : lambda x, n: x*x+n,
                      'quad'      : lambda x, n: .3*x*x*x*x+np.abs(n),
                      'log'       : lambda x, n: np.log(np.abs(x))+n,
                      'product'   : lambda x, n: (np.abs(x))*np.abs(n),
                      'arctan'    : lambda x, n: (np.arctan(x))**2 + np.abs(n),
                      'hoyer2009' : lambda x, n: x+x*x*x+n
                      }
                      #'non-additive: '}


    N = ( np.random.laplace(  loc=0, scale=1, size=( nObs, 2 ) ) )

    X = np.zeros(( nObs, 2))
    X[:,0] = N[:,0]
    X[:,1] = causalFuncDict[causalFunc]( X[:,0], N[:,1] ) 

    return X 

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
  one dimensional application of sigmoid activation function
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



def generateUniformMat_causal2( Ncomp, condT ):
  """  
  generate a random matrix by sampling each element uniformly at random 
  check condition number versus a condition threshold

  we make a small amendment for the matrix to encode causal structure!
  """
  print('new code')
  # A = np.random.uniform(0,1, (Ncomp, Ncomp))
  A = np.eye( Ncomp )#np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
  A[0,1] = np.random.uniform(low=-1, high=1.)
  for i in range(Ncomp):
    A[:,i] /= np.sqrt( (A[:,i]**2).sum())
  
  while np.linalg.cond(A) > condT:
    # generate a new A matrix!
    A[0,1] = np.random.uniform(low=-1, high=1.)
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
      
  return A



def generateUniformMat_causal( Ncomp, condT ):
  """  
  generate a random matrix by sampling each element uniformly at random 
  check condition number versus a condition threshold

  we make a small amendment for the matrix to encode causal structure!
  """
  # A = np.random.uniform(0,1, (Ncomp, Ncomp))
  A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
  A[1,0] = 0
  for i in range(Ncomp):
    A[:,i] /= np.sqrt( (A[:,i]**2).sum())
  
  while np.linalg.cond(A) > condT:
    # generate a new A matrix!
    A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1 
    A[1,0] = 0
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
      
  return A





def gen_2dcausalTCL_data_1nonLinLayer(  Nsegment, NsegmentObs, NonLin='leaky', nonLinType='additiveNoise',  negSlope=.2, Niter4condThresh =1e4, control=False):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
  with the DIFFERENCE that data has causal relationship as well!

  By default we consider 2d data for now

  INPUT
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid       
        Specifically for leaky activation we also have:
        	- negSlope: slope for x < 0 in leaky ReLU
	        - Niter4condThresh: number of random matricies to generate to ensure well conditioned 
      - NonLinType: where should the causal matrix be applied. One of 'additiveNoise' or 'generalNonLin'.
        For the latter we apply causal matrix before applying nonlinearity!
      - control: should we generate nonlinear data without causal structure, to see benchmark!

  OUTPUT:
    - output is a dictionary with the following values: 
  	  - sources: original non-stationary source
	    - obs: mixed sources 
	    - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']
  Ncomp = 2 # focus on special case for now

  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each obsevation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
  dat = scale( dat ) # set to zero mean and unit variance 
  
  # get modulation parameters
  modMat = np.random.uniform( 0 , 1, (Ncomp, Nsegment) )
  
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg])
    labels[ segID] = seg 
    
  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)

  # generate mixing matrices:  
  # will generate random uniform matricies and check thier condition number based on following simulations:
  condList = []
  for i in range(int(Niter4condThresh)):
    # A = np.random.uniform(0,1, (Ncomp, Ncomp))
    A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
    if control==False:
      A[1,0] = 0 # because we wish for this to be causal!
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
    condList.append( np.linalg.cond( A ))

  condThresh = np.percentile( condList, 15 ) # only accept those below 25% percentile 

  # now we apply the matrix multiplication - which enforces causality:
  if control:
    A = generateUniformMat( Ncomp, condThresh )
  else: 
    A = generateUniformMat_causal( Ncomp, condThresh ) # generateUniformMat( Ncomp, condThresh )

  # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity! (either additive or more general!)  
  if nonLinType == 'additiveNoise':
    # we first apply non-linear function, then causal matrix!
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )

    mixedDat = np.dot( mixedDat, A )
  elif nonLinType == 'generalNonLin':
    # we apply causal matrix first, then nonlinear function!
    mixedDat = np.dot( mixedDat, A )
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )

  # finally apply causal bit:
  caus_mat = np.eye(Ncomp)
  #caus_mat[0,1] = .5
  return {'source': dat, 'obs': mixedDat.dot(caus_mat), 'labels': labels}

  
#X = gen_2dcausalTCL_data_1nonLinLayer( Nsegment=5, NsegmentObs=100, NonLin='leaky', negSlope=.2, Niter4condThresh=1e4 )



def gen_2dcausalTCL_data(  Nlayer, Nsegment, NsegmentObs, NonLin='leaky',  negSlope=.2, Niter4condThresh =1e4, control=False, ThresPerc=15, takeAbs=True, source='laplace'):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
  with the DIFFERENCE that data has causal relationship as well!

  By default we consider 2d data for now

  This function extends the function 'gen_2dcausalTCL_data_1nonLinLayer' by allowing for 
  multiple non-linear layers!

  INPUT
      - Nlayer: number of non-linear layers!
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid       
        Specifically for leaky activation we also have:
          - negSlope: slope for x < 0 in leaky ReLU
          - Niter4condThresh: number of random matricies to generate to ensure well conditioned 
      - control: should we generate nonlinear data without causal structure, to see benchmark!

  OUTPUT:
    - output is a dictionary with the following values: 
      - sources: original non-stationary source
      - obs: mixed sources 
      - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']
  Ncomp = 2 # focus on special case for now

  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each obsevation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  if source=='laplace':
    dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat ) # set to zero mean and unit variance 
    # get modulation parameters
    modMat = np.random.uniform( 0 , 1, (Ncomp, Nsegment) )

  else:
    #print 'gaussian sources'
    # gaussian sources:
    dat = np.random.normal( loc=0.0, scale=1.0, size=(Nobs, Ncomp) )
    dat = scale( dat )
    # get modulation parameters
    modMat = np.random.uniform( .5 , 2, (Ncomp, Nsegment) )
    if control==False:
      modMat[0,:] = 1

  if (takeAbs):
    dat = np.abs( dat )
    print('taking abs')
  
  
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg])
    labels[ segID] = seg 
    
  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)

  # generate mixing matrices:  
  # will generate random uniform matricies and check thier condition number based on following simulations:
  condList = []
  for i in range(int(Niter4condThresh)):
    # A = np.random.uniform(0,1, (Ncomp, Ncomp))
    A = np.eye( Ncomp ) #= np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
    A[0,1] = np.random.uniform(low=-1, high=1)
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
    condList.append( np.linalg.cond( A ))

  condThresh = np.percentile( condList, ThresPerc ) # only accept those below 15% percentile 

  # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity! (either additive or more general!)  
  for n in range(Nlayer-1):
    # we first apply non-linear function, then causal matrix!
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )

    # now we apply the matrix multiplication - which enforces causality:
    if control:
      A = generateUniformMat( Ncomp, condThresh )
    else: 
      A = generateUniformMat_causal2( Ncomp, condThresh )
      #A = generateUniformMat_causal( Ncomp, condThresh ) # generateUniformMat( Ncomp, condThresh )

    # apply matrix multiplication
    mixedDat = np.dot( mixedDat, A )


  # finally apply causal bit:
  caus_mat = np.eye(Ncomp)
  if control:
    A  = generateUniformMat( Ncomp, condThresh )
    A += A.T
    mixedDat = np.dot( mixedDat, A )
  else:
    caus_mat[0,1] = .5 
  # caus_mat[0,1] = .5
  return {'source': dat, 'obs': mixedDat.dot(caus_mat), 'labels': labels}




def gen_2dcausalTCL_data_staircase(  Nlayer, Nsegment, NsegmentObs, NonLin='leaky',  negSlope=.2, Niter4condThresh =1e4, control=False, ThresPerc=15, takeAbs=True, source='laplace', shuffle=False):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
  with the DIFFERENCE that data has causal relationship as well!

  By default we consider 2d data for now

  This function extends the function 'gen_2dcausalTCL_data_1nonLinLayer' by allowing for 
  multiple non-linear layers!

  INPUT
      - Nlayer: number of non-linear layers!
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid       
        Specifically for leaky activation we also have:
          - negSlope: slope for x < 0 in leaky ReLU
          - Niter4condThresh: number of random matricies to generate to ensure well conditioned 
      - control: should we generate nonlinear data without causal structure, to see benchmark!

  OUTPUT:
    - output is a dictionary with the following values: 
      - sources: original non-stationary source
      - obs: mixed sources 
      - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']
  Ncomp = 2 # focus on special case for now

  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each obsevation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  if source=='laplace':
    dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat ) # set to zero mean and unit variance 
    # get modulation parameters
    modMat = np.random.uniform( 0 , 1, (Ncomp, Nsegment) )

  else:
    #print 'gaussian sources'
    # gaussian sources:
    dat = np.random.normal( loc=0.0, scale=1.0, size=(Nobs, Ncomp) )
    dat = scale( dat )
    # get modulation parameters
    modMat = np.random.uniform( .5 , 2, (Ncomp, Nsegment) )
    if control==False:
      modMat[0,:] = 1

  if (takeAbs):
    dat = np.abs( dat )
    print('taking abs')
  
  
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg])
    labels[ segID] = seg 
    
  # finally we add staircasing:
  if shuffle:
    # add with shuffled staircase
    print('staircase shuffling')
    shuffle = np.random.permutation( np.unique(labels))
    new_labels_dict = { i: shuffle[i] for i in range(Nsegment) }
    new_labels = np.array( [new_labels_dict[x] for x in labels] )
    dat += new_labels.reshape((-1,1)) * 2
  else:
    dat += labels.reshape((-1,1)) * 2

  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)


  # generate mixing matrices:  
  # will generate random uniform matricies and check thier condition number based on following simulations:
  condList = []
  for i in range(int(Niter4condThresh)):
    # A = np.random.uniform(0,1, (Ncomp, Ncomp))
    A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
    if control==False:
      A[1,0] = 0 # because we wish for this to be causal!
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
    condList.append( np.linalg.cond( A ))

  condThresh = np.percentile( condList, ThresPerc ) # only accept those below 15% percentile 

  # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity! (either additive or more general!)  
  for n in range(Nlayer-1):
    # we first apply non-linear function, then causal matrix!
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )

    # now we apply the matrix multiplication - which enforces causality:
    if control:
      A = generateUniformMat( Ncomp, condThresh )
    else: 
      A = generateUniformMat_causal( Ncomp, condThresh ) # generateUniformMat( Ncomp, condThresh )

    # apply matrix multiplication
    mixedDat = np.dot( mixedDat, A )


  # finally apply causal bit:
  caus_mat = np.eye(Ncomp)
  if control:
    A  = generateUniformMat( Ncomp, condThresh )
    A += A.T
    mixedDat = np.dot( mixedDat, A )
  else:
    caus_mat[0,1] = .5 
  # caus_mat[0,1] = .5
  return {'source': dat, 'obs': mixedDat.dot(caus_mat), 'labels': labels}









def gen_ANM_data(  Nlayer, Nsegment, NsegmentObs, NonLin='leaky',  negSlope=.2, Niter4condThresh =1e4, control=False, ThresPerc=15, takeAbs=True, source='laplace'):
  """
  
  generate multivariate data based on the non-stationary non-linear ICA model of Hyvarinen & Morioka (2016)
  with the DIFFERENCE that data has causal relationship as well!
  Here the data follows the ADDITIVE NOISE MODEL (ie we apply nonlin and then add original sources as well!)

  By default we consider 2d data for now

  This function extends the function 'gen_2dcausalTCL_data_1nonLinLayer' by allowing for 
  multiple non-linear layers!

  INPUT
      - Nlayer: number of non-linear layers!
      - Nsegment: number of data segments to generate
      - NsegmentObs: number of observations per segment 
      - NonLin: linearity employed in non-linear mixing. Can be one of "leaky" = leakyReLU or "sigmoid"=sigmoid       
        Specifically for leaky activation we also have:
          - negSlope: slope for x < 0 in leaky ReLU
          - Niter4condThresh: number of random matricies to generate to ensure well conditioned 
      - control: should we generate nonlinear data without causal structure, to see benchmark!

  OUTPUT:
    - output is a dictionary with the following values: 
      - sources: original non-stationary source
      - obs: mixed sources 
      - labels: segment labels (indicating the non stationarity in the data)
  
  
  """
  
  # check input is correct
  assert NonLin in ['leaky', 'sigmoid']
  Ncomp = 2 # focus on special case for now

  # generate non-stationary data:
  Nobs = NsegmentObs * Nsegment # total number of observations
  labels = np.array( [0] * Nobs ) # labels for each obsevation (populate below)
  
  # generate data, which we will then modulate in a non-stationary manner:
  if source=='laplace':
    dat = np.random.laplace( 0, 1, (Nobs, Ncomp) )
    dat = scale( dat ) # set to zero mean and unit variance 
    # get modulation parameters
    modMat = np.random.uniform( 0 , 1, (Ncomp, Nsegment) )

  else:
    #print 'gaussian sources'
    # gaussian sources:
    dat = np.random.normal( loc=0.0, scale=1.0, size=(Nobs, Ncomp) )
    dat = scale( dat )
    # get modulation parameters
    modMat = np.random.uniform( .5 , 2, (Ncomp, Nsegment) )
    if control==False:
      modMat[0,:] = 1

  if (takeAbs):
    dat = np.abs( dat )
    print('taking abs')
  
  
  # now we adjust the variance within each segment in a non-stationary manner
  for seg in range(Nsegment):
    segID = range( NsegmentObs*seg, NsegmentObs*(seg+1) )
    dat[ segID, :] = np.multiply( dat[ segID, :], modMat[:, seg])
    labels[ segID] = seg 
    
  # now we are ready to apply the non-linear mixtures: 
  mixedDat = np.copy(dat)

  # generate mixing matrices:  
  # will generate random uniform matricies and check thier condition number based on following simulations:
  condList = []
  for i in range(int(Niter4condThresh)):
    # A = np.random.uniform(0,1, (Ncomp, Ncomp))
    A = np.random.uniform(0,2, (Ncomp, Ncomp)) - 1
    if control==False:
      A[1,0] = 0 # because we wish for this to be causal!
    for i in range(Ncomp):
      A[:,i] /= np.sqrt( (A[:,i]**2).sum())
    condList.append( np.linalg.cond( A ))

  condThresh = np.percentile( condList, ThresPerc ) # only accept those below 15% percentile 

  # now we apply layers of non-linearity (just one for now!). Note the order will depend on natural of nonlinearity! (either additive or more general!)  
  for n in range(Nlayer-1):
    # we first apply non-linear function, then causal matrix!
    if NonLin == 'leaky':
      mixedDat = leaky_ReLU( mixedDat, negSlope )
    elif NonLin == 'sigmoid':
      mixedDat = sigmoidAct( mixedDat )

    # now we apply the matrix multiplication - which enforces causality:
    if control:
      A = generateUniformMat( Ncomp, condThresh )
    else: 
      A = generateUniformMat_causal( Ncomp, condThresh ) # generateUniformMat( Ncomp, condThresh )

    # apply matrix multiplication
    mixedDat = np.dot( mixedDat, A )


  # finally apply causal bit:
  caus_mat = np.eye(Ncomp)
  if control:
    A  = generateUniformMat( Ncomp, condThresh )
    A += A.T
    mixedDat = np.dot( mixedDat, A )
  else:
    caus_mat[0,1] = .5 
  # caus_mat[0,1] = .5
  return {'source': dat, 'obs': mixedDat.dot(caus_mat) + dat, 'labels': labels}



