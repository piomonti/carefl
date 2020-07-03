import numpy as np
from sklearn.preprocessing import scale

def sigmoid( x ):
  return 1./(1 + np.exp(-x))

def gen_synth_causal_dat( nObs=100, causalFunc='square' ):
    """
    generate causal data where one variable causes another 
    Inputs:
        - nObs: number of observations
        - causalFunc: specify causal function
    """

    causalFuncDict = {'linear'       : lambda x, n: 1*x+n,
                      'hoyer2009'    : lambda x, n: x+ (.5)*x*x*x + (n),
                      'nueralnet_l1' : lambda x, n: sigmoid( sigmoid( np.random.normal(loc=1)*x ) + n )
                      }

    N = ( np.random.laplace(  loc=0, scale=1./np.sqrt(2), size=( nObs, 2 ) ) ) # scale divided by np.sqrt(2) to ensure std of 1 

    X = np.zeros(( nObs, 2))
    X[:,0] = N[:,0]
    X[:,1] = causalFuncDict[causalFunc]( X[:,0], N[:,1] ) 

    if np.random.uniform() < .5:
      mod_dir = 'y->x'
      X = X[ :, [1,0] ]
    else:
      mod_dir = 'x->y'

    return X, mod_dir