# Hungarian algorithm for correlating true with recovered sources
#
#

import numpy as np 
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr 

def SolveHungarian( recov, source ):
	"""
	compute maximum correlations between true indep components and estimated components 
	"""
	Ncomp = source.shape[1]
	CorMat = (np.abs(np.corrcoef( recov.T, source.T ) ) )[:Ncomp, Ncomp:]
	ii = linear_sum_assignment( -1*CorMat )

	return CorMat[ii].mean(), CorMat, ii


def SolveHungarianSpearman( recov, source ):
	"""
	correlation measure using spearman correlation
	"""

	Ncomp = source.shape[1]
	CorMat = np.zeros( ( Ncomp, Ncomp) )

	for i in range( Ncomp ):
		for j in range( Ncomp ):
			CorMat[i,j] = np.abs( spearmanr( recov[:,i], source[:,j] )[0] )


	(np.abs(np.corrcoef( recov.T, source.T ) ) )[:Ncomp, Ncomp:]
	ii = linear_sum_assignment( -1*CorMat )

	return CorMat[ii].mean(), CorMat, ii
