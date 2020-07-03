### plot results for causal disc simulations
#
#
#

import seaborn as sns
import pylab as plt 
import pickle 
import os
import numpy as np 

sns.set_style("whitegrid")

#sns.color_palette()[ color_dict[a] ]

os.chdir('/Users/ricardo/Documents/Projects/FlowCausalDirection/results')

# define some plotting dicts
title_dic = {'nueralnet_l1': "Neural network" + "\n" + r"$x_2 = \sigma \left ( \sigma ( x_1) + n_2 \right)$",
             'linear': "Linear SEM\n" + r"$x_2 = x_1 + n_2 $",
             'hoyer2009': "Nonlinear SEM\n"+r"$x_2 = x_1 + \frac{1}{2} x_1^3 + n_2 $"}

label_dict = {'FlowCD': 'Affine flow LR',
             'LRHyv': 'Linear LR',
             'RECI': 'RECI',
             'ANM': 'ANM',
             'notears': 'NO-TEARS'}

# define some parameters
nvals    = [ 25, 50, 75, 100,  150, 250, 500 ]
algos    = ['FlowCD', 'LRHyv', 'notears', 'RECI', 'ANM' ] 
sim_type = ['linear', 'hoyer2009', 'nueralnet_l1']

res_all  = {s:{a:[] for a in algos } for s in sim_type}

for s in sim_type:
    results = pickle.load( open(s + '_results.p', 'rb'))
    for a in algos:
        for n in range(len(nvals)):
            res_all[s][a].append( np.mean( results[n][a] == results[n]['true'] ) )
    


# prepare plot
sns.set_style("whitegrid")
sns.set_palette('deep')
#sns.set_palette( sns.color_palette("muted", 8))# sns.diverging_palette(255, 133, l=60, n=7, center="dark") )#  sns.color_palette("coolwarm", 6) )

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4), sharey=True)


for a in algos:
    ax1.plot( nvals, res_all['linear'][a],  marker='o')
    ax2.plot( nvals, res_all['hoyer2009'][a],  marker='o')
    ax3.plot( nvals, res_all['nueralnet_l1'][a],  marker='o', label = label_dict[a])

fontsize = 12
font_xlab = 10

ax1.set_title( title_dic['linear'], fontsize=fontsize)
ax2.set_title( title_dic['hoyer2009'], fontsize=fontsize)
ax3.set_title( title_dic['nueralnet_l1'], fontsize=fontsize)

ax1.set_xlabel('Sample size', fontsize=font_xlab)
ax2.set_xlabel('Sample size', fontsize=font_xlab)
ax3.set_xlabel('Sample size', fontsize=font_xlab)

ax1.set_ylabel('Proportion correct', fontsize=font_xlab)
ax2.set_ylabel('Proportion correct', fontsize=font_xlab)
ax3.set_ylabel('Proportion correct', fontsize=font_xlab)

fig.legend(  # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.2,    # Small spacing around legend box
           title="Algorithm"  # Title for the legend
           #fontsize=8
           )

plt.tight_layout()
plt.subplots_adjust(right=0.87)

os.chdir('/Users/ricardo/Documents/Projects/FlowCausalDirection/drafts/draft1')
plt.savefig('CausalDiscSims.pdf', dpi=300)


