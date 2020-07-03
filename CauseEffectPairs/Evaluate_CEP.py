### CEP result evaluation
#
#

import numpy as np 
import os 
import pandas as pd 
import pylab as plt 

# load in results
os.chdir('CauseEffectPairs/results_split_v3/')

SplitPerc = int( 100 * .8 )

# get a list of files
files = [x for x in os.listdir( os.getcwd() ) if 'split'+str(SplitPerc) in x]

pair_ids = [x.split('pair')[1].split('_')[0] for x in files]#[:86]

results = pd.DataFrame({'pid': pair_ids,
                        'trueM': ['na'] * len(pair_ids),
                        'predM': ['na'] * len(pair_ids),
                        'sampleS': ['na'] * len(pair_ids) })


method = 'max' # or 'max'
netL = 2
netH = 5
fileAppend = '_PureFlow_laplace_Scaled.csv' # '_PureFlow__outlierRemoval__Scaled.csv' #'.csv'

for pid in pair_ids:
    res = pd.read_csv('pair' + pid + '_split' + str(SplitPerc) + fileAppend)

    # add some filtering here
    #res = res[ res.L==netL ]
    res = res[ res.L !=2 ]
    res = res[ res.nh==netH ]

	# get the true model
    pair_id = '0' * (4-len(pid)) + pid
    dir_id = open('../pairs/pair' + str(pair_id) + '_des.txt', 'r').read().lower()#.split('ground truth:')[1].strip() #split('\n')[1]

    # determine causal direction (from dir_id file):
    dir_id = dir_id.replace('\n', '')
    dir_id = dir_id.replace(':' , '')
    dir_id = dir_id.replace(' ', '')

    if ('x-->y' in dir_id) | ('x->y' in dir_id):
        dir_id = 'x->y'
    elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
        dir_id = 'y->x'

    dat_id = np.loadtxt('../pairs/pair' + str(pair_id) + '.txt') 

    # store results
    if method == 'vote':
        results.loc[ results.pid == pid, 'predM' ] = 'x->y' if (res['x->y'] > res['y->x']).mean() >= .5 else 'y->x'
    else:
        results.loc[ results.pid == pid, 'predM' ] = 'x->y' if res.max()['x->y'] > res.max()['y->x'] else 'y->x'
    results.loc[ results.pid == pid, 'trueM' ] = dir_id
    results.loc[ results.pid == pid, 'sampleS'] = dat_id.shape[0]


# manual fixes
results.iloc[82]['trueM'] = 'x->y'
results.iloc[84]['trueM'] = 'x->y'

results2 = results[ results.pid.astype('int') < 88 ]

print( ( results['trueM'] == results['predM']).mean() )
print( ( results2['trueM'] == results2['predM']).mean() )



# lets go through each possible combination
res_nets = res
del res_nets['Unnamed: 0']
del res_nets['x->y']
del res_nets['y->x']

res_nets['acc'] = 0
res_nets['acc88'] = 0

for i in range( res_nets.shape[0] ):
    netL = res_nets.iloc[i].L
    netH = res_nets.iloc[i].nh

    results = pd.DataFrame({'pid': pair_ids,
                        'trueM': ['na'] * len(pair_ids),
                        'predM': ['na'] * len(pair_ids),
                        'sampleS': ['na'] * len(pair_ids) })

    for pid in pair_ids:
        res = pd.read_csv('pair' + pid + '_split' + str(SplitPerc) + fileAppend)

        # add some filtering here
        res = res[ res.L==netL ]
        res = res[ res.nh==netH ]

        # get the true model
        pair_id = '0' * (4-len(pid)) + pid
        dir_id = open('../pairs/pair' + str(pair_id) + '_des.txt', 'r').read().lower()#.split('ground truth:')[1].strip() #split('\n')[1]

        # determine causal direction (from dir_id file):
        dir_id = dir_id.replace('\n', '')
        dir_id = dir_id.replace(':' , '')
        dir_id = dir_id.replace(' ', '')

        if ('x-->y' in dir_id) | ('x->y' in dir_id):
            dir_id = 'x->y'
        elif ('y-->x' in dir_id) | ('y->x' in dir_id) | ('x<-y' in dir_id):
            dir_id = 'y->x'

        dat_id = np.loadtxt('../pairs/pair' + str(pair_id) + '.txt') 

        # store results
        if method == 'vote':
            results.loc[ results.pid == pid, 'predM' ] = 'x->y' if (res['x->y'] > res['y->x']).mean() >= .5 else 'y->x'
        else:
            results.loc[ results.pid == pid, 'predM' ] = 'x->y' if res.max()['x->y'] > res.max()['y->x'] else 'y->x'
        results.loc[ results.pid == pid, 'trueM' ] = dir_id
        results.loc[ results.pid == pid, 'sampleS'] = dat_id.shape[0]

    # manual fixes
    results.iloc[81]['trueM'] = 'x->y'
    results.iloc[83]['trueM'] = 'x->y'

    results2 = results[ results.pid.astype('int') < 88 ]

    res_nets.loc[ (res_nets.L==netL) & (res_nets.nh==netH), 'acc' ] = ( results['trueM'] == results['predM']).mean() 
    res_nets.loc[ (res_nets.L==netL) & (res_nets.nh==netH), 'acc88' ] = ( results2['trueM'] == results2['predM']).mean() 





















# ---------------------
# study the mistakes
import pylab as plt; plt.ion()
incorrectCEpair = np.where( results['trueM'] != results['predM'])[0]

for pid in incorrectCEpair:
    pid = str(pid)
    pair_id = '0' * (4-len(pid)) + pid
    dat_id = np.loadtxt('../pairs/pair' + str(pair_id) + '.txt') 

    #print('True association: ' +  )

    plt.scatter( dat_id[:,0], dat_id[:,1] )
    plt.title( str(results.loc[results.pid==pid, ['trueM']]) )

    input('press any key to continue\n')
    plt.close()






# ---------------------
possibleCut = np.unique( results.sampleS)

acc = []
for p in possibleCut:
    ii = np.where( results.sampleS > p )[0]

    acc.append( (results.iloc[ ii ]['trueM']==results.iloc[ii]['predM']).mean() )



plt.plot( possibleCut, acc )


