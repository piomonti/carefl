## Exploratory analysis and some causal discovery/etc on interventional fMRI data
#
#
#
#

import numpy as np 
import pandas as pd
import os 
import pylab as plt; plt.ion()
import seaborn as sns
from sklearn.preprocessing import scale
import yaml

os.chdir('/Users/ricardo/Documents/Projects/AffineFlowCausalInf/Application/data')

# load some data in to see whats going on

labels = np.array( pd.read_table('labels.txt') ).squeeze()

# 
# from Robs notes, looks like:
#
# Run1 Cingulate Gyrus anterior division
# Run2 No notes (maybe no stimulation)
# Run3 Cingulate Gyrus anterior division
# Run4 Heschl's Gyrus 
# Run5 Central Opercular Cortex 


# lets load in data for 

dat = [np.array(pd.read_csv(f"Run{i}.txt", header=0)) for i in range(1,6)]

scale_dat = [scale(d, with_mean=True) for d in dat]

intervention_regions = ['Cingulate Gyrus, anterior division', None, 
						'Cingulate Gyrus, anterior division', "Heschl's Gyrus (includes H1 and H2)",
						'Central Opercular Cortex']


interventional_id = [np.argmax(labels==i) if i is not None else None for i in intervention_regions]

rois = np.unique([roi for roi in interventional_id if roi is not None])
rois = np.array([28,44]) # just take 2 ROIs for now

# plot resting state data
int_dset = 2 # interventional dataset
df_12 = pd.DataFrame(np.vstack((scale_dat[1][:, rois], 
								scale_dat[3][:, rois],
								scale_dat[2][:, rois])))
df_12.columns = labels[rois]
df_12['type'] = ['rest'] * 253 + ['intvene cingulate'] * 253 + ['intvene Heschl'] * 253


sns.pairplot(df_12, kind='scatter', diag_kind='kde', hue='type', markers=["o", "s", "D"]) 

# load in the yaml
config = yaml.load(open('/Users/ricardo/Documents/Projects/AffineFlowCausalInf/Application/Scripts/fmri_app.yaml'), Loader=yaml.FullLoader)

import argparse
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

config = dict2namespace(config)


# perform causal discovery using the resting dataset
os.chdir('../../')
from models.carefl import CAReFl
#from models.affine_flow_cd import BivariateFlowLR

train_dat = scale_dat[1][:, rois]

np.random.seed(1)

mod = CAReFl(config)

p = mod.flow_lr(train_dat)

dag = np.zeros((2,2))
dag[1,0] = 1

mod.fit_to_sem(train_dat, dag)

intervene_dat = scale_dat[2][:, rois]
xvals = intervene_dat[:,0]

int_pred = np.array([mod.predict_intervention(x, n_samples=500)[1][0,1] for x in xvals])


## compare with GPs and Linear Regression

from sklearn.linear_model import LinearRegression, Ridge

lm = Ridge() # LinearRegression()
lm.fit(X=train_dat[:,0].reshape((-1,1)), y=train_dat[:,1])


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=25, n_restarts_optimizer=10)

gp.fit(X=train_dat[:,0].reshape((-1,1)), y=train_dat[:,1])


results_df = pd.DataFrame({'carefl': np.abs((int_pred)-(intervene_dat[:,1])),
						   'linear_regression': np.abs( (lm.predict(xvals.reshape((-1,1)))) - (intervene_dat[:,1])),
						   'gaussian_process': np.abs( (gp.predict(xvals.reshape((-1,1)))) - (intervene_dat[:,1]))})


results_df.median(axis=0)

# can also just compare at the times of actual events (stimulations)
# stim_times = pd.read_table('sub-292_ses-postop_task-es_run-02_events.tsv')
# stim_times['onsets_tr'] = np.floor(stim_times['onset'] / 3.1)
# results_df.iloc[stim_times['onsets_tr']].median(axis=0)


### repeat for the second subject !
os.chdir('/Users/ricardo/Documents/Projects/AffineFlowCausalInf/Application/data')

dat_second = [np.array(pd.read_csv(f"sub-314run-0{i}.txt", header=0)) for i in range(1,6)]

scale_dat = [scale(d, with_mean=True) for d in dat_second]

#train_dat = scale_dat[2][:, rois]
train_dat = scale_dat[0][:, rois]


mod = CAReFl(config)

p = mod.flow_lr(train_dat)

dag = np.zeros((2,2))
dag[1,0] = 1

mod.fit_to_sem(train_dat, dag)

#intervene_dat = scale_dat[3][:, rois]
intervene_dat = scale_dat[2][:, rois]
xvals = intervene_dat[:,0]

int_pred = np.array([mod.predict_intervention(x, n_samples=500)[1][0,1] for x in xvals])


#print(np.median(np.abs(int_pred-(intervene_dat[:,1]))))

## comparison with linear models and GPs (for ANM)

lm = Ridge() # LinearRegression()
lm.fit(X=train_dat[:,0].reshape((-1,1)), y=train_dat[:,1])

#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=25, n_restarts_optimizer=10)

gp.fit(X=train_dat[:,0].reshape((-1,1)), y=train_dat[:,1])


results2_df = pd.DataFrame({'carefl': np.abs((int_pred)-(intervene_dat[:,1])),
						   'linear_regression': np.abs( (lm.predict(xvals.reshape((-1,1)))) - (intervene_dat[:,1])),
						   'gaussian_process': np.abs( (gp.predict(xvals.reshape((-1,1)))) - (intervene_dat[:,1]))})


results2_df.median(axis=0)

# compare only at stimulation times
# stim_times = pd.read_table('sub-314_ses-postop_task-es_run-01_events.tsv')
# stim_times['onsets_tr'] = np.floor(stim_times['onset'] / 3.1)
# results2_df.iloc[stim_times['onsets_tr']].median(axis=0)



print('Subject 1')
print(results_df.median(axis=0))

print('\n\nSubject 2')
print(results2_df.median(axis=0))

