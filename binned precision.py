import os
import numpy as np
import sys
import matplotlib.pyplot as plt
try:
    os.chdir('./topic_models')
except:
    pass
sys.path.append(os.getcwd())
from sherlock_helpers.scoring import *
from scipy.spatial.distance import cdist
from numpy.random import permutation
import matplotlib
import matplotlib.pyplot as plt
DATA_DIR = './result_models'
IMG_DIR = './result_plots'
"""
binned precisions (all transcripts)
"""
n_bins = 5
precisions = []
for id in ['eyespy','pieman','baseball']:
    subfolder = '%s_t40_v55_r55_s21' % id
    prec = np.load(os.path.join(DATA_DIR,subfolder,'precision_array.npy'), allow_pickle=True)
    # binning
    n_seg = prec.shape[1]//n_bins + 1  # number of points in one bin
    starts = np.arange(0,prec.shape[1],n_seg)
    # averaging based on bins
    avg_prec = []
    for s in starts:
        avg_prec.append(np.mean(prec[:,s:min(s+n_seg,prec.shape[1])], axis=1))
    precisions.append(np.array(avg_prec).T)

precisions = np.concatenate(precisions)
# plot

y = np.mean(precisions,axis=0)
ci = 1.96 * np.std(precisions,axis=0)/np.sqrt(len(precisions))
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(np.arange(1,n_bins+1),y)
ax.set_xticks(np.arange(1,n_bins+1))
ax.fill_between(np.arange(1,n_bins+1), (y-ci), (y+ci), color='b', alpha=.1)
ax.set_title('mean precision 95% CI')
