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

"""
permutation baseline (each story)
"""
def generate_baseline(data, iterations=10000):
    n_people, n_locations = data.shape
    baseline = np.empty((iterations, n_locations))
    for j in range(iterations):
        iteration = np.array([permutation(data[i,:]) for i in range(n_people)])
        baseline[j] = np.nanmean(iteration,axis=0)
    sig = 5/n_locations  # bonferroni in percentage
    baseline_vec_upper = np.percentile(baseline,100-sig/2,axis=0)
    baseline_vec_lower = np.percentile(baseline,sig/2,axis=0)
    return baseline_vec_upper, baseline_vec_lower

for id in ['eyespy','pieman','baseball','oregon']:
    subfolder = '%s_t40_v55_r55_s21' % id
    filename = [x for x in os.listdir(os.path.join(DATA_DIR, subfolder)) if 'precision_array' in x][0]
    precisions = np.load(os.path.join(DATA_DIR, subfolder, filename), allow_pickle=True)
    precisions[precisions>0] = 1  # turn precision matrix into probability of recall
    baseline_upper, baseline_lower = generate_baseline(precisions)
    # save baseline
    np.save(os.path.join(DATA_DIR,subfolder,'probrecall_baseline_upper'),  np.array(baseline_upper))
    np.save(os.path.join(DATA_DIR,subfolder,'probrecall_baseline_lower'),  np.array(baseline_lower))

    # plot
    fig, ax = plt.subplots(figsize=(20, 20))
    y = np.mean(precisions, axis=0)
    ci = 1.96 * np.std(precisions, axis=0) / np.sqrt(len(precisions))
    ax.plot(np.arange(1,precisions.shape[1]+1), y)
    ax.plot(np.arange(1,precisions.shape[1]+1), baseline_upper, color='r')
    ax.plot(np.arange(1,precisions.shape[1]+1), baseline_lower, color='r')
    ax.fill_between(np.arange(1, precisions.shape[1]+1), (y - ci), (y + ci),
                    color='b',
                    alpha=.1)
    fig.savefig(os.path.join(IMG_DIR, subfolder, 'probrecall_baseline.png'))

"""
probability of first recall (each story)
"""
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1,2, figsize=(12,4.5))
colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3']
ax = axes.flat[0]
k = 0
for id in ['eyespy','pieman','baseball','oregon']:
    subfolder = '%s_t40_v55_r55_s21' % id
    # filename = [x for x in os.listdir(os.path.join(DATA_DIR, subfolder)) if 'precision_array' in x][0]
    recall = np.load(os.path.join(DATA_DIR, subfolder, 'recall_events.npy'), allow_pickle=True)
    video_events = np.load(os.path.join(DATA_DIR, subfolder, 'video_events.npy'), allow_pickle=True)
    if id == 'pieman':
        video_events = video_events[0:24]
    save = np.zeros((len(recall),video_events.shape[0]))
    n=0
    # compute precision mat
    for recall_event in recall:
        corrmat = 1 - cdist(video_events, recall_event, 'correlation')  # this is the correlation matrix
        precise_mat = precise_matches_mat(corrmat, 'recall')
        ind = np.argmax(precise_mat[:,0])  # the first recall corresponds to which original event?
        save[n,ind] = 1
        n += 1
    # plot
    if id == 'oregon':
        id = 'oregontrail'
    y = np.mean(save,axis=0)
    ci = 1.96 * np.std(save,axis=0)/np.sqrt(len(save))  #std error
    # ax = axes.flat[k]
    ax.plot(np.arange(1,len(y)+1),y, color=colors[k])
    # ax.set_xticks(np.arange(1,len(y)+1))
    ax.fill_between(np.arange(1,len(y)+1), (y-ci), (y+ci), color=colors[k], alpha=.1)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.set_ylabel('probability of recall')
    ax.set_xlabel('event number')
    # ax.set_title(id)
    k = k+1
ax.set_title('probability of first recall')
# fig.savefig(os.path.join('final_plots', 'first_recall.png'))

"""
lag recency (each story)
method: For each recall transition (following the first recall), we computed 
the lag between the current recall event and the next recall event, normalizing 
by the total number of possible transitions. 
This yielded a number-of-participants (17) by number-of-lags (−29 to +29; 58 lags in total excluding lags of 0) matrix. 
"""
method = 'recall'
ax = axes.flat[1]
colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3']
k = 0
for id in ['eyespy','pieman','baseball','oregon']:
    subfolder = '%s_t40_v55_r55_s21' % id
    filename = [x for x in os.listdir(os.path.join(DATA_DIR,subfolder)) if 'precision_array' in x][0]
    recall = np.load(os.path.join(DATA_DIR,subfolder,'recall_events.npy'), allow_pickle=True)
    video_events = np.load(os.path.join(DATA_DIR, subfolder, 'video_events.npy'), allow_pickle=True)
    if id == 'pieman':
        video_events = video_events[0:24]
    save = np.zeros((len(recall),2*video_events.shape[0]-1))  # n_participant x (2*event-1)  for saving the result
    n = 0
    for recall_event in recall:
        # compute precision mat
        corrmat = 1 - cdist(video_events, recall_event, 'correlation')  # this is the correlation matrix
        precise_mat = precise_matches_mat(corrmat,method)
        # Compute lag
        for recall_idx in range(len(recall_event) - 1):
            ind1 = np.argmax(precise_mat[:, recall_idx])
            ind2 = np.argmax(precise_mat[:, recall_idx + 1])
            lag = ind2 - ind1
            lag_index = lag + (2*video_events.shape[0]-1) // 2
            save[n, lag_index] += 1
        save[n] = save[n]/np.sum(save[n])
        n += 1
    # plot
    if id == 'oregon':
        id = 'oregontrail'

    # Compute the mean recall probability at each lag across all participants
    # Compute the mean across participants for each lag
    save[:,video_events.shape[0]-1] = np.nan
    y = np.nanmean(save, axis=0)
    # bootstrap CI
    ci = 1.96 * np.std(save,axis=0)/np.sqrt(len(save))
    # ax = axes.flat[k]
    ax.plot(np.arange(-video_events.shape[0]+1,video_events.shape[0]),y,color=colors[k],label=id)
    ax.set_xlabel('lag')
    ax.set_ylabel('conditional response probability')
    ax.fill_between(np.arange(-video_events.shape[0]+1,video_events.shape[0]), (y-ci), (y+ci), color=colors[k], alpha=.1)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    k+=1
fig.legend()
ax.set_title('lag recency')
fig.tight_layout()
fig.savefig(os.path.join('final_plots', 'listLearning.png'))
