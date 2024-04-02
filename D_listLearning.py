"""
Creates plots to examine effects shown in the list learning literature: probability of first recall, and lag recency.
"""
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from sherlock_helpers.scoring import *
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
data_dir = './result_models'
IMG_DIR = './result_plots'
story_ids = ['pieman','eyespy','oregon','baseball']
"""
probability of first recall (each story)
"""
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1,2, figsize=(12,4.5))
colors = ['#fc8d62','#66c2a5','#e78ac3','#8da0cb']
ax = axes.flat[0]
k = 0
for id in story_ids:
    subfolder = '%s_t40_v55_r55_s21' % id
    # filename = [x for x in os.listdir(os.path.join(data_dir, subfolder)) if 'precision_array' in x][0]
    recall = np.load(os.path.join(data_dir, subfolder, 'recall_events.npy'), allow_pickle=True)
    story_events = np.load(os.path.join(data_dir, subfolder, 'video_events.npy'), allow_pickle=True)  # TODO: refactor instances of 'video*.file_ext'
    if id == 'pieman':
        story_events = story_events[0:24]
    save = np.zeros((len(recall),story_events.shape[0]))
    n=0
    # compute precision mat
    for recall_event in recall:
        corrmat = 1 - cdist(story_events, recall_event, 'correlation')  # this is the correlation matrix
        precise_mat = precise_matches_mat(corrmat, 'recall')
        ind = np.argmax(precise_mat[:,0])  # the first recall corresponds to which original event?
        save[n,ind] = 1
        n += 1
    # plot
    if id == 'oregon':
        id = 'oregontrail'
    y = np.mean(save,axis=0)
    # bootstrap CI
    ci = []
    for n_ev in range(len(save[0])):
        rng = np.random.default_rng()
        res = bootstrap((save[:, n_ev],), np.nanmean, confidence_level=0.95,
                        random_state=rng, method='percentile')
        ci.append(y[n_ev] - res.confidence_interval.low)
    # ax = axes.flat[k]
    ax.plot(np.arange(1,len(y)+1),y, color=colors[k])
    # ax.set_xticks(np.arange(1,len(y)+1))
    ax.fill_between(np.arange(1,len(y)+1), (y-ci), (y+ci), color=colors[k], alpha=.1)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.set_ylabel('Probability of Recall')
    ax.set_xlabel('Event Number')
    # ax.set_title(id)
    k = k+1
ax.set_title('Probability of First Recall')
# fig.savefig(os.path.join('final_plots', 'first_recall.png'))

"""
lag recency (each story)
method: For each recall transition (following the first recall), we computed 
the lag between the current recall event and the next recall event, normalizing 
by the total number of possible transitions. 
This yielded a number-of-participants (17) by number-of-lags (−29 to +29; 58 lags in total excluding lags of 0) matrix.
"""  # TODO: fix the line above -- the numbers seem wrong.
method = 'recall'
ax = axes.flat[1]
colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3']
k = 0
for id in ['eyespy','pieman','baseball','oregon']:
    subfolder = '%s_t40_v55_r55_s21' % id
    # filename = [x for x in os.listdir(os.path.join(data_dir,subfolder)) if 'precision_array' in x][0]
    recall = np.load(os.path.join(data_dir,subfolder,'recall_events.npy'), allow_pickle=True)
    story_events = np.load(os.path.join(data_dir, subfolder, 'video_events.npy'), allow_pickle=True)  # TODO: refactor instances of 'video*.file_ext'
    if id == 'pieman':
        story_events = story_events[0:24]
    save = np.zeros((len(recall),2*story_events.shape[0]-1))  # n_participant x (2*event-1)  for saving the result
    n = 0
    for recall_event in recall:
        # compute precision mat
        corrmat = 1 - cdist(story_events, recall_event, 'correlation')  # this is the correlation matrix
        precise_mat = precise_matches_mat(corrmat,method)
        # Compute lag
        for recall_idx in range(len(recall_event) - 1):
            ind1 = np.argmax(precise_mat[:, recall_idx])
            ind2 = np.argmax(precise_mat[:, recall_idx + 1])
            lag = ind2 - ind1
            lag_index = lag + (2*story_events.shape[0]-1) // 2
            save[n, lag_index] += 1
        save[n] = save[n]/np.sum(save[n])
        n += 1
    # plot
    if id == 'oregon':
        id = 'oregontrail'

    # Compute the mean recall probability at each lag across all participants
    # Compute the mean across participants for each lag
    save[:,story_events.shape[0]-1] = np.nan
    y = np.nanmean(save, axis=0)
    # bootstrap CI
    ci = []
    for n_lag in range(len(save[0])):
        rng = np.random.default_rng()
        res = bootstrap((save[:, n_lag],), np.nanmean, confidence_level=0.95,
                        random_state=rng, method='percentile')
        ci.append(y[n_lag] - res.confidence_interval.low)
    # ax = axes.flat[k]
    ax.plot(np.arange(-story_events.shape[0]+1,story_events.shape[0]),y,color=colors[k],label=id)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Conditional Response Probability')
    ax.fill_between(np.arange(-story_events.shape[0]+1,story_events.shape[0]), (y-ci), (y+ci), color=colors[k], alpha=.1)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    k+=1
fig.legend()
ax.set_title('Lag Recency')
fig.tight_layout()
fig.savefig(os.path.join('final_plots', 'listLearning.svg'))
