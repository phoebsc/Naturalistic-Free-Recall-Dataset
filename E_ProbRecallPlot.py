"""
Generates plots to examine (1) probability of recall and (2) precision of recall. Also computes a permutation
baseline by scrambling tevents within each participant.
"""
from scipy.spatial.distance import cdist
import os
import sys
sys.path.append(os.getcwd())
from sherlock_helpers.scoring import precise_matches_mat, distinct_matches_mat
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap
from numpy.random import permutation
data_dir = './result_models'
img_dir = './result_plots'
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
story_ids = ['pieman','eyespy','oregon','baseball']
for id in story_ids:
    subfolder = '%s_t40_v55_r55_s21' % id
    filename = [x for x in os.listdir(os.path.join(data_dir, subfolder)) if 'precision_array' in x][0]
    precisions = np.load(os.path.join(data_dir, subfolder, filename), allow_pickle=True)
    precisions[precisions>0] = 1  # turn precision matrix into probability of recall
    baseline_upper, baseline_lower = generate_baseline(precisions)
    # save baseline
    np.save(os.path.join(data_dir,subfolder,'probrecall_baseline_upper'),  np.array(baseline_upper))
    np.save(os.path.join(data_dir,subfolder,'probrecall_baseline_lower'),  np.array(baseline_lower))

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
    fig.savefig(os.path.join(img_dir, subfolder, 'probrecall_baseline.png'))
"""
plotting precision plot
"""
plt.rcParams.update({'font.size': 13})
story_ids = ['pieman','eyespy','oregon','baseball']
colors = ['#fc8d62','#66c2a5','#e78ac3','#8da0cb']
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
FIG_DIR = 'final_plots'
plot_n=0
for story_id in story_ids:
    data_dir = 'result_models'
    img_dir = 'result_plots'
    subfolder = '%s_t40_v55_r55_s21' % story_id
    story_events = np.load(os.path.join(data_dir,subfolder,'story_events.npy'), allow_pickle=True)
    recall_events = np.load(os.path.join(data_dir,subfolder,'recall_events.npy'), allow_pickle=True)
    event_mappings = np.load(os.path.join(data_dir,subfolder,'labels.npy'), allow_pickle=True)
    _, _, recall_ids = np.load(os.path.join(data_dir,subfolder+'.npy'),
                                         allow_pickle=True)
    baseline_upper = np.load(os.path.join(data_dir, subfolder,
                       [x for x in os.listdir(os.path.join(data_dir, subfolder)) if 'baseline_upper' in x][0]))
    # produce a grand average baseline
    baseline_upper = np.mean(baseline_upper)
    baseline_lower = np.load(os.path.join(data_dir, subfolder,
                       [x for x in os.listdir(os.path.join(data_dir, subfolder)) if 'baseline_lower' in x][0]))

    if story_id == 'pieman':
        story_events = story_events[0:24]
    """
    scoring the correlation matrix (testing different methods)
    """

    """
    scoring all recalls
    """

    ##
    precisions = []
    distincts = []
    method = 'recall'

    for recall_event in recall_events:
        corrmat = 1 - cdist(story_events, recall_event, 'correlation')  # this is the correlation matrix
        # plotting the story-recall matrix
        # computing scores
        precise_mat = precise_matches_mat(corrmat,method)
        precise = np.max(precise_mat,axis=1)
        distinct_mat = distinct_matches_mat(corrmat,method)
        distinct = np.max(distinct_mat, axis=1)
        precisions.append(precise)
        distincts.append(distinct)
    # average plot for precisions
    # some confidence interval
    precisions = np.array(precisions)
    precisions[precisions>0] = 1  # turn precision matrix into probability of recall
    y = np.mean(precisions,axis=0)
    # bootstrapping Confidence Interval
    ci = []
    for sub in range(len(precisions[0])):
        rng = np.random.default_rng()
        res = bootstrap((precisions[:,sub],), np.mean, confidence_level=0.95,
                        random_state=rng, method='percentile')
        ci.append(y[sub]-res.confidence_interval.low)
    axes.flat[plot_n].plot(np.arange(1,len(precise)+1),y,color=colors[plot_n])
    # add asterisks (not used anymore)
    # for b in range(len(y)):
    #     if y[b]>baseline_upper[b]:
    #         axes.flat[plot_n].scatter([b+1], [y[b]+0.15], marker='*',color='red')
        # if y[b]<baseline_lower[b]:
        #     axes.flat[plot_n].scatter([b+1], [y[b]-0.1], marker='*',color='black')
    # plot upper baseline as a line
    axes.flat[plot_n].plot(np.arange(1,len(precise)+1), [baseline_upper]*(len(precise)),linestyle='--',color='black',alpha=0.7,linewidth=1)
    # plot
    ticks = np.arange(0,len(precise)+1,10)
    ticks[0] = 1
    axes.flat[plot_n].set_xticks(ticks)
    axes.flat[plot_n].fill_between(np.arange(1,len(precise)+1), (y-ci), (y+ci), color=colors[plot_n], alpha=.1)
    axes.flat[plot_n].set_xlabel("Event Number")
    axes.flat[plot_n].set_ylabel("Probability of Recall")
    if story_id == 'oregon':
        story_id = 'oregontrail'
    axes.flat[plot_n].set_title(story_id.capitalize())
    axes.flat[plot_n].spines[['right', 'top']].set_visible(False)
    axes.flat[plot_n].set_ylim(ymax=1)

    plot_n += 1
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'probRecall.svg'))
