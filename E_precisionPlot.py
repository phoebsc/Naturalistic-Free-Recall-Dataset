
from scipy.spatial.distance import cdist
import more_itertools as mit
import os
import sys
try:
    os.chdir('./topic_models')
except:
    pass
sys.path.append(os.getcwd())
from sherlock_helpers.functions import *
from sherlock_helpers.constants import *
from sherlock_helpers.scoring import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap
plt.rcParams.update({'font.size': 13})
import pandas as pd
story_ids = ['pieman','eyespy','oregon','baseball']
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
FIG_DIR = 'final_plots'
plot_n=0
for story_id in story_ids:
    DATA_DIR = 'result_models'
    IMG_DIR = 'result_plots'
    subfolder = '%s_t40_v55_r55_s21' % story_id
    video_events = np.load(os.path.join(DATA_DIR,subfolder,'video_events.npy'), allow_pickle=True)
    recall_events = np.load(os.path.join(DATA_DIR,subfolder,'recall_events.npy'), allow_pickle=True)
    event_mappings = np.load(os.path.join(DATA_DIR,subfolder,'labels.npy'), allow_pickle=True)
    _, _, recall_ids = np.load(os.path.join(DATA_DIR,subfolder+'.npy'),
                                         allow_pickle=True)
    baseline_upper = np.load(os.path.join(DATA_DIR, subfolder,
                       [x for x in os.listdir(os.path.join(DATA_DIR, subfolder)) if 'baseline_upper' in x][0]))
    baseline_lower = np.load(os.path.join(DATA_DIR, subfolder,
                       [x for x in os.listdir(os.path.join(DATA_DIR, subfolder)) if 'baseline_lower' in x][0]))

    if story_id == 'pieman':
        video_events = video_events[0:24]
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
        corrmat = 1 - cdist(video_events, recall_event, 'correlation')  # this is the correlation matrix
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
    axes.flat[plot_n].plot(np.arange(1,len(precise)+1),y)
    # add asterisks (not used anymore)
    # for b in range(len(y)):
    #     if y[b]>baseline_upper[b]:
    #         axes.flat[plot_n].scatter([b+1], [y[b]+0.15], marker='*',color='red')
        # if y[b]<baseline_lower[b]:
        #     axes.flat[plot_n].scatter([b+1], [y[b]-0.1], marker='*',color='black')
    # plot upper baseline as a line
    axes.flat[plot_n].plot(np.arange(1,len(precise)+1), baseline_upper,linestyle='--',color='black',alpha=0.7,linewidth=1)
    # plot
    ticks = np.arange(0,len(precise)+1,10)
    ticks[0] = 1
    axes.flat[plot_n].set_xticks(ticks)
    axes.flat[plot_n].fill_between(np.arange(1,len(precise)+1), (y-ci), (y+ci), color='b', alpha=.1)
    axes.flat[plot_n].set_xlabel("event number")
    axes.flat[plot_n].set_ylabel("probability of recall")
    if story_id == 'oregon':
        story_id = 'oregontrail'
    axes.flat[plot_n].set_title(story_id)
    axes.flat[plot_n].spines[['right', 'top']].set_visible(False)
    axes.flat[plot_n].set_ylim(ymax=1)

    plot_n += 1
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR,'probRecall.png'))










