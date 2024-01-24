from scipy.spatial.distance import cdist
import more_itertools as mit
import os
import sys
sys.path.append(os.getcwd())
from sherlock_helpers.scoring import *
import matplotlib.pyplot as plt
import numpy as np

def scoring_func(story_id):
    DATA_DIR = 'result_models'
    IMG_DIR = 'result_plots'
    subfolder = '%s_t40_v55_r55_s21' % story_id
    video_events = np.load(os.path.join(DATA_DIR,subfolder,'video_events.npy'), allow_pickle=True)
    recall_events = np.load(os.path.join(DATA_DIR,subfolder,'recall_events.npy'), allow_pickle=True)
    event_mappings = np.load(os.path.join(DATA_DIR,subfolder,'labels.npy'), allow_pickle=True)
    _, _, recall_ids = np.load(os.path.join(DATA_DIR,subfolder+'.npy'),
                                         allow_pickle=True)
    if story_id=='pieman':
        video_events = video_events[0:24]  # remove the final event (which is too short) of pieman

    """
    scoring all recalls
    """
    ##
    n = 0
    plot_n = 1
    precisions = []
    distincts = []
    method = 'recall'

    fig1, axes1 = plt.subplots(3, 6,figsize=(20, 20))
    fig2, axes2 = plt.subplots(3, 3,figsize=(20, 20))
    fig3, axes3 = plt.subplots(3, 3,figsize=(20, 20))
    for recall_event in recall_events:
        corrmat = 1 - cdist(video_events, recall_event, 'correlation')  # this is the correlation matrix
        # plotting the story-recall matrix
        im = axes1.flat[2*n].imshow(corrmat, cmap=plt.get_cmap("Greys"))
        clim=im.properties()['clim']
        axes1.flat[2*n].set_title(os.path.basename(recall_ids[n])[0:5])
        # computing scores
        precise_mat = precise_matches_mat(corrmat,method)
        precise = np.max(precise_mat,axis=1)
        distinct_mat = distinct_matches_mat(corrmat,method)
        distinct = np.max(distinct_mat, axis=1)
        precisions.append(precise)
        distincts.append(distinct)
        # plotting precise mat
        im = axes1.flat[2*n+1].imshow(precise_mat, cmap=plt.get_cmap("Greys"))
        clim = im.properties()['clim']
        # plotting the scores
        axes2.flat[n].plot(np.arange(1, 1+len(precise)),precise, '-o')
        axes3.flat[n].plot(np.arange(1, 1 + len(precise)), distinct, '-o')
        axes2.flat[n].set_xticks(np.arange(1,1+len(precise)))
        axes3.flat[n].set_xticks(np.arange(1,1+len(precise)))

        axes2.flat[n].set_xlabel('event number')
        axes3.flat[n].set_xlabel('event number')

        axes2.flat[n].set_title('precision - '+os.path.basename(recall_ids[n])[0:5])
        axes3.flat[n].set_title('distinct - '+os.path.basename(recall_ids[n])[0:5])
        n+=1
        if n>8:
            fig1.savefig(os.path.join(IMG_DIR,subfolder,'matched'+str(plot_n)+'.png'))
            fig2.savefig(os.path.join(IMG_DIR,subfolder,'precise'+str(plot_n)+'.png'))
            fig3.savefig(os.path.join(IMG_DIR,subfolder,'distinct'+str(plot_n)+'.png'))
            fig1, axes1 = plt.subplots(3, 6, figsize=(20, 20))
            fig2, axes2 = plt.subplots(3, 3, figsize=(20, 20))
            fig3, axes3 = plt.subplots(3, 3, figsize=(20, 20))
            n = 0
            plot_n+=1

    # average plot for precisions
    # some confidence interval
    precisions = np.array(precisions)
    y = np.mean(precisions,axis=0)
    ci = 1.96 * np.std(precisions,axis=0)/np.sqrt(len(precisions))
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.plot(np.arange(1,len(precise)+1),y)
    ax.set_xticks(np.arange(1,len(precise)+1))
    ax.fill_between(np.arange(1,len(precise)+1), (y-ci), (y+ci), color='b', alpha=.1)
    ax.set_title('mean precision 95% CI')
    fig.savefig(os.path.join(IMG_DIR,subfolder,'mean_precise.png'))

    y = np.mean(distincts,axis=0)
    ci = 1.96 * np.std(distincts,axis=0)/np.sqrt(len(distincts))
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.plot(np.arange(1,1+len(precise)),y)
    ax.set_xticks(np.arange(1,1+len(precise)))
    ax.fill_between(np.arange(1,1+len(precise)), (y-ci), (y+ci), color='b', alpha=.1)
    ax.set_title('mean distinctness 95% CI')
    fig.savefig(os.path.join(IMG_DIR,subfolder,'mean_distinct.png'))

    """
    save precision as an array
    """
    np.save(os.path.join(DATA_DIR,subfolder,'precision_array'),  np.array(precisions))

story_ids = ['pieman','eyespy','oregon','baseball']
for story_id in story_ids:
    scoring_func(story_id)








