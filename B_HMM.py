"""
This script will load a topic model trained on a story and apply a Hidden Markov Model to the topic vectors to segment
it into events. To do this, it searches over a range of event numbers $k$ and chooses the value of $k$ that maximizes
the Wasserstein distance of topic weights within an event versus across events. It will then generate some plots to show
the output. Finally, the best k value is used to perform the event segmentation on the story.

The same procedure is then repeated to segment the story recalls.
Finally, each recall event is matched to one of the original story events. This is done slightly different than in
Heusser et al. 2021.
# TODO: add the detail about why the event matching is different
This is repeated for each participant. Then, the average recall per original story event is computed across participants.
Several output files are saved at each stage, and can be viewed in the output folder result_models.
"""

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pickle
import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, pearsonr
sys.path.append(os.getcwd())
from sherlock_helpers.functions import create_diag_mask
from sherlock_helpers.scoring import precise_matches_mat
from eventSeg_helpers import event  # TODO: figure out where this import comes from!

def reduce_model(m, ev):
    """Reduce a model based on event labels"""
    w = (np.round(ev.segments_[0]) == 1).astype(bool)
    return np.array([m[wi, :].mean(axis=0) for wi in w.T])

def HMM_func(story_id):
    #######################
    n_topics = 40
    video_size = 55
    step_size = 21
    subfolder = f'{story_id}_t{n_topics}_v{video_size}_r{video_size}_s{step_size}'
    data_dir = 'result_models'
    img_dir = 'result_plots'
    #######################

    video_model, recall_models, recall_ids = np.load(os.path.join(os.getcwd(),data_dir,subfolder+'.npy'),
                                         allow_pickle=True)
    # create folder to save data
    isExist = os.path.exists(os.path.join(data_dir,subfolder))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(data_dir,subfolder))
        print("The new directory is created!")
    isExist = os.path.exists(os.path.join(img_dir,subfolder))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(img_dir,subfolder))
        print("The new directory is created!")

    """
    finding the optimal k for story
    """
    n_events = np.arange(2,min(50,len(video_model)-1))
    wd = np.zeros(len(n_events))
    corrmat = np.corrcoef(video_model)

    for i, events in enumerate(tqdm(n_events, leave=False)):
        ev = event.EventSegment(events)
        ev.fit(video_model)

        i1, i2 = np.where(np.round(ev.segments_[0]) == 1)
        w = np.zeros_like(ev.segments_[0])
        w[i1, i2] = 1
        mask = np.dot(w, w.T).astype(bool)

        # Create mask such that the maximum temporal distance
        # for within and across correlations is the same
        local_mask = create_diag_mask(mask)

        within_vals = corrmat[mask * local_mask]
        within_vals = within_vals[~np.isnan(within_vals)]
        across_vals = corrmat[~mask * local_mask]
        across_vals = across_vals[~np.isnan(across_vals)]
        # within_vals = np.reshape(corrmat[mask * local_mask], (-1, 1))
        # across_vals = np.reshape(corrmat[~mask * local_mask], (-1, 1))
        wd[i] = wasserstein_distance(within_vals, across_vals)

    """
    plot the distance
    """
    plt.figure()
    plt.plot(n_events, wd)
    maxk_video = n_events[np.argmax(wd)]
    plt.ylabel('Wasserstein distance')
    plt.xlabel('Number of events ($K$)')
    plt.title(f'Video: optimal $K$ = {maxk_video}')
    plt.savefig(os.path.join(img_dir,subfolder,'storyk.png'))
    # plt.show()

    plt.figure()
    plt.imshow(corrmat,cmap=plt.get_cmap("Greys"))
    plt.title('self-corr')
    plt.savefig(os.path.join(img_dir,subfolder,'storycorr.png'))

    """
    fitting the model to the story (using maxk_video, i.e. the optimal # of events)
    """
    ev = event.EventSegment(maxk_video)
    ev.fit(video_model)
    video_events = reduce_model(video_model, ev)

    video_event_times = []
    for s in ev.segments_[0].T:
        tp = np.where(np.round(s) == 1)[0]
        video_event_times.append((tp[0], tp[-1]))

    # save story stuff
    # TODO: refactor this to replace 'video' with 'story' in the filenames. Will also need to rename files in the repo...
    np.save(os.path.join(data_dir,subfolder,'video_events'), video_events)
    np.save(os.path.join(data_dir,subfolder,'video_event_times'), video_event_times)
    with open(os.path.join(data_dir,subfolder,'video_eventseg_models'),'wb') as f:
        pickle.dump(ev, f)
    """
    finding the optimal k for recall
    """
    # fig, axes = plt.subplots(5,7,figsize=(20,20))
    # fig2, axes2 = plt.subplots(5,7,figsize=(20,20))
    maxk = []
    n=0
    for recall_model in recall_models:
        n_events = np.arange(2,min(35,len(recall_model)-1))
        wd = np.zeros(len(n_events))
        corrmat = np.corrcoef(recall_model)
        for i, events in enumerate(tqdm(n_events, leave=False)):
            if recall_model.shape[0]<=events:  # if the number of recall windows is less than # of events
                wd[i] = 0
            else:
                ev = event.EventSegment(events)
                ev.fit(recall_model)

                i1, i2 = np.where(np.round(ev.segments_[0]) == 1)
                w = np.zeros_like(ev.segments_[0])
                w[i1, i2] = 1
                mask = np.dot(w, w.T).astype(bool)
                # Create mask such that the maximum temporal distance
                # for within and across correlations is the same
                local_mask = create_diag_mask(mask)
                within_vals = corrmat[mask * local_mask]
                within_vals = within_vals[~np.isnan(within_vals)]
                across_vals = corrmat[~mask * local_mask]
                across_vals = across_vals[~np.isnan(across_vals)]

                # within_vals = np.reshape(corrmat[mask * local_mask], (-1, 1))
                # across_vals = np.reshape(corrmat[~mask * local_mask], (-1, 1))
                wd[i] = wasserstein_distance(within_vals, across_vals)
        """
        plot the distance
        """
        fig, axes = plt.subplots(1, 2,figsize=(10,4))
        axes[0].plot(n_events, wd)
        maxk_recall = n_events[np.argmax(wd)]
        maxk.append(maxk_recall)
        axes[0].set_ylabel('Wasserstein distance')
        axes[0].set_title(f'$K$ = {maxk_recall}, '+os.path.basename(recall_ids[n])[0:5])
        axes[1].imshow(corrmat,cmap=plt.get_cmap("Greys"))
        fig.savefig(os.path.join(img_dir, subfolder, os.path.basename(recall_ids[n])[0:5]+'_recallk.png'))
        """
        write down the k just in case
        """
        print(n,os.path.basename(recall_ids[n])[0:5],maxk)
        n += 1


    """
    fitting back to the recall models
    """
    recall_events = []
    recall_event_times = []
    recall_eventseg_models = []

    for i, k in enumerate(maxk):
        ev = event.EventSegment(k)
        ev.fit(recall_models[i])
        m = reduce_model(recall_models[i], ev)
        recall_events.append(m)
        recall_times = []
        for s in ev.segments_[0].T:
            tp = np.where(np.round(s) == 1)[0]
            recall_times.append((tp[0], tp[-1]))

        recall_event_times.append(recall_times)
        recall_eventseg_models.append(ev)

    """
    save average recall: for each recall, find the matching event and only keep the vectors of those events. 
    and then average across participants
    """

    # average recall
    method = 'recall'
    matches = []
    for i,r in enumerate(recall_events):
        corrmat = 1 - cdist(video_events, r, 'correlation')
        precise_mat = precise_matches_mat(corrmat, method)
        match = []
        for column in precise_mat.T:
            if np.sum(column)==0:
                match.append(np.nan)
            else:
                match.append(np.argmax(column))
        matches.append(match)

    matches = [list(m) for m in matches]
    avg_recalls = [[] for _ in video_events]
    for match, r in zip(matches, recall_events):
        for i, m in enumerate(match):
            try:
                avg_recalls[m].append(r[i, :])
            except:
                continue

    avg_recall_events = np.array(list(map(lambda r: np.nanmean(r, 0) if len(r) > 0 else np.zeros((n_topics,)), avg_recalls)))

    # save
    np.save(os.path.join(data_dir,subfolder,'avg_recall_events'), avg_recall_events)
    np.save(os.path.join(data_dir,subfolder,'labels'), np.array(matches, dtype=object), allow_pickle=True)
    np.save(os.path.join(data_dir,subfolder,'recall_events'),  np.array(recall_events, dtype=object), allow_pickle=True)
    np.save(os.path.join(data_dir,subfolder,'recall_event_times'), np.array(recall_event_times, dtype=object), allow_pickle=True)
    with open(os.path.join(data_dir,subfolder,'recall_eventseg_models'),'wb') as f:
        pickle.dump(recall_eventseg_models, f)

story_ids = ['pieman','eyespy','oregon','baseball']
for story_id in story_ids:
    HMM_func(story_id)
