"""
Loads the event segmentations for both story and recall. Exports an Excel .xlsx file that contains the original story
event text that best matches the recall text.
"""

# loop over subjects
import numpy as np
from scipy.spatial.distance import cdist
import more_itertools as mit
import os
import sys
try:
    os.chdir('./topic_models')
except:
    pass
sys.path.append(os.getcwd())
from sherlock_helpers.functions import format_text, parse_windows_reg
from sherlock_helpers.scoring import precise_matches_mat
import pandas as pd

def get_story_text(corpus, bounds, winds):
    start_window, end_window = winds
    end_window = end_window+2
    windows = bounds[start_window:end_window]
    if end_window>=len(bounds):
        end = len(corpus)
    else:
        end = max([s for s, e in windows])
    start = min([s for s,e in windows])
    return ' '.join(corpus[start:end])


# load HMM info
data_dir = 'result_models'
txt_dir = 'result_text'
############################3
story_id='baseball'
n_topics = 40
story_size = 55
step_size = 21
subfolder = f'{story_id}_t{n_topics}_v{story_size}_r{story_size}_s{step_size}'
#########################3#### if os.path.isfile(os.path.join(txt_dir,subfolder,'story.txt')):
#     return None
story_events_times = np.load(os.path.join(data_dir,subfolder,'video_event_times.npy'), allow_pickle=True)  # TODO: refactor instances of 'video*.file_ext'
recall_events_times = np.load(os.path.join(data_dir,subfolder,'recall_event_times.npy'), allow_pickle=True)
story_events = np.load(os.path.join(data_dir,subfolder,'video_events.npy'), allow_pickle=True)  # TODO: refactor instances of 'video*.file_ext'
recall_events = np.load(os.path.join(data_dir,subfolder,'recall_events.npy'), allow_pickle=True)
# event_mappings = np.load(os.path.join(data_dir,subfolder,'labels.npy'), allow_pickle=True)
story_model, recall_models, recall_ids = np.load(os.path.join(os.getcwd(),data_dir,subfolder+'.npy'),
                                     allow_pickle=True)
recall_ids = [os.path.join('./recall_transcript/recall_transcript_%s' % story_id,os.path.basename(i)) for i in recall_ids]

# create folder to save
# create folder to save data
isExist = os.path.exists(os.path.join(txt_dir,subfolder))
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(os.path.join(txt_dir,subfolder))
    print("The new directory is created!")
# story - load original texts
story_path = './%s_transcript.txt' % story_id
with open(story_path, encoding="utf8") as f:
    story = f.read().strip()
# story_fmt = format_text(story).replace('.','').split(' ')
story_fmt = format_text(story).replace('\n', ' ').replace('.', '').strip()
story_fmt = " ".join(story_fmt.split()).split(' ')
story_size = int(subfolder.split('_')[2][1:])
step_size = int(subfolder.split('_')[4][1:])
sub_story, bounds_story = parse_windows_reg(story_fmt, story_size, step_size)
# check whether the windows match
assert len(sub_story) == len(story_model)
# create segmented story
story_seg = [str(i+1)+'. '+get_story_text(story_fmt, bounds_story, story_events_times[i]) for i in range(len(story_events_times))]
story_seg = '\n\n'.join(story_seg)
with open(os.path.join(txt_dir,subfolder,'story.txt'), 'w') as f:
    f.write(story_seg)

# recalls - original texts
writer = pd.ExcelWriter(os.path.join(txt_dir, subfolder,'matched.xlsx'))
for idx, recall_id in enumerate(recall_ids):
    with open(recall_id) as f:
        recall = f.read()
        # create windows of n words
    # recall_fmt = [sent.split(' ') for sent in format_text(recall).split('\n')]
    # recall_fmt = [item for sublist in recall_fmt for item in sublist]
    recall_fmt = format_text(recall).replace('\n', ' ').replace('.', '').strip()
    recall_fmt = " ".join(recall_fmt.split()).split(' ')
    step_size = int(subfolder.split('_')[4][1:])
    sub_recall, bounds_recall = parse_windows_reg(recall_fmt, story_size, step_size)
    assert len(sub_recall) == len(recall_models[idx])
    # create segmented recall
    recall_seg = [get_story_text(recall_fmt, bounds_recall, recall_events_times[idx][i])
                  for i in range(len(recall_events_times[idx]))]
    # save text
    recall_seg = '\n\n'.join(recall_seg)
    with open(os.path.join(txt_dir, subfolder, os.path.basename(recall_ids[idx])[0:5]+'.txt'), 'w') as f:
        f.write(recall_seg)

    """
    save the matched text (using the precision matrix)
    """
    corrmat = 1 - cdist(story_events, recall_events[idx], 'correlation')  # this is the correlation matrix
    precise_mat = precise_matches_mat(corrmat,'recall')
    story_segs = [get_story_text(story_fmt, bounds_story, story_events_times[i]) for i in np.where(precise_mat > 0)[0]]
    # recall_segs = [get_story_text(recall_fmt, bounds_recall,(j,j+1)) for j in np.where(precise_mat>0)[1]]
    matched = pd.DataFrame(data=dict(story_id=np.where(precise_mat>0)[0],story_segs=story_segs,
                           recall_id=np.where(precise_mat>0)[1]))
    # building matched df
    final_matched = pd.DataFrame(columns = ['story_id','story_segs','recall_id','recall_segs'])
    n=0
    for key, val in matched.groupby('story_id').groups.items():
        recall_idx = matched.loc[val,'recall_id'].values
        for seg in mit.consecutive_groups(recall_idx):
            s = list(seg)
            txt_recall = ' '.join(recall_seg.split('\n\n')[s[0]:(s[-1]+1)])
            final_matched.loc[n] = [key]+[get_story_text(story_fmt, bounds_story,
                                                         story_events_times[key])]+[(s[0],s[-1]+1)]+[txt_recall]
            n+=1
    # write dataframe to excel sheet
    final_matched.to_excel(writer, os.path.basename(recall_ids[idx])[0:5])
    # save the excel file
writer.save()

