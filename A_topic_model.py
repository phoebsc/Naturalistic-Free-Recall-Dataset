import hypertools as hyp
import os
import sys
sys.path.append(os.getcwd())
from sherlock_helpers.functions import *
from sherlock_helpers.constants import *
from sherlock_helpers.scoring import *

DATA_DIR = 'result_models'
story_ids = ['pieman','eyespy','oregon','baseball']
def get_story_text(corpus, bounds, winds):
    start_window, end_window = winds
    windows = bounds[start_window:end_window]
    start = min([s for s,e in windows])
    end = max([e for s,e in windows])
    return ' '.join(corpus[start:end])
"""
using sliding window to segment story
"""
# helper function to replace codes
def clean_text(text):
    unwanted_char = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
    text = "".join([(" " if n in unwanted_char else n) for n in text if n not in unwanted_char])
    return text

# function to generate
def topic_model(story_id, n_topics, VIDEO_WSIZE, step_size):
    SEMANTIC_PARAMS = {
        'model': 'LatentDirichletAllocation',
        'params': {
            'n_components': n_topics,
            'learning_method': 'batch',
            'random_state': 0
        }
    }

    story_path = './story_transcript/%s_transcript.txt' % story_id
    with open(story_path, encoding="utf8") as f:
        story = f.read().strip()
    # create overlapping windows
    # stop_words = set(stopwords.words('english'))
    story_fmt = format_text(story).replace('\n',' ').replace('.','').strip()
    story_fmt = " ".join(story_fmt.split()).split(' ')

    sub_story, bounds_story = parse_windows_reg(story_fmt, VIDEO_WSIZE, step_size)
    # create the story model (77x100)
    story_model = hyp.tools.format_data(sub_story,
                                          vectorizer=VECTORIZER_PARAMS,
                                          semantic=SEMANTIC_PARAMS,
                                          corpus=sub_story)[0]

    """
    then used the topic model already trained on the episode scenes 
    to compute the most probable topic proportions for each sliding window.
    """
    recall_paths = [os.path.join('./recall_transcript/recall_transcript_%s/' % story_id,x)
                    for x in os.listdir('./recall_transcript/recall_transcript_%s/' % story_id)]
    recall_models = []
    for recall_path in recall_paths:
        with open(recall_path, encoding="utf8",errors='ignore') as f:
            recall = f.read()

        # create windows of n words
        recall_fmt = format_text(recall).replace('\n', ' ').replace('.', '').strip()
        recall_fmt = clean_text(recall_fmt)
        recall_fmt = recall_fmt.replace('im done','').replace('i am done','')
        recall_fmt = " ".join(recall_fmt.split()).split(' ')

        # use avg number of words in the event as recall window
        # RECALL_WSIZE = int(np.mean([len(s.split(' ')) for s in story_fmt]))
        RECALL_WSIZE = VIDEO_WSIZE
        sub_recall, bounds_recall = parse_windows_reg(recall_fmt, RECALL_WSIZE,step_size)
        # create the story model (77x100)
        recall_model = hyp.tools.format_data(sub_recall,
                                              vectorizer=VECTORIZER_PARAMS,
                                              semantic=SEMANTIC_PARAMS,
                                              corpus=sub_story)[0]
        recall_models.append(recall_model)
    # save the models
    n_topics = SEMANTIC_PARAMS['params'].get('n_components')
    np.save(os.path.join(os.getcwd(),DATA_DIR,f'%s_t{n_topics}_v{VIDEO_WSIZE}_r{RECALL_WSIZE}_s{step_size}' % story_id),
            [story_model, recall_models,recall_paths])

n_topics = 40
video_size = 55
step_size = 21
for story_id in story_ids:
    topic_model(story_id, n_topics, video_size, step_size)


