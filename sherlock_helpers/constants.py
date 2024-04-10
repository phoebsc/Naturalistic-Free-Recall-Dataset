from pathlib import Path
import numpy as np

# paths for loading data and saving figures

# video sliding window length (in annotations)
VIDEO_WSIZE = 10  # modified

# recall sliding window length (in sentences)
RECALL_WSIZE = 5  # modified

# text vectorizer parameters
VECTORIZER_PARAMS = {
    'model': 'CountVectorizer',
    'params': {
        'stop_words': 'english'
    }
}

# topic model parameters
SEMANTIC_PARAMS = {
    'model': 'LatentDirichletAllocation',
    'params': {
        'n_components': 30,  # modified
        'learning_method': 'batch',
        'random_state': 0
    }
}