import numpy as np
# for each row, record the largest number
def precise_matches_mat(corr_mat, axis):
    if axis=='story':
        temp = np.zeros(corr_mat.shape)
        for r in range(corr_mat.shape[0]):
            temp[r, np.nanargmax(corr_mat, 1)[r]] = np.nanmax(corr_mat, 1)[r]
    elif axis=='recall':
        temp = np.zeros(corr_mat.shape)
        try:
            for r in range(corr_mat.shape[1]):
                temp[np.nanargmax(corr_mat, 0)[r], r] = np.nanmax(corr_mat, 0)[r]
        except:  # in case on NaN columns
            for r in range(corr_mat.shape[1]-1):
                temp[np.nanargmax(corr_mat[:,0:-1], 0)[r], r] = np.nanmax(corr_mat[:,0:-1], 0)[r]
    return temp


# after zscore along the columns, recor largest number for each row
def distinct_matches_mat(corr_mat, axis):
    from scipy.stats import zscore
    if axis=='story':
        temp = np.zeros(corr_mat.shape)
        std = zscore(corr_mat, 0)
        for r in range(temp.shape[0]):
            temp[r, np.argmax(std, 1)[r]] = np.max(std, 1)[r]

    elif axis=='recall':
        temp = np.zeros(corr_mat.shape)
        std = zscore(corr_mat, 0, nan_policy='omit')
        try:
            for r in range(temp.shape[1]):
                temp[np.nanargmax(std, 0)[r], r] = np.nanmax(std, 0)[r]
        except:  # in case on NaN columns
            for r in range(temp.shape[1]-1):
                temp[np.nanargmax(std[:,0:-1], 0)[r], r] = np.nanmax(std[:,0:-1], 0)[r]

    return temp


def unique_match_mat(corr_mat):
    temp = find_unique_match_events(corr_mat)
    return temp

# compute unique matched events
# corrmat is a matrix with size num_chapter_events X num_recall_events
def find_unique_match_events(corrmat):
    unique_match_weighted_sim = np.copy(corrmat)

    # for each event in the chapter
    for event in range(corrmat.shape[0]):

        # loop over each event in the recall until you find the recall event
        # that is the best match for the chapter event and the chapter event is the best match for the recall event
        match = 0
        for recall_event in range(unique_match_weighted_sim.shape[1]):
            try:
                # get the best recall event for this chapter event
                best_recall_match = np.nanargmax(unique_match_weighted_sim[event, :])

                # get the best chapter event for the previously found best recall event
                best_book_match = np.nanargmax(unique_match_weighted_sim[:, best_recall_match])

                if best_book_match != event:
                    unique_match_weighted_sim[event, best_recall_match] = 0
                else:
                    match = 1
                    # if the two best events match, then zero out every other recall event for the same chapter event
                    for e in range(unique_match_weighted_sim.shape[1]):
                        if e != best_recall_match:
                            unique_match_weighted_sim[event, e] = 0
                    break
            except:
                unique_match_weighted_sim[event, :] = 0
        # if there was no match for this chapter event, then zero out all of the columns for it
        if match == 0:
            unique_match_weighted_sim[event, :] = 0
    return unique_match_weighted_sim