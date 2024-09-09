import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

seg_data_path = './segmentation/alldata_run2.mat'  # the original path from Michelmann is './data-and-code-moment-by-moment-tracking/SegmentationData (Figure1)/alldata_run2.mat'

# load behavioral data containing word onsets and offsets
rep_results = pd.read_csv('./segmentation/rep_results.csv')

# panda to hold the boundary detection results
results_boundary = pd.DataFrame()
word_boundary_agreement = pd.DataFrame()

# these file contains all subjects (n = 205) by ms in story (450000). 0 represents no click, while 1 represents click.
# using Run2
all_subject_data_run1 = sio.loadmat(seg_data_path)['alldata_run2']  # actually using run2!!

# permutation test to find a threshold for significant events (5 second windows)
wsize = 5000
all_subject_data_run1 = all_subject_data_run1.reshape((205,int(all_subject_data_run1.shape[1]/wsize),wsize))
data_run1 = np.sum(all_subject_data_run1, axis=2) > 0
permut = []
for i in range(1000):
    data_run1_perm = np.sum(shuffle_along_axis(data_run1,1), axis=0)
    permut.extend(data_run1_perm)
plt.hist(permut)
threshold = np.percentile(permut,95)
bounds = np.sum(data_run1, axis=0) > threshold  # the time point indices where agreement is above threshold


agreement_run1 = sio.loadmat('./segmentation/agreement_run2.mat')['agreement_run2']  # using run2
results_boundary['agreement_run1'] = agreement_run1[0].tolist()
results_boundary['time (s)'] = np.arange(len(agreement_run1[0]))/1000


# find the peak POINT within the 5s windows
bounds_peak = np.zeros(450000)  # total number of time points
for bound in np.where(bounds)[0]:
    start, end = bound * wsize, (bound + 1) * wsize
    peak = np.argmax(agreement_run1[0][start:end])
    bounds_peak[start+peak] = 1
# add the boundary info to the excel sheet in the "peak" column. Boundaries are filled with 1s, and the rest is nan.
results_boundary['peak'] = agreement_run1[0]*bounds_peak
results_boundary.loc[results_boundary.peak==0,'peak']=np.nan

# plotting the boundaries before removal
plt.figure()
plt.plot(results_boundary['time (s)'],results_boundary['agreement_run1'])
plt.scatter(results_boundary['time (s)'],results_boundary['peak'],s=50,color='red')
plt.xlabel("Time in story (s)", fontsize = 15)
plt.ylabel("Boundary agreement", fontsize = 15)

# remove peaks that are too close
peak_position = np.where(bounds_peak)[0]
peak_keep = []
peak_rm = []
for i,interval in enumerate(np.diff(peak_position)):
    if interval>wsize:
        peak_keep.append(peak_position[i])
    else:
        # for two events that are too close, select one that has higher agreement
        idx = np.argmax([agreement_run1[0][peak_position[i]],
                        agreement_run1[0][peak_position[i + 1]]])
        peak_keep.append([peak_position[i],peak_position[i+1]][idx])
        peak_rm.append([peak_position[i],peak_position[i+1]][~bool(idx)])
peak_keep = np.array([x for x in sorted(list(set(peak_keep))) if x not in peak_rm])
# plotting peaks after removal
plt.figure()
plt.plot(results_boundary['time (s)'],results_boundary['agreement_run1'])
plt.scatter(results_boundary['time (s)'][peak_keep],results_boundary['agreement_run1'][peak_keep],s=50,color='red')
plt.xlabel("Time in story (s)", fontsize = 15)
plt.ylabel("Boundary agreement", fontsize = 15)

# add the boundary info to excel
onsets = rep_results['word_onsets']*1000
onsets = onsets.values
onsets = np.concatenate((onsets,[430885])) # final timestamp is appended to the end
peak_result = []
peaK_agreement = []
for word_n in range(len(onsets)-1):
    # if the current word contains a peak
    if any(onsets[word_n] < peak < onsets[word_n+1]
           for peak in peak_keep):
        peak_n = peak_keep[np.where([onsets[word_n] < peak < onsets[word_n+1] for peak in peak_keep])[0][0]]
        peak_result.append(1)
        # agg = np.mean([agreement_run1[0][round(onsets[word_n])],agreement_run1[0][round(onsets[word_n+1])]])
        agg = results_boundary['agreement_run1'][peak_n]
        peaK_agreement.append(agg)
    else:
        peak_result.append(np.nan)
        peaK_agreement.append(np.nan)
rep_results['boundary'] = peak_result
rep_results['agreement'] = peaK_agreement
rep_results.to_csv('rep_results.csv')

# print the story with the boundaries
bound_word_lvl = np.where(~rep_results['boundary'].isnull())[0]
bound_word_lvl = np.concatenate(([0], bound_word_lvl, [len(rep_results)]))
peak_word_lvl = np.concatenate(([0], peak_keep))
txt = []
for start,end,peak in zip(bound_word_lvl[0:-1], bound_word_lvl[1:],peak_word_lvl):
    txt.append(str(round(agreement_run1[0][peak],2))+'\n'+' '.join(rep_results['word'].tolist()[start:end]))
txt = '\n\n'.join(txt)
with open('./segmentation/pieman_bounds_run2_wsize5.txt', 'w') as f:
    f.write(txt)

"""
save all agreement data
"""
# add the boundary info to excel
onsets = rep_results['word_onsets'].values
onsets = np.concatenate((onsets,[430.885])) # final timestamp is appended to the end
agreements = []
for word_n in range(len(onsets)-1):
    st = onsets[word_n]
    end = onsets[word_n+1]
    agg = results_boundary.loc[(results_boundary['time (s)']>=st) & (results_boundary['time (s)']<=end),'agreement_run1'].mean()
    agreements.append(agg)
rep_results['agreement'] = agreements
rep_results.to_excel('event_segmentation_pieman.xlsx')