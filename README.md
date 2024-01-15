This repository contains code used to produce the paper 
The data in the project is at https://osf.io/h2pkv/. To reproduce the results, place the recall transcripts in folder /recall_transcript/recall_transcript_eyespy.
The code follows the processing steps in the paper "Geometric models reveal behavioural and neural signatures of transforming naturalistic experiences into episodic memories" by Andrew C. Heusser, Paxton C. Fitzpatrick, and Jeremy R. Manning., with certain modifications outlined in our paper.

The repository is organized as follows:

code
- A_topic_model.py : Python script for topic modeling
- B_HMM.py : Python script for Hidden Markov Models
- C_scoring.py : Python script for scoring algorithms
- D_listLearning.py : Python script for list learning analysis
- E_precisionPlot.py : Python script for generating precision plots
- F_check_text.py : Python script for text checking
folders
- eventSeg_helpers : Python package with support code for event segmentation
- result_models : Folder containing model results
- result_text : Folder containing textual results
- ridge_utils : Python utility scripts for ridge regression analysis
- sherlock_helpers : Python package with support code for 'Sherlock' analyses
- story_transcript : Transcripts of stories used in analyses
