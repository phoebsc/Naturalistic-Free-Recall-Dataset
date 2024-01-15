This repository contains code used to produce the paper "The Narrative Memory Dataset: four stories, hundreds of participants, and high-fidelity transcriptions" by Omri Raccah, Phoebe Chen, David Poeppel, and Vy A. Vo.
The data in the project is at https://osf.io/h2pkv/. To reproduce the results, place the recall transcripts in respective folders such as /recall_transcript/recall_transcript_eyespy.
The code follows the processing steps in the paper "Geometric models reveal behavioural and neural signatures of transforming naturalistic experiences into episodic memories" by Andrew C. Heusser, Paxton C. Fitzpatrick, and Jeremy R. Manning., with certain modifications outlined in our paper.

The repository is organized as follows:

code
- A_topic_model.py : Python script for topic modeling (output saved in "result_models")
- B_HMM.py : Python script for event detection using Hidden Markov Models (output saved in "result_models")
- C_scoring.py : Python script for scoring the recalls (output saved in "result_models")
- D_listLearning.py : Python script for list learning analysis (figure 3)
- E_precisionPlot.py : Python script for generating precision plots (figure 2)
- F_check_text.py : Python script for text checking (output saved in "result_text")
folders
- eventSeg_helpers : Python package with support code for event segmentation
- result_models : Folder containing model results
- result_text : Folder containing text results
- ridge_utils : Python utility scripts for ridge regression analysis
- sherlock_helpers : Python package with support code for 'Sherlock' analyses
- story_transcript : Transcripts of stories used in analyses
