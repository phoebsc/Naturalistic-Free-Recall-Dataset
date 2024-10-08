# Narrative Free Recall Dataset - Processing Code

This repository contains code used to reproduce analyses in the paper "The Narrative Free Recall Dataset: four stories, hundreds of participants, and high-fidelity transcriptions" by Omri Raccah, Phoebe Chen, David Poeppel, and Vy A. Vo.
The data in the project is available at https://osf.io/h2pkv/. To reproduce the results, place the recall transcripts in respective folders such as `/recall_transcript/recall_transcript_eyespy`.

The code borrows heavily from the processing steps in the published paper ["Geometric models reveal behavioural and neural signatures of transforming naturalistic experiences into episodic memories"](https://www.nature.com/articles/s41562-021-01051-6.epdf?sharing_token=zBNF7ExvsNAn6dwRV2wbatRgN0jAjWel9jnR3ZoTv0Os85t-vR-u-Efaty0-uoqOJVLSCaVoppMqs8h0fibLcqGN8-6I_NPhCJMoHMR5VvrNcBfBoco7C6Yp3vJJfeQhUOvYBnwv3BSjY0N1-ytdd_S-DhUyYmokmB3dfE-NX_Q%3D), with certain modifications outlined in our paper.

The repository is organized as follows:

## code
- A_topic_model.py : Python script for topic modeling (output saved in "result_models")
- B_HMM.py : Python script for event detection using Hidden Markov Models (output saved in "result_models")
- C_scoring.py : Python script for scoring the recalls (output saved in "result_models")
- D_listLearning.py : Python script for list learning analysis (figure 3)
- E_ProbRecallPlot.py : Python script for generating precision plots (figure 2)
- F_check_text.py : Python script for text checking (output saved in "result_text")
- G_semantic_centrality.py and G_semantic_centrality_LMM.Rmd: python and R script for generating the semantic network plot and semantic centrality LMM analysis
  
## folders
- sherlock_helpers : Python package with support code for 'Sherlock' analyses
- result_models : Folder containing model results
- result_text : Folder containing text results
- story_transcript : Original transcripts of stories used in analyses
- eventSeg_helpers : contains the exact code copied from brainiak.eventseg.event from the Python software [Brainiak](https://brainiak.org/)
- segmentation: code for extracting event boundaries from the annotation data of Pieman. The attached data (agreement_run2.mat and alldata_run2.mat) is copied from the [publicly available dataset](https://doi.org/10.5281/zenodo.5071942) associated with [Michelmann et al. 2021](https://www.nature.com/articles/s41562-021-01051-6.epdf?sharing_token=zBNF7ExvsNAn6dwRV2wbatRgN0jAjWel9jnR3ZoTv0Os85t-vR-u-Efaty0-uoqOJVLSCaVoppMqs8h0fibLcqGN8-6I_NPhCJMoHMR5VvrNcBfBoco7C6Yp3vJJfeQhUOvYBnwv3BSjY0N1-ytdd_S-DhUyYmokmB3dfE-NX_Q%3D).

## Python Version
This project requires Python `3.9.0`. You can check your Python version by running:

```bash
python --version
```

## Installation

### 1. Clone the Repository
First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/phoebsc/Naturalistic-Free-Recall-Dataset.git
cd yourrepository
```

### 2. Create a Virtual Environment (Optional but Recommended)
It's recommended to use a virtual environment to manage your project's dependencies. You can create one using `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies
Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

## Usage
After installation, you can run the scripts in alphabetical order to replicate results of the paper using for example:

```bash
python A_topic_model.py
```
