# 50.038 Final Project

## /MUStARD
Dataset repo taken from: https://github.com/soujanyaporia/MUStARD

## /BERT
Go to the google colab link to view and run code: https://colab.research.google.com/drive/1XO7RJ_wLqxiFBstDuGd13xdTYu2-Ar-C?usp=sharing

To run locally, the same code can be found in BERT.py, but it runs much slower without a GPU.

## /svm
Contents
- `run.py`: basic svm implementation
- `results_run.py`: results of `run.py`
- Other results files: result of `train_svm.py`

Notes on dataset repo's svm implementation
- Defaults to using no modalities, yes bert: `config.py`
- Utterances are strings, which can't be used in svm. Text embeddings are numbers representing a string's meaning, pre-calculated by BERT (`bert-input.txt`), in the same order as `sarcasm.json`, and used as a featre: https://github.com/TaliaWang/50.038-FINAL-PROJECT/blob/e847a364bade9e3ef254ae1f57992b4fc5edafd7/MUStARD/data_loader.py#L45

## /video-features
Folder to play around with video features