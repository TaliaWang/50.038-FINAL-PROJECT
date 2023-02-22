# 50.038 Final Project

## /MUStARD
Dataset repo taken from: https://github.com/soujanyaporia/MUStARD

## /BERT
Folder to play around with BERT model using sarcasm_data.json text

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