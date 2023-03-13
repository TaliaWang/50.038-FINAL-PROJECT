import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
import jsonlines
import numpy as np

CLS_TOKEN_INDEX = 0

# partition data
df = pd.read_json("../MUStARD/data/sarcasm_data.json")
df = df.transpose()

embeddings = []

with jsonlines.open("../MUStARD/data/bert-output.jsonl") as utterances:
    for utterance in utterances:
        features = utterance["features"][CLS_TOKEN_INDEX]
        bert_embedding_target = np.mean([np.array(features["layers"][layer]["values"])
                                            for layer in range(4)], axis=0)
        embeddings.append(np.copy(bert_embedding_target))

test_size = 50
train_size = len(df) - test_size

# train
train_input = embeddings[:train_size]
train_output = df.iloc[:train_size]["sarcasm"].astype(int)
clf = svm.SVC()
clf.fit(train_input, train_output)

# test
test_input = embeddings[train_size:train_size+test_size]
test_output = df.iloc[train_size:train_size+test_size]["sarcasm"].astype(int)
predicted = clf.predict(test_input)

# results
print(classification_report(test_output, predicted, output_dict=True, digits=3))