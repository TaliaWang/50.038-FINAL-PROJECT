import pandas as pd
from sklearn.metrics import classification_report
import jsonlines
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

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

# define a new scaler: 
x_scaler = MinMaxScaler()

# fit the normalization on the training set: 
x_scaler.fit(train_input)

# then create new and normalized training/test sets: 
X_train_norm = x_scaler.transform(train_input)

model = LogisticRegression(C=1.0, multi_class='auto', solver='lbfgs') 
model.fit(train_input, train_output) # Training the model

# test
test_input = embeddings[train_size:train_size+test_size]
test_output = df.iloc[train_size:train_size+test_size]["sarcasm"].astype(int)

X_test_norm = x_scaler.transform(test_input)

predicted = model.predict(X_test_norm) # Predicting labels for our test set using trained model

# results
classification_report(test_output, predicted, output_dict=True, digits=3)