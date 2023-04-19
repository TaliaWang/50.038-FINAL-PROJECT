
import pandas as pd
from sklearn.metrics import classification_report
import jsonlines
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from numpy import mean
from numpy import std

import random

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# function definition
def evaluate_on_training_set(y_test, y_pred):
  # Calculate AUC
  print("AUC is: ", roc_auc_score(y_test, y_pred))
  
  # print out recall and precision
  print(classification_report(y_test, y_pred))
  
  # print out confusion matrix
  print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

  # # calculate points for ROC curve
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  
  # Plot ROC curve
  plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_score(y_test, y_pred))
  plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate or (1 - Specifity)')
  plt.ylabel('True Positive Rate or (Sensitivity)')
  plt.title('Receiver Operating Characteristic')
  plt.show()


# data pre processing
CLS_TOKEN_INDEX = 0

df = pd.read_json("../MUStARD/data/sarcasm_data.json")

df = df.transpose()

embeddings = []

with jsonlines.open("../MUStARD/data/bert-output.jsonl") as utterances:
   
    for utterance in utterances:

        features = utterance["features"][CLS_TOKEN_INDEX]

        bert_embedding_target = np.mean([np.array(features["layers"][layer]["values"])
                                            for layer in range(4)], axis=0)
        
        embeddings.append(np.copy(bert_embedding_target))


test_size = 207
train_size = len(df) - test_size

# shuffle inputs
# random.shuffle(embeddings)
# need to do exactly the same shuffling for outputs to preserve ordering
# try 
"""
train, test, output_train, output_test = train_test_split(input,
                                                            output,
                                                            test_size=0.3,
                                                            random_state=1)
"""

# train
train_input = embeddings[:train_size]
train_output = df.iloc[:train_size]["sarcasm"].astype(int)

# define a new scaler: 
x_scaler = MinMaxScaler()

# fit the normalization on the training set: 
x_scaler.fit(train_input)

# then create new and normalized training/test sets: 
X_train_norm = x_scaler.transform(train_input)

# test
test_input = embeddings[train_size:train_size+test_size]
test_output = df.iloc[train_size:train_size+test_size]["sarcasm"].astype(int)
X_test_norm = x_scaler.transform(test_input)


solvers = ["liblinear", "sag", "saga", "newton-cg", "lbfgs"]
c_values = [1.0, 3.0, 4.0, 5.0, 6.0, 10.0, 100.0 ]

f = open('LR_results.txt', 'w')
f.write("File cleared. \n\n")
f.close()

# searching for the best solver
for solver in solvers:
    for c in c_values:
        model = LogisticRegression(C=c, multi_class='auto', solver=solver) 
        model.fit(train_input, train_output)    # Training the model 
        predicted = model.predict(X_test_norm)  # Predicting labels for our test set using trained model

        # 5-fold cross validation
        accuracy_scores = cross_val_score(model, X_test_norm, test_output, cv=5, scoring='accuracy')
        f1_scores = cross_val_score(model, X_test_norm, test_output, cv=5, scoring='f1')

        # results
        #print(f"Classification with solver {solver} and C value {c}")
        #print('Accuracy: %.3f (%.3f)' % (mean(accuracy_scores), std(accuracy_scores)))
        #print('f1 score: %.3f (%.3f)' % (mean(f1_scores), std(f1_scores)))
        print(classification_report(test_output, predicted))#, output_dict=True, digits=3))
        evaluate_on_training_set(test_output, predicted)
        reportDict = classification_report(test_output, predicted, output_dict=True)

        f = open('LR_results.txt', 'a')
        f.write(f"\nClassification with solver {solver} and C value {c}")

        # results with 5-fold cv
        f.write('\nAccuracy: %.3f (%.3f)' % (mean(accuracy_scores), std(accuracy_scores)))
        f.write('\nf1 score: %.3f (%.3f)' % (mean(f1_scores), std(f1_scores)))

        # results without cv
        f.write('\nAccuracy without cv: ')
        f.write(str(reportDict['accuracy']))
        f.write('\n')

        #f.write(classification_report(test_output, predicted))
        #f.write('\n')
        #f.write(evaluate_on_training_set(test_output, predicted))
        f.close()

# only best solvers
solvers = ["liblinear", "newton-cg", "lbfgs"]
c_values = [1.0]

for solver in solvers:
    for c in c_values:
        model = LogisticRegression(C=c, multi_class='auto', solver=solver) 
        model.fit(train_input, train_output)    # Training the model 
        predicted = model.predict(X_test_norm)  # Predicting labels for our test set using trained model

        # 5-fold cross validation
        accuracy_scores = cross_val_score(model, X_test_norm, test_output, cv=5, scoring='accuracy')
        f1_scores = cross_val_score(model, X_test_norm, test_output, cv=5, scoring='f1')

        # results
        print(f"Classification with solver {solver} and C value {c}")
        print('Accuracy: %.3f (%.3f)' % (mean(accuracy_scores), std(accuracy_scores)))
        print('F1 score: %.3f (%.3f)' % (mean(f1_scores), std(f1_scores)))
        #print(classification_report(test_output, predicted))#, output_dict=True, digits=3))
        evaluate_on_training_set(test_output, predicted)
        reportDict = classification_report(test_output, predicted, output_dict=True)