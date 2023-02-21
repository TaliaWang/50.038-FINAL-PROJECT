import numpy as np
import pandas as pd
import logging
import sklearn
from simpletransformers.classification import ClassificationModel

df = pd.read_json("./MUStARD/data/sarcasm_data.json")
df = df.transpose()
#print(df.head())

# TODO: for now, keep only utterance and sarcasm columns
df.drop(['speaker', 'context', 'context_speakers', 'show'], axis=1, inplace=True)
#print(df.head())

# create training, validate, and test sets
validate_size = 50
test_size = 50
train_size = len(df) - validate_size - test_size

train_df = df[:train_size]
validate_df = df[train_size:train_size+validate_size]
test_df = df[train_size+validate_size:len(df)]
# print(len(train_df))
# print(len(validate_df))
# print(len(test_df))

if __name__ == "__main__":
    #### CODE BELOW TAKEN FROM ###
    # https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sarcasm_Detection_Twitter.ipynb#scrollTo=biJ0N3ZB1IB9

    train_args = {
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'sliding_window': False,
        'max_seq_length': 75,
        'learning_rate': 0.00005,
        'weight_decay': 0.01,
        'warmup_ratio': 0.2,
        'max_grad_norm': 1.0,
        'num_train_epochs': 1,
        'train_batch_size': 32,
        'save_model_every_epoch': False,
        'save_steps': 4000,
        'fp16': True,
        'output_dir': './outputs/',
        'evaluate_during_training': True,
    }

    logging.basicConfig(level=logging.DEBUG)
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.WARNING)

    # We use the RoBERTa base pre-trained model.
    model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=train_args, use_cuda=False) 

    # Train the model, use the validate set as the development set as per the paper.
    # When training to 1 epoch this is not that essential, however, if you decide to 
    # train more and configure early stopping, do check out the simple transformers
    # documentation: https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
    model.train_model(train_df, eval_df=validate_df)

    # Evaluate the model in terms of accuracy score
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

    #print(result)