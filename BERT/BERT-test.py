#### CODE ADAPTED FROM ###
# https://colab.research.google.com/github/eugenesiow/practical-ml/blob/master/notebooks/Sarcasm_Detection_Twitter.ipynb#scrollTo=biJ0N3ZB1IB9

import numpy as np
import pandas as pd
import logging
import sklearn
from simpletransformers.classification import ClassificationModel
import matplotlib.pyplot as plot

class BERT_Model:
    # size is number of either positive or negative examples (both should be same size based on original dataset)
    # so, actual sizes are twice of those indicated
    DEFAULT_VALIDATE_SIZE = 30
    DEFAULT_TEST_SIZE = 20
    def __init__(self, validate_size=DEFAULT_VALIDATE_SIZE, test_size=DEFAULT_TEST_SIZE):
        self.df = pd.read_json("../MUStARD/data/sarcasm_data.json")
        self.df = self.df.transpose()
        # TODO: for now, combine utterance and context and drop other features
        self.df['context'] = [' '.join(map(str, l)) for l in self.df['context']]
        self.df['text'] = self.df['context'] + ' ' + self.df['utterance']
        self.df = self.df.rename(columns={'sarcasm': 'labels'})
        self.df['labels'] = self.df['labels'].astype(int)

        # create training, validate, and test sets
        self.negative_df = self.df[self.df['labels']==0]
        self.positive_df = self.df[self.df['labels']==1]
        assert(len(self.negative_df) == len(self.positive_df))
        
        self.validate_size = validate_size
        self.test_size = test_size
        self.train_size = len(self.positive_df) - self.validate_size - self.test_size

        self.train_df = pd.concat([self.positive_df[:self.train_size], self.negative_df[:self.train_size]])
        self.validate_df = pd.concat([self.positive_df[self.train_size:self.train_size+self.validate_size], self.negative_df[self.train_size:self.train_size+self.validate_size]])
        self.test_df = pd.concat([self.positive_df[self.train_size+self.validate_size:len(self.positive_df)], self.negative_df[self.train_size+self.validate_size:len(self.negative_df)]])


    def graph_examples_distribution(self):
        distribution = [[self.train_df.labels.value_counts()[0], self.test_df.labels.value_counts()[0], self.validate_df.labels.value_counts()[0]], 
                [self.train_df.labels.value_counts()[1], self.test_df.labels.value_counts()[1], self.validate_df.labels.value_counts()[1]]]
        # Prints out the dataset sizes of train test and validate as per the table.
        print(pd.DataFrame(distribution, columns=["Train", "Test", "Validate"]))
        # TODO: graph this
        
    def train_model(self, train_args):
        # We use the RoBERTa base pre-trained model.
        model = ClassificationModel('roberta', 'roberta-base', num_labels=2, args=train_args, use_cuda=False) 

        # Train the model, use the validate set as the development set as per the paper.
        # When training to 1 epoch this is not that essential, however, if you decide to 
        # train more and configure early stopping, do check out the simple transformers
        # documentation: https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
        model.train_model(self.train_df, eval_df=self.validate_df)

        # Evaluate the model in terms of accuracy score
        result, model_outputs, wrong_predictions = model.eval_model(self.test_df, acc=sklearn.metrics.accuracy_score)

        return result, model_outputs, wrong_predictions

    def train_with_varied_learning_rates(self, train_args):
        learning_rates = np.arange(0.00001, 0.01, 0.0005)
        accuracies = []
        aurocs = []
        auprcs = []
        for learning_rate in learning_rates:
            train_args['learning_rate'] = learning_rate
            result, model_outputs, wrong_predictions = self.train_model(train_args)
            accuracies.append(result['acc'])
            aurocs.append(result['auroc'])
            auprcs.append(result['auprc'])
        
        plot.scatter(learning_rates, accuracies)
        plot.xlabel("Learning rate")
        plot.ylabel("Accuracy")
        plot.show()

        plot.scatter(learning_rates, aurocs)
        plot.xlabel("Learning rate")
        plot.ylabel("AUROC value")
        plot.show()

        plot.scatter(learning_rates, auprcs)
        plot.xlabel("Learning rate")
        plot.ylabel("AUPRC value")
        plot.show()

    def train_with_varied_epochs(self, train_args):
        num_train_epochs = np.arange(1, 6, 1)
        accuracies = []
        aurocs = []
        auprcs = []
        for epochs in num_train_epochs:
            train_args['num_train_epochs'] = int(epochs)
            result, model_outputs, wrong_predictions = self.train_model(train_args)
            accuracies.append(result['acc'])
            aurocs.append(result['auroc'])
            auprcs.append(result['auprc'])
        
        plot.scatter(num_train_epochs, accuracies)
        plot.xlabel("Number of epochs")
        plot.ylabel("Accuracy")
        plot.show()

        plot.scatter(num_train_epochs, aurocs)
        plot.xlabel("Number of epochs")
        plot.ylabel("AUROC value")
        plot.show()

        plot.scatter(num_train_epochs, auprcs)
        plot.xlabel("Number of epochs")
        plot.ylabel("AUPRC value")
        plot.show()

if __name__ == "__main__":

    BERT_model = BERT_Model()
    BERT_model.graph_examples_distribution()

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

    BERT_model.train_model(train_args)
    #BERT_model.train_with_varied_learning_rates(train_args)
    #train_with_varied_epochs(train_args)

