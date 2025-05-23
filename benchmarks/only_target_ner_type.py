import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset_class.token_dataset import TokenDataset
from data_transformation.label_transformation import label_transformation
from data_transformation.token_transformation import *
import random

from models.bertic_nn import Bertic_NN
from models.fine_tune_framework import DLFramework

from feature_selection.token_selection import *
from feature_selection.token_aggregation import *
from feature_selection.layer_strategy import *

from data_transformation.feature_transformation import FeatureTransformation


def split_dataset(X, y, test_proportion):
    random_state =  random.randint(0, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state = random_state)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    # Set up GPU
    if torch.cuda.is_available():   
        # Choose between cuda:0 or cuda:1 based on GPU availability  
        device = torch.device("cuda:1")            
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(1))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    
    # Load data
    data_path = 'data/gold_label/gold_label_local.csv'
    data = pd.read_csv(data_path)

    X = data[['document_id', 'text', 'ner', 'target_order', 'ner_type']]

    # Set sentiment labels as target labels
    label = 'aggregated_sentiment'
    y = data[label]

    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.3)


    # Train and test shapes
    print('Train and test shapes:\n')
    print('X_train dimension = ', X_train.shape)
    print('y_train dimension = ', y_train.shape)

    print()

    print('X_test dimension = ', X_test.shape)
    print('y_train dimension = ', y_test.shape)

    print('..........................................')

    # Label distribution
    print(f'Train label distribution:\n\n{y_train.value_counts()}')
    print('\n----------------------------------------')
    print(f'Test label distribution:\n\n{y_test.value_counts()}')

    train_dataset = TokenDataset(X_train, y_train,
                            token_transformation,
                            label_transformation)


    test_dataset = TokenDataset(X_test, y_test,
                                token_transformation,
                                label_transformation)

    
    # Only target tokens 
    ft_only_target = FeatureTransformation(only_target, average_aggregation, last_layer)

    # CONFIG_1
    config = {}
    config['batch_size'] = 16
    config['epochs'] = 50
    config['gradient_clipping'] = True

    loss = nn.CrossEntropyLoss()

    # Init model
    bertic = Bertic_NN(ft_only_target, 769, 3)
    bertic.to(device)


    # Init train/eval module
    bertic_module = DLFramework(bertic, loss, config, device)

    # Train and eval model
    bertic_stats, clf_reports = bertic_module.run(train_dataset, test_dataset)

    # Save stats as json
    with open("benchmarks/results/ner_type_only_target_stats.json", "w") as fp:
        json.dump(bertic_stats,fp, indent=8) 

    # Save clf reports
    with open('benchmarks/results/ner_type_only_target_clf_report.pkl', 'wb') as handle:
        pickle.dump(clf_reports, handle)

    # Plot loss and F1
    train_loss = bertic_stats['train_loss']
    test_loss = bertic_stats['val_loss']
    f1 = bertic_stats['val_f1']

    epochs = range(1, config['epochs']+1)

    fig1 = plt.figure("Figure 1")
    plt.plot(epochs, train_loss, label='Train loss')
    plt.plot(epochs, test_loss, label='Test loss')
    plt.legend(loc="upper left")
    plt.savefig('benchmarks/results/ner_type_only_target_loss.jpg')
    plt.show()
    
    fig2 = plt.figure("Figure 2")
    plt.plot(epochs, f1, label='Test F1')
    plt.legend(loc="upper left")
    plt.savefig('benchmarks/results/ner_type_only_target_f1.jpg')
    plt.show()
    
    