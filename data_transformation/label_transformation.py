import numpy as np
import pandas as pd
import torch

def label_transformation(labels, label_names:list):

    transformed_labels = []

    transformation_dict = {}

    for i in range(len(label_names)):
        transformation_dict[label_names[i]] = i

    for label in labels:
        transformed_labels.append(transformation_dict.get(label))

    print(transformation_dict)

    return torch.tensor(transformed_labels)


if __name__ == '__main__':
    data_path = 'data/gold_label/gold_label_local.csv'
    data = pd.read_csv(data_path)

    label_names = ['Sentiment - NEG', 'Sentiment - NEUT', 'Sentiment - POZ']

    labels = data['aggregated_sentiment']

    transformed_labels = label_transformation(labels, label_names)

    for i in range(len(labels)):
        print(labels[i], transformed_labels[i])