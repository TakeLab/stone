import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import re


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append([ind,ind+sll-1])
    return results

def get_target_indices(target_entity, target_id, tokens, tokenizer):

    target_indices = []

    for idx in range(len(tokens)):

        # Get target tokens
        target_tokens = tokenizer.encode(target_entity.iloc[idx], add_special_tokens=False)

        # Get indices of target tokens
        print(list(target_tokens))
        
        print(list(tokens[idx]))
        print(find_sub_list(list(target_tokens), list(tokens[idx])))
        target_idx = find_sub_list(list(target_tokens), list(tokens[idx]))[target_id.iloc[idx]]
        
        target_indices.extend([target_idx])

    return torch.tensor(target_indices)


def token_transformation(data : list, target_entities, target_id):

    # Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic")

    # Encode texts
    tokens = tokenizer(data, padding='max_length', truncation=True, return_tensors="pt")

    # Get target indices
    target_indices = get_target_indices(target_entities, target_id, tokens['input_ids'], tokenizer)


    return tokens, target_indices


if __name__ == '__main__':

    data_path = 'data/annotated_data/full_dataset.csv'
    data = pd.read_csv(data_path)

    texts = list(data['text'])[:10]
    target_entities = data['target_entity'][:10]
    target_id = data['target_id'][:10]


    tokens, target_indices = token_transformation(texts, target_entities, target_id)

    print('Texts length:', len(texts))
    print('Embeddings shape', tokens['input_ids'].shape)
    print('Target indices length:', len(target_indices))

    print(tokens)


