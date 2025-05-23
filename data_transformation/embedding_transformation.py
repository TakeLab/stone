import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import pipeline, AutoTokenizer, AutoModelForPreTraining
from tqdm import tqdm

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append([ind,ind+sll-1])

    return results

def get_target_indices(target_entity, tokens, target_id, tokenizer):

    target_indices = []

    for idx in range(len(tokens)):

        # Get target tokens
        target_tokens = tokenizer.encode(target_entity.iloc[idx], add_special_tokens=False)

        # Get indices of target tokens
        #print(target_entity.iloc[idx])
        #print(target_tokens)
        target_idx = find_sub_list(list(target_tokens), list(tokens[idx]))[target_id.iloc[idx]]
        
        target_indices.extend([target_idx])


    return target_indices


def embedding_transformation(data : list, target_entities, target_id, batch_size=128):

    # Import tokenizer
    tokenizer = AutoTokenizer.from_pretrained("classla/bcms-bertic")

    # Import model
    model = AutoModelForPreTraining.from_pretrained("classla/bcms-bertic")
    model.eval()

    # Initialize embeddings
    first = True
    embeddings = torch.tensor(-1)

    # TODO - Initialize attention_weights
    attention_weights = torch.tensor(-1)

    # Initialize target token indices
    target_indices = []

    # Initialize tokens
    all_tokens = []

    # Initialize counter
    batch_start = 0

    # Get batch embeddings
    for batch_end in tqdm(range(batch_size, len(data) + batch_size, batch_size)):

        # Slice data
        if batch_end > len(data):
            batch_end = len(data)
            
        data_batch = data[batch_start:batch_end]
        target_id_batch = target_id[batch_start:batch_end]

        #print(f'Batch {batch_end} generated.')

        # Encode texts
        tokens = tokenizer(data_batch, padding='max_length', truncation=True, return_tensors="pt")
        all_tokens.extend(tokens['input_ids'])

        # Get target indices
        target_batch_index = get_target_indices(target_entities[batch_start:batch_end], tokens['input_ids'], target_id_batch, tokenizer)
        target_indices.extend(target_batch_index)

        # Get embeddings
        with torch.no_grad():
            output = model(**tokens, output_hidden_states=True, output_attentions=True)

        #print(f'Batch {batch_end} embedded.')

        # TODO - Get attention weights

        # Get hidden states
        hidden_states = output.hidden_states

        # Transform hidden states shape to [#data_points, #tokens, #layers, #features]:

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,2,0,3)

        # Concatenate batches
        if first:
            first = False
            embeddings = token_embeddings

        else:
            embeddings = torch.cat([embeddings, token_embeddings], dim=0)

        batch_start = batch_end


    return embeddings, all_tokens, target_indices


if __name__ == '__main__':

    data_path = 'data/annotated_data/full_dataset.csv'
    data = pd.read_csv(data_path)

    texts = list(data['original_text'])[90:100]
    target_entities = data['target_entity'][90:100]
    target_id = data['target_id'][90:100]


    embeddings, tokens, target_indices = embedding_transformation(texts, target_entities, target_id)

    print('Texts length:', len(texts))
    print('Embeddings shape', embeddings.shape)
    print('Target indices length:', len(target_indices))

    print(target_indices)