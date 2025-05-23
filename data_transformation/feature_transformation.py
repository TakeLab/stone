from feature_selection.token_selection import *
from feature_selection.token_aggregation import *
from feature_selection.layer_strategy import *

class FeatureTransformation:
    def __init__(self, token_selection=None, token_aggregation=None, layer_selection=None):
        self.token_selection = token_selection
        self.token_aggregation = token_aggregation
        self.layer_selection = layer_selection

        self.token_selection_flag = False
        self.token_aggregation_flag = False
        self.layer_selection_flag = False

        if self.token_selection is not None:
            self.token_selection_flag = True

        if self.token_aggregation is not None:
            self.token_aggregation_flag = True

        if self.layer_selection is not None:
            self.layer_selection_flag = True

    def apply_layer_strategy(self, embeddings, return_pt):
        # Stores the token vectors, with shape [max_length x 768]
        token_vecs = []

        # embeddings is a [max_length x 12 x 768] tensor

        # For each token in the sentence...
        for token in embeddings:
            # token is a [12 x 768] tensor

            # Apply strategy funtction
            vec = self.layer_selection(token)

            # Use vec to represent token
            token_vecs.append(vec)

        # dim(tokens_vec_sum) = [max_length x 768]
        if return_pt:
            return torch.stack(token_vecs)
        else:
            return token_vecs
    
    def transform(self, embeddings, target_indices=None, swap_dim=False, return_pt=True):

        target_embeddings = embeddings

        # Get target embeddings
        if self.token_selection_flag:
            target_embeddings = self.token_selection(target_embeddings, target_indices)         
            #print(f'Target embeddings shape: ({len(target_embeddings)}, {target_embeddings[0].shape})')                   # dim = (#data, #tokens, #layers, #features)

        if swap_dim:
            # Swap dimensions 0 and 1.
            target_embeddings = [emb.permute(1,0,2) for emb in target_embeddings]                                                 # dim = (#data, #layers, #tokens, #features)

        # Aggregate target embeddings
        if self.token_aggregation_flag:
            target_embeddings = self.token_aggregation(target_embeddings)                       
            #print(f'Aggregated target embeddings shape: ({len(target_embeddings)}, {target_embeddings[0].shape})')        # dim = (#data, #layers, #features)

        # Apply layer strategy to target embeddings
        if self.layer_selection_flag:
            target_embeddings = self.apply_layer_strategy(target_embeddings, return_pt)                    
            #print(f'Layer strategy target embeddings shape: ({len(target_embeddings)}, {target_embeddings[0].shape})')    # dim = (#data, #features)

        return target_embeddings

