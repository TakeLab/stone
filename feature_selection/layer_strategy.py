import torch

def apply_strategy(embeddings, strategy_functon, return_pt=True):
    # Stores the token vectors, with shape [max_length x 768]
    token_vecs = []

    # embeddings is a [max_length x 12 x 768] tensor

    # For each token in the sentence...
    for token in embeddings:
        # token is a [12 x 768] tensor

        # Apply strategy funtction
        vec = strategy_functon(token)

        # Use vec to represent token
        token_vecs.append(vec)

    # dim(tokens_vec_sum) = [max_length x 768]
    if return_pt:
        return torch.stack(token_vecs)
    else:
        return token_vecs


# Strategy functions:

# Sums last four hidden layers
def sum_last_four(token):
    return torch.sum(token[-4:], dim=0)

# Sums all hidden layers
def sum_all(token):
    return torch.sum(token[1:], dim=0)

# Mean all hidden layers
def mean_all(token):
    return torch.mean(token[1:], dim=0)

# Returns 11th hidden layer
def second_to_last(token):
    return token[-2]

# Returns last hidden layer
def last_layer(token):
    return token[-1]

# TODO - Add more layer strategies