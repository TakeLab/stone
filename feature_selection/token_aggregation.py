import torch

# Aggregation method: AVERAGE

def average_aggregation(token_embeddings):  # returns list
    aggregated_embeddings = []

    for embedding in token_embeddings:
        aggregated_embeddings.append(torch.mean(embedding, dim=0))

    return aggregated_embeddings   # dim = (#data_points, #layers, #features)


# Aggregation method: SUM
def sum_aggregation(token_embeddings):  # returns list
    aggregated_embeddings = []

    for embedding in token_embeddings:
        aggregated_embeddings.append(torch.sum(embedding, dim=0))

    return aggregated_embeddings   # dim = (#data_points, #layers, #features)