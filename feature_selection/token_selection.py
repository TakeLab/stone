import torch

def only_target(embeddings, target_indices):
    # Only target tokens
    target_embeddings = []
    
    for i in range(len(embeddings)):

        begin = target_indices[i][0]
        end = target_indices[i][1] + 1
        target = embeddings[i][begin:end]

        target_embeddings.append(target)
    
    return target_embeddings


def all_except_target(embeddings, target_indices):
    # All except target tokens
    target_embeddings = []

    for i in range(len(embeddings)):
        begin = target_indices[i][0]
        end = target_indices[i][1] + 1
        target_begin = embeddings[i][:begin]
        target_end = embeddings[i][end:]

        target = torch.cat((target_begin, target_end), dim=0)

        target_embeddings.append(target)
        
    # ne možemo vratiti tensor jer se dimenzije razlikuju
    return target_embeddings

def all_with_aggr_target(embeddings, target_indices): #TODO - provjeriti dimenzije
    target_embeddings = []

    for i in range(len(embeddings)):
        begin = target_indices[i][0]
        end = target_indices[i][1] 

        left = embeddings[i][:begin]
        right = embeddings[i][end:]

        aggregated_target = torch.mean(embeddings[i][begin:end+1], dim=0).unsqueeze(0)
        
        target = torch.cat((left, target, right), dim=0)

        target_embeddings.append(target)
    
    return target_embeddings

    
def all_concatenated(embeddings, target_indices):

    target_embeddings = []

    for i in range(len(embeddings)):
        begin = target_indices[i][0]
        end = target_indices[i][1] 

        left = embeddings[i][:begin]
        right = embeddings[i][end:]

        aggregated_target = torch.mean(embeddings[i][begin:end+1], dim=0).unsqueeze(0)
        
        target = torch.cat((aggregated_target, left, right), dim=0)

        target_embeddings.append(target)
    
    return target_embeddings

def cls_target(embeddings, target_indices):
    target_embeddings = []

    for i in range(len(embeddings)):
        begin = target_indices[i][0]
        end = target_indices[i][1] 

        #print('c',embeddings[i][0].shape)
        #print('a', embeddings[i][begin:end+1].shape)
        cls = embeddings[i][0].unsqueeze(0)
        
        target = torch.cat((cls, embeddings[i][begin:end+1]), dim=0)
        #print(target.shape)

        target_embeddings.append(target)
    
    return target_embeddings


def only_cls(embeddings, target_indices):
    target_embeddings = []

    for i in range(len(embeddings)):

        cls = embeddings[i][0].unsqueeze(0)
        
        target_embeddings.append(cls)
    
    return target_embeddings

