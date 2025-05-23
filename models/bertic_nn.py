import numpy as np
import torch
import torch.nn as nn

from transformers import pipeline, AutoTokenizer, AutoModelForPreTraining
from models.dl_baseline import DLBaseline

class Bertic_NN(DLBaseline):
    def __init__(self, feature_transformation, input_size, output_size=3, freeze_bert=False):
        super().__init__()

        # Init Bertić
        self.bertic = AutoModelForPreTraining.from_pretrained("classla/bcms-bertic")

        # Init feed-forward classifier
        self.ffnn = nn.Sequential(nn.Linear(input_size, output_size))                        
        
        # Init feature transformation function
        self.feature_transformation = feature_transformation

        # Freeze the BERT model - if we want to freeze params
        if freeze_bert:
            for param in self.bertic.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_masks, token_type_ids, target_indices, ner_type=None):

        # Unpack inputs
        #input_ids = input_ids
        #attention_masks = attention_masks
        #token_type_ids = token_type_ids
        #target_indices = target_indices
        
        # Get Bertic embeddings
        outputs = self.bertic(input_ids=input_ids,
                              attention_mask=attention_masks, 
                              token_type_ids = token_type_ids,
                              output_hidden_states=True)

        # Get hidden states
        hidden_states = outputs.hidden_states

        # Transform hidden states shape to [#data_points, #tokens, #layers, #features]
        hidden_states = torch.stack(hidden_states, dim=0)
        hidden_states = torch.squeeze(hidden_states, dim=1)
        hidden_states = hidden_states.permute(1,2,0,3)

        
        # Apply output transformation (token selection, token aggregation, layer selection)
        transformed_hs = self.feature_transformation.transform(hidden_states, target_indices)


        if ner_type != None:
            transformed_hs = torch.cat((transformed_hs, ner_type.unsqueeze(1)), dim=1)

        # Feed transformed hidden states to feed-forward classifier
        logits = self.ffnn(transformed_hs)

        return logits


    def reset_parameters(self):
        pass

    