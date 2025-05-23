# author: ddukic

import torch
import torch.nn as nn
from transformers.activations import get_activation
from transformers.models.electra.modeling_electra import (
    ElectraClassificationHead,
    ElectraForSequenceClassification,
    SequenceClassifierOutput,
)


class BerticSTONESequenceClassification(ElectraForSequenceClassification):
    # adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/models/electra
    def __init__(self, config, id2labels):
        super().__init__(config=config)

        self.id2labels = id2labels

        self.classification_heads = nn.ModuleDict()

        self.classification_heads["sentiment"] = ElectraClassificationHeadSentiment(
            config
        )
        self.classification_heads["tone"] = ElectraClassificationHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        task=None,
        ner_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # take only the last state
        sequence_output = discriminator_hidden_states[0]

        if task == "sentiment":
            logits = self.classification_heads[task](sequence_output, ner_ids=ner_ids)
        else:
            logits = self.classification_heads[task](sequence_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ElectraClassificationHeadSentiment(nn.Module):
    """Head for sentence-level classification tasks."""

    # adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/models/electra

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    # TODO make this better
    def forward(self, features, ner_ids, **kwargs):
        starts = ner_ids[:, 0]
        ends = ner_ids[:, 1]

        slices = [
            torch.mean(
                torch.index_select(
                    features[i],
                    0,
                    torch.arange(starts[i], ends[i] + 1, device=ner_ids.device),
                ),
                dim=0,
            )
            for i in range(ner_ids.shape[0])
        ]
        # x is average of NER position tokens
        x = torch.stack(slices)
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
