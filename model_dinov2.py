from torch import nn
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class LinearClassifier(nn.Module):
    def __init__(self, in_channels, num_labels):
        super(LinearClassifier, self).__init__()
        #self.classifier = nn.Linear(in_channels, num_labels)
        self.classifier = nn.Sequential(
                            nn.Linear(in_channels, 384),
                            nn.Dropout(.2),
                            nn.ReLU(),
                            nn.Linear(384, num_labels),
                            nn.Dropout(.2),
                            nn.ReLU(),
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


class Dinov2ForImageClassification(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        
        # Use the CLS token for classification
        cls_token_embeddings = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_token_embeddings)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )