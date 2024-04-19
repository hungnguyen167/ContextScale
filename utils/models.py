import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_utils import PreTrainedModel

## Main model


class TLRRPredict(PreTrainedModel):
    def __init__(self, config, roberta_model, topic_count, lr_count, 
                 dropout=None, hidden_dim=None, hidden_dim_2=None):
        super(TLRRPredict, self).__init__(config)
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model)
        roberta_dim = config.hidden_size


        if hidden_dim is None:
            hidden_dim = int(roberta_dim // 2)

        if hidden_dim_2 is None:
            hidden_dim_2 = int(topic_count // 2)
        self.dropout = dropout if dropout is not None else 0.2

        self.mlp1 = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(hidden_dim, topic_count)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(topic_count, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),  
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2), 
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim_2, lr_count)
        )
        self.decoder = nn.Sequential(
            nn.Linear(lr_count, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(hidden_dim_2, topic_count),
            nn.BatchNorm1d(topic_count),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(topic_count, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(hidden_dim, roberta_dim)
        )


    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        logits_topic = self.mlp1(roberta_output)
        logits_lr = self.mlp2(logits_topic)
        logits_reconstruct = self.decoder(logits_lr)
        return logits_topic, logits_lr, logits_reconstruct, roberta_output
    
    def freeze_roberta_layers(self):
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False


   



