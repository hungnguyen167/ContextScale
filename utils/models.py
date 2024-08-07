import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_utils import PreTrainedModel
from peft import LoraConfig, get_peft_model

## Main model

class TLRRPredict(PreTrainedModel):
    def __init__(self, config, roberta_model, topic_count, lr_count, lora=False, 
                 dropout=None, hidden_dim=None):
        super(TLRRPredict, self).__init__(config)
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model)
        if lora is True:
            lora_config=LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, use_rslora=True)
            self.roberta = get_peft_model(self.roberta, lora_config)

        roberta_dim = config.hidden_size

        self.topic_count = topic_count
        self.lr_count = lr_count
        if hidden_dim is None:
            hidden_dim = int((roberta_dim + topic_count + lr_count)// 2)

        self.dropout = dropout if dropout is not None else 0.1

        self.mlp1 = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(hidden_dim, topic_count)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(roberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim, lr_count)
        )
        


    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        logits_topic = self.mlp1(roberta_output)
        logits_lr = self.mlp2(roberta_output)
        return logits_topic, logits_lr

    def get_trainable_params(self, return_count=False):
        if return_count:
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return total_params
        else:
            print("Trainable Parameters:")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name) 



