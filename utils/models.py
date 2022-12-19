import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification   

class MetaBERT(nn.Module):
    def __init__(self, bert_model, labels_count, extra_dim=None, dropout=0.25, unq_year=None, hidden_dim=25, pooler=True, bert_dim=None):
        super().__init__()
        if pooler is True:
            bert_dim = 768
        else:
            bert_dim = bert_dim
        self.extra_dim = extra_dim
        self.bert = BertModel.from_pretrained(bert_model) ## get information from a pre-trained BERT model
        self.dropout = nn.Dropout(p=dropout) ## dropout rate
        embed_size = sum((unq_year, 1))//2
        self.lin1 = nn.Linear(bert_dim + extra_dim + embed_size, hidden_dim)
        self.relu =  nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, labels_count)
        self.embed = nn.Embedding(unq_year, embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(bert_dim + extra_dim + embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, labels_count)
        )
        self.pooler = pooler
    def forward(self, input_ids, attention_mask, token_type_ids, extras=None, year=None):
        if self.pooler:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['pooler_output'] 
        else:
            hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)['hidden_states'] 
            pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
            #mean_states = torch.mean(last_hidden_states, dim=0) ## averaging the last four hidden states
            #mean_cls = mean_states[:,0,:] ## take only the vector of the CLS token (for classification)
            bert_output = pooled_output[:,0,:]
        dropout_bert = self.dropout(bert_output)
        year_embed = self.embed(year)
        combined_output = torch.cat((dropout_bert, year_embed, extras), dim = 1).float()
        dropout_output = self.dropout(combined_output)
        logits = self.mlp(dropout_output)
        return logits



        
