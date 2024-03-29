import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

## Main model

class TLRRPredict(PreTrainedModel):
    def __init__(self, config, roberta_model, topic_count, lr_count, 
                 dropout=None, hidden_dim=None, hidden_dim_2=None, hidden_dim_3=None):
        super(TLRRPredict, self).__init__(config)
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model) ## get information from a pre-trained model
        roberta_dim = config.hidden_size
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = int(roberta_dim//2)
        
        if hidden_dim_2:
            self.hidden_dim_2 = hidden_dim_2
        else:
            self.hidden_dim_2 = int(topic_count//2)


        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.2

        self.mlp1 = nn.Sequential(
            nn.Linear(roberta_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim), 
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(self.hidden_dim, topic_count)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(topic_count, self.hidden_dim_2),
            nn.BatchNorm1d(self.hidden_dim_2),  
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(self.hidden_dim_2, self.hidden_dim_2),
            nn.BatchNorm1d(self.hidden_dim_2), 
            nn.ReLU(),
            nn.Dropout(self.dropout), 
            nn.Linear(self.hidden_dim_2, lr_count)
        )
        self.decoder = nn.Sequential(
            nn.Linear(lr_count, self.hidden_dim_2),
            nn.BatchNorm1d(self.hidden_dim_2),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(self.hidden_dim_2, topic_count),
            nn.BatchNorm1d(topic_count),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(topic_count, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),  
            nn.ReLU(),
            nn.Dropout(self.dropout),  
            nn.Linear(self.hidden_dim, roberta_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)['pooler_output'] 
        logits_topic = self.mlp1(roberta_output)
        logits_lr = self.mlp2(logits_topic)
        logits_reconstruct = self.decoder(logits_lr)
        return logits_topic, logits_lr, logits_reconstruct, roberta_output






## Legacy models


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


class BERTLSTM(nn.Module):
    def __init__(self, bert_model, labels_count, dropout=0.1, lstm_lay=1, lstm_dim=128,bidirectional=True,num_last_layers=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.lin = nn.Linear(lstm_dim*2, labels_count)
        else:
            self.lin = nn.Linear(lstm_dim, labels_count)
        self.lstm_lay = lstm_lay
        self.num_last_layers = num_last_layers
        if bidirectional: 
            self.lstm = nn.LSTM(768*num_last_layers,lstm_dim,lstm_lay,batch_first=True,bidirectional=bidirectional)
        else: 
            self.lstm = nn.LSTM(768*num_last_layers,lstm_dim,lstm_lay,batch_first=True,bidirectional=bidirectional)
        self.bilstm = bidirectional
    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)['hidden_states'] 
        bert_output = torch.cat(tuple([hidden_states[i] for i in range(-self.num_last_layers,0)]), dim=-1)
        _, (hidden_last, _) = self.lstm(bert_output)
        if self.bilstm:
            hidden_last_out = torch.cat([hidden_last[-2],hidden_last[-1]],dim=-1)
        else: 
            hidden_last_out = hidden_last[-1]
        dropout_output = self.dropout(hidden_last_out)
        logits = self.lin(dropout_output)
        return logits

class SavedModel(PreTrainedModel):
    def __init__(self, config, roberta_model, topic_count, lr_count, roberta_dim, per_topic_dim, 
                 dropout=None, hidden_dim=None, hidden_dim_2=None, hidden_dim_3=None):
        super(TLRRPredict, self).__init__(config)
        self.roberta = XLMRobertaModel.from_pretrained(roberta_model) ## get information from a pre-trained model
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = int(roberta_dim//2)
        
        if hidden_dim_2:
            self.hidden_dim_2 = hidden_dim_2
        else:
            self.hidden_dim_2 = int(topic_count//2)

        if hidden_dim_3:
            self.hidden_dim_3 = hidden_dim_3
        else:
            self.hidden_dim_3 = int((lr_count+per_topic_dim)//2)

        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.2

        self.mlp1 = nn.Sequential(
            nn.Linear(config.hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim, topic_count)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(topic_count, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),  
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim_2, lr_count)
        )
        self.dim_redu = nn.Sequential(
            nn.Linear(lr_count, hidden_dim_3),
            nn.BatchNorm1d(hidden_dim_3),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_3, per_topic_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(per_topic_dim, hidden_dim_3),
            nn.BatchNorm1d(hidden_dim_3),  
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim_3, lr_count),
            nn.BatchNorm1d(lr_count),  
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(lr_count, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),  
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim_2, topic_count),
            nn.BatchNorm1d(topic_count),  
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(topic_count, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim, roberta_dim)
        )