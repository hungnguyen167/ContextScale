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


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, n_components):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, n_components),
            nn.Tanh()
            
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_components, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded, x
    
class ClassifierSimple(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, n_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, n_labels),
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits