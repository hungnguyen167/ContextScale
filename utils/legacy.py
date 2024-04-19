def scale_func(train_dataloader, eval_dataloader, full_dataloader, model, 
               optimizer, scheduler, loss_fn_topic, loss_fn_lr, loss_fn_reconstruct,
               device, topic_var, lr_var, party_election_var, train_topic=True,n_epochs=3):
    print("")
    print('Training with RoBERTa layers frozen and concatenated party_election labels')
    for epoch in range(n_epochs):
      print(f'Epoch: {epoch+1}')
      ## Scale funcion first optimizes the model without roberta layers but using party_election labels concatenated to roberta_output
      # Put the model into training mode. 
      size = len(train_dataloader.dataset)
      model.train()
      train_loss = 0
      # For each batch of training data...optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
      for batch_num, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        if train_topic is True:
          y_topic = batch[topic_var].long()

        y_lr = batch[lr_var].long()
        logits_topic, logits_lr, logits_reconstruct, roberta_output, _ = model(input_ids = batch['input_ids'], 
                                        attention_mask = batch['attention_mask'],
                                        party_election_ids = batch[party_election_var].long())
        
        loss_lr = loss_fn_lr(logits_lr, y_lr)
        loss_reconstruct = loss_fn_reconstruct(logits_reconstruct, roberta_output)
        if train_topic:
          loss_topic = loss_fn_topic(logits_topic, y_topic)
          loss = 0.4*loss_topic  + 0.4*loss_lr + 0.1*loss_reconstruct
          
        else:
          loss = 0.7*loss_lr + 0.3*loss_reconstruct
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        # Report
        if batch_num % 100 == 0 and batch_num != 0:
          current_loss = train_loss/batch_num
          current = batch_num * len(batch['input_ids'])
          print(f"loss: {current_loss:>7f}  [{current:>5d}/{size:>5d}].")
      

      print("")
      print("Evaluation...")
      size = len(eval_dataloader.dataset)
      num_batches = len(eval_dataloader)
      test_loss, correct_topic, correct_lr = 0.0, 0.0, 0.0
      model.eval()
      with torch.no_grad():  
          for batch in eval_dataloader:    
              batch = {k: v.to(device) for k,v in batch.items()}
              if train_topic is True:
                  y_topic = batch[topic_var].long()

              y_lr = batch[lr_var].long()
              logits_topic, logits_lr, logits_reconstruct, roberta_output,_ = model(input_ids = batch['input_ids'], 
                                          attention_mask = batch['attention_mask'],
                                          party_election_ids = batch[party_election_var].long())
              loss_lr = loss_fn_lr(logits_lr, y_lr)
              loss_reconstruct = loss_fn_reconstruct(logits_reconstruct, roberta_output)
              if train_topic:
                  loss_topic = loss_fn_topic(logits_topic, y_topic)
                  loss = 0.4*loss_topic  + 0.4*loss_lr + 0.1*loss_reconstruct
                  correct_topic += (logits_topic.argmax(1) == batch[topic_var]).type(torch.float).sum().item()
              else:
                  loss = 0.7*loss_lr + 0.3*loss_reconstruct

              test_loss += loss
              
              correct_lr += (logits_lr.argmax(1) == batch[lr_var]).type(torch.float).sum().item()

      test_loss /= num_batches
      correct_topic /= size
      correct_lr /= size
      correct = (correct_topic+correct_lr)/2
      print(f"Test Error: \n Accuracy: {(correct*100):>0.1f}%, Avg loss: {test_loss:>8f} \n")
      print(f"Test Error: \n Accuracy - LRN: {(correct_lr*100):>0.1f}, Avg loss: {test_loss:>8f} \n")
      print(f"Test Error: \n Accuracy - Topic: {(correct_topic*100):>0.1f}, Avg loss: {test_loss:>8f} \n")

    print('Start scaling...')
    model.eval()
    res_topic = []
    res_lr = []
    party_election_embeddings = []
    with torch.no_grad():  
        for batch in full_dataloader:    
            batch = {k: v.to(device) for k,v in batch.items()}
            logits_topic, logits_lr,_,_, party_election_embedding = model(input_ids = batch['input_ids'], 
                                            attention_mask = batch['attention_mask'],
                                            party_election_ids = batch[party_election_var].long())
            pred_topic = logits_topic.argmax(1)
            pred_lr = logits_lr.argmax(1)
            res_topic.append(pred_topic)
            res_lr.append(pred_lr)
            party_election_embeddings.append(party_election_embedding)
    pred_topics = torch.cat(res_topic, dim=0).cpu().detach().numpy()
    pred_lrs =  torch.cat(res_lr, dim=0).cpu().detach().numpy()
    party_election_embeddings = torch.cat(party_election_embeddings, dim=0).cpu().detach().numpy()
    return pred_topics, pred_lrs, party_election_embeddings

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