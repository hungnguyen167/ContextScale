import torch
import torch.nn as nn
from torch.utils.data import Dataset


class AutoencoderClassifier(nn.Module):
    def __init__(self, tf_model, num_last_layers, hidden_dim, num_classes, n_components):
        super(AutoencoderClassifier, self).__init__()
        input_dim = 768*num_last_layers
        self.tf = tf_model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.encoder = nn.Linear(hidden_dim, n_components)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.num_last_layers = num_last_layers
    def forward(self, input_ids, attention_mask):
        ## TF embeddings
        hidden_states = self.tf(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)['hidden_states'] 
        inputs = torch.cat(tuple([hidden_states[i] for i in range(-self.num_last_layers,0)]), dim=-1)
        ## Normalize
        inputs_norm = inputs.view(inputs.size(0),-1)
        inputs_norm -= inputs_norm.min(1, keepdim=True)[0]
        inputs_norm /= inputs_norm.max(1, keepdim=True)[0]
        inputs_norm = inputs_norm.view(inputs.size(0),-1,self.num_last_layers)
        # Encoder
        _, (hidden, _) = self.lstm(inputs_norm)
        encoded = self.fc1(hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1))
        encoded = self.relu(encoded)

        # Decoder 
        decoded = self.decoder(encoded)
        decoded = self.sigmoid(decoded)
        # Classifier
        _, (hidden_cls, _) = self.lstm(inputs)
        logits = self.classifier(hidden_cls.permute(1, 0, 2).contiguous().view(hidden_cls.size(1), -1))
        logits = self.relu(logits)
        
        return inputs_norm, decoded, logits



class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, text, device, tokenizer, labels):
        self.encodings = tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=256)
        self.labels = labels
        self.device = device
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(self.device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item
    def __len__(self):
        return len(self.labels)
