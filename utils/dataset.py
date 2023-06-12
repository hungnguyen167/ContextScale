import torch
import numpy as np
from transformers import BertTokenizerFast

class DatasetWithExtraVar(torch.utils.data.Dataset):
    def __init__(self, dataset, list_extra_dims, device, tokenizer, label=None):
        self.encodings = tokenizer.batch_encode_plus(dataset['sentence_text'].tolist(), padding=True, truncation=True, max_length=256)
        self.labels = dataset[label].tolist()
        trimmed_dataset = dataset[list_extra_dims]
        list_columns = [col for col in trimmed_dataset]
        for col in list_columns:
          arr = np.array(trimmed_dataset[col]).reshape((len(dataset), 1))
          dict_temp = dict({col: arr})
          self.encodings.update(dict_temp)
        self.year = dataset['year_recoded'].tolist()
        self.device = device
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(self.device) for key, val in self.encodings.items()}
        item['year'] = torch.tensor(self.year[idx]).to(self.device).long()
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item
    def __len__(self):
        return len(self.labels)


class BareDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device, tokenizer, label=None):
        self.encodings = tokenizer.batch_encode_plus(dataset['sentence_text'].tolist(), padding=True, truncation=True, max_length=256)
        self.labels = dataset[label].tolist()
        self.device = device
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(self.device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item
    def __len__(self):
        return len(self.labels)


class TDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device, tokenizer, label=None):
        self.encodings = tokenizer.batch_encode_plus(dataset['text'].tolist(), padding=True, truncation=True, max_length=300)
        self.labels = dataset[label].tolist()
        self.device = device
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(self.device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(self.device)
        return item
    def __len__(self):
        return len(self.labels)
