import torch





class AEDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, y):
        self.embeddings = embeddings
        self.y = y
    def __getitem__(self, idx):
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return {'embeddings':embeddings,'labels': y}
    def __len__(self):
        return len(self.embeddings)