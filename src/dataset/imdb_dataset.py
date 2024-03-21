import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, split):
        super(IMDBDataset, self).__init__()
        imdb = load_dataset("imdb")
        self.data = imdb[split]
        # negative (label = 0): Tensor([1, 0]), positive (label=1): Tensor([0, 1])
        self.labels = [torch.Tensor([1, 0]), torch.Tensor([0, 1])]

    def __getitem__(self, item):
        return self.data[item]["text"], self.labels[self.data[item]["label"]]

    def __len__(self):
        return len(self.data)
