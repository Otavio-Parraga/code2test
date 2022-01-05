from pathlib import Path
from torch.utils.data import Dataset
from utils import squeeze_dict


def load_dataset(path):
    """
    Loads the dataset from the given path.
    """
    with open(path / 'input.methods.txt', 'r', encoding='latin') as f:
        data = f.readlines()
    with open(path / 'output.tests.txt', 'r', encoding='latin') as f:
        labels = f.readlines()
    return data, labels


class Code2TestDataset(Dataset):
    # TODO: enable train with prefix
    def __init__(self, path, split='train', tokenizer=None):
        self.path = Path(path / split)
        self.data, self.labels = load_dataset(self.path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data[idx]
        target = self.labels[idx]

        if self.tokenizer:
            source = self.tokenizer(source, padding='max_length', truncation=True, return_tensors='pt')
            target = self.tokenizer(target, padding='max_length', truncation=True, return_tensors='pt')
            source = squeeze_dict(source)
            target = squeeze_dict(target)

        return source, target
