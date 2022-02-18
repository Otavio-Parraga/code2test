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
    def __init__(self, path, split='train', tokenizer=None, add_prefix=False):
        self.path = Path(path)
        self.full_path = self.path / split
        self.data, self.labels = load_dataset(self.full_path)
        self.tokenizer = tokenizer
        if add_prefix: 
            self.data = [f'Code to test: {d}' for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data[idx]
        target = self.labels[idx]

        if self.tokenizer:
            source = self.tokenizer(source, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            target = self.tokenizer(target, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            source = squeeze_dict(source)
            target = squeeze_dict(target)

        return source, target
