# data/utils.py

import torch
from torch.utils.data import Dataset, DataLoader
import string

CHARS = string.printable
VOCAB_SIZE = len(CHARS)
MAX_LEN = 20

char_to_int = {char: i for i, char in enumerate(CHARS)}
int_to_char = {i: char for i, char in enumerate(CHARS)}

class PasswordDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the PRE-PROCESSED tensor data.
    """
    def __init__(self, file_path):
        
        # Load the pre-processed tensor data
        self.data = torch.load(file_path, weights_only=False)

    def __len__(self):
        # The length is the first dimension of the tensor
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Simply return the tensor at the given index
        return self.data[idx]

def get_dataloader(file_path, batch_size):
    """
    Creates a PasswordDataset and wraps it in a DataLoader.
    """
    dataset = PasswordDataset(file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    return dataloader