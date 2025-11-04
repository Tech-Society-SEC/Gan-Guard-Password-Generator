import torch
import torch.nn as nn

from data.utils import VOCAB_SIZE, MAX_LEN

LATENT_DIM = 128
HIDDEN_DIM = 256

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=LATENT_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, MAX_LEN, 1)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=VOCAB_SIZE,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(hidden)
        return out