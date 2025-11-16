import torch
import torch.nn as nn
from data.utils import VOCAB_SIZE, MAX_LEN

# --- Hyperparameters ---
LATENT_DIM = 128
LSTM_HIDDEN = 128
LSTM_LAYERS = 2

# --- Generator (Unchanged) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embedding = nn.Linear(LATENT_DIM, LSTM_HIDDEN)
        self.lstm = nn.LSTM(
            input_size=LSTM_HIDDEN,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=False
        )
        self.fc_out = nn.Linear(LSTM_HIDDEN, VOCAB_SIZE)

    def forward(self, z):
        # z shape: (batch_size, LATENT_DIM)
        z = self.embedding(z) # (batch_size, LSTM_HIDDEN)
        z = z.unsqueeze(1).repeat(1, MAX_LEN, 1) # (batch_size, MAX_LEN, LSTM_HIDDEN)
        
        lstm_out, _ = self.lstm(z) # (batch_size, MAX_LEN, LSTM_HIDDEN)
        output = self.fc_out(lstm_out) # (batch_size, MAX_LEN, VOCAB_SIZE)
        return output

# --- Critic (The WGAN-GP Version - NO SIGMOID) ---
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(
            input_size=VOCAB_SIZE,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True # Bidirectional for more power
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(2 * LSTM_HIDDEN, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
            # ‼️ --- NO nn.Sigmoid() layer --- ‼️
            # The Critic outputs a raw score, not a probability.
        )

    def forward(self, x_one_hot):
        # x_one_hot shape: (batch_size, MAX_LEN, VOCAB_SIZE)
        
        _, (h_n, _) = self.lstm(x_one_hot)
        
        # h_n shape is (num_layers*num_directions, batch_size, hidden_size)
        forward_last = h_n[-2,:,:] # (BATCH_SIZE, 128)
        backward_last = h_n[-1,:,:] # (BATCH_SIZE, 128)
        
        combined_hidden = torch.cat((forward_last, backward_last), dim=1) # (BATCH_SIZE, 256)
        
        score = self.fc_out(combined_hidden) # (BATCH_SIZE, 1)
        return score