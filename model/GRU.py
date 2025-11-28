import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 4, 1)
        
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        weights = torch.softmax(energy.squeeze(2), dim=1)
        return torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

class GRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, output_size=24, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, dropout=0.3)
        self.attention = Attention(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        outputs, hidden = self.gru(x)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden*2]
        context = self.attention(last_hidden, outputs.permute(1,0,2))
        return self.fc(context)