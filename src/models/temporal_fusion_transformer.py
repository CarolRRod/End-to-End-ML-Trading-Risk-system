import torch 
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.gate = nn.Linear(output_size, output_size)
        self.skip = nn.Linear(input_size, output_size)

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        residual = self.skip(x)

        out = self.fc1(x)
        out = F.elu(out)
        out = self.fc2(out)

        gate = self.gate(out)
        gate = F.sigmoid(gate)

        out = gate * out + (1-gate) * residual

        return self.layer_norm(out)

