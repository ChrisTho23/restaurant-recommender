import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)