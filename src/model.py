import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model_components import MLP

class NCF(nn.Module):
    def __init__(self, user_dim, item_dim, embedding_dim, dropout):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(user_dim, embedding_dim)
        self.item_embedding = nn.Embedding(item_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(2 * embedding_dim)
        self.mlp = nn.Sequential(
            MLP(2 * embedding_dim, 1 * embedding_dim, dropout),
            MLP(1 * embedding_dim, int(.5 * embedding_dim), dropout),
            MLP(int(.5 * embedding_dim), int(.25 * embedding_dim), dropout),
            MLP(int(.25 * embedding_dim), int(.1 * embedding_dim), dropout),
            nn.Linear(int(.1 * embedding_dim), 1)
        )
    def forward(self, user: torch.Tensor, item: torch.Tensor, label: torch.Tensor = None) -> Optional[torch.Tensor]:
        """_summary_

        Args:
            user : Tensor containing batch of user data (B, user_dim)
            item : Tensor containing bacth of item data (B, item_dim)
            label : Tensor containing batch of labels which are ratings ranging from 0.0 to 5.0 (B, 1)

        Returns:
            loss: If label is provided, returns the loss
        """
        user_embedding = self.user_embedding(user) # (B, embedding_dim)
        item_embedding = self.item_embedding(item) # (B, embedding_dim)
        x = torch.cat([user_embedding, item_embedding], dim=1) # (B, 2 * embedding_dim)
        x = self.dropout(self.batch_norm(x))
        logits = self.mlp(x) # (B, 1)

        if label is None:
            loss = None
        else:
            loss = F.mse_loss(logits, label) # (1, 1)

        return logits, loss
