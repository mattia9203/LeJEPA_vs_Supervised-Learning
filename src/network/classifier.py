import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)
