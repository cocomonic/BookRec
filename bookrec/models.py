import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self, users, items, dim):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def users(self, ids):
        return F.normalize(self.user_mlp(self.user(ids)), dim=-1)

    def items(self, ids):
        return F.normalize(self.item_mlp(self.item(ids)), dim=-1)


class DeepFM(nn.Module):
    """Shared field embeddings for user, item and TRAIN-only popularity bucket."""
    def __init__(self, users, items, dim, deep=True):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, dim) for n in [users, items, 10]])
        self.linear = nn.ModuleList([nn.Embedding(n, 1) for n in [users, items, 10]])
        self.bias = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim*3, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.deep = deep

    def forward(self, x):
        e = torch.stack([m(x[:, j]) for j, m in enumerate(self.emb)], 1)
        first = sum(m(x[:, j]).squeeze(-1) for j, m in enumerate(self.linear))
        fm = 0.5 * (e.sum(1).square() - e.square().sum(1)).sum(1)
        return self.bias + first + fm + (self.mlp(e.flatten(1)).squeeze(-1) if self.deep else 0)
