"""Modular-arithmetic grokking dataset: (a + b) mod p with token sequence [a, '+', b, '=']."""
import torch

OP, EQ = 0, 1


def build_dataset(p, train_frac, seed):
    g = torch.Generator().manual_seed(seed)
    pairs = torch.stack(torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"), dim=-1).reshape(-1, 2)
    targets = (pairs[:, 0] + pairs[:, 1]) % p
    seq = torch.stack([pairs[:, 0] + 2, torch.full((len(pairs),), OP), pairs[:, 1] + 2, torch.full((len(pairs),), EQ)], dim=1)
    perm = torch.randperm(len(pairs), generator=g)
    n_train = int(len(pairs) * train_frac)
    return seq[perm[:n_train]], targets[perm[:n_train]], seq[perm[n_train:]], targets[perm[n_train:]]
