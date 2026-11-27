"""Tiny instrumented decoder-only transformer that exposes per-head attention maps."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        A = att.softmax(dim=-1)
        y = (A @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x, A


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=2, n_heads=4, max_len=8):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        attn_maps = []
        for blk in self.blocks:
            x, A = blk(x)
            attn_maps.append(A)
        x = self.ln_f(x)
        return self.head(x[:, -1]), attn_maps, x

    def param_groups(self, wd):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (no_decay if (p.ndim <= 1 or "ln" in name) else decay).append(p)
        return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
