"""SpectralCircuit-Former — consolidated Colab edition (auto GPU/CPU)."""
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OP, EQ = 0, 1
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def build_dataset(p=53, train_frac=0.4, seed=0, op="+"):
    g = torch.Generator().manual_seed(seed)
    pairs = torch.stack(torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij"), dim=-1).reshape(-1, 2)
    targets = (pairs[:, 0] + pairs[:, 1]) % p if op == "+" else (pairs[:, 0] * pairs[:, 1]) % p
    op_tok = OP if op == "+" else OP + 1
    seq = torch.stack([pairs[:, 0] + 3, torch.full((len(pairs),), op_tok),
                       pairs[:, 1] + 3, torch.full((len(pairs),), EQ)], dim=1)
    perm = torch.randperm(len(pairs), generator=g)
    n_train = int(len(pairs) * train_frac)
    return seq[perm[:n_train]], targets[perm[:n_train]], seq[perm[n_train:]], targets[perm[n_train:]]


def sym_laplacian(A):
    d = A.sum(dim=-2).clamp_min(1e-8).rsqrt()
    B = 0.5 * (A / d.unsqueeze(-2) + (A / d.unsqueeze(-1)).transpose(-1, -2))
    N = A.shape[-1]
    return torch.eye(N, device=A.device, dtype=A.dtype) - B, B


def fiedler_value(A, differentiable=False):
    with torch.set_grad_enabled(differentiable):
        L, _ = sym_laplacian(A)
        return torch.linalg.eigvalsh(L)[..., 1]


def spectral_entropy(A):
    _, B = sym_laplacian(A)
    ev = torch.linalg.eigvalsh(B).abs().clamp_min(1e-12)
    p = ev / ev.sum(dim=-1, keepdim=True)
    N = A.shape[-1]
    return -(p * p.log()).sum(dim=-1) / math.log(N)


def mean_row_entropy(A):
    q = A.clamp_min(1e-12)
    return -(q * q.log()).sum(dim=-1).mean(dim=-1) / math.log(A.shape[-1])


def sink_fraction(A):
    return A.max(dim=-2).values.mean(dim=-1)


def effective_rank(X):
    Xc = X - X.mean(dim=-2, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    return (s.sum(dim=-1) ** 2) / s.pow(2).sum(dim=-1).clamp_min(1e-12)


def fiedler_regularizer(A_probe, tau):
    L, _ = sym_laplacian(A_probe)
    lam2 = torch.linalg.eigvalsh(L)[..., 1]
    return torch.relu(tau - lam2).mean(), lam2.detach()


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult=4, use_residual=True):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.use_residual = use_residual
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model)

    def forward(self, x, head_mask=None):
        B, N, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        A = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (A @ v).transpose(1, 2).reshape(B, N, D)
        if head_mask is not None:
            m = head_mask.repeat_interleave(self.d_head).view(1, 1, D)
            y = y * m
        x = self.proj(y) + (x if self.use_residual else 0)
        x = self.fc2(F.gelu(self.fc1(self.ln2(x)))) + (x if self.use_residual else 0)
        return x, A


class TinyTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=2, n_heads=4, max_len=16, use_residual=True):
        super().__init__()
        self.n_layers = n_layers
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, use_residual=use_residual) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx, head_mask=None, full_logits=False):
        B, N = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(N, device=idx.device))
        attn_maps = []
        for i, blk in enumerate(self.blocks):
            hm = None if head_mask is None else head_mask[i]
            x, A = blk(x, head_mask=hm)
            attn_maps.append(A)
        z = self.head(self.ln_f(x))
        if full_logits:
            return z, attn_maps, x
        return z[:, -1], attn_maps, x

    def param_groups(self, wd):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (no_decay if (p.ndim <= 1 or "ln" in name) else decay).append(p)
        return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, seq, targets, bs=4096):
    correct = 0
    for i in range(0, len(seq), bs):
        logits, _, _ = model(seq[i:i + bs])
        correct += (logits.argmax(-1) == targets[i:i + bs]).sum().item()
    return correct / len(seq)


@torch.no_grad()
def spectral_probe(model, probe):
    _, attn_maps, hidden = model(probe)
    L, H, N = len(attn_maps), attn_maps[0].shape[1], attn_maps[0].shape[-1]
    lam2 = np.zeros((L, H)); sent = np.zeros((L, H)); rent = np.zeros((L, H))
    sink = np.zeros((L, H)); spectra = np.zeros((L, H, N))
    for l, A in enumerate(attn_maps):
        lam2[l] = fiedler_value(A).mean(dim=0).cpu().numpy()
        sent[l] = spectral_entropy(A).mean(dim=0).cpu().numpy()
        rent[l] = mean_row_entropy(A).mean(dim=0).cpu().numpy()
        sink[l] = sink_fraction(A).mean(dim=0).cpu().numpy()
        _, B = sym_laplacian(A)
        spectra[l] = torch.linalg.eigvalsh(B).mean(dim=0).cpu().numpy()
    h = model.tok(probe) + model.pos(torch.arange(probe.shape[1], device=probe.device))
    eff_ranks = [float(effective_rank(h).mean())]
    for blk in model.blocks:
        h, _ = blk(h)
        eff_ranks.append(float(effective_rank(h).mean()))
    return lam2, sent, rent, sink, spectra, eff_ranks


def train_run(seed, regularize=False, p=53, train_frac=0.4, d_model=128, n_layers=2,
              n_heads=4, lr=1e-3, wd=2.0, max_steps=12000, log_every=100,
              tau=0.35, beta1=1.0, dataset=None, op="+", use_residual=True):
    torch.manual_seed(seed)
    if dataset is None:
        dataset = build_dataset(p, train_frac, seed, op)
    train_x, train_y, test_x, test_y = (t.to(DEVICE) for t in dataset)
    model = TinyTransformer(p + 3, d_model, n_layers, n_heads, max_len=train_x.shape[1],
                            use_residual=use_residual).to(DEVICE)
    opt = torch.optim.AdamW(model.param_groups(wd), lr=lr, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(seed + 999)
    idx = torch.randperm(len(train_x), generator=g)[:64].to(DEVICE)
    probe = train_x[idx]
    log = {"step": [], "train_loss": [], "train_acc": [], "test_acc": [], "wall_s": [],
           "lam2": [], "spec_ent": [], "row_ent": [], "sink": [], "spectra": [], "eff_rank": [],
           "A_mean": []}
    transition_step = None
    t0 = time.time()
    for step in range(1, max_steps + 1):
        logits, attn_maps, hidden = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        if regularize:
            reg, _ = fiedler_regularizer(torch.stack(attn_maps, dim=1)[probe], tau)
            loss = loss + beta1 * reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                tr_logits, _, _ = model(train_x)
                tr_acc = (tr_logits.argmax(-1) == train_y).float().mean().item()
            te_acc = evaluate(model, test_x, test_y)
            lam2, sent, rent, sink, spectra, eff = spectral_probe(model, probe)
            log["step"].append(step); log["train_loss"].append(float(loss.detach()))
            log["train_acc"].append(tr_acc); log["test_acc"].append(te_acc)
            log["wall_s"].append(round(time.time() - t0, 1))
            log["lam2"].append(lam2.tolist()); log["spec_ent"].append(sent.tolist())
            log["row_ent"].append(rent.tolist()); log["sink"].append(sink.tolist())
            log["spectra"].append(spectra.tolist()); log["eff_rank"].append(eff)
            log["A_mean"].append(torch.stack(attn_maps, dim=1).mean(dim=0).cpu().tolist())
            print(f"  step {step:5d} | loss {loss.item():.4f} | test_acc {te_acc:.3f} | lam2 {lam2.mean():.3f} | {time.time()-t0:.0f}s", flush=True)
            if te_acc >= 0.95 and transition_step is None:
                transition_step = step
            if te_acc >= 0.999 and step >= (transition_step or 0) + 2000:
                break
    return {"config": {"seed": seed, "regularize": regularize, "p": p, "tau": tau, "op": op,
                       "use_residual": use_residual, "n_layers": n_layers},
            "transition_step": transition_step, "final_test_acc": log["test_acc"][-1],
            "steps_run": log["step"][-1], "log": log, "model": model}
