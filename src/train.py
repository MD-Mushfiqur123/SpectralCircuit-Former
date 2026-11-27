"""Training loop with spectral instrumentation and optional Fiedler-regularizer."""
import time
import numpy as np
import torch
import torch.nn.functional as F

from data import build_dataset
from model import TinyTransformer
from spectral import (fiedler_value, spectral_entropy, mean_row_entropy,
                      sink_fraction, effective_rank, fiedler_regularizer)


@torch.no_grad()
def evaluate(model, seq, targets, bs=2048):
    correct = 0
    for i in range(0, len(seq), bs):
        logits, _, _ = model(seq[i:i + bs])
        correct += (logits.argmax(-1) == targets[i:i + bs]).sum().item()
    return correct / len(seq)


@torch.no_grad()
def spectral_probe(model, probe):
    logits, attn_maps, hidden = model(probe)
    L = len(attn_maps)
    H = attn_maps[0].shape[1]
    N = attn_maps[0].shape[-1]
    lam2 = np.zeros((L, H)); sent = np.zeros((L, H)); rent = np.zeros((L, H))
    sink = np.zeros((L, H)); spectra = np.zeros((L, H, N))
    for l, A in enumerate(attn_maps):
        d = A.sum(dim=-2).clamp_min(1e-8).rsqrt()
        B = 0.5 * (A / d.unsqueeze(-2) + (A / d.unsqueeze(-1)).transpose(-1, -2))
        lam2[l] = fiedler_value(A).mean(dim=0).numpy()
        sent[l] = spectral_entropy(A).mean(dim=0).numpy()
        rent[l] = mean_row_entropy(A).mean(dim=0).numpy()
        sink[l] = sink_fraction(A).mean(dim=0).numpy()
        spectra[l] = torch.linalg.eigvalsh(B).mean(dim=0).numpy()
    h = model.tok(probe) + model.pos(torch.arange(probe.shape[1]))
    eff_ranks = [float(effective_rank(h).mean())]
    for blk in model.blocks:
        h, _ = blk(h)
        eff_ranks.append(float(effective_rank(h).mean()))
    return lam2, sent, rent, sink, spectra, eff_ranks


def train_run(seed, regularize=False, p=53, train_frac=0.4, d_model=128, n_layers=2,
              n_heads=4, lr=1e-3, wd=2.0, max_steps=12000, log_every=100,
              tau=0.05, beta1=1.0, eval_bs=2048):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    train_x, train_y, test_x, test_y = build_dataset(p, train_frac, seed)
    model = TinyTransformer(p + 2, d_model, n_layers, n_heads, max_len=train_x.shape[1])
    opt = torch.optim.AdamW(model.param_groups(wd), lr=lr, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(seed + 999)
    probe = train_x[torch.randperm(len(train_x), generator=g)[:64]]
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
            te_acc = evaluate(model, test_x, test_y, eval_bs)
            lam2, sent, rent, sink, spectra, eff = spectral_probe(model, probe)
            log["step"].append(step); log["train_loss"].append(float(loss.detach()))
            log["train_acc"].append(tr_acc); log["test_acc"].append(te_acc)
            log["wall_s"].append(round(time.time() - t0, 1))
            log["lam2"].append(lam2.tolist()); log["spec_ent"].append(sent.tolist())
            log["row_ent"].append(rent.tolist()); log["sink"].append(sink.tolist())
            log["spectra"].append(spectra.tolist()); log["eff_rank"].append(eff)
            log["A_mean"].append(torch.stack(attn_maps, dim=1).mean(dim=0).tolist())
            if te_acc >= 0.95 and transition_step is None:
                transition_step = step
            if te_acc >= 0.999 and step >= (transition_step or 0) + 2000:
                break
    result = {"config": {"seed": seed, "regularize": regularize, "p": p, "train_frac": train_frac,
                         "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                         "lr": lr, "wd": wd, "tau": tau, "beta1": beta1},
              "transition_step": transition_step, "final_test_acc": log["test_acc"][-1],
              "final_train_acc": log["train_acc"][-1], "steps_run": log["step"][-1],
              "log": log}
    return result
