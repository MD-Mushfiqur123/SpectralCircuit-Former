"""Notebook-specific extra functions (curvature, clustering, tasks, circuits, ablation)."""
import numpy as np
import torch
import torch.nn.functional as F


def marchenko_pastur_pdf(x, q):
    x = np.asarray(x, float)
    lam_m, lam_p = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    y = np.zeros_like(x)
    m = (x >= lam_m) & (x <= lam_p)
    y[m] = np.sqrt(np.maximum((lam_p - x[m]) * (x[m] - lam_m), 0)) / (2 * np.pi * q * x[m])
    return y


def fiedler_partition(A):
    """2-way spectral clustering of a (N,N) attention matrix via Fiedler vector sign."""
    d = A.sum(-2).clamp_min(1e-8).rsqrt()
    B = 0.5 * (A / d.unsqueeze(-2) + (A / d.unsqueeze(-1)).transpose(-1, -2))
    L = torch.eye(A.shape[-1], device=A.device) - B
    evals, evecs = torch.linalg.eigh(L)
    partition = (evecs[:, 1] > 0).long()
    return partition, evals


def _hop_distance(A):
    An = (np.asarray(A, float) > 0.01).astype(float)
    An = np.maximum(An, An.T)
    An = np.maximum(An, np.eye(len(An)))
    D = np.full((len(An), len(An)), np.inf)
    for s in range(len(An)):
        D[s, s] = 0.0
        seen = np.zeros(len(An), bool); seen[s] = True
        frontier = np.array([s])
        d = 0
        while len(frontier):
            d += 1
            nxt = np.unique(np.where(An[frontier].any(0))[0])
            nxt = nxt[~seen[nxt]]
            if not len(nxt):
                break
            D[s, nxt] = d
            seen[nxt] = True
            frontier = nxt
    D[~np.isfinite(D)] = len(An)
    return D


def ollivier_ricci(A, alpha=0.5):
    """Exact Ollivier-Ricci curvature per edge using lazy random walks and W1
    over the hop-distance ground metric (small graphs only, N <= ~16)."""
    from scipy.optimize import linprog
    A = np.asarray(A, float)
    N = len(A)
    P = A / np.maximum(A.sum(-1, keepdims=True), 1e-12)
    Dg = _hop_distance(A)
    kappa = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j or P[i, j] < 1e-6:
                continue
            mi = np.zeros(N); mi[i] = 1 - alpha; mi += alpha * P[i]
            mj = np.zeros(N); mj[j] = 1 - alpha; mj += alpha * P[j]
            c = Dg.reshape(-1)
            Amat = np.kron(np.eye(N), np.ones((1, N))) - np.kron(np.ones((1, N)), np.eye(N))
            res = linprog(c, A_eq=Amat, b_eq=mi - mj, bounds=[(0, None)] * (N * N), method="highs")
            if res.success:
                kappa[i, j] = 1 - res.fun / max(Dg[i, j], 1e-9)
    return kappa


def forman_ricci(A):
    """Simplified weighted Forman-Ricci on the symmetrized attention graph."""
    W = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    np.fill_diagonal(W, 0)
    N = len(W)
    F = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if W[i, j] < 1e-9:
                continue
            w_e = W[i, j]
            s_i = W[i].sum() + 1e-12; s_j = W[j].sum() + 1e-12
            t_i = sum(np.sqrt(w_e * W[i, k]) for k in range(N) if k != j and W[i, k] > 1e-9)
            t_j = sum(np.sqrt(w_e * W[j, k]) for k in range(N) if k != i and W[j, k] > 1e-9)
            F[i, j] = F[j, i] = w_e * (w_e / s_i + w_e / s_j - t_i / s_i - t_j / s_j)
    return F


def head_roles(model, probe, feature_names=None):
    """Classify heads by dominant attention behaviour. Returns dict of arrays."""
    with torch.no_grad():
        _, attn_maps, _ = model(probe)
    rows = []
    for l, A in enumerate(attn_maps):
        Am = A.mean(dim=0)
        for h in range(Am.shape[0]):
            col = Am[h].mean(0)
            dom = int(col.argmax())
            rows.append({"layer": l, "head": h, "dominant_col": dom,
                         "lam2": float(fiedler_value(Am[h:h + 1])[0]),
                         "spec_ent": float(spectral_entropy(Am[h:h + 1])[0]),
                         "sink": float(sink_fraction(Am[h:h + 1])[0])})
    return rows


def layer_circuit_corr(model, probe):
    """Head-to-head attention-pattern cosine similarity between consecutive layers
    (candidates for the multi-layer circuit DAG)."""
    with torch.no_grad():
        _, attn_maps, _ = model(probe)
    L = len(attn_maps)
    H = attn_maps[0].shape[1]
    C = np.zeros((L - 1, H, H))
    for l in range(L - 1):
        X = attn_maps[l].mean(0).reshape(H, -1)
        Y = attn_maps[l + 1].mean(0).reshape(H, -1)
        X = X - X.mean(1, keepdims=True); Y = Y - Y.mean(1, keepdims=True)
        C[l] = (X @ Y.T) / (np.linalg.norm(X, axis=1, keepdims=True) *
                            np.linalg.norm(Y, axis=1, keepdims=True).T + 1e-9)
    return C


@torch.no_grad()
def ablation_importance(model, seq, targets, bs=4096):
    """Per-head causal importance: accuracy drop when head output is zeroed."""
    L = model.n_layers
    H = model.blocks[0].n_heads
    base = evaluate(model, seq, targets, bs)
    imp = np.zeros((L, H))
    for l in range(L):
        for h in range(H):
            hm = torch.ones(L, H, device=seq.device)
            hm[l, h] = 0.0
            correct, n = 0, 0
            for i in range(0, len(seq), bs):
                logits, _, _ = model(seq[i:i + bs], head_mask=hm)
                correct += (logits.argmax(-1) == targets[i:i + bs]).sum().item()
                n += len(seq[i:i + bs])
            imp[l, h] = base - correct / n
    return base, imp


@torch.no_grad()
def head_mask_eval(model, seq, targets, head_mask, bs=4096):
    correct, n = 0, 0
    for i in range(0, len(seq), bs):
        logits, _, _ = model(seq[i:i + bs], head_mask=head_mask)
        correct += (logits.argmax(-1) == targets[i:i + bs]).sum().item()
        n += len(seq[i:i + bs])
    return correct / n


def rank_profile(model, probe):
    """Per-layer effective rank and mean pairwise token cosine similarity."""
    with torch.no_grad():
        N = probe.shape[1]
        h = model.tok(probe) + model.pos(torch.arange(N, device=probe.device))
        eff = [float(effective_rank(h).mean())]
        cos = []
        hc = h - h.mean(dim=1, keepdim=True)
        hc = hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cos.append(float((hc @ hc.transpose(-1, -2)).mean()))
        for blk in model.blocks:
            h, _ = blk(h)
            eff.append(float(effective_rank(h).mean()))
            hc = h - h.mean(dim=1, keepdim=True)
            hc = hc / hc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            cos.append(float((hc @ hc.transpose(-1, -2)).mean()))
    return np.array(eff), np.array(cos)


def dyck1_dataset(n_samples, max_prefix=8, seed=0, train_frac=0.7):
    """Dyck-1 balanced parentheses: sequence + next-token targets (LM style).
    Vocab: 0='(' 1=')' 2=BOS. Target at position t is token t+1 (shifted)."""
    g = np.random.default_rng(seed)
    seqs, tgts = [], []
    for _ in range(n_samples):
        depth, s = 0, [2]
        for _ in range(max_prefix):
            if depth == 0:
                s.append(0); depth += 1
            elif depth < max_prefix // 2 and g.random() < 0.55:
                s.append(0); depth += 1
            else:
                s.append(1); depth -= 1
        s.append(1); depth -= 1
        seqs.append(s[:-1]); tgts.append(s[1:])
    seqs = torch.tensor(seqs); tgts = torch.tensor(tgts)
    perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed))
    n_train = int(n_samples * train_frac)
    return seqs[perm[:n_train]], tgts[perm[:n_train]], seqs[perm[n_train:]], tgts[perm[n_train:]]


def reach_dataset(n_graphs=40, n_nodes=8, p_edge=0.4, n_queries=150, k=2, seed=0, train_frac=0.7):
    """k-hop reachability: fixed random graph per dataset; query [u, ?, v] -> bit.
    Vocab: 0..n_nodes-1 = nodes, n_nodes = QUERY token, n_nodes+1 = unused; target in {0,1}."""
    g = np.random.default_rng(seed)
    G = (g.random((n_nodes, n_nodes)) < p_edge).astype(float)
    np.fill_diagonal(G, 1.0)
    reach = G.copy()
    for _ in range(k - 1):
        reach = ((reach @ G) > 0).astype(float)
    pairs = np.array([(u, v) for u in range(n_nodes) for v in range(n_nodes)])
    sel = g.choice(len(pairs), size=min(n_queries, len(pairs)), replace=False)
    seq = torch.stack([torch.tensor(pairs[sel, 0]), torch.full((len(sel),), n_nodes),
                       torch.tensor(pairs[sel, 1])], dim=1)
    tgt = torch.tensor([int(reach[u, v]) for u, v in pairs[sel]])
    perm = torch.randperm(len(sel), generator=torch.Generator().manual_seed(seed))
    n_train = int(len(sel) * train_frac)
    return seq[perm[:n_train]], tgt[perm[:n_train]], seq[perm[n_train:]], tgt[perm[n_train:]]


def train_lm_dyck(train_x, train_y, test_x, test_y, d_model=64, n_layers=2, n_heads=4,
                  lr=1e-3, wd=1.0, max_steps=4000, log_every=200):
    """Next-token LM training for Dyck-1 with full-sequence cross-entropy.
    Requires TinyTransformer, DEVICE, spectral_probe in the namespace (notebook core cell)."""
    import torch.nn.functional as Fn
    torch.manual_seed(0)
    model = TinyTransformer(3, d_model, n_layers, n_heads, max_len=train_x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.param_groups(wd), lr=lr, betas=(0.9, 0.98))
    probe = train_x[:64]
    log = {"step": [], "loss": [], "val_acc": [], "lam2": [], "spec_ent": []}
    for step in range(1, max_steps + 1):
        logits, attn_maps, _ = model(train_x, full_logits=True)
        loss = Fn.cross_entropy(logits.reshape(-1, 3), train_y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                tl, am, _ = model(test_x, full_logits=True)
                acc = (tl.argmax(-1) == test_y).float().mean().item()
            lam2, sent, _, _, _, _ = spectral_probe(model, probe)
            log["step"].append(step); log["loss"].append(float(loss)); log["val_acc"].append(acc)
            log["lam2"].append(float(lam2.mean())); log["spec_ent"].append(float(sent.mean()))
            print(f"  step {step:5d} | lm_loss {loss.item():.4f} | val_next_token_acc {acc:.3f} | lam2 {lam2.mean():.3f}", flush=True)
    return model, log
