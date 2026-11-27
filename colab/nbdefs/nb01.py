#!md
# 01 — Spectral Metrics Laboratory
**SpectralCircuit-Former** গবেষণার ভিত্তি: অ্যাটেনশন ম্যাট্রিক্সের ওপর Spectral Graph Theory।

এই নোটবুকে canonical attention topologies-এর (uniform, sink-star, block-structured, ring, random) সম্পূর্ণ spectral fingerprint তৈরি হয়:
- **Fiedler eigenvalue λ₂** (algebraic connectivity — information mixing speed)
- **Von Neumann spectral entropy 𝓗_spec** (star/sink vs modular structure)
- **Ollivier-Ricci curvature** (bottleneck detection, exact W1/LP-based)
- **Forman-Ricci curvature** (closed-form)

Runtime: ~2 min (T4)। কোনো দীর্ঘ ট্রেনিং নেই।
#####CELL#####
#!code
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import nb_extras

def graph_zoo(N=8):
    zoo = {}
    zoo["uniform (complete)"] = torch.full((N, N), 1.0 / N)
    A = torch.full((N, N), 0.02); A[:, 0] = 0.86; A[0, 0] = 0.14
    zoo["sink-star"] = A
    B = torch.zeros(N, N); half = N // 2
    B[:half, :half] = 0.45; B[half:, half:] = 0.45
    B += 0.05 * torch.eye(N); B += 0.05
    zoo["block-structured"] = B / B.sum(-1, keepdim=True)
    C = torch.zeros(N, N)
    for i in range(N):
        C[i, (i - 1) % N] = 0.5; C[i, i] = 0.5
    zoo["ring"] = C
    g = torch.Generator().manual_seed(0)
    R = torch.rand(N, N, generator=g) + 0.01
    zoo["random"] = R / R.sum(-1, keepdim=True)
    return zoo

zoo = graph_zoo(8)
rows = []
for name, A in zoo.items():
    ko = nb_extras.ollivier_ricci(A.numpy())
    kf = nb_extras.forman_ricci(A.numpy())
    iu = np.triu_indices(len(A), 1)
    rows.append({"graph": name,
                 "Fiedler lambda2": float(fiedler_value(A)),
                 "spectral entropy": float(spectral_entropy(A)),
                 "row entropy": float(mean_row_entropy(A)),
                 "sink fraction": float(sink_fraction(A)),
                 "mean Ollivier kappa": float(ko[iu].mean()),
                 "mean Forman kappa": float(kf[iu].mean())})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
#####CELL#####
#!code
colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].bar(df["graph"], df["Fiedler lambda2"], color=colors)
axes[0].set_ylabel(r"$\lambda_2$"); axes[0].tick_params(axis="x", rotation=20)
axes[0].set_title("Algebraic connectivity")
axes[1].bar(df["graph"], df["spectral entropy"], color=colors)
axes[1].set_ylabel(r"$\mathcal{H}_{spec}$"); axes[1].tick_params(axis="x", rotation=20)
axes[1].set_title("Spectral entropy")
fig.suptitle("FIG A — Spectral fingerprint of canonical attention topologies")
fig.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(5.5, 4))
for c, (_, r) in zip(colors, df.iterrows()):
    ax.scatter(r["spectral entropy"], r["Fiedler lambda2"], s=140, color=c)
    ax.annotate(r["graph"], (r["spectral entropy"], r["Fiedler lambda2"]),
                fontsize=7, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel(r"spectral entropy $\mathcal{H}_{spec}$"); ax.set_ylabel(r"Fiedler $\lambda_2$")
ax.set_title("FIG B — Spectral phase plane of attention topologies")
fig.tight_layout(); plt.show()
#####CELL#####
#!md
### একটি বাস্তব (untrained + ২০০-স্টেপ ট্রেইন করা) মিনি-মডেলের অ্যাটেনশন গ্রাফ
প্রতিটি head-এর attention graph, Fiedler partition অনুযায়ী community-reordered করে দেখানো হয় —
sink-star (uniform) বনাম modular block structure চোখে দেখা যায়।
#####CELL#####
#!code
torch.manual_seed(0)
probe = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3], [2, 1, 0, 3], [5, 1, 7, 3]], device=DEVICE)
res = train_run(seed=0, p=13, d_model=64, n_layers=2, n_heads=4, max_steps=200, log_every=100)
model = res["model"]
with torch.no_grad():
    _, attn_maps, _ = model(probe)
L, H = len(attn_maps), attn_maps[0].shape[1]
fig, axes = plt.subplots(L, H, figsize=(3 * H, 3 * L))
for l in range(L):
    for h in range(H):
        A = attn_maps[l][0, h]
        part, evals = fiedler_partition(A)
        order = torch.argsort(part)
        ax = axes[l, h] if L > 1 else axes[h]
        ax.imshow(A.cpu().numpy()[order][:, order], cmap="viridis", vmin=0)
        ax.set_title(f"L{l}H{h}  l2={float(fiedler_value(A)):.2f}  H={float(spectral_entropy(A)):.2f}", fontsize=8)
        ax.axis("off")
fig.suptitle("FIG C — Per-head attention graphs, community-reordered (Fiedler partition)")
fig.tight_layout(); plt.show()
print("Interpretation: sink heads (low H, star pattern) vs modular heads (block pattern).")
