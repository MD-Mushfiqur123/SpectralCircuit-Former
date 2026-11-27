#!md
# 05 — Automated Circuit Discovery via Spectral Subgraph Decomposition
স্পেকট্রাল পদ্ধতিতে স্বয়ংক্রিয় circuit extraction:
1. **Head roles** — কোন head কোন টোকেন/ভূমিকা দেখছে (dominant column, sink, entropy)
2. **Fiedler partition** — প্রতিটি head-এর attention graph-এর token communities
3. **Layer-to-layer circuit DAG** — কোন (l,h) head কোন (l+1,h') head-কে তথ্য পাঠাচ্ছে (attention-pattern correlation)

Runtime: T4-তে ~10 min।
#####CELL#####
#!code
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import nb_extras

res = train_run(seed=0, p=53, d_model=128, n_layers=3, n_heads=4, max_steps=12000)
model, probe = res["model"], None
train_x, _, _, _ = build_dataset(p=53, train_frac=0.4, seed=0)
probe = train_x[:64].to(DEVICE)
t = res["transition_step"]
print("transition @", t, "| final test acc", round(res["final_test_acc"], 4))
#####CELL#####
#!code
rows = nb_extras.head_roles(model, probe)
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 0.6 * (len(rows) // 4) + 2))
sc = ax.scatter(df["head"] + 0.18 * df["layer"], df["layer"], c=df["spec_ent"],
                s=90 + 260 * df["sink"], cmap="coolwarm", edgecolor="k", linewidth=0.4)
fig.colorbar(sc, ax=ax, label=r"spectral entropy $\mathcal{H}_{spec}$")
ax.set_yticks(range(res["config"]["n_layers"]))
ax.set_xlabel("head index (x-offset by layer)"); ax.set_ylabel("layer")
ax.set_title("FIG 5.1 — Head specialization map (size = sink fraction)")
plt.show()
#####CELL#####
#!code
with torch.no_grad():
    _, attn_maps, _ = model(probe)
L, H = len(attn_maps), attn_maps[0].shape[1]
fig, axes = plt.subplots(L, H, figsize=(3 * H, 3 * L))
for l in range(L):
    for h in range(H):
        A = attn_maps[l][:, h].mean(0)
        part, _ = fiedler_partition(A)
        order = torch.argsort(part)
        ax = axes[l, h]
        ax.imshow(A.cpu().numpy()[order][:, order], cmap="viridis", vmin=0)
        ax.set_title(f"L{l}H{h} lam2={float(fiedler_value(A)):.2f}", fontsize=8)
        ax.axis("off")
fig.suptitle("FIG 5.2 — Community-reordered attention graphs per head (Fiedler spectral clustering)")
fig.tight_layout(); plt.show()
#####CELL#####
#!code
C = nb_extras.layer_circuit_corr(model, probe)
fig, axes = plt.subplots(1, max(L - 1, 1), figsize=(4 * max(L - 1, 1), 3.4))
axes = np.atleast_1d(axes)
for l in range(L - 1):
    im = axes[l].imshow(C[l], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[l].set_title(f"layer {l} -> {l+1}\nhead-to-head correlation")
    axes[l].set_xlabel(f"head in layer {l+1}"); axes[l].set_ylabel(f"head in layer {l}")
    fig.colorbar(im, ax=axes[l], fraction=0.046)
fig.suptitle("FIG 5.3 — Multi-layer circuit DAG weights (attention-pattern correlation)")
fig.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 3.5))
ys = {l: 1.0 - l / max(L - 1, 1) for l in range(L)}
for l in range(L - 1):
    flat = np.dstack(np.unravel_index(np.argsort(-np.abs(C[l]), axis=None), C[l].shape))[0]
    for (h1, h2) in flat[:3]:
        w = C[l][h1, h2]
        ax.annotate("", xy=(h2, ys[l + 1] - 0.06), xytext=(h1, ys[l] + 0.06),
                    arrowprops=dict(arrowstyle="->", lw=1 + 6 * abs(w),
                                    color="crimson" if w > 0 else "royalblue",
                                    alpha=0.2 + 0.8 * abs(w)))
for l in range(L):
    ax.scatter(range(H), [ys[l]] * H, s=400, c="seagreen", zorder=3)
    for h in range(H):
        ax.text(h, ys[l], f"L{l}H{h}", ha="center", va="center", color="white", fontsize=7, zorder=4)
ax.set_ylim(-0.2, 1.2); ax.axis("off")
ax.set_title("FIG 5.4 — Top circuit edges per layer (red=excitatory, blue=inhibitory)")
plt.show()
print("Interpretation: উচ্চ |correlation| এজ = একসাথে কাজ করা head-জোড় = candidate reasoning circuit।")
