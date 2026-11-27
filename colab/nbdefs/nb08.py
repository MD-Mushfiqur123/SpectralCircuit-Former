#!md
# 08 — Causal Circuit Surgery (Ablation & Intervention)
স্পেকট্রালভাবে চিহ্নিত head-গুলো কি সত্যিই **causally important**?
1. **Exhaustive ablation** — প্রতিটি head আলাদাভাবে zero-out করে accuracy-drop মাপা (ground truth causal importance)
2. **Spectral ranking** — শুধু spectral metrics দিয়ে head importance অনুমান করা যায় কিনা (Spearman correlation)
3. **Surgery** — top-3 causal heads zero করলে কী হয় vs bottom-3 vs random-3; boost (mask=2.0) করলে কী হয়

Runtime: T4-তে ~12-15 min।
#####CELL#####
#!code
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
import nb_extras

res = train_run(seed=0, p=53, d_model=128, n_layers=2, n_heads=4, max_steps=12000)
model = res["model"]
_, _, tx, ty = build_dataset(p=53, train_frac=0.4, seed=0)
tx, ty = tx.to(DEVICE), ty.to(DEVICE)
base = res["final_test_acc"]
print("base test acc:", round(base, 4))
#####CELL#####
#!code
base_eval, imp = nb_extras.ablation_importance(model, tx, ty)
L, H = imp.shape
fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(imp, cmap="Reds")
ax.set_xlabel("head"); ax.set_ylabel("layer")
ax.set_title("FIG 8.1 — Causal importance (accuracy drop when head zeroed)")
for l in range(L):
    for h in range(H):
        ax.text(h, l, f"{imp[l, h]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="acc drop"); plt.show()

flat = [(l, h, imp[l, h]) for l in range(L) for h in range(H)]
print("\nHead ranking by causal importance (top 5):")
for l, h, v in sorted(flat, key=lambda x: -x[2])[:5]:
    print(f"  L{l}H{h}: drop = {v:.4f}")

roles = nb_extras.head_roles(model, tx[:64])
lam2 = np.array([r["lam2"] for r in roles]).reshape(L, H)
sink = np.array([r["sink"] for r in roles]).reshape(L, H)
rho_l2 = stats.spearmanr(imp.flatten(), lam2.flatten()).statistic
rho_sink = stats.spearmanr(imp.flatten(), sink.flatten()).statistic
print(f"\nSpearman(causal importance, head lambda2)  = {rho_l2:+.3f}")
print(f"Spearman(causal importance, sink fraction) = {rho_sink:+.3f}")
print("-> Spectral metrics শুধু দেখে দেখে causal head চেনা যায় কিনা তার পরিমাপ।")
#####CELL#####
#!code
top3 = [(l, h) for l, h, v in sorted(flat, key=lambda x: -x[2])[:3]]
bot3 = [(l, h) for l, h, v in sorted(flat, key=lambda x: x[2])[:3]]
rng = np.random.default_rng(0)
all_heads = [(l, h) for l in range(L) for h in range(H)]
rand3 = [rng.permutation(all_heads)[:3].tolist() for _ in range(5)]

def surgery(pairs, value):
    hm = torch.ones(L, H, device=DEVICE)
    for (l, h) in pairs:
        hm[l, h] = value
    return nb_extras.head_mask_eval(model, tx, ty, hm)

acc_top0 = surgery(top3, 0.0)
acc_bot0 = surgery(bot3, 0.0)
acc_rand0 = float(np.mean([surgery(p, 0.0) for p in rand3]))
acc_top2 = surgery(top3, 2.0)

df = pd.DataFrame([
    {"intervention": "baseline (no surgery)", "acc": base},
    {"intervention": "zero top-3 causal heads", "acc": acc_top0},
    {"intervention": "zero bottom-3 heads", "acc": acc_bot0},
    {"intervention": "zero random-3 heads (avg of 5)", "acc": acc_rand0},
    {"intervention": "boost top-3 (mask=2.0)", "acc": acc_top2},
])
print(df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 3.2))
colors = ["gray", "crimson", "seagreen", "gold", "royalblue"]
ax.bar(df["intervention"], df["acc"], color=colors)
ax.set_ylim(0, 1.05); ax.set_ylabel("test accuracy")
ax.set_title("FIG 8.2 — Causal circuit surgery")
ax.tick_params(axis="x", rotation=12)
for i, v in enumerate(df["acc"]):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
plt.show()
print("\nExpected: top-3 causal heads zero করলে accuracy ধসে পড়ে, bottom/random এ প্রায় অপরিবর্তিত")
print("— এটাই 'discovered circuit is causally necessary' দাবির প্রমাণ। Boost-এ error pattern বদলায়।")
