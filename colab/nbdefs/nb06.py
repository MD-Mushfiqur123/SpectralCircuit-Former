#!md
# 06 — Deep Transformers: Rank Collapse & Effective Rank Decay
গভীর ট্রান্সফরমারে (১২ লেয়ার) যেতে যেতে token representation কি একরকম হয়ে যায়
(**rank collapse / over-smoothing**)? তিনটি ভ্যারিয়েন্ট তুলনা:
1. **standard** (pre-LN + residual, Dyck-1 LM task, seq len ~15)
2. **no-residual** (residual সংযোগ ছাড়া — collapse-এর theoretical worst case)
3. **spectral-regularized** (λ₂ regularizer সহ)

পরিমাপ: per-layer **effective rank** (participation ratio) + mean pairwise **token cosine similarity**।

Runtime: T4-তে ~20-30 min (৩টি ১২-লেয়ার রান)।
#####CELL#####
#!code
import numpy as np, matplotlib.pyplot as plt
import nb_extras

MAXP = 14
tx, ty, sx, sy = nb_extras.dyck1_dataset(2048, max_prefix=MAXP, seed=0)
tx, ty, sx, sy = tx.to(DEVICE), ty.to(DEVICE), sx.to(DEVICE), sy.to(DEVICE)
print("Dyck-1 LM: train", tuple(tx.shape), "test", tuple(sx.shape))
SEQ = tx.shape[1]
#####CELL#####
#!code
def train_lm(train_x, train_y, test_x, test_y, n_layers=12, d_model=64, n_heads=4,
             lr=1e-3, wd=1.0, max_steps=3000, log_every=500, regularize=False,
             tau=0.35, beta1=1.0, use_residual=True):
    import torch.nn.functional as Fn
    torch.manual_seed(0)
    model = TinyTransformer(3, d_model, n_layers, n_heads, max_len=train_x.shape[1],
                            use_residual=use_residual).to(DEVICE)
    opt = torch.optim.AdamW(model.param_groups(wd), lr=lr, betas=(0.9, 0.98))
    probe = train_x[:64]
    log = {"step": [], "loss": [], "val_acc": [], "eff": [], "cos": [], "layer_ent": []}
    for step in range(1, max_steps + 1):
        logits, attn_maps, _ = model(train_x, full_logits=True)
        loss = Fn.cross_entropy(logits.reshape(-1, 3), train_y.reshape(-1))
        if regularize:
            reg, _ = fiedler_regularizer(torch.stack(attn_maps, dim=1)[probe], tau)
            loss = loss + beta1 * reg
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                tl, am, _ = model(test_x, full_logits=True)
                acc = (tl.argmax(-1) == test_y).float().mean().item()
            eff, cos = nb_extras.rank_profile(model, probe)
            lent = np.array([float(mean_row_entropy(A).mean()) for A in am])
            log["step"].append(step); log["loss"].append(float(loss)); log["val_acc"].append(acc)
            log["eff"].append(eff); log["cos"].append(cos); log["layer_ent"].append(lent)
            print(f"  step {step:5d} | loss {loss.item():.4f} | val_acc {acc:.3f} | eff_rank_last {eff[-1]:.2f}", flush=True)
    return model, log

print("=== (1) standard (residual) ===")
m_std, log_std = train_lm(tx, ty, sx, sy, use_residual=True)
print("\n=== (2) no-residual (collapse demo) ===")
m_nores, log_nores = train_lm(tx, ty, sx, sy, use_residual=False, max_steps=1500)
print("\n=== (3) spectral-regularized ===")
m_spec, log_spec = train_lm(tx, ty, sx, sy, use_residual=True, regularize=True)
#####CELL#####
#!code
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
depths = np.arange(len(log_std["eff"][0]))
for log, color, label in [(log_std, "gray", "standard"), (log_nores, "crimson", "no-residual"),
                          (log_spec, "royalblue", "spectral-regularized")]:
    eff = np.stack(log["eff"]).mean(0)
    cos = np.stack(log["cos"]).mean(0)
    axes[0].plot(depths, eff, "o-", color=color, label=label)
    axes[1].plot(depths, cos, "o-", color=color, label=label)
axes[0].set_xlabel("residual depth"); axes[0].set_ylabel("effective rank (participation ratio)")
axes[0].set_title("FIG 6.1 — Effective rank vs depth")
axes[1].set_xlabel("residual depth"); axes[1].set_ylabel("mean pairwise token cosine")
axes[1].set_title("FIG 6.2 — Token homogeneity vs depth (collapse detector)")
for ax in axes:
    ax.legend(fontsize=8)
plt.show()

print("\nFinal values (avg over training tail):")
for log, label in [(log_std, "standard"), (log_nores, "no-residual"), (log_spec, "spectral")]:
    eff = np.stack(log["eff"])[:, -1].mean()
    cos = np.stack(log["cos"])[:, -1].mean()
    print(f"  {label:20s} last-layer eff_rank = {eff:.2f}/{SEQ}   token cosine = {cos:.3f}")
print("\nExpected: no-residual model-এ rank collapse স্পষ্ট; residual + pre-LN এ নিয়ন্ত্রিত;")
print("long-context (RULER/NIAH, N>=512) এ collapse আরও প্রকট হয়।")
