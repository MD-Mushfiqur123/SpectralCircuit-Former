#!md
# 03 — Eigenvalue Spectral Density & Marchenko-Pastur Manifolds
ট্রেনিং জুড়ে অ্যাটেনশন অ্যাডজাসেন্সির eigenvalue density-র বিবর্তন, র‍্যান্ডম ম্যাট্রিক্স
তত্ত্বের **Marchenko-Pastur বণ্টনের** সাথে তুলনা, representation spectrum, এবং
(λ₂, 𝓗_spec) phase-space-এ ট্রেইনিং ট্র্যাজেক্টরি।

Runtime: T4-তে ~5-8 min।
#####CELL#####
#!code
import numpy as np, matplotlib.pyplot as plt
import nb_extras

res = train_run(seed=0, p=97, train_frac=0.4, d_model=128, n_layers=2, n_heads=4, max_steps=8000)
steps = np.array(res["log"]["step"])
spectra = np.array(res["log"]["spectra"])
print("logged", len(steps), "checkpoints; spectra shape", spectra.shape)
#####CELL#####
#!code
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for l in range(spectra.shape[1]):
    H_l = spectra[:, l].reshape(len(steps), -1)
    im = axes[l].imshow(H_l.T, aspect="auto", origin="lower", cmap="magma",
                        extent=[steps[0], steps[-1], 0, H_l.shape[1]])
    axes[l].set_title(f"layer {l}: sym-adjacency eigenvalues (heads sorted)")
    axes[l].set_xlabel("training step")
axes[0].set_ylabel("eigenvalue index")
fig.colorbar(im, ax=axes, label="eigenvalue")
fig.suptitle("FIG 3.1 — Eigenvalue spectrum evolution of attention manifolds")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
t = res["transition_step"] or steps[-1]
for idx, color, label in [(1, "gray", "step 1 (init)"),
                          (int(np.searchsorted(steps, t * 0.6)), "orange", "memorization"),
                          (-1, "royalblue", "generalization")]:
    h = ax.hist(spectra[idx].reshape(-1), bins=40, alpha=0.55, color=color, label=f"{label} (step {steps[idx]})", density=True)
q = 1.0
xs = np.linspace(0.01, 3.5, 300)
ax.plot(xs, nb_extras.marchenko_pastur_pdf(xs, q) / 3, color="k", ls="--", lw=1.2, label="Marchenko-Pastur (scaled)")
ax.set_xlabel(r"eigenvalue of $\tilde A$"); ax.set_ylabel("density")
ax.set_title("FIG 3.2 — Spectral density: memorization vs generalization vs MP law")
ax.legend(fontsize=8); plt.show()
#####CELL#####
#!code
lam2 = np.array(res["log"]["lam2"]).reshape(len(steps), -1).mean(1)
se = np.array(res["log"]["spec_ent"]).reshape(len(steps), -1).mean(1)
acc = np.array(res["log"]["test_acc"])
eff = np.array(res["log"]["eff_rank"]).mean(1)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sc = axes[0].scatter(se, lam2, c=steps, cmap="plasma", s=22)
if res["transition_step"]:
    i = int(np.searchsorted(steps, res["transition_step"]))
    axes[0].plot(se[i], lam2[i], "k*", ms=16, label="transition")
axes[0].legend()
axes[0].set_xlabel(r"spectral entropy $\mathcal{H}_{spec}$"); axes[0].set_ylabel(r"Fiedler $\lambda_2$")
axes[0].set_title("FIG 3.3 — Training trajectory in spectral phase space")
fig.colorbar(sc, ax=axes[0], label="training step")

axes[1].plot(steps, acc, color="crimson", lw=1.8, label="test accuracy")
axes[1].plot(steps, eff / max(eff.max(), 1e-9), color="seagreen", lw=1.8, label="effective rank (norm.)")
axes[1].set_xlabel("training step"); axes[1].legend(fontsize=8)
axes[1].set_title("FIG 3.4 — Generalization vs representation compression")
plt.show()
print("Grokking এর সাথে representation compress হয় (effective rank কমে) — স্পেকট্রাল থ্রেশহোল্ড ক্রসিংয়ের পাশাপাশি।")
