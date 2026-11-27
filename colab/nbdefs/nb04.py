#!md
# 04 — SGSA: Spectral-Guided Attention Regularizer (τ Ablation)
Differentiable Fiedler regularizer: `L_total = L_CE + β₁ · relu(τ − λ₂)`,
যেখানে λ₂ প্রতিটি অ্যাটেনশন গ্রাফের algebraic connectivity (torch.linalg.eigvalsh
দিয়ে GPU-তে, fully differentiable backprop)।

এই নোটবুকে **τ sweep** করে দেখি কোন τ মডেলের ট্রানজিশন, final λ₂ এবং accuracy-তে প্রভাব ফেলে।
**গুরুত্বপূর্ণ (sandbox ফলাফল):** regularizer বড় speedup দেয় না (~1.05×),
কিন্তু λ₂-কে উঁচুতে ধরে রাখে — এটিই আসল বৈজ্ঞানিক ফাইন্ডিং।

Runtime: T4-তে ~25-35 min (৫টি রান)।
#####CELL#####
#!code
TAUS = [None, 0.05, 0.20, 0.35, 0.50]
RUNS = {}
for tau in TAUS:
    name = f"tau_{tau}" if tau is not None else "baseline"
    print(f"\n=== {name} ===", flush=True)
    if tau is None:
        RUNS[name] = train_run(seed=0, p=53, max_steps=12000)
    else:
        RUNS[name] = train_run(seed=0, p=53, regularize=True, tau=tau, beta1=1.0, max_steps=12000)
#####CELL#####
#!code
import numpy as np, matplotlib.pyplot as plt, pandas as pd

rows = []
for name, r in RUNS.items():
    steps = np.array(r["log"]["step"])
    lam2 = np.array(r["log"]["lam2"]).reshape(len(steps), -1)
    rows.append({"config": name,
                 "tau": r["config"]["tau"],
                 "transition step": r["transition_step"],
                 "final lam2 (mean)": float(lam2[-1].mean()),
                 "final test acc": r["final_test_acc"]})
df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(df)))
for c, (_, r) in zip(cmap, df.iterrows()):
    rr = RUNS[r["config"]]; steps = np.array(rr["log"]["step"])
    axes[0].plot(steps, rr["log"]["test_acc"], color=c, lw=1.6, label=f"{r['config']} (t={r['transition step']})")
    axes[1].plot(steps, np.array(rr["log"]["lam2"]).reshape(len(steps), -1).mean(1), color=c, lw=1.6)
axes[0].set_xlabel("step"); axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=6); axes[0].set_ylim(-0.05, 1.05)
axes[0].set_title("Test accuracy vs τ")
axes[1].set_xlabel("step"); axes[1].set_ylabel(r"mean $\lambda_2$")
axes[1].set_title("Algebraic connectivity held up by regularizer")
fig.suptitle("FIG 4.1 — SGSA τ-sweep: accuracy and λ₂ dynamics"); plt.show()

sub = df[df["config"] != "baseline"]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(sub["tau"], sub["transition step"], "o-", color="crimson", label="transition step")
ax2 = ax.twinx()
ax2.plot(sub["tau"], sub["final lam2 (mean)"], "s--", color="royalblue", label=r"final $\lambda_2$")
ax2.grid(False); ax2.set_ylabel(r"final $\lambda_2$")
ax.set_xlabel(r"regularizer threshold $\tau$"); ax.set_ylabel("transition step")
ax.set_title("FIG 4.2 — Transition step and final λ₂ vs τ")
ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right"); plt.show()
print("\nNote: relu(τ−λ2) শুধু তখনই active যখন λ2 < τ. τ বাড়ালে regularizer বেশি bind করে এবং final λ2 বাড়ে,")
print("কিন্তু grokking-এর গতি উল্লেখযোগ্যভাবে বাড়ে না — এটিই আমাদের replicated finding.")
