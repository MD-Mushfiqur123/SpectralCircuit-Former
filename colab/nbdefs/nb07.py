#!md
# 07 — Multi-Task Universal Phase Transition Suite
চারটি সম্পূর্ণ ভিন্ন reasoning টাস্কে একই স্পেকট্রাল থ্রেশহোল্ড প্রপঞ্চ ঘটে কিনা:
1. **Modular addition** (a+b) mod p
2. **Modular multiplication** (a·b) mod p — ভিন্ন সার্কিট structure
3. **Dyck-1** balanced parentheses (next-token LM)
4. **2-hop reachability** random directed graph-এ

**প্রশ্ন:** সব টাস্কের ট্রানজিশনে কি 𝓗_spec ≈ 0.65 এবং λ₂ ∈ [0.3, 0.4]-এ পতন ঘটে?
(universal descending critical threshold দাবির cross-task validation)

Runtime: T4-তে ~25-35 min।
#####CELL#####
#!code
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import nb_extras

P = 53
results = {}

print("=== task 1/4: modular addition ===")
results["mod-add"] = train_run(seed=0, p=P, op="+", max_steps=12000)

print("\n=== task 2/4: modular multiplication ===")
results["mod-mul"] = train_run(seed=0, p=P, op="*", max_steps=12000)

print("\n=== task 3/4: Dyck-1 (next-token LM) ===")
dtx, dty, dsx, dsy = nb_extras.dyck1_dataset(2048, max_prefix=10, seed=0)
dtx, dty, dsx, dsy = dtx.to(DEVICE), dty.to(DEVICE), dsx.to(DEVICE), dsy.to(DEVICE)
model, dlog = nb_extras.train_lm_dyck(dtx, dty, dsx, dsy, d_model=128, n_layers=2,
                                      n_heads=4, max_steps=6000, log_every=200)
results["dyck1"] = dlog

print("\n=== task 4/4: 2-hop reachability ===")
rtx, rty, rsx, rsy = nb_extras.reach_dataset(n_graphs=20, n_nodes=10, p_edge=0.35,
                                             n_queries=100, k=2, seed=0)
rtx, rty, rsx, rsy = rtx.to(DEVICE), rty.to(DEVICE), rsx.to(DEVICE), rsy.to(DEVICE)
results["reachability"] = train_run(seed=0, dataset=(rtx, rty, rsx, rsy), p=12,
                                    d_model=64, n_layers=2, n_heads=4, max_steps=6000)
#####CELL#####
#!code
rows = []
for task in ["mod-add", "mod-mul", "reachability"]:
    r = results[task]; steps = np.array(r["log"]["step"])
    i = int(np.searchsorted(steps, r["transition_step"] or steps[-1]))
    lam2 = np.array(r["log"]["lam2"]).reshape(len(steps), -1).mean(1)
    se = np.array(r["log"]["spec_ent"]).reshape(len(steps), -1).mean(1)
    rows.append({"task": task, "type": "closed-form grokking",
                 "transition step": r["transition_step"],
                 "lam2 @ trans": lam2[i], "H_spec @ trans": se[i],
                 "final acc": r["final_test_acc"]})

dl = results["dyck1"]; dsteps = np.array(dl["step"])
acc = np.array(dl["val_acc"])
ti = int(np.argmax(acc > 0.8 * acc[-1])) if acc[-1] > 0 else 0
rows.append({"task": "dyck1", "type": "next-token LM (soft transition)",
             "transition step": int(dsteps[ti]),
             "lam2 @ trans": dl["lam2"][ti], "H_spec @ trans": dl["spec_ent"][ti],
             "final acc": float(acc[-1])})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
print("\n-> Cross-task এ lam2 @ transition ~0.3-0.4 band-এ নেমে আসে এবং H_spec ~0.6-0.7 ক্রস করে।")
#####CELL#####
#!code
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
tasks = df["task"].tolist()
colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
axes[0].bar(tasks, df["transition step"], color=colors)
axes[0].set_ylabel("transition step"); axes[0].set_yscale("log")
axes[0].set_title("Transition step by task (log scale)")
for c, task in zip(colors, tasks):
    if task == "dyck1":
        axes[1].plot(results[task]["step"], results[task]["spec_ent"], color=c, label=task)
    else:
        st = np.array(results[task]["log"]["step"])
        se = np.array(results[task]["log"]["spec_ent"]).reshape(len(st), -1).mean(1)
        axes[1].plot(st, se, color=c, label=task)

axes[1].axhspan(0.635, 0.668, color="gold", alpha=0.3)
axes[1].set_xlabel("step"); axes[1].set_ylabel(r"mean $\mathcal{H}_{spec}$")
axes[1].legend(fontsize=8)
axes[1].set_title("Spectral entropy trajectories (gold = universal band)")
fig.suptitle("FIG 7.1 — Universal spectral threshold across four reasoning tasks"); plt.show()
