"""Analysis + publication figures for the SpectralCircuit-Former experiment suite."""
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def load_all():
    runs = {}
    for f in glob.glob(os.path.join(RESULTS, "*.json")):
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            runs[name] = json.load(fh)
    return runs


def mean_curve(runs, prefix, key):
    steps = None
    curves = []
    for name, r in runs.items():
        if name.startswith(prefix):
            steps = np.array(r["log"]["step"])
            v = np.array(r["log"][key])
            curves.append(v.reshape(len(v), -1).mean(axis=1))
    n = min(len(c) for c in curves)
    return steps[:n], np.stack([c[:n] for c in curves])


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def fig_phase_transition(runs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, seed in zip(axes, [0, 1, 2]):
        r0 = runs[f"baseline_seed{seed}"]
        steps = np.array(r0["log"]["step"])
        lam2 = np.array(r0["log"]["lam2"]).reshape(len(steps), -1).mean(axis=1)
        acc = np.array(r0["log"]["test_acc"])
        ax.plot(steps, acc, color="crimson", lw=1.8, label="test accuracy")
        ax.plot(steps, np.array(runs[f"baseline_seed{seed}"]["log"]["train_acc"]),
                color="crimson", lw=1.0, ls="--", alpha=0.5, label="train accuracy")
        ax2 = ax.twinx()
        ax2.plot(steps, lam2, color="royalblue", lw=1.8, label=r"$\bar\lambda_2$ (Fiedler)")
        ax2.grid(False)
        t = runs[f"baseline_seed{seed}"]["transition_step"]
        if t:
            ax.axvline(t, color="k", ls=":", lw=1)
        ax.set_title(f"seed {seed}  (transition @ {t})")
        ax.set_xlabel("training step")
        ax.set_ylim(-0.05, 1.05)
        ax2.set_ylim(0, None)
    axes[0].set_ylabel("accuracy")
    axes[0].text(0.02, 0.05, "red: accuracy", transform=axes[0].transAxes, color="crimson")
    axes[0].text(0.02, 0.12, "blue: mean $\\lambda_2$", transform=axes[0].transAxes, color="royalblue")
    fig.suptitle("Spectral phase transition: Fiedler eigenvalue vs. generalization (mod-$p$ addition, 3 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_phase_transition.png"))
    plt.close(fig)


def fig_spectral_density(runs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, seed in zip(axes, [0, 1, 2]):
        r = runs[f"baseline_seed{seed}"]
        steps = np.array(r["log"]["step"])
        spec = np.array(r["log"]["spectra"])
        t = r["transition_step"] or steps[-1]
        pre_idx = int(np.searchsorted(steps, max(t * 0.5, steps[1])))
        pre = spec[pre_idx].reshape(-1)
        post = spec[-1].reshape(-1)
        bins = np.linspace(-0.2, 2.0, 45)
        ax.hist(pre, bins=bins, alpha=0.6, color="gray", label=f"memorization (step {steps[pre_idx]})")
        ax.hist(post, bins=bins, alpha=0.6, color="royalblue", label=f"generalization (step {steps[-1]})")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel(r"eigenvalue of $\tilde A$ (sym. norm. adjacency)")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("count (heads x batch-mean)")
    fig.suptitle("Spectral density shift of attention manifolds across the grokking phase transition")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_spectral_density.png"))
    plt.close(fig)


def fig_correlations(runs):
    metrics = {"lam2": r"$\bar\lambda_2$ (Fiedler)", "spec_ent": "spectral entropy",
               "row_ent": "row attention entropy", "sink": "sink fraction",
               "eff_rank": "effective rank (last layer)"}
    rows = []
    for key in metrics:
        rs = []
        for seed in [0, 1, 2]:
            r = runs[f"baseline_seed{seed}"]
            acc = np.array(r["log"]["test_acc"])
            v = np.array(r["log"][key])
            rs.append(pearson(v.reshape(len(v), -1).mean(axis=1), acc))
        rows.append((metrics[key], np.mean(rs), np.std(rs)))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = [x[0] for x in rows]
    mus = [x[1] for x in rows]; sds = [x[2] for x in rows]
    order = np.argsort(mus)
    ax.barh(np.array(names)[order], np.array(mus)[order], xerr=np.array(sds)[order],
            color="royalblue", alpha=0.8, capsize=3)
    ax.set_xlabel("Pearson r with test accuracy (3 seeds, mean ± sd)")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_metric_correlations.png"))
    plt.close(fig)
    return rows


def fig_speedup(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for prefix, color, label in [("baseline", "gray", "standard transformer"),
                                 ("spectral_tau35", "seagreen", "$\\lambda_2$ reg., $\\tau$=0.35 (critical band)"),
                                 ("spectral", "royalblue", "$\\lambda_2$ reg., $\\tau$=0.05")]:
        steps, acc = mean_curve(runs, prefix + "_seed", "test_acc")
        m = acc.mean(axis=0); s = acc.std(axis=0)
        axes[0].plot(steps, m, color=color, lw=1.8, label=label)
        axes[0].fill_between(steps, m - s, m + s, color=color, alpha=0.2)
    axes[0].set_xlabel("training step"); axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=8); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Generalization curves (mean ± sd, 3 seeds)")
    groups = {"standard": [], "spectral (tau=0.05)": [], "spectral (tau=0.35)": []}
    for name, r in runs.items():
        if name.startswith("snapshot"):
            continue
        for g, pref in [("standard", "baseline_seed"), ("spectral (tau=0.05)", "spectral_seed"),
                        ("spectral (tau=0.35)", "spectral_tau35_seed")]:
            if name.startswith(pref) and r["transition_step"]:
                groups[g].append(r["transition_step"])
    bars = [np.mean(v) if v else 0 for v in groups.values()]
    errs = [np.std(v) if v else 0 for v in groups.values()]
    axes[1].bar(list(groups.keys()), bars, yerr=errs, capsize=4,
                color=["gray", "royalblue", "seagreen"], alpha=0.85)
    for i, (k, v) in enumerate(groups.items()):
        if v:
            axes[1].text(i, bars[i] + errs[i] + 80, f"{np.mean(v):.0f}±{np.std(v):.0f}", ha="center", fontsize=8)
    axes[1].set_ylabel("transition step (test acc ≥ 0.95)")
    axes[1].set_title("Grokking transition step")
    axes[1].tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_regularization_speedup.png"))
    plt.close(fig)


def fig_rank_decay(runs):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for prefix, color, label in [("deep6_baseline", "gray", "standard (6-layer)"),
                                 ("deep6_spectral", "royalblue", "spectral-regularized (6-layer)")]:
        if prefix + "_seed0" in runs:
            r = runs[prefix + "_seed0"]
            eff = np.array(r["log"]["eff_rank"]).mean(axis=0)
            ax.plot(range(len(eff)), eff, marker="o", color=color, label=label)
    ax.set_xlabel("residual stream depth (0 = embedding, then post-block)")
    ax.set_ylabel("effective rank (participation ratio)")
    ax.set_title("Representation rank vs. depth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_rank_decay.png"))
    plt.close(fig)


def fig_attention_topology(runs):
    cands = [v for v in runs.values() if "A_mean" in v["log"]]
    r = cands[0] if cands else runs.get("baseline_seed0") or next(iter(runs.values()))
    steps = np.array(r["log"]["step"])
    A_mean = np.array(r["log"]["A_mean"])
    t = r["transition_step"] or steps[-1]
    pre_idx = int(np.searchsorted(steps, max(t * 0.5, steps[1])))
    toks = ["a", "+", "b", "="]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, idx, title in [(axes[0], pre_idx, f"memorization phase (step {steps[pre_idx]})"),
                           (axes[1], -1, f"generalization phase (step {steps[-1]})")]:
        A = A_mean[idx].mean(axis=(0, 1))
        N = A.shape[0]
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
        xy = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        for i in range(N):
            for j in range(N):
                w = A[i, j]
                if w > 0.02:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], "-",
                            lw=1 + 12 * w, color="royalblue", alpha=min(0.85, 0.15 + w), zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=600, c="crimson", zorder=2)
        for i, tk in enumerate(toks[:N]):
            ax.text(xy[i, 0], xy[i, 1], tk, ha="center", va="center", color="white", fontsize=9, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("Mean attention topology: sink-star vs. structured circuit graph (layer/head-averaged)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_attention_topology.png"))
    plt.close(fig)


def fig_critical_threshold(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for name, r in runs.items():
        if name.startswith("snapshot"):
            continue
        steps = np.array(r["log"]["step"])
        lam2 = np.array(r["log"]["lam2"]).reshape(len(steps), -1).mean(1)
        se = np.array(r["log"]["spec_ent"]).reshape(len(steps), -1).mean(1)
        t = r["transition_step"]
        color = "royalblue" if name.startswith("baseline") else ("seagreen" if "tau35" in name else "crimson")
        style = "-" if name.startswith("baseline") else "--"
        axes[0].plot(steps, lam2, style, color=color, alpha=0.7, lw=1.2)
        axes[1].plot(steps, se, style, color=color, alpha=0.7, lw=1.2)
        if t:
            i = int(np.searchsorted(steps, t))
            axes[0].plot(steps[i], lam2[i], "k*", ms=9)
            axes[1].plot(steps[i], se[i], "k*", ms=9)
    axes[0].axhspan(0.30, 0.41, color="gold", alpha=0.25)
    axes[1].axhspan(0.635, 0.668, color="gold", alpha=0.25)
    axes[0].set_xlabel("training step"); axes[1].set_xlabel("training step")
    axes[0].set_ylabel(r"mean $\lambda_2$ (Fiedler)")
    axes[1].set_ylabel("mean spectral entropy $\\mathcal{H}_{spec}$")
    axes[0].set_title("Algebraic connectivity descends through critical band")
    axes[1].set_title("Spectral entropy crosses universal threshold")
    fig.suptitle("Critical spectral thresholds at the grokking transition (stars = transition points, gold = critical band)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_critical_threshold.png"))
    plt.close(fig)


def main():
    runs = load_all()
    print("loaded runs:", sorted(runs.keys()))
    fig_phase_transition(runs)
    fig_spectral_density(runs)
    rows = fig_correlations(runs)
    print("\nMetric correlations with test accuracy (mean ± sd over 3 seeds):")
    for name, m, s in sorted(rows, key=lambda x: -x[1]):
        print(f"  {name:34s} r = {m:+.3f} ± {s:.3f}")
    fig_speedup(runs)
    fig_rank_decay(runs)
    fig_attention_topology(runs)
    fig_critical_threshold(runs)
    print("\nTransition steps:")
    for name in sorted(runs):
        r = runs[name]
        print(f"  {name:26s} transition={r['transition_step']}  final_test_acc={r['final_test_acc']:.4f}  steps={r['steps_run']}")
    g = {"standard": [], "spectral": []}
    for name, r in runs.items():
        for k in g:
            if name.startswith(k + "_seed") and r["transition_step"]:
                g[k].append(r["transition_step"])
    if g["standard"] and g["spectral"]:
        b, sp = np.mean(g["standard"]), np.mean(g["spectral"])
        print(f"\nMean transition: standard {b:.0f} vs spectral {sp:.0f}  ->  speedup {b/sp:.2f}x")


if __name__ == "__main__":
    main()
