"""Exec every code cell of every notebook in fast-patched mode (CPU smoke test)."""
import json
import glob
import os
import sys
import traceback
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ONLY = sys.argv[1:] if len(sys.argv) > 1 else None
results = {}

for path in sorted(glob.glob(os.path.join(HERE, "notebooks", "*.ipynb"))):
    name = os.path.basename(path).replace(".ipynb", "")
    if ONLY and not any(o in name for o in ONLY):
        continue
    nb = json.load(open(path))
    ns = {"__name__": "__main__"}
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    core_src = "".join(cells[0]["source"])
    extras_src = "".join(cells[1]["source"]) if len(cells) > 1 and "nb_extras" in "".join(cells[1]["source"])[:200] else None
    exec(compile(core_src, "core", "exec"), ns)
    if extras_src:
        exec(compile(extras_src, "extras", "exec"), ns)

    real_train = ns["train_run"]

    def fast_train(seed, regularize=False, p=53, train_frac=0.4, d_model=128, n_layers=2,
                   n_heads=4, max_steps=12000, log_every=100, tau=0.35, beta1=1.0,
                   dataset=None, op="+", use_residual=True, **kw):
        return real_train(seed=seed, regularize=regularize, p=p, train_frac=train_frac,
                          d_model=32, n_layers=min(n_layers, 2), n_heads=4, max_steps=201,
                          log_every=100, tau=tau, beta1=beta1, dataset=dataset, op=op,
                          use_residual=use_residual)

    ns["train_run"] = fast_train
    real_T = ns["TinyTransformer"]
    ns["TinyTransformer"] = (lambda vocab, d_model=128, n_layers=2, n_heads=4, max_len=16,
                             use_residual=True: real_T(vocab, 32, min(n_layers, 2), 4, max_len, use_residual))
    if "train_lm_dyck" in ns:
        real_lm = ns["train_lm_dyck"]
        ns["train_lm_dyck"] = lambda *a, **k: real_lm(*a, d_model=32, max_steps=81, log_every=40)
    if "dyck1_dataset" in ns:
        real_dyck = ns["dyck1_dataset"]
        ns["dyck1_dataset"] = lambda *a, **k: real_dyck(96, max_prefix=8, seed=0)

    skip = 2 if extras_src else 1
    ok, fail = 0, 0
    t0 = __import__("time").time()
    for i, cell in enumerate(cells):
        if i < skip:
            continue
        src = "".join(cell["source"])
        try:
            exec(compile(src, f"{name}:cell{i}", "exec"), ns)
            ok += 1
        except Exception:
            fail += 1
            print(f"---- FAILURE in {name} cell {i} ----")
            traceback.print_exc()
    results[name] = (ok, fail, round(__import__("time").time() - t0, 1))
    print(f"[{name}] cells OK={ok} FAIL={fail} ({results[name][2]}s)", flush=True)

print("\n===== SMOKE SUMMARY =====")
for k, (ok, fail, t) in results.items():
    print(f"  {k}: {'PASS' if fail == 0 else 'FAIL'} ({ok} cells, {t}s)")
