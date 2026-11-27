"""Run all experiments and save JSON results."""
import json
import os
import sys
import torch
from train import train_run

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run_suite():
    torch.set_num_threads(4)
    jobs = []
    for seed in [0, 1, 2]:
        jobs.append({"name": f"baseline_seed{seed}", "kwargs": {"seed": seed, "regularize": False}})
        jobs.append({"name": f"spectral_seed{seed}", "kwargs": {"seed": seed, "regularize": True, "tau": 0.05, "beta1": 1.0}})
    jobs.append({"name": "deep6_baseline_seed0", "kwargs": {"seed": 0, "regularize": False, "n_layers": 6, "d_model": 64, "max_steps": 12000}})
    jobs.append({"name": "deep6_spectral_seed0", "kwargs": {"seed": 0, "regularize": True, "tau": 0.05, "beta1": 1.0, "n_layers": 6, "d_model": 64, "max_steps": 12000}})
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for job in jobs:
        if only and job["name"] not in only:
            continue
        out = os.path.join(RESULTS, job["name"] + ".json")
        if os.path.exists(out):
            print(f"skip {job['name']} (exists)")
            continue
        print(f"=== running {job['name']} ===", flush=True)
        res = train_run(**job["kwargs"])
        with open(out, "w") as f:
            json.dump(res, f)
        print(f"{job['name']}: transition={res['transition_step']} "
              f"test_acc={res['final_test_acc']:.4f} steps={res['steps_run']}", flush=True)


if __name__ == "__main__":
    run_suite()
