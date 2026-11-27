import json
from train import train_run
for seed in [0, 1, 2]:
    res = train_run(seed=seed, regularize=True, tau=0.35, beta1=1.0)
    json.dump(res, open(f"../results/spectral_tau35_seed{seed}.json", "w"))
    print(f"tau35 seed{seed}: transition={res['transition_step']} steps={res['steps_run']}", flush=True)
