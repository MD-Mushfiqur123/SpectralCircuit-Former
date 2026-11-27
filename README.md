# SpectralCircuit-Former (Compressed Research Replication)

Spectral graph theory of attention manifolds applied to transformer grokking:
instrumented tiny transformers, differentiable graph-Laplacian metrics on
attention matrices, and phase-transition analysis.

## Layout

```
src/
  spectral.py   symmetrized normalized attention Laplacian, Fiedler λ2,
                spectral entropy, sink fraction, effective rank, differentiable λ2 hinge regularizer
  model.py      decoder-only transformer exposing per-head attention maps
  data.py       mod-p addition grokking dataset ([a, '+', b, '='] -> c)
  train.py      training loop + spectral instrumentation + logging
  run_all.py    experiment suite (seeds x {standard, spectral-regularized}, deep variant)
  run_tau.py    τ ablation for the λ2 regularizer
  analyze.py    statistics + figures fig1..fig7
results/        JSON training logs per run
figures/        publication-style PNG figures
FINDINGS.md     quantitative results and which blueprint claims replicated
```

## Run

```bash
pip install torch numpy matplotlib
python3 src/run_all.py && python3 src/run_tau.py && python3 src/analyze.py
```

Runs on CPU (~40 min total for the full suite; T4 GPU would run it ~10× faster).
Task is programmatic (no downloads): modular arithmetic mod 53, 40% train split
to induce grokking.
