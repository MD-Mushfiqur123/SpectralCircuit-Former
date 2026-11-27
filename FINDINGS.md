# FINDINGS — SpectralCircuit-Former Compressed Replication

**Setup**: mod-53 addition grokking, 2-layer/4-head/d=128 decoder-only transformer
(6-layer/d=64 variant for depth analysis), full-batch AdamW (lr 1e-3, wd 2.0),
train fraction 0.4, CPU-only (no GPU in sandbox). 12 training runs total
(3 seeds x {standard, λ2-reg τ=0.05, λ2-reg τ=0.35} + 2 deep runs + 1 snapshot run).
Transition defined as first step with test accuracy ≥ 0.95.

## Result 1 — Grokking phase transition replicated (12/12 runs reach 100% test acc)

| config | transition steps (mean ± sd) | final test acc |
|---|---|---|
| standard | 2800 ± 650 | 1.000 |
| λ2-reg, τ=0.05 | 2667 ± 650 | 1.000 |
| λ2-reg, τ=0.35 | 2600 ± 265 | 1.000 |
| 6-layer standard | 1900 | 1.000 |
| 6-layer λ2-reg | 1800 | 1.000 |

## Result 2 — The Fiedler eigenvalue does NOT jump up at the transition; it descends through a critical band

The blueprint hypothesized λ2 jumping from ~0.05 to ~0.6 at the transition.
The data show the **opposite direction**: mean λ2 falls monotonically
(≈0.58 at init → ≈0.15 after grokking) because initialized attention is a
well-connected near-uniform graph, while learned circuits are sharp and sparse.
The transition fires when λ2 **descends through** a consistent critical band:

- λ2 at transition: **0.35 ± 0.05** (baseline family, n=4 seeds; range 0.31–0.41)
- λ2 keeps descending after the transition (0.15–0.18 at convergence)

Correlation of mean λ2 with test accuracy: r = −0.863 ± 0.060 (3 seeds).
Effective rank of representations: r = −0.847 ± 0.059 (compression during grokking).

## Result 3 — Spectral entropy is the most consistent universal threshold marker

Across **all 10 shallow runs in 4 different configurations**, mean spectral
entropy of the symmetrized attention adjacency at the transition:
**0.652 ± 0.017** — the tightest marker measured (transition *steps* vary 2×,
the entropy threshold varies ~2.5%).

## Result 4 — Attention topology shift replicated: sink-star → structured graph

Head-averaged attention graphs (fig6): memorization phase shows a star-like
sink topology centered on operand token `a`; post-transition topology is
distributed/structured. Sink fraction at transition: ≈0.47–0.56 across all runs.

## Result 5 — The λ2 hinge regularizer does not deliver the claimed 8–12× speedup

Both τ=0.05 (rarely active, since λ2 rarely drops below 0.05 before transition)
and τ=0.35 (active at the critical band) yield only ≈1.05–1.08× mean speedup,
within seed noise. Pushing λ2 up also holds the final model at higher λ2
(0.38–0.62 vs 0.15 baseline) without accuracy loss, but buys no acceleration.
**Claim not replicated in this compressed setup.**

## Result 6 — Rank collapse not observed in this regime

Effective rank along the 6-layer residual stream stays flat (≈2.7 of max 4)
for both standard and regularized models. With N=4 tokens and pre-LN residual
architecture, depth-dependent over-smoothing does not manifest. A longer-context
task (RULER/NIAH with N ≥ 512) is required to test the rank-collapse claim;
out of scope for this CPU sandbox run.

## Deliverables

- `results/*.json` — 12 full training logs (step, losses, accuracies, per-head λ2,
  spectral entropy, row entropy, sink fraction, adjacency eigen-spectra, effective rank)
- `figures/fig1..fig7*.png` — publication-style figures
- `src/` — spectral.py, model.py, data.py, train.py, run_all.py, run_tau.py, analyze.py

## Reproduction

```bash
python3 src/run_all.py      # main suite (~35 min CPU)
python3 src/run_tau.py      # tau ablation
python3 src/analyze.py      # figures + statistics
```
