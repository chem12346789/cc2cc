# E3NN Training Recorder

## Model Comparison (2026-09-24)

The goal of the corrected implementation is faster execution without changing
useful model behavior. The current variants are not yet equivalent replacements.
Names below abbreviate the `transformer+dense_mix_e3nn_4` model family.

| Property | Original | `_correct` | `_correct_no_skip` |
|---|---|---|---|
| E3NN implementation | `model_e3nn.py` | `model_e3nn_correct.py` | Same as corrected |
| `lmax` | 2 | 0 | 0 |
| Cube aggregation | Sum | Mean | Mean |
| Dense residual connections | Off | On | Off |
| First tensor-product initialization, current source | Uniform(-1, 1) | Uniform(-1, 1) | Uniform(-1, 1) |
| Readout initialization | Standard-normal tensor-product weights | Standard-normal E3NN linear weights | Same as corrected |
| Dense, transformer, and gate initialization | PyTorch Linear defaults | Same defaults | Same defaults |

### Equivalence and Initialization

- A small float64 CPU probe confirmed that both E3NN blocks depend only on
  the four cube-channel means. The original readout uses spherical harmonics
  at the zero vector, eliminating the higher-angular contributions.
- The E3NN blocks have equivalent linear representational capacity up to
  reparameterization. This does not imply equal initial outputs or optimization.
- With 27 cube points, mean pooling divides the pooled features by 27 relative
  to sum pooling at fixed upstream weights. Learned weights can absorb that
  factor, but initialization, gate logits, clipping, and AdamW dynamics differ.
- The initial comparison was performed BEFORE the corrected first tensor product
  was changed from default Normal(0, 1) to Uniform(-1, 1). Its synthetic feature
  RMS was 2.10 for original versus 0.137 for corrected; these are historical
  probe results, not current-source or training-data measurements.
- In that same-seed CPU probe, corrected and no-skip state dictionaries matched
  exactly, while none of the 42 downstream parameter tensors matched original.
  Different parameter counts and initialization calls consume different random
  draws. Matching the seed or initializer distribution does not align weights.
- No-skip is the closer starting point for a speed-only replacement. Exact
  forward equivalence requires mapping effective E3NN weights, accounting for
  path normalization and pooling, and copying downstream weights. Verify both
  outputs and input gradients; forward equivalence alone does not ensure
  identical optimizer trajectories under a different parameterization.

### Logged Training Evidence

| Run | Model at run time | Objective / LR | Median interval, seconds | Median Eval | Best Eval in supplied log |
|---|---|---|---|---|---|
| 1822511 | Corrected | L1 / 3e-4 | 479 | 8.73 | 3.53 |
| 1563701 | Original | L1 / 1e-4 | 609 | 7.68 | 3.66 |
| 1560444 | Original | MSE / 1e-4 | 605 | 6.97 | 3.36 |

Medians use epochs 805-2400. Each logged interval includes five training epochs,
evaluation, and logging/checkpoint overhead, not just model execution. `Eval`
is mean absolute energy error regardless of the training objective. Best values
use the full supplied logs, which have different lengths. All runs used seed 42;
1822511 was supplied twice, not as two independent corrected runs.

The corrected run was faster per interval but generally converged more slowly
per epoch. At the epoch-2400 LR restart, its Eval increased from 4.45 to 30.4,
versus 5.40 to 10.4 for the L1 original. Higher LR is a plausible contributor,
not an isolated causal finding. Historical logs do not validate later edits.

### Recommended Comparisons

1. Match LR at 1e-4, objective, seed, split, weighting, and scheduler first.
2. Compare residual-off and pooling changes independently, with an unchanged
   baseline. Record the initializer change separately from historical runs.
3. Monitor feature RMS, gate entropy/weights, pre-clipping gradient norms,
   clipping frequency, weighted loss components, and per-molecule validation.
4. Compare time to a fixed validation threshold and equal wall-clock budgets,
   as well as cycle-end validation and consistently selected best checkpoints.
5. Repeat the strongest configurations with the same 3-5 seeds and report
   mean and standard deviation. A second original seed alone does not measure
   corrected-model variability. Treat MSE as a separate objective experiment.
6. Test reduced restart peaks or monotonic decay separately if spikes persist.
   Increasing `lmax` alone cannot recover useful angular information here;
   that requires a different invariant contraction and is an architecture change.