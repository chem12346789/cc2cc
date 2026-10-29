# Meaning of `grid_response`

In this project, `grid_response=1` means **standard PySCF grid response**.
For no-cube models, this flag now also enables the corresponding learned-model
response, including explicit relative-position dependence.

## Standard PySCF response

Standard PySCF grid response already includes nuclear derivatives of the
atom-centered quadrature weights and the motion of the quadrature points
when evaluating the conventional XC functional. These are parts of standard
grid response, not separate alternatives to it.

It does not know about the learned model's neighbor gathering or explicit
relative-position inputs, and does not automatically differentiate them.

## Current CPU model implementation

The CPU RKS/UKS gradient hooks pass the standard PySCF gradient object's
`grid_response` flag through the model gradient evaluators.
The no-cube entry modules are
[modelscf_rks_no_cube.py](../cc2cc/utils/modelscf_rks_no_cube.py) and
[modelscf_uks_no_cube.py](../cc2cc/utils/modelscf_uks_no_cube.py).
Each contains its own clustered quadrature and grid-response implementation;
neither imports the shared no-cube module or the other spin module.
Both reuse standard PySCF J/K, hybrid, NLC and total-gradient assembly.
The original cube RKS/UKS modules are unchanged.

- **Off (`0`):** baseline and learned correction use fixed quadrature
  coordinates and weights.
- **On (`1`), no-cube:** both baseline and learned correction include weight
  and grid-sampling response. Explicit edge-vector response is included too.
  Shared-neighbor derivatives are summed at their original grid points, and
  sampling/edge motion is assigned to each point's owning atom. Existing NLC
  response follows the standard PySCF flag as well.

The standard semilocal XC derivative is replaced by the full model XC
derivative, without double counting. Core, nuclear, J/K, Pulay, NLC and
density-fitting auxiliary response remain in PySCF's assembly.

Use `--if_grad 1 --grid_response 1` for full no-cube moving-grid response.
Use `--grid_response 0` for entirely fixed-grid XC derivatives.

## Interpretation and validation

This replaces the earlier baseline-only mixed convention for no-cube models.
Neighbor membership and pruning selections are discrete and held fixed locally.
Derivatives are valid on smooth branches, not at neighbor/pruning changes.
For moving-grid finite differences, rebuild grids while retaining the same
selection branch. For fixed-grid checks, preserve reference coordinates,
weights and neighbor indices, including relevant NLC quadrature.
Unmatched active response points, coincident physical points and inconsistent
weights are explicitly rejected rather than silently assigning wrong owners.
PySCF's zero-weight alignment padding at `(1e-4, 1e-4, 1e-4)` is an exception:
these points remain fixed in the laboratory frame. If used as neighbor halo
inputs, their AO/feature derivatives are retained, but they receive no
atom-owned sampling motion or weight response. Edge motion at their physical
neighbors is still included. Padding is not removed or silently ignored.
