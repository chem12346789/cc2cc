# Meaning of `grid_response`

In this project, `grid_response=1` means **standard PySCF grid response**.
It does not enable a custom moving-grid derivative of the learned model.

## Standard PySCF response

Standard PySCF grid response already includes nuclear derivatives of the
atom-centered quadrature weights and the motion of the quadrature points
when evaluating the conventional XC functional. These are parts of standard
grid response, not separate alternatives to it.

It does not know about the learned model's neighbor gathering or explicit
relative-position inputs, and does not automatically differentiate them.

## Current CPU model implementation

The CPU RKS/UKS gradient hooks pass the standard PySCF gradient object's
`grid_response` flag through the model gradient evaluators. Both legacy
cube models and no-cube models use the shared
[baseline_grid_response](../cc2cc/utils/modelscf_no_cube.py) calculation.

- **Off (`0`):** baseline and learned correction use fixed quadrature
  coordinates and weights.
- **On (`1`):** the baseline uses PySCF's standard `get_vxc_full_response`;
  the learned correction still uses fixed coordinates, weights and (for
  no-cube models) neighbor membership. No learned weight, grid-motion or relative-position response
  is added. Existing NLC response follows the standard PySCF flag as well.

To avoid double counting, the implementation adds the difference between
PySCF's moving-grid and fixed-grid baseline XC derivatives to the existing
fixed-grid model XC derivative. Baseline coefficients are taken from
`modeldict.b3lyp_weights`. The usual core, nuclear, J/K and Pulay contributions
remain in PySCF's total-gradient assembly.

Use `--if_grad 1 --grid_response 1` for this baseline-only standard response.
Use `--grid_response 0` for entirely fixed-grid XC derivatives.

## Interpretation and validation

This is an intentional mixed convention: with a nonzero learned correction,
the result is **not the full derivative of the model energy on rebuilt
atom-centered grids**. Such finite differences include learned grid/geometry
response that this convention deliberately omits. For fully fixed-grid
finite differences, keep reference grid coordinates and weights unchanged.
