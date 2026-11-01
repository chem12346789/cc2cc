# Meaning of `grid_response`

In this project, `grid_response=1` means **standard PySCF grid response**.
The CPU RKS/UKS model gradient hooks include the corresponding response of
the learned-model XC energy.

## Standard PySCF response

Standard PySCF grid response already includes nuclear derivatives of the
atom-centered quadrature weights and the motion of the quadrature points
when evaluating the conventional XC functional. These are parts of standard
grid response, not separate alternatives to it.

## Current CPU model implementation

The CPU RKS/UKS gradient hooks pass the standard PySCF gradient object's
`grid_response` flag through the model gradient evaluators in
[modelscf_rks.py](../cc2cc/utils/modelscf_rks.py) and
[modelscf_uks.py](../cc2cc/utils/modelscf_uks.py).
They retain standard PySCF J/K, hybrid, NLC and total-gradient assembly.

- **Off (`0`):** baseline and learned correction use fixed quadrature
  coordinates and weights.
- **On (`1`):** baseline and learned correction include quadrature-weight
  and grid-sampling response. NLC response follows the standard PySCF flag.

The standard semilocal XC derivative is replaced by the full model XC
derivative, without double counting. Core, nuclear, J/K, Pulay, NLC and
density-fitting auxiliary response remain in PySCF's assembly.

Use `--if_grad 1 --grid_response 1` for moving-grid response.
Use `--grid_response 0` for entirely fixed-grid XC derivatives.

## Interpretation and validation

For moving-grid finite differences, rebuild the quadrature grids consistently.
For fixed-grid checks, preserve the reference coordinates and weights,
including the relevant NLC quadrature.
