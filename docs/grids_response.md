# Meaning of `grid_response`

In this project, `grid_response=1` means **standard PySCF grid response**.
The CPU and GPU RKS/UKS model gradient hooks include the corresponding
response of the learned-model XC energy.

## Standard PySCF response

Standard PySCF grid response already includes nuclear derivatives of the
atom-centered quadrature weights and the motion of the quadrature points
when evaluating the conventional XC functional. These are parts of standard
grid response, not separate alternatives to it.

## Model implementation

The CPU and GPU RKS/UKS gradient hooks pass the standard gradient object's
`grid_response` flag through the model gradient evaluators in
[modelscf.py](../cc2cc/utils/modelscf.py) and
[modelscf_gpu.py](../cc2cc/utils/modelscf_gpu.py), with `unrestricted` passed
explicitly. They retain the respective PySCF or GPU4PySCF J/K, hybrid, NLC and
total-gradient assembly.

- **Off (`0`):** baseline and learned correction use fixed quadrature
  coordinates and weights.
- **On (`1`):** baseline and learned correction include quadrature-weight
  and grid-sampling response. NLC response follows the standard PySCF flag.

The standard semilocal XC derivative is replaced by the full model XC
derivative, without double counting. Core, nuclear, J/K, Pulay, NLC and
density-fitting auxiliary response remain in PySCF's assembly.

Use `--if_grad 1 --grid_response 1` for moving-grid response.
Use `--grid_response 0` for entirely fixed-grid XC derivatives.

## Force-training data

`get_dft_grad` includes atom-centered grid motion in `grad2force` and saves
the quadrature-weight nuclear derivatives as `grid_weight_response`.
Force training contracts the energy derivatives with respect to the original
features and weights with these two arrays, including models that apply
weights before their forward pass. These are explicit nuclear derivatives;
they do not introduce orbital relaxation.
When `--if_grad 1` is enabled, force loss is included on every training epoch
for systems with force mappings.

Legacy force-training archives must be regenerated. Running `gen_data.py`
with `--if_continue 1` in non-evaluation mode refreshes the B3LYP and
B3LYP-D3BJ reference gradients, their difference target, and both DFT/ZMP
force mappings. Use the original geometry and grid settings. Other stored
reference data is retained. Archives without weight-response arrays are
rejected when force training is enabled.

Run `python -m scripts.gen_check --train` to check the saved
DFT/ZMP mapping and weight-response shapes, finite values, translation
residuals, and the CC-minus-B3LYP gradient target. `--force-threshold` controls
these residuals in their displayed units, independently of the energy
threshold in kcal/mol. This reads the large mappings one at a time; it does
not recompute forces or prove that the saved reference gradients enabled
grid response. Force checks are mandatory for every selected archive, including
evaluation archives; missing force arrays are reported as invalid data.

## Interpretation and validation

For moving-grid finite differences, rebuild the quadrature grids consistently.
For fixed-grid checks, preserve the reference coordinates and weights,
including the relevant NLC quadrature.
