# Test Scripts

This directory contains reproducible evaluation and benchmark submission scripts.

## Post-B3LYP-D3 evaluation

In `test_diet100.bash`, set `TEST_MODE=post-b3lyp-d3` with `MODEL_INDEX=0`
for `atom-3259973` or `MODEL_INDEX=1` for `atom-1279985` (epoch 2395).
This evaluates the model once on the B3LYP density, without model SCF or VV10,
using standard B3LYP-D3BJ parameters rather than the launcher's custom D3 values.
Use `scf_d3bj_ene` for the total energy:
`E_B3LYP + E_D3BJ + E_model[rho_B3LYP]`.
The `scf_ene` column excludes D3BJ; do not add D3 again to `scf_d3bj_ene`.
For `collect_info.py`, use `--model_load atom-3259973-post-b3lyp-d3` or
`--model_load atom-1279985-post-b3lyp-d3` with the default energy column.
Existing model selections and their default modes are unchanged.

## Comparing density choices without rerunning SCF

From the repository root, compare the same checkpoint evaluated self-consistently
and on fixed B3LYP densities:

```bash
python -m scripts.analyze_density_effect \
  --scf 'validate/def2-QZVP(D)_atom-3259973/2395' \
  --b3lyp 'validate/def2-QZVP(D)_atom-3259973-post-b3lyp/2395'
```

Replace both paths to analyze other checkpoints. Each input accepts a checkpoint
directory containing molecular validation CSVs, or one aggregated molecular CSV.
The default dataset is DIET100; `--dataset PATH` selects another GMTKN/DIET
reaction JSON, `--basis` selects the molecule-name suffix, `--columns` selects
matching energy columns (default: `scf_ene scf_d3bj_ene`), and `--top` controls
the number of ranked reactions. Compare identical functional/dispersion settings;
the script cannot infer checkpoint or numerical-setting equivalence from CSVs.

The read-only report includes unweighted reaction MAE, collector-compatible
WTMAD-2, paired prediction shifts, gains/losses, subset contributions, and
within-reaction cancellation of molecular energy shifts. Positive gains mean
SCF is more accurate. Missing molecules, duplicate names, or non-finite selected
energies fail explicitly rather than silently comparing a partial benchmark.
Convergence metadata are reported; blank post-SCF flags are not SCF failures.
Redirect stdout to save the human-readable report. No density-quality claim can
be made from energy comparisons alone.

## ZMP energy-gradient reference

The module currently rejects CLI execution; the following CPU-only RKS/UKS
reference command and JSON output described below are not available:

```bash
python -m cc2cc.utils.reference.reference_zmp \
  --atom 'H 0 0 0; H 0 0 0.8' --basis sto-3g \
  --steps 0.002 0.001 --max-l 20 --output zmp_reference.json
```

Use `--spin` for open-shell systems (alpha minus beta electron count).
This reruns RHF/UHF, B3LYP, the same CCSD(T) target-density
construction as `generate_cc_data.cc`, and RZMP/UZMP at every displaced
geometry. It does not generate training data or change the existing generation
workflow. Every SCF, CC/lambda, and ZMP stage must converge; failures raise
instead of returning a previous density.
The reference tightens solver tolerances relative to data generation and keeps
the full inversion schedule fixed across geometries.

The JSON records displaced energies (minus, plus), gradients and forces for both
ZMP-density B3LYP and self-consistent B3LYP, and their gradient difference.
Steps/coordinates are in Bohr, energies in Hartree, gradients in Hartree/Bohr;
forces are negative gradients. Output is written only after all calculations
succeed. The cost is `6 * natm * nsteps` full energy pipelines.
Compare gradients at successive step sizes and repeat with increased grid
level and `--max-l` before treating results as converged. A low `--max-l` is
only a smoke check, not the production inversion limit. Comparing with B3LYP
does not eliminate ZMP orbital response. No analytical ZMP response is assumed.

Training-mode `cc()` also uses this full-pipeline numerical method for
`grad_zmp`, with a 0.001 Bohr central-difference step, the supplied grid's
level, and a fixed inversion schedule through `max_l=20`. This costs six full
energy pipelines per atom; isolated-atom gradients are zero by translation
invariance. Evaluation mode still skips gradients. The reference requires at
least two electrons for molecular CCSD(T) calculations.
ZMP DIIS filters linearly dependent error-vector modes using the same
eigenvalue threshold as PySCF DIIS, avoiding singular solves at displaced
geometries without lowering the inversion limit.

The shared inversion loop and informational output live in
`cc2cc/utils/reference/reference_zmp.py:get_zmp`. Both data generation and numerical
references use it, including per-lambda density, I-value, dipole, Hartree-grid,
and nuclear-grid comparisons. Strict numerical references retain tight
tolerances and require every stage to converge; normal generation retains
its divergence-stop behavior.

CCSD solver settings, triples-corrected lambda solving, and target one-particle
density construction are shared by `cc()` and displaced references through
`cc2cc/utils/reference/reference_cc.py`. Change CC iteration/tolerance parameters
and lambda iteration/tolerance parameters there. Generation and displaced
references use identical CC settings: energy tolerance `1e-12`, amplitude and
lambda tolerances `1e-10`, and 200 iterations for both CC and lambda solvers.
Reference-mode convergence failure checks remain enabled.
The helper also supplies the separate full-block triples corrections required
by the CC nuclear gradient, without constructing a two-particle density for
energy-only displaced references.
`ccsd_t_target_gradient()` in the same module evaluates the generation-time CC
nuclear gradient for both spin cases, including intermediate compression,
nuclear contributions, symmetry projection, and isolated-atom/one-electron
branches. The entry points retain construction of the two-particle density
needed for training data.

- `test_*.bash`: Slurm evaluation wrappers. Scheduler directives and experiment
  variables stay here; common setup and execution live in `lib/test_job.sh`.
- `benchmark*.bash`: CPU and GPU benchmark jobs.
- `calculator*.bash`: calculator-based validation jobs.
- `run_task.sh` and `submit_diet100.bash`: local launch helpers for GPU-aware submission.
- `config/`: measured runtime inputs for array-job scheduling.
- `cluster.local.sh`: optional, untracked cluster-specific environment overrides. Copy
  `cluster.local.sh.example` to create it.

Run Slurm jobs from the repository root, for example:

```bash
sbatch scripts/test/test_diet30.bash
```

`test_force.bash` enables grid response by default. Disable it with
`sbatch --export=ALL,GRID_RESPONSE=0 scripts/test/test_force.bash`.
Other evaluation scripts and `test.py` default to disabled grid response.
When invoking `test.py` directly, use
`--if_grad 1 --grid_response 1` (enabled) or `--if_grad 1 --grid_response 0`
(disabled). The grid-response setting has no effect without force calculations.

## Rotation validation

Analyze the rotation CSV without running SCF:

```bash
python -m scripts.rotation_validation \
    --csv path/to/rotation-results.csv \
    --dataset cc2cc/utils/mol_dataset/rotation-def2.json \
    --basis def2-QZVPPD
```

The report compares `scf_ene` to each reference in Hartree and kcal/mol and,
when present, compares rotated `scf_grad` in Hartree/Bohr. Diagnostic energy
and maximum-component gradient tolerances default to 1e-5; adjust them with
`--energy-tol` and `--gradient-tol`. Energy-only comparisons mark gradients
unavailable; partially missing gradient pairs are pending. `--require-gradients`
also marks energy-only comparisons pending. `--completed-only` reports available
comparisons and marks missing or unconverged runs pending. The command exits 1
for any tolerance failure, otherwise 2 for pending results, and 0 when all
available comparisons pass.

The CSV cannot verify model, grid, SCF tolerances, precision, geometry/atom
order, units, or grid-response settings; check these separately. Rotations rebuild
the quadrature grid, so energy differences include numerical integration effects.
Tighten SCF and grid settings before interpreting the diagnostic tolerances.

## Detailed force validation

`force-def2` retains the five original methane geometries in
`molecule_Force_legacy`. Its default `molecule_Force` now includes 388 geometries:
those five plus five centers and 378 displacements. The new cases are methane,
distorted water, OH, HAL59-BrBr_FCCH, and W4_11-sif4; only OH is open shell.
Each Cartesian component has positive and negative steps of 0.004, 0.002, and
0.001 Bohr. `molecule_Force_centers` contains only the five new centers.
The water geometry starts from `W4_11-h2o` with atom 1 shifted by
(0.13, 0.08, 0.04) Angstrom to break equal-bond symmetry.

The force wrapper uses `--conv_tol 1e-11 --conv_tol_grad 1e-7
--require_convergence 1` and up to 250 SCF cycles. Other `test.py` invocations
retain the original tolerances and unconverged-result policy by default.
No validation jobs are launched by generating the dataset or opening the notebook.

When generating both force sets, run these jobs **sequentially**, waiting for
each to finish. They write separate full-set and center-only CSVs:

```bash
sbatch --export=ALL,GRID_RESPONSE=1 scripts/test/test_force.bash
# After completion of the first job:
sbatch --export=ALL,GRID_RESPONSE=0,FORCE_SET=molecule_Force_centers scripts/test/test_force.bash
```

The notebook `notebooks/dgx1_test_force.ipynb` reads a validation CSV and reports
max/RMS Cartesian errors, net gradient, and torque for completed steps. Set
`CSV_PATH` and `BASIS` for the calculation to inspect. The CSV reader uses
`scf_ene` in Hartree and `scf_grad` in Hartree/Bohr (without D3) and ignores
unconverged runs when `completed_only=True`. Missing calculations must be
completed for a full report.

CSVs do not record enough information to verify the model, grid, SCF tolerances,
precision, rotation, geometry/atom order, units, or grid-response setting. Check
these separately before interpreting results; the notebook labels grid response
as `unknown`. Start with an absolute gradient target of 1e-5 Hartree/Bohr and
verify stability against tighter SCF and grid settings, particularly when
decreasing the step. Energy uncertainty is amplified by division by the
displacement.

CSV records from different sources are compared only when the caller explicitly
supplies the same `settings` dictionary, including `basis`, to `read_force_csv`;
the caller must verify that those settings apply to each source.

To print a table directly from CSV:

```bash
python -m scripts.force_validation \
    --csv validate/def2-QZVPPD_atom-35580/2395/ccdft_def2-QZVPPD_atom-35580_force-def2_molecule_Force.csv \
    --dataset cc2cc/utils/mol_dataset/force-def2.json \
    --basis def2-QZVPPD --completed-only
```

The table includes each case and step's grid-response label, max/RMS gradient
error, observed convergence order, and maximum net-gradient/torque components.
Grid response is labeled `unknown` by default. After verifying the setting, use
`--grid-response on` or `--grid-response off` to declare it.

Regenerate the displacement dataset without running SCF:

```bash
python -m scripts.force_validation \
    --source cc2cc/utils/mol_dataset/gmtkn-def2.json \
    --output cc2cc/utils/mol_dataset/force-def2.json
```

Keep model names, epochs, datasets, and Slurm resource requests in the individual
job scripts. Keep site-specific Python paths, scratch directories, and module setup
in `scripts/cluster.local.sh` and shared runtime helpers in `scripts/__lib__/`.