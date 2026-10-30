# Test Scripts

This directory contains reproducible evaluation and benchmark submission scripts.

## Comparing density choices without rerunning SCF

From the repository root, compare the same checkpoint evaluated self-consistently
and on fixed B3LYP densities:

```bash
python -m scripts.analyze_density_effect \
  --scf 'validate/def2-QZVP(D)_atom-3259973/2395' \
  --b3lyp 'validate/def2-QZVP(D)_atom-3259973-scf-b3-post-scf-b3lyp/2395'
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

Run the CPU-only RKS/UKS reference explicitly from the repository root:

```bash
python -m cc2cc.utils.zmp_reference \
  --atom 'H 0 0 0; H 0 0 0.8' --basis sto-3g \
  --steps 0.002 0.001 --max-l 20 --output zmp_reference.json
```

Use `--spin` for open-shell systems (alpha minus beta electron count).
This reruns RHF/UHF, B3LYP, the same CCSD(T) target-density construction as
`gen_cc.cc`/`gen_ucc.ucc`, and RZMP/UZMP at every displaced geometry. It does
not generate training
data or change the existing generation workflow. Every SCF, CC/lambda, and ZMP
stage must converge; failures raise instead of returning a previous density.
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

Training-mode `cc()` and `ucc()` also use this full-pipeline numerical method
for `grad_zmp`, with a 0.001 Bohr central-difference step, the supplied grid's
level, and a fixed inversion schedule through `max_l=20`. This costs six full
energy pipelines per atom; isolated-atom gradients are zero by translation
invariance. Evaluation mode still skips gradients. The reference requires at
least two electrons for molecular CCSD(T) calculations.
ZMP DIIS filters linearly dependent error-vector modes using the same
eigenvalue threshold as PySCF DIIS, avoiding singular solves at displaced
geometries without lowering the inversion limit.

The shared inversion loop and informational output live in
`cc2cc/utils/zmp_reference.py:get_zmp`. Both data generation and numerical
references use it, including per-lambda density, I-value, dipole, Hartree-grid,
and nuclear-grid comparisons. Strict numerical references retain tight
tolerances and require every stage to converge; normal generation retains
its divergence-stop behavior. `get_zmp.py` only forwards the legacy
`get_zmp_rks`/`get_zmp_uks` entry points.

CCSD solver settings, triples-corrected lambda solving, and target one-particle
density construction are shared by `cc()`, `ucc()`, and displaced references
through `cc2cc/utils/ccsd_t_target.py`. Change CC iteration/tolerance parameters
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

## Detailed force validation

`force-def2` retains the five original methane geometries in
`molecule_Force_legacy`. Its default `molecule_Force` now includes 188 geometries:
those five plus three centers and 180 displacements. The new cases are methane,
distorted water (closed shell), and OH (open shell). Each Cartesian component has
positive and negative steps of 0.004, 0.002, and 0.001 Bohr.
`molecule_Force_centers` contains only the three new centers.
The water geometry starts from `W4_11-h2o` with atom 1 shifted by
(0.13, 0.08, 0.04) Angstrom to break equal-bond symmetry.

The force wrapper uses `--conv_tol 1e-11 --conv_tol_grad 1e-7
--require_convergence 1` and up to 250 SCF cycles. Other `test.py` invocations
retain the original tolerances and unconverged-result policy by default.
No validation jobs are launched by generating the dataset or opening the notebook.

The shared launcher overwrites its group log. Run these jobs **sequentially**,
waiting for each to finish and checking its error log before copying the output:

```bash
sbatch --export=ALL,GRID_RESPONSE=1 scripts/test/test_force.bash
# After completion:
cp log/atom-7297/1115/Force_0.out log/atom-7297/1115/Force_validation_on.out
sbatch --export=ALL,GRID_RESPONSE=0,FORCE_SET=molecule_Force_centers scripts/test/test_force.bash
# After completion:
cp log/atom-7297/1115/Force_centers_0.out log/atom-7297/1115/Force_validation_off.out
```

The notebook `notebooks/dgx1_test_force.ipynb` reads these two saved logs and reports
max/RMS Cartesian errors, step convergence, on/off comparisons using the same
energies, net gradient, and torque. Missing or unconverged required calculations
must be completed; the historical loose-SCF log is not sufficient.
Start with an absolute gradient target of 1e-5 Hartree/Bohr; verify stability
against tighter SCF and grid settings, particularly when decreasing the step.
Energy uncertainty is amplified by division by the displacement.
On/off logs must use the same model checkpoint, basis, grid, and precision.

The notebook also supports checking an existing `Force_0.out` immediately:
its default `ON_LOGS` selects that file and permits the historical SCF limits
(1e-7 energy, 1e-3 orbital gradient), printing an explicit accuracy warning.
Set `SCF_CONV_TOL=1e-11` and `SCF_CONV_TOL_GRAD=1e-7` after tighter reruns.
The OFF log is optional: its absence skips only the on/off comparison, not the
Cartesian, step-convergence, or invariance reports. All required ON energies
must still exist and be converged.

Regenerate the displacement dataset without running SCF:

```bash
python -m scripts.force_validation \
    --source cc2cc/utils/mol_dataset/gmtkn-def2.json \
    --output cc2cc/utils/mol_dataset/force-def2.json
```

Keep model names, epochs, datasets, and Slurm resource requests in the individual
job scripts. Keep site-specific Python paths, scratch directories, and module setup
in `scripts/cluster.local.sh` and shared runtime helpers in `scripts/__lib__/`.