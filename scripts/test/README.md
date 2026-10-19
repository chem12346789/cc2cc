# Test Scripts

This directory contains reproducible evaluation and benchmark submission scripts.

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