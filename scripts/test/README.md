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

Keep model names, epochs, datasets, and Slurm resource requests in the individual
job scripts. Keep site-specific Python paths, scratch directories, and module setup
in `scripts/cluster.local.sh` and shared runtime helpers in `scripts/__lib__/`.