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
sbatch scripts/test/script_test/test_diet30.bash
```

Keep model names, epochs, datasets, and Slurm resource requests in the individual
job scripts. Keep site-specific Python paths, scratch directories, and module setup
in `cluster.local.sh`.