# cc2cc

cc2cc is a PyTorch + PySCF workflow for learning and evaluating DFT-inspired correction models for molecular energetics.

## Repository layout

- `cc2cc/`: core model and training code.
- `data/`: cached DFT and benchmark data.
- `configs/`: dataset and job configuration JSON files.
- `checkpoints/`: saved model checkpoints.
- `validate/`, `validate_hkqai/`, `validate_hkqai_done/`: validation outputs and comparison tables.
- `scripts/test/`: Slurm evaluation and benchmark jobs.
- `scripts/train/`: training launch scripts.
- `scripts/work/`: operational helper scripts.
- `notebooks/`: exploratory analysis notebooks.
- `manuscript/`: manuscript and LaTeX sources.
- `docs/`: architecture notes and project docs.

## Typical workflow

1. Generate or load grid data with the Python entry points in the repo root.
2. Train a model with a script in `scripts/train/`.
3. Validate or benchmark with scripts in `scripts/test/`.
4. Inspect outputs in `validate*` and `checkpoints/`.

## Important conventions

- Keep site-specific Slurm paths and environment overrides in `scripts/test/cluster.local.sh`.
- Keep project-wide importable code free of cluster-specific assumptions.
- Use the repo root as the working directory when launching batch jobs.

## Quick start

```bash
python train.py
python test.py
```

For Slurm jobs:

```bash
sbatch scripts/test/script_test/test_diet30.bash
```
