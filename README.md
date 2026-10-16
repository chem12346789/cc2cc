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

When `gen_data.py --if_continue true` compares cached data for a multi-atom
molecule, it reports analytical/central-finite-difference maximum gradient
differences for both Post-DFT VV10 and DFT (B3LYP), in Hartree/Bohr.
The B3LYP check uses a 0.001 Bohr displacement and six additional B3LYP
calculations per atom.

## Important conventions

- Keep site-specific Slurm paths and environment overrides in `scripts/cluster.local.sh`.
- Keep project-wide importable code free of cluster-specific assumptions.
- Use the repo root as the working directory when launching batch jobs.

### Custom model gradients

CPU RKS/UKS model gradients retain PySCF's default `grid_response=False`.
After installing the model gradient with `get_veff_grad_modified_rks` or
`get_veff_grad_modified_uks`, set `g.grid_response = True` before `g.kernel()`
to include atom-centered grid motion and Becke-weight derivatives.
AO-overlap Pulay terms remain handled by PySCF in either mode.
This option does not add grid response to the separate GPU implementations.

## Quick start

```bash
python train.py
python test.py
```

For Slurm jobs:

```bash
sbatch scripts/test/test_diet30.bash
```

### Two-stage cosine warm restarts

Use `--scheduler cosine_warm2 --cosine_restart_step 160 --cosine_restart_lr 1e-4`
to keep the existing `cosine_warm` schedule until 160 epochs have
completed, then use the new peak learning rate at the first natural restart
at or after that threshold. The current cycle is unchanged; cycle timing and
growth by `--cosine_T_mult` continue without resetting. `--cosine_eta_min` stays unchanged. Parameter-group
peak ratios (including Muon) are preserved relative to `--lr`. Both restart
arguments are required. `--cosine_restart_step` is multiplied by the number of
optimizer updates per epoch, like `--cosine_T`. Resumed training uses the same
absolute update count.
