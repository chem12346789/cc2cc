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

## Installing PyTorch

Use the Python interpreter of the environment you want to modify:

```bash
python scripts/install_torch.py --dry-run
python scripts/install_torch.py
python -m pip install -r requirements.txt
```

The installer pins torch 2.8.0, torchvision 0.23.0, and torchaudio 2.8.0,
using the official CPU, CUDA 12.6, or CUDA 12.8 index. It installs only this
stack and its dependencies, before the remaining project requirements.
It supports Linux/Windows x86_64 with Python 3.9 through 3.13.
Automatic selection reads `CUDA Version` from the standard `nvidia-smi` output:
>= 12.8 selects cu128, and 12.6 through 12.7 selects cu126. Lower versions
cause an error. This field describes driver CUDA support, not the installed
system CUDA Toolkit. GPU compute capabilities are queried separately.
Automatic GPU support is limited to compute capabilities 7.5 through 12.0;
Blackwell GPUs require cu128. Unsupported hardware, old drivers, and failed
queries cause an error rather than a silent CPU fallback.

Without `nvidia-smi`, CPU is explicitly selected. On GPU-less login nodes,
select for the intended compute node instead:

```bash
python scripts/install_torch.py --backend cu128 --dry-run
python scripts/install_torch.py --backend cu128
```

`--backend cpu`, `cu126`, or `cu128` bypasses GPU detection; the caller must
verify target hardware compatibility. `--dry-run` only prints the selection
and install command: it does not invoke pip, download packages, or initialize
CUDA. Driver probing does not account for GPU visibility masks or validate
CUDA kernels; test the installed stack on the actual compute node.

## Typical workflow

1. Generate or load grid data with the Python entry points in the repo root.
2. Train a model with a script in `scripts/train/`.
3. Validate or benchmark with scripts in `scripts/test/`.
4. Inspect outputs in `validate*` and `checkpoints/`.

When `gen_data.py --if_continue true --if_eval false` compares cached data for a multi-atom
molecule, it reports analytical/central-finite-difference maximum gradient
differences for both Post-DFT VV10 and DFT (B3LYP), in Hartree/Bohr.
The B3LYP check uses a 0.001 Bohr displacement and six additional B3LYP
calculations per atom.
Evaluation (`--if_eval true`) skips gradient generation and checks, and does not
add or update gradients in the addon archive; existing cached fields are preserved.
It also skips electronic density difference calculations and reporting while
retaining energy comparisons.

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

### GPU SCF integration

The GPU RKS/UKS hooks exchange model inputs and outputs with PyTorch through
stream-aware DLPack, without staging model outputs through CPU NumPy arrays.
CPU model callers still receive NumPy arrays by default.
The numerical-integration optimizer is reused on each grid object across SCF
iterations and rebuilt when molecular integral data or the CUDA device changes.
This does not cache J/K objects or change the large-system CPU J/K fallback.

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
