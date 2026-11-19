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
- `docs/`: architecture notes and project docs, including the [training log summary](docs/training_summary.md) (also referred to as `training.md`).

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

## Common-first nested training splits

The following configs retain the [mol1_add](configs/mol1_add.json) baseline
and progressively add eligible `molecule2-W4_11` systems:

| Config | Added systems | Unique training systems |
| --- | ---: | ---: |
| [mol2_add_ex1.json](configs/mol2_add_ex1.json) | 0 | 77 |
| [mol2_add_ex2.json](configs/mol2_add_ex2.json) | 8 | 85 |
| [mol2_add_ex3.json](configs/mol2_add_ex3.json) | 16 | 93 |
| [mol2_add_ex4.json](configs/mol2_add_ex4.json) | 32 | 109 |
| [mol2_add_exf.json](configs/mol2_add_exf.json) | 54 | 131 |

The curated order starts with nitrogen, oxygen, carbon monoxide, acetylene,
ethylene, ethane, formaldehyde, and methanol, then broadens to other molecules
and reactive species. "Common-first" denotes chemical familiarity, not measured
abundance or numerical difficulty. Each addition list is a prefix of the full
list in `mol2_add_exf.json`; it controls membership, not training order.
The loader still sorts identifiers and shuffles training samples.
All five configs preserve the validation entries and five training exclusions
from [mol2_ex_add](configs/mol2_ex_add.json). The first and last splits match the
effective training membership of `mol1_add` and `mol2_ex_add`, respectively.
Counts are before single-atom repetition and any missing-data skips.
Select a config with `--split_config mol2_add_ex2.json`, for example.
Use the same three training seeds across all five splits; seed and atomic
weighting settings remain training arguments, not fields in these configs.
This ladder measures common-first augmentation, not random-subset scaling.

For [train_direct.bash](scripts/train/train_direct.bash), edit `SEED`,
`SPLIT_CONFIG`, and `CUDA_VISIBLE_DEVICES` in the top-level run-selection
section, or override them when launching:

```bash
SEED=0 SPLIT_CONFIG=mol2_add_ex2.json CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_direct.bash
```

Defaults remain seed 1, `mol1_add.json`, and GPU 7. Select one GPU index (0-7)
from the launcher's existing GPU-to-NUMA mapping; its NUMA node is selected
automatically. The launcher logs all three selections. These overrides do not
change epochs, atomic weighting, or the single-GPU process count.

## Typical workflow

1. Generate or load grid data with the Python entry points in the repo root.
2. Train a model with a script in `scripts/train/`.
3. Validate or benchmark with scripts in `scripts/test/`.
4. Inspect outputs in `validate*` and `checkpoints/`.

Direct calls to `cc()` select restricted CCSD(T) by default; pass
`unrestricted=True` for unrestricted CCSD(T), including when `mol.spin == 0`.
Restricted CCSD(T) requires `mol.spin == 0`. `gen_data.py` passes
`unrestricted=False` for zero spin and `True` otherwise.
Output filenames do not include the method, so `--if_continue true` skips an
existing archive regardless of which method produced it.
The generator calculates Post-DFT VV10 and self-consistent SCF-VV10 energies
and saves `e_post_vv10`, `e_scf_vv10`, and `dm1_scf_vv10` in the main archive.
Unsuffixed VV10 results also save `grad_post_vv10` and
`grad_scf_vv10` in Hartree/Bohr; forces are their negatives.
`gen_data.py --if_continue true` uses the same calculation for cached data,
saving results in the addon archive.
For multi-atom molecules, unsuffixed VV10 calculations additionally report
analytical/central-finite-difference maximum gradient differences for Post-DFT
VV10 and B3LYP and saves `grad_post_vv10_fd`. Each check uses a 0.001 Bohr
displacement and six additional B3LYP calculations per atom, starting from the
converged reference density to track the selected spin solution.
VV10 policy depends on the result suffix, independently of `--if_eval`.
The default suffix `""` enables all VV10 calculations and checks. The `"_zmp"`
suffix saves post-DFT energy and `grad_post_vv10_zmp` but skips SCF-VV10,
finite-difference checks, and density comparisons. Existing cached fields not
recomputed by this policy are preserved.

### PBE baseline supplements

Pass `--baseline pbe` to `gen_data.py`, retaining the original dataset, basis,
grid level, and generation configuration. This requires an existing
`data_<name>.npz` and writes `data_<name>_pbe.npz` without modifying the original
archive or rerunning CC or ZMP inversion. `--if_continue true` skips an existing
PBE supplement; `false` replaces only that supplement.
PBE-specific quantities use the same DFT and ZMP field names as the original
archive. Supplements generated with earlier PBE-prefixed fields must be
regenerated.

The supplement contains self-consistent PBE energies, densities, input features,
energy-density labels and gradients. Training-data generation also evaluates PBE
at the **saved ZMP density**, with PBE-specific energy/grid/gradient targets and
optional response data. Shared CC references and ZMP input features remain in
the original archive, identified by `source_archive`. Evaluation mode writes
only the self-consistent PBE energy, density, features and reference metadata.

ZMP constraint flags retain their normal defaults. Use `--zmp_feature_ov false`
to omit full OV Jacobians; `--zmp_observable_response false` also disables compact
response data. PBE response sidecars use the distinct
`data_<name>_pbe_zmp_*` prefix. Response gaps and projections are recomputed
from the PBE Fock, not copied from B3LYP. PBE supplements currently exclude VV10.

Train with `--baseline pbe --rho_input dft` or `--baseline pbe --rho_input zmp`.
The latter can also use `--if_zmp_response true` with PBE response sidecars.
Every selected system requires its PBE supplement. The baseline is saved in
the checkpoint and restored for SCF evaluation; new B3LYP checkpoints record
`b3lyp`, and missing or empty baselines in older checkpoints mean B3LYP. For PBE,
`modeldict.input_level` is at least six; the neural network retains its
configured input width.

PBE checkpoints support `--test_mode post-pbe` (fixed PBE density),
`post-pbe-nl` (fixed PBE density plus post-SCF VV10),
`post-pbe-vv10` (self-consistent PBE-VV10 reference density), and
`post-pbe-d3` (fixed PBE density with default PBE-D3BJ parameters).
Post-SCF modes are energy-only. Mode names select the reference density;
the checkpoint still selects the model energy baseline. PBE reference densities
and energies use separate fields in the test cache, leaving B3LYP fields intact.
Use `--baseline pbe-vv10` for VV10 in the model SCF; the option must match the
checkpoint baseline. Adding VV10 to a plain-PBE-trained model changes the trained
energy definition and can double-count correlation corrections.

### D3 parameter sweep

Run one sequential sweep per model, using the GPU assigned by the direct runner:

```bash
python scripts/direct_run_per_gpu.py --script train/train_d3_para.bash --gpus 0 --foreground
```

Use `--dry-run` to inspect the plan without launching fitting. Select models in
`scripts/train/train_d3_para.bash` via its `MODEL_INDEX` cases. Keep array task 0:
sweep points share the model's validation CSVs and fitted-parameter filename.
Each run prints its directory:
`log/d3-sweep_<model>_epoch-<epoch>_<basis>_bj_<loss>_<YYYYMMDD-HHMMSS>_pid-<pid>/`.
Existing run directories are left unchanged.
After each completed inner-loop point it appends `results.csv` and redraws
`heatmap.png` (best intermol training loss and DIET100 test score); unfinished
points stay blank. The Bash `rs6_values` and `rs18_values` arrays supply both
the loop values and heatmap axes/total count; change their `seq` ranges to adjust
the sweep. Per-point parameter JSONs and training/test logs are retained.
The script also supports direct `bash scripts/train/train_d3_para.bash` execution.

## Important conventions

- Keep site-specific Slurm paths and environment overrides in `scripts/cluster.local.sh`.
- Keep project-wide importable code free of cluster-specific assumptions.
- Use the repo root as the working directory when launching batch jobs.

### Custom model gradients

CPU and GPU RKS/UKS model gradients retain PySCF's default
`grid_response=False`. Install `get_veff_grad_modified` on CPU or
`get_veff_grad_modified_gpu` on GPU, passing `unrestricted=False` for RKS or
`unrestricted=True` for UKS. Set `g.grid_response = True` before `g.kernel()`
to include atom-centered grid motion and Becke-weight derivatives.
AO-overlap Pulay terms remain handled by PySCF in either mode. The validation
runner uses GPU gradients when CUDA is available and `--device cuda` is set;
use `--cpu 1` for CPU SCF and gradients.
GPU `g.get_veff()` returns GPU4PySCF's per-atom half-gradient, not the CPU
AO-matrix derivative.
Potential and gradient derivatives include the chain rule when model inputs
are weighted before inference.

### GPU SCF integration

The GPU RKS/UKS hooks exchange model inputs and outputs with PyTorch through
stream-aware DLPack, without staging model outputs through CPU NumPy arrays.
Use `get_veff_modified_gpu(..., unrestricted=False)` for RKS or
`unrestricted=True` for UKS.
CPU model callers still receive NumPy arrays by default.
The numerical-integration optimizer is reused on each grid object across SCF
iterations and rebuilt when molecular integral data or the CUDA device changes.
This does not cache J/K objects or change the large-system CPU J/K fallback.
The same AO threshold selects CPU J/K derivatives for model gradients.

## Quick start

```bash
python train.py
python test.py
```

For Slurm jobs:

```bash
sbatch scripts/test/test_diet30.bash
```

### VV10 model validation

Data generation with `gen_data.py --if_eval true` retains CCSD(T) total
energies but skips CC densities and gradients. CC density comparisons and
CC-dependent ZMP data are therefore skipped; non-CC VV10 checks remain enabled.

Pass `--baseline b3lyp-vv10` (B3LYP checkpoint) or `--baseline pbe-vv10`
(PBE checkpoint) to `test.py` to include VV10 energy and potential
in the model SCF on CPU or GPU. The log reports `NLC (vv10) energy` during
integration. With this baseline, `scf_ene` already includes VV10 and is also
saved as `scf_vv10_ene`. Without VV10 in the SCF, `scf_vv10_ene` is blank.
Without this baseline, the runners calculate and save `post_scf_vv10_ene`.
With `--baseline b3lyp-vv10`, they skip the post-SCF VV10 calculation and
leave `post_scf_vv10_ene` blank.

When switching an existing validation run to this baseline, pass
`--if_continue 0` (or set `if_continue_args=0` in the launch script).
Resume checks molecule names, not baseline settings, so otherwise saved
results can be skipped without any new SCF calculation.
