# Architecture Reference

Detailed module map and environment reference for the cc2cc DFT codebase.
Codex reads this only when it needs full structural detail — keeping AGENTS.md lean.

## Full Repository Layout

### `cc2cc/` — importable package
- `gen_cc.py`, `gen_ucc.py`: closed-/open-shell CC data generation
- `train_model.py`: training loop, W&B logging, distributed barriers
- `test_model_rks.py`, `test_model_uks.py`: model-in-SCF validation
- `benchmark_rks.py`, `benchmark_uks.py`: benchmark helpers

### `cc2cc/utils/` — shared utilities and domain logic
- `parser.py`: CLI arg definitions; reuse `add_args()` / `gen_name_args()`
- `env_var.py`: project paths, grid env settings, thread/GPU info
- `mol.py`: molecule/dataset definitions — source of truth for molecule names
- `Grids.py`, `GridsGPU.py`: CPU/GPU grid construction. `Grid` is CPU by
  default; keep GPU classes lazily imported
- `modelscf_rks.py`, `modelscf_uks.py`, `_gpu` variants: custom effective-
  potential/gradient hooks for PySCF SCF
- `get_dft_energy_*.py`, `get_dft_grad_*.py`, `get_zmp.py`, `zmp.py`: energy,
  gradient, and ZMP helpers
- `DataBase.py`, `TestDataDFT.py`, `DataRecord.py`, `ModelClass.py`: data
  loading, record keeping, model init, checkpoints, losses
- `model/`: neural model definitions; `--model NAME` → `model/NAME.py` with `Model`
- `*.json`: dataset/split definitions (`gmtkn-def2`, `gmtkn-diet30-def2`,
  `gmtkn-diet100-def2`, `dft-fitset-def2`)
- PySCF CCSD(T) intermediate files: vendor-like scientific code — patch minimally

### Top-level entry points
- `gen_data.py`: molecule selection → `gen_mole()` → optional MD/rotation →
  `Grid` → `cc()`/`ucc()` → `.npz` grid data
- `train.py`: train/eval list setup → `train_model()`
- `test.py`: checkpoint loading and RKS/UKS model validation
- `benchmark_dft.py`: baseline DFT benchmarking
- `d3_para.py`: D3 parameter fitting/testing
- `collect_info.py`: collect validation CSV summaries
- `submit_direct_array_per_gpu.py`: per-GPU job dispatch helper

### Script directories (Slurm/HPC)
`gen_data_script/`, `train_script/`, `test_script/`, `work_script/` assume
local paths (`~/anaconda3/envs/pyscf`, `~/backup-hd/tmp`, jemalloc). Never move
those assumptions into importable Python modules.

## Data & Checkpoint Conventions
| Item | Default | Override |
|---|---|---|
| Main project path | repo root via `env_var.MAIN_PATH` | `DFT2CC_MAIN_PATH` |
| Training/grid data | `data/grids_dft` | `DFT2CC_DATA_DIR` |
| Test data | `data/test` | `DFT2CC_DATA_TEST_DIR` |
| Checkpoints | `checkpoints/checkpoint_<save_dir>` | `--load <save_dir> --load_epoch <epoch>` |

- Dataset names: `gmtkn-def2`, `gmtkn-diet30-def2`, `gmtkn-diet100-def2`, `dft-fitset-def2`
- Basis names: `def2-QZVPPD`, `def2-TZVPPD`, `def2-QZVP(D)`

## Full Environment Variables
| Variable | Purpose |
|---|---|
| `DFT2CC_MAIN_PATH` | override repository root |
| `DFT2CC_DATA_DIR` | data subdirectory under `data/` |
| `DFT2CC_DATA_TEST_DIR` | test-data subdirectory under `data/` |
| `DFT2CC_EDGE_SIZE`, `DFT2CC_EDGE_LEN` | cube/grid stencil settings |
| `CUDA_VISIBLE_DEVICES`, `NUMBER_OF_GPU` | GPU selection and DDP sizing |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `NUMBA_NUM_THREADS` | CPU threading |
| `PYSCF_MAX_MEMORY`, `PYSCF_TMPDIR` | PySCF memory and scratch location |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA allocator tuning for large training jobs |
