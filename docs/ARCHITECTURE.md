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
  - Existing-cache comparisons report B3LYP/post-DFT VV10 and SCF-VV10
    integrated absolute density differences against cached `dm1_cc` (electrons).
    Post-DFT VV10 energy density uses the same `Grid` coordinates, ordering,
    and weights as the density comparisons. Weights are loaded from the cache
    and must match the rebuilt grid's shape.
    Its weighted integral is checked against PySCF `nr_nlc_vxc` on the same
    density and grid (`rtol=1e-10`, `atol=1e-12` Hartree); mismatches are errors.
    After this check, `exc_post_grid` (VV10 energy per volume, in atomic units)
    is saved to `data_{name}_addon.npz`, preserving existing addon fields.
    Caches missing `dm1_dft`, `e_dft`, `dm1_cc`, or `weights` are logged and skipped
    without recomputing DFT.
- `train.py`: train/eval list setup → `train_model()`
- `test.py`: checkpoint loading and RKS/UKS model validation
- `benchmark_dft.py`: baseline DFT benchmarking
- `d3_para.py`: D3 parameter fitting/testing
- `collect_info.py`: collect validation CSV summaries
- `submit_direct_array_per_gpu.py`: per-GPU job dispatch helper

### Script directories (Slurm/HPC)
`scripts/train/`, `scripts/test/`, and `scripts/work/` assume local paths
(`~/anaconda3/envs/pyscf`, `~/backup-hd/tmp`, jemalloc). Keep those assumptions
out of importable Python modules.

## Data & Checkpoint Conventions
| Item | Default | Override |
|---|---|---|
| Main project path | repo root via `env_var.MAIN_PATH` | `DFT2CC_MAIN_PATH` |
| Training/grid data | `data/grids_dft` | `DFT2CC_DATA_DIR` |
| Test data | `data/test` | `DFT2CC_DATA_TEST_DIR` |
| Checkpoints | `checkpoints/checkpoint_<save_dir>` | `--load <save_dir> --load_epoch <epoch>` |

- Dataset names: `gmtkn-def2`, `gmtkn-diet30-def2`, `gmtkn-diet100-def2`, `dft-fitset-def2`
- Basis names: `def2-QZVPPD`, `def2-TZVPPD`, `def2-QZVP(D)`

### Training Logs and Loss Snapshots
- `log/train-<run-id>.log`: startup arguments and dataset sizes, checkpoint path,
  and epoch-level train/eval loss, learning rate, and elapsed-time summaries.
- `checkpoints/checkpoint_<save_dir>/loss/train-<epoch>.csv` and
  `eval-<epoch>.csv`: per-sample loss records for the training and evaluation
  sets. The run identifier in the log often matches the checkpoint directory;
  verify the recorded checkpoint path and arguments when associating artifacts.
- Loss CSVs are written when a best-loss metric improves or the checkpoint
  stride is reached (`eval_step * 32`); they are not necessarily emitted at
  every logged evaluation. The last CSV epoch can therefore precede the last
  epoch in the log and should not be treated as proof that training stopped.
- Interpret zero-valued loss columns with the run configuration and loss
  construction in `cc2cc/utils/ModelClass.py`; a CSV value alone does not show
  whether that component was active for the run.

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
