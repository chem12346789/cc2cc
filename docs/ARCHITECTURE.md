# Architecture Reference

Detailed module map and environment reference for the cc2cc DFT codebase.
Codex reads this only when it needs full structural detail — keeping AGENTS.md lean.

## Full Repository Layout

### `cc2cc/` — importable package
- `gen_cc.py`, `gen_ucc.py`: closed-/open-shell CC data generation
- `train_model.py`: training loop, W&B logging, distributed barriers
- `test_model_rks.py`, `test_model_uks.py`: model-in-SCF validation, including
  post-SCF VV10 energy corrections
- `benchmark_rks.py`, `benchmark_uks.py`: benchmark helpers

### `cc2cc/utils/` — shared utilities and domain logic
- `parser.py`: CLI arg definitions; reuse `add_args()` / `gen_name_args()`
- `env_var.py`: project paths, grid env settings, thread/GPU info
- `mol.py`: molecule/dataset definitions — source of truth for molecule names
- `Grids.py`, `GridsGPU.py`: CPU/GPU grid construction. `Grid` is CPU by
  default; keep GPU classes lazily imported
- `modelscf_rks.py`, `modelscf_uks.py`, `_gpu` variants: custom effective-
  potential/gradient hooks for PySCF SCF
- `get_dft_energy_*.py`, `get_dft_grad_*.py`: energy and gradient helpers
- `grid_neighbors.py`: opt-in, data-only nearest-grid prototype. Run
  `python -m cc2cc.utils.grid_neighbors input.npz --coords-key coords --output neighbors.npz`
  using the selected project interpreter and coordinates exported from the actual
  B3LYP `mdft.grids`. Saves coordinates and center-first indices for 26 nearest
  other entries, without changing training or generating electronic-structure data.
  Gather features by these indices; reconstruct relative positions as
  `coords[indices] - coords[:, None, :]` in blocks. Keep the center's quadrature
  weight for energy integration, not the sum of its neighbors' weights.
  Coincident grid entries are retained; equal-distance ordering is unspecified.
  These irregular neighborhoods are not Cartesian cubes and are not compatible
  with existing cube checkpoints. Force derivatives must include moving-grid
  effects; neighbor exchanges introduce discrete boundaries.
- `zmp_reference.py`: shared ZMP inversion, diagnostic prints, and numerical
  nuclear-gradient reference. Generation stores `grad_zmp` using
  `zmp_baseline_gradient(mdft, dm1_zmp)`: an analytic B3LYP baseline gradient
  at the supplied ZMP determinant, with moving-grid response and a baseline
  occupied-Fock-block energy-weighted density for the Pulay term. It excludes
  rebuilt-ZMP orbital response; the learned correction must contribute its
  explicit derivative, Pulay term, and grid response separately. Existing
  cached `grad_zmp` values must be regenerated before using this convention.
  Full-pipeline finite differences remain available as trajectory diagnostics.
  `get_zmp.py` forwards legacy entry points and `zmp.py` implements the inversion
  SCF equations.
- `ccsd_t_target.py`: shared CCSD solver settings, CCSD(T) lambda solving,
  triples-corrected target density and CC-gradient one-particle intermediates.
  `ccsd_t_target_gradient()` handles restricted/unrestricted two-particle
  intermediate compression, electronic/nuclear gradient assembly, and symmetry
  projection without modifying the density intermediates used for training.
  Both generation entry points and displaced ZMP references use this helper;
  generation/reference CC convergence settings are identical: energy `1e-12`,
  amplitudes/lambda `1e-10`, and CC/lambda iteration limits of 200.
- `DataBase.py`, `TestDataDFT.py`, `DataRecord.py`, `ModelClass.py`: data
  loading, record keeping, model init, checkpoints, losses
  - Select `--model transformer+dense_mix_e3nn_4_no_cube` for opt-in
    nearest-grid **energy training** (`--if_grad 0`). Its density input remains
    `(N_centers, 4, 27)` and output `(N_centers, 1)`, with a separate
    `relative_positions` tensor `(N_centers, 27, 3)` in Bohr. Slot zero is the
    center. Four `model_e3nn_correct_no_cube.E3nn` experts use the
    `model_e3nn_correct` radial tensor-product/readout architecture, accepting
    per-input `edge_vec` instead of fixed cube geometry. Radii are normalized
    by `EDGE_LEN`, with `lmax=0` matching the corrected cube model.
    Mean pooling makes neighbor order irrelevant when features and positions
    are permuted together. This replaces the initial no-cube moment encoder;
    its earlier checkpoints are incompatible.
  - Both generators now save `coords` in Bohr aligned with their `weights` and
    feature grid (not a separately reconstructed B3LYP SCF grid). The dataset
    extracts central features from existing cubes and builds/caches 27-neighbor
    indices once per loaded molecule, before center filtering. It keeps all
    central features on CPU and gathers neighborhood inputs on batch transfer;
    filtered centers can still use any original grid point as a neighbor.
    Center integration weights are unchanged. Existing archives need matching
    coordinates in the base or addon archive; missing coordinates are an error.
    Loading prints each molecule's unweighted mean neighbor-center distance in
    Bohr over the 26 neighbors of retained centers, excluding the center itself.
  - This mode does not eliminate cube generation from the existing generators;
    it reuses cube centers. Nearest-neighbor distances can vary strongly across
    atom-centered quadrature grids. No-cube checkpoints are distinct from cube
    checkpoints. Force training and SCF XC integration are not implemented for
    this mode and are explicitly unsupported; cube Jacobians are not reused.
  - Dataset tensors remain on CPU; batch device transfers do not evict them.
    Base/addon NPZ arrays are read lazily, with addon keys taking precedence,
    and both archives close after loading each system. Grid targets are retained
    only for absolute loss. Atomic forwards transfer model inputs, weights,
    and optional neighborhood geometry.
  - Loaded checkpoint references are released after model/optimizer restoration.
    Checkpoint-size checks stream serialization into a byte counter rather than
    retaining a full serialized checkpoint in RAM.
- `model/`: neural model definitions; `--model NAME` → `model/NAME.py` with `Model`
- `*.json`: dataset/split definitions (`gmtkn-def2`, `gmtkn-diet30-def2`,
  `gmtkn-diet100-def2`, `dft-fitset-def2`)
- PySCF CCSD(T) intermediate files: vendor-like scientific code — patch minimally

### GPU SCF XC profiling

Set `DFT2CC_PROFILE_XC=1` before the existing test command to log stage totals
at INFO verbosity for every modified GPU RKS/UKS effective-potential evaluation.
The default is `0` (no events, synchronization, or profiling output).
This profiles SCF integration only, not the separate nuclear-gradient hooks.
Stages cover setup, cube coordinates, AO evaluation (including input-buffer
allocation), density, component XC functionals, cube reshaping, model
forward/backward plus DLPack exchange, XC contraction, AO weighting, matrix GEMM, and
symmetrization/orbital unsorting. Counts for the per-block stages give the
number of integration blocks; wall and GPU elapsed seconds are summed per stage.
Each report also lists every block's `base_grids` (integration centers) and
`cube_points` (expanded AO evaluation points), including the final partial block.
Block records are printed after timing, not inside the integration loop.
The former matrix-assembly stage is now split into `AO weighting` and
`matrix GEMM`; UKS totals include both spins. Both hooks use the original
float64 `cp.dot` followed by addition into the potential matrices. The
`matrix GEMM` timer includes both the matrix product and this addition;
direct cuBLAS accumulation has been rolled back.
Weighted-AO scratch storage is reused across blocks, growing only when needed.
UKS processes alpha then beta through the same scratch buffer; `AO weighting`
and `matrix GEMM` therefore each have two calls per block in UKS (one in RKS).
Their total times still include both spins. All scratch storage is local to one
integration call; no AO cache or additional screening approximation is introduced.
The old `AO weighting` region is further separated into `AO scratch` (buffer
allocation/checks), `AO weight preparation` (density-channel half weighting and
layout bookkeeping), and `AO weighting` (the native `_scale_ao` call). Sum these
three stages when comparing to older logs. Scratch and preparation each count
once per block, including UKS. First-block AO/weight shapes, byte strides, dtypes,
contiguity flags and scratch bytes are printed after timing. No layout copies,
custom weighting kernels, or screening changes are enabled by this diagnostic.

Profiling joins the current CuPy and PyTorch streams and synchronizes at every
boundary. GPU elapsed time includes host launch gaps, not just kernel execution.
This deliberately perturbs overlap and adds overhead: use it to locate expensive
stages, then disable it for throughput comparisons. Compare warmed-up SCF
iterations separately from the first call, which can include model compilation.
Stage wall totals exclude profiler bookkeeping, logging, and the initial stream
drain, so they need not equal the enclosing `vxc` timer exactly.

### Top-level entry points

Validation CSV cleanup: `python scripts/clean_validation_csv.py --dry-run`
recursively previews removal of rows whose `scf_d3bj_ene` is empty or whitespace
under `validate/def2-QZVP(D)_atom-*`. Use `--apply` to rewrite affected files;
without either flag it only previews. Each removed row and a summary are printed.
Missing-column files are reported and skipped; malformed rows raise an error.
Nonempty values (including `0` and literal `NaN`) are retained. Retained CSV
records keep their original formatting. Writes use atomic replacement and preserve
permission bits; run only when validation writers are stopped. No backups are
created, and failures may occur after earlier files have been updated.

- `gen_data.py`: molecule selection → `gen_mole()` → optional MD/rotation →
  `Grid` → `cc()`/`ucc()` → `.npz` grid data
  - `cc()` and `ucc()` call the shared `get_vv10_data()` helper in
    `utils/b3lyp_reference.py` and save Post-DFT VV10 and SCF-VV10 energies,
    `dm1_scf_vv10`, and gradients in the main archive. They also
    save separate VV10 results based on `dm1_zmp` with `_zmp`-suffixed keys.
    The helper separates `get_post_vv10_data()` and `get_scf_vv10_data()`.
    `get_vv10_data()` derives its policy from `suffix`, not caller flags:
    `"_zmp"` sets evaluation on, SCF-VV10 off, and gradient checks off;
    the default `""` enables SCF-VV10, all gradients, and gradient checks.
    Existing cached SCF-VV10 ZMP fields are preserved, not refreshed.
    The existing-cache branch uses the same helper and saves to the addon,
    including `_zmp`-suffixed results when cached ZMP density and energy exist.
    Existing-cache comparisons report B3LYP/post-DFT VV10 and SCF-VV10
    integrated absolute density differences against cached `dm1_cc` (electrons).
    Post-DFT VV10 energy density uses the same `Grid` coordinates, ordering,
    and weights as the density comparisons. Weights are loaded from the cache
    and must match the rebuilt grid's shape.
  - VV10 calculation policy is independent of `--if_eval`: unsuffixed results
    include SCF-VV10 gradients, finite-difference checks, and density comparisons.
    ZMP results include `grad_post_vv10_zmp` but omit SCF-VV10, finite-difference
    checks, and density comparisons; energy comparisons remain enabled.
    `grad_post_vv10` is analytical: B3LYP is reconverged from the cached density
    at tight tolerances, and PySCF's coupled-perturbed RKS/UKS solver supplies
    its orbital response. The VV10 and B3LYP atom-centered grid-motion terms
    are included. `grad_scf_vv10` uses PySCF's SCF-VV10 gradient.
    SCF-VV10 uses tight SCF tolerances and moving-grid response; force
    generation requires converged wavefunctions even when optional evaluation
    convergence checks are disabled.
    Gradients have shape `(natm, 3)`, in Hartree/Bohr; forces are their negatives.
    Unsuffixed results also save `grad_post_vv10_fd` for multi-atom molecules,
    using central differences with a 0.001 Bohr step (six B3LYP single points
    per atom), and reports maximum errors for both Post-DFT VV10 and B3LYP.
    The VV10 energy-density weighted integral is checked against PySCF
    `nr_nlc_vxc` on the same
    density and grid (`rtol=1e-10`, `atol=1e-12` Hartree); mismatches are errors.
    After this check, `exc_post_vv10_grid` (VV10 energy per volume, in atomic
    units), `e_post_vv10` (B3LYP plus the VV10 correction), and `e_scf_vv10`
    are saved. Cached updates preserve existing addon fields while removing
    legacy `exc_post_grid` and `enlc_post` fields.
    Caches missing `dm1_dft`, `e_dft`, `dm1_cc`, or `weights` are logged and skipped
    without recomputing DFT.
- `train.py`: train/eval list setup → `train_model()`
- `test.py`: checkpoint loading and RKS/UKS model validation
- `benchmark_dft.py`: baseline DFT benchmarking
- `d3_para.py`: D3 parameter fitting/testing
- `collect_info.py`: collect validation CSV summaries
  - Select the corrected model energy with
    `--energy_column post_scf_vv10_ene` (Python:
    `Collect_info(..., energy_column="post_scf_vv10_ene")`).
    The default is `scf_d3bj_ene`, preserving automatic D3BJ correction.
    Other columns must already be present and are reported under their own names;
    no D3BJ correction is generated. Non-default selections use separate completion
    cache entries and summary backups, but keep the same input CSV path.
  - `scripts/calculator/calculator_diet100.bash` accepts
    `model:epoch:energy_column[:note]` entries in `model_load_epoch_pairs`, for
    example `atom-dft:1000:post_scf_vv10_ene`; it forwards the energy column to
    the collector and displays an optional note in the results table.
- `scripts/submit_direct_array_per_gpu.py`: per-GPU job dispatch helper. GPU probes
  sample roughly once per second for `--probe-timeout-sec` (default 60 seconds)
  after starting, with roughly up to two extra seconds for clock rounding and
  the final sampling interval. Selection uses
  minimum sampled free memory (`--min-free-memory`, GiB) and maximum sampled
  power (`--max-gpu-power`, watts), not averages. Peaks between samples can be
  missed; measurements do not reserve GPU resources. The total wait is bounded
  by twice the sampling duration plus 10 seconds to allow queue/startup time;
  incomplete probes are rejected.

- `scripts/direct_run_per_gpu.py`: background-by-default, single-host alternative without
  Slurm. For example:
  `python scripts/direct_run_per_gpu.py --script test/test_diet100.bash --gpus 0,1`.
  Uses the script's array and CPU directives, supplies `MODEL_INDEX`,
  `SLURM_ARRAY_TASK_ID`, and `SLURM_CPUS_PER_TASK` for the existing Bash helpers,
  and binds each job to a GPU UUID. The array's `%limit` sets the maximum
  concurrent jobs per GPU (`--array-concurrency` overrides it); without a limit,
  one job runs per GPU.
  Optional `--time-array` prioritizes longer tasks and skips zero runtimes.
  Memory (default 15 GiB) and optional power filters are instantaneous checks
  before dispatch, not reservations; coordinate GPU ownership with other users.
  With no `--gpus`, respects `CUDA_VISIBLE_DEVICES` or discovers all local GPUs.
  `--dry-run --gpus 0,1` previews without GPU access, file writes, or job execution.
  Default run log directories use the run's timestamp.
  Each run has separate stdout/stderr logs and per-job PySCF scratch directories
  under `PYSCF_TMPDIR`, `TMPDIR`, or repository `tmp` (retained for inspection).
  `--log-dir` must name a new directory. Job failures are logged and produce a
  nonzero worker exit status, saved in `exit-code`. By default the command returns
  after starting a detached launcher; its stdout/stderr go to `launcher.log`
  and its PID is saved in `launcher.pid` inside the run's log directory.
  Monitor with `tail -f <log-directory>/launcher.log`; stop with
  `kill -TERM <launcher-pid>` (also terminates active job process groups).
  The matching `[STOP] kill -TERM <launcher-pid>` command is recorded in `launcher.log`.
  The initial command's success means the launcher started, not that jobs passed.
  Use `--foreground` to stay attached and receive the final job exit status directly.
  For force testing, use `--script test/test_force.bash --gpus 0`;
  its array runs one job with 20 CPU threads. The `force` molecule profile selects
  `molecule_Force` and redirects calculation output to
  `log/<checkpoint>/<epoch>/Force_0.out` and `.err`, like the other profiles.
  Dry runs always stay foreground.

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
- `--load <save_dir> --load_epoch N --if_resume 1` restores the model and
  optimizer after completed epoch N and starts training at N + 1. Scheduler
  progress is reconstructed using N + 1 completed epochs and the current
  schedule/data settings; those settings must match for schedule continuity.
- `log/train-<run-id>.log`: startup arguments and dataset sizes, checkpoint path,
  and epoch-level train/eval loss, learning rate, and elapsed-time summaries.
- `checkpoints/checkpoint_<save_dir>/loss/train-<epoch>.csv` and
  `eval-<epoch>.csv`: per-sample loss records for the training and evaluation
  sets. The run identifier in the log often matches the checkpoint directory;
  verify the recorded checkpoint path and arguments when associating artifacts.
- Loss CSVs are written when a best-loss metric improves or the checkpoint
  stride is reached (`eval_step * 32`). For `cosine_warm` and `cosine_warm2`,
  the last scheduled evaluation before each LR restart forces a checkpoint
  and loss CSVs, without adding evaluations outside `eval_step`. With
  zero-based epoch labels, `cosine_T=160`, `cosine_T_mult=2`, and `eval_step=5`
  produce pre-restart checkpoints at 155, 475, 1115, etc. If a cycle's final
  low-LR update falls on a scheduled evaluation, that epoch is saved instead;
  its optimizer LR already holds the next cycle's LR.
  Loss CSVs are not necessarily emitted at
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
