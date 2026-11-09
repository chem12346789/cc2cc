# Architecture Reference

Detailed module map and environment reference for the cc2cc DFT codebase.
Codex reads this only when it needs full structural detail — keeping AGENTS.md lean.

## Full Repository Layout

### `cc2cc/` — importable package
- `generate_cc_data.py`: shared restricted/unrestricted CC data generation. `cc()` takes
  an explicit `unrestricted` argument, defaulting to `False`. Restricted CCSD(T)
  requires `mol.spin == 0`; unrestricted CCSD(T) can also use spin-zero systems.
- `train_model.py`: training loop, W&B logging, distributed barriers
- `test_model.py`: shared restricted/unrestricted model-in-SCF validation,
  including post-SCF VV10 energy corrections. `validate_model()` takes an
  explicit `unrestricted` argument.
- `benchmark_rks.py`, `benchmark_uks.py`: benchmark helpers

### `cc2cc/utils/` — shared utilities and domain logic
- `parser.py`: CLI arg definitions; reuse `add_args()` / `gen_name_args()`
- `helper/runtime/env_var.py`: project paths, grid env settings, thread/GPU info
- `mol.py`: dataset loading, molecule rotation, and PySCF molecule construction;
  source of truth for molecule names
- `Grids.py`, `GridsGPU.py`: CPU/GPU grid construction. `Grid` is CPU by
  default; keep GPU classes lazily imported
- `grids/`: `Grids_rotate.py` and `GridsGPU_rotate.py` grid variants; the GPU
  variant is opt-in.
- `modelscf.py`: shared CPU model potential and gradient hooks with an explicit
  `unrestricted` argument.
- `modelscf_gpu.py`: shared GPU model potential hooks with explicit spin mode.
  GPU gradients are not implemented.
- `reference/`: shared B3LYP, CCSD(T), HF, VV10, and ZMP reference modules,
  plus cached DFT reference data utilities.
- `helper/units.py`: energy and dipole conversion factors.
- `helper/chemistry/basis.py`: basis/ECP element tables and generation.
- `helper/chemistry/`: basis/ECP generation, grid energy, geometry, tensor, and
  density helpers.
- `helper/chemistry/tensor_helper.py`: public `parallel_einsum` and `w_einsum`
  contractions shared by CCSD(T) modules.
- `helper/workflow/`: data records, run support, and schedulers.
- `helper/runtime/`: environment, system information, timing, and opt-in XC
  profiling. Import helper modules from their subpackages.
- `ccsd_t/`: PySCF CCSD(T) density and lambda helper modules, including
  unrestricted 1- and 2-RDM intermediates and public shared fused helpers.
- `zmp/`: inversion, constraint generation, and occupied-virtual training modules.
- `reference/reference_cc.py`: shared CCSD(T) setup, target densities and gradients,
  and restricted/unrestricted CC energy grids; callers pass `unrestricted`.
- `reference/reference_hf.py`: shared restricted/unrestricted HF energy grids; callers
  pass `unrestricted` explicitly.
- `reference/reference_b3lyp.py`: shared restricted/unrestricted DFT energy, grid input,
  and gradient helpers; callers pass `unrestricted` explicitly.
  `helper/chemistry/grid_energy.py` contains shared grid-energy helpers.
  - Restricted and unrestricted energy helpers return `ecp_cc_grids`,
    `ecp_dft_grids`, and `ecp_hf_grids` for scalar pseudopotentials, or zeros
    for all-electron molecules. CC and DFT totals include these terms; the
    DFT-to-ZMP key conversion also produces `ecp_zmp_grids`. Previously
    generated archives do not gain these fields automatically, and
    `--if_continue` skips existing archives.
    UKS supplies its two spin densities to LibXC with `spin=1`, including
    net-spin-zero unrestricted states.
  - The `nuc_*_grids` use the point-nucleus potential, except that
    contributions within 1e-8 Bohr are omitted. The `exc_kinl` diagnostics use
    these grids with the gradient-based kinetic-energy density. Newly generated
    archives no longer include `nuc_erf_*_grids`; existing archives may retain
    them, and `--if_continue` skips existing files.
  - The nonlocal pseudopotential uses the AO-projected gauge
    `phi(r)^T S^-1 V_pp D phi(r)`, with `V_pp = hcore - kinetic - nuclear`.
    Its integral is `Tr(V_pp D)` up to grid quadrature error; it is not a
    pointwise local potential times the electron density.
- `reference/reference_zmp.py`: shared ZMP inversion and diagnostic prints. Generation
  evaluates `e_zmp` with the non-density-fitted B3LYP reference at the ZMP
  density, matching the exact Coulomb and exchange energy-density grids. It
  stores `grad_zmp` using `zmp_baseline_gradient(mdft, dm1_zmp)`: an analytic
  baseline gradient with moving-grid response and an occupied-Fock-block
  energy-weighted density for the Pulay term. It excludes rebuilt-ZMP orbital
  response; the learned correction must contribute its explicit derivative,
  Pulay term, and grid response separately. Regenerate archives from the old
  density-fitted baseline to update `e_zmp`, `grad_zmp`, `energy_train`, and
  `e_post_vv10_zmp`; `--if_continue` skips existing archives.
  No dedicated full-pipeline ZMP finite-difference routine is currently implemented.
  Generation stops at an SCF-converged stage once the integrated
  density error is at most 10% of the DFT error and the I-value error is at
  most 0.1% of the DFT error, both measured against the target density.
  These thresholds are based on the 16 runs in `log/gen_data-*.log`, which
  first satisfy both at lambda 128--1024, before later stagnation or divergence.
  Comparisons use scaled DFT errors rather than division; a zero DFT error
  requires zero ZMP error. The existing divergence fallback is unchanged.
  `zmp/zmp.py` implements the inversion SCF equations. RZMP and UZMP use PySCF's
  native `_DFHF` mixin, as its dynamically generated DFRKS/DFUKS classes do.
  Their DIIS extrapolation uses all available Fock/error pairs, up to the
  configured `diis_space`; history starts on the second SCF cycle.
  Both Coulomb J and exchange K in the inversion use the standard
  `.density_fit()` auxiliary basis and shared `with_df` object. Dataset
  baseline energies and gradients instead use the non-density-fitted reference.
- `zmp/zmp_constraints.py`: optional full-OV stationarity and compact observable
  constraints for ZMP generation and training.
  Both `--zmp_feature_ov` and `--zmp_observable_response` default to `true`;
  pass `false` explicitly to disable either option.
  `gen_data.py --zmp_feature_ov true` streams float64
  `(center, feature, cube_sample, virtual, occupied)` Jacobians to per-spin
  `.npy` sidecars beside the normal dataset. Load with `np.load(..., mmap_mode="r")`.
  The derivative is `J = dx/dkappa_ai`, with
  `dP = n*(Cv*kappa*Co.T + Co*kappa.T*Cv.T)`, `n=2` (RKS) or `1` (UKS).
  Thus `F_correction_ai = sum_x (dE_correction/dx)*J_xai/(2*n)`;
  use the derivative of the complete quadrature-weighted model energy.
  Accumulate all grid blocks before forming a nonlinear stationarity loss.
  A full AO-pair Jacobian is never constructed.
  `--zmp_observable_response true` also/instead saves three total dipoles,
  six raw electronic second moments (`xx,xy,xz,yy,yz,zz`, not traceless
  quadrupoles), and one electronic Gaussian projection per atom
  (`exp(-|r-R_atom|^2/(1 Bohr)^2)`). Dipoles include the nuclear contribution;
  moments/projections are positive-electron-number observables. The origin is
  zero; dipoles are in e*Bohr and second moments in electrons*Bohr^2.
  Dipole/moment AO operators are analytic; atom projections use the saved grid.
  The compact local potential basis consists of these same electronic operators.
  Its coefficient units are Hartree divided by the corresponding operator unit.
  At fixed ZMP density both baseline Fock diagonal blocks are diagonalized;
  saved energies/gaps are not inversion eigenvalues. The response Hamiltonian
  is this block-diagonal baseline, with independent-particle response only:
  no induced Hartree/XC coupling, SCF reconstruction, or stability guarantee.
  `R_mk = -2*n*sum_ai O_mai*B_kai/max(gap_ai,gap_floor)`.
  Nonpositive gaps are errors; small positive gaps use
  `--zmp_response_gap_floor` (default `1e-3` Hartree). Raw gaps are retained.
  Each `zmp_potential_feature_file_s*` stores
  `T_kx = pinv(B.T)_k,ai * J_xai/(2*n)` in
  `(center, feature, cube_sample, potential)` order. This defines an unweighted
  OV least-squares projection with pseudoinverse cutoff `1e-10`; dependent
  potential directions use the minimum-norm solution.
  Contract model energy derivatives with `T`, add
  `zmp_baseline_potential_coeff_s*`, and multiply by `zmp_observable_response_s*`
  to approximate the change from `zmp_observable_reference` toward
  `zmp_observable_target` (the CC density). Sum the UKS spin contributions;
  a common scalar potential uses the same coefficients for both spins.
  This compact constraint misses density errors outside the selected subspace.
  Metadata includes the determinant, MO basis/occupations, baseline Fock,
  overlap, serialized molecule/basis/ECP and density-fitting auxiliary molecule,
  both quadrature grids and cube layout.
  Sidecar filenames are relative to the dataset directory; keep them together.
  `--zmp_constraint_metadata true --zmp_feature_ov false
  --zmp_observable_response false` includes fixed-reference metadata when
  generating datasets without the other ZMP constraints. Existing datasets are
  skipped with `--if_continue true`; regenerating them requires
  `--if_continue false` and reruns the CC calculation.
  `--zmp_response_block_size` (default 32 centers) bounds transient Jacobian
  blocks, not disk size or model autograd memory. No training losses or force
  conventions are changed by these generation options.
- `--zmp_ov_loss true` is no longer supported and is rejected.
- `reference/reference_cc.py`: shared CCSD solver settings, CCSD(T) lambda solving,
  triples-corrected target density and CC-gradient one-particle intermediates.
  `ccsd_t_target_gradient()` handles restricted/unrestricted two-particle
  intermediate compression, electronic/nuclear gradient assembly, and symmetry
  projection without modifying the density intermediates used for training.
  Both generation entry points and displaced ZMP references use this helper;
  generation/reference CC convergence settings are identical: energy `1e-12`,
  amplitudes/lambda `1e-10`, and CC/lambda iteration limits of 200.
  Unrestricted lambda intermediates reuse repeated contractions without
  reordering the numerical updates, releasing reused results within each block.
  Same-spin lambda and unrestricted 1-/2-RDM outer products and triples
  symmetry/denominator operations
  use fused Numba kernels, avoiding full-size accumulation temporaries and
  denominator arrays. Kernels preserve accumulation order without fast math,
  use the smaller of the active Numba and PySCF thread counts, and restore the
  caller's Numba thread mask. First-use execution includes JIT compilation;
  performance comparisons should use warmed kernels. The 1-RDM combined
  amplitudes use addition/division without the 2-RDM's symmetry operation.
  Mixed-spin outer-product chains also use fused kernels, including in-place
  accumulation through permuted output views. Mixed-spin division computes
  denominators and numerator combinations in parallel without full-size
  denominator or numerator temporaries. It preserves the 1-RDM's plain sum,
  the lambda/2-RDM's doubled-W sum, and left-to-right energy addition.
  Output traversal follows W's strides; downstream reduction layout can cause
  roundoff-level differences. The 1-RDM's pre-/post-W-update divisions remain
  separate. Reduction contractions and block-size selection are unchanged.
  Restricted 1-/2-RDM triples updates fuse the six disconnected outer products,
  W/V division, and their combination in one parallel Numba pass over each
  active block. Both virtual-blocked and occupied-blocked 1-RDM passes retain
  their original coefficients and addition order. W accumulation reuses the
  shared parallel update kernel; BLAS reductions, PySCF's compiled triples
  symmetry routine, and block-size selection remain unchanged. No full-size
  denominator or disconnected outer-product temporaries are allocated.
  `W` updates retain BLAS contractions and use a cached parallel Numba
  kernel for in-place accumulation, fusing scaling when present without a
  scaled temporary. Traversal follows the destination strides without copying
  operands; multiplication precedes addition/subtraction without fast math.
  The caller's Numba thread mask is restored, and the active Numba/PySCF
  minimum limits parallelism. Initial terms without an accumulation destination
  retain out-of-place scaling. Per-term and stage profiling instrumentation is
  removed; standard PySCF solver timing and block-size diagnostics remain.
  The triples debug `max_memory` value is the configured budget minus RSS sampled
  before the passes, not peak usage or system-free RAM. This budget guides block
  sizing rather than reserving memory; orbital dimensions and block-splitting
  heuristics also limit block sizes.
- `DataBase.py`, `ModelClass.py`: data loading, model init, checkpoints, losses.
  The `training_stage` context manager is shared with the training entry point.
- `reference/test_data_dft.py`: cached DFT reference generation.
- `helper/chemistry/density_metrics.py`: density comparisons.
- `helper/workflow/DataRecord.py`: CSV record keeping
  - Both generators save `coords` in Bohr aligned with their `weights` and
    feature grid (not a separately reconstructed B3LYP SCF grid), as well as
    PySCF `grid_atom_indices` (parent atom per quadrature entry; -1 is fixed
    padding). Point filtering is disabled: zero-feature centers, weights,
    grid targets and cube force Jacobians retain their original order.
  - Dataset tensors remain on CPU; batch device transfers do not evict them.
    Base/addon NPZ arrays are read lazily, with addon keys taking precedence,
    and both archives close after loading each system. Grid targets are retained
    only for absolute loss. Atomic forwards transfer model inputs and weights.
    Model XC evaluation returns energy and feature derivatives, with optional
    weight derivatives for the CPU RKS/UKS moving-grid gradient hooks.
  - Loaded checkpoint references are released after model/optimizer restoration.
    Checkpoint-size checks stream serialization into a byte counter rather than
    retaining a full serialized checkpoint in RAM.
- `model/`: neural model definitions; `--model NAME` → `model/NAME.py` with `Model`
  - `transformer+dense_mix_e3nn_4_correct_5` uses the `cube5` input contract:
    four alternating corners and the center in `ModelClass.CUBE_INDEX` order
    `[0, 8, 13, 24, 20]`, with `cube_type="cube5"` and `cube_size=5`.
    Each spacing-normalized e3nn expert uses those five points; the center
    branch and six-way output mixture are unchanged. `EDGE_SIZE=3` is required.
    This variant requires a fresh checkpoint. Its e3nn experts use the same
    point ordering when selecting their geometry through `point_indices`.
- `*.json`: dataset/split definitions (`gmtkn-def2`, `gmtkn-diet30-def2`,
  `gmtkn-diet100-def2`, `dft-fitset-def2`)
- PySCF CCSD(T) intermediate files: vendor-like scientific code — patch minimally

### Training stall diagnostics

Set `DFT2CC_TRACE_TRAIN=1` in the training job environment to emit flushed
`[train-progress]` start/done messages for logger setup, RNG restoration, batch
fetching, molecular/atomic forward passes, backward, and optimizer updates.
Default is `0`; other values are rejected. Each message identifies the PID
and stage; tensor stages also include the system, device, and grid count.

An unfinished stage dumps Python thread stacks to stderr every 120 seconds.
The watchdog is cancelled on success or exception; exceptions propagate without
a misleading `done` message. Tensor stages synchronize CUDA before reporting
completion, so diagnostic wall times include queued GPU work and tracing adds
synchronization overhead. Training mathematics is unchanged.
Enable tracing in a newly launched process; it cannot instrument an already
running worker. These diagnostics identify the slow stage, not a speedup.

### Training peak GPU memory

Set `DFT2CC_PROFILE_TRAIN_MEMORY=1` in the training job environment to log one
`[train-memory]` line per local training step, on every rank. Default is `0`;
other values are rejected, and enabling it requires a CUDA model.
Measurement starts before batch transfer and ends after backward, gradient
clipping, optimizer/scheduler updates and gradient clearing. CUDA synchronization
and peak-counter reset delimit each step; profiling adds synchronization overhead.
Logs include device, local step, molecule, whether force loss is active, initial
allocated MiB, peak allocated/reserved MiB, and the batch Jacobian's logical MiB.
These are process-local PyTorch allocator statistics, not total device usage or
memory allocated by other libraries. Reserved memory includes cached blocks from
earlier steps; no cache is emptied. The allocated peak includes resident model
and optimizer state, not just allocations made during the step.
Compare identical molecules, precision, model, and force-loss settings; report
the first-step peak separately from warmed-up steps because optimizer state is
initialized lazily. Profiling does not enable nuclear-gradient evaluation.

### GPU SCF XC profiling

`setup_gpu_test_job CPU_MEMORY GPU_MEMORY [USE_NUMBA]` exports its second
argument as `MAX_MEMORY_GPU` as well as using it for the free-GPU-memory
selection threshold. `run_test_job` forwards it as `--max_memory_gpu` (MB),
defaulting to 4000 without GPU setup. An explicit `--max_memory_gpu` in
`run_test_job` arguments comes last and overrides this value; GPU selection
still uses its original threshold in MiB.

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
The previous `AO scratch` interval is split into `block handoff`,
`AO scratch allocation`, and `AO scratch check`. Allocation timing is emitted
only when storage is actually allocated or grown, so its call count records the
allocation count. `AO weight preparation` covers density-channel half weighting
and layout bookkeeping; `AO weighting` covers the native `_scale_ao` call.
After matrix assembly, caller AO/weight views are explicitly released and timed
as `caller release`. When the generator resumes, it releases its retained block
arrays under `generator release` before constructing the next block. Deleting
references does not guarantee immediate device-memory reclamation; CuPy's memory
pool and stream ordering still apply. First-block AO/weight shapes, byte strides, dtypes,
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
  `Grid` → `cc(..., unrestricted=unrestricted)` → `.npz` grid data.
  It passes `unrestricted=False` for zero spin and `True` otherwise.
  - `cc()` calls the shared `get_vv10_data()` helper in
    `utils/reference/reference_vv10.py` and saves Post-DFT VV10 and SCF-VV10 energies,
    `dm1_scf_vv10`, and gradients in the main archive. It also
    saves separate VV10 results based on `dm1_zmp` with `_zmp`-suffixed keys.
    The helper separates `get_post_vv10_data()` and `get_scf_vv10_data()`.
    Shared B3LYP SCF and finite-difference helpers live in
    `utils/reference/reference_b3lyp.py`.
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
  - `--test_mode post-b3lyp` evaluates the model once on cached `dm1_dft`,
    using the B3LYP baseline. `--test_mode post-b3lyp-d3` uses that same
    density and reports `E_B3LYP + E_D3BJ + E_model[rho_B3LYP]` in
    `scf_d3bj_ene`; `scf_ene` remains the energy without D3BJ.
    The D3 mode uses B3LYP D3BJ damping unless all six custom D3 parameters
    are supplied.
    `--test_mode post-b3lyp-nl` also uses the B3LYP density and reports
    the post-SCF VV10 energy in `post_scf_vv10_ene`.
    `--test_mode post-b3lyp-vv10` uses cached `dm1_scf_vv10` with the
    B3LYP+VV10 baseline. None of these modes runs model SCF; `--max_cycle` is
    ignored for model evaluation.
    The default `--test_mode scf` preserves model SCF.
  - All post-SCF modes read and write only `${name}_dft.npz` in `DATA_TEST_PATH`
    (default `data/test/`), never training-data or addon caches.
    If the file or requested density/energy pair is missing, plain B3LYP or
    B3LYP+VV10 reference SCF runs with the requested grid level (RKS or UKS).
    Reference SCF uses GPU when CUDA is available, respecting `--cpu`; otherwise
    it uses CPU. Cached density matrices remain NumPy arrays on disk.
    The directory and file are created as needed. B3LYP saves `dm1_dft-b3lyp`,
    `dm1_dft`, and `e_dft-b3lyp`; VV10 saves `dm1_scf_vv10` and `e_scf_vv10`.
    Updates are atomic, preserve existing fields, and are reused on later runs.
    Unconverged references raise an error without being cached.
    These energy-only modes require the cached molecular orientation and reject
    nuclear gradients and rotation options.
  - Post-SCF CSVs retain the existing energy columns, add `test_mode`, and leave
    `converged` empty (no model SCF was attempted). The output model label is
    `${load}-${test_mode}`, isolating CSVs and continuation from model SCF.
    Use that label as `collect_info.py --model_load`; checkpoint loading still
    uses the original `--load`. For example, `--load atom-393567 --load_epoch 1115
    --test_mode post-b3lyp` writes under
    `validate/${basis}_atom-393567-post-b3lyp/1115/`.
  - `scripts/test/test_diet100.bash` accepts `TEST_MODE=post-b3lyp-d3` for
    either checkpoint; it uses standard B3LYP-D3BJ parameters for that mode.
    Model selections 0 and 1 default to `post-b3lyp`, and selection 2 to `scf`.
    Other `TEST_MODE` values can also override these defaults.
- `benchmark_dft.py`: baseline DFT benchmarking
- `d3_para.py`: D3 parameter fitting/testing
- `collect_info.py`: collect validation CSV summaries using the shared
  `AU2KCALMOL` conversion from `helper/units.py` (previously 627.5094740631).
  Importing it now initializes `cc2cc.utils` and its scientific dependencies.
  Existing completed cache entries and saved summaries are not recalculated.
  The density-effects analysis script also uses `Collect_info.subset_alias()`.
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

- `scripts/gen_data/gen_data_direct.bash`: submit from the repository root with
  `sbatch scripts/gen_data/gen_data_direct.bash`. Slurm runs one generator per
  array task: `MP_NUMBER=SLURM_ARRAY_TASK_ID` and `MP_TOTAL=SLURM_ARRAY_TASK_COUNT`.
  Use contiguous, zero-based arrays (for example `0-3%1`); there is no Bash
  worker loop or standalone fan-out.

- `scripts/direct_run_per_gpu.py`: background-by-default, single-host alternative without
  Slurm. For example:
  `python scripts/direct_run_per_gpu.py --script test/test_diet100.bash --gpus 0,1`.
  Uses the script's array and CPU directives (`--array` overrides the array),
  supplies `MODEL_INDEX`, `SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_TASK_COUNT`, and
  `SLURM_CPUS_PER_TASK` for the existing Bash helpers, and binds each job to a
  GPU UUID. The array's `%limit` sets the maximum concurrent jobs per GPU
  (`--array-concurrency` overrides it); without a limit, one job runs per GPU.
  Optional `--time-array` prioritizes longer tasks and skips zero runtimes; the
  task count remains the size of the configured array before filtering.
  Memory (default 15 GiB) and optional power filters are instantaneous checks
  before dispatch, not reservations; coordinate GPU ownership with other users.
  With no `--gpus`, respects `CUDA_VISIBLE_DEVICES` or discovers all local GPUs.
  If `nvidia-smi` is missing, it warns and skips GPU selection and filters;
  jobs inherit CUDA visibility and must choose their own device. In that mode,
  the array limit or `--array-concurrency` is a global limit (default 1).
  `--dry-run --gpus 0,1` previews without probing `nvidia-smi`, so it may not
  reflect the fallback; it does not access GPUs, write files, or execute jobs.
  Default run log directories use the run's timestamp.
  `[TASK-LOG]` lines show the actual stdout/stderr paths for test jobs and
  `gen_data_direct.bash`; `[LAUNCHER-LOG]` identifies the launcher log itself.
  Per-job PySCF scratch directories use `PYSCF_TMPDIR`, `TMPDIR`, or repository
  `tmp` (retained for inspection).
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
  For rotation checks, use `--script test/test_rotate.bash --gpus 0`.
  The `rotate` profile selects `molecule_Rotation` from `rotation-def2.json`:
  the CH4, H2O and open-shell OH center geometries from `force-def2.json`,
  each in its reference orientation, rotated 45 degrees about y, and rotated
  30 degrees about x followed by 45 degrees about z (active rotations about
  the origin, coordinates in Angstrom). Charges, spins and atom order are
  preserved. Distinct molecule names keep results for all nine geometries.
  The script defaults to energy-only evaluation (`IF_GRAD=0`); optionally use
  `IF_GRAD=1 GRID_RESPONSE=1` for moving-grid response, or
  `GRID_RESPONSE=0` for fixed-grid CPU gradient evaluation.
  `ROTATION_SET` overrides the selected molecule/group. No extra runtime
  rotation is applied. Rotations rebuild atom-centered quadrature grids;
  numerical energy differences therefore include grid discretization effects.
  Matching rotation energies/forces are not asserted automatically.
  `notebooks/dgx1_test_rotate.ipynb` reads the rotation log, reports energy
  differences and optional gradient covariance (`g_rotated = g_reference @ R.T`).
  `notebooks/dgx1_test_force.ipynb` uses `force-def2` and checks Cartesian
  finite differences only when all required converged runs are available.
  Both are analysis-only: missing runs are reported as PENDING and neither
  notebook launches SCF. The shared log parser accepts explicitly logged CLI
  convergence settings when GPU4PySCF UKS omits its per-run SCF banner.
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
