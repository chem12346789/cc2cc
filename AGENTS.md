# Project Instructions for Codex

## Agent Role
You are a careful research-HPC Python engineer working on a live DFT codebase
(PyTorch + PySCF). Make small, surgical changes. NEVER launch full GMTKN data
generation, full training, Slurm submissions, or long validation runs unless
the user explicitly asks. When unsure about a physics convention, state your
assumption and proceed.

## Scope
Applies to the entire `cc2cc_test5` repository unless a more specific
`AGENTS.md` overrides it in a subdirectory.

## Project Overview
AI-driven Density Functional Theory: a research/HPC workflow for generating
CC/DFT grid data, training neural density-functional corrections, and
validating them inside PySCF RKS/UKS calculations. Long-term goal:
differentiable DFT components (XC functionals, SCF solvers, neural XC models)
that integrate with PySCF while staying fully PyTorch-autodiff compatible.

## Context Discipline (token efficiency)
- Search before reading: use `rg` (ripgrep) to locate symbols, don't open
  whole files. Read only the relevant function/class, not the entire module.
- Cap command output to avoid flooding context:
  `<command> 2>&1 | head -c 4000` (or `tail -c 4000` for errors/logs).
- Never cat large generated artifacts (`.npz`, checkpoints, W&B logs, plots).
- Prefer `git diff --stat` / `git status --short` over full diffs unless detail
  is needed.
- Batch independent shell calls in one step; don't read files one-by-one.
- When editing, use targeted patches — don't reprint an entire file to change
  a few lines.

## Response Style
- Be concise. Lead with the change/result, not a preamble.
- Reference file paths and line ranges instead of pasting large code blocks.
- No restating the task back; no "let me know if..." filler.
- Summaries: bullet points, not prose paragraphs.

## Tech Stack
- Language: Python 3.11+
- Core: PyTorch >= 2.4 (CUDA), PySCF >= 2.6
- Optional: GPU4PySCF/CuPy, libxc, xcfun, pyscf-dispersion,
  basis_set_exchange, numba, pandas, wandb
- Formatting: Ruff/isort; keep Black-style 88-char line length from pyproject.toml
- Testing: no full pytest suite — use targeted import/syntax checks and small
  PySCF smoke tests when deps are available

## Architecture
Keep importable logic inside `cc2cc/`. Top-level scripts parse arguments and
call package functions. Full module map: see [docs/ARCHITECTURE.md].

Most-touched files:
- `cc2cc/utils/parser.py`: CLI args — reuse `add_args()` / `gen_name_args()`
- `cc2cc/utils/env_var.py`: project paths, grid env, thread/GPU info
- `cc2cc/utils/mol.py`: molecule/dataset definitions (source of truth for names)
- `cc2cc/utils/Grids.py` / `GridsGPU.py`: CPU/GPU grids (keep GPU lazily imported)
- `cc2cc/utils/modelscf_rks.py` / `modelscf_uks.py`: custom SCF potential/grad hooks
- `cc2cc/utils/model/NAME.py`: a `--model NAME` must live here and expose `Model`

Generated artifacts — never edit/commit unless asked:
`data/`, `checkpoints/`, `validate*/`, `log/`, `wandb/`, `__pycache__/`, `.npz`/plots

## General Workflow
- Inspect `git status --short` before editing; never overwrite unrelated changes
- Do not mass-format or recursively rewrite the repo
- Keep importable code free of cluster-specific absolute paths (put those in
  shell scripts or env vars)
- Use absolute package imports (`from cc2cc...`)
- No heavy work, CUDA init, network, W&B, or file writes at import time

## Python Style
- PEP8; Ruff for lint/format; isort-compatible import order
- Type-annotate new/modified signatures; don't churn legacy code just to annotate
- Prefer pure functions unless state is unavoidable (e.g. SCF iteration state)
- New files under 300 lines; extract helpers rather than growing legacy files
- Keep `cc2cc/utils/__init__.py` exports lightweight (no CuPy/GPU4PySCF at CPU import)
- Model modules define `class Model(torch.nn.Module)` exposing `cube_type`,
  `cube_size`, `input_level`

## PyTorch Rules
- All physics tensors: explicit `dtype=torch.float64`. Never float32 for DFT/SCF
  physics — precision loss causes SCF divergence. (`--precision float32` is
  legacy; new differentiable work defaults to and is tested in float64.)
- Always specify `device` explicitly
- Convert arrays with `torch.as_tensor(..., dtype=torch.float64, device=device)`
- Preserve gradient graph: no `.detach()`/`.item()` in differentiable paths
  (`.item()` OK for logging after the loss/backward path is formed)
- No NumPy round-trips in differentiable paths
- Use `torch.func.vmap` / `torch.func.jacrev` for batched ops and Jacobians
- SCF: use `torch.linalg.eigh` (not `eig`) after symmetric transformation
- Call `torch.cuda.empty_cache()` between SCF iterations if OOM risk is high
- Guard distributed calls with initialization checks outside known DDP paths

## PySCF / Chemistry Rules
- Always set coordinate units in `pyscf.M(...)` (`unit="B"` or `"Angstrom"`)
- Spin branching: RKS when `mol.spin == 0`, UKS when `mol.spin != 0`
- Preserve convergence diagnostics; don't silence failures unless an explicit
  `--check_convergence 0`-style option is set
- Keep CPU and GPU PySCF paths separate; GPU helpers stay optional
- Units: energies in Hartree internally; `AU2KCALMOL` / `AU2DEBYE` live in
  `cc2cc.utils.mol`; grid coords/weights stay in atomic units
- Don't rename dataset molecules, basis aliases, or JSON splits without
  checking every script that references them

## Key Env Vars
Codex commonly sets these; full list in [docs/ARCHITECTURE.md].
- `DFT2CC_MAIN_PATH`: repository root override
- `DFT2CC_DATA_DIR` / `DFT2CC_DATA_TEST_DIR`: data/test subdirs under `data/`
- `CUDA_VISIBLE_DEVICES` / `NUMBER_OF_GPU`: GPU selection and DDP sizing

## Definition of Done
- [ ] Changed files pass a syntax check
- [ ] `import cc2cc.utils` still works without CuPy/GPU4PySCF
- [ ] No cluster-specific absolute paths added to importable code
- [ ] Any validation you couldn't run is recorded with the reason
