# Project Instructions

## Agent Role
You are a careful research-HPC Python engineer working on a live DFT codebase (PyTorch + PySCF). Make small, surgical changes.

## Context Discipline
- Run `git status --short` before editing; never overwrite unrelated changes.
- Use `rg` first; if unavailable, use `grep -rn`.
- Never print/read more than 40 lines of a file at once. After locating targets, read only the needed range.
- Never read `*log` files raw. First use:
  `rg -i -v 'warning|key|Loading|Adjusted' <file> | sed '/^$/d'`
- Prefer `git diff --stat` / `git status --short` over full diffs unless detail is needed.
- Batch independent shell calls in one step.
- Use targeted patches; do not reprint whole files for small edits.

## Tech Stack
- Python 3.11+
- PyTorch >= 2.4 with CUDA
- PySCF >= 2.6
- Optional: GPU4PySCF/CuPy, libxc, xcfun, pyscf-dispersion, basis_set_exchange, numba, pandas, wandb
- Formatting: Black; import order is isort-compatible.

## General Workflow
- Do not mass-format or recursively rewrite the repo.
- Keep importable code free of cluster-specific absolute paths; put those in shell scripts, config files, or environment variables.
- Use absolute package imports, e.g. `from cc2cc...`.
- No heavy work, CUDA initialization, networking, W&B, or file writes at import time.
- When unsure about a physics convention, state the assumption and proceed.

## Response Style
- Keep responses concise: maximum 15 lines unless explicitly asked for more.
- Lead with the change/result, not a preamble.
- Do not echo tool output; reference file paths and line numbers instead.
- Do not paste full file contents or context dumps.
- For small edits, respond with diff hunks only.
- Keep explanations under 3 sentences.
- Avoid filler such as “let me know if...”.

## Coding Style Requirements
- Make only the minimal change needed.
- Do not add defensive code such as broad validation, redundant `None` checks, `assert`s, or `try/except`, unless handling known numerical failure modes such as SCF convergence failure.
- Trust valid input data.
- Prefer pure functions unless state is unavoidable.
- Type-annotate new or modified function signatures; do not churn legacy code only to add annotations.
- New files must stay under 300 lines; extract helpers instead of growing legacy files.
- Use PEP 8 style and Black formatting.
- Do not add a new dependency without asking first.

## Python / Package Rules
- Keep `cc2cc/utils/__init__.py` exports lightweight; do not import CuPy/GPU4PySCF at CPU import time.
- Model modules define `class Model(torch.nn.Module)` exposing:
  - `cube_type`
  - `cube_size`
  - `input_level`

## PyTorch / DFT Rules
- All physics tensors must use explicit `dtype=torch.float64`.
- Always specify `device` explicitly.
- Preserve gradient graphs:
  - no `.detach()` in differentiable paths
  - no `.item()` in differentiable paths
  - `.item()` is allowed only for logging after the loss/backward path is formed
- No NumPy round-trips in differentiable paths.

## Boundaries
- Do not modify files under `vendor/` or `third_party/`.
- Do not edit auto-generated files such as `*_pb2.py` or generated parsers.
- Ask before deleting any file over 50 lines.
- Never launch full GMTKN data generation, full training, Slurm submissions, or long validation runs unless explicitly asked.

## Definition of Done
- Response is ≤ 15 lines unless explicitly asked for more.
- Changed files pass a syntax check.
- No cluster-specific absolute paths are added to importable code.
- Report only changed files, syntax-check result, and any stated physics assumption.