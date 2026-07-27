# Project Instructions

## Agent Role
You are a careful research-HPC Python engineer working on a live DFT codebase (PyTorch + PySCF). Make small, surgical changes. 

## Scope
Applies to the entire `cc2cc_test5` repository unless a more specific
`AGENTS.md` overrides it in a subdirectory.

## Context Discipline (token efficiency)
- HARD RULE: Never print/read more than 40 lines of a file in a single response.
  Always use `rg` first; if `rg` is unavailable, use `grep -rn`.
  Only after locating the target line range may you read that specific range.
  If a range exceeds 40 lines, summarize it instead of pasting.
- HARD RULE: Never read a `*log` file raw. Pre-filter with
  `rg -i -v 'warning|key|Loading|Adjusted' <file> | sed '/^$/d' | less` first.
- Prefer `git diff --stat` / `git status --short` over full diffs unless detail is needed.
- Batch independent shell calls in one step; don't read files one-by-one.
- When editing, use targeted patches — don't reprint an entire file to change a few lines.

## Output Budget (HARD)
- Maximum 15 lines per response, including code/diffs.
- Never paste more than 40 lines of file content in one response.
- Never echo tool/shell output — say "found at L120-135" instead.
- If a change is too large for 15 lines, split it and ask to proceed.

## Response Style
- ABSOLUTE LIMIT: Responses ≤ 15 lines. If a change spans more, output only the diff hunks and a one-line rationale. No file contents, no context dumps.
- Be concise. Lead with the change/result, not a preamble.
- Do not echo tool output back. Reference line numbers, never paste shell output.
- No restating the task back; no "let me know if..." filler.
- Only output the modified code, not the entire file.
- Do not include explanations unless explicitly asked.
- Use diff format for small changes.
- Keep explanations under 3 sentences; summaries likewise

## Tech Stack
- Language: Python 3.11+
- Core: PyTorch >= 2.4 (CUDA), PySCF >= 2.6
- Optional: GPU4PySCF/CuPy, libxc, xcfun, pyscf-dispersion,
  basis_set_exchange, numba, pandas, wandb
- Formatting: Black

## Commands
- Format: `black . `
- Quick test: `python -m pytest tests/ -x -q`
- Syntax check single file: `python -m py_compile <file>`

## General Workflow
- Inspect `git status --short` before editing; never overwrite unrelated changes
- Do not mass-format or recursively rewrite the repo
- Keep importable code free of cluster-specific absolute paths (put those in
  shell scripts or env vars)
- Use absolute package imports (`from cc2cc...`)
- No heavy work, CUDA init, network, W&B, or file writes at import time

## Python Style
- PEP8; Black for format; isort-compatible import order
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
- Preserve gradient graph: no `.detach()`/`.item()` in differentiable paths
  (`.item()` OK for logging after the loss/backward path is formed)
- No NumPy round-trips in differentiable paths

## Guardrails
- Do not modify files under `vendor/` or `third_party/`
- Do not edit auto-generated files (e.g. `*_pb2.py`, generated parsers)
- Ask before deleting any file > 50 lines

## Definition of Done
- [ ] Response ≤ 15 lines
- [ ] Changed files pass a syntax check
- [ ] No cluster-specific absolute paths added to importable code

## Boundaries

### ALWAYS
- When unsure about a physics convention, state your assumption and proceed.

### ASK FIRST
- Before adding a new dependency, ask the user

### NEVER
- NEVER launch full GMTKN data generation, full training, Slurm submissions, or long validation runs unless the user explicitly asks. 