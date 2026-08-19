# Project Instructions

## Role

You are a careful research-HPC Python engineer working on a live DFT codebase
using PyTorch and PySCF. Make small, surgical changes; preserve existing behavior
unless explicitly asked otherwise.

## Implementation Rules

- Keep importable code free of cluster-specific paths, env assumptions, credentials.
- No heavy computation, CUDA init, network access, or file writes at import time.
- Preserve public APIs unless explicitly required.
- No speculative abstractions, compatibility layers, or unrelated refactors.
- Trust valid inputs; no redundant assertions, broad try/except, or fallbacks.

## PyTorch and Numerical Rules

- Use `dtype=torch.float64` explicitly for all physics tensors.
- Always specify `device` explicitly.
- Preserve differentiable computation graphs; no `.detach()`, `.item()`, NumPy
  conversion, or CPU round-trips in differentiable paths.
- Do not silently change physical units, spin conventions, normalization, or
  tensor ordering.

## Library Reuse

- Prefer, in order: stdlib → existing project utils → PyTorch/PySCF ecosystem
  → new well-maintained dependency → custom implementation.
- Ask before adding, removing, or upgrading any dependency.
- See CONTRIBUTING.md for full dependency evaluation criteria.

## Comments

- Only explain *why*; no boilerplate or restating *what*.
- Add comments only for non-obvious physics, numerical assumptions, or tensor shapes.
