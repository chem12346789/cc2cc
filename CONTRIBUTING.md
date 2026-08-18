# Contributing to cc2cc

This document covers code style, compaction rules, dependency evaluation criteria,
and commenting conventions. For behavioral constraints applied to every request,
see `.github/copilot-instructions.md`. For core engineering rules, see `AGENTS.md`.

## Python Style

- PEP 8, formatted with `black` and sorted with `isort` (profile `black`).
- Type annotations: essential for public APIs and complex functions.
  Internal logic in a clearly-typed function may omit redundant annotations.
- Prefer pure functions for new code. Class-based logic is acceptable when it
  models stateful objects (e.g. model definitions, training state).
- Keep individual files under 300 lines unless holding model classes or
  configuration schemas. Split large files by responsibility.
- `utils/__init__.py` must remain lightweight — avoid heavy imports or
  side effects. Import from submodules directly.
- Model classes: define `forward()` and helper methods only. Keep
  initialization logic minimal and explicit. Avoid dynamic attribute creation.
- Prefer composition over inheritance. Avoid deep hierarchies.
- Use `pathlib.Path` for filesystem paths. No string concatenation for paths.
- Raise specific exceptions; avoid bare `except:` or catching `Exception`.

## Code Compaction Rules

### Redundancy Elimination
- Remove duplicate logic. If two functions differ only by a constant,
  unify with a parameter.
- Eliminate dead branches and unreachable code.
- Inline single-use variables when the name adds no clarity.

### Prefer Standard Library and Existing Utilities
- Use `itertools`, `collections`, `functools`, `pathlib`, `dataclasses`
  before reimplementing equivalent logic.
- Check existing project utilities (`cc2cc.utils.*`) before writing new helpers.

### Merge and Simplify
- Combine adjacent blocks that operate on the same data into a single pass.
- Replace verbose patterns (manual loops where a comprehension or `map`
  suffices) with idiomatic equivalents.
- Reduce nesting: flatten via early returns, guard clauses, or extraction.

### Naming and Structure
- Names should describe intent, not implementation (`compute_overlap`
  not `loop_func`).
- Consistent naming within a module; avoid abbreviations that are not
  domain-standard.
- One responsibility per function. If a function does two distinct things,
  split it.

### Output Requirements
- After compaction, verify the code still passes the same targeted tests.
- Do not change public behavior during compaction — only structure.
- Report what was merged, removed, or inlined.

## Library Reuse — Full Evaluation Criteria

Before adding any new dependency, evaluate along these dimensions:

1. **Maintenance status**: Active development within the last 12 months,
   responsive issue tracker, healthy release cadence.
2. **License**: Compatible with the project license (check for copyleft
   restrictions, academic-only clauses, or attribution requirements).
3. **Numerical precision**: Supports `float64` (or arbitrary precision).
   Libraries that force `float32` or low-precision internals are unacceptable
   for physics computations.
4. **Autograd compatibility**: Operations must integrate with PyTorch's
   autograd or expose differentiable interfaces. Avoid libraries that break
   the computation graph (e.g. NumPy-only internals without custom wrappers).
5. **GPU behavior**: Explicit device support. No implicit CPU transfers.
   Verify `cuda` and multi-GPU scenarios if relevant.
6. **HPC installation feasibility**: Must be installable via `pip` or `conda`
   on the cluster environment without root access. Avoid packages requiring
   system-level compilers not available on the HPC nodes, or packages with
   heavy native build chains unless clearly justified.
7. **Scope**: Prefer focused libraries over monolithic frameworks. A 200-line
   utility should not pull in a 500 MB dependency tree.

If all criteria are met, ask the maintainer before adding the dependency.
Record the justification in the PR description.

## Comments

### What deserves a comment
- Non-obvious physics: conventions that differ from the standard literature,
  sign choices, unit systems, or approximations applied.
- Numerical assumptions: expected magnitude ranges, conditioning notes,
  regularization thresholds and why that value.
- Tensor shapes: when a tensor's shape is not obvious from the variable name
  and surrounding code (e.g. batched vs. flattened orbital indices).

### What does NOT deserve a comment
- Restating what the code does in prose.
- Boilerplate (`# initialize counter`, `# return result`).
- Section dividers in short files (`# --- Helper Functions ---`).
- TODOs without context or owner. Use the issue tracker instead.

### Style
- Comments explain *why*, not *what*.
- Keep comments to one or two lines. For longer explanations, reference a
  docstring, paper, or design document.
- Place the comment on its own line above the code, not trailing.
- Update or remove comments when the associated code changes.
