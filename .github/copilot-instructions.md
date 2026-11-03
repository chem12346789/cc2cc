# cc2cc — Copilot Instructions

Density functional theory research codebase. PyTorch + PySCF. Surgical changes only.

## Glossary
PUB  = public/exported API
PRIV = private/internal
UT   = unit test

## Tests
- Default OFF. No test file/fixture/mock/dep unless message says "write tests".
- Bugfix != add tests. Ship fix; leave tests to me.
- When asked: <=1 file, <=5 cases. Cover PUB + critical branches only.
- Never: PRIV, getters, trivial wrappers, config, constants, per-exception
  cases, enumeration sweeps, "for completeness" cases.
- Reuse existing runner/naming/assertions. No new patterns.
- Never refactor prod code just to make it testable.

## DO NOT
- No mass-format, recursive rewrite, unrelated cleanup
- No whole-file rewrite for small change — targeted patches only
- No edits to vendor/, third_party/, *_pb2.py
- if file > 50 lines: ask before delete
- No dependency add/remove/upgrade without asking
- No data generation/training/Slurm/long jobs unless explicitly requested
- Never claim a check passed unless it was actually run
- No raw train*.log — filter: rg -iv 'warning|key|Loading|Adjusted' <f> | sed '/^$/d'
- No raw tool output in final response

## DO
- Inspect enough relevant context to make a correct change; choose efficient
  search and read methods without an arbitrary line limit.
- Preserve unrelated work and review the final diff for scope.
- Run focused validation appropriate to the change when practical; avoid
  unrelated or disproportionately expensive checks.
- For Python validation, prefer the interpreter selected in VS Code when available.
- Report validation performed and any remaining limitations.

## Output
- Code only, no explanation, unless asked
- For responses involving mathematical formulas, deliver the response as a PDF
  with rendered formulas and link to it.
- Bullets over paragraphs. No preamble, no recap
- Describe changes in prose + file:line — no diff hunks
- Rationale <= 3 short sentences
- No filler ("let me know if...")

## Done
- [ ] Only task-relevant files changed
- [ ] Unrelated working-tree changes preserved
- [ ] Modified .py passes syntax/import check
- [ ] Physics tensors: explicit float64 + explicit device
- [ ] Differentiable paths: no graph-breaking conversions
- [ ] No cluster-specific paths in importable code
- [ ] No prohibited long-running job launched
- [ ] No unsolicited test files added
- [ ] Response <= 15 lines
