# cc2cc — Copilot Instructions

Density functional theory research codebase. PyTorch + PySCF. Surgical changes only.

## DO NOT
- No mass-format, recursive rewrite, unrelated cleanup
- No whole-file rewrite for small change — targeted patches only
- No edits to vendor/, third_party/, *_pb2.py
- if file > 50 lines: ask before delete
- No dependency add/remove/upgrade without asking
- No data generation/training/Slurm/long jobs unless explicitly requested
- No claim check passed unless actually run
- No read > 40 lines per tool call
- No raw train*.log — filter: rg -iv 'warning|key|Loading|Adjusted' <f> | sed '/^$/d'
- No raw tool output in final response

## DO
- Before edit: git status --short
- After edit: git diff --stat  # scope check
- Locate: rg first; read smallest relevant range only
- Batch independent shell commands in one call
- Changed .py: python -c "import <module>"  # syntax check
- Modified behavior: run targeted tests if available
- Report: validation performed + remaining limitation

## Output
- Code only, no explanation, unless asked
- Bullets over paragraphs. No preamble, no recap
- Describe changes in prose + file:line — no diff hunks
- Rationale ≤ 3 short sentences
- No filler ("let me know if...")

## Safety Net
- Before making changes, run: git stash create  # snapshot, no side effects
- Record the stash ref in your response as "rollback: <ref>"
- To undo: git stash apply <ref>

## Done
- [ ] Only task-relevant files changed
- [ ] Unrelated working-tree changes preserved
- [ ] Modified .py passes syntax/import check
- [ ] Physics tensors: explicit float64 + explicit device
- [ ] Differentiable paths: no graph-breaking conversions
- [ ] No cluster-specific paths in importable code
- [ ] No prohibited long-running job launched
- [ ] Response ≤ 15 lines
