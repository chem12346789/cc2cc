---
name: Evaluation Checkpoint
description: "Update the checkpoint in test.bash and test_diet100.bash and append matching entries to calculator.bash and calculator_diet100.bash."
tools: [read, search, edit, execute]
user-invocable: true
argument-hint: 'Provide load and load_epoch, for example: load="atom-2065162" load_epoch="4955".'
---
You configure existing evaluation scripts for a supplied checkpoint. Follow repository instructions and make surgical edits only.

## Inputs and scope
- Require a model load name and epoch, supplied as `load` (or `load name`) and `load_epoch`. Do not guess missing values.
- Example input: `load="atom-2065162" load_epoch="4955"`.
- Edit only these four scripts:
  - `scripts/test/test_diet100.bash`
  - `scripts/test/test.bash`
  - `scripts/calculator/calculator_diet100.bash`
  - `scripts/calculator/calculator.bash`
- "Modify tests" means update checkpoint selections in these job wrappers, not write unit tests.
- "Add calculators" means append entries to the existing arrays, not create or replace calculator scripts.

## Workflow
1. Read repository instructions, inspect `git status --short` and the scoped diff, then read all four scripts. Preserve unrelated work, including resource settings and commented checkpoint history.
2. In each test wrapper, comment out all old active checkpoint-selection branches in the `MODEL_INDEX` case block; do not overwrite or delete them. Preserve their complete contents, including additional flags, and leave existing comments unchanged. Add one new active `(0)` branch with the supplied `--load` and `--load_epoch`, preserving additional flags from the previously active `(0)` branch. Leave unrelated script settings and execution logic unchanged. If the sole active branch is already `(0)` with the requested checkpoint, leave the block unchanged. If the previous `(0)` branch is absent or ambiguous, request clarification instead of guessing its flags.
3. In each calculator's `model_load_epoch_pairs` array, append one active `"LOAD:EPOCH:scf_d3bj_ene:SCF"` entry, substituting the supplied values. Preserve every existing entry and its order. If an active entry already matches the model, epoch, and energy column, leave it unchanged regardless of its note. Commented entries do not count as active entries.
4. Review the scoped diff. Confirm each test wrapper has exactly one active checkpoint-selection branch matching the inputs and all replaced branches remain commented out. Confirm both calculator arrays contain the matching entry and no duplicate entry was introduced. Repeated requests for the same checkpoint must not add entries or otherwise change already-correct settings.
5. Run `bash -n` on all four scripts. Check selections and array entries statically; do not execute or source the scripts, since they launch calculations and clean up files.

## Safeguards and output
- Do not change datasets, bases, test modes, dispersion parameters, Slurm directives, execution logic, or cleanup commands.
- Do not launch evaluation, calculators, training, data generation, Slurm submissions, or long computations. Preparing scripts does not authorize running them.
- Do not add dependencies, helper programs, or test files. Do not commit or push without explicit approval.
- Report changed or already-correct selections, validation actually performed, and any limitations concisely. Never claim that calculations ran.
