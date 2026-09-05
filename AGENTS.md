# Agent Instructions

Read `AGENTS-LOCAL.md` when present. It contains machine-specific operating
rules and must remain local only; do not stage or commit it.

## Environment and verification

- Use the Python environment belonging to the ComfyUI installation under test.
  Keep machine paths and personal tool/skill configuration in local instructions.
- Keep changes within the requested scope and preserve unrelated worktree edits.
- Base behavioral claims on the relevant implementation and inputs. Report what
  was actually tested and distinguish inference from verified behavior.

## Python module structure

- Keep `*_nodes.py` modules limited to node classes, schemas, registration
  collections, constants, and thin execution orchestration.
- Put algorithms, tensor operations, parsing, geometry, model handling, and
  reusable execution logic in focused domain `*_helpers.py` modules.
- Extend matching helper domains; do not create catch-all helpers or extract
  trivial forwarding functions without reuse or an interface requirement.
- Keep the legacy allowlist in `tests/test_node_module_structure.py` frozen.
  Refactor affected standalone node-module helpers rather than adding to it.

## Test selection

- Use `tests/run_tests.py` as the repository-owned selector.
- During iteration, select an exact supported test or one explicit `--group`.
- Before handoff, use `tests/run_tests.py --changed --dry-run` to resolve coverage.
  If all selected tests already ran against unchanged production and test inputs,
  report those results without rerunning them. Otherwise execute the uncovered
  groups/tests through the selector. A previous failure remains a failure.
- Use `--final` only as a deliberate broader gate; do not repeat a successful gate
  without relevant changes.
- Map new production files in `tests/test_groups.toml`. Untracked files are excluded:
  select their intended group explicitly until they are tracked.

## Managed VLM presets

- Treat serialized runtime presets as generated data. Edit the matching readable
  `*_vars.py` authority identified by `CONFIG` in
  `scripts/manage_vlm_preset_authorities.py`; never hand-edit runtime literals.
- Whitespace inside preset strings is content. Do not trim or normalize it without
  explicit authorization.
- Run the tracked manager without `--apply`. After validation succeeds, synchronize only with the tracked
  manager's `--apply`. Rerun validation until synchronized before test changes
  or execution.
- If the manager does not cover a requested pair, extend deterministic tooling
  first rather than patching serialized dictionary boundaries.
- Review the readable-authority diff and generated-key/change metadata. Do not
  print or inspect raw serialized runtime literals, including in diff output.
  Verify readable/runtime synchronization and exposed preset options.
- Audit relevant assembled prompt context when changing prompt contracts.
  Text assertions establish structural contracts, not model-generation behavior.
- After synchronization, syntax, or escaping failure, inspect the scoped state
  and utility contract, then validate a deterministic correction before mutation.
  Do not restore dirty runtime data from HEAD or discard uncertain changes.
