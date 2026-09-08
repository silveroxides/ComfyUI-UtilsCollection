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
- Use the installation's Python environment. Before running tests, identify the
  changed behavior and the uncertainty execution would resolve; choose the
  smallest useful verification. These rules govern routine test selection even
  when a skill suggests broader automatic gates.
- Comments, prose, labels, and tooltip wording normally need diff inspection
  only. For simple constants, defaults, forwarding, or conditions, trace the
  caller and consumer; use a syntax check only when useful. Static inspection
  is sufficient when it resolves the relevant question.
- For a behavioral bug, prefer an existing exact regression:
  `--test tests/test_example.py::test_behavior`. Repeat `--test` for several
  relevant cases. A Python file or `.mjs` file can also be selected explicitly.
- For shared algorithms/interfaces, inspect affected callers and select tests
  for those consumers. Use an explicit `--group` only when narrower selections
  cannot reasonably cover the changed behavior; use `--final` deliberately for
  broad coverage, never as a routine completion or commit requirement.
- `--changed` and `--changed --dry-run` provide optional coverage suggestions.
  They are not mandatory gates. Multiple matching groups do not require running
  them all; editing a test file does not require running that entire file.
  README prose does not require registration, module-structure, or SAM3 tests.
- Do not add tests merely because a function changed. Avoid assertions that
  mirror constants, arithmetic, or obvious return structures without protecting
  a meaningful external contract. Reuse existing coverage; prefer one regression
  for the actual failure/interaction over tests for each helper return.
- A test must address an identifiable failure mode or uncertainty. Passing
  mocked tests establishes only their assertions, not model-generation behavior.
- Reuse completed results. Rerun only when relevant behavior, dependencies, or
  assertions changed, or a diagnosed failure requires another run. Unrelated
  edits in the same source file do not invalidate all prior results. A failing
  targeted test does not automatically justify a broader suite. Do not repeat
  successful selections merely to produce a final report; unresolved failures
  remain failures.
- Verification cost includes agent tokens, user money, and runtime. Prefer
  narrower evidence when testing would greatly exceed the edit's cost, unless
  a concrete risk requires broader coverage. Exercise judgment without routine
  permission questions.
- Keep `tests/test_groups.toml` as an advisory coverage catalog. Missing mappings
  are not a demand to create tests or run `--final`. Existing untracked test
  files may be selected explicitly without staging unrelated artifacts.

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
