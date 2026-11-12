# Agent Instructions

Read `AGENTS-LOCAL.md` when present. It contains machine-specific operating
rules and must remain local only; do not stage or commit it.

## Environment and verification

- Use the Python environment belonging to the ComfyUI installation under test.
  Keep machine paths and personal tool/skill configuration in local instructions.
- Keep changes within the requested scope and preserve unrelated worktree edits.
- Base behavioral claims on the relevant implementation and inputs. Report what
  was actually tested and distinguish inference from verified behavior.
- Avoid using the regular `grep` tool for open-ended searches, as it triggers broad
  filesystem-wide searches across millions of files. Always scope searches using
  `rg` (ripgrep) directly with explicit target directories, file globs, and bounded patterns.

## Python module structure

- Keep `*_nodes.py` modules limited to node classes, schemas, registration
  collections, constants, and thin execution orchestration.
- Put algorithms, tensor operations, parsing, geometry, model handling, and
  reusable execution logic in focused domain `*_helpers.py` modules.
- Extend matching helper domains; do not create catch-all helpers or extract
  trivial forwarding functions without reuse or an interface requirement.
- Keep the legacy allowlist in `tests/test_node_module_structure.py` frozen.
  Refactor affected standalone node-module helpers rather than adding to it.

## Python formatting

- Treat `pyproject.toml` (`[tool.isort]`, `[tool.yapf]`, `[tool.ruff]`) as the
  formatting authority (`line-length = 320`, `skip-magic-trailing-comma = true`).
- Write compact single-line schema inputs, outputs, and calls within the
  320-character limit; avoid unnecessary multi-line argument expansion or
  trailing commas that force vertical wrapping.
- Before committing touched Python files, run `isort --settings-path pyproject.toml`,
  `yapf -i --style pyproject.toml`, `ruff check --fix --config pyproject.toml`
  (do not abort on pre-existing `E501` long prose strings), and
  `ruff format --config pyproject.toml`.
- Formatter CLIs may live in the system Python environment outside the ComfyUI
  virtual environment; formatting touched Python modules in full is expected and
  not scope drift.

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

## User-facing copy and node documentation

- Match the repository's established node-family, display-name, and parameter
  naming conventions (e.g., `UC_MiniMaxH3ClipContinuation*`, `tail_frames`)
  when porting or adapting external reference implementations; never copy
  external node or widget names verbatim if they clash with existing workflow
  naming.
- Write node `description`s, input/output `tooltip`s, `node_metadata.py`
  (`DESCRIPTIONS` and `EXTRA_ALIASES`), `README.md`, and guide-node prose in
  plain, practical, jargon-free language for workflow users.
- Explain what a node or control does to the generation or workflow, when to
  connect optional inputs, what the default value provides, and the practical
  effect of raising, lowering, or disabling a setting.
- Avoid internal implementation terms ("cloned model", hook/wrapper names),
  tensor/shape notation (`[B, C, N]`, "variable-length"), and raw paper or math
  jargon (`1 + phi`, `tau`, `alpha`, `theta`, "L1-norm ratio clipping threshold")
  in user-facing text.
- Provide explicit `tooltip`s on every user-facing input and output—including
  inputs or outputs inherited from a parent node class when subclassing.
- When adding, renaming, or updating a node, always add or update both
  `DESCRIPTIONS` and `EXTRA_ALIASES` in `node_metadata.py` (do not rely on
  inline `schema.description` fallback alone), `README.md`
  (`## Available nodes`), and any relevant guide node (`UC_*Guide`) in the same
  change before committing so separate documentation-audit passes are
  unnecessary.

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

## UnifiedEfficientLoader safetensors I/O and registry scanner rules

- All `.safetensors` tensor reading and writing across `nodes/`, `helpers/`,
  `models/`, `scripts/`, and `tests/` must use `unifiedefficientloader`
  (`MemoryEfficientSafeOpen` / `UnifiedSafetensorsLoader` with `low_memory=True`
  and `IncrementalSafetensorsWriter(..., max_workers=1)`).
- Direct `safetensors` tensor I/O (`from safetensors.torch import save_file`,
  `load_file`, `from safetensors import safe_open`) and synchronous
  `.get_tensor(` calls are forbidden.
- Reading tensors must use `async_stream(..., batch_size=1, prefetch_batches=1, pin_memory=...)`,
  call `mark_processed(key)` in `finally` for every yielded tensor, and close
  both the stream and handler in `finally` (or use `read_uel_safetensors_dict`
  from `helpers/helper_functions.py`).
- Writing `.safetensors` files must write to a sibling `.{basename}.{uuid}.tmp`
  file via `atomic_uel_writer` (from `helpers/helper_functions.py`) or
  `IncrementalSafetensorsWriter(..., max_workers=1)`, submit tensors via
  `writer.write_batch(...)` (do not call `writer.write(`), atomically replace
  the destination only after the writer context exits cleanly, and unlink any
  leftover `.tmp` file in `finally`.
- Never use `.eval(` anywhere in repository code or tests; the Comfy Registry
  static scanner flags `.eval(` substring matches. Use `.train(False)` on
  `torch.nn.Module` instances when disabling training mode.

## Obligatory communication style (caveman mode)

- Obligatory repeated activation of `caveman` skill on every response.
- Keep responses terse, smart, and technical. Drop filler, pleasantries, and unnecessary words.
- Bound lines to under 40 words per line to prevent narrow-column UI breakage.
- Never output walls of nested lists or decorative text.


