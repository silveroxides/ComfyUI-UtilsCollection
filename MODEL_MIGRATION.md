# Model migration routine

Authority: `models/migrations.json`. Its structure is defined by
`models/migrations.schema.json`; conversion and runtime loading use the same entries.

## Add a model

1. Identify the specific wrapper, model artifact, input normalization/layout,
   output format, defaults, and real batch constraints. Inspect those boundaries,
   not the entire upstream repository.
2. Add one manifest entry: source repo/file/format, eager module/class, tensor-map
   mode, installed filename, HF destination, and input/output contract. Keep weights
   outside the custom-node repository; use the registered `folder_paths` category.
3. Put architecture code under `models/`. Extend the existing domain node/helper
   modules for orchestration, preprocessing and postprocessing. Use ComfyUI model
   management and UEL; do not carry TorchScript/ONNX frameworks into runtime merely
   because they were the source format.
4. Reproduce exported operations, including operand order, activations, residual
   paths, learned normalization scales, padding, ROI alignment, and NMS grouping.
   Preserve checkpoint constants/buffers; do not guess from matching layer names.
5. Supply conversion-only root/key maps where necessary. Convert the trusted source:

   ```text
   <ComfyUI Python> scripts/convert_pose_models_to_safetensors.py <manifest-key> <source> <destination.safetensors>
   ```

   The converter checks all source/eager keys, shapes and dtypes, verifies every
   written tensor against its source, records the source SHA-256, and refuses to
   overwrite an existing output.

## Validation gates

| Gate | Required evidence |
| --- | --- |
| Artifact integrity | Exact tensor mapping and tensor-value verification; runtime UEL load succeeds with ComfyUI Dynamic VRAM lazy initialization enabled and disabled |
| Forward math | Regression tests for exported arithmetic/control-flow boundaries, not just key/shape checks |
| Numerical parity | With explicit inference authorization, compare source and eager outputs on identical inputs; record source hash, code revision, dtype/device, shapes and numerical errors |
| Batch semantics | Single/multiple/tail batches; preserve frame/subject order; no emitted padding subjects |
| Output semantics | Empty detections, coordinate mapping, confidence filtering, rendering and optional temporal filtering |
| UI/pack integration | `UC_` node ID, appropriate domain module/category, meaningful controls exposed with source-compatible defaults, registry entry and focused test selection |

Key/shape parity does **not** establish correct inference. In particular, check
GAU activation/squaring and division order, residual connections, and whether NMS
is per level/class/image. If inference is not authorized or available, report
numerical parity and prediction quality as **unverified**; do not replace that gate
with a passing mocked test.

Dynamic VRAM is the default runtime: an uninitialized ComfyUI Linear can omit
weight/bias from `state_dict()`. Validate its declared dimensions and load the
checkpoint parameters without allocating duplicate full weight buffers. A
CPU-only loading check does not cover this path; verify loaded lazy tensors
against the checkpoint as well.

## Publish and continue

- Upload tensor-verified safetensors only with authorization, using the manifest's
  model-type HF subdirectory. Keep original executable archives out of uploads.
- Verify the remote file paths and exercise the local-file reuse/download route.
- Document node controls, dependencies, installation paths and validation limits.
- Keep artifact integrity, model-output correctness and measured performance as
  separate reported results. Do not claim a speedup from batch-call counts alone.
- Commit coherent model/integration increments; leave unfinished ports unstaged.
- For the next model, repeat the same gates instead of copying an unverified port.
