# ComfyUI-UtilsCollection

A collection of ComfyUI nodes for modern text and multimodal conditioning, image and mask processing, prompt presets, workflow parameters, loading, and general utilities. The encoder nodes track current ComfyUI Core behavior while retaining compatible legacy node IDs where practical.

## Available nodes

The list below uses the canonical node IDs. Deprecated compatibility aliases remain registered for existing workflows but are not duplicated here.

### Text encoding and conditioning

- `UC_TextEncodeSystemPrompt`
- `UC_TextEncodeLtxv2SystemPrompt`
- `UC_WeightedTextEncodeSystemPrompt`
- `UC_TextEncodeSystemEditAdvanced`
- `UC_TextEncodeGemmaSystemEditAdvanced`
- `UC_AdvancedVisualConditioningEncode`
- `UC_AdvancedVisualConditioningEncodeTokenFusion`
- `UC_AdvancedMiniMaxH3ImageToVideo`
- `UC_AdvMiniMaxH3ImageToVideoTokenFusion`
- `UC_AdvMiniMaxH3ImageToVideoTemporalFusion`
- `UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion`
- `UC_MiniMaxH3VLMGuide`
- `UC_MiniMaxH3MediaConfig`
- `UC_MiniMaxH3RefExtract`
- `UC_MiniMaxH3AudioRefExtract`
- `UC_MiniMaxH3RefLoad`
- `UC_MiniMaxH3RefSave`
- `UC_MiniMaxH3RefApply`
- `UC_AdvancedVisConEncoder`
- `UC_AdvancedVisConEncoderTokenFusion`
- `UC_VisualConsensusConfiguration`
- `UC_AdvancedConsensusConfiguration`
- `UC_Krea2TokenAttentionWeight`
- `UC_Krea2TokenAttentionWeightTokenFusion`
- `UC_AttentionBiasTextEncode`
- `UC_TextConsensusBlendConfig`
- `UC_VisualFusionConfig`
- `UC_ConditioningConsensusBlend`
- `UC_VLMInputEmbeds`
- `UC_Krea2LayerProbe`
- `UC_Krea2LayerAblator`
- `UC_MiniMaxH3ClipProjectionPatcher`
- `UC_EncoderNodesGuide`

#### MiniMax H3 experiments

All four H3 image-to-video encoders preserve joint Qwen encoding regardless of enable_caching. Enabled modes cache complete joint encoded results; changing their prompt or media invalidates those results. Image/video modes select matching VAE outputs; all also includes audio VAE outputs. Raw vision, DeepStack, and pre-Qwen tokens are never saved. Guide retains its established independent encoding and caches in images_only and all. Entries use UnifiedEfficientLoader under ComfyUI temporary storage.

The temporal encoders fuse offset video samples into the ordinary video token budget. Set temporal density and consensus/spatial method on `UC_MiniMaxH3MediaConfig`; density 1 preserves ordinary sampling. Consensus uses `UC_TextConsensusBlendConfig`, spatial fusion uses `UC_VisualFusionConfig`.

`UC_MiniMaxH3VLMGuide` inserts an independently encoded timestamp/image block before the prompt in compatible H3 conditioning. Chained guides retain insertion order. This experiment does not re-encode the original prompt jointly with the guide.

`UC_MiniMaxH3FirstFrameReferences`, `UC_AdvancedMiniMaxH3ImageToVideoCombined`, and `UC_AdvMiniMaxH3ImageToVideoCombinedTokenFusion` were removed. Workflows using these IDs report missing nodes; no aliases or migration are provided.

#### MiniMax H3 Ref

**MiniMax H3 Ref Extract** encodes each image in an `IMAGE` batch as a separate native H3 reference. Select `video` only when the ordered batch is one 24 fps clip; the clip needs at least five frames. **MiniMax H3 Audio Ref Extract** creates an independent audio reference with a matching H3 audio VAE. Compression can reduce token cost but loses detail; refined compression optimizes only the compressed latent and does not train a model.

Use **MiniMax H3 Ref Save** to write individual `.safetensors` artifacts under `ComfyUI/models/minimax_h3_refs`, then select an artifact with **MiniMax H3 Ref Load**. **MiniMax H3 Ref Apply** appends saved or newly extracted refs to existing MiniMax H3 conditioning. It uses ordinary native conditioning only—no model patcher or sampler wrapper—and does not add prompt labels, learned trigger words, or voice-cloning guarantees.

Ref Save is an output node, so extraction can run without a sampler. Save uses the original `refmod_meta` file header. Load accepts original-format references and earlier files saved here with `ref_meta`; the shorter node names do not change file compatibility. Saved descriptions and settings are retained as metadata, not automatically applied as controls.

Refs are applied independently in socket order. `retention` controls detail retained in newly supplied refs; it is not attention strength or a denoising curve. `max_ref_tokens` rejects excess total reference tokens instead of silently resizing or dropping refs.

Custom merging, time-varying reference curves, synchronized audio/video identity binding, and library or preview UI are future work. Use the original reference-node package if you need its package-specific behavior.

#### MiniMax H3 CLIP projection models

`UC_MiniMaxH3ClipProjectionPatcher` projects a Qwen3-VL 4B or 8B text encoder into MiniMax H3's 32B conditioning space. Load the encoder with Core's **Load CLIP** node using type `minimax`, then connect it to the projection patcher.

Download one projection matching the encoder size into `ComfyUI/models/clip_projections/`:

- [Qwen3-VL 4B v3.1](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/blob/main/mmh3-4b-ClipProj-v3.1.safetensors)
- [Qwen3-VL 4B v3.1 with residual MLP](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/blob/main/mmh3-4b-ClipProj-v3.1-mlp.safetensors)
- [Qwen3-VL 8B v3.1](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/blob/main/mmh3-8b-ClipProj-v3.1.safetensors)
- [Qwen3-VL 8B v3.1 with residual MLP](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3/blob/main/mmh3-8b-ClipProj-v3.1-mlp.safetensors)

Only one projection is used at a time. The control and `obsolete/` files in the model repository are not normal generation models.

#### UC_AdvancedMiniMaxH3ImageToVideo: Qwen-only 1024 VLM example

The optional MiniMax H3 Media Configurator controls separate Picture and Video timestamp syntax. Configured Video timestamps map one-to-one to already-selected images; without them, the Video input is a full 24 fps batch using ComfyUI Core H3 sampling, syntax, and native conditioning. `vlm_resolution` controls Picture token detail while `vlm_video_resolution` independently controls Video token detail. Video blocks remain outside Picture fusion. Audio and its MiniMax H3 audio VAE connect directly to each advanced main node and produce a standalone native audio reference plus Qwen Audio label.

[Workflow JSON](workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_Workflow.json) | [API workflow JSON](workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_API.json) | [Workflow overview](workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_Overview.png) | [Reference images](https://github.com/silveroxides/ComfyUI-UtilsCollection/releases/download/advanced-minimax-h3-qwen-only-assets-v1/UC_AdvancedMiniMaxH3ImageToVideo_QwenOnly_8Image_1024VLM_References.zip) | [Turbo LoRA used](https://huggingface.co/silveroxides/MiniMax-H3_tests/blob/main/minimax_h3_fl2v_lightx2v_v0.1_dareties_v4_step600_comfy_fro.safetensors)

This example uses eight chronological storyboard frames as 1024-resolution Qwen3-VL/DeepStack references. The prompt associates each ordered `<Picture N>` entry with a target timestamp. With `ref_image_size` set to `none`, the images provide visual-token conditioning without VAE reference encoding.

The workflow demonstrates strong subject, composition, and approximate timeline control without a native reference video. Its eight images reproduced the main framing and progression of a 12.25-second source sequence in seven sampling steps on a 16 GB GPU. Picture timestamps are prompt instructions, not fixed frame anchors, so results remain stochastic.

Extract the separately hosted reference-image ZIP into `ComfyUI/input` before loading either workflow.

The workflow uses Core's **Create Video** and **Save Video** nodes and requires no other custom-node collection.

For headless use, start ComfyUI with its API reachable, extract the reference ZIP, then run:

```powershell
python workflows/UC_AdvancedMiniMaxH3ImageToVideo/run_api_workflow.py C:\path\to\reference-images
```

The standard-library runner uploads the eight images, substitutes the returned server filenames into the unchanged API workflow, queues it, waits for completion, and prints the saved-output metadata. Use `--server http://host:8188` for another ComfyUI server and `--seed N` to override the workflow seed.

<img src="workflows/UC_AdvancedMiniMaxH3ImageToVideo/QwenOnly_8Image_1024VLM_Overview.png" alt="UC Advanced MiniMax H3 Image to Video Qwen-only eight-image 1024 VLM workflow" width="1200">

#### Advanced visual consensus

`UC_AdvancedVisConEncoder` runs two sequential stages: it first constructs a
complete spatially fused conditioning independently at every selected VLM
resolution, then passes those complete conditionings through the same consensus
mathematics as `UC_ConditioningConsensusBlend`. Spatial fusion and consensus
are not alternatives and are never crossfaded.

`UC_AdvancedVisConEncoderTokenFusion` is the additive token-first alternative.
At each lane and resolution it fuses per-source visual and DeepStack tokens,
runs one conditioning encode, then applies the same complete-conditioning
consensus across resolution samples. The original node remains unchanged.

Use `UC_VisualConsensusConfiguration` to combine one complete
`UC_VisualFusionConfig` with one `UC_AdvancedConsensusConfiguration`. Fusion
method `off` disables the spatial stage; consensus preset `off` disables the
cross-resolution consensus stage. Advanced Consensus Configuration inherits
the complete Text Consensus Blend Configurator contract and adds
`resolution_samples` plus a 32-aligned `sample_offset`.

`block_size` is specific to block-interleave. `dither_ratio` and
`dither_secondary_pattern` are specific to random-dither. Advanced Consensus
Configuration exposes `resolution_samples` and `sample_offset`. Offset defaults
to `32` and supports `32` through `512` in 32-unit steps. The configured sample
count is exact, so `1` remains one
resolution sample regardless of visual-source or batch-lane count. Original VLM
resolution supports one sample but cannot construct adjacent resolution
variants.

A batch in the only connected image socket behaves like its images were
connected as separate visual sources. With multiple connected batched sockets,
equal indices form independent lanes, singleton sockets broadcast, and all
other batch lengths must match. Raw visual export uses the same spatial mask as
the base-resolution conditioning fusion.

### Image, mask, and compositing

- `UC_Image_Color_Noise`
- `UC_ExtractPrevalentColors`
- `UC_ModifyMask`
- `UC_SAM31CheckpointLoader`
- `UC_SAM3Detect`
- `UC_MaskToBoundingBox`
- `UC_ImageBlendByMask`
- `UC_ImagePad`
- `UC_NoHaloLoHaloDownscale`
- `UC_CropByMask`
- `UC_StagedLayerCrops`
- `UC_ImageCropMerge`
- `UC_ExtractMask`
- `UC_ExtractImage`
- `UC_ImageAndMaskResize`
- `UC_ResizeMask`
- `UC_BackgroundRemovalPreserveAlpha`
- `UC_FaceRemovalPreserveAlpha`
- `UC_UnifiedBackgroundReplace`
- `UC_StagedLayeredBackgroundComposite`
- `UC_StagedIndividualComposites`
- `UC_StagedLayeredBackgroundCompositeOptions`
- `UC_StagedMediaPipeFaceBackgroundComposite`
- `UC_StagedMediaPipeFaceOptions`
- `UC_LayeredBackgroundComposite`
- `UC_MediaPipeFaceCompositeOptions`
- `UC_MediaPipeFaceComposite`
- `UC_ListToImageBatch`
- `UC_ImageMatchProperties`
- `UC_OpticalFlowComposite`
- `UC_ImageInwardEdgeFill`
- `UC_ImageIterativeStretchFill`
- `UC_TextOverlayNode`
- `UC_CompositeNodesGuide`
- `UC_LaMaInpaint`
- `UC_BatchedOpenPose`
- `UC_DWPoseEstimator`

`UC_BatchedOpenPose` batches video frames for body inference and person crops for hand/face inference. `UC_DWPoseEstimator` batches YOLOX frames and RTMPose person crops, including partial batches without padding. Both return `IMAGE` and `POSE_KEYPOINT`, use ComfyUI's selected device/model management, and expose a batch size to control VRAM use. These are independent eager implementations loaded through UEL; `comfyui_controlnet_aux`, Ultralytics, ONNX Runtime, MMPose, and TorchScript are not runtime dependencies. Speed and prediction parity need real-model validation.

Checkpoints belong in `ComfyUI/models/controlnet/preprocessors` (also discovered under additional `controlnet` roots registered through `folder_paths`). Executing a pose node downloads missing weights from the pack's HF repository through `huggingface_hub` and reuses existing local files. No model weights are stored in this node repository.

DWPose exposes separate detection and keypoint-confidence thresholds (both default to `0.3`). OpenPose exposes body (`0.1`), hand/face (`0.05`), limb affinity (`0.05`), limb support (`0.8`), minimum connected body parts (`4`), and minimum assembled body score (`0.4`). OpenPose preserves face-landmark slots when points are below threshold so temporal matching cannot shift landmark identities.

Both nodes share optional `temporal_filter`: it treats the input batch as ordered video frames and prunes individual unsupported keypoints before rendering and exporting, without removing person entries or interpolating replacement joints. Defaults inspect two frames on either side, require support from two neighbors where available, allow movement up to `0.1` of the person's box diagonal, and match person boxes at IoU `0.3`. Increase the distance allowance for faster motion; leave filtering disabled for unrelated still images. Matching spans processing-chunk boundaries.

| Node | Checkpoints | Path in `silveroxides/ComfyUI-UtilsCollection-Models` |
| --- | --- | --- |
| OpenPose | `openpose_body.safetensors`; `openpose_hand.safetensors` when hands enabled; `openpose_face.safetensors` when face enabled | `preprocessors/openpose/` |
| DWPose detector | `dwpose_yolox_l.safetensors` | `detectors/` |
| DWPose pose | `dwpose_ucoco_384.safetensors` | `preprocessors/dwpose/` |

`scripts/convert_pose_models_to_safetensors.py` converts the original trusted `.pth`/TorchScript sources, maps eager tensor names, and verifies shapes, dtypes, and tensor values before accepting the output. Runtime loaders accept the converted safetensors, not the original executable archives.

`UC_StagedLayeredBackgroundComposite` builds a scene from a background and ordered foreground sockets. Use `run_staging` to retain cutouts and populate the placement editor. Use `run_staged` to composite retained cutouts without loading models or evaluating foreground branches. Use `full_run` to restage and composite in one queue. `foreground_0` is the backmost layer. Retained cutouts are held in server memory and must be recreated after restarting ComfyUI.

`UC_StagedMediaPipeFaceBackgroundComposite` detects faces in each foreground and adds them as independently placeable layers. The background and face options nodes contain removal, extraction, feathering, and blend settings. `UC_StagedIndividualComposites` provides the same ordinary foreground staging editor but returns one full-background image, placement mask, and box per included foreground without stacking them. `UC_BackgroundRemovalPreserveAlpha` directly returns source-resolution RGBA images and their soft alpha masks; existing RGBA inputs keep their supplied alpha without model execution. `UC_FaceRemovalPreserveAlpha` returns expanded face crops as RGBA images with matching alpha masks and transparent padding for differently sized batched crops.

`UC_LoadLaMaModel` loads Big LaMa `.safetensors` files from `ComfyUI/models/lama` through Unified Efficient Loader. Connect its `LAMA_MODEL` output to `UC_LaMaInpaint`. Device choices include ComfyUI's default device, CPU, and every visible GPU. Models are never downloaded automatically. Download [Big LaMa](https://huggingface.co/silveroxides/ComfyUI-UtilsCollection-Models/blob/main/big-lama/big-lama.safetensors) or [Anime/Manga Big LaMa](https://huggingface.co/silveroxides/ComfyUI-UtilsCollection-Models/blob/main/big-lama/anime-manga-big-lama.safetensors), then place the selected file in `ComfyUI/models/lama`.

Each staged foreground (including detected faces) supports **Brush** and **Text** in its right-click menu after staging. Additions use the foreground's rectangular canvas, including transparent pixels, and follow its transforms. Brush erasing leaves the original foreground and text untouched. Text is one editable multiline block; click the foreground in Text mode to position it. **Show Brush/Text** controls preview and output visibility independently of editing. **Reset Brush/Text** clears only that content; placement Reset preserves both. Additions stay attached to the same foreground socket when its source changes.

Brush/text PNGs save automatically to ComfyUI input storage; queueing waits for saves. Keep those PNGs when moving workflows. Text stays editable in workflow data, while its saved PNG preserves appearance during backend execution. Tool-local Undo/Redo history lasts for the current editor session.

The Brush panel's **Object Eraser** removes original foreground pixels without altering text. Brush opacity and hardness control erasure strength and edges. Undo/Redo covers painting and object erasures in order; **Reset Brush** restores the original foreground and clears painted content. **Show Brush** also controls whether object erasures apply.

The left sidebar stays visible and shows the selected foreground's context actions when no drawing tool is active. Brush and text share an HSL color picker below their controls. Hold numeric or layer-order arrows to repeat changes; releasing stops the repeat.

### Staged compositor example

[Workflow JSON](workflows/CompositorExampleWorkflow.json) | [Workflow overview](workflows/CompositorExampleWorkflow.jpg) | [Source assets](workflow_assets/)

<img src="workflows/CompositorExampleWorkflow.jpg" alt="Staged MediaPipe face background compositor workflow" width="1200">

### Resolution and workflow parameters

- `UC_AdjustedResolutionParameters`
- `UC_ResolutionSelectorExtended`
- `UC_VideoResolutionSelector`
- `UC_ImageScaleAndResolutionPicker`
- `UC_SwitchInverseNode`
- `UC_SoftSwitchInverseNode`
- `UC_IntegerRangeRandom`
- `UC_RandInt`
- `UC_StaticInt`
- `UC_StaticFloat`
- `UC_RandIntRange`
- `UC_ColorConvertNode`
- `UC_SeedCluster`
- `UC_FromSeedCluster`
- `UC_ExtractBoundingBox`
- `UC_AdjustBoundingBox`
- `UC_Ideogram4BoundingBoxCrop`
- `UC_Ideogram4DebannerPatch`
- `UC_HighResolutionTileSplit`
- `UC_HighResolutionTileAccumulator`
- `UC_HighResolutionTilingGuide`

### Prompt presets

- `UC_SystemMessagePresets`
- `UC_SystemMessageVideoPresets`
- `UC_InstructPromptPresets`
- `UC_InstructPromptVideoPresets`
- `UC_BonusPromptPresets`
- `UC_BonusPromptVideoPresets`
- `UC_EditTargetPresets`
- `UC_EditOpPresets`
- `UC_CameraShotPresets`
- `UC_VLMSysInstrPresets`
- `UC_VLMSysInstrPresetsExperimental`
- `UC_VLMSysInstrLegacyPresets`
- `UC_VLMSysQueryAddPresets`
- `UC_VLMSysQueryRawPresets`
- `UC_VLMSysInstrAdvPresets`
- `UC_VLMSysInstrAdvPresetsExperimental`
- `UC_MiniMaxH3VLMSysInstrPresets`
- `UC_MiniMaxH3VLMSysInstrPresetsExperimental`
- `UC_MiniMaxH3VLMSysInstrAdvPresets`
- `UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental`
- `UC_LegacyPromptPresets`
- `UC_UnifiedPresets`

### Loading, text generation, and text utilities

- `UC_LoadImagePath`
- `UC_LoadImageDirectory`
- `UC_LoadImageWithAlpha`
- `UC_SampleVideoFramesAsImages`
- `UC_MiniMaxH3RefVid` — **H3 Reference Video Components** prepares 24 fps reference frames, H3-ready audio, width, height, and frame count. Its final `video` output preserves the source resolution and framing, adjusting only timing and audio for nodes that accept VIDEO. Set the duration in seconds (`0` uses the whole clip); the selected duration rounds up to H3's supported frame count. Only the separate `frames` output is matched to the nearest standard aspect ratio from Video Resolution Selector and center-cropped to its selected resolution. Audio is resampled to 32 kHz and padded only at the end to an 800-sample boundary, avoiding Core's H3 audio input-cropping issue; missing audio is filled with silence.
- `UC_ImagesToVideoTimeline`
- `UC_VideoTimelineText`
- `UC_LoraLoaderCLIPOnly`
- `UC_LoadLaMaModel`
- `UC_TextGenerate`
- `UC_TextGenerateQwen35SystemPrompt`
- `UC_EmbeddingDetokenizerAnalysis`
- `UC_ImageToVideoPrompt`
- `UC_TagNormalizeCombine`
- `UC_FromList`
- `UC_GetJsonValue`
- `UC_MiniMaxH3Cache`
- `UC_MiniMaxH3SlaAttentionConfig`
- `UC_MiniMaxH3Spectrum`
- `UC_MiniMaxH3PDDAcc`
- `UC_UnifiedAttentionPatcher`
- `UC_MarkdownPreview`
- `UC_BoldFrakturTextStyle`
- `UC_UnBoldFrakturTextStyle`
- `UC_WordJoiner`
- `UC_UnWordJoiner`
- `UC_JSONMinifyRepair`
- `UC_StringUnescape`
- `UC_TextConcatenateAutogrow`
- `UC_TextConcatenateListsAutogrow`
- `UC_Newline`

### MiniMax H3 PDD Acc models

Download the PDD Acc file matching the MiniMax H3 diffusion model:

- [MiniMax H3 FL2VA PDD Acc 8-step](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI/blob/main/minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors)
- [MiniMax H3 Ref2VA PDD Acc 8-step](https://huggingface.co/aptech0081/MiniMax-H3-Acc-LoRAs-ComfyUI/blob/main/minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors)

Place the downloaded `.safetensors` file in `ComfyUI/models/loras`, or in the
configured external directory used by ComfyUI's `loras` model category. Restart
ComfyUI or refresh model files, then select it in `UC_MiniMaxH3PDDAcc`.

### Unified Attention Patcher

`UC_UnifiedAttentionPatcher` returns a cloned model with one selected attention
backend. Connect its `model` output in place of the original model. The
`disabled` mode returns the input model unchanged.

| Attention mode | Applies to | Optional runtime requirement | Behavior |
| --- | --- | --- | --- |
| `FlashAttention` | Attention calls without a mask | A compatible package providing `flash_attn` or `flash_attn_interface` | Uses FlashAttention. `allow_compile` permits compilation after the initial run. |
| `SageAttention` | General model attention | `sageattention`; `sageattn3` or `sageattn3_per_block_mean` additionally need `sageattn3` | Select a Sage kernel from `sage_mode`. `allow_compile` permits compilation after the initial run. |
| `Sparse / MiniMax H3 SLA` | MiniMax H3 self-attention only | CUDA and Triton | Routes each H3 attention block to selected key blocks while retaining dense attention where sparse routing is unsuitable. |

SLA uses its selected `dense_backend` for deliberate dense steps and sparse
fallbacks. `auto` retains the incoming ComfyUI-selected attention callable. No
attention package, model checkpoint, or LoRA is downloaded by this node.

#### SageAttention MiniMax H3 memory option

`h3_memory_optimizations` is available only inside the `SageAttention` mode. It
requires a CUDA MiniMax H3 model and a compatible SageAttention installation.
It reduces the H3 attention path's peak memory use; selecting it for another
model raises an error rather than silently applying a different patch.

#### MiniMax H3 SLA controls

SLA is experimental and only patches MiniMax H3 models with 128-dimensional
attention heads. It does not modify ComfyUI Core files or model weights. The
`minimax_h3_sla_config` input on `UC_UnifiedAttentionPatcher` is optional.
Connect `UC_MiniMaxH3SlaAttentionConfig` to configure its documented controls;
omitting it uses their defaults.

- Main SLA controls:
  - `sparsity`: fraction of ordinary key blocks skipped. Start with the
  default `0.90`; compare output and speed against dense attention for each
  model, resolution, duration, and sampler.
  - `block_size`: routing granularity. Smaller blocks retain finer temporal
  and audio detail at additional routing cost.
  - `dense_tail_steps`: final sampler steps retained on dense attention for
  detail recovery. Set `0` to allow sparse routing at every eligible step.
  - `protect_reference_media`: `Off` adds no visual-reference quota; `Light`
  retains the best-scoring 15% of each reference block range; `Heavy
  Enforcement` retains every reference block.
  - `dense_backend`: `comfy_kitchen`, `auto`, `pytorch`, or a value from
  `CUSTOM_SAGE_MODES`. A selected unavailable backend raises an actionable
  error before sampling.
- `UC_MiniMaxH3SlaAttentionConfig` controls:
  - `minimum_sequence_length`: sequences below this threshold stay on the
  selected dense attention path.
  - `dense_steps`: comma-separated zero-based steps or inclusive ranges kept
  dense, such as `0,3-5`. Default `0` preserves the first sampling step.
  - `protect_audio`: preserves text and audio ranges in every sparse key
  selection.
  - `disable_fp16_accumulation`: disables FP16/BF16 reduced-precision matmul
  accumulation for this SLA sampling run, then restores prior settings.
  - `stabilize_routing`: biases near-cutoff block selection toward the prior
  sampling step. Use it only when motion detail is unstable; it retains a
  bounded routing history while sampling.

SLA calls stay dense when the call is masked, not MiniMax H3 packed
self-attention, uses an unsupported dtype/device, falls below the minimum
sequence length, falls in `dense_steps` or the configured dense tail, lacks
MiniMax H3 layout metadata, or when the sparse kernel fails. Each reason is
reported once in the ComfyUI console for that sampling run. A sparse-kernel
failure also disables further sparse attempts for the remainder of that run.
This preserves a usable model path when SLA cannot apply, but it also means a
run may receive less acceleration than its selected sparsity suggests.

### Scheduler presets

- `Ideogram4SchedulerPreset`
- `UC_SigmaRescale`
- `UC_DiscardPenultimateSigma`
- `UC_SigmoidOffsetScheduler`
- `UC_PowerShiftScheduler`
- `UC_RadianceShiftScheduler`
- `UC_SigmaCurveFromPointsScheduler`
- `UC_SigmaCurvePchipScheduler`

The migrated schedulers also register `sigmoid_offset`, `power_shift`,
`radiance_shift`, `sigma_curve_from_points`, and `sigma_curve_pchip` for Core
scheduler selectors. The Power Shift scheduler was inspired by
[InverserSquaredScheduler](https://github.com/Clybius/ComfyUI-ClybsChromaNodes/blob/main/clyb_Schedulers.py).

`UC_SigmaRescale` maps an existing schedule to exact start and end sigma
values without changing its shape or number of steps.

The dedicated scheduler nodes do not include Core-style denoise controls or
optional penultimate-sigma controls. Connect `UC_SigmaRescale` after a
scheduler when setting image-to-image noise levels. Connect
`UC_DiscardPenultimateSigma` when the selected sampler requires penultimate
sigma removal. Radiance Shift performs its required compensated removal
internally. Sigmoid Offset retains its model-specific `start_sigma`
adjustment.

### Logic and math

- `UC_LogicIF`
- `UC_LogicAND`
- `UC_LogicOR`
- `UC_LogicNOT`
- `UC_LogicXOR`
- `UC_MathAdd`
- `UC_MathSubtract`
- `UC_MathMultiply`
- `UC_MathDivide`
- `UC_MathPower`
- `UC_MathFloor`
- `UC_MathCeil`
- `UC_MathRound`
- `UC_MathModulo`
- `UC_MathAbs`
- `UC_MathSqrt`
- `UC_MathSin`
- `UC_MathCos`
- `UC_MathTan`
- `UC_MathMin`
- `UC_MathMax`
- `UC_MathClamp`
- `UC_MathNumberConvert`
- `UC_StringToNumber`
- `UC_NumberToString`
- `UC_MathCompare`
- `UC_MathOperation`
- `UC_MathAspectRatio`

These nodes replace the equivalent nodes from ComfyUI-LogicMath, ComfyUI_SigmoidOffsetScheduler,
and ComfyUI_PowerShiftScheduler. Remove the standalone pack before accepting ComfyUI's workflow
replacement prompt.
