"""ComfyUI nodes for model-scoped diffusion-model patches."""

import folder_paths
from comfy_api.latest import io

from ..helpers.patcher_helpers import (
    CUSTOM_SAGE_MODES,
    MiniMaxH3RadialAttentionConfig,
    MiniMaxH3SlaAttentionConfig,
    SLA_DENSE_BACKENDS,
    SLA_REFERENCE_PROTECTION_MODES,
    list_minimax_h3_projections,
    patch_ideogram4_debanner,
    patch_krea2_nag_model,
    patch_minimax_h3_clip_projection,
    patch_minimax_h3_pdd_model,
    patch_unified_attention_model,
    patch_minimax_h3_cache_model,
    patch_minimax_h3_diffusion_cache_model,
    MOSS_IMAGE_EDIT_PRESETS,
    patch_moss_image_edit_model,
    KREA2_EDIT_BOOST_MODES,
    KREA2_EDIT_ROPE_POSITIONS,
    patch_krea2_edit_model,
    patch_minimax_h3_context_windows,
)


MiniMaxH3RadialAttentionConfigType = io.Custom("MINIMAX_H3_RADIAL_ATTENTION_CONFIG")
MiniMaxH3SlaAttentionConfigType = io.Custom("MINIMAX_H3_SLA_ATTENTION_CONFIG")


class UC_MiniMaxH3RadialAttentionConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3RadialAttentionConfig",
            display_name="MiniMax H3 Radial Attention Config",
            category="advanced/model/patches",
            description="Configures sparse radial attention for the target video rows of a MiniMax H3 model.",
            inputs=[
                io.Int.Input("dense_blocks", default=1, min=0, max=56, step=1, tooltip="Keep this many early transformer blocks unchanged. Higher values are safer but reduce the speed gain."),
                io.Int.Input("dense_start_steps", default=1, min=0, max=100, step=1, tooltip="Keep this many early sampling steps unchanged. Helps protect the initial composition."),
                io.Int.Input("dense_end_steps", default=1, min=0, max=100, step=1, tooltip="Keep this many final sampling steps unchanged. Helps protect finishing detail."),
                io.Combo.Input("block_size", options=[64, 128], default=128, tooltip="128 usually runs faster. 64 keeps a finer sparse-attention pattern."),
                io.Float.Input("decay_factor", default=0.2, min=0.0, max=1.0, step=0.1, tooltip="How much distant video frames stay connected. Lower is faster; higher keeps more temporal detail."),
                io.Boolean.Input("allow_compile", default=False, tooltip="May speed up later runs after a slower first run. Leave off if compilation causes problems."),
            ],
            outputs=[MiniMaxH3RadialAttentionConfigType.Output("minimax_h3_radial_config", display_name="Radial Config", tooltip="Connect to Unified Attention Patcher when using Sparse / MiniMax H3 Radial.")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, dense_blocks, dense_start_steps, dense_end_steps, block_size, decay_factor, allow_compile) -> io.NodeOutput:
        return io.NodeOutput(MiniMaxH3RadialAttentionConfig(dense_blocks=dense_blocks, dense_start_steps=dense_start_steps, dense_end_steps=dense_end_steps, block_size=block_size, decay_factor=decay_factor, allow_compile=allow_compile).validate())


class UC_MiniMaxH3SlaAttentionConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3SlaAttentionConfig",
            display_name="MiniMax H3 SLA Attention Config",
            category="advanced/model/patches",
            description="Configures block-selected sparse attention for MiniMax H3.",
            inputs=[
                io.Int.Input("minimum_sequence_length", default=8192, min=0, max=1000000, step=1024, tooltip="Sequences below this length keep the existing dense attention path."),
                io.String.Input("dense_steps", default="0", multiline=False, tooltip="Comma-separated zero-based sampling steps or inclusive ranges kept dense, for example 0,3-5."),
                io.Boolean.Input("protect_audio", default=True, tooltip="Keep text and audio token ranges in every sparse attention selection."),
                io.Boolean.Input("disable_fp16_accumulation", default=True, tooltip="Disable FP16 and BF16 reduced-precision matmul accumulation for this SLA sampling run."),
                io.Boolean.Input("stabilize_routing", default=False, tooltip="Bias near-cutoff routing toward the prior sampling step to reduce unstable motion detail."),
            ],
            outputs=[MiniMaxH3SlaAttentionConfigType.Output("minimax_h3_sla_config", display_name="SLA Config", tooltip="Connect to Unified Attention Patcher when using Sparse / MiniMax H3 SLA.")],
        )

    @classmethod
    def execute(cls, minimum_sequence_length, dense_steps, protect_audio, disable_fp16_accumulation, stabilize_routing) -> io.NodeOutput:
        return io.NodeOutput(MiniMaxH3SlaAttentionConfig(minimum_sequence_length=minimum_sequence_length, dense_steps=dense_steps, protect_audio=protect_audio, disable_fp16_accumulation=disable_fp16_accumulation, stabilize_routing=stabilize_routing).validate())


class UC_MiniMaxH3ClipProjectionPatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3ClipProjectionPatcher",
            display_name="MiniMax H3 CLIP Projection Patcher",
            category="advanced/model/patches",
            description="Projects a Qwen3-VL 4B or 8B encoder into MiniMax H3's 32B conditioning space.",
            inputs=[io.Clip.Input("clip"), io.Combo.Input("projection", options=list_minimax_h3_projections(), tooltip=("Download one matching 4B or 8B .safetensors projection into ComfyUI/models/clip_projections. If none are listed, see README.md for model links and loader settings."))],
            outputs=[io.Clip.Output("clip")],
        )

    @classmethod
    def execute(cls, clip, projection) -> io.NodeOutput:
        return io.NodeOutput(patch_minimax_h3_clip_projection(clip, projection))


class UC_UnifiedAttentionPatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        modes = [
            io.DynamicCombo.Option(key="disabled", inputs=[]),
            io.DynamicCombo.Option(key="FlashAttention", inputs=[io.Boolean.Input("allow_compile", default=False, tooltip="May speed up later runs after a slower first run. Leave off if compilation causes problems.")]),
            io.DynamicCombo.Option(
                key="SageAttention",
                inputs=[
                    io.Combo.Input("sage_mode", options=list(CUSTOM_SAGE_MODES), default="auto", tooltip="SageAttention version to use. Auto is the normal choice; select another only when your GPU or installed SageAttention version needs it."),
                    io.Boolean.Input("allow_compile", default=False, tooltip="May speed up later runs after a slower first run. Leave off if compilation causes problems."),
                    io.Boolean.Input("h3_memory_optimizations", default=False, tooltip="MiniMax H3 only. Reduces peak VRAM use during attention. Requires CUDA and a compatible SageAttention install."),
                ],
            ),
            io.DynamicCombo.Option(
                key="Sparse / MiniMax H3 SLA",
                inputs=[
                    io.Float.Input("sparsity", default=0.9, min=0.0, max=0.95, step=0.05, tooltip="Fraction of ordinary key blocks skipped after protected media is retained."),
                    io.Combo.Input("block_size", options=[32, 64, 128], default=32, tooltip="Tokens represented by each SLA routing block. 128 uses 128-token query blocks and 64-token key blocks."),
                    io.Int.Input("dense_tail_steps", default=1, min=0, max=8, tooltip="Final sampler steps that keep dense attention for detail recovery."),
                    io.Combo.Input("protect_reference_media", options=list(SLA_REFERENCE_PROTECTION_MODES), default="Light", tooltip="Off keeps no visual-reference quota; Light keeps the reference SLA quota; Heavy Enforcement keeps every reference block."),
                    io.Combo.Input("dense_backend", options=list(SLA_DENSE_BACKENDS), default="comfy_kitchen", tooltip="Dense attention backend used for deliberate dense steps and SLA fallbacks."),
                    MiniMaxH3SlaAttentionConfigType.Input("minimax_h3_sla_config", display_name="SLA Config", tooltip="Connect MiniMax H3 SLA Attention Config to override the documented defaults.", optional=True),
                ],
            ),
            # io.DynamicCombo.Option(
            #     key="Sparse / MiniMax H3 Radial",
            #     inputs=[
            #         MiniMaxH3RadialAttentionConfigType.Input("minimax_h3_radial_config", display_name="Radial Config", tooltip="Connect MiniMax H3 Radial Attention Config. Radial attention is experimental and applies only to MiniMax H3 video generation."),
            #     ],
            # ),
        ]
        return io.Schema(
            node_id="UC_UnifiedAttentionPatcher",
            display_name="Unified Attention Patcher",
            category="advanced/model/patches",
            description="Applies a selected attention backend to a cloned model.",
            inputs=[
                io.Model.Input("model", tooltip=("MiniMax H3 model to accelerate. Match an FL2VA model with an FL2VA PDD file, or a Ref2VA model with a Ref2VA PDD file.")),
                io.DynamicCombo.Input("attention_mode", options=modes, display_name="Attention Mode", tooltip="Choose an attention backend. It needs its matching installed package."),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, attention_mode) -> io.NodeOutput:
        return io.NodeOutput(patch_unified_attention_model(model, attention_mode))


class UC_Ideogram4DebannerPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Ideogram4DebannerPatch",
            display_name="Ideogram 4 Debanner Patch",
            category="advanced/model/patches",
            description="Applies correction on every invocation of this patched model. Use it only for the first split-sigma segment, then continue with the original conditional model.",
            inputs=[io.Model.Input("model"), io.Float.Input("strength", default=0.6, min=0.0, max=2.0, step=0.01)],
            outputs=[io.Model.Output("model", display_name="model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, strength) -> io.NodeOutput:
        if strength == 0.0:
            return io.NodeOutput(model)
        return io.NodeOutput(patch_ideogram4_debanner(model, strength))


class UC_Krea2NAG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Krea2NAG",
            display_name="Krea2 Normalized Attention Guidance",
            category="advanced/model/patches",
            description="Enables negative prompts on Krea2 even at CFG 1.0 by steering attention away from unwanted concepts without breaking image quality.",
            inputs=[
                io.Model.Input("model", tooltip="Krea2 model to guide. Connect the output to your sampler with CFG set to 1.0."),
                io.Float.Input("nag_scale", default=5.0, min=0.0, max=50.0, step=0.1, tooltip="How strongly to push away from the negative prompt. Higher values remove unwanted elements more aggressively; 1.0 or lower turns guidance off."),
                io.Float.Input("nag_tau", default=2.5, min=1.0, max=10.0, step=0.01, tooltip="Safety limit that prevents strong guidance from causing color burn or visual artifacts. Lower values keep the image safer; higher values allow stronger changes."),
                io.Float.Input("nag_alpha", default=0.25, min=0.0, max=1.0, step=0.01, tooltip="How much of the guided result is blended into the original image features. 0.25 keeps natural Krea2 quality while applying negative guidance; 0.0 turns guidance off."),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Point in sampling where negative guidance begins (0.0 starts from the very first step)."),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Point in sampling where negative guidance stops. Lowering this (for example to 0.5) speeds up generation while keeping most of the negative prompt effect."),
                io.Conditioning.Input("nag_negative", optional=True, tooltip="Optional negative prompt conditioning. Leave disconnected to automatically use the negative input on your sampler."),
            ],
            outputs=[io.Model.Output("model", display_name="model", tooltip="Krea2 model with Normalized Attention Guidance enabled.")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, nag_scale: float = 5.0, nag_tau: float = 2.5, nag_alpha: float = 0.25, start_percent: float = 0.0, end_percent: float = 1.0, nag_negative=None) -> io.NodeOutput:
        if nag_alpha == 0.0 or nag_scale <= 1.0 or start_percent >= end_percent:
            return io.NodeOutput(model)
        return io.NodeOutput(patch_krea2_nag_model(model=model, nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha, start_percent=start_percent, end_percent=end_percent, nag_negative=nag_negative))


class UC_MiniMaxH3PDDAcc(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3PDDAcc",
            display_name="MiniMax H3 PDD Acc (Experimental)",
            category="advanced/model/patches",
            description=("Applies a MiniMax H3 PDD Acc trunk LoRA and Core-managed step-conditioned output heads. Use the SIGMAS output with Euler."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "pdd_lora",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip=("Choose the PDD Acc file matching your MiniMax H3 model: FL2VA for first/last-frame generation or Ref2VA for reference-image generation. Download links and installation instructions are in README.md under MiniMax H3 PDD Acc models."),
                ),
                io.Combo.Input("nfe", options=["8", "7", "6", "5", "4"], default="8", tooltip=("Sampling steps: 8 gives the intended quality, 5-7 provide trained-envelope middle grounds, and 4 is fastest. An extra step can compensate when using cache. Use this node's sigmas output.")),
                io.String.Input("partition", default="", tooltip=("Advanced custom sampling schedule. Leave empty for the selected step count; otherwise enter comma-separated groups of 4 or 8 that total 32.")),
                io.Float.Input("lora_strength", default=1.0, min=-2.0, max=2.0, step=0.01, tooltip=("Strength of the PDD changes inside the model. Keep 1.0 for the intended result; lower values weaken acceleration tuning and higher values may distort output.")),
                io.Float.Input("head_strength", default=1.0, min=0.0, max=2.0, step=0.01, tooltip=("Strength of the PDD output correction. Keep 1.0 for the intended result; lower values blend toward the original model output.")),
                io.Combo.Input("on_off_grid", options=["error", "clamp"], default="error", tooltip=("Error stops sampling when the sampler uses unsupported steps. Clamp forces the nearest supported step but can reduce quality.")),
            ],
            outputs=[io.Model.Output("model", display_name="model"), io.Sigmas.Output("sigmas", display_name="sigmas")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, pdd_lora: str, nfe: str, partition: str, lora_strength: float, head_strength: float, on_off_grid: str) -> io.NodeOutput:
        patched, sigmas = patch_minimax_h3_pdd_model(model, pdd_lora, int(nfe), partition, lora_strength, head_strength, on_off_grid)
        return io.NodeOutput(patched, sigmas)


class UC_MiniMaxH3Cache(io.ComfyNode):
    """Apply the MiniMax H3 whole-block-stack cache to a cloned model patcher.

    Implementation invariant: current Core exposes per-block replacements but no
    replacement boundary around the complete H3 block stack. The local ``_forward``
    adaptation deliberately adds that missing boundary through
    ``ModelPatcher.add_object_patch``. Core installs it only while the output clone
    is patched and restores the original bound method when unpatching; this is not
    a global class monkey-patch. Replacing it with chained per-block wrappers is not
    equivalent unless preservation of existing replacements, prefetch behavior,
    whole-stack residual capture, and ModelPatcher lifecycle is proven end to end.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3Cache",
            display_name="MiniMax H3 Cache",
            category="advanced/model/patches",
            description=("Applies an approximate block-stack residual cache to a cloned MiniMax H3 model without globally modifying the Core model class."),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("reuse_threshold", default=0.05, min=0.0, max=1.0, step=0.01, tooltip=("Maximum accumulated relative feature change allowed for reuse. Higher values skip more work and may reduce fidelity.")),
                io.Float.Input("start_percent", default=0.15, min=0.0, max=1.0, step=0.01, tooltip="Sampling progress at which cache reuse may begin."),
                io.Float.Input("end_percent", default=0.90, min=0.0, max=1.0, step=0.01, tooltip="Sampling progress after which cache reuse stops."),
                io.Int.Input("max_steps", default=2, min=1, max=10, step=1, tooltip="Maximum number of consecutive block-stack skips."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto", tooltip=("auto keeps the residual with the model, cuda requires CUDA, and cpu offloads the cached residual.")),
                io.Boolean.Input("verbose", default=False, tooltip="Print per-step cache decisions and a final skip summary."),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
        )

    @classmethod
    def execute(cls, model, reuse_threshold: float, start_percent: float, end_percent: float, max_steps: int, device: str, verbose: bool) -> io.NodeOutput:
        return io.NodeOutput(patch_minimax_h3_cache_model(model=model, reuse_threshold=reuse_threshold, start_percent=start_percent, end_percent=end_percent, max_steps=max_steps, device=device, verbose=verbose))


class UC_MiniMaxH3DiffusionCache(io.ComfyNode):
    """Applies a diffusion-aware residual cache to a cloned MiniMax H3 model.

    Unlike loop-counter caching which counts arbitrary step indices, this node
    anchors cache reuse directly to physical diffusion noise phases (sigma).
    This ensures identical, predictable caching behavior regardless of whether
    a normal, karras, exponential, or custom scheduler is selected.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3DiffusionCache",
            display_name="MiniMax H3 Diffusion-Aware Cache",
            category="advanced/model/patches",
            description=("Applies a scheduler-invariant residual cache to MiniMax H3 by coupling cache reuse directly to physical diffusion noise phases (sigma) instead of arbitrary step indices."),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "preset",
                    options=["balanced", "high_fidelity", "high_speed", "custom"],
                    default="balanced",
                    tooltip=(
                        "Physical diffusion profile presets.\n"
                        "• 'balanced' (Recommended): ~1.5x speedup with negligible perceptual loss; "
                        "caches during middle semantic formation (sigma 0.75 -> 0.15).\n"
                        "• 'high_fidelity': Restricts caching to deep mid-steps (sigma 0.65 -> 0.25) "
                        "with tight drift tolerance to protect subtle motion and textures.\n"
                        "• 'high_speed': Aggressive ~2.0x speedup caching across a wider window "
                        "(sigma 0.85 -> 0.10).\n"
                        "• 'custom': Unlocks the specific noise thresholds below."
                    ),
                ),
                io.Float.Input(
                    "composition_cutoff",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Normalized noise level (0.0 = clean, 1.0 = pure noise) below which cache reuse may begin. "
                        "During early high-noise steps (> cutoff), the model establishes global composition, "
                        "character anatomy, and camera motion; keeping this at or below 0.80 prevents double "
                        "heads, anatomical warping, and motion jitter. Active in 'custom' preset."
                    ),
                ),
                io.Float.Input(
                    "detail_protection",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Normalized noise level (0.0 = clean, 1.0 = pure noise) below which cache reuse stops. "
                        "During late low-noise steps (< protection), the model renders micro-textures, fine hair, "
                        "skin pores, and sharp audio transients; setting this above 0.10 prevents smearing and blurriness. "
                        "Setting to 0.0 caches until the very last step. Active in 'custom' preset."
                    ),
                ),
                io.Float.Input(
                    "drift_tolerance",
                    default=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    tooltip=(
                        "Sensitivity multiplier for feature drift (1.0 = standard calibrated tolerance). "
                        "Lower values (< 1.0) enforce strict feature matching, forcing fresh transformer runs "
                        "whenever fast motion or sudden visual changes occur. Higher values (> 1.0) permit more "
                        "skips in static or slow-motion scenes. Scales preset thresholds as well."
                    ),
                ),
                io.Float.Input(
                    "max_noise_span",
                    default=0.20,
                    min=0.01,
                    max=0.60,
                    step=0.01,
                    tooltip=(
                        "Maximum cumulative noise distance (delta-sigma fraction) the cache may skip consecutively "
                        "before forcing a full model refresh. Automatically adapts across step counts: on a 15-step "
                        "run it limits skips to 1-2 steps, while on a 50-step run it safely allows skipping 7-8 micro-steps. "
                        "Active in 'custom' preset."
                    ),
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    tooltip=("Storage location for the cached block residual tensor.\n• 'auto': Keeps residual in VRAM alongside the model for maximum speed.\n• 'cuda': Requires CUDA VRAM.\n• 'cpu': Offloads residual to system RAM between steps to conserve VRAM (small transfer overhead)."),
                ),
                io.Boolean.Input("verbose", default=False, tooltip="Print detailed per-step cache decisions, physical noise values (sigma), and cumulative speedup to the console."),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
        )

    @classmethod
    def execute(cls, model, preset: str, composition_cutoff: float, detail_protection: float, drift_tolerance: float, max_noise_span: float, device: str, verbose: bool) -> io.NodeOutput:
        return io.NodeOutput(patch_minimax_h3_diffusion_cache_model(model=model, preset=preset, composition_cutoff=composition_cutoff, detail_protection=detail_protection, drift_tolerance=drift_tolerance, max_noise_span=max_noise_span, device=device, verbose=verbose))


class UC_MOSSImageEditPatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MOSSImageEditPatcher",
            display_name="MOSS-Image Edit Patcher",
            category="advanced/model/patches",
            description=("Configures a MOSS-Image diffusion model for reference-guided image editing, character/style transfer, and masked inpainting using clean reference attention across all 42 transformer blocks and spatial structure retention."),
            inputs=[
                io.Model.Input("model", tooltip="MOSS-Image diffusion model to configure for reference-guided image editing. Connect the output to your sampler."),
                io.Combo.Input(
                    "preset",
                    options=list(MOSS_IMAGE_EDIT_PRESETS.keys()),
                    default="balanced_edit",
                    tooltip=(
                        "Editing profile. 'balanced_edit' combines prompt-driven edits with reference layout and appearance preservation; "
                        "'preserve_structure' locks pose, framing, and composition tightly to the reference image; "
                        "'subject_reference' transfers a reference character or object into a new pose or scene using the reference index axis; "
                        "'style_reference' transfers color palette and brushwork into a new composition; "
                        "and 'inpaint_only' edits inside a connected mask while keeping unmasked areas locked to the reference."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=0.65,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=("Overall reference guidance strength across the 42 transformer blocks. The default 0.65 balances prompt changes with reference details; raising it pulls stronger appearance and texture from the reference image, while 0.0 disables reference attention."),
                ),
                io.Float.Input(
                    "structure_retention",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "How strongly the reference image's spatial layout, pose, and unedited background are preserved during the early composition steps. "
                        "The default 0.50 keeps the overall composition while allowing object or clothing edits; raise toward 1.0 to lock layout tightly, "
                        "or set to 0.0 when placing a reference subject into a completely new pose or scene."
                    ),
                ),
                io.Boolean.Input("uncond_skip_layer_22", default=True, tooltip=("Bypasses transformer block 22 on the negative prompt pass, matching OpenMOSS's recommended two-pass guidance setting for sharper structure without adding a third sampling pass. Turn off to run all blocks on the negative pass.")),
                io.Latent.Input(
                    "reference_latent",
                    optional=True,
                    tooltip=(
                        "Optional VAE-encoded reference or source image latent (encoded with the Flux 2 VAE). You can also attach reference images to conditioning with ReferenceLatent or Advanced Visual Conditioning Encode, or pass the encoded image directly into your sampler's latent_image input at denoise 1.0."
                    ),
                ),
                io.Mask.Input("mask", optional=True, tooltip=("Optional edit mask where white (1.0) marks regions to edit according to the prompt and black (0.0) preserves the reference image. Leave disconnected to edit across the full image.")),
            ],
            outputs=[io.Model.Output("model", display_name="model", tooltip="MOSS-Image model configured for reference-guided image editing.")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, preset: str = "balanced_edit", strength: float = 0.65, structure_retention: float = 0.50, uncond_skip_layer_22: bool = True, reference_latent=None, mask=None, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(patch_moss_image_edit_model(model=model, preset=preset, strength=strength, structure_retention=structure_retention, uncond_skip_layer_22=uncond_skip_layer_22, reference_latent=reference_latent, mask=mask, **kwargs))


class UC_Krea2EditPatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Krea2EditPatcher",
            display_name="Krea2 Edit Patcher",
            category="advanced/model/patches",
            description="Configures a Krea2 diffusion model for reference-guided image editing, character transfer, and scene composition using clean ModelPatcher object patches and comfy-kitchen kernel integration.",
            inputs=[
                io.Model.Input("model", tooltip="Krea2 model to configure for reference-guided editing. Connect the output to your sampler."),
                io.Latent.Input("reference_latent", optional=True, tooltip="Optional primary reference latent (subject or scene image)."),
                io.Latent.Input("reference_latent_b", optional=True, tooltip="Optional secondary reference latent (subject photo in two-image composition)."),
                io.Image.Input("source_image", optional=True, tooltip="Optional primary source image (pre-encoded at node execution time when VAE is connected, avoiding mid-sampling VRAM eviction)."),
                io.Image.Input("source_image_b", optional=True, tooltip="Optional secondary source image."),
                io.Vae.Input("vae", optional=True, tooltip="Optional VAE for pre-encoding source images cleanly before sampling begins."),
                io.Float.Input("ref_boost", default=1.0, min=0.0, max=20.0, step=0.05, tooltip="Reference attention fidelity dial for the primary/subject reference (1.0 = standard, >1.0 strengthens appearance and likeness)."),
                io.Float.Input("ref_boost_a", default=1.0, min=0.0, max=20.0, step=0.05, tooltip="Reference attention fidelity dial for the scene reference in two-image workflows (1.0 = standard)."),
                io.Combo.Input(
                    "boost_mode",
                    options=list(KREA2_EDIT_BOOST_MODES),
                    default="Fast / Key Scaling (FlashAttention)",
                    tooltip="Attention boosting method. 'Fast / Key Scaling' scales keys directly without an attention mask, keeping FlashAttention and SageAttention active with zero memory overhead. 'Exact / Logit Bias' reproduces training logit bias with session-cached masks.",
                ),
                io.Combo.Input(
                    "ref_position", options=list(KREA2_EDIT_ROPE_POSITIONS), default="aligned (trained)", tooltip="RoPE position space placement for the primary reference. 'aligned' pins onto the output layout; 'beside' shifts right of the canvas for flexible restaging; 'bag of features' collapses spatial coordinates."
                ),
                io.Combo.Input("ref_position_a", options=list(KREA2_EDIT_ROPE_POSITIONS), default="aligned (trained)", tooltip="RoPE position space placement for the scene reference in two-image workflows."),
                io.Combo.Input("fit_mode", options=["fit", "crop"], default="fit", tooltip="How source images fit output aspect ratio in pixel space before sampling."),
                io.Int.Input("target_width", default=1024, min=64, max=4096, step=16, tooltip="Target generation width in pixels (used for pre-sampling image fitting)."),
                io.Int.Input("target_height", default=1024, min=64, max=4096, step=16, tooltip="Target generation height in pixels."),
                io.Boolean.Input("cache_text_fusion", default=True, tooltip="Caches invariant TextFusion transformer output on step 0 to eliminate redundant compute across remaining sampling steps."),
            ],
            outputs=[io.Model.Output("model", display_name="model", tooltip="Krea2 model configured for reference-guided editing.")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        reference_latent=None,
        reference_latent_b=None,
        source_image=None,
        source_image_b=None,
        vae=None,
        ref_boost: float = 1.0,
        ref_boost_a: float = 1.0,
        boost_mode: str = "Fast / Key Scaling (FlashAttention)",
        ref_position: str = "aligned (trained)",
        ref_position_a: str = "aligned (trained)",
        fit_mode: str = "fit",
        target_width: int = 1024,
        target_height: int = 1024,
        cache_text_fusion: bool = True,
        **kwargs,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            patch_krea2_edit_model(
                model=model,
                reference_latent=reference_latent,
                reference_latent_b=reference_latent_b,
                source_image=source_image,
                source_image_b=source_image_b,
                vae=vae,
                ref_boost=ref_boost,
                ref_boost_a=ref_boost_a,
                boost_mode=boost_mode,
                ref_position=ref_position,
                ref_position_a=ref_position_a,
                fit_mode=fit_mode,
                target_width=target_width,
                target_height=target_height,
                cache_text_fusion=cache_text_fusion,
                **kwargs,
            )
        )


class UC_MiniMaxH3ContextPatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3ContextPatcher",
            display_name="MiniMax H3 Context Patcher",
            category="advanced/model/patches",
            description=("Patches a MiniMax H3 model with sliding temporal context windows. Enables generating longer continuous video latents (10s, 15s+) without out-of-memory errors."),
            search_aliases=["minimax context", "h3 context windows", "context patcher", "sliding window", "long video patcher"],
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 model to patch with sliding context windows."),
                io.Combo.Input(
                    "window_size",
                    options=["124 frames (~5.2s - Standard)", "73 frames (~3.0s - Low VRAM)", "141 frames (~5.9s - High VRAM)", "158 frames (~6.6s - Extra High VRAM)", "Disabled (Full Latent)"],
                    default="124 frames (~5.2s - Standard)",
                    tooltip=("Maximum temporal window evaluated in a single DiT pass. 124 frames fits standard GPU memory. Total video length is controlled by your latent or encoder node, not this setting."),
                ),
                io.Combo.Input(
                    "overlap",
                    options=["39 frames (~1.6s - Standard)", "22 frames (~0.9s - Fast)", "56 frames (~2.3s - Deep Blend)", "5 frames (~0.2s - Minimal)"],
                    default="39 frames (~1.6s - Standard)",
                    tooltip=("Overlap between adjacent sliding windows. 39 frames provides smooth seamless blending on the H3 grid; 22 is faster; 56 gives maximum seam smoothing."),
                ),
                io.Combo.Input("blend_profile", options=["pyramid", "flat"], default="pyramid", optional=True, advanced=True, tooltip=("Window weighting profile. Pyramid linearly fades window boundaries to zero for smooth, artifact-free transitions.")),
                io.Boolean.Input("freenoise", default=True, optional=True, advanced=True, tooltip="Applies FreeNoise noise shuffling across sliding windows to prevent repetitive patterns."),
            ],
            outputs=[io.Model.Output("model", display_name="model", tooltip="Patched MiniMax H3 model ready to connect to Guider or Sampler.")],
        )

    @classmethod
    def execute(cls, model, window_size="124 frames (~5.2s - Standard)", overlap="39 frames (~1.6s - Standard)", blend_profile="pyramid", freenoise=True) -> io.NodeOutput:
        patched = patch_minimax_h3_context_windows(model=model, window_size=window_size, overlap=overlap, blend_profile=blend_profile, freenoise=freenoise)
        return io.NodeOutput(patched)
