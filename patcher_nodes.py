"""ComfyUI nodes for model-scoped diffusion-model patches."""

import folder_paths
from comfy_api.latest import io

from .patcher_helpers import (
    CUSTOM_SAGE_MODES,
    MiniMaxH3RadialAttentionConfig,
    MiniMaxH3SlaAttentionConfig,
    SLA_DENSE_BACKENDS,
    SLA_REFERENCE_PROTECTION_MODES,
    SpectrumH3Config,
    effective_bootstrap_first_forecast,
    list_minimax_h3_projections,
    patch_ideogram4_debanner,
    patch_minimax_h3_clip_projection,
    patch_minimax_h3_pdd_model,
    patch_unified_attention_model,
    patch_minimax_h3_cache_model,
    patch_minimax_h3_spectrum_model,
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
        return io.NodeOutput(MiniMaxH3RadialAttentionConfig(
            dense_blocks=dense_blocks,
            dense_start_steps=dense_start_steps,
            dense_end_steps=dense_end_steps,
            block_size=block_size,
            decay_factor=decay_factor,
            allow_compile=allow_compile,
        ).validate())


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
        return io.NodeOutput(MiniMaxH3SlaAttentionConfig(
            minimum_sequence_length=minimum_sequence_length,
            dense_steps=dense_steps,
            protect_audio=protect_audio,
            disable_fp16_accumulation=disable_fp16_accumulation,
            stabilize_routing=stabilize_routing,
        ).validate())


class UC_MiniMaxH3ClipProjectionPatcher(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3ClipProjectionPatcher",
            display_name="MiniMax H3 CLIP Projection Patcher",
            category="advanced/model/patches",
            description="Projects a Qwen3-VL 4B or 8B encoder into MiniMax H3's 32B conditioning space.",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "projection",
                    options=list_minimax_h3_projections(),
                    tooltip=(
                        "Download one matching 4B or 8B .safetensors projection into "
                        "ComfyUI/models/clip_projections. If none are listed, see README.md "
                        "for model links and loader settings."
                    ),
                ),
            ],
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
            io.DynamicCombo.Option(
                key="FlashAttention",
                inputs=[
                    io.Boolean.Input("allow_compile", default=False, tooltip="May speed up later runs after a slower first run. Leave off if compilation causes problems."),
                ],
            ),
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
                io.Model.Input(
                    "model",
                    tooltip=(
                        "MiniMax H3 model to accelerate. Match an FL2VA model with "
                        "an FL2VA PDD file, or a Ref2VA model with a Ref2VA PDD file."
                    ),
                ),
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
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("strength", default=0.6, min=0.0, max=2.0, step=0.01),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, strength) -> io.NodeOutput:
        if strength == 0.0:
            return io.NodeOutput(model)
        return io.NodeOutput(patch_ideogram4_debanner(model, strength))


class UC_MiniMaxH3PDDAcc(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3PDDAcc",
            display_name="MiniMax H3 PDD Acc (Experimental)",
            category="advanced/model/patches",
            description=(
                "Applies a MiniMax H3 PDD Acc trunk LoRA and Core-managed "
                "step-conditioned output heads. Use the SIGMAS output with Euler."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "pdd_lora",
                    options=folder_paths.get_filename_list("loras"),
                    tooltip=(
                        "Choose the PDD Acc file matching your MiniMax H3 model: "
                        "FL2VA for first/last-frame generation or Ref2VA for "
                        "reference-image generation. Download links and installation "
                        "instructions are in README.md under MiniMax H3 PDD Acc models."
                    ),
                ),
                io.Combo.Input(
                    "nfe",
                    options=["8", "7", "6", "5", "4"],
                    default="8",
                    tooltip=(
                        "Sampling steps: 8 gives the intended quality, 5-7 provide "
                        "trained-envelope middle grounds, and 4 is fastest. An extra "
                        "step can compensate when using cache. Use this node's sigmas output."
                    ),
                ),
                io.String.Input(
                    "partition",
                    default="",
                    tooltip=(
                        "Advanced custom sampling schedule. Leave empty for the selected "
                        "step count; otherwise enter comma-separated groups of 4 or 8 "
                        "that total 32."
                    ),
                ),
                io.Float.Input(
                    "lora_strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Strength of the PDD changes inside the model. Keep 1.0 for the "
                        "intended result; lower values weaken acceleration tuning and "
                        "higher values may distort output."
                    ),
                ),
                io.Float.Input(
                    "head_strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Strength of the PDD output correction. Keep 1.0 for the intended "
                        "result; lower values blend toward the original model output."
                    ),
                ),
                io.Combo.Input(
                    "on_off_grid",
                    options=["error", "clamp"],
                    default="error",
                    tooltip=(
                        "Error stops sampling when the sampler uses unsupported steps. "
                        "Clamp forces the nearest supported step but can reduce quality."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model", display_name="model"),
                io.Sigmas.Output("sigmas", display_name="sigmas"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        pdd_lora: str,
        nfe: str,
        partition: str,
        lora_strength: float,
        head_strength: float,
        on_off_grid: str,
    ) -> io.NodeOutput:
        patched, sigmas = patch_minimax_h3_pdd_model(
            model,
            pdd_lora,
            int(nfe),
            partition,
            lora_strength,
            head_strength,
            on_off_grid,
        )
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
            description=(
                "Applies an approximate block-stack residual cache to a cloned "
                "MiniMax H3 model without globally modifying the Core model class."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "reuse_threshold",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Maximum accumulated relative feature change allowed for "
                        "reuse. Higher values skip more work and may reduce fidelity."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.15,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Sampling progress at which cache reuse may begin.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=0.90,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Sampling progress after which cache reuse stops.",
                ),
                io.Int.Input(
                    "max_steps",
                    default=2,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Maximum number of consecutive block-stack skips.",
                ),
                io.Combo.Input(
                    "device",
                    options=["auto", "cuda", "cpu"],
                    default="auto",
                    tooltip=(
                        "auto keeps the residual with the model, cuda requires CUDA, "
                        "and cpu offloads the cached residual."
                    ),
                ),
                io.Boolean.Input(
                    "verbose",
                    default=False,
                    tooltip="Print per-step cache decisions and a final skip summary.",
                ),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_steps: int,
        device: str,
        verbose: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            patch_minimax_h3_cache_model(
                model=model,
                reuse_threshold=reuse_threshold,
                start_percent=start_percent,
                end_percent=end_percent,
                max_steps=max_steps,
                device=device,
                verbose=verbose,
            )
        )


class UC_MiniMaxH3Spectrum(io.ComfyNode):
    """Apply experimental spectral feature forecasting to a cloned H3 model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3Spectrum",
            display_name="MiniMax H3 Spectrum (Experimental)",
            category="advanced/model/patches",
            description=(
                "Forecasts complete post-transformer MiniMax H3 features with a "
                "Chebyshev ridge model while retaining native output reconstruction."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "execution_mode",
                    options=["spectrum", "forced_actual"],
                    default="spectrum",
                    tooltip="Spectrum forecasts eligible steps; forced_actual measures patch overhead without forecasts.",
                ),
                io.Int.Input("degree", default=1, min=1, max=16, step=1, tooltip="Chebyshev polynomial degree. Values other than 1 disable first-forecast bootstrap."),
                io.Float.Input("ridge_lambda", default=0.1, min=0.0, max=10.0, step=0.01, tooltip="Regularization that stabilizes the history fit."),
                io.Float.Input("window_size", default=2.0, min=1.0, max=16.0, step=0.05, tooltip="Initial interval between actual transformer evaluations."),
                io.Float.Input("flex_window", default=0.75, min=0.0, max=8.0, step=0.05, tooltip="Amount added to the actual-evaluation interval after each refresh."),
                io.Int.Input("warmup_steps", default=1, min=0, max=64, step=1, tooltip="Initial solver steps that always run the transformer. Values above 1 disable first-forecast bootstrap."),
                io.Int.Input("tail_actual_steps", default=1, min=0, max=64, step=1, tooltip="Final solver steps that always run the transformer."),
                io.Int.Input("max_history", default=8, min=2, max=64, step=1, tooltip="Maximum actual post-transformer features retained for fitting."),
                io.Float.Input("video_blend_weight", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Video spectral share; 1 uses pure spectral prediction and 0 uses local linear prediction."),
                io.Float.Input("audio_blend_weight", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Audio spectral share; default 0 avoids direct spectral mixing of audio rows."),
                io.Combo.Input("history_storage", options=["system_ram", "vram"], default="system_ram", tooltip="Device used for bounded causal feature history."),
                io.Boolean.Input("bootstrap_first_forecast", default=True, tooltip="Forecast step 1 from step 0 only when degree is 1 and warmup is at most 1."),
                io.Boolean.Input("offline_smoothing_replay", default=True, tooltip="Run optional two-pass smoothing replay to protect H3 audio continuity."),
                io.Combo.Input("offline_archive_storage", options=["system_ram", "vram"], default="system_ram", tooltip="Device used for replay anchors retained through both passes."),
                io.Boolean.Input("debug", default=False, tooltip="Log forecast decisions, timing, retained memory, and run summary."),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        execution_mode: str,
        degree: int,
        ridge_lambda: float,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        tail_actual_steps: int,
        max_history: int,
        video_blend_weight: float,
        audio_blend_weight: float,
        history_storage: str,
        bootstrap_first_forecast: bool,
        offline_smoothing_replay: bool,
        offline_archive_storage: str,
        debug: bool,
    ) -> io.NodeOutput:
        config = SpectrumH3Config(
            enabled=True,
            force_actual=execution_mode == "forced_actual",
            degree=degree,
            ridge_lambda=ridge_lambda,
            window_size=window_size,
            flex_window=flex_window,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            max_history=max_history,
            blend_weight=video_blend_weight,
            audio_blend_weight=audio_blend_weight,
            history_storage=history_storage,
            bootstrap_first_forecast=effective_bootstrap_first_forecast(
                bootstrap_first_forecast, degree, warmup_steps
            ),
            offline_smoothing_replay=(
                offline_smoothing_replay if execution_mode == "spectrum" else False
            ),
            offline_archive_storage=offline_archive_storage,
            debug=debug,
        )
        return io.NodeOutput(patch_minimax_h3_spectrum_model(model, config))
