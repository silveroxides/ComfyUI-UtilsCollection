import re
import math
import os
import gc
import types
import logging
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import comfy
import nodes
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io
from .helper_functions import get_token_count, get_token_count_scaled, resize_nchw
from .encoder_helpers import(
    encode_embedding_scaled_bias,
    is_image_token,
    evaluate_formula,
    evaluate_conditioning_formula,
    evaluate_conditioning_consensus_blend,
    save_source_visual_embeddings,
    blend_text_vectors,
    find_visual_token_range,
    build_token_to_conditioning_map,
    encode_embedding_classical_scaled_bias,
    strip_contextual_weight_syntax,
    load_vlm_image_tensor,
    krea2_user_content_span,
    krea2_token_ids,
    find_subsequence,
    krea2_attn_forward_weight,
    ImageInputMapping,
    VISION_BLOCK,
    prepare_image_placeholder_prompt,
    extract_and_flatten_images,
    resolve_embedding_output_path,
    visual_fusion_grid,
    prepare_vlm_image,
    prepare_vae_reference_image,
    qwen3vl_visual_encoder_path,
    is_klein_vl_text_encoder,
    is_minimax_h3_text_encoder,
    tokenize_minimax_h3_prompt,
    format_minimax_h3_prompt,
    CONSENSUS_BLEND_PRESETS,
    execute_advanced_visual_consensus,
    execute_advanced_minimax_h3_image_to_video,
    execute_minimax_h3_vlm_guide,
    build_minimax_h3_media_config,
    MINIMAX_H3_MEDIA_STRUCTURE,
    MINIMAX_H3_VIDEO_LATENT_MODES,
    execute_token_fusion_visual_conditioning,
)
from .image_helpers import VIDEO_FRAME_TIMESTAMP_FORMATS

def apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode):
    if not ref_latents:
        return conditioning

    stage = getattr(clip, "cond_stage_model", None)
    stage_names = " ".join(
        type(value).__name__
        for value in (stage, getattr(stage, "clip_model", None), getattr(stage, "clip", None))
        if value is not None
    ).lower()
    if "parallel" in ref_latent_mode and "krea2" not in stage_names:
        # Keep semantic conditioning and reference-latent conditioning as separate
        # Comfy conditioning entries. Sequence concatenation is not a parallel stream.
        tokens_neutral = clip.tokenize("")
        conditioning_neutral = clip.encode_from_tokens_scheduled(tokens_neutral)
        out = [[tensor, metadata.copy()] for tensor, metadata in conditioning]
        for tensor, metadata in conditioning_neutral:
            neutral_meta = metadata.copy()
            neutral_meta["reference_latents"] = list(ref_latents)
            out.append([tensor, neutral_meta])
        return out
    else:
        # Standard append mode
        return node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)


def multiply_conditioning(conditioning, multiplier):
    if multiplier == 1.0:
        return conditioning
    output = []
    for tensor, metadata in conditioning:
        new_metadata = metadata.copy()
        pooled = new_metadata.get("pooled_output")
        if pooled is not None:
            new_metadata["pooled_output"] = pooled * multiplier
        output.append([tensor * multiplier, new_metadata])
    return output


class UC_AttentionBiasTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_AttentionBiasTextEncode",
            category="advanced/conditioning",
            display_name="Attention Bias Encode",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("text", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        if '<' not in text and '>' not in text and '=' not in text:
            tokens = clip.tokenize(text)
            return io.NodeOutput(clip.encode_from_tokens_scheduled(tokens))

        bias_pattern = re.compile(r"<([^>]+)=([0-9.-]+)>")
        split_pattern = re.compile(r"(<[^>]+=[0-9.-]+>)")
        segments = split_pattern.split(text)

        clean_text = ""
        biases_to_apply = []

        for segment in segments:
            if not segment:
                continue

            match = bias_pattern.fullmatch(segment)
            if match:
                bias_text, strength_str = match.groups()
                before = clip.tokenize(clean_text)
                key = next(iter(before))
                start_index = len(before[key][0])
                strength = float(strength_str)
                clean_text += bias_text
                after = clip.tokenize(clean_text)
                end_index = len(after[key][0])
                if end_index > start_index:
                    biases_to_apply.append({"start": start_index, "end": end_index, "strength": strength})
            else:
                clean_text += segment

        tokens = clip.tokenize(clean_text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if not biases_to_apply:
            return io.NodeOutput(conditioning)

        output = []
        for cond, metadata in conditioning:
            seq_len = cond.shape[1]
            key = next(iter(tokens))
            mapping = build_token_to_conditioning_map(tokens[key][0], cond)
            attn_mask = torch.zeros((1, seq_len, seq_len), dtype=cond.dtype, device=cond.device)
            for bias in biases_to_apply:
                strength = bias["strength"]
                if not math.isfinite(strength) or strength < 0:
                    raise ValueError("Attention weights must be finite and non-negative.")
                value = math.log(max(strength, 1e-6))
                if bias["start"] >= len(mapping):
                    continue
                start = mapping[bias["start"]][0]
                end = mapping[min(bias["end"] - 1, len(mapping) - 1)][1]
                if start < end:
                    # Key-column odds scaling. Row scaling would square the
                    # weighted intersection and is intentionally not applied.
                    attn_mask[:, :, start:end] += value
            new_metadata = metadata.copy()
            existing = new_metadata.get("attention_mask")
            if existing is not None and torch.is_tensor(existing):
                if existing.shape[-1] != seq_len:
                    raise ValueError("Existing attention mask does not match the encoded sequence length.")
                existing = existing.to(device=cond.device)
                if existing.dtype == torch.bool:
                    additive = torch.zeros(existing.shape, device=cond.device, dtype=cond.dtype)
                    additive.masked_fill_(~existing, -torch.finfo(cond.dtype).max)
                else:
                    additive = existing.to(dtype=cond.dtype)
                while additive.ndim < attn_mask.ndim:
                    additive = additive.unsqueeze(-2)
                attn_mask = attn_mask + additive
            new_metadata["attention_mask"] = attn_mask
            new_metadata["attention_mask_img_shape"] = (1, 1)
            output.append([cond, new_metadata])
        return io.NodeOutput(output)

# --- Type Definitions for Modular Configurations ---
TextBlendConfig = io.Custom("TEXT_BLEND_CONFIG")
VisualFusionConfig = io.Custom("VISUAL_FUSION_CONFIG")
AdvancedConsensusConfig = io.Custom("ADVANCED_CONSENSUS_CONFIG")
VisualConsensusConfig = io.Custom("VISUAL_CONSENSUS_CONFIG")
MiniMaxH3MediaConfig = io.Custom("MINIMAX_H3_MEDIA_CONFIG")

class UC_TextConsensusBlendConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_TextConsensusBlendConfig",
            display_name="Text Consensus Blend Configurator",
            category="advanced/conditioning",
            inputs=[
                io.Combo.Input(
                    "blend_preset",
                    options=[
                        "off", "custom", "baseline", "power_blend", "high_clarity", "smooth", "varied_merge", "diverse_concept", "high_diversity_concept",
                        "dsc_baseline", "dsc_high_clarity", "dsc_smooth", "dsc_varied_merge", "dsc_diverse_concept", "dsc_high_diversity_concept"
                    ],
                    default="baseline",
                    tooltip="Preset configuration for Text Consensus-Weighted Blending. Set to 'off' to bypass CWB, or 'custom' to use the manual parameters below."
                ),
                io.Combo.Input(
                    "blend_method",
                    options=["linear", "consensus"],
                    default="consensus",
                    tooltip="Active only in 'custom' preset. 'consensus' aligns prompts and filters noise; 'linear' averages them."
                ),
                io.Combo.Input(
                    "consensus_type",
                    options=["mean", "median"],
                    default="median",
                    tooltip="Active only in 'custom' preset. 'median' rejects up to 50% outlying noise; 'mean' is smooth averaging."
                ),
                io.Combo.Input(
                    "alignment_method",
                    options=["index", "similarity"],
                    default="similarity",
                    tooltip="Active only in 'custom' preset. 'similarity' aligns shifted prompt concepts; 'index' aligns them sequentially."
                ),
                io.Float.Input("alignment_threshold", default=0.4, min=0.0, max=1.0, step=0.01, tooltip="Active only in similarity alignment. Minimum similarity to match words."),
                io.Float.Input("similarity_threshold", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Prunes passing words if similarity to consensus falls below this."),
                io.Float.Input("power_alpha", default=2.0, min=0.0, max=10.0, step=0.1, tooltip="Soft-masking exponent. Higher values penalize outliers (e.g. 2.0)."),
                io.Float.Input("diversity_beta", default=0.0, min=0.0, max=10.0, step=0.1, tooltip="Diversity exponent. Dampens hyper-frequent details to boost variety (e.g. 1.5)."),
                io.Boolean.Input("rescale_norm", default=True, tooltip="Norm Rescaling. Keeps activation energy high to prevent washed-out colors."),
                io.Float.Input("global_scale", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Global scale multiplier applied to the blended outputs."),
                io.Boolean.Input("dynamic_similarity_contrast", default=False, tooltip="Stretches similarities to soft [0.7, 1.0] band to boost contrast."),
                io.Boolean.Input("soft_comfort_bandpass", default=False, tooltip="Softens the diversity bandpass ceiling to prevent clipping."),
                io.Float.Input("position_weight", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Bias similarity alignment toward nearby normalized token positions. Zero preserves current behavior."),
                io.Boolean.Input("preserve_common_prefix", default=False, tooltip="Keep the longest numerically identical conditioning prefix exactly from the first input."),
            ],
            outputs=[
                TextBlendConfig.Output("text_blend_config", display_name="Blend Config")
            ],
        )

    @classmethod
    def execute(
        cls,
        blend_preset: str,
        blend_method: str,
        consensus_type: str,
        alignment_method: str,
        alignment_threshold: float,
        similarity_threshold: float,
        power_alpha: float,
        diversity_beta: float,
        rescale_norm: bool,
        global_scale: float,
        dynamic_similarity_contrast: bool = False,
        soft_comfort_bandpass: bool = False,
        position_weight: float = 0.0,
        preserve_common_prefix: bool = False,
    ) -> io.NodeOutput:
        config = {
            "blend_preset": blend_preset,
            "blend_method": blend_method,
            "consensus_type": consensus_type,
            "alignment_method": alignment_method,
            "alignment_threshold": alignment_threshold,
            "similarity_threshold": similarity_threshold,
            "power_alpha": power_alpha,
            "diversity_beta": diversity_beta,
            "rescale_norm": rescale_norm,
            "global_scale": global_scale,
            "dynamic_similarity_contrast": dynamic_similarity_contrast,
            "soft_comfort_bandpass": soft_comfort_bandpass,
            "position_weight": position_weight,
            "preserve_common_prefix": preserve_common_prefix,
        }
        return io.NodeOutput(config)


class UC_VisualFusionConfig(io.ComfyNode):
    """
    Configuration node for visual component fusion.
    Specifies methods for blending or spatially interleaving isolated visual token vectors,
    and provides controls to save dynamically blended visual embeddings directly to disk.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_VisualFusionConfig",
            display_name="Visual Component Fusion Configurator",
            category="advanced/conditioning",
            inputs=[
                io.Combo.Input(
                    "visual_fusion_method",
                    options=["off", "linear", "spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"],
                    default="spatial-checkerboard",
                    tooltip="Method to combine isolated visual-token vectors. Spatial methods select source vectors according to a reproducible token-grid pattern; generation quality is model and prompt dependent."
                ),
                io.Int.Input("visual_block_size", default=2, min=1, max=8, step=1, tooltip="Active for spatial-block-interleave. Size of the spatial token patches to group and switch together."),
                io.Float.Input("dither_ratio", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="Active for spatial-dither-random. Probability of selecting the first image. Remaining images are selected with a checkerboard pattern."),
                io.Boolean.Input("save_blended_embeds", default=False, tooltip="Enable to save the blended visual tokens as a standalone .safetensors embedding."),
                io.String.Input("save_path", default="blended_visual_embeds.safetensors", tooltip="Target filename/path under models/embeddings to save the .safetensors file."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True, tooltip="Seed for the spatial-dither-random pattern."),
                io.Combo.Input(
                    "visual_encoder_path",
                    options=["grid-deepstack", "legacy-flat"],
                    default="grid-deepstack",
                    tooltip="Qwen3-VL encoder route used by visual fusion. grid-deepstack uses current Core grid MRoPE and DeepStack injection; legacy-flat reproduces the pre-d0008a89 flat 1D route."
                ),
                io.Combo.Input("dither_secondary_pattern", options=["checkerboard", "block-interleave", "dither-random-reverse", "dither-random-forward"], default="checkerboard", tooltip="How images 2+ participate in spatial-dither-random. Reverse starts with the last pair and works toward image 1; forward starts with images 1 and 2 and accumulates later images."),
                io.Boolean.Input("dither_mask_cleanup", default=False, tooltip="Swap paired one-token image-1 islands and holes with a deterministic 3x3 pass while preserving every source's token count."),
                io.Float.Input("spatial_perturbation", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="Seeded spatial variation for hard fusion methods. Exchanges cells between sources without changing any source's token count; higher values may reduce spatial coherence."),
            ],
            outputs=[
                VisualFusionConfig.Output("visual_fusion_config", display_name="Fusion Config")
            ]
        )

    @classmethod
    def execute(
        cls,
        visual_fusion_method: str,
        visual_block_size: int,
        dither_ratio: float,
        save_blended_embeds: bool = False,
        save_path: str = "blended_visual_embeds.safetensors",
        seed: int = 0,
        visual_encoder_path: str = "grid-deepstack",
        dither_secondary_pattern: str = "checkerboard",
        dither_mask_cleanup: bool = False,
        spatial_perturbation: float = 0.0,
    ) -> io.NodeOutput:
        config = {
            "visual_fusion_method": visual_fusion_method,
            "visual_block_size": visual_block_size,
            "dither_ratio": dither_ratio,
            "seed": seed,
            "visual_encoder_path": visual_encoder_path,
            "dither_secondary_pattern": dither_secondary_pattern,
            "dither_mask_cleanup": dither_mask_cleanup,
            "spatial_perturbation": spatial_perturbation,
            "save_blended_embeds": save_blended_embeds,
            "save_path": save_path
        }
        return io.NodeOutput(config)


class UC_AdvancedConsensusConfiguration(UC_TextConsensusBlendConfig):
    _EXPERIMENTAL = None
    # Core fills these together; this subclass has a different output socket.
    _RETURN_TYPES = None
    _RETURN_NAMES = None
    _OUTPUT_IS_LIST = None
    _OUTPUT_TOOLTIPS = None

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "UC_AdvancedConsensusConfiguration"
        schema.is_experimental = True
        schema.display_name = "Advanced Consensus Configuration"
        schema.inputs.append(
            io.Int.Input(
                "resolution_samples",
                default=1,
                min=1,
                max=15,
                step=2,
                tooltip="Exact odd adjacent-resolution sample count used only when consensus is enabled. A value of 1 remains one resolution sample.",
            )
        )
        schema.inputs.append(
            io.Int.Input(
                "sample_offset",
                default=32,
                min=32,
                max=512,
                step=32,
                tooltip="Equivalent-square resolution distance between adjacent samples. Larger values create more distinct visual grids at greater scale separation.",
            )
        )
        schema.outputs = [
            AdvancedConsensusConfig.Output(
                "advanced_consensus_config",
                display_name="Advanced Consensus Config",
            )
        ]
        return schema

    @classmethod
    def execute(
        cls,
        blend_preset,
        blend_method,
        consensus_type,
        alignment_method,
        alignment_threshold,
        similarity_threshold,
        power_alpha,
        diversity_beta,
        rescale_norm,
        global_scale,
        dynamic_similarity_contrast,
        soft_comfort_bandpass,
        position_weight,
        preserve_common_prefix,
        resolution_samples,
        sample_offset,
    ) -> io.NodeOutput:
        config = super().execute(
            blend_preset,
            blend_method,
            consensus_type,
            alignment_method,
            alignment_threshold,
            similarity_threshold,
            power_alpha,
            diversity_beta,
            rescale_norm,
            global_scale,
            dynamic_similarity_contrast,
            soft_comfort_bandpass,
            position_weight,
            preserve_common_prefix,
        ).args[0]
        config["resolution_samples"] = resolution_samples
        config["sample_offset"] = sample_offset
        return io.NodeOutput(config)


class UC_VisualConsensusConfiguration(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_VisualConsensusConfiguration",
            display_name="Visual Consensus Configuration",
            category="advanced/conditioning",
            inputs=[
                VisualFusionConfig.Input(
                    "visual_fusion_config",
                    display_name="Visual Fusion Configuration",
                    tooltip="Complete visual-stage configuration from Visual Component Fusion Configurator. Its method is authoritative; off disables spatial fusion.",
                ),
                AdvancedConsensusConfig.Input(
                    "consensus_config",
                    display_name="Consensus Configuration",
                    tooltip="Complete consensus configuration from Advanced Consensus Configuration. Its preset is authoritative; off disables cross-resolution consensus.",
                ),
            ],
            outputs=[
                VisualConsensusConfig.Output("visual_consensus_config", display_name="Visual Consensus Config"),
            ],
        )

    @classmethod
    def execute(
        cls,
        visual_fusion_config,
        consensus_config,
    ) -> io.NodeOutput:
        visual = dict(visual_fusion_config)
        consensus = dict(consensus_config)
        return io.NodeOutput({
            "enable_spatial_fusion": visual.get("visual_fusion_method") != "off",
            "enable_consensus": consensus.get("blend_preset") != "off",
            "visual": visual,
            "consensus": consensus,
        })

class UC_ConditioningConsensusBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Conditioning.Input("conditioning", optional=True),
            prefix="conditioning_"
        )
        return io.Schema(
            node_id="UC_ConditioningConsensusBlend",
            display_name="Conditioning Consensus",
            category="advanced/conditioning",
            inputs=[
                TextBlendConfig.Input("text_blend_config", display_name="Blend Config", optional=True, tooltip="Optional configuration from UC_TextConsensusBlendConfig. Defaults to baseline CWB if disconnected."),
                io.Autogrow.Input("conditioning_inputs", template=autogrow_template, tooltip="Conditionings are blended in ascending socket order; disconnected sockets are ignored."),
            ],
            outputs=[
                io.Conditioning.Output("conditioning")
            ],
        )

    @classmethod
    def execute(cls, conditioning_inputs: io.Autogrow.Type, text_blend_config: dict = None) -> io.NodeOutput:
        """
        Blends stock or custom ComfyUI conditioning outputs post-encoder using CWB math.
        """
        if not conditioning_inputs:
            raise ValueError("At least one conditioning input must be connected to UC_ConditioningConsensusBlend.")

        active_conds = []
        if conditioning_inputs is not None:
            for k in sorted(conditioning_inputs.keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)):
                v = conditioning_inputs[k]
                if v is not None:
                    active_conds.append(v)

        if not active_conds:
            raise ValueError("All connected conditioning inputs to UC_ConditioningConsensusBlend are empty or None.")

        if len(active_conds) == 1:
            return io.NodeOutput(active_conds[0])

        if text_blend_config is None:
            text_blend_config = {"blend_preset": "baseline"}

        if text_blend_config.get("blend_preset") == "off":
            return io.NodeOutput(active_conds[0])

        schedule_lengths = {len(conditioning) for conditioning in active_conds}
        if len(schedule_lengths) != 1:
            raise ValueError("All conditioning inputs must have the same number of scheduled entries.")

        def compatible_metadata(metadata_items):
            layout_keys = {"attention_mask", "attention_mask_img_shape", "embeds_info"}
            result = {}
            common_keys = set.intersection(*(set(item) for item in metadata_items)) - layout_keys - {"pooled_output"}
            for key in common_keys:
                values = [item[key] for item in metadata_items]
                first = values[0]
                if torch.is_tensor(first):
                    if all(torch.is_tensor(value) and value.shape == first.shape and torch.equal(value, first) for value in values[1:]):
                        result[key] = first
                elif all(value is first for value in values[1:]):
                    result[key] = first
                elif isinstance(first, (str, int, float, bool, type(None))) and all(value == first for value in values[1:]):
                    result[key] = first
            return result

        device = comfy.model_management.get_torch_device()
        compute_dtype = comfy.model_management.intermediate_dtype()
        blended_conditioning = []
        for schedule_index in range(next(iter(schedule_lengths))):
            entries = [conditioning[schedule_index] for conditioning in active_conds]
            sequence_tensors = {}
            pooled_tensors = {}
            for index, (tensor, metadata) in enumerate(entries):
                key = chr(97 + index)
                sequence_tensors[key] = tensor
                pooled = metadata.get("pooled_output") if metadata else None
                pooled_tensors[key] = pooled
            C_blended, P_blended = blend_text_vectors(
                sequence_tensors,
                text_blend_config,
                pooled_tensors=pooled_tensors,
                device=device,
                compute_dtype=compute_dtype,
            )
            metadata = compatible_metadata([entry[1] for entry in entries])
            if P_blended is not None:
                metadata["pooled_output"] = P_blended
            blended_conditioning.append([C_blended, metadata])
        return io.NodeOutput(blended_conditioning)

class UC_ScaledBiasTextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, skip_template=True)
        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override. MiniMax H3 requires off and uses Core's dedicated H3 reference conditioning instead.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, image=None, vae_dimension_multiple=8) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, image=image)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off" and image is not None:
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
            s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
            ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template)
        else:
            conditioning = encode_embedding_scaled_bias(clip, prompt)

        return io.NodeOutput(conditioning)


class UC_ScaledBiasTextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ScaledBiasTextEncodeSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode System Prompt (Scaled Bias)",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "z-image"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="(Klein only) Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        skip_template = False
        if model_type == "klein" and len(thinking_content) > 0:
            # Klein with custom thinking content
            if len(system_prompt) > 0:
                llama_template = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n" +
                    f"<|im_start|>user\n{{}}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            else:
                llama_template = (
                    "<|im_start|>user\n{}<|im_end|>\n" +
                    f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
                )
            skip_template = True
        elif len(system_prompt) > 0:
            template = SYSTEM_PROMPT_TEMPLATES.get(model_type, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
            llama_template = f"{template['prefix']}{system_prompt}{template['suffix']}"
            skip_template = (model_type == "klein")
        else:
            llama_template = None

        conditioning = encode_embedding_scaled_bias(clip, prompt, llama_template=llama_template, skip_template=skip_template)
        return io.NodeOutput(conditioning)


class UC_TextEncodeFlux2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeFlux2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 dev System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = r"[SYSTEM_PROMPT]"
            template_suffix = r"[/SYSTEM_PROMPT][INST]{}[/INST]"
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeKleinSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeKleinSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Flux2 Klein System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content to inject. Leave empty for default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )
        else:
            llama_template = (
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n" + thinking_content + "\n</think>\n\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeKrea2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeKrea2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Krea2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="") -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if len(system_prompt) > 0:
            llama_template = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            llama_template = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n{}<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class TextEncodeSystemEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeSystemEditPlus",
            display_name="TextEncodeSystemEditPlus",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path). 'Fast' = 1024x1024, 'Balanced' = 1280x1280, 'Detailed' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, vae_resolution, ref_latent_mode="off", vae=None, image1=None, image2=None, image3=None, vae_dimension_multiple=8) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        image_prompt = ""

        VLM_RESOLUTIONS = {
            "Fast (384)": 384,
            "Balanced (512)": 512,
            "Detailed (768)": 768,
            "Large (1024)": 1024,
            "X-Large (1280)": 1280,
            "XX-Large (1536)": 1536
        }

        VAE_RESOLUTIONS = {
            "Ultra (512)": 512,
            "Turbo (768)": 768,
            "Fast (1024)": 1024,
            "Balanced (1280)": 1280,
            "Detailed (1536)": 1536
        }

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)

                # 1. Semantic Path Scaling (VLM)
                if vlm_resolution == "Original":
                    images_vl.append(image)
                else:
                    vlm_size = VLM_RESOLUTIONS[vlm_resolution]
                    total_vlm = vlm_size * vlm_size
                    scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                    width_vlm = round(samples.shape[3] * scale_by_vlm)
                    height_vlm = round(samples.shape[2] * scale_by_vlm)

                    s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
                    images_vl.append(s_vlm.movedim(1, -1))

                # 2. Structural Path Scaling (VAE)
                if vae is not None and ref_latent_mode != "off":
                    if "multi" in ref_latent_mode or len(ref_latents) == 0:
                        target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                        s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                        ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        # Construct the complete template string via safe concatenation to prevent formatting errors
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + image_prompt + prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + image_prompt + prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeSystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeSystemEditPlusAdvanced",
            display_name="TextEncodeSystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Images are flattened in ascending socket order; every image in a connected batch becomes the next sequential image input."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, vae_dimension_multiple=8) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in sorted order of raw_images keys
            for num in sorted(raw_images.keys()):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                image = raw_images[num]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeKrea2SystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeKrea2SystemEditPlusAdvanced",
            display_name="TextEncodeKrea2SystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Main text prompt. Supports visual math blending: |formula| to blend image inputs at pixel-tensor level before encoding. "
                        "Example: |((image_input_1 * 1.075) + (image_input_2 * 1.025)) / 1.5| to blend styles/concepts. "
                        "Supported math operations: +, -, *, /, clamp, min, max, abs, on variables image_input_1 to image_input_16."
                    ),
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, vae_dimension_multiple=8) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\n" + system_prompt + "<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                "<|im_start|>user\n" + "<|im_end|>\n" +
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                "<|im_start|>user\n" + modified_prompt + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )

        # Pass skip_template=True so the tokenizer doesn't try to wrap or append extra blocks
        tokens = clip.tokenize(full_prompt, images=images_vl, skip_template=True)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in sorted order of raw_images keys
            for num in sorted(raw_images.keys()):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                image = raw_images[num]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeEditPlusAdvanced",
            display_name="TextEncodeEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Main text prompt. Supports visual math blending: |formula| to blend image inputs at pixel-tensor level before encoding. "
                        "Example: |((image_input_1 * 1.075) + (image_input_2 * 1.025)) / 1.5| to blend styles/concepts. "
                        "Supported math operations: +, -, *, /, clamp, min, max, abs, on variables image_input_1 to image_input_16."
                    ),
                ),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Autogrow.Input("image_inputs", template=autogrow_template),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, vae_dimension_multiple=8) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)

        images_vl = []

        def process_vlm_image(image, res):
            if image is None:
                return None
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            if res == "Original":
                return image
            else:
                vlm_size = VLM_RESOLUTIONS[res]
                total_vlm = vlm_size * vlm_size
                scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                width_vlm = round(samples.shape[3] * scale_by_vlm)
                height_vlm = round(samples.shape[2] * scale_by_vlm)

                s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
                return s_vlm.movedim(1, -1)

        # Create dict of preprocessed VLM images for math evaluation
        processed_images = {}
        for num, img in raw_images.items():
            name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            processed_images[name] = process_vlm_image(img, vlm_resolution)

        # Parse and replace any math formulas enclosed in pipes |formula|
        math_pattern = re.compile(r"\|([^|]+)\|")

        def replace_formula(match):
            expression = match.group(1).strip()
            result_tensor = evaluate_formula(expression, processed_images)
            images_vl.append(result_tensor)
            return "<|vision_start|><|image_pad|><|vision_end|>"

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # Re-check for keywords in the modified prompt
        has_keywords = bool(pattern.search(modified_prompt)) or len(images_vl) > 0

        if has_keywords:
            # Replace keywords dynamically and build images_vl in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    processed_img = processed_images.get(ImageInputMapping.get_display_name(dict_key, is_zero_indexed), process_vlm_image(img, vlm_resolution))
                    images_vl.append(processed_img)
                    return "<|vision_start|><|image_pad|><|vision_end|>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, modified_prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[name]
                images_vl.append(processed_img)
                image_prompt += f"<|vision_start|><|image_pad|><|vision_end|>"

            modified_prompt = image_prompt + modified_prompt

        # Pass standard tokens to tokenize (with images mapped to tags) and encode
        tokens = clip.tokenize(modified_prompt, images=images_vl)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in sorted order of raw_images keys
            for num in sorted(raw_images.keys()):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                image = raw_images[num]
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeGemmaSystemEditPlusAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeGemmaSystemEditPlusAdvanced",
            display_name="TextEncodeGemmaSystemEditPlusAdvanced",
            category="model/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path). 'Fast' = 384x384, 'Balanced' = 512x512, 'Detailed' = 768x768, 'Large' = 1024x1024, 'X-Large' = 1280x1280, 'XX-Large' = 1536x1536, 'Original' uses native resolution.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Images are flattened in ascending socket order; every image in a connected batch becomes the next sequential image input."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, vae_dimension_multiple=8) -> io.NodeOutput:
        # Collect, extract, and parse all autogrow keys (including batched images)
        raw_images, flat_images, is_zero_indexed = extract_and_flatten_images(image_inputs)

        # Check if the prompt has any image_input_ keyword matches (case-insensitive)
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)
        has_keywords = bool(pattern.search(prompt))

        images_vl_raw = []

        if has_keywords:
            # Replace keywords dynamically and build images_vl_raw in order of appearance
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = num - 1 if is_zero_indexed else num
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    images_vl_raw.append(img)
                    return "<img><image_soft_token><end_of_image>"
                return ""

            modified_prompt = pattern.sub(replace_keyword, prompt)
        else:
            # Fallback: prepend all connected images in numerical order of their slots
            image_prompt = ""
            for num in sorted(raw_images.keys()):
                img = raw_images[num]
                images_vl_raw.append(img)
                display_num = num + 1 if is_zero_indexed else num
                image_prompt += f"<img><image_soft_token><end_of_image>"

            modified_prompt = image_prompt + prompt

        # Construct the complete template string via safe concatenation
        if len(system_prompt) > 0:
            full_prompt = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n" + modified_prompt + "<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            full_prompt = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n" +
                "<start_of_turn>user\n" + modified_prompt + "<end_of_turn>\n<start_of_turn>model\n"
            )

        # 1. First tokenize the text without passing images, getting raw 262144 token IDs
        tokens = clip.tokenize(full_prompt, skip_template=True)

        # 2. Helper to process image for VLM
        def process_vlm_image(image, res):
            if image is None:
                return None
            if res == "Original":
                return image[:, :, :, :3]
            VLM_RESOLUTIONS = {
                "Fast (384)": 384,
                "Balanced (512)": 512,
                "Detailed (768)": 768,
                "Large (1024)": 1024,
                "X-Large (1280)": 1280,
                "XX-Large (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            vlm_size = VLM_RESOLUTIONS[res]
            total_vlm = vlm_size * vlm_size
            scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
            width_vlm = round(samples.shape[3] * scale_by_vlm)
            height_vlm = round(samples.shape[2] * scale_by_vlm)

            s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
            return s_vlm.movedim(1, -1)[:, :, :, :3]

        # 3. Process the images and manually inject them sequentially into the 262144 tokens
        if len(images_vl_raw) > 0:
            processed_images = [process_vlm_image(img, vlm_resolution) for img in images_vl_raw]

            # Loop over all tokenizer sections (e.g. 'gemma3_12b')
            for key, val in tokens.items():
                if isinstance(val, list):
                    embed_count = 0
                    for r in val:
                        if isinstance(r, list):
                            for i, token in enumerate(r):
                                if isinstance(token, tuple) and len(token) > 0:
                                    if token[0] == 262144 and embed_count < len(processed_images):
                                        # Replace the token ID (index 0 of the tuple) with the visual payload dict
                                        r[i] = ({"type": "image", "data": processed_images[embed_count]},) + token[1:]
                                        embed_count += 1

        # 4. Encode from the modified tokens dict
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process in order of images_vl_raw
            for i, image in enumerate(images_vl_raw):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class UC_TextEncodeLtxv2SystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeLtxv2SystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with LTXV 2 System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image", optional=True),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt="", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, image=None, vae_dimension_multiple=8) -> io.NodeOutput:
        # Build template with string concat (ComfyUI pattern)
        if image is not None:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n\n<image_soft_token>{}<end_of_turn>\n\n<start_of_turn>model\n"
            )
        elif len(system_prompt) > 0:
            llama_template = (
                "<start_of_turn>system\n" + system_prompt + "<end_of_turn>\n" +
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )
        else:
            llama_template = (
                "<start_of_turn>system\nYou are a helpful assistant.<end_of_turn>\n<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
            )

        if image is not None:
            tokens = clip.tokenize(prompt, llama_template=llama_template, image=image)
        else:
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        ref_latents = []
        if vae is not None and ref_latent_mode != "off" and image is not None:
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            samples = image.movedim(-1, 1)
            target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
            s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
            ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)

        return io.NodeOutput(conditioning)


class UC_TextEncodeZITSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeZITSystemPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image System Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt=None) -> io.NodeOutput:
        if len(system_prompt) > 0:
            template_prefix = "<|im_start|>system\n"
            template_suffix = (
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
            llama_template = f"{template_prefix}{system_prompt}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_TextEncodeZImageThinkPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeZImageThinkPrompt",
            category="advanced/conditioning",
            display_name="Text Encode with Z-Image Thinking Prompt",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("thinking", multiline=True, dynamic_prompts=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, thinking=None) -> io.NodeOutput:
        if len(thinking) > 0:
            template_prefix = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            template_suffix = "\n</think>\n\n"
            llama_template = f"{template_prefix}{thinking}{template_suffix}"
            tokens = clip.tokenize(prompt, llama_template=llama_template)
        else:
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


# Template definitions for unified node
SYSTEM_PROMPT_TEMPLATES = {
    "flux2dev": {
        "prefix": r"[SYSTEM_PROMPT]",
        "suffix": r"[/SYSTEM_PROMPT][INST]{}[/INST]",
    },
    "klein": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    "z-image": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    },
    "krea2": {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    },
}


def system_prompt_template(model_type, system_prompt, thinking_content=""):
    if model_type == "klein" and thinking_content:
        if system_prompt:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
            ), True
        return (
            "<|im_start|>user\n{}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{thinking_content}\n</think>\n\n"
        ), True
    if model_type == "z-image-thinking" and thinking_content:
        system_block = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
        return (
            system_block
            + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            + f"{thinking_content}\n</think>\n\n"
        ), False
    if system_prompt:
        template_key = "z-image" if model_type == "z-image-thinking" else model_type
        template = SYSTEM_PROMPT_TEMPLATES.get(template_key, SYSTEM_PROMPT_TEMPLATES["flux2dev"])
        return f"{template['prefix']}{system_prompt}{template['suffix']}", model_type == "klein"
    return None, False


class UC_TextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_TextEncodeSystemPrompt",
            display_name="System Prompt Encode",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "krea2", "z-image", "z-image-thinking"],
                    default="flux2dev",
                    tooltip="Select the model type to use the correct template format.",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input(
                    "thinking_content",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Custom thinking content for Klein or the z-image-thinking profile. Leave empty for the model default.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="") -> io.NodeOutput:
        llama_template, skip_template = system_prompt_template(model_type, system_prompt, thinking_content)
        if llama_template is None:
            tokens = clip.tokenize(prompt)
        else:
            tokens = clip.tokenize(prompt, llama_template=llama_template, skip_template=skip_template)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)


class UC_WeightedTextEncodeSystemPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_WeightedTextEncodeSystemPrompt",
            display_name="Weighted System Prompt Text Encode",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "model_type",
                    options=["flux2dev", "klein", "krea2", "z-image", "z-image-thinking"],
                    default="flux2dev",
                ),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.String.Input("thinking_content", multiline=True, dynamic_prompts=True, default=""),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, model_type, prompt, system_prompt="", thinking_content="", multiplier=1.0):
        llama_template, skip_template = system_prompt_template(model_type, system_prompt, thinking_content)
        conditioning = encode_embedding_classical_scaled_bias(
            clip,
            prompt,
            llama_template=llama_template,
            skip_template=skip_template,
        )
        return io.NodeOutput(multiply_conditioning(conditioning, multiplier))




class UC_AdvancedVisualConditioningEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="UC_AdvancedVisualConditioningEncode",
            display_name="Advanced Visual Conditioning Encode",
            category="advanced/conditioning",
            inputs=[
                # --- Primary Inputs ---
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main prompt. With fusion off, image_input_N places active image N inline. With fusion on, use image_input_fusion (image_input_1 is accepted as an alias).",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default="", tooltip="System prompt injected prior to user description."),
                io.Int.Input(
                    "vlm_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Equivalent-square VLM target from 256 to 3584. Values outside that range preserve original resolution.",
                ),

                # --- Modular Configurations ---
                VisualFusionConfig.Input("visual_fusion_config", display_name="Fusion Config", optional=True, tooltip="Optional spatial visual fusion configuration from UC_VisualFusionConfig. Blends isolated visual blocks without coordinate blur."),

                # --- Fallback Controls ---
                io.String.Input(
                    "formula",
                    default="",
                    multiline=False,
                    tooltip="Optional formula used only with fusion off when no numbered inline placeholders are present. Empty selects the first image pass.",
                ),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. Active ONLY if visual_fusion_config is disconnected or set to 'off'.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Boolean.Input("semantic_anchor", default=False, tooltip="Prefixes each encoded visual slot with its numbered <Picture N>: semantic anchor."),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Multimodal images. Maps active inputs sequentially to variables (a, b, c, ...)."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, system_prompt, vlm_resolution, image_inputs: io.Autogrow.Type, visual_fusion_config: dict = None, formula: str = "", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0, vae_dimension_multiple=8, semantic_anchor: bool = False) -> io.NodeOutput:
        # Collect, extract, and parse all active (non-null) connected images sequentially (including batched images)
        _, active_images, _ = extract_and_flatten_images(image_inputs)
        minimax_h3 = is_minimax_h3_text_encoder(clip)
        klein_vl = is_klein_vl_text_encoder(clip)
        if minimax_h3 and ref_latent_mode != "off":
            raise ValueError(
                "MiniMax H3 reference latents require Core's MiniMax H3 reference conditioning node; set ref_latent_mode to off."
            )

        def format_krea_prompt(user_prompt):
            if minimax_h3:
                return format_minimax_h3_prompt(user_prompt, system_prompt)
            if klein_vl and not system_prompt:
                return user_prompt
            if system_prompt or active_images:
                return (
                    "<|im_start|>user\n<|im_end|>\n"
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            return (
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        def add_semantic_anchors(user_prompt, picture_numbers):
            if not semantic_anchor or minimax_h3:
                return user_prompt
            prompt_segments = user_prompt.split(VISION_BLOCK)
            anchored_prompt = [prompt_segments[0]]
            for index, segment in enumerate(prompt_segments[1:]):
                if index < len(picture_numbers):
                    anchored_prompt.append(
                        f"<Picture {picture_numbers[index]}>: {VISION_BLOCK}"
                    )
                else:
                    anchored_prompt.append(VISION_BLOCK)
                anchored_prompt.append(segment)
            return "".join(anchored_prompt)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            logging.warning("AdvancedVisualConditioning: no images are connected; encoding the prompt as text only.")
            clean_prompt, _ = prepare_image_placeholder_prompt(
                prompt,
                image_count=0,
                fusion_active=False,
                context="AdvancedVisualConditioning",
            )
            full_prompt = format_krea_prompt(clean_prompt)
            conditioning = multiply_conditioning(
                encode_embedding_classical_scaled_bias(clip, full_prompt, skip_template=True),
                multiplier,
            )
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        visual_ranges = {}
        visual_grids = {}
        tokens_dict = {}
        reference_cond_dict = None

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "off"}
        visual_method = visual_fusion_config.get("visual_fusion_method", "off")
        visual_encoder_path = visual_fusion_config.get("visual_encoder_path", "grid-deepstack")
        save_visual_embedding = bool(
            visual_fusion_config.get("save_blended_embeds", False)
        )

        processed_active_images = [
            prepare_vlm_image(image, vlm_resolution) for image in active_images
        ]
        prepared_prompt, inline_numbers = prepare_image_placeholder_prompt(
            prompt,
            image_count=len(processed_active_images),
            fusion_active=visual_method != "off",
            context="AdvancedVisualConditioning",
        )
        if klein_vl:
            prepared_prompt = prepared_prompt.replace(VISION_BLOCK, "")

        inline_mode = False
        if visual_method == "off" and inline_numbers:
            inline_images = [processed_active_images[number - 1] for number in inline_numbers]
            inline_prompt = format_krea_prompt(
                add_semantic_anchors(prepared_prompt, inline_numbers)
            )
            if strip_contextual_weight_syntax(inline_prompt) != inline_prompt:
                logging.warning(
                    "AdvancedVisualConditioning: custom contextual vector scaling is disabled for native multi-image inline encoding."
                )
            try:
                with qwen3vl_visual_encoder_path(clip, visual_encoder_path):
                    inline_tokens = (
                        tokenize_minimax_h3_prompt(clip, inline_prompt, inline_images)
                        if minimax_h3
                        else clip.tokenize(
                            inline_prompt, images=inline_images, skip_template=True,
                        )
                    )
                    inline_cond = clip.encode_from_tokens_scheduled(inline_tokens)
                if len(inline_cond) != 1:
                    raise ValueError("Inline image encoding requires a single conditioning schedule entry.")
            except (TypeError, ValueError) as exc:
                logging.warning(
                    "AdvancedVisualConditioning: inline placeholder encoding failed (%s); falling back to per-image formula encoding.",
                    exc,
                )
                prepared_prompt, _ = prepare_image_placeholder_prompt(
                    prompt,
                    image_count=0,
                    fusion_active=False,
                    context="AdvancedVisualConditioning fallback",
                )
            except Exception:
                logging.exception("AdvancedVisualConditioning: inline image encoding failed and cannot be safely recovered.")
                raise
            else:
                inline_mode = True
                sequence_tensors["a"] = inline_cond[0][0]
                pooled_output = inline_cond[0][1].get("pooled_output")
                if pooled_output is not None:
                    pooled_tensors["a"] = pooled_output
                reference_cond_dict = inline_cond[0][1]
                if save_visual_embedding:
                    tokens_dict["a"] = inline_tokens
                    visual_ranges["a"] = find_visual_token_range(
                        inline_tokens,
                        sequence_tensors["a"],
                    )
                if formula.strip() not in {"", "a"}:
                    logging.warning(
                        "AdvancedVisualConditioning: numbered inline placeholders encode one native multimodal sequence; formula '%s' was ignored.",
                        formula,
                    )
                formula = "a"

        multipass_images = [] if inline_mode else active_images
        for idx, source_image in enumerate(multipass_images):
            letter = chr(97 + idx)  # 0 -> 'a', 1 -> 'b', 2 -> 'c', ...

            # Ensure prompt has image pad tokens so tokenizer knows where to inject the image
            modified_prompt = prepared_prompt
            if not klein_vl and not any(tag in modified_prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                modified_prompt = VISION_BLOCK + modified_prompt
            picture_number = 1 if visual_method != "off" else idx + 1
            modified_prompt = add_semantic_anchors(
                modified_prompt, (picture_number,)
            )
            full_prompt = format_krea_prompt(modified_prompt)

            processed_img = prepare_vlm_image(source_image, vlm_resolution)
            tokenize_callback = None
            encode_kwargs = {"images": [processed_img]}
            if not klein_vl:
                encode_kwargs["skip_template"] = True
            if minimax_h3:
                tokenize_callback = lambda text, image=processed_img: (
                    tokenize_minimax_h3_prompt(clip, text, [image])
                )
                encode_kwargs = {}
            cond_X = encode_embedding_classical_scaled_bias(
                clip,
                full_prompt,
                tokenize_callback=tokenize_callback,
                visual_encoder_path=(
                    visual_encoder_path
                    if visual_method != "off"
                    else "grid-deepstack"
                ),
                **encode_kwargs,
            )
            if len(cond_X) != 1:
                raise ValueError(
                    "Advanced visual fusion requires a single conditioning schedule entry."
                )
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output")
            cond_metadata = cond_X[0][1]

            if visual_method != "off" or save_visual_embedding:
                try:
                    tokens = (
                        tokenize_minimax_h3_prompt(clip, full_prompt, [processed_img])
                        if minimax_h3
                        else clip.tokenize(
                            full_prompt, images=[processed_img], skip_template=True
                        )
                    )
                    tokens_dict[letter] = tokens
                    visual_ranges[letter] = find_visual_token_range(
                        tokens,
                        C_X,
                        legacy_krea_spatial=(
                            visual_method != "off"
                            and visual_encoder_path == "legacy-flat"
                        ),
                    )
                    if visual_method != "off":
                        visual_grids[letter] = visual_fusion_grid(
                            processed_img,
                            visual_ranges[letter][1] - visual_ranges[letter][0],
                            visual_encoder_path == "legacy-flat",
                        )
                except Exception as exc:
                    raise ValueError(
                        f"Could not locate the visual token range for image {idx + 1}: {exc}"
                    ) from exc

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            if reference_cond_dict is None:
                reference_cond_dict = cond_metadata

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        fusion_mask_cache = {}
        if visual_method != "off":
            device = comfy.model_management.get_torch_device()

            key_name = "qwen3vl_8b"
            if "tokens" in locals() and tokens:
                key_name = next(iter(tokens.keys()))

            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors,
                pooled_tensors,
                visual_fusion_config=visual_fusion_config,
                device=device,
                visual_ranges=visual_ranges,
                embedding_key=key_name,
                clip=clip,
                tokens_dict=tokens_dict,
                mask_cache=fusion_mask_cache,
                visual_grids=visual_grids,
            )
        else:
            C_blended, P_blended = evaluate_conditioning_formula(formula.strip() or "a", sequence_tensors, pooled_tensors, padding_method=padding_method)
            if save_visual_embedding:
                default_key = "a"
                source_tokens = tokens_dict.get(default_key)
                if source_tokens is None or default_key not in visual_ranges:
                    raise ValueError(
                        "Saving an unfused visual embedding requires the default visual input."
                    )
                save_source_visual_embeddings(
                    clip,
                    source_tokens,
                    visual_fusion_config,
                    next(iter(source_tokens.keys())),
                    comfy.model_management.get_torch_device(),
                )

        # Build final conditioning dictionary
        final_cond_dict = reference_cond_dict.copy()
        attention_mask = final_cond_dict.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != C_blended.shape[1]:
            final_cond_dict.pop("attention_mask", None)
            final_cond_dict.pop("attention_mask_img_shape", None)
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended
        minimax_tags = final_cond_dict.get("minimax_token_tags")
        if minimax_h3 and (
            not torch.is_tensor(minimax_tags)
            or minimax_tags.numel() != C_blended.shape[1]
        ):
            raise ValueError(
                "MiniMax H3 modality tags do not match the fused conditioning sequence length."
            )

        if multiplier != 1.0:
            C_blended *= multiplier
            if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                final_cond_dict["pooled_output"] *= multiplier

        conditioning = [[C_blended, final_cond_dict]]

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process sequentially from active_images
            for i, image in enumerate(active_images):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class TextEncodeEditScaledAdv(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="TextEncodeEditScaledAdv",
            display_name="Text Scaled Encoder (Advanced)",
            category="advanced/conditioning",
            inputs=[
                # --- Primary Inputs ---
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main user text prompt. Supports classical weight syntax: (prompt:weight), e.g. (sunset:1.2).",
                ),
                io.Int.Input(
                    "vlm_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Equivalent-square VLM target from 256 to 3584. Values outside that range preserve original resolution.",
                ),

                # --- Modular Configurations ---
                VisualFusionConfig.Input("visual_fusion_config", display_name="Fusion Config", optional=True, tooltip="Optional spatial visual fusion configuration from UC_VisualFusionConfig. Blends isolated visual blocks without coordinate blur."),

                # --- Fallback Controls ---
                io.String.Input(
                    "formula",
                    default="",
                    multiline=False,
                    tooltip="Optional conditioning formula used only when visual fusion is off. Empty selects the first image pass.",
                ),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. Active ONLY if visual_fusion_config is disconnected or set to 'off'.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Multimodal images. Maps active inputs sequentially to variables (a, b, c, ...)."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vlm_resolution, image_inputs: io.Autogrow.Type, visual_fusion_config: dict = None, formula: str = "", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0, vae_dimension_multiple=8) -> io.NodeOutput:
        # Collect, extract, and parse all active (non-null) connected images sequentially (including batched images)
        _, active_images, _ = extract_and_flatten_images(image_inputs)

        if not active_images:
            # Fallback if no images are connected: encode prompt as plain text
            conditioning = multiply_conditioning(encode_embedding_classical_scaled_bias(clip, prompt), multiplier)
            return io.NodeOutput(conditioning)

        # Map active images sequentially to letter variables (a, b, c, ...) and encode each pass
        sequence_tensors = {}
        pooled_tensors = {}
        visual_ranges = {}
        visual_grids = {}
        tokens_dict = {}
        reference_cond_dict = None

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "off"}
        visual_method = visual_fusion_config.get("visual_fusion_method", "off")
        visual_encoder_path = visual_fusion_config.get("visual_encoder_path", "grid-deepstack")

        for idx, img in enumerate(active_images):
            letter = chr(97 + idx)  # 0 -> 'a', 1 -> 'b', 2 -> 'c', ...

            # Ensure prompt has image pad tokens so tokenizer knows where to inject the image
            modified_prompt = prompt
            if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            processed_img = prepare_vlm_image(img, vlm_resolution)
            cond_X = encode_embedding_classical_scaled_bias(
                clip,
                modified_prompt,
                images=[processed_img],
                visual_encoder_path=(
                    visual_encoder_path
                    if visual_method != "off"
                    else "grid-deepstack"
                ),
            )
            if len(cond_X) != 1:
                raise ValueError(
                    "Advanced visual fusion requires a single conditioning schedule entry."
                )
            C_X = cond_X[0][0]
            P_X = cond_X[0][1].get("pooled_output")
            cond_metadata = cond_X[0][1]

            if visual_method != "off":
                try:
                    tokens = clip.tokenize(
                        modified_prompt, images=[processed_img], skip_template=True
                    )
                    tokens_dict[letter] = tokens
                    visual_ranges[letter] = find_visual_token_range(
                        tokens,
                        C_X,
                        legacy_krea_spatial=visual_encoder_path == "legacy-flat",
                    )
                    visual_grids[letter] = visual_fusion_grid(
                        processed_img,
                        visual_ranges[letter][1] - visual_ranges[letter][0],
                        visual_encoder_path == "legacy-flat",
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Could not locate the visual token range for image {idx + 1}: {exc}"
                    ) from exc

            sequence_tensors[letter] = C_X
            if P_X is not None:
                pooled_tensors[letter] = P_X

            if reference_cond_dict is None:
                reference_cond_dict = cond_metadata

        # Evaluate mathematical formula or consensus on sequence and pooled tensors
        fusion_mask_cache = {}
        if visual_method != "off":
            device = comfy.model_management.get_torch_device()

            key_name = "qwen3vl_8b"
            if "tokens" in locals() and tokens:
                key_name = next(iter(tokens.keys()))

            C_blended, P_blended = evaluate_conditioning_consensus_blend(
                sequence_tensors,
                pooled_tensors,
                visual_fusion_config=visual_fusion_config,
                device=device,
                visual_ranges=visual_ranges,
                embedding_key=key_name,
                clip=clip,
                tokens_dict=tokens_dict,
                mask_cache=fusion_mask_cache,
                visual_grids=visual_grids,
            )
        else:
            C_blended, P_blended = evaluate_conditioning_formula(formula.strip() or "a", sequence_tensors, pooled_tensors, padding_method=padding_method)

        # Build final conditioning dictionary
        final_cond_dict = reference_cond_dict.copy()
        attention_mask = final_cond_dict.get("attention_mask")
        if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != C_blended.shape[1]:
            final_cond_dict.pop("attention_mask", None)
            final_cond_dict.pop("attention_mask_img_shape", None)
        if P_blended is not None:
            final_cond_dict["pooled_output"] = P_blended

        if multiplier != 1.0:
            C_blended *= multiplier
            if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                final_cond_dict["pooled_output"] *= multiplier

        conditioning = [[C_blended, final_cond_dict]]

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process sequentially from active_images
            for i, image in enumerate(active_images):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        return io.NodeOutput(conditioning)


class UC_Krea2InputEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_Krea2InputEmbeds",
            display_name="Krea 2 Input Embeddings",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Input text prompt. Important: skips any template wrapping."),
                io.String.Input("image_paths", multiline=True, default="", placeholder="C:/paths/to/image1.png\nC:/paths/to/image2.png", tooltip="Line-separated list of paths to image files. Must map 1-to-1 with file_names."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_names", multiline=True, default="", placeholder="bulbasaur\nivysaur", tooltip="Line-separated list of file names to save as (without .safetensors). Can include nested subfolders. Must map 1-to-1 with image_paths."),
                io.Boolean.Input(
                    "slice_visual_tokens",
                    default=False,
                    tooltip="If True, removes the first validated visual-token span. If False, preserves the full interleaved sequence.",
                ),
            ],
            outputs=[
                io.AnyType.Output("state_dict", display_name="State Dict", tooltip="Dictionary structure: {'qwen3vl_4b': tensor_2d} of shape [num_tokens, 2560]"),
                io.AnyType.Output("tensor_2d", display_name="Embeddings", tooltip="Raw PyTorch 2D tensor of shape [num_tokens, 2560]"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, image_paths, vlm_resolution, file_names, slice_visual_tokens=False) -> io.NodeOutput:
        # 1. Parse image paths and file names
        img_paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        file_names_list = [n.strip() for n in file_names.split("\n") if n.strip()]

        if not img_paths_list and not file_names_list:
            raise ValueError("Both image_paths and file_names are empty.")

        # If only text prompt encoding is desired (no images)
        if not img_paths_list:
            if not file_names_list:
                raise ValueError("No file_names specified to save the text embedding.")
            img_paths_list = [None] * len(file_names_list)
        elif not file_names_list:
            raise ValueError("No file_names specified for the provided image paths.")

        if len(img_paths_list) != len(file_names_list):
            raise ValueError(f"Count mismatch: Got {len(img_paths_list)} image paths and {len(file_names_list)} file names.")

        # 2. Pre-Execution Path Validation
        for path in img_paths_list:
            if path is not None:
                normalized_path = path.strip().replace('\\', '/')
                normalized_path = os.path.normpath(normalized_path)
                if not os.path.isabs(normalized_path):
                    normalized_path = os.path.abspath(normalized_path)
                if not os.path.isfile(normalized_path):
                    raise FileNotFoundError(
                        f"Validation aborted: Image file does not exist: '{path}' (resolved to: '{normalized_path}'). "
                        "No processing has started, ensuring safe memory state."
                    )

        # 3. Call clip.load_model() once to register the model as active for comfy-aimdo
        clip.load_model()
        cond_stage = clip.cond_stage_model
        clip_model = None

        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        if clip_model is None or not hasattr(clip_model, "process_tokens"):
            raise AttributeError("Could not locate underlying model wrapper with 'process_tokens' method in cond_stage_model.")


        # Locate ComfyUI embeddings directory
        try:
            embed_paths = folder_paths.get_folder_paths("embeddings")
            if embed_paths:
                embeddings_dir = embed_paths[0]
            else:
                embeddings_dir = os.path.join(os.path.dirname(folder_paths.__file__), "models", "embeddings")
        except Exception:
            embeddings_dir = "models/embeddings"

        os.makedirs(embeddings_dir, exist_ok=True)

        last_state_dict = None
        last_tensor_2d = None

        # 4. Process loop under inference_mode
        for img_path, f_name in zip(img_paths_list, file_names_list):
            # Load and preprocess image if present
            images_vl = []
            if img_path is not None:
                image_tensor = load_vlm_image_tensor(img_path)

                # Image resolution downscaling helper
                def process_vlm_image(img, res):
                    VLM_RESOLUTIONS = {
                        "Fast (384)": 384,
                        "Balanced (512)": 512,
                        "Detailed (768)": 768,
                        "Large (1024)": 1024,
                        "X-Large (1280)": 1280,
                        "XX-Large (1536)": 1536
                    }
                    samples = img.movedim(-1, 1)
                    if res == "Original":
                        return img
                    else:
                        vlm_size = VLM_RESOLUTIONS[res]
                        total_vlm = vlm_size * vlm_size
                        scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                        width_vlm = round(samples.shape[3] * scale_by_vlm)
                        height_vlm = round(samples.shape[2] * scale_by_vlm)

                        s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
                        return s_vlm.movedim(1, -1)

                processed_img = process_vlm_image(image_tensor, vlm_resolution)
                images_vl.append(processed_img)

            # Tokenize prompt using skip_template=True so no template wrapping is saved
            modified_prompt = prompt
            if img_path is not None:
                if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Prepend <|im_start|> to trigger skip_template internally
            modified_prompt = "<|im_start|>" + modified_prompt
            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            key_name = next(iter(tokens.keys()))
            token_list = tokens[key_name]
            # Slice off the first token (the <|im_start|> trigger token) to match skip_template behavior
            for i in range(len(token_list)):
                token_list[i] = token_list[i][1:]
            tokens_only = [[t[0] for t in b] for b in token_list]
            device = comfy.model_management.get_torch_device()

            with torch.inference_mode():
                embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

                if slice_visual_tokens:
                    vis_start, vis_end = find_visual_token_range(tokens, embeds)
                    if vis_start < vis_end:
                        prefix = embeds[:, :vis_start, :]
                        suffix = embeds[:, vis_end:, :]
                        embeds_sliced = torch.cat([prefix, suffix], dim=1)
                    else:
                        embeds_sliced = embeds
                else:
                    embeds_sliced = embeds

                tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

            state_dict = {key_name: tensor_2d}

            # Save the safetensors file (ensure nested directories exist)
            target_path = resolve_embedding_output_path(embeddings_dir, f"{f_name}.safetensors")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

            last_state_dict = state_dict
            last_tensor_2d = tensor_2d

            # Clean VRAM loop references
            del embeds, embeds_sliced, tokens, tokens_only
            if img_path is not None:
                del image_tensor, processed_img

        # 5. Final VRAM release and soft_empty_cache
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(last_state_dict, last_tensor_2d)


class UC_Qwen3VLInputEmbeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_Qwen3VLInputEmbeds",
            display_name="Qwen3-VL Unified Input Embeddings",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Input text prompt. Important: skips any template wrapping."),
                io.String.Input("image_paths", multiline=True, default="", placeholder="C:/paths/to/image1.png\nC:/paths/to/image2.png", tooltip="Line-separated list of paths to image files. Must map 1-to-1 with file_names."),
                io.Combo.Input(
                    "vlm_resolution",
                    options=["Fast (384)", "Balanced (512)", "Detailed (768)", "Large (1024)", "X-Large (1280)", "XX-Large (1536)", "Original"],
                    default="Fast (384)",
                    tooltip="Resolution of the image passed to the VLM (semantic path).",
                ),
                io.String.Input("file_names", multiline=True, default="", placeholder="bulbasaur\nivysaur", tooltip="Line-separated list of file names to save as (without .safetensors). Can include nested subfolders. Must map 1-to-1 with image_paths."),
                io.Boolean.Input(
                    "slice_visual_tokens",
                    default=False,
                    tooltip="If True, removes the first validated visual-token span. If False, preserves the full interleaved sequence.",
                ),
            ],
            outputs=[
                io.AnyType.Output("state_dict", display_name="State Dict", tooltip="Dictionary structure: {key_name: tensor_2d} of shape [num_tokens, hidden_size]"),
                io.AnyType.Output("tensor_2d", display_name="Embeddings", tooltip="Raw PyTorch 2D tensor of shape [num_tokens, hidden_size]"),
            ]
        )

    @classmethod
    def execute(cls, clip, prompt, image_paths, vlm_resolution, file_names, slice_visual_tokens=False) -> io.NodeOutput:
        # 1. Parse image paths and file names
        img_paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        file_names_list = [n.strip() for n in file_names.split("\n") if n.strip()]

        if not img_paths_list and not file_names_list:
            raise ValueError("Both image_paths and file_names are empty.")

        # If only text prompt encoding is desired (no images)
        if not img_paths_list:
            if not file_names_list:
                raise ValueError("No file_names specified to save the text embedding.")
            img_paths_list = [None] * len(file_names_list)
        elif not file_names_list:
            raise ValueError("No file_names specified for the provided image paths.")

        if len(img_paths_list) != len(file_names_list):
            raise ValueError(f"Count mismatch: Got {len(img_paths_list)} image paths and {len(file_names_list)} file names.")

        # 2. Pre-Execution Path Validation
        for path in img_paths_list:
            if path is not None:
                normalized_path = path.strip().replace('\\', '/')
                normalized_path = os.path.normpath(normalized_path)
                if not os.path.isabs(normalized_path):
                    normalized_path = os.path.abspath(normalized_path)
                if not os.path.isfile(normalized_path):
                    raise FileNotFoundError(
                        f"Validation aborted: Image file does not exist: '{path}' (resolved to: '{normalized_path}'). "
                        "No processing has started, ensuring safe memory state."
                    )

        # 3. Call clip.load_model() once to register the model as active for comfy-aimdo
        clip.load_model()
        cond_stage = clip.cond_stage_model
        clip_model = None

        if hasattr(cond_stage, "clip") and isinstance(cond_stage.clip, str) and hasattr(cond_stage, cond_stage.clip):
            clip_model = getattr(cond_stage, cond_stage.clip)
        elif hasattr(cond_stage, "clip_model"):
            clip_model = cond_stage.clip_model
        elif hasattr(cond_stage, "clip_d"):
            clip_model = cond_stage.clip_d
        else:
            clip_model = cond_stage

        if clip_model is None or not hasattr(clip_model, "process_tokens"):
            raise AttributeError("Could not locate underlying model wrapper with 'process_tokens' method in cond_stage_model.")


        # Locate ComfyUI embeddings directory
        try:
            embed_paths = folder_paths.get_folder_paths("embeddings")
            if embed_paths:
                embeddings_dir = embed_paths[0]
            else:
                embeddings_dir = os.path.join(os.path.dirname(folder_paths.__file__), "models", "embeddings")
        except Exception:
            embeddings_dir = "models/embeddings"

        os.makedirs(embeddings_dir, exist_ok=True)

        last_state_dict = None
        last_tensor_2d = None

        # 4. Process loop under inference_mode
        for img_path, f_name in zip(img_paths_list, file_names_list):
            # Load and preprocess image if present
            images_vl = []
            if img_path is not None:
                image_tensor = load_vlm_image_tensor(img_path)

                # Image resolution downscaling helper
                def process_vlm_image(img, res):
                    VLM_RESOLUTIONS = {
                        "Fast (384)": 384,
                        "Balanced (512)": 512,
                        "Detailed (768)": 768,
                        "Large (1024)": 1024,
                        "X-Large (1280)": 1280,
                        "XX-Large (1536)": 1536
                    }
                    samples = img.movedim(-1, 1)
                    if res == "Original":
                        return img
                    else:
                        vlm_size = VLM_RESOLUTIONS[res]
                        total_vlm = vlm_size * vlm_size
                        scale_by_vlm = math.sqrt(total_vlm / (samples.shape[3] * samples.shape[2]))
                        width_vlm = round(samples.shape[3] * scale_by_vlm)
                        height_vlm = round(samples.shape[2] * scale_by_vlm)

                        s_vlm = resize_nchw(samples, width_vlm, height_vlm, "bicubic")
                        return s_vlm.movedim(1, -1)

                processed_img = process_vlm_image(image_tensor, vlm_resolution)
                images_vl.append(processed_img)

            # Tokenize prompt using skip_template=True so no template wrapping is saved
            modified_prompt = prompt
            if img_path is not None:
                if not any(tag in prompt for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>"]):
                    modified_prompt = "<|vision_start|><|image_pad|><|vision_end|>" + modified_prompt

            # Prepend <|im_start|> to trigger skip_template internally
            modified_prompt = "<|im_start|>" + modified_prompt
            tokens = clip.tokenize(modified_prompt, images=images_vl, skip_template=True)

            # Retrieve the key name dynamically (typically "qwen3vl_4b" or "qwen3vl_8b")
            key_name = "qwen3vl_8b"
            if tokens:
                key_name = next(iter(tokens.keys()))
            token_list = tokens.get(key_name, [])
            # Slice off the first token (the <|im_start|> trigger token) to match skip_template behavior
            for i in range(len(token_list)):
                token_list[i] = token_list[i][1:]
            tokens_only = [[t[0] for t in b] for b in token_list]
            device = comfy.model_management.get_torch_device()

            with torch.inference_mode():
                embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)

                if slice_visual_tokens:
                    vis_start, vis_end = find_visual_token_range(tokens, embeds)
                    if vis_start < vis_end:
                        prefix = embeds[:, :vis_start, :]
                        suffix = embeds[:, vis_end:, :]
                        embeds_sliced = torch.cat([prefix, suffix], dim=1)
                    else:
                        embeds_sliced = embeds
                else:
                    embeds_sliced = embeds

                tensor_2d = embeds_sliced.squeeze(0).clone().cpu()

            state_dict = {key_name: tensor_2d}

            # Save the safetensors file (ensure nested directories exist)
            target_path = resolve_embedding_output_path(embeddings_dir, f"{f_name}.safetensors")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            state_dict_safe = {k: v.contiguous() for k, v in state_dict.items()}
            save_file(state_dict_safe, target_path)

            last_state_dict = state_dict
            last_tensor_2d = tensor_2d

            # Clean VRAM loop references
            del embeds, embeds_sliced, tokens, tokens_only
            if img_path is not None:
                del image_tensor, processed_img

        # 5. Final VRAM release and soft_empty_cache
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(state_dict, tensor_2d)



_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645

class Krea2WeightPatch:
    def __get__(self, obj, objtype=None):
        return types.MethodType(krea2_attn_forward_weight, obj)

class UC_Krea2TokenAttentionWeight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )
        return io.Schema(
            node_id="UC_Krea2TokenAttentionWeight",
            display_name="Krea2 Token Attention Weight",
            category="advanced/conditioning",
            inputs=[
                # --- Primary Inputs ---
                io.Model.Input("model", tooltip="Diffusion model to apply the attention monkeypatch to."),
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Main prompt. Fusion accepts image_input_fusion or image_input_1 for its single visual slot. Numbered multi-image inline placement is intentionally unavailable in this attention node.",
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default="", tooltip="System prompt injected prior to user description."),
                io.String.Input(
                    "attention_weights",
                    multiline=False,
                    default="",
                    tooltip="Space-separated non-negative attention odds weights. Example: (arms:1.5) (painting:0) (photo:2)",
                ),
                io.Int.Input(
                    "vlm_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Equivalent-square VLM target from 256 to 3584. Values outside that range preserve original resolution.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=4.0, step=0.05, tooltip="Global multiplier on the weighting effect. Effect compounds over all blocks."),

                # --- Modular Configurations ---
                VisualFusionConfig.Input("visual_fusion_config", display_name="Fusion Config", optional=True, tooltip="Optional spatial visual fusion configuration from UC_VisualFusionConfig. Blends isolated visual blocks without coordinate blur."),

                # --- Fallback Controls ---
                io.String.Input(
                    "formula",
                    default="",
                    multiline=False,
                    tooltip="Optional conditioning formula used only when visual fusion is off. Empty selects the first image pass.",
                ),
                io.Combo.Input(
                    "padding_method",
                    options=["zero-pad", "interpolate"],
                    default="zero-pad",
                    tooltip="Alignment method for images with different aspect ratios/resolutions. Active ONLY if visual_fusion_config is disconnected or set to 'off'.",
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                    tooltip="Resolution of the reference latent encoded by the VAE (structural path).",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Reference latent encoding mode. 'single'/'multi' append latents; 'parallel-single'/'parallel-multi' run them in a separate conditioning stream to prevent semantic override.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1, tooltip="Overall multiplier applied to the final conditioning vector."),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True, tooltip="Pixel multiple used to align reference images before VAE encoding."),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Multimodal images. Maps active inputs sequentially to variables (a, b, c, ...)."),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, clip, prompt, system_prompt, attention_weights, image_inputs: io.Autogrow.Type, vlm_resolution: int, visual_fusion_config: dict = None, formula: str = "", padding_method: str = "zero-pad", vae_resolution="Fast (1024)", ref_latent_mode="off", vae=None, multiplier: float = 1.0, strength: float = 1.0, vae_dimension_multiple=8) -> io.NodeOutput:
        # Collect, extract, and parse all active (non-null) connected images sequentially (including batched images)
        _, active_images, _ = extract_and_flatten_images(image_inputs)

        # 1. Parse weights from the attention_weights widget using regex
        pattern = re.compile(r"\(([^():]+):(-?\d*\.?\d+)\)")
        terms = [(m.group(1).strip(), float(m.group(2))) for m in pattern.finditer(attention_weights)]
        if any(not math.isfinite(weight) or weight < 0 for _, weight in terms):
            raise ValueError("Krea2 attention weights must be finite and non-negative.")

        weighted_prompt = prompt
        weighted_system_prompt = system_prompt

        def format_krea_prompt(user_text, system_text):
            if system_text:
                return (
                    "<|im_start|>user\n<|im_end|>\n"
                    f"<|im_start|>system\n{system_text}<|im_end|>\n"
                    f"<|im_start|>user\n{user_text}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            return (
                "<|im_start|>user\n<|im_end|>\n"
                "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        def add_image_marker(user_text):
            if any(tag in user_text for tag in ["<|image_pad|>", "<|image|>", "<|vision_start|>", "image_input_"]):
                return user_text
            return "<|vision_start|><|image_pad|><|vision_end|>" + user_text

        if visual_fusion_config is None:
            visual_fusion_config = {"visual_fusion_method": "off"}
        visual_method = visual_fusion_config.get("visual_fusion_method", "off")
        visual_encoder_path = visual_fusion_config.get("visual_encoder_path", "grid-deepstack")

        if active_images and visual_method != "off":
            weighted_prompt, _ = prepare_image_placeholder_prompt(
                weighted_prompt,
                image_count=len(active_images),
                fusion_active=True,
                context="Krea2TokenAttentionWeight",
            )
        elif re.search(r"\bimage_input_(?:fusion|\d+)\b", weighted_prompt, re.IGNORECASE):
            logging.warning(
                "Krea2TokenAttentionWeight: numbered inline images are not supported because attention positions cannot be mapped safely across multiple visual spans; using the existing per-image path."
            )
            weighted_prompt, _ = prepare_image_placeholder_prompt(
                weighted_prompt,
                image_count=0,
                fusion_active=False,
                context="Krea2TokenAttentionWeight",
            )

        clean_prompt = strip_contextual_weight_syntax(weighted_prompt)
        clean_system_prompt = strip_contextual_weight_syntax(weighted_system_prompt)

        # 2. Get tokens mapping on clean prompt with representative (first) image or fallback
        if active_images:
            first_img = active_images[0]
            processed_first_img = prepare_vlm_image(first_img, vlm_resolution)

            modified_clean_prompt = add_image_marker(clean_prompt)
            modified_weighted_prompt = add_image_marker(weighted_prompt)
            clean_full_prompt = format_krea_prompt(modified_clean_prompt, clean_system_prompt)
            weighted_full_prompt = format_krea_prompt(modified_weighted_prompt, weighted_system_prompt)
            tok = clip.tokenize(clean_full_prompt, images=[processed_first_img], skip_template=True)
        else:
            clean_full_prompt = format_krea_prompt(clean_prompt, clean_system_prompt)
            weighted_full_prompt = format_krea_prompt(weighted_prompt, weighted_system_prompt)
            tok = clip.tokenize(clean_full_prompt, skip_template=True)

        key = next(iter(tok))
        token_list = tok[key][0]
        ids = []
        for t in token_list:
            if isinstance(t, tuple) and len(t) > 0:
                ids.append(t[0])
            elif isinstance(t, dict):
                ids.append(-1)
            else:
                ids.append(t)

        cond = clip.encode_from_tokens_scheduled(tok)
        cond_len = cond[0][0].shape[1]

        mapping = build_token_to_conditioning_map(token_list, cond[0][0])

        weight_pairs = []
        for phrase, w in terms:
            k_bias = math.log(max(w, 1e-6)) * strength
            positions = []
            for variant in (" " + phrase, phrase):
                sub = krea2_token_ids(clip, variant)
                ps, pe = krea2_user_content_span(sub)
                if ps is not None:
                    sub = sub[ps:pe]
                matches = find_subsequence(ids, sub, 0, len(ids))
                if matches:
                    for mi in matches:
                        for off in range(len(sub)):
                            t_idx = mi + off
                            if t_idx < len(mapping):
                                positions.append(mapping[t_idx][0])
                    break
            if not positions:
                logging.warning(f"Krea2PromptWeight: phrase '{phrase}' not found in prompt or system prompt; skipped.")
                continue
            for cp in positions:
                if 0 <= cp < cond_len:
                    weight_pairs.append((cp, k_bias))

        # 3. Patch model
        model_clone = model.clone()
        if weight_pairs:
            logging.info(f"Krea2PromptWeight (Attn): weighting {weight_pairs}")
            diffusion_model = model_clone.get_model_object("diffusion_model")
            transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
            transformer_options["krea2_token_weights"] = weight_pairs
            model_clone.model_options["transformer_options"] = transformer_options

            for idx, block in enumerate(diffusion_model.blocks):
                if hasattr(block, "attn"):
                    patched_attn = Krea2WeightPatch().__get__(block.attn, block.attn.__class__)
                    model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.attn.forward", patched_attn)

        # 4. Multipass encoding and blending
        if active_images:
            sequence_tensors = {}
            pooled_tensors = {}
            visual_ranges = {}
            visual_grids = {}
            tokens_dict = {}
            reference_cond_dict = None
            for idx, img in enumerate(active_images):
                letter = chr(97 + idx)

                clean_pass_prompt = format_krea_prompt(add_image_marker(clean_prompt), clean_system_prompt)
                weighted_pass_prompt = format_krea_prompt(add_image_marker(weighted_prompt), weighted_system_prompt)

                processed_img = prepare_vlm_image(img, vlm_resolution)
                cond_X = encode_embedding_classical_scaled_bias(
                    clip,
                    weighted_pass_prompt,
                    images=[processed_img],
                    skip_template=True,
                    visual_encoder_path=visual_encoder_path,
                )
                if len(cond_X) != 1:
                    raise ValueError(
                        "Krea2 attention visual fusion requires a single conditioning schedule entry."
                    )
                C_X = cond_X[0][0]
                P_X = cond_X[0][1].get("pooled_output")
                cond_metadata = cond_X[0][1]

                if visual_method != "off":
                    try:
                        tokens = clip.tokenize(
                            clean_pass_prompt,
                            images=[processed_img],
                            skip_template=True,
                        )
                        tokens_dict[letter] = tokens
                        visual_ranges[letter] = find_visual_token_range(
                            tokens,
                            C_X,
                            legacy_krea_spatial=visual_encoder_path == "legacy-flat",
                        )
                        visual_grids[letter] = visual_fusion_grid(
                            processed_img,
                            visual_ranges[letter][1] - visual_ranges[letter][0],
                            visual_encoder_path == "legacy-flat",
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"Could not locate the visual token range for image {idx + 1}: {exc}"
                        ) from exc

                sequence_tensors[letter] = C_X
                if P_X is not None:
                    pooled_tensors[letter] = P_X

                if reference_cond_dict is None:
                    reference_cond_dict = cond_metadata

            fusion_mask_cache = {}
            if visual_method != "off":
                device = comfy.model_management.get_torch_device()

                key_name = "qwen3vl_8b"
                if "tokens" in locals() and tokens:
                    key_name = next(iter(tokens.keys()))

                C_blended, P_blended = evaluate_conditioning_consensus_blend(
                    sequence_tensors,
                    pooled_tensors,
                    visual_fusion_config=visual_fusion_config,
                    device=device,
                    visual_ranges=visual_ranges,
                    embedding_key=key_name,
                    clip=clip,
                    tokens_dict=tokens_dict,
                    mask_cache=fusion_mask_cache,
                    visual_grids=visual_grids,
                )
            else:
                C_blended, P_blended = evaluate_conditioning_formula(formula.strip() or "a", sequence_tensors, pooled_tensors, padding_method=padding_method)

            # Build final conditioning dictionary
            final_cond_dict = reference_cond_dict.copy()
            attention_mask = final_cond_dict.get("attention_mask")
            if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != C_blended.shape[1]:
                final_cond_dict.pop("attention_mask", None)
                final_cond_dict.pop("attention_mask_img_shape", None)
            if P_blended is not None:
                final_cond_dict["pooled_output"] = P_blended

            if multiplier != 1.0:
                C_blended *= multiplier
                if "pooled_output" in final_cond_dict and final_cond_dict["pooled_output"] is not None:
                    final_cond_dict["pooled_output"] *= multiplier

            conditioning = [[C_blended, final_cond_dict]]
        else:
            conditioning = encode_embedding_classical_scaled_bias(clip, weighted_full_prompt, skip_template=True)
            if multiplier != 1.0:
                for i in range(len(conditioning)):
                    conditioning[i][0] *= multiplier
                    if "pooled_output" in conditioning[i][1] and conditioning[i][1]["pooled_output"] is not None:
                        conditioning[i][1]["pooled_output"] *= multiplier

        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            VAE_RESOLUTIONS = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536
            }
            # Process sequentially from active_images
            for i, image in enumerate(active_images):
                if "single" in ref_latent_mode and len(ref_latents) > 0:
                    break
                if image is not None:
                    samples = image.movedim(-1, 1)
                    target_size = None if vae_resolution == "Original" else VAE_RESOLUTIONS[vae_resolution]
                    s_vae = prepare_vae_reference_image(samples, target_size, vae_dimension_multiple)
                    ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))

        conditioning = apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)

        return io.NodeOutput(model_clone, conditioning)


# Canonical 0.10 node IDs. Compatibility classes below remain registered for
# one release and delegate to the same execution implementations.
class UC_TextEncodeSystemEditAdvanced(TextEncodeSystemEditPlusAdvanced):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_TextEncodeSystemEditAdvanced"
        schema.display_name = "System Edit Text Encode (Advanced)"
        schema.is_deprecated = False
        return schema


class UC_TextEncodeGemmaSystemEditAdvanced(TextEncodeGemmaSystemEditPlusAdvanced):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_TextEncodeGemmaSystemEditAdvanced"
        schema.display_name = "Gemma System Edit Text Encode (Advanced)"
        schema.is_deprecated = False
        return schema


class TextEncodeKrea2SystemEditScaledAdv(UC_AdvancedVisualConditioningEncode):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "TextEncodeKrea2SystemEditScaledAdv"
        schema.display_name = "Krea2 System Prompt Scaled Encoder (Advanced)"
        schema.is_deprecated = True
        return schema


class UC_MiniMaxH3MediaConfig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MiniMaxH3MediaConfig", display_name="MiniMax H3 Media Configurator",
            category="advanced/conditioning", is_input_list=True,
            description="Sets Picture timestamp syntax, Qwen Video sampling, and Video motion-guidance memory use for the Advanced MiniMax H3 nodes.",
            inputs=[
                io.AnyType.Input("timestamps", optional=True, tooltip="Optional sequential timestamps for existing Picture slots. Leave disconnected to keep the default Core Picture presentation."),
                io.Combo.Input("timestamp_format", options=list(VIDEO_FRAME_TIMESTAMP_FORMATS), default="0.0s", tooltip="Formatting used when the Picture structure contains <<time>>."),
                io.String.Input("structure", multiline=True, dynamic_prompts=False, default=MINIMAX_H3_MEDIA_STRUCTURE, tooltip="Picture constructor using required <<picture>> and <<visual>> tags. Default matches Core: <<picture>>: <<visual>>. Timestamped example: At <<time>>, <<picture>>: <<visual>> (from <<shot>>) is fully anchored."),
                io.Int.Input("video_fps", default=2, min=1, max=24, step=1, tooltip="VLM presentation sampling rate for the 24 fps Video input. Latent video usage is unchanged."),
                io.Combo.Input(
                    "video_latent_mode",
                    options=list(MINIMAX_H3_VIDEO_LATENT_MODES),
                    default="even keyframes",
                    tooltip="Controls Video motion guidance. Full Video uses the most sampling memory. Even keyframes keep spaced points from start to end with less sampling memory. Off sends Video only to Qwen.",
                ),
                io.Int.Input(
                    "video_latent_keyframes",
                    default=4,
                    min=2,
                    max=213,
                    step=1,
                    tooltip="Number of evenly spaced Video points kept from beginning through end in even keyframes mode. Lower values use less sampling memory.",
                ),
                io.Int.Input("temporal_density", default=1, min=1, max=24, step=1, tooltip="Offset sample density used only by the experimental temporal encoders."),
                io.Combo.Input("temporal_fusion_method", options=["consensus", "spatial"], default="consensus", tooltip="Video-block fusion used only by the experimental temporal encoders."),
            ],
            outputs=[MiniMaxH3MediaConfig.Output(display_name="media_config", tooltip="Runtime media configuration for the Advanced MiniMax H3 encoder nodes.")],
        )

    @classmethod
    def execute(
        cls,
        timestamps=None,
        timestamp_format="0.0s",
        structure=MINIMAX_H3_MEDIA_STRUCTURE,
        video_fps=2,
        video_latent_mode="even keyframes",
        video_latent_keyframes=4,
        temporal_density=1,
        temporal_fusion_method="consensus",
    ):
        return io.NodeOutput(build_minimax_h3_media_config(
            timestamps,
            timestamp_format,
            structure,
            video_fps,
            video_latent_mode,
            video_latent_keyframes,
            temporal_density,
            temporal_fusion_method,
        ))


class UC_MiniMaxH3VLMGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MiniMaxH3VLMGuide", display_name="MiniMax H3 VLM Guide",
            category="advanced/conditioning", is_experimental=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Clip.Input("clip"),
                io.Image.Input("image"),
                io.Float.Input("timestamp", default=0.0, min=0.0, step=0.1, tooltip="Guide time in seconds."),
                io.Int.Input("vlm_resolution", default=384, min=0, max=4096, step=32, tooltip="Equivalent-square Qwen target from 256 to 3584. Values outside that range preserve original resolution."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning, clip, image, timestamp, vlm_resolution=384):
        return io.NodeOutput(execute_minimax_h3_vlm_guide(conditioning, clip, image, timestamp, vlm_resolution))


class UC_AdvancedMiniMaxH3ImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        reference_template = io.Autogrow.TemplateNames(
            io.Image.Input(
                "reference_image",
                tooltip="Ordered native H3 image reference and separately numbered Qwen picture.",
            ),
            names=[f"reference_image_{index}" for index in range(1, 33)],
            min=0,
        )
        fusion_template = io.Autogrow.TemplateNames(
            io.Image.Input(
                "fusion_image",
                tooltip="Ordered Qwen-only visual source. Active fusion combines only this group.",
            ),
            names=[f"fusion_image_{index}" for index in range(1, 33)],
            min=0,
        )
        return io.Schema(
            node_id="UC_AdvancedMiniMaxH3ImageToVideo",
            display_name="Advanced MiniMax H3 Image to Video",
            category="advanced/conditioning",
            description=(
                "Creates coordinated MiniMax H3 Qwen conditioning from independent frame, reference, and fusion "
                "inputs together with native visual controls and the matching joint video/audio latent. Optional media "
                    "configuration controls optional Picture timestamp syntax, Qwen Video sampling, and Video motion guidance."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="MiniMax H3 Qwen3-VL 32B text encoder (qwen3vl_32b).",
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Encodes first/last frames, reference images, and a complete Video. Not required when reference image size is none.",
                ),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Optional frame-zero VAE anchor and the first numbered Qwen picture.",
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Optional final-frame VAE anchor and the next numbered Qwen picture.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="Raw MiniMax H3 prompt. Picture labels are supplied by Core from the connected image roles.",
                ),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Frame count at 24 fps, snapped upward to MiniMax H3's 17k+5 temporal grid.",
                ),
                VisualFusionConfig.Input(
                    "visual_fusion_config",
                    display_name="Fusion Config",
                    optional=True,
                    tooltip=(
                        "Optional spatial method. With frame inputs, fusion_image_1 targets Picture 1 and "
                        "fusion_image_2 targets Picture 2; disconnected or off keeps fusion images as separate "
                        "numbered Qwen pictures outside native-reference mode. See fusion_images for the complete routing contract."
                    ),
                ),
                io.Float.Input(
                    "multiplier",
                    default=1.0,
                    min=-1000.0,
                    max=1000.0,
                    step=0.1,
                    tooltip="Scales the final Qwen conditioning and pooled output; does not alter VAE keyframes or the H3 latent.",
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max", "none"],
                    default="match",
                    tooltip=(
                        "Match limits each native reference to the generation pixel area; max limits its short edge to "
                        "2048 pixels. None keeps frame, reference, and Video inputs available to the text encoder but does not "
                        "VAE-encode them. All image sizing preserves aspect ratio; final 32-pixel "
                        "alignment can marginally enlarge a dimension."
                    ),
                ),
                io.Int.Input(
                    "vlm_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip=(
                        "Equivalent-square Qwen3-VL target from 256 to 3584. Values outside that range preserve "
                        "the original image resolution. This is independent of VAE frame and reference sizing."
                    ),
                ),
                io.Int.Input(
                    "vlm_video_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip=(
                        "Qwen3-VL resolution for Video frames. Higher values use more visual tokens. "
                        "Values outside 256 to 3584 preserve the input resolution."
                    ),
                ),
                io.Autogrow.Input(
                    "reference_images",
                    template=reference_template,
                    optional=True,
                    tooltip=(
                        "Ordered native H3 references and numbered Qwen pictures. This mode cannot be combined with "
                        "explicit first/last frame inputs. See fusion_images for supported reference-picture fusion."
                    ),
                ),
                io.Autogrow.Input(
                    "fusion_images",
                    template=fusion_template,
                    optional=True,
                    tooltip=(
                        "Qwen-only fusion contract. Active method: with frames, socket N targets Picture N and every batch "
                        "item is another source; an unmatched socket errors. With native references, one image on "
                        "fusion_image_1 broadcasts to every reference Picture; otherwise flattened fusion images pair by "
                        "index and extras beyond the reference count are ignored. Without frames or references, all fusion "
                        "images combine into Picture 1. Method off keeps them as separate Pictures, except native-reference "
                        "mode ignores them. Video blocks are never fusion targets."
                    ),
                ),
                MiniMaxH3MediaConfig.Input(
                    "media_config", optional=True,
                    tooltip="Optionally formats Picture timestamps, sets Qwen Video sampling, and controls Video motion guidance. Its default Picture constructor matches Core handling.",
                ),
                io.Image.Input("video", optional=True, tooltip="Complete Video frame batch at 24 fps. The configurator controls Qwen sampling and full, spaced, or disabled VAE motion guidance."),
                io.Audio.Input("audio", optional=True, tooltip="Optional H3 reference audio. Missing audio from a video is ignored."),
                io.Vae.Input("audio_vae", optional=True, lazy=True, tooltip="Required only when audio is present. Skipped when audio is absent; otherwise resamples and encodes the reference audio."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def check_lazy_status(cls, audio=None, audio_vae=None, **kwargs):
        return ["audio_vae"] if audio is not None and audio_vae is None else []

    @classmethod
    def execute(
        cls,
        clip,
        vae=None,
        prompt=None,
        width=None,
        height=None,
        length=None,
        first_frame=None,
        last_frame=None,
        reference_images: io.Autogrow.Type = None,
        fusion_images: io.Autogrow.Type = None,
        visual_fusion_config=None,
        multiplier=1.0,
        ref_image_size="match",
        vlm_resolution=384,
        vlm_video_resolution=384,
        media_config=None,
        video=None,
        audio=None,
        audio_vae=None,
    ) -> io.NodeOutput:
        conditioning, latent = execute_advanced_minimax_h3_image_to_video(
            clip,
            vae,
            prompt,
            width,
            height,
            length,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            fusion_images=fusion_images,
            visual_fusion_config=visual_fusion_config,
            multiplier=multiplier,
            ref_image_size=ref_image_size,
            vlm_resolution=vlm_resolution,
            vlm_video_resolution=vlm_video_resolution,
            media_config=media_config,
            video=video,
            audio=audio,
            audio_vae=audio_vae,
        )
        return io.NodeOutput(conditioning, latent)


class UC_AdvancedVisConEncoder(io.ComfyNode):
    NODE_ID = "UC_AdvancedVisConEncoder"
    DISPLAY_NAME = "Advanced Visual Consensus Encoder"
    TOKEN_FUSION = False
    CONFIG_TOOLTIP = "Required joint configuration. Spatial fusion completes independently at every resolution before complete-conditioning consensus."

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16,
        )
        return io.Schema(
            node_id=cls.NODE_ID,
            display_name=cls.DISPLAY_NAME,
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP/T5 dual text encoder reference."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Int.Input(
                    "vlm_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Equivalent-square VLM target. Values outside 256-3584 preserve Original resolution.",
                ),
                VisualConsensusConfig.Input(
                    "visual_consensus_config",
                    display_name="Visual Consensus Configuration",
                    tooltip=cls.CONFIG_TOOLTIP,
                ),
                io.Combo.Input(
                    "vae_resolution",
                    options=["Ultra (512)", "Turbo (768)", "Fast (1024)", "Balanced (1280)", "Detailed (1536)", "Original"],
                    default="Fast (1024)",
                ),
                io.Combo.Input(
                    "ref_latent_mode",
                    options=["off", "single", "multi", "parallel-single", "parallel-multi"],
                    default="off",
                    tooltip="Generic reference-latent mode. MiniMax H3 requires off and uses Core's dedicated H3 reference conditioning instead.",
                ),
                io.Vae.Input("vae", optional=True),
                io.Float.Input("multiplier", default=1.0, min=-1000.0, max=1000.0, step=0.1),
                io.Int.Input("vae_dimension_multiple", default=8, min=4, max=256, step=4, advanced=True),
                io.Boolean.Input("semantic_anchor", default=False, tooltip="Prefixes each encoded visual slot with its numbered <Picture N>: semantic anchor."),
                io.Autogrow.Input(
                    "image_inputs",
                    template=autogrow_template,
                    tooltip="One batched socket equals separate visual sources. Multiple batched sockets form index-aligned lanes; singleton sockets broadcast.",
                ),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        system_prompt,
        vlm_resolution,
        visual_consensus_config,
        image_inputs: io.Autogrow.Type,
        vae_resolution="Fast (1024)",
        ref_latent_mode="off",
        vae=None,
        multiplier=1.0,
        vae_dimension_multiple=8,
        semantic_anchor=False,
    ) -> io.NodeOutput:
        conditioning = execute_advanced_visual_consensus(
            clip,
            prompt,
            system_prompt,
            vlm_resolution,
            image_inputs,
            visual_consensus_config,
            vae_resolution,
            ref_latent_mode,
            vae,
            multiplier,
            vae_dimension_multiple,
            apply_parallel_ref_latents,
            token_fusion=cls.TOKEN_FUSION,
            semantic_anchor=semantic_anchor,
        )
        return io.NodeOutput(conditioning)


class UC_AdvancedVisConEncoderTokenFusion(UC_AdvancedVisConEncoder):
    NODE_ID = "UC_AdvancedVisConEncoderTokenFusion"
    DISPLAY_NAME = "Advanced Visual Consensus Encoder (TokenFusion)"
    TOKEN_FUSION = True
    CONFIG_TOOLTIP = "Required joint configuration. At each resolution, TokenFusion fuses visual and DeepStack tokens before one conditioning encode; complete-conditioning consensus then combines the encoded resolution samples."


class UC_VLMInputEmbeds(UC_Qwen3VLInputEmbeds):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_VLMInputEmbeds"
        schema.display_name = "VLM Input Embedding Export"
        schema.is_deprecated = False
        return schema


class TextEncodeKrea2SysEditScaledAdvAttn(UC_Krea2TokenAttentionWeight):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "TextEncodeKrea2SysEditScaledAdvAttn"
        schema.display_name = "Krea2 System Prompt Scaled Attention Encoder (Advanced)"
        schema.is_deprecated = True
        return schema


def _mark_deprecated_node(node_class):
    original = node_class.define_schema.__func__

    @classmethod
    def deprecated_schema(cls):
        schema = original(cls)
        schema.is_deprecated = True
        return schema

    node_class.define_schema = deprecated_schema


for _deprecated_node in (
    UC_TextEncodeFlux2SystemPrompt,
    UC_TextEncodeKleinSystemPrompt,
    UC_TextEncodeKrea2SystemPrompt,
    UC_TextEncodeZITSystemPrompt,
    UC_TextEncodeZImageThinkPrompt,
    UC_ScaledBiasTextEncodeFlux2SystemPrompt,
    UC_ScaledBiasTextEncodeKleinSystemPrompt,
    UC_ScaledBiasTextEncodeLtxv2SystemPrompt,
    UC_ScaledBiasTextEncodeZITSystemPrompt,
    UC_ScaledBiasTextEncodeZImageThinkPrompt,
    UC_ScaledBiasTextEncodeSystemPrompt,
    TextEncodeSystemEditPlus,
    TextEncodeSystemEditPlusAdvanced,
    TextEncodeKrea2SystemEditPlusAdvanced,
    TextEncodeEditPlusAdvanced,
    TextEncodeKrea2SystemEditScaledAdv,
    TextEncodeEditScaledAdv,
    TextEncodeGemmaSystemEditPlusAdvanced,
    UC_Krea2InputEmbeds,
    UC_Qwen3VLInputEmbeds,
    TextEncodeKrea2SysEditScaledAdvAttn,
):
    _mark_deprecated_node(_deprecated_node)


class _TokenFusionConditioningNode(io.ComfyNode):
    BASE_NODE = None
    NODE_ID = ""
    DISPLAY_NAME = ""
    USE_SYSTEM_PROMPT = False
    DEPRECATED = False

    @classmethod
    def define_schema(cls):
        schema = cls.BASE_NODE.define_schema()
        schema.node_id = cls.NODE_ID
        schema.display_name = cls.DISPLAY_NAME
        schema.is_deprecated = cls.DEPRECATED
        return schema

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        vlm_resolution,
        image_inputs,
        visual_fusion_config=None,
        formula="",
        padding_method="zero-pad",
        vae_resolution="Fast (1024)",
        ref_latent_mode="off",
        vae=None,
        multiplier=1.0,
        vae_dimension_multiple=8,
        semantic_anchor=False,
        system_prompt="",
    ):
        config = visual_fusion_config or {"visual_fusion_method": "off"}
        if config.get("visual_fusion_method", "off") == "off":
            kwargs = dict(
                clip=clip,
                prompt=prompt,
                vlm_resolution=vlm_resolution,
                image_inputs=image_inputs,
                visual_fusion_config=visual_fusion_config,
                formula=formula,
                padding_method=padding_method,
                vae_resolution=vae_resolution,
                ref_latent_mode=ref_latent_mode,
                vae=vae,
                multiplier=multiplier,
                vae_dimension_multiple=vae_dimension_multiple,
                semantic_anchor=semantic_anchor,
            )
            if cls.USE_SYSTEM_PROMPT:
                kwargs["system_prompt"] = system_prompt
            return cls.BASE_NODE.execute(**kwargs)

        _, active_images, _ = extract_and_flatten_images(image_inputs)
        if not active_images:
            kwargs = dict(
                clip=clip,
                prompt=prompt,
                vlm_resolution=vlm_resolution,
                image_inputs=image_inputs,
                visual_fusion_config=visual_fusion_config,
                formula=formula,
                padding_method=padding_method,
                vae_resolution=vae_resolution,
                ref_latent_mode=ref_latent_mode,
                vae=vae,
                multiplier=multiplier,
                vae_dimension_multiple=vae_dimension_multiple,
                semantic_anchor=semantic_anchor,
            )
            if cls.USE_SYSTEM_PROMPT:
                kwargs["system_prompt"] = system_prompt
            return cls.BASE_NODE.execute(**kwargs)
        if is_minimax_h3_text_encoder(clip) and ref_latent_mode != "off":
            raise ValueError(
                "MiniMax H3 reference latents require Core's MiniMax H3 reference conditioning node; set ref_latent_mode to off."
            )
        conditioning, _ = execute_token_fusion_visual_conditioning(
            clip,
            prompt,
            active_images,
            config,
            vlm_resolution,
            system_prompt if cls.USE_SYSTEM_PROMPT else None,
            multiplier,
            cls.USE_SYSTEM_PROMPT,
        )
        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            resolutions = {
                "Ultra (512)": 512,
                "Turbo (768)": 768,
                "Fast (1024)": 1024,
                "Balanced (1280)": 1280,
                "Detailed (1536)": 1536,
            }
            for image in active_images:
                if "single" in ref_latent_mode and ref_latents:
                    break
                samples = image.movedim(-1, 1)
                target = None if vae_resolution == "Original" else resolutions[vae_resolution]
                prepared = prepare_vae_reference_image(samples, target, vae_dimension_multiple)
                ref_latents.append(vae.encode(prepared.movedim(1, -1)[:, :, :, :3]))
        return io.NodeOutput(
            apply_parallel_ref_latents(clip, conditioning, ref_latents, ref_latent_mode)
        )


class UC_AdvancedVisualConditioningEncodeTokenFusion(_TokenFusionConditioningNode):
    BASE_NODE = UC_AdvancedVisualConditioningEncode
    NODE_ID = "UC_AdvancedVisualConditioningEncodeTokenFusion"
    DISPLAY_NAME = "Advanced Visual Conditioning Encode (TokenFusion)"
    USE_SYSTEM_PROMPT = True


class UC_Krea2TokenAttentionWeightTokenFusion(UC_Krea2TokenAttentionWeight):
    NODE_ID = "UC_Krea2TokenAttentionWeightTokenFusion"
    DISPLAY_NAME = "Krea2 Token Attention Weight (TokenFusion)"
    DEPRECATED = False

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = cls.NODE_ID
        schema.display_name = cls.DISPLAY_NAME
        schema.is_deprecated = cls.DEPRECATED
        schema.is_experimental = True
        return schema

    @classmethod
    def execute(
        cls, model, clip, prompt, system_prompt, attention_weights, image_inputs,
        vlm_resolution, visual_fusion_config=None, formula="",
        padding_method="zero-pad", vae_resolution="Fast (1024)",
        ref_latent_mode="off", vae=None, multiplier=1.0, strength=1.0,
        vae_dimension_multiple=8,
    ):
        config = visual_fusion_config or {"visual_fusion_method": "off"}
        if config.get("visual_fusion_method", "off") == "off":
            return super().execute(
                model, clip, prompt, system_prompt, attention_weights, image_inputs,
                vlm_resolution, visual_fusion_config, formula, padding_method,
                vae_resolution, ref_latent_mode, vae, multiplier, strength,
                vae_dimension_multiple,
            )
        _, active_images, _ = extract_and_flatten_images(image_inputs)
        if not active_images:
            return super().execute(
                model, clip, prompt, system_prompt, attention_weights, image_inputs,
                vlm_resolution, visual_fusion_config, formula, padding_method,
                vae_resolution, ref_latent_mode, vae, multiplier, strength,
                vae_dimension_multiple,
            )
        terms = [
            (match.group(1).strip(), float(match.group(2)))
            for match in re.finditer(r"\(([^():]+):(-?\d*\.?\d+)\)", attention_weights)
        ]
        if any(not math.isfinite(weight) or weight < 0 for _, weight in terms):
            raise ValueError("Krea2 attention weights must be finite and non-negative.")
        conditioning, token_sources = execute_token_fusion_visual_conditioning(
            clip, prompt, active_images, config, vlm_resolution, system_prompt, multiplier, True
        )
        token_list = token_sources[0][next(iter(token_sources[0]))][0]
        ids = [
            -1 if isinstance(token, dict)
            else int(token[0] if isinstance(token, tuple) else token)
            for token in token_list
        ]
        mapping = build_token_to_conditioning_map(token_list, conditioning[0][0])
        weight_pairs = []
        for phrase, weight in terms:
            bias = math.log(max(weight, 1e-6)) * strength
            for variant in (" " + phrase, phrase):
                sub = krea2_token_ids(clip, variant)
                start, end = krea2_user_content_span(sub)
                if start is not None:
                    sub = sub[start:end]
                matches = find_subsequence(ids, sub, 0, len(ids))
                if matches:
                    for match in matches:
                        for offset in range(len(sub)):
                            index = match + offset
                            if index < len(mapping) and mapping[index][0] >= 0:
                                weight_pairs.append((mapping[index][0], bias))
                    break
        model_clone = model.clone()
        if weight_pairs:
            diffusion_model = model_clone.get_model_object("diffusion_model")
            transformer_options = model_clone.model_options.get("transformer_options", {}).copy()
            transformer_options["krea2_token_weights"] = weight_pairs
            model_clone.model_options["transformer_options"] = transformer_options
            for index, block in enumerate(diffusion_model.blocks):
                if hasattr(block, "attn"):
                    patched = Krea2WeightPatch().__get__(block.attn, block.attn.__class__)
                    model_clone.add_object_patch(f"diffusion_model.blocks.{index}.attn.forward", patched)
        ref_latents = []
        if vae is not None and ref_latent_mode != "off":
            resolutions = {
                "Ultra (512)": 512, "Turbo (768)": 768, "Fast (1024)": 1024,
                "Balanced (1280)": 1280, "Detailed (1536)": 1536,
            }
            for image in active_images:
                if "single" in ref_latent_mode and ref_latents:
                    break
                samples = image.movedim(-1, 1)
                target = None if vae_resolution == "Original" else resolutions[vae_resolution]
                prepared = prepare_vae_reference_image(samples, target, vae_dimension_multiple)
                ref_latents.append(vae.encode(prepared.movedim(1, -1)[:, :, :, :3]))
        conditioning = apply_parallel_ref_latents(
            clip, conditioning, ref_latents, ref_latent_mode
        )
        return io.NodeOutput(model_clone, conditioning)


class UC_AdvMiniMaxH3ImageToVideoTokenFusion(UC_AdvancedMiniMaxH3ImageToVideo):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_AdvMiniMaxH3ImageToVideoTokenFusion"
        schema.display_name = "Adv MiniMax H3 Image to Video (TokenFusion)"
        return schema

    @classmethod
    def execute(
        cls, clip, vae=None, prompt=None, width=None, height=None, length=None, first_frame=None,
        last_frame=None, reference_images=None, fusion_images=None,
        visual_fusion_config=None, multiplier=1.0, ref_image_size="match",
        vlm_resolution=384, vlm_video_resolution=384, media_config=None,
        video=None, audio=None, audio_vae=None,
    ):
        conditioning, latent = execute_advanced_minimax_h3_image_to_video(
            clip, vae, prompt, width, height, length,
            first_frame=first_frame, last_frame=last_frame,
            reference_images=reference_images, fusion_images=fusion_images,
            visual_fusion_config=visual_fusion_config, multiplier=multiplier,
            ref_image_size=ref_image_size, vlm_resolution=vlm_resolution,
            vlm_video_resolution=vlm_video_resolution,
            media_config=media_config, video=video, audio=audio,
            audio_vae=audio_vae, token_fusion=True,
        )
        return io.NodeOutput(conditioning, latent)


class UC_AdvMiniMaxH3ImageToVideoTemporalFusion(UC_AdvancedMiniMaxH3ImageToVideo):
    _EXPERIMENTAL = None
    TEMPORAL_TOKEN_FUSION = False

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_AdvMiniMaxH3ImageToVideoTemporalFusion"
        schema.is_experimental = True
        schema.display_name = "Adv MiniMax H3 Image to Video (Temporal Fusion)"
        schema.description = "Experimentally fuses corresponding video visual blocks after separate Qwen encodes, preserving the ordinary video token budget."
        schema.inputs = [value for value in schema.inputs if value.id != "fusion_images"]
        schema.inputs.append(TextBlendConfig.Input("text_blend_config", optional=True, tooltip="Temporal consensus settings. Disconnected uses custom index consensus with norm rescaling."))
        return schema

    @classmethod
    def execute(
        cls, clip, vae=None, prompt=None, width=None, height=None, length=None,
        first_frame=None, last_frame=None, reference_images=None,
        visual_fusion_config=None, multiplier=1.0, ref_image_size="match",
        vlm_resolution=384, vlm_video_resolution=384, media_config=None,
        video=None, audio=None, audio_vae=None, text_blend_config=None,
    ):
        conditioning, latent = execute_advanced_minimax_h3_image_to_video(
            clip, vae, prompt, width, height, length,
            first_frame=first_frame, last_frame=last_frame, reference_images=reference_images,
            visual_fusion_config=visual_fusion_config, multiplier=multiplier,
            ref_image_size=ref_image_size, vlm_resolution=vlm_resolution,
            vlm_video_resolution=vlm_video_resolution, media_config=media_config,
            video=video, audio=audio, audio_vae=audio_vae,
            temporal_fusion=True, temporal_token_fusion=cls.TEMPORAL_TOKEN_FUSION,
            text_blend_config=text_blend_config,
        )
        return io.NodeOutput(conditioning, latent)


class UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion(UC_AdvMiniMaxH3ImageToVideoTemporalFusion):
    TEMPORAL_TOKEN_FUSION = True

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion"
        schema.display_name = "Adv MiniMax H3 Image to Video (Temporal TokenFusion)"
        schema.description = "Experimentally fuses corresponding video features and DeepStack before one Qwen encode per schedule, preserving the ordinary video token budget."
        return schema


