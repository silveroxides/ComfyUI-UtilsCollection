import re
import logging
import torch
from contextlib import nullcontext
from enum import Enum

from comfy_api.latest import io
import comfy.ops
from comfy import model_management
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids

from ..helpers.encoder_helpers import evaluate_tensor_expression, fuse_visual_token_sources, fuse_deepstack_layers, prepare_vlm_image
from ..helpers.textgen_helpers import (
    _qwen_execution_context,
    conform_and_validate_response_schema,
    generate_qwen3vl_video,
    prepare_qwen3vl_video_frames,
    qwen3vl_video_prompt,
    schema_guided_generation_context,
)

VisualFusionConfig = io.Custom("VISUAL_FUSION_CONFIG")
ResponseSchema = io.Custom("GEMINI_RESPONSE_SCHEMA")

# -----------------------------------------------------------------------------
# 1. Helper classes, mappings and functions
# -----------------------------------------------------------------------------

class ImageInputMapping(Enum):
    ZERO_INDEXED_OFFSET = 1
    ONE_INDEXED_OFFSET = 0

    @classmethod
    def get_display_num(cls, num, is_zero_indexed):
        offset = cls.ZERO_INDEXED_OFFSET.value if is_zero_indexed else cls.ONE_INDEXED_OFFSET.value
        return num + offset

    @classmethod
    def get_display_name(cls, num, is_zero_indexed):
        return f"image_input_{cls.get_display_num(num, is_zero_indexed)}"

    @classmethod
    def get_dict_key(cls, num, is_zero_indexed):
        offset = cls.ZERO_INDEXED_OFFSET.value if is_zero_indexed else cls.ONE_INDEXED_OFFSET.value
        return num - offset

def evaluate_formula(expression: str, processed_images: dict) -> torch.Tensor:
    try:
        result = evaluate_tensor_expression(expression, processed_images)
        return torch.clamp(result, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Error evaluating textgen visual math expression '|{expression}|': {e}") from e

def _aligned_image_values(images):
    """Return temporary common-sized copies for pixel arithmetic only."""
    if not images:
        return images
    height = max(value.shape[1] for value in images.values())
    width = max(value.shape[2] for value in images.values())
    aligned = {}
    for key, value in images.items():
        if value.shape[1:3] == (height, width):
            aligned[key] = value
        else:
            aligned[key] = torch.nn.functional.interpolate(
                value.movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False
            ).movedim(1, -1)
    return aligned


def _qwen_clip_model(clip):
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None:
        return None
    for name in (getattr(stage, "clip", None), getattr(stage, "clip_name", None)):
        candidate = getattr(stage, name, None) if isinstance(name, str) else None
        if candidate is not None and hasattr(candidate, "process_tokens") and hasattr(candidate, "transformer"):
            return candidate
    if hasattr(stage, "process_tokens") and hasattr(stage, "transformer"):
        return stage
    return None


def _token_rows(tokens):
    if isinstance(tokens, dict):
        tokens = next(iter(tokens.values()))
    return [[item[0] for item in batch] for batch in tokens]


def _visual_grid_metadata(info):
    extra = info.get("extra")
    grid = extra.get("grid") if isinstance(extra, dict) else extra
    if grid is None:
        raise ValueError("Qwen visual fusion source is missing its exact visual grid metadata.")
    values = grid.reshape(-1).tolist() if torch.is_tensor(grid) else list(grid)
    if len(values) < 3:
        raise ValueError(f"Qwen visual fusion received a malformed grid: {grid!r}.")
    shape = (int(values[-2]) // 2, int(values[-1]) // 2)
    if shape[0] * shape[1] != info["size"]:
        raise ValueError(f"Qwen visual grid {shape} does not match its {info['size']} embeddings.")
    return grid, shape


def _load_qwen_generation_model(clip):
    model = _qwen_clip_model(clip)
    if model is None:
        raise ValueError("Active visual fusion requires a supported Core Qwen multimodal model wrapper.")
    if hasattr(model, "reset_clip_options"):
        model.reset_clip_options()
    if hasattr(clip, "load_model"):
        clip.load_model()
    device = getattr(getattr(clip, "patcher", None), "load_device", None) or getattr(model, "execution_device", None)
    if device is None:
        device = model.transformer.get_input_embeddings().weight.device
    if hasattr(model, "set_clip_options"):
        model.set_clip_options({"layer": None, "execution_device": device})
    return model, device


def _encode_qwen_visual_sources(clip, model, device, full_prompt, images, thinking):
    passes = []
    for image in images:
        tokens = clip.tokenize(full_prompt, skip_template=True, min_length=1, thinking=thinking, images=[image], image=image)
        embeds, _, _, info = model.process_tokens(_token_rows(tokens), device)
        visual = [entry for entry in info if entry.get("type") == "image"]
        if len(visual) != 1:
            raise ValueError(f"Qwen visual fusion expected one visual block per source pass, received {len(visual)}.")
        entry = visual[0]
        grid, merged_grid = _visual_grid_metadata(entry)
        passes.append((embeds, entry, grid, merged_grid))
    return passes


def _fuse_qwen_primary_sources(passes, config, device, mask_cache):
    canonical, canonical_info, _, _ = passes[0]
    start, size = canonical_info["index"], canonical_info["size"]
    primary = []
    grids = []
    for embeds, info, _, grid in passes:
        source_start, source_size = info["index"], info["size"]
        if source_start != start or embeds.shape[1] - source_start - source_size != canonical.shape[1] - start - size:
            raise ValueError("Qwen fusion source prompts produced incompatible text layouts.")
        if embeds.shape[-1] != canonical.shape[-1]:
            raise ValueError("Qwen fusion source prompts produced incompatible embedding dimensions.")
        primary.append(embeds[0, source_start:source_start + source_size].to(device))
        grids.append(grid)
    fused = fuse_visual_token_sources(primary, config, device, mask_cache, size, grids)
    canonical = canonical.clone()
    canonical[0, start:start + size] = fused
    return canonical, canonical_info, grids, size


def generate_fused_qwen3vl(clip, full_prompt, images, config, generation_args, thinking=False):
    """Encode isolated Qwen3-VL images, fuse primary/DeepStack blocks, then generate."""
    model, device = _load_qwen_generation_model(clip)
    if model is None or not hasattr(model.transformer, "build_image_inputs"):
        raise ValueError("Active visual fusion requires a Core Qwen3-VL model wrapper with DeepStack support.")
    if config.get("save_blended_embeds", False):
        logging.warning("UC_TextGenerate ignores save_blended_embeds: a primary block alone cannot save the complete DeepStack generation state.")

    with _qwen_execution_context(clip, device):
        passes = _encode_qwen_visual_sources(clip, model, device, full_prompt, images, thinking)
        deepstacks = {}
        for index, (_, entry, _, _) in enumerate(passes):
            deepstack = entry.get("extra", {}).get("deepstack")
            if not deepstack:
                raise ValueError("Qwen3-VL visual fusion source is missing DeepStack tensors.")
            deepstacks[index] = deepstack

        mask_cache = {}
        canonical, canonical_info, grids, size = _fuse_qwen_primary_sources(passes, config, device, mask_cache)
        fused_deepstack = fuse_deepstack_layers(deepstacks, config, device, mask_cache, size, grids)
        logging.info(
            "TokenFusion visual-token fusion complete: fused %d Qwen3-VL visual sources into 1 block (%d visual tokens, %d DeepStack layers).",
            len(passes), size, len(fused_deepstack),
        )
        fused_info = dict(canonical_info)
        fused_info["extra"] = dict(canonical_info["extra"])
        fused_info["extra"]["deepstack"] = fused_deepstack
        position_ids, visual_mask, rebuilt_deepstack = model.transformer.build_image_inputs(canonical, [fused_info])
        transformer_args = {k: v for k, v in generation_args.items() if k != "mtp"}
        return model.transformer.generate(
            canonical, **transformer_args, position_ids=position_ids,
            visual_pos_masks=visual_mask, deepstack_embeds=rebuilt_deepstack
        )


def generate_fused_qwen35(clip, full_prompt, images, config, generation_args, thinking=False):
    """Encode and fuse Qwen3.5's primary visual blocks, then generate with rebuilt MRoPE."""
    model, device = _load_qwen_generation_model(clip)
    if hasattr(model.transformer, "build_image_inputs"):
        raise ValueError("Qwen3.5 visual fusion received a Qwen3-VL model wrapper.")
    if config.get("save_blended_embeds", False):
        logging.warning("UC_TextGenerate ignores save_blended_embeds in the fused text-generation path.")
    logging.info("UC_TextGenerate: Qwen3.5 visual fusion uses the primary visual block only; this model has no configured DeepStack outputs.")

    with _qwen_execution_context(clip, device):
        passes = _encode_qwen_visual_sources(clip, model, device, full_prompt, images, thinking)
        canonical, canonical_info, _, _ = _fuse_qwen_primary_sources(passes, config, device, {})
        logging.info(
            "TokenFusion visual-token fusion complete: fused %d Qwen3.5 visual sources into 1 block (%d visual tokens, no DeepStack).",
            len(passes), canonical_info["size"],
        )
        fused_info = dict(canonical_info)
        fused_info["extra"] = passes[0][2]
        position_ids = qwen2vl_mrope_position_ids([fused_info], canonical.shape[1], canonical.device)
        if position_ids is None:
            raise ValueError("Qwen3.5 visual fusion could not rebuild multimodal MRoPE positions.")
        return model.transformer.generate(canonical, **generation_args, position_ids=position_ids)

# -----------------------------------------------------------------------------
# 2. Template Definitions for Unified Text Generation
# -----------------------------------------------------------------------------

MODEL_TEMPLATES = {
    "qwen35": {
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "visual_token": "<|vision_start|><|image_pad|><|vision_end|>",
        "suppress_thinking": "<think>\n</think>\n"
    },
    "qwen3vl": {
        "system_prefix": "<|im_start|>system\n",
        "system_suffix": "<|im_end|>\n",
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "visual_token": "<|vision_start|><|image_pad|><|vision_end|>",
        "suppress_thinking": None
    },
    "llama3": {
        "system_prefix": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_suffix": "<|eot_id|>",
        "user_prefix": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_suffix": "<|eot_id|>",
        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "visual_token": "<image_soft_token>",
        "suppress_thinking": None
    },
    "gemma": {
        "system_prefix": "<start_of_turn>system\n",
        "system_suffix": "<end_of_turn>\n",
        "user_prefix": "<start_of_turn>user\n",
        "user_suffix": "<end_of_turn>\n",
        "assistant_prefix": "<start_of_turn>model\n",
        "visual_token": "<img><image_soft_token><end_of_image>",
        "suppress_thinking": "<think>\n\n</think>\n\n"
    },
    "custom": {
        "system_prefix": "",
        "system_suffix": "\n",
        "user_prefix": "\nUser: ",
        "user_suffix": "",
        "assistant_prefix": "\nAssistant: ",
        "visual_token": "<image>",
        "suppress_thinking": None
    }
}


def detect_textgen_template(clip) -> str:
    """Resolve the chat-template family from the connected tokenizer."""
    tokenizer = getattr(clip, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("UC_TextGenerate requires a clip with tokenizer metadata.")

    candidates = [getattr(tokenizer, "clip_name", ""), type(tokenizer).__name__]
    inner_name = getattr(tokenizer, "clip", None)
    if isinstance(inner_name, str):
        inner = getattr(tokenizer, inner_name, None)
        if inner is not None:
            candidates.extend([type(inner).__name__, getattr(inner, "embedding_key", "")])
    identity = " ".join(str(value).lower() for value in candidates if value)

    if "qwen35" in identity:
        return "qwen35"
    if "qwen3vl" in identity or "qwen3_vl" in identity:
        return "qwen3vl"
    if "llama" in identity:
        return "llama3"
    if "gemma" in identity:
        return "gemma"
    raise ValueError(
        "UC_TextGenerate could not identify a supported tokenizer family "
        f"from the connected clip ({identity or 'no tokenizer identity'})."
    )

# -----------------------------------------------------------------------------
# 3. Class Implementations
# -----------------------------------------------------------------------------

class UC_TextGenerate(io.ComfyNode):
    """
    Advanced text generation node that implements:
    - Multiple image handling via dynamic autogrow input.
    - Timestamped Qwen3-VL video handling from chronological image batches.
    - Image rescaling limits to optimize performance and prevent out-of-memory.
    - Safe system prompt template selection for different standard formats.
    - Safe manual concatenation to prevent formatting errors with user characters.
    """
    @classmethod
    def define_schema(cls):
        # Sampling Mode Options
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001, tooltip="Controls variety. Lower is more predictable; higher is more varied."),
                    io.Int.Input("top_k", default=64, min=0, max=1000, tooltip="Restricts choices to the K most likely options. 0 turns it off."),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01, tooltip="Restricts choices to likely options. Higher values allow more variety."),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Ignores unlikely options. 0 turns it off."),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01, tooltip="Discourages repeated text. 1 turns it off."),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01, tooltip="Discourages words already used in the response. 0 turns it off."),
                    io.Int.Input(
                        "empty_response_retries",
                        default=4,
                        min=0,
                        max=32,
                        advanced=True,
                        tooltip="Retry blank generations with consecutive seeds and return the successful seed.",
                    ),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        # Dynamic Autogrow Setup for up to 16 images
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix="image",
            min=1,
            max=16
        )

        return io.Schema(
            node_id="UC_TextGenerate",
            display_name="Text Generate (TokenFusion)",
            category="advanced/textgen",
            search_aliases=["LLM", "VLM", "textgen", "generate", "chat", "qwen", "gemma", "llama"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip="Main instruction for the generated text. Braces can be used normally."
                ),
                io.String.Input("system_prompt", multiline=True, dynamic_prompts=False, default=""),
                io.Int.Input(
                    "vlm_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Image analysis size. Larger values retain more detail but use more memory."
                ),
                io.Int.Input(
                    "vlm_video_resolution",
                    default=384,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="Qwen3-VL / Qwen3.5 resolution for video frames. Higher values use more visual tokens.",
                ),
                io.Int.Input("max_length", default=512, min=1, max=32768, tooltip="Maximum response length."),
                io.String.Input(
                    "formula",
                    default="",
                    multiline=False,
                    tooltip="Optional image blend. Leave empty to keep images separate."
                ),
                io.Boolean.Input("thinking", optional=True, default=False, tooltip="Include model thinking text when available."),
                io.Combo.Input(
                    "mtp",
                    options=["auto", "off", "2", "3", "4", "5"],
                    default="auto",
                    optional=True,
                    tooltip="Multi-token speculative decoding when supported by the loaded model (auto adapts depth; off disables it).",
                ),
                io.Boolean.Input(
                    "escape_parentheses",
                    optional=True,
                    default=False,
                    tooltip="Prevents errors from parentheses left in generated text.",
                ),
                VisualFusionConfig.Input("visual_fusion_config", display_name="Fusion Config", optional=True, tooltip="Optional settings for combining images before generation."),
                ResponseSchema.Input(
                    "response_schema",
                    display_name="Response Schema",
                    optional=True,
                    tooltip="Optional JSON Schema (such as from VLM System Instruction Presets or Configure Gemini Response Schema) to enforce structured JSON output.",
                ),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode", tooltip="On varies results using the seed. Off gives repeatable results."),
                io.Image.Input(
                    "video",
                    optional=True,
                    tooltip="Chronological frame batch from Get Video Components.",
                ),
                io.Float.Input(
                    "video_fps",
                    default=24.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Source frame rate from Get Video Components.",
                ),
                io.Float.Input(
                    "sample_fps",
                    default=2.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Frames per second presented to Qwen3-VL or Qwen3.5.",
                ),
                io.Autogrow.Input("image_inputs", template=autogrow_template, tooltip="Add images in order. Each image in a batch is added in order."),
            ],
            outputs=[
                io.String.Output("generated_text", display_name="Generated Text"),
                io.Int.Output("seed", display_name="Seed"),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        system_prompt,
        vlm_resolution,
        max_length,
        sampling_mode,
        formula="",
        thinking=False,
        mtp="auto",
        escape_parentheses=False,
        image_inputs=None,
        visual_fusion_config=None,
        response_schema=None,
        video=None,
        video_fps=24.0,
        sample_fps=2.0,
        vlm_video_resolution=384,
    ) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: CLIP/TextEncoder input is invalid: None")

        mtp_val = False if mtp == "off" else (True if mtp == "auto" else int(mtp))

        template_name = detect_textgen_template(clip)

        # 1. Parse Autogrow Inputs
        raw_images = {}
        if image_inputs is not None:
            for k, v in image_inputs.items():
                if v is not None:
                    digits = re.findall(r'\d+', k)
                    idx = int(digits[0]) if digits else 1
                    raw_images[idx] = v

        is_zero_indexed = 0 in raw_images

        # 2. Rescale Images to Target Resolutions & Create Variable Mapping
        processed_images = {}
        for idx, (num, img) in enumerate(sorted(raw_images.items())):
            display_name = ImageInputMapping.get_display_name(num, is_zero_indexed)
            scaled_img = prepare_vlm_image(img, vlm_resolution)
            processed_images[display_name] = scaled_img

            # Map sequentially to letter variables a, b, c, d... (matching TextEncodeKrea2SystemEditScaledAdv)
            letter = chr(97 + idx) # 0 -> 'a', 1 -> 'b', 2 -> 'c'...
            processed_images[letter] = scaled_img

        fusion_method = (visual_fusion_config or {}).get("visual_fusion_method", "off")
        fusion_active = fusion_method != "off"

        if not fusion_active and formula and formula.strip():
            try:
                blended_image = evaluate_formula(formula.strip(), _aligned_image_values(processed_images))
                # Override raw_images and processed_images to contain only this single blended image
                raw_images = {1: blended_image}
                processed_images = {
                    "image_input_1": blended_image,
                    "a": blended_image
                }
                is_zero_indexed = False
            except Exception as e:
                raise RuntimeError(f"Error evaluating global textgen blending formula '{formula}': {e}")

        # 3. Parse and Evaluate Math Formulas inside |pipes|
        images_vl = []
        resolved_visuals = {}
        visual_marker_pattern = re.compile(r"__UC_TEXTGEN_VISUAL_(\d+)__")
        math_pattern = re.compile(r"\|([^|]+)\|")
        temp = MODEL_TEMPLATES[template_name]
        v_token = temp["visual_token"]

        def register_visual(image):
            index = len(resolved_visuals)
            resolved_visuals[index] = image
            return f"__UC_TEXTGEN_VISUAL_{index}__"

        def replace_formula(match):
            expression = match.group(1).strip()
            # Build list of valid active variables (e.g. "image_input_1", "a", "b", etc.)
            valid_vars = list(processed_images.keys())

            # Check if any valid variable is present as a word boundary inside the expression
            has_valid_var = any(re.search(r'\b' + re.escape(var) + r'\b', expression, re.IGNORECASE) for var in valid_vars)

            if not raw_images or not has_valid_var:
                return match.group(0)

            try:
                result_tensor = evaluate_formula(expression, _aligned_image_values(processed_images))
                return register_visual(result_tensor)
            except Exception:
                # Fallback to preserving the original text in case of evaluation error
                return match.group(0)

        modified_prompt = math_pattern.sub(replace_formula, prompt)

        # 4. Check for keyword references like 'image_input_1'
        pattern = re.compile(r'image_input_(\d+)', re.IGNORECASE)
        has_keywords = bool(pattern.search(modified_prompt)) or bool(resolved_visuals)

        if has_keywords:
            def replace_keyword(match):
                num = int(match.group(1))
                dict_key = ImageInputMapping.get_dict_key(num, is_zero_indexed)
                if dict_key in raw_images:
                    img = raw_images[dict_key]
                    display_name = ImageInputMapping.get_display_name(dict_key, is_zero_indexed)
                    processed_img = processed_images.get(display_name, prepare_vlm_image(img, vlm_resolution))
                    return register_visual(processed_img)
                return ""
            modified_prompt = pattern.sub(replace_keyword, modified_prompt)

            def resolve_visual(match):
                image = resolved_visuals[int(match.group(1))]
                tokens = []
                for index in range(image.shape[0]):
                    images_vl.append(image[index:index + 1])
                    tokens.append(v_token)
                return "".join(tokens)

            modified_prompt = visual_marker_pattern.sub(resolve_visual, modified_prompt)
        else:
            # Fallback behavior: prepend all connected images in index-sorted order
            image_prompt_prefix = ""
            for num in sorted(raw_images.keys()):
                display_name = ImageInputMapping.get_display_name(num, is_zero_indexed)
                processed_img = processed_images[display_name]
                for index in range(processed_img.shape[0]):
                    images_vl.append(processed_img[index:index + 1])
                    image_prompt_prefix += v_token
            modified_prompt = image_prompt_prefix + modified_prompt

        video_frames = None
        if video is not None:
            if template_name not in {"qwen3vl", "qwen35"}:
                raise ValueError("Video input is supported only by Core Qwen3-VL and Qwen3.5 models.")
            video_frames, sampled_indices = prepare_qwen3vl_video_frames(
                video,
                vlm_video_resolution,
                video_fps,
                sample_fps,
            )
            modified_prompt = qwen3vl_video_prompt(sampled_indices, video_fps) + modified_prompt

        if fusion_active:
            if template_name not in {"qwen3vl", "qwen35"}:
                raise ValueError(f"Active visual fusion is supported only by Core Qwen3-VL and Qwen3.5, not {template_name}.")
            if not images_vl:
                raise ValueError("Active visual fusion requires at least one resolved image source.")
            first = modified_prompt.find(v_token)
            if first < 0:
                raise ValueError("Active visual fusion could not locate the resolved visual marker in the prompt.")
            modified_prompt = modified_prompt[:first] + v_token + modified_prompt[first + len(v_token):].replace(v_token, "")

        if template_name in {"qwen3vl", "qwen35"}:
            picture_number = 0

            def add_picture_id(_match):
                nonlocal picture_number
                picture_number += 1
                return f"Picture {picture_number}: {v_token}"

            modified_prompt = re.sub(re.escape(v_token), add_picture_id, modified_prompt)

        # 5. Build Safe Non-Formatting Chat Template
        # We completely avoid standard formatting or f-strings which fail if prompts contain { } braces.
        full_prompt = ""

        # System prompt segment
        if system_prompt and system_prompt.strip():
            full_prompt += temp["system_prefix"] + system_prompt + temp["system_suffix"]

        # User query segment
        full_prompt += temp["user_prefix"] + modified_prompt + temp["user_suffix"]

        # Assistant block entry
        full_prompt += temp["assistant_prefix"]

        # If thinking is disabled, explicitly append reasoning suppression sequences if the model template has them
        if not thinking and temp["suppress_thinking"]:
            full_prompt += temp["suppress_thinking"]

        schema_active = isinstance(response_schema, dict) and bool(response_schema)
        prefill_open = ""
        if schema_active and not thinking:
            prefill_open = "[" if response_schema.get("type") == "array" else "{"
            full_prompt += prefill_open

        # 6. Setup Tokenizer Dispatch Arguments
        kwargs = {}
        if len(images_vl) > 0:
            kwargs["images"] = images_vl
            if len(images_vl) == 1:
                kwargs["image"] = images_vl[0]

        # Use skip_template=True because we have assembled perfect, model-specific delimiters ourselves.
        tokens = None
        if not fusion_active and video_frames is None:
            tokens = clip.tokenize(
                full_prompt,
                skip_template=True,
                min_length=1,
                thinking=thinking,
                **kwargs
            )

        # 7. Extract Parameters from sampling_mode dynamic combo
        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)
        empty_response_retries = sampling_mode.get("empty_response_retries", 0)

        # 8. Generation & Decoding
        generation_args = dict(
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            mtp=mtp_val,
        )
        attempts = empty_response_retries + 1 if do_sample and seed is not None else 1
        used_seed = int(seed) if seed is not None else 0
        generated_text = ""
        for attempt in range(attempts):
            if seed is not None:
                used_seed = (int(seed) + attempt) & 0xffffffffffffffff
                generation_args["seed"] = used_seed

            with schema_guided_generation_context(clip, response_schema if schema_active else None, prefilled_open=prefill_open):
                if video_frames is not None:
                    generated_ids = generate_qwen3vl_video(
                        clip,
                        full_prompt,
                        images_vl,
                        video_frames,
                        visual_fusion_config if fusion_active else None,
                        generation_args,
                        thinking,
                    )
                elif fusion_active:
                    if template_name == "qwen3vl":
                        generated_ids = generate_fused_qwen3vl(clip, full_prompt, images_vl, visual_fusion_config, generation_args, thinking)
                    else:
                        generated_ids = generate_fused_qwen35(clip, full_prompt, images_vl, visual_fusion_config, generation_args, thinking)
                else:
                    generated_ids = clip.generate(tokens, **generation_args)

            generated_text = prefill_open + clip.decode(generated_ids, skip_special_tokens=True)
            if schema_active:
                generated_text, schema_valid = conform_and_validate_response_schema(generated_text, response_schema)
                if schema_valid and generated_text.strip():
                    break
            elif generated_text.strip():
                break

        if escape_parentheses:
            generated_text = generated_text.replace("(", r"\(").replace(")", r"\)")
        return io.NodeOutput(generated_text, used_seed)


class UC_TextGenerateQwen35SystemPrompt(io.ComfyNode):
    """
    TextGenerate variant for Qwen3.5 models with custom system message support.
    Builds the chat template via string concatenation (never .format()) so any
    characters in user input, including { } \\ and control sequences, are safe.
    """

    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="UC_TextGenerateQwen35SystemPrompt",
            display_name="Text Generate Qwen3.5 (System Prompt)",
            category="advanced/textgen",
            search_aliases=["LLM", "VLM", "qwen", "qwen35", "system prompt", "textgen"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="",
                    tooltip="Message sent to the model. Braces can be used normally."),
                io.String.Input("system_message", multiline=True, dynamic_prompts=False, default="",
                    tooltip="Optional instructions applied before the user message."),
                io.Image.Input("image", optional=True),
                io.Int.Input("max_length", default=512, min=1, max=8192),
                io.Boolean.Input("thinking", optional=True, default=False,
                    tooltip="Include model thinking text when available."),
                io.Combo.Input(
                    "mtp",
                    options=["auto", "off", "2", "3", "4", "5"],
                    default="auto",
                    optional=True,
                    tooltip="Multi-token speculative decoding when supported by the loaded model (auto adapts depth; off disables it).",
                ),
                ResponseSchema.Input(
                    "response_schema",
                    display_name="Response Schema",
                    optional=True,
                    tooltip="Optional JSON Schema (such as from VLM System Instruction Presets or Configure Gemini Response Schema) to enforce structured JSON output.",
                ),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
            ],
        )

    @classmethod
    def _build_prompt(cls, prompt: str, system_message: str, has_image: bool, thinking: bool) -> str:
        """
        Build the full Qwen3.5 chat-template string using concatenation only.
        No .format() / %-formatting; user-supplied strings are never interpolated,
        so { } and any other characters are completely safe.

        Template structure (matches qwen35.py Qwen35ImageTokenizer):
            [<|im_start|>system\\n{system_message}<|im_end|>\\n]   ... optional
            <|im_start|>user\\n
            [<|vision_start|><|image_pad|><|vision_end|>]          ... if image
            {prompt}<|im_end|>\\n
            <|im_start|>assistant\\n
            [<think>\\n</think>\\n]                                  ... if not thinking
        """
        result = ""

        # Optional system block
        if system_message and system_message.strip():
            result = (
                "<|im_start|>system\n"
                + system_message
                + "<|im_end|>\n"
            )

        # User block: image token placed before text per qwen35.py:761
        result += "<|im_start|>user\n"
        if has_image:
            result += "<|vision_start|><|image_pad|><|vision_end|>"
        result += prompt + "<|im_end|>\n"

        # Assistant block
        result += "<|im_start|>assistant\n"

        # Suppress thinking unless thinking mode requested (matches qwen35.py:784-785)
        if not thinking:
            result += "<think>\n</think>\n"

        return result

    @classmethod
    def execute(cls, clip, prompt, system_message, max_length, sampling_mode,
                image=None, thinking=False, mtp="auto", response_schema=None) -> io.NodeOutput:

        mtp_val = False if mtp == "off" else (True if mtp == "auto" else int(mtp))

        formatted_prompt = cls._build_prompt(
            prompt=prompt,
            system_message=system_message,
            has_image=image is not None,
            thinking=thinking,
        )
        schema_active = isinstance(response_schema, dict) and bool(response_schema)
        prefill_open = ""
        if schema_active and not thinking:
            prefill_open = "[" if response_schema.get("type") == "array" else "{"
            formatted_prompt += prefill_open

        # skip_template=True because we built the full template ourselves.
        # The tokenizer detects <|im_start|> prefix and skips its own template (qwen35.py:769).
        tokens = clip.tokenize(
            formatted_prompt,
            image=image,
            skip_template=True,
            min_length=1,
            thinking=thinking,
        )

        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        with schema_guided_generation_context(clip, response_schema if schema_active else None, prefilled_open=prefill_open):
            generated_ids = clip.generate(
                tokens,
                do_sample=do_sample,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                mtp=mtp_val,
            )

        generated_text = prefill_open + clip.decode(generated_ids, skip_special_tokens=True)
        if schema_active:
            generated_text, _ = conform_and_validate_response_schema(generated_text, response_schema)
        return io.NodeOutput(generated_text)
