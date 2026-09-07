import ast
from contextlib import contextmanager
import hashlib
import operator
import os
import math
import re
import torch
import torchaudio
import logging
import numbers
import threading
from einops import rearrange
from safetensors.torch import save_file
from enum import Enum
from fractions import Fraction
from pathlib import Path
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence
import numpy as np

import folder_paths
import node_helpers
import comfy
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
from .helper_functions import resize_nchw
from .image_helpers import VIDEO_FRAME_TIMESTAMP_FORMATS, format_video_timestamp, parse_video_timestamps
from .minimax_h3_guide_helpers import LAYOUT_KEY, build_layout, splice_conditioning
from .minimax_h3_cache_helpers import H3EncoderCache, spatial_cache_settings, temporal_cache_settings
from .minimax_h3_temporal_helpers import (
    encode_temporal_conditioning, fuse_temporal_block, minimax_h3_temporal_frame_pairs,
)

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention
from comfy.sd1_clip import token_weights, escape_important, unescape_important
from comfy.text_encoders.minimax import token_tags_from_embeds_info


_VISUAL_ENCODER_PATH_LOCK = threading.RLock()
MINIMAX_H3_MEDIA_STRUCTURE = "<<picture>>: <<visual>>"
MINIMAX_H3_VIDEO_LATENT_MODES = (
    "full video",
    "even keyframes",
    "off",
)
_MINIMAX_H3_MEDIA_KEYWORDS = {"time", "picture", "visual", "shot"}
_MINIMAX_H3_REQUIRED_MEDIA_KEYWORDS = {"picture", "visual"}


def prepare_vae_reference_image(samples, target_size, dimension_multiple, upscale_method="bicubic"):
    """Resize BCHW image samples for VAE encoding with configurable alignment."""
    multiple = int(dimension_multiple)
    if multiple < 4:
        raise ValueError("VAE dimension multiple must be at least 4.")
    height, width = samples.shape[-2:]
    if target_size is None:
        target_width = width
        target_height = height
    else:
        total_pixels = int(target_size) ** 2
        scale = math.sqrt(total_pixels / (width * height))
        target_width = width * scale
        target_height = height * scale
    aligned_width = max(multiple, round(target_width / multiple) * multiple)
    aligned_height = max(multiple, round(target_height / multiple) * multiple)
    return resize_nchw(samples, aligned_width, aligned_height, upscale_method)


def _resolve_clip_transformer(clip):
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None:
        return None
    if hasattr(stage, "clip") and isinstance(stage.clip, str) and hasattr(stage, stage.clip):
        clip_model = getattr(stage, stage.clip)
    elif hasattr(stage, "clip_model"):
        clip_model = stage.clip_model
    elif hasattr(stage, "clip_d"):
        clip_model = stage.clip_d
    else:
        clip_model = stage
    return getattr(clip_model, "transformer", None)


@contextmanager
def qwen3vl_visual_encoder_path(clip, path: str):
    """Select current grid/DeepStack or pre-d0008a89 flat Qwen3-VL encoding."""
    if path == "grid-deepstack":
        with _VISUAL_ENCODER_PATH_LOCK:
            yield
        return
    if path != "legacy-flat":
        raise ValueError(f"Unsupported visual encoder path: {path}")

    transformer = _resolve_clip_transformer(clip)
    if transformer is None or not hasattr(transformer, "build_image_inputs"):
        raise ValueError("legacy-flat requires a Qwen3-VL text encoder with build_image_inputs support.")

    # Core d0008a89 made Qwen3-VL build grid MRoPE, a visual-position mask,
    # and DeepStack inputs here. Returning empty inputs reproduces the inherited
    # pre-update BaseLlama forward while leaving image preprocessing unchanged.
    with _VISUAL_ENCODER_PATH_LOCK:
        original = transformer.build_image_inputs
        transformer.build_image_inputs = lambda embeds, embeds_info: (None, None, None)
        try:
            logging.warning("Visual fusion is using the pre-d0008a89 legacy flat Qwen3-VL encoder path.")
            yield
        finally:
            transformer.build_image_inputs = original


def _encode_scheduled_with_visual_path(clip, tokens, visual_encoder_path: str, cache=None):
    with qwen3vl_visual_encoder_path(clip, visual_encoder_path):
        if cache is not None:
            return cache.encode_scheduled(clip, tokens, visual_encoder_path, lambda: clip.encode_from_tokens_scheduled(tokens))
        return clip.encode_from_tokens_scheduled(tokens)


def encode_embedding_scaled_bias(clip, text, llama_template=None, **kwargs):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    if "<" not in text and ">" not in text and "=" not in text:
        tokens = clip.tokenize(text, llama_template=llama_template, **kwargs)
        return clip.encode_from_tokens_scheduled(tokens)

    # Permissive regex for whitespace inside tags
    bias_pattern = re.compile(r"<\s*([^>=]+?)\s*=\s*([0-9.-]+)\s*>")
    split_pattern = re.compile(r"(<\s*[^>=]+?\s*=\s*[0-9.-]+\s*>)")
    segments = split_pattern.split(text)

    clean_text = ""
    biases_to_apply = []

    # Use prefix-only template for measurements to avoid suffix-induced shifts
    prefix_template = "{}"
    if llama_template:
        prefix_template = llama_template.split("{}")[0] + "{}"

    for segment in segments:
        if not segment:
            continue

        match = bias_pattern.fullmatch(segment)
        if match:
            # Count before adding biased segment
            start_count = get_token_count_scaled(clip, clean_text, llama_template=prefix_template)

            bias_text, strength_str = match.groups()
            clean_text += bias_text

            # Count after adding biased segment
            end_count = get_token_count_scaled(clip, clean_text, llama_template=prefix_template)

            if end_count > start_count:
                # BOS is at index 0, so tokens are at indices 1 to count
                start_index = 1 + start_count
                end_index = 1 + end_count
                biases_to_apply.append({"start": start_index, "end": end_index, "strength": float(strength_str)})
        else:
            clean_text += segment

    tokens = clip.tokenize(clean_text, llama_template=llama_template, **kwargs)
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if not biases_to_apply:
        return conditioning

    # Apply contextual vector scaling directly to each schedule. Pooled output is
    # deliberately unchanged: local token weights do not define a pooled weight.
    new_conditioning = []

    for i in range(len(conditioning)):
        cond, cond_dict = conditioning[i]

        # Directly scale the embeddings for the biased tokens
        new_cond = cond.clone()

        for bias in biases_to_apply:
            strength = bias["strength"]
            start = min(bias["start"], new_cond.shape[1])
            end = min(bias["end"], new_cond.shape[1])

            if start >= end:
                continue

            new_cond[:, start:end, :] *= strength

        new_conditioning.append([new_cond, cond_dict.copy()])

    return new_conditioning


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


_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\bimage_input_(fusion|\d+)\b", re.IGNORECASE)
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def visual_text_encoder_key(clip) -> str | None:
    """Return the connected encoder's sole token key when it can be identified."""
    stage = getattr(clip, "cond_stage_model", None)
    for owner in (getattr(clip, "tokenizer", None), stage):
        key_name = getattr(owner, "clip_name", None)
        if isinstance(key_name, str):
            return key_name
    # Real ComfyUI CLIP objects declare clip_name. The fallback keeps lightweight
    # test doubles and compatible wrappers detectable without probing a known
    # model inside an unrelated visual-path context.
    if stage is not None:
        return None
    tokenize = getattr(clip, "tokenize", None)
    if tokenize is None:
        return None
    tokens = tokenize("")
    if not isinstance(tokens, dict) or len(tokens) != 1:
        return None
    return next(iter(tokens))


def is_klein_vl_text_encoder(clip) -> bool:
    tokenizer_type = type(getattr(clip, "tokenizer", None))
    return (
        tokenizer_type.__module__ == "comfy.text_encoders.flux"
        and tokenizer_type.__name__ in {"KleinVLTokenizer", "KleinVLTokenizer8B"}
    )


def visual_embedding_key(clip, tokens: dict) -> str:
    """Return the source model key required to load saved visual embeddings."""
    source_key = getattr(getattr(clip, "cond_stage_model", None), "clip_name", None)
    if isinstance(source_key, str):
        return source_key
    return next(iter(tokens))


def is_minimax_h3_text_encoder(clip) -> bool:
    return visual_text_encoder_key(clip) == "qwen3vl_32b"


def _token_entries(tokens, key_name: str) -> list:
    try:
        batches = tokens[key_name]
        if len(batches) != 1:
            raise ValueError
        return list(batches[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected one {key_name} token sequence from the connected encoder."
        ) from exc


def _minimax_h3_text_entries(clip, text: str) -> list:
    if not text:
        return []
    return _token_entries(clip.tokenize(text), "qwen3vl_32b")


def _minimax_h3_visual_token_entries(clip, image) -> list:
    full = _token_entries(
        clip.tokenize("", images=[image]), "qwen3vl_32b"
    )
    picture_one = _minimax_h3_text_entries(clip, "<Picture 1>: ")
    if len(full) <= len(picture_one):
        raise ValueError("MiniMax H3 tokenizer returned an incomplete picture block.")
    if [item[0] for item in full[: len(picture_one)]] != [
        item[0] for item in picture_one
    ]:
        raise ValueError("MiniMax H3 tokenizer returned an unexpected picture prefix.")
    visual = full[len(picture_one) :]
    if sum(is_image_token(item) for item in visual) != 1:
        raise ValueError("MiniMax H3 picture block must contain exactly one image entry.")
    return visual


def _minimax_h3_visual_token_blocks(clip, images) -> list[list]:
    if not images:
        return []
    entries = _token_entries(
        clip.tokenize("", images=list(images)), "qwen3vl_32b"
    )
    blocks = []
    for index, entry in enumerate(entries):
        if not is_image_token(entry):
            continue
        if index < 1 or index + 1 >= len(entries):
            raise ValueError("MiniMax H3 tokenizer returned an incomplete visual block.")
        blocks.append(entries[index - 1:index + 2])
    if len(blocks) != len(images):
        raise ValueError(
            f"MiniMax H3 tokenizer returned {len(blocks)} visual blocks for {len(images)} Pictures."
        )
    return blocks


def _minimax_h3_visual_entries(clip, image, picture_number: int) -> list:
    """Build one numbered H3 picture block through Core's active tokenizer."""
    return _minimax_h3_text_entries(
        clip, f"<Picture {picture_number}>: "
    ) + _minimax_h3_visual_token_entries(clip, image)


def _validate_minimax_h3_media_structure(structure):
    if not isinstance(structure, str) or not structure.strip():
        raise ValueError("MiniMax H3 media structure must not be empty.")
    keywords = re.findall(r"<<([^<>]+)>>", structure)
    unknown = sorted(set(keywords) - _MINIMAX_H3_MEDIA_KEYWORDS)
    if unknown:
        raise ValueError(f"Unknown MiniMax H3 media structure keyword: <<{unknown[0]}>>.")
    missing = sorted(_MINIMAX_H3_REQUIRED_MEDIA_KEYWORDS - set(keywords))
    if missing:
        raise ValueError(f"MiniMax H3 media structure is missing <<{missing[0]}>>.")
    if keywords.count("visual") != 1:
        raise ValueError("MiniMax H3 media structure requires exactly one <<visual>>.")
    return structure


def _minimax_h3_timestamped_video_entries(
    clip, frames, timestamps, timestamp_format
):
    timestamps = list(timestamps)
    if frames.shape[0] != len(timestamps):
        raise ValueError(
            "MiniMax H3 video frame and timestamp counts must match."
        )
    if frames.shape[0] % 2 == 1:
        frames = torch.cat([frames, frames[-1:]], dim=0)
        timestamps.append(timestamps[-1])
    source_entries = _token_entries(
        clip.tokenize(
            "",
            minimax_ref_items=[{
                "type": "video",
                "data": frames,
                "timestamps": timestamps,
            }],
        ),
        "qwen3vl_32b",
    )
    visual_blocks = []
    for index, item in enumerate(source_entries):
        value = item[0]
        if not (
            isinstance(value, dict)
            and value.get("type") == "image"
            and value.get("minimax_video_block", False)
        ):
            continue
        if (
            index == 0
            or index + 1 >= len(source_entries)
            or source_entries[index - 1][0] != 151652
            or source_entries[index + 1][0] != 151653
        ):
            raise ValueError(
                "MiniMax H3 tokenizer returned an invalid temporal video block."
            )
        visual_blocks.append(source_entries[index - 1:index + 2])
    if len(visual_blocks) != frames.shape[0] // 2:
        raise ValueError(
            "MiniMax H3 tokenizer returned an unexpected temporal video block count."
        )
    entries = _minimax_h3_text_entries(clip, "<Video 1>: ")
    for index, visual in enumerate(visual_blocks):
        timestamp = (timestamps[index * 2] + timestamps[index * 2 + 1]) / 2
        formatted = format_video_timestamp(timestamp, timestamp_format)
        entries.extend(_minimax_h3_text_entries(clip, f"<{formatted}>"))
        entries.extend(visual)
    return entries


def tokenize_minimax_h3_media_prompt(
    clip, text, pictures, picture_timestamps, timestamp_format, picture_structure,
    video_frames=None, video_timestamps=(),
    audio=False, default_single_visual=False, default_video_frames=None,
    default_video_timestamps=(),
):
    configured_video_frames = () if video_frames is None else video_frames
    if (
        default_single_visual
        and pictures
        and len(pictures) != 1
        and not configured_video_frames
        and default_video_frames is None
    ):
        raise ValueError(
            "MiniMax H3 default media config requires exactly one visual source."
        )
    effective_picture_timestamps = (
        () if default_single_visual and not pictures else picture_timestamps
    )
    if len(effective_picture_timestamps) > len(pictures):
        raise ValueError(
            f"MiniMax H3 media config received {len(effective_picture_timestamps)} timestamps "
            f"for {len(pictures)} available Pictures."
        )
    if len(configured_video_frames) != len(video_timestamps):
        raise ValueError(
            "MiniMax H3 configured video frame and timestamp counts must match."
        )
    entries = []
    visual_blocks = _minimax_h3_visual_token_blocks(clip, pictures)
    for index, visual in enumerate(visual_blocks, start=1):
        if index > len(effective_picture_timestamps):
            entries.extend(_minimax_h3_text_entries(clip, f"<Picture {index}>: "))
            entries.extend(visual)
            continue
        timestamp = effective_picture_timestamps[index - 1]
        expanded = picture_structure.replace(
            "<<time>>", format_video_timestamp(timestamp, timestamp_format)
        )
        expanded = expanded.replace("<<picture>>", f"<Picture {index}>")
        expanded = expanded.replace("<<shot>>", f"[Shot {index}]")
        before, after = expanded.split("<<visual>>")
        entries.extend(_minimax_h3_text_entries(clip, before))
        entries.extend(visual)
        entries.extend(_minimax_h3_text_entries(clip, after))
    if configured_video_frames:
        frames = torch.cat(tuple(configured_video_frames), dim=0)
        entries.extend(_minimax_h3_timestamped_video_entries(
            clip, frames, video_timestamps, timestamp_format
        ))
    if default_video_frames is not None:
        entries.extend(_minimax_h3_timestamped_video_entries(
            clip,
            default_video_frames,
            default_video_timestamps,
            timestamp_format,
        ))
    if audio:
        entries.extend(_minimax_h3_text_entries(clip, "<Audio 1>: "))
    entries.extend(_minimax_h3_text_entries(clip, text))
    return {"qwen3vl_32b": [entries]}


def build_minimax_h3_media_config(
    timestamps, timestamp_format="0.0s", structure=MINIMAX_H3_MEDIA_STRUCTURE,
    video_fps=2, video_latent_mode="even keyframes",
    video_latent_keyframes=4, temporal_density=1, temporal_fusion_method="consensus",
):
    if isinstance(timestamp_format, list):
        timestamp_format = timestamp_format[0] if timestamp_format else "0.0s"
    if isinstance(structure, list):
        structure = structure[0] if structure else MINIMAX_H3_MEDIA_STRUCTURE
    default_single_visual = timestamps is None or timestamps == []
    timestamps = [Fraction(0)] if default_single_visual else parse_video_timestamps(timestamps)
    if isinstance(video_fps, list):
        video_fps = video_fps[0] if video_fps else 2
    if isinstance(video_fps, bool) or not isinstance(video_fps, numbers.Integral):
        raise ValueError("MiniMax H3 video_fps must be an integer from 1 to 24.")
    video_fps = int(video_fps)
    if not 1 <= video_fps <= 24:
        raise ValueError("MiniMax H3 video_fps must be an integer from 1 to 24.")
    if isinstance(video_latent_mode, list):
        video_latent_mode = video_latent_mode[0] if video_latent_mode else "even keyframes"
    if video_latent_mode not in MINIMAX_H3_VIDEO_LATENT_MODES:
        raise ValueError("Unsupported MiniMax H3 video latent mode.")
    if isinstance(video_latent_keyframes, list):
        video_latent_keyframes = video_latent_keyframes[0] if video_latent_keyframes else 4
    if isinstance(video_latent_keyframes, bool) or not isinstance(video_latent_keyframes, numbers.Integral):
        raise ValueError("MiniMax H3 video latent keyframes must be an integer from 2 to 213.")
    video_latent_keyframes = int(video_latent_keyframes)
    if not 2 <= video_latent_keyframes <= 213:
        raise ValueError("MiniMax H3 video latent keyframes must be an integer from 2 to 213.")
    if timestamp_format not in VIDEO_FRAME_TIMESTAMP_FORMATS:
        raise ValueError(f"Unsupported video timestamp format: {timestamp_format}")
    structure = _validate_minimax_h3_media_structure(structure)
    if isinstance(temporal_density, list):
        temporal_density = temporal_density[0] if temporal_density else 1
    if isinstance(temporal_fusion_method, list):
        temporal_fusion_method = temporal_fusion_method[0] if temporal_fusion_method else "consensus"
    if isinstance(temporal_density, bool) or not isinstance(temporal_density, numbers.Integral) or not 1 <= temporal_density <= 24:
        raise ValueError("MiniMax H3 temporal density must be an integer from 1 to 24.")
    if temporal_fusion_method not in ("consensus", "spatial"):
        raise ValueError("Unsupported MiniMax H3 temporal fusion method.")
    return {
        "schema_version": 3,
        "timestamps_seconds": tuple(timestamps),
        "timestamp_format": timestamp_format,
        "structure": structure,
        "default_single_visual": default_single_visual,
        "video_fps": video_fps,
        "video_latent_mode": video_latent_mode,
        "video_latent_keyframes": video_latent_keyframes,
        "temporal_density": int(temporal_density),
        "temporal_fusion_method": temporal_fusion_method,
    }


def _validate_minimax_h3_media_config(media_config, output_frame_count):
    if not isinstance(media_config, dict) or media_config.get("schema_version") != 3:
        raise ValueError("Unsupported MiniMax H3 media config payload.")
    timestamps = list(media_config.get("timestamps_seconds", ()))
    if not timestamps:
        raise ValueError("MiniMax H3 media config requires Picture timestamps.")
    if any(not isinstance(timestamp, Fraction) or timestamp < 0 for timestamp in timestamps):
        raise ValueError("MiniMax H3 media config timestamps must be parsed nonnegative exact seconds.")
    output_duration = Fraction(output_frame_count, 24)
    if any(timestamp > output_duration for timestamp in timestamps):
        raise ValueError(
            "MiniMax H3 Picture timestamps must not exceed output duration "
            f"{float(output_duration):.3f}s."
        )
    video_fps = media_config.get("video_fps")
    if (
        isinstance(video_fps, bool)
        or not isinstance(video_fps, numbers.Integral)
        or not 1 <= video_fps <= 24
    ):
        raise ValueError("MiniMax H3 media config requires video_fps from 1 to 24.")
    video_latent_mode = media_config.get("video_latent_mode", "even keyframes")
    if video_latent_mode not in MINIMAX_H3_VIDEO_LATENT_MODES:
        raise ValueError("MiniMax H3 media config has an unsupported video latent mode.")
    video_latent_keyframes = media_config.get("video_latent_keyframes", 4)
    if (
        isinstance(video_latent_keyframes, bool)
        or not isinstance(video_latent_keyframes, numbers.Integral)
        or not 2 <= video_latent_keyframes <= 213
    ):
        raise ValueError("MiniMax H3 media config requires video_latent_keyframes from 2 to 213.")
    timestamp_format = media_config.get("timestamp_format")
    if timestamp_format not in VIDEO_FRAME_TIMESTAMP_FORMATS:
        raise ValueError("MiniMax H3 media config has an unsupported timestamp format.")
    structure = _validate_minimax_h3_media_structure(media_config.get("structure"))
    return (
        timestamps,
        timestamp_format,
        structure,
        bool(media_config.get("default_single_visual", False)),
        int(video_fps),
        video_latent_mode,
        int(video_latent_keyframes),
    )


def _encode_minimax_h3_audio_reference(audio, audio_vae, cache=None):
    if audio is None:
        return None
    if audio_vae is None:
        raise ValueError("MiniMax H3 requires audio_vae when audio is provided.")
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        raise ValueError("MiniMax H3 audio must contain a waveform tensor.")
    waveform = audio["waveform"]
    sample_rate = audio.get("sample_rate")
    if waveform.ndim != 3 or waveform.shape[-1] < 1 or not torch.isfinite(waveform).all() or not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError("MiniMax H3 audio requires a finite [batch, channels, samples] waveform and positive sample rate.")
    target_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    samples = waveform[:1].movedim(1, -1)
    latent = audio_vae.encode(samples) if cache is None else cache.encode_vae(audio_vae, samples, media="audio")
    if not torch.is_tensor(latent) or latent.ndim < 1 or latent.shape[-1] < 1:
        raise ValueError("MiniMax H3 audio VAE returned an invalid latent.")
    return {"kind": "audio", "ref_audio_t": latent.shape[-1], "audio_latent": latent}


def tokenize_minimax_h3_prompt(clip, text: str, images) -> dict:
    """Replace internal visual-slot sentinels with H3 picture token entries."""
    segments = text.split(VISION_BLOCK)
    marker_count = len(segments) - 1
    if marker_count > len(images):
        raise ValueError(
            "MiniMax H3 prompt contains more visual slots than supplied images."
        )
    entries = []
    for index, segment in enumerate(segments):
        entries.extend(_minimax_h3_text_entries(clip, segment))
        if index < marker_count:
            entries.extend(
                _minimax_h3_visual_entries(clip, images[index], index + 1)
            )
    if not entries:
        entries = _token_entries(clip.tokenize(""), "qwen3vl_32b")
    return {"qwen3vl_32b": [entries]}


def format_minimax_h3_prompt(prompt: str, system_prompt: str) -> str:
    """Keep an implicit prefix picture before all raw H3 prompt text."""
    leading_slots = ""
    while prompt.startswith(VISION_BLOCK):
        leading_slots += VISION_BLOCK
        prompt = prompt[len(VISION_BLOCK) :]
    text = f"{system_prompt}\n{prompt}" if system_prompt else prompt
    return leading_slots + text


def minimax_h3_frame_count(length: int) -> int:
    """Snap a requested H3 frame count to Core's 17k+5 temporal grid."""
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def minimax_h3_empty_av_latent(width: int, height: int, length: int) -> tuple[dict, int]:
    """Create the batch-one joint video/audio latent expected by MiniMax H3."""
    width = int(width)
    height = int(height)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be multiples of 32.")
    frame_count = minimax_h3_frame_count(length)
    video_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    audio_t = round((frame_count / 24) * 40)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([1, 24, video_t, height // 16, width // 16], device=device)
    audio = torch.zeros([1, 32, 2, audio_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def prepare_minimax_h3_frame(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    """Prepare one BHWC image using the same geometry operation as Core H3."""
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Each MiniMax H3 visual source must contain exactly one image.")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", crop)
    return samples.movedim(1, -1)


def minimax_h3_reference_size(
    image_width: int,
    image_height: int,
    generation_width: int,
    generation_height: int,
    ref_image_size: str,
) -> tuple[int, int]:
    """Match Core MiniMax H3 image-reference sizing and canvas alignment."""
    image_width = int(image_width)
    image_height = int(image_height)
    if image_width < 1 or image_height < 1:
        raise ValueError("MiniMax H3 reference images must have non-zero dimensions.")
    if ref_image_size == "match":
        scale = min(
            1.0,
            math.sqrt(
                (int(generation_width) * int(generation_height))
                / (image_width * image_height)
            ),
        )
    elif ref_image_size == "max":
        scale = min(1.0, 2048 / min(image_width, image_height))
    else:
        raise ValueError(
            f"Unsupported MiniMax H3 reference image size: {ref_image_size}"
        )
    target_width = max(32, round(image_width * scale / 32) * 32)
    target_height = max(32, round(image_height * scale / 32) * 32)
    return target_width, target_height


def prepare_minimax_h3_reference_image(
    image: torch.Tensor,
    width: int,
    height: int,
    ref_image_size: str,
) -> torch.Tensor:
    """Prepare the shared Qwen/VAE pixels for one H3 image reference."""
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Each MiniMax H3 visual source must contain exactly one image.")
    target_width, target_height = minimax_h3_reference_size(
        image.shape[2], image.shape[1], width, height, ref_image_size
    )
    return prepare_minimax_h3_frame(
        image, target_width, target_height, "disabled"
    )


def _minimax_h3_reference_video_frame_count(
    video: torch.Tensor, maximum_frames: int
) -> int:
    if (
        not torch.is_tensor(video)
        or video.ndim != 4
        or video.shape[0] < 1
        or video.shape[1] < 1
        or video.shape[2] < 1
        or video.shape[3] < 3
        or not torch.isfinite(video).all()
    ):
        raise ValueError(
            "MiniMax H3 video must be a finite BHWC frame batch with at least three channels."
        )
    frame_count = min(video.shape[0], int(maximum_frames))
    if frame_count < 5:
        raise ValueError(
            "MiniMax H3 reference video needs at least 5 frames (~0.2s at 24 fps)."
        )
    while frame_count % 17 != 5:
        frame_count -= 1
    return frame_count


def prepare_minimax_h3_reference_video(
    video: torch.Tensor,
    vae,
    maximum_frames: int,
    encode_reference: bool = True,
    cache=None,
) -> tuple[torch.Tensor, dict | None]:
    """Prepare one 24-fps H3 reference video using Core's ref2va contract."""
    frame_count = _minimax_h3_reference_video_frame_count(video, maximum_frames)
    if encode_reference and vae is None:
        raise ValueError("MiniMax H3 video requires the video VAE input.")
    source_height, source_width = video.shape[1:3]
    ratio = source_width / source_height
    if ratio >= 1.0:
        target_width, target_height = 768 * ratio, 768
    else:
        target_width, target_height = 768, 768 / ratio
    if target_width * target_height > 768 * 1344:
        scale = math.sqrt((768 * 1344) / (target_width * target_height))
        target_width *= scale
        target_height *= scale
    target_width = max(32, round(target_width / 32) * 32)
    target_height = max(32, round(target_height / 32) * 32)
    if source_width * source_height < target_width * target_height:
        target_width = max(32, round(source_width / 32) * 32)
        target_height = max(32, round(source_height / 32) * 32)
    samples = video[:frame_count, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, target_width, target_height, "lanczos", "disabled"
    )
    frames = samples.movedim(1, -1)
    if not encode_reference:
        return frames, None
    latent = vae.encode(frames) if cache is None else cache.encode_vae(vae, frames, media="video")
    if not torch.is_tensor(latent) or latent.ndim < 5:
        raise ValueError("MiniMax H3 video VAE returned an invalid latent.")
    return frames, {
        "kind": "video",
        "latent_t": latent.shape[2],
        "latent_h": target_height // 16,
        "latent_w": target_width // 16,
        "ref_audio_t": 0,
        "latent": latent,
        "audio_latent": None,
    }


def prepare_minimax_h3_positioned_video_keyframes(
    video: torch.Tensor,
    vae,
    maximum_frames: int,
    width: int,
    height: int,
    keyframe_count: int,
    cache=None,
) -> list[dict]:
    """Encode one complete H3 video and retain evenly positioned temporal chunks."""
    frame_count = _minimax_h3_reference_video_frame_count(video, maximum_frames)
    if vae is None:
        raise ValueError("MiniMax H3 even Video keyframes require the video VAE input.")
    samples = video[:frame_count, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(
        samples, int(width), int(height), "lanczos", "center"
    )
    pixels = samples.movedim(1, -1)
    latent = vae.encode(pixels) if cache is None else cache.encode_vae(vae, pixels, media="video")
    if not torch.is_tensor(latent) or latent.ndim < 5:
        raise ValueError("MiniMax H3 video VAE returned an invalid latent.")
    full_chunk_count = (frame_count - 5) // 17
    expected_latent_t = full_chunk_count * 5 + 2
    if latent.shape[2] != expected_latent_t:
        raise ValueError(
            "MiniMax H3 video VAE returned an unexpected temporal length: "
            f"expected {expected_latent_t}, received {latent.shape[2]}."
        )
    expected_spatial = (int(height) // 16, int(width) // 16)
    if latent.shape[3:5] != expected_spatial:
        raise ValueError(
            "MiniMax H3 video VAE returned an unexpected spatial shape: "
            f"expected {expected_spatial}, received {tuple(latent.shape[3:5])}."
        )
    available = full_chunk_count + 1
    selected = min(int(keyframe_count), available)
    if available == 1:
        chunk_indices = [0]
    else:
        chunk_indices = [
            (index * (available - 1) + (selected - 1) // 2) // (selected - 1)
            for index in range(selected)
        ]
    if len(set(chunk_indices)) != len(chunk_indices):
        raise ValueError("MiniMax H3 even Video keyframe selection produced duplicate points.")
    keyframes = []
    for chunk_index in chunk_indices:
        latent_start = chunk_index * 5
        latent_stop = latent_start + (2 if chunk_index == full_chunk_count else 5)
        keyframes.append({
            "resolved_frame_index": chunk_index * 17,
            "latent": latent[:, :, latent_start:latent_stop].clone(),
        })
    return keyframes


def minimax_h3_video_sample_indices(frame_count: int, video_fps: int) -> list[int]:
    """Select nearest 24 fps source positions at a uniform presentation rate."""
    if frame_count < 1:
        return []
    indices = []
    sample_index = 0
    while True:
        source_index = (sample_index * 24 + video_fps // 2) // video_fps
        if source_index >= frame_count:
            break
        if not indices or source_index != indices[-1]:
            indices.append(source_index)
        sample_index += 1
    return indices


def prepare_image_placeholder_prompt(prompt: str, image_count: int, fusion_active: bool, context: str) -> tuple[str, tuple[int, ...]]:
    """Normalize custom image placeholders without leaving invalid names as text."""
    matches = list(_IMAGE_PLACEHOLDER_PATTERN.finditer(prompt))

    if fusion_active:
        if any(tag in prompt for tag in ("<|image_pad|>", "<|image|>", "<|vision_start|>")):
            if matches:
                logging.warning(
                    "%s: native visual tokens already exist; stripped %d image_input placeholder(s).",
                    context,
                    len(matches),
                )
            return _IMAGE_PLACEHOLDER_PATTERN.sub("", prompt), ()

        chosen = next((match for match in matches if match.group(1).lower() == "fusion"), None)
        if chosen is None:
            chosen = next((match for match in matches if match.group(1) == "1"), None)

        if chosen is None:
            if matches:
                logging.warning(
                    "%s: fusion accepts image_input_fusion or image_input_1; stripped %d unsupported placeholder(s).",
                    context,
                    len(matches),
                )
            logging.warning("%s: no fusion placeholder found; prepended the fused visual slot.", context)
            return VISION_BLOCK + _IMAGE_PLACEHOLDER_PATTERN.sub("", prompt), ()

        if chosen.group(1).lower() == "1":
            logging.warning("%s: treating image_input_1 as image_input_fusion.", context)
        if len(matches) > 1:
            logging.warning(
                "%s: fusion uses one visual slot; stripped %d additional image_input placeholder(s).",
                context,
                len(matches) - 1,
            )

        def replace_fusion(match):
            return VISION_BLOCK if match.start() == chosen.start() else ""

        return _IMAGE_PLACEHOLDER_PATTERN.sub(replace_fusion, prompt), ()

    valid_numbers = []
    removed = []

    def validate_numbered(match):
        suffix = match.group(1).lower()
        if suffix == "fusion":
            removed.append(match.group(0))
            return ""
        number = int(suffix)
        if number < 1 or number > image_count:
            removed.append(match.group(0))
            return ""
        valid_numbers.append(number)
        return VISION_BLOCK

    rewritten = _IMAGE_PLACEHOLDER_PATTERN.sub(validate_numbered, prompt)
    if removed:
        logging.warning(
            "%s: stripped unavailable or fusion-only placeholder(s): %s.",
            context,
            ", ".join(removed),
        )
    return rewritten, tuple(valid_numbers)

def _token_value(token):
    return token[0] if isinstance(token, tuple) and token else token


def is_image_token(t):
    val = _token_value(t)

    if isinstance(val, dict) and val.get("type") == "image":
        return True

    if isinstance(val, numbers.Integral) and val in (151655, 262144): # Qwen & Gemma image pad IDs
        return True

    return False


_QWEN_IM_START, _QWEN_USER, _QWEN_NL, _QWEN_IM_END = 151644, 872, 198, 151645


def _token_id(token):
    value = _token_value(token)
    return int(value) if isinstance(value, numbers.Integral) else None


def _released_qwen3vl_prefix_end(token_list, expanded_length: int) -> int:
    """Mirror released Core's Qwen3-VL prefix slice for Krea2 and Mage Flow."""
    template_end = -1
    count_im_start = 0
    ids = [_token_id(token) for token in token_list]
    for index, token_id in enumerate(ids):
        if token_id == _QWEN_IM_START and count_im_start < 2:
            template_end = index
            count_im_start += 1

    if template_end < 0:
        raise ValueError("Could not locate the Qwen3-VL template prefix marker used by released Core.")
    if expanded_length > template_end + 3:
        if ids[template_end + 1:template_end + 3] == [_QWEN_USER, _QWEN_NL]:
            template_end += 3
    return template_end


def _qwen3vl_resized_dimensions(height: int, width: int) -> tuple[int, int]:
    """Replicate released Core's Qwen3-VL resize arithmetic locally."""
    patch_size = 16
    merge_size = 2
    min_pixels = 3136
    max_pixels = 12845056
    factor = patch_size * merge_size
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor

    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    return resized_height, resized_width


VLM_RESOLUTION_MIN = 256
VLM_RESOLUTION_MAX = 3584
VLM_RESOLUTION_STEP = 32


def resolve_vlm_resolution(value) -> int | None:
    """Return a valid equivalent-square side length, or None for Original."""
    if isinstance(value, bool):
        return None
    try:
        resolution = int(value)
    except (TypeError, ValueError):
        return None
    if resolution < VLM_RESOLUTION_MIN or resolution > VLM_RESOLUTION_MAX:
        return None
    return round(resolution / VLM_RESOLUTION_STEP) * VLM_RESOLUTION_STEP


def vlm_target_dimensions(height: int, width: int, resolution: int) -> tuple[int, int]:
    """Fit an aspect-preserving target area and align both axes to Qwen's 32px grid."""
    if height < 1 or width < 1:
        raise ValueError("VLM image dimensions must be positive.")
    scale = math.sqrt((resolution * resolution) / (height * width))
    target_height = max(
        VLM_RESOLUTION_STEP,
        round(height * scale / VLM_RESOLUTION_STEP) * VLM_RESOLUTION_STEP,
    )
    target_width = max(
        VLM_RESOLUTION_STEP,
        round(width * scale / VLM_RESOLUTION_STEP) * VLM_RESOLUTION_STEP,
    )
    return target_height, target_width


def prepare_vlm_image(image: torch.Tensor, resolution) -> torch.Tensor:
    """Resize a BHWC VLM image to a numeric target, or preserve it for Original."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("VLM image must have shape [batch, height, width, channels].")
    target = resolve_vlm_resolution(resolution)
    if target is None:
        return image
    height, width = image.shape[1:3]
    target_height, target_width = vlm_target_dimensions(height, width, target)
    samples = image.movedim(-1, 1)
    return resize_nchw(
        samples, target_width, target_height, "bicubic"
    ).movedim(1, -1)


def prepare_minimax_h3_vlm_video_frames(
    frames: torch.Tensor, resolution,
) -> torch.Tensor:
    """Apply the Qwen3-VL resolution to a chronological BHWC frame batch."""
    prepared = [
        prepare_vlm_image(frames[index:index + 1], resolution)
        for index in range(frames.shape[0])
    ]
    return torch.cat(prepared, dim=0)


def vlm_resolution_samples(
    image: torch.Tensor, resolution, sample_count: int, sample_offset: int = 32
) -> list[int | None]:
    """Build distinct alternating VLM grids without crossing the supported lower bound."""
    count = int(sample_count)
    if count < 1 or count > 15 or count % 2 == 0:
        raise ValueError("VLM resolution samples must be an odd integer from 1 to 15.")
    offset = int(sample_offset)
    if offset < VLM_RESOLUTION_STEP or offset > 512 or offset % VLM_RESOLUTION_STEP:
        raise ValueError("VLM resolution sample offset must be a multiple of 32 from 32 to 512.")
    base = resolve_vlm_resolution(resolution)
    if base is None or count == 1:
        return [base]

    lower_slots = math.ceil((count - 1) / 2)
    base = max(base, VLM_RESOLUTION_MIN + lower_slots * offset)
    base = min(base, VLM_RESOLUTION_MAX)
    height, width = image.shape[1:3]
    samples = []
    grids = set()

    def append_if_distinct(candidate):
        if candidate < VLM_RESOLUTION_MIN or candidate > VLM_RESOLUTION_MAX:
            return
        target_height, target_width = vlm_target_dimensions(
            height, width, candidate
        )
        grid = _qwen3vl_resized_dimensions(target_height, target_width)
        if grid in grids:
            return
        grids.add(grid)
        samples.append(candidate)

    append_if_distinct(base)
    radius = 1
    while len(samples) < count and (
        base - radius * offset >= VLM_RESOLUTION_MIN
        or base + radius * offset <= VLM_RESOLUTION_MAX
    ):
        append_if_distinct(base - radius * offset)
        if len(samples) < count:
            append_if_distinct(base + radius * offset)
        radius += 1
    return samples


def _qwen3vl_image_span(token) -> int | None:
    value = _token_value(token)
    if not isinstance(value, dict) or value.get("type") != "image":
        return None
    image = value.get("data")
    if not torch.is_tensor(image) or image.ndim != 4:
        return None
    height, width = image.shape[1:3]
    resized_height, resized_width = _qwen3vl_resized_dimensions(height, width)
    return (resized_height // 16) * (resized_width // 16) // 4


def _conditioning_token_span(
    token, allow_literal_image_token: bool = False
) -> int | None:
    if is_image_token(token):
        image_span = _qwen3vl_image_span(token)
        if image_span is not None:
            return image_span
        if allow_literal_image_token and isinstance(
            _token_value(token), numbers.Integral
        ):
            return 1
        return None
    value = _token_value(token)
    if not torch.is_tensor(value):
        return 1
    if value.ndim < 1 or value.shape[-1] < 1:
        return None
    return value.numel() // value.shape[-1]


def qwen3vl_visual_grid(image) -> tuple[int, int]:
    """Return the exact post-patch, post-merge Qwen visual token grid."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Visual token layout error: processed image must have shape [batch, height, width, channels].")
    resized_height, resized_width = _qwen3vl_resized_dimensions(*image.shape[1:3])
    return resized_height // 32, resized_width // 32


def visual_fusion_grid(image, visual_length: int, legacy_flat: bool = False) -> tuple[int, int]:
    """Describe the usable visual layout without inventing legacy spatial coordinates."""
    if visual_length < 1:
        raise ValueError("Visual token layout error: visual range must contain at least one token.")
    if legacy_flat:
        return 1, visual_length
    grid = qwen3vl_visual_grid(image)
    if grid[0] * grid[1] != visual_length:
        raise ValueError(f"Visual token layout error: grid {grid} does not match range length {visual_length}.")
    return grid


def build_token_to_conditioning_map(
    token_list, cond_tensor, embedding_key=None
) -> list[tuple[int, int]]:
    """Map raw tokenizer entries to conditioning spans, validating all inferred lengths."""
    cond_len = cond_tensor.shape[1]
    payload_backed_klein = (
        embedding_key in {"qwen3_4b", "qwen3_8b"}
        and any(_qwen3vl_image_span(token) is not None for token in token_list)
    )
    exact_spans = [
        _conditioning_token_span(
            token, allow_literal_image_token=payload_backed_klein
        )
        for token in token_list
    ]
    if not all(span is not None for span in exact_spans):
        raise ValueError("Cannot derive token positions because an image token has no usable Qwen3-VL tensor payload.")

    total_length = sum(exact_spans)
    klein_tail_padding = (
        payload_backed_klein
        and total_length < cond_len
        and cond_len == 512
    )
    if total_length == cond_len or klein_tail_padding:
        prefix_len = 0
    else:
        try:
            prefix_len = _released_qwen3vl_prefix_end(token_list, total_length)
        except ValueError as error:
            raise ValueError(
                "Released Core's Qwen3-VL prefix rule cannot explain the returned conditioning length; "
                "refusing to guess a visual range "
                f"(conditioning_length={cond_len}, expanded_length={total_length})."
            ) from error
    if any(is_image_token(token) for token in token_list[:prefix_len]):
        raise ValueError("Released Core's Qwen3-VL prefix slice crosses a visual token; mapping is unsafe.")

    token_spans = exact_spans[prefix_len:]
    expected_length = sum(token_spans)
    if expected_length != cond_len and not klein_tail_padding:
        image_details = [
            (index, exact_spans[index])
            for index, token in enumerate(token_list)
            if is_image_token(token)
        ]
        nearby_ids = [_token_id(token) for token in token_list[max(0, prefix_len - 3):prefix_len + 4]]
        raise ValueError(
            "Released Core's Qwen3-VL prefix rule does not match the returned conditioning length; "
            "refusing to guess a visual range "
            f"(conditioning_length={cond_len}, expected_length={expected_length}, "
            f"expanded_length={total_length}, prefix_end={prefix_len}, "
            f"image_spans={image_details}, nearby_token_ids={nearby_ids})."
        )

    mapping = []
    current = 0
    retained_index = 0
    for index, token in enumerate(token_list):
        if index < prefix_len:
            mapping.append((-1, -1))
            continue
        size = token_spans[retained_index]
        retained_index += 1
        mapping.append((current, current + size))
        current += size
    if current != cond_len and not klein_tail_padding:
        raise ValueError(f"Token mapping ended at {current}, expected conditioning length {cond_len}.")
    return mapping

_FORMULA_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_FORMULA_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _formula_min(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=b.device, dtype=b.dtype)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
        return torch.minimum(a, b)
    return min(a, b)


def _formula_max(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, device=b.device, dtype=b.dtype)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
        return torch.maximum(a, b)
    return max(a, b)


_FORMULA_FUNCTIONS = {
    "abs": abs,
    "min": _formula_min,
    "max": _formula_max,
    "clamp": torch.clamp,
}


def evaluate_tensor_expression(expression: str, variables: dict):
    """Evaluate the documented tensor-expression grammar without Python eval."""
    try:
        root = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc.msg}") from exc

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown expression variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_BINOPS:
            return _FORMULA_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _FORMULA_UNARYOPS:
            return _FORMULA_UNARYOPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORMULA_FUNCTIONS:
            if node.keywords:
                raise ValueError("Keyword arguments are not supported in expressions.")
            return _FORMULA_FUNCTIONS[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    result = visit(root)
    if torch.is_tensor(result) and not torch.isfinite(result).all():
        raise ValueError("Expression produced NaN or infinite values.")
    return result


def evaluate_formula(expression: str, processed_images: dict) -> torch.Tensor:
    try:
        result = evaluate_tensor_expression(expression, processed_images)
        if not torch.is_tensor(result):
            reference = next(iter(processed_images.values()), None)
            if reference is None:
                raise ValueError("A visual formula requires at least one image variable.")
            result = torch.full_like(reference, float(result))
        return torch.clamp(result, 0.0, 1.0)
    except Exception as e:
        raise RuntimeError(f"Error evaluating visual math expression '{expression}': {e}") from e

def evaluate_conditioning_formula(expression: str, sequence_tensors: dict, pooled_tensors: dict, padding_method: str = "zero-pad") -> tuple:
    # Preprocess classic weighting syntax inside math expression to math scaling
    # e.g., (image_input_1:10) -> (image_input_1 * 10)
    expression = re.sub(
        r"\(\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.-]+)\s*\)",
        r"(\1 * \2)",
        expression
    )

    # Determine max sequence length across all tensors
    max_len = max(tensor.shape[1] for tensor in sequence_tensors.values())

    # Pad/interpolate all tensors to match max length exactly
    aligned_sequence_tensors = {}
    for name, tensor in sequence_tensors.items():
        if tensor.shape[1] < max_len:
            if padding_method == "interpolate":
                tensor_perm = tensor.permute(0, 2, 1)
                tensor_interp = F.interpolate(tensor_perm, size=max_len, mode='linear', align_corners=False)
                tensor = tensor_interp.permute(0, 2, 1)
            else: # zero-pad
                pad_size = max_len - tensor.shape[1]
                padding = torch.zeros((tensor.shape[0], pad_size, tensor.shape[2]), device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=1)
        # Cast the padded tensor to the target device via Comfy's non-blocking, aimdo-aware pipeline
        aligned_sequence_tensors[name] = comfy.model_management.cast_to_device(tensor, tensor.device, tensor.dtype)

    try:
        C_blended = evaluate_tensor_expression(expression, aligned_sequence_tensors)
        P_blended = None
        if any(v is not None for v in pooled_tensors.values()):
            pooled_variables = {name: tensor for name, tensor in pooled_tensors.items() if tensor is not None}
            missing = set(aligned_sequence_tensors) - set(pooled_variables)
            if missing:
                raise ValueError(f"Formula references conditioning sources without pooled outputs: {sorted(missing)}")
            P_blended = evaluate_tensor_expression(expression, pooled_variables)
        if not torch.is_tensor(C_blended) or C_blended.ndim != 3:
            raise ValueError("Conditioning formula must produce a [batch, tokens, channels] tensor.")
        return C_blended, P_blended
    except Exception as e:
        raise RuntimeError(f"Error evaluating conditioning math expression '{expression}': {e}") from e

def reconstruct_2d_grid(N: int) -> tuple:
    """
    Determines the closest 2D grid dimensions (H, W) for N tokens.
    """
    root = int(math.sqrt(N))
    if root * root == N:
        return root, root
    for w in range(root, 0, -1):
        if N % w == 0:
            return N // w, w
    return N, 1

SPATIAL_FUSION_METHODS = {"spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"}
VISUAL_FUSION_METHODS = SPATIAL_FUSION_METHODS | {"linear"}


def _spatial_perturbation_seed(seed: int) -> int:
    payload = f"utils-collection-spatial-perturbation:{seed}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _perturb_spatial_assignments(mask: torch.Tensor, amount: float, seed: int) -> torch.Tensor:
    """Exchange differently labelled cells without changing any source count."""
    if amount <= 0.0 or mask.numel() < 2:
        return mask

    flat = mask.flatten().clone()
    requested_pairs = int(flat.numel() * amount) // 2
    if requested_pairs < 1:
        return mask

    generator = torch.Generator(device=flat.device).manual_seed(_spatial_perturbation_seed(seed))
    randomized = torch.randperm(flat.numel(), generator=generator, device=flat.device).tolist()
    labels = flat.tolist()
    buckets = {}
    for index in randomized:
        buckets.setdefault(labels[index], []).append(index)

    changed_pairs = 0
    while changed_pairs < requested_pairs:
        available = [label for label, indices in buckets.items() if indices]
        if len(available) < 2:
            break
        available.sort(key=lambda label: (-len(buckets[label]), label))
        first_label, second_label = available[:2]
        first_index = buckets[first_label].pop()
        second_index = buckets[second_label].pop()
        first_value = flat[first_index].clone()
        flat[first_index] = flat[second_index]
        flat[second_index] = first_value
        changed_pairs += 1
    return flat.reshape(mask.shape)


def _cleanup_primary_pairs(mask: torch.Tensor) -> torch.Tensor:
    """Swap complementary primary islands and holes while preserving source counts."""
    h, w = mask.shape
    primary = mask.eq(0)
    padded = F.pad(primary, (1, 1, 1, 1))
    neighbors = torch.stack([
        padded[row:row + h, column:column + w]
        for row in range(3)
        for column in range(3)
        if (row, column) != (1, 1)
    ])
    isolated = (primary & ~neighbors.any(dim=0)).flatten().nonzero().flatten()
    holes = (~primary & neighbors.all(dim=0)).flatten().nonzero().flatten()
    pair_count = min(isolated.numel(), holes.numel())
    if pair_count == 0:
        return mask

    flat = mask.flatten().clone()
    island_indices = isolated[:pair_count]
    hole_indices = holes[:pair_count]
    hole_values = flat[hole_indices].clone()
    flat[hole_indices] = flat[island_indices]
    flat[island_indices] = hole_values
    return flat.reshape(mask.shape)


def _validate_spatial_fusion_mask(N, num_sources, method, block_size, dither_ratio, seed, grid_shape, dither_secondary_pattern, spatial_perturbation):
    if N < 0:
        raise ValueError("Visual token count cannot be negative.")
    if num_sources < 1:
        raise ValueError("Visual fusion requires at least one source.")
    if method not in SPATIAL_FUSION_METHODS:
        raise ValueError(f"Unsupported spatial fusion method: {method}")
    if method == "spatial-block-interleave" and block_size < 1:
        raise ValueError("Visual block size must be at least 1.")
    if method == "spatial-dither-random" and not 0.0 <= dither_ratio <= 1.0:
        raise ValueError("Dither ratio must be between 0.0 and 1.0.")
    if not 0.0 <= spatial_perturbation <= 1.0:
        raise ValueError("Spatial perturbation must be between 0.0 and 1.0.")
    if not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("Visual fusion seed must be between 0 and 18446744073709551615.")
    if num_sources == 1:
        return None

    h, w = grid_shape if grid_shape is not None else reconstruct_2d_grid(N)
    if h < 1 or w < 1 or h * w != N:
        raise ValueError(f"Visual token layout error: grid {grid_shape} does not contain {N} tokens.")
    if method == "spatial-dither-random":
        if dither_secondary_pattern not in {"checkerboard", "block-interleave", "dither-random-reverse", "dither-random-forward"}:
            raise ValueError(f"Unsupported dither secondary pattern: {dither_secondary_pattern}")
        if block_size < 1:
            raise ValueError("Visual block size must be at least 1.")
    return h, w


def generate_spatial_fusion_mask(N: int, num_sources: int, method: str, block_size: int = 2, dither_ratio: float = 0.5, device: str = "cpu", seed: int = 0, grid_shape=None, dither_secondary_pattern: str = "checkerboard", dither_mask_cleanup: bool = False, spatial_perturbation: float = 0.0) -> torch.Tensor:
    """Generate a seeded token-to-source mapping after validating its parameters."""
    grid = _validate_spatial_fusion_mask(N, num_sources, method, block_size, dither_ratio, seed, grid_shape, dither_secondary_pattern, spatial_perturbation)
    if num_sources == 1:
        return torch.zeros(N, dtype=torch.long, device=device)
    h, w = grid
    rows = torch.arange(h, device=device).unsqueeze(1)
    columns = torch.arange(w, device=device).unsqueeze(0)

    if method == "spatial-checkerboard":
        mask = (rows + columns) % num_sources
    elif method == "spatial-block-interleave":
        mask = (rows // block_size + columns // block_size) % num_sources
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        if dither_secondary_pattern == "dither-random-reverse":
            mask = torch.full((N,), num_sources - 1, dtype=torch.long, device=device)
            for base_source in range(num_sources - 2, -1, -1):
                random = torch.rand(N, generator=generator, device=device)
                mask = torch.where(random < dither_ratio, base_source, mask)
            mask = mask.reshape(h, w)
        elif dither_secondary_pattern == "dither-random-forward":
            mask = torch.zeros(N, dtype=torch.long, device=device)
            for secondary_source in range(1, num_sources):
                random = torch.rand(N, generator=generator, device=device)
                mask = torch.where(
                    random < dither_ratio, mask, secondary_source
                )
            mask = mask.reshape(h, w)
        else:
            random = torch.rand(N, generator=generator, device=device)
            if dither_secondary_pattern == "block-interleave":
                secondary = rows // block_size + columns // block_size
            else:
                secondary = rows + columns
            other_sources = 1 + (secondary % (num_sources - 1)).flatten()
            mask = torch.where(random < dither_ratio, 0, other_sources).reshape(h, w)

    mask = _perturb_spatial_assignments(mask, spatial_perturbation, seed)
    if method == "spatial-dither-random" and dither_mask_cleanup and 0.0 < dither_ratio < 1.0:
        mask = _cleanup_primary_pairs(mask)
    return mask.flatten()


def _visual_fusion_mask(config, grid_shape, num_sources, mask_device, output_device, mask_cache):
    method = config.get("visual_fusion_method", "spatial-checkerboard")
    block_size = config.get("visual_block_size", 2)
    dither_ratio = config.get("dither_ratio", 0.5)
    seed = config.get("seed", 0)
    secondary = config.get("dither_secondary_pattern", "checkerboard")
    cleanup = config.get("dither_mask_cleanup", False)
    perturbation = config.get("spatial_perturbation", 0.0)
    key = (tuple(grid_shape), num_sources, method, secondary, cleanup, perturbation, block_size, dither_ratio, seed)
    if key not in mask_cache:
        mask_cache[key] = generate_spatial_fusion_mask(grid_shape[0] * grid_shape[1], num_sources, method, block_size, dither_ratio, mask_device, seed, grid_shape, secondary, cleanup, perturbation)
    return mask_cache[key].to(output_device)


def _align_visual_sources(sources, grids, canonical_grid, compute_float):
    canonical_length = canonical_grid[0] * canonical_grid[1]
    aligned = []
    for source, grid in zip(sources, grids):
        value = source.to(dtype=torch.float32) if compute_float else source
        if grid != canonical_grid:
            value = F.interpolate(
                value.reshape(grid[0], grid[1], -1).permute(2, 0, 1)[None],
                size=canonical_grid,
                mode="nearest",
            )[0].permute(1, 2, 0).reshape(canonical_length, -1)
        aligned.append(value)
    return torch.stack(aligned, dim=1)


def _baseline_visual_weights(
    visual_fusion_config, canonical_grid, source_count, mask_device, output_device, mask_cache
):
    method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")
    length = canonical_grid[0] * canonical_grid[1]
    if method == "linear":
        return torch.full(
            (length, source_count),
            1.0 / source_count,
            device=output_device,
            dtype=torch.float32,
        )
    mask = _visual_fusion_mask(
        visual_fusion_config,
        canonical_grid,
        source_count,
        mask_device,
        output_device,
        mask_cache,
    )
    return F.one_hot(mask, num_classes=source_count).to(dtype=torch.float32)


def fuse_visual_token_sources(
    sources,
    visual_fusion_config,
    mask_device,
    mask_cache=None,
    expected_length=None,
    source_grids=None,
    *,
    weights_override=None,
    return_weights=False,
    cache=None,
):
    if not sources:
        raise ValueError("Visual fusion requires at least one visual token source.")

    method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")
    if method not in VISUAL_FUSION_METHODS:
        raise ValueError(f"Unsupported visual fusion method: {method}")
    if any(source.ndim != 2 for source in sources):
        raise ValueError("Visual fusion sources must have shape [tokens, dimensions].")
    if any(source.shape[1] != sources[0].shape[1] for source in sources[1:]):
        raise ValueError("Visual fusion sources must have matching embedding dimensions.")
    if any(source.device != sources[0].device for source in sources[1:]):
        raise ValueError("Visual fusion sources must be on the same device.")
    if source_grids is None or len(source_grids) != len(sources):
        raise ValueError("Visual token layout error: every fusion source requires an explicit grid.")
    grids = [tuple(grid) for grid in source_grids]
    for source, grid in zip(sources, grids):
        if len(grid) != 2 or grid[0] < 1 or grid[1] < 1 or grid[0] * grid[1] != source.shape[0]:
            raise ValueError(f"Visual token layout error: grid {grid} is inconsistent with {source.shape[0]} tokens.")
    canonical_grid = grids[0]
    canonical_length = canonical_grid[0] * canonical_grid[1]
    if expected_length is not None and canonical_length != expected_length:
        raise ValueError(f"Visual token layout mismatch: expected {expected_length} tokens, received canonical grid {canonical_grid}.")

    if weights_override is None and method != "linear":
        _validate_spatial_fusion_mask(
            canonical_length, len(sources), method,
            visual_fusion_config.get("visual_block_size", 2), visual_fusion_config.get("dither_ratio", .5),
            visual_fusion_config.get("seed", 0), canonical_grid,
            visual_fusion_config.get("dither_secondary_pattern", "checkerboard"),
            visual_fusion_config.get("spatial_perturbation", 0.),
        )

    output_dtype = sources[0].dtype
    stacked = _align_visual_sources(
        sources, grids, canonical_grid, method == "linear"
    )
    if mask_cache is None:
        mask_cache = {}
    if weights_override is None:
        baseline = _baseline_visual_weights(
            visual_fusion_config,
            canonical_grid,
            len(sources),
            mask_device,
            stacked.device,
            mask_cache,
        )
        weights = baseline
    else:
        weights = weights_override.to(device=stacked.device, dtype=torch.float32)
        if weights.shape != stacked.shape[:2]:
            raise ValueError("Visual fusion weights do not match the aligned visual grid.")

    fused = (stacked.to(dtype=torch.float32) * weights[:, :, None]).sum(dim=1)
    fused = fused.to(dtype=output_dtype)
    return (fused, weights) if return_weights else fused


def fuse_deepstack_layers(deepstack_tensors, visual_fusion_config, device, mask_cache, expected_length, source_grids, cache=None):
    active_keys = sorted(deepstack_tensors)
    if not active_keys:
        return None

    num_layers = len(deepstack_tensors[active_keys[0]])
    if any(len(deepstack_tensors[key]) != num_layers for key in active_keys[1:]):
        raise ValueError("Visual fusion sources produced different DeepStack layer counts.")

    blended = []
    for layer in range(num_layers):
        sources = [deepstack_tensors[key][layer].to(device=device) for key in active_keys]
        blended.append(fuse_visual_token_sources(sources, visual_fusion_config, device, mask_cache, expected_length, source_grids, cache=cache))
    return blended


def _active_clip_model(clip):
    stage = clip.cond_stage_model
    model_name = getattr(stage, "clip", None)
    model = getattr(stage, model_name, None) if isinstance(model_name, str) else None
    if model is None:
        model = getattr(stage, "clip_model", None) or getattr(stage, "clip_d", None) or stage
    if not hasattr(model, "process_tokens") or not hasattr(model, "transformer"):
        raise ValueError("TokenFusion requires a Core multimodal text encoder with process_tokens support.")
    return model


def _token_rows_for_process(tokens):
    key = next(iter(tokens))
    return [[entry[0] for entry in batch] for batch in tokens[key]]


def _encode_preprocessed_clip_model(clip_model, embeds, attention_mask, num_tokens, embeds_info, cache=None, visual_encoder_path="grid-deepstack", hooks=None):
    if cache is not None:
        return cache.encode_preprocessed(
            clip_model, embeds, attention_mask, num_tokens, embeds_info, visual_encoder_path, hooks,
            lambda: _encode_preprocessed_clip_model(clip_model, embeds, attention_mask, num_tokens, embeds_info),
        )
    attention_mask_model = attention_mask if clip_model.enable_attention_masks else None
    if isinstance(clip_model.layer, list):
        intermediate_output = clip_model.layer
    elif clip_model.layer == "all":
        intermediate_output = "all"
    else:
        intermediate_output = clip_model.layer_idx
    outputs = clip_model.transformer(
        None,
        attention_mask_model,
        embeds=embeds,
        num_tokens=num_tokens,
        intermediate_output=intermediate_output,
        final_layer_norm_intermediate=clip_model.layer_norm_hidden_state,
        dtype=torch.float32,
        embeds_info=embeds_info,
    )
    conditioning = outputs[0].float() if clip_model.layer == "last" else outputs[1].float()
    if clip_model.zero_out_masked:
        conditioning *= attention_mask.unsqueeze(-1).float()
    pooled = None
    if len(outputs) >= 3:
        if not clip_model.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
            pooled = outputs[3].float()
        elif outputs[2] is not None:
            pooled = outputs[2].float()
    metadata = {"pooled_output": pooled}
    if clip_model.return_attention_masks:
        metadata["attention_mask"] = attention_mask
    minimax_tags = getattr(clip_model.transformer, "last_token_tags", None)
    if getattr(clip_model.transformer, "model_type", None) == "qwen3vl_32b":
        if conditioning.ndim != 3:
            raise ValueError("MiniMax H3 TokenFusion requires a three-dimensional conditioning tensor.")
        minimax_tags = token_tags_from_embeds_info(conditioning.shape[1], embeds_info)
    if minimax_tags is not None:
        metadata["minimax_token_tags"] = minimax_tags
    return conditioning.to(comfy.model_management.intermediate_device()), metadata


def _normalize_token_fused_conditioning(clip, tokens, conditioning, metadata):
    """Apply outer text-encoder shaping normally performed after the inner Qwen encode."""
    key = next(iter(tokens))
    if key != "qwen3vl_4b" or conditioning.ndim != 4:
        return conditioning, metadata

    token_pairs = tokens[key][0]
    template_end = -1
    im_start_count = 0
    for index, value in enumerate(token_pairs):
        token = value[0]
        if not torch.is_tensor(token) and isinstance(token, numbers.Integral):
            if token == 151644 and im_start_count < 2:
                template_end = index
                im_start_count += 1
    if conditioning.shape[2] > template_end + 3:
        if token_pairs[template_end + 1][0] == 872 and token_pairs[template_end + 2][0] == 198:
            template_end += 3
    conditioning = conditioning[:, :, template_end:]
    batch, layers, sequence, hidden = conditioning.shape
    conditioning = conditioning.permute(0, 2, 1, 3).reshape(
        batch, sequence, layers * hidden
    )
    attention_mask = metadata.get("attention_mask")
    if torch.is_tensor(attention_mask):
        attention_mask = attention_mask[:, template_end:]
        if attention_mask.sum() == torch.numel(attention_mask):
            metadata.pop("attention_mask", None)
        else:
            metadata["attention_mask"] = attention_mask
    return conditioning, metadata


def _prepare_fused_visual_inputs(
    clip_model,
    token_sources,
    visual_fusion_config,
    device,
    source_grids=None,
    visual_indices=None,
):
    processed = [clip_model.process_tokens(_token_rows_for_process(tokens), device) for tokens in token_sources]
    if any(len(value[3]) == 0 for value in processed):
        raise ValueError("TokenFusion source produced no multimodal metadata.")
    visual_indices = visual_indices or [0] * len(processed)
    image_entries = []
    for value, visual_index in zip(processed, visual_indices):
        images = [entry for entry in value[3] if entry.get("type") == "image"]
        if not 0 <= visual_index < len(images):
            raise ValueError("TokenFusion could not locate the requested visual source.")
        image_entries.append(images[visual_index])

    if source_grids is None:
        source_grids = []
        for entry in image_entries:
            grid = (entry.get("extra") or {}).get("grid")
            values = grid.reshape(-1).tolist() if torch.is_tensor(grid) else list(grid or [])
            if len(values) < 3:
                raise ValueError("TokenFusion source is missing exact visual grid metadata.")
            source_grids.append((int(values[-2]) // 2, int(values[-1]) // 2))

    canonical_embeds, attention_mask, num_tokens, canonical_info = processed[0]
    canonical_entry = image_entries[0]
    start = canonical_entry["index"]
    size = canonical_entry["size"]
    sources = []
    for (embeds, _, _, _), entry in zip(processed, image_entries):
        source_start = entry["index"]
        source_size = entry["size"]
        if embeds.shape[-1] != canonical_embeds.shape[-1]:
            raise ValueError("TokenFusion sources produced incompatible embedding dimensions.")
        sources.append(embeds[0, source_start:source_start + source_size].to(device))

    mask_cache = {}
    fused = fuse_visual_token_sources(
        sources,
        visual_fusion_config,
        device,
        mask_cache,
        size,
        source_grids,
    )
    fused_embeds = canonical_embeds.clone()
    fused_embeds[0, start:start + size] = fused
    fused_info = [dict(entry) for entry in canonical_info]
    canonical_position = next(
        index for index, entry in enumerate(fused_info)
        if entry.get("type") == "image" and entry.get("index") == start
    )
    fused_entry = dict(fused_info[canonical_position])
    fused_entry["extra"] = dict(fused_entry.get("extra") or {})
    deepstacks = {
        index: entry.get("extra", {}).get("deepstack")
        for index, entry in enumerate(image_entries)
    }
    if all(deepstacks.values()):
        fused_entry["extra"]["deepstack"] = fuse_deepstack_layers(
            deepstacks,
            visual_fusion_config,
            device,
            mask_cache,
            size,
            source_grids,
        )
    fused_info[canonical_position] = fused_entry
    return fused_embeds, attention_mask, num_tokens, fused_info, fused


def encode_token_fused_visual_sources(
    clip,
    token_sources,
    visual_fusion_config,
    source_grids=None,
    visual_indices=None,
    visual_encoder_path="grid-deepstack",
):
    """Fuse pre-transformer visual tokens, then run one language encode per CLIP schedule."""
    if not token_sources:
        raise ValueError("TokenFusion requires at least one token source.")
    clip.cond_stage_model.reset_clip_options()
    if clip.layer_idx is not None:
        clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})
    clip.load_model(token_sources[0])
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"execution_device": device})
    clip_model = _active_clip_model(clip)

    def encode_once():
        prepared = _prepare_fused_visual_inputs(
            clip_model,
            token_sources,
            visual_fusion_config,
            device,
            source_grids,
            visual_indices,
        )
        embeds, attention_mask, num_tokens, embeds_info, fused = prepared
        with qwen3vl_visual_encoder_path(clip, visual_encoder_path):
            conditioning, metadata = _encode_preprocessed_clip_model(
                clip_model, embeds, attention_mask, num_tokens, embeds_info
            )
            conditioning, metadata = _normalize_token_fused_conditioning(
                clip, token_sources[0], conditioning, metadata
            )
        return conditioning, metadata, embeds, embeds_info, fused

    schedules = None
    hooks = clip.patcher.forced_hooks
    if hooks is not None and clip.use_clip_schedule:
        schedules = hooks.get_hooks_for_clip_schedule()
    output = []
    last_prepared = None
    with comfy.model_management.cuda_device_context(device):
        if schedules is None:
            conditioning, metadata, *last_prepared = encode_once()
            clip.add_hooks_to_dict(metadata)
            output.append([conditioning, metadata])
        else:
            hooks.reset()
            clip.patcher.patch_hooks(None)
            for time_range, scheduled_hooks in schedules:
                for hook, keyframe in scheduled_hooks:
                    hook.hook_keyframe._current_keyframe = keyframe
                clip.patcher.patch_hooks(hooks)
                conditioning, metadata, *last_prepared = encode_once()
                metadata["clip_start_percent"] = time_range[0]
                metadata["clip_end_percent"] = time_range[1]
                clip.add_hooks_to_dict(metadata)
                output.append([conditioning, metadata])
            hooks.reset()
    if visual_fusion_config.get("save_blended_embeds", False):
        embeds, embeds_info, fused = last_prepared
        image_entry = next(entry for entry in embeds_info if entry.get("type") == "image")
        start = image_entry["index"]
        block = torch.cat([embeds[:, start - 1:start], fused[None], embeds[:, start + image_entry["size"]:start + image_entry["size"] + 1]], dim=1)
        save_blended_visual_embeddings(
            [block[batch].detach() for batch in range(block.shape[0])],
            visual_fusion_config,
            next(iter(token_sources[0])),
        )
    fused_tokens = last_prepared[2].shape[0] if last_prepared else 0
    logging.info(
        "TokenFusion visual-token fusion complete: fused %d visual sources into 1 block (%d visual tokens), then ran %d conditioning encode%s.",
        len(token_sources),
        fused_tokens,
        len(output),
        "s" if len(output) != 1 else "",
    )
    return output


def encode_token_fused_visual_slots(
    clip,
    canonical_tokens,
    slot_sources,
    visual_fusion_config,
    visual_encoder_path="grid-deepstack",
    cache=None,
):
    """Fuse one or more canonical visual slots before one transformer encode."""
    clip.cond_stage_model.reset_clip_options()
    if clip.layer_idx is not None:
        clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})
    clip.load_model(canonical_tokens)
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"execution_device": device})
    clip_model = _active_clip_model(clip)

    def encode_once():
        canonical = clip_model.process_tokens(_token_rows_for_process(canonical_tokens), device)
        embeds, attention_mask, num_tokens, info = canonical
        fused_embeds = embeds.clone()
        fused_info = [dict(entry) for entry in info]
        canonical_images = [entry for entry in fused_info if entry.get("type") == "image"]
        saved_blocks = []
        for visual_index, alternative_tokens in slot_sources:
            if not 0 <= visual_index < len(canonical_images):
                raise ValueError("TokenFusion could not locate the canonical MiniMax picture slot.")
            canonical_entry = canonical_images[visual_index]
            alternatives = [
                clip_model.process_tokens(_token_rows_for_process(tokens), device)
                for tokens in alternative_tokens
            ]
            selected = [canonical_entry]
            selected_embeds = [embeds]
            for value in alternatives:
                images = [entry for entry in value[3] if entry.get("type") == "image"]
                if not 0 <= visual_index < len(images):
                    raise ValueError("TokenFusion branch changed the MiniMax picture layout.")
                selected.append(images[visual_index])
                selected_embeds.append(value[0])
            grids = []
            sources = []
            for source_embeds, entry in zip(selected_embeds, selected):
                grid = (entry.get("extra") or {}).get("grid")
                values = grid.reshape(-1).tolist() if torch.is_tensor(grid) else list(grid or [])
                if len(values) < 3:
                    raise ValueError("TokenFusion MiniMax source is missing visual grid metadata.")
                grids.append((int(values[-2]) // 2, int(values[-1]) // 2))
                start = entry["index"]
                sources.append(source_embeds[0, start:start + entry["size"]].to(device))
            start = canonical_entry["index"]
            size = canonical_entry["size"]
            mask_cache = {}
            fused = fuse_visual_token_sources(
                sources, visual_fusion_config, device, mask_cache, size, grids, cache=cache
            )
            fused_embeds[0, start:start + size] = fused
            replacement = dict(canonical_entry)
            replacement["extra"] = dict(replacement.get("extra") or {})
            deepstacks = {
                index: entry.get("extra", {}).get("deepstack")
                for index, entry in enumerate(selected)
            }
            if all(deepstacks.values()):
                replacement["extra"]["deepstack"] = fuse_deepstack_layers(
                    deepstacks, visual_fusion_config, device, mask_cache, size, grids, cache=cache
                )
            position = next(
                index for index, entry in enumerate(fused_info)
                if entry is canonical_entry
            )
            fused_info[position] = replacement
            canonical_images[visual_index] = replacement
            saved_blocks.append((start, size, fused))
        with qwen3vl_visual_encoder_path(clip, visual_encoder_path):
            conditioning, metadata = _encode_preprocessed_clip_model(
                clip_model, fused_embeds, attention_mask, num_tokens, fused_info,
                **({"cache": cache, "visual_encoder_path": visual_encoder_path, "hooks": clip.patcher.forced_hooks} if cache is not None else {}),
            )
            conditioning, metadata = _normalize_token_fused_conditioning(
                clip, canonical_tokens, conditioning, metadata
            )
            if next(iter(canonical_tokens)) == "qwen3vl_32b":
                if conditioning.ndim != 3:
                    raise ValueError(
                        "MiniMax H3 TokenFusion requires a three-dimensional conditioning tensor."
                    )
                metadata["minimax_token_tags"] = token_tags_from_embeds_info(
                    conditioning.shape[1], fused_info
                )
                logging.info(
                    "MiniMax H3 TokenFusion metadata rebuilt: conditioning=%s tags=%s embeds=%s pictures=%s",
                    tuple(conditioning.shape),
                    tuple(metadata["minimax_token_tags"].shape),
                    tuple(fused_embeds.shape),
                    [(entry["index"], entry["size"]) for entry in fused_info if entry.get("type") == "image"],
                )
        return conditioning, metadata, fused_embeds, saved_blocks

    hooks = clip.patcher.forced_hooks
    schedules = hooks.get_hooks_for_clip_schedule() if hooks is not None and clip.use_clip_schedule else None
    output = []
    last_payload = None
    with comfy.model_management.cuda_device_context(device):
        if schedules is None:
            conditioning, metadata, *last_payload = encode_once()
            clip.add_hooks_to_dict(metadata)
            output.append([conditioning, metadata])
        else:
            hooks.reset()
            clip.patcher.patch_hooks(None)
            for time_range, scheduled_hooks in schedules:
                for hook, keyframe in scheduled_hooks:
                    hook.hook_keyframe._current_keyframe = keyframe
                clip.patcher.patch_hooks(hooks)
                conditioning, metadata, *last_payload = encode_once()
                metadata["clip_start_percent"], metadata["clip_end_percent"] = time_range
                clip.add_hooks_to_dict(metadata)
                output.append([conditioning, metadata])
            hooks.reset()
    if visual_fusion_config.get("save_blended_embeds", False) and last_payload:
        fused_embeds, saved_blocks = last_payload
        for output_index, (start, size, fused) in enumerate(saved_blocks):
            config = visual_fusion_config
            if output_index:
                config = dict(config)
                stem, suffix = os.path.splitext(config.get("save_path", "blended_visual_embeds"))
                config["save_path"] = f"{stem}_{output_index + 1}{suffix}"
            block = torch.cat([fused_embeds[:, start - 1:start], fused[None], fused_embeds[:, start + size:start + size + 1]], dim=1)
            save_blended_visual_embeddings(
                [block[batch].detach() for batch in range(block.shape[0])],
                config,
                next(iter(canonical_tokens)),
            )
    completed_blocks = last_payload[1] if last_payload else []
    summary = ", ".join(
        f"Picture {visual_index + 1}: {len(alternatives) + 1} sources -> {block[1]} tokens"
        for (visual_index, alternatives), block in zip(slot_sources, completed_blocks)
    )
    logging.info(
        "TokenFusion visual-token fusion complete: %s, then ran %d conditioning encode%s.",
        summary or "no Picture slots",
        len(output),
        "s" if len(output) != 1 else "",
    )
    return output


def encode_token_fused_text(
    clip,
    text,
    images,
    visual_fusion_config,
    tokenize_source,
    visual_encoder_path="grid-deepstack",
):
    clean_text = text
    biases = []
    if "(" in text and ")" in text:
        clean_text = ""
        for segment, strength in _contextual_token_weights(text):
            start_tokens = tokenize_source(clean_text, images[0])
            key = next(iter(start_tokens))
            start = len(start_tokens[key][0])
            clean_text += segment
            end_tokens = tokenize_source(clean_text, images[0])
            end = len(end_tokens[key][0])
            if strength != 1.0 and end > start:
                if not math.isfinite(strength):
                    raise ValueError("Contextual prompt weights must be finite.")
                biases.append((start, end, float(strength)))
    token_sources = [tokenize_source(clean_text, image) for image in images]
    conditioning = encode_token_fused_visual_sources(
        clip,
        token_sources,
        visual_fusion_config,
        visual_encoder_path=visual_encoder_path,
    )
    if biases:
        token_list = token_sources[0][next(iter(token_sources[0]))][0]
        for tensor, _ in conditioning:
            mapping = build_token_to_conditioning_map(token_list, tensor)
            for start_token, end_token, strength in biases:
                if start_token >= len(mapping):
                    continue
                start = mapping[start_token][0]
                end = mapping[min(end_token - 1, len(mapping) - 1)][1]
                if 0 <= start < end:
                    tensor[:, start:end, :] *= strength
    return conditioning, token_sources


def execute_token_fusion_visual_conditioning(
    clip,
    prompt,
    images,
    visual_fusion_config,
    vlm_resolution,
    system_prompt=None,
    multiplier=1.0,
    skip_template=True,
):
    processed = [prepare_vlm_image(image, vlm_resolution) for image in images]
    prepared_prompt, _ = prepare_image_placeholder_prompt(
        prompt,
        image_count=len(processed),
        fusion_active=True,
        context="TokenFusion",
    )
    if not any(tag in prepared_prompt for tag in ("<|image_pad|>", "<|image|>", "<|vision_start|>")):
        prepared_prompt = VISION_BLOCK + prepared_prompt
    minimax_h3 = is_minimax_h3_text_encoder(clip)
    if system_prompt is None:
        full_prompt = prepared_prompt
    elif minimax_h3:
        full_prompt = format_minimax_h3_prompt(prepared_prompt, system_prompt)
    else:
        full_prompt = (
            "<|im_start|>user\n<|im_end|>\n"
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prepared_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def tokenize_source(text, image):
        if minimax_h3:
            return tokenize_minimax_h3_prompt(clip, text, [image])
        kwargs = {"skip_template": True} if skip_template else {}
        return clip.tokenize(text, images=[image], **kwargs)

    conditioning, token_sources = encode_token_fused_text(
        clip,
        full_prompt,
        processed,
        visual_fusion_config,
        tokenize_source,
        visual_fusion_config.get("visual_encoder_path", "grid-deepstack"),
    )
    if multiplier != 1.0:
        for tensor, metadata in conditioning:
            tensor *= multiplier
            pooled = metadata.get("pooled_output")
            if pooled is not None:
                metadata["pooled_output"] = pooled * multiplier
    return conditioning, token_sources


# =====================================================================
# Visual embedding exports contain one complete prompt-reloadable visual block:
# the native vision-start input vector, expanded visual input vectors, and the
# native vision-end input vector. They never contain template or prompt text.
def save_blended_visual_embeddings(
    blended_vis_all_batches: list,
    visual_fusion_config: dict,
    embedding_key: str = "qwen3vl_8b"
) -> None:
    """
    Saves complete visual prompt blocks as standalone Core embeddings.

    The input blocks contain vision-start, expanded visual rows, and vision-end.
    They contain no surrounding prompt or template text. Core's `embedding:`
    loader receives the contiguous block under the active VLM embedding key.

    Parameters:
        blended_vis_all_batches: List of 2D torch.Tensor representing the
                                   visual blocks across the execution batches.
        visual_fusion_config: Dictionary containing target path options.
        embedding_key: The string identifier (e.g., 'qwen3vl_8b' or 'krea2_vlm')
                       required by the model checkpoint reader.
    """

    if not blended_vis_all_batches:
        raise ValueError("[save_blended_visual_embeddings] Cannot save empty visual token list.")

    # Stack batches into [B, max_vis_len, D] and move to CPU in float32 precision
    stacked_vis = torch.stack(blended_vis_all_batches, dim=0).to(device="cpu", dtype=torch.float32)

    # Squeeze batch dimension if B == 1 to keep it as a standard 2D embedding tensor
    if stacked_vis.shape[0] == 1:
        stacked_vis = stacked_vis.squeeze(0)

    save_name = visual_fusion_config.get("save_path", "blended_visual_embeds").strip()
    if not save_name.endswith(".safetensors"):
        save_name += ".safetensors"

    embed_paths = folder_paths.get_folder_paths("embeddings")
    if not embed_paths:
        raise ValueError("No ComfyUI embeddings directory is configured.")
    embeddings_dir = embed_paths[0]
    full_save_path = resolve_embedding_output_path(embeddings_dir, save_name)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    # State dict must contain exactly one layer matching the VLM's dynamic embedding_key
    state_dict = {embedding_key: stacked_vis.contiguous()}
    save_file(state_dict, full_save_path)
    logging.info(f"[UC_VisualFusionConfig] Saved visual prompt block as {embedding_key} embedding to: {full_save_path}")


def _visual_token_embedding_blocks(clip, tokens, device: str, cache=None) -> list[dict]:
    """Return validated Qwen visual blocks from one already-tokenized source."""
    if cache is not None:
        clip.load_model(tokens)
    cond_stage = clip.cond_stage_model
    clip_model = getattr(cond_stage, cond_stage.clip)
    key_name = next(iter(tokens))
    token_batches = tokens[key_name]
    tokens_only = [[token[0] for token in batch] for batch in token_batches]
    embeds, _, _, embeds_info = clip_model.process_tokens(tokens_only, device)
    visual_entries = [entry for entry in embeds_info if entry["type"] == "image"]
    image_positions = [
        index
        for index, token in enumerate(tokens_only[0])
        if isinstance(token, dict) and token.get("type") == "image"
    ]
    if len(image_positions) != len(visual_entries):
        raise ValueError(
            "Visual embedding export could not map image placeholders to expanded visual spans."
        )

    blocks = []
    for visual_index, visual in enumerate(visual_entries):
        placeholder = image_positions[visual_index]
        if placeholder == 0 or placeholder + 1 >= len(tokens_only[0]):
            raise ValueError("Visual embedding export found an unframed image placeholder.")
        if tokens_only[0][placeholder - 1] != 151652 or tokens_only[0][placeholder + 1] != 151653:
            raise ValueError("Visual embedding export requires Qwen vision-start and vision-end tokens.")

        start = visual["index"]
        end = start + visual["size"]
        if start <= 0 or end >= embeds.shape[1]:
            raise ValueError("Visual embedding export found an incomplete expanded visual block.")
        blocks.append({
            "interior": embeds[:, start:end, :],
            "block": embeds[:, start - 1:end + 1, :],
        })
    return blocks


def _raw_visual_token_embeddings(clip, tokens, device: str, visual_index: int = 0) -> torch.Tensor:
    """Return one expanded visual input-embedding span from a source sequence."""
    blocks = _visual_token_embedding_blocks(clip, tokens, device)
    if not 0 <= visual_index < len(blocks):
        raise ValueError("Visual embedding export could not locate the requested visual token span.")
    return blocks[visual_index]["interior"]


def save_source_visual_embeddings(
    clip,
    tokens,
    visual_fusion_config: dict,
    embedding_key: str,
    device: str,
    visual_indices: list[int] | None = None,
    cache=None,
) -> None:
    """Save one or more unfused complete visual prompt blocks."""
    blocks = _visual_token_embedding_blocks(clip, tokens, device, cache=cache)
    for output_index, visual_index in enumerate(visual_indices or [0]):
        if not 0 <= visual_index < len(blocks):
            raise ValueError("Visual embedding export could not locate the requested visual block.")
        visual_block = blocks[visual_index]["block"]
        config = visual_fusion_config
        if output_index:
            config = dict(visual_fusion_config)
            save_name = config.get("save_path", "blended_visual_embeds")
            stem, suffix = os.path.splitext(save_name)
            config["save_path"] = f"{stem}_{output_index + 1}{suffix}"
        save_blended_visual_embeddings(
            [visual_block[batch].detach() for batch in range(visual_block.shape[0])],
            config,
            embedding_key,
        )


def resolve_embedding_output_path(embeddings_dir: str, file_name: str) -> str:
    """Resolve a relative output name and prove that it remains below embeddings_dir."""
    if not file_name or not file_name.strip():
        raise ValueError("Embedding file name cannot be empty.")
    relative = Path(file_name.strip())
    if relative.is_absolute():
        raise ValueError("Embedding file name must be relative to the embeddings directory.")
    root = Path(embeddings_dir).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Embedding output path escapes the embeddings directory.") from exc
    return str(target)


def evaluate_conditioning_consensus_blend(
    sequence_tensors: dict,
    pooled_tensors: dict,
    visual_fusion_config: dict = None,
    device: str = "cpu",
    visual_ranges: dict = None,
    embedding_key: str = "qwen3vl_8b",
    clip=None,
    tokens_dict: dict = None,
    visual_indices: dict = None,
    mask_cache: dict = None,
    visual_grids: dict = None,
    cache=None,
) -> tuple:
    """
    Decoupled blending engine focused entirely on isolated visual token spatial fusion.
    Saves the exact isolated visual tensor used in the fused conditioning when
    save_blended_embeds is configured.
    """

    if visual_fusion_config is None:
        visual_fusion_config = {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 2, "dither_ratio": 0.5, "seed": 0}

    visual_method = visual_fusion_config.get("visual_fusion_method", "spatial-checkerboard")
    if visual_method not in VISUAL_FUSION_METHODS:
        raise ValueError(f"Unsupported visual fusion method: {visual_method}")
    if visual_ranges is None:
        raise ValueError("Visual token ranges are required for visual fusion.")
    if visual_grids is None:
        raise ValueError("Visual token layout error: visual grids are required for fusion.")
    if mask_cache is None:
        mask_cache = {}

    active_keys = sorted(list(sequence_tensors.keys()))
    tensors_list = [sequence_tensors[k] for k in active_keys]
    if not tensors_list:
        return None, None

    B = tensors_list[0].shape[0]
    if any(key not in visual_grids for key in active_keys):
        raise ValueError("Visual token layout error: every fusion source requires a grid.")
    source_grids = [visual_grids[key] for key in active_keys]
    expected_visual_length = source_grids[0][0] * source_grids[0][1]
    if expected_visual_length <= 0 or any(visual_ranges.get(key, (0, 0)) == (0, 0) for key in active_keys):
        raise ValueError("Every visual fusion source must have a valid visual token range.")

    raw_visual_blocks = None
    if visual_fusion_config.get("save_blended_embeds", False):
        if clip is None or tokens_dict is None:
            raise ValueError("Saving visual embeddings requires the text encoder and source tokens.")
        raw_visual_blocks = []
        for key in active_keys:
            blocks = _visual_token_embedding_blocks(clip, tokens_dict[key], device, cache=cache)
            visual_index = (visual_indices or {}).get(key, 0)
            if not 0 <= visual_index < len(blocks):
                raise ValueError("Visual embedding export could not locate the requested fused visual block.")
            raw_visual_blocks.append(blocks[visual_index])

    C_blended_list = []
    for b in range(B):
        batch_tensors_dict = {k: sequence_tensors[k][b].to(device=device) for k in active_keys}
        ref_key = active_keys[0]

        prefixes = {}
        visuals = {}
        suffixes = {}

        for k in active_keys:
            t = batch_tensors_dict[k]
            v_start, v_end = visual_ranges[k]
            prefixes[k] = t[:v_start, :]
            visuals[k] = t[v_start:v_end, :]
            suffixes[k] = t[v_end:, :]

        sources = [visuals[key] for key in active_keys]
        blended_vis_2d = fuse_visual_token_sources(
            sources,
            visual_fusion_config,
            device,
            mask_cache,
            expected_visual_length,
            source_grids,
            cache=cache,
        )
        # Surrounding text (prefixes & suffixes) are kept 100% pure from the reference pass
        blended_prefix = prefixes[ref_key]
        blended_suffix = suffixes[ref_key]

        # Stitch segments back together
        C_blended_list.append(torch.cat([blended_prefix, blended_vis_2d, blended_suffix], dim=0))

    C_blended = torch.stack(C_blended_list, dim=0).to(dtype=tensors_list[0].dtype, device=tensors_list[0].device)

    if visual_fusion_config.get("save_blended_embeds", False):
        reference_block = raw_visual_blocks[0]["block"]
        if any(
            source["block"].shape != reference_block.shape
            or not torch.equal(source["block"][:, :1, :], reference_block[:, :1, :])
            or not torch.equal(source["block"][:, -1:, :], reference_block[:, -1:, :])
            for source in raw_visual_blocks[1:]
        ):
            raise ValueError("Visual embedding export requires matching Qwen vision block boundaries.")
        save_blended_visual_embeddings(
            [
                torch.cat(
                    [
                        reference_block[batch, :1, :],
                        fuse_visual_token_sources(
                            [source["interior"][batch] for source in raw_visual_blocks],
                            visual_fusion_config,
                            device,
                            mask_cache,
                            expected_visual_length,
                            source_grids,
                            cache=cache,
                        ),
                        reference_block[batch, -1:, :],
                    ],
                    dim=0,
                ).detach()
                for batch in range(reference_block.shape[0])
            ],
            visual_fusion_config,
            embedding_key,
        )

    # Pooled output is kept pure from reference pass since text is identical
    ref_key = active_keys[0]
    P_blended = pooled_tensors.get(ref_key, None)

    return C_blended, P_blended


POWER_BLEND_PRESET = {
    "method": "consensus",
    "type": "median",
    "align": "similarity",
    "alignment_threshold": 0.9,
    "thresh": 0.75,
    "alpha": 8.0,
    "beta": 0.0,
    "norm": True,
    "scale": 1.0,
    "dsc": True,
    "soft_comfort": False,
}

CONSENSUS_BLEND_PRESETS = {
    "baseline": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
    "high_clarity": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 3.0, "thresh": 0.3, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
    "smooth": {"method": "consensus", "type": "mean", "align": "similarity", "alpha": 1.5, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": False, "soft_comfort": False},
    "varied_merge": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
    "diverse_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 1.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
    "high_diversity_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 2.0, "scale": 0.7, "norm": True, "dsc": False, "soft_comfort": False},
    "dsc_baseline": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": True, "soft_comfort": True},
    "dsc_high_clarity": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 4.0, "thresh": 0.3, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": True, "soft_comfort": True},
    "dsc_smooth": {"method": "consensus", "type": "mean", "align": "similarity", "alpha": 1.0, "thresh": 0.0, "beta": 0.0, "scale": 1.0, "norm": False, "dsc": True, "soft_comfort": True},
    "dsc_varied_merge": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.5, "thresh": 0.0, "beta": 0.0, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True},
    "dsc_diverse_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 1.5, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True},
    "dsc_high_diversity_concept": {"method": "consensus", "type": "median", "align": "similarity", "alpha": 2.0, "thresh": 0.0, "beta": 3.0, "scale": 0.7, "norm": True, "dsc": True, "soft_comfort": True},
    "power_blend": POWER_BLEND_PRESET,
}


def resolve_consensus_blend_settings(blend_config):
    if not isinstance(blend_config, dict):
        raise ValueError("Consensus configuration must be a TEXT_BLEND_CONFIG value.")
    preset = blend_config.get("blend_preset", "baseline")
    settings = {
        "blend_preset": preset,
        "blend_method": blend_config.get("blend_method", "consensus"),
        "consensus_type": blend_config.get("consensus_type", "median"),
        "alignment_method": blend_config.get("alignment_method", "similarity"),
        "alignment_threshold": float(blend_config.get("alignment_threshold", 0.4)),
        "similarity_threshold": float(blend_config.get("similarity_threshold", 0.0)),
        "power_alpha": float(blend_config.get("power_alpha", 2.0)),
        "diversity_beta": float(blend_config.get("diversity_beta", 0.0)),
        "rescale_norm": bool(blend_config.get("rescale_norm", True)),
        "global_scale": float(blend_config.get("global_scale", 1.0)),
        "dynamic_similarity_contrast": False,
        "soft_comfort_bandpass": False,
        "position_weight": float(blend_config.get("position_weight", 0.0)),
        "preserve_common_prefix": bool(blend_config.get("preserve_common_prefix", False)),
    }
    if not 0.0 <= settings["position_weight"] <= 1.0:
        raise ValueError("Position weight must be between 0.0 and 1.0.")
    if preset in CONSENSUS_BLEND_PRESETS:
        selected = CONSENSUS_BLEND_PRESETS[preset]
        settings.update(
            blend_method=selected["method"],
            consensus_type=selected["type"],
            alignment_method=selected["align"],
            alignment_threshold=float(
                selected.get("alignment_threshold", settings["alignment_threshold"])
            ),
            similarity_threshold=float(selected["thresh"]),
            power_alpha=float(selected["alpha"]),
            diversity_beta=float(selected["beta"]),
            rescale_norm=bool(selected["norm"]),
            dynamic_similarity_contrast=bool(selected.get("dsc", False)),
            soft_comfort_bandpass=bool(selected.get("soft_comfort", False)),
        )
        if settings["global_scale"] == 1.0:
            settings["global_scale"] = float(selected["scale"])
    elif preset == "custom":
        settings["dynamic_similarity_contrast"] = bool(
            blend_config.get("dynamic_similarity_contrast", False)
        )
        settings["soft_comfort_bandpass"] = bool(
            blend_config.get("soft_comfort_bandpass", False)
        )
    elif preset != "off":
        raise ValueError(f"Unsupported consensus blend preset: {preset}")
    return settings


def _common_conditioning_prefix_length(tensors, rtol=1e-5, atol=1e-6):
    """Return the longest leading token span numerically shared by every [tokens, dims] tensor."""
    if not tensors:
        return 0
    limit = min(tensor.shape[0] for tensor in tensors)
    if limit == 0:
        return 0
    reference = tensors[0][:limit]
    common = torch.ones(limit, dtype=torch.bool, device=reference.device)
    for tensor in tensors[1:]:
        common &= torch.isclose(reference, tensor[:limit], rtol=rtol, atol=atol).all(dim=-1)
    mismatch = (~common).nonzero(as_tuple=False)
    return int(mismatch[0].item()) if mismatch.numel() else limit


def _position_biased_similarity_scores(similarities, position_weight):
    """Blend cosine scores with a narrowing Gaussian over normalized sequence positions."""
    weight = float(position_weight)
    if weight <= 0.0:
        return similarities
    n_ref, n_source = similarities.shape
    ref_positions = torch.linspace(0.0, 1.0, n_ref, device=similarities.device, dtype=torch.float32)
    source_positions = torch.linspace(0.0, 1.0, n_source, device=similarities.device, dtype=torch.float32)
    distance = ref_positions[:, None] - source_positions[None, :]
    sigma = max(1.0 - min(weight, 1.0), 1.0 / max(n_ref, n_source, 1))
    affinity = torch.exp(-0.5 * (distance / sigma) ** 2).to(similarities)
    return similarities * (1.0 - weight) + affinity * weight


def blend_text_vectors(sequence_tensors: dict, blend_config: dict, pooled_tensors: dict = None, device=None, compute_dtype=None) -> tuple:
    """
    Consensus-Weighted Blending math engine for language space sequences and pooled embeddings.
    Used ONLY post-encoder inside UC_ConditioningConsensusBlend.
    """
    active_keys = sorted(list(sequence_tensors.keys()))
    if not active_keys:
        raise ValueError("At least one sequence tensor is required.")
    tensors_list = [sequence_tensors[k] for k in active_keys]
    if any(t.ndim != 3 for t in tensors_list):
        raise ValueError("Every sequence tensor must have [batch, tokens, channels] shape.")
    if len({(t.shape[0], t.shape[2]) for t in tensors_list}) != 1:
        raise ValueError("All sequence tensors must have matching batch and channel dimensions.")
    if device is None:
        device = tensors_list[0].device
    if compute_dtype is None:
        compute_dtype = tensors_list[0].dtype

    B = tensors_list[0].shape[0]
    D = tensors_list[0].shape[2]

    settings = resolve_consensus_blend_settings(blend_config)
    blend_preset = settings["blend_preset"]
    if blend_preset == "off":
        first_pooled = pooled_tensors.get(active_keys[0]) if pooled_tensors else None
        return tensors_list[0], first_pooled
    blend_method = settings["blend_method"]
    consensus_type = settings["consensus_type"]
    alignment_method = settings["alignment_method"]
    alignment_threshold = settings["alignment_threshold"]
    similarity_threshold = settings["similarity_threshold"]
    power_alpha = settings["power_alpha"]
    diversity_beta = settings["diversity_beta"]
    rescale_norm = settings["rescale_norm"]
    global_scale = settings["global_scale"]
    dsc_enabled = settings["dynamic_similarity_contrast"]
    soft_comfort_enabled = settings["soft_comfort_bandpass"]
    position_weight = settings["position_weight"]
    preserve_common_prefix = settings["preserve_common_prefix"]

    C_blended_list = []
    for b in range(B):
        batch_tensors = [comfy.model_management.cast_to_device(t[b], device, compute_dtype) for t in tensors_list]
        prefix_length = _common_conditioning_prefix_length(batch_tensors) if preserve_common_prefix else 0
        preserved_prefix = batch_tensors[0][:prefix_length]
        if prefix_length:
            batch_tensors = [tensor[prefix_length:] for tensor in batch_tensors]
        if all(tensor.shape[0] == 0 for tensor in batch_tensors):
            C_blended_list.append(preserved_prefix)
            continue

        if blend_method == "linear":
            max_len = max(t.shape[0] for t in batch_tensors)
            padded = []
            for t in batch_tensors:
                if t.shape[0] < max_len:
                    padding = t.new_zeros((max_len - t.shape[0], D))
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            stacked = torch.stack(padded, dim=0)
            merged_seq = torch.mean(stacked, dim=0)
            if global_scale != 1.0:
                merged_seq *= global_scale
            C_blended_list.append(torch.cat([preserved_prefix, merged_seq], dim=0))
            continue

        if alignment_method == "similarity":
            ref_idx = max(range(len(batch_tensors)), key=lambda idx: batch_tensors[idx].shape[0])
            ref_tensor = batch_tensors[ref_idx]
            N_ref = ref_tensor.shape[0]

            ref_norm = torch.nn.functional.normalize(ref_tensor, p=2, dim=1)
            aligned_groups = [[] for _ in range(N_ref)]

            for idx, t in enumerate(batch_tensors):
                N_k = t.shape[0]
                if idx == ref_idx:
                    for i in range(N_ref):
                        aligned_groups[i].append(t[i])
                    continue

                t_norm = torch.nn.functional.normalize(t, p=2, dim=1)
                sim_matrix = torch.mm(ref_norm, t_norm.t())

                matched_tk_idx = [-1] * N_ref
                sim_matrix_tmp = sim_matrix.clone()
                if position_weight > 0.0:
                    sim_matrix_tmp = _position_biased_similarity_scores(sim_matrix_tmp, position_weight)
                    sim_matrix_tmp = sim_matrix_tmp.masked_fill(sim_matrix < alignment_threshold, -100.0)

                for _ in range(min(N_ref, N_k)):
                    flat_idx = torch.argmax(sim_matrix_tmp)
                    max_val = sim_matrix_tmp.flatten()[flat_idx].item()

                    if (position_weight == 0.0 and max_val < alignment_threshold) or max_val <= -100.0:
                        break

                    r_idx = flat_idx // N_k
                    c_idx = flat_idx % N_k

                    matched_tk_idx[r_idx.item()] = c_idx.item()

                    sim_matrix_tmp[r_idx, :] = -100.0
                    sim_matrix_tmp[:, c_idx] = -100.0

                for r_idx in range(N_ref):
                    matched_c = matched_tk_idx[r_idx]
                    if matched_c != -1:
                        aligned_groups[r_idx].append(t[matched_c])

            merged_seq = ref_tensor.new_zeros((N_ref, D))
            for r_idx in range(N_ref):
                row_tensors = aligned_groups[r_idx]
                if not row_tensors:
                    continue
                stacked = torch.stack(row_tensors, dim=0)

                if consensus_type == "median":
                    consensus = torch.median(stacked, dim=0).values
                else:
                    consensus = torch.mean(stacked, dim=0)

                stacked_norm = torch.nn.functional.normalize(stacked, p=2, dim=1, eps=1e-8)
                consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=0, eps=1e-8)
                similarities = torch.mv(stacked_norm, consensus_norm)

                if dsc_enabled:
                    min_sim = similarities.min()
                    max_sim = similarities.max()
                    if max_sim > min_sim:
                        stretched_sims = 0.7 + 0.3 * (similarities - min_sim) / (max_sim - min_sim + 1e-8)
                    else:
                        stretched_sims = similarities
                else:
                    stretched_sims = similarities

                row_weights = torch.zeros_like(similarities)
                mask = similarities >= similarity_threshold

                if mask.any():
                    if diversity_beta > 0.0:
                        distance_base = 1.5 if soft_comfort_enabled else 1.001
                        safe_sims = stretched_sims[mask].clamp(min=0.0, max=1.0)
                        row_weights[mask] = torch.pow(safe_sims, power_alpha) * torch.pow((distance_base - safe_sims).clamp(min=0.0), diversity_beta)
                    else:
                        row_weights[mask] = torch.pow(stretched_sims[mask].clamp(min=0.0), power_alpha)
                    w_sum = row_weights.sum()
                    if w_sum > 0:
                        row_weights /= w_sum
                    else:
                        row_weights = torch.ones_like(similarities) / len(similarities)
                else:
                    row_weights = torch.ones_like(similarities) / len(similarities)

                merged_vec = (stacked * row_weights.unsqueeze(1)).sum(dim=0)

                if rescale_norm:
                    avg_norm = torch.norm(stacked, p=2, dim=1).mean()
                    merged_norm = torch.norm(merged_vec, p=2)
                    if merged_norm > 0:
                        merged_vec = (merged_vec / merged_norm) * avg_norm

                if global_scale != 1.0:
                    merged_vec *= global_scale
                merged_seq[r_idx] = merged_vec
            C_blended_list.append(torch.cat([preserved_prefix, merged_seq], dim=0))
        else:
            # Index-Based Sequential Matching
            max_len = max(t.shape[0] for t in batch_tensors)
            merged_seq = batch_tensors[0].new_zeros((max_len, D))
            for i in range(max_len):
                row_tensors = []
                for t in batch_tensors:
                    if t.shape[0] > i:
                        row_tensors.append(t[i])
                if not row_tensors:
                    continue
                stacked = torch.stack(row_tensors, dim=0)

                if consensus_type == "median":
                    consensus = torch.median(stacked, dim=0).values
                else:
                    consensus = torch.mean(stacked, dim=0)

                stacked_norm = torch.nn.functional.normalize(stacked, p=2, dim=1, eps=1e-8)
                consensus_norm = torch.nn.functional.normalize(consensus, p=2, dim=0, eps=1e-8)
                similarities = torch.mv(stacked_norm, consensus_norm)

                row_weights = torch.zeros_like(similarities)
                mask = similarities >= similarity_threshold

                if mask.any():
                    if diversity_beta > 0.0:
                        safe_sims = similarities[mask].clamp(min=0.0, max=1.0)
                        row_weights[mask] = torch.pow(safe_sims, power_alpha) * torch.pow((1.001 - safe_sims).clamp(min=0.0), diversity_beta)
                    else:
                        row_weights[mask] = torch.pow(similarities[mask].clamp(min=0.0), power_alpha)
                    w_sum = row_weights.sum()
                    if w_sum > 0:
                        row_weights /= w_sum
                    else:
                        row_weights = torch.ones_like(similarities) / len(similarities)
                else:
                    row_weights = torch.ones_like(similarities) / len(similarities)

                merged_vec = (stacked * row_weights.unsqueeze(1)).sum(dim=0)

                if rescale_norm:
                    avg_norm = torch.norm(stacked, p=2, dim=1).mean()
                    merged_norm = torch.norm(merged_vec, p=2)
                    if merged_norm > 0:
                        merged_vec = (merged_vec / merged_norm) * avg_norm

                if global_scale != 1.0:
                    merged_vec *= global_scale
                merged_seq[i] = merged_vec
            C_blended_list.append(torch.cat([preserved_prefix, merged_seq], dim=0))

    C_blended = comfy.model_management.cast_to_device(
        torch.stack(C_blended_list, dim=0), tensors_list[0].device, tensors_list[0].dtype
    )

    # Blend metadata pooled outputs
    P_blended = None
    if pooled_tensors and any(p is not None for p in pooled_tensors.values()):
        pooled_list_active = [pooled_tensors[k] for k in active_keys if pooled_tensors.get(k) is not None]
        pooled_reference = pooled_list_active[0]
        P_blended_batches = []
        for b in range(B):
            stacked_p = torch.stack([
                comfy.model_management.cast_to_device(pooled[b], device, compute_dtype)
                for pooled in pooled_list_active
            ])
            if consensus_type == "median":
                consensus_p = torch.median(stacked_p, dim=0).values
            else:
                consensus_p = torch.mean(stacked_p, dim=0)

            if blend_method == "linear":
                merged_p = torch.mean(stacked_p, dim=0) * global_scale
                P_blended_batches.append(merged_p)
                continue

            stacked_p_norm = torch.nn.functional.normalize(stacked_p, p=2, dim=1, eps=1e-8)
            consensus_p_norm = torch.nn.functional.normalize(consensus_p, p=2, dim=0, eps=1e-8)
            similarities_p = torch.mv(stacked_p_norm, consensus_p_norm)

            weights_p = torch.zeros_like(similarities_p)
            mask_p = similarities_p >= similarity_threshold

            if mask_p.any():
                if diversity_beta > 0.0:
                    safe_sims = similarities_p[mask_p].clamp(min=0.0, max=1.0)
                    weights_p[mask_p] = torch.pow(safe_sims, power_alpha) * torch.pow((1.001 - safe_sims).clamp(min=0.0), diversity_beta)
                else:
                    weights_p[mask_p] = torch.pow(similarities_p[mask_p].clamp(min=0.0), power_alpha)
                w_sum_p = weights_p.sum()
                if w_sum_p > 0:
                    weights_p /= w_sum_p
                else:
                    weights_p = torch.ones_like(similarities_p) / len(similarities_p)
            else:
                weights_p = torch.ones_like(similarities_p) / len(similarities_p)

            merged_p = (stacked_p * weights_p.unsqueeze(1)).sum(dim=0)

            if rescale_norm:
                avg_norm_p = torch.norm(stacked_p, p=2, dim=1).mean()
                merged_p_norm = torch.norm(merged_p, p=2)
                if merged_p_norm > 0:
                    merged_p = (merged_p / merged_p_norm) * avg_norm_p

            if global_scale != 1.0:
                merged_p *= global_scale
            P_blended_batches.append(merged_p)
        P_blended = comfy.model_management.cast_to_device(
            torch.stack(P_blended_batches, dim=0), pooled_reference.device, pooled_reference.dtype
        )

    return C_blended, P_blended


def find_visual_token_range(
    tokens,
    cond_tensor,
    legacy_krea_spatial=False,
    minimax_token_tags=None,
    minimax_visual_index=None,
) -> tuple:
    key_name = next(iter(tokens.keys()))
    token_list = tokens[key_name][0]
    if not any(is_image_token(token) for token in token_list):
        return 0, 0

    if minimax_token_tags is not None:
        if not torch.is_tensor(minimax_token_tags) or minimax_token_tags.ndim != 1 or minimax_token_tags.numel() != cond_tensor.shape[1]:
            raise ValueError("MiniMax H3 modality tags do not match the conditioning sequence length.")
        visual_positions = torch.nonzero(minimax_token_tags == 0).flatten().tolist()
        visual_blocks = []
        for position in visual_positions:
            if not visual_blocks or position != visual_blocks[-1][-1] + 1:
                visual_blocks.append([position])
            else:
                visual_blocks[-1].append(position)
        image_count = sum(is_image_token(token) for token in token_list)
        if len(visual_blocks) != image_count or any(
            len(block) < 3 for block in visual_blocks
        ):
            raise ValueError(
                "MiniMax H3 modality tags do not match the numbered visual blocks."
            )
        if minimax_visual_index is None:
            if len(visual_blocks) != 1:
                raise ValueError(
                    "MiniMax H3 visual fusion requires a selected visual block."
                )
            visual_block = visual_blocks[0]
        else:
            try:
                visual_block = visual_blocks[minimax_visual_index]
            except (IndexError, TypeError) as exc:
                raise ValueError(
                    "MiniMax H3 visual fusion selected an unavailable visual block."
                ) from exc
        return visual_block[0] + 1, visual_block[-1]

    if legacy_krea_spatial and cond_tensor.shape[-1] == 12 * 2560:
        image_indices = [index for index, token in enumerate(token_list) if is_image_token(token)]
        if len(image_indices) != 1:
            raise ValueError("Legacy Krea2 spatial mapping requires exactly one image per encoder pass.")
        text_count = len(token_list) - 1
        visual_length = cond_tensor.shape[1] - text_count
        if visual_length <= 0:
            raise ValueError("Legacy Krea2 spatial mapping produced a non-positive visual span.")
        visual_start = image_indices[0]
        visual_end = visual_start + visual_length
        trailing_text = len(token_list) - image_indices[0] - 1
        if visual_end + trailing_text != cond_tensor.shape[1]:
            raise ValueError("Legacy Krea2 spatial mapping does not cover the conditioning sequence.")
        return visual_start, visual_end

    mapping = build_token_to_conditioning_map(
        token_list, cond_tensor, embedding_key=key_name
    )
    for i, t in enumerate(token_list):
        if is_image_token(t):
            return mapping[i][0], mapping[i][1]

    return 0, 0

def encode_embedding_classical_scaled_bias(
    clip,
    text,
    llama_template=None,
    visual_encoder_path="grid-deepstack",
    tokenize_callback=None,
    encode_callback=None,
    cache=None,
    **kwargs,
):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    def tokenize(value):
        if tokenize_callback is not None:
            return tokenize_callback(value)
        return clip.tokenize(value, llama_template=llama_template, **kwargs)

    if "(" not in text or ")" not in text:
        tokens = tokenize(text)
        return encode_callback(tokens) if encode_callback is not None else _encode_scheduled_with_visual_path(clip, tokens, visual_encoder_path, cache=cache)

    clean_text = ""
    biases_to_apply = []
    for segment, strength in _contextual_token_weights(text):
        start_tokens = tokenize(clean_text)
        key_name = next(iter(start_tokens.keys()))
        start_count = len(start_tokens[key_name][0])
        clean_text += segment
        end_tokens = tokenize(clean_text)
        end_count = len(end_tokens[key_name][0])
        if strength != 1.0 and end_count > start_count:
            if not math.isfinite(strength):
                raise ValueError("Contextual prompt weights must be finite.")
            biases_to_apply.append({"start": start_count, "end": end_count, "strength": float(strength)})

    tokens = tokenize(clean_text)
    conditioning = encode_callback(tokens) if encode_callback is not None else _encode_scheduled_with_visual_path(clip, tokens, visual_encoder_path, cache=cache)

    if not biases_to_apply:
        return conditioning

    # Apply contextual vector scaling directly to each schedule. This is a
    # custom operation for modern encoders that disable Core prompt weights.
    new_conditioning = []

    for i in range(len(conditioning)):
        cond, cond_dict = conditioning[i]
        new_cond = cond.clone()

        key_name = next(iter(tokens.keys()))
        token_list = tokens[key_name][0]
        mapping = build_token_to_conditioning_map(token_list, new_cond)

        # Scale embeddings using mapped ranges
        for bias in biases_to_apply:
            strength = bias["strength"]

            # Map token indices to embedding indices
            t_start = bias["start"]
            t_end = bias["end"]

            if t_start >= len(mapping):
                continue

            start = mapping[t_start][0]
            end = mapping[min(t_end - 1, len(mapping) - 1)][1]

            if start < 0 or start >= end:
                continue

            new_cond[:, start:end, :] *= strength

        new_conditioning.append([new_cond, cond_dict.copy()])

    return new_conditioning


def strip_contextual_weight_syntax(text: str) -> str:
    """Return the exact clean text consumed by contextual vector scaling."""
    if "(" not in text or ")" not in text:
        return text
    return "".join(segment for segment, _ in _contextual_token_weights(text))


def _contextual_token_weights(text: str):
    """Parse weights with the same backslash escaping contract as Core's tokenizer."""
    return [(unescape_important(segment), weight) for segment, weight in token_weights(escape_important(text), 1.0)]

def load_vlm_image_tensor(path_str):
    if not path_str:
        return None
    normalized_path = path_str.strip().replace('\\', '/')
    normalized_path = os.path.normpath(normalized_path)
    if not os.path.isabs(normalized_path):
        normalized_path = os.path.abspath(normalized_path)

    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"Invalid image path: {path_str} (resolved to: {normalized_path})")


    img = node_helpers.pillow(Image.open, normalized_path)
    for i in ImageSequence.Iterator(img):
        i = node_helpers.pillow(ImageOps.exif_transpose, i)
        if i.mode == 'I':
            i = i.point(lambda x: x * (1 / 65535))
        image = i.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]  # Returns [1, H, W, C]

def krea2_user_content_span(ids):
    best_start, best_end = None, None
    for i in range(len(ids) - 2):
        if ids[i] == _QWEN_IM_START and ids[i + 1] == _QWEN_USER and ids[i + 2] == _QWEN_NL:
            start = i + 3
            end = start
            while end < len(ids) and ids[end] != _QWEN_IM_END:
                end += 1
            if end > start:  # prioritize non-empty block
                best_start, best_end = start, end
    if best_start is not None:
        return best_start, best_end
    for i in range(len(ids) - 2):
        if ids[i] == _QWEN_IM_START and ids[i + 1] == _QWEN_USER and ids[i + 2] == _QWEN_NL:
            start = i + 3
            end = start
            while end < len(ids) and ids[end] != _QWEN_IM_END:
                end += 1
            return start, end
    return None, None

def extract_and_flatten_images(image_inputs) -> tuple:
    """
    Extracts individual images from batched image tensors within the image_inputs dict.
    Returns:
        - raw_images: Dict mapping sequential index to a single image tensor of shape [1, H, W, C]
        - flat_images: List of individual image tensors
        - is_zero_indexed: Boolean indicating if the original keys started at 0
    """
    is_zero_indexed = False
    if image_inputs is not None:
        for k in image_inputs.keys():
            digits = re.findall(r'\d+', k)
            if digits and int(digits[0]) == 0:
                is_zero_indexed = True
                break

    flat_images = []
    if image_inputs is not None:
        # Sort keys numerically by their suffix to ensure correct sequential order
        def get_num(k):
            digits = re.findall(r'\d+', k)
            return int(digits[0]) if digits else 0
        sorted_keys = sorted(image_inputs.keys(), key=get_num)

        for k in sorted_keys:
            v = image_inputs[k]
            if v is not None:
                if isinstance(v, torch.Tensor) and len(v.shape) == 4:
                    # Shape is [B, H, W, C]. Slice into B individual tensors of [1, H, W, C]
                    for i in range(v.shape[0]):
                        flat_images.append(v[i:i+1])
                elif isinstance(v, list):
                    # Handle lists of tensors if passed
                    for item in v:
                        if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                            for i in range(item.shape[0]):
                                flat_images.append(item[i:i+1])
                        elif isinstance(item, torch.Tensor):
                            flat_images.append(item)
                else:
                    flat_images.append(v)

    start_idx = 0 if is_zero_indexed else 1
    raw_images = {}
    for idx, img in enumerate(flat_images):
        raw_images[start_idx + idx] = img

    return raw_images, flat_images, is_zero_indexed


def extract_image_socket_batches(image_inputs) -> list[tuple[int, list]]:
    """Expand image batches while retaining each numbered input socket."""
    if not image_inputs:
        return []

    def socket_number(name):
        digits = re.findall(r"\d+", name)
        return int(digits[0]) if digits else 0

    socket_batches = []
    for name in sorted(image_inputs, key=socket_number):
        value = image_inputs[name]
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        expanded = []
        for tensor in values:
            if not torch.is_tensor(tensor) or tensor.ndim != 4:
                raise ValueError("Every visual input must be a BHWC IMAGE tensor.")
            expanded.extend(tensor[index:index + 1] for index in range(tensor.shape[0]))
        if expanded:
            socket_batches.append((socket_number(name), expanded))
    return socket_batches


def build_visual_consensus_batch_lanes(image_inputs):
    """Keep visual sources and index-aligned execution batches as separate axes."""
    if not image_inputs:
        return [], []

    sockets = [images for _, images in extract_image_socket_batches(image_inputs)]

    if not sockets:
        return [], []

    # A batch in the only connected socket is explicitly equivalent to the
    # same images connected as separate visual sources.
    if len(sockets) == 1:
        return [sockets[0]], list(sockets[0])

    non_singleton_lengths = {len(socket) for socket in sockets if len(socket) > 1}
    if len(non_singleton_lengths) > 1:
        lengths = ", ".join(str(len(socket)) for socket in sockets)
        raise ValueError(f"Visual input batch lengths must match or be 1 (got {lengths}).")

    lane_count = next(iter(non_singleton_lengths), 1)
    lanes = [
        [socket[0] if len(socket) == 1 else socket[lane] for socket in sockets]
        for lane in range(lane_count)
    ]
    reference_images = [image for socket in sockets for image in socket]
    return lanes, reference_images


def blend_complete_conditionings(conditionings, blend_config):
    """Apply the same complete-conditioning contract as UC_ConditioningConsensusBlend."""
    if not conditionings:
        raise ValueError("Consensus requires at least one complete conditioning.")
    if len(conditionings) == 1:
        return conditionings[0]

    schedule_lengths = {len(conditioning) for conditioning in conditionings}
    if len(schedule_lengths) != 1:
        raise ValueError("All conditioning samples must have the same schedule length.")

    device = comfy.model_management.get_torch_device()
    compute_dtype = comfy.model_management.intermediate_dtype()
    blended = []
    for schedule_index in range(next(iter(schedule_lengths))):
        entries = [conditioning[schedule_index] for conditioning in conditionings]
        sequence_tensors = {
            chr(97 + index): entry[0] for index, entry in enumerate(entries)
        }
        pooled_tensors = {
            chr(97 + index): entry[1].get("pooled_output")
            for index, entry in enumerate(entries)
        }
        tensor, pooled = blend_text_vectors(
            sequence_tensors,
            blend_config,
            pooled_tensors=pooled_tensors,
            device=device,
            compute_dtype=compute_dtype,
        )

        layout_keys = {
            "attention_mask",
            "attention_mask_img_shape",
            "embeds_info",
            "minimax_token_tags",
            "pooled_output",
        }
        metadata_items = [entry[1] for entry in entries]
        common_keys = set.intersection(
            *(set(metadata) for metadata in metadata_items)
        ) - layout_keys
        metadata = {}
        for key in common_keys:
            values = [item[key] for item in metadata_items]
            first = values[0]
            if torch.is_tensor(first):
                if all(
                    torch.is_tensor(value)
                    and value.shape == first.shape
                    and torch.equal(value, first)
                    for value in values[1:]
                ):
                    metadata[key] = first
            elif all(value is first for value in values[1:]):
                metadata[key] = first
            elif isinstance(first, (str, int, float, bool, type(None))) and all(
                value == first for value in values[1:]
            ):
                metadata[key] = first
        if pooled is not None:
            metadata["pooled_output"] = pooled
        reference_entry = max(entries, key=lambda entry: entry[0].shape[1])
        minimax_tags = reference_entry[1].get("minimax_token_tags")
        if minimax_tags is not None:
            if not torch.is_tensor(minimax_tags) or minimax_tags.numel() != tensor.shape[1]:
                raise ValueError(
                    "MiniMax H3 modality tags do not match the consensus sequence length."
                )
            metadata["minimax_token_tags"] = minimax_tags
        blended.append([tensor, metadata])
    return blended


def batch_complete_conditionings(conditionings):
    """Combine independent execution lanes without treating them as consensus samples."""
    if not conditionings:
        raise ValueError("At least one completed conditioning batch lane is required.")
    if len(conditionings) == 1:
        return conditionings[0]
    schedule_lengths = {len(conditioning) for conditioning in conditionings}
    if len(schedule_lengths) != 1:
        raise ValueError("All conditioning batch lanes must have the same schedule length.")

    batched = []
    for schedule_index in range(next(iter(schedule_lengths))):
        entries = [conditioning[schedule_index] for conditioning in conditionings]
        tensors = [entry[0] for entry in entries]
        if any(tensor.shape[1:] != tensors[0].shape[1:] for tensor in tensors[1:]):
            raise ValueError(
                "Visual batch lanes produced incompatible conditioning shapes."
            )
        metadata_items = [entry[1] for entry in entries]
        metadata = {}
        common_keys = set.intersection(*(set(item) for item in metadata_items))
        for key in common_keys:
            values = [item[key] for item in metadata_items]
            first = values[0]
            if torch.is_tensor(first):
                if all(
                    torch.is_tensor(value)
                    and value.shape[1:] == first.shape[1:]
                    for value in values[1:]
                ):
                    metadata[key] = torch.cat(values, dim=0)
            elif all(value is first for value in values[1:]):
                metadata[key] = first
            elif isinstance(first, (str, int, float, bool, type(None))) and all(
                value == first for value in values[1:]
            ):
                metadata[key] = first
        batched.append([torch.cat(tensors, dim=0), metadata])
    return batched


def _format_advanced_visual_consensus_prompt(prompt, system_prompt):
    if system_prompt:
        return (
            "<|im_start|>user\n<|im_end|>\n"
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, "
        "size, texture, quantity, text, spatial relationships of the objects "
        "and background:<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _encode_visual_consensus_source(
    clip, source_image, resolution, prompt, visual_encoder_path
):
    processed = prepare_vlm_image(source_image, resolution)
    minimax_h3 = is_minimax_h3_text_encoder(clip)
    tokenize_callback = None
    encode_kwargs = {"images": [processed], "skip_template": True}
    if minimax_h3:
        tokenize_callback = lambda text: tokenize_minimax_h3_prompt(
            clip, text, [processed]
        )
        encode_kwargs = {}
    conditioning = encode_embedding_classical_scaled_bias(
        clip,
        prompt,
        visual_encoder_path=visual_encoder_path,
        tokenize_callback=tokenize_callback,
        **encode_kwargs,
    )
    if len(conditioning) != 1:
        raise ValueError("Visual consensus encoding requires one schedule entry.")
    tensor, metadata = conditioning[0]
    tokens = (
        tokenize_minimax_h3_prompt(clip, prompt, [processed])
        if minimax_h3
        else clip.tokenize(prompt, images=[processed], skip_template=True)
    )
    visual_range = find_visual_token_range(
        tokens,
        tensor,
        legacy_krea_spatial=visual_encoder_path == "legacy-flat",
        minimax_token_tags=metadata.get("minimax_token_tags"),
    )
    return {
        "conditioning": [[tensor, metadata]],
        "tensor": tensor,
        "metadata": metadata,
        "pooled": metadata.get("pooled_output"),
        "tokens": tokens,
        "visual_range": visual_range,
        "grid": visual_fusion_grid(
            processed,
            visual_range[1] - visual_range[0],
            visual_encoder_path == "legacy-flat",
        ),
        "image": processed,
    }


def _tokenize_visual_consensus_source(clip, source_image, resolution, prompt):
    processed = prepare_vlm_image(source_image, resolution)
    tokens = (
        tokenize_minimax_h3_prompt(clip, prompt, [processed])
        if is_minimax_h3_text_encoder(clip)
        else clip.tokenize(prompt, images=[processed], skip_template=True)
    )
    return tokens


def _spatially_fuse_visual_consensus_sources(
    branches, visual_config, clip, allow_export, cache=None
):
    keys = [chr(97 + index) for index in range(len(branches))]
    config = dict(visual_config)
    config["save_blended_embeds"] = bool(
        allow_export and config.get("save_blended_embeds", False)
    )
    tensor, pooled = evaluate_conditioning_consensus_blend(
        {key: branch["tensor"] for key, branch in zip(keys, branches)},
        {
            key: branch["pooled"]
            for key, branch in zip(keys, branches)
            if branch["pooled"] is not None
        },
        visual_fusion_config=config,
        device=comfy.model_management.get_torch_device(),
        visual_ranges={
            key: branch["visual_range"] for key, branch in zip(keys, branches)
        },
        embedding_key=next(iter(branches[0]["tokens"])),
        clip=clip,
        tokens_dict={key: branch["tokens"] for key, branch in zip(keys, branches)},
        visual_indices={key: branch.get("raw_visual_index", 0) for key, branch in zip(keys, branches)},
        mask_cache={},
        visual_grids={
            key: branch["grid"] for key, branch in zip(keys, branches)
        },
        cache=cache,
    )
    metadata = branches[0]["metadata"].copy()
    attention_mask = metadata.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.shape[-1] != tensor.shape[1]:
        metadata.pop("attention_mask", None)
        metadata.pop("attention_mask_img_shape", None)
    if pooled is not None:
        metadata["pooled_output"] = pooled
    minimax_tags = metadata.get("minimax_token_tags")
    if minimax_tags is not None and (
        not torch.is_tensor(minimax_tags) or minimax_tags.numel() != tensor.shape[1]
    ):
        raise ValueError(
            "MiniMax H3 modality tags do not match the spatially fused sequence length."
        )
    return [[tensor, metadata]]


def execute_minimax_h3_vlm_guide(conditioning, clip, image, timestamp, vlm_resolution=384, cache=None, enable_caching="all"):
    if not is_minimax_h3_text_encoder(clip):
        raise ValueError("MiniMax H3 VLM Guide requires the qwen3vl_32b text encoder.")
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("MiniMax H3 guide timestamp must be finite nonnegative seconds.")
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError("MiniMax H3 VLM Guide requires exactly one BHWC image.")
    if cache is None:
        with H3EncoderCache(enable_caching) as invocation:
            return execute_minimax_h3_vlm_guide(conditioning, clip, image, timestamp, vlm_resolution, cache=invocation, enable_caching=enable_caching)
    clip = cache.prepare_clip(clip)
    prepared = prepare_vlm_image(image, vlm_resolution)
    entries = _minimax_h3_text_entries(clip, f"<{timestamp:.1f} seconds>")
    entries += _minimax_h3_visual_token_entries(clip, prepared)
    guide = _encode_scheduled_with_visual_path(clip, {"qwen3vl_32b": [entries]}, "grid-deepstack", cache=cache)
    return splice_conditioning(conditioning, guide)


def execute_advanced_minimax_h3_image_to_video(
    clip,
    vae,
    prompt,
    width,
    height,
    length,
    first_frame=None,
    last_frame=None,
    reference_images=None,
    fusion_images=None,
    visual_fusion_config=None,
    multiplier=1.0,
    ref_image_size="match",
    vlm_resolution=384,
    vlm_video_resolution=384,
    media_config=None,
    video=None,
    audio=None,
    audio_vae=None,
    token_fusion=False,
    temporal_fusion=False,
    temporal_token_fusion=False,
    text_blend_config=None,
    cache=None,
    enable_caching="all",
):
    """Build coordinated Qwen conditioning, H3 image controls, and AV latent."""
    if not is_minimax_h3_text_encoder(clip):
        raise ValueError(
            "Advanced MiniMax H3 Image to Video requires the qwen3vl_32b text encoder."
        )
    if cache is None:
        with H3EncoderCache(enable_caching) as invocation:
            return execute_advanced_minimax_h3_image_to_video(
                clip, vae, prompt, width, height, length,
                first_frame=first_frame, last_frame=last_frame, reference_images=reference_images,
                fusion_images=fusion_images, visual_fusion_config=visual_fusion_config,
                multiplier=multiplier, ref_image_size=ref_image_size,
                vlm_resolution=vlm_resolution, vlm_video_resolution=vlm_video_resolution,
                media_config=media_config, video=video, audio=audio, audio_vae=audio_vae,
                token_fusion=token_fusion, temporal_fusion=temporal_fusion,
                temporal_token_fusion=temporal_token_fusion, text_blend_config=text_blend_config,
                cache=invocation, enable_caching=enable_caching,
            )
    _, flat_references, _ = extract_and_flatten_images(reference_images)
    _, flat_fusion_images, _ = extract_and_flatten_images(fusion_images)
    fusion_socket_batches = extract_image_socket_batches(fusion_images)
    keyframe_mode = first_frame is not None or last_frame is not None
    if keyframe_mode and flat_references:
        raise ValueError(
            "MiniMax H3 frame inputs cannot be combined with native reference images."
        )
    if keyframe_mode and video is not None:
        raise ValueError(
            "MiniMax H3 frame inputs cannot be used with reference video."
        )
    native_reference_mode = bool(flat_references)
    if native_reference_mode and ref_image_size not in ("match", "max", "none"):
        raise ValueError(f"Unsupported MiniMax H3 reference image size: {ref_image_size}")

    frame_vae_enabled = ref_image_size != "none"
    visual_sources = []
    if first_frame is not None:
        visual_sources.append(("first frame", first_frame))
    if last_frame is not None:
        visual_sources.append(("last frame", last_frame))
    visual_sources.extend(
        (f"reference image {index}", image)
        for index, image in enumerate(flat_references, start=1)
    )
    visual_sources.extend(
        (f"fusion image {index}", image)
        for index, image in enumerate(flat_fusion_images, start=1)
    )
    for label, image in visual_sources:
        if (
            not torch.is_tensor(image)
            or image.ndim != 4
            or image.shape[0] != 1
            or image.shape[1] < 1
            or image.shape[2] < 1
            or image.shape[3] < 3
        ):
            raise ValueError(
                f"MiniMax H3 {label} must contain exactly one BHWC image with at least three channels."
            )

    clip = cache.prepare_clip(clip)

    latent, frame_count = minimax_h3_empty_av_latent(width, height, length)
    picture_timestamps = []
    media_timestamp_format = None
    picture_structure = None
    default_single_visual = False
    video_fps = 2
    video_latent_mode = None
    video_latent_keyframes = 4
    if media_config is not None:
        (
            picture_timestamps,
            media_timestamp_format,
            picture_structure,
            default_single_visual,
            video_fps,
            video_latent_mode,
            video_latent_keyframes,
        ) = _validate_minimax_h3_media_config(
            media_config,
            frame_count,
        )
    video_frames = None
    video_reference = None
    positioned_video_keyframes = []
    if video is not None:
        resolved_video_latent_mode = video_latent_mode
        if resolved_video_latent_mode is None:
            resolved_video_latent_mode = "full video" if frame_vae_enabled else "off"
        if (
            resolved_video_latent_mode == "even keyframes"
            and frame_vae_enabled
            and flat_references
        ):
            raise ValueError(
                "MiniMax H3 even Video keyframes with native reference images require ref_image_size none."
            )
        video_frames, video_reference = prepare_minimax_h3_reference_video(
            video,
            vae,
            frame_count,
            encode_reference=resolved_video_latent_mode == "full video",
            cache=cache,
        )
        if resolved_video_latent_mode == "even keyframes":
            positioned_video_keyframes = prepare_minimax_h3_positioned_video_keyframes(
                video,
                vae,
                frame_count,
                width,
                height,
                video_latent_keyframes,
                cache=cache,
            )
    audio_reference = _encode_minimax_h3_audio_reference(audio, audio_vae, cache=cache)
    prepared_first = (
        prepare_minimax_h3_frame(first_frame, width, height, "disabled")
        if first_frame is not None and frame_vae_enabled
        else None
    )
    prepared_last = (
        prepare_minimax_h3_frame(last_frame, width, height, "center")
        if last_frame is not None and frame_vae_enabled
        else None
    )
    prepared_references = (
        [
            prepare_minimax_h3_reference_image(
                image, width, height, ref_image_size
            )
            for image in flat_references
        ]
        if native_reference_mode and frame_vae_enabled
        else []
    )

    base_vlm_images = []
    if first_frame is not None:
        base_vlm_images.append(prepare_vlm_image(first_frame, vlm_resolution))
    if last_frame is not None:
        base_vlm_images.append(prepare_vlm_image(last_frame, vlm_resolution))
    base_vlm_images.extend(
        prepare_vlm_image(image, vlm_resolution) for image in flat_references
    )
    default_media_image = None
    if default_single_visual:
        if first_frame is not None:
            default_media_image = base_vlm_images[0]
        elif flat_references:
            default_media_image = base_vlm_images[int(last_frame is not None)]
        elif video_frames is None:
            raise ValueError(
                "MiniMax H3 default media config requires a first frame, reference image 1, or video."
            )
    config = dict(visual_fusion_config or {})
    visual_method = config.get("visual_fusion_method", "off")
    fusion_vlm_images = [
        prepare_vlm_image(image, vlm_resolution) for image in flat_fusion_images
    ]
    if native_reference_mode and visual_method == "off":
        fusion_vlm_images = []
    fusion_active = visual_method != "off" and bool(fusion_vlm_images)
    visual_encoder_path = config.get("visual_encoder_path", "grid-deepstack")
    actual_prompt = prompt if token_fusion and fusion_active else strip_contextual_weight_syntax(prompt)
    prompt_entries = _minimax_h3_text_entries(clip, actual_prompt)
    prompt_spans = [_conditioning_token_span(entry) for entry in prompt_entries]
    if any(span is None for span in prompt_spans):
        raise ValueError("MiniMax H3 prompt contains an unsupported embedding span.")
    prompt_length = sum(prompt_spans)
    def tokenize_presentation(text, images):
        if media_config is not None:
            collapse_default_picture = default_single_visual and video_frames is None
            presentation_images = (
                [default_media_image]
                if collapse_default_picture and default_media_image is not None
                else ([] if collapse_default_picture else images)
            )
            default_video_frames = None
            default_video_timestamps = []
            if video_frames is not None:
                sample_indices = minimax_h3_video_sample_indices(
                    video_frames.shape[0], video_fps
                )
                default_video_frames = prepare_minimax_h3_vlm_video_frames(
                    video_frames[sample_indices], vlm_video_resolution
                )
                default_video_timestamps = [
                    Fraction(index, 24) for index in sample_indices
                ]
            return tokenize_minimax_h3_media_prompt(
                clip,
                text,
                presentation_images,
                picture_timestamps,
                media_timestamp_format,
                picture_structure,
                audio=audio_reference is not None,
                default_single_visual=default_single_visual,
                default_video_frames=default_video_frames,
                default_video_timestamps=default_video_timestamps,
            )
        if video_frames is not None or audio_reference is not None:
            reference_items = [
                {"type": "image", "data": image} for image in images
            ]
            if video_frames is not None:
                sample_indices = list(range(0, video_frames.shape[0], 12))
                reference_items.append({
                    "type": "video",
                    "data": prepare_minimax_h3_vlm_video_frames(
                        video_frames[sample_indices], vlm_video_resolution
                    ),
                    "timestamps": [Fraction(index, 24) for index in sample_indices],
                })
            if audio_reference is not None:
                reference_items.append({"type": "audio"})
            return clip.tokenize(text, minimax_ref_items=reference_items)
        if native_reference_mode:
            return clip.tokenize(
                text,
                minimax_ref_items=[
                    {"type": "image", "data": image} for image in images
                ],
            )
        return clip.tokenize(text, images=images)

    temporal_encode = None
    if temporal_fusion and video_frames is not None:
        temporal_config = media_config or {}
        density = temporal_config.get("temporal_density", 1)
        method = temporal_config.get("temporal_fusion_method", "consensus")
        if isinstance(density, bool) or not isinstance(density, numbers.Integral) or not 1 <= density <= 24:
            raise ValueError("MiniMax H3 temporal density must be an integer from 1 to 24.")
        if method not in ("consensus", "spatial"):
            raise ValueError("Unsupported MiniMax H3 temporal fusion method.")
        default_consensus = {
            "blend_preset": "custom", "alignment_method": "index", "consensus_type": "median",
            "power_alpha": 2.0, "diversity_beta": 0.0, "rescale_norm": True, "global_scale": 1.0,
        }
        settings = resolve_consensus_blend_settings(default_consensus if text_blend_config is None else text_blend_config)
        enabled = settings["blend_preset"] != "off" if method == "consensus" else visual_method != "off"
        if density > 1 and enabled:
            indices = minimax_h3_video_sample_indices(video_frames.shape[0], video_fps) if media_config is not None else list(range(0, video_frames.shape[0], 12))
            frame_pairs = minimax_h3_temporal_frame_pairs(video_frames.shape[0], indices, int(density))

            def fuse_video_block(sources, grids, deepstack):
                def compute():
                    return fuse_temporal_block(
                        sources, method, settings, config, grids,
                        spatial_fuse_callback=fuse_visual_token_sources,
                        position_score_callback=_position_biased_similarity_scores,
                        deepstack_layers=deepstack,
                    )
                if len(sources) == 1 or temporal_token_fusion:
                    return compute()
                return cache.get_or_compute("encoded_section", {
                    "sources": sources, "grids": grids, "deepstack": deepstack,
                    "method": method, "settings": temporal_cache_settings(method, settings, config),
                    "section": "temporal_post_qwen", "device": str(sources[0].device),
                }, compute)

            def temporal_encode(tokens):
                return encode_temporal_conditioning(
                    clip, tokens, frame_pairs,
                    lambda pair: prepare_minimax_h3_vlm_video_frames(video_frames[list(pair)], vlm_video_resolution),
                    token_fusion=temporal_token_fusion,
                    fusion_callback=fuse_video_block,
                    encode_tokens_callback=lambda value: _encode_scheduled_with_visual_path(clip, value, "grid-deepstack", cache=cache),
                    active_clip_model_callback=_active_clip_model,
                    encode_preprocessed_callback=_encode_preprocessed_clip_model,
                    visual_context_callback=lambda: qwen3vl_visual_encoder_path(clip, "grid-deepstack"),
                    video_grid_callback=lambda data, size: visual_fusion_grid(data, size, False),
                    token_spans_callback=build_token_to_conditioning_map,
                    cache=cache,
                )

    if fusion_active and (keyframe_mode or native_reference_mode):
        if keyframe_mode:
            if any(
                socket_number < 1 or socket_number > len(base_vlm_images)
                for socket_number, _ in fusion_socket_batches
            ):
                raise ValueError(
                    "MiniMax H3 frame fusion has a fusion image without a matching picture slot."
                )
            fusion_slot_batches = [
                (socket_number - 1, socket_images)
                for socket_number, socket_images in fusion_socket_batches
            ]
        elif (
            len(fusion_socket_batches) == 1
            and fusion_socket_batches[0][0] == 1
            and len(fusion_socket_batches[0][1]) == 1
        ):
            fusion_slot_batches = [
                (index, fusion_socket_batches[0][1])
                for index in range(len(base_vlm_images))
            ]
        else:
            fusion_slot_batches = [
                (index, [image])
                for index, image in enumerate(fusion_vlm_images[:len(base_vlm_images)])
            ]

        tokenize_callback = lambda text: tokenize_presentation(text, base_vlm_images)
        if token_fusion:
            canonical_tokens = tokenize_callback(prompt)
            slot_sources = []
            for visual_index, socket_images in fusion_slot_batches:
                alternatives = []
                for fusion_image in socket_images:
                    branch_images = list(base_vlm_images)
                    branch_images[visual_index] = fusion_image
                    alternatives.append(tokenize_presentation(prompt, branch_images))
                slot_sources.append((visual_index, alternatives))
            conditioning = encode_token_fused_visual_slots(
                clip, canonical_tokens, slot_sources, config, visual_encoder_path, cache=cache
            )
        else:
            conditioning = encode_embedding_classical_scaled_bias(
                clip,
                prompt,
                tokenize_callback=tokenize_callback,
                visual_encoder_path=visual_encoder_path,
                cache=cache,
            )
            if len(conditioning) != 1:
                raise ValueError(
                    "MiniMax H3 visual conditioning requires one schedule entry."
                )

            base_tensor, base_metadata = conditioning[0]
            base_tokens = tokenize_callback(prompt)
            fused_tensor = base_tensor.clone()
            for visual_index, socket_images in fusion_slot_batches:
                branches = []
                branch_sources = [(base_vlm_images[visual_index], base_tensor, base_metadata, base_tokens)]
                for fusion_image in socket_images:
                    branch_images = list(base_vlm_images)
                    branch_images[visual_index] = fusion_image
                    branch_callback = lambda text, images=branch_images: tokenize_presentation(text, images)
                    branch_conditioning = encode_embedding_classical_scaled_bias(
                        clip, prompt, tokenize_callback=branch_callback,
                        visual_encoder_path=visual_encoder_path,
                        cache=cache,
                    )
                    if len(branch_conditioning) != 1:
                        raise ValueError("MiniMax H3 visual fusion requires one conditioning schedule entry.")
                    branch_tensor, branch_metadata = branch_conditioning[0]
                    branch_sources.append((fusion_image, branch_tensor, branch_metadata, branch_callback(prompt)))
                for image, tensor, metadata, tokens in branch_sources:
                    visual_range = find_visual_token_range(
                        tokens, tensor,
                        legacy_krea_spatial=visual_encoder_path == "legacy-flat",
                        minimax_token_tags=metadata.get("minimax_token_tags"),
                        minimax_visual_index=visual_index,
                    )
                    if visual_range == (0, 0):
                        raise ValueError(f"Could not locate MiniMax H3 picture {visual_index + 1} for fusion.")
                    branches.append({
                        "tensor": tensor, "pooled": metadata.get("pooled_output"),
                        "metadata": metadata, "tokens": tokens, "visual_range": visual_range,
                        "grid": visual_fusion_grid(image, visual_range[1] - visual_range[0], visual_encoder_path == "legacy-flat"),
                        "raw_visual_index": visual_index,
                    })
                slot_conditioning = _spatially_fuse_visual_consensus_sources(branches, config, clip, allow_export=True, cache=cache)
                slot_tensor = slot_conditioning[0][0]
                base_range = branches[0]["visual_range"]
                fused_tensor[:, base_range[0]:base_range[1], :] = slot_tensor[:, base_range[0]:base_range[1], :]
            conditioning = [[fused_tensor, base_metadata]]
    elif fusion_active:
        if token_fusion:
            canonical_images = [*base_vlm_images, fusion_vlm_images[0]]
            canonical_tokens = tokenize_presentation(prompt, canonical_images)
            visual_index = len(canonical_images) - 1
            alternatives = [
                tokenize_presentation(prompt, [*base_vlm_images, image])
                for image in fusion_vlm_images[1:]
            ]
            conditioning = encode_token_fused_visual_slots(
                clip,
                canonical_tokens,
                [(visual_index, alternatives)],
                config,
                visual_encoder_path,
                cache=cache,
            )
        else:
            branches = []
            for index, image in enumerate(fusion_vlm_images):
                branch_images = [*base_vlm_images, image]
                tokenize_callback = lambda text, images=branch_images: tokenize_presentation(text, images)
                conditioning = encode_embedding_classical_scaled_bias(
                    clip, prompt, tokenize_callback=tokenize_callback,
                    visual_encoder_path=visual_encoder_path,
                    cache=cache,
                )
                if len(conditioning) != 1:
                    raise ValueError("MiniMax H3 visual fusion requires one conditioning schedule entry.")
                tensor, metadata = conditioning[0]
                tokens = tokenize_callback(prompt)
                visual_range = find_visual_token_range(
                    tokens, tensor,
                    legacy_krea_spatial=visual_encoder_path == "legacy-flat",
                    minimax_token_tags=metadata.get("minimax_token_tags"),
                    minimax_visual_index=len(branch_images) - 1,
                )
                if visual_range == (0, 0):
                    raise ValueError(f"Could not locate the MiniMax H3 visual span for fusion image {index + 1}.")
                branches.append({
                    "tensor": tensor, "pooled": metadata.get("pooled_output"),
                    "metadata": metadata, "tokens": tokens, "visual_range": visual_range,
                    "grid": visual_fusion_grid(image, visual_range[1] - visual_range[0], visual_encoder_path == "legacy-flat"),
                    "raw_visual_index": len(branch_images) - 1,
                })
            conditioning = _spatially_fuse_visual_consensus_sources(branches, config, clip, allow_export=True, cache=cache)
    else:
        presentation_images = [*base_vlm_images, *fusion_vlm_images]
        tokenize_callback = lambda text: tokenize_presentation(
            text, presentation_images
        )
        conditioning = encode_embedding_classical_scaled_bias(
            clip,
            prompt,
            tokenize_callback=tokenize_callback,
            visual_encoder_path="grid-deepstack",
            encode_callback=temporal_encode,
            cache=cache,
        )
        if len(conditioning) != 1 and not temporal_fusion:
            raise ValueError(
                "MiniMax H3 visual conditioning requires one schedule entry."
            )
        if config.get("save_blended_embeds", False) and presentation_images:
            tokens = tokenize_callback(prompt)
            save_source_visual_embeddings(
                clip,
                tokens,
                config,
                visual_embedding_key(clip, tokens),
                comfy.model_management.get_torch_device(),
                list(range(len(presentation_images))),
                cache,
            )

    layout_conditioning = []
    for tensor, metadata in conditioning:
        tags = metadata.get("minimax_token_tags")
        if not torch.is_tensor(tags) or tags.numel() != tensor.shape[1]:
            raise ValueError(
                "MiniMax H3 modality tags do not match the conditioning sequence length: "
                f"conditioning={tuple(tensor.shape)}, "
                f"tags={tuple(tags.shape) if torch.is_tensor(tags) else type(tags).__name__}, "
                f"metadata_keys={sorted(metadata)}."
            )
        metadata = metadata.copy()
        empty_padding = not actual_prompt and not base_vlm_images and not fusion_vlm_images and video_frames is None and audio_reference is None
        boundary = 0 if empty_padding else tensor.shape[1] - prompt_length
        metadata[LAYOUT_KEY] = build_layout(tensor, tags, boundary)
        layout_conditioning.append([tensor, metadata])
    conditioning = layout_conditioning

    if multiplier != 1.0:
        scaled = []
        for tensor, metadata in conditioning:
            metadata = metadata.copy()
            pooled = metadata.get("pooled_output")
            if pooled is not None:
                metadata["pooled_output"] = pooled * multiplier
            scaled.append([tensor * multiplier, metadata])
        conditioning = scaled

    keyframes = []
    if prepared_first is not None:
        keyframes.append(
            {"resolved_frame_index": 0, "latent": cache.encode_vae(vae, prepared_first)}
        )
    if prepared_last is not None:
        keyframes.append(
            {
                "resolved_frame_index": frame_count - 1,
                "latent": cache.encode_vae(vae, prepared_last),
            }
        )
    references = [
        {
            "kind": "image",
            "latent_h": image.shape[1] // 16,
            "latent_w": image.shape[2] // 16,
            "latent": cache.encode_vae(vae, image),
        }
        for image in prepared_references
    ] if native_reference_mode else []
    if video_reference is not None:
        references.append(video_reference)
    if audio_reference is not None:
        references.append(audio_reference)
    if positioned_video_keyframes:
        keyframes.extend(positioned_video_keyframes)
        keyframes.sort(key=lambda keyframe: keyframe["resolved_frame_index"])
    metadata = {}
    if keyframes:
        metadata["minimax_keyframes"] = keyframes
        metadata["minimax_frame_count"] = frame_count
    if references:
        metadata["minimax_refs"] = references
    if metadata:
        conditioning = node_helpers.conditioning_set_values(conditioning, metadata)
    return conditioning, latent


def execute_advanced_visual_consensus(
    clip,
    prompt,
    system_prompt,
    vlm_resolution,
    image_inputs,
    joint_config,
    vae_resolution,
    ref_latent_mode,
    vae,
    multiplier,
    vae_dimension_multiple,
    apply_reference_latents,
    token_fusion=False,
    semantic_anchor=False,
):
    """Run spatial fusion per resolution, then consensus over complete outputs."""
    if not isinstance(joint_config, dict):
        raise ValueError("Connect a Visual Consensus Configuration.")

    minimax_h3 = is_minimax_h3_text_encoder(clip)
    if minimax_h3 and ref_latent_mode != "off":
        raise ValueError(
            "MiniMax H3 reference latents require Core's MiniMax H3 reference conditioning node; set ref_latent_mode to off."
        )

    lanes, reference_images = build_visual_consensus_batch_lanes(image_inputs)
    if not lanes:
        clean_prompt, _ = prepare_image_placeholder_prompt(
            prompt, 0, False, "Advanced Visual Consensus Encoder"
        )
        formatted_prompt = (
            format_minimax_h3_prompt(clean_prompt, system_prompt)
            if minimax_h3
            else _format_advanced_visual_consensus_prompt(
                clean_prompt, system_prompt
            )
        )
        conditioning = encode_embedding_classical_scaled_bias(
            clip,
            formatted_prompt,
            skip_template=True,
        )
        return _scale_and_attach_visual_consensus_references(
            clip,
            conditioning,
            reference_images,
            vae_resolution,
            ref_latent_mode,
            vae,
            multiplier,
            vae_dimension_multiple,
            apply_reference_latents,
        )

    spatial_enabled = bool(joint_config.get("enable_spatial_fusion", True))
    consensus_enabled = bool(joint_config.get("enable_consensus", True))
    if minimax_h3 and not consensus_enabled and len(lanes) > 1:
        raise ValueError(
            "MiniMax H3 supports batch size 1; enable consensus or provide one batch lane."
        )
    visual_config = dict(joint_config["visual"])
    consensus_config = dict(joint_config["consensus"])
    visual_encoder_path = visual_config.get("visual_encoder_path", "grid-deepstack")

    prepared_prompt, _ = prepare_image_placeholder_prompt(
        prompt,
        max(len(lane) for lane in lanes),
        spatial_enabled or consensus_enabled,
        "Advanced Visual Consensus Encoder",
    )
    if not any(
        marker in prepared_prompt
        for marker in ("<|image_pad|>", "<|image|>", "<|vision_start|>")
    ):
        prepared_prompt = VISION_BLOCK + prepared_prompt
    if semantic_anchor and not minimax_h3:
        segments = prepared_prompt.split(VISION_BLOCK)
        anchored_prompt = [segments[0]]
        for index, segment in enumerate(segments[1:], start=1):
            anchored_prompt.extend((f"<Picture {index}>: {VISION_BLOCK}", segment))
        prepared_prompt = "".join(anchored_prompt)
    full_prompt = (
        format_minimax_h3_prompt(prepared_prompt, system_prompt)
        if minimax_h3
        else _format_advanced_visual_consensus_prompt(prepared_prompt, system_prompt)
    )

    if consensus_enabled:
        requested_samples = int(consensus_config.get("resolution_samples", 1))
        if (
            requested_samples < 1
            or requested_samples > 15
            or requested_samples % 2 == 0
        ):
            raise ValueError("Resolution samples must be an odd integer from 1 to 15.")
        effective_samples = requested_samples
    else:
        effective_samples = 1
    if (
        consensus_enabled
        and effective_samples > 1
        and resolve_vlm_resolution(vlm_resolution) is None
    ):
        raise ValueError(
            "Original VLM resolution cannot construct adjacent resolution samples; "
            "set resolution_samples to 1 or select a numeric VLM resolution."
        )

    completed_conditionings = []
    base_conditioning = None
    export_pending = True
    for lane in lanes:
        targets = vlm_resolution_samples(
            lane[0],
            vlm_resolution,
            effective_samples,
            consensus_config.get("sample_offset", VLM_RESOLUTION_STEP),
        )
        if len(targets) < effective_samples:
            raise ValueError(
                f"Could not construct {effective_samples} distinct adjacent VLM "
                "resolution samples for this batch lane."
            )
        for target in targets:
            sources = lane if spatial_enabled or consensus_enabled else lane[:1]
            if spatial_enabled and token_fusion:
                token_sources = [
                    _tokenize_visual_consensus_source(
                        clip, source, target, full_prompt
                    )
                    for source in sources
                ]
                token_fusion_config = dict(visual_config)
                token_fusion_config["save_blended_embeds"] = bool(
                    export_pending
                    and token_fusion_config.get("save_blended_embeds", False)
                )
                completed = encode_token_fused_visual_sources(
                    clip,
                    token_sources,
                    token_fusion_config,
                    visual_encoder_path=visual_encoder_path,
                )
                export_pending = False
                completed_conditionings.append(completed)
                if base_conditioning is None:
                    base_conditioning = completed
                continue
            branches = [
                _encode_visual_consensus_source(
                    clip, source, target, full_prompt, visual_encoder_path
                )
                for source in sources
            ]
            if spatial_enabled:
                completed = _spatially_fuse_visual_consensus_sources(
                    branches, visual_config, clip, export_pending
                )
                export_pending = False
                completed_conditionings.append(completed)
                if base_conditioning is None:
                    base_conditioning = completed
            else:
                source_conditionings = [
                    branch["conditioning"] for branch in branches
                ]
                completed_conditionings.extend(source_conditionings)
                if base_conditioning is None:
                    base_conditioning = source_conditionings[0]

    conditioning = (
        blend_complete_conditionings(completed_conditionings, consensus_config)
        if consensus_enabled
        else batch_complete_conditionings(completed_conditionings)
    )
    return _scale_and_attach_visual_consensus_references(
        clip,
        conditioning,
        reference_images,
        vae_resolution,
        ref_latent_mode,
        vae,
        multiplier,
        vae_dimension_multiple,
        apply_reference_latents,
    )


def _scale_and_attach_visual_consensus_references(
    clip,
    conditioning,
    reference_images,
    vae_resolution,
    ref_latent_mode,
    vae,
    multiplier,
    vae_dimension_multiple,
    apply_reference_latents,
):
    if multiplier != 1.0:
        conditioning = [
            [
                tensor * multiplier,
                {
                    **metadata,
                    **(
                        {"pooled_output": metadata["pooled_output"] * multiplier}
                        if metadata.get("pooled_output") is not None
                        else {}
                    ),
                },
            ]
            for tensor, metadata in conditioning
        ]

    ref_latents = []
    if vae is not None and ref_latent_mode != "off":
        resolutions = {
            "Ultra (512)": 512,
            "Turbo (768)": 768,
            "Fast (1024)": 1024,
            "Balanced (1280)": 1280,
            "Detailed (1536)": 1536,
        }
        for image in reference_images:
            if "single" in ref_latent_mode and ref_latents:
                break
            target = (
                None if vae_resolution == "Original" else resolutions[vae_resolution]
            )
            prepared = prepare_vae_reference_image(
                image.movedim(-1, 1), target, vae_dimension_multiple
            )
            ref_latents.append(vae.encode(prepared.movedim(1, -1)[..., :3]))
    return apply_reference_latents(
        clip, conditioning, ref_latents, ref_latent_mode
    )

def krea2_token_ids(clip, text):
    tok = clip.tokenize(text)
    key = next(iter(tok))
    return [t[0] if isinstance(t, tuple) else t for t in tok[key][0]]

def find_subsequence(seq, sub, lo, hi):
    out = []
    n = len(sub)
    if n == 0:
        return out
    for i in range(lo, hi - n + 1):
        if seq[i:i + n] == sub:
            out.append(i)
    return out


def krea2_attn_forward_weight(self, x, freqs=None, mask=None, transformer_options={}):

    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

    weights = transformer_options.get("krea2_token_weights")
    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        rep = self.heads // self.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    bias = None
    if weights and any(kb != 0.0 for _, kb in weights):
        bias = q.new_zeros(1, 1, 1, k.shape[2])
        for pos, kb in weights:
            if kb != 0.0 and pos < bias.shape[-1]:
                bias[..., pos] = kb
    if bias is not None:
        if mask is None:
            mask = bias
        else:
            if mask.dtype == torch.bool:
                additive_mask = torch.zeros(mask.shape, device=mask.device, dtype=q.dtype)
                additive_mask.masked_fill_(~mask, -torch.finfo(q.dtype).max)
            else:
                additive_mask = mask.to(dtype=q.dtype)
            mask = additive_mask + bias.to(device=mask.device, dtype=q.dtype)
    out = optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
    return self.wo(out * torch.sigmoid(gate))
