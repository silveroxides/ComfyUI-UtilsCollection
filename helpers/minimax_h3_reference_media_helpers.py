from __future__ import annotations

from fractions import Fraction
import math
import numbers
import re
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio

import comfy.model_management
import comfy.nested_tensor
import comfy.utils
from comfy.text_encoders.minimax import token_tags_from_embeds_info
import node_helpers

from .minimax_h3_cache_helpers import H3EncoderCache, H3_CACHE_MODES, temporal_cache_settings
from .minimax_h3_guide_helpers import LAYOUT_KEY, build_layout
from .minimax_h3_temporal_helpers import (
    encode_temporal_conditioning,
    fuse_temporal_block,
    minimax_h3_temporal_frame_pairs,
)
from .encoder_helpers import (
    _active_clip_model,
    _encode_preprocessed_clip_model,
    _encode_scheduled_with_visual_path,
    _position_biased_similarity_scores,
    build_token_to_conditioning_map,
    encode_embedding_classical_scaled_bias,
    fuse_visual_token_sources,
    qwen3vl_visual_encoder_path,
    resolve_consensus_blend_settings,
    visual_fusion_grid,
)


def is_minimax_h3_text_encoder(clip) -> bool:
    """Return True if the connected CLIP object is MiniMax H3 (qwen3vl_32b)."""
    stage = getattr(clip, "cond_stage_model", None)
    for owner in (getattr(clip, "tokenizer", None), stage):
        key_name = getattr(owner, "clip_name", None)
        if isinstance(key_name, str):
            return key_name == "qwen3vl_32b"
    if stage is not None:
        model = getattr(stage, "clip_model", stage)
        return type(model).__name__.lower() == "qwen3vl_32b"
    return False


def minimax_h3_frame_count(length: int) -> int:
    """Calculate the snapped frame count on MiniMax H3's 17k+5 temporal grid."""
    length = int(length)
    if length <= 5:
        return 5
    return ((length - 5 + 16) // 17) * 17 + 5


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


def prepare_vlm_image(image: torch.Tensor, vlm_resolution: int) -> torch.Tensor:
    """Prepare a visual tensor for Qwen VLM presentation."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("Expected 4D image tensor.")
    target = image[:1]
    if target.shape[-1] > 3:
        target = target[..., :3]
    b, h, w, c = target.shape
    if 256 <= vlm_resolution <= 4096:
        area = float(h * w)
        scale = math.sqrt(float(vlm_resolution * vlm_resolution) / area)
        th = max(28, round(h * scale / 28) * 28)
        tw = max(28, round(w * scale / 28) * 28)
        if (th, tw) != (h, w):
            chw = target.permute(0, 3, 1, 2)
            resized = F.interpolate(chw, size=(th, tw), mode="bilinear", align_corners=False)
            target = resized.permute(0, 2, 3, 1)
    return target


def prepare_minimax_h3_vlm_video_frames(frames: torch.Tensor, vlm_video_resolution: int) -> torch.Tensor:
    """Rescale video frames for Qwen VLM video conditioning."""
    if not torch.is_tensor(frames) or frames.ndim != 4:
        raise ValueError("Expected 4D video tensor.")
    if frames.shape[-1] > 3:
        frames = frames[..., :3]
    f, h, w, c = frames.shape
    if 256 <= vlm_video_resolution <= 4096:
        area = float(h * w)
        scale = math.sqrt(float(vlm_video_resolution * vlm_video_resolution) / area)
        th = max(28, round(h * scale / 28) * 28)
        tw = max(28, round(w * scale / 28) * 28)
        if (th, tw) != (h, w):
            chw = frames.permute(0, 3, 1, 2)
            resized = F.interpolate(chw, size=(th, tw), mode="bilinear", align_corners=False)
            frames = resized.permute(0, 2, 3, 1)
    return frames


def minimax_h3_video_sample_indices(frame_count: int, video_fps: int) -> list[int]:
    """Sample indices from video frames at specified FPS (assuming 24 fps native)."""
    step = max(1, round(24 / max(1, video_fps)))
    return list(range(0, frame_count, step))


def _encode_minimax_h3_audio_reference(audio, audio_vae, cache=None) -> dict | None:
    """Encode an audio track into an acoustic reference latent."""
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
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    hop_length = getattr(getattr(audio_vae, "first_stage_model", audio_vae), "hop_length", getattr(audio_vae, "downscale_ratio", None))
    if isinstance(hop_length, (int, float)) and hop_length > 1:
        hop = int(hop_length)
        rem = waveform.shape[-1] % hop
        if rem != 0:
            waveform = F.pad(waveform, (0, hop - rem))
    samples = waveform[:1].movedim(1, -1)
    latent = audio_vae.encode(samples) if cache is None else cache.encode_vae(audio_vae, samples, media="audio")
    if not torch.is_tensor(latent) or latent.ndim < 1 or latent.shape[-1] < 1:
        raise ValueError("MiniMax H3 audio VAE returned an invalid latent.")
    return {
        "kind": "audio",
        "audio_latent": latent,
        "ref_audio_t": latent.shape[-1],
    }


def prepare_minimax_h3_reference_video(
    video: torch.Tensor,
    vae,
    frame_count: int,
    encode_reference: bool = True,
    cache=None,
) -> tuple[torch.Tensor, dict | None]:
    """Format and VAE-encode video frames into an H3 reference block."""
    if not torch.is_tensor(video) or video.ndim != 4 or video.shape[0] < 1:
        raise ValueError("MiniMax H3 video must contain at least one frame.")
    frames = video
    if frames.shape[-1] > 3:
        frames = frames[..., :3]

    # Snap frames to H3's 17k+5 temporal grid
    n = min(frames.shape[0], int(frame_count))
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps).")
    while n % 17 != 5:
        n -= 1
    frames = frames[:n]

    # Canvas adaptation and strict 32-pixel spatial alignment for DiT 2x2 patches on 16x VAE
    source_height, source_width = frames.shape[1:3]
    ratio = source_width / source_height
    if ratio >= 1.0:
        target_width, target_height = 768.0 * ratio, 768.0
    else:
        target_width, target_height = 768.0, 768.0 / ratio
    if target_width * target_height > 768.0 * 1344.0:
        scale = math.sqrt((768.0 * 1344.0) / (target_width * target_height))
        target_width *= scale
        target_height *= scale
    target_width = max(32, round(target_width / 32) * 32)
    target_height = max(32, round(target_height / 32) * 32)
    if source_width * source_height < target_width * target_height:
        target_width = max(32, round(source_width / 32) * 32)
        target_height = max(32, round(source_height / 32) * 32)

    if (frames.shape[2], frames.shape[1]) != (target_width, target_height):
        samples = frames.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled")
        frames = samples.movedim(1, -1)

    ref_dict = None
    if encode_reference and vae is not None:
        latent = vae.encode(frames) if cache is None else cache.encode_vae(vae, frames, media="video")
        ref_dict = {
            "kind": "video",
            "latent": latent,
            "latent_t": latent.shape[2],
            "latent_h": latent.shape[3],
            "latent_w": latent.shape[4],
            "ref_audio_t": 0,
            "audio_latent": None,
        }
    return frames, ref_dict


def extract_minimax_h3_reference_media(reference_media) -> tuple[list, list]:
    """
    Extracts and validates video and audio references from a reference_media container.
    Returns:
        parsed_videos: list of (video_tensor, paired_soundtrack_dict_or_None)
        parsed_audios: list of standalone_audio_dict
    """
    if not isinstance(reference_media, dict) and reference_media is not None:
        raise ValueError("MiniMax H3 reference_media must be a dictionary.")

    ref_videos = (reference_media or {}).get("reference_videos") or {}
    ref_video_audios = (reference_media or {}).get("reference_video_audios") or {}
    ref_audios = (reference_media or {}).get("reference_audios") or {}

    valid_videos = {k: v for k, v in ref_videos.items() if v is not None}
    valid_audios = {k: v for k, v in ref_audios.items() if v is not None}

    parsed_videos = []
    if valid_videos:
        def get_ordinal(key):
            digits = re.findall(r"\d+", key)
            return int(digits[0]) if digits else 0

        sorted_video_keys = sorted(valid_videos.keys(), key=get_ordinal)
        for k in sorted_video_keys:
            vid = valid_videos[k]
            digits = re.findall(r"\d+", k)
            ordinal_str = digits[0] if digits else ""
            soundtrack = None
            if ordinal_str:
                for a_key, a_val in ref_video_audios.items():
                    if a_val is not None:
                        a_digits = re.findall(r"\d+", a_key)
                        if a_digits and a_digits[0] == ordinal_str:
                            soundtrack = a_val
                            break
            parsed_videos.append((vid, soundtrack))

    parsed_audios = []
    if valid_audios:
        def get_ordinal(key):
            digits = re.findall(r"\d+", key)
            return int(digits[0]) if digits else 0

        sorted_audio_keys = sorted(valid_audios.keys(), key=get_ordinal)
        for k in sorted_audio_keys:
            parsed_audios.append(valid_audios[k])

    return parsed_videos, parsed_audios


def execute_advanced_minimax_h3_reference_media_image_to_video(
    clip,
    vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    first_frame=None,
    last_frame=None,
    reference_images=None,
    multiplier: float = 1.0,
    ref_image_size: str = "match",
    vlm_resolution: int = 384,
    vlm_video_resolution: int = 384,
    media_config=None,
    audio_vae=None,
    cache=None,
    enable_caching: str = "all",
    reference_media=None,
    visual_fusion_config=None,
    temporal_fusion=False,
    temporal_token_fusion=False,
    text_blend_config=None,
    **kwargs,
):
    """Build coordinated Qwen conditioning, H3 controls, and AV latent strictly from reference_media.

    Execution Pipeline:
        1. Validates CLIP text encoder matches Qwen3-VL 32B (qwen3vl_32b).
        2. Unpacks reference_media into parsed reference videos and audios.
        3. Encodes latent video and audio references with VAE (pooled, refined, or full).
        4. Gathers static images (first/last frame, reference images) and dynamic video frames
           into reference items for Qwen multimodal tokenization.
        5. If temporal_fusion is active and reference videos exist:
           - Derives temporal_density (1-24) and temporal_fusion_method ("consensus" vs "spatial")
             from media_config.
           - Resolves mathematical consensus settings from text_blend_config (or default median).
           - When temporal_fusion_method is "spatial", spatial interleaving is governed by
             visual_fusion_config.
           - Builds multi-lane temporal frame pairs with minimax_h3_temporal_frame_pairs.
           - If temporal_token_fusion is True (token_fusion), fuses visual features and DeepStack
             before a single joint Qwen encode pass.
           - If temporal_token_fusion is False (conds_fusion), encodes each temporal lane separately
             and fuses conditioning tensors post-Qwen.
        6. Processes classical prompt weights e.g. (word:1.2) via encode_embedding_classical_scaled_bias.
        7. Attaches layout metadata, keyframes, reference descriptors, and AV empty latents.

    Returns:
        tuple[list[list[Tensor, dict]], dict]: (conditioning, latent)
    """
    if not is_minimax_h3_text_encoder(clip):
        raise ValueError(
            "Advanced MiniMax H3 Reference Media to Video requires the qwen3vl_32b text encoder."
        )
    if cache is None:
        with H3EncoderCache(enable_caching) as invocation:
            return execute_advanced_minimax_h3_reference_media_image_to_video(
                clip, vae, prompt, width, height, length,
                first_frame=first_frame, last_frame=last_frame, reference_images=reference_images,
                multiplier=multiplier, ref_image_size=ref_image_size,
                vlm_resolution=vlm_resolution, vlm_video_resolution=vlm_video_resolution,
                media_config=media_config, audio_vae=audio_vae,
                cache=invocation, enable_caching=enable_caching,
                reference_media=reference_media,
                visual_fusion_config=visual_fusion_config,
                temporal_fusion=temporal_fusion,
                temporal_token_fusion=temporal_token_fusion,
                text_blend_config=text_blend_config,
                **kwargs,
            )

    parsed_videos, parsed_audios = extract_minimax_h3_reference_media(reference_media)
    clip = cache.prepare_clip(clip)
    latent, frame_count = minimax_h3_empty_av_latent(width, height, length)

    video_fps = 2
    video_latent_mode = "full video"
    video_reference_resolution = 256
    video_latent_fps = 2.0
    refine_steps = 100
    if media_config is not None:
        video_fps = int(media_config.get("video_fps", 2))
        video_latent_mode = media_config.get("video_latent_mode", "full video")
        video_reference_resolution = int(media_config.get("video_reference_resolution", 256))
        video_latent_fps = float(media_config.get("video_latent_fps", 2.0))
        refine_steps = int(media_config.get("refine_steps", 100))

    encode_reference = (
        video_latent_mode in ("full video", "pooled", "refined")
        and vae is not None
    )

    prepared_reference_videos = []
    prepared_reference_audios = []

    if parsed_videos:
        for vid, soundtrack in parsed_videos:
            v_frames, v_ref = prepare_minimax_h3_reference_video(
                vid, vae, frame_count, encode_reference=encode_reference, cache=cache
            )
            if encode_reference and video_latent_mode in ("pooled", "refined") and v_ref is not None:
                source_latent_t = v_ref["latent"].shape[2]
                target_latent_frames = max(
                    1, min(source_latent_t, round(source_latent_t * (video_latent_fps / 24.0)))
                )
                target_h = max(2, (video_reference_resolution // 32) * 2)
                target_w = max(2, (video_reference_resolution // 32) * 2)
                target_shape = (target_latent_frames, target_h, target_w)
                if target_shape == tuple(v_ref["latent"].shape[2:]):
                    pooled_latent = v_ref["latent"].clone()
                else:
                    pooled_latent = F.adaptive_avg_pool3d(
                        v_ref["latent"].to(torch.float32), target_shape
                    ).to(v_ref["latent"].dtype)
                final_latent = pooled_latent
                v_ref = {
                    **v_ref,
                    "latent": final_latent,
                    "latent_t": final_latent.shape[2],
                    "latent_h": final_latent.shape[3],
                    "latent_w": final_latent.shape[4],
                }
            a_ref = None
            if soundtrack is not None and audio_vae is not None:
                a_ref = _encode_minimax_h3_audio_reference(soundtrack, audio_vae, cache=cache)
            if v_ref is not None:
                v_ref["kind"] = "video"
                v_ref["ref_audio_t"] = 0
                v_ref["audio_latent"] = None
            prepared_reference_videos.append((v_frames, soundtrack, v_ref, a_ref))

    for aud in parsed_audios:
        a_ref = None
        if aud is not None and audio_vae is not None:
            a_ref = _encode_minimax_h3_audio_reference(aud, audio_vae, cache=cache)
        if aud is not None:
            prepared_reference_audios.append((aud, a_ref))

    # Build reference items for presentation tokenization
    reference_items = []
    # 1. Frames & Images
    if first_frame is not None:
        vlm_first = prepare_vlm_image(first_frame, vlm_resolution)
        reference_items.append({"type": "image", "data": vlm_first})
    if last_frame is not None:
        vlm_last = prepare_vlm_image(last_frame, vlm_resolution)
        reference_items.append({"type": "image", "data": vlm_last})
    flat_images = []
    if reference_images is not None:
        if isinstance(reference_images, dict):
            def get_num(k):
                digits = re.findall(r"\d+", k)
                return int(digits[0]) if digits else 0

            for k in sorted(reference_images.keys(), key=get_num):
                v = reference_images[k]
                if torch.is_tensor(v):
                    if v.ndim == 4:
                        for i in range(v.shape[0]):
                            flat_images.append(v[i:i+1])
                    else:
                        flat_images.append(v)
        elif isinstance(reference_images, (list, tuple)):
            for v in reference_images:
                if torch.is_tensor(v):
                    if v.ndim == 4:
                        for i in range(v.shape[0]):
                            flat_images.append(v[i:i+1])
                    else:
                        flat_images.append(v)
        elif torch.is_tensor(reference_images):
            if reference_images.ndim == 4:
                for i in range(reference_images.shape[0]):
                    flat_images.append(reference_images[i:i+1])
            else:
                flat_images.append(reference_images)
    for img in flat_images:
        vlm_img = prepare_vlm_image(img, vlm_resolution)
        reference_items.append({"type": "image", "data": vlm_img})

    # 2. Videos
    for v_frames, _s_track, _v_ref, _a_ref in prepared_reference_videos:
        sample_indices = minimax_h3_video_sample_indices(v_frames.shape[0], video_fps)
        vlm_frames = prepare_minimax_h3_vlm_video_frames(
            v_frames[sample_indices], vlm_video_resolution
        )
        timestamps = [Fraction(index, 24) for index in sample_indices]
        if vlm_frames.shape[0] % 2 == 1:
            vlm_frames = torch.cat([vlm_frames, vlm_frames[-1:]], dim=0)
            timestamps.append(timestamps[-1])
        reference_items.append({
            "type": "video",
            "data": vlm_frames,
            "timestamps": timestamps,
        })

    # 3. Audio items (both soundtracks and standalone)
    for _v_frames, s_track, _v_ref, _a_ref in prepared_reference_videos:
        if s_track is not None:
            reference_items.append({"type": "audio"})
    for _aud, _ in prepared_reference_audios:
        reference_items.append({"type": "audio"})

    temporal_encode = None
    if temporal_fusion and prepared_reference_videos:
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
        visual_config = visual_fusion_config or {}
        visual_method = visual_config.get("visual_fusion_method", "off")
        enabled = settings["blend_preset"] != "off" if method == "consensus" else visual_method != "off"

        if density > 1 and enabled:
            all_frame_pairs = []
            for v_frames, _s_track, _v_ref, _a_ref in prepared_reference_videos:
                indices = minimax_h3_video_sample_indices(v_frames.shape[0], video_fps)
                v_pairs = minimax_h3_temporal_frame_pairs(v_frames.shape[0], indices, int(density))
                for block in v_pairs:
                    all_frame_pairs.append([(v_frames, pair) for pair in block])

            def fuse_video_block(sources, grids, deepstack):
                def compute():
                    return fuse_temporal_block(
                        sources, method, settings, visual_config, grids,
                        spatial_fuse_callback=fuse_visual_token_sources,
                        position_score_callback=_position_biased_similarity_scores,
                        deepstack_layers=deepstack,
                    )
                if len(sources) == 1 or temporal_token_fusion:
                    return compute()
                return cache.get_or_compute("encoded_section", {
                    "sources": sources, "grids": grids, "deepstack": deepstack,
                    "method": method, "settings": temporal_cache_settings(method, settings, visual_config),
                    "section": "temporal_post_qwen", "device": str(sources[0].device),
                }, compute)

            def temporal_encode(tokens, encode_tokens_callback=None):
                return encode_temporal_conditioning(
                    clip, tokens, all_frame_pairs,
                    lambda pair_info: prepare_minimax_h3_vlm_video_frames(
                        pair_info[0][list(pair_info[1])], vlm_video_resolution
                    ),
                    token_fusion=temporal_token_fusion,
                    fusion_callback=fuse_video_block,
                    encode_tokens_callback=encode_tokens_callback or (
                        lambda value: _encode_scheduled_with_visual_path(clip, value, "grid-deepstack", cache=cache)
                    ),
                    active_clip_model_callback=_active_clip_model,
                    encode_preprocessed_callback=_encode_preprocessed_clip_model,
                    visual_context_callback=lambda: qwen3vl_visual_encoder_path(clip, "grid-deepstack"),
                    video_grid_callback=lambda data, size: visual_fusion_grid(data, size, False),
                    token_spans_callback=build_token_to_conditioning_map,
                    cache=cache,
                )

    # Presentation tokenization
    tokenize_callback = lambda text: clip.tokenize(text, minimax_ref_items=reference_items)
    conditioning = encode_embedding_classical_scaled_bias(
        clip,
        prompt,
        tokenize_callback=tokenize_callback,
        visual_encoder_path="grid-deepstack",
        encode_callback=temporal_encode,
        cache=cache,
    )

    # Wrap conditioning with modality tags and layout
    layout_conditioning = []
    for tensor, metadata in conditioning:
        tags = metadata.get("minimax_token_tags")
        if not torch.is_tensor(tags) or tags.numel() != tensor.shape[1]:
            tags = token_tags_from_embeds_info(tensor.shape[1], metadata.get("embeds_info", {}))
            metadata["minimax_token_tags"] = tags
        metadata = metadata.copy()
        boundary = tensor.shape[1]
        metadata[LAYOUT_KEY] = build_layout(tensor, tags, boundary)
        layout_conditioning.append([tensor, metadata])
    conditioning = layout_conditioning

    if multiplier != 1.0:
        scaled = []
        for tensor, metadata in conditioning:
            meta = metadata.copy()
            pooled = meta.get("pooled_output")
            if pooled is not None:
                meta["pooled_output"] = pooled * multiplier
            scaled.append([tensor * multiplier, meta])
        conditioning = scaled

    # Collect keyframes and references
    keyframes = []
    if first_frame is not None and ref_image_size != "none" and vae is not None:
        samples = first_frame[..., :3].movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
        prepared_first = samples.movedim(1, -1)
        latent_first = vae.encode(prepared_first) if cache is None else cache.encode_vae(vae, prepared_first, media="image")
        keyframes.append({"resolved_frame_index": 0, "latent": latent_first})
    if last_frame is not None and ref_image_size != "none" and vae is not None:
        samples = last_frame[..., :3].movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
        prepared_last = samples.movedim(1, -1)
        latent_last = vae.encode(prepared_last) if cache is None else cache.encode_vae(vae, prepared_last, media="image")
        keyframes.append({"resolved_frame_index": frame_count - 1, "latent": latent_last})

    references = []
    for img in flat_images:
        if ref_image_size != "none" and vae is not None:
            img_h, img_w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (img_w * img_h)))
            elif ref_image_size == "max":
                scale = min(1.0, 2048.0 / min(img_w, img_h))
            else:
                scale = 1.0
            tw = max(32, round(img_w * scale / 32) * 32)
            th = max(32, round(img_h * scale / 32) * 32)

            prepared_img = img
            if (img.shape[2], img.shape[1]) != (tw, th):
                samples = img[..., :3].movedim(-1, 1)
                samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
                prepared_img = samples.movedim(1, -1)

            latent_img = vae.encode(prepared_img) if cache is None else cache.encode_vae(vae, prepared_img, media="image")
            references.append({
                "kind": "image",
                "latent_h": th // 16,
                "latent_w": tw // 16,
                "latent": latent_img,
            })
    for _, _, v_ref, _ in prepared_reference_videos:
        if v_ref is not None:
            references.append(v_ref)
    for _, _, _, a_ref in prepared_reference_videos:
        if a_ref is not None:
            references.append(a_ref)
    for _, a_ref in prepared_reference_audios:
        if a_ref is not None:
            references.append(a_ref)

    metadata = {}
    if keyframes:
        metadata["minimax_keyframes"] = keyframes
    if references:
        metadata["minimax_refs"] = references
    metadata["minimax_frame_count"] = frame_count
    conditioning = node_helpers.conditioning_set_values(conditioning, metadata)

    return conditioning, latent
