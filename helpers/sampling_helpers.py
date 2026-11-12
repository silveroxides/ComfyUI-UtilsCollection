"""Sampling helpers for windowed and looping generation workflows."""

import copy
import logging
import math
from collections.abc import Mapping
from typing import Any, Optional, Sequence

import comfy.model_management
import comfy.utils
import torch
import torch.nn.functional as F
import torchaudio
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas

from .video_helpers import _H3SegmentInt as _H3SegmentInt
from .video_helpers import split_h3_video_components_into_segments as split_h3_video_components_into_segments

LOGGER = logging.getLogger("ComfyUI-UtilsCollection.sampling")

# MiniMax H3 grid constants
H3_FPS = 24
H3_AUDIO_FPS = 40
H3_FRAME_GROUP = 17
H3_FRAME_BASE = 5
H3_LATENT_GROUP = 5
H3_LATENT_BASE = 2

H3_VIDEO_T_DIM = 2
H3_AUDIO_T_DIM = 3

H3_CHUNK_SECONDS_OPTIONS = [
    "0.00s (single pass)",
    "2.33s (56 frames)",
    "3.04s (73 frames)",
    "3.75s (90 frames)",
    "4.46s (107 frames)",
    "5.17s (124 frames)",
    "5.88s (141 frames)",
    "6.58s (158 frames)",
    "7.29s (175 frames)",
    "8.00s (192 frames)",
    "8.71s (209 frames)",
    "9.42s (226 frames)",
    "10.12s (243 frames)",
    "10.83s (260 frames)",
    "12.25s (294 frames)",
    "13.67s (328 frames)",
    "15.08s (362 frames)",
    "16.50s (396 frames)",
    "17.92s (430 frames)",
    "20.04s (481 frames)",
]

H3_OVERLAP_SECONDS_OPTIONS = ["0.00s (no overlap)", "0.21s (5 frames)", "0.92s (22 frames)", "1.62s (39 frames)", "2.33s (56 frames)", "3.04s (73 frames)", "3.75s (90 frames)"]


def parse_h3_seconds_option(option_str: Any, default_frames: int = 0) -> int:
    """Parse integer frame count from a seconds combo option string like '5.17s (124 frames)'."""
    if isinstance(option_str, (int, float)):
        return max(0, int(option_str))
    if not isinstance(option_str, str):
        return default_frames
    import re

    match = re.search(r"\((\d+)\s*frames?\)", option_str)
    if match:
        return int(match.group(1))
    if "single pass" in option_str or "no overlap" in option_str:
        return 0
    # Try parsing direct number string
    try:
        val = float(option_str.replace("s", "").strip())
        if val <= 0:
            return 0
        from .parameter_helpers import h3_video_length_from_seconds

        return h3_video_length_from_seconds(val)
    except Exception:
        return default_frames


def h3_snap_latent_t(n: int) -> int:
    """Snap latent count to H3's 5j+2 temporal grid (minimum 2)."""
    if n < H3_LATENT_BASE:
        return H3_LATENT_BASE
    return H3_LATENT_GROUP * ((n - H3_LATENT_BASE) // H3_LATENT_GROUP) + H3_LATENT_BASE


def h3_snap_frame_count(n: int) -> int:
    """Snap frame count to H3's 17j+5 temporal grid (minimum 5)."""
    frames = max(H3_FRAME_BASE, int(n))
    while frames % H3_FRAME_GROUP != H3_FRAME_BASE:
        frames += 1
    return frames


def h3_latents_to_frames(latent_t: int) -> int:
    """Convert H3 video latent count (5j+2) to frame count (17j+5)."""
    return H3_FRAME_GROUP * ((int(latent_t) - H3_LATENT_BASE) // H3_LATENT_GROUP) + H3_FRAME_BASE


def h3_frames_to_latents(frames: int) -> int:
    """Convert H3 frame count (17j+5) to video latent count (5j+2)."""
    frames = int(frames)
    if frames <= H3_FRAME_BASE:
        return H3_LATENT_BASE
    return ((frames - H3_FRAME_BASE) // H3_FRAME_GROUP) * H3_LATENT_GROUP + H3_LATENT_BASE


def h3_frame_at_latent(latent_idx: int) -> int:
    """Frame index corresponding to video latent index on (1,4,4,4,4) pattern."""
    idx = int(latent_idx)
    groups = idx // H3_LATENT_GROUP
    rem = idx % H3_LATENT_GROUP
    offsets = (0, 1, 5, 9, 13)
    return groups * H3_FRAME_GROUP + offsets[rem]


def h3_frames_to_audio_t(frames: int) -> int:
    """Calculate H3 audio latents from frame count (40Hz audio vs 24fps video)."""
    return int(round(int(frames) / H3_FPS * H3_AUDIO_FPS))


def h3_audio_idx_at_video_latent(v_latent: int, total_v: int, total_a: int) -> int:
    """Compute matching audio latent index from video latent index."""
    if total_v <= 0 or total_a <= 0 or v_latent <= 0:
        return 0
    if v_latent >= total_v:
        return total_a
    target_frames = h3_frame_at_latent(v_latent)
    a_idx = h3_frames_to_audio_t(target_frames)
    return max(0, min(total_a, a_idx))


def h3_unpack_av(latent_dict: dict[str, Any], label: str = "latent") -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpack H3 joint AV latent nested tensor into video and audio tensors."""
    samples = latent_dict.get("samples") if isinstance(latent_dict, dict) else latent_dict
    if isinstance(samples, NestedTensor):
        sub_tensors = samples.unbind()
        if len(sub_tensors) < 2:
            return sub_tensors[0], None
        return sub_tensors[0], sub_tensors[-1]
    if torch.is_tensor(samples):
        if samples.ndim == 5:
            return samples, None
        raise ValueError(f"{label} has unrecognized tensor shape {tuple(samples.shape)}")
    raise TypeError(f"{label} has unexpected type {type(samples)}")


def h3_pack_av(orig_dict: dict[str, Any], video_tensor: torch.Tensor, audio_tensor: Optional[torch.Tensor] = None, noise_mask: Optional[Any] = None) -> dict[str, Any]:
    """Package video and audio tensors into an H3 AV latent dictionary."""
    out = dict(orig_dict) if isinstance(orig_dict, dict) else {}
    if audio_tensor is not None:
        out["samples"] = NestedTensor([video_tensor, audio_tensor])
    else:
        out["samples"] = video_tensor
    if noise_mask is not None:
        out["noise_mask"] = noise_mask
    elif "noise_mask" in out:
        del out["noise_mask"]
    return out


def h3_split_noise_mask(latent_dict: dict[str, Any]) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract individual video and audio noise masks from latent dict."""
    m = latent_dict.get("noise_mask") if isinstance(latent_dict, dict) else None
    if m is None:
        return None, None
    if isinstance(m, NestedTensor):
        parts = m.unbind()
        return parts[0], parts[-1]
    if getattr(m, "ndim", 0) == 5:
        return m, None
    return None, m


def prepare_h3_source_audio(audio: dict[str, Any], audio_vae: Any, audio_template: torch.Tensor, total_frames: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit raw source audio to an H3 clip and encode it into its audio latent."""
    if not isinstance(audio, Mapping) or not torch.is_tensor(audio.get("waveform")):
        raise ValueError("UC_H3LoopSampler source_audio must contain a waveform tensor.")
    waveform = audio["waveform"]
    sample_rate = audio.get("sample_rate")
    if waveform.ndim != 3 or waveform.shape[1] not in (1, 2) or waveform.shape[-1] < 1 or not torch.is_floating_point(waveform) or not torch.isfinite(waveform).all() or not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError("UC_H3LoopSampler source_audio requires a finite mono or stereo waveform and positive sample rate.")
    if audio_vae is None or not callable(getattr(audio_vae, "encode", None)):
        raise ValueError("UC_H3LoopSampler source_audio requires an audio_vae input.")

    source = waveform[:1]
    if source.shape[1] == 1:
        source = source.repeat(1, 2, 1)
    target_samples = int(round(int(total_frames) * float(sample_rate) / H3_FPS))
    if source.shape[-1] > target_samples:
        source = source[..., :target_samples]
    elif source.shape[-1] < target_samples:
        source = F.pad(source, (0, target_samples - source.shape[-1]))
    passthrough = {"waveform": source, "sample_rate": int(sample_rate)}

    vae_rate = getattr(audio_vae, "audio_sample_rate", sample_rate)
    encoded_source = source
    if vae_rate != sample_rate:
        encoded_source = torchaudio.functional.resample(encoded_source, sample_rate, vae_rate)
    hop = getattr(getattr(audio_vae, "first_stage_model", audio_vae), "hop_length", getattr(audio_vae, "downscale_ratio", None))
    if isinstance(hop, (int, float)) and hop > 1:
        remainder = encoded_source.shape[-1] % int(hop)
        if remainder:
            encoded_source = F.pad(encoded_source, (0, int(hop) - remainder))

    encoded = audio_vae.encode(encoded_source.movedim(1, -1))
    if not torch.is_tensor(encoded) or encoded.ndim != audio_template.ndim or encoded.shape[:-1] != audio_template.shape[:-1]:
        raise ValueError("UC_H3LoopSampler audio_vae returned an incompatible audio latent.")
    encoded = encoded.to(dtype=audio_template.dtype, device=audio_template.device)
    target_t = audio_template.shape[-1]
    if encoded.shape[-1] > target_t:
        encoded = encoded[..., :target_t]
    elif encoded.shape[-1] < target_t:
        encoded = F.pad(encoded, (0, target_t - encoded.shape[-1]))
    return encoded.contiguous(), passthrough


def check_core_per_row_masking() -> bool:
    """Check if installed ComfyUI core provides per-row continuous masking."""
    try:
        import comfy.ldm.minimax.model as mm

        return hasattr(mm, "mask_row_values")
    except Exception:
        return False


def check_core_any_index_guides() -> bool:
    """Check if installed ComfyUI core allows keyframe guides at arbitrary indices."""
    try:
        import comfy.ldm.minimax.model as mm

        return hasattr(mm, "PackedLayout")
    except Exception:
        return False


def chunk_noise(noise: Any, index: int) -> Any:
    """Return a unique noise object per chunk with index added to seed.

    Prevents identical noise initialization across chunks which causes motion stutter and seam rewinds.
    """
    if index == 0 or noise is None or not hasattr(noise, "seed"):
        return noise
    n = copy.copy(noise)
    try:
        n.seed = int(noise.seed) + index
    except Exception:
        pass
    return n


def plan_h3_schedule(total_frames: int, window_frames: int = 124, overlap_frames: int = 22, segment_lengths: Optional[Sequence[int]] = None) -> tuple[int, int, list[dict[str, Any]], list[tuple[int, int]]]:
    """Single authoritative scheduler for MiniMax H3 windowed sampling and reference slicing.

    Returns:
        (total_latents, total_audio_latents, window_records, frame_spans)
        where each window_record is a dict:
            {
                'chunk_index': int,
                'v0': int, 'v1': int,
                'a0': int, 'a1': int,
                'carried_v': int, 'carried_a': int,
                'frame_start': int, 'frame_end': int,
            }
        and frame_spans is a list of (start_frame, end_frame) inclusive pixel ranges.
    """
    total_f = h3_snap_frame_count(int(total_frames))
    total_v = h3_frames_to_latents(total_f)
    total_a = h3_frames_to_audio_t(total_f)

    # 1. Determine latent window bounds (v0, v1)
    if segment_lengths:
        if isinstance(segment_lengths, (int, float)):
            segment_lengths = [int(segment_lengths)]
        elif not isinstance(segment_lengths, (list, tuple)):
            try:
                segment_lengths = list(segment_lengths)
            except Exception:
                segment_lengths = [int(segment_lengths)]

        ov_latents = 0
        if overlap_frames > 0:
            raw_ov = h3_frames_to_latents(int(overlap_frames))
            ov_latents = h3_snap_latent_t(raw_ov) if raw_ov >= H3_LATENT_BASE else 0

        raw_windows = []
        curr_f = 0
        for i, seg_f in enumerate(segment_lengths):
            seg_len = int(seg_f)
            start_f = curr_f
            end_f = min(total_f, curr_f + seg_len)
            nominal_v0 = h3_frames_to_latents(start_f) if start_f > 0 else 0
            v1 = min(total_v, h3_frames_to_latents(end_f))

            v0 = nominal_v0
            if i > 0 and ov_latents > 0:
                v0 = max(0, nominal_v0 - ov_latents)

            if v1 > v0:
                raw_windows.append((v0, v1))
            curr_f = end_f
            if curr_f >= total_f:
                break
    else:
        w_frames = max(H3_FRAME_BASE, int(window_frames)) if window_frames > 0 else total_f
        w_latents = min(total_v, h3_snap_latent_t(h3_frames_to_latents(w_frames)))
        if w_latents <= H3_LATENT_BASE or w_latents >= total_v:
            raw_windows = [(0, total_v)]
        else:
            ov_latents = 0
            if overlap_frames > 0:
                raw_ov = h3_frames_to_latents(int(overlap_frames))
                ov_latents = h3_snap_latent_t(raw_ov) if raw_ov >= H3_LATENT_BASE else 0
                ov_latents = min(ov_latents, max(0, w_latents - H3_LATENT_GROUP))

            step = max(H3_LATENT_GROUP, w_latents - ov_latents)
            step = (step // H3_LATENT_GROUP) * H3_LATENT_GROUP
            if step < H3_LATENT_GROUP:
                step = H3_LATENT_GROUP

            raw_windows = []
            start = 0
            while start < total_v:
                end = min(total_v, start + w_latents)
                raw_windows.append((start, end))
                if end >= total_v:
                    break
                start += step
                if start + w_latents > total_v:
                    final_start = max(0, total_v - w_latents)
                    if final_start > raw_windows[-1][0]:
                        raw_windows.append((final_start, total_v))
                    break

    # 2. Build detailed records with exact audio alignment and carry calculations
    window_records = []
    frame_spans = []

    for i, (v0, v1) in enumerate(raw_windows):
        f_start = h3_frame_at_latent(v0)
        f_end = min(total_f, h3_frame_at_latent(v1))

        a0 = h3_audio_idx_at_video_latent(v0, total_v, total_a)
        a1 = h3_audio_idx_at_video_latent(v1, total_v, total_a)

        # Clamped actual tail carry: previous window's true reach minus this window's start
        prev_end = raw_windows[i - 1][1] if i > 0 else 0
        carried_v = min(max(0, prev_end - v0), v1 - v0) if i > 0 else 0
        carried_a = (h3_audio_idx_at_video_latent(v0 + carried_v, total_v, total_a) - a0) if carried_v > 0 else 0

        window_records.append({"chunk_index": i, "v0": v0, "v1": v1, "a0": a0, "a1": a1, "carried_v": carried_v, "carried_a": carried_a, "frame_start": f_start, "frame_end": f_end})
        frame_spans.append((f_start, f_end))

    return total_v, total_a, window_records, frame_spans


def plan_h3_windows(total_frames: int, window_frames: int, overlap_frames: int, segment_lengths: Optional[Sequence[int]] = None) -> list[tuple[int, int]]:
    """Backward compatible wrapper around plan_h3_schedule."""
    _, _, records, _ = plan_h3_schedule(total_frames=total_frames, window_frames=window_frames, overlap_frames=overlap_frames, segment_lengths=segment_lengths)
    return [(r["v0"], r["v1"]) for r in records]


def prepare_chunk_guider(guider: Any, positive_cond: Any, frame0: Optional[int] = None) -> Any:
    """Create isolated clone of guider with swapped positive cond and timeline frame0 offsets."""
    chunk_g = copy.copy(guider)
    chunk_g.original_conds = dict(getattr(guider, "original_conds", {}) or {})

    if frame0 is not None:
        opts = dict(getattr(guider, "model_options", {}) or {})
        t_opts = dict(opts.get("transformer_options", {}))
        # Set both custom and standard H3 control offsets so motion/control models track time correctly
        t_opts["h3_ctrl_frame0"] = int(frame0)
        t_opts["mmh3_control_frame0"] = int(frame0)
        opts["transformer_options"] = t_opts
        chunk_g.model_options = opts

    raw_conds = getattr(guider, "raw_conds", None)
    if raw_conds is None:
        c_dict = getattr(guider, "original_conds", {}) or {}
        raw_conds = (c_dict.get("positive"), c_dict.get("negative"))

    _, negative = raw_conds
    if hasattr(chunk_g, "inner_set_conds") and callable(chunk_g.inner_set_conds):
        chunk_g.inner_set_conds({"positive": positive_cond})
    elif negative is None:
        chunk_g.set_conds(positive_cond)
    else:
        chunk_g.set_conds(positive_cond, negative)
    chunk_g.raw_conds = (positive_cond, negative)
    return chunk_g


def strip_stale_keyframes(cond: Any) -> list[Any]:
    """Strip prior or stale keyframe metadata from conditioning list."""
    if not isinstance(cond, list):
        return cond
    out = []
    for item in cond:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
            d = {k: v for k, v in item[1].items() if k not in ("minimax_keyframes", "minimax_frame_count")}
            out.append([item[0], d])
        else:
            out.append(item)
    return out


def build_carry_noise_mask(sub_v: torch.Tensor, sub_a: Optional[torch.Tensor], carried_v: int, carried_a: int, strength_v: float, strength_a: float, master_mask_v: Optional[torch.Tensor], master_mask_a: Optional[torch.Tensor], v0: int, v1: int, a0: int, a1: int) -> Any:
    """Build nested carry noise mask ensuring previous chunk tail is preserved."""
    vm = torch.ones([sub_v.shape[0], 1] + list(sub_v.shape[2:]), dtype=torch.float32, device=sub_v.device)
    if master_mask_v is not None:
        vm = master_mask_v[:, :, v0:v1].to(dtype=vm.dtype, device=vm.device).clone()

    am = None
    if sub_a is not None:
        am = torch.ones([sub_a.shape[0], 1, sub_a.shape[2], sub_a.shape[3]], dtype=torch.float32, device=sub_a.device)
        if master_mask_a is not None:
            am = master_mask_a[:, :, :, a0:a1].to(dtype=am.dtype, device=am.device).clone()

    if carried_v > 0:
        vm[:, :, :carried_v] = torch.minimum(vm[:, :, :carried_v], torch.full_like(vm[:, :, :carried_v], 1.0 - float(strength_v)))

    if am is not None and carried_a > 0:
        am[:, :, :, :carried_a] = torch.minimum(am[:, :, :, :carried_a], torch.full_like(am[:, :, :, :carried_a], 1.0 - float(strength_a)))

    return NestedTensor([vm, am]) if am is not None else vm


def run_chunk_sampling(noise: Any, guider: Any, sampler: Any, sigmas: torch.Tensor, latent_dict: dict[str, Any], start_step: int = 0, end_step: int = 1000, phase2_sampler: Optional[Any] = None, phase2_guider: Optional[Any] = None, phase2_start_step: int = 0) -> dict[str, Any]:
    """Run one chunk's schedule using SamplerCustomAdvanced and SplitSigmas."""
    start = max(0, int(start_step))
    curr_sigmas = sigmas
    if start > 0:
        if end_step <= start:
            raise ValueError(f"sampling start step ({start}) must be strictly less than end step ({end_step})")
        _, curr_sigmas = SplitSigmas().get_sigmas(curr_sigmas, start)
        if len(curr_sigmas) <= 1:
            raise ValueError(f"sampling start step {start} leaves no sigmas to sample")

    use_phase2 = phase2_sampler is not None and int(phase2_start_step) > 0
    phase2_local = int(phase2_start_step) - start
    effective_end = int(end_step) - start

    cut_points = sorted(p for p in ({phase2_local} if use_phase2 else set()) if 0 < p < effective_end)

    segments = []
    rem = curr_sigmas
    prev = 0
    for p in cut_points:
        seg, rem = SplitSigmas().get_sigmas(rem, p - prev)
        segments.append((prev, seg))
        prev = p
    tail, _ = SplitSigmas().get_sigmas(rem, effective_end - prev)
    segments.append((prev, tail))

    current = latent_dict
    for seg_start, seg_sigmas in segments:
        if len(seg_sigmas) <= 1:
            continue
        seg_sampler = sampler
        seg_guider = guider
        if use_phase2 and seg_start >= phase2_local:
            seg_sampler = phase2_sampler
            seg_guider = phase2_guider if phase2_guider is not None else guider
        _, current = SamplerCustomAdvanced().sample(noise, seg_guider, seg_sampler, seg_sigmas, current)

    return current


def reset_model_caches(guider: Any) -> None:
    """Reset any residual or attention caches attached to the model before sampling a chunk."""
    if guider is None:
        return
    model_patcher = getattr(guider, "model_patcher", None)
    if model_patcher is None:
        model_patcher = getattr(guider, "model", None)
    if model_patcher is None:
        return

    opts = getattr(model_patcher, "model_options", {}) or {}
    to = opts.get("transformer_options", {}) or {}
    patches_replace = to.get("patches_replace", {}) or {}
    dit_patches = patches_replace.get("dit", {}) or {}

    for k, patch_obj in dit_patches.items():
        if hasattr(patch_obj, "reset") and callable(patch_obj.reset):
            try:
                patch_obj.reset()
            except Exception:
                pass


def start_sampling_loop(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    cond_list: Sequence[Any],
    latent: dict[str, Any],
    chunk_frames: int = 124,
    overlap_frames: int = 22,
    segment_lengths: Optional[Sequence[int]] = None,
    carry_mode: str = "mask",
    overlap_strength_video: float = 1.0,
    overlap_strength_audio: float = 0.9,
    audio_mode: str = "preserve_input",
    sampling_start_step: int = 0,
    sampling_end_step: int = 1000,
    phase2_start_step: int = 0,
    phase2_sampler: Optional[Any] = None,
    phase2_guider: Optional[Any] = None,
    denoise_mask: Optional[torch.Tensor] = None,
    audio_denoise_mask: Optional[torch.Tensor] = None,
    source_audio: Optional[dict[str, Any]] = None,
    audio_vae: Optional[Any] = None,
) -> tuple[dict[str, Any], int, str, Optional[dict[str, Any]]]:
    """Execute looping sampling over an H3 whole-clip AV latent."""
    if not cond_list:
        raise ValueError("start_sampling_loop: conditioning list is empty")

    master_v, master_a = h3_unpack_av(latent, "latent")
    total_v = int(master_v.shape[H3_VIDEO_T_DIM])
    total_a = 0 if master_a is None else int(master_a.shape[H3_AUDIO_T_DIM])
    total_f = h3_latents_to_frames(total_v)

    # Check if target latent is smaller than requested segments/conds and expand it automatically
    needed_frames = 0
    if segment_lengths:
        needed_frames = sum(int(x) for x in segment_lengths)
    elif chunk_frames > 0 and len(cond_list) > 1:
        needed_frames = chunk_frames + max(0, len(cond_list) - 1) * max(1, chunk_frames - overlap_frames)

    if needed_frames > total_f:
        target_f = h3_snap_frame_count(needed_frames)
        target_v = h3_frames_to_latents(target_f)
        target_a = h3_frames_to_audio_t(target_f)
        LOGGER.info(f"UC_H3LoopSampler: expanding master latent from {total_f} frames ({total_v} latents) to {target_f} frames ({target_v} latents) to accommodate all {len(cond_list)} chunks.")
        expanded_v = torch.zeros([master_v.shape[0], master_v.shape[1], target_v, master_v.shape[3], master_v.shape[4]], dtype=master_v.dtype, device=master_v.device)
        expanded_v[:, :, :total_v] = master_v
        master_v = expanded_v
        total_v = target_v
        total_f = target_f

        if master_a is not None and target_a > total_a:
            expanded_a = torch.zeros([master_a.shape[0], master_a.shape[1], master_a.shape[2], target_a], dtype=master_a.dtype, device=master_a.device)
            expanded_a[:, :, :, :total_a] = master_a
            master_a = expanded_a
            total_a = target_a

    passthrough_audio = None
    if source_audio is not None:
        if audio_mode != "preserve_input":
            raise ValueError("UC_H3LoopSampler source_audio requires audio_mode 'preserve_input'.")
        if master_a is None:
            raise ValueError("UC_H3LoopSampler source_audio requires a joint video/audio latent.")
        master_a, passthrough_audio = prepare_h3_source_audio(source_audio, audio_vae, master_a, total_f)

    total_v, total_a, window_records, _ = plan_h3_schedule(total_frames=total_f, window_frames=chunk_frames, overlap_frames=overlap_frames, segment_lengths=segment_lengths)
    num_chunks = len(window_records)

    out_v = master_v.clone()
    out_a = None if master_a is None else master_a.clone()
    in_mask_v, in_mask_a = h3_split_noise_mask(latent)
    if source_audio is not None:
        in_mask_a = torch.zeros([master_a.shape[0], 1, master_a.shape[2], master_a.shape[3]], dtype=torch.float32, device=master_a.device)

    lines = [f"Sampling {num_chunks} chunk(s) over {total_v} latents ({total_f} frames, {total_f / float(H3_FPS):.2f}s), carry mode: {carry_mode}, audio mode: {audio_mode}"]

    for i, w in enumerate(window_records):
        v0, v1 = w["v0"], w["v1"]
        a0, a1 = w["a0"], w["a1"]
        carried_v, carried_a = w["carried_v"], w["carried_a"]

        sub_v = out_v[:, :, v0:v1].clone()
        sub_a = None if out_a is None else out_a[:, :, :, a0:a1].clone()

        chunk_latent = h3_pack_av(latent, sub_v, sub_a)
        if audio_mode == "full_generation":
            # Strip audio carry so model generates audio cleanly without pinning
            if carried_v > 0:
                chunk_latent["noise_mask"] = build_carry_noise_mask(
                    sub_v,
                    sub_a,
                    carried_v,
                    0,  # no carried audio pinning
                    overlap_strength_video,
                    0.0,
                    in_mask_v,
                    in_mask_a,
                    v0,
                    v1,
                    a0,
                    a1,
                )
        elif carried_v > 0:
            chunk_latent["noise_mask"] = build_carry_noise_mask(sub_v, sub_a, carried_v, carried_a, overlap_strength_video, overlap_strength_audio, in_mask_v, in_mask_a, v0, v1, a0, a1)
        elif in_mask_v is not None or in_mask_a is not None:
            vm = in_mask_v[:, :, v0:v1].clone() if in_mask_v is not None else None
            am = in_mask_a[:, :, :, a0:a1].clone() if in_mask_a is not None else None
            if vm is not None and am is not None:
                chunk_latent["noise_mask"] = NestedTensor([vm, am])
            else:
                chunk_latent["noise_mask"] = vm if vm is not None else am

        cond_idx = min(i, len(cond_list) - 1)
        raw_cond = cond_list[cond_idx]
        chunk_cond = strip_stale_keyframes(raw_cond)

        frame0 = h3_frame_at_latent(v0)
        chunk_g = prepare_chunk_guider(guider, chunk_cond, frame0=frame0)
        chunk_g2 = None
        if phase2_guider is not None:
            chunk_g2 = prepare_chunk_guider(phase2_guider, chunk_cond, frame0=frame0)

        # Reset any block residual caches so each chunk gets a clean schedule start
        reset_model_caches(chunk_g)
        if chunk_g2 is not None:
            reset_model_caches(chunk_g2)

        # Generate unique noise per chunk to prevent repeated motion loops
        c_noise = chunk_noise(noise, i)

        sampled = run_chunk_sampling(c_noise, chunk_g, sampler, sigmas, chunk_latent, start_step=sampling_start_step, end_step=sampling_end_step, phase2_sampler=phase2_sampler, phase2_guider=chunk_g2, phase2_start_step=phase2_start_step)

        res_v, res_a = h3_unpack_av(sampled, f"chunk_{i}_output")
        out_v[:, :, v0:v1] = res_v.to(out_v.dtype)
        if out_a is not None and res_a is not None:
            out_a[:, :, :, a0:a1] = res_a.to(out_a.dtype)

        lines.append(f"  chunk {i}: prompt {cond_idx}, frames {w['frame_start']}-{w['frame_end']} (latents {v0}-{v1}), carried {carried_v}")

    # Restore clean input audio if audio_mode is 'preserve_input'
    if audio_mode == "preserve_input" and master_a is not None:
        out_a = master_a.clone()

    final_mask = None
    if source_audio is not None:
        final_v_mask = in_mask_v
        if final_v_mask is None:
            final_v_mask = torch.ones([out_v.shape[0], 1] + list(out_v.shape[2:]), dtype=torch.float32, device=out_v.device)
        final_mask = NestedTensor([final_v_mask, in_mask_a])
    final_latent = h3_pack_av(latent, out_v, out_a, noise_mask=final_mask)
    report = "\n".join(lines)
    return final_latent, num_chunks, report, passthrough_audio


def _unwrap_first_scalar(val: Any) -> Any:
    while isinstance(val, list) and len(val) > 0:
        if isinstance(val[0], dict) and "samples" in val[0]:
            return val[0]
        val = val[0]
    return val


def _unwrap_conditioning_list(conditioning: Any) -> list[Any]:
    if isinstance(conditioning, dict) and "conds" in conditioning:
        return list(conditioning["conds"])
    if isinstance(conditioning, list):
        unwrapped = conditioning
        if len(conditioning) == 1 and isinstance(conditioning[0], list):
            if len(conditioning[0]) > 0 and isinstance(conditioning[0][0], list):
                unwrapped = conditioning[0]
        if len(unwrapped) > 0 and isinstance(unwrapped[0], list):
            if len(unwrapped[0]) > 0 and isinstance(unwrapped[0][0], list):
                return list(unwrapped)
            if len(unwrapped[0]) == 2 and isinstance(unwrapped[0][1], dict):
                return [unwrapped]
            return list(unwrapped)
        return [unwrapped]
    return [conditioning]


def _unwrap_latent_list(latent: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def _collect(item: Any) -> None:
        if isinstance(item, dict) and "samples" in item:
            out.append(item)
        elif isinstance(item, list):
            for sub in item:
                _collect(sub)

    _collect(latent)
    return out


def _unwrap_guider_list(guider: Any) -> list[Any]:
    out: list[Any] = []

    def _collect(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, list):
            for sub in item:
                _collect(sub)
        else:
            out.append(item)

    _collect(guider)
    return out


def _extract_positive_cond_from_guider(guider: Any) -> list[Any]:
    raw_conds = getattr(guider, "raw_conds", None)
    if raw_conds is not None and raw_conds[0] is not None:
        return raw_conds[0]
    orig = getattr(guider, "original_conds", None) or {}
    pos = orig.get("positive")
    if not pos:
        return []
    out = []
    for item in pos:
        if isinstance(item, dict):
            emb = item.get("cross_attn")
            extra = {k: v for k, v in item.items() if k not in ("cross_attn", "uuid")}
            out.append([emb, extra])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append([item[0], dict(item[1])])
    return out


def _unwrap_scalar_or_indexed(val: Any, index: int) -> Any:
    if isinstance(val, list) and len(val) > 0:
        item = val[min(index, len(val) - 1)]
        while isinstance(item, list) and len(item) > 0:
            item = item[0]
        return item
    return val


def sample_minimax_h3_clip_continuation_chain(noise: Any, guider: Any, sampler: Any, sigmas: Any, latent: Any, tail_frames: int | str | Sequence[int | str] = "22", audio_tail_frames: int | Sequence[int] = 24) -> tuple[list[dict[str, Any]], list[int], list[int], dict[str, Any], dict[str, Any]]:
    """Sample a list of MiniMax H3 segment guiders and latents in one run, chaining each segment's sampled AV latent tail into the next."""
    from .model_helpers import apply_minimax_h3_clip_continuation, h3_steps_for_frames

    noise_val = _unwrap_first_scalar(noise)
    sampler_val = _unwrap_first_scalar(sampler)
    sigmas_val = _unwrap_first_scalar(sigmas)
    guider_list = _unwrap_guider_list(guider)
    default_tail_f = int(_unwrap_first_scalar(tail_frames) or 22)
    default_aud_tail_f = int(_unwrap_first_scalar(audio_tail_frames) if audio_tail_frames is not None else 24)

    if not guider_list:
        raise ValueError("UC_MiniMaxH3ClipContinuationSampler requires a 'guider' input (from CFGGuider or BasicGuider).")

    cond_list = [_extract_positive_cond_from_guider(g) for g in guider_list]
    cond_list = [c for c in cond_list if c]
    latent_list = _unwrap_latent_list(latent)
    if not cond_list:
        raise ValueError("UC_MiniMaxH3ClipContinuationSampler guider has no positive conditioning.")
    if not latent_list:
        raise ValueError("UC_MiniMaxH3ClipContinuationSampler requires at least one latent_image input.")

    num_segments = max(len(cond_list), len(guider_list), len(latent_list))
    sampled_latents: list[dict[str, Any]] = []
    trim_leading_list: list[int] = []
    trim_trailing_list: list[int] = []

    for i in range(num_segments):
        base_guider = guider_list[min(i, len(guider_list) - 1)]
        seg_cond = cond_list[min(i, len(cond_list) - 1)]
        raw_lat = latent_list[min(i, len(latent_list) - 1)]
        sub_v, sub_a = h3_unpack_av(raw_lat, f"segment_{i}_latent")
        seg_latent = h3_pack_av(raw_lat, sub_v.clone(), None if sub_a is None else sub_a.clone())

        cond_meta = seg_cond[0][1] if seg_cond and len(seg_cond[0]) > 1 and isinstance(seg_cond[0][1], dict) else {}
        indexed_tail_f = int(_unwrap_scalar_or_indexed(tail_frames, i) or default_tail_f)
        indexed_aud_tail_f = int(_unwrap_scalar_or_indexed(audio_tail_frames, i) if audio_tail_frames is not None else default_aud_tail_f)

        lat_ov = raw_lat.get("uc_overlap_frames", cond_meta.get("uc_overlap_frames"))
        seg_tail_frames = int(lat_ov) if lat_ov is not None and int(lat_ov) > 0 else indexed_tail_f
        lat_pad = raw_lat.get("uc_padded_frames", cond_meta.get("uc_padded_frames", 0))
        seg_padded_frames = int(lat_pad or 0)

        if i == 0:
            existing_trim = int(cond_meta.get("minimax_trim_frames", 0))
            active_cond = seg_cond
            trim_i = existing_trim
        else:
            prev_sampled = sampled_latents[i - 1]
            active_cond, trim_i = apply_minimax_h3_clip_continuation(seg_cond, seg_latent, tail_frames=seg_tail_frames, audio_tail_frames=indexed_aud_tail_f, continuation_latent=prev_sampled)

        chunk_g = prepare_chunk_guider(base_guider, active_cond)
        reset_model_caches(chunk_g)

        c_noise = chunk_noise(noise_val, i)
        sampled = run_chunk_sampling(c_noise, chunk_g, sampler_val, sigmas_val, seg_latent)
        sampled_latents.append(sampled)
        trim_leading_list.append(int(trim_i))
        trim_trailing_list.append(seg_padded_frames)

    # Assemble combined full-length AV latent by removing the pinned overlap head from segments 1..N-1
    comb_v_parts: list[torch.Tensor] = []
    comb_a_parts: list[torch.Tensor] = []
    for i, (seg_out, trim_f) in enumerate(zip(sampled_latents, trim_leading_list)):
        sv, sa = h3_unpack_av(seg_out, f"sampled_segment_{i}")
        if i == 0 or trim_f <= 0:
            comb_v_parts.append(sv)
            if sa is not None:
                comb_a_parts.append(sa)
        else:
            ov_v = h3_steps_for_frames(trim_f) or 0
            ov_a = int(round(float(trim_f) / float(H3_FPS) * float(H3_AUDIO_FPS)))
            comb_v_parts.append(sv[:, :, ov_v:] if ov_v < sv.shape[2] else sv[:, :, :0])
            if sa is not None:
                comb_a_parts.append(sa[..., ov_a:] if ov_a < sa.shape[-1] else sa[..., :0])

    combined_v = torch.cat(comb_v_parts, dim=2) if len(comb_v_parts) > 1 else comb_v_parts[0].clone()
    combined_a = (torch.cat(comb_a_parts, dim=-1) if len(comb_a_parts) > 1 else comb_a_parts[0].clone()) if comb_a_parts else None
    combined_latent = h3_pack_av(sampled_latents[0], combined_v, combined_a)
    last_latent = sampled_latents[-1]
    return sampled_latents, trim_leading_list, trim_trailing_list, combined_latent, last_latent


def _generate_noise_like(target: Any, generator: Optional[torch.Generator] = None) -> Any:
    """Generate Gaussian noise matching target tensor or nested tensor shape and device."""

    def _draw(t: torch.Tensor) -> torch.Tensor:
        if generator is not None:
            gen_device = getattr(generator, "device", None)
            gen_type = getattr(gen_device, "type", str(gen_device)) if gen_device is not None else "cpu"
            if gen_type == "cpu" and t.device.type != "cpu":
                return torch.randn(t.shape, dtype=t.dtype, device="cpu", generator=generator).to(t.device)
            return torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=generator)
        return torch.randn(t.shape, dtype=t.dtype, device=t.device)

    if isinstance(target, NestedTensor):
        return NestedTensor([_draw(t) for t in target.unbind()])
    return _draw(target)


def _blend_renoise(denoised: Any, noise: Any, sigma: float, s_noise: float = 1.0) -> Any:
    """Blend clean endpoint prediction with fresh noise: (1 - sigma) * x0 + sigma * (e * s_noise)."""
    weight_denoised = 1.0 - float(sigma)
    weight_noise = float(sigma) * float(s_noise)
    if isinstance(denoised, NestedTensor):
        d_parts = denoised.unbind()
        n_parts = noise.unbind() if isinstance(noise, NestedTensor) else [noise]
        blended = [d * weight_denoised + n * weight_noise for d, n in zip(d_parts, n_parts)]
        return NestedTensor(blended)
    return denoised * weight_denoised + noise * weight_noise


@torch.no_grad()
def sample_dmad_renoise(model: Any, x: Any, sigmas: torch.Tensor, extra_args: Optional[dict[str, Any]] = None, callback: Optional[Any] = None, disable: Optional[bool] = None, s_noise: float = 1.0, noise_sampler: Optional[Any] = None) -> Any:
    """Few-step re-noise sampler for DMAD/DMD2 distilled models (e.g. 4-step MiniMax H3).

    At each intermediate step:
      1. Evaluates model to predict clean endpoint x0 (denoised).
      2. If next sigma is 0, outputs x0 directly.
      3. Otherwise, discards prior noise and re-noises x0 with fresh Gaussian noise:
         x_{next} = (1 - sigma_{next}) * x0 + sigma_{next} * (noise * s_noise)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    total_steps = len(sigmas) - 1
    if total_steps <= 0:
        return x

    from comfy.utils import model_trange as trange

    s_in = x.new_ones([x.shape[0]]) if hasattr(x, "new_ones") else torch.ones((1,), device=getattr(x, "device", "cpu"))

    for i in trange(total_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        if float(sigma_next) == 0.0:
            x = denoised
        else:
            if noise_sampler is not None:
                try:
                    e = noise_sampler(sigma, sigma_next)
                except Exception:
                    e = _generate_noise_like(denoised, generator=generator)
            else:
                e = _generate_noise_like(denoised, generator=generator)

            x = _blend_renoise(denoised, e, float(sigma_next), s_noise=s_noise)

    return x


H3_TILED_VAE_DOWNSAMPLE = 16
H3_TILED_PATCH_ALIGN = 2
H3_TILED_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
H3_TILED_FRAME_RESCALE = 5.0 / 3.0
H3_TILED_MAX_IDENTITY_ANCHORS = 6
H3_TILED_POLISH_HALO = 16
H3_TILED_COLOR_CLAMP = 0.05
H3_TILED_SEAM_CORR_GATE = 0.85
H3_TILED_SEAM_DC_GATE = 0.60
H3_MOTION_ANCHOR_OPTIONS = ("0", "5", "22", "39", "56")


def _h3_tiled_frames_for_tokens(token_count: int) -> int:
    return sum(H3_TILED_FRAME_PER_TOKEN[i % 5] for i in range(int(token_count)))


def _h3_tiled_tokens_for_frames(frame_count: int) -> int:
    n, acc = 0, 0
    while acc < frame_count:
        acc += H3_TILED_FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def _h3_tiled_steps_for_frames(frame_count: int) -> int | None:
    k, covered = 0, 0
    while covered < frame_count:
        covered += H3_TILED_FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == frame_count else None


def _h3_tiled_token_start_at_or_before(frame_idx: int) -> int:
    k = 0
    while _h3_tiled_frames_for_tokens(k + 1) <= frame_idx:
        k += 1
    return k


def snap_h3_tiled_chunk_frames(frames: int) -> int:
    v = int(frames)
    if v < 5:
        return 5
    return 5 + 17 * max(0, round((v - 5) / 17))


def snap_h3_tiled_overlap_frames(frames: int) -> int:
    v = int(frames)
    if v <= 0:
        return 0
    return 5 + 17 * max(0, round((v - 5) / 17))


def build_h3_temporal_split_config(chunk_frames: int = 73, overlap_frames: int = 22, anchor_strength: float = 0.999, motion_anchor_frames: str | int = "22", identity_anchor_frames: int = 24) -> dict[str, Any]:
    chunk = snap_h3_tiled_chunk_frames(chunk_frames)
    overlap = snap_h3_tiled_overlap_frames(overlap_frames)
    if overlap >= chunk:
        overlap = snap_h3_tiled_overlap_frames(chunk - 17) if chunk > 5 else 0
    return {"chunk_frames": chunk, "overlap_frames": overlap, "anchor_strength": max(0.0, min(1.0, float(anchor_strength))), "motion_anchor_frames": max(0, int(motion_anchor_frames)), "identity_anchor_frames": max(0, int(identity_anchor_frames))}


def _px_to_h3_lat(px: int) -> int:
    return max(H3_TILED_PATCH_ALIGN, (round(int(px) / H3_TILED_VAE_DOWNSAMPLE) // H3_TILED_PATCH_ALIGN) * H3_TILED_PATCH_ALIGN)


def _snap_h3_lat_align(val: float) -> int:
    return max(0, int(round(float(val) / H3_TILED_PATCH_ALIGN)) * H3_TILED_PATCH_ALIGN)


def build_h3_spatial_split_config(tile_width: int = 512, tile_height: int = 512, overlap_ratio: float = 0.25, fade_ratio: float = 0.50, min_tile_size: int = 256, seam_denoise: float = 1.0) -> tuple[dict[str, Any], str]:
    tw = _px_to_h3_lat(tile_width)
    th = _px_to_h3_lat(tile_height)
    ol_w = min(tw - H3_TILED_PATCH_ALIGN, _snap_h3_lat_align(tw * float(overlap_ratio)))
    ol_h = min(th - H3_TILED_PATCH_ALIGN, _snap_h3_lat_align(th * float(overlap_ratio)))
    fw = min(ol_w, int(round(ol_w * float(fade_ratio))))
    fh = min(ol_h, int(round(ol_h * float(fade_ratio))))
    mt = min(_px_to_h3_lat(min_tile_size), th, tw) if int(min_tile_size) > 0 else 0
    cap = max(0.1, min(1.0, float(seam_denoise)))
    config = {"tw": tw, "th": th, "ol_w": ol_w, "ol_h": ol_h, "fw": fw, "fh": fh, "mt": mt, "cap": cap}
    summary = f"Tile: {tw * H3_TILED_VAE_DOWNSAMPLE}x{th * H3_TILED_VAE_DOWNSAMPLE}px ({tw}x{th} lat) | Overlap: {ol_w}x{ol_h} lat | Fade: {fw}x{fh} lat | Seam denoise cap: {cap:.2f}"
    return config, summary


def compute_h3_tiled_segments(total_tokens: int, chunk_frames: int, overlap_frames: int) -> list[tuple[int, int, int, int]]:
    """Plan phase-aligned [k0, f0, k1, f1] segments with k0 % 5 == 0 and (k1 - k0) = 5m + 2."""
    tv = int(total_tokens)
    if tv <= 2:
        return [(0, 0, tv, _h3_tiled_frames_for_tokens(tv))]
    if (tv - 2) % 5 != 0:
        raise ValueError(f"MiniMax H3 video latent temporal length must satisfy 5k + 2 (or <= 2 for still/short clips), got {tv}.")

    chunk_f = snap_h3_tiled_chunk_frames(chunk_frames)
    tc = ((chunk_f - 5) // 17) * 5 + 2
    if tc >= tv:
        return [(0, 0, tv, _h3_tiled_frames_for_tokens(tv))]

    overlap_f = snap_h3_tiled_overlap_frames(overlap_frames)
    to = ((overlap_f - 5) // 17) * 5 + 2 if overlap_f > 0 else 0
    if to >= tc:
        to = max(0, tc - 5)
    hop = max(5, ((tc - to) // 5) * 5)

    bounds: list[tuple[int, int, int, int]] = []
    k0 = 0
    while True:
        if k0 + tc >= tv:
            last_k0 = tv - tc
            bounds.append((last_k0, _h3_tiled_frames_for_tokens(last_k0), tv, _h3_tiled_frames_for_tokens(tv)))
            break
        k1 = k0 + tc
        bounds.append((k0, _h3_tiled_frames_for_tokens(k0), k1, _h3_tiled_frames_for_tokens(k1)))
        k0 += hop
    return bounds


def trim_h3_tiled_keyframe(kf: Mapping[str, Any], f0: int, f1: int) -> dict[str, Any] | None:
    idx = int(kf["resolved_frame_index"])
    latent = kf.get("latent")
    audio_latent = kf.get("audio_latent")
    if latent is None and audio_latent is None:
        return None if (idx < f0 or idx >= f1) else {"resolved_frame_index": idx - f0}

    out: dict[str, Any] = {}
    if latent is not None:
        t_start = None
        t_end = None
        pos = idx
        total_k = int(latent.shape[2])
        for k in range(total_k):
            span = H3_TILED_FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None and (total_k == 1 or k % 5 == 0):
                    t_start = k
                if t_start is not None:
                    t_end = k + 1
            pos += span
        if t_start is None or t_end is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + _h3_tiled_frames_for_tokens(t_start) - f0

    if audio_latent is not None:
        rt = int(audio_latent.shape[-1])
        a_start = max(0, math.ceil((f0 - idx) * H3_TILED_FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) * H3_TILED_FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)

    return out if ("latent" in out or "audio_latent" in out) else None


def _reanchor_h3_conditioning(cond: Any, f0: int, f1: int, spatial_hw: tuple[int, int] | None) -> Any:
    if cond is None:
        return None
    out = []
    for tensor, meta in cond:
        nd = dict(meta)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [tkf for kf in kfs if (tkf := trim_h3_tiled_keyframe(kf, f0, f1)) is not None]
            if trimmed:
                if spatial_hw is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial_hw[0] or lt.shape[4] != spatial_hw[1]):
                            b, c, t, h, w = lt.shape
                            kf["latent"] = F.interpolate(lt.reshape(b * t, c, h, w), size=spatial_hw, mode="bilinear", align_corners=False).reshape(b, c, t, spatial_hw[0], spatial_hw[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def _prepend_h3_keyframes(cond: Any, kfs: list[dict[str, Any]]) -> Any:
    if not kfs or cond is None:
        return cond
    out = []
    for tensor, meta in cond:
        nd = dict(meta)
        nd["minimax_keyframes"] = list(kfs) + list(nd.get("minimax_keyframes") or [])
        out.append([tensor, nd])
    return out


def _motion_h3_keyframes(prev_video: torch.Tensor, prev_tokens: int, f0: int, n_frames: int) -> list[dict[str, Any]]:
    chosen = next((g for g in (56, 39, 22, 5) if g <= n_frames), 0)
    steps = _h3_tiled_steps_for_frames(chosen) if chosen else None
    if not steps or steps > prev_tokens or (prev_tokens - steps) % 5 != 0:
        return []
    start = prev_tokens - steps
    return [{"resolved_frame_index": _h3_tiled_frames_for_tokens(start + k) - f0, "latent": prev_video[:, :, start + k : start + k + 1].contiguous()} for k in range(steps)]


def _identity_h3_keyframes(source: torch.Tensor, f0: int, f1: int, spacing: int) -> list[dict[str, Any]]:
    kfs: list[dict[str, Any]] = []
    pos = f0 + spacing
    while pos < f1:
        k = _h3_tiled_token_start_at_or_before(pos)
        if k < source.shape[2]:
            kfs.append({"resolved_frame_index": _h3_tiled_frames_for_tokens(k) - f0, "latent": source[:, :, k : k + 1].contiguous()})
        pos += spacing
    if len(kfs) > H3_TILED_MAX_IDENTITY_ANCHORS:
        kfs = [kfs[i * len(kfs) // H3_TILED_MAX_IDENTITY_ANCHORS] for i in range(H3_TILED_MAX_IDENTITY_ANCHORS)]
    return kfs


def _anchor_h3_conditioning(cond: Any, prev_video: torch.Tensor, f0: int, strength: float) -> Any:
    t_idx = _h3_tiled_tokens_for_frames(f0)
    if t_idx >= prev_video.shape[2]:
        return cond
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t_idx : t_idx + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, meta in cond:
        nd = dict(meta)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def _crop_h3_keyframes_to_tile(cond: Any, src_h: int, src_w: int, r0: int, c0: int, tr: int, tc: int) -> Any:
    if cond is None:
        return None
    out = []
    for tensor, meta in cond:
        nd = dict(meta)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    if lt.shape[3] == src_h and lt.shape[4] == src_w:
                        nkf["latent"] = lt[:, :, :, r0 : r0 + tr, c0 : c0 + tc].contiguous()
                    else:
                        b, c, t, h, w = lt.shape
                        lt_resized = F.interpolate(lt.to(torch.float32).reshape(b * t, c, h, w), size=(src_h, src_w), mode="bilinear", align_corners=False).to(lt.dtype).reshape(b, c, t, src_h, src_w)
                        nkf["latent"] = lt_resized[:, :, :, r0 : r0 + tr, c0 : c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


def _spatial_grid_1d(size: int, tile: int, overlap: int, min_tile: int) -> tuple[list[int], list[int], list[int]]:
    if size <= tile:
        return [0], [size], [0]
    stride = max(H3_TILED_PATCH_ALIGN, tile - overlap)
    count = math.ceil((size - overlap) / stride)
    if (count - 1) * stride + tile < size:
        count += 1
    starts = [i * stride for i in range(count)]
    lengths = [min(tile, size - s) for s in starts]
    if min_tile > 0 and count >= 2:
        edge = size - starts[-1]
        if edge < min_tile:
            new_last = _snap_h3_lat_align(size - min_tile)
            if starts[-2] < new_last < starts[-2] + lengths[-2]:
                starts[-1] = new_last
                lengths[-1] = size - new_last
    overlaps = [0] * count
    for i in range(1, count):
        overlaps[i] = max(0, starts[i - 1] + lengths[i - 1] - starts[i])
    return starts, lengths, overlaps


def compute_h3_spatial_grid(height: int, width: int, th: int, tw: int, ol_h: int, ol_w: int, min_tile: int = 0) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    rows, trows, row_ovl = _spatial_grid_1d(height, th, ol_h, min_tile)
    cols, tcols, col_ovl = _spatial_grid_1d(width, tw, ol_w, min_tile)
    return rows, cols, trows, tcols, row_ovl, col_ovl


def build_h3_spatial_fade_mask(tile_h: int, tile_w: int, ovh: int, ovw: int, done_top: bool, done_left: bool, fade_h: int = 0, fade_w: int = 0, seam_cap: float = 1.0, device: torch.device | str = "cpu") -> torch.Tensor:
    mask = torch.ones((tile_h, tile_w), dtype=torch.float32, device=device)

    def _profile(length: int, ov: int, fade: int) -> torch.Tensor:
        prof = torch.ones(length, dtype=torch.float32, device=device)
        f = min(fade, ov)
        frozen = ov - f
        if frozen > 0:
            prof[:frozen] = 0.0
        if f > 0:
            prof[frozen:ov] = torch.linspace(0.0, float(seam_cap), f, device=device, dtype=torch.float32)
        ramp = min(ov, length - ov)
        if ramp > 0 and seam_cap < 1.0:
            start = float(seam_cap) if f > 0 else 0.0
            prof[ov : ov + ramp] = torch.linspace(start, 1.0, ramp, device=device, dtype=torch.float32)
        return prof

    if done_left and ovw > 0:
        mask = torch.minimum(mask, _profile(tile_w, ovw, fade_w).unsqueeze(0))
    if done_top and ovh > 0:
        mask = torch.minimum(mask, _profile(tile_h, ovh, fade_h).unsqueeze(1))
    return mask


def _dc_median_correct(new: torch.Tensor, refs: Sequence[tuple[torch.Tensor, torch.Tensor] | None], clamp: float = H3_TILED_COLOR_CLAMP, min_samples: int = 256) -> tuple[torch.Tensor, torch.Tensor | None]:
    pairs = [pair for pair in refs if pair is not None and pair[0] is not None and pair[0].numel() >= min_samples]
    if not pairs:
        return new, None
    pa = torch.cat([a.float().permute(0, 2, 3, 4, 1).reshape(-1, a.shape[1]) for a, _ in pairs], dim=0)
    pb = torch.cat([b.float().permute(0, 2, 3, 4, 1).reshape(-1, b.shape[1]) for _, b in pairs], dim=0)
    dc = (pa - pb).median(dim=0).values.clamp(-clamp, clamp)
    return new - dc.view(1, -1, 1, 1, 1).to(device=new.device, dtype=new.dtype), dc


def _grade_pin_chunk(chunk: torch.Tensor, ref: torch.Tensor, clamp: float = H3_TILED_COLOR_CLAMP) -> torch.Tensor:
    a = chunk.float().permute(0, 2, 3, 4, 1).reshape(-1, chunk.shape[1])
    b = ref.float().permute(0, 2, 3, 4, 1).reshape(-1, ref.shape[1])
    dc = (a - b).median(dim=0).values.clamp(-clamp, clamp)
    return chunk - dc.view(1, -1, 1, 1, 1).to(device=chunk.device, dtype=chunk.dtype)


def _should_polish_seam(mode: str, sub: torch.Tensor, region: torch.Tensor) -> bool:
    if mode == "all":
        return True
    if sub.numel() < 4096:
        return False
    sp = sub.float().permute(0, 2, 3, 4, 1).reshape(-1, sub.shape[1])
    rp = region.float().permute(0, 2, 3, 4, 1).reshape(-1, region.shape[1])
    dc = (sp - rp).median(dim=0).values.clamp(-H3_TILED_COLOR_CLAMP, H3_TILED_COLOR_CLAMP)
    a = sp - sp.mean(dim=0)
    b = rp - rp.mean(dim=0)
    corr = float(((a * b).mean(dim=0) / (a.std(dim=0) * b.std(dim=0) + 1e-6)).median().item())
    return corr < H3_TILED_SEAM_CORR_GATE or float(dc.abs().max().item()) > H3_TILED_SEAM_DC_GATE * H3_TILED_COLOR_CLAMP


def _linear_crossfade(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    length = a.shape[dim]
    weights = torch.linspace(0.0, 1.0, length, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = length
    return a + (b - a) * weights.view(shape)


def _temporal_append_h3_av(acc_v: torch.Tensor | None, acc_a: torch.Tensor | None, chunk_v: torch.Tensor, chunk_a: torch.Tensor, index: int, k0: int, f0: int, color_match: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if acc_v is None or acc_a is None:
        return chunk_v, chunk_a
    gi = k0
    agi = round(f0 * H3_TILED_FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    rv = torch.empty((acc_v.shape[0], acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]), device=acc_v.device, dtype=acc_v.dtype)
    ra = torch.empty((acc_a.shape[0], acc_a.shape[1], acc_a.shape[2], total_a), device=acc_a.device, dtype=acc_a.dtype)
    rv[:, :, : acc_v.shape[2]] = acc_v
    ra[:, :, :, : acc_a.shape[-1]] = acc_a
    v, a = chunk_v, chunk_a
    if index > 0:
        ov = min(acc_v.shape[2] - gi, v.shape[2])
        if ov > 0:
            if color_match:
                v, _ = _dc_median_correct(v, [(v[:, :, :ov], rv[:, :, gi : gi + ov])])
            rv[:, :, gi : gi + ov] = _linear_crossfade(rv[:, :, gi : gi + ov], v[:, :, :ov], dim=2)
            v = v[:, :, ov:]
            gi += ov
        ova = min(acc_a.shape[-1] - agi, a.shape[-1])
        if ova > 0:
            ra[:, :, :, agi : agi + ova] = _linear_crossfade(ra[:, :, :, agi : agi + ova], a[:, :, :, :ova], dim=3)
            a = a[:, :, :, ova:]
            agi += ova
    if v.shape[2] > 0:
        rv[:, :, gi : gi + v.shape[2]] = v
    if a.shape[-1] > 0:
        ra[:, :, :, agi : agi + a.shape[-1]] = a
    return rv, ra


def _build_tiled_guider(base_guider: Any, model: Any, pos_cond: Any, neg_cond: Any, cfg: float, frame0: int) -> Any:
    if base_guider is not None:
        tile_guider = prepare_chunk_guider(base_guider, pos_cond, frame0=frame0)
        if neg_cond is not None and hasattr(tile_guider, "set_conds"):
            tile_guider.set_conds(pos_cond, neg_cond)
            tile_guider.raw_conds = (pos_cond, neg_cond)
        return tile_guider

    import comfy.samplers

    tile_guider = comfy.samplers.CFGGuider(model)
    opts = dict(getattr(tile_guider, "model_options", {}) or {})
    t_opts = dict(opts.get("transformer_options", {}))
    t_opts["h3_ctrl_frame0"] = int(frame0)
    t_opts["mmh3_control_frame0"] = int(frame0)
    opts["transformer_options"] = t_opts
    tile_guider.model_options = opts
    if neg_cond is not None:
        tile_guider.set_conds(pos_cond, neg_cond)
        tile_guider.set_cfg(float(cfg))
    else:
        tile_guider.inner_set_conds({"positive": pos_cond})
    return tile_guider


def _sample_h3_tile_piece(piece: dict[str, Any], tile_guider: Any, tile_noise: NestedTensor, seed: int, sampler: Any, sigmas: torch.Tensor, callback: Any) -> Any:
    import comfy.sample

    reset_model_caches(tile_guider)
    latent_image = comfy.sample.fix_empty_latent_channels(model=tile_guider.model_patcher.model, latent_image=piece["samples"])
    samples = tile_guider.sample(tile_noise, latent_image, sampler, sigmas, denoise_mask=piece.get("noise_mask"), callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())


def sample_minimax_h3_tiled_upscale(
    *,
    latent: dict[str, Any],
    noise: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    model: Any = None,
    conditioning: Any = None,
    negative: Any = None,
    guider: Any = None,
    cfg: float = 1.0,
    temporal_split_config: Mapping[str, Any] | None = None,
    spatial_split_config: Mapping[str, Any] | None = None,
    seam_polish: str = "off",
    color_match: bool = True,
) -> dict[str, Any]:
    if guider is None and model is None:
        raise ValueError("Connect either model + conditioning or an external guider to MiniMax H3 Tiled Upscale Sampler.")

    base_pos = conditioning
    base_neg = negative
    if guider is not None:
        if base_pos is None:
            base_pos = _extract_positive_cond_from_guider(guider)
        if base_neg is None:
            raw_conds = getattr(guider, "raw_conds", None)
            if raw_conds is not None and len(raw_conds) > 1:
                base_neg = raw_conds[1]
            else:
                orig = getattr(guider, "original_conds", {}) or {}
                base_neg = orig.get("negative")
    if base_pos is None:
        raise ValueError("Positive conditioning is required when no guider with positive conditioning is connected.")

    video, audio = h3_unpack_av(latent, label="latent")
    if audio is None:
        raise ValueError("MiniMax H3 Tiled Upscale Sampler expects a joint MiniMax H3 audio-video latent (NestedTensor with video and audio streams).")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax H3 Tiled Upscale Sampler supports batch size 1 only.")

    _, _, tv, height, width = video.shape
    if temporal_split_config is not None:
        bounds = compute_h3_tiled_segments(tv, int(temporal_split_config["chunk_frames"]), int(temporal_split_config["overlap_frames"]))
        anchor_strength = float(temporal_split_config.get("anchor_strength", 0.999))
        motion_n = int(temporal_split_config.get("motion_anchor_frames", 22))
        identity_n = int(temporal_split_config.get("identity_anchor_frames", 24))
    else:
        bounds = [(0, 0, tv, _h3_tiled_frames_for_tokens(tv))]
        anchor_strength, motion_n, identity_n = 0.999, 0, 0

    if spatial_split_config is not None:
        rows, cols, trows, tcols, row_ovl, col_ovl = compute_h3_spatial_grid(height, width, int(spatial_split_config["th"]), int(spatial_split_config["tw"]), int(spatial_split_config["ol_h"]), int(spatial_split_config["ol_w"]), int(spatial_split_config.get("mt", 0)))
        fade_h = int(spatial_split_config.get("fh", 0))
        fade_w = int(spatial_split_config.get("fw", 0))
        seam_cap = float(spatial_split_config.get("cap", 1.0))
    else:
        rows, cols, trows, tcols, row_ovl, col_ovl = [0], [0], [height], [width], [0], [0]
        fade_h = fade_w = 0
        seam_cap = 1.0

    nrows, ncols = len(rows), len(cols)
    steps = max(int(sigmas.shape[-1]) - 1, 1)
    n_tiles = len(bounds) * nrows * ncols

    import latent_preview

    patcher_ref = guider.model_patcher if guider is not None else model
    previewer = latent_preview.get_previewer(patcher_ref.load_device, patcher_ref.model.latent_format)
    total_progress = steps * n_tiles
    pbar = comfy.utils.ProgressBar(total_progress)

    def _make_tile_cb(idx: int):

        def _cb(step, x0, x, total_steps):
            preview = None
            if previewer is not None and x0 is not None:
                px0 = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                try:
                    preview = previewer.decode_latent_to_preview_image("JPEG", px0)
                except Exception:
                    preview = None
            pbar.update_absolute(idx * steps + step + 1, total_progress, preview)

        return _cb

    acc_v: torch.Tensor | None = None
    acc_a: torch.Tensor | None = None
    tile_idx = 0
    base_seed = int(getattr(noise, "seed", 0))

    for i, (k0, f0, k1, f1) in enumerate(bounds):
        chunk_v = video[:, :, k0:k1].contiguous()
        a0 = round(f0 * H3_TILED_FRAME_RESCALE)
        a1 = min(round(f1 * H3_TILED_FRAME_RESCALE), int(audio.shape[-1]))
        if a1 <= a0:
            a0 = max(0, int(audio.shape[-1]) - 1)
            a1 = int(audio.shape[-1])
        chunk_a = audio[:, :, :, a0:a1].contiguous()

        cond_pos_i = _reanchor_h3_conditioning(base_pos, f0, f1, (height, width))
        cond_neg_i = _reanchor_h3_conditioning(base_neg, f0, f1, (height, width))
        if i > 0 and acc_v is not None:
            if motion_n > 0:
                cond_pos_i = _prepend_h3_keyframes(cond_pos_i, _motion_h3_keyframes(acc_v, k0, f0, motion_n))
            if identity_n > 0:
                cond_pos_i = _prepend_h3_keyframes(cond_pos_i, _identity_h3_keyframes(video, f0, f1, identity_n))
            if anchor_strength > 0.0:
                cond_pos_i = _anchor_h3_conditioning(cond_pos_i, acc_v, f0, anchor_strength)
        elif identity_n > 0:
            cond_pos_i = _prepend_h3_keyframes(cond_pos_i, _identity_h3_keyframes(video, f0, f1, identity_n))

        c_noise = chunk_noise(noise, i)
        noise_vc = c_noise.generate_noise({"samples": torch.empty_like(chunk_v, dtype=torch.float32)})
        noise_ac = c_noise.generate_noise({"samples": torch.empty_like(chunk_a, dtype=torch.float32)})
        chunk_seed = int(getattr(c_noise, "seed", base_seed + i))

        chunk_out = chunk_v.clone()
        chunk_polish: list[tuple[str, int, int]] = []
        frozen_audio_mask = torch.zeros((1, chunk_a.shape[1], chunk_a.shape[2], chunk_a.shape[-1]), device=chunk_a.device, dtype=torch.float32)

        for ri in range(nrows):
            for cj in range(ncols):
                comfy.model_management.throw_exception_if_processing_interrupted()
                r0, c0 = rows[ri], cols[cj]
                tr, tc = trows[ri], tcols[cj]
                ovh, ovw = row_ovl[ri], col_ovl[cj]

                tile = chunk_out[:, :, :, r0 : r0 + tr, c0 : c0 + tc].clone()
                mv = build_h3_spatial_fade_mask(tr, tc, ovh, ovw, done_top=(ri > 0), done_left=(cj > 0), fade_h=fade_h, fade_w=fade_w, seam_cap=seam_cap, device=chunk_out.device)[None, None, None]
                piece = {"samples": NestedTensor((tile, chunk_a)), "noise_mask": NestedTensor((mv, frozen_audio_mask))}
                tile_noise = NestedTensor((noise_vc[:, :, :, r0 : r0 + tr, c0 : c0 + tc].contiguous(), noise_ac.contiguous()))

                cond_tile = _crop_h3_keyframes_to_tile(cond_pos_i, height, width, r0, c0, tr, tc)
                neg_tile = _crop_h3_keyframes_to_tile(cond_neg_i, height, width, r0, c0, tr, tc)
                tile_guider = _build_tiled_guider(guider, model, cond_tile, neg_tile, cfg, f0)

                out_sample = _sample_h3_tile_piece(piece, tile_guider, tile_noise, chunk_seed, sampler, sigmas, _make_tile_cb(tile_idx))
                tile_v = (out_sample.tensors[0] if getattr(out_sample, "is_nested", False) else out_sample).to(device=chunk_out.device, dtype=chunk_out.dtype)
                region = chunk_out[:, :, :, r0 : r0 + tr, c0 : c0 + tc].clone()

                if color_match:
                    tile_v, _ = _dc_median_correct(
                        tile_v, [(tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw]) if (cj > 0 and ovw > 0) else None, (tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :]) if (ri > 0 and ovh > 0) else None, (tile_v, chunk_v[:, :, :, r0 : r0 + tr, c0 : c0 + tc]) if (ri == 0 and cj == 0) else None]
                    )

                if seam_polish != "off":
                    if cj > 0 and ovw > 0 and _should_polish_seam(seam_polish, tile_v[:, :, :, :, :ovw], region[:, :, :, :, :ovw]):
                        chunk_polish.append(("W", c0, ovw))
                    if ri > 0 and ovh > 0 and _should_polish_seam(seam_polish, tile_v[:, :, :, :ovh, :], region[:, :, :, :ovh, :]):
                        chunk_polish.append(("H", r0, ovh))

                if cj > 0 and ovw > 0:
                    wts = torch.linspace(0.0, 1.0, ovw, device=region.device, dtype=region.dtype).view(1, 1, 1, 1, ovw)
                    region[:, :, :, :, :ovw] = region[:, :, :, :, :ovw] * (1.0 - wts) + tile_v[:, :, :, :, :ovw] * wts
                if ri > 0 and ovh > 0:
                    wts = torch.linspace(0.0, 1.0, ovh, device=region.device, dtype=region.dtype).view(1, 1, 1, ovh, 1)
                    region[:, :, :, :ovh, :] = region[:, :, :, :ovh, :] * (1.0 - wts) + tile_v[:, :, :, :ovh, :] * wts

                band = torch.zeros((1, 1, 1, tr, tc), dtype=torch.bool, device=region.device)
                if cj > 0 and ovw > 0:
                    band[:, :, :, :, :ovw] = True
                if ri > 0 and ovh > 0:
                    band[:, :, :, :ovh, :] = True
                chunk_out[:, :, :, r0 : r0 + tr, c0 : c0 + tc] = torch.where(band, region, tile_v)
                tile_idx += 1

        if color_match:
            chunk_out = _grade_pin_chunk(chunk_out, chunk_v)

        if chunk_polish:
            dedup_polish = list(dict.fromkeys(chunk_polish))
            pbar_polish = comfy.utils.ProgressBar(steps * len(dedup_polish))
            t_len = k1 - k0
            for pi, (axis, s0, band_w) in enumerate(dedup_polish):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if axis == "W":
                    w0 = max(0, s0 - H3_TILED_POLISH_HALO)
                    w1 = min(width, s0 + band_w + H3_TILED_POLISH_HALO)
                    win = chunk_out[:, :, :, :, w0:w1].clone()
                    b0, b1 = s0 - w0, s0 - w0 + band_w
                    mv = torch.zeros((1, 1, t_len, height, w1 - w0), dtype=torch.float32, device=win.device)
                    mv[:, :, :, :, b0:b1] = 1.0
                    r0c, c0c, trc, tcc, ax = 0, w0, height, w1 - w0, 4
                    tn_v = noise_vc[:, :, :, :, w0:w1].contiguous()
                else:
                    h0 = max(0, s0 - H3_TILED_POLISH_HALO)
                    h1 = min(height, s0 + band_w + H3_TILED_POLISH_HALO)
                    win = chunk_out[:, :, :, h0:h1, :].clone()
                    b0, b1 = s0 - h0, s0 - h0 + band_w
                    mv = torch.zeros((1, 1, t_len, h1 - h0, width), dtype=torch.float32, device=win.device)
                    mv[:, :, :, b0:b1, :] = 1.0
                    r0c, c0c, trc, tcc, ax = h0, 0, h1 - h0, width, 3
                    tn_v = noise_vc[:, :, :, h0:h1, :].contiguous()

                piece = {"samples": NestedTensor((win, chunk_a)), "noise_mask": NestedTensor((mv, frozen_audio_mask))}
                tn = NestedTensor((tn_v, noise_ac.contiguous()))
                cond_p = _crop_h3_keyframes_to_tile(cond_pos_i, height, width, r0c, c0c, trc, tcc)
                neg_p = _crop_h3_keyframes_to_tile(cond_neg_i, height, width, r0c, c0c, trc, tcc)
                polish_guider = _build_tiled_guider(guider, model, cond_p, neg_p, cfg, f0)

                def _polish_cb(step, x0, x, ts, _pi=pi, _tot=steps * len(dedup_polish)):
                    pbar_polish.update_absolute(_pi * steps + step + 1, _tot)

                out_p = _sample_h3_tile_piece(piece, polish_guider, tn, chunk_seed, sampler, sigmas, _polish_cb)
                pv = (out_p.tensors[0] if getattr(out_p, "is_nested", False) else out_p).to(device=chunk_out.device, dtype=chunk_out.dtype)

                nlen = (w1 - w0) if ax == 4 else (h1 - h0)
                alpha = torch.ones(nlen, dtype=chunk_out.dtype, device=chunk_out.device)
                if b0 > 0:
                    alpha[:b0] = torch.linspace(0.0, 1.0, b0, device=chunk_out.device, dtype=chunk_out.dtype)
                if nlen - b1 > 0:
                    alpha[b1:] = torch.linspace(1.0, 0.0, nlen - b1, device=chunk_out.device, dtype=chunk_out.dtype)
                view = [1, 1, 1, 1, 1]
                view[ax] = nlen
                avv = alpha.view(view)
                if ax == 4:
                    chunk_out[:, :, :, :, w0:w1] = avv * pv + (1.0 - avv) * win
                else:
                    chunk_out[:, :, :, h0:h1, :] = avv * pv + (1.0 - avv) * win

        acc_v, acc_a = _temporal_append_h3_av(acc_v, acc_a, chunk_out, chunk_a, i, k0, f0, color_match)

    out = dict(latent)
    out["samples"] = NestedTensor((acc_v, acc_a))
    out.pop("noise_mask", None)
    return out
