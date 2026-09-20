"""Sampling helpers for windowed and looping generation workflows."""

import copy
import logging
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas

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

H3_OVERLAP_SECONDS_OPTIONS = [
    "0.00s (no overlap)",
    "0.21s (5 frames)",
    "0.92s (22 frames)",
    "1.62s (39 frames)",
    "2.33s (56 frames)",
    "3.04s (73 frames)",
    "3.75s (90 frames)",
]


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


def h3_pack_av(
    orig_dict: dict[str, Any],
    video_tensor: torch.Tensor,
    audio_tensor: Optional[torch.Tensor] = None,
    noise_mask: Optional[Any] = None,
) -> dict[str, Any]:
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


def plan_h3_windows(total_frames: int, window_frames: int, overlap_frames: int) -> list[tuple[int, int]]:
    """Plan H3 window intervals on latent grid (start_idx, end_idx) inclusive-exclusive.
    
    Ensures step stride maintains 5-group phase alignment.
    """
    total_f = h3_snap_frame_count(int(total_frames))
    total_latents = h3_frames_to_latents(total_f)
    if total_latents <= H3_LATENT_BASE:
        return [(0, total_latents)]

    w_frames = max(H3_FRAME_BASE, int(window_frames)) if window_frames > 0 else total_f
    w_latents = min(total_latents, h3_snap_latent_t(h3_frames_to_latents(w_frames)))
    if w_latents <= H3_LATENT_BASE or w_latents >= total_latents:
        return [(0, total_latents)]

    ov_latents = 0
    if overlap_frames > 0:
        raw_ov = h3_frames_to_latents(int(overlap_frames))
        ov_latents = h3_snap_latent_t(raw_ov) if raw_ov >= H3_LATENT_BASE else 0
        ov_latents = min(ov_latents, max(0, w_latents - H3_LATENT_GROUP))

    step = max(H3_LATENT_GROUP, w_latents - ov_latents)
    # Align step to multiple of H3_LATENT_GROUP (5) to maintain phase
    step = (step // H3_LATENT_GROUP) * H3_LATENT_GROUP
    if step < H3_LATENT_GROUP:
        step = H3_LATENT_GROUP

    windows = []
    start = 0
    while start < total_latents:
        end = min(total_latents, start + w_latents)
        windows.append((start, end))
        if end >= total_latents:
            break
        start += step
        if start + w_latents > total_latents:
            final_start = max(0, total_latents - w_latents)
            if final_start > windows[-1][0]:
                windows.append((final_start, total_latents))
            break

    return windows


def prepare_chunk_guider(guider: Any, positive_cond: Any, frame0: Optional[int] = None) -> Any:
    """Create isolated clone of guider with swapped positive cond and h3_ctrl_frame0."""
    chunk_g = copy.copy(guider)
    chunk_g.original_conds = dict(guider.original_conds)

    if frame0 is not None:
        opts = dict(getattr(guider, "model_options", {}) or {})
        t_opts = dict(opts.get("transformer_options", {}))
        t_opts["h3_ctrl_frame0"] = int(frame0)
        opts["transformer_options"] = t_opts
        chunk_g.model_options = opts

    raw_conds = getattr(guider, "raw_conds", None)
    if raw_conds is None:
        c_dict = getattr(guider, "original_conds", {}) or {}
        raw_conds = (c_dict.get("positive"), c_dict.get("negative"))

    _, negative = raw_conds
    if negative is None:
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


def build_carry_noise_mask(
    sub_v: torch.Tensor,
    sub_a: Optional[torch.Tensor],
    carried_v: int,
    carried_a: int,
    strength_v: float,
    strength_a: float,
    master_mask_v: Optional[torch.Tensor],
    master_mask_a: Optional[torch.Tensor],
    v0: int,
    v1: int,
    a0: int,
    a1: int,
) -> Any:
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
        vm[:, :, :carried_v] = 1.0 - float(strength_v)

    if am is not None and carried_a > 0:
        am[:, :, :, :carried_a] = 1.0 - float(strength_a)

    return NestedTensor([vm, am]) if am is not None else vm


def run_chunk_sampling(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_dict: dict[str, Any],
    start_step: int = 0,
    end_step: int = 1000,
    phase2_sampler: Optional[Any] = None,
    phase2_guider: Optional[Any] = None,
    phase2_start_step: int = 0,
) -> dict[str, Any]:
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


def start_sampling_loop(
    noise: Any,
    guider: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    cond_list: Sequence[Any],
    latent: dict[str, Any],
    chunk_frames: int,
    overlap_frames: int,
    carry_mode: str = "mask",
    overlap_strength_video: float = 1.0,
    overlap_strength_audio: float = 0.9,
    sampling_start_step: int = 0,
    sampling_end_step: int = 1000,
    phase2_start_step: int = 0,
    phase2_sampler: Optional[Any] = None,
    phase2_guider: Optional[Any] = None,
    denoise_mask: Optional[torch.Tensor] = None,
    audio_denoise_mask: Optional[torch.Tensor] = None,
) -> tuple[dict[str, Any], int, str]:
    """Execute looping sampling over an H3 whole-clip AV latent."""
    if not cond_list:
        raise ValueError("start_sampling_loop: conditioning list is empty")

    master_v, master_a = h3_unpack_av(latent, "latent")
    total_v = int(master_v.shape[H3_VIDEO_T_DIM])
    total_a = 0 if master_a is None else int(master_a.shape[H3_AUDIO_T_DIM])
    total_f = h3_latents_to_frames(total_v)

    windows = plan_h3_windows(total_f, chunk_frames, overlap_frames)
    num_chunks = len(windows)

    out_v = master_v.clone()
    out_a = None if master_a is None else master_a.clone()
    in_mask_v, in_mask_a = h3_split_noise_mask(latent)

    lines = [
        f"Sampling {num_chunks} chunk(s) over {total_v} latents ({total_f} frames, {total_f / float(H3_FPS):.2f}s), "
        f"carry mode: {carry_mode}"
    ]

    for i, (v0, v1) in enumerate(windows):
        a0 = h3_audio_idx_at_video_latent(v0, total_v, total_a)
        a1 = h3_audio_idx_at_video_latent(v1, total_v, total_a)

        # Actual carry from previous window
        prev_end = windows[i - 1][1] if i > 0 else 0
        carried_v = min(max(0, prev_end - v0), v1 - v0) if i > 0 else 0
        carried_a = (h3_audio_idx_at_video_latent(v0 + carried_v, total_v, total_a) - a0) if carried_v > 0 else 0

        sub_v = out_v[:, :, v0:v1].clone()
        sub_a = None if out_a is None else out_a[:, :, :, a0:a1].clone()

        chunk_latent = h3_pack_av(latent, sub_v, sub_a)
        if carried_v > 0:
            chunk_latent["noise_mask"] = build_carry_noise_mask(
                sub_v,
                sub_a,
                carried_v,
                carried_a,
                overlap_strength_video,
                overlap_strength_audio,
                in_mask_v,
                in_mask_a,
                v0,
                v1,
                a0,
                a1,
            )
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

        sampled = run_chunk_sampling(
            noise,
            chunk_g,
            sampler,
            sigmas,
            chunk_latent,
            start_step=sampling_start_step,
            end_step=sampling_end_step,
            phase2_sampler=phase2_sampler,
            phase2_guider=chunk_g2,
            phase2_start_step=phase2_start_step,
        )

        res_v, res_a = h3_unpack_av(sampled, f"chunk_{i}_output")
        out_v[:, :, v0:v1] = res_v.to(out_v.dtype)
        if out_a is not None and res_a is not None:
            out_a[:, :, :, a0:a1] = res_a.to(out_a.dtype)

        lines.append(f"  chunk {i}: prompt {cond_idx}, latents {v0}-{v1}, carried {carried_v}")

    final_latent = h3_pack_av(latent, out_v, out_a)
    report = "\n".join(lines)
    return final_latent, num_chunks, report


def split_h3_video_components_into_segments(
    video: Any,
    megapixels: float = 0.5,
    duration_seconds: float = 0.0,
    start_at_timestamp: float = 0.0,
    segment_count: int = 0,
    whisper_model: Any = None,
    timestamp_format: str = "00.000s",
    enable_whisper: bool = True,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], int, int, list[int], list[Any], list[str]]:
    """Split reference video components across sequential segments, returning lists."""
    from fractions import Fraction
    from comfy_api.latest import InputImpl, Types
    from .image_helpers import cached_h3_reference_components, prepare_h3_reference_components, h3_video_length_from_seconds
    from .model_helpers import transcribe_reference_audio

    components = cached_h3_reference_components(video, megapixels)
    source_frames = components.images
    source_rate = float(components.frame_rate)
    source_count = source_frames.shape[0]
    source_seconds = source_count / source_rate

    # If segment_count is specified, divide into that many equal H3 segments
    # If segment_count is 0, use duration_seconds to step across available duration
    if segment_count > 0:
        segment_indices = list(range(segment_count))
        use_fixed_segments = True
    else:
        dur = float(duration_seconds) if duration_seconds > 0 else 5.17
        start_sec = float(start_at_timestamp) if start_at_timestamp > 0 else 0.0
        remaining_sec = max(0.0, source_seconds - start_sec)
        num_segments = max(1, int(round(remaining_sec / dur)))
        segment_indices = []
        curr = start_sec
        while curr < source_seconds:
            segment_indices.append((curr, dur))
            curr += dur
        use_fixed_segments = False

    frames_list = []
    audio_list = []
    length_list = []
    video_list = []
    transcript_list = []
    out_w = 0
    out_h = 0

    if use_fixed_segments:
        for seg_idx in segment_indices:
            frames, audio, width, height, length, _, _ = prepare_h3_reference_components(
                components,
                megapixels,
                duration_seconds=0.0,
                start_at_timestamp=0.0,
                spatially_prepared=True,
                segment_count=segment_count,
                segment_index=seg_idx,
            )
            out_w, out_h = width, height
            t_audio = transcribe_reference_audio(whisper_model, components.audio, audio, timestamp_format, length) if enable_whisper else ""
            prep_video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=audio, frame_rate=Fraction(24)))
            frames_list.append(frames)
            audio_list.append(audio)
            length_list.append(length)
            video_list.append(prep_video)
            transcript_list.append(t_audio)
    else:
        for s_time, s_dur in segment_indices:
            frames, audio, width, height, length, _, _ = prepare_h3_reference_components(
                components,
                megapixels,
                duration_seconds=s_dur,
                start_at_timestamp=s_time,
                spatially_prepared=True,
            )
            out_w, out_h = width, height
            t_audio = transcribe_reference_audio(whisper_model, components.audio, audio, timestamp_format, length) if enable_whisper else ""
            prep_video = InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=audio, frame_rate=Fraction(24)))
            frames_list.append(frames)
            audio_list.append(audio)
            length_list.append(length)
            video_list.append(prep_video)
            transcript_list.append(t_audio)

    return frames_list, audio_list, out_w, out_h, length_list, video_list, transcript_list
