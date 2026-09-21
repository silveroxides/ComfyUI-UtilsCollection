"""Sampling helpers for windowed and looping generation workflows."""

import copy
import logging
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
import torchaudio

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


def prepare_h3_source_audio(
    audio: dict[str, Any],
    audio_vae: Any,
    audio_template: torch.Tensor,
    total_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit raw source audio to an H3 clip and encode it into its audio latent."""
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        raise ValueError("UC_H3LoopSampler source_audio must contain a waveform tensor.")
    waveform = audio["waveform"]
    sample_rate = audio.get("sample_rate")
    if (
        waveform.ndim != 3
        or waveform.shape[1] not in (1, 2)
        or waveform.shape[-1] < 1
        or not torch.is_floating_point(waveform)
        or not torch.isfinite(waveform).all()
        or not isinstance(sample_rate, (int, float))
        or sample_rate <= 0
    ):
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


def plan_h3_schedule(
    total_frames: int,
    window_frames: int = 124,
    overlap_frames: int = 22,
    segment_lengths: Optional[Sequence[int]] = None,
) -> tuple[int, int, list[dict[str, Any]], list[tuple[int, int]]]:
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

        window_records.append({
            "chunk_index": i,
            "v0": v0,
            "v1": v1,
            "a0": a0,
            "a1": a1,
            "carried_v": carried_v,
            "carried_a": carried_a,
            "frame_start": f_start,
            "frame_end": f_end,
        })
        frame_spans.append((f_start, f_end))

    return total_v, total_a, window_records, frame_spans


def plan_h3_windows(
    total_frames: int,
    window_frames: int,
    overlap_frames: int,
    segment_lengths: Optional[Sequence[int]] = None,
) -> list[tuple[int, int]]:
    """Backward compatible wrapper around plan_h3_schedule."""
    _, _, records, _ = plan_h3_schedule(
        total_frames=total_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        segment_lengths=segment_lengths,
    )
    return [(r["v0"], r["v1"]) for r in records]


def prepare_chunk_guider(guider: Any, positive_cond: Any, frame0: Optional[int] = None) -> Any:
    """Create isolated clone of guider with swapped positive cond and timeline frame0 offsets."""
    chunk_g = copy.copy(guider)
    chunk_g.original_conds = dict(guider.original_conds)

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
        vm[:, :, :carried_v] = torch.minimum(vm[:, :, :carried_v], torch.full_like(vm[:, :, :carried_v], 1.0 - float(strength_v)))

    if am is not None and carried_a > 0:
        am[:, :, :, :carried_a] = torch.minimum(am[:, :, :, :carried_a], torch.full_like(am[:, :, :, :carried_a], 1.0 - float(strength_a)))

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
        LOGGER.info(
            f"UC_H3LoopSampler: expanding master latent from {total_f} frames ({total_v} latents) "
            f"to {target_f} frames ({target_v} latents) to accommodate all {len(cond_list)} chunks."
        )
        expanded_v = torch.zeros(
            [master_v.shape[0], master_v.shape[1], target_v, master_v.shape[3], master_v.shape[4]],
            dtype=master_v.dtype,
            device=master_v.device,
        )
        expanded_v[:, :, :total_v] = master_v
        master_v = expanded_v
        total_v = target_v
        total_f = target_f

        if master_a is not None and target_a > total_a:
            expanded_a = torch.zeros(
                [master_a.shape[0], master_a.shape[1], master_a.shape[2], target_a],
                dtype=master_a.dtype,
                device=master_a.device,
            )
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

    total_v, total_a, window_records, _ = plan_h3_schedule(
        total_frames=total_f,
        window_frames=chunk_frames,
        overlap_frames=overlap_frames,
        segment_lengths=segment_lengths,
    )
    num_chunks = len(window_records)

    out_v = master_v.clone()
    out_a = None if master_a is None else master_a.clone()
    in_mask_v, in_mask_a = h3_split_noise_mask(latent)
    if source_audio is not None:
        in_mask_a = torch.zeros(
            [master_a.shape[0], 1, master_a.shape[2], master_a.shape[3]],
            dtype=torch.float32,
            device=master_a.device,
        )

    lines = [
        f"Sampling {num_chunks} chunk(s) over {total_v} latents ({total_f} frames, {total_f / float(H3_FPS):.2f}s), "
        f"carry mode: {carry_mode}, audio mode: {audio_mode}"
    ]

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

        # Reset any block residual caches so each chunk gets a clean schedule start
        reset_model_caches(chunk_g)
        if chunk_g2 is not None:
            reset_model_caches(chunk_g2)

        # Generate unique noise per chunk to prevent repeated motion loops
        c_noise = chunk_noise(noise, i)

        sampled = run_chunk_sampling(
            c_noise,
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


def split_h3_video_components_into_segments(
    video: Any,
    megapixels: float = 0.5,
    duration_seconds: float = 0.0,
    start_at_timestamp: float = 0.0,
    segment_count: int = 0,
    overlap_frames: int = 22,
    whisper_model: Any = None,
    timestamp_format: str = "00.000s",
    enable_whisper: bool = True,
) -> tuple[list[torch.Tensor], list[dict[str, Any]], int, int, list[int], list[Any], list[str]]:
    """Split reference video components across planned H3 schedule windows, returning lists."""
    from fractions import Fraction
    from comfy_api.latest import InputImpl, Types
    from .image_helpers import cached_h3_reference_components, prepare_h3_reference_components, h3_video_length_from_seconds
    from .model_helpers import transcribe_reference_audio

    components = cached_h3_reference_components(video, megapixels)
    source_frames = components.images
    source_rate = float(components.frame_rate)
    source_count = source_frames.shape[0]
    source_seconds = source_count / source_rate
    total_frames = max(1, round(source_seconds * 24))

    # If segment_count is provided, derive window_frames to divide evenly into segment_count windows
    if segment_count > 0:
        target_f = total_frames / segment_count
        window_frames = h3_snap_frame_count(round(target_f))
    elif duration_seconds > 0:
        window_frames = h3_video_length_from_seconds(duration_seconds)
    else:
        window_frames = 124

    # Use the single authoritative schedule planner so window frame spans match the sampler 1:1
    _, _, window_records, _ = plan_h3_schedule(
        total_frames=total_frames,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
    )

    frames_list = []
    audio_list = []
    length_list = []
    video_list = []
    transcript_list = []
    out_w = 0
    out_h = 0

    for w in window_records:
        f_start = w["frame_start"]
        f_end = w["frame_end"]
        f_len = f_end - f_start
        s_time = f_start / 24.0
        s_dur = f_len / 24.0

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
