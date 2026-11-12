"""Native video-sequence alignment helpers.

This module adapts the complete comparison/correlation portion of the
``video-offset-finder`` reference implementation to ComfyUI BHWC tensors.
Alignment operates directly on tensors without third-party hashing.
File discovery uses ComfyUI's PyAV decoder to identify video streams.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction
import json
import logging
import math
import os
import re
from typing import Any

import av
from comfy.cli_args import args
import folder_paths
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".avi", ".flv", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg",
    ".mpg", ".mts", ".ts", ".webm", ".wmv",
}


def normalize_video_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    path = path.strip().replace("\\", "/")
    return os.path.abspath(os.path.normpath(path)) if path else ""


def is_video_file(path: str) -> bool:
    """Identify video content through the same decoder used by VideoFromFile."""
    try:
        with av.open(path, mode="r") as container:
            return bool(container.streams.video)
    except (av.FFmpegError, OSError):
        return False


def list_video_files(directory: str) -> list[str]:
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
        and is_video_file(os.path.join(directory, name))
    )


class VideoCompareType(str, Enum):
    PHASH = "phash"
    DHASH = "dhash"
    AHASH = "ahash"
    WHASH = "whash"
    SAD = "sad"


@dataclass(frozen=True)
class VideoCorrelationResult:
    offset_frames: int
    distance: float
    second_best_distance: float | None
    overlap_frames: int


def _validate_frames(frames: torch.Tensor) -> torch.Tensor:
    if (
        not torch.is_tensor(frames)
        or frames.ndim != 4
        or frames.shape[0] < 1
        or min(frames.shape[1:3]) < 1
        or frames.shape[3] < 3
        or not torch.isfinite(frames).all()
    ):
        raise ValueError("Video alignment frames must be a finite BHWC RGB tensor.")
    return frames[..., :3].to(torch.float32)


def _gray(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    samples = F.interpolate(
        frames.movedim(-1, 1), size=(height, width), mode="area"
    )
    return (
        0.2126 * samples[:, :1]
        + 0.7152 * samples[:, 1:2]
        + 0.0722 * samples[:, 2:3]
    ).squeeze(1)


def _dct_basis(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(size, device=device, dtype=dtype)
    basis = torch.cos(math.pi / size * (index[:, None] + 0.5) * index[None, :])
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _phash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    side = hash_size * 4
    pixels = _gray(frames, side, side)
    basis = _dct_basis(side, pixels.device, pixels.dtype)
    coefficients = basis @ pixels @ basis.t()
    low = coefficients[:, :hash_size, :hash_size]
    median = low.flatten(1).median(dim=1, keepdim=True).values
    return (low.flatten(1) > median).to(torch.uint8)


def _dhash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    pixels = _gray(frames, hash_size, hash_size + 1)
    return (pixels[..., 1:] > pixels[..., :-1]).flatten(1).to(torch.uint8)


def _ahash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    pixels = _gray(frames, hash_size, hash_size)
    return (pixels > pixels.flatten(1).mean(dim=1, keepdim=True).view(-1, 1, 1)).flatten(1).to(torch.uint8)


def _whash(frames: torch.Tensor, hash_size: int) -> torch.Tensor:
    size = hash_size * 2
    pixels = _gray(frames, size, size)
    while pixels.shape[-1] > hash_size:
        pixels = (
            pixels[..., 0::2, 0::2]
            + pixels[..., 1::2, 0::2]
            + pixels[..., 0::2, 1::2]
            + pixels[..., 1::2, 1::2]
        ) * 0.25
    return (pixels > pixels.flatten(1).mean(dim=1, keepdim=True).view(-1, 1, 1)).flatten(1).to(torch.uint8)


def _sad(frames: torch.Tensor, width: int = 64, height: int = 64) -> torch.Tensor:
    return (_gray(frames, height, width) * 255.0).round().flatten(1)


def compute_video_signatures(
    frames: torch.Tensor,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    hash_size: int = 16,
) -> torch.Tensor:
    """Compute one native signature row per BHWC video frame."""
    frames = _validate_frames(frames)
    compare_type = VideoCompareType(compare_type)
    if compare_type == VideoCompareType.SAD:
        return _sad(frames)
    if isinstance(hash_size, bool) or not isinstance(hash_size, int) or hash_size < 1:
        raise ValueError("Video alignment hash_size must be a positive integer.")
    if compare_type == VideoCompareType.PHASH:
        return _phash(frames, hash_size)
    if compare_type == VideoCompareType.DHASH:
        return _dhash(frames, hash_size)
    if compare_type == VideoCompareType.AHASH:
        return _ahash(frames, hash_size)
    if compare_type == VideoCompareType.WHASH:
        return _whash(frames, hash_size)
    return _sad(frames)


def _candidate_offsets(
    reference_count: int,
    query_count: int,
    min_overlap_fraction: float,
    min_overlap_frames: int,
    min_offset_frames: int | None,
    max_offset_frames: int | None,
) -> tuple[range, int]:
    if reference_count < 1 or query_count < 1:
        raise ValueError("Cannot correlate empty video signatures.")
    shortest = min(reference_count, query_count)
    required = max(1, min_overlap_frames, math.ceil(shortest * min_overlap_fraction))
    lower_bound = -query_count + required if min_offset_frames is None else min_offset_frames
    upper_bound = reference_count - required if max_offset_frames is None else max_offset_frames
    lower = max(-query_count + required, lower_bound)
    upper = min(reference_count - required, upper_bound)
    if lower > upper:
        return range(0), required
    return range(lower, upper + 1), required


def cross_correlate_video_signatures(
    reference: torch.Tensor,
    query: torch.Tensor,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    min_overlap_fraction: float = 0.5,
    min_overlap_frames: int = 2,
    min_offset_frames: int | None = None,
    max_offset_frames: int | None = None,
) -> VideoCorrelationResult:
    """Find the globally best temporal offset and report ambiguity."""
    compare_type = VideoCompareType(compare_type)
    reference = reference.to(torch.uint8) if compare_type != VideoCompareType.SAD else reference.to(torch.float32)
    query = query.to(reference.dtype)
    offsets, _ = _candidate_offsets(
        reference.shape[0], query.shape[0], min_overlap_fraction,
        min_overlap_frames, min_offset_frames, max_offset_frames,
    )
    candidates: list[tuple[float, int, int]] = []
    for offset in offsets:
        reference_start = max(offset, 0)
        query_start = max(-offset, 0)
        overlap = min(reference.shape[0] - reference_start, query.shape[0] - query_start)
        if overlap < 1:
            continue
        first = reference[reference_start:reference_start + overlap]
        second = query[query_start:query_start + overlap]
        if compare_type == VideoCompareType.SAD:
            distance = (first.to(torch.float32) - second.to(torch.float32)).abs().mean(dim=1)
        else:
            distance = (first != second).to(torch.float32).mean(dim=1)
        candidates.append((float(distance.mean().item()), offset, overlap))
    if not candidates:
        return VideoCorrelationResult(0, float("inf"), None, 0)
    candidates.sort(key=lambda item: (item[0], -item[2], abs(item[1])))
    best_distance, best_offset, best_overlap = candidates[0]
    return VideoCorrelationResult(
        best_offset,
        best_distance,
        candidates[1][0] if len(candidates) > 1 else None,
        best_overlap,
    )


def rank_video_overlap_candidates(
    previous: torch.Tensor,
    current: torch.Tensor,
    maximum_overlap_frames: int,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    hash_size: int = 16,
) -> list[tuple[int, float]]:
    """Rank previous-tail/current-head overlaps using inexpensive signatures.

    Callers can validate the ranked candidates with more expensive visual or
    audio comparisons without recomputing the initial signatures.
    """
    previous = _validate_frames(previous)
    current = _validate_frames(current)
    maximum_overlap_frames = min(maximum_overlap_frames, previous.shape[0], current.shape[0] - 1)
    if maximum_overlap_frames < 1:
        return []
    previous_signatures = compute_video_signatures(previous, compare_type, hash_size)
    current_signatures = compute_video_signatures(current, compare_type, hash_size)
    candidates: list[tuple[float, int]] = []
    for overlap in range(maximum_overlap_frames, 0, -1):
        first = previous_signatures[-overlap:]
        second = current_signatures[:overlap]
        if compare_type == VideoCompareType.SAD:
            distance = (first.to(torch.float32) - second.to(torch.float32)).abs().mean().item() / 255.0
        else:
            distance = (first != second).to(torch.float32).mean().item()
        candidates.append((float(distance), overlap))
    return [(overlap, distance) for distance, overlap in sorted(
        candidates, key=lambda item: (item[0], -item[1])
    )]


def find_video_overlap(
    previous: torch.Tensor,
    current: torch.Tensor,
    maximum_overlap_frames: int,
    compare_type: VideoCompareType | str = VideoCompareType.PHASH,
    hash_size: int = 16,
) -> VideoCorrelationResult:
    """Find the best previous-tail/current-head overlap."""
    candidates = rank_video_overlap_candidates(
        previous, current, maximum_overlap_frames, compare_type, hash_size
    )
    if not candidates:
        return VideoCorrelationResult(0, float("inf"), None, 0)
    best_overlap, best_distance = candidates[0]
    return VideoCorrelationResult(
        best_overlap,
        best_distance,
        candidates[1][1] if len(candidates) > 1 else None,
        best_overlap,
    )


def audio_overlap_similarity(
    previous: torch.Tensor,
    current: torch.Tensor,
    overlap_frames: int,
    fps: int,
    sample_rate: int,
) -> float | None:
    """Compare suffix/prefix audio over a video-frame duration."""
    sample_count = round(overlap_frames * sample_rate / fps)
    if sample_count < max(256, sample_rate // 10):
        return None
    first = previous[:, -sample_count:].to(torch.float32)
    second = current[:, :sample_count].to(torch.float32)
    first = first - first.mean(dim=-1, keepdim=True)
    second = second - second.mean(dim=-1, keepdim=True)
    if first.norm() < 1e-5 or second.norm() < 1e-5:
        return None
    correlation = F.cosine_similarity(first.reshape(1, -1), second.reshape(1, -1)).item()
    return max(0.0, min(1.0, (correlation + 1.0) * 0.5))


def resample_video_frames_to_24fps(frames: torch.Tensor, source_fps: float) -> tuple[torch.Tensor, float]:
    """Resample video frames to 24 fps preserving real-time duration and playback speed.

    If source_fps < 24 fps, fills in with clones of neighboring frames.
    If source_fps > 24 fps, uniformly strips frames.
    """
    if frames is None or not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[0] == 0:
        return frames, source_fps
    source_rate = float(source_fps)
    if not math.isfinite(source_rate) or source_rate <= 0:
        return frames, 24.0
    if abs(source_rate - 24.0) <= 0.01:
        return frames, 24.0

    orig_count = frames.shape[0]
    target_count = max(1, round(orig_count * 24.0 / source_rate))
    indices = [min(orig_count - 1, round(i * source_rate / 24.0)) for i in range(target_count)]
    resampled = frames[indices]
    return resampled, 24.0


def prepare_video_audio_for_h3(audio: Any) -> dict[str, Any] | None:
    """Format extracted video audio into a valid MiniMax H3 audio structure (32kHz stereo, multiple of 800 samples).

    Does not trim the end of the audio.
    """
    if audio is None:
        return None

    waveform = None
    sample_rate = 44100

    if isinstance(audio, Mapping):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate", 44100)
    elif hasattr(audio, "waveform"):
        waveform = getattr(audio, "waveform")
        sample_rate = getattr(audio, "sample_rate", 44100)

    if waveform is None or not torch.is_tensor(waveform) or waveform.numel() == 0:
        return None

    waveform = waveform.to(dtype=torch.float32)

    # Ensure shape is [batch, channels, samples] with batch=1
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 3:
        waveform = waveform.reshape(1, waveform.shape[-2], waveform.shape[-1])

    # Ensure stereo (2 channels)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2, :]

    # Resample to 32000 Hz if needed
    sample_rate = int(sample_rate) if sample_rate and math.isfinite(sample_rate) else 32000
    if sample_rate != 32000 and waveform.numel() > 0:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 32000)
        sample_rate = 32000

    # Ensure sample length is a multiple of 800 without trimming the end
    remainder = waveform.shape[-1] % 800
    if remainder != 0:
        pad = 800 - remainder
        waveform = F.pad(waveform, (0, pad))

    return {"waveform": waveform, "sample_rate": 32000}


_FILENAME_DATE_TOKEN = re.compile(r"%date:([^%]+)%")
_FILENAME_DATE_PART = re.compile(r"(yyyy|yy|(\w)\2?)")


def expand_filename_date_tokens(filename_prefix: str, now: datetime | None = None) -> str:
    """Expand ``%date:FORMAT%`` the way the ComfyUI frontend does.

    The frontend only rewrites these tokens for core save nodes, and
    ``folder_paths.get_save_image_path`` does not know them, so custom save
    nodes otherwise write the literal token (with its colon) into the path.
    """
    now = now or datetime.now()
    parts = {"d": now.day, "M": now.month, "h": now.hour, "m": now.minute, "s": now.second}

    def format_part(match: re.Match) -> str:
        token, letter = match.group(1), match.group(2)
        if token == "yyyy":
            return str(now.year)
        if token == "yy":
            return str(now.year)[2:]
        if letter in parts:
            return str(parts[letter]).zfill(len(token))
        return token

    return _FILENAME_DATE_TOKEN.sub(lambda m: _FILENAME_DATE_PART.sub(format_part, m.group(1)), filename_prefix)


def save_keyframe_optimized_video(
    images: torch.Tensor,
    fps: float = 24.0,
    codec: str = "auto",
    gop_size: int = 100,
    b_frames: int = 0,
    bitrate_mbps: float = 10.0,
    crf: float = 19.0,
    audio: Any = None,
    filename_prefix: str = "video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default",
    prompt: Any = None,
    extra_pnginfo: Any = None,
    save_workflow_image: bool = True,
) -> tuple[str, str, str]:
    """Encode video frames to MP4 using native PyAV with Avidemux-matching GOP and scene-cut keyframes."""
    if images is None or not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("save_keyframe_optimized_video requires a non-empty image batch [B, H, W, C].")

    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0

    batch, height, width, channels = images.shape
    out_width = width - (width % 2)
    out_height = height - (height % 2)
    if out_width <= 0 or out_height <= 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")

    if not filename_prefix:
        filename_prefix = "video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default"
    filename_prefix = expand_filename_date_tokens(filename_prefix)

    has_audio = False
    audio_np = None
    audio_sample_rate = 44100
    if audio is not None:
        has_audio = True
        waveform = None
        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            audio_sample_rate = int(audio.get("sample_rate", 44100))
        elif hasattr(audio, "waveform"):
            waveform = getattr(audio, "waveform")
            audio_sample_rate = int(getattr(audio, "sample_rate", 44100))

        if torch.is_tensor(waveform) and waveform.numel() > 0:
            if waveform.ndim == 3:
                waveform = waveform[0]
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).repeat(2, 1)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            audio_np = waveform.detach().cpu().to(dtype=torch.float32).contiguous().numpy()

    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), out_width, out_height
    )
    os.makedirs(full_output_folder, exist_ok=True)

    if os.path.exists(full_output_folder):
        pattern = re.compile(rf"^{re.escape(filename)}_(\d+)")
        existing_counters = []
        for f in os.listdir(full_output_folder):
            m = pattern.match(f)
            if m:
                try:
                    existing_counters.append(int(m.group(1)))
                except ValueError:
                    pass
        if existing_counters:
            counter = max(counter, max(existing_counters) + 1)

    suffix = "-audio" if has_audio else ""
    file = f"{filename}_{counter:05}{suffix}.mp4"
    output_path = os.path.join(full_output_folder, file)

    available = av.codecs_available
    target_codec = codec.lower() if codec else "auto"
    if target_codec == "auto":
        if "h264_nvenc" in available and torch.cuda.is_available() and out_width >= 128 and out_height >= 128:
            target_codec = "h264_nvenc"
        else:
            target_codec = "libx264"
    elif "av1" in target_codec:
        if "nvenc" in target_codec:
            if "av1_nvenc" in available and torch.cuda.is_available() and out_width >= 144 and out_height >= 144:
                target_codec = "av1_nvenc"
            elif "libsvtav1" in available:
                target_codec = "libsvtav1"
            else:
                target_codec = "libx264"
        else:
            target_codec = "libsvtav1" if "libsvtav1" in available else "libx264"
    elif "nvenc" in target_codec and (out_width < 128 or out_height < 128 or not torch.cuda.is_available()):
        target_codec = "libx265" if "hevc" in target_codec else "libx264"

    container = av.open(
        output_path,
        mode="w",
        options={"movflags": "use_metadata_tags+faststart"},
    )
    try:
        if prompt is not None:
            container.metadata["prompt"] = prompt if isinstance(prompt, str) else json.dumps(prompt)
        if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
            for k, v in extra_pnginfo.items():
                container.metadata[k] = v if isinstance(v, str) else json.dumps(v)

        rate = Fraction(round(fps * 1000), 1000)
        try:
            stream = container.add_stream(target_codec, rate=rate)
        except Exception as exc:
            if target_codec == "av1_nvenc" and "libsvtav1" in available:
                logging.warning(
                    "[UC_SaveVideo] av1_nvenc stream initialization failed (%s); falling back to libsvtav1.",
                    exc,
                )
                target_codec = "libsvtav1"
                stream = container.add_stream(target_codec, rate=rate)
            else:
                raise
        stream.width = out_width
        stream.height = out_height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = int(gop_size)
        stream.max_b_frames = int(b_frames)

        if "nvenc" in target_codec:
            stream.bit_rate = int(bitrate_mbps * 1_000_000)
            stream.options = {
                "preset": "p5",
                "tune": "hq",
                "forced-idr": "1",
            }
        elif "264" in target_codec:
            stream.options = {
                "crf": str(int(crf)),
                "preset": "medium",
                "x264-params": f"scenecut=40:keyint={int(gop_size)}:min-keyint=1",
            }
        elif "265" in target_codec:
            stream.options = {
                "crf": str(int(crf)),
                "preset": "medium",
                "x265-params": f"no-open-gop=1:keyint={int(gop_size)}",
            }
        elif "svtav1" in target_codec or "av1" in target_codec:
            stream.options = {
                "crf": str(int(crf)),
                "preset": "6",
                "svtav1-params": f"keyint={int(gop_size)}",
            }
        else:
            stream.options = {"crf": str(int(crf))}

        a_stream = None
        if audio_np is not None:
            a_stream = container.add_stream("aac", rate=audio_sample_rate, layout="stereo")

        prev_thumb = None
        cut_count = 0
        for i in tqdm(range(batch), desc=f"Encoding Video ({target_codec})", unit="frame"):
            frame_tensor = images[i, :out_height, :out_width, :3]
            thumb = F.adaptive_avg_pool2d(frame_tensor.permute(2, 0, 1).unsqueeze(0), (16, 16))
            is_cut = (i == 0)
            if prev_thumb is not None:
                if torch.mean(torch.abs(thumb - prev_thumb)).item() > 0.25:
                    is_cut = True
            prev_thumb = thumb

            uint8_img = (frame_tensor.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8, device="cpu").numpy()
            frame = av.VideoFrame.from_ndarray(uint8_img, format="rgb24")
            frame.pts = i
            if is_cut:
                cut_count += 1
                frame.pict_type = av.video.frame.PictureType.I
                frame.key_frame = True

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        if a_stream is not None and audio_np is not None:
            frame_size = a_stream.codec_context.frame_size or 1024
            num_samples = audio_np.shape[1]
            for start in range(0, num_samples, frame_size):
                chunk = audio_np[:, start : start + frame_size]
                if chunk.shape[1] < frame_size:
                    padded = np.zeros((2, frame_size), dtype=np.float32)
                    padded[:, : chunk.shape[1]] = chunk
                    chunk = padded
                a_frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                a_frame.sample_rate = audio_sample_rate
                a_frame.pts = start
                for packet in a_stream.encode(a_frame):
                    container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)

    finally:
        container.close()

    if save_workflow_image:
        first_frame = 255.0 * images[0, ..., :3].detach().cpu().numpy()
        img = Image.fromarray(np.clip(first_frame, 0, 255).astype(np.uint8))
        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", prompt if isinstance(prompt, str) else json.dumps(prompt))
            if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                for x in extra_pnginfo:
                    metadata.add_text(x, extra_pnginfo[x] if isinstance(extra_pnginfo[x], str) else json.dumps(extra_pnginfo[x]))
        image_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.png")
        img.save(image_path, pnginfo=metadata, compress_level=4)

    audio_desc = f"aac {audio_sample_rate}Hz stereo" if audio_np is not None else "none"
    logging.info(
        "[UC_SaveVideo] Saved %d frame(s) (%.2fs at %.2f fps) at %dx%d | codec=%s, scene_keyframes=%d, audio=%s -> %s",
        batch, batch / fps, fps, out_width, out_height, target_codec, cut_count, audio_desc, output_path,
    )
    return output_path, file, subfolder


def encode_keyframe_optimized_video_clip(
    images: torch.Tensor,
    fps: float = 24.0,
    audio: Any = None,
    codec: str = "auto",
    gop_size: int = 100,
    b_frames: int = 0,
    bitrate_mbps: float = 10.0,
    crf: float = 19.0,
):
    """Re-encode a segment clip with PyAV (h264_nvenc + scene-cut IDR keyframes) and return a VideoFromFile."""
    import io as pyio
    from comfy_api.latest import InputImpl, Types

    if images is None or not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("encode_keyframe_optimized_video_clip requires a non-empty image batch [B, H, W, C].")

    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0

    batch, height, width, _ = images.shape
    out_width = width - (width % 2)
    out_height = height - (height % 2)
    if out_width <= 0 or out_height <= 0:
        return InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(round(fps * 1000), 1000))
        )

    audio_np = None
    audio_sample_rate = 32000
    if audio is not None:
        waveform = None
        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            audio_sample_rate = int(audio.get("sample_rate", 32000))
        elif hasattr(audio, "waveform"):
            waveform = getattr(audio, "waveform")
            audio_sample_rate = int(getattr(audio, "sample_rate", 32000))

        if torch.is_tensor(waveform) and waveform.numel() > 0:
            if waveform.ndim == 3:
                waveform = waveform[0]
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).repeat(2, 1)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            exact_samples = max(1, round((batch / fps) * audio_sample_rate))
            waveform = waveform[:, :exact_samples]
            audio_np = waveform.detach().cpu().to(dtype=torch.float32).contiguous().numpy()

    available = av.codecs_available
    target_codec = codec.lower() if codec else "auto"
    if target_codec == "auto":
        if "h264_nvenc" in available and torch.cuda.is_available() and out_width >= 128 and out_height >= 128:
            target_codec = "h264_nvenc"
        else:
            target_codec = "libx264"
    elif "nvenc" in target_codec and (out_width < 128 or out_height < 128 or not torch.cuda.is_available()):
        target_codec = "libx265" if "hevc" in target_codec else "libx264"

    buffer = pyio.BytesIO()
    container = av.open(buffer, mode="w", format="mp4", options={"movflags": "use_metadata_tags"})
    try:
        rate = Fraction(round(fps * 1000), 1000)
        stream = container.add_stream(target_codec, rate=rate)
        stream.width = out_width
        stream.height = out_height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = int(gop_size)
        stream.max_b_frames = int(b_frames)

        if "nvenc" in target_codec:
            stream.bit_rate = int(bitrate_mbps * 1_000_000)
            stream.options = {
                "preset": "p5",
                "tune": "hq",
                "forced-idr": "1",
            }
        elif "264" in target_codec:
            stream.options = {
                "crf": str(int(crf)),
                "preset": "medium",
                "x264-params": f"scenecut=40:keyint={int(gop_size)}:min-keyint=1",
            }
        elif "265" in target_codec:
            stream.options = {
                "crf": str(int(crf)),
                "preset": "medium",
                "x265-params": f"no-open-gop=1:keyint={int(gop_size)}",
            }
        else:
            stream.options = {"crf": str(int(crf))}

        a_stream = None
        if audio_np is not None:
            a_stream = container.add_stream("aac", rate=audio_sample_rate, layout="stereo")

        prev_thumb = None
        for i in range(batch):
            frame_tensor = images[i, :out_height, :out_width, :3]
            thumb = F.adaptive_avg_pool2d(frame_tensor.permute(2, 0, 1).unsqueeze(0), (16, 16))
            is_cut = (i == 0)
            if prev_thumb is not None:
                if torch.mean(torch.abs(thumb - prev_thumb)).item() > 0.25:
                    is_cut = True
            prev_thumb = thumb

            uint8_img = (frame_tensor.clamp(0.0, 1.0) * 255.0).to(dtype=torch.uint8, device="cpu").numpy()
            frame = av.VideoFrame.from_ndarray(uint8_img, format="rgb24")
            frame.pts = i
            if is_cut:
                frame.pict_type = av.video.frame.PictureType.I
                frame.key_frame = True

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        if a_stream is not None and audio_np is not None:
            frame_size = a_stream.codec_context.frame_size or 1024
            num_samples = audio_np.shape[1]
            for start in range(0, num_samples, frame_size):
                chunk = audio_np[:, start : start + frame_size]
                if chunk.shape[1] < frame_size:
                    padded = np.zeros((2, frame_size), dtype=np.float32)
                    padded[:, : chunk.shape[1]] = chunk
                    chunk = padded
                a_frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                a_frame.sample_rate = audio_sample_rate
                a_frame.pts = start
                for packet in a_stream.encode(a_frame):
                    container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)
    finally:
        container.close()

    buffer.seek(0)
    return InputImpl.VideoFromFile(buffer)


class _H3SegmentInt(int):
    """Integer subclass that preserves per-segment overlap and padding metadata across ComfyUI INT wires."""
    _uc_overlap_frames: int = 0
    _uc_padded_frames: int = 0


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
    *,
    return_trim_metadata: bool = False,
    cache_large_clips: bool = False,
):
    """Split reference video components across sequential H3 windows with exact overlap and trailing-padding tracking."""
    from comfy_api.latest import InputImpl
    from .image_helpers import cached_h3_reference_components, h3_video_length_from_seconds
    from .model_helpers import transcribe_reference_audio
    from .sampling_helpers import h3_snap_frame_count

    components = cached_h3_reference_components(video, megapixels, cache_large_clips=cache_large_clips)
    source_frames = components.images
    source_rate = float(components.frame_rate)
    source_count = source_frames.shape[0]
    source_seconds = getattr(source_frames, "source_seconds", None)
    if source_seconds is None:
        source_seconds = source_count / source_rate
    total_frames = max(1, round(source_seconds * 24))
    out_h, out_w = int(source_frames.shape[1]), int(source_frames.shape[2])

    start_offset = max(0, round(float(start_at_timestamp) * 24)) if start_at_timestamp > 0 else 0
    if start_offset >= total_frames:
        raise ValueError("Start timestamp rounds past the end of the reference video.")

    ov_frames = max(0, int(overlap_frames))
    rem_total = max(1, total_frames - start_offset)
    if segment_count > 0:
        target_f = (rem_total + max(0, int(segment_count) - 1) * ov_frames) / int(segment_count)
        window_frames = h3_snap_frame_count(math.ceil(target_f))
    elif duration_seconds > 0:
        window_frames = h3_video_length_from_seconds(duration_seconds)
    else:
        window_frames = 124

    if ov_frames >= window_frames:
        ov_frames = max(0, window_frames - 17)

    windows: list[tuple[int, int, int, int, int]] = []
    curr = start_offset
    while curr < total_frames:
        if segment_count > 0 and len(windows) == int(segment_count) - 1:
            real_end = total_frames
        else:
            real_end = min(total_frames, curr + window_frames)
        real_len = max(1, real_end - curr)
        seg_overlap = 0 if not windows else max(0, windows[-1][1] - curr)
        min_needed_len = max(real_len, seg_overlap + 1) if seg_overlap > 0 else real_len
        seg_length = h3_snap_frame_count(min_needed_len)
        padded_frames = max(0, seg_length - real_len)
        windows.append((curr, real_end, seg_length, seg_overlap, padded_frames))
        if real_end >= total_frames or (segment_count > 0 and len(windows) >= int(segment_count)):
            break
        next_curr = real_end - ov_frames
        if next_curr <= curr:
            next_curr = curr + max(1, window_frames - ov_frames)
        curr = next_curr

    frames_list = []
    audio_list = []
    length_list = []
    video_list = []
    transcript_list = []
    trim_trailing_list = []
    soundtrack = components.audio

    for f_start, real_end, seg_length, seg_overlap, padded_frames in tqdm(windows, desc="Preparing H3 Video Segments", unit="seg"):
        frame_positions = [min(pos, max(f_start, real_end - 1)) for pos in range(f_start, f_start + seg_length)]
        frame_indices = [min(max(0, round(pos * source_rate / 24)), source_count - 1) for pos in frame_positions]
        frames = source_frames[frame_indices]

        audio_window = round(seg_length / 24 * 32000)
        aligned_samples = ((audio_window + 799) // 800) * 800
        if soundtrack is not None and soundtrack.get("waveform") is not None and soundtrack["waveform"].numel() > 0:
            sr = int(soundtrack["sample_rate"])
            start_sample = round(f_start / 24 * sr)
            end_sample = round(real_end / 24 * sr)
            audio_samples = soundtrack["waveform"][..., start_sample:max(start_sample, end_sample)]
            if sr != 32000 and audio_samples.numel():
                audio_samples = torchaudio.functional.resample(audio_samples, sr, 32000)
            audio_samples = audio_samples[..., :audio_window]
            if audio_samples.shape[-1] < aligned_samples:
                audio_samples = F.pad(audio_samples, (0, aligned_samples - audio_samples.shape[-1]))
        else:
            audio_samples = torch.zeros(1, 2, aligned_samples)

        audio = {
            "waveform": audio_samples,
            "sample_rate": 32000,
            "_uc_overlap_frames": int(seg_overlap),
            "_uc_padded_frames": int(padded_frames),
        }
        t_audio = transcribe_reference_audio(whisper_model, components.audio, audio, timestamp_format, seg_length) if enable_whisper else ""
        prep_video = encode_keyframe_optimized_video_clip(images=frames, fps=24.0, audio=audio)
        frames._uc_overlap_frames = int(seg_overlap)
        frames._uc_padded_frames = int(padded_frames)
        seg_len_obj = _H3SegmentInt(seg_length)
        seg_len_obj._uc_overlap_frames = int(seg_overlap)
        seg_len_obj._uc_padded_frames = int(padded_frames)
        try:
            prep_video._uc_overlap_frames = int(seg_overlap)
            prep_video._uc_padded_frames = int(padded_frames)
        except Exception:
            pass
        frames_list.append(frames)
        audio_list.append(audio)
        length_list.append(seg_len_obj)
        video_list.append(prep_video)
        transcript_list.append(t_audio)
        trim_trailing_list.append(int(padded_frames))

    has_src_audio = soundtrack is not None and soundtrack.get("waveform") is not None and soundtrack["waveform"].numel() > 0
    logging.info(
        "[UC_H3RefVideoSegments] Source %d frame(s) (%.2fs at %.2f fps) -> %d segment(s) at %dx%d | lengths=%s | overlap=%d frame(s) | padded=%s | audio=%s",
        source_count, source_seconds, source_rate, len(windows), out_w, out_h,
        [int(x) for x in length_list], ov_frames, trim_trailing_list,
        "32kHz" if has_src_audio else "silence-filled",
    )

    if return_trim_metadata:
        return (
            frames_list,
            audio_list,
            out_w,
            out_h,
            length_list,
            video_list,
            transcript_list,
            str(ov_frames if ov_frames in (5, 22, 39, 56) else 22),
            trim_trailing_list,
        )
    return frames_list, audio_list, out_w, out_h, length_list, video_list, transcript_list


_ALLOWED_VIDEO_EXTS = {
    "mp4", "mov", "mkv", "webm", "avi", "m4v", "gif",
    "mpg", "mpeg", "wmv", "flv", "ogv", "ts",
}


def generate_video_preview(video_path: str, file_input: str, preview_mode: str) -> dict[str, Any] | None:
    """Generate or retrieve a lightweight preview UI output based on preview_mode.

    Supported preview_mode options:
      - 'none': returns None (no preview sent to frontend)
      - 'thumbnail': single extracted static frame saved as compressed WebP in temp
      - 'reduced_proxy': downscaled low-fps proxy clip encoded in temp
      - 'full': raw original file reference (matches Core LoadVideo behavior)
    """
    from comfy_api.latest import io, ui

    if preview_mode == "none":
        return None

    if preview_mode == "full":
        name, _ = folder_paths.annotated_filepath(file_input)
        subfolder, _, filename = name.replace("\\", "/").rpartition("/")
        result = ui.SavedResult(filename, subfolder, io.FolderType.input)
        return ui.PreviewVideo([result]).as_dict()

    if not os.path.isfile(video_path):
        return None

    mtime = int(os.path.getmtime(video_path))
    temp_dir = folder_paths.get_temp_directory()
    path_stem = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(video_path))[0])[:32]

    if preview_mode == "thumbnail":
        thumb_filename = f"UC_thumb_{path_stem}_{mtime}.webp"
        thumb_path = os.path.join(temp_dir, thumb_filename)
        if not os.path.isfile(thumb_path):
            try:
                with av.open(video_path) as container:
                    stream = next((s for s in container.streams if s.type == "video"), None)
                    if stream is None:
                        return None
                    frame = next(container.decode(stream), None)
                    if frame is None:
                        return None
                    img = frame.to_image()
                    w, h = img.size
                    max_dim = max(w, h)
                    if max_dim > 512:
                        scale = 512.0 / max_dim
                        new_w = max(1, round(w * scale))
                        new_h = max(1, round(h * scale))
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    img.save(thumb_path, format="WEBP", quality=80)
            except Exception as e:
                logging.warning("[UC_LoadVideo] Failed to generate thumbnail preview: %s", e)
                return None

        result = ui.SavedResult(thumb_filename, "", io.FolderType.temp)
        return ui.SavedImages([result]).as_dict()

    if preview_mode == "reduced_proxy":
        proxy_filename = f"UC_proxy_{path_stem}_{mtime}.mp4"
        proxy_path = os.path.join(temp_dir, proxy_filename)
        if not os.path.isfile(proxy_path):
            try:
                with av.open(video_path) as in_container:
                    in_stream = next((s for s in in_container.streams if s.type == "video"), None)
                    if in_stream is None:
                        return None

                    orig_w = in_stream.width
                    orig_h = in_stream.height
                    target_h = 360
                    target_w = max(2, round((orig_w * (target_h / orig_h)) / 2) * 2)

                    with av.open(proxy_path, mode="w") as out_container:
                        out_stream = out_container.add_stream("libx264", rate=10)
                        out_stream.width = target_w
                        out_stream.height = target_h
                        out_stream.pix_fmt = "yuv420p"
                        out_stream.options = {"crf": "28", "preset": "ultrafast"}

                        max_frames = 100
                        frame_count = 0
                        source_fps = float(in_stream.average_rate or 24.0)
                        step = max(1, round(source_fps / 10.0))

                        for idx, frame in enumerate(in_container.decode(in_stream)):
                            if idx % step != 0:
                                continue
                            re_frame = frame.reformat(width=target_w, height=target_h)
                            for packet in out_stream.encode(re_frame):
                                out_container.mux(packet)
                            frame_count += 1
                            if frame_count >= max_frames:
                                break

                        for packet in out_stream.encode():
                            out_container.mux(packet)
            except Exception as e:
                logging.warning("[UC_LoadVideo] Failed to generate reduced proxy preview: %s", e)
                return None

        result = ui.SavedResult(proxy_filename, "", io.FolderType.temp)
        return ui.PreviewVideo([result]).as_dict()

    return None


async def handle_uc_video_upload(request: Any) -> Any:
    """Stream incoming video file upload straight to input/ without buffering whole body in memory."""
    from aiohttp import web
    import uuid

    reader = await request.multipart()
    field = None
    subfolder = ""

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "subfolder":
            subfolder = (await part.text()).strip()
        elif part.name == "file":
            field = part
            break

    if field is None or not field.filename:
        return web.json_response({"error": "No video file field in upload request."}, status=400)

    base = os.path.basename(field.filename.replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower().lstrip(".")
    if ext not in _ALLOWED_VIDEO_EXTS:
        return web.json_response(
            {"error": f"Video extension {ext!r} not allowed. Supported: {', '.join(sorted(_ALLOWED_VIDEO_EXTS))}"},
            status=400,
        )

    clean_stem = re.sub(r"[^A-Za-z0-9_\- ]", "_", stem).strip().strip(".") or "video"
    input_base = os.path.realpath(folder_paths.get_input_directory())
    target_dir = os.path.realpath(os.path.join(input_base, subfolder)) if subfolder else input_base

    if os.path.commonpath([input_base, target_dir]) != input_base:
        return web.json_response({"error": "Invalid subfolder path (path traversal blocked)."}, status=400)

    os.makedirs(target_dir, exist_ok=True)

    candidate = f"{clean_stem}.{ext}"
    target_path = os.path.join(target_dir, candidate)
    counter = 1
    while os.path.exists(target_path):
        candidate = f"{clean_stem}_{counter}.{ext}"
        target_path = os.path.join(target_dir, candidate)
        counter += 1

    temp_path = os.path.join(target_dir, f".{clean_stem}_{uuid.uuid4().hex}.tmp")

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await field.read_chunk(262144)  # 256 KB
                if not chunk:
                    break
                f.write(chunk)

        os.replace(temp_path, target_path)
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        logging.error("[UC_LoadVideo] Video upload failed: %s", e)
        return web.json_response({"error": f"Failed to save uploaded video: {e}"}, status=500)

    return web.json_response({"name": candidate, "subfolder": subfolder})


_UC_UPLOAD_ROUTE_REGISTERED = False


def register_uc_video_upload_route() -> None:
    """Register /uc/api/upload_video route on ComfyUI's PromptServer if available."""
    global _UC_UPLOAD_ROUTE_REGISTERED
    if _UC_UPLOAD_ROUTE_REGISTERED:
        return
    try:
        from server import PromptServer
        routes = PromptServer.instance.routes
        routes.post("/uc/api/upload_video")(handle_uc_video_upload)
        _UC_UPLOAD_ROUTE_REGISTERED = True
        logging.info("[UC_LoadVideo] Registered /uc/api/upload_video streaming upload endpoint.")
    except Exception as e:
        logging.debug("[UC_LoadVideo] PromptServer not available for route registration: %s", e)

