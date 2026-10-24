"""Native video-sequence alignment helpers.

This module adapts the complete comparison/correlation portion of the
``video-offset-finder`` reference implementation to ComfyUI BHWC tensors.
Alignment operates directly on tensors without third-party hashing.
File discovery uses ComfyUI's PyAV decoder to identify video streams.
"""

from collections.abc import Mapping
from dataclasses import dataclass
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


