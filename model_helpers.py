import json
import base64
import zlib
from dataclasses import dataclass, field, fields, replace
from functools import cached_property, lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from torch import Tensor
from torch.distributions import Categorical
import torchaudio.functional
try:
    import tiktoken
except ImportError:
    tiktoken = None
from comfy.model_management import throw_exception_if_processing_interrupted
from .models.whisper import ModelDimensions, Whisper
import numpy as np
import comfy.model_patcher
import comfy.ops
from .model_assets import download_huggingface_model, get_model_migration, MODEL_MIGRATIONS
from .models.openpose import BodyPoseModel, HandPoseModel, FacePoseModel
from .models.yolox import YOLOXDetector
from .models.rtmpose import RTMPoseEstimator, AP10KPoseEstimator
from .models.densepose import DensePoseModel
import math
import numbers
import os
import tempfile
from contextlib import closing
from pathlib import Path

import torch
import torch.nn.functional as F
from unifiedefficientloader import IncrementalSafetensorsWriter, MemoryEfficientSafeOpen

import comfy.model_management
import comfy.utils
import folder_paths

from .encoder_helpers import (
    _encode_minimax_h3_audio_reference,
    prepare_minimax_h3_reference_image,
    prepare_minimax_h3_reference_video,
)


SAM3_WORKING_SIZE = 1008
SAM3_EDGE_PADDING = 32


MODEL_FOLDER = MODEL_MIGRATIONS["folder_category"]
MODEL_REPO = MODEL_MIGRATIONS["hf_repo"]
CHECKPOINTS = {
    "body": (get_model_migration("openpose_body")["filename"], BodyPoseModel),
    "hand": (get_model_migration("openpose_hand")["filename"], HandPoseModel),
    "face": (get_model_migration("openpose_face")["filename"], FacePoseModel),
}


def register_openpose_paths():
    base, *relative = Path(MODEL_MIGRATIONS["folder_relative"]).parts
    for directory in folder_paths.get_folder_paths(base):
        folder_paths.add_model_folder_path(MODEL_FOLDER, os.path.join(directory, *relative))
    paths, extensions = folder_paths.folder_names_and_paths[MODEL_FOLDER]
    folder_paths.folder_names_and_paths[MODEL_FOLDER] = (paths, extensions | {".safetensors"})


def load_openpose_model(kind):
    return load_migrated_pose_model(f"openpose_{kind}", CHECKPOINTS[kind][1])


def load_migrated_pose_model(kind, architecture):
    specification = get_model_migration(kind)
    if specification["architecture"] != architecture.__name__:
        raise ValueError(f"Migration architecture mismatch for {kind}")
    path = download_huggingface_model(MODEL_FOLDER, specification["filename"], MODEL_REPO, specification["hf_path"])
    return load_pose_safetensors(path, architecture)


def load_pose_safetensors(path, architecture):
    model = architecture()
    handler = MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        expected = model.state_dict()
        lazy_parameters = set()
        # Dynamic VRAM Linear leaves parameters unset until checkpoint loading.
        # Describe those tensors without allocating a second full weight buffer.
        for name, module in model.named_modules():
            if isinstance(module, comfy.ops.disable_weight_init.Linear) and module.weight is None:
                prefix = f"{name}." if name else ""
                dtype = module.weight_comfy_model_dtype or torch.get_default_dtype()
                expected[prefix + "weight"] = torch.empty((module.out_features, module.in_features), device="meta", dtype=dtype)
                lazy_parameters.add(prefix + "weight")
                if module.comfy_need_lazy_init_bias:
                    expected[prefix + "bias"] = torch.empty(module.out_features, device="meta", dtype=dtype)
                    lazy_parameters.add(prefix + "bias")
        actual_keys = set(handler.keys())
        missing = sorted(set(expected) - actual_keys)
        unexpected = sorted(actual_keys - set(expected))
        if missing or unexpected:
            raise ValueError(
                f"Invalid {architecture.__name__} checkpoint keys in {path}: "
                f"missing ({len(missing)})={missing[:10]}, "
                f"unexpected ({len(unexpected)})={unexpected[:10]}."
            )
        metadata = handler.metadata() or {}
        if metadata.get("architecture") != architecture.__name__:
            raise ValueError(f"Expected architecture metadata {architecture.__name__}")
        for key, tensor in expected.items():
            if tuple(handler.get_shape(key)) != tuple(tensor.shape) or handler.get_dtype(key) != tensor.dtype:
                raise ValueError(f"Invalid {architecture.__name__} tensor: {key}")
        keys = list(expected)
        stream = handler.async_stream(keys, batch_size=1, prefetch_batches=1, pin_memory=False)
        consumed = 0
        try:
            for batch in stream:
                for key, tensor in batch:
                    if key != keys[consumed]:
                        raise RuntimeError(f"Unexpected pose checkpoint stream key: {key}")
                    if key in lazy_parameters:
                        comfy.utils.set_attr_param(model, key, tensor)
                    else:
                        comfy.utils.copy_to_param(model, key, tensor)
                    handler.mark_processed(key)
                    consumed += 1
            if consumed != len(keys):
                raise RuntimeError("Incomplete pose checkpoint stream")
        finally:
            stream.close()
    finally:
        handler.close()
    model.eval()
    offload = comfy.model_management.unet_offload_device()
    model.to(offload)
    return comfy.model_patcher.CoreModelPatcher(
        model, load_device=comfy.model_management.get_torch_device(), offload_device=offload,
    )


def openpose_forward(patcher, images):
    comfy.model_management.throw_exception_if_processing_interrupted()
    comfy.model_management.load_models_gpu([patcher])
    array = np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2), dtype=np.float32)
    tensor = (torch.from_numpy(array) / 256.0 - 0.5).to(patcher.load_device)
    output = patcher.model(tensor)
    if isinstance(output, tuple):
        return tuple(value.detach().movedim(1, -1).cpu().numpy() for value in output)
    return output.detach().movedim(1, -1).cpu().numpy()


DWPOSE_CHECKPOINTS = {
    "detector": get_model_migration("dwpose_detector")["filename"],
    "pose": get_model_migration("dwpose_pose")["filename"],
}


def load_dwpose_model(kind):
    return load_migrated_pose_model(f"dwpose_{kind}", YOLOXDetector if kind == "detector" else RTMPoseEstimator)


def dwpose_forward(patcher, images):
    comfy.model_management.throw_exception_if_processing_interrupted()
    comfy.model_management.load_models_gpu([patcher])
    dtype = next(patcher.model.parameters()).dtype
    tensor = torch.from_numpy(np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2))).to(device=patcher.load_device, dtype=dtype)
    output = patcher.model(tensor)
    if isinstance(output, (tuple, list)):
        return tuple(value.detach().float().cpu().numpy() for value in output)
    return output.detach().float().cpu().numpy()


def load_animal_pose_model(kind):
    if kind == "detector":
        return load_dwpose_model("detector")
    return load_migrated_pose_model("animalpose", AP10KPoseEstimator)


def load_densepose_model():
    return load_migrated_pose_model("densepose_r50", DensePoseModel)


def densepose_forward(patcher, images, **options):
    comfy.model_management.throw_exception_if_processing_interrupted()
    comfy.model_management.load_models_gpu([patcher])
    tensor = torch.from_numpy(np.ascontiguousarray(np.stack(images).transpose(0, 3, 1, 2))).to(patcher.load_device)
    return patcher.model(tensor, **options)


def _extract_text_prompts(conditioning, device, dtype):
    cond_meta = conditioning[0][1]
    multi = cond_meta.get("sam3_multi_cond")
    prompts = []
    if multi is not None:
        for entry in multi:
            embedding = entry["cond"].to(device=device, dtype=dtype)
            mask = entry["attention_mask"]
            if mask is None:
                mask = torch.ones(embedding.shape[:2], dtype=torch.int64, device=device)
            else:
                mask = mask.to(device)
            prompts.append((embedding, mask, entry.get("max_detections", 1)))
    else:
        embedding = conditioning[0][0].to(device=device, dtype=dtype)
        mask = cond_meta.get("attention_mask")
        if mask is None:
            mask = torch.ones(embedding.shape[:2], dtype=torch.int64, device=device)
        else:
            mask = mask.to(device)
        prompts.append((embedding, mask, 1))
    return prompts


def _sam3_axis_ranges(length, count, tile_size):
    if count == 1:
        return [(0, length)]
    center_start = (length - tile_size) / 2
    stride = length / count
    offsets = [0]
    for offset in range(1, count // 2 + 1):
        offsets.extend((-offset, offset))
    return [
        (
            max(0, round(center_start + offset * stride)),
            min(length, round(center_start + offset * stride) + tile_size),
        )
        for offset in offsets
    ]


def _sam3_long_axis_tile_count(longest, shortest):
    count = max(3, -(-longest // shortest))
    return count if count % 2 else count + 1


def _sam3_tile_records(height, width, edge_padding):
    if height <= width:
        rows, columns = 1, _sam3_long_axis_tile_count(width, height)
    else:
        rows, columns = _sam3_long_axis_tile_count(height, width), 1
    padded_height = height + edge_padding * 2
    padded_width = width + edge_padding * 2
    tile_size = min(padded_height, padded_width)
    return [
        (y0, y1, x0, x1)
        for y0, y1 in _sam3_axis_ranges(padded_height, rows, tile_size)
        for x0, x1 in _sam3_axis_ranges(padded_width, columns, tile_size)
    ]


def _padded_sam3_tile(image, y0, y1, x0, x1, device, dtype):
    tile = image[:, :, y0:y1, x0:x1]
    tile_height, tile_width = tile.shape[-2:]
    square_size = max(tile_height, tile_width)
    pad_height = square_size - tile_height
    pad_width = square_size - tile_width
    pad_top = pad_height if y0 == 0 else 0 if y1 == image.shape[-2] else pad_height // 2
    pad_left = pad_width if x0 == 0 else 0 if x1 == image.shape[-1] else pad_width // 2
    padded = F.pad(
        tile,
        (pad_left, square_size - tile_width - pad_left, pad_top, square_size - tile_height - pad_top),
        mode="replicate",
    )
    if square_size > SAM3_WORKING_SIZE:
        frame = comfy.utils.common_upscale(padded, SAM3_WORKING_SIZE, SAM3_WORKING_SIZE, "lanczos", crop="disabled")
    else:
        frame = F.interpolate(padded, size=(SAM3_WORKING_SIZE, SAM3_WORKING_SIZE), mode="bilinear", align_corners=False)
    return frame.to(device=device, dtype=dtype), (tile_height, tile_width, square_size, pad_top, pad_left)


def _padded_sam3_image(image, edge_padding):
    image = image[..., :3].movedim(-1, 1)
    return F.pad(image, (edge_padding,) * 4, mode="replicate") if edge_padding else image


def _refine_mask(sam3_model, frame, coarse_mask, iterations):
    mask_logit = coarse_mask[None, None]
    for _ in range(iterations):
        mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
    return (mask_logit[0] > 0).float()


MINIMAX_H3_REF_FOLDER = "minimax_h3_refs"
MINIMAX_H3_REF_METADATA_KEY = "refmod_meta"
MINIMAX_H3_REF_FORMAT = 4


def _register_minimax_h3_ref_folder() -> None:
    path = os.path.join(folder_paths.models_dir, MINIMAX_H3_REF_FOLDER)
    if MINIMAX_H3_REF_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MINIMAX_H3_REF_FOLDER] = (
            [path], {".safetensors"}
        )
    else:
        folder_paths.add_model_folder_path(MINIMAX_H3_REF_FOLDER, path)


_register_minimax_h3_ref_folder()


def list_minimax_h3_refs() -> list[str]:
    return [
        name for name in folder_paths.get_filename_list(MINIMAX_H3_REF_FOLDER)
        if name.lower().endswith(".safetensors")
    ]


def get_minimax_h3_ref_input_fingerprint(filename: str) -> str:
    path = _minimax_h3_ref_load_path(filename)
    stat = os.stat(path)
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


def _validate_visual_latent(latent: torch.Tensor, kind: str) -> torch.Tensor:
    if not torch.is_tensor(latent) or latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"MiniMax H3 {kind} VAE must return a [1, 24, T, H, W] latent.")
    if not torch.is_floating_point(latent) or not torch.isfinite(latent).all():
        raise ValueError(f"MiniMax H3 {kind} VAE must return finite floating-point values.")
    if latent.shape[1] != 24 or latent.shape[2] < 1 or min(latent.shape[3:]) < 2 or latent.shape[3] % 2 or latent.shape[4] % 2:
        raise ValueError(f"MiniMax H3 {kind} VAE must return a non-empty [1, 24, T, H, W] latent.")
    if kind == "image" and latent.shape[2] != 1:
        raise ValueError("MiniMax H3 image VAE must return exactly one latent frame per image.")
    return latent.detach()


def _validate_audio_latent(latent: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(latent) or latent.ndim != 4 or tuple(latent.shape[:3]) != (1, 32, 2) or latent.shape[-1] < 1:
        raise ValueError("MiniMax H3 audio VAE must return a non-empty [1, 32, 2, T] latent.")
    if not torch.is_floating_point(latent) or not torch.isfinite(latent).all():
        raise ValueError("MiniMax H3 audio VAE must return finite floating-point values.")
    return latent.detach()


def _minimax_h3_ref_metadata(ref: dict) -> dict:
    metadata = ref.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("MiniMax H3 Ref metadata must be a dictionary.")
    return dict(metadata)


def _validate_minimax_h3_ref(ref: dict) -> dict:
    if not isinstance(ref, dict):
        raise ValueError("MiniMax H3 Ref must be a reference dictionary.")
    kind = ref.get("kind")
    if kind not in {"image", "video", "audio"}:
        raise ValueError("MiniMax H3 Ref kind must be image, video, or audio.")
    latent = ref.get("latent")
    if kind == "audio":
        latent = _validate_audio_latent(latent)
    else:
        latent = _validate_visual_latent(latent, kind)
    return {"kind": kind, "latent": latent, "metadata": _minimax_h3_ref_metadata(ref)}


def minimax_h3_ref_resolution_grid(resolution: int) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, numbers.Integral) or resolution < 32 or resolution % 32:
        raise ValueError("Reference resolution must be at least 32 pixels, in multiples of 32.")
    return resolution // 16


def _minimax_h3_ref_pool_shape(latent: torch.Tensor, grid_long_edge: int, latent_frames: int | None) -> tuple[int, int, int]:
    if grid_long_edge < 2 or grid_long_edge % 2:
        raise ValueError("MiniMax H3 Ref grid long edge must be an even integer of at least 2.")
    frames, height, width = latent.shape[2:]
    source_long_edge = max(height, width)
    scale = min(1.0, grid_long_edge / source_long_edge)
    target_height = max(2, min(height, int(round(height * scale / 2)) * 2))
    target_width = max(2, min(width, int(round(width * scale / 2)) * 2))
    target_frames = frames if latent_frames is None else min(frames, max(1, int(latent_frames)))
    return target_frames, target_height, target_width


def _pool_minimax_h3_visual_latent(latent: torch.Tensor, grid_long_edge: int, latent_frames: int | None) -> torch.Tensor:
    target_shape = _minimax_h3_ref_pool_shape(latent, grid_long_edge, latent_frames)
    if target_shape == tuple(latent.shape[2:]):
        return latent.clone()
    return F.adaptive_avg_pool3d(latent.to(torch.float32), target_shape).to(latent.dtype)


def _refine_minimax_h3_visual_latent(source: torch.Tensor, pooled: torch.Tensor, refine_steps: int) -> torch.Tensor:
    if refine_steps < 1:
        raise ValueError("MiniMax H3 Ref refinement steps must be at least 1.")
    with torch.inference_mode(False), torch.enable_grad():
        target = source.detach().to(torch.float32).clone()
        candidate = pooled.detach().to(torch.float32).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=0.02)
        for _ in range(refine_steps):
            optimizer.zero_grad()
            reconstruction = F.interpolate(candidate, size=target.shape[2:], mode="trilinear", align_corners=False)
            F.mse_loss(reconstruction, target).backward()
            optimizer.step()
    return candidate.detach().to(source.dtype)


def _compress_minimax_h3_visual_ref(ref: dict, compression: str, grid_long_edge: int, latent_frames: int | None, refine_steps: int) -> dict:
    ref = _validate_minimax_h3_ref(ref)
    if ref["kind"] == "audio":
        raise ValueError("Visual compression cannot be applied to an audio Ref.")
    if compression not in {"encode", "pooled", "refined"}:
        raise ValueError("MiniMax H3 Ref compression must be encode, pooled, or refined.")
    metadata = _minimax_h3_ref_metadata(ref)
    metadata["compression"] = compression
    metadata.setdefault("source_shape", "x".join(str(dimension) for dimension in ref["latent"].shape[2:]))
    if compression == "encode":
        return {**ref, "metadata": metadata}
    pooled = _pool_minimax_h3_visual_latent(ref["latent"], grid_long_edge, latent_frames)
    latent = pooled if compression == "pooled" else _refine_minimax_h3_visual_latent(ref["latent"], pooled, refine_steps)
    metadata.update({"grid_long_edge": grid_long_edge, "latent_frames": latent_frames, "refine_steps": refine_steps if compression == "refined" else 0})
    return {**ref, "latent": latent, "metadata": metadata}


def create_minimax_h3_image_refs(images: torch.Tensor, vae, compression: str = "encode", grid_long_edge: int = 16, refine_steps: int = 100, description: str = "") -> list[dict]:
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("MiniMax H3 Ref images must be a non-empty BHWC image batch.")
    if vae is None or not callable(getattr(vae, "encode", None)):
        raise ValueError("MiniMax H3 Ref images require a visual VAE input.")
    refs = []
    for image in images:
        prepared = prepare_minimax_h3_reference_image(image.unsqueeze(0), 2048, 2048, "max")
        ref = {"kind": "image", "latent": _validate_visual_latent(vae.encode(prepared), "image"), "metadata": {"description": description, "source": "image"}}
        refs.append(_compress_minimax_h3_visual_ref(ref, compression, grid_long_edge, None, refine_steps))
    return refs


def create_minimax_h3_video_ref(video: torch.Tensor, vae, compression: str = "encode", grid_long_edge: int = 16, latent_frames: int = 16, refine_steps: int = 100, description: str = "") -> dict:
    if not torch.is_tensor(video) or video.ndim != 4:
        raise ValueError("MiniMax H3 Ref video must be a BHWC frame batch.")
    if vae is None or not callable(getattr(vae, "encode", None)):
        raise ValueError("MiniMax H3 Ref video requires a visual VAE input.")
    _frames, block = prepare_minimax_h3_reference_video(video, vae, video.shape[0], encode_reference=True)
    latent = _validate_visual_latent(block["latent"], "video")
    ref = {"kind": "video", "latent": latent, "metadata": {"description": description, "source": "video", "source_frames": int(video.shape[0]), "prepared_frames": int(_frames.shape[0])}}
    return _compress_minimax_h3_visual_ref(ref, compression, grid_long_edge, latent_frames, refine_steps)


def create_minimax_h3_audio_ref(audio: dict, audio_vae, description: str = "") -> dict:
    if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")):
        raise ValueError("MiniMax H3 Ref audio must contain a waveform tensor.")
    waveform, sample_rate = audio["waveform"], audio.get("sample_rate")
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in {1, 2} or waveform.shape[-1] < 1 or not torch.is_floating_point(waveform) or not torch.isfinite(waveform).all() or not isinstance(sample_rate, (int, float)) or not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("MiniMax H3 Ref audio requires one finite mono or stereo waveform batch and a positive sample rate.")
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    if audio_vae is None or not callable(getattr(audio_vae, "encode", None)):
        raise ValueError("MiniMax H3 Ref audio requires an audio VAE input.")
    encoded = _encode_minimax_h3_audio_reference({"waveform": waveform, "sample_rate": sample_rate}, audio_vae)
    latent = _validate_audio_latent(encoded["audio_latent"])
    target_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    return {"kind": "audio", "latent": latent, "metadata": {"description": description, "source": "audio", "source_shape": str(latent.shape[-1]), "sample_rate": int(target_rate)}}


def _minimax_h3_ref_storage_metadata(ref: dict) -> dict:
    ref = _validate_minimax_h3_ref(ref)
    latent = ref["latent"]
    metadata = _minimax_h3_ref_metadata(ref)
    compression = metadata.get("compression", "encode")
    mode = "training" if compression in {"pooled", "refined"} else "encode"
    stored = {
        "_format_version": MINIMAX_H3_REF_FORMAT,
        "kind": ref["kind"],
        "name": metadata.get("name", ""),
        "latent_t": latent.shape[-1] if ref["kind"] == "audio" else latent.shape[2],
        "latent_h": 0 if ref["kind"] == "audio" else latent.shape[3],
        "latent_w": 0 if ref["kind"] == "audio" else latent.shape[4],
        "mode": mode,
        "source": metadata.get("source", ref["kind"]),
        "source_shape": metadata.get("source_shape", "x".join(str(dimension) for dimension in latent.shape[2:])),
        "pool": f"{latent.shape[-1]}" if ref["kind"] == "audio" else f"{latent.shape[2]}x{latent.shape[3]}x{latent.shape[4]}",
        "optimize_steps": int(metadata.get("refine_steps", 0)),
        "tags": metadata.get("tags", []),
        "description": metadata.get("description", ""),
        "concept_type": metadata.get("concept_type", "generic"),
        "sample_rate": int(metadata.get("sample_rate", 32000)),
        "shape": list(latent.shape),
        "dimensions": ({"latent_t": latent.shape[2], "latent_h": latent.shape[3], "latent_w": latent.shape[4]} if ref["kind"] != "audio" else {"ref_audio_t": latent.shape[-1]}),
        "metadata": metadata,
    }
    if "refmod_config" in metadata:
        stored["refmod_config"] = metadata["refmod_config"]
    return stored


def _minimax_h3_ref_output_root() -> str:
    paths = folder_paths.get_folder_paths(MINIMAX_H3_REF_FOLDER)
    if not paths:
        raise ValueError("MiniMax H3 Ref folder has no registered output path.")
    return str(Path(paths[0]).resolve())


def _minimax_h3_ref_relative_prefix(filename_prefix: str) -> Path:
    if not isinstance(filename_prefix, str) or not filename_prefix:
        raise ValueError("MiniMax H3 Ref filename prefix must not be empty.")
    relative = Path(filename_prefix)
    return _minimax_h3_ref_safe_relative(relative, "filename prefix")


def _minimax_h3_ref_safe_relative(relative: Path, label: str) -> Path:
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"MiniMax H3 Ref {label} must stay inside the Ref folder.")
    for part in relative.parts:
        stem = part.split(".", 1)[0].upper()
        if part in {"", ".", ".."} or ":" in part or part[-1:] in {".", " "} or stem in reserved or any(ord(character) < 32 for character in part):
            raise ValueError(f"MiniMax H3 Ref {label} contains an unsupported path component.")
    return relative


def _minimax_h3_ref_load_path(filename: str) -> str:
    if not isinstance(filename, str) or not filename.lower().endswith(".safetensors"):
        raise ValueError("MiniMax H3 Ref files must use the .safetensors format.")
    relative = Path(filename)
    _minimax_h3_ref_safe_relative(relative, "filename")
    path = Path(folder_paths.get_full_path_or_raise(MINIMAX_H3_REF_FOLDER, filename)).resolve()
    for root in folder_paths.get_folder_paths(MINIMAX_H3_REF_FOLDER):
        root_path = Path(root).resolve()
        if root_path == path.parent or root_path in path.parents:
            return str(path)
    raise ValueError("MiniMax H3 Ref filename resolves outside the Ref folder.")


def _next_minimax_h3_ref_path(root: str, prefix: Path, index: int) -> str:
    candidate = Path(root, f"{prefix}_{index:04d}.safetensors")
    root_path = Path(root).resolve()
    resolved = candidate.resolve()
    if root_path not in resolved.parents:
        raise ValueError("MiniMax H3 Ref filename prefix must stay inside the Ref folder.")
    while resolved.exists():
        index += 1
        resolved = Path(root, f"{prefix}_{index:04d}.safetensors").resolve()
        if root_path not in resolved.parents:
            raise ValueError("MiniMax H3 Ref filename prefix must stay inside the Ref folder.")
    return str(resolved)


def save_minimax_h3_ref_collection(refs: list[dict], filename_prefix: str) -> list[str]:
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ValueError("MiniMax H3 Ref Save requires at least one Ref input.")
    root = _minimax_h3_ref_output_root()
    prefix = _minimax_h3_ref_relative_prefix(filename_prefix)
    saved = []
    next_index = 1
    for ref in refs:
        ref = _validate_minimax_h3_ref(ref)
        metadata = {MINIMAX_H3_REF_METADATA_KEY: json.dumps(_minimax_h3_ref_storage_metadata(ref), separators=(",", ":"), sort_keys=True)}
        while True:
            target = _next_minimax_h3_ref_path(root, prefix, next_index)
            next_index = int(Path(target).stem.rsplit("_", 1)[-1]) + 1
            os.makedirs(os.path.dirname(target), exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".ref-", suffix=".tmp", dir=os.path.dirname(target))
            os.close(descriptor)
            try:
                with IncrementalSafetensorsWriter(temporary, metadata=metadata, max_workers=1) as writer:
                    writer.write("latent", ref["latent"].detach())
                if os.name == "nt":
                    os.rename(temporary, target)
                else:
                    os.link(temporary, target)
                break
            except FileExistsError:
                pass
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        saved.append(os.path.relpath(target, root).replace("\\", "/"))
    return saved


def load_minimax_h3_ref(filename: str) -> dict:
    path = _minimax_h3_ref_load_path(filename)
    with MemoryEfficientSafeOpen(path, low_memory=True) as handle:
        if set(handle.keys()) != {"latent"}:
            raise ValueError("MiniMax H3 Ref file must contain exactly one latent tensor.")
        headers = handle.metadata() or {}
        header = headers.get(MINIMAX_H3_REF_METADATA_KEY, headers.get("ref_meta"))
        if header is None:
            raise ValueError("File does not contain MiniMax H3 reference metadata.")
        try:
            stored = json.loads(header)
        except json.JSONDecodeError as exc:
            raise ValueError("MiniMax H3 Ref metadata is invalid JSON.") from exc
        if not isinstance(stored, dict):
            raise ValueError("MiniMax H3 Ref metadata must be a dictionary.")
        metadata = stored.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("MiniMax H3 Ref metadata must be a dictionary.")
        metadata = dict(metadata)
        for key in ("name", "description", "source", "source_shape", "tags", "concept_type", "sample_rate", "refmod_config"):
            if key in stored:
                metadata.setdefault(key, stored[key])
        steps = stored.get("optimize_steps", 0)
        metadata.setdefault("refine_steps", steps)
        mode = stored.get("mode", "encode")
        metadata.setdefault("compression", "encode" if mode in {"encode", "full"} else "refined" if steps else "pooled")
        kind = stored.get("kind")
        shape = handle.get_shape("latent")
        if not handle.get_dtype("latent").is_floating_point:
            raise ValueError("MiniMax H3 Ref latent must contain floating-point values.")
        if kind == "audio":
            valid_shape = len(shape) == 4 and tuple(shape[:3]) == (1, 32, 2) and shape[-1] > 0
        elif kind in {"image", "video"}:
            valid_shape = len(shape) == 5 and tuple(shape[:2]) == (1, 24) and shape[2] > 0 and min(shape[3:]) >= 2 and not shape[3] % 2 and not shape[4] % 2
            valid_shape = valid_shape and (kind != "image" or shape[2] == 1)
        else:
            raise ValueError("MiniMax H3 Ref kind must be image, video, or audio.")
        if not valid_shape:
            raise ValueError(f"MiniMax H3 {kind} Ref has an invalid latent shape: {shape}.")
        if "shape" in stored and stored["shape"] != list(shape):
            raise ValueError("MiniMax H3 Ref metadata does not match its latent tensor.")
        expected_dimensions = ({"latent_t": shape[2], "latent_h": shape[3], "latent_w": shape[4]} if kind != "audio" else {"ref_audio_t": shape[-1]})
        if "dimensions" in stored and stored["dimensions"] != expected_dimensions:
            raise ValueError("MiniMax H3 Ref metadata dimensions do not match its latent tensor.")
        expected_header_dimensions = (
            {"latent_t": shape[-1], "latent_h": 0, "latent_w": 0}
            if kind == "audio"
            else {"latent_t": shape[2], "latent_h": shape[3], "latent_w": shape[4]}
        )
        if any(key in stored and stored[key] != value for key, value in expected_header_dimensions.items()):
            raise ValueError("MiniMax H3 Ref header dimensions do not match its latent tensor.")
        latent = None
        with closing(handle.async_stream(["latent"], batch_size=1, prefetch_batches=1, pin_memory=False)) as stream:
            for batch in stream:
                try:
                    if len(batch) != 1 or batch[0][0] != "latent":
                        raise ValueError("MiniMax H3 Ref stream returned an unexpected tensor.")
                    latent = batch[0][1].detach().clone()
                finally:
                    for index in range(len(batch)):
                        handle.mark_processed(batch[index][0])
                    del batch
        if latent is None:
            raise ValueError("MiniMax H3 Ref stream did not return its latent tensor.")
    return _validate_minimax_h3_ref({"kind": kind, "latent": latent, "metadata": metadata})


def _minimax_h3_ref_lowpass_visual(latent: torch.Tensor) -> torch.Tensor:
    height, width = latent.shape[-2:]
    kernel_height, kernel_width = min(8, height), min(8, width)
    flat = latent.to(torch.float32).permute(0, 1, 2, 3, 4).reshape(-1, 1, height, width)
    pooled = F.avg_pool2d(flat, (kernel_height, kernel_width), stride=(kernel_height, kernel_width), ceil_mode=True)
    restored = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
    return restored.reshape_as(latent).to(latent.dtype)


def _minimax_h3_ref_lowpass_audio(latent: torch.Tensor) -> torch.Tensor:
    frames = latent.shape[-1]
    kernel = min(8, frames)
    flat = latent.to(torch.float32).reshape(-1, 1, frames)
    pooled = F.avg_pool1d(flat, kernel, stride=kernel, ceil_mode=True)
    restored = F.interpolate(pooled, size=frames, mode="linear", align_corners=False)
    return restored.reshape_as(latent).to(latent.dtype)


def _minimax_h3_ref_native_block(ref: dict, retention: float) -> dict | None:
    ref = _validate_minimax_h3_ref(ref)
    if retention == 0.0:
        return None
    latent = ref["latent"]
    if retention != 1.0:
        lowpass = _minimax_h3_ref_lowpass_audio(latent) if ref["kind"] == "audio" else _minimax_h3_ref_lowpass_visual(latent)
        latent = torch.lerp(lowpass, latent, retention)
    if ref["kind"] == "audio":
        return {"kind": "audio", "ref_audio_t": latent.shape[-1], "audio_latent": latent}
    if ref["kind"] == "image":
        return {"kind": "image", "latent": latent, "latent_h": latent.shape[3], "latent_w": latent.shape[4]}
    return {"kind": "video", "latent": latent, "latent_t": latent.shape[2], "latent_h": latent.shape[3], "latent_w": latent.shape[4], "ref_audio_t": 0, "audio_latent": None}


def _minimax_h3_native_ref_token_count(block: dict) -> int:
    kind = block.get("kind")
    if kind == "audio":
        return int(block["ref_audio_t"]) * 2
    if kind == "image":
        return (int(block["latent_h"]) // 2) * (int(block["latent_w"]) // 2)
    if kind in {"video", "video_audio"}:
        visual = int(block["latent_t"]) * (int(block["latent_h"]) // 2) * (int(block["latent_w"]) // 2)
        return visual + int(block.get("ref_audio_t", 0)) * 2
    raise ValueError("Conditioning contains an unsupported MiniMax H3 reference block.")


def flatten_minimax_h3_ref_collections(ref_collections: dict) -> list[dict]:
    refs = []
    for _name, collection in sorted(
        (ref_collections or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[-1])
    ):
        if collection is not None:
            refs.extend(collection)
    return refs


def format_minimax_h3_ref_info(refs: list[dict]) -> str:
    summaries = []
    for ref in refs:
        ref = _validate_minimax_h3_ref(ref)
        latent = ref["latent"]
        token_cost = latent.shape[-1] * 2 if ref["kind"] == "audio" else latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2)
        summary = f"{ref['kind']} {tuple(latent.shape)} ({token_cost} tokens)"
        metadata = ref["metadata"]
        if ref["kind"] == "video" and "prepared_frames" in metadata and "source_frames" in metadata and metadata["prepared_frames"] != metadata["source_frames"]:
            summary += f"; prepared {metadata['prepared_frames']} of {metadata['source_frames']} frames"
        summaries.append(summary)
    return f"{len(refs)} ref(s): " + "; ".join(summaries)


def apply_minimax_h3_refs_to_conditioning(conditioning, refs: list[dict], retention: float = 1.0, max_ref_tokens: int = 0):
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ValueError("MiniMax H3 Ref Apply requires at least one Ref input.")
    if not 0.0 <= float(retention) <= 1.0:
        raise ValueError("MiniMax H3 Ref retention must be from 0 to 1.")
    if isinstance(max_ref_tokens, bool) or not isinstance(max_ref_tokens, numbers.Integral) or max_ref_tokens < 0:
        raise ValueError("MiniMax H3 Ref max tokens must be zero or a positive integer.")
    new_blocks = [block for ref in refs if (block := _minimax_h3_ref_native_block(ref, float(retention))) is not None]
    output = []
    for embedding, metadata in conditioning:
        if not isinstance(metadata, dict):
            raise ValueError("Conditioning entries must contain metadata dictionaries.")
        existing_blocks = list(metadata.get("minimax_refs", []))
        all_blocks = existing_blocks + new_blocks
        if max_ref_tokens and sum(_minimax_h3_native_ref_token_count(block) for block in all_blocks) > int(max_ref_tokens):
            raise ValueError(f"MiniMax H3 Ref token budget of {max_ref_tokens} would be exceeded.")
        new_metadata = dict(metadata)
        new_metadata["minimax_refs"] = all_blocks
        output.append([embedding, new_metadata])
    return output


def detect_sam3(model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False, edge_padding=True):
    batch, height, width, _ = image.shape

    def boxes_to_tensor(box_list):
        return torch.tensor([[
            [(box["x"] + box["width"] / 2) / width, (box["y"] + box["height"] / 2) / height, box["width"] / width, box["height"] / height]
            for box in box_list
        ]], dtype=torch.float32)

    per_frame_boxes = None
    if bboxes is not None:
        if isinstance(bboxes, dict):
            per_frame_boxes = [boxes_to_tensor([bboxes])] * batch
        elif isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list):
            per_frame_boxes = [boxes_to_tensor(frame_boxes) if frame_boxes else None for frame_boxes in bboxes]
            per_frame_boxes.extend([per_frame_boxes[-1] if per_frame_boxes else None] * max(0, batch - len(per_frame_boxes)))
        elif isinstance(bboxes, list) and bboxes:
            per_frame_boxes = [boxes_to_tensor(bboxes)] * batch

    positive_points = json.loads(positive_coords) if positive_coords else []
    negative_points = json.loads(negative_coords) if negative_coords else []
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    point_inputs = None
    if positive_points or negative_points:
        coordinates = [[point["x"] / width * SAM3_WORKING_SIZE, point["y"] / height * SAM3_WORKING_SIZE] for point in positive_points + negative_points]
        point_inputs = {
            "point_coords": torch.tensor([coordinates], dtype=dtype, device=device),
            "point_labels": torch.tensor([[1] * len(positive_points) + [0] * len(negative_points)], dtype=torch.int32, device=device),
        }
    conditionings = _extract_text_prompts(conditioning, device, dtype) if conditioning else []
    tile_edge_padding = SAM3_EDGE_PADDING if edge_padding else 0
    image_in = None
    if point_inputs is not None or (per_frame_boxes is not None and not conditionings):
        image_in = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), SAM3_WORKING_SIZE, SAM3_WORKING_SIZE, "bilinear", crop="disabled")
    tiled_images = None
    if conditionings:
        tiled_images = _padded_sam3_image(image, tile_edge_padding)
    all_bboxes, all_masks = [], []
    progress = comfy.utils.ProgressBar(batch)
    for index in range(batch):
        frame = image_in[index:index + 1].to(device=device, dtype=dtype) if image_in is not None else None
        frame_boxes = per_frame_boxes[index].to(device=device, dtype=dtype) if per_frame_boxes is not None and per_frame_boxes[index] is not None else None
        bbox_dicts, frame_masks = [], []
        if point_inputs is not None:
            mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs)
            for _ in range(max(0, refine_iterations - 1)):
                mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
            frame_masks.append((F.interpolate(mask_logit, size=(height, width), mode="bilinear", align_corners=False)[0] > 0).float())
        if frame_boxes is not None and not conditionings:
            for cx, cy, box_width, box_height in frame_boxes[0].tolist():
                box = torch.tensor([[[(cx - box_width / 2) * SAM3_WORKING_SIZE, (cy - box_height / 2) * SAM3_WORKING_SIZE], [(cx + box_width / 2) * SAM3_WORKING_SIZE, (cy + box_height / 2) * SAM3_WORKING_SIZE]]], device=device, dtype=dtype)
                mask_logit = sam3_model.forward_segment(frame, box_inputs=box)
                for _ in range(max(0, refine_iterations - 1)):
                    mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                frame_masks.append((F.interpolate(mask_logit, size=(height, width), mode="bilinear", align_corners=False)[0] > 0).float())
        for embedding, text_mask, max_detections in conditionings:
            candidates = []
            for y0, y1, x0, x1 in _sam3_tile_records(height, width, tile_edge_padding):
                tile, tile_geometry = _padded_sam3_tile(tiled_images[index:index + 1], y0, y1, x0, x1, device, dtype)
                results = sam3_model(tile, text_embeddings=embedding, text_mask=text_mask, threshold=threshold, orig_size=(SAM3_WORKING_SIZE, SAM3_WORKING_SIZE))
                scores = results["scores"][0].sigmoid()
                kept = scores > threshold
                tile_boxes, tile_scores, tile_masks = results["boxes"][0][kept], scores[kept], results["masks"][0][kept]
                order = tile_scores.argsort(descending=True)[:max_detections]
                for box, score, mask in zip(tile_boxes[order], tile_scores[order], tile_masks[order]):
                    candidates.append((float(score), box, mask, tile, tile_geometry, y0, y1, x0, x1))

            for score, box, mask, tile, tile_geometry, y0, y1, x0, x1 in candidates:
                tile_height, tile_width, square_size, pad_top, pad_left = tile_geometry
                box_x0 = max(0.0, min(float(box[0]) * square_size / SAM3_WORKING_SIZE - pad_left, tile_width))
                box_y0 = max(0.0, min(float(box[1]) * square_size / SAM3_WORKING_SIZE - pad_top, tile_height))
                box_x1 = max(0.0, min(float(box[2]) * square_size / SAM3_WORKING_SIZE - pad_left, tile_width))
                box_y1 = max(0.0, min(float(box[3]) * square_size / SAM3_WORKING_SIZE - pad_top, tile_height))
                if box_x1 <= box_x0 or box_y1 <= box_y0:
                    continue
                global_x0, global_y0 = x0 + box_x0, y0 + box_y0
                global_x1, global_y1 = x0 + box_x1, y0 + box_y1
                clipped_x0 = max(tile_edge_padding, global_x0)
                clipped_y0 = max(tile_edge_padding, global_y0)
                clipped_x1 = min(width + tile_edge_padding, global_x1)
                clipped_y1 = min(height + tile_edge_padding, global_y1)
                if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                    continue
                bbox_dicts.append({"x": clipped_x0 - tile_edge_padding, "y": clipped_y0 - tile_edge_padding, "width": clipped_x1 - clipped_x0, "height": clipped_y1 - clipped_y0, "score": score})
                refined = _refine_mask(sam3_model, tile, mask, refine_iterations)
                refined = F.interpolate(refined[None], size=(square_size, square_size), mode="bilinear", align_corners=False)[0]
                refined = refined[:, pad_top:pad_top + tile_height, pad_left:pad_left + tile_width]
                source_y0 = max(tile_edge_padding, y0)
                source_y1 = min(height + tile_edge_padding, y1)
                source_x0 = max(tile_edge_padding, x0)
                source_x1 = min(width + tile_edge_padding, x1)
                full_mask = torch.zeros(1, height, width, device=device, dtype=dtype)
                full_mask[:, source_y0 - tile_edge_padding:source_y1 - tile_edge_padding, source_x0 - tile_edge_padding:source_x1 - tile_edge_padding] = refined[:, source_y0 - y0:source_y1 - y0, source_x0 - x0:source_x1 - x0]
                frame_masks.append(full_mask)
        all_bboxes.append(bbox_dicts)
        if frame_masks:
            combined = torch.cat(frame_masks)
            all_masks.append(combined if individual_masks else (combined > 0).any(dim=0).float())
        elif individual_masks:
            all_masks.append(torch.zeros(0, height, width, device=comfy.model_management.intermediate_device()))
        else:
            all_masks.append(torch.zeros(height, width, device=comfy.model_management.intermediate_device()))
        progress.update(1)
    intermediate_device = comfy.model_management.intermediate_device()
    masks = [mask.to(intermediate_device) for mask in all_masks]
    return (torch.cat(masks) if individual_masks else torch.stack(masks), all_bboxes)


# Whisper helpers adapted from OpenAI Whisper (MIT); see models/whisper_assets/LICENSE.
WHISPER_SAMPLE_RATE = 16000
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_CHUNK_LENGTH = 30
WHISPER_N_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH
WHISPER_N_FRAMES = WHISPER_N_SAMPLES // WHISPER_HOP_LENGTH


def prepare_whisper_audio(audio):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim != 3 or min(waveform.shape) < 1:
        raise ValueError("Whisper AUDIO waveform must be non-empty [batch, channels, samples].")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("Whisper AUDIO sample_rate must be a positive integer.")
    waveform = waveform.to(dtype=torch.float32).mean(dim=1)
    if sample_rate != WHISPER_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, WHISPER_SAMPLE_RATE)
    return waveform


def whisper_pad_or_trim(array, length=WHISPER_N_SAMPLES):
    array = array[..., :length]
    return F.pad(array, (0, length - array.shape[-1])) if array.shape[-1] < length else array


def whisper_log_mel_spectrogram(audio, n_mels=80, padding=0):
    if n_mels not in (80, 128):
        raise ValueError(f"Unsupported Whisper mel bins: {n_mels}")
    if padding:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(WHISPER_N_FFT, device=audio.device, dtype=audio.dtype)
    stft = torch.stft(audio, WHISPER_N_FFT, WHISPER_HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs().square()
    with np.load(Path(__file__).parent / "models" / "whisper_assets" / "mel_filters.npz", allow_pickle=False) as archive:
        filters = torch.from_numpy(archive[f"mel_{n_mels}"]).to(device=audio.device, dtype=audio.dtype)
    log_spec = (filters @ magnitudes).clamp(min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


WHISPER_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
WHISPER_TO_LANGUAGE_CODE = {
    **{language: code for code, language in WHISPER_LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}


@dataclass
class WhisperTokenizer:
    """A thin wrapper around `tiktoken` providing quick access to special tokens"""

    encoding: "tiktoken.Encoding"
    num_languages: int
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(WHISPER_LANGUAGES.keys())[: self.num_languages]
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    @cached_property
    def eot(self) -> int:
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        return self.to_language_token(self.language)

    def to_language_token(self, language):
        if token := self.special_tokens.get(f"<|{language}|>", None):
            return token

        raise KeyError(f"Language {language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in WHISPER_LANGUAGES:
                result.append(token_id)
        return tuple(result)[: self.num_languages]

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([_l]).strip("<|>") for _l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))



@lru_cache(maxsize=None)
def whisper_get_encoding(name: str = "gpt2", num_languages: int = 99):
    if tiktoken is None:
        raise RuntimeError("Whisper requires tiktoken. Install tiktoken in ComfyUI's Python environment and restart ComfyUI.")
    vocab_path = os.path.join(os.path.dirname(__file__), "models", "whisper_assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(WHISPER_LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def whisper_get_tokenizer(
    multilingual: bool,
    *,
    num_languages: int = 99,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> WhisperTokenizer:
    if language is not None:
        language = language.lower()
        if language not in WHISPER_LANGUAGES:
            if language in WHISPER_TO_LANGUAGE_CODE:
                language = WHISPER_TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = whisper_get_encoding(name=encoding_name, num_languages=num_languages)

    return WhisperTokenizer(
        encoding=encoding, num_languages=num_languages, language=language, task=task
    )



def whisper_compression_ratio(text):
    encoded = text.encode("utf-8")
    return len(encoded) / len(zlib.compress(encoded))

def detect_whisper_language(
    model: "Whisper", mel: Tensor, tokenizer: WhisperTokenizer = None
) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = whisper_get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class WhisperDecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details


@dataclass(frozen=True)
class WhisperDecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class WhisperInference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class WhisperPyTorchInference(WhisperInference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        key_modules = [block.attn.key for block in self.model.decoder.blocks]
        value_modules = [block.attn.value for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()


class WhisperSequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class WhisperMaximumLikelihoodRanker(WhisperSequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class WhisperTokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class WhisperGreedyDecoder(WhisperTokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


class WhisperBeamSearchDecoder(WhisperTokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: WhisperInference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class WhisperLogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class WhisperSuppressBlank(WhisperLogitFilter):
    def __init__(self, tokenizer: WhisperTokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class WhisperSuppressTokens(WhisperLogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class WhisperApplyTimestampRules(WhisperLogitFilter):
    def __init__(
        self,
        tokenizer: WhisperTokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class WhisperDecodingTask:
    inference: WhisperInference
    sequence_ranker: WhisperSequenceRanker
    decoder: WhisperTokenDecoder
    logit_filters: List[WhisperLogitFilter]

    def __init__(self, model: "Whisper", options: WhisperDecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = whisper_get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: WhisperTokenizer = tokenizer
        self.options: WhisperDecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = WhisperPyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = WhisperMaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = WhisperBeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = WhisperGreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(WhisperSuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(WhisperSuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = WHISPER_CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                WhisperApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: WhisperDecodingOptions) -> WhisperDecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        mel = mel.to(dtype=self.model.compute_dtype)

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != self.model.compute_dtype:
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = detect_whisper_language(
                self.model, audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                throw_exception_if_processing_interrupted()
                logits = self.inference.logits(tokens, audio_features)

                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: Tensor) -> List[WhisperDecodingResult]:
        self.decoder.reset()
        tokenizer: WhisperTokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                WhisperDecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            WhisperDecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=whisper_compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode_whisper(
    model: "Whisper",
    mel: Tensor,
    options: WhisperDecodingOptions = WhisperDecodingOptions(),
    **kwargs,
) -> Union[WhisperDecodingResult, List[WhisperDecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: WhisperDecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[WhisperDecodingResult, List[WhisperDecodingResult]]
        The result(s) of decoding contained in `WhisperDecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = WhisperDecodingTask(model, options).run(mel)

    return result[0] if single else result



WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3")
WHISPER_REPO = "silveroxides/ComfyUI-UtilsCollection-Models"


def register_whisper_paths():
    folder_paths.add_model_folder_path("whisper", os.path.join(folder_paths.models_dir, "whisper"))
    paths, extensions = folder_paths.folder_names_and_paths["whisper"]
    folder_paths.folder_names_and_paths["whisper"] = (paths, extensions | {".safetensors"})


def load_whisper_model(model_name):
    if model_name not in WHISPER_MODELS:
        raise ValueError(f"Unknown Whisper model: {model_name}")
    register_whisper_paths()
    filename = f"{model_name}.safetensors"
    path = download_huggingface_model("whisper", filename, WHISPER_REPO, f"audio/whisper/{filename}")
    return load_whisper_safetensors(path, model_name)


def whisper_dimensions(metadata, model_name):
    if metadata.get("architecture") != "openai-whisper" or metadata.get("official_model") != model_name:
        raise ValueError(f"Expected native Whisper {model_name} checkpoint metadata.")
    dimensions = json.loads(metadata["dims"])
    if set(dimensions) != {field.name for field in fields(ModelDimensions)} or any(type(value) is not int or value <= 0 for value in dimensions.values()):
        raise ValueError("Invalid Whisper dimensions metadata.")
    dims = ModelDimensions(**dimensions)
    state, heads, layers = {"tiny": (384, 6, 4), "base": (512, 8, 6), "small": (768, 12, 12), "medium": (1024, 16, 24), "large-v2": (1280, 20, 32), "large-v3": (1280, 20, 32)}[model_name]
    expected = ModelDimensions(128 if model_name == "large-v3" else 80, 1500, state, heads, layers, 51866 if model_name == "large-v3" else 51865, 448, state, heads, layers)
    if dims != expected:
        raise ValueError(f"Checkpoint dimensions do not match Whisper {model_name}.")
    return dims


def load_whisper_safetensors(path, model_name):
    handler = MemoryEfficientSafeOpen(str(path), low_memory=True)
    try:
        dims = whisper_dimensions(handler.metadata() or {}, model_name)
        # Meta construction avoids an additional full-sized allocation before UEL streaming.
        with torch.device("meta"):
            model = Whisper(dims)
        expected = dict(model.state_dict())
        for name, module in model.named_modules():
            if isinstance(module, comfy.ops.disable_weight_init.Linear) and module.weight is None:
                expected[f"{name}.weight"] = torch.empty((module.out_features, module.in_features), device="meta")
                if module.comfy_need_lazy_init_bias:
                    expected[f"{name}.bias"] = torch.empty(module.out_features, device="meta")
        actual_keys = set(handler.keys())
        if actual_keys != set(expected):
            raise ValueError(f"Invalid Whisper checkpoint keys: missing={sorted(set(expected) - actual_keys)[:10]}, unexpected={sorted(actual_keys - set(expected))[:10]}")
        for name, tensor in expected.items():
            if tuple(handler.get_shape(name)) != tuple(tensor.shape) or handler.get_dtype(name) not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"Invalid Whisper checkpoint tensor: {name}")
        stream = handler.async_stream(list(expected), batch_size=1, prefetch_batches=1, pin_memory=False)
        consumed = set()
        try:
            for batch in stream:
                comfy.model_management.throw_exception_if_processing_interrupted()
                for name, tensor in batch:
                    comfy.utils.set_attr_param(model, name, tensor)
                    handler.mark_processed(name)
                    consumed.add(name)
        finally:
            stream.close()
        if consumed != set(expected):
            raise RuntimeError("Incomplete Whisper checkpoint stream.")
    finally:
        handler.close()
    offload_device = comfy.model_management.unet_offload_device()
    model.decoder.mask = torch.full((dims.n_text_ctx, dims.n_text_ctx), -torch.inf, device=offload_device).triu_(1)
    load_device = comfy.model_management.get_torch_device()
    model.compute_dtype = comfy.model_management.unet_dtype(load_device, model_params=sum(t.numel() for t in expected.values()), supported_dtypes=[torch.float16, torch.float32])
    model.device = offload_device
    model.eval().to(offload_device)
    return comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)


def whisper_decode_with_fallback(model, mel, language, task, prompt):
    for temperature in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        comfy.model_management.throw_exception_if_processing_interrupted()
        result = decode_whisper(model, mel, WhisperDecodingOptions(language=language, task=task, prompt=prompt, temperature=temperature))
        needs_fallback = result.compression_ratio > 2.4 or result.avg_logprob < -1.0
        if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
            needs_fallback = False
        if not needs_fallback:
            break
    return result


def transcribe_whisper(model, waveform, task, language):
    mel = whisper_log_mel_spectrogram(waveform, model.dims.n_mels, padding=WHISPER_N_SAMPLES)
    content_frames = mel.shape[-1] - WHISPER_N_FRAMES
    duration = waveform.shape[-1] / WHISPER_SAMPLE_RATE
    if language == "auto":
        first_window = whisper_pad_or_trim(mel, WHISPER_N_FRAMES).to(device=model.device, dtype=model.compute_dtype)
        _, probabilities = detect_whisper_language(model, first_window)
        language = max(probabilities, key=probabilities.get)
    tokenizer = whisper_get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=language, task=task)
    input_stride = WHISPER_N_FRAMES // model.dims.n_audio_ctx
    time_precision = input_stride * WHISPER_HOP_LENGTH / WHISPER_SAMPLE_RATE
    seek, prompt_reset_since = 0, 0
    all_tokens, segments = [], []
    progress = comfy.utils.ProgressBar(content_frames)
    while seek < content_frames:
        comfy.model_management.throw_exception_if_processing_interrupted()
        previous_seek = seek
        time_offset = seek * WHISPER_HOP_LENGTH / WHISPER_SAMPLE_RATE
        segment_size = min(WHISPER_N_FRAMES, content_frames - seek)
        window = whisper_pad_or_trim(mel[:, seek:seek + segment_size], WHISPER_N_FRAMES).to(device=model.device, dtype=model.compute_dtype)
        result = whisper_decode_with_fallback(model, window, language, task, all_tokens[prompt_reset_since:])
        if result.no_speech_prob > 0.6 and result.avg_logprob <= -1.0:
            seek += segment_size
            progress.update_absolute(seek)
            continue

        tokens = result.tokens
        timestamps = [token >= tokenizer.timestamp_begin for token in tokens]
        single_timestamp_ending = timestamps[-2:] == [False, True]
        consecutive = [i + 1 for i in range(len(tokens) - 1) if timestamps[i] and timestamps[i + 1]]
        current_segments = []

        def add_segment(start, end, selected_tokens):
            text = tokenizer.decode([token for token in selected_tokens if token < tokenizer.eot])
            # Timestamp tokens may extend into the padded final window.
            start, end = min(start, duration), min(end, duration)
            if end <= start or not text.strip():
                return
            current_segments.append({"start": start, "end": end, "text": text, "tokens": selected_tokens})

        if consecutive:
            if single_timestamp_ending:
                consecutive.append(len(tokens))
            last_slice = 0
            for current_slice in consecutive:
                selected = tokens[last_slice:current_slice]
                add_segment(time_offset + (selected[0] - tokenizer.timestamp_begin) * time_precision,
                            time_offset + (selected[-1] - tokenizer.timestamp_begin) * time_precision, selected)
                last_slice = current_slice
            if single_timestamp_ending:
                seek += segment_size
            else:
                seek += (tokens[last_slice - 1] - tokenizer.timestamp_begin) * input_stride
        else:
            segment_duration = segment_size * WHISPER_HOP_LENGTH / WHISPER_SAMPLE_RATE
            timestamp_values = [token for token in tokens if token >= tokenizer.timestamp_begin]
            if timestamp_values and timestamp_values[-1] != tokenizer.timestamp_begin:
                segment_duration = (timestamp_values[-1] - tokenizer.timestamp_begin) * time_precision
            add_segment(time_offset, time_offset + segment_duration, tokens)
            seek += segment_size

        if seek <= previous_seek:
            raise RuntimeError("Whisper produced non-advancing timestamps; transcription cannot continue.")
        for segment in current_segments:
            all_tokens.extend(segment.pop("tokens"))
            segments.append(segment)
        if result.temperature > 0.5:
            prompt_reset_since = len(all_tokens)
        progress.update_absolute(min(content_frames, seek))
    return {"text": tokenizer.decode(all_tokens), "segments": segments, "language": language}


def run_whisper(patcher, audio, task, language):
    if tiktoken is None:
        raise RuntimeError("Whisper requires tiktoken. Install tiktoken in ComfyUI's Python environment and restart ComfyUI.")
    if task not in ("transcribe", "translate"):
        raise ValueError(f"Unknown Whisper task: {task}")
    if language != "auto" and language not in WHISPER_LANGUAGES:
        raise ValueError(f"Unknown Whisper language: {language}")
    if language != "auto" and list(WHISPER_LANGUAGES).index(language) >= patcher.model.num_languages:
        raise ValueError(f"Whisper checkpoint does not support language: {language}")
    recordings = prepare_whisper_audio(audio)
    comfy.model_management.load_models_gpu([patcher])
    transcripts, segments, languages = [], [], []
    for waveform in recordings:
        comfy.model_management.throw_exception_if_processing_interrupted()
        result = transcribe_whisper(patcher.model, waveform, task, language)
        transcripts.append(result["text"])
        segments.append(json.dumps(result["segments"], ensure_ascii=False))
        languages.append(result["language"])
    return transcripts, segments, languages
