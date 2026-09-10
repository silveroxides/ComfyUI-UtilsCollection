import json
import numpy as np
import comfy.model_patcher
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
        if set(expected) != set(handler.keys()):
            raise ValueError(f"Invalid {architecture.__name__} checkpoint keys")
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
