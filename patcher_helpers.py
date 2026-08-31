"""Model-scoped patch helpers for diffusion-model cache nodes."""

from __future__ import annotations

import bisect
import contextvars
import hashlib
import logging
import math
import os
import re
import time
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.ldm.common_dit
from comfy.ldm.ideogram4 import model as ideogram4_model
import comfy.ldm.modules.attention
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy.text_encoders.minimax import process_video_block, token_tags_from_embeds_info
from comfy.ldm.minimax import model as minimax_model
from safetensors import safe_open


MINIMAX_H3_CACHE_OWNER_KEY = "utilscollection_minimax_h3_cache"
MINIMAX_H3_SPECTRUM_OWNER_KEY = "utilscollection_minimax_h3_spectrum"
MINIMAX_H3_PDD_OWNER_KEY = "utilscollection_minimax_h3_pdd_acc"
UNIFIED_ATTENTION_OWNER_KEY = "utilscollection_unified_attention"
MINIMAX_H3_RADIAL_WRAPPER_KEY = "utilscollection_minimax_h3_radial"
MINIMAX_H3_RADIAL_STATE_KEY = "utilscollection_minimax_h3_radial_state"
MINIMAX_H3_PROJECTION_FOLDER = "clip_projections"
MINIMAX_H3_PROJECTED_KEY = "qwen3vl_32b"
MINIMAX_H3_PROJECTION_PATCH_KEY = "utilscollection_minimax_h3_projection"
MINIMAX_H3_SOURCE_KEYS = {"qwen3vl_4b", "qwen3vl_8b"}
MINIMAX_H3_PAD_TOKEN = 151643
MINIMAX_H3_VISION_START = 151652
MINIMAX_H3_VISION_END = 151653
IDEOGRAM4_DEBANNER_WRAPPER_KEY = "utilscollection_ideogram4_debanner"
IDEOGRAM4_DEBANNER_STATE_KEY = "utilscollection_ideogram4_debanner_state"
IDEOGRAM4_DEBANNER_BUNDLE = Path(__file__).resolve().parent / "models" / "ideogram4_correction_v1.safetensors"


def _register_minimax_h3_projection_folder() -> None:
    path = os.path.join(folder_paths.models_dir, MINIMAX_H3_PROJECTION_FOLDER)
    if MINIMAX_H3_PROJECTION_FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MINIMAX_H3_PROJECTION_FOLDER] = (
            [path],
            {".safetensors"},
        )
    else:
        folder_paths.add_model_folder_path(MINIMAX_H3_PROJECTION_FOLDER, path)


_register_minimax_h3_projection_folder()


def list_minimax_h3_projections() -> list[str]:
    return [
        name
        for name in folder_paths.get_filename_list(MINIMAX_H3_PROJECTION_FOLDER)
        if name.lower().endswith(".safetensors")
    ]


def _projection_scalar(data: dict[str, Any], metadata: dict[str, str], key: str) -> int:
    value = metadata.get(key, data.get(key))
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Projection {key} must contain one value.")
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Projection is missing a valid {key} value.") from exc


def load_minimax_h3_projection(name: str) -> tuple[dict[str, torch.Tensor], int]:
    if not name.lower().endswith(".safetensors"):
        raise ValueError("MiniMax H3 projections must use the .safetensors format.")
    path = folder_paths.get_full_path_or_raise(MINIMAX_H3_PROJECTION_FOLDER, name)
    with safe_open(path, framework="pt", device="cpu") as handle:
        data = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata() or {}
    required = ("mean_in", "std_in", "mean_out", "std_out")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Projection is missing required tensors: {', '.join(missing)}.")
    tap = _projection_scalar(data, metadata, "tap")
    return data, tap


def _projection_layers(data: dict[str, torch.Tensor]) -> list[int]:
    layers = set()
    for key in data:
        if not key.startswith("mlp."):
            continue
        parts = key.split(".")
        if len(parts) != 3 or parts[2] not in {"weight", "bias"}:
            raise ValueError(f"Unsupported projection tensor: {key}.")
        try:
            layers.add(int(parts[1]))
        except ValueError as exc:
            raise ValueError(f"Unsupported projection tensor: {key}.") from exc
    return sorted(layers)


class MiniMaxH3ProjectionModel(torch.nn.Module):
    def __init__(self, data: dict[str, torch.Tensor], tap: int):
        super().__init__()
        self.tap = tap
        for key in ("mean_in", "std_in", "mean_out", "std_out"):
            value = data[key]
            if value.ndim != 1 or not torch.is_floating_point(value):
                raise ValueError(f"Projection {key} must be a floating-point vector.")
            self.register_buffer(key, value.float())
        d_in = self.mean_in.shape[0]
        d_out = self.mean_out.shape[0]
        if self.std_in.shape != self.mean_in.shape or self.std_out.shape != self.mean_out.shape:
            raise ValueError("Projection mean and standard-deviation shapes must match.")
        if torch.any(self.std_in == 0) or torch.any(self.std_out == 0):
            raise ValueError("Projection standard deviations must be nonzero.")

        weight = data.get("W")
        if weight is not None:
            if weight.ndim != 2 or tuple(weight.shape) != (d_in, d_out):
                raise ValueError(
                    f"Projection W must have shape [{d_in}, {d_out}]."
                )
            self.linear_weight = torch.nn.Parameter(weight.float())
        else:
            self.register_parameter("linear_weight", None)

        residual_layers = []
        previous_out = d_in
        layer_indices = _projection_layers(data)
        for position, index in enumerate(layer_indices):
            weight_key = f"mlp.{index}.weight"
            bias_key = f"mlp.{index}.bias"
            if weight_key not in data:
                raise ValueError(f"Projection residual layer {index} has no weight.")
            residual_weight = data[weight_key]
            if residual_weight.ndim != 2 or residual_weight.shape[1] != previous_out:
                raise ValueError(f"Projection residual layer {index} has incompatible dimensions.")
            linear = torch.nn.Linear(
                residual_weight.shape[1],
                residual_weight.shape[0],
                bias=bias_key in data,
                dtype=residual_weight.dtype,
            )
            linear.weight = torch.nn.Parameter(residual_weight)
            if bias_key in data:
                bias = data[bias_key]
                if bias.ndim != 1 or bias.shape[0] != residual_weight.shape[0]:
                    raise ValueError(f"Projection residual layer {index} has an incompatible bias.")
                linear.bias = torch.nn.Parameter(bias)
            residual_layers.append(linear)
            if position < len(layer_indices) - 1:
                residual_layers.append(torch.nn.GELU())
            previous_out = residual_weight.shape[0]
        if residual_layers and previous_out != d_out:
            raise ValueError(f"Projection residual output must have {d_out} dimensions.")
        self.residual = torch.nn.Sequential(*residual_layers) if residual_layers else None
        if self.linear_weight is None and self.residual is None:
            raise ValueError("Projection requires a linear matrix or residual network.")

        sink = data.get("sink_out")
        if sink is not None:
            if sink.ndim != 1 or sink.shape[0] != d_out:
                raise ValueError(f"Projection sink_out must have {d_out} values.")
            self.register_buffer("sink_out", sink.float())
        else:
            self.register_buffer("sink_out", None)
        self.device = torch.device("cpu")

    @property
    def input_dimensions(self) -> int:
        return self.mean_in.shape[0]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        if hidden.shape[-1] != self.input_dimensions:
            raise ValueError(
                f"Connected encoder produces {hidden.shape[-1]} dimensions, but projection expects {self.input_dimensions}."
            )
        normalized = (hidden - self.mean_in) / self.std_in
        projected = normalized @ self.linear_weight if self.linear_weight is not None else None
        if self.residual is not None:
            residual_dtype = self.residual[0].weight.dtype
            residual = self.residual(normalized.to(residual_dtype)).float()
            projected = residual if projected is None else projected + residual
        output = projected * self.std_out + self.mean_out
        if self.sink_out is not None and output.shape[1] > 0:
            output[:, 0] = self.sink_out
        return output


class _MiniMaxH3ProjectedTokenizer:
    clip_name = MINIMAX_H3_PROJECTED_KEY

    def __init__(self, source_tokenizer: Any, source_key: str):
        self.source_tokenizer = source_tokenizer
        self.source_key = source_key
        self.raw_tokenizer = getattr(source_tokenizer, source_key)

    def _text_entries(self, text: str) -> list[tuple]:
        if not text:
            return []
        batches = self.raw_tokenizer.tokenize_with_weights(
            text,
            return_word_ids=False,
            disable_weights=True,
        )
        if len(batches) != 1:
            raise ValueError("MiniMax H3 projected text exceeds the supported prompt length.")
        return list(batches[0])

    @staticmethod
    def _vision_entry(data: torch.Tensor, video_block: bool = False) -> dict[str, Any]:
        entry = {"type": "image", "data": data, "original_type": "image"}
        if video_block:
            entry["minimax_video_block"] = True
        return entry

    def tokenize_with_weights(
        self,
        text: str,
        return_word_ids: bool = False,
        images: list = [],
        minimax_ref_items=None,
        **_kwargs,
    ) -> dict[str, list[list[tuple]]]:
        entries = []

        def add_text(value: str) -> None:
            entries.extend(self._text_entries(value))

        def add_vision(data: torch.Tensor, video_block: bool = False) -> None:
            entries.append((MINIMAX_H3_VISION_START, 1.0))
            entries.append((self._vision_entry(data, video_block), 1.0))
            entries.append((MINIMAX_H3_VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                if kind not in counters:
                    raise ValueError(f"Unsupported MiniMax H3 reference type: {kind}.")
                counters[kind] += 1
                if kind == "image":
                    add_text(f"<Picture {counters[kind]}>: ")
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text(f"<Audio {counters[kind]}>: ")
                else:
                    frames = item["data"]
                    timestamps = item.get("timestamps")
                    if timestamps is None:
                        timestamps = [index / 2.0 for index in range(frames.shape[0])]
                    else:
                        timestamps = list(timestamps)
                    if frames.shape[0] % 2:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps.append(timestamps[-1])
                    add_text(f"<Video {counters[kind]}>: ")
                    for index in range(0, frames.shape[0], 2):
                        midpoint = (timestamps[index] + timestamps[index + 1]) / 2.0
                        add_text(f"<{float(midpoint):.1f} seconds>")
                        add_vision(frames[index:index + 2], video_block=True)
        else:
            for index, image in enumerate(images, start=1):
                add_text(f"<Picture {index}>: ")
                add_vision(image)
        add_text(text)
        if not entries:
            entries.append((MINIMAX_H3_PAD_TOKEN, 1.0))
        if return_word_ids:
            entries = [entry + (0,) for entry in entries]
        return {MINIMAX_H3_PROJECTED_KEY: [entries]}


class MiniMaxH3ProjectedCLIP:
    def __init__(
        self,
        base: Any,
        projection_name: str,
        projection_model: MiniMaxH3ProjectionModel | None = None,
        projection_patcher: Any = None,
        original_methods: tuple[Callable, Callable] | None = None,
    ):
        source_key = getattr(base.cond_stage_model, "clip_name", None)
        if source_key not in MINIMAX_H3_SOURCE_KEYS:
            raise ValueError("MiniMax H3 projection requires a Qwen3-VL 4B or 8B text encoder.")
        source_model = getattr(
            base.cond_stage_model,
            getattr(base.cond_stage_model, "clip", ""),
            None,
        )
        if type(source_model).__module__ != "comfy.text_encoders.qwen3vl":
            raise ValueError(
                "MiniMax H3 projection requires the generic Qwen3-VL wrapper. "
                "Load the 4B or 8B encoder with CLIP type minimax."
            )
        if projection_model is None:
            data, tap = load_minimax_h3_projection(projection_name)
            projection_model = MiniMaxH3ProjectionModel(data, tap)
        self._base = base
        self._projection_name = projection_name
        self._projection_model = projection_model
        self._source_key = source_key
        self._capture = {}
        self.tokenizer = _MiniMaxH3ProjectedTokenizer(base.tokenizer, source_key)
        self._base.clip_layer(projection_model.tap)

        if projection_patcher is None:
            offload_device = comfy.model_management.text_encoder_offload_device()
            projection_model.to(offload_device)
            projection_model.device = offload_device
            projection_patcher = comfy.model_patcher.CoreModelPatcher(
                projection_model,
                load_device=base.patcher.load_device,
                offload_device=offload_device,
            )
        self._projection_patcher = projection_patcher
        self._install_encoder_patches(original_methods)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _install_encoder_patches(
        self, original_methods: tuple[Callable, Callable] | None
    ) -> None:
        owner_name = self._base.cond_stage_model.clip
        submodel = getattr(self._base.cond_stage_model, owner_name)
        transformer = submodel.transformer
        if original_methods is None:
            original_preprocess = transformer.preprocess_embed
            original_forward = transformer.forward
        else:
            original_preprocess, original_forward = original_methods
        self._original_methods = (original_preprocess, original_forward)

        def preprocess_embed(current, embed, device):
            if embed.get("type") == "image" and embed.get("minimax_video_block", False):
                flattened, grid = process_video_block(embed["data"])
                merged, deepstack = current.visual(
                    flattened.to(device, dtype=torch.float32), grid
                )
                return merged, {"grid": grid, "deepstack": deepstack}
            return original_preprocess(embed, device)

        def forward(_current, *args, **kwargs):
            if self._capture.get("active"):
                embeds_info = kwargs.get("embeds_info", [])
                embeds = kwargs.get("embeds")
                if embeds is not None:
                    self._capture["tags"] = token_tags_from_embeds_info(
                        embeds.shape[1], embeds_info
                    )
            return original_forward(*args, **kwargs)

        prefix = f"{owner_name}.transformer"
        self._base.patcher.add_object_patch(
            f"{prefix}.preprocess_embed",
            types.MethodType(preprocess_embed, transformer),
        )
        self._base.patcher.add_object_patch(
            f"{prefix}.forward",
            types.MethodType(forward, transformer),
        )

    def clone(self):
        return MiniMaxH3ProjectedCLIP(
            self._base.clone(),
            self._projection_name,
            projection_model=self._projection_model,
            projection_patcher=self._projection_patcher.clone(),
            original_methods=self._original_methods,
        )

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids, **kwargs)

    def _source_tokens(self, tokens: dict) -> dict:
        try:
            batches = tokens[MINIMAX_H3_PROJECTED_KEY]
        except (KeyError, TypeError) as exc:
            raise ValueError("Projected MiniMax H3 CLIP requires qwen3vl_32b tokens.") from exc
        return {self._source_key: batches}

    def _capture_encode(self, function: Callable, tokens: dict):
        self._capture.clear()
        self._capture["active"] = True
        try:
            output = function(self._source_tokens(tokens))
            tags = self._capture.get("tags")
        finally:
            self._capture.clear()
        if not torch.is_tensor(tags):
            raise RuntimeError("Projected MiniMax H3 encoder did not produce modality tags.")
        return output, tags

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        comfy.model_management.load_models_gpu([self._projection_patcher])
        device = self._projection_patcher.load_device
        with comfy.model_management.cuda_device_context(device):
            projected = self._projection_model(hidden.to(device))
        return projected.to(comfy.model_management.intermediate_device())

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        output, tags = self._capture_encode(
            lambda source: self._base.encode_from_tokens(
                source, return_pooled=True, return_dict=True
            ),
            tokens,
        )
        output["cond"] = self._project(output["cond"])
        output["minimax_token_tags"] = tags
        if return_dict:
            return output
        pooled = output.get("pooled_output")
        if return_pooled:
            return output["cond"], pooled
        return output["cond"]

    def encode_from_tokens_scheduled(
        self, tokens, unprojected=False, add_dict={}, show_pbar=True
    ):
        conditioning, tags = self._capture_encode(
            lambda source: self._base.encode_from_tokens_scheduled(
                source,
                unprojected=unprojected,
                add_dict=add_dict,
                show_pbar=show_pbar,
            ),
            tokens,
        )
        projected = []
        for hidden, metadata in conditioning:
            metadata = metadata.copy()
            metadata["minimax_token_tags"] = tags
            projected.append([self._project(hidden), metadata])
        return projected


def patch_minimax_h3_clip_projection(clip: Any, projection_name: str) -> MiniMaxH3ProjectedCLIP:
    return MiniMaxH3ProjectedCLIP(clip.clone(), projection_name)


PDD_VIDEO_SHIFT = 12.0
PDD_AUDIO_SHIFT = 3.0
PDD_KNOT_TOLERANCE = 1e-4
PDD_DEFAULT_PARTITIONS = {
    5: (8, 8, 8, 4, 4),
    6: (8, 8, 4, 4, 4, 4),
    7: (8, 4, 4, 4, 4, 4, 4),
}
PDD_HEAD_KEYS = ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")
PDD_WRAPPER_KEY = "utilscollection_minimax_h3_pdd_acc"
PDD_ADDITIONAL_MODEL_KEY = "utilscollection_minimax_h3_pdd_heads"
PDD_FINAL_FORWARD_PATH = "diffusion_model.final_layer.forward"
PDD_BASIS_DIR = Path(__file__).resolve().parent / "assets" / "minimax_h3_pdd"


def shifted_pdd_sigma(shift: float, t):
    return shift * t / (1.0 + (shift - 1.0) * t)


def pdd_fine_sigmas(shift: float, num_steps: int) -> tuple[float, ...]:
    times = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    values = shifted_pdd_sigma(shift, times)
    values[0] = 1.0
    values[-1] = 0.0
    return tuple(float(value) for value in values)


def resolve_pdd_partition(num_steps: int, nfe: int, partition_text: str = "", trained_block: int = 4) -> tuple[int, ...]:
    text = (partition_text or "").strip()
    if text:
        try:
            sizes = tuple(int(part) for part in text.replace(" ", "").split(",") if part)
        except ValueError as exc:
            raise ValueError(f"partition '{partition_text}' is not a comma-separated integer list") from exc
        if any(size < 1 for size in sizes) or sum(sizes) != num_steps:
            raise ValueError(f"partition {sizes} must contain positive block sizes summing to {num_steps}; received {sum(sizes)}")
    elif num_steps % nfe == 0:
        sizes = (num_steps // nfe,) * nfe
    elif nfe in PDD_DEFAULT_PARTITIONS and sum(PDD_DEFAULT_PARTITIONS[nfe]) == num_steps:
        sizes = PDD_DEFAULT_PARTITIONS[nfe]
    else:
        raise ValueError(f"nfe {nfe} does not divide the {num_steps}-step PDD grid and has no default partition")
    allowed = (trained_block, 2 * trained_block)
    invalid = sorted({size for size in sizes if size not in allowed})
    if invalid:
        raise ValueError(f"partition {sizes} contains block sizes {invalid} outside the trained envelope {allowed}")
    return sizes


def pdd_partition_starts(sizes: tuple[int, ...]) -> tuple[int, ...]:
    starts = []
    offset = 0
    for size in sizes:
        starts.append(offset)
        offset += size
    return tuple(starts)


def pdd_block_boundaries(num_steps: int, sizes: tuple[int, ...]) -> tuple[float, ...]:
    fine = pdd_fine_sigmas(PDD_VIDEO_SHIFT, num_steps)
    return tuple(fine[start] for start in pdd_partition_starts(sizes)) + (0.0,)


def fuse_pdd_heads(bank_weight: torch.Tensor, bank_bias: torch.Tensor, fine: tuple[float, ...], sizes: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    intervals = [fine[index] - fine[index + 1] for index in range(bank_weight.shape[0])]
    weights = bank_weight.to(torch.float64)
    biases = bank_bias.to(torch.float64)
    fused_weights = []
    fused_biases = []
    for start, size in zip(pdd_partition_starts(sizes), sizes):
        indices = range(start, start + size)
        span = sum(intervals[index] for index in indices)
        fused_weights.append(sum((intervals[index] / span) * weights[index] for index in indices).to(torch.float32))
        fused_biases.append(sum((intervals[index] / span) * biases[index] for index in indices).to(torch.float32))
    return torch.stack(fused_weights).contiguous(), torch.stack(fused_biases).contiguous()


def select_pdd_block(sigma: float, bounds: tuple[float, ...], on_off_grid: str) -> int:
    count = len(bounds) - 1
    for index in range(count):
        if abs(sigma - bounds[index]) <= PDD_KNOT_TOLERANCE:
            return index
    if abs(sigma - bounds[-1]) <= PDD_KNOT_TOLERANCE:
        return count - 1
    if on_off_grid == "error":
        expected = ", ".join(f"{value:.6f}" for value in bounds)
        raise ValueError(f"MiniMax H3 PDD evaluated at sigma {sigma:.6f}, outside trained boundaries [{expected}]. Use this node's SIGMAS output with a sampler that evaluates only those boundaries.")
    if sigma >= bounds[0]:
        return 0
    for index in range(count):
        if sigma > bounds[index + 1]:
            return index
    return count - 1


def convert_pdd_lora(state_dict: dict[str, torch.Tensor], alpha: float) -> tuple[dict[str, torch.Tensor], set[str]]:
    converted = {}
    consumed = set()

    def take(key):
        consumed.add(key)
        return state_dict[key]

    def emit(destination, down, up, alpha_value):
        converted[f"{destination}.lora_A.weight"] = down.contiguous()
        converted[f"{destination}.lora_B.weight"] = up.contiguous()
        converted[f"{destination}.alpha"] = torch.tensor(float(alpha_value))

    def convert_qkv(source, destination):
        parts = [(take(f"{source}.attn.to_{name}.lora_down"), take(f"{source}.attn.to_{name}.lora_up")) for name in ("q", "k", "v")]
        rank = parts[0][0].shape[0]
        output = parts[0][1].shape[0]
        down = torch.cat([part[0] for part in parts], dim=0)
        up = torch.zeros(output * 3, rank * 3, dtype=parts[0][1].dtype)
        for index, (_, value) in enumerate(parts):
            up[index * output:(index + 1) * output, index * rank:(index + 1) * rank] = value
        emit(f"{destination}.attn.qkv_proj", down, up, alpha * 3.0)

    def convert_linear(source, destination, half_swap=False):
        down = take(f"{source}.lora_down")
        up = take(f"{source}.lora_up")
        if half_swap:
            midpoint = up.shape[0] // 2
            up = torch.cat((up[midpoint:], up[:midpoint]), dim=0)
        emit(destination, down, up, alpha)

    trunk_blocks = sorted({int(match.group(1)) for key in state_dict if (match := re.match(r"transformer_blocks\.(\d+)\.", key))})
    refiner_blocks = sorted({int(match.group(1)) for key in state_dict if (match := re.match(r"token_refiner\.refiner_blocks\.(\d+)\.", key))})
    for index in trunk_blocks:
        source = f"transformer_blocks.{index}"
        destination = f"diffusion_model.blocks.{index}"
        convert_qkv(source, destination)
        convert_linear(f"{source}.attn.to_out.0", f"{destination}.attn.out_proj")
        convert_linear(f"{source}.ff.net.0.proj", f"{destination}.mlp.fc1", True)
        convert_linear(f"{source}.ff.net.2", f"{destination}.mlp.fc2")
        if f"{source}.adaln_proj.linear.lora_down" in state_dict:
            convert_linear(f"{source}.adaln_proj.linear", f"{destination}.adaln_proj.linear")
    for index in refiner_blocks:
        source = f"token_refiner.refiner_blocks.{index}"
        destination = f"diffusion_model.token_refiner.blocks.{index}"
        convert_qkv(source, destination)
        convert_linear(f"{source}.attn.to_out.0", f"{destination}.attn.out_proj")
        convert_linear(f"{source}.ff.net.0.proj", f"{destination}.mlp.fc1", True)
        convert_linear(f"{source}.ff.net.2", f"{destination}.mlp.fc2")
    return converted, set(state_dict) - consumed


def split_pdd_state_dict(state_dict: dict[str, torch.Tensor], metadata: dict[str, str] | None, filename: str):
    metadata = metadata or {}
    config = {"num_steps": int(metadata.get("pdd_num_steps", 32)), "block_size": int(metadata.get("pdd_block_size", 4)), "alpha": float(metadata.get("lora_alpha", 64.0))}
    missing = [key for key in PDD_HEAD_KEYS if key not in state_dict]
    if missing:
        raise ValueError(f"{filename} is missing PDD head tensors: {missing}")
    heads = tuple(state_dict.pop(key) for key in PDD_HEAD_KEYS)
    video_weight, video_bias, audio_weight, audio_bias = heads
    steps = config["num_steps"]
    valid = video_weight.ndim == 3 and video_bias.shape == video_weight.shape[:2] and audio_weight.ndim == 3 and audio_bias.shape == audio_weight.shape[:2] and video_weight.shape[0] == steps and audio_weight.shape[0] == steps
    if not valid:
        raise ValueError(f"{filename} has invalid PDD head shapes: video {list(video_weight.shape)}, video bias {list(video_bias.shape)}, audio {list(audio_weight.shape)}, audio bias {list(audio_bias.shape)}")
    if any(key.startswith("diffusion_model.") for key in state_dict):
        invalid = [key for key in state_dict if not (key.startswith("diffusion_model.") and key.endswith((".lora_A.weight", ".lora_B.weight", ".alpha")))]
        if invalid:
            raise ValueError(f"{filename} contains unexpected converted keys: {invalid[:4]}")
        config["source_format"] = "converted"
        lora_state = dict(state_dict)
    else:
        config["source_format"] = "original"
        lora_state, leftovers = convert_pdd_lora(state_dict, config["alpha"])
        if leftovers:
            raise ValueError(f"{filename} contains unrecognized keys: {sorted(leftovers)[:4]}")
    return lora_state, heads, config


def pdd_table_sha(table: torch.Tensor) -> str:
    value = table.detach().to(torch.float32).contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()[:16]


def rebase_pdd_adaln(lora_state: dict[str, torch.Tensor], center: torch.Tensor, basis: torch.Tensor) -> tuple[dict[str, torch.Tensor], int]:
    rebased = dict(lora_state)
    center64 = center.to(torch.float64)
    basis64 = basis.to(torch.float64)
    count = 0
    for key in list(rebased):
        if not key.endswith(".adaln_proj.linear.lora_A.weight"):
            continue
        module = key[:-len(".lora_A.weight")]
        down = rebased.pop(f"{module}.lora_A.weight").to(torch.float64)
        up = rebased.pop(f"{module}.lora_B.weight").to(torch.float64)
        alpha = rebased.pop(f"{module}.alpha", None)
        scale = float(alpha) / down.shape[0] if alpha is not None else 1.0
        rebased[f"{module}.diff"] = (scale * (up @ (down @ basis64))).to(torch.float32).contiguous()
        rebased[f"{module}.diff_b"] = (scale * (up @ (down @ center64))).to(torch.float32).contiguous()
        count += 1
    return rebased, count


def rebase_pdd_for_pruned_model(model: Any, lora_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    diffusion_model = model.get_model_object("diffusion_model")
    if not diffusion_model.use_adaln_curves:
        return lora_state, ""
    if not any(key.endswith(".adaln_proj.linear.lora_A.weight") for key in lora_state):
        return lora_state, "pruned model without dense AdalN LoRA modules"
    table = diffusion_model.adaln_t_table.detach().to(torch.float32).cpu()
    candidates = []
    for path in sorted(PDD_BASIS_DIR.glob("basis_*.safetensors")):
        data = comfy.utils.load_torch_file(str(path), safe_load=True)
        candidates.append(path.name)
        basis_table = data["adaln_t_table"]
        if basis_table.shape == table.shape and torch.allclose(basis_table, table, atol=1e-6):
            trunk = path.stem.removeprefix("basis_")
            rebased, count = rebase_pdd_adaln(lora_state, data["c"], data["V"])
            return rebased, f"rebased {count} AdalN modules onto {trunk} curve basis"
    raise ValueError(f"MiniMax H3 PDD cannot match this pruned model's AdalN table ({list(table.shape)}, sha {pdd_table_sha(table)}) against {candidates}")


class MiniMaxH3PDDHeadBank(torch.nn.Module):
    def __init__(self, video_weight: torch.Tensor, video_bias: torch.Tensor, audio_weight: torch.Tensor, audio_bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("video_weight", video_weight)
        self.register_buffer("video_bias", video_bias)
        self.register_buffer("audio_weight", audio_weight)
        self.register_buffer("audio_bias", audio_bias)

    def project(self, video: torch.Tensor, audio: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.video_weight.device != video.device or self.audio_weight.device != audio.device:
            raise RuntimeError("MiniMax H3 PDD head bank was not loaded beside the diffusion model by Core.")
        return F.linear(video, self.video_weight[block], self.video_bias[block]), F.linear(audio, self.audio_weight[block], self.audio_bias[block])


class MiniMaxH3PDDExecutionState:
    def __init__(self) -> None:
        self.sigma = contextvars.ContextVar("utilscollection_minimax_h3_pdd_sigma", default=None)


def make_pdd_diffusion_wrapper(state: MiniMaxH3PDDExecutionState):
    def pdd_diffusion_wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options or {}
        diffusion_model = executor.class_obj
        video_shift = float(options.get("minimax_h3_sigma_shift_video", getattr(diffusion_model, "sigma_shift_video", PDD_VIDEO_SHIFT)))
        audio_shift = float(options.get("minimax_h3_sigma_shift_audio", getattr(diffusion_model, "sigma_shift_audio", PDD_AUDIO_SHIFT)))
        if not (math.isclose(video_shift, PDD_VIDEO_SHIFT, abs_tol=1e-6) and math.isclose(audio_shift, PDD_AUDIO_SHIFT, abs_tol=1e-6)):
            raise ValueError(f"MiniMax H3 PDD requires SigmaShift 12.0/3.0; received {video_shift}/{audio_shift}.")
        payload = kwargs.get("minimax_payload") or {}
        audio_scale = float(payload.get("audio_scale", PDD_VIDEO_SHIFT / PDD_AUDIO_SHIFT))
        if not math.isclose(audio_scale, PDD_VIDEO_SHIFT / PDD_AUDIO_SHIFT, abs_tol=1e-6):
            raise ValueError(f"MiniMax H3 PDD requires audio_scale {PDD_VIDEO_SHIFT / PDD_AUDIO_SHIFT}; received {audio_scale}.")
        token = state.sigma.set(float(timestep.flatten()[0]) / 1000.0)
        try:
            return executor(x, timestep, context, options, **kwargs)
        finally:
            state.sigma.reset(token)
    return pdd_diffusion_wrapper


def make_pdd_final_forward(final_layer: Any, head_bank: MiniMaxH3PDDHeadBank, state: MiniMaxH3PDDExecutionState, bounds: tuple[float, ...], on_off_grid: str, strength: float):
    def pdd_final_forward(self, x, t_emb, video_seg, audio_seg):
        sigma = state.sigma.get()
        if sigma is None:
            raise RuntimeError("MiniMax H3 PDD final layer ran outside its Core diffusion wrapper.")
        block = select_pdd_block(sigma, bounds, on_off_grid)
        shift, scale = self.adaln_proj(t_emb)

        def modulate(segment):
            start, stop, row = segment
            return (self.norm(x[start:stop]) * (1.0 + minimax_model._mod_row(scale, row, scale.dtype)) + minimax_model._mod_row(shift, row, shift.dtype)).to(torch.float32)

        video_hidden = modulate(video_seg)
        audio_hidden = modulate(audio_seg)
        video, audio = head_bank.project(video_hidden, audio_hidden, block)
        if strength != 1.0:
            native_video = self.video_out(video_hidden)
            native_audio = self.audio_out(audio_hidden)
            video = native_video + (video - native_video) * strength
            audio = native_audio + (audio - native_audio) * strength
        return video, audio
    return types.MethodType(pdd_final_forward, final_layer)


def patch_minimax_h3_pdd_model(model: Any, pdd_lora: str, nfe: int, partition: str, lora_strength: float, head_strength: float, on_off_grid: str) -> tuple[Any, torch.Tensor]:
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 PDD requires a MiniMax H3 diffusion model.")
    model_options = getattr(model, "model_options", {})
    if model_options.get(MINIMAX_H3_PDD_OWNER_KEY):
        raise ValueError("MiniMax H3 PDD is already applied to this model.")
    if model_options.get(MINIMAX_H3_SPECTRUM_OWNER_KEY):
        raise ValueError("MiniMax H3 PDD cannot be combined with MiniMax H3 Spectrum.")
    if PDD_FINAL_FORWARD_PATH in getattr(model, "object_patches", {}):
        raise ValueError("MiniMax H3 PDD will not replace an existing final-layer object patch.")

    path = folder_paths.get_full_path_or_raise("loras", pdd_lora)
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    lora_state, heads, config = split_pdd_state_dict(state_dict, metadata, pdd_lora)
    sizes = resolve_pdd_partition(config["num_steps"], int(nfe), partition, config["block_size"])
    lora_state, curve_note = rebase_pdd_for_pruned_model(model, lora_state)
    expected_keys = sum(key.endswith((".lora_A.weight", ".diff", ".diff_b")) for key in lora_state)
    loaded = comfy.lora.load_lora(lora_state, comfy.lora.model_lora_keys_unet(model.model, {}), log_missing=False)

    patched = model.clone()
    applied = patched.add_patches(loaded, lora_strength)
    if len(applied) != expected_keys:
        raise ValueError(f"MiniMax H3 PDD matched {len(applied)}/{expected_keys} patch keys; verify that PDD and diffusion-model trunks match.")
    final_layer = patched.get_model_object("diffusion_model.final_layer")
    video_weight, video_bias, audio_weight, audio_bias = heads
    if tuple(final_layer.video_out.weight.shape) != tuple(video_weight.shape[1:]):
        raise ValueError("PDD video head shape does not match this MiniMax H3 model.")
    if tuple(final_layer.audio_out.weight.shape) != tuple(audio_weight.shape[1:]):
        raise ValueError("PDD audio head shape does not match this MiniMax H3 model.")

    video_fine = pdd_fine_sigmas(PDD_VIDEO_SHIFT, config["num_steps"])
    audio_fine = pdd_fine_sigmas(PDD_AUDIO_SHIFT, config["num_steps"])
    fused_video_weight, fused_video_bias = fuse_pdd_heads(video_weight, video_bias, video_fine, sizes)
    fused_audio_weight, fused_audio_bias = fuse_pdd_heads(audio_weight, audio_bias, audio_fine, sizes)
    head_bank = MiniMaxH3PDDHeadBank(fused_video_weight, fused_video_bias, fused_audio_weight, fused_audio_bias)
    head_patcher = comfy.model_patcher.ModelPatcher(head_bank, load_device=patched.load_device, offload_device=patched.offload_device)
    patched.set_additional_models(PDD_ADDITIONAL_MODEL_KEY, [head_patcher])

    bounds = pdd_block_boundaries(config["num_steps"], sizes)
    execution_state = MiniMaxH3PDDExecutionState()
    patched.add_object_patch(PDD_FINAL_FORWARD_PATH, make_pdd_final_forward(final_layer, head_bank, execution_state, bounds, on_off_grid, head_strength))
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    patched.remove_wrappers_with_key(wrapper_type, PDD_WRAPPER_KEY)
    patched.add_wrapper_with_key(wrapper_type, PDD_WRAPPER_KEY, make_pdd_diffusion_wrapper(execution_state))
    patched.model_options[MINIMAX_H3_PDD_OWNER_KEY] = True
    if curve_note:
        LOG.info("MiniMax H3 PDD: %s", curve_note)
    return patched, torch.tensor(bounds, dtype=torch.float32)


# Cache heuristic adapted from ComfyUI-MiniMaxH3-Cache by lihaoyun6:
# https://github.com/lihaoyun6/ComfyUI-MiniMaxH3-Cache (GPL-3.0).
class MiniMaxH3Cache:
    """Reuse the residual produced by the complete MiniMax H3 block stack."""

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_steps: int,
        device: str,
        verbose: bool,
    ) -> None:
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_steps = max_steps
        self.device = device
        self.verbose = verbose
        self.total_steps = 1
        self.reset()

    def reset(self) -> None:
        self.cached_residual: torch.Tensor | None = None
        self.previous_feature_signature: torch.Tensor | None = None
        self.layout_signature: tuple[Any, ...] | None = None
        self.last_seen_timestep: float | None = None
        self.step_counter = 0
        self.accumulated_relative_l1 = 0.0
        self.consecutive_skips = 0
        self.run_count = 0
        self.skip_count = 0

    def begin(self, total_steps: int) -> None:
        self.reset()
        self.total_steps = max(1, total_steps)

    def finish(self) -> None:
        if self.verbose and self.run_count + self.skip_count:
            total = self.run_count + self.skip_count
            speedup = total / max(1, self.run_count)
            logging.info(
                "[UtilsCollection MiniMax H3 Cache] Skipped %s/%s block-stack "
                "executions (%.2fx theoretical block-stack speedup).",
                self.skip_count,
                total,
                speedup,
            )
        self.reset()

    @staticmethod
    def _feature_signature(
        hidden_states: torch.Tensor,
        cache_ranges: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        max_dim = min(64, hidden_states.shape[-1])
        signatures = []
        for start, end in cache_ranges:
            length = end - start
            if length <= 0:
                continue
            stride = max(1, length // 100)
            sampled = hidden_states[start:end:stride, :max_dim]
            signatures.append(sampled.detach().abs().mean(dim=-1))

        if not signatures:
            stride = max(1, hidden_states.shape[0] // 100)
            sampled = hidden_states[::stride, :max_dim]
            return sampled.detach().abs().mean(dim=-1).clone()
        return torch.cat(signatures).clone()

    @staticmethod
    def _timestep_value(timestep: Any) -> float | None:
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                return None
            return float(timestep.detach().flatten()[0].item())
        if isinstance(timestep, (int, float)):
            return float(timestep)
        return None

    def _store_residual(self, residual: torch.Tensor) -> None:
        if self.device == "cuda" and residual.device.type != "cuda":
            raise ValueError(
                "MiniMax H3 cache device is set to cuda, but the model is not running on CUDA."
            )

        try:
            if self.device == "cpu":
                self.cached_residual = residual.detach().to("cpu", copy=True)
            else:
                self.cached_residual = residual.detach().clone()
        except torch.OutOfMemoryError:
            if self.device == "cuda":
                raise
            self.cached_residual = residual.detach().to("cpu", copy=True)

    def _apply_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.cached_residual
        if residual is None:
            return hidden_states
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=True,
            )
        return hidden_states + residual

    def __call__(self, args: dict[str, Any], extra_options: dict[str, Any]) -> dict[str, Any]:
        original_block = extra_options["original_block"]
        hidden_states = args["img"]
        cache_ranges = tuple(tuple(pair) for pair in args.get("cache_ranges", ()))
        current_layout = (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
            args.get("block_count"),
            cache_ranges,
        )

        if self.layout_signature is None:
            self.layout_signature = current_layout
        elif self.layout_signature != current_layout:
            total_steps = self.total_steps
            self.reset()
            self.total_steps = total_steps
            self.layout_signature = current_layout

        timestep = self._timestep_value(args.get("timestep"))
        if timestep is None:
            return original_block(args)
        if self.last_seen_timestep != timestep:
            self.last_seen_timestep = timestep
            self.step_counter += 1

        progress = self.step_counter / self.total_steps
        in_cache_range = self.start_percent <= progress <= self.end_percent
        skip_reason = "initial step"

        if self.cached_residual is not None and self.previous_feature_signature is not None:
            current_signature = self._feature_signature(hidden_states, cache_ranges)
            current_float = current_signature.float()
            previous_float = self.previous_feature_signature.float()
            difference = (current_float - previous_float).abs().mean().item()
            denominator = previous_float.abs().mean().item() + 1e-6
            self.accumulated_relative_l1 += difference / denominator

            below_threshold = self.accumulated_relative_l1 < self.reuse_threshold
            below_skip_limit = self.consecutive_skips < self.max_steps
            if below_threshold and below_skip_limit and in_cache_range:
                self.skip_count += 1
                self.consecutive_skips += 1
                if self.verbose:
                    logging.info(
                        "[UtilsCollection MiniMax H3 Cache] Step %s SKIP "
                        "(relative L1 %.4f < %.4f).",
                        self.step_counter,
                        self.accumulated_relative_l1,
                        self.reuse_threshold,
                    )
                return {"img": self._apply_residual(hidden_states)}

            reasons = []
            if not below_threshold:
                reasons.append("threshold reached")
            if not below_skip_limit:
                reasons.append("maximum consecutive skips reached")
            if not in_cache_range:
                reasons.append("outside cache range")
            skip_reason = ", ".join(reasons)

        if self.verbose:
            logging.info(
                "[UtilsCollection MiniMax H3 Cache] Step %s RUN (%s).",
                self.step_counter,
                skip_reason,
            )

        self.run_count += 1
        self.consecutive_skips = 0
        self.cached_residual = None
        self.previous_feature_signature = self._feature_signature(hidden_states, cache_ranges)
        start_hidden_states = hidden_states.clone()
        result = original_block(args)
        output = result["img"]
        self._store_residual(output - start_hidden_states)
        self.accumulated_relative_l1 = 0.0
        return result


class MiniMaxH3SamplingScope:
    """Give one cache object an exact outer-sampling lifecycle."""

    def __init__(self, cache: MiniMaxH3Cache) -> None:
        self.cache = cache

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            self.cache.reset()
            raise ValueError("MiniMax H3 cache could not read the sampler sigma schedule.")

        self.cache.begin(len(sigmas) - 1)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


def run_minimax_h3_blocks(
    model: minimax_model.MiniMaxH3Model,
    hidden_states: torch.Tensor,
    timestep_embedding: torch.Tensor,
    mod_segments: list[tuple[int, int, int]],
    rope_freqs: torch.Tensor,
    transformer_options: dict[str, Any],
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Run a bounded H3 block range while preserving Core block replacements."""

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    end = len(model.blocks) if end is None else end
    blocks = list(model.blocks[start:end])
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        blocks, hidden_states.device, transformer_options
    )
    for index in range(start, end):
        block = model.blocks[index]
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, hidden_states.device, block)
        if ("double_block", index) in blocks_replace:

            def block_wrapper(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
                return {
                    "img": block(
                        block_args["img"],
                        block_args["t_emb"],
                        block_args["mod_segments"],
                        block_args["rope_freqs"],
                        transformer_options=block_args["transformer_options"],
                    )
                }

            hidden_states = blocks_replace[("double_block", index)](
                {
                    "img": hidden_states,
                    "t_emb": timestep_embedding,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrapper},
            )["img"]
        else:
            hidden_states = block(
                hidden_states,
                timestep_embedding,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue, hidden_states.device, None
        )
    return hidden_states


def minimax_h3_block_patch_forward(
    self: minimax_model.MiniMaxH3Model,
    x: list[torch.Tensor],
    timestep: torch.Tensor,
    context: torch.Tensor,
    transformer_options: dict[str, Any] = {},
    minimax_payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Current Core H3 forward with one model-patchable block-loop boundary."""

    del kwargs
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2:5]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2:5]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = minimax_model.PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )

    shift_v = float(
        transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)
    )
    shift_a = float(
        transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio)
    )
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(
        payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP)
    )
    aud_aug = float(
        payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP)
    )
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted(
        {t_v, t_a}
        | ({seg_t["cond"]} if has_vis_cond else set())
        | ({seg_t["ref_audio"]} if has_aud_cond else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "ref_audio": 2,
    }

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, end - start + 1):
                if index == end - start or tags[index] != tags[run_start]:
                    mod_segments.append(
                        (start + run_start, start + index, row_base + int(tags[run_start]))
                    )
                    run_start = index
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(
            self.condition_proj(text_states), transformer_options=transformer_options
        )

    hidden_states = torch.empty(
        layout.seq_len, self.hidden_size, dtype=dtype, device=device
    )
    video_offset = audio_offset = 0
    for start, end, kind in layout.segments:
        length = end - start
        if kind == "text":
            hidden_states[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden_states[start:end] = video_embed[video_offset : video_offset + length]
            video_offset += length
        else:
            hidden_states[start:end] = audio_embed[audio_offset : audio_offset + length]
            audio_offset += length

    t_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        position = t_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        timestep_embedding = torch.lerp(
            table[lower], table[lower + 1], (position - lower).unsqueeze(1)
        )
    else:
        timestep_embedding = self.time_embedder(t_values).to(dtype)

    rope_freqs = minimax_model.rope_rotation_table(
        self.rope_freqs(layout.position_ids, device), dtype
    )
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    cache_ranges = tuple(
        (start, end)
        for start, end, kind in layout.segments
        if kind in ("audio", "video")
    )
    if ("block_loop", 0) in blocks_replace:

        def block_loop_wrapper(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
            return {
                "img": run_minimax_h3_blocks(
                    self,
                    block_args["img"],
                    block_args["t_emb"],
                    block_args["mod_segments"],
                    block_args["rope_freqs"],
                    block_args["transformer_options"],
                    block_args.get("start", 0),
                    block_args.get("end"),
                )
            }

        hidden_states = blocks_replace[("block_loop", 0)](
            {
                "img": hidden_states,
                "timestep": timestep,
                "t_emb": timestep_embedding,
                "mod_segments": mod_segments,
                "rope_freqs": rope_freqs,
                "transformer_options": transformer_options,
                "cache_ranges": cache_ranges,
                "target_ranges": tuple(
                    (start, end, kind)
                    for start, end, kind in layout.segments
                    if kind in ("audio", "video")
                ),
                "block_count": len(self.blocks),
            },
            {"original_block": block_loop_wrapper},
        )["img"]
    else:
        hidden_states = run_minimax_h3_blocks(
            self,
            hidden_states,
            timestep_embedding,
            mod_segments,
            rope_freqs,
            transformer_options,
        )

    video_seg = next(
        (start, end, t_row[seg_t["video"]])
        for start, end, kind in layout.segments
        if kind == "video"
    )
    audio_seg = next(
        (start, end, t_row[seg_t["audio"]])
        for start, end, kind in layout.segments
        if kind == "audio"
    )
    video_result, audio_result = self.final_layer(
        hidden_states, timestep_embedding, video_seg, audio_seg
    )
    video_out = minimax_model.unpatchify_video(
        video_result,
        latent_t,
        lat_h // 2,
        lat_w // 2,
        self.latents_dim,
        self.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio_result)
    return [
        -video_out.to(video_x.dtype),
        -audio_out.to(audio_x.dtype),
    ]


def patch_minimax_h3_cache_model(
    model: Any,
    reuse_threshold: float,
    start_percent: float,
    end_percent: float,
    max_steps: int,
    device: str,
    verbose: bool,
) -> Any:
    """Return a cloned patcher with a reversible H3 block-loop cache."""

    if start_percent > end_percent:
        raise ValueError("Cache start percent must not exceed end percent.")

    if getattr(model, "model_options", {}).get(MINIMAX_H3_SPECTRUM_OWNER_KEY):
        raise ValueError("MiniMax H3 Cache cannot be combined with MiniMax H3 Spectrum.")
    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError(
            "MiniMax H3 Cache requires a MiniMax H3 diffusion model; "
            f"received {diffusion_model.__class__.__name__}."
        )

    cache = MiniMaxH3Cache(
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        max_steps=max_steps,
        device=device,
        verbose=verbose,
    )
    if hasattr(patched_model, "model_options"):
        patched_model.model_options[MINIMAX_H3_CACHE_OWNER_KEY] = True
    bound_forward = types.MethodType(minimax_h3_block_patch_forward, diffusion_model)
    patched_model.add_object_patch("diffusion_model._forward", bound_forward)
    patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    patched_model.add_wrapper(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        MiniMaxH3SamplingScope(cache),
    )
    return patched_model

# Spectrum runtime adapted from xmarre/ComfyUI-Spectrum-MiniMax-H3
# revision 1a8930d662f4f66694d06275bff40c002e0d451d (GPL-3.0-or-later).

@dataclass(frozen=True, slots=True)
class SpectrumH3Config:
    enabled: bool = True
    blend_weight: float = 0.50
    degree: int = 1
    ridge_lambda: float = 0.10
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 1
    tail_actual_steps: int = 1
    max_history: int = 8
    history_storage: str = "system_ram"
    debug: bool = False
    force_actual: bool = False
    bootstrap_first_forecast: bool = True
    anchor_residual_feedback: bool = False
    selective_rollback_correction: bool = False
    offline_smoothing_replay: bool = True
    audio_blend_weight: float = 0.0
    offline_archive_storage: str = "system_ram"

    def __post_init__(self) -> None:
        trajectory_modes = {
            "anchor_residual_feedback": self.anchor_residual_feedback,
            "selective_rollback_correction": self.selective_rollback_correction,
            "offline_smoothing_replay": self.offline_smoothing_replay,
        }
        for name, value in trajectory_modes.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        conflicts = [name for name, value in trajectory_modes.items() if value]
        if self.enabled and len(conflicts) > 1:
            raise ValueError(
                "Spectrum H3 trajectory modes are mutually exclusive; conflicting settings: "
                + ", ".join(conflicts)
            )

    @property
    def min_fit_points(self) -> int:
        return max(2, self.degree + 1)

    def validate(self) -> SpectrumH3Config:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean")
        if not isinstance(self.debug, bool):
            raise TypeError("debug must be a boolean")
        if not isinstance(self.force_actual, bool):
            raise TypeError("force_actual must be a boolean")
        if not isinstance(self.bootstrap_first_forecast, bool):
            raise TypeError("bootstrap_first_forecast must be a boolean")
        for name in (
            "anchor_residual_feedback",
            "selective_rollback_correction",
            "offline_smoothing_replay",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        conflicts = [
            name
            for name in (
                "anchor_residual_feedback",
                "selective_rollback_correction",
                "offline_smoothing_replay",
            )
            if getattr(self, name)
        ]
        if self.enabled and len(conflicts) > 1:
            raise ValueError(
                "Spectrum H3 trajectory modes are mutually exclusive; conflicting settings: "
                + ", ".join(conflicts)
            )
        if not math.isfinite(self.blend_weight) or not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be finite and in [0, 1]")
        if not math.isfinite(self.audio_blend_weight) or not 0.0 <= self.audio_blend_weight <= 1.0:
            raise ValueError("audio_blend_weight must be finite and in [0, 1]")
        if isinstance(self.degree, bool) or not isinstance(self.degree, int) or self.degree < 1:
            raise ValueError("degree must be an integer >= 1")
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be finite and >= 0")
        if not math.isfinite(self.window_size) or self.window_size < 1.0:
            raise ValueError("window_size must be finite and >= 1")
        if not math.isfinite(self.flex_window) or self.flex_window < 0.0:
            raise ValueError("flex_window must be finite and >= 0")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
        ):
            raise ValueError("warmup_steps must be an integer >= 0")
        if self.bootstrap_first_forecast and self.degree != 1:
            raise ValueError("bootstrap_first_forecast requires degree == 1")
        if self.bootstrap_first_forecast and self.warmup_steps > 1:
            raise ValueError("bootstrap_first_forecast requires warmup_steps <= 1")
        if (
            isinstance(self.tail_actual_steps, bool)
            or not isinstance(self.tail_actual_steps, int)
            or self.tail_actual_steps < 0
        ):
            raise ValueError("tail_actual_steps must be an integer >= 0")
        if (
            isinstance(self.max_history, bool)
            or not isinstance(self.max_history, int)
            or self.max_history < self.min_fit_points
        ):
            raise ValueError(
                f"max_history must be an integer >= {self.min_fit_points} for degree {self.degree}"
            )
        if not isinstance(self.history_storage, str) or self.history_storage not in {
            "system_ram",
            "vram",
        }:
            raise ValueError("history_storage must be 'system_ram' or 'vram'")
        if not isinstance(self.offline_archive_storage, str) or self.offline_archive_storage not in {
            "system_ram",
            "vram",
        }:
            raise ValueError("offline_archive_storage must be 'system_ram' or 'vram'")
        return self


CONSERVATIVE_PRESET = SpectrumH3Config()
AGGRESSIVE_PRESET = SpectrumH3Config(
    blend_weight=0.75,
    degree=4,
    flex_window=3.0,
    warmup_steps=5,
    tail_actual_steps=1,
    bootstrap_first_forecast=False,
)


def effective_bootstrap_first_forecast(
    requested: bool, degree: int, warmup_steps: int
) -> bool:
    """Disable unsupported one-point bootstrap combinations with one warning."""

    if not requested or (degree == 1 and warmup_steps <= 1):
        return requested
    logging.warning(
        "[UtilsCollection Spectrum H3] First-forecast bootstrap requires "
        "degree 1 and warmup at most 1; disabling it for this node execution."
    )
    return False

@dataclass(slots=True)
class _HistoryEntry:
    coordinate: float
    feature_flat: torch.Tensor


@dataclass(slots=True)
class ForecasterSnapshot:
    history: list[_HistoryEntry]
    feature_shape: tuple[int, ...] | None
    feature_dtype: torch.dtype | None
    history_device: torch.device | None
    generation: int
    factor_generation: int
    design: torch.Tensor | None
    cholesky: torch.Tensor | None
    factorization_count: int
    jitter_attempts: int


class HistoryWeightForecaster:
    """Chebyshev ridge forecasting without full-feature FP32 coefficients.

    Persistent large tensors are model-dtype history snapshots on the configured
    storage device. Regression work is limited to K x (M + 1) design data and a
    (M + 1)^2 factorization.
    """

    def __init__(
        self,
        degree: int = 4,
        ridge_lambda: float = 0.1,
        max_history: int = 8,
        chunk_bytes: int = 32 * 1024 * 1024,
        history_storage: str = "system_ram",
    ) -> None:
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self.max_history = int(max_history)
        self.chunk_bytes = int(chunk_bytes)
        self.history_storage = str(history_storage)
        if self.degree < 1:
            raise ValueError("degree must be >= 1")
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.max_history < max(2, self.degree + 1):
            raise ValueError("max_history is too small for the requested polynomial degree")
        if self.chunk_bytes < 4096:
            raise ValueError("chunk_bytes must be >= 4096")
        if self.history_storage not in {"system_ram", "vram"}:
            raise ValueError("history_storage must be 'system_ram' or 'vram'")
        self.reset()

    def reset(self) -> None:
        self._history: list[_HistoryEntry] = []
        self._feature_shape: tuple[int, ...] | None = None
        self._feature_dtype: torch.dtype | None = None
        self._history_device: torch.device | None = None
        self._generation = 0
        self._factor_generation = -1
        self._design: torch.Tensor | None = None
        self._cholesky: torch.Tensor | None = None
        self._factorization_count = 0
        self._jitter_attempts = 0
        self.last_prediction_chunk_count = 0
        self.last_prediction_max_fp32_elements = 0

    def snapshot(self) -> ForecasterSnapshot:
        return ForecasterSnapshot(
            history=list(self._history),
            feature_shape=self._feature_shape,
            feature_dtype=self._feature_dtype,
            history_device=self._history_device,
            generation=self._generation,
            factor_generation=self._factor_generation,
            design=self._design,
            cholesky=self._cholesky,
            factorization_count=self._factorization_count,
            jitter_attempts=self._jitter_attempts,
        )

    def restore(self, snapshot: ForecasterSnapshot) -> None:
        if not isinstance(snapshot, ForecasterSnapshot):
            raise TypeError("snapshot must be a ForecasterSnapshot")
        self._history = list(snapshot.history)
        self._feature_shape = snapshot.feature_shape
        self._feature_dtype = snapshot.feature_dtype
        self._history_device = snapshot.history_device
        self._generation = snapshot.generation
        self._factor_generation = snapshot.factor_generation
        self._design = snapshot.design
        self._cholesky = snapshot.cholesky
        self._factorization_count = snapshot.factorization_count
        self._jitter_attempts = snapshot.jitter_attempts
        self.last_prediction_chunk_count = 0
        self.last_prediction_max_fp32_elements = 0

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def feature_shape(self) -> tuple[int, ...] | None:
        return self._feature_shape

    @property
    def feature_dtype(self) -> torch.dtype | None:
        return self._feature_dtype

    @property
    def history_device(self) -> torch.device | None:
        return self._history_device

    @property
    def factorization_count(self) -> int:
        return self._factorization_count

    @property
    def jitter_attempts(self) -> int:
        return self._jitter_attempts

    @property
    def persistent_tensor_bytes(self) -> int:
        tensors = [entry.feature_flat for entry in self._history]
        tensors.extend(t for t in (self._design, self._cholesky) if t is not None)
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def history_tensor_bytes(self) -> int:
        return sum(entry.feature_flat.numel() * entry.feature_flat.element_size() for entry in self._history)

    def ready(self, minimum: int | None = None) -> bool:
        required = max(2, self.degree + 1, int(minimum or 0))
        return len(self._history) >= required

    @staticmethod
    def chebyshev_design(coordinates: torch.Tensor, degree: int) -> torch.Tensor:
        x = coordinates.reshape(-1, 1).to(device="cpu", dtype=torch.float32)
        columns = [torch.ones_like(x)]
        if degree >= 1:
            columns.append(x)
        for _ in range(2, degree + 1):
            columns.append(2.0 * x * columns[-1] - columns[-2])
        return torch.cat(columns[: degree + 1], dim=1)

    def update(self, coordinate: float, feature: torch.Tensor, *, take_ownership: bool = False) -> None:
        if not torch.is_tensor(feature) or not feature.dtype.is_floating_point:
            raise ValueError("Spectrum history features must be floating-point tensors")
        shape = tuple(int(v) for v in feature.shape)
        if len(shape) < 2:
            raise ValueError("Spectrum history features must have a branch dimension and feature dimensions")
        if self._feature_shape is None:
            self._feature_shape = shape
            self._feature_dtype = feature.dtype
        elif shape != self._feature_shape:
            raise ValueError(f"feature shape changed from {self._feature_shape} to {shape}")
        elif feature.dtype != self._feature_dtype:
            raise ValueError(f"feature dtype changed from {self._feature_dtype} to {feature.dtype}")

        detached = feature.detach()
        storage_device = torch.device("cpu") if self.history_storage == "system_ram" else detached.device
        if self._history_device is None:
            self._history_device = storage_device
        elif storage_device != self._history_device:
            raise ValueError(f"history device changed from {self._history_device} to {storage_device}")

        if take_ownership and detached.device == storage_device and detached.is_contiguous():
            archived = detached.reshape(-1)
        else:
            archived = (
                detached.to(device=storage_device, dtype=self._feature_dtype, copy=True)
                .contiguous()
                .reshape(-1)
            )
        self._history.append(_HistoryEntry(float(coordinate), archived))
        if len(self._history) > self.max_history:
            self._history.pop(0)
        self._generation += 1
        self._design = None
        self._cholesky = None

    def _ensure_factorization(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.ready():
            raise RuntimeError("Spectrum forecaster does not have enough actual history")
        if (
            self._factor_generation == self._generation
            and self._design is not None
            and self._cholesky is not None
        ):
            return self._design, self._cholesky

        coords = torch.tensor([entry.coordinate for entry in self._history], dtype=torch.float32)
        design = self.chebyshev_design(coords, self.degree)
        gram = design.transpose(0, 1) @ design
        eye = torch.eye(gram.shape[0], dtype=torch.float32)
        base = gram + self.ridge_lambda * eye
        diagonal_scale = max(float(gram.diag().abs().mean().item()), 1.0)
        jitters = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)
        last_error: RuntimeError | None = None
        cholesky = None
        for attempt, multiplier in enumerate(jitters):
            try:
                cholesky = torch.linalg.cholesky(base + (diagonal_scale * multiplier) * eye)
                self._jitter_attempts += attempt
                break
            except RuntimeError as exc:
                last_error = exc
        if cholesky is None:
            raise RuntimeError("Spectrum ridge factorization failed after bounded jitter attempts") from last_error

        self._design = design
        self._cholesky = cholesky
        self._factor_generation = self._generation
        self._factorization_count += 1
        return design, cholesky

    def _spectral_weights(self, coordinate: float) -> torch.Tensor:
        design, cholesky = self._ensure_factorization()
        phi = self.chebyshev_design(torch.tensor([float(coordinate)]), self.degree)
        solved = torch.cholesky_solve(design.transpose(0, 1), cholesky)
        return (phi @ solved).reshape(-1)

    def spectral_weights(self, coordinate: float) -> torch.Tensor:
        return self._spectral_weights(coordinate)

    def _linear_weights(self, coordinate: float) -> torch.Tensor:
        weights = torch.zeros(len(self._history), dtype=torch.float32)
        if len(self._history) == 1:
            weights[-1] = 1.0
            return weights
        previous = self._history[-2].coordinate
        latest = self._history[-1].coordinate
        spacing = latest - previous
        if abs(spacing) <= 1e-12:
            weights[-1] = 1.0
            return weights
        ratio = (float(coordinate) - latest) / spacing
        weights[-2] = -ratio
        weights[-1] = 1.0 + ratio
        return weights

    def combined_weights(self, coordinate: float, blend_weight: float) -> torch.Tensor:
        blend = float(blend_weight)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if blend <= 1e-12:
            return self._linear_weights(coordinate)
        spectral = self._spectral_weights(coordinate)
        if blend >= 1.0 - 1e-12:
            return spectral
        return blend * spectral + (1.0 - blend) * self._linear_weights(coordinate)

    def _chunk_elements(self, device: torch.device) -> int:
        target_bytes = self.chunk_bytes
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                target_bytes = min(target_bytes, max(4 * 1024 * 1024, int(free_bytes) // 32))
            except (RuntimeError, TypeError):
                pass
        return max(1024, target_bytes // torch.tensor([], dtype=torch.float32).element_size())

    def _normalize_rows(self, rows: Sequence[int] | None) -> tuple[int, ...]:
        if self._feature_shape is None:
            raise RuntimeError("Spectrum forecaster has no feature shape")
        branch_count = self._feature_shape[0]
        resolved = tuple(range(branch_count)) if rows is None else tuple(int(v) for v in rows)
        if not resolved:
            raise ValueError("row selection cannot be empty")
        if any(row < 0 or row >= branch_count for row in resolved):
            raise ValueError(f"row selection {resolved} is outside branch count {branch_count}")
        return resolved

    def predict(
        self,
        coordinate: float,
        blend_weight: float,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None:
            raise RuntimeError("Spectrum forecaster has no actual history")
        weights = self.combined_weights(coordinate, blend_weight)
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_segments(
        self,
        coordinate: float,
        segment_blends: Sequence[tuple[int, int, float]],
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        weighted_segments = [
            (start, end, self.combined_weights(coordinate, blend_weight))
            for start, end, blend_weight in segment_blends
        ]
        return self._predict_with_segment_weights(
            weighted_segments,
            rows=rows,
            device=device,
            dtype=dtype,
        )

    def predict_one_point_hold(
        self,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if len(self._history) != 1:
            raise RuntimeError("one-point hold requires exactly one actual history entry")
        weights = torch.ones(1, dtype=torch.float32)
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_latest_hold(
        self,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self._history:
            raise RuntimeError("latest hold requires at least one actual history entry")
        weights = torch.zeros(len(self._history), dtype=torch.float32)
        weights[-1] = 1.0
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_with_weights(
        self,
        weights: torch.Tensor,
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self._predict_with_weights(weights, rows=rows, device=device, dtype=dtype)

    def predict_with_segment_weights(
        self,
        weighted_segments: Sequence[tuple[int, int, torch.Tensor]],
        *,
        rows: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self._predict_with_segment_weights(
            weighted_segments,
            rows=rows,
            device=device,
            dtype=dtype,
        )

    def _predict_with_weights(
        self,
        weights: torch.Tensor,
        *,
        rows: Sequence[int] | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None:
            raise RuntimeError("Spectrum forecaster has no actual history")
        if len(self._feature_shape) < 2:
            raise RuntimeError("Spectrum forecaster feature shape is invalid")
        feature_rows = self._feature_shape[1] if len(self._feature_shape) >= 3 else 1
        return self._predict_with_segment_weights(
            ((0, feature_rows, weights),),
            rows=rows,
            device=device,
            dtype=dtype,
        )

    def _predict_with_segment_weights(
        self,
        weighted_segments: Sequence[tuple[int, int, torch.Tensor]],
        *,
        rows: Sequence[int] | None,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if self._feature_shape is None or self._feature_dtype is None:
            raise RuntimeError("Spectrum forecaster has no actual history")
        feature_rows = self._feature_shape[1] if len(self._feature_shape) >= 3 else 1
        normalized_segments = []
        expected_start = 0
        for start, end, weights in weighted_segments:
            start = int(start)
            end = int(end)
            if start != expected_start or end <= start or end > feature_rows:
                raise ValueError("prediction segments must cover feature rows contiguously")
            if weights.ndim != 1 or weights.numel() != len(self._history):
                raise ValueError("prediction weights must match actual history length")
            normalized_segments.append((start, end, tuple(float(weight) for weight in weights.tolist())))
            expected_start = end
        if expected_start != feature_rows:
            raise ValueError("prediction segments must cover every feature row")

        resolved_rows = self._normalize_rows(rows)
        target_device = torch.device(device or "cpu")
        target_dtype = dtype or self._feature_dtype
        if not target_dtype.is_floating_point:
            raise ValueError("prediction dtype must be floating point")

        tail_shape = self._feature_shape[1:]
        tail_numel = 1
        for size in tail_shape:
            tail_numel *= size
        if len(self._feature_shape) >= 3:
            row_numel = 1
            for size in tail_shape[1:]:
                row_numel *= size
        else:
            row_numel = tail_numel
        result = torch.empty((len(resolved_rows), *tail_shape), device=target_device, dtype=target_dtype)
        result_flat = result.reshape(-1)
        self.last_prediction_chunk_count = 0
        self.last_prediction_max_fp32_elements = 0

        for target_row, source_row in enumerate(resolved_rows):
            source_base = source_row * tail_numel
            target_base = target_row * tail_numel
            for start, end, weight_scalars in normalized_segments:
                segment_start = start * row_numel
                segment_numel = (end - start) * row_numel
                chunk_elements = min(self._chunk_elements(target_device), segment_numel)
                for offset in range(0, segment_numel, chunk_elements):
                    length = min(chunk_elements, segment_numel - offset)
                    accumulator = torch.zeros(length, device=target_device, dtype=torch.float32)
                    for scalar, entry in zip(weight_scalars, self._history, strict=True):
                        if scalar == 0.0:
                            continue
                        source = entry.feature_flat.narrow(
                            0,
                            source_base + segment_start + offset,
                            length,
                        )
                        source_fp32 = source.to(device=target_device, dtype=torch.float32, non_blocking=False)
                        accumulator.add_(source_fp32, alpha=scalar)
                    result_flat.narrow(
                        0,
                        target_base + segment_start + offset,
                        length,
                    ).copy_(accumulator.to(target_dtype))
                    self.last_prediction_chunk_count += 1
                    self.last_prediction_max_fp32_elements = max(
                        self.last_prediction_max_fp32_elements, accumulator.numel()
                    )
        return result

DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
OFFLINE_VALIDATION_SAMPLES = 16 * 1024


def _chunk_elements(chunk_bytes: int) -> int:
    if chunk_bytes < 4096:
        raise ValueError("chunk_bytes must be >= 4096")
    return max(1024, int(chunk_bytes) // torch.tensor([], dtype=torch.float32).element_size())


def tensor_all_finite(value: torch.Tensor, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> bool:
    flat = value.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    for offset in range(0, flat.numel(), chunk):
        if not bool(torch.isfinite(flat.narrow(0, offset, min(chunk, flat.numel() - offset))).all().item()):
            return False
    return True


@dataclass(frozen=True, slots=True)
class StreamResidualScore:
    forecast_rms: float
    hold_rms: float
    actual_rms: float
    epsilon: float
    score: float
    chunks: int


def measure_stream_residual(
    actual: torch.Tensor,
    shadow: torch.Tensor,
    hold: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> StreamResidualScore:
    if actual.shape != shadow.shape or actual.shape != hold.shape:
        raise ValueError("actual, shadow, and hold outputs must have identical shapes")
    if not actual.dtype.is_floating_point or not shadow.dtype.is_floating_point or not hold.dtype.is_floating_point:
        raise ValueError("residual measurement requires floating-point outputs")
    count = actual.numel()
    if count == 0:
        raise ValueError("residual measurement cannot reduce an empty output")

    actual_flat = actual.detach().reshape(-1)
    shadow_flat = shadow.detach().reshape(-1)
    hold_flat = hold.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    actual_sq = 0.0
    forecast_sq = 0.0
    hold_sq = 0.0
    chunks = 0
    for offset in range(0, count, chunk):
        length = min(chunk, count - offset)
        actual_chunk = actual_flat.narrow(0, offset, length).to(torch.float32)
        shadow_chunk = shadow_flat.narrow(0, offset, length).to(device=actual_chunk.device, dtype=torch.float32)
        hold_chunk = hold_flat.narrow(0, offset, length).to(device=actual_chunk.device, dtype=torch.float32)
        actual_sq += float(torch.sum(actual_chunk * actual_chunk, dtype=torch.float32).item())
        forecast_delta = actual_chunk - shadow_chunk
        hold_delta = actual_chunk - hold_chunk
        forecast_sq += float(torch.sum(forecast_delta * forecast_delta, dtype=torch.float32).item())
        hold_sq += float(torch.sum(hold_delta * hold_delta, dtype=torch.float32).item())
        chunks += 1

    actual_rms = math.sqrt(actual_sq / count)
    forecast_rms = math.sqrt(forecast_sq / count)
    hold_rms = math.sqrt(hold_sq / count)
    if not all(math.isfinite(value) for value in (actual_rms, forecast_rms, hold_rms)):
        raise ValueError("residual measurement produced a nonfinite RMS")

    epsilon = max(actual_rms * 1e-6, torch.finfo(torch.float32).eps)
    if forecast_rms <= epsilon and hold_rms <= epsilon:
        score = 0.0
    else:
        score = forecast_rms / max(hold_rms, epsilon)
    if not math.isfinite(score):
        raise ValueError("residual measurement produced a nonfinite score")
    return StreamResidualScore(
        forecast_rms=forecast_rms,
        hold_rms=hold_rms,
        actual_rms=actual_rms,
        epsilon=epsilon,
        score=score,
        chunks=chunks,
    )


@dataclass(frozen=True, slots=True)
class OfflineStepRecord:
    step_id: int
    coordinate: float
    actual: bool


@dataclass(slots=True)
class OfflineAnchor:
    step_id: int
    coordinate: float
    feature: torch.Tensor


class OfflineFeatureArchive:
    def __init__(
        self,
        *,
        total_steps: int,
        sampler_name: str,
        history_storage: str = "system_ram",
    ) -> None:
        if history_storage not in {"system_ram", "vram"}:
            raise ValueError("history_storage must be 'system_ram' or 'vram'")
        self.total_steps = int(total_steps)
        self.sampler_name = str(sampler_name)
        self.history_storage = str(history_storage)
        self.steps: list[OfflineStepRecord] = []
        self.anchors: list[OfflineAnchor] = []
        self.labels: tuple[Any, ...] | None = None
        self.topology: tuple[Any, ...] | None = None
        self.feature_shape: tuple[int, ...] | None = None
        self.feature_dtype: torch.dtype | None = None
        self.history_device: torch.device | None = None
        self.valid = True
        self.failure_reason: str | None = None

    @property
    def tensor_bytes(self) -> int:
        return sum(anchor.feature.numel() * anchor.feature.element_size() for anchor in self.anchors)

    @property
    def estimated_tensor_bytes(self) -> int:
        if not self.anchors:
            return 0
        feature = self.anchors[0].feature
        return self.total_steps * feature.numel() * feature.element_size()

    def invalidate(self, reason: str) -> None:
        if self.valid:
            self.valid = False
            self.failure_reason = str(reason)

    def record_step(self, step_id: int, coordinate: float, actual: bool) -> None:
        if not self.valid:
            return
        expected = len(self.steps)
        if int(step_id) != expected:
            self.invalidate(f"offline step sequence changed: expected {expected}, got {step_id}")
            return
        self.steps.append(OfflineStepRecord(int(step_id), float(coordinate), bool(actual)))

    def record_actual(
        self,
        step_id: int,
        coordinate: float,
        feature: torch.Tensor,
        *,
        labels: tuple[Any, ...],
        topology: tuple[Any, ...],
        take_ownership: bool,
    ) -> None:
        if not self.valid:
            return
        shape = tuple(int(value) for value in feature.shape)
        if self.labels is None:
            self.labels = tuple(labels)
            self.topology = tuple(topology)
            self.feature_shape = shape
            self.feature_dtype = feature.dtype
        elif tuple(labels) != self.labels or tuple(topology) != self.topology:
            self.invalidate("offline branch labels or topology changed across actual anchors")
            return
        elif shape != self.feature_shape or feature.dtype != self.feature_dtype:
            self.invalidate("offline actual feature shape or dtype changed")
            return

        detached = feature.detach()
        storage_device = torch.device("cpu") if self.history_storage == "system_ram" else detached.device
        if self.history_device is None:
            self.history_device = storage_device
        elif storage_device != self.history_device:
            self.invalidate("offline actual feature device changed")
            return
        if take_ownership and detached.device == storage_device and detached.is_contiguous():
            archived = detached
        else:
            archived = detached.to(device=storage_device, dtype=feature.dtype, copy=True).contiguous()
        self.anchors.append(OfflineAnchor(int(step_id), float(coordinate), archived))

    def complete(self, *, minimum_anchors: int) -> bool:
        if len(self.steps) != self.total_steps:
            self.invalidate(
                f"offline first pass recorded {len(self.steps)} of {self.total_steps} logical steps"
            )
        actual_ids = [step.step_id for step in self.steps if step.actual]
        anchor_ids = [anchor.step_id for anchor in self.anchors]
        if actual_ids != anchor_ids:
            self.invalidate("offline actual-step schedule does not match the retained anchor archive")
        if len(self.anchors) < int(minimum_anchors):
            self.invalidate(
                f"offline smoothing requires at least {minimum_anchors} actual anchors"
            )
        if any(
            not step.actual
            and not any(anchor.step_id < step.step_id for anchor in self.anchors)
            for step in self.steps
        ):
            self.invalidate("offline forecast step has no earlier actual anchor")
        if any(
            not step.actual
            and not any(anchor.step_id > step.step_id for anchor in self.anchors)
            for step in self.steps
        ):
            self.invalidate("offline forecast step has no future actual anchor")
        return self.valid

    def release(self) -> None:
        self.steps.clear()
        self.anchors.clear()
        self.labels = None
        self.topology = None
        self.feature_shape = None
        self.feature_dtype = None
        self.history_device = None


class OfflineSmoother:
    def __init__(
        self,
        archive: OfflineFeatureArchive,
        *,
        degree: int,
        ridge_lambda: float,
        blend_weight: float,
        audio_blend_weight: float = 0.0,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        if not archive.valid or not archive.anchors or archive.labels is None:
            raise ValueError("offline archive is incomplete")
        self.archive = archive
        self.blend_weight = float(blend_weight)
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        self.audio_blend_weight = float(audio_blend_weight)
        if not 0.0 <= self.audio_blend_weight <= 1.0:
            raise ValueError("audio_blend_weight must be in [0, 1]")
        self.degree = int(degree)
        self.ridge_lambda = float(ridge_lambda)
        self._anchor_ids = [anchor.step_id for anchor in archive.anchors]
        self._anchor_by_step = {anchor.step_id: anchor for anchor in archive.anchors}
        self._branch_count = int(archive.anchors[0].feature.shape[0])
        self._forecaster = HistoryWeightForecaster(
            degree=degree,
            ridge_lambda=ridge_lambda,
            max_history=max(len(archive.anchors), degree + 1, 2),
            chunk_bytes=chunk_bytes,
            history_storage=archive.history_storage,
        )
        for anchor in archive.anchors:
            self._forecaster.update(anchor.coordinate, anchor.feature, take_ownership=True)
        self._stream_ranges = self._resolve_stream_ranges()
        if (
            self._stream_ranges[0][0] == "packed"
            and not math.isclose(
                self.audio_blend_weight,
                self.blend_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "offline modality-specific blending requires target audio/video row metadata"
            )
        self.configured_stream_blends = {
            name: self.audio_blend_weight if name == "audio" else self.blend_weight
            for name, _, _ in self._stream_ranges
        }
        self.validation_stream_count = len(self._stream_ranges)
        self.validation_stream_max_scores = {
            name: 0.0 for name, _, _ in self._stream_ranges
        }
        self.validation_samples_per_branch = 0
        self.validation_anchor_count = 0
        self.attenuated_prediction_count = 0
        self.local_only_prediction_count = 0
        self.attenuated_prediction_counts = {
            name: 0 for name, _, _ in self._stream_ranges
        }
        self.local_only_prediction_counts = {
            name: 0 for name, _, _ in self._stream_ranges
        }
        self.effective_blend_stream_stats = {
            name: (blend, blend, blend)
            for name, blend in self.configured_stream_blends.items()
        }
        self.effective_blend_min = self.blend_weight
        self.effective_blend_mean = self.blend_weight
        self.effective_blend_max = self.blend_weight
        self._last_prediction_chunk_count = 0
        validation_started = time.perf_counter()
        try:
            self._validation_scores = self._build_validation_scores()
        finally:
            self.validation_seconds = time.perf_counter() - validation_started
        self._forecast_weights = self._build_forecast_weights()

    @property
    def history_length(self) -> int:
        return self._forecaster.history_length

    @property
    def history_device(self) -> torch.device | None:
        return self._forecaster.history_device

    @property
    def history_tensor_bytes(self) -> int:
        return self._forecaster.history_tensor_bytes

    @property
    def last_prediction_chunk_count(self) -> int:
        return self._last_prediction_chunk_count

    @staticmethod
    def _affine_spectral_weights(weights: torch.Tensor) -> torch.Tensor:
        normalized = weights.detach().to(device="cpu", dtype=torch.float32).clone()
        if normalized.ndim != 1 or normalized.numel() == 0 or not bool(torch.isfinite(normalized).all().item()):
            raise RuntimeError("offline spectral weights are invalid")
        normalized.add_((1.0 - float(normalized.sum().item())) / normalized.numel())
        if not bool(torch.isfinite(normalized).all().item()):
            raise RuntimeError("offline affine spectral correction is nonfinite")
        return normalized

    def _resolve_stream_ranges(self) -> list[tuple[str, int, int]]:
        feature_shape = self.archive.feature_shape
        if feature_shape is None or len(feature_shape) != 3:
            raise RuntimeError("offline smoothing requires [branch, rows, width] features")
        topology = {
            str(entry[0]): entry[1]
            for entry in (self.archive.topology or ())
            if isinstance(entry, tuple) and len(entry) == 2
        }
        audio_rows = topology.get("target_audio_rows")
        video_rows = topology.get("target_video_rows")
        if (
            isinstance(audio_rows, int)
            and isinstance(video_rows, int)
            and audio_rows > 0
            and video_rows > 0
            and audio_rows + video_rows == feature_shape[1]
        ):
            return [
                ("audio", 0, audio_rows),
                ("video", audio_rows, audio_rows + video_rows),
            ]
        return [("packed", 0, feature_shape[1])]

    def _sampled_anchors(self, start_row: int, end_row: int) -> torch.Tensor:
        width = self.archive.anchors[0].feature.shape[2]
        tail_numel = (end_row - start_row) * width
        sample_count = min(OFFLINE_VALIDATION_SAMPLES, tail_numel)
        flat_indices = torch.div(
            torch.arange(sample_count, dtype=torch.int64) * tail_numel,
            sample_count,
            rounding_mode="floor",
        )
        feature_rows = torch.div(flat_indices, width, rounding_mode="floor") + start_row
        feature_columns = flat_indices.remainder(width)
        sampled = []
        for anchor in self.archive.anchors:
            rows = feature_rows.to(device=anchor.feature.device)
            columns = feature_columns.to(device=anchor.feature.device)
            sampled.append(
                anchor.feature[:, rows, columns].to(device="cpu", dtype=torch.float32)
            )
        self.validation_samples_per_branch += sample_count
        return torch.stack(sampled, dim=0)

    def _build_validation_scores(self) -> list[list[list[float | None]]]:
        anchor_count = len(self.archive.anchors)
        scores: list[list[list[float | None]]] = [
            [[None] * self._branch_count for _ in range(anchor_count)]
            for _ in self._stream_ranges
        ]
        if anchor_count < self.degree + 2 or anchor_count < 3:
            return scores

        for stream_index, (stream_name, start_row, end_row) in enumerate(self._stream_ranges):
            samples = self._sampled_anchors(start_row, end_row)
            for target_index in range(1, anchor_count - 1):
                retained = [index for index in range(anchor_count) if index != target_index]
                validator = HistoryWeightForecaster(
                    degree=self.degree,
                    ridge_lambda=self.ridge_lambda,
                    max_history=max(len(retained), self.degree + 1, 2),
                    chunk_bytes=DEFAULT_CHUNK_BYTES,
                    history_storage="system_ram",
                )
                for index in retained:
                    validator.update(
                        self.archive.anchors[index].coordinate,
                        samples[index],
                        take_ownership=True,
                    )
                spectral = self._affine_spectral_weights(
                    validator.spectral_weights(self.archive.anchors[target_index].coordinate)
                )
                spectral_prediction = torch.einsum("k,kbs->bs", spectral, samples[retained])

                left = self.archive.anchors[target_index - 1]
                target = self.archive.anchors[target_index]
                right = self.archive.anchors[target_index + 1]
                spacing = right.coordinate - left.coordinate
                if abs(spacing) <= 1e-12:
                    raise RuntimeError("offline validation anchors have duplicate coordinates")
                ratio = (target.coordinate - left.coordinate) / spacing
                local_prediction = torch.lerp(samples[target_index - 1], samples[target_index + 1], ratio)
                target_samples = samples[target_index]

                for branch in range(self._branch_count):
                    actual = target_samples[branch]
                    spectral_rms = float(torch.sqrt(torch.mean((spectral_prediction[branch] - actual) ** 2)).item())
                    local_rms = float(torch.sqrt(torch.mean((local_prediction[branch] - actual) ** 2)).item())
                    actual_rms = float(torch.sqrt(torch.mean(actual * actual)).item())
                    epsilon = max(actual_rms * 1e-6, torch.finfo(torch.float32).eps)
                    if spectral_rms <= epsilon and local_rms <= epsilon:
                        score = 0.0
                    else:
                        score = spectral_rms / max(local_rms, epsilon)
                    if not math.isfinite(score):
                        raise RuntimeError("offline validation score is nonfinite")
                    scores[stream_index][target_index][branch] = score
                    self.validation_stream_max_scores[stream_name] = max(
                        self.validation_stream_max_scores[stream_name], score
                    )
        self.validation_anchor_count = anchor_count - 2
        return scores

    def _validation_score_for_interval(self, position: int, branch: int, stream_index: int) -> float:
        nearby = [
            self._validation_scores[stream_index][index][branch]
            for index in (position - 1, position)
            if self._validation_scores[stream_index][index][branch] is not None
        ]
        return max(nearby, default=1.0)

    def _build_forecast_weights(self) -> dict[tuple[int, int, int], torch.Tensor]:
        weights_by_step: dict[tuple[int, int, int], torch.Tensor] = {}
        effective_blends: list[float] = []
        stream_effective_blends = {
            name: [] for name, _, _ in self._stream_ranges
        }
        for record in self.archive.steps:
            if record.actual:
                continue
            spectral = self._affine_spectral_weights(
                self._forecaster.spectral_weights(record.coordinate)
            )
            position = bisect.bisect_left(self._anchor_ids, record.step_id)
            if position == 0 or position == len(self._anchor_ids):
                raise RuntimeError("offline forecast requires bracketing actual anchors")
            left = self.archive.anchors[position - 1]
            right = self.archive.anchors[position]
            spacing = right.coordinate - left.coordinate
            if abs(spacing) <= 1e-12:
                raise RuntimeError("offline bracketing anchors have duplicate coordinates")
            ratio = (record.coordinate - left.coordinate) / spacing
            local = torch.zeros(len(self.archive.anchors), dtype=torch.float32)
            local[position - 1] = 1.0 - ratio
            local[position] = ratio
            for stream_index, (stream_name, _, _) in enumerate(self._stream_ranges):
                configured_blend = self.configured_stream_blends[stream_name]
                for branch in range(self._branch_count):
                    validation_score = self._validation_score_for_interval(
                        position,
                        branch,
                        stream_index,
                    )
                    effective_blend = configured_blend / max(1.0, validation_score)
                    weights_by_step[(record.step_id, branch, stream_index)] = (
                        effective_blend * spectral + (1.0 - effective_blend) * local
                    )
                    effective_blends.append(effective_blend)
                    stream_effective_blends[stream_name].append(effective_blend)
                    if effective_blend < configured_blend - 1e-7:
                        self.attenuated_prediction_count += 1
                        self.attenuated_prediction_counts[stream_name] += 1
                    if effective_blend <= 1e-7:
                        self.local_only_prediction_count += 1
                        self.local_only_prediction_counts[stream_name] += 1
        if effective_blends:
            self.effective_blend_min = min(effective_blends)
            self.effective_blend_mean = sum(effective_blends) / len(effective_blends)
            self.effective_blend_max = max(effective_blends)
        for stream_name, values in stream_effective_blends.items():
            if values:
                self.effective_blend_stream_stats[stream_name] = (
                    min(values),
                    sum(values) / len(values),
                    max(values),
                )
        return weights_by_step

    def predict(
        self,
        step_id: int,
        *,
        rows: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        record = self.archive.steps[int(step_id)]
        anchor = self._anchor_by_step.get(record.step_id)
        if anchor is not None:
            weights = torch.zeros(len(self.archive.anchors), dtype=torch.float32)
            weights[self._anchor_ids.index(record.step_id)] = 1.0
            result = self._forecaster.predict_with_weights(
                weights,
                rows=rows,
                device=device,
                dtype=dtype,
            )
            self._last_prediction_chunk_count = self._forecaster.last_prediction_chunk_count
            return result

        predictions = []
        chunks = 0
        for row in rows:
            weighted_segments = [
                (
                    start,
                    end,
                    self._forecast_weights[(record.step_id, int(row), stream_index)],
                )
                for stream_index, (_, start, end) in enumerate(self._stream_ranges)
            ]
            prediction = self._forecaster.predict_with_segment_weights(
                weighted_segments,
                rows=(int(row),),
                device=device,
                dtype=dtype,
            )
            predictions.append(prediction)
            chunks += self._forecaster.last_prediction_chunk_count
        self._last_prediction_chunk_count = chunks
        return torch.cat(predictions, dim=0)

LOG = logging.getLogger(__name__)

_FEEDBACK_SCORE_THRESHOLD = 1.5
_FEEDBACK_MAX_REFRESHES = 3
_ROLLBACK_SCORE_THRESHOLD = 1.5
_ROLLBACK_MAX_CORRECTIONS = 3


def _as_cpu_float64_vector(value: Any) -> torch.Tensor:
    """Detach a tensor-like value, move it to CPU, then cast to float64."""
    return (
        torch.as_tensor(value)
        .detach()
        .to(device="cpu")
        .to(dtype=torch.float64)
        .reshape(-1)
    )


class ForecastRetryActual(RuntimeError):
    """Internal signal used to discard a partial forecast attempt transactionally."""


class OfflineReplayAbort(RuntimeError):
    """Internal signal used to return the valid first-pass result after replay failure."""


@dataclass(slots=True)
class RuntimeStats:
    run_id: int = 0
    sampler_name: str = "unknown"
    total_steps: int = 0
    actual_steps: int = 0
    forecast_steps: int = 0
    actual_transformer_calls: int = 0
    forecast_model_calls: int = 0
    forecast_fallbacks: int = 0
    bypassed_steps: int = 0
    causal_video_blend_weight: float = 0.0
    causal_audio_blend_weight: float = 0.0
    history_archive_seconds: float = 0.0
    history_update_seconds: float = 0.0
    forecast_prediction_seconds: float = 0.0
    direct_history_updates: int = 0
    current_window: float = 0.0
    disabled: bool = False
    disable_reason: str | None = None
    residual_measure_seconds: float = 0.0
    residual_output_head_seconds: float = 0.0
    residual_anchors: int = 0
    residual_failures: int = 0
    residual_skipped_terminal_probes: int = 0
    residual_max_score: float = 0.0
    residual_max_video_score: float = 0.0
    residual_max_audio_score: float = 0.0
    residual_policy_max_score: float = 0.0
    feedback_refreshes: int = 0
    feedback_suppressed_threshold: int = 0
    feedback_suppressed_budget: int = 0
    speculative_forecast_calls: int = 0
    discarded_actual_calls: int = 0
    rollback_count: int = 0
    rollback_suppressed_threshold: int = 0
    rollback_suppressed_budget: int = 0
    replayed_transformer_calls: int = 0
    offline_archive_bytes: int = 0
    offline_estimated_archive_bytes: int = 0
    offline_archive_seconds: float = 0.0
    offline_smoother_build_seconds: float = 0.0
    offline_replay_steps: int = 0
    offline_replay_model_calls: int = 0
    offline_replay_anchor_steps: int = 0
    offline_replay_smoothed_steps: int = 0
    offline_validation_samples_per_branch: int = 0
    offline_validation_anchors: int = 0
    offline_validation_streams: int = 0
    offline_validation_seconds: float = 0.0
    offline_validation_audio_max: float = 0.0
    offline_validation_video_max: float = 0.0
    offline_validation_packed_max: float = 0.0
    offline_attenuated_predictions: int = 0
    offline_local_only_predictions: int = 0
    offline_effective_blend_min: float = 0.0
    offline_effective_blend_mean: float = 0.0
    offline_effective_blend_max: float = 0.0
    offline_effective_audio_blend_min: float = 0.0
    offline_effective_audio_blend_mean: float = 0.0
    offline_effective_audio_blend_max: float = 0.0
    offline_effective_video_blend_min: float = 0.0
    offline_effective_video_blend_mean: float = 0.0
    offline_effective_video_blend_max: float = 0.0
    offline_attenuated_audio_predictions: int = 0
    offline_attenuated_video_predictions: int = 0
    offline_local_only_audio_predictions: int = 0
    offline_local_only_video_predictions: int = 0


@dataclass(slots=True)
class _ActualRecord:
    feature: torch.Tensor
    labels: tuple[Any, ...] | None


@dataclass(slots=True)
class ResidualProbe:
    shadow: torch.Tensor
    hold: torch.Tensor


@dataclass(slots=True)
class _ResidualRecord:
    labels: tuple[Any, ...]
    video_score: StreamResidualScore
    audio_score: StreamResidualScore


@dataclass(slots=True)
class _AggregatedResidual:
    policy_score: float
    video_score: float
    audio_score: float


@dataclass(slots=True)
class _CallState:
    topology: tuple[Any, ...]
    labels: tuple[Any, ...] | None
    expected_shape: tuple[int, ...]
    observed_actual: bool = False
    used_forecast: bool = False


@dataclass(slots=True)
class _StepState:
    step_id: int
    coordinate: float
    adaptive_recompute: bool
    mode: str
    reason: str
    bootstrap_forecast: bool = False
    calls: list[_CallState] = field(default_factory=list)
    actual_records: list[_ActualRecord] = field(default_factory=list)
    used_history_rows: set[int] = field(default_factory=set)
    fallback: bool = False
    residual_expected: bool = False
    residual_records: list[_ResidualRecord] = field(default_factory=list)
    rollback_replay: bool = False
    consumes_feedback_refresh: bool = False
    residual_skip_reason: str | None = None


@dataclass(slots=True)
class _RunState:
    run_id: int
    sampler_name: str
    total_steps: int
    sigma_min: float
    sigma_max: float
    supported_sampler: bool
    max_consecutive_forecasts: int | None
    min_actual_steps_after_forecast: int
    min_tail_actual_steps: int
    next_step_id: int = 0


@dataclass(slots=True)
class RuntimeRollbackSnapshot:
    next_step_id: int
    forecaster: ForecasterSnapshot
    history_topology: tuple[Any, ...] | None
    history_labels: tuple[Any, ...] | None
    current_window: float
    consecutive_forecasts: int
    required_actual_refreshes: int
    required_feedback_actuals: int
    disabled: bool
    disable_reason: str | None
    experiment_disabled: bool
    experiment_disable_reason: str | None
    last_completed_mode: str | None
    last_completed_step_id: int | None
    stats: RuntimeStats


class SpectrumH3Runtime:
    def __init__(self, config: SpectrumH3Config):
        self.config = config.validate()
        self.forecaster = HistoryWeightForecaster(
            degree=self.config.degree,
            ridge_lambda=self.config.ridge_lambda,
            max_history=self.config.max_history,
            history_storage=self.config.history_storage,
        )

        self.stats = RuntimeStats(current_window=self.config.window_size)
        self._run_counter = 0
        self._run: _RunState | None = None
        self._step: _StepState | None = None
        self._history_topology: tuple[Any, ...] | None = None
        self._history_labels: tuple[Any, ...] | None = None
        self._current_window = float(self.config.window_size)
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._required_feedback_actuals = 0
        self._disabled = False
        self._disable_reason: str | None = None
        self._experiment_disabled = False
        self._experiment_disable_reason: str | None = None
        self._last_completed_mode: str | None = None
        self._last_completed_step_id: int | None = None
        self._rollback_requested = False
        self._forced_actual_reason: str | None = None
        self._forced_actual_is_replay = False
        self._rollback_replay_active = False
        self._offline_phase: str | None = None
        self._offline_archive: OfflineFeatureArchive | None = None
        self._offline_smoother: OfflineSmoother | None = None
        self._offline_archive_seconds_total = 0.0
        self._offline_smoother_build_seconds_total = 0.0

    @property
    def active_run_id(self) -> int | None:
        return None if self._run is None else self._run.run_id

    @property
    def active_step_id(self) -> int | None:
        return None if self._step is None else self._step.step_id

    @property
    def supported_sampler(self) -> bool:
        return self._run is not None and self._run.supported_sampler

    @property
    def disabled_reason(self) -> str | None:
        return self._disable_reason

    @property
    def history_labels(self) -> tuple[Any, ...] | None:
        return self._history_labels

    @property
    def last_completed_mode(self) -> str | None:
        return self._last_completed_mode

    @property
    def last_completed_step_id(self) -> int | None:
        return self._last_completed_step_id

    @property
    def experiment_disabled_reason(self) -> str | None:
        return self._experiment_disable_reason

    @property
    def offline_archive(self) -> OfflineFeatureArchive | None:
        return self._offline_archive

    @property
    def offline_phase(self) -> str | None:
        return self._offline_phase

    @property
    def prediction_history_length(self) -> int:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.history_length
        return self.forecaster.history_length

    @property
    def prediction_history_device(self) -> torch.device | None:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.history_device
        return self.forecaster.history_device

    @property
    def prediction_history_tensor_bytes(self) -> int:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.history_tensor_bytes
        return self.forecaster.history_tensor_bytes

    @property
    def last_prediction_chunk_count(self) -> int:
        if self._offline_phase == "replay" and self._offline_smoother is not None:
            return self._offline_smoother.last_prediction_chunk_count
        return self.forecaster.last_prediction_chunk_count

    def record_residual_output_head_seconds(self, elapsed: float) -> None:
        self.stats.residual_output_head_seconds += max(0.0, float(elapsed))

    def _record_offline_smoother_stats(self) -> None:
        smoother = self._offline_smoother
        if smoother is None:
            return
        self.stats.offline_validation_samples_per_branch = smoother.validation_samples_per_branch
        self.stats.offline_validation_anchors = smoother.validation_anchor_count
        self.stats.offline_validation_streams = smoother.validation_stream_count
        self.stats.offline_validation_seconds = smoother.validation_seconds
        self.stats.offline_validation_audio_max = smoother.validation_stream_max_scores.get("audio", 0.0)
        self.stats.offline_validation_video_max = smoother.validation_stream_max_scores.get("video", 0.0)
        self.stats.offline_validation_packed_max = smoother.validation_stream_max_scores.get("packed", 0.0)
        self.stats.offline_attenuated_predictions = smoother.attenuated_prediction_count
        self.stats.offline_local_only_predictions = smoother.local_only_prediction_count
        self.stats.offline_effective_blend_min = smoother.effective_blend_min
        self.stats.offline_effective_blend_mean = smoother.effective_blend_mean
        self.stats.offline_effective_blend_max = smoother.effective_blend_max
        audio_blends = smoother.effective_blend_stream_stats.get("audio", (0.0, 0.0, 0.0))
        video_blends = smoother.effective_blend_stream_stats.get("video", (0.0, 0.0, 0.0))
        (
            self.stats.offline_effective_audio_blend_min,
            self.stats.offline_effective_audio_blend_mean,
            self.stats.offline_effective_audio_blend_max,
        ) = audio_blends
        (
            self.stats.offline_effective_video_blend_min,
            self.stats.offline_effective_video_blend_mean,
            self.stats.offline_effective_video_blend_max,
        ) = video_blends
        self.stats.offline_attenuated_audio_predictions = smoother.attenuated_prediction_counts.get("audio", 0)
        self.stats.offline_attenuated_video_predictions = smoother.attenuated_prediction_counts.get("video", 0)
        self.stats.offline_local_only_audio_predictions = smoother.local_only_prediction_counts.get("audio", 0)
        self.stats.offline_local_only_video_predictions = smoother.local_only_prediction_counts.get("video", 0)

    def _prediction_segments(self, call: _CallState) -> tuple[tuple[int, int, float], ...]:
        audio_blend_weight, video_blend_weight = self._causal_prediction_blends()
        topology = {
            str(entry[0]): entry[1]
            for entry in call.topology
            if isinstance(entry, tuple) and len(entry) == 2
        }
        audio_rows = topology.get("target_audio_rows")
        video_rows = topology.get("target_video_rows")
        target_rows = call.expected_shape[1]
        if (
            isinstance(audio_rows, int)
            and isinstance(video_rows, int)
            and audio_rows > 0
            and video_rows > 0
            and audio_rows + video_rows == target_rows
        ):
            return (
                (0, audio_rows, audio_blend_weight),
                (audio_rows, target_rows, video_blend_weight),
            )
        if math.isclose(
            audio_blend_weight,
            video_blend_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return ((0, target_rows, video_blend_weight),)
        raise ValueError("packed H3 topology does not expose the target audio/video boundary")

    def _causal_prediction_blends(self) -> tuple[float, float]:
        if self._offline_phase in {"first_pass", "replay"}:
            return 0.0, 0.0
        return self.config.audio_blend_weight, self.config.blend_weight

    def _residual_experiment_enabled(self) -> bool:
        return bool(
            not self._experiment_disabled
            and (
                self.config.anchor_residual_feedback
                or self.config.selective_rollback_correction
            )
        )

    def disable_experiment(self, reason: str) -> bool:
        newly_disabled = not self._experiment_disabled
        self._experiment_disabled = True
        self._experiment_disable_reason = str(reason)
        self._rollback_requested = False
        self._required_feedback_actuals = 0
        if newly_disabled:
            self.stats.residual_failures += 1
            LOG.warning("Spectrum H3 experimental mode disabled for this run: %s", reason)
        return newly_disabled

    def start_run(
        self,
        sigmas: torch.Tensor,
        sampler_name: str,
        *,
        supported_sampler: bool,
        max_consecutive_forecasts: int | None = None,
        min_actual_steps_after_forecast: int = 0,
        min_tail_actual_steps: int = 0,
    ) -> int:
        if self._run is not None:
            raise RuntimeError("Spectrum H3 runtime already has an active run")
        if max_consecutive_forecasts is not None and (
            isinstance(max_consecutive_forecasts, bool)
            or not isinstance(max_consecutive_forecasts, int)
            or max_consecutive_forecasts < 1
        ):
            raise ValueError("max_consecutive_forecasts must be None or an integer >= 1")
        for name, value in (
            ("min_actual_steps_after_forecast", min_actual_steps_after_forecast),
            ("min_tail_actual_steps", min_tail_actual_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        sigma_values = _as_cpu_float64_vector(sigmas)
        total_steps = max(0, sigma_values.numel() - 1)
        evaluated = sigma_values[:-1]
        finite_schedule = bool(evaluated.numel()) and bool(torch.isfinite(evaluated).all().item())
        sigma_min = float(evaluated.min().item()) if finite_schedule else 0.0
        sigma_max = float(evaluated.max().item()) if finite_schedule else 0.0
        schedule_valid = finite_schedule and math.isfinite(sigma_min) and math.isfinite(sigma_max) and sigma_max > sigma_min

        self._run_counter += 1
        effective_supported = bool(
            self.config.enabled and supported_sampler and schedule_valid and total_steps > 0
        )
        self._run = _RunState(
            run_id=self._run_counter,
            sampler_name=str(sampler_name),
            total_steps=total_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            supported_sampler=effective_supported,
            max_consecutive_forecasts=max_consecutive_forecasts,
            min_actual_steps_after_forecast=min_actual_steps_after_forecast,
            min_tail_actual_steps=min_tail_actual_steps,
        )
        self._step = None
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._current_window = float(self.config.window_size)
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._required_feedback_actuals = 0
        self._disabled = not effective_supported
        self._experiment_disabled = False
        self._experiment_disable_reason = None
        self._last_completed_mode = None
        self._last_completed_step_id = None
        self._rollback_requested = False
        self._forced_actual_reason = None
        self._forced_actual_is_replay = False
        self._rollback_replay_active = False
        if not self.config.enabled:
            self._disable_reason = "forecasting disabled by configuration"
        elif not supported_sampler:
            self._disable_reason = f"sampler {sampler_name!r} is not allowlisted for one-call solver-step tracking"
        elif not schedule_valid:
            self._disable_reason = "supplied sigma schedule is empty, nonfinite, or has no usable range"
        elif total_steps <= 0:
            self._disable_reason = "supplied sigma schedule has no solver steps"
        else:
            self._disable_reason = None
        causal_audio_blend, causal_video_blend = self._causal_prediction_blends()
        self.stats = RuntimeStats(
            run_id=self._run.run_id,
            sampler_name=self._run.sampler_name,
            total_steps=total_steps,
            current_window=self._current_window,
            disabled=self._disabled,
            disable_reason=self._disable_reason,
            causal_video_blend_weight=causal_video_blend,
            causal_audio_blend_weight=causal_audio_blend,
        )
        if self._offline_phase == "replay" and self._offline_archive is not None:
            self.stats.offline_archive_bytes = self._offline_archive.tensor_bytes
            self.stats.offline_estimated_archive_bytes = self._offline_archive.estimated_tensor_bytes
            self.stats.offline_archive_seconds = self._offline_archive_seconds_total
            self.stats.offline_smoother_build_seconds = self._offline_smoother_build_seconds_total
            self._record_offline_smoother_stats()
        return self._run.run_id

    def end_run(self, run_id: int) -> None:
        if self._run is None:
            return
        if self._run.run_id != int(run_id):
            raise RuntimeError("attempted to end a stale Spectrum H3 run")
        self._step = None
        self._run = None
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._consecutive_forecasts = 0
        self._required_actual_refreshes = 0
        self._required_feedback_actuals = 0
        self._rollback_requested = False
        self._forced_actual_reason = None
        self._forced_actual_is_replay = False
        self._rollback_replay_active = False
        self._last_completed_mode = None
        self._last_completed_step_id = None

    def begin_offline_capture(self, *, total_steps: int, sampler_name: str) -> None:
        self.release_offline_archive()
        self._offline_archive_seconds_total = 0.0
        self._offline_smoother_build_seconds_total = 0.0
        self._offline_phase = "first_pass"
        self._offline_archive = OfflineFeatureArchive(
            total_steps=total_steps,
            sampler_name=sampler_name,
            history_storage=self.config.offline_archive_storage,
        )

    def complete_offline_capture(self) -> bool:
        archive = self._offline_archive
        if self._offline_phase != "first_pass" or archive is None:
            raise RuntimeError("offline capture is not active")
        if self._disabled:
            archive.invalidate(self._disable_reason or "base Spectrum disabled during offline first pass")
        complete = archive.complete(minimum_anchors=self.config.min_fit_points)
        self.stats.offline_archive_bytes = archive.tensor_bytes
        self.stats.offline_estimated_archive_bytes = archive.estimated_tensor_bytes
        if not complete:
            return False
        started = time.perf_counter()
        try:
            self._offline_smoother = OfflineSmoother(
                archive,
                degree=self.config.degree,
                ridge_lambda=self.config.ridge_lambda,
                blend_weight=self.config.blend_weight,
                audio_blend_weight=self.config.audio_blend_weight,
            )
            self._record_offline_smoother_stats()
        except (RuntimeError, ValueError) as exc:
            archive.invalidate(f"offline smoother construction failed: {exc}")
            return False
        finally:
            elapsed = time.perf_counter() - started
            self._offline_smoother_build_seconds_total += elapsed
            self.stats.offline_smoother_build_seconds += elapsed
        return True

    def begin_offline_replay(self) -> None:
        if self._offline_archive is None or self._offline_smoother is None:
            raise RuntimeError("offline replay requires a complete first-pass archive")
        self._offline_phase = "replay"

    def release_offline_archive(self) -> None:
        if self._offline_archive is not None:
            self._offline_archive.release()
        self._offline_archive = None
        self._offline_smoother = None
        self._offline_phase = None

    def coordinate_for_timestep(self, timestep: torch.Tensor | float) -> float:
        if self._run is None:
            raise RuntimeError("Spectrum H3 runtime is outside a sampling run")
        value_tensor = _as_cpu_float64_vector(timestep)
        if value_tensor.numel() == 0 or not bool(torch.isfinite(value_tensor).all().item()):
            raise RuntimeError("current solver timestep is empty or nonfinite")
        if not bool(torch.allclose(value_tensor, value_tensor[0].expand_as(value_tensor))):
            raise RuntimeError("current predict_noise call contains multiple solver timesteps")
        value = float(value_tensor[0].item())
        sigma_span = self._run.sigma_max - self._run.sigma_min
        if not math.isfinite(sigma_span) or sigma_span <= 0.0:
            return 0.0
        coordinate = 2.0 * (value - self._run.sigma_min) / sigma_span - 1.0
        return float(max(-1.0, min(1.0, coordinate)))

    def begin_step(self, timestep: torch.Tensor | float) -> dict[str, Any]:
        if self._run is None:
            raise RuntimeError("Spectrum H3 runtime is outside a sampling run")
        if self._step is not None:
            raise RuntimeError("previous Spectrum H3 solver step was not finalized")
        step_id = self._run.next_step_id
        if step_id >= self._run.total_steps:
            raise RuntimeError("predict_noise call count exceeded the supplied sigma schedule")
        coordinate = self.coordinate_for_timestep(timestep)

        if self._offline_phase == "replay":
            archive = self._offline_archive
            if archive is None or self._offline_smoother is None or step_id >= len(archive.steps):
                raise OfflineReplayAbort("offline replay archive is incomplete")
            record = archive.steps[step_id]
            if not math.isclose(record.coordinate, coordinate, rel_tol=1e-6, abs_tol=1e-6):
                raise OfflineReplayAbort(
                    f"offline replay coordinate changed at step {step_id}: "
                    f"{record.coordinate:.9f} != {coordinate:.9f}"
                )
            self._step = _StepState(
                step_id=step_id,
                coordinate=coordinate,
                adaptive_recompute=False,
                mode="replay",
                reason="offline smoothing replay",
            )
            self._run.next_step_id += 1
            return {
                "run_id": self._run.run_id,
                "step_id": step_id,
                "coordinate": coordinate,
                "actual": False,
                "reason": "offline smoothing replay",
            }

        effective_tail = max(self.config.tail_actual_steps, self._run.min_tail_actual_steps)
        tail_start = max(0, self._run.total_steps - effective_tail)
        advances_window = False
        bootstrap_forecast = False
        rollback_replay = False
        consumes_feedback_refresh = False
        if self.config.force_actual:
            actual, reason = True, "forced-actual validation mode"
        elif self._disabled:
            actual, reason = True, self._disable_reason or "forecasting disabled"
        elif self._forced_actual_reason is not None:
            actual, reason = True, self._forced_actual_reason
            rollback_replay = self._forced_actual_is_replay
            self._forced_actual_reason = None
            self._forced_actual_is_replay = False
        elif self._required_feedback_actuals > 0:
            actual, reason = True, "anchor residual feedback refresh"
            rollback_replay = False
            consumes_feedback_refresh = True
        elif step_id < self.config.warmup_steps:
            actual, reason = True, "warmup"
        elif step_id >= tail_start:
            actual, reason = True, "final actual tail"
        elif (
            self.config.bootstrap_first_forecast
            and self.config.degree == 1
            and step_id == 1
            and self.forecaster.history_length == 1
        ):
            actual, reason = False, "one-point bootstrap forecast"
            bootstrap_forecast = True
        elif not self.forecaster.ready(self.config.min_fit_points):
            actual, reason = True, "insufficient actual history"
        else:
            interval = max(1, math.floor(self._current_window))
            actual = ((self._consecutive_forecasts + 1) % interval) == 0
            reason = "adaptive recompute" if actual else "adaptive forecast"
            advances_window = actual

        forecast_limit = self._run.max_consecutive_forecasts
        if not actual and forecast_limit is not None and self._consecutive_forecasts >= forecast_limit:
            actual = True
            reason = "post-forecast sampler refresh"
            advances_window = False
            bootstrap_forecast = False
        if not actual and self._required_actual_refreshes > 0:
            actual = True
            reason = "post-forecast sampler refresh"
            advances_window = False
            bootstrap_forecast = False

        self._step = _StepState(
            step_id=step_id,
            coordinate=coordinate,
            adaptive_recompute=advances_window,
            mode="actual" if actual else "forecast",
            reason=reason,
            bootstrap_forecast=bootstrap_forecast,
            rollback_replay=rollback_replay,
            consumes_feedback_refresh=consumes_feedback_refresh,
        )
        self._run.next_step_id += 1
        return {
            "run_id": self._run.run_id,
            "step_id": step_id,
            "coordinate": coordinate,
            "actual": actual,
            "reason": reason,
        }

    def _require_step(self, run_id: int, step_id: int) -> _StepState:
        if self._run is None or self._run.run_id != int(run_id):
            raise RuntimeError("Spectrum H3 run context is stale")
        if self._step is None or self._step.step_id != int(step_id):
            raise RuntimeError("Spectrum H3 solver-step context is stale")
        return self._step

    def _disable_forecasting(self, reason: str) -> bool:
        newly_disabled = not self._disabled
        if newly_disabled:
            self._disabled = True
            self._disable_reason = str(reason)
            self.stats.disabled = True
            self.stats.disable_reason = self._disable_reason
        self.forecaster.reset()
        self._history_topology = None
        self._history_labels = None
        self._rollback_requested = False
        if self._offline_phase == "first_pass" and self._offline_archive is not None:
            self._offline_archive.invalidate(reason)
        return newly_disabled

    def _fallback_or_retry(self, step: _StepState, reason: str) -> None:
        if step.mode == "replay":
            raise OfflineReplayAbort(reason)
        self._disable_forecasting(reason)
        if any(call.used_forecast for call in step.calls):
            raise ForecastRetryActual(reason)
        step.mode = "actual"
        step.reason = reason
        step.bootstrap_forecast = False
        step.fallback = True

    def fallback_current_step(self, run_id: int, step_id: int, reason: str) -> None:
        step = self._require_step(run_id, step_id)
        self._fallback_or_retry(step, reason)

    def begin_model_call(
        self,
        run_id: int,
        step_id: int,
        *,
        topology: tuple[Any, ...],
        labels: tuple[Any, ...] | None,
        expected_shape: tuple[int, ...],
    ) -> tuple[int, bool]:
        step = self._require_step(run_id, step_id)
        normalized_topology = tuple(topology)
        normalized_shape = tuple(int(v) for v in expected_shape)
        normalized_labels = None if labels is None else tuple(labels)
        if len(normalized_shape) < 2:
            self._fallback_or_retry(step, "target feature shape has no branch dimension")
        if step.mode == "replay" and self._offline_archive is not None:
            if normalized_topology != self._offline_archive.topology:
                self._fallback_or_retry(step, "offline replay topology changed")
            if self._offline_archive.feature_shape is not None and tuple(normalized_shape[1:]) != tuple(
                self._offline_archive.feature_shape[1:]
            ):
                self._fallback_or_retry(step, "offline replay target feature shape changed")
        if self._history_topology is not None and normalized_topology != self._history_topology:
            self._fallback_or_retry(step, "packed H3 topology changed within the sampling run")
        call = _CallState(normalized_topology, normalized_labels, normalized_shape)
        step.calls.append(call)
        return len(step.calls) - 1, step.mode == "actual"

    def observe_actual(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        feature: torch.Tensor,
    ) -> None:
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if step.mode != "actual":
            raise RuntimeError("actual H3 feature observed during a forecast-only step")
        if tuple(feature.shape) != call.expected_shape:
            raise RuntimeError(
                f"actual H3 feature shape {tuple(feature.shape)} does not match {call.expected_shape}"
            )
        started = time.perf_counter()
        try:
            detached = feature.detach()
            capture_storage = self.config.history_storage
            if (
                self._offline_phase == "first_pass"
                and self.config.offline_archive_storage == "vram"
            ):
                # Preserve a compact device copy when either consumer needs it.
                # Finalization can then transfer the bounded causal history to
                # CPU while the full replay archive takes ownership on device.
                capture_storage = "vram"
            if capture_storage == "vram":
                # The observed target is a view into the complete final-block
                # hidden state. A forced clone keeps only the compact target
                # storage alive and prevents later reuse of the backing tensor.
                archived = detached.clone(memory_format=torch.contiguous_format)
            else:
                archived = detached.to(device="cpu", dtype=feature.dtype, copy=True).contiguous()
        finally:
            self.stats.history_archive_seconds += time.perf_counter() - started
        call.observed_actual = True
        step.actual_records.append(_ActualRecord(archived, call.labels))

    def prepare_residual_probe(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ResidualProbe | None:
        step = self._require_step(run_id, step_id)
        if (
            step.mode != "actual"
            or step.rollback_replay
            or self._rollback_replay_active
            or not self._residual_experiment_enabled()
        ):
            return None
        if step.residual_skip_reason is not None:
            return None
        if self._last_completed_mode != "forecast":
            return None
        if self.config.anchor_residual_feedback:
            run = self._run
            if run is None:
                raise RuntimeError("residual probe lost its active run")
            if self.stats.feedback_refreshes >= _FEEDBACK_MAX_REFRESHES:
                step.residual_skip_reason = "actual-refresh budget exhausted"
                self.stats.feedback_suppressed_budget += 1
                if self.config.debug:
                    LOG.warning(
                        "Spectrum H3 residual probe skipped run_id=%s step=%s reason=%s budget=%s",
                        run_id,
                        step_id,
                        step.residual_skip_reason,
                        _FEEDBACK_MAX_REFRESHES,
                    )
                return None
            effective_tail = max(self.config.tail_actual_steps, run.min_tail_actual_steps)
            tail_start = max(0, run.total_steps - effective_tail)
            if step.step_id + 1 >= tail_start:
                step.residual_skip_reason = "no later forecast can consume feedback"
                self.stats.residual_skipped_terminal_probes += 1
                if self.config.debug:
                    LOG.warning(
                        "Spectrum H3 residual probe skipped run_id=%s step=%s reason=%s",
                        run_id,
                        step_id,
                        step.residual_skip_reason,
                    )
                return None
        if (
            self.config.selective_rollback_correction
            and self.stats.rollback_count >= _ROLLBACK_MAX_CORRECTIONS
        ):
            step.residual_skip_reason = "rollback correction budget exhausted"
            self.stats.rollback_suppressed_budget += 1
            if self.config.debug:
                LOG.warning(
                    "Spectrum H3 residual probe skipped run_id=%s step=%s reason=%s budget=%s",
                    run_id,
                    step_id,
                    step.residual_skip_reason,
                    _ROLLBACK_MAX_CORRECTIONS,
                )
            return None
        if not self.forecaster.ready(self.config.min_fit_points):
            return None

        call = step.calls[int(call_id)]
        if self._history_labels is None or call.labels is None:
            self.disable_experiment("residual measurement branch labels are missing")
            return None
        if len(call.labels) != call.expected_shape[0] or len(set(call.labels)) != len(call.labels):
            self.disable_experiment("residual measurement branch labels are duplicate or incomplete")
            return None
        positions: list[int] = []
        for label in call.labels:
            try:
                positions.append(self._history_labels.index(label))
            except ValueError:
                self.disable_experiment("residual measurement branch identity changed")
                return None
        if len(set(positions)) != len(positions):
            self.disable_experiment("residual measurement assigned a canonical row more than once")
            return None
        history_shape = self.forecaster.feature_shape
        if history_shape is None or tuple(call.expected_shape[1:]) != tuple(history_shape[1:]):
            self.disable_experiment("residual measurement target feature shape changed")
            return None

        started = time.perf_counter()
        try:
            segments = self._prediction_segments(call)
            shadow = self.forecaster.predict_segments(
                step.coordinate,
                segments,
                rows=positions,
                device=device,
                dtype=dtype,
            )
            hold = self.forecaster.predict_latest_hold(
                rows=positions,
                device=device,
                dtype=dtype,
            )
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as exc:
            self.disable_experiment(f"residual probe prediction failed: {exc}")
            return None
        finally:
            self.stats.residual_measure_seconds += time.perf_counter() - started
        if tuple(shadow.shape) != call.expected_shape or tuple(hold.shape) != call.expected_shape:
            self.disable_experiment("residual probe prediction shape is invalid")
            return None
        if not tensor_all_finite(shadow) or not tensor_all_finite(hold):
            self.disable_experiment("residual probe prediction is nonfinite")
            return None
        step.residual_expected = True
        return ResidualProbe(shadow=shadow, hold=hold)

    def record_residual_measurement(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        probe: ResidualProbe,
        *,
        actual_feature: torch.Tensor,
        actual_output: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        shadow_output: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        hold_output: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        step = self._require_step(run_id, step_id)
        if not self._residual_experiment_enabled() or step.mode != "actual":
            return
        call = step.calls[int(call_id)]
        if call.labels is None:
            self.disable_experiment("residual measurement lost branch labels")
            return
        if not all(
            isinstance(output, (list, tuple)) and len(output) == 2
            for output in (actual_output, shadow_output, hold_output)
        ):
            self.disable_experiment("residual measurement output structure changed")
            return
        started = time.perf_counter()
        try:
            video_score = measure_stream_residual(
                actual_output[0], shadow_output[0], hold_output[0]
            )
            audio_score = measure_stream_residual(
                actual_output[1], shadow_output[1], hold_output[1]
            )
        except (RuntimeError, ValueError) as exc:
            self.disable_experiment(f"residual measurement failed: {exc}")
            return
        finally:
            self.stats.residual_measure_seconds += time.perf_counter() - started
        step.residual_records.append(
            _ResidualRecord(tuple(call.labels), video_score, audio_score)
        )
        self.stats.residual_max_video_score = max(
            self.stats.residual_max_video_score, video_score.score
        )
        self.stats.residual_max_audio_score = max(
            self.stats.residual_max_audio_score, audio_score.score
        )
        self.stats.residual_max_score = max(
            self.stats.residual_max_score, video_score.score, audio_score.score
        )

    def predict(
        self,
        run_id: int,
        step_id: int,
        call_id: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        step = self._require_step(run_id, step_id)
        call = step.calls[int(call_id)]
        if step.mode == "actual":
            return None
        canonical_labels = (
            self._offline_archive.labels
            if step.mode == "replay" and self._offline_archive is not None
            else self._history_labels
        )
        if canonical_labels is None or call.labels is None:
            self._fallback_or_retry(step, "branch labels are missing; forecast row correspondence is unproven")
            return None
        if len(call.labels) != call.expected_shape[0] or len(set(call.labels)) != len(call.labels):
            self._fallback_or_retry(step, "branch labels are duplicate or do not match the model-call batch")
            return None
        positions = []
        for label in call.labels:
            try:
                position = canonical_labels.index(label)
            except ValueError:
                self._fallback_or_retry(step, "conditional branch identity changed")
                return None
            if position in step.used_history_rows:
                self._fallback_or_retry(step, "conditional branch row was assigned more than once")
                return None
            positions.append(position)

        history_shape = (
            self._offline_archive.feature_shape
            if step.mode == "replay" and self._offline_archive is not None
            else self.forecaster.feature_shape
        )
        if history_shape is None or tuple(call.expected_shape[1:]) != tuple(history_shape[1:]):
            self._fallback_or_retry(step, "target audio/video row count or hidden width changed")
            return None
        if step.bootstrap_forecast and self.forecaster.history_length != 1:
            self._fallback_or_retry(
                step,
                "one-point bootstrap forecast requires exactly one actual history entry",
            )
            return None
        segments = None
        if step.mode != "replay" and not step.bootstrap_forecast:
            try:
                segments = self._prediction_segments(call)
            except ValueError as exc:
                self._fallback_or_retry(step, str(exc))
                return None
        started = time.perf_counter()
        try:
            if step.mode == "replay":
                if self._offline_smoother is None:
                    raise OfflineReplayAbort("offline replay smoother is missing")
                predicted = self._offline_smoother.predict(
                    step.step_id,
                    rows=tuple(positions),
                    device=device,
                    dtype=dtype,
                )
            elif step.bootstrap_forecast:
                predicted = self.forecaster.predict_one_point_hold(
                    rows=positions,
                    device=device,
                    dtype=dtype,
                )
            else:
                assert segments is not None
                predicted = self.forecaster.predict_segments(
                    step.coordinate,
                    segments,
                    rows=positions,
                    device=device,
                    dtype=dtype,
                )
        except (RuntimeError, ValueError) as exc:
            if step.mode == "replay":
                raise OfflineReplayAbort(f"offline replay prediction failed: {exc}") from exc
            raise
        finally:
            self.stats.forecast_prediction_seconds += time.perf_counter() - started
        if tuple(predicted.shape) != call.expected_shape:
            self._fallback_or_retry(step, "predicted target feature shape is invalid")
            return None
        step.used_history_rows.update(positions)
        call.used_forecast = True
        return predicted

    def prepare_actual_retry(self, run_id: int, step_id: int, reason: str) -> None:
        step = self._require_step(run_id, step_id)
        step.mode = "actual"
        step.reason = str(reason)
        step.bootstrap_forecast = False
        step.fallback = True
        step.calls.clear()
        step.actual_records.clear()
        step.residual_records.clear()
        step.residual_expected = False
        step.used_history_rows.clear()
        self.stats.forecast_fallbacks += 1

    @staticmethod
    def _label_key(label: Any) -> tuple[str, str]:
        return type(label).__name__, repr(label)

    def _aggregate_actual(self, step: _StepState) -> torch.Tensor | None:
        if not step.actual_records:
            self._disable_forecasting("actual solver step produced no observable target feature")
            return None
        if any(record.labels is None for record in step.actual_records):
            self._disable_forecasting("actual solver step did not provide branch labels")
            return None

        rows: list[tuple[Any, torch.Tensor]] = []
        topology = step.calls[0].topology
        for call in step.calls:
            if call.topology != topology:
                self._disable_forecasting("packed H3 topology changed between model subcalls")
                return None
        for record in step.actual_records:
            assert record.labels is not None
            if len(record.labels) != record.feature.shape[0]:
                self._disable_forecasting("branch labels do not cover actual feature rows")
                return None
            rows.extend((label, record.feature[index]) for index, label in enumerate(record.labels))
        labels = tuple(label for label, _ in rows)
        if len(set(labels)) != len(labels):
            self._disable_forecasting("duplicate conditional branch labels make row correspondence ambiguous")
            return None

        if self._history_labels is None:
            canonical_labels = tuple(sorted(labels, key=self._label_key))
        else:
            canonical_labels = self._history_labels
            if set(labels) != set(canonical_labels) or len(labels) != len(canonical_labels):
                self._disable_forecasting("conditional branch set changed across actual solver steps")
                return None
        if len(step.actual_records) == 1 and step.actual_records[0].labels == canonical_labels:
            combined = step.actual_records[0].feature
            self.stats.direct_history_updates += 1
        else:
            row_map = {label: feature for label, feature in rows}
            combined = torch.stack([row_map[label] for label in canonical_labels], dim=0).contiguous()

        if self._history_topology is None:
            self._history_topology = topology
        elif topology != self._history_topology:
            self._disable_forecasting("packed H3 topology changed across actual history")
            return None
        self._history_labels = canonical_labels
        return combined

    def _aggregate_residual(
        self,
        step: _StepState,
    ) -> _AggregatedResidual | None:
        if not step.residual_expected:
            return None
        if len(step.residual_records) != len(step.actual_records):
            self.disable_experiment("residual measurement did not cover every actual model subcall")
            return None
        if self._history_labels is None:
            self.disable_experiment("residual measurement has no canonical branch labels")
            return None
        video_score = 0.0
        audio_score = 0.0
        for record in step.residual_records:
            video_score = max(video_score, record.video_score.score)
            audio_score = max(audio_score, record.audio_score.score)
        labels = tuple(
            label
            for record in step.residual_records
            for label in record.labels
        )
        if len(set(labels)) != len(labels) or set(labels) != set(self._history_labels):
            self.disable_experiment("residual branch set is duplicate or incomplete")
            return None
        policy_score = (
            video_score
            if self.config.anchor_residual_feedback
            else max(video_score, audio_score)
        )
        if (
            not all(math.isfinite(value) for value in (video_score, audio_score, policy_score))
        ):
            self.disable_experiment("aggregated residual score is nonfinite")
            return None
        return _AggregatedResidual(
            policy_score=policy_score,
            video_score=video_score,
            audio_score=audio_score,
        )

    def _apply_residual_policy(self, step: _StepState, result: _AggregatedResidual) -> None:
        if self._experiment_disabled:
            return
        score = result.policy_score
        action = "none"
        if self.config.anchor_residual_feedback:
            if score < _FEEDBACK_SCORE_THRESHOLD:
                self.stats.feedback_suppressed_threshold += 1
                action = "below_refresh_threshold"
            elif self.stats.feedback_refreshes >= _FEEDBACK_MAX_REFRESHES:
                self.stats.feedback_suppressed_budget += 1
                action = "refresh_budget_exhausted"
            else:
                self._required_feedback_actuals = max(self._required_feedback_actuals, 1)
                action = "actual_refresh_requested"
        elif self.config.selective_rollback_correction and self._last_completed_mode == "forecast":
            if self._rollback_replay_active:
                action = "rollback_replay_ignored"
            elif score < _ROLLBACK_SCORE_THRESHOLD:
                self.stats.rollback_suppressed_threshold += 1
                action = "below_rollback_threshold"
            elif self.stats.rollback_count >= _ROLLBACK_MAX_CORRECTIONS:
                self.stats.rollback_suppressed_budget += 1
                action = "rollback_budget_exhausted"
            else:
                self._rollback_requested = True
                action = "rollback_requested"
        self.stats.residual_policy_max_score = max(
            self.stats.residual_policy_max_score,
            score,
        )
        if self.config.debug:
            LOG.warning(
                "Spectrum H3 residual anchor run_id=%s step=%s video=%.6f audio=%.6f "
                "policy=%.6f action=%s",
                self.stats.run_id,
                step.step_id,
                result.video_score,
                result.audio_score,
                score,
                action,
            )

    def finalize_step(self, run_id: int, step_id: int) -> None:
        step = self._require_step(run_id, step_id)
        if not step.calls:
            if step.mode == "replay":
                raise OfflineReplayAbort(
                    "offline replay step did not reach the native H3 model wrapper"
                )
            if step.fallback and self._disabled:
                self._consecutive_forecasts = 0
                self.stats.actual_steps += 1
                self.stats.current_window = self._current_window
                self._step = None
                return
            reason = "solver step completed without reaching the native H3 model wrapper"
            newly_disabled = self._disable_forecasting(reason)
            self._consecutive_forecasts = 0
            self._required_actual_refreshes = 0
            self.stats.bypassed_steps += 1
            self.stats.current_window = self._current_window
            self._step = None
            if newly_disabled:
                LOG.warning(
                    "Spectrum H3 disabled for the rest of this run because a predict-noise "
                    "evaluation returned without reaching the native MiniMax H3 model wrapper; "
                    "accepting the wrapped result as a passthrough. Another model or cache patch "
                    "may have intercepted the evaluation."
                )
            return

        if step.mode == "replay":
            if any(call.observed_actual for call in step.calls) or not all(
                call.used_forecast for call in step.calls
            ):
                raise OfflineReplayAbort("offline replay model-call transaction was incomplete")
            archive_labels = self._offline_archive.labels if self._offline_archive is not None else ()
            expected_rows = set(range(len(archive_labels or ())))
            if step.used_history_rows != expected_rows:
                raise OfflineReplayAbort("offline replay branch-row allocation was incomplete")
            self.stats.offline_replay_steps += 1
            self.stats.offline_replay_model_calls += len(step.calls)
            archive = self._offline_archive
            if archive is None:
                raise OfflineReplayAbort("offline replay archive disappeared during finalization")
            if archive.steps[step.step_id].actual:
                self.stats.offline_replay_anchor_steps += 1
            else:
                self.stats.offline_replay_smoothed_steps += 1
            self._last_completed_mode = "replay"
            self._last_completed_step_id = step.step_id
            self._step = None
            return

        if step.mode == "forecast":
            if any(call.observed_actual for call in step.calls) or not all(call.used_forecast for call in step.calls):
                raise ForecastRetryActual("forecast solver step was incomplete or mixed with an actual call")
            expected_rows = set(range(len(self._history_labels or ())))
            if step.used_history_rows != expected_rows:
                raise ForecastRetryActual("forecast branch-row allocation was incomplete")
            self._consecutive_forecasts += 1
            self._required_actual_refreshes = self._run.min_actual_steps_after_forecast
            self.stats.forecast_steps += 1
            self.stats.forecast_model_calls += len(step.calls)
        else:
            if any(call.used_forecast for call in step.calls):
                raise RuntimeError("actual solver step retained a forecasted subcall")
            combined = self._aggregate_actual(step)
            residual_result = self._aggregate_residual(step)
            if combined is not None and not self._disabled:
                if self._offline_phase == "first_pass" and self._offline_archive is not None:
                    assert self._history_labels is not None
                    archive_started = time.perf_counter()
                    try:
                        self._offline_archive.record_actual(
                            step.step_id,
                            step.coordinate,
                            combined,
                            labels=self._history_labels,
                            topology=step.calls[0].topology,
                            take_ownership=True,
                        )
                    finally:
                        elapsed = time.perf_counter() - archive_started
                        self.stats.offline_archive_seconds += elapsed
                        self._offline_archive_seconds_total += elapsed
                update_started = time.perf_counter()
                try:
                    self.forecaster.update(step.coordinate, combined, take_ownership=True)
                except ValueError as exc:
                    self._disable_forecasting(f"actual H3 feature is incompatible with history: {exc}")
                finally:
                    self.stats.history_update_seconds += time.perf_counter() - update_started
            self._consecutive_forecasts = 0
            self._required_actual_refreshes = max(0, self._required_actual_refreshes - 1)
            if step.consumes_feedback_refresh:
                self._required_feedback_actuals = max(0, self._required_feedback_actuals - 1)
            self.stats.actual_steps += 1
            self.stats.actual_transformer_calls += len(step.actual_records)
            if step.consumes_feedback_refresh:
                self.stats.feedback_refreshes += 1
            if step.rollback_replay:
                self.stats.replayed_transformer_calls += len(step.actual_records)
            if (
                step.adaptive_recompute
                and not step.fallback
                and not self._disabled
                and step.step_id >= self.config.warmup_steps
            ):
                window_ceiling = max(float(self.config.window_size), float(self.config.max_history))
                self._current_window = min(
                    round(self._current_window + self.config.flex_window, 6),
                    window_ceiling,
                )
            if residual_result is not None and not self._experiment_disabled:
                self.stats.residual_anchors += 1
                self._apply_residual_policy(step, residual_result)

        if self._offline_phase == "first_pass" and self._offline_archive is not None:
            self._offline_archive.record_step(
                step.step_id,
                step.coordinate,
                step.mode == "actual",
            )

        self.stats.current_window = self._current_window
        self._last_completed_mode = step.mode
        self._last_completed_step_id = step.step_id
        self._step = None

    def abort_step(self, run_id: int, step_id: int) -> None:
        step = self._require_step(run_id, step_id)
        if self._run is not None and self._run.next_step_id == step.step_id + 1:
            self._run.next_step_id = step.step_id
        self._rollback_requested = False
        self._step = None

    def create_rollback_snapshot(self) -> RuntimeRollbackSnapshot:
        if self._run is None or self._step is not None:
            raise RuntimeError("rollback snapshots require an idle active run")
        return RuntimeRollbackSnapshot(
            next_step_id=self._run.next_step_id,
            forecaster=self.forecaster.snapshot(),
            history_topology=self._history_topology,
            history_labels=self._history_labels,
            current_window=self._current_window,
            consecutive_forecasts=self._consecutive_forecasts,
            required_actual_refreshes=self._required_actual_refreshes,
            required_feedback_actuals=self._required_feedback_actuals,
            disabled=self._disabled,
            disable_reason=self._disable_reason,
            experiment_disabled=self._experiment_disabled,
            experiment_disable_reason=self._experiment_disable_reason,
            last_completed_mode=self._last_completed_mode,
            last_completed_step_id=self._last_completed_step_id,
            stats=replace(self.stats),
        )

    def restore_rollback_snapshot(self, snapshot: RuntimeRollbackSnapshot) -> None:
        if self._run is None or self._step is not None:
            raise RuntimeError("rollback restoration requires an idle active run")
        if not isinstance(snapshot, RuntimeRollbackSnapshot):
            raise TypeError("snapshot must be a RuntimeRollbackSnapshot")
        current = self.stats
        speculative_calls = max(
            0, current.forecast_model_calls - snapshot.stats.forecast_model_calls
        )
        discarded_actual_calls = max(
            0, current.actual_transformer_calls - snapshot.stats.actual_transformer_calls
        )
        restored = replace(snapshot.stats)
        restored.forecast_model_calls += speculative_calls
        restored.actual_transformer_calls += discarded_actual_calls
        restored.speculative_forecast_calls += speculative_calls
        restored.discarded_actual_calls += discarded_actual_calls
        restored.rollback_count += 1
        for name in (
            "history_archive_seconds",
            "history_update_seconds",
            "forecast_prediction_seconds",
            "residual_measure_seconds",
            "residual_output_head_seconds",
            "offline_archive_seconds",
            "offline_smoother_build_seconds",
        ):
            setattr(
                restored,
                name,
                getattr(restored, name) + max(0.0, getattr(current, name) - getattr(snapshot.stats, name)),
            )
        for name in (
            "direct_history_updates",
            "residual_anchors",
            "residual_failures",
            "residual_skipped_terminal_probes",
            "rollback_suppressed_threshold",
            "rollback_suppressed_budget",
            "offline_replay_steps",
            "offline_replay_model_calls",
            "offline_replay_anchor_steps",
            "offline_replay_smoothed_steps",
        ):
            setattr(
                restored,
                name,
                getattr(restored, name) + max(0, getattr(current, name) - getattr(snapshot.stats, name)),
            )
        restored.residual_max_score = max(restored.residual_max_score, current.residual_max_score)
        restored.residual_max_video_score = max(
            restored.residual_max_video_score, current.residual_max_video_score
        )
        restored.residual_max_audio_score = max(
            restored.residual_max_audio_score, current.residual_max_audio_score
        )
        restored.residual_policy_max_score = max(
            restored.residual_policy_max_score, current.residual_policy_max_score
        )

        self._run.next_step_id = snapshot.next_step_id
        self.forecaster.restore(snapshot.forecaster)
        self._history_topology = snapshot.history_topology
        self._history_labels = snapshot.history_labels
        self._current_window = snapshot.current_window
        self._consecutive_forecasts = snapshot.consecutive_forecasts
        self._required_actual_refreshes = snapshot.required_actual_refreshes
        self._required_feedback_actuals = snapshot.required_feedback_actuals
        self._disabled = snapshot.disabled
        self._disable_reason = snapshot.disable_reason
        self._experiment_disabled = snapshot.experiment_disabled
        self._experiment_disable_reason = snapshot.experiment_disable_reason
        self._last_completed_mode = snapshot.last_completed_mode
        self._last_completed_step_id = snapshot.last_completed_step_id
        self._rollback_requested = False
        self._forced_actual_reason = None
        self._forced_actual_is_replay = False
        self.stats = restored

    def consume_rollback_request(self) -> bool:
        requested = self._rollback_requested
        self._rollback_requested = False
        return requested

    def force_next_actual(self, reason: str, *, rollback_replay: bool) -> None:
        if self._step is not None:
            raise RuntimeError("cannot force an actual step while another step is active")
        self._forced_actual_reason = str(reason)
        self._forced_actual_is_replay = bool(rollback_replay)

    def begin_rollback_replay(self) -> None:
        self._rollback_replay_active = True

    def end_rollback_replay(self) -> None:
        self._rollback_replay_active = False

    def debug_summary(self) -> str:
        offline_archive_device = (
            self._offline_archive.history_device
            if self._offline_archive is not None
            else None
        )
        return (
            f"run_id={self.stats.run_id} sampler={self.stats.sampler_name} "
            f"steps={self.stats.total_steps} actual_steps={self.stats.actual_steps} "
            f"forecast_steps={self.stats.forecast_steps} "
            f"actual_transformer_calls={self.stats.actual_transformer_calls} "
            f"forecast_calls={self.stats.forecast_model_calls} "
            f"fallbacks={self.stats.forecast_fallbacks} "
            f"bypassed_steps={self.stats.bypassed_steps} disabled={self.stats.disabled} "
            f"video_blend_weight={self.config.blend_weight:.6f} "
            f"audio_blend_weight={self.config.audio_blend_weight:.6f} "
            f"causal_video_blend_weight={self.stats.causal_video_blend_weight:.6f} "
            f"causal_audio_blend_weight={self.stats.causal_audio_blend_weight:.6f} "
            f"history_archive_s={self.stats.history_archive_seconds:.3f} "
            f"history_update_s={self.stats.history_update_seconds:.3f} "
            f"forecast_predict_s={self.stats.forecast_prediction_seconds:.3f} "
            f"residual_measure_s={self.stats.residual_measure_seconds:.3f} "
            f"residual_output_head_s={self.stats.residual_output_head_seconds:.3f} "
            f"residual_anchors={self.stats.residual_anchors} "
            f"residual_failures={self.stats.residual_failures} "
            f"residual_terminal_skips={self.stats.residual_skipped_terminal_probes} "
            f"residual_score_max={self.stats.residual_max_score:.6f} "
            f"residual_video_max={self.stats.residual_max_video_score:.6f} "
            f"residual_audio_max={self.stats.residual_max_audio_score:.6f} "
            f"residual_policy_max={self.stats.residual_policy_max_score:.6f} "
            f"feedback_threshold={_FEEDBACK_SCORE_THRESHOLD:.3f} "
            f"feedback_budget={_FEEDBACK_MAX_REFRESHES} "
            f"feedback_refreshes={self.stats.feedback_refreshes} "
            f"feedback_below_threshold={self.stats.feedback_suppressed_threshold} "
            f"feedback_budget_skips={self.stats.feedback_suppressed_budget} "
            f"speculative_calls={self.stats.speculative_forecast_calls} "
            f"discarded_actual_calls={self.stats.discarded_actual_calls} "
            f"rollbacks={self.stats.rollback_count} "
            f"rollback_threshold={_ROLLBACK_SCORE_THRESHOLD:.3f} "
            f"rollback_budget={_ROLLBACK_MAX_CORRECTIONS} "
            f"rollback_below_threshold={self.stats.rollback_suppressed_threshold} "
            f"rollback_budget_skips={self.stats.rollback_suppressed_budget} "
            f"replayed_transformer_calls={self.stats.replayed_transformer_calls} "
            f"offline_archive_s={self.stats.offline_archive_seconds:.3f} "
            f"offline_smoother_build_s={self.stats.offline_smoother_build_seconds:.3f} "
            f"offline_replay_steps={self.stats.offline_replay_steps} "
            f"offline_replay_calls={self.stats.offline_replay_model_calls} "
            f"offline_replay_anchor_steps={self.stats.offline_replay_anchor_steps} "
            f"offline_replay_smoothed_steps={self.stats.offline_replay_smoothed_steps} "
            f"offline_validation_samples_per_branch={self.stats.offline_validation_samples_per_branch} "
            f"offline_validation_anchors={self.stats.offline_validation_anchors} "
            f"offline_validation_streams={self.stats.offline_validation_streams} "
            f"offline_validation_s={self.stats.offline_validation_seconds:.3f} "
            f"offline_validation_audio_max={self.stats.offline_validation_audio_max:.6f} "
            f"offline_validation_video_max={self.stats.offline_validation_video_max:.6f} "
            f"offline_validation_packed_max={self.stats.offline_validation_packed_max:.6f} "
            f"offline_attenuated_predictions={self.stats.offline_attenuated_predictions} "
            f"offline_local_only_predictions={self.stats.offline_local_only_predictions} "
            f"offline_effective_blend_min={self.stats.offline_effective_blend_min:.6f} "
            f"offline_effective_blend_mean={self.stats.offline_effective_blend_mean:.6f} "
            f"offline_effective_blend_max={self.stats.offline_effective_blend_max:.6f} "
            f"offline_effective_audio_blend_min={self.stats.offline_effective_audio_blend_min:.6f} "
            f"offline_effective_audio_blend_mean={self.stats.offline_effective_audio_blend_mean:.6f} "
            f"offline_effective_audio_blend_max={self.stats.offline_effective_audio_blend_max:.6f} "
            f"offline_effective_video_blend_min={self.stats.offline_effective_video_blend_min:.6f} "
            f"offline_effective_video_blend_mean={self.stats.offline_effective_video_blend_mean:.6f} "
            f"offline_effective_video_blend_max={self.stats.offline_effective_video_blend_max:.6f} "
            f"offline_attenuated_audio_predictions={self.stats.offline_attenuated_audio_predictions} "
            f"offline_attenuated_video_predictions={self.stats.offline_attenuated_video_predictions} "
            f"offline_local_only_audio_predictions={self.stats.offline_local_only_audio_predictions} "
            f"offline_local_only_video_predictions={self.stats.offline_local_only_video_predictions} "
            f"offline_archive_mib={self.stats.offline_archive_bytes / (1024 * 1024):.1f} "
            f"offline_full_schedule_estimated_mib={self.stats.offline_estimated_archive_bytes / (1024 * 1024):.1f} "
            f"direct_history_updates={self.stats.direct_history_updates} "
            f"history_storage={self.config.history_storage} "
            f"offline_archive_storage={self.config.offline_archive_storage} "
            f"offline_archive_device={str(offline_archive_device)!r} "
            f"history_device={str(self.prediction_history_device)!r} "
            f"history_mib={self.prediction_history_tensor_bytes / (1024 * 1024):.1f} "
            f"reason={self.stats.disable_reason!r} "
            f"experimental_reason={self._experiment_disable_reason!r}"
        )

LOG = logging.getLogger(__name__)

BINDING_KEY = "spectrum_h3_binding"
RUNTIME_KEY = "spectrum_h3_runtime"
RUN_ID_KEY = "spectrum_h3_run_id"
STEP_ID_KEY = "spectrum_h3_step_id"
COORDINATE_KEY = "spectrum_h3_coordinate"
ACTUAL_KEY = "spectrum_h3_actual"
REASON_KEY = "spectrum_h3_reason"
WRAPPER_KEY = "spectrum_minimax_h3"
KJ_PREVIEW_WRAPPER_KEY = "kj_preview_override"

SUPPORTED_SINGLE_CALL_SAMPLERS = frozenset(
    {
        "_turbo_sampler",
        "sample_euler",
        "sample_er_sde",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

RES_MULTISTEP_SAMPLERS = frozenset(
    {
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    }
)

ER_SDE_SAMPLERS = frozenset({"sample_er_sde"})


@dataclass(slots=True)
class SpectrumH3Binding:
    runtime: SpectrumH3Runtime


def sampler_name(sampler: Any) -> str:
    function = getattr(sampler, "sampler_function", None)
    return str(getattr(function, "__name__", type(sampler).__name__))


def sampler_is_supported(sampler: Any) -> bool:
    return sampler_name(sampler) in SUPPORTED_SINGLE_CALL_SAMPLERS


def sampler_supports_seeded_replay(sampler: Any) -> bool:
    """Return whether a fresh invocation can reconstruct the sampler's random stream."""
    if not sampler_is_supported(sampler):
        return False
    if sampler_name(sampler) not in ER_SDE_SAMPLERS:
        return True

    options = getattr(sampler, "extra_options", {}) or {}
    if not isinstance(options, dict):
        return False
    return options.get("noise_sampler") is None and options.get("noise_scaler") is None


def max_consecutive_forecasts(sampler: Any) -> int | None:
    return 1 if sampler_is_supported(sampler) else None


def min_actual_steps_after_forecast(sampler: Any) -> int:
    name = sampler_name(sampler)
    return 1 if name in SUPPORTED_SINGLE_CALL_SAMPLERS else 0


def min_tail_actual_steps(sampler: Any) -> int:
    return 3 if sampler_name(sampler) in RES_MULTISTEP_SAMPLERS else 0


def _binding_from_model_options(model_options: dict[str, Any] | None) -> SpectrumH3Binding | None:
    binding = (model_options or {}).get(BINDING_KEY)
    return binding if isinstance(binding, SpectrumH3Binding) else None


def _copy_condition_structure(value: Any) -> Any:
    """Copy mutable conditioning containers while sharing tensor/model payloads."""
    if isinstance(value, dict):
        return {key: _copy_condition_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_condition_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_condition_structure(item) for item in value)
    return value


def _offline_progress_callbacks(callback, total_steps: int):
    """Report both passes while keeping previews and callback side effects replay-only."""
    if callback is None or total_steps <= 0:
        return None, callback, None

    import comfy.utils

    total_work = total_steps * 2
    progress = comfy.utils.ProgressBar(total_work)
    replay_finished = False

    def capture_callback(step, _x0, _x, _pass_steps):
        progress.update_absolute(step + 1, total_work)

    def replay_callback(step, x0, x, _pass_steps):
        nonlocal replay_finished
        callback(total_steps + step, x0, x, total_work)
        replay_finished = step + 1 >= total_steps

    def complete_progress():
        if not replay_finished:
            progress.update_absolute(total_work, total_work)

    return capture_callback, replay_callback, complete_progress


def copy_model_options_with_step(
    model_options: dict[str, Any] | None,
    runtime: SpectrumH3Runtime,
    decision: dict[str, Any],
) -> dict[str, Any]:
    copied = dict(model_options or {})
    transformer_options = dict(copied.get("transformer_options") or {})
    copied["transformer_options"] = transformer_options
    transformer_options[RUNTIME_KEY] = runtime
    transformer_options[RUN_ID_KEY] = int(decision["run_id"])
    transformer_options[STEP_ID_KEY] = int(decision["step_id"])
    transformer_options[COORDINATE_KEY] = float(decision["coordinate"])
    transformer_options[ACTUAL_KEY] = bool(decision["actual"])
    transformer_options[REASON_KEY] = str(decision["reason"])
    return copied


def outer_sample_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None:
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    transformer_options = (getattr(guider, "model_options", None) or {}).get("transformer_options") or {}
    if transformer_options.get("easycache") is not None:
        LOG.warning(
            "Spectrum H3 disabled for this run because EasyCache or LazyCache is active on the same model"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    runtime = binding.runtime
    name = sampler_name(sampler)

    def execute_run(
        run_noise,
        run_latent,
        run_sigmas,
        run_mask,
        run_callback,
        run_disable_pbar,
        *,
        phase: str,
        complete_offline_capture: bool = False,
    ):
        run_id = runtime.start_run(
            run_sigmas,
            name,
            supported_sampler=sampler_is_supported(sampler),
            max_consecutive_forecasts=max_consecutive_forecasts(sampler),
            min_actual_steps_after_forecast=min_actual_steps_after_forecast(sampler),
            min_tail_actual_steps=min_tail_actual_steps(sampler),
        )
        if name in ER_SDE_SAMPLERS and (
            runtime.config.anchor_residual_feedback
            or runtime.config.selective_rollback_correction
        ):
            runtime.disable_experiment(
                "ER-SDE is reviewed only for ordinary Spectrum and offline smoothing replay"
            )
        if runtime.config.debug:
            LOG.warning(
                "Spectrum H3 run start phase=%s run_id=%s sampler=%s steps=%s supported=%s",
                phase,
                run_id,
                name,
                runtime.stats.total_steps,
                runtime.supported_sampler,
            )
        capture_complete = False
        started = time.perf_counter()
        try:
            result = executor(
                run_noise,
                run_latent,
                sampler,
                run_sigmas,
                run_mask,
                run_callback,
                run_disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
            if complete_offline_capture:
                capture_complete = runtime.complete_offline_capture()
            return result, capture_complete
        finally:
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 run summary phase=%s wall_s=%.3f %s",
                    phase,
                    time.perf_counter() - started,
                    runtime.debug_summary(),
                )
            runtime.end_run(run_id)
            if runtime.config.debug:
                LOG.warning("Spectrum H3 run teardown phase=%s run_id=%s", phase, run_id)

    if not runtime.config.offline_smoothing_replay:
        result, _ = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            phase="single_pass",
        )
        return result

    if not sampler_is_supported(sampler):
        LOG.warning(
            "Spectrum H3 offline smoothing replay is unsupported for sampler %s; running one native pass",
            name,
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if not sampler_supports_seeded_replay(sampler):
        LOG.warning(
            "Spectrum H3 offline smoothing replay requires ER-SDE's native seeded "
            "noise_sampler and noise_scaler; running one native pass"
        )
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if not all(torch.is_tensor(value) for value in (noise, latent_image, sigmas)):
        LOG.warning(
            "Spectrum H3 offline smoothing replay requires tensor sampling inputs; running one ordinary pass"
        )
        result, _ = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            phase="single_pass_fallback",
        )
        return result

    replay_noise = noise.detach().clone()
    replay_latent = latent_image.detach().clone()
    replay_sigmas = sigmas.detach().clone()
    replay_mask = denoise_mask.detach().clone() if torch.is_tensor(denoise_mask) else denoise_mask
    initial_conds = _copy_condition_structure(guider.conds) if hasattr(guider, "conds") else None
    offline_steps = max(0, sigmas.numel() - 1)
    capture_callback, replay_callback, complete_progress = _offline_progress_callbacks(
        callback,
        offline_steps,
    )
    runtime.begin_offline_capture(total_steps=offline_steps, sampler_name=name)
    try:
        first_result, capture_complete = execute_run(
            noise,
            latent_image,
            sigmas,
            denoise_mask,
            capture_callback,
            disable_pbar,
            phase="offline_first_pass",
            complete_offline_capture=True,
        )
        if not capture_complete:
            if complete_progress is not None:
                complete_progress()
            reason = (
                runtime.offline_archive.failure_reason
                if runtime.offline_archive is not None
                else "offline archive was not retained"
            )
            LOG.warning(
                "Spectrum H3 offline replay skipped; returning the valid first-pass result: %s",
                reason,
            )
            return first_result

        runtime.begin_offline_replay()
        if initial_conds is not None:
            # CFGGuider.inner_sample replaces ``guider.conds`` with processed
            # conditions. Replay must begin from the same preprocessed input
            # structure as the first pass, not process its output a second time.
            guider.conds = _copy_condition_structure(initial_conds)
        try:
            replay_result, _ = execute_run(
                replay_noise,
                replay_latent,
                replay_sigmas,
                replay_mask,
                replay_callback,
                True,
                phase="offline_replay",
            )
            if complete_progress is not None:
                complete_progress()
        except OfflineReplayAbort as exc:
            if complete_progress is not None:
                complete_progress()
            LOG.warning(
                "Spectrum H3 offline replay aborted; returning the valid first-pass result: %s",
                exc,
            )
            return first_result
        return replay_result
    finally:
        runtime.release_offline_archive()


def predict_noise_wrapper(executor, x, timestep, model_options=None, seed=None):
    guider = executor.class_obj
    binding = _binding_from_model_options(getattr(guider, "model_options", None))
    if binding is None or binding.runtime.active_run_id is None or not binding.runtime.supported_sampler:
        return executor(x, timestep, model_options or {}, seed)

    if "multigpu_clones" in (model_options or {}):
        if binding.runtime.config.debug:
            LOG.warning("Spectrum H3 native fallback: multi-GPU parallel model calls are not transactionally supported")
        return executor(x, timestep, model_options or {}, seed)

    runtime = binding.runtime
    decision = runtime.begin_step(timestep)
    if runtime.config.debug:
        LOG.warning(
            "Spectrum H3 step run_id=%s step=%s coordinate=%.6f decision=%s reason=%s history=%s window=%.3f",
            decision["run_id"],
            decision["step_id"],
            decision["coordinate"],
            "actual" if decision["actual"] else "forecast",
            decision["reason"],
            runtime.prediction_history_length,
            runtime.stats.current_window,
        )

    def execute_attempt(attempt_decision: dict[str, Any]):
        patched = copy_model_options_with_step(model_options, runtime, attempt_decision)
        return executor(x, timestep, patched, seed)

    try:
        try:
            result = execute_attempt(decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
        except ForecastRetryActual as retry:
            runtime.prepare_actual_retry(decision["run_id"], decision["step_id"], str(retry))
            retry_decision = dict(decision)
            retry_decision["actual"] = True
            retry_decision["reason"] = f"forecast transaction retry: {retry}"
            if runtime.config.debug:
                LOG.warning(
                    "Spectrum H3 forecast retry run_id=%s step=%s reason=%s",
                    decision["run_id"],
                    decision["step_id"],
                    retry,
                )
            result = execute_attempt(retry_decision)
            runtime.finalize_step(decision["run_id"], decision["step_id"])
            return result
    except BaseException:
        if runtime.active_step_id == decision["step_id"]:
            runtime.abort_step(decision["run_id"], decision["step_id"])
        raise


def model_clone_callback(source_model: Any, cloned_model: Any) -> None:
    source_binding = _binding_from_model_options(getattr(source_model, "model_options", None))
    if source_binding is None:
        return
    if not hasattr(cloned_model, "model_options") or cloned_model.model_options is None:
        cloned_model.model_options = {}
    cloned_model.model_options[BINDING_KEY] = SpectrumH3Binding(
        SpectrumH3Runtime(source_binding.runtime.config)
    )


def _place_kj_preview_inside_offline_wrapper(model: Any, outer_wrapper_type: str) -> None:
    """Ensure KJ's observational preview wrapper is entered once for each offline pass."""
    outer_wrappers = (getattr(model, "wrappers", None) or {}).get(outer_wrapper_type)
    if not isinstance(outer_wrappers, dict):
        return
    keys = list(outer_wrappers)
    if KJ_PREVIEW_WRAPPER_KEY not in outer_wrappers or WRAPPER_KEY not in outer_wrappers:
        return
    if keys.index(KJ_PREVIEW_WRAPPER_KEY) > keys.index(WRAPPER_KEY):
        return

    preview_wrappers = outer_wrappers.pop(KJ_PREVIEW_WRAPPER_KEY)
    reordered = {}
    for key, wrappers in outer_wrappers.items():
        reordered[key] = wrappers
        if key == WRAPPER_KEY:
            reordered[KJ_PREVIEW_WRAPPER_KEY] = preview_wrappers
    outer_wrappers.clear()
    outer_wrappers.update(reordered)


def install_sampler_wrappers(model: Any, runtime: SpectrumH3Runtime) -> None:
    import comfy.patcher_extension

    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    model.model_options[BINDING_KEY] = SpectrumH3Binding(runtime)
    model.model_options.setdefault("transformer_options", {})

    wrapper_types = comfy.patcher_extension.WrappersMP
    existing_outer = model.get_wrappers(wrapper_types.OUTER_SAMPLE, WRAPPER_KEY)
    if not existing_outer:
        model.add_wrapper_with_key(wrapper_types.OUTER_SAMPLE, WRAPPER_KEY, outer_sample_wrapper)
    existing_predict = model.get_wrappers(wrapper_types.PREDICT_NOISE, WRAPPER_KEY)
    if not existing_predict:
        model.add_wrapper_with_key(wrapper_types.PREDICT_NOISE, WRAPPER_KEY, predict_noise_wrapper)
    callback_type = comfy.patcher_extension.CallbacksMP.ON_CLONE
    if not model.get_callbacks(callback_type, WRAPPER_KEY):
        model.add_callback_with_key(callback_type, WRAPPER_KEY, model_clone_callback)

LOG = logging.getLogger(__name__)
_H3_LABEL = (("minimax_h3_guidance_distilled",),)


def _target_ranges(args: dict[str, Any]) -> tuple[tuple[int, int, str], ...]:
    ranges = tuple(tuple(value) for value in args.get("target_ranges", ()))
    audio = tuple(value for value in ranges if value[2] == "audio")
    video = tuple(value for value in ranges if value[2] == "video")
    if len(audio) != 1 or len(video) != 1:
        raise ValueError("packed H3 layout must contain one target audio and video segment")
    return audio + video


class SpectrumH3BlockLoop:
    """Observe or replace the complete H3 transformer-stack target feature."""

    def __call__(
        self,
        args: dict[str, Any],
        extra_options: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        original = extra_options["original_block"]
        options = args["transformer_options"]
        runtime = options.get(RUNTIME_KEY)
        run_id = options.get(RUN_ID_KEY)
        step_id = options.get(STEP_ID_KEY)
        if not isinstance(runtime, SpectrumH3Runtime) or run_id is None or step_id is None:
            return original(args)

        try:
            ranges = _target_ranges(args)
        except ValueError as error:
            runtime.fallback_current_step(int(run_id), int(step_id), str(error))
            return original(args)

        hidden = args["img"]
        audio_rows = ranges[0][1] - ranges[0][0]
        video_rows = ranges[1][1] - ranges[1][0]
        expected_shape = (1, audio_rows + video_rows, hidden.shape[-1])
        topology = (
            ("target_audio_rows", audio_rows),
            ("target_video_rows", video_rows),
            ("hidden_width", int(hidden.shape[-1])),
            ("block_count", int(args.get("block_count", 0))),
        )
        call_id, actual = runtime.begin_model_call(
            int(run_id),
            int(step_id),
            topology=topology,
            labels=_H3_LABEL,
            expected_shape=expected_shape,
        )
        if actual:
            output = original(args)["img"]
            feature = torch.cat(
                [output[start:end] for start, end, _kind in ranges], dim=0
            ).unsqueeze(0)
            runtime.observe_actual(int(run_id), int(step_id), call_id, feature)
            return {"img": output}

        predicted = runtime.predict(
            int(run_id),
            int(step_id),
            call_id,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        if predicted is None:
            output = original(args)["img"]
            feature = torch.cat(
                [output[start:end] for start, end, _kind in ranges], dim=0
            ).unsqueeze(0)
            runtime.observe_actual(int(run_id), int(step_id), call_id, feature)
            return {"img": output}
        if predicted.shape != expected_shape or not torch.isfinite(predicted).all():
            reason = "forecasted H3 target feature is invalid"
            if runtime.offline_phase == "replay":
                raise OfflineReplayAbort(reason)
            runtime.fallback_current_step(int(run_id), int(step_id), reason)
            return original(args)

        result = hidden.clone()
        offset = 0
        compact = predicted[0]
        for start, end, _kind in ranges:
            length = end - start
            result[start:end] = compact[offset : offset + length]
            offset += length
        return {"img": result}


def patch_minimax_h3_spectrum_model(model: Any, config: SpectrumH3Config) -> Any:
    """Return an H3 Spectrum clone using Core-managed object patch lifecycle."""

    config.validate()
    if config.anchor_residual_feedback or config.selective_rollback_correction:
        raise ValueError("This Spectrum node does not include feedback or rollback research modes.")
    if getattr(model, "model_options", {}).get(MINIMAX_H3_CACHE_OWNER_KEY):
        raise ValueError("MiniMax H3 Spectrum cannot be combined with MiniMax H3 Cache.")
    if getattr(model, "model_options", {}).get(MINIMAX_H3_PDD_OWNER_KEY):
        raise ValueError("MiniMax H3 Spectrum cannot be combined with MiniMax H3 PDD.")
    patched = model.clone()
    diffusion_model = patched.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 Spectrum requires a MiniMax H3 diffusion model.")
    patched.model_options[MINIMAX_H3_SPECTRUM_OWNER_KEY] = True
    runtime = SpectrumH3Runtime(config)
    bound_forward = types.MethodType(minimax_h3_block_patch_forward, diffusion_model)
    patched.add_object_patch("diffusion_model._forward", bound_forward)
    patched.set_model_patch_replace(SpectrumH3BlockLoop(), "dit", "block_loop", 0)
    install_sampler_wrappers(patched, runtime)
    return patched


@dataclass(frozen=True, slots=True)
class MiniMaxH3RadialAttentionConfig:
    dense_blocks: int = 1
    dense_start_steps: int = 1
    dense_end_steps: int = 1
    block_size: int = 128
    decay_factor: float = 0.2
    allow_compile: bool = False

    def validate(self) -> MiniMaxH3RadialAttentionConfig:
        if self.dense_blocks < 0:
            raise ValueError("Dense block count must be zero or greater.")
        if self.dense_start_steps < 0 or self.dense_end_steps < 0:
            raise ValueError("Dense step counts must be zero or greater.")
        if self.block_size not in (64, 128):
            raise ValueError("MiniMax H3 Radial block size must be 64 or 128.")
        if not math.isfinite(self.decay_factor) or not 0.0 <= self.decay_factor <= 1.0:
            raise ValueError("MiniMax H3 Radial decay factor must be finite and in [0, 1].")
        if not isinstance(self.allow_compile, bool):
            raise TypeError("MiniMax H3 Radial allow_compile must be a boolean.")
        return self


UNIFIED_ATTENTION_MODES = (
    "disabled",
    "FlashAttention",
    "SageAttention",
    "Sparse / MiniMax H3 Radial",
)

CUSTOM_SAGE_MODES = (
    "auto",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++",
    "sageattn3",
    "sageattn3_per_block_mean",
)


def _call_attention_function(attention_function: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def override(_original: Callable[..., torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        wrapped = getattr(attention_function, "__wrapped__", attention_function)
        return wrapped(*args, **kwargs)

    return override


def _attention_as_nhd(q, k, v, heads: int, already_hnd: bool):
    if already_hnd:
        batch, _, _, head_dim = q.shape
        return (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), batch, head_dim
    batch, _, hidden_size = q.shape
    head_dim = hidden_size // heads
    return (q.reshape(batch, -1, heads, head_dim), k.reshape(batch, -1, heads, head_dim), v.reshape(batch, -1, heads, head_dim)), batch, head_dim


def _attention_from_nhd(output, batch: int, heads: int, head_dim: int, hnd_output: bool):
    return output.transpose(1, 2) if hnd_output else output.reshape(batch, -1, heads * head_dim)


def _sage_mask(mask):
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask[None]
    return mask[:, None] if mask.ndim == 3 else mask


def _make_sage_backend(mode: str, allow_compile: bool) -> Callable[..., torch.Tensor]:
    if mode == "auto":
        from sageattention import sageattn

        def sage_function(q, k, v, **kwargs):
            return sageattn(q, k, v, **kwargs)
    elif mode == "sageattn_qk_int8_pv_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def sage_function(q, k, v, **kwargs):
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, pv_accum_dtype="fp32", **kwargs)
    elif mode == "sageattn_qk_int8_pv_fp16_triton":
        from sageattention import sageattn_qk_int8_pv_fp16_triton

        def sage_function(q, k, v, **kwargs):
            return sageattn_qk_int8_pv_fp16_triton(q, k, v, pv_accum_dtype="fp32", **kwargs)
    elif mode in ("sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++"):
        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        accumulation = "fp32+fp16" if mode.endswith("++") else "fp32+fp32"

        def sage_function(q, k, v, **kwargs):
            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, pv_accum_dtype=accumulation, **kwargs)
    elif mode in ("sageattn3", "sageattn3_per_block_mean"):
        from sageattn3 import sageattn3_blackwell

        def sage_function(q, k, v, **kwargs):
            tensor_layout = kwargs.pop("tensor_layout", "NHD")
            if tensor_layout == "NHD":
                q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
            result = sageattn3_blackwell(
                q,
                k,
                v,
                per_block_mean=mode == "sageattn3_per_block_mean",
                **kwargs,
            )
            return result.transpose(1, 2) if tensor_layout == "NHD" else result
    else:
        raise ValueError(f"Unsupported custom SageAttention mode: {mode}")

    if not allow_compile:
        sage_function = torch.compiler.disable()(sage_function)

    def attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if kwargs.get("low_precision_attention", True) is False:
            return comfy.ldm.modules.attention.attention_pytorch.__wrapped__(
                q, k, v, heads, mask=mask, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs
            )
        input_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        (q, k, v), batch, head_dim = _attention_as_nhd(q, k, v, heads, skip_reshape)
        output = sage_function(q, k, v, attn_mask=_sage_mask(mask), is_causal=False, tensor_layout="NHD").to(input_dtype)
        return _attention_from_nhd(output, batch, heads, head_dim, skip_output_reshape)

    return attention


def _make_flash_backend(allow_compile: bool, cast_dtype: torch.dtype) -> Callable[..., torch.Tensor]:
    try:
        from flash_attn import flash_attn_func
        is_v3 = False
    except ImportError:
        try:
            from flash_attn_interface import flash_attn_func
            is_v3 = True
        except ImportError as error:
            raise RuntimeError("Custom FlashAttention requires flash_attn or flash_attn_interface.") from error

    def flash_function(q, k, v):
        result = flash_attn_func(q, k, v, causal=False) if is_v3 else flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        return result[0] if isinstance(result, tuple) else result

    if not allow_compile:
        flash_function = torch.compiler.disable()(flash_function)

    def attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **_kwargs):
        if mask is not None:
            raise RuntimeError("Custom FlashAttention does not support attention masks.")
        input_dtype = v.dtype
        if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
            q, k, v = q.to(cast_dtype), k.to(cast_dtype), v.to(cast_dtype)
        (q, k, v), batch, head_dim = _attention_as_nhd(q, k, v, heads, skip_reshape)
        output = flash_function(q, k, v).to(input_dtype)
        return _attention_from_nhd(output, batch, heads, head_dim, skip_output_reshape)

    return attention


def _model_compute_dtype(model: Any) -> torch.dtype:
    diffusion_model = model.get_model_object("diffusion_model")
    get_dtype = getattr(diffusion_model, "get_dtype_inference", None)
    dtype = get_dtype() if callable(get_dtype) else torch.float16
    return dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16


def _ensure_transformer_options(model: Any) -> dict[str, Any]:
    options = dict(getattr(model, "model_options", {}).get("transformer_options", {}))
    model.model_options["transformer_options"] = options
    return options


def _h3_sage_forward(self, x, rope_freqs=None, transformer_options=None):
    if x.device.type != "cuda":
        raise RuntimeError("MiniMax H3 memory optimizations require CUDA.")
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
    except ImportError as error:
        raise RuntimeError("MiniMax H3 memory optimizations require SageAttention.") from error

    token_count = x.shape[0]
    projected = self.qkv_proj(x).reshape(token_count, 3, self.heads, self.head_dim)
    q, k, v = (projected[:, index].unsqueeze(0) for index in range(3))
    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)
    attended = sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=False, tensor_layout="NHD", pv_accum_dtype="fp32+fp32")
    return self.out_proj(attended.to(x.dtype).flatten(2).squeeze(0))


def _h3_sparse_step(transformer_options: dict[str, Any], timestep: torch.Tensor, config: MiniMaxH3RadialAttentionConfig) -> bool:
    sigmas = transformer_options.get("sample_sigmas")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("MiniMax H3 Radial requires the sampler sigma schedule.")
    current = timestep.flatten()[0].to(sigmas.dtype)
    if current <= sigmas[-1]:
        return False
    step_index = torch.searchsorted(-sigmas, -current, right=True) - 1
    step_index = int(step_index.clamp(0, len(sigmas) - 2).item())
    return config.dense_start_steps <= step_index < len(sigmas) - 1 - config.dense_end_steps


def _h3_radial_block_mask(state: dict[str, Any], device: torch.device) -> torch.Tensor:
    block_size = state["config"].block_size
    total_blocks = math.ceil(state["sequence_length"] / block_size)
    plan = torch.ones((total_blocks, total_blocks), dtype=torch.bool, device=device)
    video_start, video_end = state["video_start"], state["video_end"]
    first_video_block = (video_start + block_size - 1) // block_size
    last_video_block = video_end // block_size - 1
    if last_video_block < first_video_block:
        return plan

    plan[first_video_block : last_video_block + 1, first_video_block : last_video_block + 1] = False
    frame_tokens = state["tokens_per_frame"]
    frame_count = state["frame_count"]

    def describe(block_index: int):
        start = max(video_start, block_index * block_size) - video_start
        end = min(video_end, (block_index + 1) * block_size) - video_start
        return min(frame_count - 1, start // frame_tokens), start % frame_tokens, (end - 1) % frame_tokens

    for output_block in range(first_video_block, last_video_block + 1):
        output_frame, output_left, output_right = describe(output_block)
        for input_block in range(first_video_block, last_video_block + 1):
            input_frame, input_left, input_right = describe(input_block)
            frame_distance = abs(output_frame - input_frame)
            if frame_distance == 0:
                plan[output_block, input_block] = True
                continue
            radius = frame_tokens // 2 if frame_distance == 1 else max(
                block_size,
                int((2 ** frame_tokens.bit_length()) * state["config"].decay_factor / 2**frame_distance),
            )
            plan[output_block, input_block] = output_left <= input_right + radius and input_left <= output_right + radius
    return plan


def _convert_sparse_mask(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    compute_capability = "sm{}{}".format(*torch.cuda.get_device_capability(mask.device))
    if block_size == 128:
        expand_axis = 0 if compute_capability == "sm90" else 1
        return mask.repeat_interleave(2, dim=expand_axis)
    rows, columns = mask.shape
    if compute_capability == "sm90":
        return mask.reshape(rows, columns // 2, 2).any(dim=2)
    return mask.reshape(rows // 2, 2, columns).any(dim=1)


def _run_h3_radial_attention(self, x, rope_freqs, state: dict[str, Any]) -> torch.Tensor:
    if x.device.type != "cuda":
        raise RuntimeError("MiniMax H3 Radial requires CUDA.")
    try:
        from spas_sage_attn import block_sparse_sage2_attn_cuda
    except ImportError as error:
        raise RuntimeError("MiniMax H3 Radial requires spas_sage_attn.") from error

    sequence_length = x.shape[0]
    block_size = state["config"].block_size
    padded_length = math.ceil(sequence_length / block_size) * block_size
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    q = q.view(1, sequence_length, self.heads, self.head_dim)
    k = k.view(1, sequence_length, self.heads, self.head_dim)
    v = v.view(1, sequence_length, self.heads, self.head_dim)
    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)
    if padded_length != sequence_length:
        padding = (0, 0, 0, 0, 0, padded_length - sequence_length)
        q, k, v = (torch.nn.functional.pad(tensor, padding) for tensor in (q, k, v))
    mask = state.setdefault("masks", {}).get((x.device, padded_length))
    if mask is None:
        mask = _convert_sparse_mask(_h3_radial_block_mask(state, x.device), block_size)
        state["masks"][(x.device, padded_length)] = mask
    mask_id = mask.unsqueeze(0).unsqueeze(0).expand(1, self.heads, -1, -1).to(torch.int8)
    output = block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask_id, tensor_layout="NHD", output_dtype=x.dtype)
    return self.out_proj(output[:, :sequence_length].to(x.dtype).reshape(sequence_length, self.heads * self.head_dim))


def _make_h3_radial_attention_forward(original: Callable[..., torch.Tensor], config: MiniMaxH3RadialAttentionConfig):
    def forward(self, x, rope_freqs=None, transformer_options=None):
        transformer_options = {} if transformer_options is None else transformer_options
        state = transformer_options.get(MINIMAX_H3_RADIAL_STATE_KEY)
        if state is None or not _h3_sparse_step(transformer_options, state["timestep"], config):
            return original(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return _run_h3_radial_attention(self, x, rope_freqs, state)

    return forward


def _make_h3_radial_wrapper(config: MiniMaxH3RadialAttentionConfig):
    def wrapper(executor, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
        transformer_options = {} if transformer_options is None else transformer_options
        model = executor.class_obj
        video_x, audio_x = x
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
        latent_t, latent_h, latent_w = video_x.shape[2:5]
        payload = minimax_payload or {}
        layout = payload.get("layout")
        signature = (context.shape[1], latent_t, latent_h, latent_w, audio_x.shape[-1])
        if layout is None or layout.signature != signature:
            layout = minimax_model.PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
        video_start, video_end, _ = next(segment for segment in layout.segments if segment[2] == "video")
        options = dict(transformer_options)
        options[MINIMAX_H3_RADIAL_STATE_KEY] = {
            "config": config,
            "timestep": timestep,
            "sequence_length": layout.seq_len,
            "video_start": video_start,
            "video_end": video_end,
            "frame_count": latent_t,
            "tokens_per_frame": (video_end - video_start) // latent_t,
            "masks": {},
        }
        return executor(x, timestep, context, options, minimax_payload=minimax_payload, **kwargs)

    return wrapper


def _patch_h3_radial_attention(model: Any, config: MiniMaxH3RadialAttentionConfig) -> Any:
    config.validate()
    patched = model.clone()
    diffusion_model = patched.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 Radial requires a MiniMax H3 diffusion model.")
    attention_head_dims = {block.attn.head_dim for block in diffusion_model.blocks}
    if not attention_head_dims <= {64, 128}:
        raise ValueError("MiniMax H3 Radial requires a 64 or 128 dimension attention head.")
    options = _ensure_transformer_options(patched)
    options[UNIFIED_ATTENTION_OWNER_KEY] = "Sparse / MiniMax H3 Radial"
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MINIMAX_H3_RADIAL_WRAPPER_KEY)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, MINIMAX_H3_RADIAL_WRAPPER_KEY, _make_h3_radial_wrapper(config))
    for index, block in enumerate(diffusion_model.blocks[config.dense_blocks :], start=config.dense_blocks):
        original = block.attn.forward
        patched_forward = _make_h3_radial_attention_forward(original, config)
        if config.allow_compile:
            patched_forward = torch.compile(patched_forward, dynamic=True)
        patched_forward = types.MethodType(patched_forward, block.attn)
        patched.add_object_patch(f"diffusion_model.blocks.{index}.attn.forward", patched_forward)
    return patched


def patch_unified_attention_model(model: Any, attention_mode: dict[str, Any]) -> Any:
    mode = attention_mode.get("attention_mode")
    if mode not in UNIFIED_ATTENTION_MODES:
        raise ValueError("Choose a supported attention mode.")
    if mode == "disabled":
        return model
    if mode == "Sparse / MiniMax H3 Radial":
        config = attention_mode.get("minimax_h3_radial_config")
        if not isinstance(config, MiniMaxH3RadialAttentionConfig):
            raise ValueError("Sparse / MiniMax H3 Radial requires MiniMax H3 Radial Attention Config.")
        return _patch_h3_radial_attention(model, config)

    patched = model.clone()
    options = _ensure_transformer_options(patched)
    options[UNIFIED_ATTENTION_OWNER_KEY] = mode
    if mode == "FlashAttention":
        options["optimized_attention_override"] = _call_attention_function(
            _make_flash_backend(bool(attention_mode.get("allow_compile", False)), _model_compute_dtype(patched))
        )
    elif mode == "SageAttention":
        h3_memory_optimizations = bool(attention_mode.get("h3_memory_optimizations", False))
        options["optimized_attention_override"] = _call_attention_function(
            _make_sage_backend(attention_mode.get("sage_mode", "auto"), bool(attention_mode.get("allow_compile", False)))
        )
        if h3_memory_optimizations:
            diffusion_model = patched.get_model_object("diffusion_model")
            if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
                raise ValueError("MiniMax H3 memory optimizations require a MiniMax H3 diffusion model.")
            for index, block in enumerate(diffusion_model.blocks):
                patched_forward = types.MethodType(_h3_sage_forward, block.attn)
                patched.add_object_patch(f"diffusion_model.blocks.{index}.attn.forward", patched_forward)
    return patched


def load_ideogram4_debanner_directions() -> dict[int, torch.Tensor]:
    if not IDEOGRAM4_DEBANNER_BUNDLE.is_file():
        raise FileNotFoundError(f"Ideogram 4 debanner bundle is missing: {IDEOGRAM4_DEBANNER_BUNDLE}")
    with safe_open(str(IDEOGRAM4_DEBANNER_BUNDLE), framework="pt", device="cpu") as handle:
        return {index: handle.get_tensor(f"direction.block_{index:02d}") for index in range(25, 29)}


def _validate_ideogram4_debanner_directions(directions: dict[int, torch.Tensor]) -> None:
    for index in range(25, 29):
        direction = directions.get(index)
        if not torch.is_tensor(direction) or tuple(direction.shape) != (1, 8, 8, 4608):
            raise ValueError(f"Ideogram 4 debanner direction for block {index} must have shape [1, 8, 8, 4608].")


def _make_ideogram4_debanner_wrapper(strength: float):
    def wrapper(executor, x, timesteps, context=None, attention_mask=None, transformer_options=None, **kwargs):
        _, _, grid_height, grid_width = x.shape
        options = {} if transformer_options is None else dict(transformer_options)
        options[IDEOGRAM4_DEBANNER_STATE_KEY] = {
            "grid_height": grid_height,
            "grid_width": grid_width,
            "image_token_count": grid_height * grid_width,
            "strength": strength,
            "is_conditional": context is not None,
        }
        return executor(x, timesteps, context, attention_mask, options, **kwargs)

    return wrapper


def _make_ideogram4_debanner_forward(original: Callable[..., torch.Tensor], direction: torch.Tensor):
    def forward(self, x, attn_mask, freqs_cis, adaln_input, transformer_options={}):
        output = original(x, attn_mask, freqs_cis, adaln_input, transformer_options=transformer_options)
        state = transformer_options.get(IDEOGRAM4_DEBANNER_STATE_KEY)
        if state is None or not state["is_conditional"]:
            return output

        grid_height = state["grid_height"]
        grid_width = state["grid_width"]
        image_token_count = state["image_token_count"]
        image_offset = output.shape[1] - image_token_count
        spatial_direction = torch.nn.functional.interpolate(
            direction.permute(0, 3, 1, 2), size=(grid_height, grid_width), mode="nearest"
        ).permute(0, 2, 3, 1).reshape(1, image_token_count, -1).to(device=output.device, dtype=output.dtype)
        image_tokens = output[:, image_offset:]
        original_norm = torch.linalg.vector_norm(image_tokens, dim=-1, keepdim=True)
        corrected = torch.nn.functional.normalize(image_tokens - state["strength"] * spatial_direction, p=2, dim=-1) * original_norm
        return torch.cat((output[:, :image_offset], corrected), dim=1)

    return forward


def patch_ideogram4_debanner_model(model: Any, directions: dict[int, torch.Tensor], strength: float) -> Any:
    strength = float(strength)
    if strength < 0.0:
        raise ValueError("Ideogram 4 debanner strength must be nonnegative.")
    _validate_ideogram4_debanner_directions(directions)

    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, ideogram4_model.Ideogram4Transformer2DModel):
        raise ValueError("Ideogram 4 Debanner requires an Ideogram4Transformer2DModel diffusion model.")

    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        IDEOGRAM4_DEBANNER_WRAPPER_KEY,
        _make_ideogram4_debanner_wrapper(strength),
    )
    for index in range(25, 29):
        block = diffusion_model.layers[index]
        patched_forward = types.MethodType(_make_ideogram4_debanner_forward(block.forward, directions[index]), block)
        patched.add_object_patch(f"diffusion_model.layers.{index}.forward", patched_forward)
    return patched


def patch_ideogram4_debanner(model: Any, strength: float) -> Any:
    return patch_ideogram4_debanner_model(model, load_ideogram4_debanner_directions(), strength)
