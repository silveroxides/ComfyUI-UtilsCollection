"""Invocation-owned H3 processing cache, stored through UnifiedEfficientLoader."""

import hashlib
import json
import logging
import os
import threading
import time
import types
import uuid
import weakref
from collections import Counter
from enum import Enum
from fractions import Fraction
from pathlib import Path
from struct import error as StructError

import torch
from safetensors import SafetensorError
from unifiedefficientloader import (
    IncrementalSafetensorsWriter,
    UnifiedSafetensorsLoader,
    dict_to_tensor,
    tensor_to_dict,
)

import folder_paths
import comfy.model_management


_IDENTITIES = {}
H3_CACHE_MODES = ("disabled", "video_only", "images_only", "all")
_IDENTITY_LOCK = threading.RLock()
_MISSING = object()
_STAGES = frozenset(("encoded_section", "vae_encode"))
_STORAGE_ERRORS = (OSError, ValueError, TypeError, KeyError, UnicodeError, SafetensorError, StructError, OverflowError, MemoryError, RuntimeError)


def _filter_h3_cache_write_log(record):
    if record.levelno != logging.INFO:
        return True
    message = record.getMessage().replace("\\", "/")
    return not (message.startswith("Finalized '") and "/utilscollection_h3_encoder_cache/" in message)


# UEL has no per-writer quiet option; retain all non-cache output and errors.
logging.getLogger("unifiedefficientloader").addFilter(_filter_h3_cache_write_log)


def lifetime_identity(value):
    """Never retain a model, nor serialize a recyclable Python address as its key."""
    address = id(value)
    with _IDENTITY_LOCK:
        existing = _IDENTITIES.get(address)
        if existing is not None and existing[0]() is value:
            return existing[1]

        def discard(reference):
            with _IDENTITY_LOCK:
                current = _IDENTITIES.get(address)
                if current is not None and current[0] is reference:
                    del _IDENTITIES[address]

        reference = weakref.ref(value, discard)
        identity = uuid.uuid4().hex
        _IDENTITIES[address] = (reference, identity)
        return identity


def tensor_fingerprint(value):
    """Hash logical bytes in first-axis slices, including BF16 without a cast."""
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode("ascii"))
    if value.numel():
        slices = value.unbind(0) if value.ndim else (value,)
        for part in slices:
            raw = part.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest.hexdigest()}


def describe_value(value, describe_tensor=tensor_fingerprint):
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, Fraction):
        return ["fraction", value.numerator, value.denominator]
    if isinstance(value, (torch.dtype, torch.device, uuid.UUID)):
        return [type(value).__name__, str(value)]
    if isinstance(value, Enum):
        return ["enum", type(value).__module__, type(value).__qualname__, describe_value(value.value)]
    if torch.is_tensor(value):
        return ["tensor", describe_tensor(value)]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [describe_value(item, describe_tensor) for item in value]]
    if isinstance(value, dict):
        items = [[describe_value(key, describe_tensor), describe_value(item, describe_tensor)] for key, item in value.items()]
        items.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return ["dict", items]
    raise TypeError(f"Unsupported H3 cache dependency: {type(value).__name__}")


def fingerprint(value):
    return hashlib.sha256(json.dumps(describe_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _configuration(value):
    """Patcher payloads are lifetime-owned weights, not media to rehash."""
    if value is None or isinstance(value, (str, bool, int, float, Fraction, Enum, torch.dtype, torch.device, uuid.UUID)):
        return describe_value(value)
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_configuration(item) for item in value]]
    if isinstance(value, dict):
        items = [[_configuration(key), _configuration(item)] for key, item in value.items()]
        return ["dict", sorted(items, key=lambda item: json.dumps(item[0], sort_keys=True))]
    if isinstance(value, types.MethodType):
        return ["method", lifetime_identity(value.__self__), lifetime_identity(value.__func__)]
    return [type(value).__module__, type(value).__qualname__, lifetime_identity(value)]


def _hook_description(group, active=False):
    if group is None:
        return None
    return [{
        "ref": _configuration(hook.hook_ref),
        "type": _configuration(hook.hook_type),
        "scope": _configuration(hook.hook_scope),
        "clip_strength": getattr(hook, "_strength_clip", None),
        "keyframes": [(frame.strength, frame.start_percent, frame.guarantee_steps)
                      for frame in hook.hook_keyframe.keyframes],
        **({"active_strength": hook.strength} if active else {}),
    } for hook in group.hooks]


def patcher_description(patcher, owned_methods=None):
    if patcher is None:
        return None
    owned_methods = owned_methods or {}
    objects = {}
    for name, value in getattr(patcher, "object_patches", {}).items():
        objects[name] = ["projection_method", name] if value is owned_methods.get(name) else _configuration(value)
    result = {"objects": objects}
    for name in ("patches_uuid", "patches", "model_options", "weight_wrapper_patches", "hook_patches",
                 "wrappers", "callbacks", "injections", "load_device", "force_cast_weights"):
        result[name] = _configuration(getattr(patcher, name, None))
    result["additional_models"] = {
        name: [[lifetime_identity(model.model), patcher_description(model)] for model in models]
        for name, models in getattr(patcher, "additional_models", {}).items()
    }
    result["forced_hooks"] = _hook_description(getattr(patcher, "forced_hooks", None))
    return result


def clip_description(clip):
    projection = None
    identity_method = getattr(type(clip), "h3_cache_identity", None)
    if identity_method is not None:
        projection = identity_method(clip)
        base = projection["base"]
    else:
        base = clip
    result = {
        "model": lifetime_identity(base.cond_stage_model),
        "tokenizer": lifetime_identity(base.tokenizer),
        "patcher": patcher_description(getattr(base, "patcher", None), projection["owned_methods"] if projection else None),
        "options": {name: _configuration(getattr(base, name, None))
                    for name in ("layer_idx", "tokenizer_options", "use_clip_schedule", "apply_hooks_to_conds")},
    }
    if projection:
        result["projection"] = {
            "model": lifetime_identity(projection["model"]),
            "patcher": patcher_description(projection["patcher"]),
            "name": projection["name"], "source": projection["source"], "tap": projection["tap"],
            "original_methods": _configuration(projection["original_methods"]),
        }
    return result


def spatial_cache_settings(config):
    method = config.get("visual_fusion_method", "spatial-checkerboard")
    result = {"method": method}
    if method == "linear":
        return result
    perturbation = config.get("spatial_perturbation", 0.0)
    result["spatial_perturbation"] = perturbation
    if perturbation or method == "spatial-dither-random":
        result["seed"] = config.get("seed", 0)
    if method == "spatial-block-interleave":
        result["visual_block_size"] = config.get("visual_block_size", 2)
    if method == "spatial-dither-random":
        secondary = config.get("dither_secondary_pattern", "checkerboard")
        ratio = config.get("dither_ratio", 0.5)
        result.update(dither_ratio=ratio, dither_secondary_pattern=secondary,
                      dither_mask_cleanup=bool(config.get("dither_mask_cleanup", False)) and 0 < ratio < 1)
        if secondary == "block-interleave":
            result["visual_block_size"] = config.get("visual_block_size", 2)
    return result


def temporal_cache_settings(method, settings, visual_config):
    if method == "spatial":
        return spatial_cache_settings(visual_config)
    result = {name: settings[name] for name in ("blend_method", "global_scale", "preserve_common_prefix")}
    if settings["blend_method"] == "linear":
        return result
    for name in ("consensus_type", "alignment_method", "similarity_threshold", "power_alpha", "diversity_beta", "rescale_norm"):
        result[name] = settings[name]
    if settings["alignment_method"] == "similarity":
        for name in ("alignment_threshold", "position_weight", "dynamic_similarity_contrast"):
            result[name] = settings[name]
        if settings["diversity_beta"] > 0:
            result["soft_comfort_bandpass"] = settings["soft_comfort_bandpass"]
    return result


def _pack(value, key, tensors):
    if torch.is_tensor(value):
        name = f"{key}.tensor.{len(tensors):06d}"
        tensors[name] = value.detach()
        return {"kind": "tensor", "key": name, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [_pack(item, key, tensors) for item in value]}
    if isinstance(value, dict) and all(isinstance(name, str) for name in value):
        return {"kind": "dict", "items": {name: _pack(item, key, tensors) for name, item in value.items()}}
    raise TypeError(f"Unsupported H3 cache result: {type(value).__name__}")


def _tensor_specs(tree):
    if not isinstance(tree, dict):
        raise ValueError("Invalid H3 cache result structure")
    kind = tree["kind"]
    if kind == "tensor":
        return [tree]
    if kind == "scalar":
        if tree["value"] is not None and not isinstance(tree["value"], (str, bool, int, float)):
            raise ValueError("Invalid H3 cache scalar")
        return []
    if kind not in ("dict", "list", "tuple"):
        raise ValueError("Invalid H3 cache result structure")
    if kind == "dict" and (not isinstance(tree["items"], dict) or not all(isinstance(key, str) for key in tree["items"])):
        raise ValueError("Invalid H3 cache dictionary")
    if kind != "dict" and not isinstance(tree["items"], list):
        raise ValueError("Invalid H3 cache sequence")
    values = tree["items"].values() if kind == "dict" else tree["items"]
    return [spec for value in values for spec in _tensor_specs(value)]


def _unpack(tree, tensors):
    kind = tree["kind"]
    if kind == "tensor":
        return tensors[tree["key"]].to(device=tree["device"])
    if kind == "scalar":
        return tree["value"]
    if kind == "dict":
        return {key: _unpack(value, tensors) for key, value in tree["items"].items()}
    values = [_unpack(value, tensors) for value in tree["items"]]
    return tuple(values) if kind == "tuple" else values


def _contains_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


class H3EncoderCache:
    def __init__(self, enable_caching="all"):
        if enable_caching not in H3_CACHE_MODES:
            raise ValueError(f"Unsupported H3 caching mode: {enable_caching}")
        self.enable_caching = enable_caching
        self.root = Path(folder_paths.get_temp_directory()) / "utilscollection_h3_encoder_cache" / "v2"
        self.hits = Counter()
        self.misses = Counter()
        self.timings = Counter()
        self._warned = False
        self._active = True
        self._clip_identity = None
        self._tensor_descriptions = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._active = False
        self._tensor_descriptions.clear()
        self._clip_identity = None
        logging.debug("H3 encoder cache: hits=%s misses=%s seconds=%s", dict(self.hits), dict(self.misses), dict(self.timings))

    def _warning(self, error):
        if not self._warned:
            logging.warning("H3 disk cache unavailable for an entry; computing normally (%s).", type(error).__name__)
            self._warned = True

    def _describe_tensor(self, tensor):
        # Inference tensors have no version counter: hash them each time rather
        # than treating their identity as evidence that their contents match.
        try:
            revision = tensor._version
        except RuntimeError:
            return tensor_fingerprint(tensor)
        address = id(tensor)
        previous = self._tensor_descriptions.get(address)
        if previous is not None and previous[0]() is tensor and previous[1] == revision:
            return previous[2]
        description = tensor_fingerprint(tensor)
        self._tensor_descriptions[address] = (weakref.ref(tensor), revision, description)
        return description

    def _read(self, path, stage, key, dependencies):
        with UnifiedSafetensorsLoader(str(path), low_memory=True) as loader:
            expected = {"uc_h3_cache_version": "2", "uc_h3_cache_stage": stage, "uc_h3_cache_key": key}
            if loader.metadata() != expected:
                raise ValueError("H3 cache header mismatch")
            properties_key = f"{key}.properties"
            if loader.get_dtype(properties_key) != torch.uint8 or loader.get_ndim(properties_key) != 1:
                raise ValueError("H3 cache properties must be uint8")
            properties = tensor_to_dict(loader.get_tensor(properties_key))
            loader.mark_processed(properties_key)
            if (properties["version"], properties["stage"], properties["fingerprint"], properties["dependencies"]) != (2, stage, key, dependencies):
                raise ValueError("H3 cache dependencies mismatch")
            specs = _tensor_specs(properties["result"])
            names = [spec["key"] for spec in specs]
            if len(set(names)) != len(names) or set(loader.keys()) != {properties_key, *names}:
                raise ValueError("H3 cache tensor manifest mismatch")
            for spec in specs:
                if list(loader.get_shape(spec["key"])) != spec["shape"] or str(loader.get_dtype(spec["key"])) != spec["dtype"]:
                    raise ValueError("H3 cache tensor declaration mismatch")
                torch.device(spec["device"])
            tensors = {}
            if names:
                for batch in loader.async_stream(names):
                    for name, tensor in batch:
                        tensors[name] = tensor
                        loader.mark_processed(name)
        return _unpack(properties["result"], tensors)

    def _write(self, path, stage, key, dependencies, result):
        tensors = {}
        properties = {"version": 2, "stage": stage, "fingerprint": key,
                      "dependencies": dependencies, "result": _pack(result, key, tensors)}
        properties_tensor = dict_to_tensor(properties)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with IncrementalSafetensorsWriter(str(temporary), metadata={
                "uc_h3_cache_version": "2", "uc_h3_cache_stage": stage, "uc_h3_cache_key": key,
            }) as writer:
                writer.write(f"{key}.properties", properties_tensor)
                for name, tensor in tensors.items():
                    writer.write(name, tensor)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                self._warning(error)

    def _allows_media(self, media):
        return self.enable_caching == "all" or (
            self.enable_caching == "video_only" and media == "video"
        ) or (self.enable_caching == "images_only" and media == "image")

    def allows_encoded_section(self, section_kind):
        if self.enable_caching == "disabled":
            return False
        if section_kind in {"text", "joint"}:
            return True
        if self.enable_caching == "all":
            return section_kind in {
                "image", "video", "regular_fusion", "regular_temporal", "guide", "token_fusion", "temporal_token_fusion",
            }
        if self.enable_caching == "images_only":
            return section_kind in {"image", "regular_fusion", "guide", "token_fusion"}
        return section_kind in {"video", "regular_temporal", "temporal_token_fusion"}

    def _allows_stage(self, stage, dependencies):
        if self.enable_caching == "disabled":
            return False
        if stage == "encoded_section":
            return self.allows_encoded_section(dependencies.get("section_kind", "joint"))
        return self._allows_media(dependencies["media"])

    def get_or_compute(self, stage, dependencies, compute, eligible=None):
        if stage not in _STAGES:
            raise ValueError(f"Unknown H3 cache stage: {stage}")
        if not self._active or not self._allows_stage(stage, dependencies):
            return compute()
        start = time.perf_counter()
        try:
            description = describe_value(dependencies, self._describe_tensor)
            key = hashlib.sha256(json.dumps([2, stage, description], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
        except (TypeError, ValueError) as error:
            self._warning(error)
            return compute()
        finally:
            self.timings["hash"] += time.perf_counter() - start
        path = self.root / stage / f"{key}.safetensors"
        value = _MISSING
        start = time.perf_counter()
        try:
            value = self._read(path, stage, key, description)
        except FileNotFoundError:
            pass
        except _STORAGE_ERRORS as error:
            self._warning(error)
        finally:
            self.timings["read"] += time.perf_counter() - start
        if value is not _MISSING:
            self.hits[stage] += 1
            return value
        self.misses[stage] += 1
        start = time.perf_counter()
        nested_before = sum(self.timings.values())
        try:
            value = compute()
        finally:
            self.timings["compute"] += max(0.0, time.perf_counter() - start - (sum(self.timings.values()) - nested_before))
        if eligible is not None and not eligible(value):
            return value
        start = time.perf_counter()
        try:
            self._write(path, stage, key, description, value)
        except _STORAGE_ERRORS as error:
            self._warning(error)
        finally:
            self.timings["write"] += time.perf_counter() - start
        return value

    def prepare_clip(self, clip):
        if self.enable_caching != "disabled":
            self._clip_identity = clip_description(clip)
        return clip

    def encode_scheduled(self, clip, tokens, visual_path, compute, *, section_kind="joint", section_id="joint", section_inputs=None):
        if not self.allows_encoded_section(section_kind):
            return compute()
        outputs = None
        hooks = clip.patcher.forced_hooks
        schedules = hooks.get_hooks_for_clip_schedule() if hooks is not None and clip.use_clip_schedule else [None]
        dependencies = {
            "section_kind": section_kind, "section_id": section_id, "tokens": tokens,
            "model": self._clip_identity or clip_description(clip), "visual_path": visual_path,
            "output_device": str(comfy.model_management.intermediate_device()),
            "section_inputs": section_inputs,
        }

        def compute_section(index):
            nonlocal outputs
            if outputs is None:
                outputs = compute()
            if len(outputs) != len(schedules):
                raise ValueError("H3 scheduled encode returned a different number of sections.")
            tensor, metadata = outputs[index]
            return [tensor, {key: value for key, value in metadata.items() if key != "hooks"}]

        result = []
        for index, schedule in enumerate(schedules):
            section = self.get_or_compute("encoded_section", {
                **dependencies, "section": index, "schedule": schedule,
            }, lambda index=index: compute_section(index))
            clip.add_hooks_to_dict(section[1])
            result.append(section)
        return result

    def encode_preprocessed(self, clip_model, embeds, attention, num_tokens, info, visual_path, hooks, compute):
        options = {name: _configuration(getattr(clip_model, name, None)) for name in (
            "enable_attention_masks", "layer", "layer_idx", "layer_norm_hidden_state",
            "zero_out_masked", "return_projected_pooled", "return_attention_masks",
        )}
        return self.get_or_compute("encoded_section", {
            "model": self._clip_identity, "options": options, "embeds": embeds,
            "attention": attention, "num_tokens": num_tokens, "info": info,
            "visual_path": visual_path, "hooks": _hook_description(hooks, active=True),
            "device": str(embeds.device), "output_device": str(comfy.model_management.intermediate_device()),
            "section": "preprocessed",
        }, compute)

    def encode_vae(self, vae, samples, media="image"):
        if not self._allows_media(media):
            return vae.encode(samples)
        identity = {
            "model": lifetime_identity(getattr(vae, "first_stage_model", vae)),
            "patcher": patcher_description(getattr(vae, "patcher", None)),
            "options": {name: _configuration(getattr(vae, name, None)) for name in (
                "vae_dtype", "device", "output_device", "latent_dim", "not_video", "disable_offload",
                "audio_sample_rate", "process_input", "vae_encode_crop_pixels", "vae_output_dtype",
            )},
        }
        return self.get_or_compute("vae_encode", {"model": identity, "samples": samples, "media": media}, lambda: vae.encode(samples))
