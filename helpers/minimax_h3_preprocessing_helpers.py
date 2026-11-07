"""Model-native still-image preprocessing for the H3 Qwen3-VL encoder."""

import threading
import types
import weakref

import torch

from comfy.text_encoders.minimax import MiniMaxQwen3VL
from comfy.text_encoders.qwen_vl import process_qwen2vl_images


_PATCHED_CLIP_CACHE = {}
_PATCHED_CLIP_LOCK = threading.RLock()


def preprocess_h3_embed(self, embed, device):
    if embed["type"] != "image" or embed.get("minimax_video_block", False):
        return MiniMaxQwen3VL.preprocess_embed(self, embed, device)
    data = embed["data"]
    if data.ndim == 3:
        data = data.unsqueeze(0)
    image, grid = process_qwen2vl_images(
        data[..., :3], min_pixels=65536, max_pixels=16777216,
        patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    )
    merged, deepstack = self.visual(image.to(device, dtype=torch.float32), grid)
    return merged, {"grid": grid, "deepstack": deepstack}


def _clip_patch_signature(clip):
    patcher = getattr(clip, "patcher", None)
    return (
        getattr(patcher, "patches_uuid", None),
        getattr(patcher, "object_patches_uuid", None),
        getattr(clip, "layer_idx", None),
        getattr(clip, "use_clip_schedule", None),
    )


def prepare_h3_preprocessing_clip(clip):
    if getattr(clip, "_uc_h3_preprocessed_clip", False):
        return clip
    stage = getattr(getattr(clip, "cond_stage_model", None), "qwen3vl_32b", None)
    transformer = getattr(stage, "transformer", None)
    if not isinstance(transformer, MiniMaxQwen3VL):
        return clip
    address = id(clip)
    sig = _clip_patch_signature(clip)
    with _PATCHED_CLIP_LOCK:
        cached = _PATCHED_CLIP_CACHE.get(address)
        if cached is not None:
            ref, cached_sig, patched_obj = cached
            if ref() is clip and cached_sig == sig and patched_obj is not None:
                return patched_obj

        patched = clip.clone()
        patched.tokenize = types.MethodType(tokenize_h3_images, patched)
        patched.patcher.add_object_patch(
            "qwen3vl_32b.transformer.preprocess_embed",
            types.MethodType(preprocess_h3_embed, transformer),
        )
        patched._uc_h3_preprocessed_clip = True

        def _cleanup(_ref, addr=address):
            with _PATCHED_CLIP_LOCK:
                entry = _PATCHED_CLIP_CACHE.get(addr)
                if entry is not None and entry[0] is _ref:
                    del _PATCHED_CLIP_CACHE[addr]

        try:
            clip_ref = weakref.ref(clip, _cleanup)
            _PATCHED_CLIP_CACHE[address] = (clip_ref, sig, patched)
        except TypeError:
            pass
        return patched



def tokenize_h3_images(self, *args, **kwargs):
    tokens = type(self).tokenize(self, *args, **kwargs)
    for rows in tokens.values():
        for row in rows:
            for token in row:
                value = token[0]
                if isinstance(value, dict) and value.get("type") == "image" and not value.get("minimax_video_block", False):
                    value["h3_image_pixel_limits"] = (65536, 16777216)
    return tokens
