"""Model-native still-image preprocessing for the H3 Qwen3-VL encoder."""

import types

import torch

from comfy.text_encoders.minimax import MiniMaxQwen3VL
from comfy.text_encoders.qwen_vl import process_qwen2vl_images


def preprocess_h3_embed(self, embed, device):
    if embed["type"] != "image" or embed.get("minimax_video_block", False):
        return MiniMaxQwen3VL.preprocess_embed(self, embed, device)
    image, grid = process_qwen2vl_images(
        embed["data"][..., :3], min_pixels=65536, max_pixels=16777216,
        patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    )
    merged, deepstack = self.visual(image.to(device, dtype=torch.float32), grid)
    return merged, {"grid": grid, "deepstack": deepstack}


def prepare_h3_preprocessing_clip(clip):
    stage = getattr(getattr(clip, "cond_stage_model", None), "qwen3vl_32b", None)
    transformer = getattr(stage, "transformer", None)
    if not isinstance(transformer, MiniMaxQwen3VL):
        return clip
    patched = clip.clone()
    patched.tokenize = types.MethodType(tokenize_h3_images, patched)
    patched.patcher.add_object_patch(
        "qwen3vl_32b.transformer.preprocess_embed",
        types.MethodType(preprocess_h3_embed, transformer),
    )
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
