"""Persistent foreground annotations; retained cutouts remain unmodified."""

import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from .composite_helpers import _COMPOSITE_RESIZE_METHODS


def parse_foreground_content(placement_data):
    if not placement_data:
        return {}
    if isinstance(placement_data, str) and placement_data.strip() in _COMPOSITE_RESIZE_METHODS:
        return {}
    payload = json.loads(placement_data) if isinstance(placement_data, str) else placement_data
    content = payload.get("foreground_content", {})
    if not isinstance(content, dict):
        raise ValueError("Foreground content must be an object keyed by foreground identifier.")
    for key, item in content.items():
        if not isinstance(item, dict):
            raise ValueError(f"Foreground content for {key} must be an object.")
        for kind in ("object_erase", "brush", "text"):
            part = item.get(kind, {})
            if not isinstance(part, dict) or not isinstance(part.get("visible", True), bool):
                raise ValueError(f"Foreground {key} {kind} must contain Boolean visibility.")
            asset = part.get("asset")
            if asset is None:
                if kind == "text" and part.get("value") and part.get("visible", True):
                    raise ValueError(f"Foreground {key} text has not been saved. Open the editor and save before queueing.")
                continue
            if (
                not isinstance(asset, dict)
                or asset.get("type") != "input"
                or not isinstance(asset.get("filename"), str)
                or not asset["filename"]
                or not isinstance(asset.get("subfolder", ""), str)
            ):
                raise ValueError(f"Foreground {key} {kind} requires a PNG asset in input storage.")
    return content


def load_foreground_rgba(asset, reference):
    root = os.path.realpath(folder_paths.get_input_directory())
    path = os.path.realpath(os.path.join(root, asset.get("subfolder", ""), asset["filename"]))
    if os.path.commonpath((root, path)) != root:
        raise ValueError("Foreground content asset must stay inside ComfyUI input storage.")
    if not os.path.isfile(path):
        raise ValueError(f"Foreground content PNG is missing: {asset['filename']}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGBA" or image.n_frames != 1:
            raise ValueError("Foreground content must be a single RGBA PNG image.")
        pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).to(reference)


def composite_foreground_content(image, alpha, overlay):
    """Straight-alpha source-over, including previously transparent cutout pixels."""
    added_alpha = overlay[0, ..., 3].clamp(0, 1)
    combined_alpha = added_alpha + alpha * (1 - added_alpha)
    premultiplied = (
        overlay[..., :3] * added_alpha[None, ..., None]
        + image * (alpha * (1 - added_alpha))[None, ..., None]
    )
    combined_image = torch.where(
        combined_alpha[None, ..., None] > 1e-8,
        premultiplied / combined_alpha[None, ..., None].clamp_min(1e-8),
        torch.zeros_like(premultiplied),
    )
    return combined_image, combined_alpha
