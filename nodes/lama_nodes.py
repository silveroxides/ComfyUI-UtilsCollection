from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

from ..helpers.lama_helpers import get_lama_device_options, get_lama_model_names, load_lama_model, run_lama_inpaint


LaMaModel = io.Custom("LAMA_MODEL")
_LAMA_MODEL_DIRECTORY = os.path.join(folder_paths.models_dir, "lama")
if "lama" in folder_paths.folder_names_and_paths:
    _paths, _extensions = folder_paths.folder_names_and_paths["lama"]
    if _LAMA_MODEL_DIRECTORY not in _paths:
        _paths.append(_LAMA_MODEL_DIRECTORY)
    folder_paths.folder_names_and_paths["lama"] = (_paths, _extensions | {".safetensors"})
else:
    folder_paths.folder_names_and_paths["lama"] = ([_LAMA_MODEL_DIRECTORY], {".safetensors"})


class UC_LoadLaMaModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadLaMaModel",
            display_name="Load LaMa Model",
            category="utils/model",
            description="Loads an eager Big LaMa inpainting model through UEL.",
            inputs=[io.Combo.Input("model_name", options=get_lama_model_names(), tooltip="Safetensors model from ComfyUI/models/lama."), io.Combo.Input("device", options=get_lama_device_options(), default="default", tooltip="ComfyUI default, CPU, or a specific available GPU.")],
            outputs=[LaMaModel.Output("lama_model")],
        )

    @classmethod
    def execute(cls, model_name: str, device: str = "default") -> io.NodeOutput:
        if not model_name.lower().endswith(".safetensors"):
            raise ValueError("LaMa loader accepts only .safetensors models.")
        model_path = folder_paths.get_full_path_or_raise("lama", model_name)
        return io.NodeOutput(load_lama_model(model_path, device))


class UC_LaMaInpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LaMaInpaint",
            display_name="LaMa Inpaint",
            category="utils/image",
            description="Inpaints masked image regions with an eager Big LaMa model.",
            inputs=[
                LaMaModel.Input("lama_model"),
                io.Image.Input("images"),
                io.Mask.Input("masks"),
                io.Int.Input("mask_threshold", default=250, min=0, max=255, step=1, display_mode=io.NumberDisplay.slider, tooltip="Threshold applied after mask blur; matches legacy 0-255 behavior."),
                io.Int.Input("gaussblur_radius", default=8, min=0, max=20, step=1, display_mode=io.NumberDisplay.slider, tooltip="Gaussian mask blur radius in pixels."),
                io.Boolean.Input("invert_mask", default=False, tooltip="Invert which mask region LaMa replaces."),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, lama_model, images, masks, mask_threshold: int = 250, gaussblur_radius: int = 8, invert_mask: bool = False) -> io.NodeOutput:
        return io.NodeOutput(run_lama_inpaint(lama_model, images, masks, mask_threshold, gaussblur_radius, invert_mask))
