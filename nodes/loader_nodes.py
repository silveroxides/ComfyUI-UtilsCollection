import comfy
from folder_paths import get_filename_list, get_folder_paths, get_full_path_or_raise
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file
from comfy_api.latest import io
from ..helpers.loader_helpers import apply_lora_stack_to_model, build_lora_stack_dynamic_combo, extract_lora_stack_entries, load_filtered_lora_for_model, load_sam31_checkpoint

_LORA_LOADER_CACHE = None
_LORA_STACK_CACHE = {}


class UC_LoraLoaderCLIPOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoraLoaderCLIPOnly",
            display_name="Load LoRA for CLIP Only",
            category="advanced/model",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("lora_name", get_filename_list("loras"), tooltip="Choose a LoRA to apply to the text encoder. LoRAs without text-encoder data cannot be used here."),
                io.Float.Input("strength_clip", default=1.0, min=-10.0, max=10.0, step=0.05, tooltip="Strength of the LoRA effect on text encoding."),
            ],
            outputs=[io.Clip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, clip, lora_name: str, strength_clip: float) -> io.NodeOutput:
        global _LORA_LOADER_CACHE
        if strength_clip == 0:
            return (io.NodeOutput(clip),)
        # Placeholder for actual LoRA loading logic
        lora_path = get_full_path_or_raise("loras", lora_name)
        lora = None

        if _LORA_LOADER_CACHE is not None:
            if _LORA_LOADER_CACHE[0] == lora_path:
                lora = _LORA_LOADER_CACHE[1]
            else:
                _LORA_LOADER_CACHE = None

        if lora is None:
            lora = load_torch_file(lora_path, safe_load=True)
            _LORA_LOADER_CACHE = (lora_path, lora)

        clip_lora = load_lora_for_models(None, clip, lora, 0, strength_clip)[1]
        return io.NodeOutput(clip_lora)


class UC_LoraLoaderModelOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoraLoaderModelOnly",
            display_name="Load LoRA for Model Only (Filtered)",
            category="advanced/model",
            description="Loads a LoRA into the diffusion model with layer regex and transformer block filtering.",
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model the LoRA will be applied to."),
                io.Combo.Input("lora_name", options=get_filename_list("loras"), tooltip="Choose a LoRA to apply to the diffusion model."),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01, tooltip="How strongly to modify the diffusion model. This value can be negative."),
                io.String.Input("layer_filter", multiline=True, default="", tooltip="Substring regex patterns (one per line) to match layer names."),
                io.String.Input("block_filter", default="", tooltip="Comma-separated block numbers or ranges (e.g. '1, 2, 3', '1,2,4-5,7') for transformer blocks."),
                io.Boolean.Input("whitelist", default=True, tooltip="When True, matching layers are loaded (whitelist). When False, matching layers are excluded (blacklist)."),
            ],
            outputs=[io.Model.Output("model", display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength_model: float, layer_filter: str = "", block_filter: str = "", whitelist: bool = True) -> io.NodeOutput:
        global _LORA_LOADER_CACHE
        new_model, _LORA_LOADER_CACHE = load_filtered_lora_for_model(model=model, lora_name=lora_name, strength_model=strength_model, layer_filter=layer_filter, block_filter=block_filter, whitelist=whitelist, cache=_LORA_LOADER_CACHE)
        return io.NodeOutput(new_model)


class UC_SAM31CheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SAM31CheckpointLoader",
            display_name="Load SAM 3.1 Checkpoint",
            category="advanced/model",
            description="Loads a SAM 3.1 checkpoint with automatic or forced FP32 model and text-encoder precision.",
            inputs=[io.Combo.Input("ckpt_name", options=get_filename_list("checkpoints"), tooltip="SAM 3.1 checkpoint from the checkpoints folder."), io.Combo.Input("precision", options=["auto", "fp32"], default="fp32", tooltip="fp32 prevents Core from downcasting model and text-encoder weights.")],
            outputs=[io.Model.Output("model"), io.Clip.Output("clip"), io.Vae.Output("vae")],
        )

    @classmethod
    def execute(cls, ckpt_name: str, precision: str = "fp32") -> io.NodeOutput:
        checkpoint_path = get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae, _ = load_sam31_checkpoint(checkpoint_path, precision, embedding_directory=get_folder_paths("embeddings"))
        return io.NodeOutput(model, clip, vae)


class UC_LoraStackLoaderModelOnly(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoraStackLoaderModelOnly",
            display_name="Load LoRA Stack (Model Only)",
            category="advanced/model",
            description="Applies a stack of LoRAs to a diffusion model using a recursive DynamicCombo template.",
            search_aliases=["lora", "lora stack", "load lora stack", "lora loader", "stack loader"],
            inputs=[io.Model.Input("model", tooltip="The diffusion model the LoRA stack will be applied to."), build_lora_stack_dynamic_combo(depth=1, max_depth=15)],
            outputs=[io.Model.Output("model", display_name="model", tooltip="The diffusion model with all enabled LoRAs applied.")],
        )

    @classmethod
    def execute(cls, model, add_lora=None, **kwargs) -> io.NodeOutput:
        global _LORA_STACK_CACHE
        entries = extract_lora_stack_entries(add_lora, **kwargs)
        new_model, _LORA_STACK_CACHE = apply_lora_stack_to_model(model=model, lora_entries=entries, cache=_LORA_STACK_CACHE)
        return io.NodeOutput(new_model)


class UC_LoraStack(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoraStack",
            display_name="LoRA Stack",
            category="advanced/model",
            description="Specifies a stack of LoRAs using a recursive DynamicCombo template matching the DynamicGroup format.",
            search_aliases=["lora", "lora stack", "stack", "dynamic group", "lora template"],
            inputs=[build_lora_stack_dynamic_combo(depth=1, max_depth=15, strength_key="strength")],
            outputs=[io.Custom("COMFY_DYNAMICGROUP_V3").Output("loras", display_name="loras", tooltip="The LoRA stack specification matching the DynamicGroup format expected by LoadLoraModel and LoadLoraTextEncoder.")],
        )

    @classmethod
    def execute(cls, add_lora=None, **kwargs) -> io.NodeOutput:
        entries = extract_lora_stack_entries(add_lora, **kwargs)
        return io.NodeOutput(entries)
