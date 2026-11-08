from typing_extensions import override
from comfy_api.latest import io

from .encoder_nodes import *
from .image_nodes import *
from .preset_nodes import *
from .vlm_nodes import *
from .parameter_nodes import *
from .utils_nodes import *
from .textgen_nodes import *
from ..helpers.lama_helpers import image_mask_to_luma, load_lama_model, run_lama_inpaint

import folder_paths


class AdjustedResolutionParameters(UC_AdjustedResolutionParameters):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "AdjustedResolutionParameters"
        schema.display_name = f"{schema.display_name or 'AdjustedResolutionParameters'} (Legacy)"
        return schema


class ResolutionSelectorExtended(UC_ResolutionSelectorExtended):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ResolutionSelectorExtended"
        schema.display_name = f"{schema.display_name or 'ResolutionSelectorExtended'} (Legacy)"
        return schema


class ImageScaleAndResolutionPicker(UC_ImageScaleAndResolutionPicker):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ImageScaleAndResolutionPicker"
        schema.display_name = f"{schema.display_name or 'ImageScaleAndResolutionPicker'} (Legacy)"
        return schema


class Image_Color_Noise(UC_Image_Color_Noise):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "Image_Color_Noise"
        schema.display_name = f"{schema.display_name or 'Image_Color_Noise'} (Legacy)"
        return schema


class TextEncodeFlux2SystemPrompt(UC_TextEncodeFlux2SystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextEncodeFlux2SystemPrompt"
        schema.display_name = f"{schema.display_name or 'TextEncodeFlux2SystemPrompt'} (Legacy)"
        return schema


class TextEncodeKleinSystemPrompt(UC_TextEncodeKleinSystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextEncodeKleinSystemPrompt"
        schema.display_name = f"{schema.display_name or 'TextEncodeKleinSystemPrompt'} (Legacy)"
        return schema


class TextEncodeLtxv2SystemPrompt(UC_TextEncodeLtxv2SystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextEncodeLtxv2SystemPrompt"
        schema.display_name = f"{schema.display_name or 'TextEncodeLtxv2SystemPrompt'} (Legacy)"
        return schema


class TextEncodeZITSystemPrompt(UC_TextEncodeZITSystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextEncodeZITSystemPrompt"
        schema.display_name = f"{schema.display_name or 'TextEncodeZITSystemPrompt'} (Legacy)"
        return schema


class TextEncodeZImageThinkPrompt(UC_TextEncodeZImageThinkPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextEncodeZImageThinkPrompt"
        schema.display_name = f"{schema.display_name or 'TextEncodeZImageThinkPrompt'} (Legacy)"
        return schema


class TextEncodeSystemPrompt(UC_TextEncodeSystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextEncodeSystemPrompt"
        schema.display_name = f"{schema.display_name or 'TextEncodeSystemPrompt'} (Legacy)"
        return schema


class ScaledBiasTextEncodeFlux2SystemPrompt(UC_ScaledBiasTextEncodeFlux2SystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ScaledBiasTextEncodeFlux2SystemPrompt"
        schema.display_name = f"{schema.display_name or 'ScaledBiasTextEncodeFlux2SystemPrompt'} (Legacy)"
        return schema


class ScaledBiasTextEncodeKleinSystemPrompt(UC_ScaledBiasTextEncodeKleinSystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ScaledBiasTextEncodeKleinSystemPrompt"
        schema.display_name = f"{schema.display_name or 'ScaledBiasTextEncodeKleinSystemPrompt'} (Legacy)"
        return schema


class ScaledBiasTextEncodeLtxv2SystemPrompt(UC_ScaledBiasTextEncodeLtxv2SystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ScaledBiasTextEncodeLtxv2SystemPrompt"
        schema.display_name = f"{schema.display_name or 'ScaledBiasTextEncodeLtxv2SystemPrompt'} (Legacy)"
        return schema


class ScaledBiasTextEncodeZITSystemPrompt(UC_ScaledBiasTextEncodeZITSystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ScaledBiasTextEncodeZITSystemPrompt"
        schema.display_name = f"{schema.display_name or 'ScaledBiasTextEncodeZITSystemPrompt'} (Legacy)"
        return schema


class ScaledBiasTextEncodeZImageThinkPrompt(UC_ScaledBiasTextEncodeZImageThinkPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ScaledBiasTextEncodeZImageThinkPrompt"
        schema.display_name = f"{schema.display_name or 'ScaledBiasTextEncodeZImageThinkPrompt'} (Legacy)"
        return schema


class ScaledBiasTextEncodeSystemPrompt(UC_ScaledBiasTextEncodeSystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ScaledBiasTextEncodeSystemPrompt"
        schema.display_name = f"{schema.display_name or 'ScaledBiasTextEncodeSystemPrompt'} (Legacy)"
        return schema


class ModifyMask(UC_ModifyMask):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ModifyMask"
        schema.display_name = f"{schema.display_name or 'ModifyMask'} (Legacy)"
        return schema


class LamaRemover(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LamaRemover",
            display_name="LamaRemover (Legacy)",
            category="LamaRemover",
            is_deprecated=True,
            inputs=[io.Image.Input("images"), io.Mask.Input("masks"), io.Int.Input("mask_threshold", default=250, min=0, max=255, step=1), io.Int.Input("gaussblur_radius", default=8, min=0, max=20, step=1), io.Boolean.Input("invert_mask", default=False)],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, masks, mask_threshold=250, gaussblur_radius=8, invert_mask=False) -> io.NodeOutput:
        model_path = folder_paths.get_full_path_or_raise("lama", "big-lama.safetensors")
        patcher = load_lama_model(model_path, "default")
        return io.NodeOutput(run_lama_inpaint(patcher, images, masks, mask_threshold, gaussblur_radius, invert_mask))


class LamaRemoverIMG(LamaRemover):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "LamaRemoverIMG"
        schema.display_name = "LamaRemoverIMG (Legacy)"
        schema.inputs[1] = io.Image.Input("masks")
        return schema

    @classmethod
    @override
    def execute(cls, images, masks, mask_threshold=250, gaussblur_radius=8, invert_mask=False) -> io.NodeOutput:
        return super().execute(images, image_mask_to_luma(masks), mask_threshold, gaussblur_radius, invert_mask)


class ImageBlendByMask(UC_ImageBlendByMask):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ImageBlendByMask"
        schema.display_name = f"{schema.display_name or 'ImageBlendByMask'} (Legacy)"
        return schema


class SystemMessagePresets(UC_SystemMessagePresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "SystemMessagePresets"
        schema.display_name = f"{schema.display_name or 'SystemMessagePresets'} (Legacy)"
        return schema


class SystemMessageVideoPresets(UC_SystemMessageVideoPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "SystemMessageVideoPresets"
        schema.display_name = f"{schema.display_name or 'SystemMessageVideoPresets'} (Legacy)"
        return schema


class InstructPromptPresets(UC_InstructPromptPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "InstructPromptPresets"
        schema.display_name = f"{schema.display_name or 'InstructPromptPresets'} (Legacy)"
        return schema


class InstructPromptVideoPresets(UC_InstructPromptVideoPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "InstructPromptVideoPresets"
        schema.display_name = f"{schema.display_name or 'InstructPromptVideoPresets'} (Legacy)"
        return schema


class BonusPromptPresets(UC_BonusPromptPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "BonusPromptPresets"
        schema.display_name = f"{schema.display_name or 'BonusPromptPresets'} (Legacy)"
        return schema


class BonusPromptVideoPresets(UC_BonusPromptVideoPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "BonusPromptVideoPresets"
        schema.display_name = f"{schema.display_name or 'BonusPromptVideoPresets'} (Legacy)"
        return schema


class EditTargetPresets(UC_EditTargetPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "EditTargetPresets"
        schema.display_name = f"{schema.display_name or 'EditTargetPresets'} (Legacy)"
        return schema


class EditOpPresets(UC_EditOpPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "EditOpPresets"
        schema.display_name = f"{schema.display_name or 'EditOpPresets'} (Legacy)"
        return schema


class CameraShotPresets(UC_CameraShotPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "CameraShotPresets"
        schema.display_name = f"{schema.display_name or 'CameraShotPresets'} (Legacy)"
        return schema


class VLMSysInstrPresets(UC_VLMSysInstrPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "VLMSysInstrPresets"
        schema.display_name = f"{schema.display_name or 'VLMSysInstrPresets'} (Legacy)"
        return schema


class VLMSysQueryAddPresets(UC_VLMSysQueryAddPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "VLMSysQueryAddPresets"
        schema.display_name = f"{schema.display_name or 'VLMSysQueryAddPresets'} (Legacy)"
        return schema


class VLMSysInstrAdvPresets(UC_VLMSysInstrAdvPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "VLMSysInstrAdvPresets"
        schema.display_name = f"{schema.display_name or 'VLMSysInstrAdvPresets'} (Legacy)"
        return schema


class UnifiedPresets(UC_UnifiedPresets):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "UnifiedPresets"
        schema.display_name = f"{schema.display_name or 'UnifiedPresets'} (Legacy)"
        return schema


class AttentionBiasTextEncode(UC_AttentionBiasTextEncode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "AttentionBiasTextEncode"
        schema.display_name = f"{schema.display_name or 'AttentionBiasTextEncode'} (Legacy)"
        return schema


class TagNormalizeCombine(UC_TagNormalizeCombine):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TagNormalizeCombine"
        schema.display_name = f"{schema.display_name or 'TagNormalizeCombine'} (Legacy)"
        return schema


class SU_LoadImagePath(UC_LoadImagePath):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "SU_LoadImagePath"
        schema.display_name = f"{schema.display_name or 'SU_LoadImagePath'} (Legacy)"
        return schema


class SU_LoadImageDirectory(UC_LoadImageDirectory):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "SU_LoadImageDirectory"
        schema.display_name = f"{schema.display_name or 'SU_LoadImageDirectory'} (Legacy)"
        return schema


class SwitchInverseNode(UC_SwitchInverseNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ComfySwitchInverseNode"
        schema.display_name = f"{schema.display_name or 'SwitchInverseNode'} (Legacy)"
        return schema


class SoftSwitchInverseNode(UC_SoftSwitchInverseNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ComfySoftSwitchInverseNode"
        schema.display_name = f"{schema.display_name or 'SoftSwitchInverseNode'} (Legacy)"
        return schema


class IntegerRangeRandom(UC_IntegerRangeRandom):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "IntegerRangeRandom"
        schema.display_name = f"{schema.display_name or 'IntegerRangeRandom'} (Legacy)"
        return schema


class ImageMatchPropertiesNode(UC_ImageMatchPropertiesNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ImageMatchProperties"
        schema.display_name = f"{schema.display_name or 'ImageMatchPropertiesNode'} (Legacy)"
        return schema


class OpticalFlowComposite(UC_OpticalFlowComposite):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "OpticalFlowComposite"
        schema.display_name = f"{schema.display_name or 'OpticalFlowComposite'} (Legacy)"
        return schema


class ImageInwardEdgeFill(UC_ImageInwardEdgeFill):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ImageInwardEdgeFill"
        schema.display_name = f"{schema.display_name or 'ImageInwardEdgeFill'} (Legacy)"
        return schema


class ImageIterativeStretchFill(UC_ImageIterativeStretchFill):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ImageIterativeStretchFill"
        schema.display_name = f"{schema.display_name or 'ImageIterativeStretchFill'} (Legacy)"
        return schema


class TextOverlayNode(UC_TextOverlayNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextOverlayNode"
        schema.display_name = f"{schema.display_name or 'TextOverlayNode'} (Legacy)"
        return schema


class RandInt(UC_RandInt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "PrimitiveRandomInt"
        schema.display_name = f"{schema.display_name or 'RandInt'} (Legacy)"
        return schema


class StaticInt(UC_StaticInt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "PrimitiveStaticInt"
        schema.display_name = f"{schema.display_name or 'StaticInt'} (Legacy)"
        return schema


class RandIntRange(UC_RandIntRange):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "PrimitiveRandomIntRange"
        schema.display_name = f"{schema.display_name or 'RandIntRange'} (Legacy)"
        return schema


class TextGenerateQwen35SystemPrompt(UC_TextGenerateQwen35SystemPrompt):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "TextGenerateQwen35SystemPrompt"
        schema.display_name = f"{schema.display_name or 'TextGenerateQwen35SystemPrompt'} (Legacy)"
        return schema


class ColorConvertNode(UC_ColorConvertNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ColorConvertNode"
        schema.display_name = f"{schema.display_name or 'ColorConvertNode'} (Legacy)"
        return schema
