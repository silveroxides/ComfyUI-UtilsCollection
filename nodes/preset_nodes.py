import sys
import re

from comfy_api.latest import ComfyExtension, io
from .. import presets_collection, presets_video_collection
from ..helpers.helper_functions import to_video_prompt


def escape_prompt_parentheses(text, enabled=False):
    """Backslash-escape unescaped parentheses for downstream prompt-weight parsers."""
    return re.sub(r"(?<!\\)([()])", r"\\\1", text) if enabled else text


class UC_SystemMessagePresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("SYSTEM_MESSAGE"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    if name == "SYSTEM_MESSAGE":
                        presets["F2_SYSTEM_MESSAGE"] = val
                    elif name == "SYSTEM_MESSAGE_UPSAMPLING_I2I":
                        presets["F2_SYSTEM_MESSAGE_UPSAMPLING_I2I"] = val
                    elif name == "SYSTEM_MESSAGE_UPSAMPLING_T2I":
                        presets["F2_SYSTEM_MESSAGE_UPSAMPLING_T2I"] = val
                    elif name.startswith("SYSTEM_MESSAGE_STYLE_"):
                        presets[name.replace("SYSTEM_MESSAGE_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_SystemMessagePresets",
            category="advanced/text",
            display_name="System Message Presets",
            inputs=[
                io.Combo.Input("preset", display_name="system_styles_preset", options=sorted(list(presets.keys())), default="F2_SYSTEM_MESSAGE" if "F2_SYSTEM_MESSAGE" in presets else sorted(list(presets.keys()))[0]),
                io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt."),
            ],
            outputs=[io.String.Output(display_name="system_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        system_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(system_prompt, escape_parentheses))


class UC_InstructPromptPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("INSTRUCT_PROMPT_STYLE_"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_InstructPromptPresets",
            category="advanced/text",
            display_name="Instruct Prompt Presets",
            inputs=[io.Combo.Input("preset", display_name="instruct_styles_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="instruct_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        instruct_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(instruct_prompt, escape_parentheses))


class UC_BonusPromptPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("BONUS_PROMPT_STYLE_"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    presets[name.replace("BONUS_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_BonusPromptPresets",
            category="advanced/text",
            display_name="Bonus Prompt Presets",
            inputs=[io.Combo.Input("preset", display_name="bonus_styles_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="bonus_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        bonus_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(bonus_prompt, escape_parentheses))


class UC_LegacyPromptPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("LEGACY_PROMPT_"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    presets[name.replace("LEGACY_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_LegacyPromptPresets",
            category="advanced/text",
            display_name="Legacy Prompt Presets",
            inputs=[io.Combo.Input("preset", display_name="legacy_prompt_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="legacy_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        legacy_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(legacy_prompt, escape_parentheses))


class UC_SystemMessageVideoPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_video_collection):
            if name.startswith("SYSTEM_MESSAGE"):
                val = getattr(presets_video_collection, name)
                if isinstance(val, str):
                    if name == "SYSTEM_MESSAGE":
                        presets["F2_SYSTEM_MESSAGE"] = val
                    elif name == "SYSTEM_MESSAGE_UPSAMPLING_I2I":
                        presets["F2_SYSTEM_MESSAGE_UPSAMPLING_I2I"] = val
                    elif name == "SYSTEM_MESSAGE_UPSAMPLING_T2I":
                        presets["F2_SYSTEM_MESSAGE_UPSAMPLING_T2I"] = val
                    elif name.startswith("SYSTEM_MESSAGE_STYLE_"):
                        presets[name.replace("SYSTEM_MESSAGE_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_SystemMessageVideoPresets",
            category="advanced/text",
            display_name="System Message Video Presets",
            inputs=[
                io.Combo.Input("preset", display_name="system_video_styles_preset", options=sorted(list(presets.keys())), default="F2_SYSTEM_MESSAGE" if "F2_SYSTEM_MESSAGE" in presets else sorted(list(presets.keys()))[0]),
                io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt."),
            ],
            outputs=[io.String.Output(display_name="system_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        system_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(system_prompt, escape_parentheses))


class UC_InstructPromptVideoPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_video_collection):
            if name.startswith("INSTRUCT_PROMPT_STYLE_"):
                val = getattr(presets_video_collection, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_InstructPromptVideoPresets",
            category="advanced/text",
            display_name="Instruct Prompt Video Presets",
            inputs=[io.Combo.Input("preset", display_name="instruct_video_styles_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="instruct_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        instruct_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(instruct_prompt, escape_parentheses))


class UC_BonusPromptVideoPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_video_collection):
            if name.startswith("BONUS_PROMPT_STYLE_"):
                val = getattr(presets_video_collection, name)
                if isinstance(val, str):
                    presets[name.replace("BONUS_PROMPT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_BonusPromptVideoPresets",
            category="advanced/text",
            display_name="Bonus Prompt Video Presets",
            inputs=[io.Combo.Input("preset", display_name="bonus_video_styles_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="bonus_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        bonus_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(bonus_prompt, escape_parentheses))


class UC_EditTargetPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("BONUS_PROMPT_EDIT_TARGET_"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    presets[name.replace("BONUS_PROMPT_EDIT_TARGET_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_EditTargetPresets",
            category="advanced/text",
            display_name="Edit Target Presets",
            inputs=[io.Combo.Input("preset", display_name="edit_target_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="edit_target_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        edit_target_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(edit_target_prompt, escape_parentheses))


class UC_EditOpPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("INSTRUCT_PROMPT_EDIT_OP_"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_EDIT_OP_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_EditOpPresets",
            category="advanced/text",
            display_name="Edit Operation Presets",
            inputs=[io.Combo.Input("preset", display_name="edit_op_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="edit_op_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        edit_op_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(edit_op_prompt, escape_parentheses))


class UC_CameraShotPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        presets = {}
        for name in dir(presets_collection):
            if name.startswith("INSTRUCT_PROMPT_CAMERA_SHOT_"):
                val = getattr(presets_collection, name)
                if isinstance(val, str):
                    presets[name.replace("INSTRUCT_PROMPT_CAMERA_SHOT_", "")] = val
        return presets

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        return io.Schema(
            node_id="UC_CameraShotPresets",
            category="advanced/text",
            display_name="Camera Shot Presets",
            inputs=[io.Combo.Input("preset", display_name="camera_shot_preset", options=sorted(list(presets.keys())), default=sorted(list(presets.keys()))[0] if presets else ""), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.String.Output(display_name="camera_shot_prompt")],
        )

    @classmethod
    def execute(cls, preset, escape_parentheses=False) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        camera_shot_prompt = presets_dict.get(preset, "")
        return io.NodeOutput(escape_prompt_parentheses(camera_shot_prompt, escape_parentheses))


class UC_UnifiedPresets(io.ComfyNode):
    """
    Primitive node that unifies shared presets between SystemMessagePresets,
    InstructPromptPresets, and BonusPromptPresets.
    Outputs selected preset as 'any' type for flexible downstream usage.
    """

    @classmethod
    def get_shared_presets(cls):
        """Get presets that are shared between all three preset sources"""
        system_presets = set(UC_SystemMessagePresets.get_presets().keys())
        instruct_presets = set(UC_InstructPromptPresets.get_presets().keys())
        bonus_presets = set(UC_BonusPromptPresets.get_presets().keys())

        # Find intersection of all three
        shared = system_presets & instruct_presets & bonus_presets
        return sorted(list(shared))

    @classmethod
    def define_schema(cls) -> io.Schema:
        shared_presets = cls.get_shared_presets()
        default_preset = shared_presets[0] if shared_presets else ""

        return io.Schema(
            node_id="UC_UnifiedPresets",
            display_name="Unified Presets (Primitive)",
            category="advanced/primitives",
            inputs=[io.Combo.Input("preset", display_name="styles_preset", options=shared_presets, default=default_preset), io.Boolean.Input("escape_parentheses", default=False, tooltip="Prevents errors from parentheses left in the prompt.")],
            outputs=[io.AnyType.Output(display_name="preset"), io.Boolean.Output(display_name="escape_parentheses")],
        )

    @classmethod
    def execute(cls, preset: str, escape_parentheses: bool = False) -> io.NodeOutput:
        """Forward the selected preset as 'any' type"""
        return io.NodeOutput(preset, escape_parentheses)
