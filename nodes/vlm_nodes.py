from comfy_api.latest import io
from ..helpers.text_helpers import get_vlm_preset_response_schema
from ..vlm_presets import additional_instructions_vlm, system_instructions_vlm, system_query_additional_vlm, system_query_raw_vlm
from ..vlm_legacy_presets import legacy_system_instructions_vlm
from ..vlm_experimental_presets import system_instructions_vlm_experimental
from ..vlm_motion_presets import motion_system_instructions_vlm

ResponseSchema = io.Custom("GEMINI_RESPONSE_SCHEMA")


class UC_VLMSysInstrPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        return system_instructions_vlm

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(list(presets.keys()))
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrPresets",
            display_name="VLM System Instruction Presets",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="vlm_system_instruction_preset", options=options, default=default, tooltip="Select the system instruction preset. JSON layout presets also output a paired Gemini response_schema.")],
            outputs=[
                io.String.Output(display_name="system_instruction", tooltip="Selected VLM system instruction text."),
                ResponseSchema.Output("response_schema", display_name="response_schema", tooltip="Paired Gemini response schema for JSON presets (moss_image_json_instruction* and ideogram_4_json_instruction*). Connect to the Gemini node's Response format input."),
            ],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        system_instruction = presets_dict.get(preset, "")
        return io.NodeOutput(system_instruction, get_vlm_preset_response_schema(preset))


class UC_VLMSysInstrPresetsExperimental(UC_VLMSysInstrPresets):
    @classmethod
    def get_presets(cls):
        return system_instructions_vlm_experimental

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(presets)
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrPresetsExperimental",
            display_name="VLM System Instruction Presets Experimental",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="vlm_system_instruction_preset_experimental", options=options, default=default)],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        return io.NodeOutput(presets_dict.get(preset, ""))


class UC_VLMSysInstrLegacyPresets(UC_VLMSysInstrPresets):
    @classmethod
    def get_presets(cls):
        return legacy_system_instructions_vlm

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(presets)
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrLegacyPresets", display_name="VLM System Instruction Legacy Presets", category="advanced/text", inputs=[io.Combo.Input("preset", display_name="vlm_system_instruction_legacy_preset", options=options, default=default)], outputs=[io.String.Output(display_name="system_instruction")]
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        return io.NodeOutput(presets_dict.get(preset, ""))


class UC_VLMSysInstrMotionPresets(UC_VLMSysInstrPresets):
    @classmethod
    def get_presets(cls):
        return motion_system_instructions_vlm

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(presets)
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrMotionPresets", display_name="VLM System Instruction Motion Presets", category="advanced/text", inputs=[io.Combo.Input("preset", display_name="vlm_system_instruction_motion_preset", options=options, default=default)], outputs=[io.String.Output(display_name="system_instruction")]
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        presets_dict = cls.get_presets()
        return io.NodeOutput(presets_dict.get(preset, ""))


class UC_VLMSysQueryAddPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        base_names = set()
        for key in system_query_additional_vlm:
            base_names.add(key.removesuffix("_prefix").removesuffix("_suffix"))
        return sorted(list(base_names))

    @classmethod
    def define_schema(cls):
        options = cls.get_presets()
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysQueryAddPresets",
            display_name="VLM System Query Add Presets",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="vlm_system_query_add_preset", options=options, default=default), io.String.Input("text", multiline=True, default="")],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset, text) -> io.NodeOutput:
        prefix = system_query_additional_vlm.get(f"{preset}_prefix", "")
        suffix = system_query_additional_vlm.get(f"{preset}_suffix", "")
        return io.NodeOutput(f"{prefix}{text}{suffix}")


class UC_VLMSysQueryRawPresets(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = sorted(system_query_raw_vlm)
        default = options[0] if options else ""
        return io.Schema(node_id="UC_VLMSysQueryRawPresets", display_name="VLM System Query Raw Presets", category="advanced/text", inputs=[io.Combo.Input("preset", display_name="vlm_system_query_raw_preset", options=options, default=default)], outputs=[io.String.Output(display_name="system_query")])

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        return io.NodeOutput(system_query_raw_vlm.get(preset, ""))


class UC_VLMSysInstrAdvPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        return system_instructions_vlm

    @classmethod
    def get_additional_instructions(cls):
        return additional_instructions_vlm

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(list(presets.keys()))
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrAdvPresets",
            display_name="VLM System Instruction Advanced Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input("preset", display_name="vlm_system_instruction_advanced_preset", options=options, default=default, tooltip="Select the system instruction preset. JSON layout presets also output a paired Gemini response_schema."),
                io.Boolean.Input("jailbreak", default=False, tooltip="Wrap the system instruction with the configured jailbreak prefix and suffix."),
                io.String.Input("system_query", multiline=True, default="", tooltip="Optional high-priority system query instructions appended to the preset."),
                io.String.Input("user_query", multiline=True, default="", tooltip="Optional user query instructions appended to the preset."),
            ],
            outputs=[
                io.String.Output(display_name="system_instruction", tooltip="Assembled VLM system instruction text."),
                ResponseSchema.Output("response_schema", display_name="response_schema", tooltip="Paired Gemini response schema for JSON presets (moss_image_json_instruction* and ideogram_4_json_instruction*). Connect to the Gemini node's Response format input."),
            ],
        )

    @classmethod
    def _assemble_instruction(cls, preset, jailbreak, system_query, user_query) -> str:
        presets_dict = cls.get_presets()
        additional = cls.get_additional_instructions()

        system_instruction = presets_dict.get(preset, "")

        # Start building the final output
        # Formula: {jailbreak_prefix} + {chosen_preset_content} + {system_query_prefix} + {system_query_string} + {user_query_prefix} + {user_query_string} + {user_query_suffix} + {system_query_suffix} + {jailbreak_suffix}

        result = system_instruction

        # Apply system query wrapping
        if system_query:
            sq_prefix = additional.get("system_query_prefix", "")
            sq_suffix = additional.get("system_query_suffix", "")
            result = result + sq_prefix + system_query

            # Apply user query nesting
            if user_query:
                uq_prefix = additional.get("user_query_prefix", "")
                uq_suffix = additional.get("user_query_suffix", "")
                result = result + uq_prefix + user_query + uq_suffix

            result = result + sq_suffix
        elif user_query:
            # If system_query is empty but user_query is not, just add user query components
            uq_prefix = additional.get("user_query_prefix", "")
            uq_suffix = additional.get("user_query_suffix", "")
            result = result + uq_prefix + user_query + uq_suffix

        # Apply jailbreak wrapping
        if jailbreak:
            jb_prefix = additional.get("jailbreak_prefix", "")
            jb_suffix = additional.get("jailbreak_suffix", "")
            result = jb_prefix + result + jb_suffix

        return result

    @classmethod
    def execute(cls, preset, jailbreak, system_query, user_query) -> io.NodeOutput:
        result = cls._assemble_instruction(preset, jailbreak, system_query, user_query)
        return io.NodeOutput(result, get_vlm_preset_response_schema(preset))


class UC_VLMSysInstrAdvPresetsExperimental(UC_VLMSysInstrAdvPresets):
    @classmethod
    def get_presets(cls):
        return system_instructions_vlm_experimental

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(presets)
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrAdvPresetsExperimental",
            display_name="VLM System Instruction Advanced Presets Experimental",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="vlm_system_instruction_advanced_preset_experimental", options=options, default=default), io.Boolean.Input("jailbreak", default=False), io.String.Input("system_query", multiline=True, default=""), io.String.Input("user_query", multiline=True, default="")],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset, jailbreak, system_query, user_query) -> io.NodeOutput:
        return io.NodeOutput(cls._assemble_instruction(preset, jailbreak, system_query, user_query))


class UC_VLMSysInstrAdvMotionPresets(UC_VLMSysInstrAdvPresets):
    @classmethod
    def get_presets(cls):
        return motion_system_instructions_vlm

    @classmethod
    def define_schema(cls):
        presets = cls.get_presets()
        options = sorted(presets)
        default = options[0] if options else ""
        return io.Schema(
            node_id="UC_VLMSysInstrAdvMotionPresets",
            display_name="VLM System Instruction Advanced Motion Presets",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="vlm_system_instruction_advanced_motion_preset", options=options, default=default), io.Boolean.Input("jailbreak", default=False), io.String.Input("system_query", multiline=True, default=""), io.String.Input("user_query", multiline=True, default="")],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset, jailbreak, system_query, user_query) -> io.NodeOutput:
        return io.NodeOutput(cls._assemble_instruction(preset, jailbreak, system_query, user_query))
