from comfy_api.latest import io

from ..minimax_h3_vlm_presets import minimax_h3_system_instructions_vlm, minimax_h3_vlm_jailbreak_prefix, minimax_h3_vlm_jailbreak_suffix
from ..minimax_h3_vlm_experimental_presets import minimax_h3_system_instructions_vlm_experimental


class UC_MiniMaxH3VLMSysInstrPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        return minimax_h3_system_instructions_vlm

    @classmethod
    def define_schema(cls):
        options = list(cls.get_presets())
        return io.Schema(
            node_id="UC_MiniMaxH3VLMSysInstrPresets",
            display_name="MiniMax H3 VLM System Instruction Presets",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="minimax_h3_vlm_system_instruction_preset", options=options, default=options[0] if options else "")],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset) -> io.NodeOutput:
        return io.NodeOutput(cls.get_presets().get(preset, ""))


class UC_MiniMaxH3VLMSysInstrAdvPresets(io.ComfyNode):
    @classmethod
    def get_presets(cls):
        return minimax_h3_system_instructions_vlm

    @classmethod
    def define_schema(cls):
        options = list(cls.get_presets())
        return io.Schema(
            node_id="UC_MiniMaxH3VLMSysInstrAdvPresets",
            display_name="MiniMax H3 VLM System Instruction Advanced Presets",
            category="advanced/text",
            inputs=[
                io.Combo.Input("preset", display_name="minimax_h3_vlm_system_instruction_advanced_preset", options=options, default=options[0] if options else ""),
                io.String.Input("system_query", multiline=True, default=""),
                io.String.Input("user_query", multiline=True, default=""),
                io.Boolean.Input("jailbreak", default=False),
                io.String.Input("system_query_additional", multiline=True, default=""),
            ],
            outputs=[io.String.Output(display_name="system_instruction")],
        )

    @classmethod
    def execute(cls, preset, system_query, user_query, jailbreak=False, system_query_additional="") -> io.NodeOutput:
        result = cls.get_presets().get(preset, "")
        if jailbreak:
            result = minimax_h3_vlm_jailbreak_prefix + "\n\n" + result
        if user_query and user_query.strip():
            result += "\n\nRequested target-video requirements:\n" + user_query.strip()
        if system_query_additional and system_query_additional.strip():
            result += "\n\n" + system_query_additional.strip()
        if jailbreak:
            result += "\n\n" + minimax_h3_vlm_jailbreak_suffix
        if system_query and system_query.strip():
            result += "\n\nHighest-priority system override:\n" + system_query.strip()
        return io.NodeOutput(result)


class UC_MiniMaxH3VLMSysInstrPresetsExperimental(UC_MiniMaxH3VLMSysInstrPresets):
    @classmethod
    def get_presets(cls):
        return minimax_h3_system_instructions_vlm_experimental

    @classmethod
    def define_schema(cls):
        options = list(cls.get_presets())
        return io.Schema(
            node_id="UC_MiniMaxH3VLMSysInstrPresetsExperimental",
            display_name="MiniMax H3 VLM System Instruction Presets Experimental",
            category="advanced/text",
            inputs=[io.Combo.Input("preset", display_name="minimax_h3_vlm_system_instruction_preset_experimental", options=options, default=options[0] if options else "")],
            outputs=[io.String.Output(display_name="system_instruction")],
        )


class UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental(UC_MiniMaxH3VLMSysInstrAdvPresets):
    @classmethod
    def get_presets(cls):
        return minimax_h3_system_instructions_vlm_experimental

    @classmethod
    def define_schema(cls):
        options = list(cls.get_presets())
        return io.Schema(
            node_id="UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental",
            display_name=("MiniMax H3 VLM System Instruction Advanced Presets Experimental"),
            category="advanced/text",
            inputs=[
                io.Combo.Input("preset", display_name=("minimax_h3_vlm_system_instruction_advanced_preset_experimental"), options=options, default=options[0] if options else ""),
                io.String.Input("system_query", multiline=True, default=""),
                io.String.Input("user_query", multiline=True, default=""),
                io.Boolean.Input("jailbreak", default=False),
                io.String.Input("system_query_additional", multiline=True, default=""),
            ],
            outputs=[io.String.Output(display_name="system_instruction")],
        )
