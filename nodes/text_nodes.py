import os

from comfy_api.latest import io
from ..helpers.helper_functions import join_words_in_text, to_bold_fraktur_style, from_bold_fraktur_style, remove_joiners, unescape_string, repair_and_minify_json
from ..helpers.text_helpers import TEXT_FILE_DELIMITER_OPTIONS, concatenate_aligned_text_inputs, load_text_file_as_list, normalize_text_file_path


class UC_BoldFrakturTextStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_BoldFrakturTextStyle", display_name="Bold Fraktur Text style", category="advanced/text", inputs=[io.String.Input("text", multiline=True, default="", placeholder="Enter text to style...")], outputs=[io.String.Output(display_name="fraktur_text")])

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = to_bold_fraktur_style(text)
        return io.NodeOutput(result)


class UC_UnBoldFrakturTextStyle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_UnBoldFrakturTextStyle", display_name="UnBoldFrakturTextStyle", category="advanced/text", inputs=[io.String.Input("text", multiline=True, default="", placeholder="Enter styled text to convert back...")], outputs=[io.String.Output(display_name="plain_text")])

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = from_bold_fraktur_style(text)
        return io.NodeOutput(result)


class UC_WordJoiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_WordJoiner", display_name="Word Joiner", category="advanced/text", inputs=[io.String.Input("text", multiline=True, default="", placeholder="Enter text to join...")], outputs=[io.String.Output(display_name="joined_text")])

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = join_words_in_text(text)
        return io.NodeOutput(result)


class UC_UnWordJoiner(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_UnWordJoiner", display_name="Remove Word Joiners", category="advanced/text", inputs=[io.String.Input("text", multiline=True, default="", placeholder="Enter text with joiners...")], outputs=[io.String.Output(display_name="unjoined_text")])

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = remove_joiners(text)
        return io.NodeOutput(result)


class UC_JSONMinifyRepair(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_JSONMinifyRepair", display_name="JSON Minify and Repair", category="advanced/text", inputs=[io.String.Input("text", multiline=True, default="", placeholder="Enter prettified or malformed JSON here...")], outputs=[io.String.Output(display_name="json_text")])

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = repair_and_minify_json(text)
        return io.NodeOutput(result)


class UC_StringUnescape(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_StringUnescape", display_name="String Unescape", category="advanced/text", inputs=[io.String.Input("text", multiline=True, default="", placeholder="Enter string with escaped characters...")], outputs=[io.String.Output(display_name="unescaped_text")])

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        result = unescape_string(text)
        return io.NodeOutput(result)


class UC_TextConcatenateAutogrow(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_template = io.Autogrow.TemplateNames(io.AnyType.Input("text", optional=True), names=[f"text_{index}" for index in range(1, 101)], min=0)
        return io.Schema(
            node_id="UC_TextConcatenateAutogrow",
            display_name="Concatenate Text (Autogrow)",
            category="advanced/text",
            inputs=[io.AnyType.Input("delimiter", optional=True, tooltip="Text placed between each connected input."), io.Autogrow.Input("text_inputs", template=text_template, optional=True, tooltip="Values joined in their input order.")],
            outputs=[io.String.Output(display_name="concatenated_text")],
        )

    @classmethod
    def execute(cls, text_inputs: io.Autogrow.Type | None = None, delimiter="") -> io.NodeOutput:
        text_inputs = text_inputs or {}
        ordered_values = [value for _, value in sorted(text_inputs.items(), key=lambda item: int(item[0].removeprefix("text_")))]
        return io.NodeOutput(str(delimiter).join(str(value) for value in ordered_values))


class UC_TextConcatenateListsAutogrow(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        text_template = io.Autogrow.TemplateNames(io.AnyType.Input("text", optional=True), names=[f"text_{index}" for index in range(1, 101)], min=0)
        return io.Schema(
            node_id="UC_TextConcatenateListsAutogrow",
            display_name="Concatenate Text Lists (Autogrow)",
            category="advanced/text",
            is_input_list=True,
            inputs=[io.AnyType.Input("delimiter", optional=True, tooltip="Use one delimiter for every output, or one delimiter per output."), io.Autogrow.Input("text_inputs", template=text_template, optional=True, tooltip="Use one value for every output, or one value per output.")],
            outputs=[io.String.Output("concatenated_text", display_name="concatenated text", is_output_list=True)],
        )

    @classmethod
    def execute(cls, text_inputs: io.Autogrow.Type | None = None, delimiter: list | None = None) -> io.NodeOutput:
        return io.NodeOutput(concatenate_aligned_text_inputs(text_inputs, delimiter))


class UC_Newline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(node_id="UC_Newline", display_name=r"\n", category="advanced/text", inputs=[], outputs=[io.String.Output(display_name=r"\n")])

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput("\n")


class UC_LoadTextFilePath(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadTextFilePath",
            display_name="Load Text (Path)",
            category="advanced/text",
            description="Loads any file as text from a path and outputs entries split by the chosen delimiter as a list.",
            inputs=[
                io.String.Input("file_path", multiline=False, placeholder="path/to/file.txt or X:/path/to/file.txt", tooltip="Path to the file. Supports absolute or relative paths with any OS slash format."),
                io.Combo.Input("delimiter", options=TEXT_FILE_DELIMITER_OPTIONS, default="Newline [\n] or [\r\n]", tooltip="Delimiter used to split the file content into list entries."),
                io.String.Input("custom_delimiter", multiline=False, default="", optional=True, tooltip="Custom delimiter string used when 'Custom' is selected."),
                io.Boolean.Input("trim_entries", default=True, tooltip="Strip leading and trailing whitespace from each split entry."),
                io.Boolean.Input("skip_empty", default=True, tooltip="Omit empty entries produced by splitting."),
            ],
            outputs=[io.String.Output("text_list", display_name="text_list", is_output_list=True, tooltip="List of text entries separated by the selected delimiter.")],
        )

    @classmethod
    def execute(cls, file_path: str, delimiter: str = "Newline [\n] or [\r\n]", custom_delimiter: str = "", trim_entries: bool = True, skip_empty: bool = True) -> io.NodeOutput:
        entries = load_text_file_as_list(file_path=file_path, delimiter=delimiter, custom_delimiter=custom_delimiter, trim_entries=trim_entries, skip_empty=skip_empty)
        return io.NodeOutput(entries)

    @classmethod
    def fingerprint_inputs(cls, file_path: str, **kwargs):
        normalized = normalize_text_file_path(file_path)
        if not normalized or not os.path.isfile(normalized):
            return ""
        stat = os.stat(normalized)
        return f"{normalized}:{stat.st_mtime_ns}:{stat.st_size}"

    @classmethod
    def validate_inputs(cls, file_path: str, **kwargs):
        normalized = normalize_text_file_path(file_path)
        if not normalized:
            return "File path cannot be empty"
        if not os.path.isfile(normalized):
            return f"Invalid file path: {file_path} (resolved to: {normalized})"
        return True
