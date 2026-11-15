import pathlib
import sys
import types

CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_text_nodes_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_text_nodes_test.nodes import text_nodes


def test_text_concatenate_autogrow_schema_uses_wildcard_links():
    schema = text_nodes.UC_TextConcatenateAutogrow.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    text_inputs = inputs["text_inputs"]

    assert schema.node_id == "UC_TextConcatenateAutogrow"
    assert schema.display_name == "Concatenate Text (Autogrow)"
    assert schema.category == "advanced/text"
    assert inputs["delimiter"].get_io_type() == "*"
    assert inputs["delimiter"].optional is True
    assert text_inputs.optional is True
    assert text_inputs.template.input.get_io_type() == "*"
    assert text_inputs.template.input.optional is True
    assert text_inputs.template.names == [f"text_{index}" for index in range(1, 101)]
    assert text_inputs.template.min == 0
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].display_name == "concatenated_text"


def test_text_concatenate_autogrow_joins_in_numeric_socket_order():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(delimiter=" | ", text_inputs={"text_10": "ten", "text_2": "two", "text_1": "one"})

    assert output.args == ("one | two | ten",)


def test_text_concatenate_autogrow_converts_arbitrary_values_to_strings():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(delimiter=0, text_inputs={"text_1": 12, "text_2": True, "text_3": None})

    assert output.args == ("120True0None",)


def test_text_concatenate_autogrow_supports_empty_and_newline_delimiters():
    compact = text_nodes.UC_TextConcatenateAutogrow.execute(delimiter="", text_inputs={"text_1": "alpha", "text_2": "beta"})
    multiline = text_nodes.UC_TextConcatenateAutogrow.execute(delimiter="\n", text_inputs={"text_1": "alpha", "text_2": "beta"})

    assert compact.args == ("alphabeta",)
    assert multiline.args == ("alpha\nbeta",)


def test_text_concatenate_autogrow_uses_empty_delimiter_when_disconnected():
    output = text_nodes.UC_TextConcatenateAutogrow.execute(text_inputs={"text_1": "alpha", "text_2": "beta"})

    assert output.args == ("alphabeta",)


def test_text_concatenate_autogrow_accepts_no_inputs():
    output = text_nodes.UC_TextConcatenateAutogrow.execute()

    assert output.args == ("",)


def test_text_concatenate_lists_schema_receives_and_returns_complete_lists():
    schema = text_nodes.UC_TextConcatenateListsAutogrow.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_TextConcatenateListsAutogrow"
    assert schema.display_name == "Concatenate Text Lists (Autogrow)"
    assert schema.is_input_list is True
    assert inputs["delimiter"].get_io_type() == "*"
    assert inputs["text_inputs"].template.input.get_io_type() == "*"
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].is_output_list is True


def test_text_concatenate_lists_broadcasts_scalars_and_aligns_lists():
    output = text_nodes.UC_TextConcatenateListsAutogrow.execute(delimiter=[" "], text_inputs={"text_1": ["prefix"], "text_2": ["one", "two", "three"], "text_3": ["suffix"]})

    assert output.args == (["prefix one suffix", "prefix two suffix", "prefix three suffix"],)


def test_text_concatenate_lists_aligns_delimiters_and_repeats_final_values():
    output = text_nodes.UC_TextConcatenateListsAutogrow.execute(delimiter=["/", " | "], text_inputs={"text_2": ["A", "B"], "text_1": [1, 2, 3]})

    assert output.args == (["1/A", "2 | B", "3 | B"],)


def test_text_concatenate_lists_accepts_no_inputs():
    output = text_nodes.UC_TextConcatenateListsAutogrow.execute()

    assert output.args == ([""],)


def test_newline_node_has_no_inputs_and_outputs_one_newline():
    schema = text_nodes.UC_Newline.define_schema()

    assert schema.node_id == "UC_Newline"
    assert schema.display_name == r"\n"
    assert schema.inputs == []
    assert schema.outputs[0].get_io_type() == "STRING"
    assert schema.outputs[0].display_name == r"\n"
    assert text_nodes.UC_Newline.execute().args == ("\n",)


def test_load_text_file_path_schema():
    schema = text_nodes.UC_LoadTextFilePath.define_schema()
    assert schema.node_id == "UC_LoadTextFilePath"
    assert schema.display_name == "Load Text (Path)"
    assert schema.category == "advanced/text"
    input_ids = [inp.id for inp in schema.inputs]
    assert input_ids == ["file_path", "delimiter", "custom_delimiter", "trim_entries", "skip_empty"]
    assert schema.outputs[0].id == "text_list"
    assert schema.outputs[0].is_output_list is True


def test_load_text_helpers_splitting():
    from utils_collection_text_nodes_test.helpers.text_helpers import split_text_content

    # Period
    assert split_text_content("First sentence. Second sentence.", "Period [.]") == ["First sentence", "Second sentence"]

    # Comma
    assert split_text_content("apple, banana , cherry,", "Comma [,]", trim_entries=True, skip_empty=True) == ["apple", "banana", "cherry"]

    # Colon
    assert split_text_content("key: value: other", "Colon [:]") == ["key", "value", "other"]

    # Semicolon
    assert split_text_content("one; two; three;", "Semicolon [;]") == ["one", "two", "three"]

    # Newline (mixed CRLF and LF)
    assert split_text_content("line 1\r\nline 2\nline 3\n\n", "Newline [\n] or [\r\n]") == ["line 1", "line 2", "line 3"]

    # Double Newline (Paragraphs)
    para_text = "Paragraph 1.\nStill para 1.\n\nParagraph 2.\n\n\nParagraph 3."
    assert split_text_content(para_text, "Double Newline [\n\n] (Paragraphs)") == ["Paragraph 1.\nStill para 1.", "Paragraph 2.", "Paragraph 3."]

    # Space
    assert split_text_content("word1 word2   word3", "Space [ ]") == ["word1", "word2", "word3"]

    # Tab
    assert split_text_content("col1\tcol2\tcol3", "Tab [\t]") == ["col1", "col2", "col3"]

    # Pipe
    assert split_text_content("item1 | item2 | item3", "Pipe [|]") == ["item1", "item2", "item3"]

    # Triple Quote
    tq_text = 'line 1\nline 2\n"""\nline 1\nline 2\nline 3\n"""\nline 1line 2\nline 3\nline 4'
    assert split_text_content(tq_text, 'Triple Quote ["""]') == ["line 1\nline 2", "line 1\nline 2\nline 3", "line 1line 2\nline 3\nline 4"]

    # Triple Equals
    te_text = "line 1\nline 2\n===\nline 1\nline 2\nline 3\n===\nline 1line 2\nline 3\nline 4"
    assert split_text_content(te_text, "Triple Equals [===]") == ["line 1\nline 2", "line 1\nline 2\nline 3", "line 1line 2\nline 3\nline 4"]

    # Triple Dash
    td_text = "section 1\n---\nsection 2\n---\nsection 3"
    assert split_text_content(td_text, "Triple Dash [---]") == ["section 1", "section 2", "section 3"]

    # None (Single Entry)
    assert split_text_content("  entire content  ", "None (Single Entry)", trim_entries=False) == ["  entire content  "]

    # Custom
    assert split_text_content("a<=>b<=>c", "Custom", custom_delimiter="<=>") == ["a", "b", "c"]


def test_load_text_file_path_execute_and_validation(tmp_path):
    from utils_collection_text_nodes_test.helpers.text_helpers import read_arbitrary_text_file

    # 1. UTF-8 file
    utf8_file = tmp_path / "test_utf8.txt"
    utf8_file.write_text("prompt 1\nprompt 2\nprompt 3\n", encoding="utf-8")

    out = text_nodes.UC_LoadTextFilePath.execute(str(utf8_file), "Newline [\n] or [\r\n]")
    assert out.args == (["prompt 1", "prompt 2", "prompt 3"],)

    # 2. UTF-8 with BOM
    bom_file = tmp_path / "test_bom.txt"
    bom_file.write_bytes(b"\xef\xbb\xbfalpha, beta, gamma")
    out = text_nodes.UC_LoadTextFilePath.execute(str(bom_file), "Comma [,]")
    assert out.args == (["alpha", "beta", "gamma"],)

    # 3. UTF-16 file with BOM
    utf16_file = tmp_path / "test_utf16.txt"
    utf16_file.write_text("entry1; entry2; entry3", encoding="utf-16")
    out = text_nodes.UC_LoadTextFilePath.execute(str(utf16_file), "Semicolon [;]")
    assert out.args == (["entry1", "entry2", "entry3"],)

    # 4. Arbitrary binary content loaded as text without crashing
    bin_file = tmp_path / "arbitrary.bin"
    bin_file.write_bytes(b"\x00\x01\x02hello\xffworld\x00\nsecond line\n")
    text_content = read_arbitrary_text_file(str(bin_file))
    assert "hello" in text_content
    assert "world" in text_content

    # 5. Missing file raises ValueError
    import pytest

    with pytest.raises(ValueError, match="Invalid file path"):
        text_nodes.UC_LoadTextFilePath.execute(str(tmp_path / "non_existent.txt"))

    # 6. fingerprint_inputs and validate_inputs
    fp = text_nodes.UC_LoadTextFilePath.fingerprint_inputs(str(utf8_file))
    assert fp != ""
    assert str(utf8_file.name) in fp or str(utf8_file.stat().st_size) in fp

    assert text_nodes.UC_LoadTextFilePath.validate_inputs(str(utf8_file)) is True
    assert text_nodes.UC_LoadTextFilePath.validate_inputs("") == "File path cannot be empty"
    assert "Invalid file path" in text_nodes.UC_LoadTextFilePath.validate_inputs(str(tmp_path / "missing.txt"))


def test_load_text_directory_schema_and_execution(tmp_path):
    import pytest

    schema = text_nodes.UC_LoadTextDirectory.define_schema()
    assert schema.node_id == "UC_LoadTextDirectory"
    assert schema.display_name == "Load Texts (Directory)"
    assert schema.category == "advanced/text"
    assert [inp.id for inp in schema.inputs] == ["directory_path", "start_index", "load_count"]
    assert schema.outputs[0].id == "text_list"
    assert schema.outputs[0].is_output_list is True

    (tmp_path / "b.txt").write_bytes(b"line 1\nline 2")
    (tmp_path / "a.TXT").write_bytes(b"\xef\xbb\xbfalpha")
    (tmp_path / "c.txt").write_text("gamma", encoding="utf-16")
    (tmp_path / "ignore.md").write_text("not txt", encoding="utf-8")
    (tmp_path / "subdir.txt").mkdir()

    out = text_nodes.UC_LoadTextDirectory.execute(str(tmp_path), start_index=0, load_count=2)
    assert out.args == (["alpha", "line 1\nline 2"],)

    sliced = text_nodes.UC_LoadTextDirectory.execute(str(tmp_path), start_index=1, load_count=5)
    assert sliced.args == (["line 1\nline 2", "gamma"],)

    fp_before = text_nodes.UC_LoadTextDirectory.fingerprint_inputs(str(tmp_path), start_index=0, load_count=2)
    assert fp_before != ""
    (tmp_path / "b.txt").write_text("line 1\nline 2 modified", encoding="utf-8")
    fp_after = text_nodes.UC_LoadTextDirectory.fingerprint_inputs(str(tmp_path), start_index=0, load_count=2)
    assert fp_after != fp_before

    assert text_nodes.UC_LoadTextDirectory.validate_inputs(str(tmp_path), 0, 1) is True
    assert "empty" in text_nodes.UC_LoadTextDirectory.validate_inputs("", 0, 1)
    assert "Invalid text directory" in text_nodes.UC_LoadTextDirectory.validate_inputs(str(tmp_path / "missing"), 0, 1)
    assert "No .txt files found" in text_nodes.UC_LoadTextDirectory.validate_inputs(str(tmp_path), 10, 1)

    with pytest.raises(ValueError, match="No .txt files found"):
        text_nodes.UC_LoadTextDirectory.execute(str(tmp_path), start_index=10, load_count=1)
