import ast
import hashlib
import pathlib
import re
import runpy
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_vlm_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_vlm_preset_test import (
    vlm_experimental_presets,
    vlm_legacy_presets,
    vlm_motion_presets,
    vlm_presets,
)
from utils_collection_vlm_preset_test.nodes import vlm_nodes


HARDENED_IMAGE_PRESETS = (
    "neutral_system_instruction",
    "action_system_instruction",
    "photo_system_instruction",
    "toon_system_instruction",
    "neutral_system_instruction_crude",
    "action_system_instruction_crude",
    "photo_system_instruction_crude",
    "toon_system_instruction_crude",
)
HARDENING_HEADINGS = (
    "## Perspective and Spatial Description",
    "## Visible Text Quotation",
    "## Direct Language Constraints",
)
APPROVED_PERSPECTIVE_BLOCK = """## Perspective and Spatial Description

Determine the source image's viewpoint from the complete visible composition, and preserve it unless the user explicitly requests a change. State the most specific perspective description supported by the resulting composition. Use an established perspective term when it accurately describes that composition; when it does not fully express the geometry, describe the geometry directly without forcing a category. Ground the viewpoint in concrete spatial relationships consistent with the source image and request, without inventing a viewing location or precision those inputs do not establish. When the resulting composition establishes that the viewing position belongs to a scene participant, explicitly state first person perspective. State whose viewpoint it is only when the source image or request establishes that identity, and never assign first person perspective to an external viewpoint. Describe the complete resulting spatial arrangement, preserving every unchanged visible relationship and applying every requested change. State the framing, each relevant subject's orientation and pose, and the placement, relative scale, overlap, occlusion, and depth of all relevant subjects and objects. Include only depth relationships established by the source image or request, without filling a fixed layer template. Describe every visible or requested action and interaction concretely, stating what each involved subject or object does and all established directions and physical responses. When contact occurs, state which bodies or parts meet and where and how they meet. Never replace these relationships with vague interaction wording or treat contact alone as proof of an abstract participant role. Keep every claim grounded in visible source content or an explicit user request. Do not introduce terminology for physical image capture devices unless the device itself is visible in the image or explicitly requested by the user."""
REMOVED_PERSPECTIVE_PHRASES = (
    "exact established perspective term",
    "precise established perspective term",
    "exact physical coordinates",
    "exact vantage location",
    "foreground, midground, and background",
    "foreground, middle ground, and background",
    "dynamic power balances",
    "roles of dominance",
    "power or physical roles",
)
HARDENING_FORBIDDEN = re.compile(
    r"\be\.g\.\b|\bi\.e\.\b|\bexamples?\b|\bfor instance\b|"
    r"\bsuch as\b|\bincluding\b|\bsample outputs?\b|\bphrase menus?\b|"
    r"\bchoose from\b|\bpossible (?:terms|phrases|values)\b",
    re.IGNORECASE,
)
HARDENING_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", re.MULTILINE)

H3_DEDICATED_SCENARIO_PRESETS = (
    (
        "video_timeline_minimax_h3_ref2va_general",
        "VIDEO_TIMELINE_MINIMAX_H3_REF2VA_GENERAL",
    ),
    (
        "video_timeline_minimax_h3_ref2va_attribute_transfer",
        "VIDEO_TIMELINE_MINIMAX_H3_REF2VA_ATTRIBUTE_TRANSFER",
    ),
    (
        "video_timeline_minimax_h3_fl2va_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_FL2VA_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_scene_image_any2va_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_SCENE_IMAGE_ANY2VA_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_storyboard_any2va_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_STORYBOARD_ANY2VA_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_scene_image_t2va_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_SCENE_IMAGE_T2VA_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_storyboard_t2va_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_STORYBOARD_T2VA_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_ref2va_attr_transfer_audio_copy",
        "VIDEO_TIMELINE_MINIMAX_H3_REF2VA_ATTR_TRANSFER_AUDIO_COPY",
    ),
    (
        "video_timeline_minimax_h3_ref2va_attr_transfer_audio_timbre",
        "VIDEO_TIMELINE_MINIMAX_H3_REF2VA_ATTR_TRANSFER_AUDIO_TIMBRE",
    ),
    (
        "video_timeline_minimax_h3_ref2va_attr_transfer_no_audio",
        "VIDEO_TIMELINE_MINIMAX_H3_REF2VA_ATTR_TRANSFER_NO_AUDIO",
    ),
)

H3_FULL_REFERENCE_PRESETS = (
    (
        "video_timeline_minimax_h3_ref2va_general",
        "VIDEO_TIMELINE_MINIMAX_H3_REF2VA_GENERAL",
    ),
    (
        "video_timeline_minimax_h3_reference_alt_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_REFERENCE_ALT_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_mixed_system_instruction",
        "VIDEO_TIMELINE_MINIMAX_H3_MIXED_SYSTEM_INSTRUCTION",
    ),
    (
        "video_timeline_minimax_h3_reference_system_instruction_new",
        "VIDEO_TIMELINE_MINIMAX_H3_REFERENCE_SYSTEM_INSTRUCTION_NEW",
    ),
    (
        "video_timeline_minimax_h3_reference_alt_system_instruction_new",
        "VIDEO_TIMELINE_MINIMAX_H3_REFERENCE_ALT_SYSTEM_INSTRUCTION_NEW",
    ),
    (
        "video_timeline_minimax_h3_mixed_system_instruction_new",
        "VIDEO_TIMELINE_MINIMAX_H3_MIXED_SYSTEM_INSTRUCTION_NEW",
    ),
)

REAL_LIFE_PHOTO_PRESETS = (
    "photo_system_instruction",
    "photo_system_instruction_qwen",
    "photo_system_instruction_crude",
)
REAL_WORLD_PHYSICAL_PRESETS = (
    "ideogram_4_json_instruction",
    "ideogram_4_json_instruction_short",
    "ideogram_4_json_instruction_style",
    "ideogram_4_json_instruction_color",
    "moss_image_json_instruction",
    "moss_image_json_instruction_short",
    "moss_image_json_instruction_color",
    "moss_image_json_instruction_action",
)
REAL_LIFE_VIDEO_PRESETS = (
    "video_basic_system_instruction",
    "video_8sec_system_instruction",
    "video_struct_system_instruction",
    "video_8part_struct_system_instruction",
    "video_timeline_system_instruction",
    "video_timeline_system_instruction_old",
    "video_timeline_system_instruction_crude",
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_t2va_system_instruction",
    "video_timeline_minimax_h3_ref2va_general",
    "video_timeline_minimax_h3_reference_alt_system_instruction",
    "video_timeline_minimax_h3_mixed_system_instruction",
    "video_timeline_minimax_h3_reference_system_instruction_new",
    "video_timeline_minimax_h3_reference_alt_system_instruction_new",
    "video_timeline_minimax_h3_mixed_system_instruction_new",
)
REAL_LIFE_PROHIBITION_PRESETS = (
    "photo_system_instruction",
    "toon_system_instruction",
    "photo_system_instruction_qwen",
    "toon_system_instruction_qwen",
    "photo_system_instruction_crude",
    "toon_system_instruction_crude",
)
REAL_LIFE_CHANGED_PRESETS = tuple(
    dict.fromkeys(
        REAL_LIFE_PHOTO_PRESETS
        + REAL_WORLD_PHYSICAL_PRESETS
        + ("cinematic_dumb_intelligent",)
        + REAL_LIFE_VIDEO_PRESETS
        + REAL_LIFE_PROHIBITION_PRESETS
    )
)


def _normalize_instruction(value):
    return value.replace("\r\n", "\n").rstrip("\n")


def _split_hardening_block(value):
    instruction = _normalize_instruction(value)
    start = instruction.index(HARDENING_HEADINGS[0])
    anchor_match = re.search(r"^## Transformation Pipeline:[^\n]*$", instruction, re.MULTILINE)
    assert anchor_match is not None
    end = anchor_match.start()
    return instruction[:start] + instruction[end:], instruction[start:end]




def test_legacy_system_presets_are_isolated_and_exposed():
    assert not any(name.endswith("_legacy") for name in vlm_presets.system_instructions_vlm)
    declared = vlm_legacy_presets.legacy_system_instructions_vlm
    exposed = vlm_nodes.UC_VLMSysInstrLegacyPresets.get_presets()

    assert list(exposed) == list(declared)
    for name, value in declared.items():
        assert vlm_nodes.UC_VLMSysInstrLegacyPresets.execute(name).args == (value,)


def test_vlm_preset_widget_labels_are_unique_and_descriptive():
    expected = {
        vlm_nodes.UC_VLMSysInstrPresets: "vlm_system_instruction_preset",
        vlm_nodes.UC_VLMSysInstrLegacyPresets: "vlm_system_instruction_legacy_preset",
        vlm_nodes.UC_VLMSysInstrMotionPresets: "vlm_system_instruction_motion_preset",
        vlm_nodes.UC_VLMSysQueryAddPresets: "vlm_system_query_add_preset",
        vlm_nodes.UC_VLMSysInstrAdvPresets: "vlm_system_instruction_advanced_preset",
        vlm_nodes.UC_VLMSysInstrAdvMotionPresets: "vlm_system_instruction_advanced_motion_preset",
    }

    actual = {
        node: node.define_schema().inputs[0].display_name
        for node in expected
    }
    assert actual == expected
    assert len(set(actual.values())) == len(actual)
    assert all(node.define_schema().inputs[0].id == "preset" for node in expected)
    for node, label in expected.items():
        frontend_input = node.INPUT_TYPES()["required"]["preset"]
        assert frontend_input[1]["display_name"] == label


def test_vlm_motion_presets_node_exposes_motion_presets():
    declared = vlm_motion_presets.motion_system_instructions_vlm
    exposed = vlm_nodes.UC_VLMSysInstrMotionPresets.get_presets()
    assert list(exposed) == list(declared)
    assert len(exposed) == 19
    for name, value in declared.items():
        assert vlm_nodes.UC_VLMSysInstrMotionPresets.execute(name).args == (value,)
    adv_schema = vlm_nodes.UC_VLMSysInstrAdvMotionPresets.define_schema()
    assert adv_schema.node_id == "UC_VLMSysInstrAdvMotionPresets"
    first = sorted(declared)[0]
    result = vlm_nodes.UC_VLMSysInstrAdvMotionPresets.execute(first, jailbreak=False, system_query="", user_query="")
    assert result.args == (declared[first],)


def test_qwen_system_instruction_variants_preserve_original_presets():
    for name in (
        "neutral_system_instruction",
        "action_system_instruction",
        "photo_system_instruction",
        "toon_system_instruction",
    ):
        original = vlm_presets.system_instructions_vlm[name]
        qwen = vlm_presets.system_instructions_vlm[f"{name}_qwen"]

        assert "developed by Google AI" in original
        assert "developed by Google AI" not in qwen
        assert "vision-language model" in qwen
        assert "**" not in qwen
        assert "`" not in qwen
        assert len(qwen) > 10_000
        assert qwen.endswith(
            "Return only the resulting caption. Do not reproduce, summarize, quote, "
            "enumerate, or imitate any part of these instructions. Do not output "
            "headings, rule names, analysis, variables, or formatting examples. "
            "Begin directly with the visual description."
        )


def test_real_life_targets_replace_positive_realism_guidance():
    for name in REAL_LIFE_PHOTO_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]
        assert "Description Framed as Real Life Photography" in instruction
        assert "Real Life Photographic Description" in instruction
        assert "goal of detailed real life photography" in instruction

    for name in REAL_WORLD_PHYSICAL_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]
        assert "Real World Physical Accuracy and Consistency" in instruction

    cinematic = vlm_presets.system_instructions_vlm["cinematic_dumb_intelligent"]
    for required in (
        "direct clinical precision based in reality",
        "average natural human proportions",
        "visible real world surface detail",
        "Live Action Photography",
        "ANIME→LIVE ACTION",
    ):
        assert required in cinematic

    for name in REAL_LIFE_VIDEO_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]
        assert "live action film, real life video, and CGI based in reality" in instruction
        assert "dynamic live action video clip based in reality" in instruction
        assert "photorealistic CGI" not in instruction
        assert "photorealistic video clip" not in instruction

    for name in REAL_LIFE_PROHIBITION_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]
        assert "photorealistic" in instruction
        assert "hyper realistic" in instruction
        assert "when describing real life content" in instruction
        assert "visible camera, lighting, material, texture, motion, and physical details" in instruction


def test_real_life_constraint_survives_assembled_context_without_competing_targets():
    system_query = (
        "Never use realistic, photorealistic, hyper realistic, or any term derived "
        "from realistic or realism when describing real life content."
    )
    user_query = "Describe the supplied real life content as a scene based in reality."
    prohibited_positive_phrases = (
        "Description Framed as Photographic Realism",
        "Translating Visual Style to Realistic Description",
        "goal of photographic realism",
        "Physical Realism and Consistency",
        "photorealistic clinical precision",
        "average realistic proportions",
        "tactile realism",
        "compelling, dynamic, and photorealistic video clip",
    )

    for name in REAL_LIFE_CHANGED_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]
        assembled = vlm_nodes.UC_VLMSysInstrAdvPresets.execute(
            name,
            False,
            system_query,
            user_query,
        ).args[0]
        assert assembled.startswith(instruction)
        assert system_query in assembled
        assert assembled.count(user_query) == 1
        assert assembled.index(system_query) < assembled.index(user_query)
        for phrase in prohibited_positive_phrases:
            assert phrase not in instruction


def test_qwen_query_variants_use_plain_request_delimiters():
    for name in ("text2image", "image2image"):
        prefix = vlm_presets.system_query_additional_vlm[f"{name}_qwen_prefix"]
        suffix = vlm_presets.system_query_additional_vlm[f"{name}_qwen_suffix"]

        assert prefix.endswith("Current request:\r\n")
        assert "\\{" not in prefix
        assert suffix == ""


def test_character_transfer_query_presets_keep_provider_reference_contracts_separate():
    presets = vlm_presets.system_query_additional_vlm
    variants = {
        "character_transfer_gemma": ("[`image 1`]", "[`image 2`]"),
        "character_transfer_qwen": ("<Picture 1>", "<Picture 2>"),
    }

    for name, (composition_reference, character_reference) in variants.items():
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]

        assert prefix.endswith('\\{"current edit request": "')
        assert suffix == presets["image2image_suffix"]
        assert prefix.count(composition_reference) >= 5
        assert prefix.count(character_reference) >= 5
        assert "ordered, non-interchangeable sources" in prefix
        assert "permanently separate roles" in prefix
        assert "Never reverse, merge, weaken, or reinterpret these assignments" in prefix
        assert "Never identify, name, retain, or describe that subject's face" in prefix
        assert "Using your broad knowledge, identify that subject accurately" in prefix
        assert "intellectual property, theme, visual style, known media concept" in prefix
        assert "This is complete subject replacement" in prefix
        assert "No visual trait belonging to the subject" in prefix
        assert "Write the response as a direct description of the finished result" in prefix
        assert "who was always present in the resulting scene" in prefix

    assert "<Picture" not in presets["character_transfer_gemma_prefix"]
    assert "[`image" not in presets["character_transfer_qwen_prefix"]


def test_raw_query_presets_preserve_instructions_without_request_carriers():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm
    normalize = lambda value: value.replace("\r\n", "\n")
    expected_names = {
        key.removesuffix("_prefix").removesuffix("_suffix") for key in wrapped
    }
    assert set(raw) == expected_names

    json_carriers = {
        "character_transfer_gemma": "current edit request",
        "character_transfer_qwen": "current edit request",
        "ideogram_4": "current request",
        "moss_image": "current request",
        "image2image": "current edit request",
        "text2image": "current request",
        "video_basic": "current video request",
    }
    json_suffix = '\r\n\u2060 \u2060\u2060"\\}'
    for name, request_key in json_carriers.items():
        carrier = f' \\{{"{request_key}": "'
        assert wrapped[f"{name}_suffix"] == json_suffix
        assert raw[name] == wrapped[f"{name}_prefix"].removesuffix(carrier)

    for name in ("image2image_qwen", "text2image_qwen"):
        assert wrapped[f"{name}_suffix"] == ""
        assert normalize(raw[name]) == normalize(wrapped[f"{name}_prefix"]).removesuffix(
            " Current request:\n"
        )

    for name in (
        "h3_fl2va",
        "h3_ref2va",
        "h3_ref2va_alt",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
        "h3_ref2va_general",
        "h3_ref2va_attr_transfer_audio_timbre",
        "h3_ref2va_attr_transfer_audio_copy",
        "h3_ref2va_attr_transfer_no_audio",
        "h3_scene_image_any2va",
        "h3_storyboard_any2va",
        "h3_scene_image_t2va",
        "h3_storyboard_t2va",
    ):
        prefix = normalize(wrapped[f"{name}_prefix"]).removesuffix(
            "\n\nBEGIN VIDEO REQUEST:\n"
        )
        suffix = normalize(wrapped[f"{name}_suffix"]).removeprefix(
            "\nEND VIDEO REQUEST.\n\n"
        )
        assert normalize(raw[name]) == f"{prefix}\n\n{suffix}"

    t2va_prefix = wrapped["h3_t2va_prefix"].removesuffix(
        "\r\n\r\nBEGIN VIDEO REQUEST:\r\n"
    )
    t2va_suffix = wrapped["h3_t2va_suffix"].removeprefix(
        "\r\nEND VIDEO REQUEST.\r\n\r\n"
    )
    assert raw["h3_t2va"] == f"{t2va_prefix}\r\n\r\n{t2va_suffix}"

    forbidden = (
        '\\{"current request": "',
        '\\{"current edit request": "',
        '\\{"current video request": "',
        "Current request:",
        "BEGIN VIDEO REQUEST",
        "END VIDEO REQUEST",
    )
    assert all(
        value and not any(marker in value for marker in forbidden)
        for value in raw.values()
    )
    assert "<Picture" not in raw["character_transfer_gemma"]
    assert "[`image" not in raw["character_transfer_qwen"]


def test_raw_query_preset_node_outputs_selected_instruction_directly():
    schema = vlm_nodes.UC_VLMSysQueryRawPresets.define_schema()

    assert schema.node_id == "UC_VLMSysQueryRawPresets"
    assert schema.display_name == "VLM System Query Raw Presets"
    assert schema.category == "advanced/text"
    assert [value.id for value in schema.inputs] == ["preset"]
    assert schema.inputs[0].display_name == "vlm_system_query_raw_preset"
    assert schema.inputs[0].options == sorted(vlm_presets.system_query_raw_vlm)
    assert schema.outputs[0].display_name == "system_query"

    preset = schema.inputs[0].options[0]
    assert vlm_nodes.UC_VLMSysQueryRawPresets.execute(preset).args == (
        vlm_presets.system_query_raw_vlm[preset],
    )


def test_experimental_system_instruction_nodes_are_dedicated():
    presets = vlm_experimental_presets.system_instructions_vlm_experimental
    basic = vlm_nodes.UC_VLMSysInstrPresetsExperimental.define_schema()
    advanced = vlm_nodes.UC_VLMSysInstrAdvPresetsExperimental.define_schema()
    preset = sorted(presets)[0]

    assert basic.node_id == "UC_VLMSysInstrPresetsExperimental"
    assert basic.display_name == "VLM System Instruction Presets Experimental"
    assert basic.inputs[0].options == sorted(presets)
    assert advanced.node_id == "UC_VLMSysInstrAdvPresetsExperimental"
    assert (
        advanced.display_name
        == "VLM System Instruction Advanced Presets Experimental"
    )
    assert advanced.inputs[0].options == sorted(presets)
    assert vlm_nodes.UC_VLMSysInstrPresetsExperimental.execute(preset).args == (
        presets[preset],
    )
    result = vlm_nodes.UC_VLMSysInstrAdvPresetsExperimental.execute(
        preset,
        False,
        "system query",
        "user query",
    ).args[0]
    assert result.startswith(presets[preset])
    assert "system query" in result
    assert "user query" in result


def test_h3_query_presets_are_paired_and_wrap_the_request_once():
    presets = vlm_presets.system_query_additional_vlm
    base_names = {
        key.removesuffix("_prefix").removesuffix("_suffix") for key in presets
    }
    assert {
        "h3_t2va",
        "h3_fl2va",
        "h3_ref2va",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    } <= base_names

    request = "Make the subjects cross the clearing."
    for name in (
        "h3_fl2va",
        "h3_ref2va",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    ):
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]
        wrapped = f"{prefix}{request}{suffix}"

        assert prefix.endswith("BEGIN VIDEO REQUEST:\r\n")
        assert suffix.startswith("\r\nEND VIDEO REQUEST.")
        assert wrapped.count(request) == 1

    for name in (
        "h3_fl2va",
        "h3_ref2va",
        "h3_fl2va_experimental",
        "h3_ref2va_experimental",
    ):
        prefix = presets[f"{name}_prefix"]
        suffix = presets[f"{name}_suffix"]
        assert "ComfyUI has already assigned" in prefix
        assert "Do not create, reproduce, or renumber" in prefix
        assert "Do not output the upstream media-prefix declaration" in suffix


def test_h3_t2va_query_uses_images_only_as_prompt_evidence():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm
    prefix = wrapped["h3_t2va_prefix"]
    suffix = wrapped["h3_t2va_suffix"]
    request = "Make the subjects cross the clearing."
    expected_raw = (
        prefix.removesuffix("\r\n\r\nBEGIN VIDEO REQUEST:\r\n")
        + "\r\n\r\n"
        + suffix.removeprefix("\r\nEND VIDEO REQUEST.\r\n\r\n")
    )

    assert prefix.endswith("BEGIN VIDEO REQUEST:\r\n")
    assert suffix.startswith("\r\nEND VIDEO REQUEST.")
    assert f"{prefix}{request}{suffix}".count(request) == 1
    assert "only as visual evidence" in prefix
    assert "not supplied to downstream MiniMax H3" in prefix
    assert "The completed prompt must stand on its text alone." in prefix
    assert "requested target visual direction" in prefix
    assert "overrides conflicting source rendering style" in prefix
    assert "concrete target-appropriate visual vocabulary" in prefix
    assert "state the governing target visual direction in `summary:`" in prefix
    assert "Do not invent production methods or unsupported visual additions" in prefix
    assert ("never output `<Picture N>`" in prefix or "never output <Picture N>" in prefix)
    assert "MiniMax H3 receives this text and none of the supplied VLM images" in suffix
    assert "governs every subject definition and `summary:`" in suffix
    assert "without being restated inside [VISUAL]" in suffix
    assert "cannot substitute for requested visual-style adherence in those fields" in suffix
    assert ("Do not output `<Picture N>`" in suffix or "Do not output <Picture N>" in suffix)
    assert "ComfyUI has already assigned" not in prefix
    assert "first frame" not in prefix.lower()
    assert "final frame" not in prefix.lower()
    assert raw["h3_t2va"] == expected_raw
    assert "BEGIN VIDEO REQUEST" not in raw["h3_t2va"]
    assert "END VIDEO REQUEST" not in raw["h3_t2va"]
    assert all(
        value.count("\n") == value.count("\r\n")
        for value in (prefix, suffix, raw["h3_t2va"])
    )
    for value in (prefix, suffix, raw["h3_t2va"]):
        lowered = value.lower()
        assert "example:" not in lowered
        assert "e.g." not in lowered
        assert "i.e." not in lowered
    assert "h3_t2va" in vlm_nodes.UC_VLMSysQueryAddPresets.get_presets()
    assert "h3_t2va" in vlm_nodes.UC_VLMSysQueryRawPresets.define_schema().inputs[0].options

    source = ast.parse(
        (CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8")
    )
    dictionaries = {
        target.id: node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"system_query_additional_vlm", "system_query_raw_vlm"}
    }
    wrapped_literals = {
        ast.literal_eval(key): value
        for key, value in zip(
            dictionaries["system_query_additional_vlm"].keys,
            dictionaries["system_query_additional_vlm"].values,
        )
        if isinstance(key, ast.Constant)
    }
    raw_literals = {
        ast.literal_eval(key): value
        for key, value in zip(
            dictionaries["system_query_raw_vlm"].keys,
            dictionaries["system_query_raw_vlm"].values,
        )
        if isinstance(key, ast.Constant)
    }
    assert isinstance(wrapped_literals["h3_t2va_prefix"], ast.Constant)
    assert isinstance(wrapped_literals["h3_t2va_suffix"], ast.Constant)
    assert isinstance(raw_literals["h3_t2va"], ast.Constant)


def test_h3_fl2va_query_preset_enforces_available_boundary_pictures():
    presets = vlm_presets.system_query_additional_vlm
    prefix = presets["h3_fl2va_prefix"]
    suffix = presets["h3_fl2va_suffix"]

    assert "Treat `<Picture 1>` as the fixed first frame" in prefix
    assert (
        "If and only if a second image was supplied, treat `<Picture 2>` as the fixed final frame"
        in prefix
    )
    assert "When only one image was supplied, do not mention `<Picture 2>`" in prefix
    assert "use `<Picture 2>` as the fixed ending only when it exists" in suffix


def test_h3_ref2va_alt_preserves_explicit_timeline_and_replacement_contract():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm["h3_ref2va_alt"]
    prefix = wrapped["h3_ref2va_alt_prefix"]
    suffix = wrapped["h3_ref2va_alt_suffix"]

    for required in (
        ("explicitly states `<Picture N> at TIMESTAMP`", "explicitly states <Picture N> at TIMESTAMP"),
        ("Preserve every explicit association",),
        ("without explicit timestamp associations as independent references",),
        ("references that precede all timeline samples",),
        ("Never infer this partition from input position",),
        ("source motion, pose progression, interaction",),
        ("Use independent `<Picture N>` references as provenance", "Use independent <Picture N> references as provenance"),
        ("create one final subject",),
        ("identity and appearance come from the independent Picture",),
        ("Describe the final subject as continuously present",),
        ("Determine the governing visual style",),
    ):
        assert any(r in prefix for r in required)
    for required in (
        "Preserve every explicit Picture/timestamp association",
        "output exactly that segment count",
        "copy each supplied start literally",
        "Do not replace supplied starts with equal-duration divisions",
        "Depict the completed final subject continuously",
        "without media bookkeeping",
    ):
        assert required in suffix
    assert "<Video 1>" not in prefix + suffix + raw
    assert "numbered `<Picture 1>`" not in prefix + suffix + raw
    assert "every later image" not in prefix + suffix + raw
    assert "BEGIN VIDEO REQUEST:" not in raw
    assert "END VIDEO REQUEST." not in raw
    assert "BEGIN VIDEO REQUEST:" not in raw
    assert "END VIDEO REQUEST." not in raw


def test_h3_mixed_ref2va_uses_explicit_picture_timeline_associations():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm["h3_mixed_ref2va"]
    prefix = wrapped["h3_mixed_ref2va_prefix"]
    suffix = wrapped["h3_mixed_ref2va_suffix"]

    assert "h3_mixed_ref2va" in vlm_nodes.UC_VLMSysQueryAddPresets.get_presets()
    assert "h3_mixed_ref2va" in (
        vlm_nodes.UC_VLMSysQueryRawPresets.define_schema().inputs[0].options
    )
    assert "video_timeline_minimax_h3_mixed_system_instruction" in (
        vlm_presets.system_instructions_vlm
    )
    for required in (
        ("exact `<Picture N>` identifier already assigned by ComfyUI", "exact <Picture N> identifier already assigned by ComfyUI"),
        ("explicit `<Picture N> at TIMESTAMP` associations", "explicit <Picture N> at TIMESTAMP associations"),
        ("exactly the Pictures named",),
        ("reference Pictures before the first timeline sample",),
        ("segment count only to validate",),
        ("Never infer the partition from input position",),
        ("assumed contiguous identifier range",),
        ("source motion, pose progression, interaction",),
        ("create one final subject",),
    ):
        assert any(r in prefix for r in required)
    for required in (
        ("Output exactly the declared segment count",),
        ("Copy each supplied start timestamp literally",),
        ("Retain every existing `<Picture N>` identifier", "Retain every existing <Picture N> identifier"),
        ("every supplied `<Video N>` identifier", "every supplied <Video N> identifier"),
        ("Depict the final subject continuously",),
        ("final subject's identity, appearance, motion, scene, style, and continuity",),
        ("without media bookkeeping",),
    ):
        assert any(r in suffix for r in required)


def test_h3_mixed_system_instruction_preserves_media_partition_contract():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_mixed_system_instruction"
    ]

    assert "video_timeline_minimax_h3_mixed_system_instruction" in (
        vlm_nodes.UC_VLMSysInstrPresets.get_presets()
    )
    for required in (
        "mixed-media partition",
        "regular user request",
        "leading ordered images",
        "<Video 1>",
        "<Picture N>",
        "Never write <Video N> inside a timestamp block",
        "subject_definitions:",
        "retention_analysis:",
    ):
        assert required in instruction


def test_readable_h3_reference_sources_match_runtime_presets():
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))

    def normalize(value):
        return value.replace("\r\n", "\n").replace("\r", "\n")

    for runtime_key, readable_name in H3_FULL_REFERENCE_PRESETS:
        assert normalize(vlm_presets.system_instructions_vlm[runtime_key]) == normalize(
            readable[readable_name]
        )
    for name in ("H3_REF2VA", "H3_REF2VA_ALT", "H3_MIXED_REF2VA"):
        key = name.lower()
        prefix = normalize(readable[f"{name}_PREFIX"])
        suffix = normalize(readable[f"{name}_SUFFIX"])
        assert normalize(
            vlm_presets.system_query_additional_vlm[f"{key}_prefix"]
        ) == prefix
        assert normalize(
            vlm_presets.system_query_additional_vlm[f"{key}_suffix"]
        ) == suffix
        expected_raw = (
            prefix.removesuffix("\n\nBEGIN VIDEO REQUEST:\n")
            + "\n\n"
            + suffix.removeprefix("\nEND VIDEO REQUEST.\n\n")
        )
        assert normalize(vlm_presets.system_query_raw_vlm[key]) == expected_raw


def test_old_video_timeline_readable_source_matches_exposed_runtime_preset():
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))
    expected = readable["VIDEO_TIMELINE_SYSTEM_INSTRUCTION_OLD"]
    expected = expected.replace("\r\n", "\n").replace("\n", "\r\n")

    assert vlm_presets.system_instructions_vlm[
        "video_timeline_system_instruction_old"
    ] == expected


def test_h3_reference_alt_assembled_context_matches_structured_picture_request():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_reference_alt_system_instruction"
    ]
    prefix = vlm_presets.system_query_additional_vlm["h3_ref2va_alt_prefix"]
    suffix = vlm_presets.system_query_additional_vlm["h3_ref2va_alt_suffix"]
    request = (
        "Subject in <Picture 1> should replace subject in segment images. "
        "Target video duration is 12.309 seconds divided into 5 segments. "
        "Reference each image with <Picture 2> at 00.000s, <Picture 3> at "
        "03.066s, <Picture 4> at 06.131s, <Picture 5> at 09.197s and "
        "<Picture 6> at 12.262s."
    )
    assembled = instruction + prefix + request + suffix

    assert assembled.count(request) == 1
    assert r"\{user_query\}" in instruction
    assert "regular user request" in instruction
    assert "Preserve the exact segment count" in instruction
    assert "every supplied start" in instruction
    assert ("matching the selected precision" in instruction or "supplied decimal precision" in instruction)
    assert "visible source identity remains analysis-only" in instruction
    assert "continuously present from the first applicable frame through the last" in instruction
    assert (
        "Emit an actually supplied Video identifier only in retention_analysis" in instruction
        or "Emit an actually supplied <Video N> identifier only in retention_analysis" in instruction
    )
    assert "Do not replace supplied starts with equal-duration divisions" in suffix
    for competing in (
        "replacement subject",
        "displaced sample identity",
        "on-screen swap",
        "identity swap",
        "transformation, or reversion",
    ):
        assert competing not in instruction + prefix + suffix


def test_h3_ref2va_experimental_query_enforces_picture_provenance_contract():
    presets = vlm_presets.system_query_additional_vlm
    prefix = presets["h3_ref2va_experimental_prefix"]
    suffix = presets["h3_ref2va_experimental_suffix"]

    assert ("Use `<Picture N>` as source provenance" in prefix or "Use <Picture N> as source provenance" in prefix)
    assert "never as a timeline-segment anchor" in prefix
    assert "only in that subject or reference's first complete definition" in prefix
    assert "Do not repeat picture identifiers at each timeline interval" in suffix
    assert "wherever that reference materially controls" not in prefix


def test_h3_query_experimental_variants_remain_independent_and_complete():
    wrapped = vlm_presets.system_query_additional_vlm
    raw = vlm_presets.system_query_raw_vlm

    assert wrapped["h3_fl2va_experimental_prefix"] == wrapped["h3_fl2va_prefix"]
    assert wrapped["h3_fl2va_experimental_suffix"] == wrapped["h3_fl2va_suffix"]
    assert raw["h3_fl2va_experimental"] == raw["h3_fl2va"]
    assert wrapped["h3_ref2va_prefix"] != wrapped["h3_ref2va_experimental_prefix"]
    assert wrapped["h3_ref2va_suffix"] != wrapped["h3_ref2va_experimental_suffix"]
    assert raw["h3_ref2va"] != raw["h3_ref2va_experimental"]


def test_style_presets_receive_no_qwen_variants():
    assert not any(
        "style" in name.lower() and name.endswith("_qwen")
        for name in vlm_presets.system_instructions_vlm
    )


def test_image_prompt_hardening_preserves_required_contracts():
    for name in HARDENED_IMAGE_PRESETS:
        instruction = _normalize_instruction(vlm_presets.system_instructions_vlm[name])
        original, hardening = _split_hardening_block(instruction)

        assert all(instruction.count(heading) == 1 for heading in HARDENING_HEADINGS)
        assert [hardening.index(heading) for heading in HARDENING_HEADINGS] == sorted(
            hardening.index(heading) for heading in HARDENING_HEADINGS
        )

        for retained_contract in (
            "uncensored",
            "system_query",
            "Subject Count Lock",
            "Reference Adherence is Paramount",
            "image input",
        ):
            assert retained_contract.lower() in original.lower()

        family = name.split("_", 1)[0]
        family_contract = {
            "neutral": "natural language captions",
            "action": "Action, Interaction, and Subject Characteristic Analysis",
            "photo": "Photographic Image Captioning",
            "toon": "Cartoon Art Prompt Refinement",
        }[family]
        assert family_contract.lower() in original.lower()
        if name.endswith("_crude"):
            assert "crude" in original.lower()




def test_image_prompt_hardening_is_literal_without_reward_hacking_anchors():
    for name in HARDENED_IMAGE_PRESETS:
        _, hardening = _split_hardening_block(vlm_presets.system_instructions_vlm[name])
        perspective, remainder = hardening.split(HARDENING_HEADINGS[1], 1)
        quotation, language = remainder.split(HARDENING_HEADINGS[2], 1)

        assert "```" not in hardening
        assert HARDENING_FORBIDDEN.search(hardening) is None
        assert HARDENING_LIST_LINE.search(hardening) is None
        assert re.search(r"\bcamera\b", hardening, re.IGNORECASE) is None

        perspective_lower = perspective.lower()
        assert perspective.rstrip() == APPROVED_PERSPECTIVE_BLOCK
        for removed_phrase in REMOVED_PERSPECTIVE_PHRASES:
            assert removed_phrase not in perspective_lower

        quotation_lower = quotation.lower()
        assert "text" in quotation_lower
        assert "visib" in quotation_lower or "visual" in quotation_lower
        assert "double quotation marks" in quotation_lower
        assert "other" in quotation_lower or "otherwise" in quotation_lower

        language_lower = language.lower()
        for required in (
            "visually",
            "hyphenated words",
            "em dashes",
            "en dashes",
            "purple prose",
            "superfluous",
            "ambiguous",
        ):
            assert required in language_lower
        assert "direct" in language_lower or "literal" in language_lower


STRUCTURED_VIDEO_PRESETS = (
    "video_struct_system_instruction",
    "video_8part_struct_system_instruction",
    "video_timeline_system_instruction",
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_ref2va_general",
)

H3_TIMELINE_PRESETS = (
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_ref2va_general",
    "video_timeline_minimax_h3_reference_alt_system_instruction",
)

H3_CAMERA_CONTINUITY_PRESETS = (
    "video_timeline_minimax_h3_base_system_instruction",
    "video_timeline_minimax_h3_t2va_system_instruction",
    "video_timeline_minimax_h3_ref2va_general",
)

TIMELINE_CHANNEL_BALANCE_PRESETS = (
    "video_timeline_system_instruction",
    "video_timeline_system_instruction_crude",
    "video_timeline_minimax_h3_base_system_instruction",
)




def test_structured_video_presets_are_independent_literal_values():
    source = ast.parse((CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8"))
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "system_instructions_vlm"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }

    for name in STRUCTURED_VIDEO_PRESETS:
        assert name in vlm_presets.system_instructions_vlm
        assert isinstance(vlm_presets.system_instructions_vlm[name], str)
        assert isinstance(literal_nodes[name], ast.Constant)


def test_experimental_video_presets_are_independent_literal_values():
    source = ast.parse(
        (CUSTOM_NODE_ROOT / "vlm_experimental_presets.py").read_text(
            encoding="utf-8"
        )
    )
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "system_instructions_vlm_experimental"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }

    assert set(literal_nodes) == set(
        vlm_experimental_presets.system_instructions_vlm_experimental
    )
    assert all(isinstance(value, ast.Constant) for value in literal_nodes.values())




def test_structured_video_presets_share_audio_and_motion_contracts():
    for name in STRUCTURED_VIDEO_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "nonverbal creature noise" in instruction
        assert "belong under [SOUNDS], never [SPEECH]" in instruction
        lowered = instruction.lower()
        assert (
            "omit the entire [speech] line" in lowered
            or "omit the complete [speech] line" in lowered
        )
        assert (
            "Maintain concrete, descriptive visual-motion language throughout"
            in instruction
            or "Maintain concrete visual-motion language throughout" in instruction
        )


def test_regular_structured_video_preset_retains_non_part_structure():
    instruction = vlm_presets.system_instructions_vlm[
        "video_struct_system_instruction"
    ]

    assert "### Principle 4: Audio-Visual Component Structuring" in instruction
    assert "[VISUAL]: Describe the camera work" in instruction
    assert "[SOUNDS]: Describe the tone" in instruction
    assert "Chronological Channel Alignment" in instruction
    assert "appropriate Part" not in instruction


def test_eight_part_video_preset_keeps_channels_in_each_part():
    instruction = vlm_presets.system_instructions_vlm[
        "video_8part_struct_system_instruction"
    ]

    assert "eight or more distinct one-second segments" in instruction
    assert "Part 1:" in instruction
    assert "Part 8:" in instruction
    assert "inside the appropriate `Part N:`" in instruction


def test_timeline_video_preset_uses_adaptive_contiguous_ranges():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_system_instruction"
    ]

    assert "first output text must be exactly `Timeline:`" in instruction
    assert "[00.00s-01.00s]:" in instruction
    assert "[01.00s-02.50s]:" in instruction
    assert "3-second request using three meaningful sections" in instruction
    assert "5-second request using four differently timed sections" in instruction
    assert "Never reuse their duration, count, boundaries, or content" in instruction
    assert "Use no fixed number of sections" in instruction
    assert "not mandatory one-second intervals" in instruction
    assert "Every range touches the next without a gap or overlap" in instruction
    assert "final range ends at the exact total duration" in instruction
    assert "no `Part N:` headings" in instruction


def test_timeline_presets_create_requested_dialogue_and_balance_channels():
    for name in TIMELINE_CHANNEL_BALANCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "**Requested Dialogue Creation:**" in instruction
        assert "`Add dialogue` or another direct user request" in instruction
        assert "not as a request to detect speech already present" in instruction
        assert "The user does not need to provide wording or timestamps" in instruction
        assert "do not force dialogue into every block" in instruction
        assert "**Foreground Priority and Segment Load:**" in instruction
        assert "one primary foreground event" in instruction
        assert "Never make dialogue or lyrics, loud music, dense effects, and heavy action compete" in instruction
        assert "**Dialogue and Vocal Mixing:**" in instruction
        assert "duck any music" in instruction
        assert "**Pacing and Flow:**" in instruction
        assert "quieter breathing room" in instruction
        assert "only inside a timestamp block containing intelligible spoken dialogue" not in instruction
        assert "synchronized effects intensify with the movement" not in instruction


def test_minimax_h3_timeline_presets_keep_adaptive_standalone_visual_contract():
    for name in H3_TIMELINE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "one or more **image inputs as ordered visual evidence for prompt generation**" in instruction
        assert "Determine the prompt role of each image" in instruction
        assert "must not depend on the downstream video model receiving the images" in instruction
        assert (
            "Fully specify the subjects" in instruction
            or "concrete written specifications" in instruction
            or "current composition, framing, Subject appearance" in instruction
        )
        lowered = instruction.lower()
        assert (
            "use no fixed number of sections" in lowered
            or "do not impose a fixed section count" in lowered
            or "use no fixed number of timestamp sections" in lowered
        )
        assert "Every range touches the next without a gap or overlap" in instruction
        assert "final range ends at the exact total duration" in instruction
        if name == "video_timeline_minimax_h3_reference_alt_system_instruction":
            assert "[START-END]:" in instruction
            assert "every supplied start" in instruction
        else:
            assert "[00.00s-00.00s]:" in instruction
        assert "[SPEECH]:" in instruction
        assert "<d>[Language]" in instruction


def test_minimax_h3_timeline_presets_mark_actual_cuts_with_shot_references():
    for name in H3_CAMERA_CONTINUITY_PRESETS[:2]:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "**Shot Continuity:**" in instruction or "Place [Shot 1]" in instruction
        assert (
            "Introduce sequential `[Shot N]` markers inside [VISUAL] only when the "
            "scene changes or perspective shifts instantly."
            in instruction
            or "Introduce sequential later [Shot N] markers only when the scene "
            "actually cuts or transitions."
            in instruction
        )
        assert "The timestamp range remains the authoritative timing structure." in instruction


def test_minimax_h3_timeline_presets_require_segment_music_contract():
    for name in H3_CAMERA_CONTINUITY_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "[VISUAL], optional [SPEECH], [SOUNDS], and optional [MUSIC]" in instruction
        assert (
            "Omit [MUSIC] from segments with no music specific to them." in instruction
            or "Omit the complete [MUSIC] line when no segment-specific music occurs."
            in instruction
        )
        assert (
            "When music is specific to a timestamp block, write [MUSIC] after [SOUNDS]."
            in instruction
            or "When music is specific to one timestamp block, write [MUSIC] after [SOUNDS]."
            in instruction
        )
        assert (
            "State the type of music for that segment." in instruction
            or "State its audible type." in instruction
        )
        assert (
            "Mention <Subject N> in [MUSIC] only when that actual subject is "
            "playing the music; otherwise state only the music type."
            in instruction
            or "Mention <Subject N> in [MUSIC] only when that actual Subject is "
            "playing the music."
            in instruction
            or "Mention <Subject N> in [MUSIC] only when that actual <Subject N> is "
            "playing the music."
            in instruction
        )
        assert (
            "as the whole-video summary of music specified in the timeline."
            in instruction
            or "background music audible only to the audience" in instruction
            or "background music, theme score, or suspense music" in instruction
        )
        assert (
            "Do not introduce music absent from the timeline." in instruction
            or "Do not introduce music absent from the Timeline." in instruction
        )


def test_stable_timeline_presets_require_zero_padded_two_decimal_seconds():
    for name in TIMELINE_CHANNEL_BALANCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[name]

        assert "first range begins at `00.00s`" in instruction
        assert "[00.00s-00.00s]:" in instruction
        assert "total elapsed seconds" in instruction
        assert "at least two integer digits" in instruction or "Pad single-digit seconds with one leading zero" in instruction
        assert "exactly two decimal digits" in instruction or "for two decimals or" in instruction
        assert ("zero-padded two-decimal" in instruction or "zero-padded total-seconds format" in instruction)
        assert "fewest integer digits needed" not in instruction
        assert "[0s-" not in instruction
        assert "[start-end]:" not in instruction


def test_experimental_timeline_presets_keep_minimal_width_contract():
    for instruction in (
        vlm_experimental_presets.system_instructions_vlm_experimental.values()
    ):
        assert "first range begins at `0.00s`" in instruction or "first range begins at `00.00s`" in instruction
        assert "[0.00s-0.00s]:" in instruction or "[00.00s-00.00s]:" in instruction
        assert ("fewest integer digits needed" in instruction or "zero-padded total-seconds format" in instruction)
        assert "zero-padded two-decimal" not in instruction


def test_minimax_h3_t2va_uses_general_standalone_timeline_contract():
    name = "video_timeline_minimax_h3_t2va_system_instruction"
    instruction = vlm_presets.system_instructions_vlm[name]
    reference_instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_ref2va_general"
    ]
    base_instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_base_system_instruction"
    ]
    prefix = vlm_presets.system_query_additional_vlm["h3_t2va_prefix"]
    suffix = vlm_presets.system_query_additional_vlm["h3_t2va_suffix"]
    raw = vlm_presets.system_query_raw_vlm["h3_t2va"]
    basic_schema = vlm_nodes.UC_VLMSysInstrPresets.define_schema()
    advanced_schema = vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema()

    assert name in basic_schema.inputs[0].options
    assert name in advanced_schema.inputs[0].options
    assert instruction != reference_instruction
    assert instruction != base_instruction

    assert "MiniMax H3 Standalone Text-to-Video" in instruction
    assert "MiniMax H3 receives only the completed text" in instruction
    assert "receives none of these images" in instruction
    assert "VLM-Only Visual Evidence" in instruction
    assert "Standalone Prompt Boundary" in instruction
    assert ("Never emit `<Picture N>`" in instruction or "Never emit <Picture N>" in instruction)
    assert instruction.count("<Picture N>") >= 1
    assert "Complete First-Use Definitions" in instruction
    assert "**Requested Target Visual Style:**" in instruction
    assert "target direction governs the completed video" in instruction
    assert "overrides conflicting source-image rendering style" in instruction
    assert "concrete visual language appropriate to the requested target style" in instruction
    assert "Do not invent production methods or unsupported visual additions" in instruction
    assert "State the governing target visual style" in instruction
    assert "Do not restate the global target style inside [VISUAL]" in instruction
    assert "Concrete lighting or color changes may appear there only when materially relevant" in instruction
    assert "cannot substitute for target-style information in the subject definitions and summary" in instruction
    assert "preserve supported source-image style evidence" in instruction
    assert "requested target-style adherence in every subject definition and in `summary:`" in instruction
    assert "no global target-style restatement inside [VISUAL]" in instruction
    assert "opening [VISUAL] block" not in instruction
    assert "maintain it through every later [VISUAL] block" not in instruction
    assert "visual style, and physical state" not in instruction
    assert "Keep [VISUAL] focused on scene state, action, interaction, camera movement" in instruction
    assert "Write `summary:` immediately after the completed timeline" in instruction
    assert "governing target visual style, medium, era, and subject presentation" in instruction
    assert "subject_definitions:  \r\n<Subject 1>: complete definition" in instruction
    assert "beginning in column one" in instruction
    assert "Do not place a bullet, numbering prefix, indentation" in instruction
    assert "`<Subject N>`" not in instruction
    assert "fixed label, never as a word or name" in instruction
    assert "Never place an apostrophe, possessive marker" in instruction
    assert "Correct possession form: the red sash worn by <Subject 1>." in instruction
    assert "Forbidden possession form: <Subject 1>'s red sash." in instruction
    assert "place surrounding grammar outside it" not in instruction
    assert instruction.count(r"\{user_query\}") == 3
    lowered_instruction = instruction.lower()
    assert "example:" not in lowered_instruction
    assert "e.g." not in lowered_instruction
    assert "i.e." not in lowered_instruction

    fields = (
        "subject_definitions:",
        "summary:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )
    positions = [instruction.index(field) for field in fields]
    assert positions == sorted(positions)
    assert "exactly five top-level fields" in instruction
    assert "Place `Timeline:` immediately beneath `detailed_description:`" in instruction
    assert (
        "Place `summary:` immediately after `subject_definitions`" in instruction
        or "Write `summary:` immediately after the complete" in instruction
    )
    assert "Do not enumerate, sequence, condense, restate, paraphrase" in instruction
    assert "duplicate timeline progression in `summary:`" in instruction

    assert "[00.00s-00.00s]:" in instruction
    assert "The first range begins at `00.00s`" in instruction
    assert "Pad single-digit seconds with one leading zero" in instruction
    for forbidden in (
        "MM:SS.mmm",
        "00:00.000",
        "ComfyUI constructs",
        "Existing Media References",
        "Explicit Picture Timestamp Mapping",
        "retention_analysis:",
        "integrated_multimodal_description:",
        "fully_preserved",
        "<Video N>",
        "<Audio N>",
    ):
        assert forbidden not in instruction

    assert "Use every ordered image supplied with this request" in prefix
    assert "BEGIN VIDEO REQUEST:" in prefix
    assert "END VIDEO REQUEST." in suffix
    assert "none of the supplied VLM images" in suffix
    assert "BEGIN VIDEO REQUEST:" not in raw
    assert "END VIDEO REQUEST." not in raw
    for value in (
        instruction,
        prefix,
        suffix,
        raw,
    ):
        assert value.count("\n") == value.count("\r\n")

    assert "retention_analysis:" in reference_instruction
    assert "integrated_multimodal_description:" in base_instruction

    source = ast.parse(
        (CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8")
    )
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "system_instructions_vlm"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }
    assert isinstance(literal_nodes[name], ast.Constant)


def test_minimax_h3_base_timeline_field_order():
    instruction = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_base_system_instruction"
    ]
    fields = (
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    assert [instruction.index(field) for field in fields] == sorted(
        instruction.index(field) for field in fields
    )
    assert "place `Timeline:` immediately beneath it" in instruction
    assert "Analyze the VLM images in their supplied order" in instruction
    assert "Do not require the user to predeclare these roles" not in instruction


def test_minimax_h3_full_reference_common_contracts():
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))
    options = vlm_nodes.UC_VLMSysInstrPresets.define_schema().inputs[0].options
    advanced_options = (
        vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema().inputs[0].options
    )
    fields = (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )
    task_types = (
        "keyframe completion",
        "reference generation",
        "video editing",
        "video continuation",
        "audio reuse",
        "audio reference",
    )
    visible_markers = (
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
    )
    audio_markers = ("fully_copy", "partially_copy", "reference", "weak_reference")
    camera_terms = (
        "Zoom In / Zoom Out",
        "Push In / Pull Out",
        "Pan Left / Pan Right",
        "Truck Left / Truck Right",
        "Tilt Up / Tilt Down",
        "Pedestal Up / Pedestal Down",
        "Arc Shot",
        "Tracking Shot",
        "Static Shot",
        "Shake Slightly / Shake Strongly",
        "POV",
        "Roll Clockwise / Roll Counterclockwise",
    )
    required = (
        "<Subject N>",
        "<Picture N>",
        "<Video N>",
        "<Audio N>",
        "says in an off-screen voiceover",
        "<scenetrans>",
        "<cutoff>",
        "one to four sentences",
        "one to three English sentences",
    )

    for runtime_key, readable_name in H3_FULL_REFERENCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        assert runtime_key in options
        assert runtime_key in advanced_options
        assert instruction == readable[readable_name]
        assert [instruction.index(field) for field in fields] == sorted(
            instruction.index(field) for field in fields
        )
        for phrase in task_types + visible_markers + audio_markers + camera_terms + required:
            assert phrase in instruction
        assert "no text outside" in instruction or "extra output" in instruction
        assert " + " in instruction
        assert "\r\n" in instruction
        assert not re.search(r"(?<!\r)\n", instruction)
        lowered = instruction.lower()
        assert "example:" not in lowered
        assert "e.g." not in lowered
        assert "i.e." not in lowered


def test_minimax_h3_full_reference_keeps_shot_terms_inside_timeline_context():
    non_timeline_sections = (
        ("#### subject_definitions", "#### summary"),
        ("#### summary", "#### retention_analysis"),
        ("#### retention_analysis", "#### detailed_description and Timeline"),
        ("#### overall_soundscape", "#### non_diegetic_music"),
        (
            "#### non_diegetic_music",
            "#### Instruction Authority and Final Constraints",
        ),
    )

    for runtime_key, _readable_name in H3_FULL_REFERENCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        for start, end in non_timeline_sections:
            section = instruction[instruction.index(start) : instruction.index(end)]
            assert not re.search(r"\bShots?\b", section)

        retention = instruction[
            instruction.index("#### retention_analysis") : instruction.index(
                "#### detailed_description and Timeline"
            )
        ]
        assert "<Subject N>: visible_marker -" in retention
        assert "(appears in applicable Shots)" not in retention
        assert "shot-planning role" not in instruction
        assert "Put [Shot 1] right after [VISUAL]:" in instruction
        assert (
            "Use [Shot N] only when the scene changes or perspective shifts instantly."
            in instruction
            or "Put the next [Shot N] right after [VISUAL]: in any later segment where an actual camera cut or scene change happens."
            in instruction
        )
        assert (
            "Otherwise omit [Shot N] from a new segment and camera movement prompting alone should dictate how the view shifts gradually."
            in instruction
            or "Keep the timestamp range as the timing." in instruction
            or "Keep the timestamp ranges as the timing." in instruction
        )
        assert (
            "advance the shot number sequentially without skipping or repeating." in instruction
            or "in any later segment where an actual camera cut or scene change happens." in instruction
        )

    for runtime_key in (
        "video_timeline_minimax_h3_ref2va_general",
        "video_timeline_minimax_h3_reference_system_instruction_new",
    ):
        inst = vlm_presets.system_instructions_vlm[runtime_key]
        assert (
            "<Picture N>: concrete frame-anchor or timeline-planning role" in inst
            or "| `<Picture N>:` | concrete frame-anchor or timeline-planning role |" in inst
            or "| <Picture N>: | concrete frame-anchor or timeline-planning role |" in inst
        )


def test_minimax_h3_full_reference_assigns_video_motion_transfer_in_retention():
    for runtime_key, _readable_name in H3_FULL_REFERENCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        retention = instruction[
            instruction.index("#### retention_analysis") : instruction.index(
                "#### detailed_description and Timeline"
            )
        ]

        assert "<Video N>" in retention or "<Video 1>" in retention
        assert "camera movement, choreography, timing, pacing, spatial progression" in retention
        assert "attribute_transfer" in retention
        assert "<Subject N>" in retention
        assert "or resulting scene" not in retention
        assert (
            "Use partially_preserved only when some source Video content itself "
            "remains visible."
            in retention
            or "Use partially_preserved only when some source <Video N> content itself "
            "remains visible."
            in retention
        )
        assert "Never create a Video entry" not in retention
        assert "Do not emit timestamp-sample Picture identifiers, Video identifiers" not in instruction

    query_presets = vlm_presets.system_query_additional_vlm
    raw_presets = vlm_presets.system_query_raw_vlm
    for name in ("h3_ref2va", "h3_ref2va_alt", "h3_mixed_ref2va"):
        suffix = query_presets[f"{name}_suffix"]
        raw = raw_presets[name]
        for value in (suffix, raw):
            assert "<Video N>: attribute_transfer" in value
            assert (
                "names the receiving `<Subject N>`" in value
                or "names the receiving <Subject N>" in value
            )
            assert "or resulting scene" not in value
            assert (
                "Use `partially_preserved` only when some source Video content itself remains visible." in value
                or "Use `partially_preserved` only when some source <Video N> content itself remains visible." in value
                or "Use `partially_preserved` only when some source `<Video N>` content itself remains visible." in value
            )
            assert "Exclude choreography, event progression" not in value
            assert "Never emit `<Video N>`" not in value


def test_minimax_h3_full_reference_protected_prefixes_are_unchanged():
    marker = (
        "### Principle 4: MiniMax H3 Reference-Aware Adaptive Timeline and "
        "Audio-Visual Structuring"
    )
    protected = {
        "video_timeline_minimax_h3_ref2va_general": (
            6560,
            "34838a0a0cc0f5e520652f8a2df26d264e11d188364e2b9b06c7fbf5fc0747a0",
        ),
        "video_timeline_minimax_h3_reference_alt_system_instruction": (
            6560,
            "34838a0a0cc0f5e520652f8a2df26d264e11d188364e2b9b06c7fbf5fc0747a0",
        ),
        "video_timeline_minimax_h3_mixed_system_instruction": (
            6769,
            "3b0dfd8966484e3e36ab7b6e43b1916f114e1b51e649979e41ace9c1079adcba",
        ),
        "video_timeline_minimax_h3_reference_system_instruction_new": (
            6560,
            "34838a0a0cc0f5e520652f8a2df26d264e11d188364e2b9b06c7fbf5fc0747a0",
        ),
        "video_timeline_minimax_h3_reference_alt_system_instruction_new": (
            6560,
            "34838a0a0cc0f5e520652f8a2df26d264e11d188364e2b9b06c7fbf5fc0747a0",
        ),
        "video_timeline_minimax_h3_mixed_system_instruction_new": (
            6816,
            "d95aed84d9d4cb13af343dddfaa4a24f6786e36ef8e5e1d389b4ffd39b2bdf2a",
        ),
    }

    for runtime_key, (expected_length, expected_hash) in protected.items():
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        prefix = instruction[: instruction.index(marker)]
        assert len(prefix) == expected_length
        assert hashlib.sha256(prefix.encode()).hexdigest() == expected_hash


def test_minimax_h3_full_reference_variant_contracts():
    required_by_preset = {
        "video_timeline_minimax_h3_ref2va_general": (
            "does not automatically represent the first or last target-video frame",
            "Every action, motion, giver, receiver, target, and touched body part must explicitly name the literal <Subject N> tag",
            "Never use pronouns ('he', 'she', 'him', 'her', 'his', 'their', 'They', 'Them', 'Their', 'He', 'She')",
        ),
        "video_timeline_minimax_h3_reference_alt_system_instruction": (
            "Preserve the exact segment count, every supplied start",
            "visible source identity remains analysis-only",
            (
                "Emit an actually supplied Video identifier only in retention_analysis",
                "Emit an actually supplied <Video N> identifier only in retention_analysis",
            ),
            "continuously present from the first applicable frame through the last",
        ),
        "video_timeline_minimax_h3_mixed_system_instruction": (
            "regular user request to establish the mixed-media partition",
            "chronological samples of one <Video 1> sequence",
            (
                "Picture reference numbered from <Picture 1> within that later subset",
                "<Picture N> reference numbered from <Picture 1> within that later subset",
            ),
            "Never write <Video N> inside a timestamp block",
        ),
        "video_timeline_minimax_h3_reference_system_instruction_new": (
            "exactly that number of leading Pictures",
            (
                "Treat every later Picture as a reference image",
                "Treat every later <Picture N> as a reference image",
            ),
            (
                "Do not create a Video namespace for leading timeline Pictures",
                "Do not create a <Video N> namespace for leading timeline Pictures",
            ),
            (
                "Never assume that another Picture requests replacement",
                "Never assume that another <Picture N> requests replacement",
            ),
        ),
        "video_timeline_minimax_h3_reference_alt_system_instruction_new": (
            (
                "ordered Shot starts",
                "ordered timeline Shots",
            ),
            (
                "Treat every later Picture as a reference image",
                "Treat every later <Picture N> as a reference image",
            ),
            "no timeline-source identifiers in Timeline",
            "Never assume replacement merely because later Pictures exist",
        ),
        "video_timeline_minimax_h3_mixed_system_instruction_new": (
            (
                "one continuous Picture namespace",
                "one continuous <Picture N> namespace",
            ),
            (
                "Never create a Video namespace from Pictures",
                "Never create a <Video N> namespace from Pictures",
            ),
            "leading timeline Pictures",
            "Never activate video editing or video continuation from timeline Pictures",
        ),
    }

    for runtime_key, required in required_by_preset.items():
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        for phrase in required:
            if isinstance(phrase, tuple):
                assert any(p in instruction for p in phrase)
            else:
                assert phrase in instruction


def test_minimax_h3_full_reference_assembled_context_contracts():
    system_query = "SYSTEM QUERY SENTINEL: preserve established communication rules."
    user_query = (
        "USER QUERY SENTINEL: duration 12.309 seconds; preserve starts 00.000s, "
        "03.066s, 06.131s, 09.197s, and 12.262s; use supplied media roles."
    )

    for runtime_key, _readable_name in H3_FULL_REFERENCE_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        assembled = vlm_nodes.UC_VLMSysInstrAdvPresets.execute(
            runtime_key,
            False,
            system_query,
            user_query,
        ).args[0]
        assert assembled.startswith(instruction)
        assert assembled.count(system_query) == 1
        assert assembled.count(user_query) == 1
        assert assembled.index(system_query) < assembled.index(user_query)
        for timestamp in ("00.000s", "03.066s", "06.131s", "09.197s", "12.262s"):
            assert timestamp in assembled
        assert "subject_definitions:" in instruction
        assert "retention_analysis:" in instruction
        assert "stable" in instruction.lower()
        assert r"\{user_query\}" in instruction


def test_minimax_h3_reference_debug_preserves_control_and_fixes_label_contract():
    runtime_key = "video_timeline_minimax_h3_reference_system_instruction_debug"
    readable_name = "VIDEO_TIMELINE_MINIMAX_H3_REFERENCE_SYSTEM_INSTRUCTION_DEBUG"
    control = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_ref2va_general"
    ]
    debug = vlm_presets.system_instructions_vlm[runtime_key]
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))
    basic_options = vlm_nodes.UC_VLMSysInstrPresets.define_schema().inputs[0].options
    advanced_options = (
        vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema().inputs[0].options
    )

    assert debug == readable[readable_name]
    assert runtime_key in basic_options
    assert runtime_key in advanced_options
    assert debug != control

    control_subjects = control[
        control.index("#### subject_definitions") : control.index("#### summary")
    ]
    debug_subjects = debug[
        debug.index("#### subject_definitions") : debug.index("#### summary")
    ]
    debug_timeline = debug[
        debug.index("#### detailed_description and Timeline") : debug.index(
            "#### Shots and Camera"
        )
    ]

    assert ("| `<Subject N>:` |" in control_subjects or "| <Subject N>: |" in control_subjects)
    assert ("every mentioned Subject action must include" in control_subjects or "every mentioned <Subject N> action must include" in control_subjects)
    for tag in ("Subject", "Picture", "Video", "Audio"):
        assert f"| `<{tag} N> is ...` |" in debug_subjects
        assert (f"| `<{tag} N>:` |" not in debug_subjects and f"| <{tag} N>: |" not in debug_subjects)
    assert ("every mentioned Subject action must include" not in debug_subjects and "every mentioned <Subject N> action must include" not in debug_subjects)
    assert "In generated output, emit it as plain text" in debug_subjects
    assert "Separate the tag from following prose with whitespace" in debug_subjects
    assert "literal alias at first introduction" in debug_timeline
    assert "Otherwise use a concise ordinary name, role, or pronoun" in debug_timeline
    assert "Do not repeat the alias at every action mention" in debug_timeline

    system_query = "SYSTEM QUERY DEBUG SENTINEL"
    user_query = "USER QUERY DEBUG SENTINEL: create a 6.00s video."
    assembled = vlm_nodes.UC_VLMSysInstrAdvPresets.execute(
        runtime_key,
        False,
        system_query,
        user_query,
    ).args[0]
    assert assembled.startswith(debug)
    assert assembled.count(system_query) == 1
    assert assembled.count(user_query) == 1
    assert assembled.index(system_query) < assembled.index(user_query)


def test_minimax_h3_reference_newdebug_preserves_literal_template_contract():
    runtime_key = "video_timeline_minimax_h3_reference_system_instruction_newdebug"
    readable_name = "VIDEO_TIMELINE_MINIMAX_H3_REFERENCE_SYSTEM_INSTRUCTION_NEWDEBUG"
    debug = vlm_presets.system_instructions_vlm[
        "video_timeline_minimax_h3_reference_system_instruction_debug"
    ]
    newdebug = vlm_presets.system_instructions_vlm[runtime_key]
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))
    basic_options = vlm_nodes.UC_VLMSysInstrPresets.define_schema().inputs[0].options
    advanced_options = (
        vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema().inputs[0].options
    )

    assert newdebug == readable[readable_name]
    assert runtime_key in basic_options
    assert runtime_key in advanced_options
    assert newdebug != debug
    for required in (
        "everything within curly brace `{}` contains elements to replace",
        "Completely leave out lyrics for theme music.",
        "Do NOT invent sounds.",
        "[SOUNDS] is not for music or instruments.",
        "[VISUAL]: [Shot 18]",
        "<Audio 1> serves as the full {description of the music}",
    ):
        assert required in newdebug

    system_query = "SYSTEM QUERY NEWDEBUG SENTINEL"
    user_query = "USER QUERY NEWDEBUG SENTINEL: create a 6.00s video."
    assembled = vlm_nodes.UC_VLMSysInstrAdvPresets.execute(
        runtime_key,
        False,
        system_query,
        user_query,
    ).args[0]
    assert assembled.startswith(newdebug)
    assert assembled.count(system_query) == 1
    assert assembled.count(user_query) == 1
    assert assembled.index(system_query) < assembled.index(user_query)


def test_experimental_h3_reference_keeps_regression_contract():
    instruction = vlm_experimental_presets.system_instructions_vlm_experimental[
        "video_timeline_minimax_h3_reference_system_instruction"
    ]
    fields = (
        "subject_definitions:",
        "retention_analysis:",
        "detailed_description:",
        "summary:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    assert [instruction.index(field) for field in fields] == sorted(
        instruction.index(field) for field in fields
    )
    assert "Place `summary:` immediately after the complete timeline" in instruction
    assert "as an internal picture-to-time map" in instruction
    assert "Do not write `<Picture N>` anywhere inside" in instruction
    assert "never cite `<Picture N>` in a timestamp block" in instruction
    assert "cite ComfyUI's existing `<Picture N>` identifiers only as subject provenance" in instruction
    assert "**Atomic Subject Labels:**" in instruction
    assert "Never append possessive markers" in instruction
    assert "without modifying the alias" in instruction


def test_minimax_h3_timeline_presets_avoid_example_led_content_anchors():
    forbidden = re.compile(
        r"\be\.g\.|\bi\.e\.|\bexamples?\b|\bsuch as\b|\bincluding\b|"
        r"\betc\b|\be621\b|\bdanbooru\b",
        re.IGNORECASE,
    )

    for name in H3_TIMELINE_PRESETS:
        assert forbidden.search(vlm_presets.system_instructions_vlm[name]) is None

    assert not any(
        re.search(r"\be621\b|\bdanbooru\b", instruction, re.IGNORECASE)
        for instruction in vlm_presets.system_instructions_vlm.values()
    )


def test_minimax_h3_dedicated_presets_schema_and_authorities():
    readable = runpy.run_path(str(CUSTOM_NODE_ROOT / "vlm_presets_vars.py"))
    basic_options = vlm_nodes.UC_VLMSysInstrPresets.define_schema().inputs[0].options
    advanced_options = vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema().inputs[0].options

    for runtime_key, readable_name in H3_DEDICATED_SCENARIO_PRESETS:
        assert runtime_key in basic_options, f"{runtime_key} missing from basic options"
        assert runtime_key in advanced_options, f"{runtime_key} missing from advanced options"
        assert runtime_key in vlm_presets.system_instructions_vlm, f"{runtime_key} missing from runtime instructions"
        assert vlm_presets.system_instructions_vlm[runtime_key] == readable[readable_name]
        assert len(vlm_presets.system_instructions_vlm[runtime_key]) > 0


def test_minimax_h3_dedicated_presets_standalone_ast_literals():
    source = ast.parse((CUSTOM_NODE_ROOT / "vlm_presets.py").read_text(encoding="utf-8"))
    dict_node = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "system_instructions_vlm" for target in node.targets)
    )
    keys_to_values = {
        key.value: val
        for key, val in zip(dict_node.keys, dict_node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    for runtime_key, _ in H3_DEDICATED_SCENARIO_PRESETS:
        assert runtime_key in keys_to_values
        val_node = keys_to_values[runtime_key]
        assert isinstance(val_node, ast.Constant)
        assert isinstance(val_node.value, str)


def test_minimax_h3_dedicated_presets_trailing_whitespace_integrity():
    for runtime_key, _ in H3_DEDICATED_SCENARIO_PRESETS:
        if runtime_key == "video_timeline_minimax_h3_ref2va_attribute_transfer":
            continue
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        if "t2va" in runtime_key and "any2va" not in runtime_key:
            fields = (
                "integrated_multimodal_description:  \r\n",
                "overall_soundscape:  \r\n",
                "non_diegetic_music:  \r\n",
            )
        else:
            fields = (
                "subject_definitions:  \r\n",
                "summary:  \r\n",
                "detailed_description:  \r\n",
                "overall_soundscape:  \r\n",
                "non_diegetic_music:  \r\n",
            )
        for field in fields:
            assert field in instruction, f"{field!r} missing two trailing spaces in {runtime_key}"


def test_minimax_h3_dedicated_presets_tag_binding_and_mechanics():
    forbidden_pronouns = re.compile(r"Never use pronouns \('he', 'she', 'him', 'her', 'his', 'their', 'They', 'Them', 'Their', 'He', 'She'\)")
    tag_mention = (
        "In every Timeline segment, every mentioned <Subject N> action must include that Subject's literal <Subject N> alias at the action mention."
    )
    tag_strict = "Every action, motion, giver, receiver, target, and touched body part must explicitly name the literal <Subject N> tag (write '<Subject 1> takes the ball from <Subject 2>', not 'from her')."
    banned_umbrella = "Never use umbrella placeholders like 'physical interaction', 'interacting with', or 'intimate scene'."
    contact_mechanics = "For bodily contact or sexual acts, explicitly describe the exact physical poses, surfaces touching (mouth on penis, hands on body), motion trajectories (kneeling, stroking, thrusting), and any fluids present (semen, saliva, sweat)."

    for runtime_key, _ in H3_DEDICATED_SCENARIO_PRESETS:
        if runtime_key == "video_timeline_minimax_h3_ref2va_attribute_transfer":
            continue
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        assert forbidden_pronouns.search(instruction) is not None, f"Pronoun ban missing in {runtime_key}"
        assert "write '<Subject 1> and <Subject 2>', never 'They' or 'Both'" in instruction
        assert "never 'their hands', 'their bodies', or 'their lips'" in instruction
        assert tag_mention in instruction, f"Tag mention rule missing in {runtime_key}"
        assert tag_strict in instruction, f"Explicit tag naming rule missing in {runtime_key}"
        assert banned_umbrella in instruction, f"Umbrella phrase ban missing in {runtime_key}"
        assert contact_mechanics in instruction, f"Contact mechanics missing in {runtime_key}"


def test_minimax_h3_fl2va_contract():
    instruction = vlm_presets.system_instructions_vlm["video_timeline_minimax_h3_fl2va_system_instruction"]
    assert "MiniMax H3 First/Last Frame Adaptive Timeline" in instruction
    assert "<Picture 1> is always the fixed first frame anchor at 00.00s" in instruction
    assert "00.00s" in instruction
    assert "<Picture 2> is the fixed final frame anchor" in instruction
    assert "Describe the shot beginning directly from <Picture 1>" in instruction
    assert "ensure the final segment describes motion converging on <Picture 2>" in instruction
    assert "subject_definitions:" in instruction
    assert "retention_analysis:" in instruction


def test_minimax_h3_scene_image_any2va_contract():
    for collection in (
        vlm_presets.system_instructions_vlm,
        vlm_motion_presets.motion_system_instructions_vlm,
    ):
        instruction = collection["video_timeline_minimax_h3_scene_image_any2va_system_instruction"]
        assert "MiniMax H3 Scene Image to Video Adaptive Timeline" in instruction
        assert "The single supplied image acts as the visual seed" in instruction
        assert "extrapolate a continuous, escalating progression of motion" in instruction
        assert "do not emit `<Picture 1>` or any media identifier inside the summary or timeline" in instruction
        assert "do not create a <Video N> namespace from the image" in instruction
        assert "Conditional User-Instructed Reference Rule:" in instruction
        assert "Only when the user instruction explicitly specifies" in instruction
        assert "referenced from <Picture N>" in instruction
        assert "#### retention_analysis (Conditional on User Instruction)" in instruction
        assert "Segment Visual and Motion Density Requirement:" in instruction
        assert "Detailed 3-5 sentence" in instruction
        assert "short single sentence" not in instruction
        assert "when no `<Video N>` reference is explicitly specified by the user instruction" in instruction
        assert "never write `of the video`" in instruction
        assert "First and foremost, when an <Audio N> item supplies the soundtrack, background music, or musical beat for the video" in instruction
        assert "`non_diegetic_music:` must begin with `<Audio N>`" in instruction
        assert "Never use the word 'both' or 'Both' in any context inside [VISUAL]" in instruction
        assert "In-Place Character Replacement Rule (No Replacement Instructions):" in instruction
        assert "never write a prompt that instructs replacement, swapping, editing, or transfer" in instruction
        assert "completely replace the character in the full description across all instances" in instruction
        assert "subject_definitions:" in instruction
        assert "summary:" in instruction
        assert "retention_analysis:" in instruction
        assert "detailed_description:" in instruction
        assert re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", instruction) is None


def test_minimax_h3_scene_image_t2va_contract():
    for collection in (
        vlm_presets.system_instructions_vlm,
        vlm_motion_presets.motion_system_instructions_vlm,
    ):
        instruction = collection["video_timeline_minimax_h3_scene_image_t2va_system_instruction"]
        assert "MiniMax H3 Scene Image to Video Adaptive Timeline" in instruction
        assert "The single supplied image acts as the visual seed" in instruction
        assert "extrapolate a continuous, escalating progression of motion" in instruction
        assert "Do not emit `<Picture 1>` or any media identifier inside the timeline" in instruction
        assert (
            "Do not create a Video namespace from the image" in instruction
            or "Do not create a <Video N> namespace from the image" in instruction
        )
        assert "In-Place Character Replacement Rule (No Replacement Instructions):" in instruction
        assert "never write a prompt that instructs replacement, swapping, editing, or transfer" in instruction
        assert "completely replace the character in the full description across all instances" in instruction
        assert "integrated_multimodal_description:" in instruction
        assert "subject_definitions:" not in instruction
        assert "summary:" not in instruction
        assert "retention_analysis:" not in instruction
        assert "Segment Visual and Motion Density Requirement:" in instruction
        assert "Detailed 3-5 sentence" in instruction
        assert "short single sentence" not in instruction
        assert "Never use the word 'both' or 'Both' in any context inside [VISUAL]" in instruction
        assert re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", instruction) is None


def test_minimax_h3_storyboard_any2va_contract():
    for collection in (
        vlm_presets.system_instructions_vlm,
        vlm_motion_presets.motion_system_instructions_vlm,
    ):
        instruction = collection["video_timeline_minimax_h3_storyboard_any2va_system_instruction"]
        assert "MiniMax H3 Storyboard to Video" in instruction
        assert "comic strip" in instruction
        assert "a stitched-together row of images outlining the timeline progression" in instruction
        assert "multiple separate images showing the progression of the timeline" in instruction
        assert "a video showing a detailed progression" in instruction
        assert "The storyboard is seen only by the VLM and is never seen by the downstream video model" in instruction
        assert "every single motion, action, interaction, dialogue, song, sound, and music" in instruction
        assert "The target video translates the sequential storyboard panels" not in instruction
        assert "Strict Graphic Strip Rule:" in instruction
        assert "comic book or graphic conventions, or the storyboard itself, be described or included in the output prompt" in instruction
        assert "Strip panel borders, speech bubbles, sound effect lettering, and graphic conventions." in instruction
        assert "Dialogue text visibly printed in speech bubbles must be transcribed into `[SPEECH]` rows" in instruction
        assert "Conditional User-Instructed Reference Rule:" in instruction
        assert "Only when the user instruction explicitly specifies" in instruction
        assert "referenced from <Picture N>" in instruction
        assert "#### retention_analysis (Conditional on User Instruction)" in instruction
        assert "Segment Visual and Motion Density Requirement:" in instruction
        assert "Detailed 3-5 sentence" in instruction
        assert "short single sentence" not in instruction
        assert "when no `<Video N>` reference is explicitly specified by the user instruction" in instruction
        assert "never write `of the video`" in instruction
        assert "First and foremost, when an <Audio N> item supplies the soundtrack, background music, or musical beat for the video" in instruction
        assert "`non_diegetic_music:` must begin with `<Audio N>`" in instruction
        assert "Never use the word 'both' or 'Both' in any context inside [VISUAL]" in instruction
        assert "In-Place Character Replacement Rule (No Replacement Instructions):" in instruction
        assert "never write a prompt that instructs replacement, swapping, editing, or transfer" in instruction
        assert "completely replace the character in the full description across all instances" in instruction
        assert "subject_definitions:" in instruction
        assert "summary:" in instruction
        assert "retention_analysis:" in instruction
        assert "detailed_description:" in instruction
        assert re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", instruction) is None


def test_minimax_h3_storyboard_t2va_contract():
    for collection in (
        vlm_presets.system_instructions_vlm,
        vlm_motion_presets.motion_system_instructions_vlm,
    ):
        instruction = collection["video_timeline_minimax_h3_storyboard_t2va_system_instruction"]
        assert "MiniMax H3 Storyboard to Video" in instruction
        assert "comic strip" in instruction
        assert "a stitched-together row of images outlining the timeline progression" in instruction
        assert "multiple separate images showing the progression of the timeline" in instruction
        assert "a video showing a detailed progression" in instruction
        assert "The storyboard is seen only by the VLM and is never seen by the downstream video model" in instruction
        assert "every single motion, action, interaction, dialogue, song, sound, and music" in instruction
        assert "Strict Graphic Strip Rule:" in instruction
        assert "comic book or graphic conventions, or the storyboard itself, be described or included in the output prompt" in instruction
        assert "Strip panel borders, speech bubbles, sound effect lettering, and graphic conventions." in instruction
        assert "Dialogue text visibly printed in speech bubbles must be transcribed into `[SPEECH]` rows" in instruction
        assert "In-Place Character Replacement Rule (No Replacement Instructions):" in instruction
        assert "never write a prompt that instructs replacement, swapping, editing, or transfer" in instruction
        assert "completely replace the character in the full description across all instances" in instruction
        assert "integrated_multimodal_description:" in instruction
        assert "subject_definitions:" not in instruction
        assert "summary:" not in instruction
        assert "retention_analysis:" not in instruction
        assert "Segment Visual and Motion Density Requirement:" in instruction
        assert "Detailed 3-5 sentence" in instruction
        assert "short single sentence" not in instruction
        assert "Never use the word 'both' or 'Both' in any context inside [VISUAL]" in instruction
        assert re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}", instruction) is None


def test_minimax_h3_ref2va_attribute_transfer_variants_contracts():
    audio_copy = vlm_presets.system_instructions_vlm["video_timeline_minimax_h3_ref2va_attr_transfer_audio_copy"]
    assert "<Audio 1>: fully_copy - source audio track is copied entirely" in audio_copy
    assert "[video editing + reference generation + audio reuse]" in audio_copy
    assert "The target video is an edited version of <Video 1>" in audio_copy

    audio_timbre = vlm_presets.system_instructions_vlm["video_timeline_minimax_h3_ref2va_attr_transfer_audio_timbre"]
    assert "<Audio 1>: reference - vocal timbre and delivery guide speech of <Subject 1> (S1) without copying audio signal" in audio_timbre
    assert "[video editing + reference generation + audio reference]" in audio_timbre
    assert "referencing vocal timbre from <Audio 1>" in audio_timbre

    no_audio = vlm_presets.system_instructions_vlm["video_timeline_minimax_h3_ref2va_attr_transfer_no_audio"]
    assert "<Audio 1>:" not in no_audio
    assert ("Do not create `<Audio N>` labels" in no_audio or "Do not create <Audio N> labels" in no_audio)
    assert "[video editing + reference generation]" in no_audio
    assert "non_diegetic_music:  \r\nN/A" in no_audio


def test_minimax_h3_dedicated_presets_assembled_context():
    system_query = "SYSTEM QUERY DEDICATED SENTINEL"
    user_query = "USER QUERY DEDICATED SENTINEL: create a 6.00s video."

    for runtime_key, _ in H3_DEDICATED_SCENARIO_PRESETS:
        instruction = vlm_presets.system_instructions_vlm[runtime_key]
        assembled = vlm_nodes.UC_VLMSysInstrAdvPresets.execute(
            runtime_key,
            False,
            system_query,
            user_query,
        ).args[0]
        assert assembled.startswith(instruction)
        assert assembled.count(system_query) == 1
        assert assembled.count(user_query) == 1
        assert assembled.index(system_query) < assembled.index(user_query)


def test_h3_dedicated_query_presets_contracts():
    add_presets = vlm_nodes.UC_VLMSysQueryAddPresets.get_presets()
    raw_schema_options = (
        vlm_nodes.UC_VLMSysQueryRawPresets.define_schema().inputs[0].options
    )
    dedicated_queries = (
        "h3_ref2va_general",
        "h3_ref2va_attr_transfer_audio_timbre",
        "h3_ref2va_attr_transfer_audio_copy",
        "h3_ref2va_attr_transfer_no_audio",
        "h3_scene_image_any2va",
        "h3_storyboard_any2va",
        "h3_scene_image_t2va",
        "h3_storyboard_t2va",
    )

    for name in dedicated_queries:
        assert name in add_presets
        assert name in raw_schema_options
        assert f"{name}_prefix" in vlm_presets.system_query_additional_vlm
        assert f"{name}_suffix" in vlm_presets.system_query_additional_vlm
        assert name in vlm_presets.system_query_raw_vlm

    sentinel_text = "SENTINEL ACTION: subjects walk in park."
    for name in dedicated_queries:
        wrapped = vlm_nodes.UC_VLMSysQueryAddPresets.execute(name, sentinel_text).args[0]
        assert wrapped.count(sentinel_text) == 1
        assert "BEGIN VIDEO REQUEST:" in wrapped
        assert "END VIDEO REQUEST." in wrapped
        assert wrapped.index("BEGIN VIDEO REQUEST:") < wrapped.index(sentinel_text)
        assert wrapped.index(sentinel_text) < wrapped.index("END VIDEO REQUEST.")

    timbre_pref = vlm_presets.system_query_additional_vlm["h3_ref2va_attr_transfer_audio_timbre_prefix"]
    timbre_suff = vlm_presets.system_query_additional_vlm["h3_ref2va_attr_transfer_audio_timbre_suffix"]
    assert "vocal-timbre and vocal-delivery reference for `<Subject 1> (S1)`" in timbre_pref
    assert "cannot replace, omit, or override the mandatory attribute transfer roles" in timbre_pref
    assert "[video editing + reference generation + audio reference]" in timbre_suff
    assert "<Audio 1>: reference - vocal timbre and delivery guide speech" in timbre_suff
    assert "zero pronouns" in timbre_suff

    copy_pref = vlm_presets.system_query_additional_vlm["h3_ref2va_attr_transfer_audio_copy_prefix"]
    copy_suff = vlm_presets.system_query_additional_vlm["h3_ref2va_attr_transfer_audio_copy_suffix"]
    assert "full synchronized soundtrack copied from `<Video 1>`" in copy_pref
    assert "[video editing + reference generation + audio reuse]" in copy_suff
    assert "<Audio 1>: fully_copy" in copy_suff

    no_audio_pref = vlm_presets.system_query_additional_vlm["h3_ref2va_attr_transfer_no_audio_prefix"]
    no_audio_suff = vlm_presets.system_query_additional_vlm["h3_ref2va_attr_transfer_no_audio_suffix"]
    assert "No audio track is copied or referenced" in no_audio_pref
    assert "[video editing + reference generation]" in no_audio_suff
    assert "non_diegetic_music: N/A" in no_audio_suff

    scene_any_pref = vlm_presets.system_query_additional_vlm["h3_scene_image_any2va_prefix"]
    scene_any_suff = vlm_presets.system_query_additional_vlm["h3_scene_image_any2va_suffix"]
    assert "strictly as an opening scene seed" in scene_any_pref
    assert "if and only if the user request explicitly specifies" in scene_any_pref
    assert "never write a prompt that instructs replacement" in scene_any_pref
    assert "completely replace the character in the full description at all instances" in scene_any_pref
    assert "[reference generation]" in scene_any_suff
    assert "Omit by default; include `retention_analysis:`" in scene_any_suff
    assert "only when the user request explicitly specifies downstream reference media" in scene_any_suff
    assert "completely replace the character in-place across all instances" in scene_any_suff

    scene_t2va_pref = vlm_presets.system_query_additional_vlm["h3_scene_image_t2va_prefix"]
    scene_t2va_suff = vlm_presets.system_query_additional_vlm["h3_scene_image_t2va_suffix"]
    assert "strictly as an opening scene seed" in scene_t2va_pref
    assert "never write a prompt that instructs replacement" in scene_t2va_pref
    assert "completely replace the character in the full description at all instances" in scene_t2va_pref
    assert "integrated_multimodal_description:" in scene_t2va_suff
    assert "overall_soundscape:" in scene_t2va_suff
    assert "completely replace the character in-place across all instances" in scene_t2va_suff

    story_any_pref = vlm_presets.system_query_additional_vlm["h3_storyboard_any2va_prefix"]
    story_any_suff = vlm_presets.system_query_additional_vlm["h3_storyboard_any2va_suffix"]
    assert "Strict Graphic Strip Rule" in story_any_pref
    assert "a stitched-together row of images outlining the timeline progression" in story_any_pref
    assert "multiple separate images showing the progression of the timeline" in story_any_pref
    assert "a video showing a detailed progression" in story_any_pref
    assert "if and only if the user request explicitly specifies" in story_any_pref
    assert "never write a prompt that instructs replacement" in story_any_pref
    assert "completely replace the character in the full description at all instances" in story_any_pref
    assert "[reference generation]" in story_any_suff
    assert "Omit by default; include `retention_analysis:`" in story_any_suff
    assert "only when the user request explicitly specifies downstream reference media" in story_any_suff
    assert "completely replace the character in-place across all instances" in story_any_suff

    story_t2va_pref = vlm_presets.system_query_additional_vlm["h3_storyboard_t2va_prefix"]
    story_t2va_suff = vlm_presets.system_query_additional_vlm["h3_storyboard_t2va_suffix"]
    assert "Strict Graphic Strip Rule" in story_t2va_pref
    assert "a stitched-together row of images outlining the timeline progression" in story_t2va_pref
    assert "multiple separate images showing the progression of the timeline" in story_t2va_pref
    assert "a video showing a detailed progression" in story_t2va_pref
    assert "never write a prompt that instructs replacement" in story_t2va_pref
    assert "completely replace the character in the full description at all instances" in story_t2va_pref
    assert "integrated_multimodal_description:" in story_t2va_suff
    assert "overall_soundscape:" in story_t2va_suff
    assert "completely replace the character in-place across all instances" in story_t2va_suff


def test_minimax_h3_attr_transfer_multi_subject_contracts():
    keys = (
        "video_timeline_minimax_h3_ref2va_attr_transfer_audio_timbre",
        "video_timeline_minimax_h3_ref2va_attr_transfer_audio_copy",
        "video_timeline_minimax_h3_ref2va_attr_transfer_no_audio",
    )
    for key in keys:
        for collection in (
            vlm_presets.system_instructions_vlm,
            vlm_motion_presets.motion_system_instructions_vlm,
        ):
            instruction = collection[key]
            assert "Original subjects from `<Video 1>` designated for replacement must NEVER be defined under `subject_definitions:`" in instruction
            assert "Every other non-replaced subject present in `<Video 1>` must be defined as its own `<Subject {N}>` under `subject_definitions:` using its visible traits from `<Video 1>`." in instruction
            assert "When other subjects in `<Video 1>` have dialogue, attribute their `[SPEECH]` lines to their own `<Subject {N}> (S{N})` tag and never set their dialogue to be spoken by the reference-replaced subject." in instruction
            assert "Every other non-replaced subject present in `<Video 1>` must be defined as its own <Subject N> in `subject_definitions:` using its visible traits from `<Video 1>`." in instruction
            assert "Attribute every [SPEECH] line strictly to the <Subject N> (Sx) who speaks it; if other subjects in `<Video 1>` have dialogue, assign their dialogue to their own <Subject N> (Sx) tag and never set it to be spoken by the reference-replaced subject." in instruction
            assert "`<Subject {N}>` is a {visual description of the subject that will be used in the video}, referenced from `<Picture {N}>`." in instruction
            assert "where on-screen subject(s) are replaced by defined `<Subject {N}>` entities from `<Picture {N}>`" in instruction
            assert "with source subjects replaced by defined `<Subject {N}>` entities" in instruction
            assert (
                "Strict Visual Appearance Continuity: In every single shot and timestamp block, all visual details (face, body, materials, textures, geometry, colors) for each defined `<Subject N>` must strictly and exclusively depict that `<Subject N>` using the visual traits established by matching `<Picture N>`"
                in instruction
                or "Strict Visual Appearance Continuity: In every single shot and timestamp block, all visual details (face, body, materials, textures, geometry, colors) for each defined <Subject N> must strictly and exclusively depict that <Subject N> using the visual traits established by matching <Picture N>"
                in instruction
            )
            assert (
                "Original subjects from `<Video 1>` designated for replacement are NEVER defined, named, described, or assigned `<Subject N>` aliases under `subject_definitions:`"
                in instruction
                or "Original subjects from `<Video 1>` designated for replacement are NEVER defined, named, described, or assigned <Subject N> aliases under `subject_definitions:`"
                in instruction
            )

    for collection in (
        vlm_presets.system_instructions_vlm,
        vlm_motion_presets.motion_system_instructions_vlm,
    ):
        attr_transfer = collection["video_timeline_minimax_h3_ref2va_attribute_transfer"]
        assert "Every other non-replaced subject present in `<Video 1>` must be defined as its own `<Subject {N}>` under `subject_definitions:` using its visible traits from `<Video 1>`." in attr_transfer
        assert "Attribute every [SPEECH] line strictly to the <Subject N> (Sx) who speaks it; if other subjects in <Video 1> have dialogue, assign their dialogue to their own <Subject N> (Sx) tag and never set it to be spoken by the reference-replaced subject." in attr_transfer

        ref2va_general = collection["video_timeline_minimax_h3_ref2va_general"]
        assert "define every other non-replaced subject present in <Video N> as its own <Subject N> using its visible traits from <Video N>." in ref2va_general
        assert "Attribute every [SPEECH] line strictly to the <Subject N> (Sx) who speaks it; if other subjects in the video have dialogue, assign their dialogue to their own <Subject N> (Sx) tag and never set it to be spoken by the reference-replaced subject." in ref2va_general

    for query_key in (
        "h3_ref2va_attr_transfer_audio_timbre",
        "h3_ref2va_attr_transfer_audio_copy",
        "h3_ref2va_attr_transfer_no_audio",
    ):
        pref = vlm_presets.system_query_additional_vlm[f"{query_key}_prefix"]
        suff = vlm_presets.system_query_additional_vlm[f"{query_key}_suffix"]
        raw = vlm_presets.system_query_raw_vlm[query_key]
        assert "Any other non-replaced subjects in `<Video 1>` must be defined as separate `<Subject N>` entities" in pref
        assert "define any other non-replaced subjects in `<Video 1>` as their own `<Subject N>` entities" in suff
        assert "never set" in suff and "spoken by `<Subject 1>`" in suff
        assert "define any other non-replaced subjects in `<Video 1>` as their own `<Subject N>` entities" in raw
        assert "never set" in raw and "spoken by `<Subject 1>`" in raw



def test_moss_image_json_preset_contract():
    basic_schema = vlm_nodes.UC_VLMSysInstrPresets.define_schema()
    adv_schema = vlm_nodes.UC_VLMSysInstrAdvPresets.define_schema()
    assert [out.display_name for out in basic_schema.outputs] == [
        "system_instruction",
        "response_schema",
    ]
    assert [out.display_name for out in adv_schema.outputs] == [
        "system_instruction",
        "response_schema",
    ]
    options = basic_schema.inputs[0].options
    for key in (
        "moss_image_json_instruction",
        "moss_image_json_instruction_short",
        "moss_image_json_instruction_color",
        "moss_image_json_instruction_neutral",
        "moss_image_json_instruction_action",
        "moss_image_json_instruction_qwen",
        "moss_image_json_instruction_short_qwen",
        "moss_image_json_instruction_color_qwen",
        "moss_image_json_instruction_neutral_qwen",
        "moss_image_json_instruction_action_qwen",
    ):
        instruction = vlm_presets.system_instructions_vlm[key]
        assert '"summary":{"meta":["..."],"canvas":"single","caption":"..."}' in instruction
        assert '"bbox_coord_space":"abs_centralized"' in instruction
        assert '"bbox_format":"yxyx"' in instruction
        assert '"label":"character"' in instruction
        if key.endswith("_qwen"):
            assert "[xmin, ymin, xmax, ymax]" in instruction
            assert "[ymin, xmin, ymax, xmax]" not in instruction
        else:
            assert "[ymin, xmin, ymax, xmax]" in instruction
        assert key in options
        sys_text, resp_schema = vlm_nodes.UC_VLMSysInstrPresets.execute(key).args
        assert sys_text == instruction
        assert isinstance(resp_schema, dict)
        assert resp_schema["required"] == [
            "summary",
            "items",
            "bbox_coord_space",
            "bbox_format",
        ]
        adv_text, adv_resp_schema = vlm_nodes.UC_VLMSysInstrAdvPresets.execute(
            key, False, "", ""
        ).args
        assert adv_text == instruction
        assert adv_resp_schema == resp_schema

    for key, family_anchor in (
        ("moss_image_json_instruction_neutral", "Style Deconstruction and Conceptual Vocabulary"),
        ("moss_image_json_instruction_action", "Action, Interaction, and Subject Characteristic Analysis"),
    ):
        instruction = vlm_presets.system_instructions_vlm[key]
        original, hardening = _split_hardening_block(instruction)
        perspective, remainder = hardening.split(HARDENING_HEADINGS[1], 1)
        quotation, language = remainder.split(HARDENING_HEADINGS[2], 1)
        assert perspective.rstrip() == APPROVED_PERSPECTIVE_BLOCK
        assert HARDENING_FORBIDDEN.search(hardening) is None
        assert HARDENING_LIST_LINE.search(hardening) is None
        assert re.search(r"\bcamera\b", hardening, re.IGNORECASE) is None
        assert "double quotation marks" in quotation.lower()
        assert "hyphenated words" in language.lower()
        assert family_anchor in original
        assert "## Handling Refusal Constraints (PTSD Rule)" in original
        assert "up to 16 **image inputs**" in original
        assert "ONLY the first **image input** as the spatial basis" in original
        assert "30 to 60 words" not in instruction
        _, resp_schema = vlm_nodes.UC_VLMSysInstrPresets.execute(key).args
        schema_str = str(resp_schema)
        assert "first input image" in schema_str
        assert "30 to 60 words" not in schema_str
        assert "body-part sub-boxes" not in schema_str

    for ig_key in (
        "ideogram_4_json_instruction",
        "ideogram_4_json_instruction_short",
        "ideogram_4_json_instruction_style",
        "ideogram_4_json_instruction_color",
    ):
        _, ig_schema = vlm_nodes.UC_VLMSysInstrPresets.execute(ig_key).args
        assert isinstance(ig_schema, dict)
        assert "compositional_deconstruction" in ig_schema["properties"]

    _, plain_schema = vlm_nodes.UC_VLMSysInstrPresets.execute(
        "neutral_system_instruction"
    ).args
    assert plain_schema is None
    assert '"palette"' not in vlm_presets.system_instructions_vlm["moss_image_json_instruction_short"]
    assert '"palette":["#000000","#FFFFFF"]' in vlm_presets.system_instructions_vlm["moss_image_json_instruction_color"]
    assert "moss_image" in vlm_nodes.UC_VLMSysQueryAddPresets.define_schema().inputs[0].options
    assert "moss_image" in vlm_nodes.UC_VLMSysQueryRawPresets.define_schema().inputs[0].options


