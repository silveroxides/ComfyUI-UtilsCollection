import asyncio
import importlib.util
import json
import pathlib
import re
import sys

TARGETED_DESCRIPTION_IDS = {
    "UC_VLMSysInstrPresetsExperimental",
    "UC_VLMSysInstrLegacyPresets",
    "UC_VLMSysInstrAdvPresetsExperimental",
    "UC_MiniMaxH3VLMSysInstrPresetsExperimental",
    "UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental",
    "UC_HighResolutionTilingGuide",
    "UC_ExtractMask",
    "UC_Ideogram4BoundingBoxCrop",
    "UC_TextConcatenateListsAutogrow",
    "UC_MathAdd",
    "UC_MathSubtract",
    "UC_MathMultiply",
    "UC_MathDivide",
    "UC_MathPower",
    "UC_MathFloor",
    "UC_MathCeil",
    "UC_MathRound",
    "UC_MathModulo",
    "UC_MathAbs",
    "UC_MathSqrt",
    "UC_MathSin",
    "UC_MathCos",
    "UC_MathTan",
    "UC_MathMin",
    "UC_MathMax",
    "UC_MathClamp",
    "UC_MathNumberConvert",
    "UC_StringToNumber",
    "UC_NumberToString",
    "UC_MathCompare",
    "UC_MathOperation",
    "UC_MathAspectRatio",
    "UC_LogicIF",
    "UC_LogicAND",
    "UC_LogicOR",
    "UC_LogicNOT",
    "UC_LogicXOR",
    "UC_SigmoidOffsetScheduler",
}

ACTION_REQUIRED_TOOLTIP_INPUTS = {
    "UC_AdvancedConsensusConfiguration": {"blend_method", "consensus_type", "alignment_threshold", "similarity_threshold", "power_alpha", "diversity_beta", "rescale_norm", "global_scale", "dynamic_similarity_contrast", "soft_comfort_bandpass", "position_weight", "preserve_common_prefix"},
    "UC_ModifyMask": {"expand", "incremental_expandrate", "tapered_corners", "flip_input", "blur_radius", "lerp_alpha", "decay_factor", "fill_holes", "lower_clamp", "upper_clamp"},
    "UC_ImageBlendByMask": {"destination", "source", "mode", "blend_percentage", "resize_source", "mask"},
    "UC_ImagePad": {"left", "right", "top", "bottom", "extra_padding", "mask", "target_width", "target_height"},
    "UC_CropByMask": {"padding"},
    "UC_ImageCropMerge": {"cropped_image", "original_image", "crop_x", "crop_y", "crop_width", "crop_height", "resize_method", "mask"},
    "UC_ImageAndMaskResize": {"target", "resize_method", "crop", "mask_blur_radius", "width", "height"},
    "UC_ResizeMask": {"keep_proportions", "upscale_method", "crop"},
    "UC_StagedLayeredBackgroundCompositeOptions": {"border_cleanup_width", "artifact_cleanup_radius", "gap_fill_radius", "feather_radius", "image_resize_method", "mask_resize_method"},
    "UC_StagedMediaPipeFaceOptions": {"detection_threshold", "maximum_faces", "bbox_expansion", "mask_expansion", "face_feather_radius", "initial_face_scale"},
    "UC_ImageMatchProperties": {"original_image", "generated_image", "overall_weight", "color_weight", "lighting_weight"},
    "UC_TextGenerate": {"max_length", "sampling_mode", "seed", "image_inputs"},
    "UC_PowerShiftScheduler": {"power", "midpoint_shift"},
    "UC_RadianceShiftScheduler": {"power", "midpoint_shift"},
    "UC_StagedLayerCrops": {"layer_masks"},
    "UC_FromList": {"items", "start_index", "number_of_entries"},
    "UC_ConditioningConsensusBlend": {"conditioning_inputs"},
    "UC_TextEncodeSystemEditAdvanced": {"image_inputs"},
    "UC_TextEncodeGemmaSystemEditAdvanced": {"image_inputs"},
}

CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_metadata_test"


def _load_extension_package():
    spec = importlib.util.spec_from_file_location(PACKAGE_NAME, CUSTOM_NODE_ROOT / "__init__.py", submodule_search_locations=[str(CUSTOM_NODE_ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(PACKAGE_NAME, package)
    spec.loader.exec_module(package)
    return package


def test_registered_non_deprecated_nodes_have_search_metadata():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    checked = []

    for node_class in node_classes:
        assert node_class.__module__.startswith(f"{PACKAGE_NAME}.nodes.")
        schema = node_class.define_schema()
        if schema.is_deprecated or "(Legacy)" in (schema.display_name or ""):
            continue
        checked.append(schema.node_id)
        assert schema.description and not schema.description.startswith("Provides UC_")
        assert schema.search_aliases
        assert len(schema.search_aliases) == len(set(schema.search_aliases))

    assert "UC_StaticFloat" in checked
    assert "UC_ConditioningConsensusBlend" in checked
    assert "UC_VLMSysQueryRawPresets" in checked
    assert len(checked) == len(set(checked))


def test_readme_available_nodes_match_current_registration():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    expected = set()
    for node_class in node_classes:
        schema = node_class.define_schema()
        if not schema.is_deprecated and "(Legacy)" not in (schema.display_name or ""):
            expected.add(schema.node_id)

    readme = (CUSTOM_NODE_ROOT / "README.md").read_text(encoding="utf-8")
    available_nodes = readme.split("## Available nodes", 1)[1]
    listed = re.findall(r"^- `([^`]+)`$", available_nodes, re.MULTILINE)

    assert len(listed) == len(set(listed))
    assert set(listed) == expected


def test_executable_guides_reference_registered_node_ids():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    schemas = {node_class.define_schema().node_id: node_class for node_class in node_classes}
    references = set()

    for guide_id in {"UC_EncoderNodesGuide", "UC_CompositeNodesGuide", "UC_HighResolutionTilingGuide"}:
        guide = schemas[guide_id]
        topic = next(value for value in guide.define_schema().inputs if value.id == "topic")
        for option in topic.options:
            markdown = guide.execute(option).args[0]
            references.update(node_id for node_id in re.findall(r"\bUC_[A-Za-z0-9_]+\b", markdown) if any(character.islower() for character in node_id.removeprefix("UC_")))

    assert references
    assert references <= schemas.keys()


def test_shipped_workflows_reference_registered_node_ids():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    registered = {node_class.define_schema().node_id for node_class in node_classes}
    references = set()

    for workflow_path in (CUSTOM_NODE_ROOT / "workflows").glob("*.json"):
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        references.update(node["type"] for node in workflow["nodes"] if node.get("type", "").startswith("UC_"))

    assert references
    assert references <= registered


def test_legacy_nodes_do_not_inherit_canonical_search_metadata():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    legacy_static = next(node for node in node_classes if node.__name__ == "StaticInt")
    schema = legacy_static.define_schema()

    assert "(Legacy)" in schema.display_name
    assert not schema.search_aliases


def test_targeted_nodes_have_specific_plain_language_descriptions():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    schemas = {node.define_schema().node_id: node.define_schema() for node in node_classes}

    assert TARGETED_DESCRIPTION_IDS <= schemas.keys()
    for node_id in TARGETED_DESCRIPTION_IDS:
        schema = schemas[node_id]
        assert schema.description != f"Provides {schema.display_name} functionality."

    descriptions = {node_id: schemas[node_id].description.lower() for node_id in TARGETED_DESCRIPTION_IDS}
    assert "broadcasting" in descriptions["UC_TextConcatenateListsAutogrow"]
    assert "left to right" in descriptions["UC_MathDivide"]
    assert "divisor is zero" in descriptions["UC_MathDivide"]
    assert "divisor is zero" in descriptions["UC_MathModulo"]
    assert "negative input" in descriptions["UC_MathSqrt"]
    assert "odd number" in descriptions["UC_LogicXOR"]
    assert "simplest whole-number aspect ratio" in descriptions["UC_MathAspectRatio"]
    assert "originally made for the chroma model" in descriptions["UC_SigmoidOffsetScheduler"]


def test_action_required_controls_have_targeted_tooltips():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    schemas = {node.define_schema().node_id: node.define_schema() for node in node_classes}

    for node_id, input_ids in ACTION_REQUIRED_TOOLTIP_INPUTS.items():
        inputs = {value.id: value for value in schemas[node_id].inputs}
        assert input_ids <= inputs.keys()
        assert all(inputs[input_id].tooltip for input_id in input_ids)

    sampling_mode = next(value for value in schemas["UC_TextGenerate"].inputs if value.id == "sampling_mode")
    sampling_inputs = [value for option in sampling_mode.options for value in option.inputs]
    assert sampling_inputs
    assert all(value.tooltip for value in sampling_inputs)

    attention_mode = next(value for value in schemas["UC_UnifiedAttentionPatcher"].inputs if value.id == "attention_mode")
    attention_inputs = [value for option in attention_mode.options for value in option.inputs]
    assert attention_inputs
    assert all(value.tooltip for value in attention_inputs)


def test_corrected_metadata_matches_current_behavior():
    package = _load_extension_package()
    node_classes = asyncio.run(package.SamplingUtils().get_node_list())
    schemas = {node.define_schema().node_id: node.define_schema() for node in node_classes}

    unified = schemas["UC_UnifiedBackgroundReplace"]
    foregrounds = next(value for value in unified.inputs if value.id == "foreground_images")
    assert "independently places each" in unified.description
    assert "each flattened image produces an independent output" in foregrounds.tooltip
    assert "center" not in unified.description.lower()
    assert "center" not in foregrounds.tooltip.lower()

    adjusted_box = schemas["UC_AdjustBoundingBox"]
    assert "around its center" in adjusted_box.description
    assert "boundary" not in adjusted_box.description.lower()
