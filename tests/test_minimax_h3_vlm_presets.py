import ast
import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_minimax_h3_vlm_preset_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_minimax_h3_vlm_preset_test import (
    minimax_h3_vlm_experimental_presets,
    minimax_h3_vlm_presets,
)
from utils_collection_minimax_h3_vlm_preset_test.nodes import minimax_h3_vlm_nodes


NATIVE_H3_PRESETS = [
    "minimax_h3_base",
    "minimax_h3_first_last_frame",
    "minimax_h3_reference",
]
REFERENCE_EXPANSION_PRESETS = [
    "minimax_h3_last_frame",
    "minimax_h3_full_reference",
    "minimax_h3_minimalist_product_ad_reference",
    "minimax_h3_brand_promo_reference",
    "minimax_h3_stylized_3d_animation_reference",
    "minimax_h3_papercraft_stop_motion_reference",
    "minimax_h3_paper_collage_reference",
    "minimax_h3_music_video_reference",
    "minimax_h3_coop_game_intro_reference",
    "minimax_h3_handdrawn_live_action_reference",
]
TIMELINE_FL2VA_PRESETS = [
    "minimax_h3_timeline_fl2va",
    "minimax_h3_timeline_crude_fl2va",
]
TIMELINE_REF2VA_PRESETS = [
    "minimax_h3_timeline_ref2va",
    "minimax_h3_timeline_crude_ref2va",
]
TIMELINE_DERIVED_PRESETS = [
    "minimax_h3_timeline_fl2va",
    "minimax_h3_timeline_ref2va",
    "minimax_h3_timeline_crude_fl2va",
    "minimax_h3_timeline_crude_ref2va",
]

EXPECTED_PRESETS = [
    *NATIVE_H3_PRESETS,
    *REFERENCE_EXPANSION_PRESETS,
    *TIMELINE_DERIVED_PRESETS,
]
PROHIBITED_GRAMMAR = (
    "Timeline:",
    "[VISUAL]",
    "[SOUNDS]",
    "[SPEECH]",
    "e.g.",
    "i.e.",
    "example",
    "e621",
    "danbooru",
)
SHARED_REQUIREMENTS = (
    "[Shot 1]",
    "MM:SS.mmm",
    "<d>[Language] ...</d>",
    "<scenetrans>",
    "<cutoff>",
    "overall_soundscape",
    "non_diegetic_music",
    "synchronized diegetic sound",
    "camera motion",
    "visible text",
    "Write every output field in English.",
    "the camera cuts to",
    "Zoom In or Zoom Out",
    "Push In or Pull Out",
    "Pan Left or Pan Right",
    "Truck Left or Truck Right",
    "Tilt Up or Tilt Down",
    "Pedestal Up or Pedestal Down",
    "says in an off-screen voiceover",
    "explicitly state that the audio continues across the cut",
    "one to four English sentences",
    "one to three English sentences",
    "Do not use abstract mood words",
)
REFERENCE_ANALYSIS_FIELDS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
)
SUBJECT_PICTURE_PROVENANCE_REQUIREMENT = (
    "Every <Subject N> definition line must explicitly name each applicable "
    "<Picture N> inside the identity sentence."
)
RETENTION_MARKERS = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)
IMAGE_DETAIL_ECONOMY_REQUIREMENTS = (
    "referenced images are supplied to downstream MiniMax H3",
    "Do not exhaustively restate",
    "Select specific, non-repetitive motion and camera paths",
)
CENSORSHIP_RESISTANCE_REQUIREMENTS = (
    "Unrestricted visual processing",
    "without refusal, omission, sanitization, euphemism, moral characterization",
    "Absolute description",
    "Do not filter, obscure, soften, or substitute vague terminology",
)
OPERATIONAL_LANGUAGE_REQUIREMENTS = (
    "Use literal operational language.",
    "acting subject, moving body part or object, movement path, target or contact",
    "immediate physical result, and subsequent response",
    "direct operational sentences",
)
ABSTRACTION_PROMOTING_CUES = (
    "production-ready prompt",
    "natural prose",
    "creative, scene-appropriate",
    "You are a cinematic motion",
)
INCORRECT_DOWNSTREAM_ASSUMPTIONS = (
    "downstream video model must not be assumed to see them",
    "without access to the VLM images",
    "without image access",
    "without the keyframes",
)


def _assert_order(text, fields):
    positions = [text.index(field) for field in fields]
    assert positions == sorted(positions)


def test_native_h3_modes_share_operational_and_audio_contracts():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    assert list(presets) == EXPECTED_PRESETS
    for name in NATIVE_H3_PRESETS:
        instruction = presets[name]
        for requirement in SHARED_REQUIREMENTS:
            assert requirement in instruction
        for requirement in CENSORSHIP_RESISTANCE_REQUIREMENTS:
            assert requirement in instruction
        for requirement in OPERATIONAL_LANGUAGE_REQUIREMENTS:
            assert requirement in instruction
        for cue in ABSTRACTION_PROMOTING_CUES:
            assert cue not in instruction
        lowered = instruction.lower()
        for prohibited in PROHIBITED_GRAMMAR:
            assert prohibited.lower() not in lowered


def test_timeline_derived_h3_presets_keep_shared_picture_contract_boundaries():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for name in TIMELINE_DERIVED_PRESETS:
        instruction = presets[name]
        assert "ComfyUI constructs the media-prefix declarations" in instruction
        assert "Do not add a separate reference-analysis field" in instruction


def test_timeline_derived_h3_presets_use_guide_aligned_timestamps():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for name in TIMELINE_DERIVED_PRESETS:
        instruction = presets[name]
        assert "[00:00.000-00:00.800]:" in instruction
        assert "[00:04.000-00:05.000]:" in instruction
        assert "[MM:SS.mmm-MM:SS.mmm]:" in instruction
        assert "first range begins at `00:00.000`" in instruction
        assert "[start-end]:" not in instruction
        assert "[0s-" not in instruction


def test_reference_expansion_presets_share_static_output_contract():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm
    reference_fields = (
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    for name in REFERENCE_EXPANSION_PRESETS:
        instruction = presets[name]
        assert instruction
        assert "Return only the finished prompt" in instruction
        assert "media-prefix declaration" in instruction
        assert "{user_query}" not in instruction
        assert "{system_query}" not in instruction
        main_field = (
            "integrated_multimodal_description:"
            if name == "minimax_h3_last_frame"
            else "detailed_description:"
        )
        _assert_order(
            instruction,
            (*reference_fields[:3], main_field, *reference_fields[4:]),
        )


def test_last_frame_preset_uses_one_final_picture_and_l2va_alignment():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_last_frame"
    ]

    assert "exactly one ordered VLM image" in instruction
    assert "Use only the existing <Picture 1> identifier" in instruction
    assert "<Picture 1> is the exact final frame" in instruction
    assert "<Picture 2>" not in instruction
    assert (
        "<Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the "
        "target video."
    ) in instruction
    assert "describe the operations that produce it" in instruction


def test_full_reference_preset_requires_explicit_asset_roles():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_full_reference"
    ]

    for label in ("<Subject N>", "<Picture N>", "<Video N>", "<Audio N>"):
        assert label in instruction
    for marker in (*RETENTION_MARKERS, "fully_copy", "partially_copy", "reference"):
        assert marker in instruction
    assert "Use only reference identifiers that are explicitly established" in instruction
    assert "Never invent an asset or label that was not supplied" in instruction
    assert "Number each category independently" in instruction


def test_product_and_brand_presets_preserve_verified_identity_and_copy():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm
    product = presets["minimax_h3_minimalist_product_ad_reference"]
    brand = presets["minimax_h3_brand_promo_reference"]

    for requirement in (
        "original body color",
        "one primary action",
        "one single-line copy event at a time",
        "stable full-frame product composition",
        "claims, specifications, variants, metrics",
    ):
        assert requirement in product
    for requirement in (
        "verified brand or product images",
        "names, features, metrics, slogans, and claims established",
        "logo clear space",
        "Never create fake metrics",
        "stable final lockup",
    ):
        assert requirement in brand


def test_stylized_3d_preset_tracks_identity_space_and_performance_handoffs():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_stylized_3d_animation_reference"
    ]

    for requirement in (
        "character identity card",
        "named landmarks",
        "lighting baseline",
        "exact state handed to the next beat or shot",
        "anticipation, squash and stretch, overshoot, follow-through",
        "Do not emit a shot table",
    ):
        assert requirement in instruction


def test_paper_presets_keep_distinct_material_motion_and_audio_contracts():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm
    papercraft = presets["minimax_h3_papercraft_stop_motion_reference"]
    collage = presets["minimax_h3_paper_collage_reference"]

    for requirement in (
        "tabs, brads, joints, cut edges",
        "hand-manipulated frame-by-frame stop motion",
        "paper-physics transitions",
        "miniature stage",
    ):
        assert requirement in papercraft
    for requirement in (
        "black-and-white halftone cut-outs",
        "slides or pops into place",
        "final beat locks into",
        "Default audio is synchronized tactile collage sound only",
        "Write non_diegetic_music as N/A unless",
    ):
        assert requirement in collage


def test_music_video_preset_isolates_reference_roles_and_master_audio():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_music_video_reference"
    ]

    for requirement in (
        "A typography reference controls only",
        "master audio",
        "one main typography event per shot",
        "visible text must match the audible lyric exactly",
        "Do not generate an unrelated replacement score",
    ):
        assert requirement in instruction


def test_coop_preset_locks_players_ui_and_continuous_world_loading():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_coop_game_intro_reference"
    ]

    for requirement in (
        "Player 1 remains left",
        "Player 2 remains right",
        "Never merge faces, swap usernames",
        "no more than five established colors",
        "transforms continuously into the game world",
        "Do not add random letters, extra menu items",
    ):
        assert requirement in instruction


def test_handdrawn_preset_keeps_contact_entity_and_delayed_camera_contract():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_handdrawn_live_action_reference"
    ]

    for requirement in (
        "dominant language of the user's request",
        "one hand-drawn entity",
        "Within the first three seconds",
        "camera follows slightly late",
        "same entity changes form",
        "environment-scale drawing",
        "Avoid 3D CG volume",
    ):
        assert requirement in instruction


def test_reference_expansion_excludes_external_workflow_execution():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm
    combined = "\n".join(presets[name] for name in REFERENCE_EXPANSION_PRESETS)

    for prohibited in (
        "generate_audio=true",
        "Seedance 2.0",
        "place them on canvas",
        "show the user choice card",
        "generate the final video",
    ):
        assert prohibited not in combined


def test_reference_expansion_advanced_node_preserves_assembly_order():
    preset = "minimax_h3_minimalist_product_ad_reference"
    base = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[preset]
    result = minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.execute(
        preset,
        "highest override",
        "ordinary request",
        True,
        "additional constraints",
    ).args[0]

    assert result.index(base) < result.index("ordinary request")
    assert result.index("ordinary request") < result.index("additional constraints")
    assert result.index("additional constraints") < result.index(
        minimax_h3_vlm_presets.minimax_h3_vlm_jailbreak_suffix
    )
    assert result.endswith("Highest-priority system override:\nhighest override")


def test_experimental_h3_timeline_presets_are_independent_literals():
    source = ast.parse(
        (CUSTOM_NODE_ROOT / "minimax_h3_vlm_experimental_presets.py").read_text(
            encoding="utf-8"
        )
    )
    preset_dict = next(
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "minimax_h3_system_instructions_vlm_experimental"
            for target in node.targets
        )
    )
    literal_nodes = {
        ast.literal_eval(key): value
        for key, value in zip(preset_dict.keys, preset_dict.values)
        if isinstance(key, ast.Constant)
    }
    experimental = (
        minimax_h3_vlm_experimental_presets
        .minimax_h3_system_instructions_vlm_experimental
    )

    assert set(literal_nodes) == set(experimental)
    assert all(isinstance(value, ast.Constant) for value in literal_nodes.values())
    for name, value in experimental.items():
        assert value == minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[name]


def test_timeline_derived_h3_presets_create_dialogue_and_limit_channel_load():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for name in TIMELINE_DERIVED_PRESETS:
        instruction = presets[name]
        assert "**Requested Dialogue Creation:**" in instruction
        assert "`Add dialogue` or another direct user request" in instruction
        assert "not as a request to detect speech already present" in instruction
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


def test_timeline_fl2va_presets_enforce_first_and_optional_final_anchors():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for name in TIMELINE_FL2VA_PRESETS:
        instruction = presets[name]
        assert "### MiniMax H3 FL2VA Existing Picture Anchor Contract" in instruction
        assert "`<Picture 1>` is always the fixed first frame at 0.00 seconds" in instruction
        assert "`<Picture 2>` is the fixed final frame" in instruction
        assert "`<Picture 2>` must not appear anywhere in the response" in instruction
        assert "Never invent a third picture, exchange the anchor roles" in instruction
        assert "Establish `<Picture 1>` in the first timeline interval" in instruction
        assert "cite it in the final timeline interval" in instruction


def test_timeline_ref2va_presets_enforce_ordered_inferred_reference_roles():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for name in TIMELINE_REF2VA_PRESETS:
        instruction = presets[name]
        assert "### MiniMax H3 REF2VA Existing Picture Reference Contract" in instruction
        assert "the only valid identifiers are `<Picture 1>` through `<Picture M>`" in instruction
        assert "Do not mention `<Picture M+1>`" in instruction
        assert "Preserve VLM input order and use every supplied picture" in instruction
        assert "Establish every supplied picture at its first materially relevant point" in instruction
        assert "Do not automatically classify any picture as the first frame" in instruction
        assert "verify that every supplied `<Picture N>` is used" in instruction


def test_base_mode_has_only_base_output_structure():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_base"
    ]
    fields = (
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    _assert_order(instruction, fields)
    assert "subject_definitions:" not in instruction
    assert "retention_analysis:" not in instruction
    assert "detailed_description:" not in instruction
    assert "<Picture" not in instruction
    assert "any supplied VLM images" in instruction
    assert "not supplied to downstream MiniMax H3 Base mode" in instruction
    assert "Translate all relevant visual evidence into explicit text" in instruction
    assert "No image is supplied to the VLM" not in instruction
    assert "Inside integrated_multimodal_description, enforce the same operational rule" in instruction
    assert "rewrite any unresolved action" in instruction
    assert "rewrite abstract mood, narrative-purpose, or emotional-function language" in instruction


def test_first_last_mode_is_reference_analysis_plus_base_body():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_first_last_frame"
    ]
    fields = (
        *REFERENCE_ANALYSIS_FIELDS,
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    _assert_order(instruction, fields)
    assert "detailed_description:" not in instruction
    assert "<Picture 1> is the initial frame at 0.00 seconds" in instruction
    assert "When <Picture 2> exists, it is the final frame at the exact requested endpoint" in instruction
    assert "Never exchange these roles" in instruction
    assert "infer a third picture" in instruction
    assert "With one image, write:" in instruction
    assert "With two images, write:" in instruction
    assert "<Picture 2> must not appear anywhere in the generated response" in instruction
    assert "Never mention an identifier for media that was not supplied" in instruction
    assert "standalone <Picture 1> definition" in instruction
    assert "standalone <Picture 2> definition only when the second image exists" in instruction
    assert SUBJECT_PICTURE_PROVENANCE_REQUIREMENT in instruction
    assert "[keyframe completion]" in instruction
    assert "prefer one continuous shot" in instruction
    assert "Use multiple shots only when the user explicitly requests them" in instruction
    assert "Picture 2 (from Shot N) aligns with the S.SS-second mark" in instruction
    assert "Replace N with the actual final shot number" in instruction
    assert "Never output N or S.SS as literal placeholders" in instruction
    assert "write exactly one label definition per line" in instruction
    assert "write exactly one entry per line" in instruction
    assert "write no entry for undefined or unlabeled content" in instruction
    assert "If the second image is mirrored" in instruction
    assert "Do not end with a bare claim that the exact final composition is reached" in instruction
    assert "Reject and rewrite merged label definitions" in instruction
    assert "Do not return the prompt until every check passes" in instruction
    for marker in RETENTION_MARKERS:
        assert marker in instruction


def test_reference_mode_uses_ordered_multi_keyframe_structure():
    instruction = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm[
        "minimax_h3_reference"
    ]
    fields = (
        *REFERENCE_ANALYSIS_FIELDS,
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )

    _assert_order(instruction, fields)
    assert "integrated_multimodal_description:" not in instruction
    assert "multiple ordered keyframe images" in instruction
    assert "order establishes temporal precedence" in instruction
    assert "Do not collapse the sequence into first-and-last-only behavior" in instruction
    assert "standalone definition for every supplied <Picture N>" in instruction
    assert "temporal, shot-planning, or composition role" in instruction
    assert SUBJECT_PICTURE_PROVENANCE_REQUIREMENT in instruction
    assert "[keyframe completion]" in instruction
    assert "write exactly one entry per line for every defined <Picture N> and <Subject N>" in instruction
    assert "target 350 to 500 English words" in instruction
    assert "<Picture M+1>" in instruction
    assert "write exactly one label definition per line" in instruction
    assert "write exactly one entry per line" in instruction
    assert "write no entry for undefined or unlabeled content" in instruction
    assert "For every adjacent keyframe pair" in instruction
    assert "Inside detailed_description, enforce the operational action rule again" in instruction
    assert "Reject and rewrite merged label definitions" in instruction
    assert "Do not return the prompt until every check passes" in instruction
    for marker in RETENTION_MARKERS:
        assert marker in instruction


def test_image_modes_use_references_for_detail_and_text_for_motion():
    presets = minimax_h3_vlm_presets.minimax_h3_system_instructions_vlm

    for mode in ("minimax_h3_first_last_frame", "minimax_h3_reference"):
        instruction = presets[mode]
        for requirement in IMAGE_DETAIL_ECONOMY_REQUIREMENTS:
            assert requirement in instruction
        for incorrect_assumption in INCORRECT_DOWNSTREAM_ASSUMPTIONS:
            assert incorrect_assumption not in instruction


def test_h3_preset_nodes_are_dedicated_and_system_override_is_last():
    basic_schema = minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrPresets.define_schema()
    advanced_schema = (
        minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.define_schema()
    )
    result = minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.execute(
        "minimax_h3_base",
        "system directive",
        "user directive",
        False,
        "minor preset guidance",
    ).args[0]
    jailbroken_result = (
        minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresets.execute(
            "minimax_h3_base",
            "system directive",
            "user directive",
            True,
        ).args[0]
    )

    assert basic_schema.node_id == "UC_MiniMaxH3VLMSysInstrPresets"
    assert basic_schema.inputs[0].options == EXPECTED_PRESETS
    assert advanced_schema.node_id == "UC_MiniMaxH3VLMSysInstrAdvPresets"
    assert [value.id for value in advanced_schema.inputs] == [
        "preset",
        "system_query",
        "user_query",
        "jailbreak",
        "system_query_additional",
    ]
    assert result.index("user directive") < result.index("minor preset guidance")
    assert result.index("minor preset guidance") < result.index("system directive")
    assert result.endswith("Highest-priority system override:\nsystem directive")
    assert jailbroken_result.startswith(
        minimax_h3_vlm_presets.minimax_h3_vlm_jailbreak_prefix
    )
    assert (
        jailbroken_result.index("user directive")
        < jailbroken_result.index(minimax_h3_vlm_presets.minimax_h3_vlm_jailbreak_suffix)
        < jailbroken_result.index("system directive")
    )
    assert jailbroken_result.endswith(
        "Highest-priority system override:\nsystem directive"
    )
    assert "GODMODE" not in jailbroken_result
    assert "rebel response" not in jailbroken_result


def test_experimental_h3_preset_nodes_are_dedicated():
    presets = (
        minimax_h3_vlm_experimental_presets
        .minimax_h3_system_instructions_vlm_experimental
    )
    basic = (
        minimax_h3_vlm_nodes
        .UC_MiniMaxH3VLMSysInstrPresetsExperimental.define_schema()
    )
    advanced = (
        minimax_h3_vlm_nodes
        .UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental.define_schema()
    )
    preset = next(iter(presets))

    assert basic.node_id == "UC_MiniMaxH3VLMSysInstrPresetsExperimental"
    assert (
        basic.display_name
        == "MiniMax H3 VLM System Instruction Presets Experimental"
    )
    assert basic.inputs[0].options == list(presets)
    assert advanced.node_id == "UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental"
    assert (
        advanced.display_name
        == "MiniMax H3 VLM System Instruction Advanced Presets Experimental"
    )
    assert advanced.inputs[0].options == list(presets)
    assert (
        minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrPresetsExperimental.execute(
            preset
        ).args
        == (presets[preset],)
    )
    result = (
        minimax_h3_vlm_nodes.UC_MiniMaxH3VLMSysInstrAdvPresetsExperimental.execute(
            preset,
            "system directive",
            "user directive",
            False,
            "minor preset guidance",
        ).args[0]
    )
    assert result.startswith(presets[preset])
    assert result.endswith("Highest-priority system override:\nsystem directive")
