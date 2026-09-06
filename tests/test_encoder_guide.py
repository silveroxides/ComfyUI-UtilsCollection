import pathlib
import sys
import types


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_encoder_guide_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_encoder_guide_test.utils_nodes import UC_EncoderNodesGuide
finally:
    cli_args.cpu = prior_cpu


EXPECTED_TOPICS = [
    "node_catalog",
    "prompt_templates_and_weighting",
    "image_inputs_and_placeholders",
    "resolution_and_reference_latents",
    "visual_fusion",
    "consensus_settings",
    "formulas_and_alignment",
    "embedding_export",
    "compatibility_nodes",
]


def test_encoder_guide_topics_are_complete_and_render_markdown():
    topic_input = {
        value.id: value for value in UC_EncoderNodesGuide.define_schema().inputs
    }["topic"]

    assert topic_input.options == EXPECTED_TOPICS
    assert topic_input.default == "node_catalog"
    for topic in EXPECTED_TOPICS:
        markdown = UC_EncoderNodesGuide.execute(topic).args[0]
        assert markdown.startswith("## ")
        assert "Unknown topic" not in markdown


def test_encoder_guide_documents_minimax_h3_visual_contract():
    markdown = UC_EncoderNodesGuide.execute(
        "image_inputs_and_placeholders"
    ).args[0]

    assert "`qwen3vl_32b`" in markdown
    assert "raw `<Picture N>:` presentation" in markdown
    assert "Keep `ref_latent_mode` off" in markdown


def test_encoder_guide_covers_current_visual_fusion_contract():
    resolution = UC_EncoderNodesGuide.execute(
        "resolution_and_reference_latents"
    ).args[0]
    fusion = UC_EncoderNodesGuide.execute("visual_fusion").args[0]
    consensus = UC_EncoderNodesGuide.execute("consensus_settings").args[0]

    assert all(
        value in resolution
        for value in (
            "256",
            "3584",
            "1` through `15",
            "UC_AdvancedVisualConditioningEncode",
            "encode only the selected base VLM resolution",
            "`1` remains one resolution sample",
            "complete conditioning until cross-resolution consensus",
            "visual and DeepStack tokens fuse before",
        )
    )
    assert all(
        value in fusion
        for value in (
            "UC_AdvancedVisualConditioningEncode",
            "UC_Krea2TokenAttentionWeight",
            "spatial-checkerboard",
            "spatial-block-interleave",
            "spatial-dither-random",
            "`dither-random-reverse`",
            "`dither-random-forward`",
            "optional spatial-only configuration",
            "UC_AdvancedVisConEncoder",
            "UC_AdvancedVisConEncoderTokenFusion",
            "sequential stages",
            "never crossfaded",
        )
    )
    assert "Text Scaled Encoder (Advanced)" not in fusion
    assert "consumed directly by `UC_ConditioningConsensusBlend`" in consensus
    assert "complete per-lane and per-resolution conditionings" in consensus
    assert "does not replace, disable, or crossfade against spatial fusion" in consensus


def test_encoder_guide_makes_primary_and_specialized_hierarchy_explicit():
    catalog = UC_EncoderNodesGuide.execute("node_catalog").args[0]

    assert "### Primary encoder: use this by default" in catalog
    assert "the recommended encoder for nearly all conditioning workflows" in catalog
    assert "no image connected" in catalog
    assert "text-only conditioning" in catalog
    assert "It is not a replacement for `UC_AdvancedVisualConditioningEncode`" in catalog
    assert "`UC_AdvancedVisConEncoderTokenFusion`" in catalog
    assert "fuses visual and DeepStack tokens before one encode" in catalog
    assert "### Specialized encoders" in catalog
    assert "### Embedding export" in catalog

    primary_position = catalog.index("`UC_AdvancedVisualConditioningEncode`")
    consensus_position = catalog.index("`UC_AdvancedVisConEncoder`")
    specialized_position = catalog.index("### Specialized encoders")
    export_position = catalog.index("### Embedding export")
    assert primary_position < consensus_position < specialized_position < export_position

    assert catalog.index("`UC_VisualFusionConfig`") < consensus_position
    assert "`UC_AdvancedConsensusConfiguration`" in catalog
    assert "extends Text Consensus Blend Configurator" in catalog
    assert "`UC_ConditioningConsensusBlend`" in catalog


def test_encoder_guide_separates_flattened_sources_from_consensus_lanes():
    images = UC_EncoderNodesGuide.execute("image_inputs_and_placeholders").args[0]

    assert "flattens active autogrow sockets and image batches" in images
    assert "does not use global image flattening" in images
    assert "Visual sources, batch lanes, and resolution variants remain separate" in images
    assert "pair equal-index images into independent lanes" in images
    assert "Singleton image sockets broadcast into every batch lane" in images


def test_encoder_guide_explains_advanced_minimax_h3_role_contract():
    catalog = UC_EncoderNodesGuide.execute("node_catalog").args[0]
    images = UC_EncoderNodesGuide.execute("image_inputs_and_placeholders").args[0]

    assert "`UC_AdvancedMiniMaxH3ImageToVideo`" in catalog
    assert "Connected `first_frame` or `last_frame` inputs select the keyframe path" in images
    assert "`reference_images` autogrow selects native-reference mode" in images
    assert "`fusion_images` autogrow is Qwen-only" in images
    assert "cannot be combined with frame or fusion inputs" in images
    assert "`ref_image_size`" in images
    assert "`vlm_resolution` independently prepares every Qwen copy" in images
    assert "passed directly to the native H3 tokenizer without a system template" in images
    assert "Qwen-only visual conditioning" in images
    assert "do not create `minimax_keyframes` or an H3 latent" in images


def test_encoder_guide_lists_only_registered_compatibility_migrations_as_mappings():
    compatibility = UC_EncoderNodesGuide.execute("compatibility_nodes").args[0]
    mappings = (
        "`TextEncodeSystemEditPlusAdvanced` → `UC_TextEncodeSystemEditAdvanced`",
        "`TextEncodeGemmaSystemEditPlusAdvanced` → `UC_TextEncodeGemmaSystemEditAdvanced`",
        "`TextEncodeKrea2SystemEditScaledAdv` → `UC_AdvancedVisualConditioningEncode`",
        "`UC_Krea2InputEmbeds` → `UC_VLMInputEmbeds`",
        "`UC_Qwen3VLInputEmbeds` → `UC_VLMInputEmbeds`",
        "`TextEncodeKrea2SysEditScaledAdvAttn` → `UC_Krea2TokenAttentionWeight`",
    )

    assert all(mapping in compatibility for mapping in mappings)
    assert "no interface-preserving migration is registered" in compatibility
    assert "Do not infer a widget mapping" in compatibility
    assert "`UC_TextEncodeLtxv2SystemPrompt` remains a current specialized node" in compatibility
