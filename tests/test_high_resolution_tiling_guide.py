import pathlib
import sys
import types

CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_high_resolution_tiling_guide_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_high_resolution_tiling_guide_test.nodes.utils_nodes import UC_HighResolutionTilingGuide
finally:
    cli_args.cpu = prior_cpu

EXPECTED_TOPICS = ["workflow", "split_settings", "overlap_masks", "depth_structure_mask", "visual_conditioning", "sampling", "accumulation", "minimax_h3_latent_upscale"]


def test_high_resolution_tiling_guide_schema_and_topics():
    schema = UC_HighResolutionTilingGuide.define_schema()
    topic_input = {value.id: value for value in schema.inputs}["topic"]

    assert schema.node_id == "UC_HighResolutionTilingGuide"
    assert schema.display_name == "High Resolution Tiling Guide"
    assert topic_input.options == EXPECTED_TOPICS
    assert topic_input.default == "workflow"
    for topic in EXPECTED_TOPICS:
        markdown = UC_HighResolutionTilingGuide.execute(topic).args[0]
        assert markdown.startswith("## ")
        assert "Unknown topic" not in markdown


def test_high_resolution_tiling_guide_documents_execution_contract():
    workflow = UC_HighResolutionTilingGuide.execute("workflow").args[0]
    masks = UC_HighResolutionTilingGuide.execute("overlap_masks").args[0]
    depth = UC_HighResolutionTilingGuide.execute("depth_structure_mask").args[0]
    conditioning = UC_HighResolutionTilingGuide.execute("visual_conditioning").args[0]
    sampling = UC_HighResolutionTilingGuide.execute("sampling").args[0]
    accumulation = UC_HighResolutionTilingGuide.execute("accumulation").args[0]
    minimax_upscale = UC_HighResolutionTilingGuide.execute("minimax_h3_latent_upscale").args[0]

    assert "true ComfyUI lists" in workflow
    assert "exact padded tensor sent to VAE encoding" in workflow
    assert "non-overlapping interior is solid `1.0`" in masks
    assert "`depth_influence` ranges from `-1` to `1`" in depth
    assert "multiplies the existing overlap mask" in depth
    assert "accumulator continues using the original overlap feather" in depth
    assert "paired by index" in conditioning
    assert "full-image caption" in conditioning
    assert "`advanced` uses the KJNodes-compatible threshold multiplier" in sampling
    assert "Connect its `model` output to the sampler guider" in sampling
    assert "`differential_diffusion_value` is mode-dependent" in sampling
    assert "does not implement a sampler" in sampling
    assert "exactly the tiles described by that layout" in accumulation
    assert "UC_MiniMaxH3LatentUpscaleLoader" in minimax_upscale
    assert "UC_MiniMaxH3LatentUpscale" in minimax_upscale
    assert "UC_MiniMaxH3SpatialTilePatcher" in minimax_upscale
    assert "SplitSigmasDenoise" in minimax_upscale
    assert "tile_width=768" in minimax_upscale
