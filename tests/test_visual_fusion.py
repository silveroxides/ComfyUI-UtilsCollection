import pathlib
import sys
import types

import pytest
import torch
from comfy.sd1_clip import load_embed


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_test import embedding_helpers, encoder_helpers, encoder_nodes
    from utils_collection_test.encoder_nodes import (
        UC_AdvancedVisualConditioningEncode,
        UC_Krea2TokenAttentionWeight,
        UC_VisualFusionConfig,
    )
finally:
    cli_args.cpu = prior_cpu


@pytest.mark.parametrize("method,invalid,message", [
    ("spatial-checkerboard", {"seed": -1}, "seed"),
    ("spatial-dither-random", {"visual_block_size": 0}, "block size"),
])
def test_cached_fusion_preserves_validation_of_inactive_parameters(monkeypatch, tmp_path, method, invalid, message):
    monkeypatch.setattr(encoder_helpers.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    sources = [torch.zeros(4, 3), torch.ones(4, 3)]
    config = {"visual_fusion_method": method}
    with encoder_helpers.H3EncoderCache() as cache:
        encoder_helpers.fuse_visual_token_sources(sources, config, "cpu", source_grids=[(2, 2)] * 2, cache=cache)
        with pytest.raises(ValueError, match=message):
            encoder_helpers.fuse_visual_token_sources(sources, {**config, **invalid}, "cpu", source_grids=[(2, 2)] * 2, cache=cache)


def _config(method="spatial-dither-random", ratio=0.5, seed=0):
    return {
        "visual_fusion_method": method,
        "visual_block_size": 2,
        "dither_ratio": ratio,
        "seed": seed,
        "dither_secondary_pattern": "checkerboard",
        "dither_mask_cleanup": False,
        "spatial_perturbation": 0.0,
    }


def test_dither_seed_and_secondary_checkerboard():
    first = encoder_helpers.generate_spatial_fusion_mask(256, 3, "spatial-dither-random", dither_ratio=0.5, seed=7)
    repeated = encoder_helpers.generate_spatial_fusion_mask(256, 3, "spatial-dither-random", dither_ratio=0.5, seed=7)
    changed = encoder_helpers.generate_spatial_fusion_mask(256, 3, "spatial-dither-random", dither_ratio=0.5, seed=8)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=0.0).tolist() == [1, 2, 2, 1, 1, 2]
    assert encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=1.0).tolist() == [0] * 6


def test_old_mask_device_position_remains_compatible():
    mask = encoder_helpers.generate_spatial_fusion_mask(4, 2, "spatial-checkerboard", 2, 0.5, "cpu")
    assert mask.device.type == "cpu"
    assert mask.tolist() == [0, 1, 1, 0]


def test_spatial_fusion_matches_mask_and_preserves_dtype():
    sources = [
        torch.arange(12, dtype=torch.float16).reshape(6, 2),
        torch.arange(100, 112, dtype=torch.float16).reshape(6, 2),
        torch.arange(200, 212, dtype=torch.float16).reshape(6, 2),
    ]
    config = _config(seed=11)
    mask = encoder_helpers.generate_spatial_fusion_mask(6, 3, "spatial-dither-random", dither_ratio=0.5, seed=11, grid_shape=(2, 3))
    expected = torch.stack([sources[int(mask[index])][index] for index in range(6)])

    output = encoder_helpers.fuse_visual_token_sources(sources, config, "cpu", source_grids=[(2, 3)] * 3)

    assert output.dtype == torch.float16
    assert torch.equal(output, expected)


def test_nearest_grid_remap_selects_exact_tokens():
    short = torch.tensor([[0.0], [2.0]], dtype=torch.float16)
    long = torch.tensor([[10.0], [11.0], [12.0], [13.0]], dtype=torch.float16)

    output = encoder_helpers.fuse_visual_token_sources([long, short], _config(ratio=0.0), "cpu", source_grids=[(2, 2), (1, 2)])

    assert output.shape == (4, 1)
    assert output.dtype == torch.float16
    assert output.flatten().tolist() == [0.0, 2.0, 0.0, 2.0]


def test_deepstack_reuses_main_spatial_mask():
    config = _config(seed=23)
    cache = {}
    main_sources = [torch.zeros(16, 1), torch.ones(16, 1)]
    main = encoder_helpers.fuse_visual_token_sources(main_sources, config, "cpu", cache, 16, [(4, 4)] * 2)
    deepstack = {
        "a": [torch.full((16, 1), 10.0), torch.full((16, 1), 100.0)],
        "b": [torch.full((16, 1), 20.0), torch.full((16, 1), 200.0)],
    }

    layers = encoder_helpers.fuse_deepstack_layers(deepstack, config, "cpu", cache, 16, [(4, 4)] * 2)

    assert len(cache) == 1
    assert torch.equal(main.bool(), layers[0].eq(20.0))
    assert torch.equal(main.bool(), layers[1].eq(200.0))


def test_saved_embedding_is_complete_fused_visual_block(monkeypatch):
    config = {
        **_config(seed=31),
        "save_blended_embeds": True,
    }
    cache = {}
    sequence_tensors = {
        "a": torch.stack([torch.zeros(6, 1), torch.full((6, 1), 2.0)]),
        "b": torch.stack([torch.ones(6, 1), torch.full((6, 1), 3.0)]),
    }
    visual_ranges = {"a": (1, 5), "b": (1, 5)}
    raw_sources = {
        1: torch.stack([torch.arange(4), torch.arange(10, 14)]).unsqueeze(-1).float(),
        2: torch.stack([torch.arange(20, 24), torch.arange(30, 34)]).unsqueeze(-1).float(),
    }

    class RawClipModel:
        @staticmethod
        def process_tokens(tokens, _device):
            source_id = tokens[0][1]["source"]
            raw = raw_sources[source_id]
            embeds = torch.cat([torch.full((raw.shape[0], 1, 1), -10.0), raw, torch.full((raw.shape[0], 1, 1), 10.0)], dim=1)
            return embeds, None, None, [{"type": "image", "index": 1, "size": 4}]

    class Clip:
        cond_stage_model = type("CondStage", (), {"clip": "clip_model", "clip_model": RawClipModel()})()

    tokens_dict = {
        "a": {"qwen3vl_4b": [[(151652, 1.0), ({"type": "image", "source": 1}, 1.0), (151653, 1.0)]]},
        "b": {"qwen3vl_4b": [[(151652, 1.0), ({"type": "image", "source": 2}, 1.0), (151653, 1.0)]]},
    }

    saved = []
    monkeypatch.setattr(
        encoder_helpers,
        "save_blended_visual_embeddings",
        lambda tensors, config, key: saved.extend(tensors),
    )

    conditioning, _ = encoder_helpers.evaluate_conditioning_consensus_blend(
        sequence_tensors,
        {},
        config,
        "cpu",
        visual_ranges,
        clip=Clip(),
        tokens_dict=tokens_dict,
        mask_cache=cache,
        visual_grids={"a": (2, 2), "b": (2, 2)},
    )

    assert len(cache) == 1
    assert len(saved) == 2
    for batch in range(2):
        expected = encoder_helpers.fuse_visual_token_sources(
            [raw_sources[1][batch], raw_sources[2][batch]],
            config,
            "cpu",
            cache,
            source_grids=[(2, 2), (2, 2)],
        )
        assert torch.equal(saved[batch][0], torch.tensor([-10.0]))
        assert torch.equal(saved[batch][-1], torch.tensor([10.0]))
        assert torch.equal(expected, saved[batch][1:-1])
        assert not torch.equal(conditioning[batch, 1:5, :], saved[batch][1:-1])


def test_unfused_visual_export_preserves_complete_visual_blocks(monkeypatch):
    raw = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    saved = []
    monkeypatch.setattr(
        encoder_helpers,
        "save_blended_visual_embeddings",
        lambda tensors, config, key: saved.extend(tensors),
    )

    class RawClipModel:
        @staticmethod
        def process_tokens(_tokens, _device):
            embeds = torch.cat([torch.full((2, 1, 2), -1.0), raw, torch.full((2, 1, 2), 1.0)], dim=1)
            return embeds, None, None, [{"type": "image", "index": 1, "size": 4}]

    class Clip:
        cond_stage_model = type("CondStage", (), {"clip": "clip_model", "clip_model": RawClipModel()})()

    encoder_helpers.save_source_visual_embeddings(
        Clip(),
        {"qwen3vl_4b": [[(151652, 1.0), ({"type": "image"}, 1.0), (151653, 1.0)]]},
        {"save_path": "unused.safetensors"},
        "qwen3vl_4b",
        "cpu",
    )

    assert len(saved) == 2
    assert torch.equal(saved[0][1:-1], raw[0])
    assert torch.equal(saved[1][1:-1], raw[1])
    assert torch.equal(saved[0][0], torch.full((2,), -1.0))
    assert torch.equal(saved[0][-1], torch.full((2,), 1.0))


def test_saved_visual_block_detokenizes_without_template_tokens(monkeypatch):
    saved = []
    monkeypatch.setattr(
        encoder_helpers,
        "save_blended_visual_embeddings",
        lambda tensors, _config, _key: saved.extend(tensors),
    )

    class RawClipModel:
        @staticmethod
        def process_tokens(_tokens, _device):
            embeds = torch.tensor([[[0.0], [2.0], [3.0], [1.0]]])
            return embeds, None, None, [{"type": "image", "index": 1, "size": 2}]

    class Clip:
        cond_stage_model = type("CondStage", (), {"clip": "clip_model", "clip_model": RawClipModel()})()

    encoder_helpers.save_source_visual_embeddings(
        Clip(),
        {"qwen3vl_4b": [[(151652, 1.0), ({"type": "image"}, 1.0), (151653, 1.0)]]},
        {"save_path": "unused.safetensors"},
        "qwen3vl_4b",
        "cpu",
    )

    embedding_module = torch.nn.Embedding.from_pretrained(
        torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    )
    branch = embedding_helpers.ModelBranch((), embedding_module, 4, 1)
    candidates = embedding_helpers.nearest_vocabulary_tokens(
        saved[0], branch, [0, 1, 2, 3], 1, "euclidean"
    )

    assert [row[0].token_id for row in candidates] == [0, 2, 3, 1]


def test_saved_visual_block_loads_through_core_embedding_key(monkeypatch, tmp_path):
    block = torch.tensor([[-1.0, -1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 1.0]])
    monkeypatch.setattr(
        encoder_helpers.folder_paths,
        "get_folder_paths",
        lambda category: [str(tmp_path)] if category == "embeddings" else [],
    )

    encoder_helpers.save_blended_visual_embeddings(
        [block],
        {"save_path": "visual/block"},
        "qwen3vl_4b",
    )

    loaded = load_embed("visual/block", [str(tmp_path)], 2, "qwen3vl_4b")
    assert torch.equal(loaded, block)


def test_visual_block_export_rejects_unframed_placeholder():
    class RawClipModel:
        @staticmethod
        def process_tokens(_tokens, _device):
            return torch.zeros(1, 4, 2), None, None, [{"type": "image", "index": 1, "size": 2}]

    class Clip:
        cond_stage_model = type("CondStage", (), {"clip": "clip_model", "clip_model": RawClipModel()})()

    with pytest.raises(ValueError, match="unframed image placeholder"):
        encoder_helpers.save_source_visual_embeddings(
            Clip(),
            {"qwen3vl_4b": [[({"type": "image"}, 1.0)]]},
            {"save_path": "unused.safetensors"},
            "qwen3vl_4b",
            "cpu",
        )


def test_visual_embedding_key_uses_projected_h3_source_clip():
    projected_clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(clip_name="qwen3vl_8b")
    )

    assert encoder_helpers.visual_embedding_key(
        projected_clip, {"qwen3vl_32b": [[(0, 1.0)]]}
    ) == "qwen3vl_8b"


@pytest.mark.parametrize("method", ["index-consensus", "similarity-consensus", "unknown"])
def test_unsupported_methods_raise(method):
    with pytest.raises(ValueError, match="Unsupported visual fusion method"):
        encoder_helpers.fuse_visual_token_sources([torch.zeros(4, 1), torch.ones(4, 1)], _config(method), "cpu", source_grids=[(2, 2)] * 2)


def test_config_seed_and_legacy_call_compatibility():
    schema_inputs = UC_VisualFusionConfig.define_schema().inputs
    inputs = {value.id: value for value in schema_inputs}
    legacy = UC_VisualFusionConfig.execute("spatial-dither-random", 2, 0.5, False, "legacy.safetensors").args[0]
    seeded = UC_VisualFusionConfig.execute("spatial-dither-random", 2, 0.5, seed=123).args[0]

    assert [value.id for value in schema_inputs] == [
        "visual_fusion_method",
        "visual_block_size",
        "dither_ratio",
        "save_blended_embeds",
        "save_path",
        "seed",
        "visual_encoder_path",
        "dither_secondary_pattern",
        "dither_mask_cleanup",
        "spatial_perturbation",
    ]
    assert inputs["seed"].control_after_generate is True
    assert inputs["dither_secondary_pattern"].options == [
        "checkerboard",
        "block-interleave",
        "dither-random-reverse",
        "dither-random-forward",
    ]
    assert "index-consensus" not in inputs["visual_fusion_method"].options
    assert "similarity-consensus" not in inputs["visual_fusion_method"].options
    assert legacy["seed"] == 0
    assert legacy["save_path"] == "legacy.safetensors"
    assert seeded["seed"] == 123
    assert legacy["spatial_perturbation"] == 0.0
    assert "text_blend_config" not in legacy
    assert "fusion_strength" not in legacy
    assert "resolution_samples" not in legacy


def test_visual_fusion_consumers_use_aligned_integer_resolution():
    for node in (
        UC_AdvancedVisualConditioningEncode,
        UC_Krea2TokenAttentionWeight,
    ):
        resolution = {
            value.id: value for value in node.define_schema().inputs
        }["vlm_resolution"]
        assert resolution.io_type == "INT"
        assert resolution.default == 384
        assert resolution.min == 0
        assert resolution.max == 4096
        assert resolution.step == 32


def test_vlm_resolution_boundaries_and_original_sentinels():
    assert encoder_helpers.resolve_vlm_resolution(256) == 256
    assert 256 * 256 == 65_536
    assert encoder_helpers.resolve_vlm_resolution(3584) == 3584
    assert encoder_helpers.resolve_vlm_resolution(255) is None
    assert encoder_helpers.resolve_vlm_resolution(3585) is None
    assert encoder_helpers.resolve_vlm_resolution(0) is None


def test_vlm_target_is_aspect_preserving_and_grid_aligned():
    height, width = encoder_helpers.vlm_target_dimensions(600, 1200, 384)
    assert height % 32 == width % 32 == 0
    assert width / height == pytest.approx(2.0, rel=0.1)
    resized = encoder_helpers.prepare_vlm_image(torch.zeros(1, 600, 1200, 3), 384)
    assert resized.shape[1:3] == (height, width)


def test_resolution_samples_alternate_and_raise_low_base():
    image = torch.zeros(1, 512, 512, 3)
    assert encoder_helpers.vlm_resolution_samples(image, 384, 5) == [
        384,
        352,
        416,
        320,
        448,
    ]
    assert encoder_helpers.vlm_resolution_samples(image, 256, 5) == [
        320,
        288,
        352,
        256,
        384,
    ]
    assert encoder_helpers.vlm_resolution_samples(image, 512, 5, 64) == [
        512,
        448,
        576,
        384,
        640,
    ]
    assert encoder_helpers.vlm_resolution_samples(image, 0, 15) == [None]
    assert len(encoder_helpers.vlm_resolution_samples(image, 512, 15)) == 15
    with pytest.raises(ValueError, match="odd integer"):
        encoder_helpers.vlm_resolution_samples(image, 512, 4)
    with pytest.raises(ValueError, match="multiple of 32 from 32 to 512"):
        encoder_helpers.vlm_resolution_samples(image, 512, 5, 48)


def test_stale_consensus_keys_do_not_change_spatial_result():
    sources = [torch.zeros(4, 1), torch.ones(4, 1)]
    config = _config(method="spatial-checkerboard")
    expected = encoder_helpers.fuse_visual_token_sources(
        sources, config, "cpu", source_grids=[(2, 2)] * 2
    )
    actual = encoder_helpers.fuse_visual_token_sources(
        sources,
        {
            **config,
            "text_blend_config": {"blend_preset": "high_diversity_concept"},
            "fusion_strength": 0.0,
            "resolution_samples": 15,
        },
        "cpu",
        source_grids=[(2, 2)] * 2,
    )
    assert torch.equal(actual, expected)


def test_reverse_dither_random_recursively_accumulates_sources():
    token_count = 4096
    source_count = 8
    ratio = 0.5
    seed = 83
    generator = torch.Generator().manual_seed(seed)
    expected = torch.full((token_count,), source_count - 1, dtype=torch.long)
    for base_source in range(source_count - 2, -1, -1):
        random = torch.rand(token_count, generator=generator)
        expected = torch.where(random < ratio, base_source, expected)

    actual = encoder_helpers.generate_spatial_fusion_mask(
        token_count,
        source_count,
        "spatial-dither-random",
        dither_ratio=ratio,
        seed=seed,
        grid_shape=(64, 64),
        dither_secondary_pattern="dither-random-reverse",
    )
    repeated = encoder_helpers.generate_spatial_fusion_mask(
        token_count,
        source_count,
        "spatial-dither-random",
        dither_ratio=ratio,
        seed=seed,
        grid_shape=(64, 64),
        dither_secondary_pattern="dither-random-reverse",
    )

    assert torch.equal(actual, expected)
    assert torch.equal(repeated, actual)
    assert torch.bincount(actual, minlength=source_count).gt(0).all()


def test_forward_dither_random_recursively_accumulates_sources():
    token_count = 4096
    source_count = 8
    ratio = 0.5
    seed = 97
    generator = torch.Generator().manual_seed(seed)
    expected = torch.zeros(token_count, dtype=torch.long)
    for secondary_source in range(1, source_count):
        random = torch.rand(token_count, generator=generator)
        expected = torch.where(random < ratio, expected, secondary_source)

    actual = encoder_helpers.generate_spatial_fusion_mask(
        token_count,
        source_count,
        "spatial-dither-random",
        dither_ratio=ratio,
        seed=seed,
        grid_shape=(64, 64),
        dither_secondary_pattern="dither-random-forward",
    )

    assert torch.equal(actual, expected)
    assert torch.bincount(actual, minlength=source_count).gt(0).all()


@pytest.mark.parametrize(
    "pattern",
    ["dither-random-reverse", "dither-random-forward"],
)
@pytest.mark.parametrize("ratio, expected_source", [(0.0, 7), (1.0, 0)])
def test_iterative_dither_random_endpoints(pattern, ratio, expected_source):
    mask = encoder_helpers.generate_spatial_fusion_mask(
        64,
        8,
        "spatial-dither-random",
        dither_ratio=ratio,
        seed=19,
        grid_shape=(8, 8),
        dither_secondary_pattern=pattern,
    )

    assert mask.eq(expected_source).all()


def test_advanced_visual_encoder_uses_one_base_resolution_pass_per_source(
    monkeypatch,
):
    prepared_resolutions = []
    encoded_images = []
    encoded_prompts = []

    def prepare(image, resolution):
        prepared_resolutions.append(resolution)
        return image

    def encode(_clip, prompt, images, **_kwargs):
        encoded_prompts.append(prompt)
        encoded_images.append(images[0])
        return [[torch.ones(1, 6, 2), {}]]

    class Clip:
        @staticmethod
        def tokenize(*_args, **_kwargs):
            return {"qwen": [[(0, 1.0)]]}

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", prepare)
    monkeypatch.setattr(
        encoder_nodes, "encode_embedding_classical_scaled_bias", encode
    )
    monkeypatch.setattr(
        encoder_nodes, "find_visual_token_range", lambda *_args, **_kwargs: (1, 5)
    )
    monkeypatch.setattr(
        encoder_nodes, "visual_fusion_grid", lambda *_args, **_kwargs: (2, 2)
    )
    monkeypatch.setattr(
        encoder_nodes,
        "evaluate_conditioning_consensus_blend",
        lambda sequences, pooled, **_kwargs: (sequences["a"], pooled.get("a")),
    )

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={
            "image_1": torch.zeros(1, 2, 2, 3),
            "image_2": torch.ones(1, 2, 2, 3),
        },
        visual_fusion_config={
            **_config(method="spatial-checkerboard"),
            "text_blend_config": {"blend_preset": "high_diversity_concept"},
            "fusion_strength": 0.0,
            "resolution_samples": 15,
        },
        semantic_anchor=True,
    )

    assert prepared_resolutions[:2] == [384, 384]
    assert len(encoded_images) == 2
    assert all(
        f"<Picture 1>: {encoder_nodes.VISION_BLOCK}" in prompt
        for prompt in encoded_prompts
    )


def test_advanced_visual_encoder_does_not_duplicate_klein_vision_block(
    monkeypatch,
):
    encoded_prompts = []
    encoded_kwargs = []

    def encode(_clip, prompt, **kwargs):
        encoded_prompts.append(prompt)
        encoded_kwargs.append(kwargs)
        return [[torch.ones(1, 512, 2), {}]]

    class Clip:
        cond_stage_model = types.SimpleNamespace(clip_name="qwen3_4b")

    monkeypatch.setattr(
        encoder_nodes, "encode_embedding_classical_scaled_bias", encode
    )
    monkeypatch.setattr(encoder_nodes, "is_klein_vl_text_encoder", lambda _clip: True)

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.zeros(1, 2, 2, 3)},
        visual_fusion_config=_config(method="off"),
    )

    assert len(encoded_prompts) == 1
    assert encoded_prompts[0] == ""
    assert "skip_template" not in encoded_kwargs[0]


def test_advanced_visual_encoder_spatial_output_contains_every_source(monkeypatch):
    def encode(_clip, _prompt, images, **_kwargs):
        value = float(images[0].flatten()[0])
        tensor = torch.tensor(
            [[[-1.0], [value], [value], [value], [value], [-2.0]]]
        )
        return [[tensor, {}]]

    class Clip:
        @staticmethod
        def tokenize(*_args, **_kwargs):
            return {"qwen": [[(0, 1.0)]]}

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)
    monkeypatch.setattr(encoder_nodes, "encode_embedding_classical_scaled_bias", encode)
    monkeypatch.setattr(
        encoder_nodes, "find_visual_token_range", lambda *_args, **_kwargs: (1, 5)
    )
    monkeypatch.setattr(
        encoder_nodes, "visual_fusion_grid", lambda *_args, **_kwargs: (2, 2)
    )
    monkeypatch.setattr(
        encoder_nodes.comfy.model_management,
        "get_torch_device",
        lambda: torch.device("cpu"),
    )

    output = UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={
            "image_1": torch.ones(1, 2, 2, 3),
            "image_2": torch.full((1, 2, 2, 3), 10.0),
        },
        visual_fusion_config={
            **_config(method="spatial-checkerboard"),
            "visual_block_size": 1,
        },
    ).result[0]

    assert set(output[0][0][:, 1:5, 0].flatten().tolist()) == {1.0, 10.0}


def test_advanced_visual_encoder_exports_default_unfused_visual_input(monkeypatch):
    encoded = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    exported = []

    class Clip:
        @staticmethod
        def tokenize(*_args, **_kwargs):
            return {"qwen3vl_4b": [[(0, 1.0)]]}

    monkeypatch.setattr(
        encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image
    )
    monkeypatch.setattr(
        encoder_nodes,
        "encode_embedding_classical_scaled_bias",
        lambda *_args, **_kwargs: [[encoded, {}]],
    )
    monkeypatch.setattr(
        encoder_nodes, "find_visual_token_range", lambda *_args, **_kwargs: (1, 5)
    )
    monkeypatch.setattr(
        encoder_nodes,
        "save_source_visual_embeddings",
        lambda clip, tokens, config, key, device: exported.append(
            (clip, tokens, key, device)
        ),
    )

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.zeros(1, 2, 2, 3)},
        visual_fusion_config={
            **_config(method="off"),
            "save_blended_embeds": True,
        },
    )

    assert len(exported) == 1
    _, tokens, key, device = exported[0]
    assert tokens == {"qwen3vl_4b": [[(0, 1.0)]]}
    assert key == "qwen3vl_4b"
    assert device is not None


def test_token_fusion_preprocesses_each_source_then_encodes_once(monkeypatch, caplog):
    calls = {"process": 0, "forward": 0}
    caplog.set_level("INFO")

    class Transformer:
        last_token_tags = None

        def __call__(self, _ids, _mask, embeds=None, embeds_info=None, **_kwargs):
            calls["forward"] += 1
            calls["embeds"] = embeds.clone()
            calls["info"] = embeds_info
            return embeds + 100.0, None, None

    class ClipModel:
        layer = "last"
        layer_idx = None
        layer_norm_hidden_state = False
        enable_attention_masks = False
        zero_out_masked = False
        return_projected_pooled = True
        return_attention_masks = False

        def __init__(self):
            self.transformer = Transformer()

        def process_tokens(self, rows, device):
            calls["process"] += 1
            value = float(next(item["data"] for item in rows[0] if isinstance(item, dict)))
            embeds = torch.tensor([[[10.0], [value], [value], [value], [value], [20.0], [30.0]]], device=device)
            info = [{
                "type": "image",
                "index": 1,
                "size": 4,
                "extra": {
                    "grid": torch.tensor([1, 4, 4]),
                    "deepstack": [torch.full((4, 1), value * 10.0, device=device)],
                },
            }]
            return embeds, torch.ones((1, 7), dtype=torch.long, device=device), [7], info

    model = ClipModel()

    class Stage:
        clip = "qwen"
        qwen = model

        @staticmethod
        def reset_clip_options():
            pass

        @staticmethod
        def set_clip_options(_options):
            pass

    class Patcher:
        forced_hooks = None
        load_device = torch.device("cpu")

    class Clip:
        cond_stage_model = Stage()
        patcher = Patcher()
        layer_idx = None
        use_clip_schedule = False

        @staticmethod
        def load_model(_tokens):
            pass

        @staticmethod
        def add_hooks_to_dict(_metadata):
            pass

    def tokens(value):
        return {"qwen": [[(151652, 1.0), ({"type": "image", "data": value}, 1.0), (151653, 1.0), (42, 1.0)]]}

    output = encoder_helpers.encode_token_fused_visual_sources(
        Clip(),
        [tokens(1.0), tokens(10.0)],
        {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 1},
    )

    assert calls["process"] == 2
    assert calls["forward"] == 1
    assert set(calls["embeds"][0, 1:5, 0].tolist()) == {1.0, 10.0}
    deepstack = calls["info"][0]["extra"]["deepstack"][0]
    assert set(deepstack[:, 0].tolist()) == {10.0, 100.0}
    assert torch.equal(output[0][0], calls["embeds"] + 100.0)
    assert "fused 2 visual sources into 1 block (4 visual tokens)" in caplog.text


def test_token_fusion_alternative_nodes_have_distinct_ids_and_matching_sockets():
    pairs = [
        (encoder_nodes.UC_AdvancedVisualConditioningEncodeTokenFusion, encoder_nodes.UC_AdvancedVisualConditioningEncode),
        (encoder_nodes.UC_Krea2TokenAttentionWeightTokenFusion, encoder_nodes.UC_Krea2TokenAttentionWeight),
        (encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTokenFusion, encoder_nodes.UC_AdvancedMiniMaxH3ImageToVideo),
    ]
    for alternative, original in pairs:
        alternative_schema = alternative.define_schema()
        original_schema = original.define_schema()
        assert "TokenFusion" in alternative_schema.display_name
        assert alternative_schema.node_id != original_schema.node_id
        assert [value.id for value in alternative_schema.inputs] == [value.id for value in original_schema.inputs]
        assert [value.id for value in alternative_schema.outputs] == [value.id for value in original_schema.outputs]


def test_token_fusion_minimax_slots_share_one_final_encode(monkeypatch, caplog):
    calls = {"process": 0, "encode": 0}
    caplog.set_level("INFO")

    class ClipModel:
        transformer = object()

        @staticmethod
        def process_tokens(rows, device):
            calls["process"] += 1
            values = [float(item["data"]) for item in rows[0] if isinstance(item, dict)]
            chunks = [torch.tensor([[10.0]], device=device)]
            info = []
            index = 1
            for value in values:
                chunks.extend([
                    torch.full((4, 1), value, device=device),
                    torch.tensor([[20.0]], device=device),
                ])
                info.append({
                    "type": "image", "index": index, "size": 4,
                    "extra": {
                        "grid": torch.tensor([1, 4, 4]),
                        "deepstack": [torch.full((4, 1), value * 10.0, device=device)],
                    },
                })
                index += 5
            chunks.append(torch.tensor([[30.0]], device=device))
            embeds = torch.cat(chunks)[None]
            return embeds, torch.ones((1, embeds.shape[1]), dtype=torch.long), [embeds.shape[1]], info

    model = ClipModel()

    class Stage:
        clip = "qwen"
        qwen = model

        @staticmethod
        def reset_clip_options():
            pass

        @staticmethod
        def set_clip_options(_options):
            pass

    class Patcher:
        forced_hooks = None
        load_device = torch.device("cpu")

    class Clip:
        cond_stage_model = Stage()
        patcher = Patcher()
        layer_idx = None
        use_clip_schedule = False

        @staticmethod
        def load_model(_tokens):
            pass

        @staticmethod
        def add_hooks_to_dict(_metadata):
            pass

    def tokens(first, second):
        return {"qwen3vl_32b": [[
            (151652, 1.0), ({"type": "image", "data": first}, 1.0), (151653, 1.0),
            (151652, 1.0), ({"type": "image", "data": second}, 1.0), (151653, 1.0),
            (42, 1.0),
        ]]}

    def encode(_model, embeds, _mask, _counts, info):
        calls["encode"] += 1
        calls["embeds"] = embeds
        calls["info"] = info
        return embeds, {"minimax_token_tags": torch.zeros(embeds.shape[1])}

    monkeypatch.setattr(encoder_helpers, "_encode_preprocessed_clip_model", encode)
    output = encoder_helpers.encode_token_fused_visual_slots(
        Clip(),
        tokens(1.0, 2.0),
        [(0, [tokens(10.0, 2.0)]), (1, [tokens(1.0, 20.0)])],
        {"visual_fusion_method": "spatial-checkerboard", "visual_block_size": 1},
    )

    assert calls["process"] == 3
    assert calls["encode"] == 1
    assert set(calls["embeds"][0, 1:5, 0].tolist()) == {1.0, 10.0}
    assert set(calls["embeds"][0, 6:10, 0].tolist()) == {2.0, 20.0}
    assert output[0][1]["minimax_token_tags"].numel() == calls["embeds"].shape[1]
    assert "Picture 1: 2 sources -> 4 tokens, Picture 2: 2 sources -> 4 tokens" in caplog.text


def test_token_fusion_minimax_rebuilds_stale_modality_tags():
    class Transformer:
        model_type = "qwen3vl_32b"
        last_token_tags = torch.ones(6, dtype=torch.long)

        def __call__(self, *_args, **_kwargs):
            return torch.zeros(1, 5, 2), None

    class ClipModel:
        transformer = Transformer()
        enable_attention_masks = False
        layer = "last"
        layer_idx = None
        layer_norm_hidden_state = False
        zero_out_masked = False
        return_projected_pooled = True
        return_attention_masks = False

    conditioning, metadata = encoder_helpers._encode_preprocessed_clip_model(
        ClipModel(),
        torch.zeros(1, 6, 2),
        torch.ones(1, 6, dtype=torch.long),
        [6],
        [{"type": "image", "index": 1, "size": 2}],
    )

    assert conditioning.shape[1] == 5
    assert metadata["minimax_token_tags"].tolist() == [0, 0, 0, 0, 1]


def test_token_fusion_applies_krea2_outer_shape_and_attention_mask_contract():
    tokens = {"qwen3vl_4b": [[
        (151644, 1.0), (1, 1.0), (151644, 1.0),
        (872, 1.0), (198, 1.0), (42, 1.0), (43, 1.0),
    ]]}
    tapped_layers = torch.arange(1 * 12 * 7 * 2, dtype=torch.float32).reshape(1, 12, 7, 2)
    metadata = {"attention_mask": torch.ones((1, 7), dtype=torch.long)}

    conditioning, normalized = encoder_helpers._normalize_token_fused_conditioning(
        object(), tokens, tapped_layers, metadata
    )

    expected = tapped_layers[:, :, 5:].permute(0, 2, 1, 3).reshape(1, 2, 24)
    assert torch.equal(conditioning, expected)
    assert conditioning.ndim == 3
    assert "attention_mask" not in normalized


def test_token_fusion_keeps_nontrivial_krea2_attention_mask_after_template_slice():
    tokens = {"qwen3vl_4b": [[
        (151644, 1.0), (1, 1.0), (151644, 1.0),
        (872, 1.0), (198, 1.0), (42, 1.0), (43, 1.0),
    ]]}
    tapped_layers = torch.zeros((1, 12, 7, 2))
    metadata = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]])}

    _, normalized = encoder_helpers._normalize_token_fused_conditioning(
        object(), tokens, tapped_layers, metadata
    )

    assert torch.equal(normalized["attention_mask"], torch.tensor([[1, 0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cached_cuda_mask_matches_cpu_raw_fusion():
    config = _config(seed=47)
    cache = {}
    gpu = encoder_helpers.fuse_visual_token_sources(
        [torch.zeros(64, 1, device="cuda"), torch.ones(64, 1, device="cuda")],
        config,
        "cuda",
        cache, source_grids=[(8, 8)] * 2,
    )
    cpu = encoder_helpers.fuse_visual_token_sources([torch.zeros(64, 1), torch.ones(64, 1)], config, "cuda", cache, source_grids=[(8, 8)] * 2)

    assert torch.equal(gpu.cpu(), cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cleanup_mask_runs_on_cuda_without_cudnn_integer_convolution():
    mask = encoder_helpers.generate_spatial_fusion_mask(
        64, 3, "spatial-dither-random", dither_ratio=0.5, device="cuda", seed=5,
        grid_shape=(8, 8), dither_mask_cleanup=True,
    )
    assert mask.device.type == "cuda"
    assert mask.shape == (64,)


def test_equal_area_landscape_and_portrait_keep_canonical_orientation():
    landscape = torch.arange(6.0).reshape(2, 3, 1).flatten(0, 1)
    portrait = torch.arange(100.0, 106.0).reshape(3, 2, 1).flatten(0, 1)
    output = encoder_helpers.fuse_visual_token_sources(
        [landscape, portrait], _config(ratio=0.0), "cpu", source_grids=[(2, 3), (3, 2)]
    )
    expected = torch.nn.functional.interpolate(portrait.reshape(3, 2, 1).permute(2, 0, 1)[None], size=(2, 3), mode="nearest")[0].permute(1, 2, 0).flatten(0, 1)
    assert output.shape == (6, 1)
    assert torch.equal(output, expected)


def test_block_secondary_cleanup_endpoints_and_cache_separation():
    block = encoder_helpers.generate_spatial_fusion_mask(16, 4, "spatial-dither-random", 2, 0.0, grid_shape=(4, 4), dither_secondary_pattern="block-interleave")
    assert block.reshape(4, 4).tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 3, 3], [2, 2, 3, 3]]
    for ratio, expected in [(0.0, False), (1.0, True)]:
        cleaned = encoder_helpers.generate_spatial_fusion_mask(9, 3, "spatial-dither-random", dither_ratio=ratio, grid_shape=(3, 3), dither_mask_cleanup=True)
        assert cleaned.eq(0).all().item() is expected
    cache = {}
    sources = [torch.zeros(16, 1), torch.ones(16, 1)]
    encoder_helpers.fuse_visual_token_sources(sources, _config(seed=1), "cpu", cache, source_grids=[(4, 4)] * 2)
    changed = {**_config(seed=1), "dither_mask_cleanup": True}
    encoder_helpers.fuse_visual_token_sources(sources, changed, "cpu", cache, source_grids=[(4, 4)] * 2)
    perturbed = {**_config(seed=1), "spatial_perturbation": 0.5}
    encoder_helpers.fuse_visual_token_sources(sources, perturbed, "cpu", cache, source_grids=[(4, 4)] * 2)
    assert len(cache) == 3


@pytest.mark.parametrize("method", ["spatial-checkerboard", "spatial-block-interleave", "spatial-dither-random"])
def test_spatial_perturbation_is_seeded_and_preserves_source_counts(method):
    kwargs = {"grid_shape": (8, 8), "seed": 19, "spatial_perturbation": 0.5}
    base = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, grid_shape=(8, 8), seed=19)
    first = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, **kwargs)
    repeated = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, **kwargs)
    changed_seed = encoder_helpers.generate_spatial_fusion_mask(64, 3, method, **{**kwargs, "seed": 20})

    assert torch.equal(first, repeated)
    assert not torch.equal(first, base)
    assert not torch.equal(first, changed_seed)
    assert torch.equal(torch.bincount(first, minlength=3), torch.bincount(base, minlength=3))
    assert first.ne(base).sum().item() == 32


def test_spatial_perturbation_saturates_without_ratio_drift():
    mask = torch.tensor([0] * 9 + [1])
    changed = encoder_helpers._perturb_spatial_assignments(mask, 1.0, seed=7)

    assert changed.ne(mask).sum().item() == 2
    assert torch.equal(torch.bincount(changed), torch.bincount(mask))
    assert torch.equal(encoder_helpers._perturb_spatial_assignments(mask, 0.0, seed=7), mask)


def test_spatial_perturbation_selects_only_exact_source_tokens():
    config = {**_config(method="spatial-checkerboard", seed=29), "spatial_perturbation": 0.75}
    sources = [torch.full((16, 2), float(index), dtype=torch.float16) for index in range(3)]
    mask = encoder_helpers.generate_spatial_fusion_mask(
        16, 3, "spatial-checkerboard", seed=29, grid_shape=(4, 4), spatial_perturbation=0.75,
    )
    output = encoder_helpers.fuse_visual_token_sources(sources, config, "cpu", source_grids=[(4, 4)] * 3)

    assert output.dtype == torch.float16
    assert torch.equal(output[:, 0], mask.to(dtype=output.dtype))
    assert torch.equal(output[:, 0], output[:, 1])


def test_cleanup_swaps_only_complementary_pairs_and_preserves_each_source():
    mask = torch.ones((7, 7), dtype=torch.long)
    mask[0, 0] = 0
    mask[3:6, 3:6] = 0
    mask[4, 4] = 2
    before_counts = torch.bincount(mask.flatten(), minlength=3)

    cleaned = encoder_helpers._cleanup_primary_pairs(mask)

    assert cleaned[0, 0].item() == 2
    assert cleaned[4, 4].item() == 0
    assert torch.equal(torch.bincount(cleaned.flatten(), minlength=3), before_counts)
    assert cleaned.ne(mask).sum().item() == 2


def test_cleanup_leaves_unpaired_sparse_islands_unchanged():
    sparse = torch.ones((8, 8), dtype=torch.long)
    sparse[1, 1] = 0
    sparse[6, 6] = 0

    assert torch.equal(encoder_helpers._cleanup_primary_pairs(sparse), sparse)


def test_combined_perturbation_and_cleanup_is_deterministic_and_balanced():
    kwargs = {
        "grid_shape": (12, 12),
        "seed": 37,
        "dither_ratio": 0.25,
        "spatial_perturbation": 0.4,
    }
    before = encoder_helpers.generate_spatial_fusion_mask(144, 4, "spatial-dither-random", **kwargs)
    cleaned = encoder_helpers.generate_spatial_fusion_mask(
        144, 4, "spatial-dither-random", dither_mask_cleanup=True, **kwargs,
    )
    repeated = encoder_helpers.generate_spatial_fusion_mask(
        144, 4, "spatial-dither-random", dither_mask_cleanup=True, **kwargs,
    )

    assert torch.equal(cleaned, repeated)
    assert torch.equal(torch.bincount(cleaned, minlength=4), torch.bincount(before, minlength=4))


@pytest.mark.parametrize("amount", [-0.01, 1.01])
def test_spatial_perturbation_rejects_invalid_amount(amount):
    with pytest.raises(ValueError, match="Spatial perturbation"):
        encoder_helpers.generate_spatial_fusion_mask(
            4, 2, "spatial-checkerboard", grid_shape=(2, 2), spatial_perturbation=amount,
        )


def test_missing_and_malformed_grids_rejected():
    sources = [torch.zeros(4, 1)]
    with pytest.raises(ValueError, match="explicit grid"):
        encoder_helpers.fuse_visual_token_sources(sources, _config(), "cpu")
    with pytest.raises(ValueError, match="inconsistent"):
        encoder_helpers.fuse_visual_token_sources(sources, _config(), "cpu", source_grids=[(1, 3)])
