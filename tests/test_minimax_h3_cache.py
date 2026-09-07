"""Focused tests for persisted post-Qwen H3 sections."""

import pathlib
import sys
import types

import pytest
import torch
from comfy.cli_args import args


PACKAGE = "utils_collection_cache_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(pathlib.Path(__file__).parents[1])]
sys.modules.setdefault(PACKAGE, package)
prior_cpu = args.cpu
args.cpu = True
try:
    from utils_collection_cache_test import encoder_helpers as encoder
    from utils_collection_cache_test import encoder_nodes as nodes
    from utils_collection_cache_test import minimax_h3_cache_helpers as cache_module
finally:
    args.cpu = prior_cpu


@pytest.fixture(autouse=True)
def cache_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_module.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    monkeypatch.setattr(encoder.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))


class _Patcher:
    forced_hooks = None
    patches_uuid = "patches"
    object_patches = {}
    model_options = {}
    load_device = torch.device("cpu")


class _Clip:
    def __init__(self, schedules=None):
        self.patcher = _Patcher()
        self.patcher.forced_hooks = schedules
        self.use_clip_schedule = schedules is not None
        self.hooks_added = []

    def add_hooks_to_dict(self, metadata):
        self.hooks_added.append(metadata)


def test_only_encoded_sections_and_vae_outputs_are_persisted():
    assert cache_module._STAGES == frozenset(("encoded_section", "vae_encode"))
    with cache_module.H3EncoderCache() as cache:
        with pytest.raises(ValueError, match="Unknown H3 cache stage"):
            cache.get_or_compute("vision", {}, lambda: torch.ones(1))
        with pytest.raises(ValueError, match="Unknown H3 cache stage"):
            cache.get_or_compute("text_tokens", {}, lambda: [1])


def test_prepare_clip_does_not_clone_or_patch_raw_vision():
    clip = object()
    with cache_module.H3EncoderCache("disabled") as cache:
        assert cache.prepare_clip(clip) is clip


def test_each_scheduled_qwen_result_has_its_own_cache_file():
    schedules = types.SimpleNamespace(get_hooks_for_clip_schedule=lambda: [((0.0, 0.5), []), ((0.5, 1.0), [])])
    clip = _Clip(schedules)
    calls = []

    def compute():
        calls.append(True)
        return [
            [torch.full((1, 2, 4), 1.0), {"clip_start_percent": 0.0, "clip_end_percent": 0.5}],
            [torch.full((1, 2, 4), 2.0), {"clip_start_percent": 0.5, "clip_end_percent": 1.0}],
        ]

    with cache_module.H3EncoderCache() as cache:
        cache._clip_identity = {"model": "test"}
        first = cache.encode_scheduled(clip, {"qwen3vl_32b": [[(1, 1.0)]]}, "grid-deepstack", compute, section_kind="text", section_id="prompt")
        second = cache.encode_scheduled(clip, {"qwen3vl_32b": [[(1, 1.0)]]}, "grid-deepstack", compute, section_kind="text", section_id="prompt")
        assert len(calls) == 1
        assert len(list((cache.root / "encoded_section").glob("*.safetensors"))) == 2
        torch.testing.assert_close(first[0][0], second[0][0])
        torch.testing.assert_close(first[1][0], second[1][0])


def test_preprocessed_qwen_result_is_not_persisted():
    embeds = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    info = [{"type": "image", "index": 0, "size": 2, "extra": {"deepstack": [torch.ones(2, 4)]}}]
    model = types.SimpleNamespace(layer="last", layer_idx=None, enable_attention_masks=False,
                                  layer_norm_hidden_state=False, zero_out_masked=False,
                                  return_projected_pooled=True, return_attention_masks=False)
    calls = []
    with cache_module.H3EncoderCache() as cache:
        cache._clip_identity = {"model": "test"}
        first = cache.encode_preprocessed(
            model, embeds, torch.ones((1, 3), dtype=torch.long), [3], info, "grid-deepstack", None,
            lambda: (calls.append(True) or (embeds, {"minimax_token_tags": torch.zeros(3, dtype=torch.long)})),
        )
        second = cache.encode_preprocessed(
            model, embeds, torch.ones((1, 3), dtype=torch.long), [3], info, "grid-deepstack", None,
            lambda: (calls.append(True) or (embeds, {"minimax_token_tags": torch.zeros(3, dtype=torch.long)})),
        )
        assert len(calls) == 2
        torch.testing.assert_close(first[0], second[0])
        assert not (cache.root / "encoded_section").exists()


@pytest.mark.parametrize(
    ("mode", "kind", "cached"),
    [
        ("disabled", "text", False), ("images_only", "text", True),
        ("video_only", "text", True), ("images_only", "image", True),
        ("video_only", "image", False), ("video_only", "video", True),
        ("images_only", "video", False), ("images_only", "regular_fusion", True),
        ("video_only", "regular_temporal", True), ("video_only", "guide", False),
        ("images_only", "guide", True), ("all", "video", True), ("all", "joint", False),
    ],
)
def test_encoded_section_selection_follows_mode(mode, kind, cached):
    with cache_module.H3EncoderCache(mode) as cache:
        assert cache.allows_encoded_section(kind) is cached


@pytest.mark.parametrize(
    ("mode", "media", "cached"),
    [
        ("disabled", "image", False), ("video_only", "image", False),
        ("images_only", "image", True), ("all", "image", True),
        ("images_only", "video", False), ("video_only", "video", True),
        ("all", "video", True), ("all", "audio", True),
    ],
)
def test_vae_cache_selection_follows_media_mode(mode, media, cached):
    calls = []
    samples = torch.ones(1, 2, 2, 3)

    class VAE:
        def encode(self, value):
            calls.append(value)
            return torch.ones(1, 4, 1, 1)

    vae = VAE()
    with cache_module.H3EncoderCache(mode) as cache:
        first = cache.encode_vae(vae, samples, media)
        second = cache.encode_vae(vae, samples, media)
        assert len(calls) == (1 if cached else 2)
        torch.testing.assert_close(first, second)


def test_disabled_mode_never_builds_cache_identity_or_directory(monkeypatch):
    monkeypatch.setattr(cache_module, "clip_description", lambda value: pytest.fail("unexpected cache identity"))
    with cache_module.H3EncoderCache("disabled") as cache:
        assert cache.prepare_clip(object()) is not None
        assert cache.get_or_compute("encoded_section", {}, lambda: [1]) == [1]
        assert not cache.root.exists()


@pytest.mark.parametrize("node_type", [
    nodes.UC_AdvancedMiniMaxH3ImageToVideo,
    nodes.UC_AdvMiniMaxH3ImageToVideoTokenFusion,
    nodes.UC_AdvMiniMaxH3ImageToVideoTemporalFusion,
    nodes.UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion,
    nodes.UC_MiniMaxH3VLMGuide,
])
def test_h3_nodes_expose_the_cache_mode_combo(node_type):
    combo = next(value for value in node_type.define_schema().inputs if value.id == "enable_caching")
    assert list(combo.options) == ["disabled", "video_only", "images_only", "all"]
    assert combo.default == "all"
