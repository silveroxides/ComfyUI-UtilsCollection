"""Expected source pairs independent of model execution and encoder imports."""

import pathlib
import sys
import types
from contextlib import nullcontext

import pytest
import torch


PACKAGE_NAME = "utils_collection_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(pathlib.Path(__file__).parents[1])]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_test.minimax_h3_temporal_helpers import (
    fuse_temporal_block,
    encode_temporal_conditioning,
    minimax_h3_temporal_frame_pairs,
)


@pytest.mark.parametrize(
    "frame_count, indices, expected",
    [
        (0, [], []),
        (1, [0], [[(0, 0)]]),
        (25, [0, 12, 24], [[(0, 12)], [(24, 24)]]),
        # Configured 5 FPS rounds 24 FPS source positions to nearest frames.
        (25, [0, 5, 10, 14, 19, 24], [[(0, 5)], [(10, 14)], [(19, 24)]]),
        # Configured 24 FPS preserves all source frames, including odd tail.
        (5, [0, 1, 2, 3, 4], [[(0, 1)], [(2, 3)], [(4, 4)]]),
    ],
)
def test_density_one_exact_canonical_pairs(frame_count, indices, expected):
    assert minimax_h3_temporal_frame_pairs(frame_count, indices) == expected


def test_native_two_fps_lanes_and_exhausted_odd_tail():
    assert minimax_h3_temporal_frame_pairs(25, [0, 12, 24], 3) == [
        [(0, 12), (4, 16), (8, 20)],
        [(24, 24)],
    ]


def test_configured_five_fps_uses_each_actual_sample_interval():
    assert minimax_h3_temporal_frame_pairs(25, [0, 5, 10, 14, 19, 24], 2) == [
        [(0, 5), (3, 8)],
        [(10, 14), (12, 17)],
        [(19, 24)],
    ]


def test_short_native_video_samples_tail_and_repeats_each_frame():
    assert minimax_h3_temporal_frame_pairs(4, [0], 4) == [
        [(0, 0), (1, 1), (2, 2), (3, 3)],
    ]


def test_odd_tail_uses_remaining_source_interval():
    assert minimax_h3_temporal_frame_pairs(29, [0, 12, 24], 2) == [
        [(0, 12), (6, 18)],
        [(24, 24), (27, 27)],
    ]


def test_high_density_deduplicates_and_omits_interval_endpoints():
    assert minimax_h3_temporal_frame_pairs(4, [0, 2], 24) == [
        [(0, 2), (1, 3)],
    ]
    assert minimax_h3_temporal_frame_pairs(1, [0], 24) == [[(0, 0)]]
    assert minimax_h3_temporal_frame_pairs(3, [0, 1, 2], 24) == [
        [(0, 1)], [(2, 2)],
    ]


@pytest.mark.parametrize("density", range(1, 25))
def test_density_never_expands_blocks_or_crosses_sample_intervals(density):
    indices = [0, 5, 10, 14, 19, 24]
    blocks = minimax_h3_temporal_frame_pairs(29, indices, density)
    assert len(blocks) == 3
    assert blocks == minimax_h3_temporal_frame_pairs(29, indices, density)
    for pairs, first, second, stop in zip(blocks, [0, 10, 19], [5, 14, 24], [10, 19, 29]):
        assert pairs[0] == (first, second)
        assert len(pairs) == len(set(pairs))
        assert len(pairs) <= density
        assert all(first <= left < second <= right < stop for left, right in pairs)


@pytest.mark.parametrize("density", [0, 25, -1, 1.5, True, None])
def test_invalid_density_rejected(density):
    with pytest.raises(ValueError, match="density"):
        minimax_h3_temporal_frame_pairs(1, [0], density)


@pytest.mark.parametrize("frame_count", [-1, 1.5, True])
def test_invalid_source_count_rejected(frame_count):
    with pytest.raises(ValueError, match="frame count"):
        minimax_h3_temporal_frame_pairs(frame_count, [])


@pytest.mark.parametrize("indices", [[-1], [3], [0, 0], [2, 1], [0.5], [True]])
def test_invalid_canonical_indices_rejected(indices):
    with pytest.raises(ValueError, match="canonical indices"):
        minimax_h3_temporal_frame_pairs(3, indices)


def test_empty_canonical_sampling_has_no_blocks_at_any_density():
    assert minimax_h3_temporal_frame_pairs(0, [], 24) == []
    assert minimax_h3_temporal_frame_pairs(10, [], 24) == []


@pytest.fixture(scope="module")
def encoder_callbacks():
    from comfy.cli_args import args
    previous = args.cpu
    args.cpu = True
    try:
        from utils_collection_test import encoder_helpers
    finally:
        args.cpu = previous
    return encoder_helpers


def _consensus_config(**overrides):
    return dict({
        "blend_preset": "custom", "blend_method": "consensus",
        "alignment_method": "index", "consensus_type": "median",
        "power_alpha": 2.0, "diversity_beta": 0.0,
        "rescale_norm": True, "global_scale": 1.0,
    }, **overrides)


@pytest.mark.parametrize("alignment", ["index", "similarity"])
@pytest.mark.parametrize("overrides", [
    {}, {"consensus_type": "mean"}, {"blend_method": "linear", "global_scale": 1.7},
    {"rescale_norm": False, "global_scale": 0.8},
    {"similarity_threshold": 1.1}, {"alignment_threshold": 1.1},
    {"position_weight": 0.6},
    {"dynamic_similarity_contrast": True, "soft_comfort_bandpass": True,
     "diversity_beta": 2.0, "power_alpha": 1.5},
    {"preserve_common_prefix": True, "global_scale": 2.0},
])
def test_consensus_matches_existing_engine(encoder_callbacks, alignment, overrides):
    generator = torch.Generator().manual_seed(42)
    sources = [torch.randn(5, 6, generator=generator) for _ in range(3)]
    for source in sources[1:]:
        source[0] = sources[0][0]
    config = _consensus_config(alignment_method=alignment, **overrides)
    settings = encoder_callbacks.resolve_consensus_blend_settings(config)
    actual, deepstack = fuse_temporal_block(
        sources, "consensus", settings,
        position_score_callback=encoder_callbacks._position_biased_similarity_scores,
    )
    expected, _ = encoder_callbacks.blend_text_vectors(
        {i: source.unsqueeze(0) for i, source in enumerate(sources)}, config,
    )
    torch.testing.assert_close(actual, expected[0])
    assert actual.shape == sources[0].shape
    assert deepstack is None


def test_all_consensus_presets_match_existing_engine(encoder_callbacks):
    sources = [torch.tensor([[1., 2., 3.], [3., 2., 1.]]),
               torch.tensor([[2., 1., 3.], [3., 1., 2.]])]
    for preset in encoder_callbacks.CONSENSUS_BLEND_PRESETS:
        config = {"blend_preset": preset}
        actual, _ = fuse_temporal_block(
            sources, "consensus", encoder_callbacks.resolve_consensus_blend_settings(config),
            position_score_callback=encoder_callbacks._position_biased_similarity_scores,
        )
        expected, _ = encoder_callbacks.blend_text_vectors(
            {i: source.unsqueeze(0) for i, source in enumerate(sources)}, config,
        )
        torch.testing.assert_close(actual, expected[0], msg=preset)


def test_similarity_selection_and_weights_reused_on_deepstack(encoder_callbacks):
    # Primary source 1 is reversed. DeepStack values intentionally cannot imply
    # that matching: expected values require the primary permutation and weights.
    sources = [torch.eye(2), torch.eye(2).flip(0)]
    layers = [[torch.tensor([[10.], [20.]]), torch.tensor([[40.], [30.]])]]
    settings = encoder_callbacks.resolve_consensus_blend_settings(
        _consensus_config(alignment_method="similarity", rescale_norm=False),
    )
    primary, deepstack = fuse_temporal_block(sources, "consensus", settings, deepstack_layers=layers)
    torch.testing.assert_close(primary, torch.eye(2))
    torch.testing.assert_close(deepstack[0], torch.tensor([[20.], [30.]]))
    torch.testing.assert_close(layers[0][0], torch.tensor([[10.], [20.]]))


def test_primary_rejection_weights_are_not_recomputed_on_deepstack(encoder_callbacks):
    # Coordinate median [0, 0] gives similarities zero; threshold forces uniform
    # fallback. DeepStack vectors would produce very different cosine weights.
    sources = [torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])]
    layers = [[torch.tensor([[9., 0.]]), torch.tensor([[0., 1.]])]]
    settings = encoder_callbacks.resolve_consensus_blend_settings(
        _consensus_config(rescale_norm=False, similarity_threshold=0.8),
    )
    primary, deepstack = fuse_temporal_block(sources, "consensus", settings, deepstack_layers=layers)
    torch.testing.assert_close(primary, torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(deepstack[0], torch.tensor([[4.5, 0.5]]))


def test_deepstack_norm_uses_its_own_selected_features(encoder_callbacks):
    settings = encoder_callbacks.resolve_consensus_blend_settings(_consensus_config())
    _, deepstack = fuse_temporal_block(
        [torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])], "consensus", settings,
        deepstack_layers=[[torch.tensor([[6., 0.]]), torch.tensor([[0., 2.]])]],
    )
    torch.testing.assert_close(deepstack[0].norm(dim=-1), torch.tensor([4.]))


def test_common_prefix_uses_primary_decision_for_every_layer(encoder_callbacks):
    settings = encoder_callbacks.resolve_consensus_blend_settings(
        _consensus_config(preserve_common_prefix=True, global_scale=3.),
    )
    primary, deepstack = fuse_temporal_block(
        [torch.ones(2, 3), torch.ones(2, 3)], "consensus", settings,
        deepstack_layers=[[torch.ones(2, 1), torch.full((2, 1), 9.)]],
    )
    torch.testing.assert_close(primary, torch.ones(2, 3))
    torch.testing.assert_close(deepstack[0], torch.ones(2, 1))


@pytest.mark.parametrize("method", ["linear", "spatial-checkerboard", "spatial-block-interleave"])
def test_spatial_weights_reused_and_deterministic(encoder_callbacks, method):
    calls = []
    def spatial(*args, **kwargs):
        result = encoder_callbacks.fuse_visual_token_sources(*args, **kwargs)
        calls.append((kwargs, result))
        return result
    config = {"visual_fusion_method": method, "visual_block_size": 1}
    sources = [torch.ones(4, 3), torch.full((4, 3), 3.)]
    layers = [[torch.ones(4, 2), torch.full((4, 2), 3.)]]
    actual, deepstack = fuse_temporal_block(
        sources, "spatial", visual_config=config, grids=[(2, 2), (2, 2)],
        spatial_fuse_callback=spatial, deepstack_layers=layers,
    )
    assert calls[1][0]["weights_override"] is calls[0][1][1]
    torch.testing.assert_close(actual[:, :2], deepstack[0])
    repeated, _ = fuse_temporal_block(
        sources, "spatial", visual_config=config, grids=[(2, 2), (2, 2)],
        spatial_fuse_callback=spatial,
    )
    torch.testing.assert_close(actual, repeated)


def test_off_and_single_source_are_identity_without_callbacks(encoder_callbacks):
    first, second = torch.ones(2, 3), torch.zeros(2, 3)
    settings = encoder_callbacks.resolve_consensus_blend_settings({"blend_preset": "off"})
    primary, layers = fuse_temporal_block(
        [first, second], "consensus", settings, deepstack_layers=[[first, second]],
    )
    assert primary is first and layers[0] is first
    assert fuse_temporal_block([first], "spatial")[0] is first


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_consensus_preserves_dtype_and_device(encoder_callbacks, dtype):
    sources = [torch.eye(2, dtype=dtype), torch.eye(2, dtype=dtype) * 2]
    config = _consensus_config()
    actual, _ = fuse_temporal_block(
        sources, "consensus", encoder_callbacks.resolve_consensus_blend_settings(config),
    )
    expected, _ = encoder_callbacks.blend_text_vectors(
        {i: source.unsqueeze(0) for i, source in enumerate(sources)}, config,
    )
    assert actual.dtype == dtype and actual.device == sources[0].device
    torch.testing.assert_close(actual, expected[0])


def test_incompatible_temporal_shapes_fail_before_callback():
    with pytest.raises(ValueError, match="match shape"):
        fuse_temporal_block([torch.ones(2, 3), torch.ones(3, 3)], "spatial")
    with pytest.raises(ValueError, match="source count"):
        fuse_temporal_block([torch.ones(2, 3)], "spatial",
                            deepstack_layers=[[torch.ones(2, 1), torch.ones(2, 1)]])
    with pytest.raises(ValueError, match="token count"):
        fuse_temporal_block([torch.ones(2, 3)], "spatial", deepstack_layers=[[torch.ones(3, 1)]])


def _lane_fixture():
    def image(value, video=False):
        return {"type": "image", "data": torch.full((2, 32, 64, 3), float(value)),
                "minimax_video_block": video, "kept": "entry metadata"}
    row = [(71, 1.), (151652, 1.), (image(2), 1.), (151653, 1.), (72, 1.),
           (151652, 1.), (image(10, True), 0.8, 4), (151653, 1.),
           # No text between video wrappers: tag runs merge, mapping must not.
           (151652, 1.), (image(20, True), 1.), (151653, 1.), (73, 1.)]
    return {"qwen3vl_32b": [row]}, [[(0, 1), (1, 2), (2, 3)], [(4, 5)]]


def _fake_process(row):
    vectors, tags, info, spans = [], [], [], []
    for entry in row:
        value = entry[0]
        start = len(vectors)
        if isinstance(value, dict):
            pixel = float(value["data"][0, 0, 0, 0])
            vectors.extend([[pixel, pixel]] * 2)
            tags.extend([0, 0])
            info.append({"type": "image", "index": start, "size": 2,
                         "extra": {"grid": torch.tensor([[1, 2, 4]]),
                                   "deepstack": [torch.full((2, 1), pixel * 10)]}})
        else:
            vectors.append([float(value), float(value)])
            tags.append(0 if value in (151652, 151653) else 1)
        spans.append((start, len(vectors)))
    tensor = torch.tensor([vectors])
    return (tensor, torch.ones(1, len(vectors)), len(vectors), info), torch.tensor(tags), spans


def _mean_fusion(sources, grids, layers):
    assert all(grid == (1, 2) for grid in grids)
    return torch.stack(sources).mean(0), (None if layers is None else
                                        [torch.stack(layer).mean(0) for layer in layers])


def _prepare_test_pair(pair):
    return torch.full((2, 32, 64, 3), float(sum(pair)))


def _test_token_spans(row, tensor):
    processed, _, spans = _fake_process(row)
    assert processed[0].shape[1] == tensor.shape[1]
    return spans


def test_post_lane_encoding_preserves_metadata_and_all_nonvideo_slices():
    tokens, pairs = _lane_fixture()
    calls = []
    pooled = torch.tensor([42.])
    def encode(lane_tokens):
        calls.append(lane_tokens)
        processed, tags, _ = _fake_process(lane_tokens["qwen3vl_32b"][0])
        return [[processed[0], {"minimax_token_tags": tags, "pooled_output": pooled,
                                "latent_reference": "preserved"}]]
    result = encode_temporal_conditioning(
        None, tokens, pairs, _prepare_test_pair, token_fusion=False,
        fusion_callback=_mean_fusion, encode_tokens_callback=encode,
        video_grid_callback=lambda data, length: (1, length), token_spans_callback=_test_token_spans,
    )
    assert len(calls) == 3
    original, _, spans = _fake_process(tokens["qwen3vl_32b"][0])
    expected = original[0].clone()
    start, end = spans[6]
    expected[:, start:end] = 6.  # mean of canonical10, shifted3, shifted5
    torch.testing.assert_close(result[0][0], expected)
    assert result[0][1]["pooled_output"] is pooled
    assert result[0][1]["latent_reference"] == "preserved"
    for lane in calls[1:]:
        row = lane["qwen3vl_32b"][0]
        assert row[6][1:] == (0.8, 4)
        assert row[6][0]["kept"] == "entry metadata"
        assert row[9] is tokens["qwen3vl_32b"][0][9]  # unavailable alternative stays canonical
        assert all(row[i] is tokens["qwen3vl_32b"][0][i] for i in range(len(row)) if i != 6)
    torch.testing.assert_close(tokens["qwen3vl_32b"][0][6][0]["data"], torch.full((2, 32, 64, 3), 10.))


@pytest.mark.parametrize("mismatch", [None, "count", "boundary", "layout"])
def test_post_conditioning_schedules_are_preserved_or_rejected(mismatch):
    tokens, pairs = _lane_fixture()
    calls = 0
    def encode(lane_tokens):
        nonlocal calls
        calls += 1
        processed, tags, _ = _fake_process(lane_tokens["qwen3vl_32b"][0])
        output = [[processed[0].clone(), {"minimax_token_tags": tags,
                   "clip_start_percent": start, "clip_end_percent": end}]
                  for start, end in [(0., 0.5), (0.5, 1.)]]
        if calls > 1:
            if mismatch == "count":
                output.pop()
            elif mismatch == "boundary":
                output[0][1]["clip_end_percent"] = 0.4
            elif mismatch == "layout":
                output[0][0] = output[0][0].repeat(2, 1, 1)
        return output
    kwargs = dict(token_fusion=False, fusion_callback=_mean_fusion, encode_tokens_callback=encode,
                  video_grid_callback=lambda data, length: (1, length), token_spans_callback=_test_token_spans)
    if mismatch:
        with pytest.raises(ValueError, match="schedule|layout"):
            encode_temporal_conditioning(None, tokens, pairs, _prepare_test_pair, **kwargs)
    else:
        result = encode_temporal_conditioning(None, tokens, pairs, _prepare_test_pair, **kwargs)
        assert [(meta["clip_start_percent"], meta["clip_end_percent"]) for _, meta in result] == [(0., 0.5), (0.5, 1.)]


@pytest.mark.parametrize("scheduled", [False, True])
def test_pre_lane_processing_runs_one_qwen_per_schedule_and_fuses_deepstack(scheduled):
    tokens, pairs = _lane_fixture()
    processed_calls, qwen_calls, options, patches = [], [], [], []
    hooks = types.SimpleNamespace(get_hooks_for_clip_schedule=lambda: [((0., 0.4), []), ((0.4, 1.), [])],
                                  reset=lambda: patches.append("reset"))
    class Model:
        def process_tokens(self, rows, device):
            processed_calls.append(rows)
            return _fake_process([(value, 1.) for value in rows[0]])[0]
    model = Model()
    clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(reset_clip_options=lambda: options.append("reset"),
                                              set_clip_options=options.append),
        layer_idx=7, load_model=lambda value: options.append("load"),
        patcher=types.SimpleNamespace(load_device=torch.device("cpu"), forced_hooks=hooks if scheduled else None,
                                      patch_hooks=patches.append),
        use_clip_schedule=scheduled, add_hooks_to_dict=lambda meta: meta.update(hooks_added=True),
    )
    def preprocessed(clip_model, embeds, attention, num_tokens, info):
        assert clip_model is model
        qwen_calls.append((embeds.clone(), info))
        return embeds, {"preserved": True}
    def ordinary(_):
        pytest.fail("Active pre-fusion must not use ordinary Qwen lane encoding")
    result = encode_temporal_conditioning(
        clip, tokens, pairs, _prepare_test_pair, token_fusion=True,
        fusion_callback=_mean_fusion, encode_tokens_callback=ordinary,
        active_clip_model_callback=lambda value: model, encode_preprocessed_callback=preprocessed,
        visual_context_callback=nullcontext,
    )
    expected_schedules = 2 if scheduled else 1
    assert len(qwen_calls) == len(result) == expected_schedules
    assert len(processed_calls) == 3 * expected_schedules
    original, _, spans = _fake_process(tokens["qwen3vl_32b"][0])
    expected = original[0].clone()
    start, end = spans[6]
    expected[:, start:end] = 6.
    for tensor, info in qwen_calls:
        torch.testing.assert_close(tensor, expected)
        torch.testing.assert_close(info[1]["extra"]["deepstack"][0], torch.full((2, 1), 60.))
        torch.testing.assert_close(info[0]["extra"]["deepstack"][0], torch.full((2, 1), 20.))
        torch.testing.assert_close(info[2]["extra"]["deepstack"][0], torch.full((2, 1), 200.))
    assert options == ["reset", {"layer": 7}, "load", {"execution_device": torch.device("cpu")}]
    assert all(meta["hooks_added"] for _, meta in result)
    if scheduled:
        assert [(meta["clip_start_percent"], meta["clip_end_percent"]) for _, meta in result] == [(0., 0.4), (0.4, 1.)]
        assert patches[0] == patches[-1] == "reset"


@pytest.mark.parametrize("token_fusion", [False, True])
def test_no_alternatives_bypass_preprocessing_and_fusion(token_fusion):
    tokens, pairs = _lane_fixture()
    output = object()
    def forbidden(*args, **kwargs):
        pytest.fail("Bypass called alternative or model processing")
    result = encode_temporal_conditioning(
        None, tokens, [pair[:1] for pair in pairs], forbidden, token_fusion=token_fusion,
        fusion_callback=forbidden, encode_tokens_callback=lambda value: output,
        active_clip_model_callback=forbidden, encode_preprocessed_callback=forbidden,
    )
    assert result is output


def test_no_video_bypasses_even_without_preprocessing_callbacks():
    tokens = {"qwen3vl_32b": [[(42, 1.)]]}
    output = object()
    assert encode_temporal_conditioning(
        None, tokens, [], None, token_fusion=True,
        fusion_callback=None, encode_tokens_callback=lambda value: output,
    ) is output


@pytest.mark.parametrize("failure", ["empty", "shape"])
def test_post_rejects_empty_schedules_and_broadcastable_wrong_fusion_shape(failure):
    tokens, pairs = _lane_fixture()
    def encode(lane_tokens):
        if failure == "empty":
            return []
        processed, tags, _ = _fake_process(lane_tokens["qwen3vl_32b"][0])
        return [[processed[0], {"minimax_token_tags": tags}]]
    with pytest.raises(ValueError, match="schedules|shape"):
        encode_temporal_conditioning(
            None, tokens, pairs, _prepare_test_pair, token_fusion=False,
            fusion_callback=lambda sources, grids, layers: (torch.ones(1, 2), None),
            encode_tokens_callback=encode, video_grid_callback=lambda data, length: (1, length),
            token_spans_callback=_test_token_spans,
        )
