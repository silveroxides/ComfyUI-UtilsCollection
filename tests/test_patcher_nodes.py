import pathlib
import sys
import types

import pytest
import torch

import comfy.model_patcher


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_patcher_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_patcher_test import patcher_helpers, patcher_nodes


def test_unified_attention_schema_keeps_mode_settings_separate():
    schema = patcher_nodes.UC_UnifiedAttentionPatcher.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    attention_mode = inputs["attention_mode"]
    options = {option.key: option for option in attention_mode.options}

    assert schema.node_id == "UC_UnifiedAttentionPatcher"
    assert [option.key for option in attention_mode.options] == ["disabled", "FlashAttention", "SageAttention"]
    assert [value.id for value in options["FlashAttention"].inputs] == ["allow_compile"]
    assert [value.id for value in options["SageAttention"].inputs] == [
        "sage_mode",
        "allow_compile",
        "h3_memory_optimizations",
    ]
    assert "Sparse / MiniMax H3 Radial" not in options
    assert len(options["SageAttention"].inputs) + 1 == 4


def test_minimax_h3_projection_schema_exposes_clip_contract(monkeypatch):
    monkeypatch.setattr(
        patcher_nodes,
        "list_minimax_h3_projections",
        lambda: ["mmh3-4b.safetensors"],
    )

    schema = patcher_nodes.UC_MiniMaxH3ClipProjectionPatcher.define_schema()

    assert schema.node_id == "UC_MiniMaxH3ClipProjectionPatcher"
    assert [value.id for value in schema.inputs] == ["clip", "projection"]
    assert schema.inputs[1].options == ["mmh3-4b.safetensors"]
    assert "ComfyUI/models/clip_projections" in schema.inputs[1].tooltip
    assert "README.md" in schema.inputs[1].tooltip
    assert len(schema.outputs) == 1


def test_minimax_h3_radial_config_returns_typed_runtime_value():
    result = patcher_nodes.UC_MiniMaxH3RadialAttentionConfig.execute(1, 2, 3, 128, 0.4, True)
    config = result.args[0]

    assert isinstance(config, patcher_helpers.MiniMaxH3RadialAttentionConfig)
    assert config == patcher_helpers.MiniMaxH3RadialAttentionConfig(1, 2, 3, 128, 0.4, True)


def test_unified_attention_disabled_returns_original_model():
    model = object()

    assert patcher_helpers.patch_unified_attention_model(model, {"attention_mode": "disabled"}) is model


def test_unified_h3_radial_uses_clone_scoped_wrapper_and_block_patches(monkeypatch):
    class FakeAttention:
        head_dim = 128

        def forward(self, x, rope_freqs=None, transformer_options=None):
            return x

    class FakeBlock:
        def __init__(self):
            self.attn = FakeAttention()

    class FakeH3:
        head_dim = 128

        def __init__(self):
            self.blocks = [FakeBlock() for _ in range(3)]

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}
            self.wrappers = {}

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def get_model_object(self, _name):
            return self.model.diffusion_model

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.get(wrapper_type, {}).pop(key, None)

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.setdefault(wrapper_type, {})[key] = [wrapper]

    original = FakePatcher(FakeH3())
    config = patcher_helpers.MiniMaxH3RadialAttentionConfig(dense_blocks=1)
    patched = patcher_helpers.patch_unified_attention_model(
        original,
        {"attention_mode": "Sparse / MiniMax H3 Radial", "minimax_h3_radial_config": config},
    )

    assert original.object_patches == {}
    assert sorted(patched.object_patches) == [
        "diffusion_model.blocks.1.attn.forward",
        "diffusion_model.blocks.2.attn.forward",
    ]
    wrappers = patched.wrappers[patcher_helpers.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    assert patcher_helpers.MINIMAX_H3_RADIAL_WRAPPER_KEY in wrappers


def test_unified_h3_memory_optimizations_use_clone_scoped_block_patches(monkeypatch):
    class FakeAttention:
        def forward(self, x, rope_freqs=None, transformer_options=None):
            return x

    class FakeH3:
        def __init__(self):
            self.blocks = [types.SimpleNamespace(attn=FakeAttention()) for _ in range(2)]

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    monkeypatch.setattr(patcher_helpers, "_make_sage_backend", lambda *_args: lambda *args, **_kwargs: args[2])

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def get_model_object(self, _name):
            return self.model.diffusion_model

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

    patched = patcher_helpers.patch_unified_attention_model(
        FakePatcher(FakeH3()),
        {
            "attention_mode": "SageAttention",
            "sage_mode": "auto",
            "h3_memory_optimizations": True,
        },
    )

    assert sorted(patched.object_patches) == [
        "diffusion_model.blocks.0.attn.forward",
        "diffusion_model.blocks.1.attn.forward",
    ]


def test_h3_radial_block_mask_keeps_cross_segment_blocks_dense():
    config = patcher_helpers.MiniMaxH3RadialAttentionConfig(block_size=64)
    state = {
        "config": config,
        "sequence_length": 256,
        "video_start": 128,
        "video_end": 256,
        "frame_count": 2,
        "tokens_per_frame": 64,
    }
    mask = patcher_helpers._h3_radial_block_mask(state, torch.device("cpu"))

    assert mask.shape == (4, 4)
    assert mask[:2].all()
    assert mask[:, :2].all()
    assert mask[2:, 2:].all()


def test_h3_radial_maps_adjusted_sigma_to_its_schedule_step():
    config = patcher_helpers.MiniMaxH3RadialAttentionConfig(dense_start_steps=0)
    transformer_options = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}

    assert patcher_helpers._h3_sparse_step(transformer_options, torch.tensor([0.9999]), config)
    assert patcher_helpers._h3_sparse_step(transformer_options, torch.tensor([0.75]), config)
    assert not patcher_helpers._h3_sparse_step(transformer_options, torch.tensor([0.0]), config)


def _cache(
    *,
    threshold=0.1,
    start=0.0,
    end=1.0,
    max_steps=2,
    device="auto",
    verbose=False,
):
    cache = patcher_helpers.MiniMaxH3Cache(
        reuse_threshold=threshold,
        start_percent=start,
        end_percent=end,
        max_steps=max_steps,
        device=device,
        verbose=verbose,
    )
    cache.begin(10)
    return cache


def _cache_args(image, timestep, cache_ranges=((0, 4),)):
    return {
        "img": image,
        "timestep": torch.tensor([timestep]),
        "cache_ranges": cache_ranges,
        "block_count": 2,
    }


def test_schema_exposes_stable_cache_controls():
    schema = patcher_nodes.UC_MiniMaxH3Cache.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_MiniMaxH3Cache"
    assert schema.is_experimental
    assert [value.id for value in schema.inputs] == [
        "model",
        "reuse_threshold",
        "start_percent",
        "end_percent",
        "max_steps",
        "device",
        "verbose",
    ]
    assert inputs["reuse_threshold"].default == 0.05
    assert inputs["start_percent"].default == 0.15
    assert inputs["end_percent"].default == 0.9
    assert inputs["max_steps"].default == 2
    assert inputs["device"].default == "auto"


def test_spectrum_schema_exposes_measurable_experimental_controls():
    schema = patcher_nodes.UC_MiniMaxH3Spectrum.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_MiniMaxH3Spectrum"
    assert schema.is_experimental
    assert inputs["execution_mode"].default == "spectrum"
    assert inputs["degree"].default == 1
    assert inputs["ridge_lambda"].default == 0.1
    assert inputs["warmup_steps"].default == 1
    assert inputs["video_blend_weight"].default == 0.5
    assert inputs["offline_smoothing_replay"].default is True


def test_spectrum_ridge_weights_reproduce_known_polynomial():
    coordinates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    forecaster = patcher_helpers.HistoryWeightForecaster(
        degree=2, ridge_lambda=0.0, max_history=5
    )
    for coordinate in coordinates:
        feature = torch.tensor([[[2.0 + 3.0 * coordinate - coordinate * coordinate]]])
        forecaster.update(coordinate, feature)

    predicted = forecaster.predict(0.25, 1.0)
    assert predicted.item() == pytest.approx(
        2.0 + 3.0 * 0.25 - 0.25**2, abs=1e-5
    )


def _spectrum_config(**overrides):
    values = {
        "enabled": True,
        "force_actual": False,
        "degree": 2,
        "ridge_lambda": 0.1,
        "window_size": 2.0,
        "flex_window": 0.75,
        "warmup_steps": 3,
        "tail_actual_steps": 1,
        "max_history": 8,
        "blend_weight": 1.0,
        "audio_blend_weight": 0.0,
        "history_storage": "system_ram",
        "bootstrap_first_forecast": True,
        "offline_smoothing_replay": False,
        "offline_archive_storage": "system_ram",
        "debug": False,
    }
    values.update(overrides)
    return patcher_helpers.SpectrumH3Config(**values)


def test_spectrum_config_rejects_insufficient_history():
    with pytest.raises(ValueError, match="max_history"):
        _spectrum_config(degree=4, max_history=4, bootstrap_first_forecast=False).validate()


def test_spectrum_forced_actual_disables_replay(monkeypatch):
    captured = {}

    def patch(model, config):
        captured["config"] = config
        return model

    monkeypatch.setattr(patcher_nodes, "patch_minimax_h3_spectrum_model", patch)
    model = object()
    patcher_nodes.UC_MiniMaxH3Spectrum.execute(
        model,
        "forced_actual",
        1,
        0.1,
        2.0,
        0.75,
        1,
        1,
        8,
        0.5,
        0.0,
        "system_ram",
        True,
        True,
        "system_ram",
        False,
    )

    assert captured["config"].force_actual is True
    assert captured["config"].offline_smoothing_replay is False


def test_spectrum_block_loop_observes_named_audio_then_video_targets(monkeypatch):
    class FakeRuntime:
        offline_phase = None

        def begin_model_call(self, *_args, **kwargs):
            assert kwargs["expected_shape"] == (1, 3, 2)
            assert dict(kwargs["topology"])["target_audio_rows"] == 1
            assert dict(kwargs["topology"])["target_video_rows"] == 2
            return 0, True

        def observe_actual(self, _run, _step, _call, feature):
            self.feature = feature

    monkeypatch.setattr(patcher_helpers, "SpectrumH3Runtime", FakeRuntime)
    runtime = FakeRuntime()
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    options = {
        patcher_helpers.RUNTIME_KEY: runtime,
        patcher_helpers.RUN_ID_KEY: 1,
        patcher_helpers.STEP_ID_KEY: 2,
    }
    output = patcher_helpers.SpectrumH3BlockLoop()(
        {
            "img": hidden,
            "transformer_options": options,
            "target_ranges": ((4, 6, "video"), (2, 3, "audio")),
            "block_count": 2,
        },
        {"original_block": lambda args: {"img": args["img"] + 10}},
    )

    assert torch.equal(output["img"], hidden + 10)
    assert torch.equal(runtime.feature, torch.cat(((hidden + 10)[2:3], (hidden + 10)[4:6])).unsqueeze(0))


def test_spectrum_block_loop_inserts_forecast_into_named_targets(monkeypatch):
    class FakeRuntime:
        offline_phase = None

        def begin_model_call(self, *_args, **_kwargs):
            return 0, False

        def predict(self, *_args, **_kwargs):
            return torch.tensor([[[20.0], [40.0], [50.0]]])

    monkeypatch.setattr(patcher_helpers, "SpectrumH3Runtime", FakeRuntime)
    runtime = FakeRuntime()
    hidden = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    result = patcher_helpers.SpectrumH3BlockLoop()(
        {
            "img": hidden,
            "transformer_options": {
                patcher_helpers.RUNTIME_KEY: runtime,
                patcher_helpers.RUN_ID_KEY: 1,
                patcher_helpers.STEP_ID_KEY: 2,
            },
            "target_ranges": ((4, 6, "video"), (2, 3, "audio")),
            "block_count": 2,
        },
        {"original_block": lambda _args: pytest.fail("forecast ran transformer")},
    )["img"]

    assert result[:, 0].tolist() == [0.0, 1.0, 20.0, 3.0, 40.0, 50.0]


def test_cache_reuses_residual_then_honors_maximum_skip_count():
    cache = _cache(max_steps=1)
    calls = []

    def original(args):
        calls.append(args["img"].clone())
        return {"img": args["img"] + 2.0}

    first = torch.ones((4, 8))
    assert torch.equal(
        cache(_cache_args(first, 1000.0), {"original_block": original})["img"],
        first + 2.0,
    )

    second = first + 0.001
    skipped = cache(_cache_args(second, 900.0), {"original_block": original})["img"]
    assert torch.allclose(skipped, second + 2.0)
    assert len(calls) == 1

    third = first + 0.002
    cache(_cache_args(third, 800.0), {"original_block": original})
    assert len(calls) == 2


def test_cache_recomputes_for_threshold_range_and_layout_changes():
    calls = []

    def original(args):
        calls.append(args["img"].clone())
        return {"img": args["img"] + 1.0}

    cache = _cache(threshold=0.01)
    cache(_cache_args(torch.ones((4, 8)), 1000.0), {"original_block": original})
    cache(_cache_args(torch.full((4, 8), 2.0), 900.0), {"original_block": original})
    assert len(calls) == 2

    range_cache = _cache(start=0.5)
    range_cache(
        _cache_args(torch.ones((4, 8)), 1000.0), {"original_block": original}
    )
    range_cache(
        _cache_args(torch.ones((4, 8)), 900.0), {"original_block": original}
    )
    assert len(calls) == 4

    layout_cache = _cache()
    layout_cache(
        _cache_args(torch.ones((4, 8)), 1000.0), {"original_block": original}
    )
    layout_cache(
        _cache_args(torch.ones((4, 8)), 900.0, ((1, 4),)),
        {"original_block": original},
    )
    assert len(calls) == 6


def test_cpu_cache_preserves_output_shape_and_dtype():
    cache = _cache(device="cpu")

    def original(args):
        return {"img": args["img"] + 0.5}

    image = torch.ones((4, 8), dtype=torch.float32)
    cache(_cache_args(image, 1000.0), {"original_block": original})
    output = cache(
        _cache_args(image + 0.001, 900.0), {"original_block": original}
    )["img"]

    assert cache.cached_residual.device.type == "cpu"
    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_sampling_scope_always_clears_cache_state():
    cache = _cache()
    scope = patcher_helpers.MiniMaxH3SamplingScope(cache)

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    sigmas = torch.tensor([1.0, 0.5, 0.0])
    with pytest.raises(RuntimeError, match="sampling failed"):
        scope(fail, None, None, None, sigmas)

    assert cache.cached_residual is None
    assert cache.step_counter == 0


def test_block_runner_preserves_double_block_replacements(monkeypatch):
    prefetch_events = []
    monkeypatch.setattr(
        patcher_helpers.comfy.model_prefetch,
        "make_prefetch_queue",
        lambda blocks, device, options: "queue",
    )
    monkeypatch.setattr(
        patcher_helpers.comfy.model_prefetch,
        "prefetch_queue_pop",
        lambda queue, device, block: prefetch_events.append(block),
    )

    class Block:
        def __call__(self, image, *args, **kwargs):
            return image + 1.0

    blocks = [Block(), Block()]
    model = types.SimpleNamespace(blocks=blocks)

    def replacement(args, extra_options):
        result = extra_options["original_block"](args)
        return {"img": result["img"] + 10.0}

    output = patcher_helpers.run_minimax_h3_blocks(
        model,
        torch.zeros((2, 4)),
        torch.zeros((1, 4)),
        [],
        torch.zeros((1, 4)),
        {"patches_replace": {"dit": {("double_block", 0): replacement}}},
    )

    assert torch.equal(output, torch.full((2, 4), 12.0))
    assert prefetch_events == [blocks[0], blocks[1], None]


def test_cached_forward_matches_current_core_audio_output_contract(monkeypatch):
    layout = types.SimpleNamespace(
        signature=(1, 1, 2, 2, 1),
        segments=[(0, 1, "text"), (1, 3, "audio"), (3, 4, "video")],
        img_update=torch.ones(1, dtype=torch.bool),
        audio_update=torch.ones(2, dtype=torch.bool),
        seq_len=4,
        position_ids=torch.zeros((4, 3)),
    )
    video_result = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    audio_result = torch.arange(1.0, 9.0).reshape(2, 4)
    final_layer_args = {}

    def final_layer(
        hidden, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts
    ):
        final_layer_args.update(
            sigma=sigma,
            sample_sigmas=sample_sigmas,
            shifts=shifts,
        )
        return video_result, audio_result

    model = types.SimpleNamespace(
        patch_size=(1, 2, 2),
        sigma_shift_video=1.0,
        sigma_shift_audio=1.0,
        hidden_size=4,
        use_adaln_curves=False,
        blocks=[],
        latents_dim=1,
        video_patch_proj=lambda rows: rows,
        audio_patch_proj=lambda rows: rows,
        _cond_video_rows=lambda payload, device: None,
        _cond_audio_rows=lambda payload, device: None,
        time_embedder=lambda values: values[:, None].expand(-1, 4),
        rope_freqs=lambda position_ids, device: torch.zeros((1, 1)),
        final_layer=final_layer,
    )
    monkeypatch.setattr(
        patcher_helpers.minimax_model,
        "rope_rotation_table",
        lambda frequencies, dtype: frequencies,
    )

    video = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float16)
    audio = torch.zeros((1, 4, 2, 1), dtype=torch.float16)
    context = torch.zeros((1, 1, 4), dtype=torch.float32)
    sample_sigmas = torch.tensor([1.0, 0.5, 0.0])

    assert not hasattr(patcher_helpers.minimax_model, "time_shift_slope")
    output = patcher_helpers.minimax_h3_block_patch_forward(
        model,
        [video, audio],
        torch.tensor([500.0]),
        context,
        transformer_options={
            "sample_sigmas": sample_sigmas,
            "minimax_h3_sigma_shift_video": 1.25,
            "minimax_h3_sigma_shift_audio": 0.75,
        },
        minimax_payload={"layout": layout},
    )

    expected_video = -patcher_helpers.minimax_model.unpatchify_video(
        video_result, 1, 1, 1, 1, (1, 2, 2)
    ).to(video.dtype)
    expected_audio = -patcher_helpers.minimax_model.unpack_audio(audio_result).to(
        audio.dtype
    )
    assert torch.equal(output[0], expected_video)
    assert torch.equal(output[1], expected_audio)
    assert final_layer_args["sigma"].item() == pytest.approx(0.5)
    assert final_layer_args["sample_sigmas"] is sample_sigmas
    assert final_layer_args["shifts"] == (1.25, 0.75)


def test_cached_forward_matches_current_core_final_layer_without_sigma_args(monkeypatch):
    layout = types.SimpleNamespace(
        signature=(1, 1, 2, 2, 1),
        segments=[(0, 1, "text"), (1, 3, "audio"), (3, 4, "video")],
        img_update=torch.ones(1, dtype=torch.bool),
        audio_update=torch.ones(2, dtype=torch.bool),
        seq_len=4,
        position_ids=torch.zeros((4, 3)),
    )
    video_result = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    audio_result = torch.arange(1.0, 9.0).reshape(2, 4)
    received = {}

    class CurrentFinalLayer(torch.nn.Module):
        def forward(self, x, t_emb, video_seg, audio_seg):
            received["nargs"] = 4
            received["video_seg"] = video_seg
            received["audio_seg"] = audio_seg
            return video_result, audio_result

    model = types.SimpleNamespace(
        patch_size=(1, 2, 2),
        sigma_shift_video=1.0,
        sigma_shift_audio=1.0,
        hidden_size=4,
        use_adaln_curves=False,
        blocks=[],
        latents_dim=1,
        video_patch_proj=lambda rows: rows,
        audio_patch_proj=lambda rows: rows,
        _cond_video_rows=lambda payload, device: None,
        _cond_audio_rows=lambda payload, device: None,
        time_embedder=lambda values: values[:, None].expand(-1, 4),
        rope_freqs=lambda position_ids, device: torch.zeros((1, 1)),
        final_layer=CurrentFinalLayer(),
    )
    monkeypatch.setattr(
        patcher_helpers.minimax_model,
        "rope_rotation_table",
        lambda frequencies, dtype: frequencies,
    )

    video = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float16)
    audio = torch.zeros((1, 4, 2, 1), dtype=torch.float16)
    context = torch.zeros((1, 1, 4), dtype=torch.float32)

    output = patcher_helpers.minimax_h3_block_patch_forward(
        model,
        [video, audio],
        torch.tensor([500.0]),
        context,
        transformer_options={
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            "minimax_h3_sigma_shift_video": 1.25,
            "minimax_h3_sigma_shift_audio": 0.75,
        },
        minimax_payload={"layout": layout},
    )

    expected_video = -patcher_helpers.minimax_model.unpatchify_video(
        video_result, 1, 1, 1, 1, (1, 2, 2)
    ).to(video.dtype)
    expected_audio = -patcher_helpers.minimax_model.unpack_audio(audio_result).to(
        audio.dtype
    )
    assert torch.equal(output[0], expected_video)
    assert torch.equal(output[1], expected_audio)
    assert received["nargs"] == 4


def test_model_helper_adds_only_reversible_instance_patch(monkeypatch):
    class FakeH3:
        def _forward(self):
            return "original"

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    diffusion_model = FakeH3()

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.object_patches = {}
            self.replacements = []
            self.wrappers = []

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def set_model_patch_replace(self, *args):
            self.replacements.append(args)

        def add_wrapper(self, *args):
            self.wrappers.append(args)

    original = FakePatcher(diffusion_model)
    original_class_forward = FakeH3.__dict__["_forward"]
    patched = patcher_helpers.patch_minimax_h3_cache_model(
        original, 0.1, 0.15, 0.9, 2, "auto", False
    )

    assert original.object_patches == {}
    assert set(patched.object_patches) == {"diffusion_model._forward"}
    assert patched.object_patches["diffusion_model._forward"].__self__ is diffusion_model
    assert patched.replacements[0][1:] == ("dit", "block_loop", 0)
    assert patched.wrappers[0][0] == patcher_helpers.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    assert FakeH3.__dict__["_forward"] is original_class_forward


def test_spectrum_helper_uses_clone_scoped_object_patch_and_hybrid_boundary(monkeypatch):
    class FakeH3:
        def _forward(self):
            return "original"

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    monkeypatch.setattr(patcher_helpers, "install_sampler_wrappers", lambda model, runtime: model.wrappers.append(runtime))

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.model_options = {}
            self.object_patches = {}
            self.replacements = []
            self.wrappers = []

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def set_model_patch_replace(self, *args):
            self.replacements.append(args)

    original = FakePatcher(FakeH3())
    original_class_forward = FakeH3.__dict__["_forward"]
    patched = patcher_helpers.patch_minimax_h3_spectrum_model(
        original, _spectrum_config(degree=1, warmup_steps=1)
    )

    assert original.model_options == {}
    assert patched.model_options[patcher_helpers.MINIMAX_H3_SPECTRUM_OWNER_KEY] is True
    assert set(patched.object_patches) == {"diffusion_model._forward"}
    assert patched.object_patches["diffusion_model._forward"].__self__ is original.model.diffusion_model
    assert isinstance(patched.replacements[0][0], patcher_helpers.SpectrumH3BlockLoop)
    assert patched.replacements[0][1:] == ("dit", "block_loop", 0)
    assert len(patched.wrappers) == 1
    assert FakeH3.__dict__["_forward"] is original_class_forward


def test_cache_and_spectrum_reject_both_stacking_orders():
    cache_model = types.SimpleNamespace(
        model_options={patcher_helpers.MINIMAX_H3_SPECTRUM_OWNER_KEY: True}
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        patcher_helpers.patch_minimax_h3_cache_model(
            cache_model, 0.1, 0.1, 0.9, 2, "auto", False
        )

    spectrum_model = types.SimpleNamespace(
        model_options={patcher_helpers.MINIMAX_H3_CACHE_OWNER_KEY: True}
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        patcher_helpers.patch_minimax_h3_spectrum_model(
            spectrum_model, _spectrum_config(degree=1, warmup_steps=1)
        )

    pdd_model = types.SimpleNamespace(
        model_options={patcher_helpers.MINIMAX_H3_PDD_OWNER_KEY: True}
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        patcher_helpers.patch_minimax_h3_spectrum_model(
            pdd_model, _spectrum_config(degree=1, warmup_steps=1)
        )


def test_model_helper_rejects_invalid_inputs(monkeypatch):
    class FakeH3:
        pass

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)

    class FakePatcher:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=object())

        def clone(self):
            return self

    with pytest.raises(ValueError, match="start percent"):
        patcher_helpers.patch_minimax_h3_cache_model(
            FakePatcher(), 0.1, 0.9, 0.1, 2, "auto", False
        )
    with pytest.raises(ValueError, match="requires a MiniMax H3"):
        patcher_helpers.patch_minimax_h3_cache_model(
            FakePatcher(), 0.1, 0.1, 0.9, 2, "auto", False
        )


def test_core_object_patch_restores_original_bound_method():
    class Diffusion:
        def _forward(self):
            return "original"

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.diffusion_model = Diffusion()

    tiny_model = TinyModel()
    patcher = comfy.model_patcher.ModelPatcher(
        tiny_model,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        size=1,
    )

    def replacement(self):
        return "patched"

    patcher.add_object_patch(
        "diffusion_model._forward",
        types.MethodType(replacement, tiny_model.diffusion_model),
    )
    patcher.patch_model(load_weights=False)
    assert tiny_model.diffusion_model._forward() == "patched"

    patcher.unpatch_model(unpatch_weights=False)
    assert tiny_model.diffusion_model._forward() == "original"


def _projection_data(d_in=2, d_out=3):
    return {
        "W": torch.arange(d_in * d_out, dtype=torch.float32).reshape(d_in, d_out),
        "mean_in": torch.tensor([1.0, 2.0]),
        "std_in": torch.tensor([2.0, 4.0]),
        "mean_out": torch.tensor([0.5, 1.0, 1.5]),
        "std_out": torch.tensor([1.0, 2.0, 3.0]),
        "sink_out": torch.tensor([9.0, 8.0, 7.0]),
    }


def test_minimax_h3_projection_model_matches_checkpoint_formula_and_sink():
    data = _projection_data()
    model = patcher_helpers.MiniMaxH3ProjectionModel(data, tap=7)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])

    output = model(hidden)
    normalized = (hidden.float() - data["mean_in"]) / data["std_in"]
    expected = normalized @ data["W"] * data["std_out"] + data["mean_out"]
    expected[:, 0] = data["sink_out"]

    assert model.tap == 7
    assert torch.equal(output, expected)


def test_minimax_h3_projection_model_supports_residual_only():
    data = _projection_data()
    data.pop("W")
    data["mlp.0.weight"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data["mlp.0.bias"] = torch.tensor([0.5, 1.0, 1.5])
    data.pop("sink_out")
    model = patcher_helpers.MiniMaxH3ProjectionModel(data, tap=-1)
    hidden = torch.tensor([[[3.0, 6.0]]])

    output = model(hidden)
    normalized = (hidden - data["mean_in"]) / data["std_in"]
    residual = normalized @ data["mlp.0.weight"].T + data["mlp.0.bias"]
    expected = residual * data["std_out"] + data["mean_out"]

    assert torch.equal(output, expected)


def test_minimax_h3_projected_tokenizer_uses_h3_key_and_video_blocks():
    class RawTokenizer:
        def tokenize_with_weights(self, text, **_kwargs):
            return [[(ord(character), 1.0) for character in text]]

    tokenizer = patcher_helpers._MiniMaxH3ProjectedTokenizer(
        types.SimpleNamespace(qwen3vl_4b=RawTokenizer()),
        "qwen3vl_4b",
    )
    frames = torch.zeros((3, 2, 2, 3))

    tokens = tokenizer.tokenize_with_weights(
        "go",
        minimax_ref_items=[{"type": "video", "data": frames}],
    )
    entries = tokens["qwen3vl_32b"][0]
    video_entries = [entry[0] for entry in entries if isinstance(entry[0], dict)]

    assert list(tokens) == ["qwen3vl_32b"]
    assert len(video_entries) == 2
    assert all(entry["minimax_video_block"] for entry in video_entries)
    assert all(entry["data"].shape[0] == 2 for entry in video_entries)


def test_minimax_h3_projected_clip_is_clone_scoped_and_returns_tags(monkeypatch):
    class RawTokenizer:
        def tokenize_with_weights(self, text, **_kwargs):
            return [[(ord(character), 1.0) for character in text]]

    class Transformer:
        def preprocess_embed(self, embed, device):
            return embed["data"], None

        def forward(self, *args, **kwargs):
            return kwargs.get("embeds")

    class Stage:
        clip_name = "qwen3vl_4b"
        clip = "qwen3vl_4b"

        def __init__(self):
            source_type = type(
                "GenericQwen3VL",
                (),
                {"__module__": "comfy.text_encoders.qwen3vl"},
            )
            self.qwen3vl_4b = source_type()
            self.qwen3vl_4b.transformer = Transformer()

    class FakePatcher:
        def __init__(self):
            self.load_device = torch.device("cpu")
            self.object_patches = {}

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

    class FakeProjectionPatcher:
        load_device = torch.device("cpu")

        def clone(self):
            return self

    class FakeClip:
        def __init__(self):
            self.cond_stage_model = Stage()
            self.tokenizer = types.SimpleNamespace(qwen3vl_4b=RawTokenizer())
            self.patcher = FakePatcher()
            self.layer_idx = None

        def clip_layer(self, layer_idx):
            self.layer_idx = layer_idx

        def encode_from_tokens(self, tokens, **_kwargs):
            assert list(tokens) == ["qwen3vl_4b"]
            hidden = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [1.0, 2.0]]])
            self.patcher.object_patches["qwen3vl_4b.transformer.forward"](
                embeds=hidden,
                embeds_info=[{"type": "image", "index": 1, "size": 1}],
            )
            return {"cond": hidden, "pooled_output": None}

    original = FakeClip()
    cloned = FakeClip()
    projection = patcher_helpers.MiniMaxH3ProjectionModel(_projection_data(), tap=7)
    projected = patcher_helpers.MiniMaxH3ProjectedCLIP(
        cloned,
        "projection.safetensors",
        projection_model=projection,
        projection_patcher=FakeProjectionPatcher(),
    )
    monkeypatch.setattr(patcher_helpers.comfy.model_management, "load_models_gpu", lambda _models: None)
    monkeypatch.setattr(
        patcher_helpers.comfy.model_management,
        "intermediate_device",
        lambda: torch.device("cpu"),
    )

    output = projected.encode_from_tokens(
        {"qwen3vl_32b": [[(1, 1.0)]]},
        return_dict=True,
    )

    assert original.patcher.object_patches == {}
    assert set(cloned.patcher.object_patches) == {
        "qwen3vl_4b.transformer.preprocess_embed",
        "qwen3vl_4b.transformer.forward",
    }
    assert cloned.layer_idx == 7
    assert output["cond"].shape == (1, 3, 3)
    assert torch.equal(output["minimax_token_tags"], torch.tensor([0, 0, 0]))


def _ideogram4_directions():
    return {index: torch.full((1, 8, 8, 4608), 0.25) for index in range(25, 29)}


def _ideogram4_patcher(monkeypatch):
    class Block:
        def forward(self, x, attn_mask, freqs_cis, adaln_input, transformer_options={}):
            return x

    class FakeIdeogram4:
        def __init__(self):
            self.layers = [Block() for _ in range(29)]

    class FakePatcher:
        def __init__(self, diffusion_model):
            self.diffusion_model = diffusion_model
            self.object_patches = {}
            self.wrappers = {}

        def clone(self):
            return FakePatcher(self.diffusion_model)

        def get_model_object(self, name):
            assert name == "diffusion_model"
            return self.diffusion_model

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers[wrapper_type] = {key: wrapper}

    monkeypatch.setattr(patcher_helpers.ideogram4_model, "Ideogram4Transformer2DModel", FakeIdeogram4)
    return FakePatcher(FakeIdeogram4())


def test_ideogram4_debanner_schema_and_zero_strength_noop(monkeypatch):
    schema = patcher_nodes.UC_Ideogram4DebannerPatch.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    model = object()

    assert schema.node_id == "UC_Ideogram4DebannerPatch"
    assert inputs["strength"].default == 0.6
    assert inputs["strength"].min == 0.0
    assert inputs["strength"].max == 2.0
    monkeypatch.setattr(patcher_nodes, "patch_ideogram4_debanner", lambda *_args: pytest.fail("zero strength must not patch"))
    assert patcher_nodes.UC_Ideogram4DebannerPatch.execute(model, 0.0).result == (model,)
    monkeypatch.setattr(patcher_nodes, "patch_ideogram4_debanner", lambda patched_model, strength: (patched_model, strength))
    assert patcher_nodes.UC_Ideogram4DebannerPatch.execute(model, 0.6).result == ((model, 0.6),)


def test_ideogram4_debanner_clone_wrapper_and_validation(monkeypatch):
    original = _ideogram4_patcher(monkeypatch)
    patched = patcher_helpers.patch_ideogram4_debanner_model(original, _ideogram4_directions(), 0.6)

    assert original.object_patches == {}
    assert original.wrappers == {}
    assert sorted(patched.object_patches) == [f"diffusion_model.layers.{index}.forward" for index in range(25, 29)]
    wrappers = patched.wrappers[patcher_helpers.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    assert set(wrappers) == {patcher_helpers.IDEOGRAM4_DEBANNER_WRAPPER_KEY}

    with pytest.raises(ValueError, match="block 25"):
        patcher_helpers.patch_ideogram4_debanner_model(original, {26: _ideogram4_directions()[26]}, 0.6)
    with pytest.raises(ValueError, match="nonnegative"):
        patcher_helpers.patch_ideogram4_debanner_model(original, _ideogram4_directions(), -0.1)


def test_ideogram4_debanner_wrapper_and_block_forward_preserve_text_and_norm(monkeypatch):
    original = _ideogram4_patcher(monkeypatch)
    patched = patcher_helpers.patch_ideogram4_debanner_model(original, _ideogram4_directions(), 0.5)
    wrapper = patched.wrappers[patcher_helpers.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL][patcher_helpers.IDEOGRAM4_DEBANNER_WRAPPER_KEY]
    options = {"unchanged": True}

    class Executor:
        class_obj = original.diffusion_model

        def __call__(self, x, timesteps, context, attention_mask, transformer_options, **_kwargs):
            return transformer_options

    state_options = wrapper(Executor(), torch.zeros(1, 128, 2, 2), torch.tensor([1.0]), torch.ones(1, 2, 3), None, options)
    assert options == {"unchanged": True}
    state = state_options[patcher_helpers.IDEOGRAM4_DEBANNER_STATE_KEY]
    assert state["grid_height"] == state["grid_width"] == 2
    assert state["image_token_count"] == 4
    assert state["is_conditional"] is True

    output = torch.ones(1, 6, 4608)
    corrected = patched.object_patches["diffusion_model.layers.25.forward"](output, None, None, None, state_options)
    assert torch.equal(corrected[:, :2], output[:, :2])
    assert torch.allclose(
        torch.linalg.vector_norm(corrected[:, 2:], dim=-1),
        torch.linalg.vector_norm(output[:, 2:], dim=-1),
    )
    assert not torch.equal(corrected[:, 2:], output[:, 2:])

    unconditional_options = wrapper(Executor(), torch.zeros(1, 128, 3, 1), torch.tensor([1.0]), None, None, {})
    unchanged = patched.object_patches["diffusion_model.layers.25.forward"](output, None, None, None, unconditional_options)
    assert unchanged is output


def test_minimax_h3_pdd_schema_uses_existing_lora_category(monkeypatch):
    monkeypatch.setattr(
        patcher_nodes.folder_paths,
        "get_filename_list",
        lambda category: ["minimax_h3_pdd.safetensors"] if category == "loras" else [],
    )

    schema = patcher_nodes.UC_MiniMaxH3PDDAcc.define_schema()

    assert schema.node_id == "UC_MiniMaxH3PDDAcc"
    assert schema.is_experimental
    assert [value.id for value in schema.inputs] == [
        "model",
        "pdd_lora",
        "nfe",
        "partition",
        "lora_strength",
        "head_strength",
        "on_off_grid",
    ]
    assert schema.inputs[1].options == ["minimax_h3_pdd.safetensors"]
    assert schema.inputs[2].options == ["8", "7", "6", "5", "4"]
    assert [value.id for value in schema.outputs] == ["model", "sigmas"]


def test_minimax_h3_pdd_partition_and_sigmas_stay_on_trained_grid():
    assert patcher_helpers.resolve_pdd_partition(32, 8) == (4,) * 8
    assert patcher_helpers.resolve_pdd_partition(32, 7) == (8, 4, 4, 4, 4, 4, 4)
    assert patcher_helpers.resolve_pdd_partition(32, 6) == (8, 8, 4, 4, 4, 4)
    assert patcher_helpers.resolve_pdd_partition(32, 5) == (8, 8, 8, 4, 4)
    assert patcher_helpers.resolve_pdd_partition(32, 4) == (8,) * 4
    assert len(patcher_helpers.pdd_block_boundaries(32, patcher_helpers.resolve_pdd_partition(32, 7))) == 8
    assert len(patcher_helpers.pdd_block_boundaries(32, patcher_helpers.resolve_pdd_partition(32, 5))) == 6
    bounds = patcher_helpers.pdd_block_boundaries(32, (8,) * 4)
    assert bounds[0] == 1.0
    assert bounds[-1] == 0.0
    assert len(bounds) == 5
    assert patcher_helpers.select_pdd_block(0.999992, bounds, "error") == 0
    with pytest.raises(ValueError, match="trained envelope"):
        patcher_helpers.resolve_pdd_partition(32, 32)


def test_minimax_h3_pdd_wrapper_resets_context_after_failure():
    state = patcher_helpers.MiniMaxH3PDDExecutionState()
    wrapper = patcher_helpers.make_pdd_diffusion_wrapper(state)

    class Executor:
        class_obj = types.SimpleNamespace(sigma_shift_video=12.0, sigma_shift_audio=3.0)

        def __call__(self, *_args, **_kwargs):
            assert state.sigma.get() == pytest.approx(0.5)
            raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        wrapper(Executor(), None, torch.tensor([500.0]), None, {})
    assert state.sigma.get() is None


def test_minimax_h3_pdd_head_bank_is_core_modelpatcher_managed():
    bank = patcher_helpers.MiniMaxH3PDDHeadBank(
        torch.ones(2, 3, 4),
        torch.zeros(2, 3),
        torch.ones(2, 2, 4),
        torch.zeros(2, 2),
    )
    patcher = comfy.model_patcher.ModelPatcher(
        bank,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )
    video, audio = bank.project(torch.ones(1, 4), torch.ones(1, 4), 1)

    assert set(bank.state_dict()) == {
        "video_weight",
        "video_bias",
        "audio_weight",
        "audio_bias",
    }
    assert patcher.model_size() == sum(tensor.nbytes for tensor in bank.state_dict().values())
    assert video.shape == (1, 3)
    assert audio.shape == (1, 2)
