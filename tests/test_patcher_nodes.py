import copy
import logging
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

from utils_collection_patcher_test.helpers import patcher_helpers
from utils_collection_patcher_test.nodes import patcher_nodes
from utils_collection_patcher_test.helpers.minimax_h3_cache_helpers import H3EncoderCache, clip_description


def test_unified_attention_schema_keeps_mode_settings_separate():
    schema = patcher_nodes.UC_UnifiedAttentionPatcher.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    attention_mode = inputs["attention_mode"]
    options = {option.key: option for option in attention_mode.options}

    assert schema.node_id == "UC_UnifiedAttentionPatcher"
    assert [option.key for option in attention_mode.options] == ["disabled", "FlashAttention", "SageAttention", "Sparse / MiniMax H3 SLA"]
    assert [value.id for value in options["FlashAttention"].inputs] == ["allow_compile"]
    assert [value.id for value in options["SageAttention"].inputs] == [
        "sage_mode",
        "allow_compile",
        "h3_memory_optimizations",
    ]
    assert [value.id for value in options["Sparse / MiniMax H3 SLA"].inputs] == [
        "sparsity",
        "block_size",
        "dense_tail_steps",
        "protect_reference_media",
        "dense_backend",
        "minimax_h3_sla_config",
    ]
    assert options["Sparse / MiniMax H3 SLA"].inputs[-1].optional
    sla_inputs = {value.id: value for value in options["Sparse / MiniMax H3 SLA"].inputs}
    assert sla_inputs["sparsity"].default == 0.9
    assert sla_inputs["block_size"].options == [32, 64, 128]
    assert sla_inputs["block_size"].default == 32
    assert sla_inputs["dense_tail_steps"].default == 1
    assert sla_inputs["protect_reference_media"].options == ["Off", "Light", "Heavy Enforcement"]
    assert sla_inputs["protect_reference_media"].default == "Light"
    assert sla_inputs["dense_backend"].options == list(patcher_helpers.SLA_DENSE_BACKENDS)
    assert sla_inputs["dense_backend"].default == "comfy_kitchen"
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


def test_minimax_h3_sla_config_returns_typed_runtime_value():
    config = patcher_nodes.UC_MiniMaxH3SlaAttentionConfig.execute(4096, "0,2-3", False, False, True).result[0]

    assert isinstance(config, patcher_helpers.MiniMaxH3SlaAttentionConfig)
    assert config == patcher_helpers.MiniMaxH3SlaAttentionConfig(
        minimum_sequence_length=4096,
        dense_steps="0,2-3",
        protect_audio=False,
        disable_fp16_accumulation=False,
        stabilize_routing=True,
    )


def test_minimax_h3_sla_config_schema_owns_only_config_controls():
    schema = patcher_nodes.UC_MiniMaxH3SlaAttentionConfig.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert list(inputs) == [
        "minimum_sequence_length",
        "dense_steps",
        "protect_audio",
        "disable_fp16_accumulation",
        "stabilize_routing",
    ]
    assert inputs["minimum_sequence_length"].default == 8192
    assert inputs["dense_steps"].default == "0"
    assert inputs["protect_audio"].default is True
    assert inputs["disable_fp16_accumulation"].default is True
    assert inputs["stabilize_routing"].default is False


def test_unified_h3_sla_combines_main_and_typed_controls(monkeypatch):
    typed_config = patcher_nodes.UC_MiniMaxH3SlaAttentionConfig.execute(4096, "1-2", False, False, True).result[0]
    monkeypatch.setattr(patcher_helpers, "_patch_h3_sla_attention", lambda _model, config: config)

    config = patcher_helpers.patch_unified_attention_model(
        object(),
        {
            "attention_mode": "Sparse / MiniMax H3 SLA",
            "sparsity": 0.85,
            "block_size": 128,
            "dense_tail_steps": 2,
            "protect_reference_media": "Heavy Enforcement",
            "dense_backend": "pytorch",
            "minimax_h3_sla_config": typed_config,
        },
    )

    assert config == patcher_helpers.MiniMaxH3SlaAttentionConfig(
        sparsity=0.85,
        block_size=128,
        dense_tail_steps=2,
        protect_reference_media="Heavy Enforcement",
        dense_backend="pytorch",
        minimum_sequence_length=4096,
        dense_steps="1-2",
        protect_audio=False,
        disable_fp16_accumulation=False,
        stabilize_routing=True,
    )


def test_h3_sla_parses_dense_steps_and_rejects_invalid_public_values(caplog):
    assert patcher_helpers._parse_h3_sla_dense_steps("0, 2-4, 7-6") == frozenset({0, 2, 3, 4, 6, 7})
    with caplog.at_level(logging.WARNING, logger=patcher_helpers.__name__):
        assert patcher_helpers._parse_h3_sla_dense_steps("bad,-1") == frozenset()
    assert "ignoring invalid dense step" in caplog.text

    with pytest.raises(ValueError, match="reference-media protection"):
        patcher_helpers.MiniMaxH3SlaAttentionConfig(protect_reference_media="invalid").validate()
    with pytest.raises(ValueError, match="dense backend"):
        patcher_helpers.MiniMaxH3SlaAttentionConfig(dense_backend="invalid").validate()


def test_h3_sla_dense_backend_selection_and_failure(monkeypatch):
    assert patcher_helpers._resolve_h3_sla_dense_backend("auto") is None
    assert callable(patcher_helpers._resolve_h3_sla_dense_backend("pytorch"))
    assert callable(patcher_helpers._resolve_h3_sla_dense_backend("comfy_kitchen"))

    for name in ("attention_comfy_kitchen_int8", "attention_ck", "attention_composable_kernel"):
        monkeypatch.setattr(patcher_helpers.comfy.ldm.modules.attention, name, None, raising=False)
    with pytest.raises(RuntimeError, match="comfy_kitchen is unavailable"):
        patcher_helpers._resolve_h3_sla_dense_backend("comfy_kitchen")


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


def test_unified_h3_sla_uses_clone_scoped_override_and_wrapper(monkeypatch):
    class FakeAttention:
        head_dim = 128

    class FakeBlock:
        def __init__(self):
            self.attn = FakeAttention()

    class FakeH3:
        def __init__(self):
            self.blocks = [FakeBlock(), FakeBlock()]

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)

    class FakePatcher:
        def __init__(self, diffusion):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.model_options = {"transformer_options": {}}
            self.wrappers = {}

        def clone(self):
            return FakePatcher(self.model.diffusion_model)

        def get_model_object(self, _name):
            return self.model.diffusion_model

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.get(wrapper_type, {}).pop(key, None)

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.setdefault(wrapper_type, {})[key] = [wrapper]

    original = FakePatcher(FakeH3())
    patched = patcher_helpers.patch_unified_attention_model(
        original,
        {"attention_mode": "Sparse / MiniMax H3 SLA", "dense_backend": "auto"},
    )

    assert patched is not original
    assert "optimized_attention_override" not in original.model_options["transformer_options"]
    assert "optimized_attention_override" in patched.model_options["transformer_options"]
    wrappers = patched.wrappers[patcher_helpers.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    assert patcher_helpers.MINIMAX_H3_SLA_WRAPPER_KEY in wrappers
    outer_wrappers = patched.wrappers[patcher_helpers.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]
    assert patcher_helpers.MINIMAX_H3_SLA_SAMPLING_SCOPE_KEY in outer_wrappers

    q = torch.zeros((1, 1, 8, 128))
    fallback = lambda q, _k, _v, _heads, **_kwargs: q
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    assert override(fallback, q, q, q, 1, skip_reshape=True, transformer_options={}) is q


def test_h3_sla_dense_fallback_and_protected_block_selection(caplog):
    config = patcher_helpers.MiniMaxH3SlaAttentionConfig(sparsity=0.5, block_size=64, minimum_sequence_length=0)
    q = torch.randn((1, 128, 1, 128))
    k = torch.randn((1, 128, 1, 128))
    selected, count, _history = patcher_helpers._sla_select_blocks(q, k, config, ((0, 64),), (), None)

    assert count == 2
    assert torch.all((selected == 0).any(dim=-1))

    bf16_selected, bf16_count, _history = patcher_helpers._sla_select_blocks(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        config,
        ((0, 64),),
        (),
        None,
    )
    assert bf16_count == 2
    assert torch.all((bf16_selected == 0).any(dim=-1))

    state = {"history": {}}
    override = patcher_helpers._make_h3_sla_override(config, state)
    dense_state = {patcher_helpers.MINIMAX_H3_SLA_STATE_KEY: {"dense": True}}
    with caplog.at_level(logging.INFO, logger=patcher_helpers.__name__):
        result = override(lambda q, *_args, **_kwargs: q, q.transpose(1, 2), k.transpose(1, 2), k.transpose(1, 2), 1, skip_reshape=True, transformer_options=dense_state)
    assert result.shape == (1, 1, 128, 128)
    assert "MiniMax H3 SLA using dense attention" in caplog.text


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_h3_sla_selects_each_exposed_block_size(block_size):
    config = patcher_helpers.MiniMaxH3SlaAttentionConfig(
        block_size=block_size,
        sparsity=0.5,
        minimum_sequence_length=0,
    )
    q = torch.randn((1, 256, 1, 128))
    k = torch.randn((1, 256, 1, 128))

    selected, count, _history = patcher_helpers._sla_select_blocks(q, k, config, (), (), None)

    expected_key_blocks = 4 if block_size == 128 else 256 // block_size
    assert selected.shape[-2:] == (256 // block_size, count)
    assert count == expected_key_blocks // 2


def test_h3_sla_reference_protection_modes_add_reference_quota():
    q = torch.randn((1, 128, 1, 128))
    k = torch.randn((1, 128, 1, 128))
    counts = {}
    for mode in patcher_helpers.SLA_REFERENCE_PROTECTION_MODES:
        _selected, counts[mode], _history = patcher_helpers._sla_select_blocks(
            q,
            k,
            patcher_helpers.MiniMaxH3SlaAttentionConfig(
                sparsity=0.5,
                block_size=32,
                minimum_sequence_length=0,
                protect_reference_media=mode,
            ),
            (),
            ((0, 128),),
            None,
        )

    assert counts == {"Off": 2, "Light": 3, "Heavy Enforcement": 4}


def test_h3_sla_triton_launch_ladder_retries_and_caches_success(monkeypatch):
    class OutOfResources(RuntimeError):
        pass

    calls = []

    class FakeKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **kwargs):
                calls.append((kwargs["num_warps"], kwargs["num_stages"]))
                if len(calls) == 1:
                    raise OutOfResources("first launch cannot fit")

            return launch

    monkeypatch.setattr(
        patcher_helpers,
        "triton",
        types.SimpleNamespace(
            cdiv=lambda value, block: (value + block - 1) // block,
            runtime=types.SimpleNamespace(errors=types.SimpleNamespace(OutOfResources=OutOfResources)),
        ),
    )
    monkeypatch.setattr(patcher_helpers, "_sla_attention_kernel", FakeKernel())
    patcher_helpers._SLA_TRITON_CHOSEN.clear()
    q = torch.zeros((1, 32, 1, 128))

    output = patcher_helpers._sla_sparse_attention(
        q,
        q,
        q,
        torch.zeros((1, 1, 1, 1), dtype=torch.int32),
        1,
        patcher_helpers.MiniMaxH3SlaAttentionConfig(block_size=32),
    )

    assert output.shape == q.shape
    assert calls == list(patcher_helpers.SLA_TRITON_LAUNCH_LADDER[(32, 32)][:2])
    assert patcher_helpers._SLA_TRITON_CHOSEN[(32, 32, 128)] == calls[-1]


def test_h3_sla_dense_step_selection_unions_explicit_and_tail_steps():
    config = patcher_helpers.MiniMaxH3SlaAttentionConfig(dense_steps="0,2-3", dense_tail_steps=1)
    options = {"sample_sigmas": torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])}

    assert patcher_helpers._sla_dense_step(options, torch.tensor([4.0]), config) == "sampling step 0 is configured dense"
    assert patcher_helpers._sla_dense_step(options, torch.tensor([2.0]), config) == "sampling step 2 is configured dense"
    assert patcher_helpers._sla_dense_step(options, torch.tensor([1.0]), config) == "sampling step 3 is configured dense"
    tail_config = patcher_helpers.MiniMaxH3SlaAttentionConfig(dense_steps="", dense_tail_steps=1)
    assert patcher_helpers._sla_dense_step(options, torch.tensor([1.0]), tail_config) == "one of the final 1 sampler steps is configured dense"


def test_h3_sla_sampling_scope_reports_each_run(caplog):
    state = {
        "history": {0: torch.ones((1, 1, 1, 1), dtype=torch.int32)},
        "reported_fallbacks": {"stale reason"},
        "sparse_disabled_reason": "stale failure",
    }
    scope = patcher_helpers.MiniMaxH3SlaSamplingScope(
        state,
        patcher_helpers.MiniMaxH3SlaAttentionConfig(disable_fp16_accumulation=False),
    )

    def sample():
        assert state["history"] == {}
        assert state["reported_fallbacks"] == set()
        assert state["sparse_disabled_reason"] is None
        patcher_helpers._report_sla_fallback(state, "configured dense step")
        return "sampled"

    with caplog.at_level(logging.INFO, logger=patcher_helpers.__name__):
        assert scope(sample) == "sampled"
        assert scope(sample) == "sampled"

    assert state["history"] == {}
    assert caplog.text.count("configured dense step") == 2


def test_h3_sla_sampling_scope_restores_reduced_precision_accumulation():
    backend = torch.backends.cuda.matmul
    attributes = [
        name
        for name in ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
        if hasattr(backend, name)
    ]
    if not attributes:
        pytest.skip("PyTorch build has no reduced-precision accumulation controls.")
    original = {name: getattr(backend, name) for name in attributes}
    state = {"history": {}, "reported_fallbacks": set(), "sparse_disabled_reason": None}
    scope = patcher_helpers.MiniMaxH3SlaSamplingScope(state, patcher_helpers.MiniMaxH3SlaAttentionConfig())

    def sample():
        assert all(getattr(backend, name) is False for name in attributes)

    scope(sample)

    assert {name: getattr(backend, name) for name in attributes} == original

    def failing_sample():
        raise RuntimeError("sampling failure")

    with pytest.raises(RuntimeError, match="sampling failure"):
        scope(failing_sample)
    assert {name: getattr(backend, name) for name in attributes} == original


def test_unified_h3_memory_optimizations_use_clone_scoped_block_patches(monkeypatch):
    class FakeAttention:
        def forward(self, x, rope_freqs=None, transformer_options=None):
            return x

    class FakeH3:
        def __init__(self):
            self.blocks = [
                types.SimpleNamespace(attn=FakeAttention()),
                types.SimpleNamespace(attn=FakeAttention()),
            ]

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    monkeypatch.setattr(patcher_helpers, "_make_sage_backend", lambda *_args: lambda *args, **_kwargs: args[2])

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
            self.wrappers.setdefault(wrapper_type, {})[key] = wrapper

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
    assert not schema.is_experimental
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


def test_cache_finish_releases_residual_buffer():
    cache = _cache(device="cpu")
    cache._store_residual(torch.ones((4, 8)))
    assert cache.cached_residual is not None
    assert cache._residual_buffer is not None

    cache.finish()

    assert cache.cached_residual is None
    assert cache._residual_buffer is None


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


def test_diffusion_cache_schema_exposes_physical_controls():
    schema = patcher_nodes.UC_MiniMaxH3DiffusionCache.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_MiniMaxH3DiffusionCache"
    assert not schema.is_experimental
    assert [value.id for value in schema.inputs] == [
        "model",
        "preset",
        "composition_cutoff",
        "detail_protection",
        "drift_tolerance",
        "max_noise_span",
        "device",
        "verbose",
    ]
    assert inputs["preset"].default == "balanced"
    assert inputs["preset"].options == ["balanced", "high_fidelity", "high_speed", "custom"]
    assert inputs["composition_cutoff"].default == 0.75
    assert inputs["detail_protection"].default == 0.15
    assert inputs["drift_tolerance"].default == 1.0
    assert inputs["max_noise_span"].default == 0.20
    assert inputs["device"].default == "auto"


def test_diffusion_cache_presets_and_custom_validation():
    balanced = patcher_helpers.MiniMaxH3DiffusionCache(preset="balanced")
    assert balanced.composition_cutoff == 0.75
    assert balanced.detail_protection == 0.15
    assert balanced.reuse_threshold == 0.05
    assert balanced.max_noise_span == 0.20

    high_fidelity = patcher_helpers.MiniMaxH3DiffusionCache(preset="high_fidelity", drift_tolerance=0.8)
    assert high_fidelity.composition_cutoff == 0.65
    assert high_fidelity.detail_protection == 0.25
    assert high_fidelity.reuse_threshold == pytest.approx(0.03 * 0.8)
    assert high_fidelity.max_noise_span == 0.12

    high_speed = patcher_helpers.MiniMaxH3DiffusionCache(preset="high_speed", drift_tolerance=1.5)
    assert high_speed.composition_cutoff == 0.85
    assert high_speed.detail_protection == 0.10
    assert high_speed.reuse_threshold == pytest.approx(0.08 * 1.5)
    assert high_speed.max_noise_span == 0.30

    custom = patcher_helpers.MiniMaxH3DiffusionCache(
        preset="custom", composition_cutoff=0.80, detail_protection=0.20, drift_tolerance=2.0, max_noise_span=0.25
    )
    assert custom.composition_cutoff == 0.80
    assert custom.detail_protection == 0.20
    assert custom.reuse_threshold == pytest.approx(0.10)
    assert custom.max_noise_span == 0.25

    with pytest.raises(ValueError, match="Composition cutoff noise level"):
        patcher_helpers.patch_minimax_h3_diffusion_cache_model(
            None, preset="custom", composition_cutoff=0.10, detail_protection=0.80
        )


def test_diffusion_cache_phase_gating_and_span_limits():
    cache = patcher_helpers.MiniMaxH3DiffusionCache(preset="balanced", drift_tolerance=1.0)
    sigmas = torch.linspace(1.0, 0.0, 11)
    cache.begin(sigmas)

    calls = []

    def original(args):
        calls.append(args["img"].clone())
        return {"img": args["img"] + 2.0}

    # Step 1: noise = 0.90 (in composition phase > 0.75) -> must RUN
    first = torch.ones((4, 8))
    out1 = cache(_cache_args(first, 900.0), {"original_block": original})
    assert len(calls) == 1
    assert torch.equal(out1["img"], first + 2.0)

    # Step 2: noise = 0.70 (in cache window [0.15, 0.75]) -> runs to establish first residual in window
    second = first + 0.001
    out2 = cache(_cache_args(second, 700.0), {"original_block": original})
    assert len(calls) == 2
    assert torch.equal(out2["img"], second + 2.0)

    # Step 3: noise = 0.65 (delta 0.70 - 0.65 = 0.05 <= 0.20 max span, low drift) -> must SKIP
    third = second + 0.0005
    out3 = cache(_cache_args(third, 650.0), {"original_block": original})
    assert len(calls) == 2  # No new call, skipped!
    assert torch.allclose(out3["img"], third + 2.0)

    # Step 4: noise = 0.45 (delta 0.70 - 0.45 = 0.25 > 0.20 max span!) -> must RUN to refresh
    fourth = third + 0.0005
    out4 = cache(_cache_args(fourth, 450.0), {"original_block": original})
    assert len(calls) == 3
    assert torch.equal(out4["img"], fourth + 2.0)

    # Step 5: noise = 0.10 (in micro detail phase < 0.15) -> must RUN
    fifth = fourth + 0.0005
    out5 = cache(_cache_args(fifth, 100.0), {"original_block": original})
    assert len(calls) == 4
    assert torch.equal(out5["img"], fifth + 2.0)


def test_diffusion_cache_sampling_scope_resets_on_error():
    cache = patcher_helpers.MiniMaxH3DiffusionCache(preset="balanced")
    scope = patcher_helpers.MiniMaxH3DiffusionSamplingScope(cache)

    def fail(*args, **kwargs):
        raise RuntimeError("diffusion sampling failed")

    sigmas = torch.tensor([1.0, 0.5, 0.0])
    with pytest.raises(RuntimeError, match="diffusion sampling failed"):
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
        lambda queue, device, block, malloc_scope=None: prefetch_events.append((block, malloc_scope)),
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
    assert prefetch_events == [(blocks[0], "block"), (blocks[1], "block"), (None, "block")]


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


def test_cached_forward_supports_core_cond_audio_segments(monkeypatch):
    layout = types.SimpleNamespace(
        signature=(1, 1, 2, 2, 1),
        segments=[
            (0, 1, "text"),
            (1, 3, "cond_audio"),
            (3, 5, "audio"),
            (5, 6, "video"),
        ],
        img_update=torch.ones(1, dtype=torch.bool),
        audio_update=torch.tensor([False, False, True, True]),
        seq_len=6,
        position_ids=torch.zeros((6, 3)),
    )
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
        _cond_audio_rows=lambda payload, device: torch.zeros((2, 4), device=device),
        time_embedder=lambda values: values[:, None].expand(-1, 4),
        rope_freqs=lambda position_ids, device: torch.zeros((1, 1)),
        final_layer=lambda *args: (torch.ones((1, 4)), torch.ones((2, 4))),
    )
    monkeypatch.setattr(
        patcher_helpers.minimax_model,
        "rope_rotation_table",
        lambda frequencies, dtype: frequencies,
    )

    output = patcher_helpers.minimax_h3_block_patch_forward(
        model,
        [
            torch.zeros((1, 1, 1, 2, 2), dtype=torch.float16),
            torch.zeros((1, 4, 2, 1), dtype=torch.float16),
        ],
        torch.tensor([500.0]),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        transformer_options={},
        minimax_payload={"layout": layout},
    )

    assert len(output) == 2


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


def test_h3_sla_and_cache_preserve_both_model_patcher_contracts(monkeypatch):
    class FakeAttention:
        head_dim = 128

    class FakeBlock:
        def __init__(self):
            self.attn = FakeAttention()

    class FakeH3:
        def __init__(self):
            self.blocks = [FakeBlock(), FakeBlock()]

    class FakePatcher:
        def __init__(self, diffusion, model_options=None, object_patches=None, replacements=None, wrappers=None):
            self.model = types.SimpleNamespace(diffusion_model=diffusion)
            self.model_options = dict(model_options or {})
            self.object_patches = dict(object_patches or {})
            self.replacements = dict(replacements or {})
            self.wrappers = {
                wrapper_type: dict(entries) for wrapper_type, entries in (wrappers or {}).items()
            }

        def clone(self):
            return FakePatcher(
                self.model.diffusion_model,
                self.model_options,
                self.object_patches,
                self.replacements,
                self.wrappers,
            )

        def get_model_object(self, _name):
            return self.model.diffusion_model

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def set_model_patch_replace(self, value, *key):
            self.replacements[key] = value

        def add_wrapper(self, wrapper_type, wrapper):
            self.wrappers.setdefault(wrapper_type, {})[f"plain-{len(self.wrappers.get(wrapper_type, {}))}"] = wrapper

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.get(wrapper_type, {}).pop(key, None)

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.setdefault(wrapper_type, {})[key] = wrapper

    monkeypatch.setattr(patcher_helpers.minimax_model, "MiniMaxH3Model", FakeH3)
    config = patcher_helpers.MiniMaxH3SlaAttentionConfig()

    def assert_combined(patched):
        assert "diffusion_model._forward" in patched.object_patches
        assert ("dit", "block_loop", 0) in patched.replacements
        assert "optimized_attention_override" in patched.model_options["transformer_options"]
        assert patcher_helpers.MINIMAX_H3_SLA_WRAPPER_KEY in patched.wrappers[
            patcher_helpers.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
        ]
        outer = patched.wrappers[patcher_helpers.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]
        assert patcher_helpers.MINIMAX_H3_SLA_SAMPLING_SCOPE_KEY in outer
        assert any(key.startswith("plain-") for key in outer)

    original = FakePatcher(FakeH3())
    cached_then_sla = patcher_helpers.patch_unified_attention_model(
        patcher_helpers.patch_minimax_h3_cache_model(original, 0.1, 0.1, 0.9, 2, "auto", False),
        {"attention_mode": "Sparse / MiniMax H3 SLA", "minimax_h3_sla_config": config},
    )
    assert_combined(cached_then_sla)

    sla_then_cached = patcher_helpers.patch_minimax_h3_cache_model(
        patcher_helpers.patch_unified_attention_model(
            original,
            {"attention_mode": "Sparse / MiniMax H3 SLA", "minimax_h3_sla_config": config},
        ),
        0.1,
        0.1,
        0.9,
        2,
        "auto",
        False,
    )
    assert_combined(sla_then_cached)


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


def test_minimax_h3_projected_clip_is_clone_scoped_and_returns_tags(monkeypatch, tmp_path):
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
        def __init__(self, model):
            self.model = model
            self.load_device = torch.device("cpu")
            self.object_patches = {}

        def add_object_patch(self, name, value):
            self.object_patches[name] = value

        def get_model_object(self, name):
            return self.object_patches.get(name, getattr(self.model.qwen3vl_4b.transformer, name.rsplit(".", 1)[-1]))

        def clone(self):
            cloned = copy.copy(self)
            cloned.object_patches = self.object_patches.copy()
            return cloned

    class FakeProjectionPatcher:
        load_device = torch.device("cpu")

        def clone(self):
            return self

    class FakeClip:
        def __init__(self):
            self.cond_stage_model = Stage()
            self.tokenizer = type("Tokenizers", (), {"qwen3vl_4b": RawTokenizer()})()
            self.patcher = FakePatcher(self.cond_stage_model)
            self.layer_idx = None

        def clone(self):
            cloned = copy.copy(self)
            cloned.patcher = self.patcher.clone()
            return cloned

        def clip_layer(self, layer_idx):
            self.layer_idx = layer_idx

        def add_hooks_to_dict(self, metadata):
            pass

        def encode_from_tokens_scheduled(self, tokens, **kwargs):
            output = self.encode_from_tokens(tokens)
            return [[output["cond"], {key: value for key, value in output.items() if key != "cond"}]]

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
    identity = clip_description(projected)
    assert identity == clip_description(projected.clone())
    monkeypatch.setattr(patcher_helpers.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    for iteration in range(2):
        with H3EncoderCache() as cache:
            cached_clip = cache.prepare_clip(projected)
            tokens = {"qwen3vl_32b": [[(1, 1.0)]]}
            cached = cache.encode_scheduled(cached_clip, tokens, "grid-deepstack", lambda: cached_clip.encode_from_tokens_scheduled(tokens))
            torch.testing.assert_close(cached[0][0], output["cond"], rtol=0, atol=0)
            assert torch.equal(cached[0][1]["minimax_token_tags"], output["minimax_token_tags"])
            assert cache.hits["encoded_section"] == iteration
    projected._projection_model.tap += 1
    assert identity != clip_description(projected)


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


def test_krea2_nag_schema_and_formula_contract(monkeypatch):
    schema = patcher_nodes.UC_Krea2NAG.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    model = object()

    assert schema.node_id == "UC_Krea2NAG"
    assert inputs["nag_scale"].default == 5.0
    assert inputs["nag_tau"].default == 2.5
    assert inputs["nag_alpha"].default == 0.25
    assert inputs["start_percent"].default == 0.0
    assert inputs["end_percent"].default == 1.0
    assert inputs["nag_negative"].optional is True
    assert all(value.tooltip for value in schema.inputs)
    assert all(value.tooltip for value in schema.outputs)

    monkeypatch.setattr(patcher_nodes, "patch_krea2_nag_model", lambda **_kwargs: pytest.fail("noop must not patch"))
    assert patcher_nodes.UC_Krea2NAG.execute(model, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.0).result == (model,)
    assert patcher_nodes.UC_Krea2NAG.execute(model, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25).result == (model,)
    assert patcher_nodes.UC_Krea2NAG.execute(model, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25, start_percent=0.5, end_percent=0.5).result == (model,)

    z_pos = torch.tensor([[[1.0, 2.0, -1.0, 2.0]]])
    z_neg = torch.tensor([[[-1.0, 0.0, 1.0, -2.0]]])
    # guided = 5*z_pos - 4*z_neg = [9, 10, -9, 18], L1(guided)=46, L1(z_pos)=6 -> ratio=46/6 > tau=2.5
    # guided_norm = guided * (2.5 / (46/6)) = guided * (15/46)
    # z_final = 0.25 * guided_norm + 0.75 * z_pos
    expected = 0.25 * (z_pos * 5.0 - z_neg * 4.0) * (15.0 / 46.0) + 0.75 * z_pos
    actual = patcher_helpers.apply_nag_guidance(z_pos, z_neg, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(
        patcher_helpers.apply_nag_guidance(z_pos, z_pos, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.25),
        z_pos,
        atol=1e-6,
    )

    # Per-head L1 normalization preserves spatial RoPE heads (head 0: z_pos == z_neg)
    # when semantic text heads (head 1: z_neg == -z_pos) exceed nag_tau.
    two_head_pos = torch.cat((torch.ones(1, 4, 4), torch.ones(1, 4, 4)), dim=-1)
    two_head_neg = torch.cat((torch.ones(1, 4, 4), -torch.ones(1, 4, 4)), dim=-1)
    guided_two_head = patcher_helpers.apply_nag_guidance(
        two_head_pos, two_head_neg, nag_scale=5.0, nag_tau=2.5, nag_alpha=0.5, heads=2
    )
    assert torch.allclose(guided_two_head[..., :4], torch.ones(1, 4, 4), atol=1e-6)
    assert not torch.allclose(guided_two_head[..., 4:], torch.ones(1, 4, 4), atol=1e-6)


def test_krea2_nag_end_to_end_variable_length_and_ref_latents():
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT

    torch.manual_seed(42)
    dit = SingleStreamDiT(
        features=32,
        tdim=16,
        txtdim=16,
        heads=2,
        kvheads=1,
        multiplier=1,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=2,
        txtheads=2,
        txtkvheads=1,
        default_ref_method="index_timestep_zero",
        device=torch.device("cpu"),
        dtype=torch.float32,
        operations=comfy.ops.manual_cast,
    )
    with torch.no_grad():
        for name, param in dit.named_parameters():
            if "scale" in name or "lin" in name:
                param.normal_(0.0, 0.05)
            else:
                param.normal_(0.0, 0.1)

    class FakeModelSampling:
        def percent_to_sigma(self, percent: float) -> float:
            return 1.0 - float(percent)

    class FakeKrea2Patcher:
        def __init__(self, diffusion_model, model_sampling):
            self.diffusion_model = diffusion_model
            self.model_sampling = model_sampling
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}

        def clone(self):
            cloned = FakeKrea2Patcher(self.diffusion_model, self.model_sampling)
            cloned.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            cloned.object_patches = dict(self.object_patches)
            return cloned

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self.diffusion_model
            if name == "model_sampling":
                return self.model_sampling
            raise KeyError(name)

        def set_model_patch(self, patch, name):
            patches = self.model_options.setdefault("transformer_options", {}).setdefault("patches", {})
            patches[name] = patches.get(name, []) + [patch]

        def set_model_post_input_patch(self, patch):
            self.set_model_patch(patch, "post_input")

        def set_model_attn1_patch(self, patch):
            self.set_model_patch(patch, "attn1_patch")

        def set_model_attn1_output_patch(self, patch):
            self.set_model_patch(patch, "attn1_output_patch")

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            wrappers = self.model_options.setdefault("wrappers", {}).setdefault(wrapper_type, {})
            wrappers[key] = [wrapper]

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def disable_model_cfg1_optimization(self):
            self.model_options["disable_cfg1_optimization"] = True

    base_patcher = FakeKrea2Patcher(dit, FakeModelSampling())
    x = torch.randn(1, 4, 4, 8)
    timesteps = torch.tensor([0.5])
    pos_context = torch.randn(1, 7, 32)
    neg_context = torch.randn(1, 3, 32)
    ref_latent = torch.randn(1, 4, 4, 8)

    baseline_out = dit(x, timesteps, pos_context, ref_latents=[ref_latent], transformer_options={})

    patched = patcher_helpers.patch_krea2_nag_model(
        base_patcher,
        nag_scale=5.0,
        nag_tau=2.5,
        nag_alpha=0.25,
        start_percent=0.0,
        end_percent=0.8,
    )
    assert patched.model_options.get("disable_cfg1_optimization") is True

    for index, block in enumerate(dit.blocks):
        block.forward = patched.object_patches[f"diffusion_model.blocks.{index}.forward"]

    # When positive and negative conditioning are identical, NAG output must equal baseline
    same_opts = comfy.model_patcher.create_model_options_clone(patched.model_options)["transformer_options"]
    same_opts["wrappers"] = patched.model_options["wrappers"]
    same_opts[patcher_helpers.KREA2_NAG_NEG_CONTEXT_KEY] = pos_context
    same_opts["sigmas"] = torch.tensor([0.5])
    same_out = dit(x, timesteps, pos_context, ref_latents=[ref_latent], transformer_options=same_opts)
    assert torch.allclose(same_out, baseline_out, atol=1e-5)

    # With variable-length negative prompt (L_pos=7 vs L_neg=3), NAG actively steers output
    nag_opts = comfy.model_patcher.create_model_options_clone(patched.model_options)["transformer_options"]
    nag_opts["wrappers"] = patched.model_options["wrappers"]
    nag_opts[patcher_helpers.KREA2_NAG_NEG_CONTEXT_KEY] = neg_context
    nag_opts["sigmas"] = torch.tensor([0.5])
    nag_out = dit(x, timesteps, pos_context, ref_latents=[ref_latent], transformer_options=nag_opts)
    assert nag_out.shape == baseline_out.shape
    assert not torch.allclose(nag_out, baseline_out, atol=1e-4)

    # Outside sigma window (sigma=0.1 < sigma_end=0.2 for end_percent=0.8), NAG is bypassed
    late_opts = comfy.model_patcher.create_model_options_clone(patched.model_options)["transformer_options"]
    late_opts["wrappers"] = patched.model_options["wrappers"]
    late_opts[patcher_helpers.KREA2_NAG_NEG_CONTEXT_KEY] = neg_context
    late_opts["sigmas"] = torch.tensor([0.1])
    late_out = dit(x, timesteps, pos_context, ref_latents=[ref_latent], transformer_options=late_opts)
    assert torch.allclose(late_out, baseline_out, atol=1e-5)

    # CALC_COND_BATCH wrapper at cfg=1.0 skips redundant uncond forward pass and injects neg_context
    calls = []

    def fake_calc_cond_batch(_model, conds, x_in, _timestep, model_options):
        calls.append((conds, model_options))
        return [torch.ones_like(x_in), torch.zeros_like(x_in)]

    calc_wrapper = patched.model_options["wrappers"][comfy.patcher_extension.WrappersMP.CALC_COND_BATCH][
        patcher_helpers.KREA2_NAG_CALC_COND_WRAPPER_KEY
    ][0]
    predict_wrapper = patched.model_options["wrappers"][comfy.patcher_extension.WrappersMP.PREDICT_NOISE][
        patcher_helpers.KREA2_NAG_PREDICT_NOISE_WRAPPER_KEY
    ][0]

    class FakeGuiderExec:
        class_obj = types.SimpleNamespace(cfg=1.0)

        def __call__(self, x_in, ts, opts, _seed):
            cond_pos = [{"model_conds": {"c_crossattn": types.SimpleNamespace(cond=pos_context)}}]
            cond_neg = [{"model_conds": {"c_crossattn": types.SimpleNamespace(cond=neg_context)}}]
            return calc_wrapper(fake_calc_cond_batch, None, [cond_pos, cond_neg], x_in, ts, opts)

    predict_wrapper(FakeGuiderExec(), x, torch.tensor([0.5]), dict(patched.model_options), None)
    assert len(calls) == 1
    recorded_conds, recorded_opts = calls[0]
    assert recorded_conds[1] is None
    assert recorded_opts["transformer_options"][patcher_helpers.KREA2_NAG_NEG_CONTEXT_KEY] is neg_context


def test_moss_image_edit_patcher_end_to_end():
    from comfy.ldm.moss_image import model as moss_model

    torch.manual_seed(7)
    raw_flux2 = torch.randn(2, 16, 4, 6)
    unpatched = patcher_helpers.unpatchify_flux2_latent(raw_flux2)
    assert unpatched.shape == (2, 4, 8, 12)
    repatched = patcher_helpers.patchify_flux2_latent(unpatched)
    assert torch.allclose(repatched, raw_flux2)

    resized = patcher_helpers.resize_moss_reference_latent(raw_flux2, 6, 8)
    assert resized.shape == (2, 16, 6, 8)
    lp = patcher_helpers.lowpass_moss_latent(raw_flux2, kernel_size=3)
    assert lp.shape == raw_flux2.shape
    assert float(lp.std()) <= float(raw_flux2.std())

    ids_aligned = patcher_helpers.build_moss_reference_rope_ids(
        target_h=4,
        target_w=6,
        ref_h=4,
        ref_w=6,
        text_len=5,
        ref_index=1,
        rope_mode="spatial_aligned",
    )
    assert ids_aligned.shape == (24, 4)
    assert torch.all(ids_aligned[:, 0] == 0.0)
    assert torch.all(ids_aligned[:, 1] == 5.0)

    ids_ref_axis = patcher_helpers.build_moss_reference_rope_ids(
        target_h=4,
        target_w=6,
        ref_h=3,
        ref_w=3,
        text_len=5,
        ref_index=2,
        rope_mode="reference_axis",
    )
    assert ids_ref_axis.shape == (9, 4)
    assert torch.all(ids_ref_axis[:, 0] == 2.0)
    assert torch.all(ids_ref_axis[:, 1] == 7.0)

    dit = moss_model.MOSSImage(
        dim=64,
        ffn_dim=128,
        num_heads=2,
        num_layers=2,
        text_dim=32,
        latent_channels=16,
        dtype=torch.float32,
        device="cpu",
        operations=torch.nn,
    ).train(False)

    class FakeModelSampling:
        def percent_to_sigma(self, percent: float) -> float:
            return 1.0 - float(percent)

    class FakeBaseModel:
        def process_latent_in(self, latent):
            return latent

        def extra_conds(self, **kwargs):
            return {}

    class FakeMOSSPatcher:
        def __init__(self, diffusion, sampling, base_model=None):
            self.diffusion = diffusion
            self.sampling = sampling
            self.model = base_model or FakeBaseModel()
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}

        def clone(self):
            cloned = FakeMOSSPatcher(self.diffusion, self.sampling, FakeBaseModel())
            cloned.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            cloned.object_patches = dict(self.object_patches)
            return cloned

        def add_object_patch(self, name: str, obj):
            self.object_patches[name] = obj
            if "." not in name:
                setattr(self.model, name, obj)

        def apply_layer_patches(self):
            for idx, layer in enumerate(self.diffusion.layers):
                key = f"diffusion_model.layers.{idx}.forward"
                if key in self.object_patches:
                    layer.forward = self.object_patches[key]

        def get_model_object(self, name: str):
            if name == "diffusion_model":
                return self.diffusion
            if name == "model_sampling":
                return self.sampling
            return None

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.model_options.get("wrappers", {}).get(wrapper_type, {}).pop(key, None)

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            wrappers = self.model_options.setdefault("wrappers", {}).setdefault(wrapper_type, {})
            wrappers[key] = [wrapper]

    base_patcher = FakeMOSSPatcher(dit, FakeModelSampling())
    ref_tensor = torch.randn(1, 16, 4, 4)
    mask_tensor = torch.zeros(1, 4, 4)
    mask_tensor[:, :2, :2] = 1.0  # Edit top-left 2x2, preserve rest

    (patched_edit,) = patcher_nodes.UC_MOSSImageEditPatcher.execute(
        base_patcher,
        preset="balanced_edit",
        strength=0.75,
        structure_retention=0.60,
        uncond_skip_layer_22=False,
        reference_latent={"samples": ref_tensor},
        mask=mask_tensor,
        uncond_skip_layer=1,
        start_percent=0.0,
        end_percent=0.85,
    )
    (patched_zero,) = patcher_nodes.UC_MOSSImageEditPatcher.execute(
        base_patcher,
        preset="balanced_edit",
        strength=0.0,
        structure_retention=0.0,
        uncond_skip_layer_22=False,
        uncond_skip_layer=-1,
    )
    (patched_subject,) = patcher_nodes.UC_MOSSImageEditPatcher.execute(
        base_patcher,
        preset="subject_reference",
        strength=0.80,
        structure_retention=0.0,
        uncond_skip_layer_22=False,
        uncond_skip_layer=-1,
    )

    # Verify extra_conds patch extracts reference_latents from conditioning
    extra = patched_edit.model.extra_conds(reference_latents=[ref_tensor])
    assert "ref_latents" in extra
    assert patcher_helpers._embed_moss_rope_ids(dit, ids_aligned).ndim == 6
    for idx in range(len(dit.layers)):
        assert f"diffusion_model.layers.{idx}.forward" in patched_edit.object_patches

    old_training = comfy.model_management.in_training
    comfy.model_management.in_training = False
    try:
        x_inplace = torch.randn(2, 8, 64)
        ptr_before = x_inplace.data_ptr()
        res_inplace = patcher_helpers._moss_gated_residual(
            x_inplace,
            torch.ones_like(x_inplace),
            torch.full((2, 1, 64), 0.5),
        )
        assert res_inplace.data_ptr() == ptr_before
    finally:
        comfy.model_management.in_training = old_training

    x = torch.randn(2, 16, 4, 4)
    timesteps = torch.tensor([0.5, 0.5])
    context = torch.randn(2, 5, 32)

    old_training = comfy.model_management.in_training
    comfy.model_management.in_training = True
    try:
        baseline_out = dit(x, timesteps, context, transformer_options={"cond_or_uncond": [0, 1]})

        # Zero strength & no mask matches baseline exactly
        patched_zero.apply_layer_patches()
        zero_opts = {
            "wrappers": patched_zero.model_options["wrappers"],
            "cond_or_uncond": [0, 1],
            "sigmas": torch.tensor([0.5]),
        }
        zero_out = dit(x, timesteps, context, transformer_options=zero_opts)
        assert torch.allclose(zero_out, baseline_out, atol=1e-5)

        # OUTER_SAMPLE anchors noise and rescues non-empty sampler latent_image
        outer_wrapper = patched_edit.model_options["wrappers"][comfy.patcher_extension.WrappersMP.OUTER_SAMPLE][
            patcher_helpers.MOSS_IMAGE_EDIT_OUTER_SAMPLE_WRAPPER_KEY
        ][0]
        captured_noise = []
        outer_wrapper(
            lambda n, *_a, **_kw: captured_noise.append(n),
            x.clone(),
            ref_tensor.expand_as(x),
            None,
            None,
        )
        assert len(captured_noise) == 1
        assert captured_noise[0].shape == x.shape
        # Inside mask (top-left 2x2) noise is unanchored, outside mask it is anchored toward reference low-pass
        assert torch.allclose(captured_noise[0][:, :, :2, :2], x[:, :, :2, :2], atol=1e-5)
        assert not torch.allclose(captured_noise[0][:, :, 2:, 2:], x[:, :, 2:, 2:], atol=1e-4)

        # Diffusion wrapper applies 42-layer reference KV attention, structure guidance, and exact unmasked preservation
        patched_edit.apply_layer_patches()
        edit_opts = {
            "wrappers": patched_edit.model_options["wrappers"],
            "cond_or_uncond": [0, 1],
            "sigmas": torch.tensor([0.5]),
        }
        edit_v = dit(x, timesteps, context, transformer_options=edit_opts)
        assert edit_v.shape == baseline_out.shape
        assert not torch.allclose(edit_v, baseline_out, atol=1e-4)

        # Verify unmasked region (bottom-right 2x2 where mask == 0) predicts exact reference x0 = x - sigma * v
        x0_edited = x - 0.5 * edit_v
        assert torch.allclose(x0_edited[:, :, 2:, 2:], ref_tensor.expand_as(x)[:, :, 2:, 2:], atol=1e-4)

        # Subject reference mode with conditioning ref_latents (even at different spatial size 3x3) runs cleanly
        patched_subject.apply_layer_patches()
        subject_opts = {
            "wrappers": patched_subject.model_options["wrappers"],
            "cond_or_uncond": [0, 1],
            "sigmas": torch.tensor([0.5]),
        }
        small_ref = torch.randn(1, 16, 3, 3)
        subject_v = dit(x, timesteps, context, transformer_options=subject_opts, ref_latents=[small_ref])
        assert subject_v.shape == baseline_out.shape
        assert not torch.allclose(subject_v, baseline_out, atol=1e-4)

        # Outside sigma window (sigma=0.05 < 0.15 for end_percent=0.85), wrapper is bypassed
        late_opts = {
            "wrappers": patched_edit.model_options["wrappers"],
            "cond_or_uncond": [0, 1],
            "sigmas": torch.tensor([0.05]),
        }
        late_out = dit(x, timesteps, context, transformer_options=late_opts)
        assert torch.allclose(late_out, baseline_out, atol=1e-5)
    finally:
        comfy.model_management.in_training = old_training


def test_krea2_edit_patcher_end_to_end():
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT

    schema = patcher_nodes.UC_Krea2EditPatcher.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert schema.node_id == "UC_Krea2EditPatcher"
    assert inputs["ref_boost"].default == 1.0
    assert inputs["boost_mode"].default == "Fast / Key Scaling (FlashAttention)"
    assert inputs["ref_position"].default == "aligned (trained)"
    assert inputs["cache_text_fusion"].default is True

    torch.manual_seed(42)
    dit = SingleStreamDiT(
        features=32,
        tdim=16,
        txtdim=16,
        heads=2,
        kvheads=1,
        multiplier=1,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=2,
        txtheads=2,
        txtkvheads=1,
        default_ref_method="index_timestep_zero",
        device=torch.device("cpu"),
        dtype=torch.float32,
        operations=comfy.ops.manual_cast,
    )
    with torch.no_grad():
        for name, param in dit.named_parameters():
            if "scale" in name or "lin" in name:
                param.normal_(0.0, 0.05)
            else:
                param.normal_(0.0, 0.1)

    class FakeBaseModel:
        def __init__(self, diffusion_model):
            self.diffusion_model = diffusion_model

        def process_latent_in(self, latent):
            return latent * 0.5

        def extra_conds(self, **kwargs):
            return {}

    class FakeKrea2Patcher:
        def __init__(self, base_model):
            self.model = base_model
            self.model_options = {"transformer_options": {}}
            self.object_patches = {}

        def clone(self):
            cloned = FakeKrea2Patcher(self.model)
            cloned.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            cloned.object_patches = dict(self.object_patches)
            return cloned

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self.model.diffusion_model
            raise KeyError(name)

        def add_object_patch(self, path, value):
            self.object_patches[path] = value

        def remove_wrappers_with_key(self, wrapper_type, key):
            wrappers = self.model_options.get("wrappers", {}).get(wrapper_type, {})
            wrappers.pop(key, None)

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            wrappers = self.model_options.setdefault("wrappers", {}).setdefault(wrapper_type, {})
            wrappers[key] = [wrapper]

        def apply_object_patches(self):
            for path, value in self.object_patches.items():
                comfy.utils.set_attr(self.model, path, value)

    class FakeVAE:
        def __init__(self):
            self.calls = []

        def encode(self, pixels):
            self.calls.append(tuple(pixels.shape))
            b, h, w, _ = pixels.shape
            return torch.ones(b, 4, h // 8, w // 8)

    base_model = FakeBaseModel(dit)
    base_patcher = FakeKrea2Patcher(base_model)
    vae = FakeVAE()
    src_img = torch.rand(1, 64, 64, 3)

    (patched,) = patcher_nodes.UC_Krea2EditPatcher.execute(
        model=base_patcher,
        source_image=src_img,
        vae=vae,
        ref_boost=4.0,
        ref_boost_a=2.0,
        boost_mode="Fast / Key Scaling (FlashAttention)",
        ref_position="beside (right of target)",
        ref_position_a="aligned (trained)",
        fit_mode="fit",
        target_width=64,
        target_height=64,
        cache_text_fusion=True,
    )

    # 1. Verify pre-sampling VAE encode happened during node execution
    assert len(vae.calls) == 1
    assert vae.calls[0] == (1, 64, 64, 3)

    # 2. Verify object patches installed cleanly
    assert "extra_conds" in patched.object_patches
    assert "diffusion_model.process_img" in patched.object_patches
    assert "diffusion_model.txtfusion" in patched.object_patches
    assert "diffusion_model.blocks.0.attn.forward" in patched.object_patches
    assert "diffusion_model.blocks.1.attn.forward" in patched.object_patches

    patched.apply_object_patches()

    # 3. Verify extra_conds injects scaled ref_latents and index_timestep_zero
    conds = base_model.extra_conds()
    assert "ref_latents" in conds
    assert "ref_latents_method" in conds

    # 4. Verify RoPE placement in process_img ("beside" shifts x by target_w)
    x = torch.randn(1, 4, 4, 8)
    ref = torch.randn(1, 4, 4, 8)
    _, tgt_ids, th, tw = dit.process_img(x, index=0)
    _, ref_ids, _, _ = dit.process_img(ref, index=1)
    assert tw == 4
    assert torch.allclose(ref_ids[..., 2], tgt_ids[..., 2] + float(tw))

    # 5. Verify TextFusion step-0 cache & fast key-scaling boost
    timesteps = torch.tensor([0.5])
    context = torch.randn(1, 5, 32)
    out_step0 = dit(x, timesteps, context, ref_latents=[ref], transformer_options={})
    assert patched.object_patches["diffusion_model.txtfusion"].cache is not None

    # Step 1 reuses cached txtfusion output
    out_step1 = dit(x, torch.tensor([0.25]), context, ref_latents=[ref], transformer_options={})
    assert out_step0.shape == x.shape
    assert out_step1.shape == x.shape

    # OUTER_SAMPLE scope clears cache on exit
    outer_wrapper = patched.model_options["wrappers"][comfy.patcher_extension.WrappersMP.OUTER_SAMPLE][
        patcher_helpers.KREA2_EDIT_OUTER_SAMPLE_WRAPPER_KEY
    ][0]
    outer_wrapper(lambda: None)
    assert patched.object_patches["diffusion_model.txtfusion"].cache is None


def test_minimax_h3_context_patcher_schema():
    schema = patcher_nodes.UC_MiniMaxH3ContextPatcher.define_schema()
    assert schema.node_id == "UC_MiniMaxH3ContextPatcher"
    assert schema.display_name == "MiniMax H3 Context Patcher"
    assert schema.category == "advanced/model/patches"
    inputs = {inp.id: inp for inp in schema.inputs}
    assert "model" in inputs
    assert inputs["window_size"].default == "124 frames (~5.2s - Standard)"
    assert "124 frames (~5.2s - Standard)" in inputs["window_size"].options
    assert "Disabled (Full Latent)" in inputs["window_size"].options
    assert inputs["overlap"].default == "39 frames (~1.6s - Standard)"
    assert "39 frames (~1.6s - Standard)" in inputs["overlap"].options
    assert inputs["blend_profile"].default == "pyramid"
    assert inputs["freenoise"].default is True
    assert len(schema.outputs) == 1
    assert schema.outputs[0].id == "model"


def test_minimax_h3_context_patcher_option_parsing():
    assert patcher_helpers.parse_h3_context_frames("124 frames (~5.2s - Standard)", 124) == 124
    assert patcher_helpers.parse_h3_context_frames("73 frames (~3.0s - Low VRAM)", 124) == 73
    assert patcher_helpers.parse_h3_context_frames(141, 124) == 141
    assert patcher_helpers.parse_h3_context_frames(None, 124) == 124


def test_minimax_h3_context_patcher_execution_and_grid_snapping():
    import comfy.context_windows

    class FakePatcher:
        def __init__(self):
            self.model_options = {}
            self.wrappers = {}

        def clone(self):
            f = FakePatcher()
            f.model_options = copy.deepcopy(self.model_options)
            f.wrappers = copy.deepcopy(self.wrappers)
            return f

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    base = FakePatcher()
    out = patcher_nodes.UC_MiniMaxH3ContextPatcher.execute(
        base,
        window_size="124 frames (~5.2s - Standard)",
        overlap="39 frames (~1.6s - Standard)",
        blend_profile="pyramid",
        freenoise=True,
    )
    patched = out.args[0]
    assert patched is not base
    handler = patched.model_options.get("context_handler")
    assert isinstance(handler, comfy.context_windows.IndexListContextHandler)
    # 124 frames -> 37 latents; 39 frames -> 12 latents
    assert handler.context_length == 37
    assert handler.context_overlap == 12
    assert handler.freenoise is True
    assert "prepare_sampling" in patched.wrappers
    assert "sampler_sample" in patched.wrappers

    # Disabled mode leaves model untouched
    out_disabled = patcher_nodes.UC_MiniMaxH3ContextPatcher.execute(
        base,
        window_size="Disabled (Full Latent)",
    )
    assert out_disabled.args[0] is base




