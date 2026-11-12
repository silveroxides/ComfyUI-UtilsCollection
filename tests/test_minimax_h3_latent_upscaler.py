import pathlib
import sys
import types

import comfy.model_patcher
import comfy.ops
import folder_paths
import pytest
import torch
import torch.nn.functional as F
from comfy.nested_tensor import NestedTensor

PACKAGE_NAME = "utils_collection_h3_upscaler_test"
PACKAGE_ROOT = pathlib.Path(__file__).parents[1]
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_h3_upscaler_test.helpers import model_helpers, patcher_helpers  # noqa: E402
from utils_collection_h3_upscaler_test.helpers.helper_functions import atomic_uel_writer  # noqa: E402
from utils_collection_h3_upscaler_test.models import minimax_h3_latent_upscaler as upscaler_mod  # noqa: E402
from utils_collection_h3_upscaler_test.models.minimax_h3_latent_upscaler import MiniMaxH3LatentUpscaler2D  # noqa: E402


def test_2d_upscaler_matches_v1_checkpoint_key_count_and_shapes():
    model = MiniMaxH3LatentUpscaler2D(in_channels=24, channels=640, in_blocks=12, out_blocks=12, embed_dim=128, dual_embed_freq=16, tier_dim=32, temporal_every=2, temporal_kernel=5, device="meta", dtype=torch.bfloat16, operations=comfy.ops.disable_weight_init)
    expected, _ = model_helpers._collect_expected_module_shapes(model)
    assert len(expected) == 262
    total_params = sum(torch.Size(shape).numel() for shape in expected.values())
    assert total_params == 182140312
    assert expected["resizer.conv_in.weight"] == (640, 24, 3, 3)
    assert expected["resizer.embed.linear.0.weight"] == (128, 96)
    assert expected["temporal_blocks.0.dwconv.weight"] == (640, 1, 5, 1, 1)
    assert expected["temporal_blocks.1.pwconv.weight"] == (640, 640, 1, 1, 1)


def _init_small_params(module: torch.nn.Module) -> torch.nn.Module:
    for param in module.parameters():
        param.data.normal_(0.0, 0.05)
    return module


def test_2d_temporal_conv_and_chunking_match_naive_conv3d_reference(monkeypatch):
    monkeypatch.setattr(upscaler_mod, "FRAME_SLAB", 3)
    torch.manual_seed(7)
    model = _init_small_params(MiniMaxH3LatentUpscaler2D(in_channels=24, channels=32, in_blocks=2, out_blocks=2, embed_dim=32, dual_embed_freq=4, tier_dim=32, temporal_every=2, temporal_kernel=3, dtype=torch.float32, operations=comfy.ops.disable_weight_init))
    model.train(False)

    tb = model.temporal_blocks[0]
    x_seq = torch.randn(1, 32, 8, 6, 6, dtype=torch.float32)
    x_2d = x_seq.permute(0, 2, 1, 3, 4).reshape(8, 32, 6, 6).clone()
    slab_out = tb(x_2d, 1, 8).view(1, 8, 32, 6, 6).permute(0, 2, 1, 3, 4)

    normed = F.silu(tb.norm(x_seq.permute(0, 2, 1, 3, 4).reshape(8, 32, 6, 6))).view(1, 8, 32, 6, 6).permute(0, 2, 1, 3, 4)
    ref_out = x_seq + tb.pwconv(tb.dwconv(normed))
    torch.testing.assert_close(slab_out, ref_out, rtol=1e-5, atol=1e-5)

    video = torch.randn(1, 24, 10, 4, 6, dtype=torch.float32)
    full_pass = model(video, (10, 8, 12), sharpness=1, chunk_frames=None)
    chunked_pass = model(video, (10, 8, 12), sharpness=1, chunk_frames=3)
    default_call = model(video, (10, 8, 12))
    sharp_tier6 = model(video, (10, 8, 12), sharpness=6, chunk_frames=3)

    torch.testing.assert_close(chunked_pass, full_pass, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(default_call, full_pass, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(chunked_pass, sharp_tier6, atol=1e-4)


def test_uel_loader_streams_2d_and_3d_checkpoints_and_rejects_pth(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"\.safetensors"):
        model_helpers.load_minimax_h3_latent_upscale_model("minimax_h3_latent_upscaler_2d_conv_v1_fp32.pth")
    with pytest.raises(ValueError, match=r"\.safetensors"):
        model_helpers.load_minimax_h3_latent_upscale_model(model_helpers.MINIMAX_H3_UPSCALER_PLACEHOLDER)

    ref_2d = _init_small_params(MiniMaxH3LatentUpscaler2D(in_channels=24, channels=32, in_blocks=2, out_blocks=2, embed_dim=32, dual_embed_freq=16, tier_dim=32, temporal_every=2, temporal_kernel=3, dtype=torch.float16, operations=comfy.ops.disable_weight_init))
    lazy_probe = MiniMaxH3LatentUpscaler2D(in_channels=24, channels=32, in_blocks=2, out_blocks=2, embed_dim=32, dual_embed_freq=16, tier_dim=32, temporal_every=2, temporal_kernel=3, dtype=torch.float16, operations=comfy.ops.disable_weight_init)
    lazy_probe.resizer.conv_in.weight = None
    lazy_probe.resizer.conv_in.bias = None
    lazy_probe.resizer.conv_in.comfy_need_lazy_init_bias = True
    expected_lazy, lazy_keys = model_helpers._collect_expected_module_shapes(lazy_probe)
    assert "resizer.conv_in.weight" in lazy_keys
    assert "resizer.conv_in.bias" in lazy_keys
    assert expected_lazy["resizer.conv_in.weight"] == (32, 24, 3, 3)

    ckpt_2d = tmp_path / "mini_2d.safetensors"
    with atomic_uel_writer(ckpt_2d, {"model": "mini_2d"}) as writer:
        writer.write_batch([(f"upscaler.{k}", v.detach().cpu()) for k, v in ref_2d.state_dict().items()])

    from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler

    ref_3d = _init_small_params(MiniMaxH3LatentUpscaler(in_channels=24, channels=32, in_blocks=2, out_blocks=2, temporal_kernel=3, dtype=torch.float16, operations=comfy.ops.disable_weight_init))
    ckpt_3d = tmp_path / "mini_3d.safetensors"
    with atomic_uel_writer(ckpt_3d) as writer:
        writer.write_batch([(k, v.detach().cpu()) for k, v in ref_3d.state_dict().items()])

    paths = {"mini_2d.safetensors": str(ckpt_2d), "mini_3d.safetensors": str(ckpt_3d)}
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda folder, name: paths[name])
    monkeypatch.setattr(comfy.model_management, "vae_dtype", lambda allowed_dtypes=None: torch.float32)

    patcher_2d = model_helpers.load_minimax_h3_latent_upscale_model("mini_2d.safetensors")
    assert isinstance(patcher_2d, comfy.model_patcher.CoreModelPatcher)
    assert isinstance(patcher_2d.model, MiniMaxH3LatentUpscaler2D)
    assert patcher_2d.model.training is False
    torch.testing.assert_close(patcher_2d.model.resizer.conv_in.weight.cpu(), ref_2d.resizer.conv_in.weight.float().cpu())

    patcher_3d = model_helpers.load_minimax_h3_latent_upscale_model("mini_3d.safetensors")
    assert isinstance(patcher_3d, comfy.model_patcher.CoreModelPatcher)
    assert isinstance(patcher_3d.model, MiniMaxH3LatentUpscaler)
    assert patcher_3d.model.training is False


def test_upscale_minimax_h3_latent_preserves_av_audio_and_aligns_modes(monkeypatch):
    model = _init_small_params(MiniMaxH3LatentUpscaler2D(in_channels=24, channels=32, in_blocks=1, out_blocks=1, embed_dim=32, dual_embed_freq=4, tier_dim=32, temporal_every=0, dtype=torch.float32, operations=comfy.ops.disable_weight_init))
    model.train(False)
    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    monkeypatch.setattr(comfy.model_management, "load_models_gpu", lambda models, memory_required=0: None)
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))

    video = torch.randn(1, 24, 2, 6, 8, dtype=torch.float32)
    audio = torch.randn(1, 32, 2, 8, dtype=torch.float32)
    av_latent = {"samples": NestedTensor((video, audio)), "noise_mask": NestedTensor((torch.ones(1, 1, 2, 6, 8), torch.ones(1, 32, 2, 8)))}

    res = model_helpers.upscale_minimax_h3_latent(patcher, av_latent, {"mode": "scale_by", "scale": 2.0}, align=32, sharpness=2, temporal_chunking={"temporal_chunking": "chunked", "chunk_frames": 4})
    assert "noise_mask" not in res
    assert res["samples"].is_nested
    up_v, up_a = res["samples"].tensors
    assert tuple(up_v.shape) == (1, 24, 2, 12, 16)
    torch.testing.assert_close(up_a, audio)

    noop = model_helpers.upscale_minimax_h3_latent(patcher, av_latent, {"mode": "scale_by", "scale": 1.0}, align=32)
    assert tuple(noop["samples"].tensors[0].shape) == (1, 24, 2, 6, 8)

    h_mp, w_mp, _ = model_helpers.compute_minimax_h3_upscale_latent_hw(22, 38, {"mode": "megapixels", "megapixels": 1.0}, align=32)
    assert (h_mp * 16) % 32 == 0 and (w_mp * 16) % 32 == 0

    with pytest.raises(ValueError, match="upscaling only"):
        model_helpers.compute_minimax_h3_upscale_latent_hw(20, 20, {"mode": "dimensions", "width": 160, "height": 160}, align=32)


def test_spatial_tile_patcher_fuses_packed_av_crops_payload_and_isolates_cache_keys():
    import comfy.conds
    import comfy.patcher_extension
    import comfy.utils

    class _DummyInnerModel(torch.nn.Module):
        def __init__(self, shapes):
            super().__init__()
            self.latent_shapes = shapes

    with pytest.raises(ValueError, match="blend_profile"):
        patcher_helpers.patch_minimax_h3_spatial_tiles(comfy.model_patcher.CoreModelPatcher(_DummyInnerModel(None), load_device=torch.device("cpu"), offload_device=torch.device("cpu")), blend_profile="invalid")

    video = torch.linspace(0.1, 1.0, 1 * 24 * 2 * 12 * 16, dtype=torch.float32).reshape(1, 24, 2, 12, 16)
    audio = torch.linspace(-0.5, 0.5, 1 * 32 * 2 * 8, dtype=torch.float32).reshape(1, 32, 2, 8)
    packed_x, full_shapes = comfy.utils.pack_latents([video, audio])

    kf_lat = torch.ones(1, 24, 1, 12, 16, dtype=torch.float32)
    ref_lat = torch.ones(1, 24, 1, 10, 10, dtype=torch.float32)
    dm_lat = torch.ones(1, 1, 2, 12, 16, dtype=torch.float32) * 0.75
    payload = {"layout": "stale_full_frame_layout", "keyframes": [{"resolved_frame_index": 0, "latent": kf_lat}], "refs": [{"kind": "image", "latent": ref_lat}], "cond_video_latents": [kf_lat, ref_lat], "cond_audio_latents": []}
    conds = [[{"model_conds": {"latent_shapes": comfy.conds.CONDConstant(full_shapes), "denoise_mask": comfy.conds.CONDRegular(dm_lat), "minimax_payload": comfy.conds.CONDConstant(payload)}}], None]

    inner = _DummyInnerModel(full_shapes)
    base_patcher = comfy.model_patcher.CoreModelPatcher(inner, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    calls_noop = []

    def _noop_exec(m, c, x, t, opts):
        calls_noop.append(x.shape)
        return [x, x]

    patched_large = patcher_helpers.patch_minimax_h3_spatial_tiles(base_patcher, tile_width=512, tile_height=512)
    exec_noop = comfy.patcher_extension.WrapperExecutor.new_executor(_noop_exec, patched_large.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH))
    out_noop = exec_noop.execute(inner, conds, packed_x, torch.tensor([0.5]), patched_large.model_options)
    assert len(calls_noop) == 1
    torch.testing.assert_close(out_noop[0], packed_x)

    seen_cache_keys = set()
    seen_kf_shapes = []

    def _tile_exec(m, c, x, t, opts):
        key = patcher_helpers._extract_window_key({"transformer_options": opts.get("transformer_options", {})})
        seen_cache_keys.add(key)
        mc = c[0][0]["model_conds"]
        tile_shapes = mc["latent_shapes"].cond
        mp = mc["minimax_payload"].cond
        assert "layout" not in mp
        assert tuple(mp["refs"][0]["latent"].shape[-2:]) == (10, 10)
        kf_crop = mp["keyframes"][0]["latent"]
        seen_kf_shapes.append(tuple(kf_crop.shape[-2:]))
        assert tuple(kf_crop.shape[-2:]) == tuple(tile_shapes[0][-2:])
        assert tuple(mc["denoise_mask"].cond.shape[-2:]) == tuple(tile_shapes[0][-2:])
        assert tuple(mp["cond_video_latents"][0].shape[-2:]) == tuple(tile_shapes[0][-2:])
        assert tuple(mp["cond_video_latents"][1].shape[-2:]) == (10, 10)
        m.latent_shapes = tile_shapes
        return [x * 2.0, x * 0.0]

    for profile in ("cosine", "pyramid"):
        seen_cache_keys.clear()
        seen_kf_shapes.clear()
        patched = patcher_helpers.patch_minimax_h3_spatial_tiles(base_patcher, tile_width=128, tile_height=128, overlap_ratio=0.25, min_tile_size=64, blend_profile=profile)
        executor = comfy.patcher_extension.WrapperExecutor.new_executor(_tile_exec, patched.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH))
        res_branches = executor.execute(inner, conds, packed_x, torch.tensor([0.5]), patched.model_options)
        assert len(seen_cache_keys) == 6
        assert all(4 <= h <= 8 and 4 <= w <= 8 and h % 2 == 0 and w % 2 == 0 for h, w in seen_kf_shapes)
        assert inner.latent_shapes == full_shapes
        torch.testing.assert_close(res_branches[0], packed_x * 2.0, rtol=1e-5, atol=1e-5)
