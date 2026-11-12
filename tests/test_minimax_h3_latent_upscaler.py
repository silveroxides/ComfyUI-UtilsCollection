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

from utils_collection_h3_upscaler_test.helpers import model_helpers, sampling_helpers  # noqa: E402
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


def test_tiled_sampler_enforces_h3_token_phase_audio_trim_and_seam_stitching(monkeypatch):
    for tv in (7, 12, 17, 22, 27, 37, 42):
        for chunk_f, ov_f in ((22, 0), (22, 5), (22, 22), (39, 22), (56, 39), (73, 22)):
            segs = sampling_helpers.compute_h3_tiled_segments(tv, chunk_f, ov_f)
            assert segs[0][0] == 0
            assert segs[-1][2] == tv
            for k0, f0, k1, f1 in segs:
                assert k0 % 5 == 0
                assert k1 - k0 >= 2
                assert (k1 - k0 - 2) % 5 == 0
                assert f0 == sampling_helpers._h3_tiled_frames_for_tokens(k0)
                assert f1 == sampling_helpers._h3_tiled_frames_for_tokens(k1)

    kf = {"resolved_frame_index": 0, "audio_latent": torch.ones(1, 32, 2, 80)}
    trimmed_kf = sampling_helpers.trim_h3_tiled_keyframe(kf, 0, 24)
    assert trimmed_kf is not None
    assert trimmed_kf["audio_latent"].shape[-1] == 40

    mask = sampling_helpers.build_h3_spatial_fade_mask(16, 16, ovh=4, ovw=4, done_top=True, done_left=True, fade_h=2, fade_w=2, seam_cap=0.6)
    assert float(mask[0, 0]) == 0.0
    assert float(mask[1, 1]) == 0.0
    assert 0.0 < float(mask[3, 3]) <= 0.6 + 1e-5
    assert float(mask[-1, -1]) == 1.0

    recorded_kf_shapes = []
    cache_resets = []

    class _DummyModelObj:
        latent_format = None

    class _DummyPatcher:
        load_device = torch.device("cpu")
        model = _DummyModelObj()
        model_options = {}

    class _DummyGuider:
        def __init__(self):
            self.model_patcher = _DummyPatcher()
            self.model_options = {}
            base_kf = {"resolved_frame_index": 0, "latent": torch.ones(1, 24, 1, 12, 12)}
            pos = [[torch.zeros(1, 4, 8), {"minimax_keyframes": [base_kf]}]]
            self.original_conds = {"positive": pos, "negative": None}
            self.raw_conds = (pos, None)

        def inner_set_conds(self, conds):
            self.original_conds = dict(conds)

        def sample(self, noise_t, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=True, seed=0):
            pos_list = self.original_conds["positive"]
            kfs = pos_list[0][1].get("minimax_keyframes", [])
            for entry in kfs:
                if "latent" in entry:
                    recorded_kf_shapes.append(tuple(entry["latent"].shape[-2:]))
            v, a = latent_image.tensors
            assert denoise_mask is not None and denoise_mask.is_nested
            if callback is not None:
                callback(0, latent_image, latent_image, 1)
            return NestedTensor((v + 0.01, a))

    class _DummyNoise:
        seed = 123

        def generate_noise(self, input_latent):
            return torch.zeros_like(input_latent["samples"])

    import comfy.sample
    import latent_preview

    monkeypatch.setattr(comfy.sample, "fix_empty_latent_channels", lambda model, latent_image: latent_image)
    monkeypatch.setattr(latent_preview, "get_previewer", lambda device, fmt: None)
    monkeypatch.setattr(comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    orig_reset = sampling_helpers.reset_model_caches
    monkeypatch.setattr(sampling_helpers, "reset_model_caches", lambda g: (cache_resets.append(1), orig_reset(g)))

    video_in = torch.zeros(1, 24, 12, 12, 12)
    audio_in = torch.zeros(1, 32, 2, 65)
    t_cfg = sampling_helpers.build_h3_temporal_split_config(chunk_frames=22, overlap_frames=5, anchor_strength=0.999, motion_anchor_frames="5", identity_anchor_frames=12)
    s_cfg, _ = sampling_helpers.build_h3_spatial_split_config(tile_width=128, tile_height=128, overlap_ratio=0.25, fade_ratio=0.5, min_tile_size=64, seam_denoise=0.7)

    stitched = sampling_helpers.sample_minimax_h3_tiled_upscale(latent={"samples": NestedTensor((video_in, audio_in))}, noise=_DummyNoise(), sampler=object(), sigmas=torch.tensor([0.5, 0.0]), guider=_DummyGuider(), temporal_split_config=t_cfg, spatial_split_config=s_cfg, seam_polish="all", color_match=True)
    out_v, out_a = stitched["samples"].tensors
    assert tuple(out_v.shape) == (1, 24, 12, 12, 12)
    assert tuple(out_a.shape) == (1, 32, 2, 65)
    assert len(cache_resets) > 4
    assert (8, 8) in recorded_kf_shapes
