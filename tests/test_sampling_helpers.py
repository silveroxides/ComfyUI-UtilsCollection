"""Unit tests for sampling helpers and H3 loop sampling logic."""

import pathlib
import sys
import types
import pytest
import torch

CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_sampling_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.nested_tensor import NestedTensor
from utils_collection_sampling_test.helpers.sampling_helpers import (
    H3_LATENT_BASE,
    H3_VIDEO_T_DIM,
    H3_AUDIO_T_DIM,
    build_carry_noise_mask,
    check_core_per_row_masking,
    h3_frame_at_latent,
    h3_frames_to_audio_t,
    h3_frames_to_latents,
    h3_latents_to_frames,
    h3_pack_av,
    h3_snap_frame_count,
    h3_snap_latent_t,
    h3_split_noise_mask,
    h3_unpack_av,
    plan_h3_windows,
    prepare_chunk_guider,
    prepare_h3_source_audio,
    sample_dmad_renoise,
    start_sampling_loop,
    strip_stale_keyframes,
    split_h3_video_components_into_segments,
)
from utils_collection_sampling_test.helpers.scheduler_helpers import (
    dmad_scheduler,
    register_scheduler_handlers,
)
from utils_collection_sampling_test.nodes.sampling_nodes import (
    UC_H3LoopSampler,
    UC_H3RefVideoSegments,
    UC_SamplerDMADReNoise,
)
from utils_collection_sampling_test.nodes.scheduler_nodes import (
    UC_DMADSchedule,
)


def test_h3_temporal_conversions():
    assert h3_snap_frame_count(1) == 5
    assert h3_snap_frame_count(5) == 5
    assert h3_snap_frame_count(6) == 22
    assert h3_snap_frame_count(22) == 22

    assert h3_snap_latent_t(1) == 2
    assert h3_snap_latent_t(2) == 2
    assert h3_snap_latent_t(5) == 2
    assert h3_snap_latent_t(7) == 7

    assert h3_latents_to_frames(2) == 5
    assert h3_latents_to_frames(7) == 22
    assert h3_frames_to_latents(5) == 2
    assert h3_frames_to_latents(22) == 7

    assert h3_frame_at_latent(0) == 0
    assert h3_frame_at_latent(1) == 1
    assert h3_frame_at_latent(2) == 5
    assert h3_frame_at_latent(3) == 9
    assert h3_frame_at_latent(4) == 13
    assert h3_frame_at_latent(5) == 17


def test_h3_pack_unpack_av():
    v = torch.zeros([1, 24, 7, 16, 16])
    a = torch.zeros([1, 32, 2, 37])
    packed = h3_pack_av({}, v, a)
    un_v, un_a = h3_unpack_av(packed)
    assert un_v.shape == v.shape
    assert un_a.shape == a.shape

    video_only = h3_pack_av({}, v, None)
    un_v2, un_a2 = h3_unpack_av(video_only)
    assert un_v2.shape == v.shape
    assert un_a2 is None


def test_plan_h3_windows():
    windows = plan_h3_windows(total_frames=73, window_frames=39, overlap_frames=22)
    assert len(windows) >= 1
    for start, end in windows:
        assert end > start
        assert start >= 0
        assert end <= 22

    # Direct segment lengths test with overlap
    custom_windows = plan_h3_windows(total_frames=73, window_frames=0, overlap_frames=22, segment_lengths=[22, 22, 29])
    assert len(custom_windows) == 3
    assert custom_windows[0][0] == 0
    assert custom_windows[-1][1] == 22


def test_plan_h3_schedule_spans_and_records():
    from utils_collection_sampling_test.helpers.sampling_helpers import plan_h3_schedule
    total_v, total_a, records, spans = plan_h3_schedule(total_frames=124, window_frames=56, overlap_frames=22)
    assert total_v == 37
    assert len(records) >= 2
    assert len(spans) == len(records)
    for r in records:
        assert r["v1"] > r["v0"]
        assert r["a1"] >= r["a0"]
        assert r["frame_end"] > r["frame_start"]


def test_prepare_chunk_guider_isolation():
    class FakeGuider:
        def __init__(self):
            self.original_conds = {"positive": [["p0", {}]], "negative": [["n0", {}]]}
            self.model_options = {}

        def set_conds(self, positive, negative=None):
            self.original_conds["positive"] = positive
            if negative is not None:
                self.original_conds["negative"] = negative

    orig = FakeGuider()
    chunk_g = prepare_chunk_guider(orig, [["p1", {}]], frame0=17)

    assert orig.original_conds["positive"] == [["p0", {}]]
    assert chunk_g.original_conds["positive"] == [["p1", {}]]
    assert chunk_g.model_options["transformer_options"]["h3_ctrl_frame0"] == 17
    assert "transformer_options" not in orig.model_options


def test_strip_stale_keyframes():
    cond = [
        [torch.zeros([1, 10, 64]), {"minimax_keyframes": [{"fake": 1}], "keep_me": "yes"}],
        [torch.zeros([1, 10, 64]), {"other": 123}],
    ]
    stripped = strip_stale_keyframes(cond)
    assert "minimax_keyframes" not in stripped[0][1]
    assert stripped[0][1]["keep_me"] == "yes"
    assert stripped[1][1]["other"] == 123


def test_start_sampling_loop_mock():
    v = torch.zeros([1, 24, 7, 4, 4])
    a = torch.zeros([1, 32, 2, 12])
    latent = h3_pack_av({}, v, a)
    cond = [[torch.zeros([1, 5, 16]), {}]]

    class MockGuider:
        def __init__(self):
            self.original_conds = {"positive": cond, "negative": None}
            self.model_options = {}

        def set_conds(self, positive, negative=None):
            self.original_conds["positive"] = positive

    class MockSampler:
        pass

    sigmas = torch.tensor([1.0, 0.5, 0.0])

    from utils_collection_sampling_test.helpers import sampling_helpers as sh
    orig_run = sh.run_chunk_sampling

    def dummy_run_chunk(noise, g, s, sig, chunk_latent, **kwargs):
        return chunk_latent

    sh.run_chunk_sampling = dummy_run_chunk
    try:
        out_latent, num_chunks, report, passthrough_audio = start_sampling_loop(
            noise=None,
            guider=MockGuider(),
            sampler=MockSampler(),
            sigmas=sigmas,
            cond_list=cond,
            latent=latent,
            chunk_frames=22,
            overlap_frames=0,
            carry_mode="mask",
        )
        assert num_chunks >= 1
        res_v, res_a = h3_unpack_av(out_latent)
        assert res_v.shape == v.shape
        assert res_a.shape == a.shape
        assert "chunk 0" in report
        assert passthrough_audio is None
    finally:
        sh.run_chunk_sampling = orig_run


def test_h3_source_audio_is_encoded_pinned_and_returned_cleanly():
    video = torch.zeros([1, 24, 17, 4, 4])
    audio_latent = torch.zeros([1, 32, 2, 93])
    source_audio = {"waveform": torch.ones([1, 1, 300]), "sample_rate": 240}

    class MockAudioVAE:
        audio_sample_rate = 240
        downscale_ratio = 8

        def encode(self, samples):
            assert samples.shape[1] % 8 == 0
            return torch.full([1, 32, 2, 20], 3.0)

    encoded, passthrough = prepare_h3_source_audio(source_audio, MockAudioVAE(), audio_latent, total_frames=56)
    assert encoded.shape == audio_latent.shape
    assert torch.all(encoded[..., :20] == 3.0)
    assert torch.all(encoded[..., 20:] == 0.0)
    assert passthrough["waveform"].shape == (1, 2, 560)
    assert passthrough["sample_rate"] == 240

    cond = [[torch.zeros([1, 5, 16]), {}]]

    class MockGuider:
        def __init__(self):
            self.original_conds = {"positive": cond, "negative": None}
            self.model_options = {}

        def set_conds(self, positive, negative=None):
            self.original_conds["positive"] = positive

    from utils_collection_sampling_test.helpers import sampling_helpers as sh
    observed_masks = []
    original_run = sh.run_chunk_sampling

    def identity_chunk(noise, guider, sampler, sigmas, chunk_latent, **kwargs):
        observed_masks.append(chunk_latent["noise_mask"].unbind()[-1])
        return chunk_latent

    sh.run_chunk_sampling = identity_chunk
    try:
        result, _, _, direct_audio = start_sampling_loop(
            noise=None,
            guider=MockGuider(),
            sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            cond_list=cond,
            latent=h3_pack_av({}, video, audio_latent),
            chunk_frames=22,
            overlap_frames=22,
            source_audio=source_audio,
            audio_vae=MockAudioVAE(),
        )
    finally:
        sh.run_chunk_sampling = original_run

    _, result_audio = h3_unpack_av(result)
    result_video_mask, result_audio_mask = h3_split_noise_mask(result)
    assert direct_audio["waveform"].shape == (1, 2, 560)
    assert torch.all(result_audio[..., :20] == 3.0)
    assert result_video_mask is not None
    assert torch.count_nonzero(result_audio_mask) == 0
    assert observed_masks and all(torch.count_nonzero(mask) == 0 for mask in observed_masks)


def test_h3_source_audio_rejects_generated_audio_modes():
    video = torch.zeros([1, 24, 7, 4, 4])
    audio_latent = torch.zeros([1, 32, 2, 37])
    with pytest.raises(ValueError, match="preserve_input"):
        start_sampling_loop(
            noise=None,
            guider=object(),
            sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            cond_list=[[torch.zeros([1, 5, 16]), {}]],
            latent=h3_pack_av({}, video, audio_latent),
            audio_mode="full_generation",
            source_audio={"waveform": torch.ones([1, 2, 10]), "sample_rate": 10},
        )


def test_h3_source_audio_requires_joint_latent_and_valid_waveform():
    video = torch.zeros([1, 24, 7, 4, 4])
    source_audio = {"waveform": torch.ones([1, 2, 10]), "sample_rate": 10}
    with pytest.raises(ValueError, match="joint video/audio latent"):
        start_sampling_loop(
            noise=None,
            guider=object(),
            sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            cond_list=[[torch.zeros([1, 5, 16]), {}]],
            latent=h3_pack_av({}, video),
            source_audio=source_audio,
        )
    with pytest.raises(ValueError, match="finite mono or stereo"):
        prepare_h3_source_audio(
            {"waveform": torch.ones([1, 3, 10]), "sample_rate": 10},
            object(),
            torch.zeros([1, 32, 2, 37]),
            total_frames=22,
        )


def test_h3_loop_sampler_execute_handles_all_input_variants():
    v = torch.zeros([1, 24, 7, 4, 4])
    a = torch.zeros([1, 32, 2, 12])
    latent = h3_pack_av({}, v, a)
    single_cond = [[torch.zeros([1, 5, 16]), {}]]
    multi_cond = [single_cond, single_cond]

    class FakeModel:
        def __init__(self):
            self.model_options = {}

        def is_dynamic(self):
            return False

        def get_non_dynamic_delegate(self):
            return self

        def model_dtype(self):
            return torch.float16

    class MockGuider:
        def __init__(self):
            self.original_conds = {"positive": single_cond, "negative": None}
            self.model_options = {}

        def set_conds(self, positive, negative=None):
            self.original_conds["positive"] = positive

    from utils_collection_sampling_test.helpers import sampling_helpers as sh
    orig_run = sh.run_chunk_sampling

    def dummy_run_chunk(noise, g, s, sig, chunk_latent, **kwargs):
        return chunk_latent

    sh.run_chunk_sampling = dummy_run_chunk
    try:
        # 1. Test when segment_lengths is a single int (caused by ComfyUI list unwrapping)
        res1 = UC_H3LoopSampler.execute(
            noise=[None],
            sampler=[object()],
            sigmas=[torch.tensor([1.0, 0.0])],
            conditioning=[multi_cond],
            latent=[latent],
            model=[FakeModel()],
            segment_lengths=[22],
        )
        assert res1.result[0] is not None

        # 2. Test when segment_lengths is list of ints
        res2 = UC_H3LoopSampler.execute(
            noise=None,
            sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            conditioning=multi_cond,
            latent=latent,
            guider=MockGuider(),
            segment_lengths=[7, 15],
        )
        assert res2.result[0] is not None

        # 3. Test when segment_lengths is list of lists
        res3 = UC_H3LoopSampler.execute(
            noise=None,
            sampler=object(),
            sigmas=torch.tensor([1.0, 0.0]),
            conditioning=single_cond,
            latent=latent,
            model=FakeModel(),
            segment_lengths=[[7], [15]],
        )
        assert res3.result[0] is not None
    finally:
        sh.run_chunk_sampling = orig_run


def test_node_schema():
    schema = UC_H3LoopSampler.define_schema()
    assert schema.node_id == "UC_H3LoopSampler"
    assert len(schema.inputs) >= 7
    assert len(schema.outputs) == 4
    input_names = [inp.id for inp in schema.inputs]
    assert "model" in input_names
    assert "guider" in input_names
    assert "audio_mode" in input_names
    assert "segment_lengths" in input_names
    assert "chunk_duration" in input_names
    assert "overlap_duration" in input_names
    assert "source_audio" in input_names
    assert "audio_vae" in input_names
    assert schema.outputs[-1].id == "audio_passthrough"
    assert UC_H3LoopSampler.check_lazy_status() == []
    assert UC_H3LoopSampler.check_lazy_status(source_audio={"waveform": torch.ones([1, 2, 1])}) == ["audio_vae"]

    seg_schema = UC_H3RefVideoSegments.define_schema()
    assert seg_schema.node_id == "UC_H3RefVideoSegments"
    mp_input = next(inp for inp in seg_schema.inputs if inp.id == "megapixels")
    assert mp_input.min == 0.0
    assert seg_schema.outputs[0].is_output_list is True
    assert seg_schema.outputs[1].is_output_list is True
    assert seg_schema.outputs[2].is_output_list is False
    assert seg_schema.outputs[3].is_output_list is False
    assert seg_schema.outputs[4].is_output_list is True
    assert seg_schema.outputs[5].is_output_list is True
    assert seg_schema.outputs[6].is_output_list is True
    assert seg_schema.outputs[7].id == "tail_frames"
    assert seg_schema.outputs[7].is_output_list is False
    assert seg_schema.outputs[8].id == "trim_trailing_frames"
    assert seg_schema.outputs[8].is_output_list is True
    assert seg_schema.outputs[9].id == "durations"
    assert seg_schema.outputs[9].is_output_list is True

    # Test split_h3_video_components_into_segments with megapixels=0.0 preserves resolution
    import types
    raw_frames = torch.zeros(24, 60, 120, 3)
    comp = types.SimpleNamespace(images=raw_frames, frame_rate=24, audio=None)
    vid = types.SimpleNamespace(get_components=lambda: comp)
    out = split_h3_video_components_into_segments(vid, megapixels=0.0, segment_count=1)
    frames_list, _, out_w, out_h, lengths, _, _ = out
    assert out_w == 120
    assert out_h == 60

    # Test UC_H3RefVideoSegments.execute returns durations list
    seg_out = UC_H3RefVideoSegments.execute(vid, megapixels=0.0, segment_count=1, enable_whisper=False)
    assert len(seg_out.args) == 10
    durations = seg_out.args[9]
    assert isinstance(durations, list)
    assert durations == [float(l) / 24.0 for l in seg_out.args[4]]
    assert (out_w, out_h) == (120, 60)
    assert tuple(frames_list[0].shape[1:3]) == (60, 120)

    # Verify exact 22-frame overlap start_frame (102, not rounded to 107) and trailing padding metadata
    frames_200 = torch.arange(200, dtype=torch.float32).view(200, 1, 1, 1).expand(200, 32, 32, 3)
    comp_200 = types.SimpleNamespace(images=frames_200, frame_rate=24, audio=None)
    vid_200 = types.SimpleNamespace(get_components=lambda: comp_200)
    f_list, _, _, _, l_list, _, _, tail_opt, trail_list = split_h3_video_components_into_segments(
        vid_200, megapixels=0.0, overlap_frames=22, enable_whisper=False, return_trim_metadata=True
    )
    assert l_list == [124, 107]
    assert tail_opt == "22"
    assert trail_list == [0, 9]
    assert float(f_list[1][0].mean()) == pytest.approx(102.0)

    # Verify 15.062s clip (361 frames at 24fps) with segment_count=3 produces exactly 3 segments (not 4)
    frames_361 = torch.zeros(361, 32, 32, 3)
    frames_361[10:] = 1.0
    comp_361 = types.SimpleNamespace(images=frames_361, frame_rate=361 / 15.062, audio=None)
    vid_361 = types.SimpleNamespace(get_components=lambda: comp_361)
    _, _, _, _, l_361, v_361, _, tail_361, trail_361 = split_h3_video_components_into_segments(
        vid_361, megapixels=0.0, segment_count=3, overlap_frames=22, enable_whisper=False, return_trim_metadata=True
    )
    assert len(l_361) == 3
    assert l_361 == [141, 141, 124]
    assert trail_361 == [0, 0, 1]
    assert [getattr(x, "_uc_overlap_frames", None) for x in l_361] == [0, 22, 22]
    assert [getattr(x, "_uc_padded_frames", None) for x in l_361] == [0, 0, 1]
    import av
    src_buf = v_361[0].get_stream_source()
    src_buf.seek(0)
    with av.open(src_buf, mode="r") as container:
        keyframe_indices = [
            idx for idx, packet in enumerate(container.demux(container.streams.video[0]))
            if packet.pts is not None and packet.is_keyframe
        ]
    assert 0 in keyframe_indices and 10 in keyframe_indices


def test_dmad_scheduler_exact_values():
    """Verify dmad_scheduler matches exact grid from reference/DMAD/dmad_h3/sampling.py."""
    sigmas_4 = dmad_scheduler(None, steps=4, shift=12.0)
    assert len(sigmas_4) == 5
    assert pytest.approx(sigmas_4[0].item(), rel=1e-5) == 1.0
    assert pytest.approx(sigmas_4[1].item(), rel=1e-5) == 12.0 * 0.75 / (1.0 + 11.0 * 0.75)  # 0.97297
    assert pytest.approx(sigmas_4[2].item(), rel=1e-5) == 12.0 * 0.50 / (1.0 + 11.0 * 0.50)  # 0.92308
    assert pytest.approx(sigmas_4[3].item(), rel=1e-5) == 12.0 * 0.25 / (1.0 + 11.0 * 0.25)  # 0.80000
    assert pytest.approx(sigmas_4[4].item(), rel=1e-5) == 0.0

    # Node output
    node_out = UC_DMADSchedule.execute(steps=4, shift=12.0).result[0]
    assert torch.allclose(node_out, sigmas_4)


def test_sample_dmad_renoise_step_math_and_terminal():
    """Verify that intermediate steps re-noise and the terminal step keeps pure clean endpoint."""
    target_x0 = torch.ones(1, 4, 8, 8) * 3.14

    def mock_model(x, sigma_input, **kwargs):
        # Always predicts target_x0 as clean state
        return target_x0.clone()

    sigmas = torch.tensor([1.0, 0.8, 0.0], dtype=torch.float32)
    init_noise = torch.randn(1, 4, 8, 8)

    # Fixed noise draw for predictable assertions
    fixed_noise = torch.ones_like(target_x0) * 2.0

    def mock_noise_sampler(s, sn):
        return fixed_noise.clone()

    result = sample_dmad_renoise(
        mock_model,
        init_noise,
        sigmas,
        noise_sampler=mock_noise_sampler,
        s_noise=1.0,
    )

    # In step 0 (sigma 1.0 -> 0.8): x = 0.2 * target_x0 + 0.8 * fixed_noise
    # In step 1 (sigma 0.8 -> 0.0): sigma_next == 0, so x = target_x0
    assert torch.allclose(result, target_x0)


def test_sample_dmad_renoise_nested_tensor():
    """Verify sample_dmad_renoise natively handles NestedTensor (MiniMax H3 video + audio)."""
    t_vid = torch.randn(1, 24, 2, 8, 8)
    t_aud = torch.randn(1, 32, 2, 10)
    nt_in = NestedTensor([t_vid, t_aud])

    def mock_model(x, sigma_input, **kwargs):
        return x

    sigmas = torch.tensor([1.0, 0.9, 0.0], dtype=torch.float32)
    out = sample_dmad_renoise(
        mock_model,
        nt_in,
        sigmas,
        extra_args={"seed": 42},
    )

    assert isinstance(out, NestedTensor)
    parts = out.unbind()
    assert len(parts) == 2
    assert parts[0].shape == t_vid.shape
    assert parts[1].shape == t_aud.shape
    assert torch.isfinite(parts[0]).all()
    assert torch.isfinite(parts[1]).all()


def test_sample_dmad_renoise_determinism():
    """Verify that seed in extra_args provides reproducible re-noise draws."""
    init_x = torch.randn(1, 4, 4, 4)

    def mock_model(x, sigma, **kwargs):
        return x * 0.5

    sigmas = torch.tensor([1.0, 0.9, 0.5, 0.0], dtype=torch.float32)

    res1 = sample_dmad_renoise(mock_model, init_x.clone(), sigmas, extra_args={"seed": 123})
    res2 = sample_dmad_renoise(mock_model, init_x.clone(), sigmas, extra_args={"seed": 123})
    res3 = sample_dmad_renoise(mock_model, init_x.clone(), sigmas, extra_args={"seed": 456})

    assert torch.allclose(res1, res2)
    assert not torch.allclose(res1, res3)


def test_dmad_sampler_node_and_registration():
    """Verify UC_SamplerDMADReNoise schema/output and custom sampler wrapping."""
    schema = UC_SamplerDMADReNoise.define_schema()
    assert schema.node_id == "UC_SamplerDMADReNoise"

    out = UC_SamplerDMADReNoise.execute(s_noise=1.0).result[0]
    import comfy.samplers
    assert isinstance(out, comfy.samplers.KSAMPLER)
    assert out.sampler_function is sample_dmad_renoise
    assert out.extra_options.get("s_noise") == 1.0

    # Scheduler registration
    register_scheduler_handlers()
    assert "dmad" in comfy.samplers.SCHEDULER_HANDLERS
    assert "dmad" in comfy.samplers.SCHEDULER_NAMES


