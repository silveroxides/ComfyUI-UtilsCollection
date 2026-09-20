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
    start_sampling_loop,
    strip_stale_keyframes,
)
from utils_collection_sampling_test.nodes.sampling_nodes import (
    UC_H3LoopSampler,
    UC_H3RefVideoSegments,
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
    assert custom_windows[1][0] == 0  # overlapped into chunk 0
    assert custom_windows[-1][1] == 22


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
        out_latent, num_chunks, report = start_sampling_loop(
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
    finally:
        sh.run_chunk_sampling = orig_run


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
    assert len(schema.outputs) == 3
    input_names = [inp.id for inp in schema.inputs]
    assert "model" in input_names
    assert "guider" in input_names
    assert "segment_lengths" in input_names
    assert "chunk_duration" in input_names
    assert "overlap_duration" in input_names

    seg_schema = UC_H3RefVideoSegments.define_schema()
    assert seg_schema.node_id == "UC_H3RefVideoSegments"
    assert seg_schema.outputs[0].is_output_list is True
    assert seg_schema.outputs[1].is_output_list is True
    assert seg_schema.outputs[2].is_output_list is False
    assert seg_schema.outputs[3].is_output_list is False
    assert seg_schema.outputs[4].is_output_list is True
    assert seg_schema.outputs[5].is_output_list is True
    assert seg_schema.outputs[6].is_output_list is True
