import datetime
import pytest
import torch
import pathlib
import sys
import types

CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_helpers_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_helpers_test.helpers.video_helpers import (
    VideoCompareType,
    compute_video_signatures,
    cross_correlate_video_signatures,
    expand_filename_date_tokens,
    find_video_overlap,
)


@pytest.mark.parametrize("compare_type", list(VideoCompareType))
def test_video_signatures_support_all_reference_comparison_modes(compare_type):
    frames = torch.rand(6, 32, 48, 3, generator=torch.Generator().manual_seed(3))
    signatures = compute_video_signatures(frames, compare_type)
    assert signatures.shape[0] == frames.shape[0]
    assert signatures.ndim == 2


@pytest.mark.parametrize("compare_type", list(VideoCompareType))
def test_video_signatures_accept_size_one_and_preserve_finite_range(compare_type):
    frames = torch.linspace(2.0, 4.0, 8).view(1, 1, 8, 1).expand(3, 8, 8, 3)
    signatures = compute_video_signatures(frames, compare_type, hash_size=1)
    assert signatures.shape[0] == 3
    if compare_type == VideoCompareType.DHASH:
        assert signatures.eq(1).all()
    if compare_type == VideoCompareType.SAD:
        torch.testing.assert_close(signatures, compute_video_signatures(frames, compare_type, hash_size=0))


def test_video_sad_preserves_values_beyond_int16_range():
    reference = compute_video_signatures(torch.full((3, 8, 8, 3), 200.0), "sad")
    query = compute_video_signatures(torch.full((3, 8, 8, 3), 201.0), "sad")
    assert reference.min() > 32767
    result = cross_correlate_video_signatures(reference, query, compare_type="sad")
    assert result.distance == pytest.approx(255.0)


def test_video_correlation_honors_supplied_minimum_without_quality_errors():
    signatures = torch.arange(8, dtype=torch.int16).view(8, 1)
    result = cross_correlate_video_signatures(
        signatures, signatures, compare_type="sad", min_overlap_fraction=0, min_overlap_frames=2,
    )
    assert result.offset_frames == 0
    assert result.overlap_frames == 8
    for fraction, frames in ((0, 100), (2, 2)):
        result = cross_correlate_video_signatures(
            signatures, signatures, compare_type="sad", min_overlap_fraction=fraction, min_overlap_frames=frames,
        )
        assert result.overlap_frames == 0
        assert result.distance == float("inf")


def test_video_cross_correlation_reports_offset_and_runner_up():
    frames = torch.rand(8, 16, 16, 3, generator=torch.Generator().manual_seed(4))
    signatures = compute_video_signatures(frames, VideoCompareType.SAD)
    result = cross_correlate_video_signatures(signatures, signatures)
    assert result.offset_frames == 0
    assert result.distance == 0.0
    assert result.second_best_distance is not None
    assert result.overlap_frames == frames.shape[0]


def test_video_overlap_finds_suffix_prefix_without_single_frame_match():
    previous = torch.rand(8, 16, 16, 3, generator=torch.Generator().manual_seed(5))
    current = torch.cat(
        (previous[-4:], torch.rand(3, 16, 16, 3, generator=torch.Generator().manual_seed(6))),
        dim=0,
    )
    result = find_video_overlap(previous, current, maximum_overlap_frames=6, compare_type="sad")
    assert result.overlap_frames == 4
    assert result.distance == 0.0


@pytest.mark.parametrize("offset", [-2, 0, 2])
def test_video_cross_correlation_signed_offsets(offset):
    signatures = torch.tensor([[1], [5], [10], [30], [70], [120], [190], [240]], dtype=torch.int16)
    reference = signatures[2:7]
    query = signatures[2 + offset:]
    result = cross_correlate_video_signatures(reference, query, compare_type="sad")
    assert result.offset_frames == offset
    assert result.distance == 0.0
    assert result.second_best_distance > 0.0
    assert result.overlap_frames == min(5, 5 - offset)


def test_filename_date_tokens_expand_like_frontend():
    now = datetime.datetime(2026, 3, 7, 9, 5, 4)
    assert expand_filename_date_tokens(
        "video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default", now,
    ) == "video/2026-03-07/vid_2026-03-07_default"
    assert expand_filename_date_tokens("%date:yy.M.d_hhmmss%_%year%", now) == "26.3.7_090504_%year%"
