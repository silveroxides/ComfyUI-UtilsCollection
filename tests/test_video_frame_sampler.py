import io as stdlib_io
import pathlib
import sys
import types
from fractions import Fraction

import numpy as np
import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_frame_sampler_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_frame_sampler_test.helpers import image_helpers
from utils_collection_video_frame_sampler_test.nodes import utils_nodes
from utils_collection_video_frame_sampler_test.helpers.image_helpers import (
    VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE,
    VIDEO_TIMELINE_TEXT_STRUCTURE,
    VideoFrameRecord,
    build_structured_video_timeline_text,
    build_text_video_timeline_text,
    build_video_timeline_text,
    format_video_timestamp,
    images_to_video_timeline,
    parse_video_timestamp,
    parse_video_timestamps,
    sample_video_frames_as_images,
    scan_video_frame_records,
    select_video_frame_records,
    video_timeline_text,
    focused_timeline_timestamps,
)


@pytest.mark.parametrize("value, expected", [
    ("01:02:03.250", Fraction(14893, 4)),
    ("02:03:250", Fraction(493, 4)),
    ("02:03.250", Fraction(493, 4)),
    ("00.125s", Fraction(1, 8)),
    (1.25, Fraction(5, 4)),
])
def test_parse_video_timestamp_formats(value, expected):
    assert parse_video_timestamp(value) == expected


def test_parse_video_timestamps_flattens_delimited_and_nested_values():
    assert parse_video_timestamps([["0; 1.2"], ("2.5s",)]) == [Fraction(0), Fraction(6, 5), Fraction(5, 2)]


@pytest.mark.parametrize("value", [True, -1, float("inf"), "", "1;;2", "00:00:60.0"])
def test_parse_video_timestamps_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_video_timestamps(value)


def test_parse_video_timestamps_rejects_decreasing_order():
    with pytest.raises(ValueError, match="earlier"):
        parse_video_timestamps([1, 0])
from utils_collection_video_frame_sampler_test.nodes.image_nodes import (
    UC_ImagesToVideoTimeline,
    UC_SampleVideoFramesAsImages,
    UC_VideoTimelineText,
)


def _records(times, keyframes=None):
    if keyframes is None:
        keyframes = [True] * len(times)
    return [
        VideoFrameRecord(index, Fraction(str(timestamp)), key_frame)
        for index, (timestamp, key_frame) in enumerate(zip(times, keyframes))
    ]


def test_schema_exposes_batch_list_and_aligned_metadata_outputs():
    schema = UC_SampleVideoFramesAsImages.define_schema()

    assert schema.node_id == "UC_SampleVideoFramesAsImages"
    assert schema.display_name == "Sample Video Frames (Images)"
    assert [value.id for value in schema.inputs] == [
        "video",
        "sampling_strategy",
        "maximum_frames",
        "focus_areas",
        "focus_one",
        "focus_two",
        "focus_three",
        "include_zero_time",
        "minimum_spacing_seconds",
        "keyframe_stride",
        "timestamp_format",
        "timeline_style",
        "timeline_text_structure",
        "structured_timeline_text_structure",
        "index_offset",
    ]
    assert schema.inputs[1].default == "codec keyframes"
    assert schema.inputs[2].default == 16
    assert schema.inputs[3].default == schema.inputs[3].min == 0
    assert schema.inputs[3].max == 3
    assert [value.default for value in schema.inputs[4:7]] == [0.5, 0.5, 0.5]
    assert schema.inputs[7].default is True
    assert schema.inputs[8].default == 0.25
    assert schema.inputs[9].default == 1
    assert schema.inputs[10].default == "00.000s"
    assert "0.0s" in schema.inputs[10].options
    assert "0.00s" in schema.inputs[10].options
    assert "00.00s" not in schema.inputs[10].options
    assert schema.inputs[11].default == "H3 alignment prefix"
    assert schema.inputs[12].default == VIDEO_TIMELINE_TEXT_STRUCTURE
    assert schema.inputs[13].default == VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE
    assert schema.inputs[14].default == 0
    assert schema.inputs[14].min == 0
    assert [output.id for output in schema.outputs] == [
        "image_batch",
        "timestamps_text",
        "timeline_text",
        "video_runtime",
        "structured_timeline_text",
    ]
    assert [output.is_output_list for output in schema.outputs] == [
        False,
        False,
        False,
        False,
        False,
    ]


def test_images_to_video_timeline_schema_exposes_manual_focus_controls():
    schema = UC_ImagesToVideoTimeline.define_schema()

    assert schema.node_id == "UC_ImagesToVideoTimeline"
    assert schema.display_name == "Images to Video Timeline"
    assert [value.id for value in schema.inputs] == [
        "duration",
        "focus_areas",
        "focus_one",
        "focus_two",
        "focus_three",
        "last_image_is_final",
        "resize_images",
        "timestamp_format",
        "timeline_style",
        "timeline_text_structure",
        "structured_timeline_text_structure",
        "index_offset",
        "image_inputs",
    ]
    assert schema.inputs[0].default == 5.0
    assert schema.inputs[1].default == schema.inputs[1].min == 0
    assert schema.inputs[1].max == 3
    assert [value.default for value in schema.inputs[2:5]] == [0.5, 0.5, 0.5]
    assert schema.inputs[5].default is False
    assert schema.inputs[6].default is True
    assert [output.id for output in schema.outputs] == [
        "image_batch",
        "timestamps_text",
        "timeline_text",
        "video_runtime",
        "structured_timeline_text",
    ]


def test_text_only_video_timeline_schema_and_outputs():
    schema = UC_VideoTimelineText.define_schema()

    assert schema.node_id == "UC_VideoTimelineText"
    assert [value.id for value in schema.inputs] == [
        "duration",
        "segment_count",
        "focus_areas",
        "focus_one",
        "focus_two",
        "focus_three",
        "timestamp_format",
        "timeline_text_structure",
        "structured_timeline_text_structure",
        "video",
    ]
    assert [output.id for output in schema.outputs] == [
        "timestamps_text",
        "timeline_text",
        "video_runtime",
        "structured_timeline_text",
    ]
    assert video_timeline_text(
        10.0,
        5,
        0,
        0.5,
        0.5,
        0.5,
        "0.0s",
        "<<shot>> at <<timestamp>>",
        "Target video duration is <<duration>> seconds divided into <<segments>> segments at <<timestamps>>",
    ) == (
        "0.0s, 2.5s, 5.1s, 7.6s, 10.1s",
        "Shot 1 at 0.0s\nShot 2 at 2.5s\nShot 3 at 5.1s\n"
        "Shot 4 at 7.6s\nShot 5 at 10.1s",
        10.125,
            "Target video duration is 10.125 seconds divided into 5 segments at "
        "0.0s, 2.5s, 5.1s, 7.6s, 10.1s",
    )
    assert video_timeline_text(
        3.0, 3, 0, 0.5, 0.5, 0.5, "0.0s",
        VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE,
        VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )[3] == (
            "Target video duration is 3.04167 seconds divided into 3 segments. "
        "Shot 1 at 0.0s, Shot 2 at 1.5s, Shot 3 at 3.0s."
    )

    class FakeVideo:
        @staticmethod
        def get_duration():
            return 8.0

    output = UC_VideoTimelineText.execute(
        segment_count=3,
        focus_areas=0,
        focus_one=0.5,
        focus_two=0.5,
        focus_three=0.5,
        timestamp_format="0.0s",
        timeline_text_structure=VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE,
        structured_timeline_text_structure=VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
        duration=99.0,
        video=FakeVideo(),
    )
    assert output[0] == "0.0s, 4.0s, 8.0s"
    assert output[2] == 8.0
    assert output[3] == (
        "Target video duration is 8 seconds divided into 3 segments. "
        "Shot 1 at 0.0s, Shot 2 at 4.0s, Shot 3 at 8.0s."
    )


def test_timeline_image_timestamps_anchor_ends_and_apply_local_focuses():
    timestamps = focused_timeline_timestamps(9, 12.0, 2, 0.25, 0.75, 0.5)

    assert timestamps[0] == 0.0
    assert timestamps[-1] == 12.0
    assert timestamps == sorted(timestamps)
    assert timestamps[2] < 3.0
    assert timestamps[-3] > 9.0


def test_focused_pts_selects_source_frames_near_local_focus_targets():
    records = _records(range(11))
    early = select_video_frame_records(records, "focused PTS", 4, True, 0, 1, 1, 0.1, 0.5, 0.5)
    late = select_video_frame_records(records, "focused PTS", 4, True, 0, 1, 1, 0.9, 0.5, 0.5)
    without_zero = select_video_frame_records(records, "focused PTS", 3, False, 0, 1)
    all_frames = select_video_frame_records(records, "focused PTS", 0, True, 0, 1, 3, 0.1, 0.5, 0.9)

    assert [record.frame_index for record in early] == [0, 1, 2, 10]
    assert [record.frame_index for record in late] == [0, 8, 9, 10]
    assert [record.frame_index for record in without_zero] == [2, 5, 7]
    assert [record.frame_index for record in all_frames] == list(range(11))


def test_images_to_video_timeline_normalizes_or_returns_black_batch():
    first = torch.zeros((2, 2, 4, 3))
    second = torch.ones((1, 4, 2, 4))
    inputs = {"image0": first, "image1": second}

    resized = images_to_video_timeline(
        inputs, 4.0, 0, 0.5, 0.5, 0.5, True, True,
        "00.000s", "custom", "<<time>>", VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )
    unresized = images_to_video_timeline(
        inputs, 4.0, 0, 0.5, 0.5, 0.5, True, False,
        "00.000s", "custom", "<<time>>", VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )
    final_image_not_anchored = images_to_video_timeline(
        inputs, 4.0, 0, 0.5, 0.5, 0.5, False, True,
        "00.000s", "custom", "<<time>>", VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )

    assert resized.image_batch.shape == (3, 2, 4, 4)
    assert [image.shape for image in resized.image_list] == [(1, 2, 4, 4)] * 3
    assert resized.timestamps == ["00.000s", "02.229s", "04.458s"]
    assert final_image_not_anchored.timestamps == ["00.000s", "01.486s", "02.972s"]
    assert unresized.image_batch.shape == (1, 64, 64, 3)
    assert torch.count_nonzero(unresized.image_batch) == 0
    assert [image.shape for image in unresized.image_list] == [
        (1, 2, 4, 3),
        (1, 2, 4, 3),
        (1, 4, 2, 4),
    ]


def test_structured_timeline_text_describes_duration_segments_and_references():
    assert build_structured_video_timeline_text(
        12.3,
        [
            "00.00s",
            "01.21s",
            "02.46s",
            "05.30s",
            "06.55s",
            "07.80s",
            "10.84s",
            "12.10s",
        ],
    ) == (
        "Target video duration is 12.3 seconds divided into 8 segments. "
        "Reference each image with <Picture 1> at 00.00s, "
        "<Picture 2> at 01.21s, <Picture 3> at 02.46s, "
        "<Picture 4> at 05.30s, <Picture 5> at 06.55s, "
        "<Picture 6> at 07.80s, <Picture 7> at 10.84s and "
        "<Picture 8> at 12.10s."
    )


def test_structured_image_timeline_can_repeat_shots_without_picture_references():
    assert build_structured_video_timeline_text(
        13.9676,
        ["0.00s", "6.98s", "13.97s"],
        (
            "Target video duration is <<duration>> seconds divided into "
            "<<segments>> segments. <<shot>> at <<timestamp>>."
        ),
    ) == (
        "Target video duration is 13.9676 seconds divided into 3 segments. "
        "Shot 1 at 0.00s, Shot 2 at 6.98s, Shot 3 at 13.97s."
    )


def test_video_timeline_placeholder_aliases_are_consistent():
    timestamps = ["0.00s", "1.25s"]

    assert build_video_timeline_text(
        timestamps, "custom", "<<shot>> at <<timestamp>> (<<time>>)"
    ) == "[Shot 1] at 0.00s (0.00s)\n[Shot 2] at 1.25s (1.25s)"
    assert build_text_video_timeline_text(
        timestamps, "<<shot>> at <<time>> (<<timestamp>>)"
    ) == "Shot 1 at 0.00s (0.00s)\nShot 2 at 1.25s (1.25s)"
    assert build_structured_video_timeline_text(
        1.25,
        timestamps,
        "<<segments>> segments at <<timestamps>>.",
    ) == "2 segments at 0.00s, 1.25s."


def test_index_offset_shifts_all_picture_references():
    timestamps = ["00.00s", "01.21s"]

    assert build_structured_video_timeline_text(
        12.3, timestamps, VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE, 1
    ) == (
        "Target video duration is 12.3 seconds divided into 2 segments. "
        "Reference each image with <Picture 2> at 00.00s and "
        "<Picture 3> at 01.21s."
    )
    assert build_video_timeline_text(timestamps, "custom", "<<picture>> at <<time>>", 1) == (
        "<Picture 2> at 00.00s\n<Picture 3> at 01.21s"
    )
    assert build_video_timeline_text(timestamps, "H3 alignment prefix", VIDEO_TIMELINE_TEXT_STRUCTURE, 1) == (
        "For the target video, at 00.00s into the target video, "
        "<Picture 2> (from [Shot 1]) is fully referenced.\n"
        "For the target video, at 01.21s into the target video, "
        "<Picture 3> (from [Shot 2]) is fully referenced."
    )
def test_zero_time_is_output_position_zero_and_spacing_removes_near_duplicate():
    selected = select_video_frame_records(
        _records([0, 0.01, 0.25, 0.49, 0.5]),
        "codec keyframes",
        maximum_frames=0,
        include_zero_time=True,
        minimum_spacing_seconds=0.25,
        keyframe_stride=1,
    )

    assert [record.frame_index for record in selected] == [0, 2, 4]
    assert [record.timestamp for record in selected] == [
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
    ]


def test_keyframe_stride_precedes_even_full_timeline_count_limiting():
    selected = select_video_frame_records(
        _records(range(10)),
        "codec keyframes",
        maximum_frames=3,
        include_zero_time=True,
        minimum_spacing_seconds=0,
        keyframe_stride=2,
    )

    assert [record.frame_index for record in selected] == [0, 2, 8]


def test_uniform_pts_uses_irregular_presentation_times():
    records = _records([0, 0.1, 0.9, 2.1, 4.0])

    with_zero = select_video_frame_records(
        records,
        "uniform PTS",
        maximum_frames=3,
        include_zero_time=True,
        minimum_spacing_seconds=0,
        keyframe_stride=1,
    )
    without_zero = select_video_frame_records(
        records,
        "uniform PTS",
        maximum_frames=2,
        include_zero_time=False,
        minimum_spacing_seconds=0,
        keyframe_stride=1,
    )

    assert [record.frame_index for record in with_zero] == [0, 3, 4]
    assert [record.frame_index for record in without_zero] == [2, 3]


@pytest.mark.parametrize(
    ("timestamp_format", "expected"),
    [
        ("HH:MM:SS.mmm", "00:02:03.456"),
        ("HH:MM:SS:mmm", "00:02:03:456"),
        ("MM:SS.mmm", "02:03.456"),
        ("MM:SS:mmm", "02:03:456"),
        ("00.000s", "123.456s"),
        ("0.0s", "123.5s"),
        ("0.00s", "123.46s"),
    ],
)
def test_timestamp_formats_are_deterministic(timestamp_format, expected):
    assert format_video_timestamp(Fraction(123456, 1000), timestamp_format) == expected


def test_two_decimal_seconds_use_minimal_integer_width():
    assert format_video_timestamp(Fraction(0), "0.0s") == "0.0s"
    assert format_video_timestamp(Fraction(1234, 1000), "0.0s") == "1.2s"
    assert format_video_timestamp(Fraction(125, 100), "0.0s") == "1.3s"
    assert format_video_timestamp(Fraction(0), "0.00s") == "0.00s"
    assert format_video_timestamp(Fraction(1234, 1000), "0.00s") == "1.23s"
    assert format_video_timestamp(Fraction(12304, 1000), "0.00s") == "12.30s"
    assert format_video_timestamp(Fraction(1234, 1000), "00.000s") == "01.234s"


def test_timestamp_rounding_carries_into_the_next_minute():
    timestamp = Fraction(599996, 10000)

    assert format_video_timestamp(timestamp, "MM:SS.mmm") == "01:00.000"
    assert format_video_timestamp(timestamp, "MM:SS:mmm") == "01:00:000"


def test_timeline_reuses_formatted_timestamp_literals_without_unit_rewriting():
    timestamps = ["00.000s", "2.50s"]

    assert build_video_timeline_text(timestamps, "custom", "<<time>>") == (
        "00.000s\n2.50s"
    )
    assert build_video_timeline_text(timestamps, "indexed", "unused") == (
        "0: 00.000s\n1: 2.50s"
    )
    assert build_video_timeline_text(timestamps, "H3 pictures", "unused") == (
        "<Picture 1> at 00.000s\n<Picture 2> at 2.50s"
    )
    assert build_video_timeline_text(timestamps, "H3 alignment prefix", "unused") == (
        "For the target video, at 00.000s into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.\n"
        "For the target video, at 2.50s into the target video, "
        "<Picture 2> (from [Shot 2]) is fully referenced."
    )


class _FakeFrame:
    def __init__(self, pts, key_frame, value, time_base=Fraction(1, 1000)):
        self.pts = pts
        self.time_base = time_base
        self.key_frame = key_frame
        self.rotation = 0
        self._value = value

    def to_ndarray(self, format):
        assert format == "rgb24"
        return np.full((2, 3, 3), self._value, dtype=np.uint8)


class _FakeStream:
    def __init__(self, start_time=0, time_base=Fraction(1, 1000)):
        self.start_time = start_time
        self.time_base = time_base


class _FakeContainer:
    def __init__(self, frames, stream):
        self._frames = frames
        self.streams = types.SimpleNamespace(video=[stream])
        self.seek_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def seek(self, pts, stream, backward, any_frame):
        self.seek_calls.append((pts, stream, backward, any_frame))

    def decode(self, stream):
        assert stream is self.streams.video[0]
        return iter(self._frames)


class _FakeVideo:
    def __init__(self, start_time=0.0, duration=0.0, runtime=1.25):
        self._trim = (start_time, duration)
        self._runtime = runtime
        self.source_calls = 0

    def get_stream_source(self):
        self.source_calls += 1
        return stdlib_io.BytesIO()

    def get_active_trim_window(self):
        return self._trim

    def get_duration(self):
        return self._runtime


def test_pts_scan_normalizes_the_first_visible_trim_frame(monkeypatch):
    frames = [
        _FakeFrame(1400, True, 1),
        _FakeFrame(1500, False, 2),
        _FakeFrame(1750, True, 3),
        _FakeFrame(2499, True, 4),
        _FakeFrame(2500, True, 5),
    ]
    stream = _FakeStream(start_time=1000)
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    records = scan_video_frame_records(_FakeVideo(start_time=0.5, duration=1.0))

    assert [record.frame_index for record in records] == [0, 1, 2]
    assert [record.timestamp for record in records] == [
        Fraction(0),
        Fraction(1, 4),
        Fraction(999, 1000),
    ]
    assert [record.key_frame for record in records] == [False, True, True]


def test_sampler_outputs_aligned_batch_list_and_metadata(monkeypatch):
    frames = [
        _FakeFrame(0, True, 0),
        _FakeFrame(10, True, 10),
        _FakeFrame(250, True, 20),
        _FakeFrame(1000, True, 30),
    ]
    stream = _FakeStream()
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    video = _FakeVideo()
    sampled = sample_video_frames_as_images(
        video,
        "codec keyframes",
        maximum_frames=0,
        include_zero_time=True,
        minimum_spacing_seconds=0.25,
        keyframe_stride=1,
        timestamp_format="00.000s",
        timeline_style="custom",
        timeline_text_structure="<<picture>> at <<time>>",
        structured_timeline_text_structure=VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )

    assert sampled.image_batch.shape == (3, 2, 3, 3)
    assert len(sampled.image_list) == 3
    assert all(image.shape == (1, 2, 3, 3) for image in sampled.image_list)
    assert sampled.image_list[1].data_ptr() == sampled.image_batch[1:2].data_ptr()
    assert sampled.timestamps == ["00.000s", "00.250s", "01.000s"]
    assert sampled.timestamps_text == "00.000s, 00.250s, 01.000s"
    assert sampled.timeline_text == (
        "<Picture 1> at 00.000s\n"
        "<Picture 2> at 00.250s\n"
        "<Picture 3> at 01.000s"
    )
    assert sampled.video_runtime == 1.625
    assert sampled.structured_timeline_text == (
        "Target video duration is 1.625 seconds divided into 3 segments. "
        "Reference each image with <Picture 1> at 00.000s, "
        "<Picture 2> at 00.250s and <Picture 3> at 01.000s."
    )
    assert torch.allclose(sampled.image_batch[1], torch.full((2, 3, 3), 20 / 255))
    assert video.source_calls == 1


def test_single_decimal_timestamps_stay_aligned_with_selected_images(monkeypatch):
    frames = [
        _FakeFrame(0, True, 0),
        _FakeFrame(250, True, 20),
        _FakeFrame(999, True, 30),
    ]
    stream = _FakeStream()
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    sampled = sample_video_frames_as_images(
        _FakeVideo(),
        "codec keyframes",
        maximum_frames=0,
        include_zero_time=True,
        minimum_spacing_seconds=0,
        keyframe_stride=1,
        timestamp_format="0.0s",
        timeline_style="custom",
        timeline_text_structure="<<time>>",
        structured_timeline_text_structure=VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )

    assert sampled.timestamps == ["0.0s", "0.3s", "1.0s"]
    assert torch.allclose(sampled.image_batch[0], torch.zeros((2, 3, 3)))
    assert torch.allclose(sampled.image_batch[1], torch.full((2, 3, 3), 20 / 255))
    assert torch.allclose(sampled.image_batch[2], torch.full((2, 3, 3), 30 / 255))


def test_uniform_pts_selects_frames_after_rounding_targets(monkeypatch):
    frames = [
        _FakeFrame(0, True, 0),
        _FakeFrame(360, False, 10),
        _FakeFrame(410, False, 20),
        _FakeFrame(740, False, 30),
    ]
    stream = _FakeStream()
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    sampled = sample_video_frames_as_images(
        _FakeVideo(runtime=0.75),
        "uniform PTS",
        maximum_frames=3,
        include_zero_time=True,
        minimum_spacing_seconds=0,
        keyframe_stride=1,
        timestamp_format="0.0s",
        timeline_style="custom",
        timeline_text_structure="<<time>>",
        structured_timeline_text_structure=VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )

    assert sampled.timestamps == ["0.0s", "0.4s", "0.7s"]
    assert torch.allclose(sampled.image_batch[0], torch.zeros((2, 3, 3)))
    assert torch.allclose(sampled.image_batch[1], torch.full((2, 3, 3), 20 / 255))
    assert torch.allclose(sampled.image_batch[2], torch.full((2, 3, 3), 30 / 255))


def test_sampler_rejects_timestamp_shift_between_selection_and_decode(monkeypatch):
    scan_frames = [_FakeFrame(0, True, 0), _FakeFrame(250, True, 20)]
    decode_frames = [_FakeFrame(0, True, 0), _FakeFrame(251, True, 20)]
    stream = _FakeStream()
    calls = iter((scan_frames, decode_frames))
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(next(calls), stream),
    )

    with pytest.raises(
        ValueError,
        match="Video timestamps changed between selection and decoding",
    ):
        sample_video_frames_as_images(
            _FakeVideo(),
            "codec keyframes",
            maximum_frames=0,
            include_zero_time=True,
            minimum_spacing_seconds=0,
            keyframe_stride=1,
            timestamp_format="0.0s",
            timeline_style="custom",
            timeline_text_structure="<<time>>",
            structured_timeline_text_structure=VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
        )


@pytest.mark.parametrize("source_rate,source_count", [(24, 778), (30, 972)])
def test_h3_indexed_segments_cover_source_without_borrowing_padding(source_rate, source_count):
    components = types.SimpleNamespace(
        images=torch.arange(source_count, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3),
        frame_rate=source_rate,
        audio={"waveform": torch.arange(round(source_count / source_rate * 32000), dtype=torch.float32).view(1, 1, -1), "sample_rate": 32000},
    )
    total = round(source_count / source_rate * 24)
    starts = [0, 260, 520]
    stops = [260, 520, total]
    lengths = [260, 260, 260]
    paddings = [0, 0, 2]
    for index in range(3):
        frames, audio, _, _, length, _, preview = image_helpers.prepare_h3_reference_components(
            components, 0.01, duration_seconds=999, start_at_timestamp=999,
            spatially_prepared=True, segment_count=3, segment_index=index,
        )
        start, stop = starts[index], stops[index]
        assert preview["start_frame"] == start
        assert preview["source_end_frame"] == stop - 1
        assert preview["padded_frames"] == paddings[index]
        assert length == lengths[index]
        expected = [min(round(frame * source_rate / 24), source_count - 1) for frame in range(start, stop)]
        assert frames[:stop - start, 0, 0, 0].tolist() == expected
        if paddings[index]:
            assert frames[stop - start:, 0, 0, 0].eq(expected[-1]).all()
        audio_stop = min(round(stop / 24 * 32000), components.audio["waveform"].shape[-1])
        samples = audio_stop - round(start / 24 * 32000)
        assert audio["waveform"][0, 0, 0] == round(start / 24 * 32000)
        assert audio["waveform"][0, 0, samples - 1] == audio_stop - 1
        if paddings[index]:
            assert audio["waveform"][..., samples:].eq(0).all()


def test_h3_reference_node_selects_zero_based_segment(monkeypatch):
    components = types.SimpleNamespace(
        images=torch.arange(72, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3),
        frame_rate=24, audio=None,
    )
    monkeypatch.setattr(utils_nodes, "cached_h3_reference_components", lambda *args: components)
    output = utils_nodes.UC_MiniMaxH3RefVid.execute(
        object(), segment_count=3, segment_index=1, enable_whisper=False,
    )
    assert output.args[0][0, 0, 0, 0] == 22
    assert output.args[0][-1, 0, 0, 0] == 43
    assert output.args[5].get_components().images is output.args[0]
    assert output.args[5].get_components().audio is output.args[1]
    with pytest.raises(ValueError, match="segment index"):
        utils_nodes.UC_MiniMaxH3RefVid.execute(object(), segment_count=3, segment_index=3)


def test_h3_reference_node_is_changed_tracks_segment_and_continuation():
    node = utils_nodes.UC_MiniMaxH3RefVid
    val0 = node.IS_CHANGED(object(), segment_count=3, segment_index=0)
    val1 = node.IS_CHANGED(object(), segment_count=3, segment_index=1)
    assert val0 != val1
    cont = {"format_version": 1, "frame_rate": 24, "frames": torch.zeros(5, 16, 16, 3), "video_merge_mode": "replace"}
    val_cont = node.IS_CHANGED(object(), segment_count=3, segment_index=0, continuation_media=cont)
    assert val0 != val_cont


@pytest.mark.parametrize("merge_mode", ["replace", "prepend"])
def test_h3_reference_node_merges_continuation_media(monkeypatch, merge_mode):
    components = types.SimpleNamespace(
        images=torch.arange(72, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3),
        frame_rate=24,
        audio={"waveform": torch.arange(72000, dtype=torch.float32).view(1, 1, -1), "sample_rate": 24000},
    )
    monkeypatch.setattr(utils_nodes, "cached_h3_reference_components", lambda *args: components)
    cont_frames = torch.full((5, 32, 32, 3), 99.0)
    cont_audio = {"waveform": torch.full((1, 1, 10000), 88.0), "sample_rate": 32000}
    continuation = {
        "format_version": 1, "frame_rate": 24,
        "frames": cont_frames, "audio": cont_audio,
        "video_merge_mode": merge_mode,
    }
    output = utils_nodes.UC_MiniMaxH3RefVid.execute(
        object(), segment_count=3, segment_index=0, enable_whisper=False, continuation_media=continuation,
    )
    frames = output.args[0]
    assert frames[:5].eq(99.0).all()
    if merge_mode == "replace":
        assert frames[5, 0, 0, 0] == 5.0
        assert frames.shape[0] == 22
    else:
        assert frames[5, 0, 0, 0] == 0.0
        assert frames.shape[0] == 22
    audio = output.args[1]
    tail_samples = round(5 / 24 * 32000)
    torch.testing.assert_close(audio["waveform"][..., :tail_samples], torch.full((1, 1, tail_samples), 88.0))


def test_h3_reference_node_audio_only_continuation_media(monkeypatch):
    components = types.SimpleNamespace(
        images=torch.arange(72, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3),
        frame_rate=24,
        audio={"waveform": torch.arange(72000, dtype=torch.float32).view(1, 1, -1), "sample_rate": 24000},
    )
    monkeypatch.setattr(utils_nodes, "cached_h3_reference_components", lambda *args: components)
    cont_audio = {"waveform": torch.full((1, 1, 10000), 88.0), "sample_rate": 32000}
    continuation = {
        "format_version": 1, "frame_rate": 24,
        "frames": None, "audio": cont_audio,
        "video_merge_mode": "replace",
    }
    output = utils_nodes.UC_MiniMaxH3RefVid.execute(
        object(), segment_count=3, segment_index=0, enable_whisper=False, continuation_media=continuation,
    )
    frames = output.args[0]
    assert frames[0, 0, 0, 0] == 0.0
    audio = output.args[1]
    assert audio["waveform"][0, 0, 0] == 88.0


def test_h3_reference_node_video_only_continuation_prepends_source_audio(monkeypatch):
    components = types.SimpleNamespace(
        images=torch.arange(72, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3),
        frame_rate=24,
        audio={"waveform": torch.arange(72000, dtype=torch.float32).view(1, 1, -1), "sample_rate": 24000},
    )
    monkeypatch.setattr(utils_nodes, "cached_h3_reference_components", lambda *args: components)
    cont_frames = torch.full((5, 32, 32, 3), 99.0)
    continuation = {
        "format_version": 1, "frame_rate": 24,
        "frames": cont_frames, "audio": None,
        "video_merge_mode": "prepend",
    }
    output = utils_nodes.UC_MiniMaxH3RefVid.execute(
        object(), segment_count=3, segment_index=1, enable_whisper=False, continuation_media=continuation,
    )
    frames = output.args[0]
    assert frames[:5].eq(99.0).all()
    assert frames[5, 0, 0, 0] == 22.0
    audio = output.args[1]
    assert audio["waveform"][0, 0, 0] > 0
    assert not audio["waveform"].eq(0).all()


@pytest.mark.parametrize("tail", [5, 22, 39, 56])
def test_h3_reference_node_prepend_continuation_zero_padding(tail):
    components = types.SimpleNamespace(
        images=torch.zeros(240, 32, 32, 3), frame_rate=24, audio=None,
    )
    cont_frames = torch.zeros(tail, 32, 32, 3)
    continuation = {
        "format_version": 1, "frame_rate": 24, "frames": cont_frames,
        "video_merge_mode": "prepend",
    }
    frames, audio, _, _, length, _, preview = image_helpers.prepare_h3_reference_components(
        components, 0.5, segment_count=5, segment_index=1,
        continuation_media=continuation, spatially_prepared=True,
    )
    assert (frames.shape[0] - 5) % 17 == 0
    assert frames.shape[0] == length
    assert preview["padded_frames"] == 0


def test_h3_reference_components_fifty_second_five_segments_chain():
    total_frames = 1199
    source_rate = 24.0
    images = torch.arange(total_frames, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3)
    waveform = torch.arange(round(total_frames / 24 * 32000), dtype=torch.float32).view(1, 1, -1).repeat(1, 2, 1)
    components = types.SimpleNamespace(images=images, frame_rate=source_rate, audio={"waveform": waveform, "sample_rate": 32000})

    prev_end = -1
    for seg_idx in range(5):
        cont = None
        if seg_idx > 0:
            cont = {
                "format_version": 1, "frame_rate": 24,
                "frames": torch.full((22, 32, 32, 3), 999.0),
                "audio": {"waveform": torch.full((1, 2, 29600), 999.0), "sample_rate": 32000},
                "video_merge_mode": "prepend",
            }
        frames, audio, _, _, length, _, preview = image_helpers.prepare_h3_reference_components(
            components, 0.5, segment_count=5, segment_index=seg_idx,
            continuation_media=cont, spatially_prepared=True,
        )
        start = preview["start_frame"]
        end = preview["source_end_frame"]
        if seg_idx < 4:
            assert preview["padded_frames"] == 0
            assert (length - 5) % 17 == 0
        if seg_idx == 0:
            assert start == 0
            assert end == 242
            assert length == 243
        else:
            assert start == prev_end + 1
            if seg_idx < 4:
                assert length == 260
        prev_end = end
    assert prev_end == 1198


def test_h3_reference_components_windowed_decoding_matches_full():
    total_frames = 1199
    source_rate = 24.0
    full_seconds = total_frames / source_rate
    images = torch.arange(total_frames, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 32, 32, 3)
    waveform = torch.arange(round(full_seconds * 32000), dtype=torch.float32).view(1, 1, -1).repeat(1, 2, 1)
    full_comp = types.SimpleNamespace(images=images, frame_rate=source_rate, audio={"waveform": waveform, "sample_rate": 32000})

    for sc in (5, 6):
        for si in range(sc):
            cont = None
            if si > 0:
                cont = {
                    "format_version": 1, "frame_rate": 24,
                    "frames": torch.full((22, 32, 32, 3), 999.0),
                    "audio": None,
                    "video_merge_mode": "prepend",
                }
            # Full reference components call
            f_frames, f_audio, _, _, f_length, _, f_prev = image_helpers.prepare_h3_reference_components(
                full_comp, 0.5, segment_count=sc, segment_index=si,
                continuation_media=cont, spatially_prepared=True,
            )
            # Simulated windowed reference components call
            sf, fc, se, pt = image_helpers.resolve_h3_reference_window(full_seconds, 0.0, 0.0, sc, si, cont)
            end_frame = se if se is not None else min(total_frames, sf + fc)
            needs_prep = pt > 0 and (cont is None or cont.get("audio") is None)
            dec_start = max(0, sf - pt) if needs_prep else sf
            w_images = images[dec_start:end_frame]
            start_s = round(dec_start / 24 * 32000)
            end_s = round(end_frame / 24 * 32000)
            w_waveform = waveform[..., start_s:end_s]
            w_comp = types.SimpleNamespace(images=w_images, frame_rate=source_rate, audio={"waveform": w_waveform, "sample_rate": 32000})

            w_frames, w_audio, _, _, w_length, _, w_prev = image_helpers.prepare_h3_reference_components(
                w_comp, 0.5, segment_count=sc, segment_index=si,
                continuation_media=cont, spatially_prepared=True,
                full_source_seconds=full_seconds,
                window_start_frame=dec_start,
            )
            assert f_length == w_length
            assert (w_length - 5) % 17 == 0
            assert f_prev["padded_frames"] == w_prev["padded_frames"]
            assert f_prev["start_frame"] == w_prev["start_frame"]
            assert torch.equal(f_frames, w_frames)
            assert torch.equal(f_audio["waveform"], w_audio["waveform"])


def test_h3_segment_count_can_exceed_available_frames():
    components = types.SimpleNamespace(
        images=torch.ones(1, 32, 32, 3), frame_rate=24,
        audio={"waveform": torch.ones(1, 1, 1333), "sample_rate": 32000},
    )
    frames, audio, _, _, length, _, preview = image_helpers.prepare_h3_reference_components(
        components, 0.01, spatially_prepared=True, segment_count=3, segment_index=0,
    )
    assert frames.shape[0] == length == 5
    assert frames.eq(1).all()
    assert audio["waveform"][..., :1333].eq(1).all()
    assert audio["waveform"][..., 1333:].eq(0).all()
    assert preview["padded_frames"] == 4


def test_h3_reference_components_round_seconds_and_preserve_audio_start():
    frames = torch.linspace(0, 1, 100).view(100, 1, 1, 1).expand(100, 8, 16, 3)
    waveform = torch.zeros(1, 2, 441000)
    waveform[..., 0] = 1.0
    components = types.SimpleNamespace(images=frames, frame_rate=10, audio={"waveform": waveform, "sample_rate": 44100})
    video = types.SimpleNamespace(get_components=lambda: components)
    result = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01, duration_seconds=9.3112024)
    prepared, audio, width, height, length, combined_video, transcript = result.result
    assert transcript == ""
    assert (width, height, length) == (160, 64, 226)
    assert tuple(prepared.shape) == (226, 64, 160, 3)
    assert float(prepared[24].mean()) == pytest.approx(10 / 99, abs=1e-6)
    assert float(prepared[-1].mean()) == pytest.approx(94 / 99, abs=1e-6)
    assert audio["sample_rate"] == 32000
    assert audio["waveform"].shape == (1, 2, 301600)
    assert torch.all(audio["waveform"][..., 0] > 0.5)
    assert torch.count_nonzero(audio["waveform"][..., 301333:]) == 0
    schema = utils_nodes.UC_MiniMaxH3RefVid.GET_SCHEMA()
    assert [output.id for output in schema.outputs] == ["frames", "audio", "width", "height", "length", "video", "transcribed_audio"]
    combined = combined_video.get_components()
    assert combined.images is prepared
    assert combined.audio is audio
    assert combined.frame_rate == 24

    # Positive start offsets use H3 rounding and move video and audio together.
    waveform[..., 71662] = 1.0  # round((39 / 24) * 44100)
    shifted = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01, duration_seconds=2.0, start_at_timestamp=1.0)
    shifted_frames, shifted_audio, _, _, shifted_length, shifted_video, _ = shifted.result
    assert shifted_length == 56
    assert shifted_video.get_components().images is shifted_frames
    assert float(shifted_frames[0].mean()) == pytest.approx(16 / 99, abs=1e-6)
    assert torch.all(shifted_audio["waveform"][..., 0] > 0.5)
    assert shifted.ui == {"h3_reference_range": [{"start_frame": 39, "length": 56, "source_seconds": 10.0}]}
    remaining = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01, start_at_timestamp=1.0)
    assert remaining.result[4] == 209
    assert remaining.result[5].get_components().images is remaining.result[0]
    with pytest.raises(ValueError, match="past the end"):
        utils_nodes.UC_MiniMaxH3RefVid.execute(video, start_at_timestamp=10.0)

    components.audio = None
    components.images = frames[:5]
    components.frame_rate = 24
    short = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01)
    assert short.result[4] == 5
    assert short.result[1]["waveform"].shape[-1] == 7200
    assert torch.count_nonzero(short.result[1]["waveform"]) == 0

    # megapixels=0.0 keeps native frame dimensions without scaling or cropping.
    zero_mp = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.0)
    assert zero_mp.result[2] == 16
    assert zero_mp.result[3] == 8
    assert tuple(zero_mp.result[0].shape[1:3]) == (8, 16)


@pytest.mark.parametrize("timestamp_format, expected", [
    ("00.000s", "[00.400s–01.800s] Shake the bottle."),
    ("MM:SS.mmm", "[00:00.400–00:01.800] Shake the bottle."),
])
def test_h3_whisper_selected_audio_and_timestamp_format(monkeypatch, timestamp_format, expected):
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    frames = torch.zeros(120, 8, 16, 3)
    waveform = torch.arange(160000, dtype=torch.float32).reshape(1, 1, -1)
    source_audio = {"waveform": waveform, "sample_rate": 32000}
    components = types.SimpleNamespace(images=frames, frame_rate=24, audio=source_audio)
    decodes = []
    video = types.SimpleNamespace(get_components=lambda: (decodes.append(1), components)[1])
    model = object()
    seen = []

    def transcribe(patcher, audio, task, language, *, word_timestamps):
        assert word_timestamps is True
        seen.append(audio)
        assert patcher is model
        assert (task, language) == ("transcribe", "auto")
        assert audio["waveform"][0, 0, 0] == 52000  # Actual H3 offset: 39 / 24 seconds.
        return ["unused plain text"], ['[{"start":0.4,"end":1.8,"text":" Shake the bottle.","words":[{"word":" Shake","start":0.4,"end":0.7},{"word":" the","start":0.7,"end":0.8},{"word":" bottle.","start":0.8,"end":1.8}]}]'], ["en"]

    monkeypatch.setattr(speech, "run_whisper", transcribe)
    result = utils_nodes.UC_MiniMaxH3RefVid.execute(
        video, megapixels=0.01, start_at_timestamp=1.0,
        whisper_model=model, timestamp_format=timestamp_format,
    )
    assert result.result[6] == expected
    assert seen == [result.result[1]]
    assert seen[0] is result.result[5].get_components().audio
    assert decodes == [1]


def test_whisper_transcribe_node_formatted_transcription_output(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.nodes import model_nodes

    schema = model_nodes.UC_WhisperTranscribe.define_schema()
    output_ids = [value.id for value in schema.outputs]
    assert output_ids == ["text", "segments", "language", "formatted_transcription"]

    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": " world.", "start": 0.5, "end": 1.0},
    ]
    segments = [{"text": "Hello world.", "start": 0.0, "end": 1.0, "words": words}]
    monkeypatch.setattr(
        model_nodes, "run_whisper",
        lambda *args, **kwargs: (["Hello world."], [json.dumps(segments)], ["en"]),
    )
    audio = {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}
    output = model_nodes.UC_WhisperTranscribe.execute(object(), audio)
    assert output.args[0] == ["Hello world."]
    assert output.args[2] == ["en"]
    assert output.args[3] == ["[00.000s–01.000s] Hello world."]


@pytest.mark.parametrize("case", ["disconnected", "no_track", "empty_track", "disabled"])
def test_h3_whisper_skips_inference_without_model_or_source_audio(monkeypatch, case):
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    def forbidden(*args):
        pytest.fail("Whisper must not run when disabled, disconnected, or without source audio")

    monkeypatch.setattr(speech, "run_whisper", forbidden)
    source_audio = None if case == "no_track" else {
        "waveform": torch.zeros(1, 2, 0 if case == "empty_track" else 32000), "sample_rate": 32000,
    }
    video = types.SimpleNamespace(get_components=lambda: types.SimpleNamespace(
        images=torch.zeros(24, 8, 16, 3), frame_rate=24, audio=source_audio,
    ))
    result = utils_nodes.UC_MiniMaxH3RefVid.execute(
        video, megapixels=0.01, whisper_model=None if case == "disconnected" else object(),
        enable_whisper=case != "disabled",
    )
    assert result.result[6] == ""
    assert result.result[1]["waveform"].numel() > 0


def test_h3_whisper_empty_speech_and_errors(monkeypatch):
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    audio = {"waveform": torch.zeros(1, 1, 32000), "sample_rate": 32000}
    monkeypatch.setattr(speech, "run_whisper", lambda *args, **kwargs: ([""], ["[]"], ["en"]))
    assert speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 39) == ""

    def fail(*args, **kwargs):
        raise RuntimeError("transcription failed")

    monkeypatch.setattr(speech, "run_whisper", fail)
    with pytest.raises(RuntimeError, match="transcription failed"):
        speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 39)


def test_h3_transcript_groups_boundary_words_once_and_omits_padding(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    words = [
        {"word": "First", "start": 0, "end": 4 / 24},
        {"word": " crossing", "start": 4 / 24, "end": 9 / 24},
        {"word": " boundary", "start": 22 / 24, "end": 22 / 24},
        {"word": " last.", "start": 37 / 24, "end": 40 / 24},
        {"word": " synthetic", "start": 39 / 24, "end": 40 / 24},
    ]
    monkeypatch.setattr(speech, "run_whisper", lambda *a, **kw: ([""], [json.dumps([{"words": words}])], ["en"]))
    audio = {"waveform": torch.zeros(1, 1, 60000), "sample_rate": 32000}
    # "boundary" has both two- and three-syllable dictionary pronunciations.
    assert speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 39) == (
        "[00.000s–01.625s] First crossing boundary last."
    )
    assert speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 5) == "[00.000s–00.208s] First crossing"
    assert speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 22) == (
        "[00.000s–00.375s] First crossing"
    )


def test_h3_transcript_text_boundaries_across_segments(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    tokens = ['say', ' hello,”', ' then', ' I', " I'm", ' Alice', ' speaks.', ' “Next?”', ' 終わり。', ' 最後、']
    words = [{"word": token, "start": i / 2, "end": (i + 1) / 2} for i, token in enumerate(tokens)]
    segments = [{"words": words[:1]}, {"words": words[1:4]}, {"words": words[4:]}]
    monkeypatch.setattr(speech, "run_whisper", lambda *a, **kw: ([""], [json.dumps(segments)], ["ja"]))
    monkeypatch.setattr(speech, "reference_syllable_counts", lambda keys: pytest.fail("Non-English must not read dictionary"))
    audio = {"waveform": torch.ones(1), "sample_rate": 24}
    result = speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 240)
    assert result.splitlines() == [
        '[00.000s–01.000s] say hello,”', '[01.000s–01.500s] then',
        '[01.500s–02.000s] I', "[02.000s–02.500s] I'm", '[02.500s–03.500s] Alice speaks.',
        '[03.500s–04.000s] “Next?”', '[04.000s–04.500s] 終わり。', '[04.500s–05.000s] 最後、',
    ]
    assert ' '.join(line.split('] ', 1)[1] for line in result.splitlines()) == ''.join(tokens)


def test_h3_transcript_dictionary_counts_and_unknown_phrases(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    counts = speech.reference_syllable_counts({'disappear', 'poof', 'addicted', 'smoking', 'something', 'fire', 'every', 'zzzxq'})
    assert counts == {'disappear': 3, 'poof': 1, 'addicted': 3, 'smoking': 2, 'something': 2, 'fire': None, 'every': None}
    assert speech.reference_syllable_key(' “I’m,” ') == "i'm"
    phrases = [
        'Disappear just like poof,', " then she's gone", ' Addicted,',
        ' it starts with smoking something strong.',
        ' zzzxq it starts with smoking something strong.',
        ' fire it starts with smoking something strong.',
    ]
    tokens = (' '.join(phrases)).split()
    words = [{"word": ' ' + token, "start": i, "end": i + 1} for i, token in enumerate(tokens)]
    monkeypatch.setattr(speech, "run_whisper", lambda *a, **kw: ([""], [json.dumps([{"words": words}])], ["en"]))
    audio = {"waveform": torch.ones(1), "sample_rate": 24}
    result = speech.transcribe_reference_audio(object(), audio, audio, "00.000s", 2400)
    assert [line.split('] ', 1)[1] for line in result.splitlines()] == [
        'Disappear just like poof,', "then she's gone",
        'Addicted, it starts with', 'smoking something strong.',
        'zzzxq it starts with smoking something strong.',
        'fire it starts with smoking something strong.',
    ]


def test_h3_transcript_single_capital_comma_uses_syllable_grouping(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    # Synthetic word timings; the supplied sample contains phrase timings only.
    tokens = ['Donovan,', ' you', ' see', ' this?', ' Women,', ' oh', ' my', ' god,',
              " he's", ' so', ' smooth.', ' Women,', ' love', ' that,', ' oh', ' my', ' god.',
              ' Women,', ' I', ' see.', ' Women,']
    words = [{"word": token, "start": i / 2, "end": (i + 1) / 2} for i, token in enumerate(tokens)]
    monkeypatch.setattr(speech, 'run_whisper', lambda *a, **kw: (
        [''], [json.dumps([{"words": words[:1]}, {"words": words[1:]}])], ['en']))
    audio = {"waveform": torch.ones(1), "sample_rate": 24}
    result = speech.transcribe_reference_audio(object(), audio, audio, '00.000s', 2400)
    assert result.splitlines() == [
        '[00.000s–02.000s] Donovan, you see this?',
        '[02.000s–04.000s] Women, oh my god,',
        "[04.000s–05.500s] he's so smooth.",
        '[05.500s–07.000s] Women, love that,',
        '[07.000s–08.500s] oh my god.',
        '[08.500s–09.000s] Women,',
        '[09.000s–10.000s] I see.',
        '[10.000s–10.500s] Women,',
    ]
    assert ' '.join(line.split('] ', 1)[1] for line in result.splitlines()) == ''.join(tokens)


@pytest.mark.parametrize('counts, expected', [
    ([2, 2, 2, 2], [[0, 1], [2, 3]]),  # 4+4 beats greedy 6+2.
    ([3, 2, 3], [[0, 1], [2]]),  # 5+3 wins the tie against 3+5.
    ([7, 2, 2], [[0], [1, 2]]),  # Never split an indivisible long word.
    ([2, 2, 2], [[0, 1, 2]]),
])
def test_h3_syllable_partition_objective(counts, expected):
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    words = [{"word": str(i)} for i in range(len(counts))]
    groups = speech.partition_reference_phrase(words, {str(i): value for i, value in enumerate(counts)})
    assert [[int(word['word']) for word in group] for group in groups] == expected


@pytest.mark.parametrize('speech_end, expected', [
    (7.28, ''),
    (4.0, '[00.000s–04.000s] Thank you for watching!'),
    (2.0, '[00.000s–02.000s] Thank you for watching!'),
])
def test_h3_transcript_rejects_low_word_rate(monkeypatch, speech_end, expected):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    # Synthetic internal timings reproduce the reported phrase span, not its alignment.
    tokens = ['Thank', ' you', ' for', ' watching!']
    words = [{"word": token, "start": i * speech_end / 4, "end": (i + 1) * speech_end / 4}
             for i, token in enumerate(tokens)]
    monkeypatch.setattr(speech, 'run_whisper', lambda *a, **kw: ([''], [json.dumps([{"words": words}])], ['en']))
    audio = {"waveform": torch.ones(1), "sample_rate": 24}
    assert speech.transcribe_reference_audio(object(), audio, audio, '00.000s', 175) == expected
    words.extend([{"word": ' You', "start": 7.3, "end": 7.4},
                  {"word": ' see', "start": 7.4, "end": 7.5},
                  {"word": ' that?', "start": 7.5, "end": 7.62}])
    result = speech.transcribe_reference_audio(object(), audio, audio, '00.000s', 192)
    assert result == '\n'.join(filter(None, [expected, '[07.300s–07.620s] You see that?']))


def test_h3_transcript_word_rate_excludes_pauses(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    # Same text and outer span as the hallucination regression, but short words
    # separated by silence must survive. These are synthetic alignment fixtures.
    words = [
        {"word": 'Thank', "start": 0.0, "end": 0.3},
        {"word": ' you', "start": 0.3, "end": 0.5},
        {"word": ' for', "start": 6.5, "end": 6.7},
        {"word": ' watching!', "start": 6.7, "end": 7.28},
    ]
    monkeypatch.setattr(speech, 'run_whisper', lambda *a, **kw: ([''], [json.dumps([{"words": words}])], ['en']))
    audio = {"waveform": torch.ones(1), "sample_rate": 24}
    assert speech.transcribe_reference_audio(object(), audio, audio, '00.000s', 175) == (
        '[00.000s–07.280s] Thank you for watching!'
    )


def test_h3_transcript_clamps_crossing_words_and_keeps_zero_duration(monkeypatch):
    import json
    from utils_collection_video_frame_sampler_test.helpers import model_helpers as speech

    words = [
        {"word": "excluded", "start": -2, "end": -1},
        {"word": " crossing,", "start": -0.2, "end": 0.1},
        {"word": " point,", "start": 0.5, "end": 0.5},
        {"word": " end.", "start": 0.8, "end": 1.2},
        {"word": " padding", "start": 1, "end": 2},
        {"word": " \n", "start": 0.1, "end": 0.2},
    ]
    monkeypatch.setattr(speech, 'run_whisper', lambda *a, **kw: ([""], [json.dumps([{"words": words}])], ['de']))
    audio = {"waveform": torch.ones(1), "sample_rate": 24}
    assert speech.transcribe_reference_audio(object(), audio, audio, '00.000s', 24) == (
        '[00.000s–00.100s] crossing,\n[00.500s–00.500s] point,\n[00.800s–01.000s] end.'
    )


def test_h3_video_disk_cache_reuses_full_decode_across_ranges(tmp_path, monkeypatch):
    from utils_collection_video_frame_sampler_test.helpers import image_helpers as cache
    monkeypatch.setattr(cache, "_frame_storage", lambda video: "npz")

    monkeypatch.setattr(cache.folder_paths, "get_temp_directory", lambda: str(tmp_path / "cache"))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"encoded source video")
    frames = torch.linspace(0, 1, 100).view(100, 1, 1, 1).expand(100, 8, 16, 3).contiguous()
    audio = {"waveform": torch.arange(320000, dtype=torch.float32).reshape(1, 1, -1), "sample_rate": 32000}
    components = cache.Types.VideoComponents(images=frames, frame_rate=Fraction(10), audio=audio)
    calls = []

    def decode(video):
        calls.append(video)
        return components

    monkeypatch.setattr(cache.InputImpl.VideoFromFile, "get_components", decode)
    video = cache.InputImpl.VideoFromFile(str(source))
    raw_components = cache.cached_video_components(video)
    assert isinstance(raw_components.images, cache.CachedVideoFrames)
    assert len(calls) == 1
    resizes = []
    original_resize = image_helpers.comfy.utils.common_upscale

    def resize(*args):
        resizes.append((args[0].shape[0], args[0].dtype, args[0].device.type))
        return original_resize(*args)

    monkeypatch.setattr(image_helpers.comfy.utils, "common_upscale", resize)
    utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01, duration_seconds=2)
    assert sum(size for size, _, _ in resizes) == len(frames)
    assert all(dtype == torch.float32 and device == "cpu" for _, dtype, device in resizes)
    resize_calls = len(resizes)
    shifted = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01, duration_seconds=2, start_at_timestamp=1)
    assert len(calls) == 1
    assert len(resizes) == resize_calls  # Neither extraction nor resizing repeats for a new range.
    assert shifted.result[5].get_components().images is shifted.result[0]
    assert shifted.result[1]["waveform"][0, 0, 0] == round(39 / 24 * 32000)
    stored = image_helpers.cached_h3_reference_components(cache.InputImpl.VideoFromFile(str(source)), 0.01)
    assert stored.images.shape == (100, 64, 160, 3)
    torch.testing.assert_close(stored.audio["waveform"], audio["waveform"])
    assert stored.frame_rate == Fraction(10)
    copied = tmp_path / "copy.mp4"
    copied.write_bytes(source.read_bytes())
    image_helpers.cached_h3_reference_components(cache.InputImpl.VideoFromFile(str(copied)), 0.01)
    assert len(calls) == 1
    assert len(list((tmp_path / "cache" / "utilscollection_video_components" / "v3").glob("*.zip"))) == 1
    changed = image_helpers.cached_h3_reference_components(video, 0.02)
    assert changed.images.shape[1:3] != stored.images.shape[1:3]
    assert len(calls) == 1
    assert len(resizes) > resize_calls
    assert len(list((tmp_path / "cache" / "utilscollection_video_components" / "v3").glob("*.zip"))) == 2
    assert (tmp_path / "cache" / "utilscollection_video_components" / "v2").exists()


def test_h3_png_cache_reads_only_selected_frames_and_preserves_audio(tmp_path, monkeypatch):
    import zipfile
    from utils_collection_video_frame_sampler_test.helpers import image_helpers as cache

    frames = torch.arange(4 * 8 * 16 * 3).reshape(4, 8, 16, 3).remainder(256).float() / 255
    audio = {"waveform": torch.tensor([[[0.1, -0.2, 0.3]]]), "sample_rate": 32000}
    path = tmp_path / "frames.zip"
    cache._write_components(path, "test", frames, frames.shape, audio, Fraction(24), "png8")
    components = cache._read_components(path, "test")
    torch.testing.assert_close(components.audio["waveform"], audio["waveform"], rtol=0, atol=0)
    reads = []
    original = zipfile.ZipFile.read

    def read(archive, name, *args, **kwargs):
        reads.append(name)
        return original(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", read)
    expected = frames[[3, 1, 3]]

    def forbidden_stack(*args, **kwargs):
        pytest.fail("Cache selection must fill its final output, not stack retained frames")

    monkeypatch.setattr(torch, "stack", forbidden_stack)
    torch.testing.assert_close(components.images[[3, 1, 3]], expected, rtol=0, atol=0)
    assert reads == ["frames/00000003.png8", "frames/00000001.png8"]
    torch.testing.assert_close(components.images[-1], frames[-1], rtol=0, atol=0)
    assert components.images[0:0].shape == (0, 8, 16, 3)


def _encode_h3_fixture(path, codec, audio_tracks=0, variable_rate=False):
    import av

    with av.open(str(path), "w") as output:
        stream = output.add_stream(codec, rate=24)
        stream.width, stream.height = 70, 46
        stream.pix_fmt = {"mjpeg": "yuvj420p", "libx264": "yuv420p"}.get(codec, "bgr0")
        stream.time_base = Fraction(1, 24)
        audio_streams = [output.add_stream("pcm_s16le", rate=48000) for _ in range(audio_tracks)]
        for track in audio_streams:
            track.layout = "stereo"
        for index in range(18):
            pixels = np.arange(46 * 70 * 3, dtype=np.uint16).reshape(46, 70, 3)
            pixels = ((pixels + index * 17) % 256).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index + (index // 3 if variable_rate else 0)
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                output.mux(packet)
            for track_index, track in enumerate(audio_streams):
                samples = np.arange(index * 2000, (index + 1) * 2000, dtype=np.int64)
                samples = ((samples * (track_index + 1) * 13) % 30000).astype(np.int16)
                packed = np.column_stack((samples, -samples)).reshape(1, -1)
                audio_frame = av.AudioFrame.from_ndarray(packed, format="s16", layout="stereo")
                audio_frame.sample_rate = 48000
                audio_frame.pts = index * 2000
                audio_frame.time_base = Fraction(1, 48000)
                for packet in track.encode(audio_frame):
                    output.mux(packet)
        for track in [stream, *audio_streams]:
            for packet in track.encode(None):
                output.mux(packet)


@pytest.mark.parametrize("codec,tracks,trim,buffered,variable_rate,storage,rotation", [
    ("ffv1", 2, (0.13, 0.4), False, False, "npz", 0),
    ("mjpeg", 0, (0, 0), True, False, "png8", 0),
    ("ffv1", 1, (-0.4, 0), True, True, "png8", 0),
    ("libx264", 0, (0.13, 0.4), False, False, "npz", 90),
])
def test_h3_streamed_decode_matches_core(tmp_path, monkeypatch, codec, tracks, trim, buffered, variable_rate, storage, rotation):
    cache = image_helpers
    source = tmp_path / "source.mkv"
    _encode_h3_fixture(source, codec, tracks, variable_rate)
    if rotation:
        import av
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("FFmpeg CLI needed to create display-matrix rotation fixture")
        rotated = tmp_path / "rotated.mp4"
        subprocess.run([ffmpeg, "-v", "error", "-display_rotation:v:0", str(rotation), "-i", str(source), "-c", "copy", str(rotated)], check=True, capture_output=True)
        source = rotated
        with av.open(str(source)) as container:
            assert abs(next(container.decode(video=0)).rotation) == rotation
    source_value = stdlib_io.BytesIO(source.read_bytes()) if buffered else str(source)
    video = cache.InputImpl.VideoFromFile(source_value, start_time=trim[0], duration=trim[1], crop=(2, 3, 62, 39))
    baseline = video.get_components()
    frames = baseline.images
    ratio = min(cache.ASPECT_RATIOS.values(), key=lambda value: abs(frames.shape[2] / frames.shape[1] - value[0] / value[1]))
    width, height = cache.select_video_resolution(*ratio, 0.01, 32, 32, cache.MAX_RESOLUTION)
    expected = cache.comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "bicubic", "center").clamp(0, 1).movedim(1, -1)
    if storage == "png8":
        expected = expected.mul(255).round().div(255)

    def forbidden(*args, **kwargs):
        pytest.fail("Streamed construction must not call Core's full-clip decoder")

    monkeypatch.setattr(cache.InputImpl.VideoFromFile, "get_components", forbidden)
    monkeypatch.setattr(cache, "_h3_batch_capacity", lambda *args: 3)
    target = tmp_path / "streamed.zip"
    cache._write_streamed_h3_components(target, "fixture", video, 0.01, storage)
    actual = cache._read_components(target, "fixture")
    torch.testing.assert_close(actual.images[:], expected, rtol=0, atol=0)
    assert actual.frame_rate == baseline.frame_rate
    if baseline.audio is None:
        assert actual.audio is None
    else:
        assert actual.audio["sample_rate"] == baseline.audio["sample_rate"]
        torch.testing.assert_close(actual.audio["waveform"], baseline.audio["waveform"], rtol=0, atol=0)
    selected = cache.prepare_h3_reference_components(actual, 0.01, 0.2, spatially_prepared=True)
    oracle = cache.Types.VideoComponents(images=expected, audio=baseline.audio, frame_rate=baseline.frame_rate)
    expected_selection = cache.prepare_h3_reference_components(oracle, 0.01, 0.2, spatially_prepared=True)
    torch.testing.assert_close(selected[0], expected_selection[0], rtol=0, atol=0)
    torch.testing.assert_close(selected[1]["waveform"], expected_selection[1]["waveform"], rtol=0, atol=0)
    assert not list(tmp_path.glob(".*.tmp"))


def test_h3_streamed_writer_failure_joins_decoder(tmp_path, monkeypatch):
    import threading

    source = tmp_path / "source.mkv"
    _encode_h3_fixture(source, "ffv1", 1)
    monkeypatch.setattr(image_helpers, "_h3_batch_capacity", lambda *args: 1)

    def fail(*args, **kwargs):
        raise OSError("test disk failure")

    monkeypatch.setattr(image_helpers.cv2, "imencode", fail)
    target = tmp_path / "failed.zip"
    with pytest.raises(OSError, match="test disk failure"):
        image_helpers._write_streamed_h3_components(target, "fixture", image_helpers.InputImpl.VideoFromFile(str(source)), 0.01, "png8")
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert not any(thread.name == "h3-video-decode" for thread in threading.enumerate())


@pytest.mark.parametrize("failure", ["producer", "cancel"])
def test_h3_prefetch_propagates_failure_and_closes(failure):
    from contextlib import closing
    import threading

    closed = threading.Event()

    def produce(stopped):
        try:
            for _ in range(20):
                if stopped.is_set():
                    return
                yield torch.zeros(1)
            raise ValueError("test decoder failure")
        finally:
            closed.set()

    with pytest.raises(ValueError, match="test decoder failure|test cancellation"):
        with image_helpers._prefetch_h3_batches(lambda stopped: closing(produce(stopped))) as batches:
            for _ in batches:
                if failure == "cancel":
                    raise ValueError("test cancellation")
    assert closed.is_set()
    assert not any(thread.name == "h3-video-decode" for thread in threading.enumerate())


def test_h3_streamed_cache_reuses_ranges(tmp_path, monkeypatch):
    source = tmp_path / "source.mkv"
    _encode_h3_fixture(source, "ffv1", 1)
    monkeypatch.setattr(image_helpers.folder_paths, "get_temp_directory", lambda: str(tmp_path / "cache"))
    video = image_helpers.InputImpl.VideoFromFile(str(source))
    first = image_helpers.prepare_h3_reference_video_components(video, 0.01, 0.2)

    def forbidden(*args, **kwargs):
        pytest.fail("Range changes must reuse the prepared cache")

    monkeypatch.setattr(image_helpers, "_write_streamed_h3_components", forbidden)
    second = image_helpers.prepare_h3_reference_video_components(video, 0.01, 0.4)
    assert second[4] > first[4]


def test_h3_legacy_video_cache_migrates_without_decoding_source(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from utils_collection_video_frame_sampler_test.helpers import image_helpers as cache

    monkeypatch.setattr(cache.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    monkeypatch.setattr(cache, "_frame_storage", lambda video: "png8")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"encoded video")
    video = cache.InputImpl.VideoFromFile(str(source))
    key = cache.video_source_hash(video)
    legacy = tmp_path / "utilscollection_video_components" / "v1" / f"{key}.safetensors"
    legacy.parent.mkdir(parents=True)
    frames = torch.ones(3, 8, 8, 3)
    audio = torch.tensor([[[0.125, -0.75]]])
    save_file({"frames": frames, "audio": audio}, str(legacy), metadata={
        "source_hash": key, "frame_rate": "24", "sample_rate": "32000",
    })

    def forbidden(video):
        pytest.fail("Migration must not decode the source video")

    monkeypatch.setattr(cache.InputImpl.VideoFromFile, "get_components", forbidden)
    components = cache.cached_video_components(video)
    torch.testing.assert_close(components.images[:], frames, rtol=0, atol=0)
    torch.testing.assert_close(components.audio["waveform"], audio, rtol=0, atol=0)
    assert components.frame_rate == Fraction(24)
    assert legacy.exists()  # Explicit verified cleanup remains separate.


def test_h3_video_cache_invalidates_content_trim_crop_and_recovers(tmp_path, monkeypatch):
    from utils_collection_video_frame_sampler_test.helpers import image_helpers as cache
    monkeypatch.setattr(cache, "_frame_storage", lambda video: "npz")

    monkeypatch.setattr(cache.folder_paths, "get_temp_directory", lambda: str(tmp_path / "cache"))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    calls = []

    def decode(video):
        calls.append(video)
        return cache.Types.VideoComponents(images=torch.zeros(2, 4, 4, 3), frame_rate=Fraction(24), audio=None)

    monkeypatch.setattr(cache.InputImpl.VideoFromFile, "get_components", decode)
    video = cache.InputImpl.VideoFromFile(str(source))
    cache.cached_video_components(video)
    source.write_bytes(b"modified")  # Same path and byte count must not imply a hit.
    cache.cached_video_components(video)
    cache.cached_video_components(cache.InputImpl.VideoFromFile(str(source), start_time=1, duration=2))
    cache.cached_video_components(cache.InputImpl.VideoFromFile(str(source), crop=(0, 0, 4, 4)))
    cache.cached_video_components(cache.InputImpl.VideoFromFile(str(source), crop=(4, 0, 4, 4)))
    assert len(calls) == 5
    path = tmp_path / "cache" / "utilscollection_video_components" / "v2" / f"{cache.video_source_hash(video)}.zip"
    path.write_bytes(b"incomplete cache")
    restored = cache.cached_video_components(video)
    assert len(calls) == 6
    assert restored.audio is None
    cache.cached_video_components(video)
    assert len(calls) == 6
    # Memory-backed encoded video shares the file's content hash.
    buffer_video = cache.InputImpl.VideoFromFile(stdlib_io.BytesIO(source.read_bytes()))
    cache.cached_video_components(buffer_video)
    assert len(calls) == 6


def test_missing_pts_is_rejected_instead_of_estimated(monkeypatch):
    frames = [_FakeFrame(None, True, 0)]
    stream = _FakeStream()
    monkeypatch.setattr(
        image_helpers.av,
        "open",
        lambda source, mode: _FakeContainer(frames, stream),
    )

    with pytest.raises(ValueError, match="has no PTS"):
        scan_video_frame_records(_FakeVideo())
