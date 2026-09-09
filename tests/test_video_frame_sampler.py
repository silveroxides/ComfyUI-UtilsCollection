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

from utils_collection_video_frame_sampler_test import image_helpers, utils_nodes
from utils_collection_video_frame_sampler_test.image_helpers import (
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
from utils_collection_video_frame_sampler_test.image_nodes import (
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
        "0.0s, 2.5s, 5.0s, 7.5s, 10.0s",
        "Shot 1 at 0.0s\nShot 2 at 2.5s\nShot 3 at 5.0s\n"
        "Shot 4 at 7.5s\nShot 5 at 10.0s",
        10.0,
        "Target video duration is 10 seconds divided into 5 segments at "
        "0.0s, 2.5s, 5.0s, 7.5s, 10.0s",
    )
    assert video_timeline_text(
        3.0, 3, 0, 0.5, 0.5, 0.5, "0.0s",
        VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE,
        VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    )[3] == (
        "Target video duration is 3 seconds divided into 3 segments. "
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
    assert resized.timestamps == ["00.000s", "02.000s", "04.000s"]
    assert final_image_not_anchored.timestamps == ["00.000s", "01.333s", "02.667s"]
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
    assert sampled.video_runtime == 1.25
    assert sampled.structured_timeline_text == (
        "Target video duration is 1.25 seconds divided into 3 segments. "
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


def test_h3_reference_components_round_seconds_and_preserve_audio_start():
    frames = torch.linspace(0, 1, 100).view(100, 1, 1, 1).expand(100, 8, 16, 3)
    waveform = torch.zeros(1, 2, 441000)
    waveform[..., 0] = 1.0
    components = types.SimpleNamespace(images=frames, frame_rate=10, audio={"waveform": waveform, "sample_rate": 44100})
    video = types.SimpleNamespace(get_components=lambda: components)
    result = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01, duration_seconds=9.3112024)
    prepared, audio, width, height, length = result.result
    assert (width, height, length) == (160, 64, 226)
    assert tuple(prepared.shape) == (226, 64, 160, 3)
    assert float(prepared[24].mean()) == pytest.approx(10 / 99, abs=1e-6)
    assert float(prepared[-1].mean()) == pytest.approx(94 / 99, abs=1e-6)
    assert audio["sample_rate"] == 32000
    assert audio["waveform"].shape == (1, 2, 301600)
    assert torch.all(audio["waveform"][..., 0] > 0.5)
    assert torch.count_nonzero(audio["waveform"][..., 301333:]) == 0
    schema = utils_nodes.UC_MiniMaxH3RefVid.GET_SCHEMA()
    assert [output.id for output in schema.outputs] == ["frames", "audio", "width", "height", "length"]

    components.audio = None
    components.images = frames[:5]
    components.frame_rate = 24
    short = utils_nodes.UC_MiniMaxH3RefVid.execute(video, megapixels=0.01)
    assert short.result[-1] == 5
    assert short.result[1]["waveform"].shape[-1] == 7200
    assert torch.count_nonzero(short.result[1]["waveform"]) == 0


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
