import pathlib
import sys
import types

import av
from PIL import Image
import pytest


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_video_nodes_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_video_nodes_test.nodes import video_nodes


@pytest.fixture
def write_video():
    def write(path):
        with av.open(str(path), mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width = 16
            stream.height = 16
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame(16, 16, "rgb24")
            frame.planes[0].update(bytes(frame.planes[0].buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path
    return write


def test_load_video_path_schema_and_execution(monkeypatch, tmp_path, write_video):
    source = tmp_path / "clip.mp4"
    write_video(source)
    monkeypatch.setattr(video_nodes.InputImpl, "VideoFromFile", lambda path: ("VIDEO", path))

    schema = video_nodes.UC_LoadVideoPath.define_schema()
    assert schema.node_id == "UC_LoadVideoPath"
    assert schema.outputs[0].display_name == "VIDEO"
    output = video_nodes.UC_LoadVideoPath.execute(str(source))
    assert output.args == (("VIDEO", str(source.resolve())),)
    assert video_nodes.UC_LoadVideoPath.validate_inputs(str(source)) is True


def test_load_video_path_rejects_non_video_file(tmp_path):
    source = tmp_path / "not-video.txt"
    source.write_text("not video")
    with pytest.raises(ValueError, match="does not contain readable video"):
        video_nodes.UC_LoadVideoPath.execute(str(source))


def test_load_video_directory_returns_sorted_video_list(monkeypatch, tmp_path, write_video):
    for name in ("10.mp4", "2.webm"):
        write_video(tmp_path / name)
    (tmp_path / "ignore.png").write_bytes(b"not video")
    monkeypatch.setattr(video_nodes.InputImpl, "VideoFromFile", lambda path: path)

    schema = video_nodes.UC_LoadVideoDirectory.define_schema()
    assert schema.outputs[0].is_output_list is True
    output = video_nodes.UC_LoadVideoDirectory.execute(str(tmp_path), 0, 2)
    assert output.args == ([str((tmp_path / "10.mp4").resolve()), str((tmp_path / "2.webm").resolve())],)


def test_load_video_directory_validates_empty_selection(tmp_path):
    with pytest.raises(ValueError, match="No videos found"):
        video_nodes.UC_LoadVideoDirectory.execute(str(tmp_path), 0, 1)


def test_load_video_directory_uses_image_directory_compatibility_hooks():
    assert hasattr(video_nodes.UC_LoadVideoDirectory, "IS_CHANGED")
    assert hasattr(video_nodes.UC_LoadVideoDirectory, "VALIDATE_INPUTS")


@pytest.mark.parametrize("name", ["clip", "clip.bin", "clip.ogv", "clip.3gp", "clip.nut"])
def test_load_video_accepts_decodable_content_regardless_of_suffix(tmp_path, write_video, name):
    source = write_video(tmp_path / name)
    assert video_nodes.UC_LoadVideoPath.validate_inputs(str(source)) is True
    video = video_nodes.UC_LoadVideoPath.execute(str(source)).args[0]
    assert video.get_dimensions() == (16, 16)
    output = video_nodes.UC_LoadVideoDirectory.execute(str(tmp_path), 0, 1).args[0]
    assert len(output) == 1
    assert output[0].get_dimensions() == (16, 16)


def test_video_extension_does_not_make_invalid_content_readable(tmp_path):
    source = tmp_path / "invalid.mp4"
    source.write_bytes(b"not video")
    assert video_nodes.UC_LoadVideoPath.validate_inputs(str(source)) is not True
    with pytest.raises(ValueError, match="does not contain readable video"):
        video_nodes.UC_LoadVideoPath.execute(str(source))
    assert video_nodes.list_video_files(str(tmp_path)) == []


def test_save_video_schema_and_execution(tmp_path, monkeypatch):
    import torch
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    schema = video_nodes.UC_SaveVideo.define_schema()
    assert schema.node_id == "UC_SaveVideo"
    assert schema.is_output_node is True

    # 40 frames: frame 0-19 dark, frame 20-39 bright (cut at frame 20)
    images = torch.cat(
        [torch.full((20, 64, 64, 3), 0.1), torch.full((20, 64, 64, 3), 0.9)],
        dim=0,
    )
    result = video_nodes.UC_SaveVideo.execute(
        images=images,
        fps=24.0,
        codec="libx264",
        gop_size=100,
        b_frames=0,
        filename_prefix="test_save",
    )
    output_images, output_path, video_obj, audio_out = result.args
    assert output_images.shape == images.shape
    assert pathlib.Path(output_path).is_file()
    assert output_path.endswith("test_save_00001.mp4")
    assert video_obj is not None
    assert audio_out is None

    # Verify keyframes placed at frame 0 and frame 20
    with av.open(output_path) as container:
        stream = container.streams.video[0]
        key_frames = [
            round(float(packet.pts * packet.time_base * 24.0))
            for packet in container.demux(stream)
            if packet.is_keyframe
        ]
    assert 0 in key_frames
    assert 20 in key_frames

    # Next execution with audio: should increment counter to 00002 and append -audio
    audio = {
        "waveform": torch.zeros((1, 2, 44100), dtype=torch.float32),
        "sample_rate": 44100,
    }
    result_audio = video_nodes.UC_SaveVideo.execute(
        images=images,
        fps=24.0,
        codec="libx264",
        audio=audio,
        filename_prefix="test_save",
    )
    _, output_path_audio, video_obj_audio, audio_out_audio = result_audio.args
    assert output_path_audio.endswith("test_save_00002-audio.mp4")
    assert pathlib.Path(output_path_audio).is_file()
    assert pathlib.Path(output_path_audio.replace("-audio.mp4", ".png")).is_file()
    assert video_obj_audio is not None
    assert audio_out_audio is audio
    with av.open(output_path_audio) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert sum(1 for _ in container.demux(container.streams.audio[0])) > 0

    # Next execution without audio and save_workflow_image=False: should increment counter to 00003 without -audio or PNG
    result_no_audio = video_nodes.UC_SaveVideo.execute(
        images=images,
        fps=24.0,
        codec="libx264",
        filename_prefix="test_save",
        save_workflow_image=False,
    )
    _, output_path_no_audio, video_obj3, _ = result_no_audio.args
    assert output_path_no_audio.endswith("test_save_00003.mp4")
    assert not pathlib.Path(output_path_no_audio.replace(".mp4", ".png")).exists()

    # Default prefix creates formatted date structure and preserves workflow/prompt metadata
    monkeypatch.setattr(
        video_nodes.UC_SaveVideo,
        "hidden",
        types.SimpleNamespace(
            prompt={"1": {"class_type": "UC_SaveVideo"}},
            extra_pnginfo={"workflow": {"nodes": [{"id": 1, "type": "UC_SaveVideo"}]}},
        ),
        raising=False,
    )
    result_default = video_nodes.UC_SaveVideo.execute(
        images=images[:2],
        fps=24.0,
        codec="libx264",
    )
    _, output_path_default, _, _ = result_default.args
    assert "vid_" in output_path_default
    assert "_default_00001.mp4" in output_path_default
    workflow_png = pathlib.Path(output_path_default.replace(".mp4", ".png"))
    assert workflow_png.is_file()
    with Image.open(workflow_png) as saved_img:
        assert "prompt" in saved_img.info
        assert "workflow" in saved_img.info
        assert "UC_SaveVideo" in saved_img.info["workflow"]
    with av.open(output_path_default) as container:
        assert "prompt" in container.metadata
        assert "workflow" in container.metadata
        assert "UC_SaveVideo" in container.metadata["workflow"]

    # Test Concatenate Video (List)
    cat_result = video_nodes.UC_ConcatenateVideoList.execute([video_obj, video_obj_audio])
    cat_video = cat_result.args[0]
    assert cat_video is not None

    # Test saving the concatenated video with keyframe optimization through UC_SaveVideo
    save_cat = video_nodes.UC_SaveVideo.execute(
        video=cat_video,
        codec="libx264",
        filename_prefix="test_save_cat",
    )
    cat_frames, cat_path, cat_obj, cat_audio = save_cat.args
    assert cat_frames.shape[0] == 80
    assert pathlib.Path(cat_path).is_file()
    assert cat_audio is not None
