import json
import os
import pathlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from comfy_execution.graph import ExecutionBlocker
from comfy_api.latest import io
from comfy_api.latest._io import build_nested_inputs, get_finalized_class_inputs
from unifiedefficientloader import IncrementalSafetensorsWriter


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_minimax_h3_refs_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_minimax_h3_refs_test.helpers import model_helpers
from utils_collection_minimax_h3_refs_test.nodes import model_nodes, utils_nodes


def _image_ref(value=1.0):
    return {"kind": "image", "latent": torch.full((1, 24, 1, 4, 4), value), "metadata": {"description": "test"}}


def _video_ref(value=1.0):
    return {"kind": "video", "latent": torch.full((1, 24, 2, 4, 4), value), "metadata": {"description": "test"}}


def _ref_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_folder_paths", lambda _name: [str(tmp_path)])
    monkeypatch.setattr(model_helpers.folder_paths, "get_full_path_or_raise", lambda _name, filename: str(tmp_path / filename))


class _VisualVae:
    def __init__(self):
        self.inputs = []

    def encode(self, pixels):
        self.inputs.append(pixels.clone())
        frames = 1 if pixels.shape[0] == 1 else 2
        return pixels.mean().expand(1, 24, frames, pixels.shape[1] // 16, pixels.shape[2] // 16).clone()


def test_image_batch_fuses_prepared_vae_latents_into_one_ref():
    vae = _VisualVae()
    images = torch.cat((torch.zeros(4, 40, 72, 3), torch.ones(4, 40, 72, 3)))

    refs = model_helpers.create_minimax_h3_image_refs(images, vae, description="subject")

    assert len(refs) == 1
    assert [tuple(image.shape) for image in vae.inputs] == [(1, 32, 64, 3)] * 8
    torch.testing.assert_close(refs[0]["latent"], torch.full_like(refs[0]["latent"], 0.5))
    assert refs[0]["kind"] == "image"
    assert refs[0]["metadata"]["source_images"] == 8
    assert "fused 8 images" in model_helpers.format_minimax_h3_ref_info(refs)
    pooled = model_helpers.create_minimax_h3_image_refs(images, _VisualVae(), "pooled", 2)[0]
    single = model_helpers.create_minimax_h3_image_refs(images[:1], _VisualVae(), "pooled", 2)[0]
    assert pooled["latent"].shape == single["latent"].shape


def test_video_uses_native_five_plus_seventeen_frame_contract():
    vae = _VisualVae()
    video = torch.arange(24, dtype=torch.float32).view(24, 1, 1, 1).expand(24, 32, 64, 3) / 24
    ref = model_helpers.create_minimax_h3_video_ref(video, vae)

    # Core Lanczos goes through 8-bit PIL pixels, even at unchanged dimensions.
    torch.testing.assert_close(vae.inputs[0], video[:22], atol=1 / 255, rtol=0)
    assert ref["metadata"]["prepared_frames"] == 22
    assert ref["metadata"]["source_frames"] == 24
    with pytest.raises(ValueError, match="at least 5 frames"):
        model_helpers.create_minimax_h3_video_ref(video[:4], vae)
    assert len(vae.inputs) == 1


def test_audio_mono_is_duplicated_and_native_resampling_is_retained(monkeypatch):
    encoded, resampled = [], []
    audio_module = sys.modules[model_helpers._encode_minimax_h3_audio_reference.__module__]
    native_resample = audio_module.torchaudio.functional.resample

    def resample(waveform, source_rate, target_rate):
        resampled.append((source_rate, target_rate))
        return native_resample(waveform, source_rate, target_rate)

    def encode(samples):
        encoded.append(samples.clone())
        return torch.full((1, 32, 2, 4), 0.25)

    monkeypatch.setattr(audio_module.torchaudio.functional, "resample", resample)
    audio_vae = types.SimpleNamespace(audio_sample_rate=32000, encode=encode)
    waveform = torch.ones((1, 1, 16))
    ref = model_helpers.create_minimax_h3_audio_ref({"waveform": waveform, "sample_rate": 16000}, audio_vae)

    assert resampled == [(16000, 32000)]
    assert encoded[0].shape == (1, 32, 2)
    torch.testing.assert_close(encoded[0][..., 0], encoded[0][..., 1])
    torch.testing.assert_close(ref["latent"], torch.full((1, 32, 2, 4), 0.25))
    assert ref["metadata"]["sample_rate"] == 32000
    for invalid in (waveform.repeat(2, 1, 1), waveform.repeat(1, 3, 1)):
        with pytest.raises(ValueError, match="one finite mono or stereo"):
            model_helpers.create_minimax_h3_audio_ref({"waveform": invalid, "sample_rate": 16000}, audio_vae)
    assert len(encoded) == 1


def test_clip_continuation_save_load_overwrite_and_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    frames = torch.arange(56, dtype=torch.float32).view(56, 1, 1, 1).expand(56, 8, 8, 3).clone()

    relative = model_helpers.save_minimax_h3_clip_continuation_media(
        frames, 22, "h3_clip_continuation/clip", 1
    )
    assert relative == "h3_clip_continuation/clip_00001.safetensors"
    first = model_helpers.load_minimax_h3_clip_continuation_media(
        "h3_clip_continuation/clip", 1
    )
    assert first["frame_rate"] == 24
    torch.testing.assert_close(first["frames"], frames[-22:, ..., :3])
    fingerprint = model_helpers.get_minimax_h3_clip_continuation_fingerprint(
        "h3_clip_continuation/clip", 1
    )

    replacement = torch.full((22, 8, 8, 3), 0.5)
    model_helpers.save_minimax_h3_clip_continuation_media(
        replacement, 22, "h3_clip_continuation/clip", 1
    )
    assert model_helpers.get_minimax_h3_clip_continuation_fingerprint(
        "h3_clip_continuation/clip", 1
    ) != fingerprint
    torch.testing.assert_close(
        model_helpers.load_minimax_h3_clip_continuation_media(
            "h3_clip_continuation/clip", 1
        )["frames"], replacement,
    )


def test_clip_continuation_save_audio_aligns_to_h3_audio_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    frames = torch.zeros(56, 8, 8, 3)
    audio = {"waveform": torch.ones(1, 2, 74400), "sample_rate": 32000}
    model_helpers.save_minimax_h3_clip_continuation_media(
        frames, 22, "h3_clip_continuation/clip", 1, audio=audio,
    )
    loaded = model_helpers.load_minimax_h3_clip_continuation_media(
        "h3_clip_continuation/clip", 1,
    )
    assert loaded["audio"]["waveform"].shape == (1, 2, 29600)
    assert loaded["audio"]["sample_rate"] == 32000


def test_clip_continuation_load_media_type_filtering(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    frames = torch.zeros(22, 8, 8, 3)
    audio = {"waveform": torch.ones(1, 2, 29600), "sample_rate": 32000}
    model_helpers.save_minimax_h3_clip_continuation_media(
        frames, 22, "h3_clip_continuation/clip", 1, audio=audio,
    )
    load_node = utils_nodes.UC_MiniMaxH3ClipContinuationLoad
    both = load_node.execute("h3_clip_continuation/clip", 1, "replace", "video+audio").args[0]
    assert both["frames"] is not None
    assert both["audio"] is not None

    video_only = load_node.execute("h3_clip_continuation/clip", 1, "replace", "video only").args[0]
    assert video_only["frames"] is not None
    assert video_only["audio"] is None

    audio_only = load_node.execute("h3_clip_continuation/clip", 1, "replace", "audio only").args[0]
    assert audio_only["frames"] is None
    assert audio_only["audio"] is not None


def test_clip_continuation_unpack_node(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    schema = utils_nodes.UC_MiniMaxH3ClipContinuationUnpack.define_schema()
    assert [out.id for out in schema.outputs] == ["images", "audio"]

    frames = torch.ones(22, 16, 16, 3)
    audio = {"waveform": torch.ones(1, 2, 29600), "sample_rate": 32000}
    media = {
        "format_version": 1, "frame_rate": 24,
        "frames": frames, "audio": audio,
    }
    unpack_node = utils_nodes.UC_MiniMaxH3ClipContinuationUnpack
    out_images, out_audio = unpack_node.execute(continuation_media=media).args
    torch.testing.assert_close(out_images, frames)
    torch.testing.assert_close(out_audio["waveform"], audio["waveform"])

    video_only_images, video_only_audio = unpack_node.execute(continuation_media=media, media_type="video only").args
    torch.testing.assert_close(video_only_images, frames)
    assert video_only_audio is None

    audio_only_images, audio_only_audio = unpack_node.execute(continuation_media=media, media_type="audio only").args
    assert audio_only_images is None
    torch.testing.assert_close(audio_only_audio["waveform"], audio["waveform"])

    model_helpers.save_minimax_h3_clip_continuation_media(
        frames, 22, "h3_clip_continuation/unpack_test", 1, audio=audio,
    )
    loaded_images, loaded_audio = unpack_node.execute(
        continuation_media=None, filename_prefix="h3_clip_continuation/unpack_test", clip_index=1,
    ).args
    torch.testing.assert_close(loaded_images, frames)
    assert loaded_audio["waveform"].shape == (1, 2, 29600)


@pytest.mark.parametrize("tail", [5, 22, 39, 56])
def test_clip_continuation_save_skips_trailing_padded_frames(monkeypatch, tmp_path, tail):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    frames = torch.arange(58, dtype=torch.float32).view(58, 1, 1, 1).expand(58, 8, 8, 3)
    pad_frames = torch.full((12, 8, 8, 3), 999.0)
    full_frames = torch.cat((frames, pad_frames), dim=0)
    pad_samples = round(12 / 24 * 40) * 800
    tail_samples = round(tail / 24 * 40) * 800
    audio_active = torch.arange(100000, dtype=torch.float32).view(1, 1, -1).repeat(1, 2, 1)
    audio_pad = torch.full((1, 2, pad_samples), -999.0)
    full_audio = {"waveform": torch.cat((audio_active, audio_pad), dim=-1), "sample_rate": 32000}

    path = model_helpers.save_minimax_h3_clip_continuation_media(
        full_frames, tail, f"h3_clip_continuation/clip_{tail}", 1,
        audio=full_audio, padded_frames=12,
    )
    loaded = model_helpers.load_minimax_h3_clip_continuation_media(
        f"h3_clip_continuation/clip_{tail}", 1,
    )
    assert loaded["frames"].shape[0] == tail
    assert not loaded["frames"].eq(999.0).any()
    torch.testing.assert_close(loaded["frames"], frames[-tail:])
    assert loaded["audio"]["waveform"].shape == (1, 2, tail_samples)
    assert not loaded["audio"]["waveform"].eq(-999.0).any()
    torch.testing.assert_close(loaded["audio"]["waveform"], audio_active[..., -tail_samples:])


def test_clip_continuation_save_node_accepts_padded_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    schema = utils_nodes.UC_MiniMaxH3ClipContinuationSave.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert inputs["padded_frames"].default == 0
    frames = torch.arange(34, dtype=torch.float32).view(34, 1, 1, 1).expand(34, 8, 8, 3)
    pad = torch.full((12, 8, 8, 3), 777.0)
    output = utils_nodes.UC_MiniMaxH3ClipContinuationSave.execute(
        torch.cat((frames, pad), dim=0), tail_frames="22", filename_prefix="h3_clip_continuation/save_pad",
        clip_index=1, padded_frames=12,
    )
    loaded = model_helpers.load_minimax_h3_clip_continuation_media("h3_clip_continuation/save_pad", 1)
    assert loaded["frames"].shape[0] == 22
    assert not loaded["frames"].eq(777.0).any()


def test_clip_continuation_rejects_short_tail_and_output_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(model_helpers.folder_paths, "get_output_directory", lambda: str(tmp_path))
    with pytest.raises(ValueError, match="needs 22 frames"):
        model_helpers.save_minimax_h3_clip_continuation_media(
            torch.ones(5, 8, 8, 3), 22, "h3_clip_continuation/clip", 1
        )
    with pytest.raises(ValueError, match="stay inside output"):
        model_helpers.save_minimax_h3_clip_continuation_media(
            torch.ones(22, 8, 8, 3), 22, "../escape", 1
        )


def test_clip_continuation_trim_removes_aligned_frame_and_audio_heads():
    frames = torch.arange(56, dtype=torch.float32).view(56, 1, 1, 1).expand(56, 8, 8, 3).clone()
    audio = {"waveform": torch.arange(560, dtype=torch.float32).view(1, 1, 560), "sample_rate": 240}

    untrimmed = utils_nodes.UC_MiniMaxH3ClipContinuationTrim.execute(
        frames, audio, 0
    )
    torch.testing.assert_close(untrimmed.args[0], frames)
    torch.testing.assert_close(untrimmed.args[1]["waveform"], audio["waveform"])

    trimmed = utils_nodes.UC_MiniMaxH3ClipContinuationTrim.execute(
        frames, audio, 22, 5
    )
    torch.testing.assert_close(trimmed.args[0], frames[22:-5])
    torch.testing.assert_close(trimmed.args[1]["waveform"], audio["waveform"][..., 220:510])

    without_audio = utils_nodes.UC_MiniMaxH3ClipContinuationTrim.execute(frames, None, 22, 5)
    torch.testing.assert_close(without_audio.args[0], frames[22:-5])
    assert without_audio.args[1] is None

    with pytest.raises(ValueError, match="leave at least one frame"):
        utils_nodes.UC_MiniMaxH3ClipContinuationTrim.execute(frames, None, 22, 34)


def test_clip_continuation_save_still_has_only_path_output():
    schema = utils_nodes.UC_MiniMaxH3ClipContinuationSave.define_schema()
    assert [output.id for output in schema.outputs] == ["path"]


def test_clip_continuation_accumulate_blocks_then_joins_and_resets():
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    schema = node.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert inputs["overlap_threshold"].display_name == "Duplicate boundary threshold (%)"
    assert inputs["overlap_threshold"].advanced is True
    assert inputs["overlap_threshold"].default == 88.0
    assert inputs["overlap_threshold"].min == 0.0
    assert inputs["overlap_threshold"].max == 100.0
    assert inputs["overlap_threshold"].step == 0.1
    assert inputs["maximum_overlap_frames"].default == 56
    assert inputs["maximum_overlap_frames"].display_name == "Maximum duplicate frames to check"
    assert inputs["near_identical_tolerance"].default == 1.0
    assert inputs["near_identical_tolerance"].min == 0.0
    assert inputs["near_identical_tolerance"].max == 5.0
    assert inputs["near_identical_tolerance"].step == 0.1
    assert inputs["near_identical_tolerance"].advanced is True
    assert inputs["first_batch_reset"].default is False
    assert inputs["auto_accumulate"].default is True
    assert inputs["auto_accumulate"].label_on == "Add next clip automatically"
    assert inputs["auto_accumulate"].label_off == "Choose clip number"
    assert inputs["current_entry"].default == 1
    assert inputs["current_entry"].min == 1
    assert inputs["current_entry"].display_name == "Clip number"
    load_schema = utils_nodes.UC_MiniMaxH3ClipContinuationLoad.define_schema()
    load_inputs = {value.id: value for value in load_schema.inputs}
    assert load_inputs["media_type"].default == "video+audio"
    assert load_inputs["media_type"].options == ["video+audio", "video only", "audio only"]
    assert load_inputs["video_merge_mode"].default == "replace"
    assert load_inputs["video_merge_mode"].options == ["replace", "prepend", "temporal fusion"]
    first_images = torch.zeros(2, 8, 8, 3)
    second_images = torch.ones(3, 8, 8, 3)
    first_audio = {"waveform": torch.zeros(1, 1, 20), "sample_rate": 240}
    second_audio = {"waveform": torch.ones(1, 1, 30), "sample_rate": 240}

    blocked = node.execute(first_images, first_audio, target_batches=2, overlap_threshold=88, first_batch_reset=True, unique_id="accumulate-main")
    assert isinstance(blocked.args[0], ExecutionBlocker)
    output = node.execute(second_images, second_audio, target_batches=2, overlap_threshold=88, first_batch_reset=False, unique_id="accumulate-main")
    torch.testing.assert_close(output.args[0], torch.cat((first_images, second_images)))
    torch.testing.assert_close(output.args[1]["waveform"], torch.cat((first_audio["waveform"], second_audio["waveform"]), dim=-1))

    assert isinstance(node.execute(first_images, None, target_batches=2, overlap_threshold=88, first_batch_reset=True, unique_id="accumulate-reset").args[0], ExecutionBlocker)
    assert isinstance(node.execute(second_images, None, target_batches=2, overlap_threshold=88, first_batch_reset=True, unique_id="accumulate-reset").args[0], ExecutionBlocker)
    output = node.execute(first_images, None, target_batches=2, overlap_threshold=88, first_batch_reset=False, unique_id="accumulate-reset")
    torch.testing.assert_close(output.args[0], torch.cat((second_images, first_images)))

    assert isinstance(node.execute(first_images, None, target_batches=2, overlap_threshold=88, first_batch_reset=True, unique_id="accumulate-main").args[0], ExecutionBlocker)
    output = node.execute(second_images, None, target_batches=2, overlap_threshold=88, first_batch_reset=False, unique_id="accumulate-main")
    torch.testing.assert_close(output.args[0], torch.cat((first_images, second_images)))
    assert output.args[1] is None


def test_clip_continuation_accumulate_manual_entry_replaces_retry():
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    first = torch.zeros(2, 8, 8, 3)
    initial = torch.ones(2, 8, 8, 3)
    retry = torch.full((2, 8, 8, 3), 0.5)
    final = torch.full((2, 8, 8, 3), 0.75)
    entry_id = "accumulate-manual"

    assert isinstance(node.execute(first, target_batches=3, overlap_threshold=100, first_batch_reset=True, auto_accumulate=False, current_entry=1, unique_id=entry_id).args[0], ExecutionBlocker)
    assert isinstance(node.execute(initial, target_batches=3, overlap_threshold=100, first_batch_reset=False, auto_accumulate=False, current_entry=2, unique_id=entry_id).args[0], ExecutionBlocker)
    assert isinstance(node.execute(retry, target_batches=3, overlap_threshold=100, first_batch_reset=False, auto_accumulate=False, current_entry=2, unique_id=entry_id).args[0], ExecutionBlocker)
    output = node.execute(final, target_batches=3, overlap_threshold=100, first_batch_reset=False, auto_accumulate=False, current_entry=3, unique_id=entry_id)

    torch.testing.assert_close(output.args[0], torch.cat((first, retry, final)))

    reset = node.execute(final, target_batches=2, overlap_threshold=100, first_batch_reset=True, auto_accumulate=False, current_entry=99, unique_id=entry_id)
    assert isinstance(reset.args[0], ExecutionBlocker)


def test_clip_continuation_accumulate_defers_overlap_until_target(monkeypatch):
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    calls = []
    media_calls = []
    original = utils_nodes.trim_minimax_h3_clip_continuation_batch

    def track(*args, **kwargs):
        calls.append(args[4:])
        media_calls.append(list(kwargs["continuation_batches"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(utils_nodes, "trim_minimax_h3_clip_continuation_batch", track)
    entry_id = "accumulate-deferred"
    for index in range(2):
        output = node.execute(torch.full((2, 8, 8, 3), index / 2), target_batches=3, first_batch_reset=index == 0, unique_id=entry_id)
        assert isinstance(output.args[0], ExecutionBlocker)
    assert calls == []

    node.execute(torch.full((2, 8, 8, 3), 1.0), target_batches=3, first_batch_reset=False, near_identical_tolerance=2.0, unique_id=entry_id)
    assert calls == [(2.0,)]
    assert media_calls == [[None, None, None]]


def test_clip_continuation_accumulate_rejects_mixed_audio_and_geometry():
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    node.execute(torch.zeros(1, 8, 8, 3), None, target_batches=2, overlap_threshold=88, first_batch_reset=True, unique_id="accumulate-audio")
    with pytest.raises(ValueError, match="all include audio or all omit"):
        node.execute(torch.zeros(1, 8, 8, 3), {"waveform": torch.zeros(1, 1, 4), "sample_rate": 240}, target_batches=2, overlap_threshold=88, first_batch_reset=False, unique_id="accumulate-audio")

    node.execute(torch.zeros(1, 8, 8, 3), None, target_batches=2, overlap_threshold=88, first_batch_reset=True, unique_id="accumulate-geometry")
    with pytest.raises(ValueError, match="matching geometry"):
        node.execute(torch.zeros(1, 9, 8, 3), None, target_batches=2, overlap_threshold=88, first_batch_reset=False, unique_id="accumulate-geometry")


def test_clip_continuation_accumulate_trims_detected_visual_overlap_and_audio():
    generator = torch.Generator().manual_seed(1)
    first_images = torch.rand(56, 8, 8, 3, generator=generator)
    second_images = torch.cat((first_images[-22:], torch.rand(10, 8, 8, 3, generator=generator)))
    first_audio = {"waveform": torch.arange(560, dtype=torch.float32).view(1, 1, 560), "sample_rate": 240}
    second_audio = {"waveform": torch.arange(320, dtype=torch.float32).view(1, 1, 320), "sample_rate": 240}
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate

    node.execute(first_images, first_audio, target_batches=2, overlap_threshold=98, maximum_overlap_frames=10, first_batch_reset=True, unique_id="accumulate-overlap")
    output = node.execute(second_images, second_audio, target_batches=2, overlap_threshold=98, maximum_overlap_frames=10, first_batch_reset=False, unique_id="accumulate-overlap")

    torch.testing.assert_close(output.args[0], torch.cat((first_images, second_images)))
    torch.testing.assert_close(output.args[1]["waveform"], torch.cat((first_audio["waveform"], second_audio["waveform"]), dim=-1))


def test_clip_continuation_overlap_rejects_an_inconsistent_sequence():
    previous = torch.zeros(4, 8, 8, 3)
    current = torch.zeros(5, 8, 8, 3)
    current[2] = 1

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(previous, current, threshold=70, maximum_frames=4) == 2


def test_clip_continuation_overlap_selects_the_strongest_alignment():
    previous = torch.tensor((0.0, 0.1, 0.2, 0.3)).view(4, 1, 1, 1).expand(4, 8, 8, 3)
    current = torch.tensor((0.2, 0.3, 0.5, 0.6, 0.7)).view(5, 1, 1, 1).expand(5, 8, 8, 3)

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(previous, current, threshold=70, maximum_frames=4) == 2


def test_clip_continuation_overlap_prioritizes_motion_over_static_scenery():
    previous = torch.full((4, 32, 32, 3), 0.5)
    current = torch.full((5, 32, 32, 3), 0.5)
    previous[:, 12:20, 12:20] = torch.tensor((0.0, 0.1, 0.2, 0.3)).view(4, 1, 1, 1)
    current[:, 12:20, 12:20] = torch.tensor((0.2, 0.3, 0.4, 0.5, 0.6)).view(5, 1, 1, 1)

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(previous, current, threshold=90, maximum_frames=4) == 2


def test_clip_continuation_overlap_requires_audio_when_audio_is_available():
    previous = torch.zeros(16, 8, 8, 3)
    current = torch.zeros(20, 8, 8, 3)
    generator = torch.Generator().manual_seed(7)
    previous_audio_waveform = torch.rand(1, 1, 160, generator=generator)
    current_audio_waveform = torch.cat(
        (torch.rand(1, 1, 120, generator=generator), torch.rand(1, 1, 40, generator=generator)),
        dim=-1,
    )
    overlap = model_helpers.find_minimax_h3_clip_continuation_overlap(
        previous,
        current,
        threshold=88,
        maximum_frames=12,
        previous_audio={"waveform": previous_audio_waveform, "sample_rate": 240},
        current_audio={"waveform": current_audio_waveform, "sample_rate": 240},
    )
    assert overlap == 0


def test_clip_continuation_overlap_skips_audio_for_visually_rejected_candidates(monkeypatch):
    feature_batches = iter((
        torch.tensor(((1.0, 0.0), (1.0, 0.0))),
        torch.tensor(((-1.0, 0.0), (-1.0, 0.0))),
    ))
    monkeypatch.setattr(
        model_helpers, "rank_video_overlap_candidates", lambda *args, **kwargs: [(2, 0.0)]
    )
    monkeypatch.setattr(
        model_helpers, "_continuation_visual_features", lambda frames: next(feature_batches)
    )
    monkeypatch.setattr(
        model_helpers,
        "audio_overlap_similarity",
        lambda *args, **kwargs: pytest.fail("audio must follow visual acceptance"),
    )
    audio = {"waveform": torch.ones(1, 1, 480), "sample_rate": 240}

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(
        torch.zeros(2, 8, 8, 3),
        torch.zeros(3, 8, 8, 3),
        threshold=88,
        previous_audio=audio,
        current_audio=audio,
    ) == 0


def test_clip_continuation_overlap_prefers_more_accurate_boundary_over_longer_loose_match(monkeypatch):
    feature_batches = iter((
        torch.tensor(((0.9, 0.4359), (1.0, 0.0), (1.0, 0.0))),
        torch.tensor(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0))),
    ))
    monkeypatch.setattr(
        model_helpers, "rank_video_overlap_candidates", lambda *args, **kwargs: [(2, 0.0), (3, 0.1)]
    )
    monkeypatch.setattr(
        model_helpers, "_continuation_visual_features", lambda frames: next(feature_batches)
    )

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(
        torch.zeros(3, 8, 8, 3),
        torch.zeros(4, 8, 8, 3),
        threshold=80,
    ) == 2


def test_clip_continuation_overlap_prefers_deepest_equally_accurate_boundary(monkeypatch):
    feature_batches = iter((
        torch.tensor(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0))),
        torch.tensor(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0))),
    ))
    monkeypatch.setattr(
        model_helpers, "rank_video_overlap_candidates", lambda *args, **kwargs: [(2, 0.0), (3, 0.1)]
    )
    monkeypatch.setattr(
        model_helpers, "_continuation_visual_features", lambda frames: next(feature_batches)
    )

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(
        torch.zeros(3, 8, 8, 3),
        torch.zeros(4, 8, 8, 3),
        threshold=80,
    ) == 3


def test_clip_continuation_overlap_prefers_deepest_near_identical_boundary(monkeypatch):
    feature_batches = iter((
        torch.tensor(((0.0,), (1.0,), (2.0,))),
        torch.tensor(((0.0,), (1.0,), (2.0,), (3.0,))),
    ))
    monkeypatch.setattr(
        model_helpers, "rank_video_overlap_candidates", lambda *args, **kwargs: [(2, 0.0), (3, 0.1)]
    )
    monkeypatch.setattr(
        model_helpers, "_continuation_visual_features", lambda frames: next(feature_batches)
    )

    assert model_helpers.find_minimax_h3_clip_continuation_overlap(
        torch.zeros(3, 8, 8, 3),
        torch.zeros(4, 8, 8, 3),
        threshold=0,
        near_identical_tolerance=1.0,
    ) == 3


def test_clip_continuation_overlap_static_dissimilar_scenes_rejected():
    previous = torch.zeros(5, 8, 8, 3)
    previous[..., 0] = 1.0
    current = torch.zeros(5, 8, 8, 3)
    current[..., 2] = 1.0
    assert model_helpers.find_minimax_h3_clip_continuation_overlap(
        previous, current, threshold=80,
    ) == 0


def _continuation_media(frames, **metadata):
    return {"format_version": 1, "frame_rate": 24, "frames": frames, **metadata}


@pytest.mark.parametrize("cut", [3, 5, 7])
@pytest.mark.parametrize("merge_mode", ["replace", "prepend"])
def test_clip_continuation_media_measures_offset_and_audio_cut(cut, merge_mode):
    generator = torch.Generator().manual_seed(183)
    previous = torch.rand(12, 24, 24, 3, generator=generator)
    current = torch.cat((previous[-cut:], torch.rand(4, 24, 24, 3, generator=generator)))
    previous_wave = torch.rand(1, 1, 12000, generator=generator)
    current_wave = torch.cat((previous_wave[..., -cut * 1000:], torch.rand(1, 1, 4000, generator=generator)), dim=-1)
    audios = [{"waveform": wave, "sample_rate": 24000} for wave in (previous_wave, current_wave)]
    images, audio = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current], audios, 99, 10,
        continuation_batches=[None, _continuation_media(previous[-5:], video_merge_mode=merge_mode)],
    )
    torch.testing.assert_close(torch.cat(images), torch.cat((previous, current[cut:])))
    torch.testing.assert_close(torch.cat([item["waveform"] for item in audio], dim=-1),
                               torch.cat((previous_wave, current_wave[..., cut * 1000:]), dim=-1))


def test_clip_continuation_media_queue_snapshot_replacement_and_reset(monkeypatch):
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    key = "continuation-snapshots"
    frames = torch.rand(6, 8, 8, 3, generator=torch.Generator().manual_seed(14))
    media = _continuation_media(frames[-5:].clone(), video_merge_mode="prepend",
                                audio={"waveform": torch.ones(1, 1, 500), "sample_rate": 2400})
    calls = []

    def track(images, audio, *args, **kwargs):
        calls.append(kwargs["continuation_batches"])
        return images, audio

    monkeypatch.setattr(utils_nodes, "trim_minimax_h3_clip_continuation_batch", track)
    node.execute(frames, target_batches=3, first_batch_reset=True, unique_id=key)
    node.execute(frames, target_batches=3, continuation_media=media, unique_id=key)
    media["frames"].zero_()
    media["audio"]["waveform"].zero_()
    stored = utils_nodes._MINIMAX_H3_CLIP_ACCUMULATION[key]["continuation_batches"][1]
    torch.testing.assert_close(stored["frames"], frames[-5:])
    assert stored["audio"]["waveform"].eq(1).all()
    assert stored["video_merge_mode"] == "prepend"
    with pytest.raises(ValueError, match="format version"):
        node.execute(frames, target_batches=3, auto_accumulate=False, current_entry=2,
                     continuation_media={"format_version": 0}, unique_id=key)
    assert utils_nodes._MINIMAX_H3_CLIP_ACCUMULATION[key]["continuation_batches"][1] is stored
    replacement = _continuation_media(frames[-5:], video_merge_mode="replace")
    node.execute(frames, target_batches=3, auto_accumulate=False, current_entry=2,
                 continuation_media=replacement, unique_id=key)
    assert not calls
    node.execute(frames, target_batches=3, continuation_media=media, unique_id=key)
    assert calls[0][0] is None
    assert calls[0][1]["video_merge_mode"] == "replace"
    assert calls[0][2]["video_merge_mode"] == "prepend"
    assert key not in utils_nodes._MINIMAX_H3_CLIP_ACCUMULATION
    node.execute(frames, target_batches=3, continuation_media=replacement, unique_id=key)
    node.execute(frames, target_batches=3, first_batch_reset=True, unique_id=key)
    assert utils_nodes._MINIMAX_H3_CLIP_ACCUMULATION[key]["continuation_batches"] == [None]
    node.execute(frames, target_batches=1, first_batch_reset=True, unique_id=key)


def test_clip_continuation_media_pixel_difference_uses_actual_queued_clips():
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    key = "continuation-mismatch"
    previous = torch.rand(8, 24, 24, 3, generator=torch.Generator().manual_seed(17))
    current = torch.cat((previous[-5:], previous[:2]))
    node.execute(previous, target_batches=2, first_batch_reset=True, unique_id=key)
    saved = (previous[-5:] * 0.95 + 0.02).clone()
    output = node.execute(current, target_batches=2, overlap_threshold=99,
                          continuation_media=_continuation_media(saved), unique_id=key)
    torch.testing.assert_close(output.args[0], torch.cat((previous, current[5:])))
    assert key not in utils_nodes._MINIMAX_H3_CLIP_ACCUMULATION


@pytest.mark.parametrize("maximum,remaining", [(4, 2), (7, 2), (0, 7)])
def test_clip_continuation_media_respects_limit_and_retains_new_frames(maximum, remaining):
    previous = torch.rand(8, 24, 24, 3, generator=torch.Generator().manual_seed(31))
    repeated_frames = 5
    current = torch.cat((previous[-repeated_frames:], previous[:2]))
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current], [None, None], 99, maximum,
        continuation_batches=[None, _continuation_media(previous[-repeated_frames:])],
    )
    expected_total = previous.shape[0] + remaining if maximum >= repeated_frames else previous.shape[0] + current.shape[0]
    assert sum(item.shape[0] for item in images) == expected_total
    one, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current[:1]], [None, None], 99, maximum,
        continuation_batches=[None, _continuation_media(previous[-repeated_frames:])],
    )
    torch.testing.assert_close(one[1], current[:1])


def test_clip_continuation_media_validates_original_clip_after_prior_trim():
    frames = torch.rand(15, 24, 24, 3, generator=torch.Generator().manual_seed(41))
    clips = [frames[:8], frames[3:10], frames[5:15]]
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        clips, [None] * 3, 99, 10,
        continuation_batches=[None, _continuation_media(clips[0][-5:]), _continuation_media(clips[1][-5:])],
    )
    torch.testing.assert_close(torch.cat(images), frames)
    with pytest.raises(ValueError, match="one continuation entry"):
        model_helpers.trim_minimax_h3_clip_continuation_batch(clips, [None] * 3, 99, 10, continuation_batches=[None])


@pytest.mark.parametrize("sample_rate,samples", [(24000, 8000), (240, 80)])
def test_clip_continuation_media_silent_and_short_audio_is_neutral(sample_rate, samples):
    previous = torch.rand(8, 24, 24, 3, generator=torch.Generator().manual_seed(21))
    current = torch.cat((previous[-5:], previous[:3]))
    audio = {"waveform": torch.zeros(1, 1, samples), "sample_rate": sample_rate}
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current], [audio, audio], 99, 7,
        continuation_batches=[None, _continuation_media(previous[-5:])],
    )
    torch.testing.assert_close(torch.cat(images), torch.cat((previous, current[5:])))


def test_clip_continuation_media_tolerance_keeps_close_correlation_candidates(monkeypatch):
    distances = iter((0.105, 0.10, 0.40))

    def correlate(*args, **kwargs):
        return types.SimpleNamespace(offset_frames=kwargs["min_offset_frames"], distance=next(distances), overlap_frames=3)

    monkeypatch.setattr(model_helpers, "cross_correlate_video_signatures", correlate)
    frames = torch.zeros(5, 8, 8, 3)
    current = torch.zeros(6, 8, 8, 3)
    assert model_helpers.rank_minimax_h3_continuation_media_candidates(frames, current, 5, 1.0) == [(5, 0.105), (4, 0.10)]
    distances = iter((0.105, 0.10, 0.40))
    assert model_helpers.rank_minimax_h3_continuation_media_candidates(frames, current, 5, 0.0) == [(4, 0.10)]


def test_clip_continuation_supplied_candidates_do_not_fall_back_or_exceed_limits():
    frames = torch.zeros(6, 8, 8, 3)
    assert model_helpers.find_minimax_h3_clip_continuation_overlap(frames, frames, 99, 5, candidate_overlaps=[]) == 0
    assert model_helpers.find_minimax_h3_clip_continuation_overlap(frames, frames, 99, 5, candidate_overlaps=[(6, 0.0), (0, 0.0)]) == 0
    assert model_helpers.find_minimax_h3_clip_continuation_overlap(frames, torch.ones_like(frames), 99, 5, candidate_overlaps=[(5, 0.0)]) == 0


def test_clip_continuation_media_schema_preserves_existing_input_order():
    schema = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate.define_schema()
    assert [value.id for value in schema.inputs] == [
        "images", "audio", "continuation_media", "target_batches", "overlap_threshold",
        "maximum_overlap_frames", "continuation_prune_range", "near_identical_tolerance", "first_batch_reset",
        "auto_accumulate", "current_entry",
    ]


@pytest.mark.parametrize("connected", [False, True])
def test_clip_continuation_join_matches_inside_both_clips(connected):
    generator = torch.Generator().manual_seed(511)
    timeline = torch.rand(16, 24, 24, 3, generator=generator)
    previous = torch.cat((timeline[:10], torch.rand(3, 24, 24, 3, generator=generator)))
    current = torch.cat((torch.rand(2, 24, 24, 3, generator=generator), timeline[5:]))
    wave = torch.rand(1, 1, 16000, generator=generator)
    previous_wave = torch.cat((wave[..., :10000], torch.rand(1, 1, 3000, generator=generator)), dim=-1)
    current_wave = torch.cat((torch.rand(1, 1, 2000, generator=generator), wave[..., 5000:]), dim=-1)
    audio = [{"waveform": item, "sample_rate": 24000} for item in (previous_wave, current_wave)]
    media = [None, _continuation_media(previous[-8:])] if connected else None
    images, sounds = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current], audio, 99, 12, continuation_batches=media,
    )
    assert images[0].shape[0] < previous.shape[0]
    assert images[1].shape[0] < current.shape[0]
    torch.testing.assert_close(torch.cat(images), timeline)
    torch.testing.assert_close(torch.cat([item["waveform"] for item in sounds], dim=-1), wave)


def test_clip_continuation_join_no_match_keeps_both_clips():
    generator = torch.Generator().manual_seed(512)
    clips = [torch.rand(8, 24, 24, 3, generator=generator) for _ in range(2)]
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(clips, [None, None], 99, 8)
    torch.testing.assert_close(torch.cat(images), torch.cat(clips))
    # Context resembling the current clip must not substitute for the actual previous clip.
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        clips, [None, None], 99, 8,
        continuation_batches=[None, _continuation_media(clips[1][-5:])],
    )
    torch.testing.assert_close(torch.cat(images), torch.cat(clips))


def test_clip_continuation_prune_range_preserves_overlap_and_aligns():
    generator = torch.Generator().manual_seed(77)
    timeline = torch.rand(18, 24, 24, 3, generator=generator)
    previous = timeline[:12]
    current = timeline[6:]
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current], [None, None], 99, 10,
        continuation_batches=[None, _continuation_media(previous[-6:])],
        continuation_prune_range=30.0,
    )
    torch.testing.assert_close(torch.cat(images), timeline)


def test_clip_continuation_join_saved_pixels_resolve_competing_offsets(caplog):
    generator = torch.Generator().manual_seed(611)
    context = torch.rand(5, 24, 24, 3, generator=generator)
    repeated = torch.rand(5, 24, 24, 3, generator=generator)
    repeated[:2] = context[:2]
    previous = torch.cat((context, torch.rand(2, 24, 24, 3, generator=generator), repeated))
    current = torch.cat((context, torch.rand(3, 24, 24, 3, generator=generator)))
    join = model_helpers.align_minimax_h3_clip_continuation_join(previous, current, 99, 12)
    assert join is None
    assert "competing temporal alignments" in caplog.text
    join = model_helpers.align_minimax_h3_clip_continuation_join(
        previous, current, 99, 12, continuation_frames=context,
    )
    assert join is not None
    torch.testing.assert_close(torch.cat((previous[:join[0]], current[join[1]:])), current)


def test_clip_continuation_join_imperfect_frame_keeps_sequence_and_avoids_bad_seam(caplog):
    generator = torch.Generator().manual_seed(610)
    previous = torch.rand(12, 24, 24, 3, generator=generator)
    current = torch.cat((previous[-8:], torch.rand(4, 24, 24, 3, generator=generator)))
    current[3, :6] = 1 - current[3, :6]
    join = model_helpers.align_minimax_h3_clip_continuation_join(previous, current, 95, 12, 5)
    assert join is not None
    assert join[0] - join[1] == 4
    torch.testing.assert_close(
        torch.cat((previous[:join[0]], current[join[1]:])),
        torch.cat((previous, current[8:])),
    )
    assert "matched 8" in caplog.text


def test_clip_continuation_join_unresolved_repetition_is_reported(caplog):
    generator = torch.Generator().manual_seed(21)
    repeated = torch.rand(4, 24, 24, 3, generator=generator)
    previous = torch.cat((repeated, repeated))
    current = torch.cat((repeated, torch.rand(3, 24, 24, 3, generator=generator)))
    images, _ = model_helpers.trim_minimax_h3_clip_continuation_batch(
        [previous, current], [None, None], 99, 8,
    )
    torch.testing.assert_close(torch.cat(images), torch.cat((previous, current)))
    assert "competing temporal alignments" in caplog.text


def test_clip_continuation_join_reports_no_match_and_insufficient_window(caplog):
    generator = torch.Generator().manual_seed(612)
    previous = torch.rand(8, 24, 24, 3, generator=generator)
    current = torch.rand(8, 24, 24, 3, generator=generator)
    assert model_helpers.align_minimax_h3_clip_continuation_join(previous, current, 99, 8) is None
    assert "no sequence meets" in caplog.text
    assert model_helpers.align_minimax_h3_clip_continuation_join(previous, current, 99, 1) is None
    assert "fewer than two frames" in caplog.text


@pytest.mark.parametrize("threshold,tolerance,matched", [(105, 1, False), (99, -1, False), (99, 6, True), (-1, 1, True)])
def test_clip_continuation_join_accepts_out_of_widget_range_values(threshold, tolerance, matched):
    generator = torch.Generator().manual_seed(613)
    previous = torch.rand(8, 24, 24, 3, generator=generator)
    current = torch.cat((previous[-5:], torch.rand(3, 24, 24, 3, generator=generator)))
    join = model_helpers.align_minimax_h3_clip_continuation_join(previous, current, threshold, 8, tolerance)
    if matched:
        assert join is not None
        torch.testing.assert_close(
            torch.cat((previous[:join[0]], current[join[1]:])),
            torch.cat((previous, current[5:])),
        )
    else:
        assert join is None


@pytest.mark.parametrize("threshold,tolerance,matched", [(99, 6, True), (105, 6, False), (99, -1, False)])
def test_clip_continuation_retained_helpers_accept_quality_values(threshold, tolerance, matched):
    previous = torch.rand(8, 24, 24, 3, generator=torch.Generator().manual_seed(714))
    current = torch.cat((previous[-5:], torch.ones(3, 24, 24, 3)))
    join = model_helpers.find_minimax_h3_clip_continuation_join(
        previous, current, threshold, 7, tolerance,
    )
    overlap = model_helpers.find_minimax_h3_clip_continuation_overlap(
        previous, current, threshold, 7, near_identical_tolerance=tolerance,
        candidate_overlaps=[(5, 0.0)], fps=23.976,
    )
    if matched:
        assert join is not None
        assert overlap == 5
    else:
        assert join is None
        assert overlap == 0


def test_clip_continuation_visual_features_preserve_supplied_range():
    frames = torch.zeros(2, 24, 24, 3)
    frames[0, :, :12] = 2.0
    frames[0, :, 12:] = 3.0
    frames[1, :, :12] = 3.0
    frames[1, :, 12:] = 2.0
    features = model_helpers._continuation_visual_features(frames)
    assert not torch.equal(features[0], features[1])


def test_refined_compression_can_create_gradients_inside_inference_mode():
    previous_grad = torch.is_grad_enabled()
    with torch.inference_mode(True):
        source = (torch.arange(16, dtype=torch.float32).square() / 225).reshape(1, 1, 1, 4, 4).expand(1, 24, 1, 4, 4).clone()
        before = source.clone()
        pooled = model_helpers._pool_minimax_h3_visual_latent(source, 2, None)
        refined = model_helpers._refine_minimax_h3_visual_latent(source, pooled, 3)
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()

    assert refined.shape == (1, 24, 1, 2, 2)
    assert refined.requires_grad is False
    assert torch.is_grad_enabled() == previous_grad
    assert not torch.is_inference_mode_enabled()
    torch.testing.assert_close(source, before)
    baseline = F.interpolate(pooled, size=(1, 4, 4), mode="trilinear", align_corners=False)
    improved = F.interpolate(refined, size=(1, 4, 4), mode="trilinear", align_corners=False)
    assert F.mse_loss(improved, source) < F.mse_loss(baseline, source)


def test_pooling_averages_reference_detail_instead_of_sampling_centers():
    source = torch.zeros((1, 24, 1, 6, 6), dtype=torch.float64)
    source[..., ::3, ::3] = 9
    pooled = model_helpers._pool_minimax_h3_visual_latent(source, 2, None)
    torch.testing.assert_close(pooled, torch.ones((1, 24, 1, 2, 2), dtype=torch.float64))


def test_save_load_collision_fingerprint_and_containment(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    first = model_helpers.save_minimax_h3_ref_collection([_image_ref()], "nested/ref")
    second = model_helpers.save_minimax_h3_ref_collection([_image_ref(2.0)], "nested/ref")

    assert first == ["nested/ref_0001.safetensors"]
    assert second == ["nested/ref_0002.safetensors"]
    loaded = model_helpers.load_minimax_h3_ref(first[0])
    assert torch.equal(loaded["latent"], _image_ref()["latent"])
    path = tmp_path / first[0]
    before = model_helpers.get_minimax_h3_ref_input_fingerprint(first[0])
    old_stat = path.stat()
    metadata = model_helpers._minimax_h3_ref_storage_metadata(_image_ref(3.0))
    with IncrementalSafetensorsWriter(str(path), metadata={"ref_meta": json.dumps(metadata)}, max_workers=1) as writer:
        writer.write("latent", _image_ref(3.0)["latent"])
    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))
    assert model_helpers.get_minimax_h3_ref_input_fingerprint(first[0]) != before
    assert str(path.resolve()) in before
    torch.testing.assert_close(loaded["latent"], _image_ref()["latent"])
    torch.testing.assert_close(model_helpers.load_minimax_h3_ref(first[0])["latent"], _image_ref(3.0)["latent"])
    with pytest.raises(ValueError, match="inside the Ref folder"):
        model_helpers.save_minimax_h3_ref_collection([_image_ref()], "../escape")
    with pytest.raises(ValueError, match="inside the Ref folder"):
        model_helpers.load_minimax_h3_ref("../outside.safetensors")


def test_save_publishes_without_overwriting_a_racing_writer(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    operation = "rename" if os.name == "nt" else "link"
    publish = getattr(os, operation)
    raced = []

    def racing_publish(source, destination):
        if not raced:
            raced.append(pathlib.Path(destination))
            raced[0].write_bytes(b"other writer")
        publish(source, destination)

    monkeypatch.setattr(model_helpers.os, operation, racing_publish)
    result = model_helpers.save_minimax_h3_ref_collection([_image_ref()], "character.v2")
    assert raced[0].read_bytes() == b"other writer"
    assert result == ["character.v2_0002.safetensors"]
    assert not list(tmp_path.glob("*.tmp"))


def test_video_and_audio_round_trip_independently(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    refs = [_video_ref(), {"kind": "audio", "latent": torch.ones((1, 32, 2, 5), dtype=torch.float64), "metadata": {"description": "audio", "sample_rate": 32000}}]
    paths = model_helpers.save_minimax_h3_ref_collection(refs, "mixed")
    loaded = [model_helpers.load_minimax_h3_ref(path) for path in paths]
    assert [ref["kind"] for ref in loaded] == ["video", "audio"]
    for original, restored in zip(refs, loaded):
        torch.testing.assert_close(original["latent"], restored["latent"])
        for key, value in original["metadata"].items():
            assert restored["metadata"][key] == value


def test_visual_ref_round_trip_restores_qwen_conditioning_and_legacy_file(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    qwen_ref = _image_ref()
    qwen_ref["vlm_embedding"] = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    qwen_ref["vlm_tags"] = torch.tensor([1, 0, 0], dtype=torch.long)
    qwen_ref["metadata"].update({"vlm_presentation": "image_numbered", "vlm_reference_number": 17})
    saved = model_helpers.save_minimax_h3_ref_collection([qwen_ref, _video_ref()], "qwen")
    loaded, legacy = [model_helpers.load_minimax_h3_ref(path) for path in saved]
    assert len(model_nodes.UC_MiniMaxH3RefLoad.execute(saved[0])[0]) == 1
    torch.testing.assert_close(loaded["vlm_embedding"], qwen_ref["vlm_embedding"])
    torch.testing.assert_close(loaded["vlm_tags"], qwen_ref["vlm_tags"])
    assert "Qwen <Picture 17> 3 tokens" in model_helpers.format_minimax_h3_ref_info([loaded])
    assert "vlm_embedding" not in legacy


def test_fused_image_batch_saves_one_file_and_applies_one_ref(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    monkeypatch.setattr(model_helpers, "fuse_minimax_h3_ref_vlm_images", lambda *_args: (
        torch.ones((1, 2, 4)), torch.tensor([1, 0], dtype=torch.long),
    ))
    images = torch.stack((torch.zeros(32, 64, 3), torch.ones(32, 64, 3)))
    refs = model_helpers.create_minimax_h3_image_refs(images, _VisualVae(), clip=object())
    paths = model_helpers.save_minimax_h3_ref_collection(refs, "fused")
    assert len(paths) == 1
    loaded = model_helpers.load_minimax_h3_ref(paths[0])
    assert loaded["metadata"]["source_images"] == 2
    base = [[torch.zeros((1, 1, 4)), {
        "minimax_token_tags": torch.ones(1, dtype=torch.long),
        "uc_minimax_h3_vlm_layout": {"version": 1, "sequence_length": 1, "prompt_start": 0},
    }]]
    applied = model_helpers.apply_minimax_h3_refs_to_conditioning(base, [loaded])
    assert len(applied[0][1]["minimax_refs"]) == 1
    assert applied[0][0].shape[1] == 3
    assert applied[0][1]["minimax_token_tags"].tolist() == [1, 0, 1]


def test_bundle_saves_four_refs_in_one_file_and_loads_for_apply(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    fused_image = _image_ref(2.0)
    fused_image["metadata"].update({"source_images": 8, "vlm_presentation": "image_numbered", "vlm_reference_number": 17})
    fused_image["vlm_embedding"] = torch.ones((1, 2, 4))
    fused_image["vlm_tags"] = torch.tensor([1, 0], dtype=torch.long)
    refs = [_image_ref(), fused_image, _video_ref(), {
        "kind": "audio", "latent": torch.ones((1, 32, 2, 5)), "metadata": {"description": "voice"},
    }]
    paths = model_nodes.UC_MiniMaxH3RefSave.execute("bundle", refs={"ref_1": refs}, save_layout="bundle")[0]
    assert len(paths) == 1
    loaded = model_nodes.UC_MiniMaxH3RefLoad.execute(paths[0])[0]
    assert [ref["kind"] for ref in loaded] == ["image", "image", "video", "audio"]
    assert loaded[1]["metadata"]["source_images"] == 8
    torch.testing.assert_close(loaded[1]["vlm_embedding"], fused_image["vlm_embedding"])
    base = [[torch.zeros((1, 1, 4)), {
        "minimax_token_tags": torch.ones(1, dtype=torch.long),
        "uc_minimax_h3_vlm_layout": {"version": 1, "sequence_length": 1, "prompt_start": 0},
    }]]
    applied = model_nodes.UC_MiniMaxH3RefApply.execute(base, refs={"ref_1": loaded})[0]
    assert len(applied[0][1]["minimax_refs"]) == 4
    assert applied[0][0].shape[1] == 3


def test_loads_original_style_bundle_members(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    metadata = {"_format_version": 5, "kind": "bundle", "name": "original", "members": [
        {"kind": "image", "latent_t": 1, "latent_h": 4, "latent_w": 4, "mode": "training", "description": "subject"},
        {"kind": "audio", "latent_t": 5, "latent_h": 0, "latent_w": 0, "mode": "encode", "description": "voice"},
    ]}
    with IncrementalSafetensorsWriter(str(tmp_path / "original_bundle.safetensors"), metadata={"refmod_meta": json.dumps(metadata)}, max_workers=1) as writer:
        writer.write("ref_0", _image_ref()["latent"])
        writer.write("ref_1", torch.ones((1, 32, 2, 5)))
    loaded = model_helpers.load_minimax_h3_ref_collection("original_bundle.safetensors")
    assert [ref["kind"] for ref in loaded] == ["image", "audio"]
    assert [ref["metadata"]["description"] for ref in loaded] == ["subject", "voice"]


def test_bundle_rejects_member_dimensions_that_disagree_with_tensor(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    metadata = {"_format_version": 5, "kind": "bundle", "members": [
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
    ]}
    with IncrementalSafetensorsWriter(str(tmp_path / "bad_bundle.safetensors"), metadata={"refmod_meta": json.dumps(metadata)}, max_workers=1) as writer:
        writer.write("ref_0", _image_ref()["latent"])
    with pytest.raises(ValueError, match="member dimensions"):
        model_helpers.load_minimax_h3_ref_collection("bad_bundle.safetensors")


def test_qwen_refs_splice_in_socket_order_and_zero_retention_omits_them():
    first, second = _image_ref(), _video_ref()
    first["metadata"]["vlm_presentation"] = "image_visual"
    second["metadata"]["vlm_presentation"] = "video_visual_2fps"
    first["vlm_embedding"] = torch.full((1, 2, 1), 3.0)
    first["vlm_tags"] = torch.tensor([1, 0], dtype=torch.long)
    second["vlm_embedding"] = torch.full((1, 1, 1), 4.0)
    second["vlm_tags"] = torch.tensor([0], dtype=torch.long)
    base = [[torch.tensor([[[1.0], [2.0]]]), {
        "minimax_token_tags": torch.ones(2, dtype=torch.long),
        "uc_minimax_h3_vlm_layout": {"version": 1, "sequence_length": 2, "prompt_start": 1},
        "start_percent": 0.25,
    }]]
    applied = model_helpers.apply_minimax_h3_refs_to_conditioning(base, [first, second])
    assert applied[0][0].flatten().tolist() == [1, 3, 3, 4, 2]
    assert applied[0][1]["minimax_token_tags"].tolist() == [1, 1, 0, 0, 1]
    assert applied[0][1]["uc_minimax_h3_vlm_layout"]["prompt_start"] == 4
    assert len(applied[0][1]["minimax_refs"]) == 2
    assert applied[0][1]["start_percent"] == 0.25
    assert base[0][0].shape[1] == 2
    skipped = model_helpers.apply_minimax_h3_refs_to_conditioning(base, [first, second], retention=0)
    assert skipped[0][0] is base[0][0]
    assert skipped[0][1]["minimax_refs"] == []


def test_ref_extract_keeps_old_inputs_and_attaches_optional_qwen(monkeypatch):
    clip = object()
    calls = []

    def encode(_clip, media_type, media, resolution, number):
        calls.append((_clip, media_type, tuple(media.shape), resolution, number))
        return torch.ones((1, 2, 4)), torch.tensor([1, 0], dtype=torch.long)

    monkeypatch.setattr(model_helpers, "encode_minimax_h3_ref_vlm", encode)
    monkeypatch.setattr(model_helpers, "fuse_minimax_h3_ref_vlm_images", lambda _clip, media, resolution, number: encode(_clip, "image", media, resolution, number))
    schema = model_nodes.UC_MiniMaxH3RefExtract.define_schema()
    assert [item.id for item in schema.inputs[:4]] == ["images", "vae", "media_type", "description"]
    assert any(item.id == "clip" and item.optional for item in schema.inputs)
    assert any(item.id == "vlm_reference_start" for item in schema.inputs)
    assert all(item.id != "timestamp" for item in schema.inputs)
    media_type = {"media_type": "image", "compression": {"compression": "encode"}}
    images = torch.zeros((2, 32, 64, 3))
    refs = model_nodes.UC_MiniMaxH3RefExtract.execute(images, _VisualVae(), media_type, clip=clip, vlm_resolution=512, vlm_reference_start=21)[0]
    assert len(refs) == 1
    assert calls == [(clip, "image", (2, 32, 64, 3), 512, 21)]
    assert refs[0]["metadata"]["vlm_reference_number"] == 21
    assert all("vlm_embedding" in ref and "vlm_tags" in ref for ref in refs)
    video_type = {"media_type": "video", "compression": {"compression": "encode"}}
    video_ref = model_nodes.UC_MiniMaxH3RefExtract.execute(torch.zeros((22, 32, 64, 3)), _VisualVae(), video_type, clip=clip, vlm_resolution=512, vlm_reference_start=30)[0][0]
    assert calls[-1] == (clip, "video", (22, 32, 64, 3), 512, 30)
    assert video_ref["metadata"]["vlm_reference_number"] == 30


def test_ref_vlm_requires_matching_tags_and_base_layout(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    ref = _image_ref()
    ref["vlm_embedding"] = torch.ones((1, 2, 4))
    ref["vlm_tags"] = torch.ones(1, dtype=torch.long)
    with pytest.raises(ValueError, match="tags must match"):
        model_helpers.save_minimax_h3_ref_collection([ref], "invalid")
    ref["vlm_tags"] = torch.ones(2, dtype=torch.long)
    ref["metadata"]["vlm_presentation"] = "image_visual"
    with pytest.raises(ValueError, match="layout metadata"):
        model_helpers.apply_minimax_h3_refs_to_conditioning([[torch.ones((1, 1, 4)), {"minimax_token_tags": torch.ones(1, dtype=torch.long)}]], [ref])
    ref["metadata"]["vlm_presentation"] = "image_guide"
    with pytest.raises(ValueError, match="outdated"):
        model_helpers.apply_minimax_h3_refs_to_conditioning([[torch.ones((1, 1, 4)), {"minimax_token_tags": torch.ones(1, dtype=torch.long)}]], [ref])


def test_ref_apply_rejects_duplicate_qwen_picture_numbers():
    refs = [_image_ref(), _image_ref()]
    for ref in refs:
        ref["metadata"].update({"vlm_presentation": "image_numbered", "vlm_reference_number": 17})
        ref["vlm_embedding"] = torch.ones((1, 2, 4))
        ref["vlm_tags"] = torch.ones(2, dtype=torch.long)
    with pytest.raises(ValueError, match="duplicate image number 17"):
        model_helpers.apply_minimax_h3_refs_to_conditioning([[torch.ones((1, 1, 4)), {"minimax_token_tags": torch.ones(1, dtype=torch.long), "uc_minimax_h3_vlm_layout": {"version": 1, "sequence_length": 1, "prompt_start": 0}}]], refs)


def test_load_rejects_incomplete_qwen_tensor_pair(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    target = tmp_path / "incomplete.safetensors"
    metadata = model_helpers._minimax_h3_ref_storage_metadata(_image_ref())
    with IncrementalSafetensorsWriter(str(target), metadata={"refmod_meta": json.dumps(metadata)}, max_workers=1) as writer:
        writer.write("latent", _image_ref()["latent"])
        writer.write("vlm_embedding", torch.ones((1, 2, 4)))
    with pytest.raises(ValueError, match="invalid tensor set"):
        model_helpers.load_minimax_h3_ref("incomplete.safetensors")


def test_load_rejects_wrong_header_shape_and_dtype(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    original_metadata = {"kind": "image", "latent_t": 1, "latent_h": 4, "latent_w": 4, "mode": "encode", "description": "original format", "_format_version": 4}
    with IncrementalSafetensorsWriter(str(tmp_path / "original.safetensors"), metadata={"refmod_meta": json.dumps(original_metadata)}, max_workers=1) as writer:
        writer.write("latent", _image_ref()["latent"])
    original = model_helpers.load_minimax_h3_ref("original.safetensors")
    torch.testing.assert_close(original["latent"], _image_ref()["latent"])
    assert original["metadata"]["description"] == "original format"
    wrong_header = tmp_path / "wrong.safetensors"
    with IncrementalSafetensorsWriter(str(wrong_header), max_workers=1) as writer:
        writer.write("latent", _image_ref()["latent"])
    with pytest.raises(ValueError, match="does not contain MiniMax H3 reference metadata"):
        model_helpers.load_minimax_h3_ref("wrong.safetensors")

    shape_header = tmp_path / "shape.safetensors"
    metadata = model_helpers._minimax_h3_ref_storage_metadata(_image_ref())
    metadata["shape"] = [1, 24, 1, 2, 2]
    with IncrementalSafetensorsWriter(str(shape_header), metadata={"refmod_meta": json.dumps(metadata)}, max_workers=1) as writer:
        writer.write("latent", _image_ref()["latent"])
    with pytest.raises(ValueError, match="metadata does not match"):
        model_helpers.load_minimax_h3_ref("shape.safetensors")

    dtype_header = tmp_path / "dtype.safetensors"
    metadata = model_helpers._minimax_h3_ref_storage_metadata(_image_ref())
    with IncrementalSafetensorsWriter(str(dtype_header), metadata={"refmod_meta": json.dumps(metadata)}, max_workers=1) as writer:
        writer.write("latent", torch.ones((1, 24, 1, 4, 4), dtype=torch.int32))
    with pytest.raises(ValueError, match="floating-point"):
        model_helpers.load_minimax_h3_ref("dtype.safetensors")


def test_apply_preserves_existing_metadata_and_never_mutates_source_refs():
    source = _image_ref()
    source["latent"][..., 0, 0] = 10.0
    original = source["latent"].clone()
    existing = {"kind": "video_audio", "latent_t": 1, "latent_h": 4, "latent_w": 4, "ref_audio_t": 3}
    conditioning = [
        [torch.zeros((1, 1)), {"minimax_refs": [existing], "keyframes": "keep", "custom": {"nested": True}, "start_percent": 0.0, "end_percent": 0.5}],
        [torch.ones((1, 1)), {"start_percent": 0.5, "end_percent": 1.0}],
    ]

    applied = model_helpers.apply_minimax_h3_refs_to_conditioning(conditioning, [source], retention=0.5)

    assert applied[0][0] is conditioning[0][0]
    assert applied[0][1]["keyframes"] == "keep"
    assert applied[0][1]["minimax_refs"][0] is existing
    assert len(applied[0][1]["minimax_refs"]) == 2
    assert applied[1][0] is conditioning[1][0]
    assert applied[1][1]["start_percent"] == 0.5
    assert len(applied[1][1]["minimax_refs"]) == 1
    assert "minimax_refs" not in conditioning[1][1]
    assert torch.equal(source["latent"], original)
    assert not torch.equal(applied[0][1]["minimax_refs"][-1]["latent"], original)
    assert model_helpers.apply_minimax_h3_refs_to_conditioning(conditioning, [source], retention=0.0)[0][1]["minimax_refs"] == [existing]
    with pytest.raises(ValueError, match="token budget"):
        model_helpers.apply_minimax_h3_refs_to_conditioning(conditioning, [source], max_ref_tokens=13)
    assert model_helpers.apply_minimax_h3_refs_to_conditioning(conditioning, [source], max_ref_tokens=14)


def test_schema_nested_dynamic_combo_autogrow_order_and_registration_source(monkeypatch, tmp_path):
    _ref_folder(monkeypatch, tmp_path)
    extract = model_nodes.UC_MiniMaxH3RefExtract.define_schema()
    media_type = next(input for input in extract.inputs if input.id == "media_type")
    assert [option.key for option in media_type.options] == ["image", "video"]
    assert all(option.inputs[0].id == "compression" for option in media_type.options)
    assert media_type.options[0].inputs[0].options[0].key == "pooled"
    flat_inputs = {"images": torch.ones((1, 32, 64, 3)), "vae": _VisualVae(), "media_type": "image", "media_type.compression": "pooled", "media_type.compression.reference_resolution": 32, "description": "schema"}
    expanded, _hidden, dynamic = get_finalized_class_inputs(model_nodes.UC_MiniMaxH3RefExtract.INPUT_TYPES(), flat_inputs)
    assert not any(name.endswith("latent_frames") for group in expanded.values() for name in group)
    extracted = model_nodes.UC_MiniMaxH3RefExtract.execute(**build_nested_inputs(flat_inputs, dynamic))
    assert extracted[0][0]["latent"].shape == (1, 24, 1, 2, 2)

    save_schema = model_nodes.UC_MiniMaxH3RefSave.define_schema()
    assert save_schema.is_output_node is True
    flat_save = {"filename_prefix": "graph", "refs.ref_10": [_image_ref(10.0)], "refs.ref_2": [_image_ref(2.0)]}
    _expanded, _hidden, save_dynamic = get_finalized_class_inputs(model_nodes.UC_MiniMaxH3RefSave.INPUT_TYPES(), flat_save)
    paths = model_nodes.UC_MiniMaxH3RefSave.execute(**build_nested_inputs(flat_save, save_dynamic))[0]
    assert [float(model_helpers.load_minimax_h3_ref(path)["latent"].mean()) for path in paths] == [2.0, 10.0]
    registered = (CUSTOM_NODE_ROOT / "__init__.py").read_text(encoding="utf-8")
    for node_id in ("UC_MiniMaxH3RefExtract", "UC_MiniMaxH3AudioRefExtract", "UC_MiniMaxH3RefLoad", "UC_MiniMaxH3RefSave", "UC_MiniMaxH3RefApply"):
        assert node_id in registered
        schema = getattr(model_nodes, node_id).GET_SCHEMA()
        assert all(output.tooltip for output in schema.outputs)
        assert all(input.tooltip for input in schema.inputs)
    for source in ((CUSTOM_NODE_ROOT / "nodes" / "model_nodes.py").read_text(encoding="utf-8"), (CUSTOM_NODE_ROOT / "helpers" / "model_helpers.py").read_text(encoding="utf-8")):
        assert "reference/" not in source
        assert "add_object_patch" not in source
