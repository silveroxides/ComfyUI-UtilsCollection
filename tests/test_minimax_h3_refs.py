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

from utils_collection_minimax_h3_refs_test import model_helpers, model_nodes, utils_nodes


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


def test_image_batch_uses_native_preparation_and_keeps_entries_independent():
    vae = _VisualVae()
    images = torch.stack((torch.zeros(40, 72, 3), torch.ones(40, 72, 3)))

    refs = model_helpers.create_minimax_h3_image_refs(images, vae, description="subject")

    assert len(refs) == 2
    assert [tuple(image.shape) for image in vae.inputs] == [(1, 32, 64, 3)] * 2
    assert torch.count_nonzero(refs[0]["latent"]) == 0
    torch.testing.assert_close(refs[1]["latent"], torch.ones_like(refs[1]["latent"]))
    assert all(ref["kind"] == "image" and ref["metadata"]["description"] == "subject" for ref in refs)
    assert refs[0]["latent"] is not refs[1]["latent"]


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
    first_images = torch.zeros(2, 8, 8, 3)
    second_images = torch.ones(3, 8, 8, 3)
    first_audio = {"waveform": torch.zeros(1, 1, 20), "sample_rate": 240}
    second_audio = {"waveform": torch.ones(1, 1, 30), "sample_rate": 240}

    blocked = node.execute(first_images, first_audio, target_batches=2, overlap_threshold=224, reset_counter=0, unique_id="accumulate-main")
    assert isinstance(blocked.args[0], ExecutionBlocker)
    output = node.execute(second_images, second_audio, target_batches=2, overlap_threshold=224, reset_counter=0, unique_id="accumulate-main")
    torch.testing.assert_close(output.args[0], torch.cat((first_images, second_images)))
    torch.testing.assert_close(output.args[1]["waveform"], torch.cat((first_audio["waveform"], second_audio["waveform"]), dim=-1))

    assert isinstance(node.execute(first_images, None, target_batches=2, overlap_threshold=224, reset_counter=1, unique_id="accumulate-main").args[0], ExecutionBlocker)
    output = node.execute(second_images, None, target_batches=2, overlap_threshold=224, reset_counter=1, unique_id="accumulate-main")
    torch.testing.assert_close(output.args[0], torch.cat((first_images, second_images)))
    assert output.args[1] is None


def test_clip_continuation_accumulate_rejects_mixed_audio_and_geometry():
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate
    node.execute(torch.zeros(1, 8, 8, 3), None, target_batches=2, overlap_threshold=224, reset_counter=0, unique_id="accumulate-audio")
    with pytest.raises(ValueError, match="all include audio or all omit"):
        node.execute(torch.zeros(1, 8, 8, 3), {"waveform": torch.zeros(1, 1, 4), "sample_rate": 240}, target_batches=2, overlap_threshold=224, reset_counter=0, unique_id="accumulate-audio")

    node.execute(torch.zeros(1, 8, 8, 3), None, target_batches=2, overlap_threshold=224, reset_counter=0, unique_id="accumulate-geometry")
    with pytest.raises(ValueError, match="matching geometry"):
        node.execute(torch.zeros(1, 9, 8, 3), None, target_batches=2, overlap_threshold=224, reset_counter=0, unique_id="accumulate-geometry")


def test_clip_continuation_accumulate_trims_detected_visual_overlap_and_audio():
    generator = torch.Generator().manual_seed(1)
    first_images = torch.rand(56, 8, 8, 3, generator=generator)
    second_images = torch.cat((first_images[-22:], torch.rand(10, 8, 8, 3, generator=generator)))
    first_audio = {"waveform": torch.arange(560, dtype=torch.float32).view(1, 1, 560), "sample_rate": 240}
    second_audio = {"waveform": torch.arange(320, dtype=torch.float32).view(1, 1, 320), "sample_rate": 240}
    node = utils_nodes.UC_MiniMaxH3ClipContinuationAccumulate

    node.execute(first_images, first_audio, target_batches=2, overlap_threshold=250, reset_counter=0, unique_id="accumulate-overlap")
    output = node.execute(second_images, second_audio, target_batches=2, overlap_threshold=250, reset_counter=0, unique_id="accumulate-overlap")

    torch.testing.assert_close(output.args[0], torch.cat((first_images, second_images[22:])))
    torch.testing.assert_close(output.args[1]["waveform"], torch.cat((first_audio["waveform"], second_audio["waveform"][..., 220:]), dim=-1))


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
        assert restored["metadata"] == original["metadata"]


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
    for source in ((CUSTOM_NODE_ROOT / "model_nodes.py").read_text(encoding="utf-8"), (CUSTOM_NODE_ROOT / "model_helpers.py").read_text(encoding="utf-8")):
        assert "reference/" not in source
        assert "add_object_patch" not in source
