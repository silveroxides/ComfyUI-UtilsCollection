import pathlib
import sys
import types

import numpy as np
import pytest
import torch

from comfy.cli_args import args as cli_args


package = types.ModuleType("utils_collection_pose_test")
package.__path__ = [str(pathlib.Path(__file__).parents[1])]
sys.modules.setdefault(package.__name__, package)
prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_pose_test import image_helpers, image_nodes, model_helpers
finally:
    cli_args.cpu = prior_cpu


def _body(offset=0):
    points = [None] * 18
    for index, point in {0: (64, 24), 1: (64, 40), 2: (44, 40), 3: (30, 65), 4: (20, 92),
                         5: (84, 40), 6: (100, 65), 7: (108, 92), 14: (57, 21), 15: (71, 21)}.items():
        points[index] = (point[0] + offset, point[1])
    return points


def test_openpose_batches_frames_and_person_crops_with_tail_and_frame_order():
    calls, loaded = [], []

    def loader(kind):
        loaded.append(kind)
        return kind

    def forward(kind, images):
        calls.append((kind, len(images), images[0].shape[:2]))
        count = len(images)
        if kind == "body":
            heat = np.zeros((count, 23, 32, 19), np.float32)
            for index, image in enumerate(images):
                heat[index, ..., 0] = image[0, 0, 0] / 255
            return np.zeros((count, 23, 32, 38), np.float32), heat
        maps = np.zeros((count, 16, 16, 22 if kind == "hand" else 71), np.float32)
        maps[:, 6:10, 6:10] = 1
        return maps

    frames = torch.stack([torch.full((128, 192, 3), index / 10) for index in range(5)])
    output, poses = image_helpers.run_openpose_batch(
        frames, resolution=128, batch_size=4, loader=loader, forward=forward,
        body_decoder=lambda heat, paf: [_body(float(heat[0, 0, 0]) * 10)],
    )
    assert output.shape == (5, 128, 192, 3)
    assert loaded == ["body", "hand", "face"]
    assert [count for kind, count, shape in calls if kind == "body"] == [4, 1]
    assert [count for kind, count, shape in calls if kind == "face"] == [4, 1]
    hands = [(count, shape) for kind, count, shape in calls if kind == "hand"]
    assert len(hands) == 12
    assert all(count in (2, 4) for count, shape in hands)
    assert {shape for count, shape in hands} == {(184, 184), (368, 368), (552, 552), (736, 736)}
    positions = [pose["people"][0]["pose_keypoints_2d"][0] for pose in poses]
    assert positions == sorted(positions) and len(set(positions)) == 5
    assert all(len(pose["people"][0]["hand_left_keypoints_2d"]) == 63 for pose in poses)
    assert all(pose["people"][0]["face_keypoints_2d"] for pose in poses)


def test_disabled_hand_and_face_models_are_not_loaded_and_empty_frames_keep_order():
    loaded = []
    def forward(kind, images):
        return np.zeros((len(images), 23, 23, 38), np.float32), np.zeros((len(images), 23, 23, 19), np.float32)
    output, poses = image_helpers.run_openpose_batch(
        torch.zeros(3, 64, 64, 3), resolution=64, batch_size=2, detect_hand=False, detect_face=False,
        loader=lambda kind: loaded.append(kind) or kind, forward=forward,
    )
    assert loaded == ["body"]
    assert output.shape == (3, 64, 64, 3) and not output.any()
    assert [pose["people"] for pose in poses] == [[], [], []]


def test_vectorized_limb_matching_and_body_assembly_preserve_connections():
    first = np.array([[2, 2, 1, 0], [2, 20, 1, 1]], dtype=np.float64)
    second = np.array([[22, 2, 1, 2], [22, 20, 1, 3]], dtype=np.float64)
    paf = np.zeros((32, 32, 38), np.float32)
    paf[..., 12] = 1
    matches = image_helpers.score_limb_pairs(first, second, paf, (12, 13))
    assert [(a, b) for a, b, score in matches] == [(0, 2), (1, 3)]
    assert [score for a, b, score in matches] == pytest.approx([0.8, 0.8])
    y, x = np.mgrid[:64, :64]
    heat = np.zeros((64, 64, 19), np.float32)
    for part, px in ((1, 8), (2, 20), (3, 32), (4, 44)):
        heat[..., part] = np.exp(-((x - px) ** 2 + (y - 20) ** 2) / 32)
    paf = np.zeros((64, 64, 38), np.float32)
    paf[..., [12, 14, 16]] = 1
    bodies = image_helpers.decode_body(heat, paf)
    assert len(bodies) == 1
    assert bodies[0][1:5] == [(8.0, 20.0), (20.0, 20.0), (32.0, 20.0), (44.0, 20.0)]


def test_model_directory_registration_preserves_custom_paths(monkeypatch, tmp_path):
    folders = model_helpers.folder_paths
    custom = str(tmp_path / "existing")
    roots = [str(tmp_path / "models" / "controlnet"), str(tmp_path / "other_controlnet")]
    monkeypatch.setitem(folders.folder_names_and_paths, model_helpers.MODEL_FOLDER, ([custom], {".bin"}))
    monkeypatch.setattr(folders, "get_folder_paths", lambda category: roots)
    model_helpers.register_openpose_paths()
    paths, extensions = folders.folder_names_and_paths[model_helpers.MODEL_FOLDER]
    assert paths == [custom, *(str(pathlib.Path(root) / "preprocessors") for root in roots)]
    assert extensions == {".bin", ".safetensors"}


def test_forward_pass_preserves_batch_dimension_and_uses_comfy_model_management(monkeypatch):
    loaded, shapes = [], []
    monkeypatch.setattr(model_helpers.comfy.model_management, "load_models_gpu", lambda models: loaded.extend(models))
    def model(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor[:, :1], tensor[:, 1:]
    patcher = types.SimpleNamespace(load_device=torch.device("cpu"), model=model)
    first, second = model_helpers.openpose_forward(patcher, [np.zeros((8, 16, 3), np.uint8)] * 3)
    assert loaded == [patcher]
    assert shapes == [(3, 3, 8, 16)]
    assert first.shape == (3, 8, 16, 1) and second.shape == (3, 8, 16, 2)


def test_openpose_node_exposes_batch_and_pose_keypoint_contract(monkeypatch):
    expected = (torch.zeros(2, 64, 64, 3), [{"people": []}, {"people": []}])
    calls = []
    monkeypatch.setattr(image_nodes, "run_openpose_batch", lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    result = image_nodes.UC_BatchedOpenPose.execute(expected[0], batch_size=2)
    assert result.result[0] is expected[0] and result.result[1] is expected[1]
    assert calls[0][0][2] == 2
    assert calls[0][1]["loader"] is model_helpers.load_openpose_model
    assert "openpose_json" in result.ui


def test_dwpose_pools_people_across_frames_without_emitting_padding_people():
    calls, loaded = [], []
    counts = [2, 0, 1, 2, 0]
    def forward(kind, images):
        calls.append((kind, len(images)))
        if kind == "detector":
            result = np.zeros((len(images), 8400, 85), np.float32)
            for index, image in enumerate(images):
                frame = round(float(image[0, 0, 0]) / 255 * 4)
                for person in range(counts[frame]):
                    anchor = 1000 + person * 30
                    result[index, anchor, 4:6] = 1
            return result
        assert len(images) in (2, 3)
        assert images[0].shape == (384, 288, 3)
        x = np.zeros((len(images), 133, 576), np.float32)
        y = np.zeros((len(images), 133, 768), np.float32)
        x[..., 288] = 1
        y[..., 384] = 1
        return x, y
    frames = torch.stack([torch.full((64, 96, 3), index / 4) for index in range(5)])
    output, poses = image_helpers.run_dwpose_batch(
        frames, resolution=64, batch_size=3, loader=lambda kind: loaded.append(kind) or kind, forward=forward,
    )
    assert output.shape == (5, 64, 96, 3)
    assert [len(pose["people"]) for pose in poses] == counts
    assert calls == [("detector", 3), ("pose", 3), ("detector", 2), ("pose", 2)]
    assert loaded == ["detector", "pose"]
    first = poses[0]["people"][0]
    assert len(first["pose_keypoints_2d"]) == 54
    assert len(first["face_keypoints_2d"]) == 210
    assert len(first["hand_left_keypoints_2d"]) == 63
    assert len(first["hand_right_keypoints_2d"]) == 63


def test_dwpose_empty_batch_does_not_load_pose_checkpoint():
    loaded = []
    output, poses = image_helpers.run_dwpose_batch(
        torch.zeros(2, 64, 64, 3), resolution=64,
        loader=lambda kind: loaded.append(kind) or kind,
        forward=lambda kind, images: np.zeros((len(images), 8400, 85), np.float32),
    )
    assert loaded == ["detector"]
    assert [pose["people"] for pose in poses] == [[], []]
    assert not output.any()


def test_dwpose_coordinates_and_confidence_follow_crop_to_canvas_mapping():
    crop, center, scale = image_helpers.prepare_dwpose_crop(np.zeros((100, 200, 3), np.uint8), [40, 10, 120, 90])
    assert crop.shape == (384, 288, 3)
    x, y = np.zeros((133, 576), np.float32), np.zeros((133, 768), np.float32)
    x[:, 288], y[:, 384] = 1, 1
    x[5] = 0
    person = image_helpers.decode_dwpose_person(x, y, center, scale, 200, 100)
    assert person["pose_keypoints_2d"][:3] == pytest.approx([0.4, 0.5, 1])
    assert person["pose_keypoints_2d"][3:6] == [0, 0, 0]


def test_dwpose_node_uses_shared_output_contract(monkeypatch):
    expected = (torch.zeros(2, 64, 64, 3), [{"people": []}, {"people": []}])
    calls = []
    monkeypatch.setattr(image_nodes, "run_dwpose_batch", lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    result = image_nodes.UC_DWPoseEstimator.execute(expected[0], batch_size=2)
    assert result.result[1] is expected[1]
    assert calls[0][1]["forward"] is model_helpers.dwpose_forward
    assert "openpose_json" in result.ui


@pytest.mark.parametrize("family,kind", [("openpose", "body"), ("openpose", "hand"), ("openpose", "face"), ("dwpose", "detector"), ("dwpose", "pose")])
def test_available_safetensors_load_through_uel_without_inference(monkeypatch, family, kind):
    filename = model_helpers.CHECKPOINTS[kind][0] if family == "openpose" else model_helpers.DWPOSE_CHECKPOINTS[kind]
    path = model_helpers.folder_paths.get_full_path(model_helpers.MODEL_FOLDER, filename)
    if path is None:
        pytest.skip("Converted local checkpoint is not installed")
    monkeypatch.setattr(model_helpers.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(model_helpers.comfy.model_management, "unet_offload_device", lambda: torch.device("cpu"))
    patcher = model_helpers.load_openpose_model(kind) if family == "openpose" else model_helpers.load_dwpose_model(kind)
    assert not patcher.model.training
    loaded = patcher.model.state_dict()
    with model_helpers.MemoryEfficientSafeOpen(path, low_memory=True) as source:
        assert set(loaded) == set(source.keys())
        first, last = next(iter(loaded)), next(reversed(loaded))
        assert torch.equal(loaded[first], source.get_tensor(first))
        assert torch.equal(loaded[last], source.get_tensor(last))


def test_pose_download_uses_registered_directory_and_reuses_existing_file(monkeypatch, tmp_path):
    from utils_collection_pose_test import model_assets
    source = tmp_path / "cache.safetensors"
    source.write_bytes(b"verified-test-checkpoint")
    destination = tmp_path / "models" / "controlnet" / "preprocessors"
    target = destination / "model.safetensors"
    calls = []
    monkeypatch.setattr(model_assets.folder_paths, "get_full_path", lambda category, filename: str(target) if target.exists() else None)
    monkeypatch.setattr(model_assets.folder_paths, "get_folder_paths", lambda category: [str(destination)])
    monkeypatch.setattr(model_assets, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or str(source))
    actual = model_assets.download_huggingface_model("controlnet_preprocessors", target.name, "owner/repo", "detectors/model.safetensors")
    assert pathlib.Path(actual) == target and target.read_bytes() == source.read_bytes()
    model_assets.download_huggingface_model("controlnet_preprocessors", target.name, "owner/repo", "detectors/model.safetensors")
    assert calls == [{"repo_id": "owner/repo", "filename": "detectors/model.safetensors"}]
