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
        body_decoder=lambda heat, paf, **kwargs: [_body(float(heat[0, 0, 0]) * 10)],
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
    assert image_helpers.decode_body(heat, paf, body_threshold=0.9) == []
    assert image_helpers.decode_body(heat, paf, min_body_parts=5) == []
    assert image_helpers.decode_body(heat, paf, min_body_score=2.0) == []
    assert image_helpers.decode_body(heat, paf, limb_threshold=1.0) == []


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
    result = image_nodes.UC_BatchedOpenPose.execute(expected[0], batch_size=2, body_threshold=0.2, face_threshold=0.15,
                                                   hand_threshold=0.12, temporal_filter=True, temporal_match_iou=0.4)
    assert result.result[0] is expected[0] and result.result[1] is expected[1]
    assert calls[0][0][2] == 2
    assert calls[0][1]["loader"] is model_helpers.load_openpose_model
    assert calls[0][1]["body_threshold"] == 0.2
    assert calls[0][1]["face_threshold"] == 0.15
    assert calls[0][1]["hand_threshold"] == 0.12
    assert calls[0][1]["temporal_filter"] is True
    assert calls[0][1]["temporal_match_iou"] == 0.4
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
    result = image_nodes.UC_DWPoseEstimator.execute(expected[0], batch_size=2, detection_threshold=0.6, keypoint_threshold=0.5)
    assert result.result[1] is expected[1]
    assert calls[0][1]["forward"] is model_helpers.dwpose_forward
    assert calls[0][1]["detection_threshold"] == 0.6
    assert calls[0][1]["keypoint_threshold"] == 0.5
    assert "openpose_json" in result.ui


def test_dwpose_thresholds_filter_detections_and_uncertain_joints_independently():
    prediction = np.zeros((8400, 85), np.float32)
    prediction[1000, 4:6] = [1, 0.4]
    prediction[1100, 4:6] = [1, 0.8]
    assert len(image_helpers.decode_dwpose_boxes(prediction, 1)) == 2
    assert len(image_helpers.decode_dwpose_boxes(prediction, 1, detection_threshold=0.6)) == 1
    x, y = np.zeros((133, 576), np.float32), np.zeros((133, 768), np.float32)
    x[:, 288], y[:, 384] = 0.4, 0.4
    x[0, 288], y[0, 384] = 0.8, 0.8
    low = image_helpers.decode_dwpose_person(x, y, np.array([50, 50]), np.array([75, 100]), 100, 100)
    high = image_helpers.decode_dwpose_person(x, y, np.array([50, 50]), np.array([75, 100]), 100, 100, keypoint_threshold=0.6)
    assert low["hand_left_keypoints_2d"] is not None
    assert high["hand_left_keypoints_2d"] is None
    assert high["face_keypoints_2d"] is None
    assert high["pose_keypoints_2d"][:3] == [0.5, 0.5, 1.0]
    assert high["pose_keypoints_2d"][3:6] == [0.0, 0.0, 0.0]
    empty = image_helpers.decode_dwpose_person(x * 0, y * 0, np.array([50, 50]), np.array([75, 100]), 100, 100, keypoint_threshold=0)
    assert not any(empty["pose_keypoints_2d"])


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


@pytest.mark.parametrize("family", ["human", "animal"])
def test_rtmpose_head_matches_exported_gated_attention_math(family):
    # Synthetic weights test the exported math without running a trained model.
    def values(shape, amplitude=0.03):
        return torch.sin(torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(shape)) * amplitude
    state = types.SimpleNamespace(**{
        "onnx_initializer_0": torch.tensor(1e-5, dtype=torch.float64),
        "onnx_initializer_1": torch.linspace(0.5, 1.5, 108, dtype=torch.float64),
        "onnx_initializer_2": values((108, 256)),
        "onnx_initializer_3": torch.linspace(0.7, 1.3, 256, dtype=torch.float64),
        "onnx_initializer_4": values((256, 1152)),
        "onnx_initializer_5": values((1, 1, 2, 128), 0.7),
        "onnx_initializer_6": values((1, 1, 2, 128), 0.2),
        "onnx_initializer_7": values((512, 256)),
        "onnx_initializer_8": torch.linspace(0.1, 0.9, 256, dtype=torch.float64),
        "onnx_initializer_9": values((256, 5)),
        "onnx_initializer_10": values((256, 7)),
    })
    head = types.SimpleNamespace(
        initializers=state, Constant_277=types.SimpleNamespace(value=108 ** -0.5),
        Constant_286=types.SimpleNamespace(value=256 ** -0.5), Constant_303=types.SimpleNamespace(value=128 ** 0.5),
        _silu=torch.nn.functional.silu,
    )
    features = values((2, 3, 6, 18), 2)
    flat = features.flatten(2)
    normalized = flat / flat.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-5)
    embedding = torch.einsum("bnf,fd->bnd", normalized * state.onnx_initializer_1, state.onnx_initializer_2)
    normalized = embedding / embedding.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-5)
    projected = torch.nn.functional.silu(torch.einsum("bnd,de->bne", normalized * state.onnx_initializer_3, state.onnx_initializer_4))
    gate, value, base = projected[..., :512], projected[..., 512:1024], projected[..., 1024:]
    query = base * state.onnx_initializer_5[..., 0, :] + state.onnx_initializer_6[..., 0, :]
    key = base * state.onnx_initializer_5[..., 1, :] + state.onnx_initializer_6[..., 1, :]
    attention = (torch.einsum("bnc,bmc->bnm", query, key) / np.sqrt(128)).clamp_min(0).square()
    attended = gate * torch.einsum("bnm,bmc->bnc", attention, value)
    result = torch.einsum("bnc,cd->bnd", attended, state.onnx_initializer_7) + embedding * state.onnx_initializer_8
    expected = (result @ state.onnx_initializer_9, result @ state.onnx_initializer_10)
    if family == "human":
        actual = model_helpers.RTMPoseEstimator._head(head, features)
    else:
        animal_state = {"onnx_initializer_4": head.Constant_277.value, "onnx_initializer_7": head.Constant_286.value,
                        "onnx_initializer_12": head.Constant_303.value}
        for source, target in zip(range(1, 11), (5, 6, 8, 9, 10, 11, 13, 14, 15, 16)):
            animal_state[f"onnx_initializer_{target}"] = getattr(state, f"onnx_initializer_{source}")
        actual = model_helpers.AP10KPoseEstimator._head(types.SimpleNamespace(initializers=types.SimpleNamespace(**animal_state)), features)
    for predicted, reference in zip(actual, expected):
        torch.testing.assert_close(predicted, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("shortcut", [True, False])
def test_rtmpose_bottleneck_activates_projection_and_respects_shortcut(shortcut):
    block = types.SimpleNamespace(
        Conv_1=lambda x: x * 2 - 1, Conv_2=lambda x: x * 0.5 + 0.3,
        Conv_3=lambda x: x * 0.25 - 0.2, _silu=torch.nn.functional.silu,
    )
    value = torch.tensor([-3., -1., 0., 2.])
    expected = torch.nn.functional.silu(torch.nn.functional.silu(torch.nn.functional.silu(value * 2 - 1) * 0.5 + 0.3) * 0.25 - 0.2)
    if shortcut:
        expected = expected + value
    actual = model_helpers.RTMPoseEstimator._residual(block, value, (1, 2, 3), shortcut=shortcut)
    torch.testing.assert_close(actual, expected)


def _temporal_person(x):
    return {"pose_keypoints_2d": [coordinate for _ in range(18) for coordinate in (x, 0.5, 1.0)],
            "face_keypoints_2d": None, "hand_left_keypoints_2d": None, "hand_right_keypoints_2d": None}


def test_temporal_filter_prunes_only_unsupported_joints_not_people():
    documents = [{"canvas_width": 100, "canvas_height": 100, "people": [_temporal_person(0.25)]} for _ in range(7)]
    boxes = [[[10, 10, 40, 90]] for _ in documents]
    documents[3]["people"][0]["pose_keypoints_2d"][30:33] = [0.9, 0.9, 1]
    documents[3]["people"].append(_temporal_person(0.75))
    boxes[3].append([60, 10, 90, 90])
    filtered = image_helpers.prune_temporal_pose_keypoints(documents, boxes)
    assert len(filtered[3]["people"]) == 2
    assert filtered[3]["people"][0]["pose_keypoints_2d"][30:33] == [0, 0, 0]
    assert filtered[3]["people"][0]["pose_keypoints_2d"][27:30] == [0.25, 0.5, 1]
    assert not any(filtered[3]["people"][1]["pose_keypoints_2d"])
    assert filtered[0]["people"][0]["pose_keypoints_2d"][30:33] == [0.25, 0.5, 1]
    assert documents[3]["people"][0]["pose_keypoints_2d"][30:33] == [0.9, 0.9, 1]
    assert documents[3]["people"][1]["pose_keypoints_2d"][2] == 1


def test_temporal_filter_matches_people_one_to_one_when_detection_order_changes():
    documents, boxes = [], []
    for index in range(5):
        people = [_temporal_person(0.25), _temporal_person(0.75)]
        frame_boxes = [[10, 10, 40, 90], [60, 10, 90, 90]]
        if index % 2:
            people.reverse()
            frame_boxes.reverse()
        documents.append({"canvas_width": 100, "canvas_height": 100, "people": people})
        boxes.append(frame_boxes)
    assert image_helpers.prune_temporal_pose_keypoints(documents, boxes) == documents
    assert image_helpers.prune_temporal_pose_keypoints(documents[:1], boxes[:1]) == documents[:1]


def test_temporal_filter_rejects_isolated_knee_label_jumps_at_person_scale():
    documents = [{"canvas_width": 1024, "canvas_height": 576, "people": [_temporal_person(0.5)]} for _ in range(7)]
    for frame in documents:
        points = frame["people"][0]["pose_keypoints_2d"]
        points[27:30] = [495 / 1024, 385 / 576, 1]
        points[30:33] = [487 / 1024, 470 / 576, 1]
        points[36:39] = [592 / 1024, 300 / 576, 1]
    middle = documents[3]["people"][0]["pose_keypoints_2d"]
    middle[27:30] = [591 / 1024, 390 / 576, 1]
    middle[36:39] = [495 / 1024, 389 / 576, 1]
    boxes = [[[440, 120, 610, 480]] for _ in documents]
    filtered = image_helpers.prune_temporal_pose_keypoints(documents, boxes)
    points = filtered[3]["people"][0]["pose_keypoints_2d"]
    assert points[27:30] == [0, 0, 0] and points[36:39] == [0, 0, 0]
    assert points[30:33] == [487 / 1024, 470 / 576, 1]
    assert filtered[2]["people"][0]["pose_keypoints_2d"][29] == 1
    assert filtered[4]["people"][0]["pose_keypoints_2d"][38] == 1


def test_temporal_filter_runs_before_rendering_and_across_processing_chunks(monkeypatch):
    sequence = {"index": 0}
    def decode(*args):
        index = sequence["index"]
        sequence["index"] += 1
        person = _temporal_person(0.25)
        if index == 3:
            person["pose_keypoints_2d"][30:33] = [0.9, 0.9, 1]
        return person
    def forward(kind, inputs):
        if kind == "detector":
            output = np.zeros((len(inputs), 8400, 85), np.float32)
            output[:, 500, 4:6] = 1
            return output
        return np.zeros((len(inputs), 133, 576), np.float32), np.zeros((len(inputs), 133, 768), np.float32)
    monkeypatch.setattr(image_helpers, "decode_dwpose_person", decode)
    monkeypatch.setattr(image_helpers, "draw_pose_frame", lambda people, h, w, *args: np.full((h, w, 3), 255 if people[0]["pose_keypoints_2d"][32] else 0, np.uint8))
    inputs = torch.zeros(7, 64, 64, 3)
    first_image, first_data = image_helpers.run_dwpose_batch(inputs, 64, 2, loader=lambda kind: kind, forward=forward, temporal_filter=True)
    sequence["index"] = 0
    second_image, second_data = image_helpers.run_dwpose_batch(inputs, 64, 5, loader=lambda kind: kind, forward=forward, temporal_filter=True)
    assert first_data == second_data
    assert torch.equal(first_image, second_image)
    assert not first_image[3].any()
    assert bool(first_image[2].all()) and bool(first_image[4].all())


def test_openpose_face_threshold_preserves_landmark_slots_for_temporal_matching():
    heatmap = np.zeros((16, 16, 3), np.float32)
    heatmap[4, 5, 0] = 0.1
    heatmap[7, 8, 1] = 0.8
    low = image_helpers.decode_face(heatmap, (10, 10, 32), 100, 100)
    high = image_helpers.decode_face(heatmap, (10, 10, 32), 100, 100, threshold=0.5)
    assert len(low) == len(high) == 9
    assert low[:3] == [0.2, 0.18, 1]
    assert high[:3] == [0, 0, 0]
    assert high[3:6] == low[3:6]
    assert high[6:] == [0, 0, 0]


def test_openpose_temporal_filter_spans_chunks_before_rendering(monkeypatch):
    sequence = {"index": 0}
    def decode(heat, paf, **kwargs):
        body = _body()
        body[9], body[10] = (65, 110), (65, 124)
        if sequence["index"] == 3:
            body[9] = (170, 110)
        sequence["index"] += 1
        return [body]
    def forward(kind, images):
        return np.zeros((len(images), 23, 32, 38), np.float32), np.zeros((len(images), 23, 32, 19), np.float32)
    monkeypatch.setattr(image_helpers, "draw_pose_frame", lambda people, h, w, *args: np.full((h, w, 3), 255 if people[0]["pose_keypoints_2d"][29] else 0, np.uint8))
    frames = torch.zeros(7, 128, 192, 3)
    outputs = []
    for batch_size in (2, 5):
        sequence["index"] = 0
        outputs.append(image_helpers.run_openpose_batch(frames, 128, batch_size, detect_hand=False, detect_face=False,
                                                       loader=lambda kind: kind, forward=forward, body_decoder=decode, temporal_filter=True))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]
    assert not outputs[0][0][3].any()
    assert bool(outputs[0][0][2].all()) and bool(outputs[0][0][4].all())


def test_animalpose_batches_only_animal_classes_and_preserves_ap10k_format():
    calls = []
    def forward(kind, frames):
        calls.append((kind, len(frames)))
        if kind == "detector":
            result = np.zeros((len(frames), 8400, 85), np.float32)
            result[:, 1000, 4] = 1
            result[:, 1000, 5 + 16] = 0.8
            result[:, 1100, 4:6] = 1
            return result
        x, y = np.zeros((len(frames), 17, 512), np.float32), np.zeros((len(frames), 17, 512), np.float32)
        x[..., 256] = 0.9
        y[..., 256] = 0.8
        x[:, 0, 256] = 0.1
        assert frames[0].shape == (256, 256, 3)
        return x, y
    output, poses = image_helpers.run_dwpose_batch(torch.zeros(5, 64, 96, 3), 64, 3,
                                                 loader=lambda kind: kind, forward=forward, animal=True)
    assert calls == [("detector", 3), ("pose", 3), ("detector", 2), ("pose", 2)]
    assert output.shape == (5, 64, 96, 3)
    assert all(frame["version"] == "ap10k" and len(frame["animals"]) == 1 for frame in poses)
    assert len(poses[0]["animals"][0]) == 17
    assert poses[0]["animals"][0][0] == [0, 0, 0]
    assert poses[0]["animals"][0][1][2] == pytest.approx(0.8)


def test_animal_temporal_filter_prunes_joint_without_removing_animal():
    frames = [{"version": "ap10k", "canvas_width": 100, "canvas_height": 100,
               "animals": [[[25., 50., 0.8] for _ in range(17)]]} for _ in range(7)]
    frames[3]["animals"][0][8] = [90., 90., 0.9]
    filtered = image_helpers.prune_temporal_animal_keypoints(frames, [[[10, 10, 40, 90]]] * 7)
    assert len(filtered[3]["animals"]) == 1
    assert filtered[3]["animals"][0][8] == [0, 0, 0]
    assert filtered[3]["animals"][0][7] == [25, 50, 0.8]
    assert frames[3]["animals"][0][8] == [90, 90, 0.9]


def test_densepose_batches_frames_renders_parts_and_handles_empty_results():
    calls = []
    def forward(model, frames, **options):
        calls.append((len(frames), options))
        output = []
        for frame in frames:
            if frame[0, 0, 0] > 100:
                output.append((torch.zeros(0, 4), torch.zeros(0, 2, 2, 2), *(torch.zeros(0, 25, 2, 2) for _ in range(3))))
            else:
                coarse = torch.zeros(1, 2, 2, 2)
                fine = torch.zeros(1, 25, 2, 2)
                coarse[:, 1] = 1
                fine[:, 4] = 1
                output.append((torch.tensor([[10., 10., 30., 30.]]), coarse, fine, fine * 0, fine * 0))
        return output
    frames = torch.stack([torch.full((64, 96, 3), value) for value in (0., 1., 0.)])
    output = image_helpers.run_densepose_batch(frames, 64, 2, "viridis", loader=lambda: object(), forward=forward, score_threshold=0.4, max_detections=20)
    assert [size for size, options in calls] == [2, 1]
    assert all(options["score_threshold"] == 0.4 and options["max_detections"] == 20 for size, options in calls)
    background = torch.tensor([68, 1, 84], dtype=torch.float32) / 255
    torch.testing.assert_close(output[1, 15, 15], background)
    assert not torch.equal(output[0, 15, 15], background)
    torch.testing.assert_close(output[0], output[2])
    blank = (torch.zeros(0, 4), torch.zeros(0, 2, 2, 2), *(torch.zeros(0, 25, 2, 2) for _ in range(3)))
    assert not image_helpers.render_densepose_frame(blank, 64, 96, "parula").any()


@pytest.mark.parametrize("kind,loader", [("animalpose", lambda: model_helpers.load_animal_pose_model("pose")),
                                         ("densepose_r50", model_helpers.load_densepose_model)])
def test_new_pose_migrations_load_installed_safetensors_without_inference(monkeypatch, kind, loader):
    specification = model_helpers.get_model_migration(kind)
    path = model_helpers.folder_paths.get_full_path(model_helpers.MODEL_FOLDER, specification["filename"])
    if path is None:
        pytest.skip("Converted local checkpoint not installed")
    monkeypatch.setattr(model_helpers.comfy.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(model_helpers.comfy.model_management, "unet_offload_device", lambda: torch.device("cpu"))
    patcher = loader()
    assert type(patcher.model).__name__ == specification["architecture"]
    with model_helpers.MemoryEfficientSafeOpen(path, low_memory=True) as source:
        loaded = patcher.model.state_dict()
        assert set(loaded) == set(source.keys())
        last = next(reversed(loaded))
        assert torch.equal(loaded[last], source.get_tensor(last))


def test_animal_and_densepose_node_controls_reach_shared_helpers(monkeypatch):
    calls = []
    image = torch.zeros(1, 64, 64, 3)
    monkeypatch.setattr(image_nodes, "run_dwpose_batch", lambda *args, **kwargs: calls.append(kwargs) or (image, []))
    monkeypatch.setattr(image_nodes, "run_densepose_batch", lambda *args, **kwargs: calls.append(kwargs) or image)
    image_nodes.UC_AnimalPoseEstimator.execute(image, detection_threshold=0.6, temporal_filter=True)
    image_nodes.UC_DensePoseEstimator.execute(image, score_threshold=0.4, rpn_nms_threshold=0.6)
    assert calls[0]["animal"] is True and calls[0]["detection_threshold"] == 0.6 and calls[0]["temporal_filter"] is True
    assert calls[1]["score_threshold"] == 0.4 and calls[1]["rpn_nms_threshold"] == 0.6
