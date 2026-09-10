import json
import pathlib
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_composite_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_composite_test import (
        background_replace_helpers,
        composite_helpers,
        composite_nodes,
        foreground_content_helpers,
        image_helpers,
        image_nodes,
        model_assets,
        staged_compositor_helpers,
        staged_face_helpers,
    )
    from utils_collection_composite_test.helper_functions import resize_nchw
finally:
    cli_args.cpu = prior_cpu


def _paint_test_stage():
    return {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.tensor([[[[1.0, 0.0, 0.0]]]]).expand(1, 4, 4, 3),
                "mask": torch.ones(1, 4, 4),
                "uses_embedded_alpha": True,
            }
        ],
    }


def _paint_placement(order, included=True):
    return json.dumps(
        {
            "version": 3,
            "workspace_padding": 0,
            "layer_order": order,
            "layers": {
                "foreground_0": {
                    "scale": 1,
                    "center_x": 0.5,
                    "center_y": 0.5,
                }
            },
            "paint_layer": {
                "included": included,
                "asset": {
                    "filename": "paint.png",
                    "subfolder": "clipspace",
                    "type": "input",
                },
            },
        }
    )


@pytest.mark.parametrize("individual", [False, True])
@pytest.mark.parametrize("socket", ["foreground_0", "foreground_0_face_0"])
def test_foreground_annotations_fill_transparency_keep_indices_and_leave_stage_unchanged(tmp_path, monkeypatch, individual, socket):
    monkeypatch.setattr(foreground_content_helpers.folder_paths, "get_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(staged_compositor_helpers, "_save_editor_preview", lambda *args: {})
    brush = np.zeros((4, 4, 4), dtype=np.uint8)
    brush[0, 0] = (0, 0, 255, 255)
    text = np.zeros_like(brush)
    text[0, 0] = (0, 255, 0, 128)
    Image.fromarray(brush).save(tmp_path / "brush.png")
    Image.fromarray(text).save(tmp_path / "text.png")
    stage = _paint_test_stage()
    stage["layers"][0]["socket"] = socket
    stage["layers"][0]["mask"][0, 0, 0] = 0
    stage["layers"][0]["is_face"] = "face" in socket
    original_image = stage["layers"][0]["image"].clone()
    original_mask = stage["layers"][0]["mask"].clone()
    data = json.loads(_paint_placement([socket]))
    data["layers"] = {socket: data["layers"]["foreground_0"]}
    data["foreground_content"] = {socket: {
        kind: {"visible": True, "asset": {"filename": f"{kind}.png", "subfolder": "", "type": "input"}}
        for kind in ("brush", "text")
    }}
    render = staged_compositor_helpers._composite_staged_individual_foregrounds if individual else staged_compositor_helpers._composite_staged_foregrounds
    result = render(torch.zeros(1, 4, 4, 3), stage, json.dumps(data), 0)
    expected = torch.tensor([0.0, 128 / 255, 127 / 255])
    torch.testing.assert_close(result.result[0][0, 0, 0], expected)
    assert result.result[1][0, 0, 0] == 1
    assert len(result.result[2][0]) == 1
    torch.testing.assert_close(stage["layers"][0]["image"], original_image)
    torch.testing.assert_close(stage["layers"][0]["mask"], original_mask)

    data["foreground_content"][socket]["text"]["visible"] = False
    brush_only = render(torch.zeros(1, 4, 4, 3), stage, json.dumps(data), 0)
    torch.testing.assert_close(brush_only.result[0][0, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    data["foreground_content"][socket]["brush"]["visible"] = False
    hidden = render(torch.zeros(1, 4, 4, 3), stage, json.dumps(data), 0)
    assert hidden.result[1][0, 0, 0] == 0
    assert not hidden.result[0][0, 0, 0].any()


@pytest.mark.parametrize("flip_h,flip_v,rotation,target", [
    (True, False, 0, (0, 3)), (False, True, 0, (3, 0)), (False, False, 180, (3, 3)),
])
def test_foreground_annotations_follow_transform_after_restaging(tmp_path, monkeypatch, flip_h, flip_v, rotation, target):
    monkeypatch.setattr(foreground_content_helpers.folder_paths, "get_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(staged_compositor_helpers, "_save_editor_preview", lambda *args: {})
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[0, 0] = (0, 0, 255, 255)
    Image.fromarray(pixels).save(tmp_path / "brush.png")
    stage = _paint_test_stage()
    stage["layers"][0]["flip_horizontal"] = flip_h
    stage["layers"][0]["flip_vertical"] = flip_v
    data = json.loads(_paint_placement(["foreground_0"]))
    data["layers"]["foreground_0"].update(flip_horizontal=flip_h, flip_vertical=flip_v, rotation=rotation)
    data["foreground_content"] = {"foreground_0": {"brush": {
        "asset": {"filename": "brush.png", "subfolder": "", "type": "input"},
    }}}
    result = staged_compositor_helpers._composite_staged_foregrounds(torch.zeros(1, 4, 4, 3), stage, json.dumps(data), 0)
    torch.testing.assert_close(result.result[0][(0, *target)], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)


def test_foreground_assets_reject_missing_files_and_input_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(foreground_content_helpers.folder_paths, "get_input_directory", lambda: str(tmp_path))
    for filename, message in [("missing.png", "missing"), ("../outside.png", "inside")]:
        with pytest.raises(ValueError, match=message):
            foreground_content_helpers.load_foreground_rgba({"filename": filename, "subfolder": ""}, torch.zeros(1))


@pytest.mark.parametrize("individual", [False, True])
@pytest.mark.parametrize("flipped", [False, True])
def test_object_eraser_reduces_original_alpha_without_erasing_text_or_mutating_stage(tmp_path, monkeypatch, individual, flipped):
    monkeypatch.setattr(foreground_content_helpers.folder_paths, "get_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(staged_compositor_helpers, "_save_editor_preview", lambda *args: {})
    eraser = np.zeros((4, 4, 4), dtype=np.uint8)
    eraser[0, 0] = (255, 255, 255, 128)
    eraser[0, 3] = (255, 255, 255, 255)
    eraser[1, 1] = (255, 255, 255, 255)
    text = np.zeros_like(eraser)
    text[1, 1] = (0, 255, 0, 255)
    Image.fromarray(eraser).save(tmp_path / "erase.png")
    Image.fromarray(text).save(tmp_path / "text.png")
    asset = lambda name: {"filename": name, "subfolder": "", "type": "input"}
    data = json.loads(_paint_placement(["foreground_0"]))
    data["layers"]["foreground_0"]["flip_horizontal"] = flipped
    data["foreground_content"] = {"foreground_0": {
        "brush": {"visible": True}, "object_erase": {"asset": asset("erase.png")},
        "text": {"asset": asset("text.png")},
    }}
    stage = _paint_test_stage()
    render = staged_compositor_helpers._composite_staged_individual_foregrounds if individual else staged_compositor_helpers._composite_staged_foregrounds
    result = render(torch.zeros(1, 4, 4, 3), stage, json.dumps(data), 0)
    soft_x, hole_x, text_x = (3, 0, 2) if flipped else (0, 3, 1)
    assert result.result[1][0, 0, soft_x] == pytest.approx(127 / 255)
    assert result.result[1][0, 0, hole_x] == 0
    assert not result.result[0][0, 0, hole_x].any()
    torch.testing.assert_close(result.result[0][0, 1, text_x], torch.tensor([0.0, 1.0, 0.0]))
    assert bool(torch.all(stage["layers"][0]["mask"] == 1))
    assert len(result.result[2][0]) == 1
    data["foreground_content"]["foreground_0"]["brush"]["visible"] = False
    restored = render(torch.zeros(1, 4, 4, 3), stage, json.dumps(data), 0)
    assert restored.result[1][0, 0, hole_x] == 1
    torch.testing.assert_close(restored.result[0][0, 0, hole_x], torch.tensor([1.0, 0.0, 0.0]))


def test_resize_mask_preserves_asymmetric_orientation():
    mask = torch.zeros(1, 2, 3)
    mask[0, 0, 0] = 1.0

    output = composite_nodes.UC_ResizeMask.execute(
        mask, 6, 4, False, "nearest-exact", "disabled"
    )
    resized, width, height = output.result

    assert (width, height) == (6, 4)
    assert torch.equal(resized[0, :2, :2], torch.ones(2, 2))
    assert resized[0, 2:, :].sum() == 0
    assert resized[0, :, 2:].sum() == 0


def test_image_and_mask_resize_uses_target_and_optional_overrides():
    image = torch.zeros(1, 2, 3, 3)
    image[0, 0, 0, 0] = 1.0
    mask = image[..., 0]
    target = torch.zeros(1, 8, 10, 3)

    target_sized = composite_nodes.UC_ImageAndMaskResize.execute(
        image, mask, target, "nearest-exact", "disabled", 0
    )
    width_overridden = composite_nodes.UC_ImageAndMaskResize.execute(
        image, mask, target, "nearest-exact", "disabled", 0, width=6
    )

    assert target_sized.result[0].shape == (1, 8, 10, 3)
    assert target_sized.result[1].shape == (1, 8, 10)
    assert width_overridden.result[0].shape == (1, 8, 6, 3)
    assert width_overridden.result[1].shape == (1, 8, 6)
    assert target_sized.result[0][0, :4, :4, 0].sum() > 0
    assert target_sized.result[0][0, 4:, :, 0].sum() == 0


def test_fp32_resize_bypasses_equal_dimensions_and_lanczos_does_not_quantize():
    image = torch.linspace(0.0003, 0.9991, 8 * 10 * 3, dtype=torch.float32).reshape(
        1, 3, 8, 10
    )

    unchanged = resize_nchw(image, 10, 8, "lanczos")
    reduced = resize_nchw(image, 5, 4, "lanczos")

    assert unchanged.data_ptr() == image.data_ptr()
    assert torch.equal(unchanged, image)
    assert ((reduced * 255.0) - (reduced * 255.0).round()).abs().max() > 1e-4


def test_exact_size_crop_merge_and_image_mask_resize_are_bit_exact():
    image = torch.rand(1, 8, 10, 3)
    mask = torch.rand(1, 8, 10)

    merged = composite_nodes.UC_ImageCropMerge.execute(
        image, torch.zeros_like(image), 0, 0, 10, 8, "lanczos"
    ).result[0]
    resized_image, resized_mask = composite_nodes.UC_ImageAndMaskResize.execute(
        image, mask, image, "lanczos", "disabled", 0
    ).result

    assert torch.equal(merged, image)
    assert torch.equal(resized_image, image)
    assert torch.equal(resized_mask, mask)


def test_image_pad_equal_target_preserves_fp32_pixels():
    image = torch.rand(1, 9, 11, 3)

    padded, _ = image_nodes.UC_ImagePad.execute(
        image,
        0,
        0,
        0,
        0,
        0,
        "0, 0, 0",
        "color",
        target_width=11,
        target_height=9,
    ).result

    assert torch.equal(padded, image)


def test_crop_by_mask_uses_batch_union_and_rejects_empty_masks():
    image = torch.zeros(2, 32, 32, 3)
    mask = torch.zeros(2, 32, 32)
    mask[0, 4:6, 4:6] = 1.0
    mask[1, 24:26, 24:26] = 1.0

    output = composite_nodes.UC_CropByMask.execute(image, mask, 0)

    assert output.result[0].shape[0] == 2
    cropped_mask, crop_x, crop_y = output.result[1:4]
    assert cropped_mask[0, 4 - crop_y : 6 - crop_y, 4 - crop_x : 6 - crop_x].sum() == 4
    assert (
        cropped_mask[1, 24 - crop_y : 26 - crop_y, 24 - crop_x : 26 - crop_x].sum() == 4
    )
    with pytest.raises(ValueError, match="Mask is empty"):
        composite_nodes.UC_CropByMask.execute(image[:1], torch.zeros(1, 32, 32), 0)


def test_crop_by_mask_exposes_dimension_multiple_without_resizing():
    image = torch.arange(64 * 64 * 3, dtype=torch.float32).reshape(1, 64, 64, 3)
    mask = torch.zeros(1, 64, 64)
    mask[:, 20:29, 22:31] = 1.0

    output = composite_nodes.UC_CropByMask.execute(image, mask, padding=2, multiple=16)
    cropped_image, cropped_mask, crop_x, crop_y, width, height = output.result

    assert (width, height) == (16, 16)
    assert cropped_image.shape == (1, 16, 16, 3)
    assert cropped_mask.shape == (1, 16, 16)
    assert torch.equal(
        cropped_image, image[:, crop_y : crop_y + height, crop_x : crop_x + width]
    )
    assert cropped_mask.sum() == 81


def test_crop_by_mask_multiple_defaults_to_legacy_eight_pixels():
    schema = composite_nodes.UC_CropByMask.define_schema()
    multiple = next(value for value in schema.inputs if value.id == "multiple")
    mask = torch.zeros(1, 64, 64)
    mask[:, 20:29, 22:31] = 1.0

    output = composite_nodes.UC_CropByMask.execute(torch.zeros(1, 64, 64, 3), mask, 2)

    assert multiple.default == 8
    assert multiple.min == 4
    assert multiple.step == 4
    assert output.result[4:] == (16, 16)


def test_staged_layer_crops_returns_selected_layers_as_ordered_image_list():
    schema = composite_nodes.UC_StagedLayerCrops.define_schema()
    image = torch.zeros((1, 24, 32, 3), dtype=torch.float32)
    image[0, :, :, 0] = torch.arange(32, dtype=torch.float32)[None, :]
    layer_masks = torch.zeros((3, 24, 32), dtype=torch.float32)
    layer_masks[0, 4:6, 2:4] = 1.0
    layer_masks[1, 8:10, 14:16] = 1.0
    layer_masks[2, 12:14, 26:28] = 1.0

    crops = composite_nodes.UC_StagedLayerCrops.execute(
        image,
        layer_masks,
        "2, 0 1",
    ).result[0]

    assert schema.node_id == "UC_StagedLayerCrops"
    assert [value.id for value in schema.inputs] == [
        "image",
        "layer_masks",
        "layer_indices",
    ]
    assert schema.inputs[2].multiline is False
    assert schema.outputs[0].is_output_list is True
    assert len(crops) == 3
    assert all(crop.shape == (1, 8, 8, 3) for crop in crops)
    means = [float(crop[..., 0].mean()) for crop in crops]
    assert means[0] > means[2] > means[1]


@pytest.mark.parametrize("layer_indices", ["", "x", "-1", "3"])
def test_staged_layer_crops_rejects_invalid_indices(layer_indices):
    image = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    layer_masks = torch.ones((3, 16, 16), dtype=torch.float32)

    with pytest.raises(ValueError):
        composite_nodes.UC_StagedLayerCrops.execute(
            image,
            layer_masks,
            layer_indices,
        )


def test_staged_layer_crops_rejects_empty_selected_mask():
    image = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    layer_masks = torch.zeros((1, 16, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match="Layer mask index 0 is empty"):
        composite_nodes.UC_StagedLayerCrops.execute(image, layer_masks, "0")


def test_crop_merge_supports_mask_and_singleton_broadcast():
    original = torch.zeros(2, 8, 8, 3)
    crop = torch.ones(1, 4, 4, 3)
    mask = torch.zeros(1, 4, 4)
    mask[:, :, :2] = 1.0

    output = composite_nodes.UC_ImageCropMerge.execute(
        crop, original, 2, 2, 4, 4, "nearest-exact", mask
    ).result[0]

    assert torch.equal(output[:, 2:6, 2:4], torch.ones(2, 4, 2, 3))
    assert output[:, 2:6, 4:6].sum() == 0


def test_mask_expansion_and_feather_support_contraction():
    mask = torch.zeros(9, 9)
    mask[2:7, 2:7] = 1.0

    expanded = composite_helpers._expand_mask(mask, 1)
    contracted = composite_helpers._expand_mask(mask, -1)
    outward = composite_helpers._feather_mask(mask, 2)
    inward = composite_helpers._feather_mask(mask, -2)

    assert expanded.sum() > mask.sum()
    assert contracted.sum() < mask.sum()
    assert outward.sum() > mask.sum()
    assert inward.sum() < mask.sum()
    assert torch.equal(outward[2:7, 2:7], mask[2:7, 2:7])
    assert inward[:2].sum() == 0
    assert inward[:, :2].sum() == 0
    assert torch.all(inward <= mask)
    assert inward[2, 4] < inward[4, 4]


class _QueuedBackgroundModel:
    def __init__(self, masks):
        self.masks = list(masks)
        self.colors = []

    def encode_image(self, image):
        self.colors.append(image[0, 0, 0].clone())
        return self.masks.pop(0).to(image)


def _replace_background(model, background, foregrounds, **overrides):
    options = {
        "foreground_scale": 0.9,
        "long_axis_shift": 0.0,
        "short_axis_shift": 0.0,
        "mask_threshold": 0.5,
        "border_cleanup_width": 0,
        "artifact_cleanup_radius": 0,
        "gap_fill_radius": 0,
        "feather_radius": 0,
        "image_resize_method": "auto",
        "mask_resize_method": "auto",
        "workspace_padding": 0.0,
    }
    options.update(overrides)
    return composite_nodes.UC_UnifiedBackgroundReplace.execute(
        model, background, foregrounds, **options
    ).result


def test_unified_background_flattens_inputs_and_centers_foreground_bounds():
    background = torch.zeros(1, 100, 160, 3, dtype=torch.float64)
    first = torch.zeros(1, 10, 10, 3)
    first[..., 0] = 1.0
    second = torch.zeros(2, 8, 12, 3)
    second[0, ..., 1] = 1.0
    second[1, ..., 2] = 1.0
    square_mask = torch.zeros(1, 5, 5)
    square_mask[:, 1:4, 1:4] = 1.0
    wide_mask = torch.zeros(1, 8, 12)
    wide_mask[:, 2:6, 2:10] = 1.0
    tall_mask = torch.zeros(1, 8, 12)
    tall_mask[:, :, 4:8] = 1.0
    model = _QueuedBackgroundModel([square_mask, wide_mask, tall_mask])

    images, masks = _replace_background(
        model,
        background,
        {"foreground_10": second, "foreground_2": first},
    )

    assert images.shape == (3, 100, 160, 3)
    assert masks.shape == (3, 100, 160)
    assert images.dtype == background.dtype
    assert images.device == background.device
    assert [color.argmax().item() for color in model.colors] == [0, 1, 2]
    assert set(masks.unique().tolist()) == {0.0, 1.0}
    assert masks[0].sum() == 90 * 90
    assert masks[1].sum() == 45 * 90
    assert masks[2].sum() == 45 * 90
    assert masks[0, 5:95, 35:125].all()
    assert masks[1, 28:73, 35:125].all()
    assert masks[2, 5:95, 58:103].all()


def test_unified_background_refines_weak_edges_gaps_and_artifacts():
    raw = torch.zeros(12, 12)
    raw[3:10, 3:10] = 0.9
    raw[6, 6] = 0.0
    raw[0, 0:3] = 0.6
    raw[1, 11] = 0.9

    refined = composite_helpers._refine_foreground_mask(raw, 0.5, 2, 1, 1)

    assert refined[0, 0:3].sum() == 0
    assert refined[1, 11] == 0
    assert refined[6, 6] == 1
    assert refined[4:9, 4:9].all()


def test_unified_background_keeps_strong_edge_subject_and_feathers_only_boundary():
    background = torch.zeros(1, 64, 80, 3)
    foreground = torch.ones(1, 16, 16, 3)
    raw = torch.zeros(1, 16, 16)
    raw[:, 0:14, 3:13] = 0.95
    model = _QueuedBackgroundModel([raw])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        border_cleanup_width=2,
        feather_radius=2,
    )

    assert masks.max() == 1
    assert masks.min() == 0
    assert ((masks > 0) & (masks < 1)).any()
    assert masks[0, 32, 40] == 1


@pytest.mark.parametrize(
    ("background_shape", "shift", "expected_bounds"),
    [
        ((60, 100), -1.0, (3, 57, 0, 54)),
        ((60, 100), 1.0, (3, 57, 46, 100)),
        ((100, 60), -1.0, (0, 54, 3, 57)),
        ((100, 60), 1.0, (46, 100, 3, 57)),
    ],
)
def test_unified_background_shifts_along_background_long_axis(
    background_shape, shift, expected_bounds
):
    height, width = background_shape
    background = torch.zeros(1, height, width, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        long_axis_shift=shift,
    )

    top, bottom, left, right = expected_bounds
    assert masks[0, top:bottom, left:right].all()
    assert masks.sum() == (bottom - top) * (right - left)


def test_unified_background_overscale_crops_to_canvas_perimeter():
    background = torch.zeros(1, 40, 60, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    images, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        foreground_scale=2.0,
    )

    assert images.shape == background.shape
    assert masks.shape == background.shape[:3]
    assert masks.all()
    assert images.all()


@pytest.mark.parametrize(
    ("background_shape", "shift", "expected_bounds"),
    [
        ((60, 100), -1.0, (0, 54, 23, 77)),
        ((60, 100), 1.0, (6, 60, 23, 77)),
        ((100, 60), -1.0, (23, 77, 0, 54)),
        ((100, 60), 1.0, (23, 77, 6, 60)),
    ],
)
def test_unified_background_shifts_along_background_short_axis(
    background_shape, shift, expected_bounds
):
    height, width = background_shape
    background = torch.zeros(1, height, width, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        short_axis_shift=shift,
    )

    top, bottom, left, right = expected_bounds
    assert masks[0, top:bottom, left:right].all()
    assert masks.sum() == (bottom - top) * (right - left)


def test_unified_background_square_canvas_uses_both_axis_shifts():
    background = torch.zeros(1, 100, 100, 3)
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 8, 8)])

    _, masks = _replace_background(
        model,
        background,
        {"foreground_0": foreground},
        long_axis_shift=-1.0,
        short_axis_shift=1.0,
    )

    assert masks[0, 10:100, 0:90].all()
    assert masks.sum() == 90 * 90


def test_composite_auto_resize_selects_direction_appropriate_methods():
    assert composite_helpers._composite_resize_method("auto", 100, 80, 50, 40) == "area"
    assert (
        composite_helpers._composite_resize_method("auto", 50, 40, 100, 80) == "bicubic"
    )
    assert (
        composite_helpers._composite_resize_method("auto", 100, 40, 50, 80) == "bicubic"
    )
    assert (
        composite_helpers._composite_resize_method("auto", 100, 80, 50, 40, mask=True)
        == "area"
    )
    assert (
        composite_helpers._composite_resize_method("auto", 50, 40, 100, 80, mask=True)
        == "bilinear"
    )
    assert (
        composite_helpers._composite_resize_method("auto", 100, 40, 50, 80, mask=True)
        == "bilinear"
    )
    assert (
        composite_helpers._composite_resize_method("nearest-exact", 100, 80, 50, 40)
        == "nearest-exact"
    )


def test_background_compositor_schemas_place_variable_foregrounds_after_static_controls():
    schemas = [
        composite_nodes.UC_UnifiedBackgroundReplace.define_schema(),
        composite_nodes.UC_LayeredBackgroundComposite.define_schema(),
        composite_nodes.UC_StagedLayeredBackgroundComposite.define_schema(),
    ]
    unified, layered, staged = (
        [value.id for value in schema.inputs] for schema in schemas
    )

    assert unified[-1] == "foreground_images"
    assert unified[-4:-1] == [
        "image_resize_method",
        "mask_resize_method",
        "workspace_padding",
    ]
    assert layered[-1] == "foreground_images"
    assert layered[-4:-1] == [
        "image_resize_method",
        "mask_resize_method",
        "placement_data",
    ]
    assert staged[-1] == "foreground_images"
    assert staged[-3:-1] == ["placement_data", "background_removal_model_name"]
    assert not {
        "mask_threshold",
        "border_cleanup_width",
        "artifact_cleanup_radius",
        "gap_fill_radius",
        "feather_radius",
        "image_resize_method",
        "mask_resize_method",
    } & set(staged)
    for schema in schemas:
        assert all(value.tooltip for value in schema.inputs)
    for schema in schemas[2:]:
        assert [value.id for value in schema.outputs] == [
            "image",
            "mask",
            "bounding_boxes",
            "layer_masks",
        ]


def test_nonstaged_background_and_face_models_are_optional():
    for node in (
        composite_nodes.UC_UnifiedBackgroundReplace,
        composite_nodes.UC_LayeredBackgroundComposite,
    ):
        model_input = next(
            value
            for value in node.define_schema().inputs
            if value.id == "background_removal_model"
        )
        assert model_input.optional is True
        assert model_input.display_name == "background_removal_model_opt"

    face_inputs = {
        value.id: value
        for value in composite_nodes.UC_MediaPipeFaceComposite.define_schema().inputs
    }
    assert face_inputs["face_detection_model"].optional is True
    assert face_inputs["face_detection_model"].display_name == "face_detection_model_opt"
    assert face_inputs["background_removal_model"].optional is True
    assert (
        face_inputs["background_removal_model"].display_name
        == "background_removal_model_opt"
    )


def test_background_model_resolver_uses_internal_birefnet_only_when_disconnected(
    monkeypatch,
):
    internal = object()
    external = object()
    requested = []
    monkeypatch.setattr(
        composite_helpers,
        "_load_internal_background_removal_model",
        lambda name: requested.append(name) or internal,
    )

    assert composite_helpers.resolve_background_removal_model(None) is internal
    assert composite_helpers.resolve_background_removal_model(external) is external
    assert requested == ["birefnet"]


def test_nonstaged_compositors_resolve_disconnected_background_model(monkeypatch):
    fallback = _QueuedBackgroundModel(
        [torch.ones(1, 4, 4), torch.ones(1, 4, 4)]
    )
    requested = []
    monkeypatch.setattr(
        composite_nodes,
        "resolve_background_removal_model",
        lambda model: requested.append(model) or fallback,
    )
    background = torch.zeros(1, 16, 16, 3)
    foreground = torch.ones(1, 4, 4, 3)

    _replace_background(
        None, background, {"foreground_0": foreground}, foreground_scale=0.5
    )
    _layered_composite(
        None, background, {"foreground_0": foreground}
    )

    assert requested == [None, None]


def test_composite_smooth_mask_resize_preserves_subpixel_coverage():
    mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    smooth = composite_helpers._resize_composite_mask(mask, 4, 4, "auto")
    hard = composite_helpers._resize_composite_mask(mask, 4, 4, "nearest-exact")

    assert ((smooth > 0) & (smooth < 1)).any()
    assert set(hard.unique().tolist()) == {0.0, 1.0}


def test_unified_workspace_padding_allows_partial_and_fully_hidden_foregrounds():
    background = torch.zeros(1, 100, 100, 3)
    foreground = torch.ones(1, 8, 8, 3)

    _, partial = _replace_background(
        _QueuedBackgroundModel([torch.ones(1, 8, 8)]),
        background,
        {"foreground_0": foreground},
        foreground_scale=0.2,
        long_axis_shift=-1.0,
        workspace_padding=0.5,
    )
    image, hidden = _replace_background(
        _QueuedBackgroundModel([torch.ones(1, 8, 8)]),
        background,
        {"foreground_0": foreground},
        foreground_scale=0.2,
        long_axis_shift=-1.0,
        workspace_padding=1.0,
    )

    assert partial.sum() == 8 * 20
    assert hidden.sum() == 0
    assert torch.equal(image, background)


def test_unified_background_validates_background_and_empty_masks():
    foreground = torch.ones(1, 8, 8, 3)
    model = _QueuedBackgroundModel([torch.zeros(1, 4, 4)])
    with pytest.raises(ValueError, match="exactly one background"):
        _replace_background(
            model, torch.zeros(2, 16, 16, 3), {"foreground_0": foreground}
        )
    with pytest.raises(ValueError, match="empty foreground mask for image 1"):
        _replace_background(
            model, torch.zeros(1, 16, 16, 3), {"foreground_0": foreground}
        )


def _layered_composite(
    model, background, foregrounds, placement_data=None, **overrides
):
    options = {
        "placement_data": placement_data or '{"version":1,"layers":{}}',
        "mask_threshold": 0.5,
        "border_cleanup_width": 0,
        "artifact_cleanup_radius": 0,
        "gap_fill_radius": 0,
        "feather_radius": 0,
    }
    options.update(overrides)
    return composite_nodes.UC_LayeredBackgroundComposite.execute(
        model, background, foregrounds, **options
    )


def test_layered_background_composites_in_socket_order(monkeypatch):
    monkeypatch.setattr(
        composite_nodes,
        "_save_editor_preview",
        lambda image, prefix: {"filename": prefix},
    )
    background = torch.zeros(1, 20, 20, 3)
    red = torch.zeros(1, 4, 4, 3)
    red[..., 0] = 1
    green = torch.zeros(1, 4, 4, 3)
    green[..., 1] = 1
    model = _QueuedBackgroundModel([torch.ones(1, 4, 4), torch.ones(1, 4, 4)])

    output = _layered_composite(
        model,
        background,
        {"foreground_1": green, "foreground_0": red},
        '{"version":1,"layers":{"foreground_0":{"scale":0.5},"foreground_1":{"scale":0.25}}}',
    )
    image, mask = output.result

    assert image.shape == (1, 20, 20, 3)
    assert mask.shape == (1, 20, 20)
    assert torch.allclose(image[0, 8:13, 8:13, 1], torch.ones(5, 5))
    assert image[0, 5:15, 5:15, 0].sum().item() == pytest.approx(75)
    assert mask.sum() == 100
    metadata = output.ui["uc_layered_scene_editor"][0]
    assert [
        (layer["socket"], layer["crop_width"], layer["crop_height"])
        for layer in metadata["layers"]
    ] == [
        ("foreground_0", 4, 4),
        ("foreground_1", 4, 4),
    ]


def test_layered_background_uses_independent_landscape_positions(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 20, 40, 3)
    left = torch.ones(1, 4, 4, 3)
    right = torch.ones(1, 4, 4, 3) * 0.5
    model = _QueuedBackgroundModel([torch.ones(1, 4, 4), torch.ones(1, 4, 4)])
    placement = (
        '{"version":1,"layers":{'
        '"foreground_0":{"scale":0.5,"long_axis_shift":-1,"short_axis_shift":0},'
        '"foreground_1":{"scale":0.5,"long_axis_shift":1,"short_axis_shift":0}}}'
    )

    image, mask = _layered_composite(
        model, background, {"foreground_0": left, "foreground_1": right}, placement
    ).result

    assert torch.allclose(image[0, 5:15, 0:10], torch.ones(10, 10, 3))
    assert torch.allclose(image[0, 5:15, 30:40], torch.full((10, 10, 3), 0.5))
    assert mask.sum() == 200


def test_layered_version_two_centers_allow_partial_off_canvas_placement(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 100, 100, 3)
    foreground = torch.ones(1, 8, 8, 3)
    placement = (
        '{"version":2,"workspace_padding":1,"layers":{'
        '"foreground_0":{"scale":0.2,"center_x":0,"center_y":0.5}}}'
    )

    image, mask = _layered_composite(
        _QueuedBackgroundModel([torch.ones(1, 8, 8)]),
        background,
        {"foreground_0": foreground},
        placement,
    ).result

    assert mask.sum() == 10 * 20
    assert image[:, :, :10].sum() == 10 * 20 * 3
    assert mask[:, :, 10:].sum() == 0


def test_layered_background_uses_explicit_layer_order(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 20, 20, 3)
    red = torch.zeros(1, 4, 4, 3)
    red[..., 0] = 1
    green = torch.zeros(1, 4, 4, 3)
    green[..., 1] = 1
    model = _QueuedBackgroundModel([torch.ones(1, 4, 4), torch.ones(1, 4, 4)])
    placement = (
        '{"version":1,"layer_order":["foreground_1","foreground_0"],"layers":{'
        '"foreground_0":{"scale":0.5},"foreground_1":{"scale":0.5}}}'
    )

    image, _ = _layered_composite(
        model, background, {"foreground_0": red, "foreground_1": green}, placement
    ).result

    assert torch.allclose(image[0, 5:15, 5:15, 0], torch.ones(10, 10))
    assert image[0, 5:15, 5:15, 1].sum().item() == pytest.approx(0, abs=1e-6)


def test_layered_background_rejects_batches_and_invalid_placements(monkeypatch):
    monkeypatch.setattr(composite_nodes, "_save_editor_preview", lambda *args: None)
    background = torch.zeros(1, 16, 16, 3)
    foreground = torch.ones(2, 4, 4, 3)
    with pytest.raises(ValueError, match="not a batch"):
        _layered_composite(
            _QueuedBackgroundModel([]), background, {"foreground_0": foreground}
        )
    with pytest.raises(ValueError, match="scale must be between"):
        _layered_composite(
            _QueuedBackgroundModel([]),
            background,
            {"foreground_0": foreground[:1]},
            '{"version":1,"layers":{"foreground_0":{"scale":20}}}',
        )


def test_layered_background_preview_failure_does_not_change_result(monkeypatch):
    def fail_preview(*args):
        raise OSError("preview unavailable")

    monkeypatch.setattr(composite_nodes, "_save_editor_preview", fail_preview)
    background = torch.zeros(1, 12, 12, 3)
    foreground = torch.ones(1, 4, 2, 3)
    model = _QueuedBackgroundModel([torch.ones(1, 4, 2)])

    output = _layered_composite(model, background, {"foreground_0": foreground})

    assert output.result[0].max().item() == pytest.approx(1)
    metadata = output.ui["uc_layered_scene_editor"][0]
    assert "preview" not in metadata["background"]
    assert "preview" not in metadata["layers"][0]
    assert metadata["layers"][0]["crop_width"] == 2
    assert metadata["layers"][0]["crop_height"] == 4


def test_staged_layered_composite_reuses_prepared_cutouts(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers,
        "_save_editor_preview",
        lambda image, prefix: {"filename": prefix},
    )
    foreground = torch.zeros(1, 6, 8, 3)
    foreground[..., 0] = 1
    mask = torch.zeros(1, 6, 8)
    mask[:, 1:5, 2:6] = 1
    model = _QueuedBackgroundModel([mask])

    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        0,
        0,
        0,
    )
    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 20, 20, 3),
        staged,
        '{"version":1,"layers":{"foreground_0":{"scale":0.5}}}',
        0,
    )
    image, placed_mask = output.result[:2]

    assert len(model.masks) == 0
    assert staged["layers"][0]["image"].shape[1:3] == (4, 4)
    assert image[..., 0].sum().item() == pytest.approx(100)
    assert placed_mask.sum().item() == pytest.approx(100)
    assert output.ui["uc_layered_scene_editor"][0]["layers"][0]["crop_width"] == 4


@pytest.mark.parametrize(
    ("model_resolution", "configured_resolution"),
    [(12, 0), (16, 12)],
)
def test_staged_foreground_masks_use_model_resolution_and_preserve_source_crop(
    monkeypatch, model_resolution, configured_resolution
):
    foreground = torch.zeros(1, 300, 200, 3)
    foreground[:, 60:240, 40:160, 0] = 1
    model = _QueuedBackgroundModel([torch.ones(1, 12, 8)])
    model.image_size = model_resolution
    received_shapes = []
    nonzero_shapes = []
    original_encode = model.encode_image
    original_nonzero = torch.nonzero

    def encode_image(image):
        received_shapes.append(tuple(image.shape))
        return original_encode(image)

    model.encode_image = encode_image
    monkeypatch.setattr(
        staged_compositor_helpers.torch,
        "nonzero",
        lambda value, *args, **kwargs: nonzero_shapes.append(tuple(value.shape))
        or original_nonzero(value, *args, **kwargs),
    )
    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        0,
        0,
        0,
        mask_processing_resolution=configured_resolution,
    )

    assert received_shapes == [(1, 12, 8, 3)]
    layer = staged["layers"][0]
    assert layer["image"].shape == foreground.shape
    assert layer["mask"].shape == (1, 300, 200)
    assert torch.equal(layer["image"], foreground)
    assert nonzero_shapes
    assert max(torch.tensor(shape).prod().item() for shape in nonzero_shapes) <= 12


def test_staged_editor_preview_is_bounded_without_changing_layer_geometry(monkeypatch):
    saved_shapes = []
    monkeypatch.setattr(
        staged_compositor_helpers,
        "_save_editor_preview",
        lambda image, prefix: saved_shapes.append(tuple(image.shape)) or {"filename": prefix},
    )
    staged = {
        "version": 1,
        "mask_processing_resolution": 12,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 30, 20, 3),
                "mask": torch.ones(1, 30, 20),
                "uses_embedded_alpha": False,
            }
        ],
    }

    output = staged_compositor_helpers._preview_staged_foregrounds(
        torch.zeros(1, 40, 40, 3), staged, 0
    )

    assert saved_shapes == [(1, 12, 8, 4)]
    metadata = output.ui["uc_layered_scene_editor"][0]["layers"][0]
    assert (metadata["crop_height"], metadata["crop_width"]) == (30, 20)


def test_staged_compositor_uses_embedded_alpha_without_background_removal(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    foreground = torch.zeros(1, 4, 5, 4)
    foreground[..., 0] = 0.75
    foreground[:, 1:3, 2:4, 3] = 0.5
    model = _QueuedBackgroundModel([])

    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        2,
        1,
        1,
    )

    layer = staged["layers"][0]
    assert len(model.masks) == 0
    assert layer["uses_embedded_alpha"] is True
    assert layer["image"].shape == (1, 2, 2, 3)
    assert torch.allclose(layer["mask"], torch.full((1, 2, 2), 0.5))


def test_staged_compositor_uses_threshold_only_for_embedded_alpha_bounds(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    foreground = torch.zeros(1, 8, 10, 4)
    foreground[..., 0] = 0.75
    foreground[..., 3] = 1.0 / 255.0
    foreground[:, 2:6, 3:8, 3] = 0.75
    model = _QueuedBackgroundModel([])

    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        0,
        0,
        0,
    )

    layer = staged["layers"][0]
    assert len(model.masks) == 0
    assert layer["image"].shape == (1, 4, 5, 3)
    assert torch.allclose(layer["mask"], torch.full((1, 4, 5), 0.75))
    preview = staged_compositor_helpers._preview_staged_foregrounds(
        torch.zeros(1, 16, 16, 3), staged, 0
    )
    metadata = preview.ui["uc_layered_scene_editor"][0]["layers"][0]
    assert (metadata["crop_width"], metadata["crop_height"]) == (5, 4)


def test_staged_compositor_uses_full_frame_when_solid_rgb_removal_is_empty():
    foreground = torch.zeros(1, 4, 5, 3)
    model = _QueuedBackgroundModel([torch.zeros(1, 4, 5)])

    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        2,
        1,
        1,
    )

    layer = staged["layers"][0]
    assert len(model.colors) == 1
    assert model.masks == []
    assert layer["uses_embedded_alpha"] is False
    assert layer["image"].shape == (1, 4, 5, 3)
    assert torch.equal(layer["mask"], torch.ones(1, 4, 5))


def test_staged_layered_composite_tracks_and_applies_horizontal_flip(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    foreground = torch.zeros(1, 2, 3, 3)
    foreground[:, :, 0, 0] = 1.0
    model = _QueuedBackgroundModel([torch.ones(1, 2, 3)])
    flipped_placement = (
        '{"version":2,"layers":{"foreground_0":{"flip_horizontal":true}}}'
    )

    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        0,
        0,
        0,
        placement_data=flipped_placement,
    )

    assert staged["layers"][0]["flip_horizontal"] is True
    assert staged["layers"][0]["image"][0, 0, 2, 0] == 1.0
    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        '{"version":2,"layers":{"foreground_0":{"flip_horizontal":false}}}',
        0,
    )
    assert (
        output.ui["uc_layered_scene_editor"][0]["layers"][0]["flip_horizontal"] is True
    )


def test_staged_layered_composite_tracks_and_applies_vertical_flip(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    foreground = torch.zeros(1, 3, 2, 3)
    foreground[:, 0, :, 0] = 1.0
    model = _QueuedBackgroundModel([torch.ones(1, 3, 2)])
    flipped_placement = '{"version":2,"layers":{"foreground_0":{"flip_vertical":true}}}'

    staged = staged_compositor_helpers._stage_layered_foregrounds(
        model,
        {"foreground_0": foreground},
        0.5,
        0,
        0,
        0,
        placement_data=flipped_placement,
    )

    assert staged["layers"][0]["flip_vertical"] is True
    assert staged["layers"][0]["image"][0, 2, 0, 0] == 1.0
    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        '{"version":2,"layers":{"foreground_0":{"flip_vertical":false}}}',
        0,
    )
    assert output.ui["uc_layered_scene_editor"][0]["layers"][0]["flip_vertical"] is True


def test_staged_layered_composite_rejects_missing_stage():
    with pytest.raises(ValueError, match="missing or incompatible"):
        staged_compositor_helpers._composite_staged_foregrounds(
            torch.zeros(1, 20, 20, 3), None, '{"version":1,"layers":{}}', 0
        )


def test_paint_resize_uses_premultiplied_alpha_without_dark_fringe():
    rgba = torch.tensor([[[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]])
    resized = staged_compositor_helpers.resize_paint_rgba(rgba, 3, 1)

    assert resized.shape == (1, 1, 3, 4)
    assert 0 < resized[0, 0, 1, 3] < 1
    assert torch.allclose(resized[0, 0, 1, :3], torch.tensor([1.0, 0.0, 0.0]))


def test_paint_loader_requires_a_single_rgba_png(tmp_path, monkeypatch):
    path = tmp_path / "paint.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
    monkeypatch.setattr(
        staged_compositor_helpers.folder_paths,
        "exists_annotated_filepath",
        lambda annotated: annotated == "clipspace/paint.png [input]",
    )
    monkeypatch.setattr(
        staged_compositor_helpers.folder_paths,
        "get_annotated_filepath",
        lambda annotated: str(path),
    )
    paint = {
        "asset": {"filename": "paint.png", "subfolder": "clipspace", "type": "input"}
    }

    with pytest.raises(ValueError, match="preserve an RGBA channel"):
        staged_compositor_helpers.load_staged_paint_rgba(
            paint, 2, 2, torch.device("cpu"), torch.float32
        )


@pytest.mark.parametrize(
    ("order", "expected_pixel"),
    [
        (["__uc_paint__", "foreground_0"], [1.0, 0.0, 0.0]),
        (["foreground_0", "__uc_paint__"], [0.0, 1.0, 0.0]),
    ],
)
def test_paint_composites_at_its_exact_layer_position(
    monkeypatch, order, expected_pixel
):
    monkeypatch.setattr(staged_compositor_helpers, "_PAINT_LAYER_ENABLED", True)
    rgba = torch.zeros(1, 4, 4, 4)
    rgba[0, 1, 1] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    monkeypatch.setattr(
        staged_compositor_helpers,
        "load_staged_paint_rgba",
        lambda *args, **kwargs: rgba,
    )

    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 4, 4, 3),
        _paint_test_stage(),
        _paint_placement(order),
        0,
    )
    image, combined, boxes, masks = output.result

    assert torch.allclose(image[0, 1, 1], torch.tensor(expected_pixel))
    assert combined[0, 1, 1] == 1
    assert masks.shape == (2, 4, 4)
    paint_index = order.index("__uc_paint__")
    assert masks[paint_index, 1, 1] == 1
    assert boxes[0][paint_index] == {"x": 1, "y": 1, "width": 1, "height": 1}


def test_excluded_paint_keeps_ordered_empty_outputs_without_affecting_rgb(monkeypatch):
    monkeypatch.setattr(staged_compositor_helpers, "_PAINT_LAYER_ENABLED", True)
    rgba = torch.ones(1, 4, 4, 4)
    monkeypatch.setattr(
        staged_compositor_helpers,
        "load_staged_paint_rgba",
        lambda *args, **kwargs: rgba,
    )
    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 4, 4, 3),
        _paint_test_stage(),
        _paint_placement(["foreground_0", "__uc_paint__"], included=False),
        0,
    )
    image, _, boxes, masks = output.result

    assert torch.allclose(image, torch.tensor([1.0, 0.0, 0.0]).expand_as(image))
    assert masks[1].sum() == 0
    assert boxes[0][1] == {"x": 0, "y": 0, "width": 0, "height": 0}


def test_transparent_paint_is_an_rgb_and_mask_noop(monkeypatch):
    monkeypatch.setattr(staged_compositor_helpers, "_PAINT_LAYER_ENABLED", True)
    monkeypatch.setattr(
        staged_compositor_helpers,
        "load_staged_paint_rgba",
        lambda *args, **kwargs: torch.zeros(1, 4, 4, 4),
    )
    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 4, 4, 3),
        _paint_test_stage(),
        _paint_placement(["foreground_0", "__uc_paint__"]),
        0,
    )
    image, combined, boxes, masks = output.result

    assert torch.allclose(image, torch.tensor([1.0, 0.0, 0.0]).expand_as(image))
    assert torch.all(combined == 1)
    assert masks[1].sum() == 0
    assert boxes[0][1] == {"x": 0, "y": 0, "width": 0, "height": 0}


def test_disabled_paint_layer_is_not_assigned_or_composited(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers,
        "load_staged_paint_rgba",
        lambda *args, **kwargs: pytest.fail("disabled paint must not load"),
    )
    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 4, 4, 3),
        _paint_test_stage(),
        _paint_placement(["__uc_paint__", "foreground_0"]),
        0,
    )
    image, combined, boxes, masks = output.result

    assert torch.allclose(image, torch.tensor([1.0, 0.0, 0.0]).expand_as(image))
    assert torch.all(combined == 1)
    assert masks.shape == (1, 4, 4)
    assert boxes[0] == [{"x": 0, "y": 0, "width": 4, "height": 4}]


def test_staged_compositor_publishes_transparent_cutout_previews(monkeypatch):
    saved = []

    def capture_preview(image, prefix):
        saved.append(image.clone())
        return {"filename": f"{prefix}.png"}

    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", capture_preview
    )
    node = composite_nodes.UC_StagedLayeredBackgroundComposite
    node._staged_by_node.clear()
    monkeypatch.setattr(
        node, "hidden", types.SimpleNamespace(unique_id="preview-compositor")
    )
    foreground = torch.ones(1, 6, 8, 3)
    mask = torch.zeros(1, 6, 8)
    mask[:, 1:5, 3] = 1
    mask[:, 3, 2:6] = 1
    background = torch.zeros(1, 12, 12, 3)
    output = node.execute(
        background=background,
        foreground_images={"foreground_0": foreground},
        execution_mode="run_staging",
        placement_data='{"version":1,"layers":{}}',
        background_removal_model=_QueuedBackgroundModel([mask]),
        background_options={
            "mask_threshold": 0.5,
            "border_cleanup_width": 0,
            "artifact_cleanup_radius": 0,
            "gap_fill_radius": 0,
            "feather_radius": 0,
        },
    )

    assert torch.equal(output.result[0], background)
    assert output.result[1].sum().item() == 0
    assert output.ui["uc_layered_scene_editor"][0]["stage_mode"] == "fresh"
    assert len(saved) == 1
    assert saved[0].shape == (1, 4, 4, 4)
    assert saved[0][..., 3].min().item() == 0
    assert saved[0][..., 3].max().item() == 1


def test_staged_compositor_uses_connected_background_options(monkeypatch):
    captured = {}
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 2, 2, 3),
                "mask": torch.ones(1, 2, 2),
                "uses_embedded_alpha": False,
            }
        ],
    }

    def capture_stage(
        model,
        foregrounds,
        threshold,
        border,
        artifact,
        gap,
        mask_resize,
        placement,
        mask_processing_resolution=0,
    ):
        captured["stage"] = (
            threshold,
            border,
            artifact,
            gap,
            mask_resize,
            placement,
            mask_processing_resolution,
        )
        return staged

    sentinel = object()

    def capture_preview(
        background, preview_stage, feather, placement, image_resize, mask_resize
    ):
        captured["preview"] = (
            preview_stage,
            feather,
            placement,
            image_resize,
            mask_resize,
        )
        return sentinel

    monkeypatch.setattr(
        composite_nodes,
        "_stage_layered_foregrounds",
        capture_stage,
    )
    monkeypatch.setattr(
        composite_nodes,
        "_preview_staged_foregrounds",
        capture_preview,
    )
    node = composite_nodes.UC_StagedLayeredBackgroundComposite
    node._staged_by_node.clear()
    monkeypatch.setattr(
        node,
        "hidden",
        types.SimpleNamespace(unique_id="options-compositor"),
    )
    options = {
        "mask_threshold": 0.75,
        "border_cleanup_width": 3,
        "artifact_cleanup_radius": 4,
        "gap_fill_radius": 5,
        "mask_processing_resolution": 1536,
        "feather_radius": 6,
        "mask_resize_method": "bilinear",
        "foreground_blend": 0.0,
    }

    result = node.execute(
        background=torch.zeros(1, 4, 4, 3),
        foreground_images={"foreground_0": torch.ones(1, 2, 2, 3)},
        execution_mode="run_staging",
        placement_data='{"version":2,"layers":{}}',
        background_removal_model=object(),
        background_options=options,
    )

    assert result is sentinel
    assert captured["stage"][:5] == (0.75, 3, 4, 5, "bilinear")
    assert captured["stage"][-1] == 1536
    preview_stage, feather, placement, image_resize, mask_resize = captured["preview"]
    assert feather == 6
    assert placement == '{"version":2,"layers":{}}'
    assert image_resize == "auto"
    assert mask_resize == "bilinear"
    assert preview_stage["layers"][0]["blend_factor"] == 0.5


def test_staged_compositor_automatically_reuses_and_refreshes_its_stage(monkeypatch):
    node = composite_nodes.UC_StagedLayeredBackgroundComposite
    schema = node.define_schema()
    model_input = next(
        value for value in schema.inputs if value.id == "background_removal_model"
    )
    options_input = next(
        value for value in schema.inputs if value.id == "background_options"
    )
    foreground_input = next(
        value for value in schema.inputs if value.id == "foreground_images"
    )
    execution_input = next(
        value for value in schema.inputs if value.id == "execution_mode"
    )
    selector_input = next(
        value for value in schema.inputs if value.id == "background_removal_model_name"
    )
    assert schema.is_output_node is True
    assert schema.display_name == "Staged Background Composite"
    assert "Automatically rebuilds retained cutouts" in schema.description
    assert execution_input.options == ["run_staging", "run_staged", "full_run"]
    assert execution_input.default == "run_staged"
    assert execution_input.advanced is True
    assert model_input.lazy is True
    assert model_input.optional is True
    assert model_input.display_name == "background_removal_model_opt"
    assert options_input.optional is True
    assert options_input.display_name == "Background Options"
    assert selector_input.options == ["birefnet", "lucida"]
    assert selector_input.default == "birefnet"
    assert foreground_input.template.input.lazy is True
    node._staged_by_node.clear()
    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="compositor-a"))
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    background = torch.zeros(1, 20, 20, 3)

    model = _QueuedBackgroundModel([torch.ones(1, 4, 4)])
    foregrounds = {"foreground_0": torch.ones(1, 4, 4, 3)}
    staging_options = {
        "border_cleanup_width": 0,
        "artifact_cleanup_radius": 0,
        "gap_fill_radius": 0,
        "feather_radius": 0,
    }
    assert node.check_lazy_status(
        "run_staging", foreground_images={"foreground_0": (None, "foreground_0")}
    ) == ["foreground_0"]
    assert node.check_lazy_status(
        "run_staging", None, {"foreground_0": (None, "foreground_0")}
    ) == ["background_removal_model", "foreground_0"]
    assert (
        node.check_lazy_status(
            "full_run",
            model,
            {"foreground_0": (foregrounds["foreground_0"], "foreground_0")},
        )
        == []
    )
    assert (
        node.check_lazy_status(
            "run_staged", None, {"foreground_0": (None, "foreground_0")}
        )
        == ["background_removal_model", "foreground_0"]
    )
    model = _QueuedBackgroundModel(
        [torch.ones(1, 4, 4), torch.ones(1, 4, 4)]
    )
    fresh = node.execute(
        background,
        foregrounds,
        "run_staging",
        '{"version":1,"layers":{}}',
        background_removal_model=model,
        background_options=staging_options,
    )
    retained = node.execute(
        background,
        foregrounds,
        "run_staged",
        '{"version":1,"layers":{"foreground_0":{"long_axis_shift":0.2}}}',
        background_removal_model=model,
        background_options=staging_options,
    )

    assert fresh.ui["uc_layered_scene_editor"][0]["stage_mode"] == "fresh"
    assert retained.ui["uc_layered_scene_editor"][0]["stage_mode"] == "retained"
    assert fresh.result[0].sum().item() == 0
    assert retained.result[0].sum().item() > 0
    assert len(model.masks) == 1

    refreshed = node.execute(
        background,
        {"foreground_0": foregrounds["foreground_0"] * 0.5},
        "run_staged",
        '{"version":1,"layers":{"foreground_0":{"long_axis_shift":0.2}}}',
        background_removal_model=model,
        background_options=staging_options,
    )
    assert refreshed.ui["uc_layered_scene_editor"][0]["stage_mode"] == "full_run"
    assert len(model.masks) == 0

    updated_model = _QueuedBackgroundModel([torch.ones(1, 4, 4)])
    full = node.execute(
        background,
        {"foreground_0": foregrounds["foreground_0"] * 0.5},
        "full_run",
        '{"version":1,"layers":{}}',
        background_removal_model=updated_model,
        background_options=staging_options,
    )
    assert full.ui["uc_layered_scene_editor"][0]["stage_mode"] == "full_run"
    assert full.result[0].sum().item() > 0
    assert len(updated_model.masks) == 0

    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="compositor-b"))
    automatic_model = _QueuedBackgroundModel([torch.ones(1, 4, 4)])
    automatic = node.execute(
        background,
        foregrounds,
        "run_staged",
        '{"version":1,"layers":{}}',
        background_removal_model=automatic_model,
        background_options=staging_options,
    )
    assert automatic.ui["uc_layered_scene_editor"][0]["stage_mode"] == "full_run"
    assert len(automatic_model.masks) == 0


def test_staged_compositor_rebuilds_serialized_layout_after_cache_loss(monkeypatch):
    node = composite_nodes.UC_StagedLayeredBackgroundComposite
    node._staged_by_node.clear()
    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="restart"))
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    background = torch.zeros(1, 20, 20, 3)
    foreground = torch.ones(1, 4, 8, 3)
    placement = json.dumps(
        {
            "version": 3,
            "layer_order": ["foreground_0"],
            "layers": {
                "foreground_0": {
                    "scale": 0.35,
                    "center_x": 0.7,
                    "center_y": 0.3,
                    "rotation": 90,
                    "corners": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                }
            },
        }
    )
    options = {
        "border_cleanup_width": 0,
        "artifact_cleanup_radius": 0,
        "gap_fill_radius": 0,
        "feather_radius": 0,
    }

    first = node.execute(
        background,
        {"foreground_0": foreground},
        "full_run",
        placement,
        background_removal_model=_QueuedBackgroundModel([torch.ones(1, 4, 8)]),
        background_options=options,
    )
    node._staged_by_node.clear()
    rebuilt = node.execute(
        background,
        {"foreground_0": foreground},
        "full_run",
        placement,
        background_removal_model=_QueuedBackgroundModel([torch.ones(1, 4, 8)]),
        background_options=options,
    )

    assert torch.equal(first.result[0], rebuilt.result[0])
    assert torch.equal(first.result[1], rebuilt.result[1])
    assert first.result[2] == rebuilt.result[2]
    assert torch.equal(first.result[3], rebuilt.result[3])


def test_internal_background_removal_loader_selects_exact_files_without_retention(
    monkeypatch, tmp_path
):
    from comfy import bg_removal_model

    loaded = []

    class DummyModel:
        def __init__(self):
            self.image_size = 256
            self.image_mean = [0.0, 0.0, 0.0]
            self.image_std = [1.0, 1.0, 1.0]
            self.config = {}

        def encode_image(self, image):
            return image

    def load(path):
        loaded.append(path)
        return DummyModel()

    monkeypatch.setattr(
        composite_helpers,
        "require_huggingface_model",
        lambda category, filename, repo_id, repo_path: str(tmp_path / filename),
    )
    monkeypatch.setattr(bg_removal_model, "load", load)
    birefnet = composite_helpers._load_internal_background_removal_model("birefnet")
    second_birefnet = composite_helpers._load_internal_background_removal_model(
        "birefnet"
    )
    lucida = composite_helpers._load_internal_background_removal_model("lucida")

    assert [pathlib.Path(path).name for path in loaded] == [
        "birefnet.safetensors",
        "birefnet.safetensors",
        "lucida.safetensors",
    ]
    assert second_birefnet is not birefnet
    assert birefnet.image_mean == [0.0, 0.0, 0.0]
    assert lucida.image_size == 1024
    assert lucida.image_mean == [0.485, 0.456, 0.406]
    assert lucida.image_std == [0.229, 0.224, 0.225]
    assert lucida.config["image_mean"] == lucida.image_mean


def test_internal_background_removal_loader_reports_missing_model(monkeypatch):
    monkeypatch.setattr(
        composite_helpers,
        "require_huggingface_model",
        lambda *_args: (_ for _ in ()).throw(
            ValueError(
                "Required model lucida.safetensors was not found. Download it from "
                "https://huggingface.co/Comfy-Org/BiRefNet/blob/main/"
                "background_removal/lucida.safetensors"
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"Comfy-Org/BiRefNet/blob/main/background_removal/lucida\.safetensors",
    ):
        composite_helpers._load_internal_background_removal_model("lucida")


def test_internal_face_loader_constructs_each_model_without_retention(
    monkeypatch, tmp_path
):
    import comfy.utils
    from comfy_extras import nodes_mediapipe

    path = tmp_path / "mediapipe_face_fp32.safetensors"
    loaded = []
    constructed = []

    monkeypatch.setattr(
        staged_face_helpers,
        "require_huggingface_model",
        lambda *_args: str(path),
    )
    monkeypatch.setattr(
        comfy.utils,
        "load_torch_file",
        lambda loaded_path, safe_load=True: loaded.append(loaded_path) or {},
    )
    monkeypatch.setattr(
        nodes_mediapipe,
        "FaceLandmarkerModel",
        lambda state: constructed.append(state) or object(),
    )

    first = staged_face_helpers.load_face_model()
    second = staged_face_helpers.load_face_model()

    assert first is not second
    normalized_path = str(path).lower()
    assert loaded == [normalized_path, normalized_path]
    assert constructed == [{}, {}]


def test_required_huggingface_model_reports_url_and_registered_locations(
    monkeypatch, tmp_path
):
    import folder_paths

    first = tmp_path / "custom-removal-location"
    second = tmp_path / "other-removal-location"

    monkeypatch.setattr(
        folder_paths,
        "get_full_path_or_raise",
        lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr(
        folder_paths,
        "get_folder_paths",
        lambda category: [str(first), str(second)],
    )

    with pytest.raises(ValueError) as raised:
        model_assets.require_huggingface_model(
            "background_removal",
            "birefnet.safetensors",
            "Comfy-Org/BiRefNet",
            "background_removal/birefnet.safetensors",
        )

    message = str(raised.value)
    assert (
        "https://huggingface.co/Comfy-Org/BiRefNet/blob/main/"
        "background_removal/birefnet.safetensors"
    ) in message
    assert str(first / "birefnet.safetensors") in message
    assert str(second / "birefnet.safetensors") in message


def test_required_huggingface_model_returns_registered_file(monkeypatch, tmp_path):
    import folder_paths

    existing = tmp_path / "detection" / "mediapipe_face_fp32.safetensors"
    existing.parent.mkdir()
    existing.write_bytes(b"present")
    monkeypatch.setattr(
        folder_paths, "get_full_path_or_raise", lambda *_args: str(existing)
    )

    result = model_assets.require_huggingface_model(
        "detection",
        existing.name,
        "Comfy-Org/mediapipe",
        f"detection/{existing.name}",
    )

    assert result == str(existing)


def test_staged_compositor_preserves_exact_size_non_overlapping_foregrounds(
    monkeypatch,
):
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    first = torch.rand(1, 8, 8, 3)
    second = torch.rand(1, 8, 8, 3)
    staged = {
        "version": 1,
        "layers": [
            {"socket": "foreground_0", "image": first, "mask": torch.ones(1, 8, 8)},
            {"socket": "foreground_1", "image": second, "mask": torch.ones(1, 8, 8)},
        ],
    }
    placement = (
        '{"version":2,"workspace_padding":0,"layers":{'
        '"foreground_0":{"scale":0.5,"center_x":0.125,"center_y":0.5},'
        '"foreground_1":{"scale":0.5,"center_x":0.875,"center_y":0.5}}}'
    )

    image, _ = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 16, 32, 3), staged, placement, 0
    ).result[:2]

    assert torch.equal(image[:, 4:12, 0:8], first)
    assert torch.equal(image[:, 4:12, 24:32], second)
    assert torch.count_nonzero(image[:, :, 8:24]) == 0


def test_tensor_blending_preserves_fp32_and_noop_pixels():
    destination = torch.rand(1, 12, 14, 3)
    source = torch.rand(1, 12, 14, 3)
    zero_mask = torch.zeros(1, 12, 14)

    masked = image_nodes.UC_ImageBlendByMask.execute(
        destination, source, "overlay", 1.0, False, zero_mask
    ).result[0]
    blended = image_nodes.UC_ImageBlendByMask.execute(
        destination, source, "multiply", 0.37, False, None
    ).result[0]

    assert torch.equal(masked, destination)
    assert ((blended * 255.0) - (blended * 255.0).round()).abs().max() > 1e-4


def test_property_match_zero_weight_is_bit_exact():
    original = torch.rand(1, 10, 12, 3)
    generated = torch.rand(1, 10, 12, 3)

    result = image_nodes.UC_ImageMatchPropertiesNode.execute(
        original, generated, 0.0, 1.0, 1.0, 0.5
    ).result[0]

    assert torch.equal(result, generated)


def test_property_match_uses_global_statistics_without_spatial_correspondence():
    source = torch.rand(1, 9, 7, 3)
    target = torch.rand(1, 6, 8, 3)
    permutation = torch.randperm(target.shape[1] * target.shape[2])
    shuffled = target.reshape(1, -1, 3)[:, permutation].reshape_as(target)

    matched = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 1.0, 1.0, 0.0
    ).result[0]
    shuffled_matched = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, shuffled, 1.0, 1.0, 1.0, 0.0
    ).result[0]
    inverse = torch.argsort(permutation)
    restored = shuffled_matched.reshape(1, -1, 3)[:, inverse].reshape_as(target)

    assert matched.shape == target.shape
    assert torch.allclose(matched, restored, atol=2e-4, rtol=0.0)


def test_property_match_transfers_global_lighting_between_different_sizes():
    source = torch.full((1, 5, 9, 3), 0.8)
    target = torch.full((1, 11, 6, 3), 0.2)

    result = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 0.0, 1.0, 1.0,
        saturation_weight=0.0, contrast_weight=0.0,
    ).result[0]

    assert result.shape == target.shape
    assert result.mean() > 0.7


def test_property_match_allows_lighting_strength_above_measured_match():
    source = torch.full((1, 5, 9, 3), 0.7)
    target = torch.full((1, 11, 6, 3), 0.3)

    measured = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 0.0, 1.0, 1.0,
        saturation_weight=0.0, contrast_weight=0.0,
    ).result[0]
    stronger = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 0.0, 2.0, 1.0,
        saturation_weight=0.0, contrast_weight=0.0,
    ).result[0]

    assert stronger.mean() > measured.mean()


def test_property_match_constant_images_remain_finite():
    source = torch.full((1, 4, 4, 3), 0.65)
    target = torch.full((1, 7, 5, 3), 0.35)

    result = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 1.0, 1.0, 0.5,
    ).result[0]

    assert torch.isfinite(result).all()


def test_property_match_ignores_removed_analysis_mask_inputs():
    source = torch.rand(1, 6, 8, 3)
    target = torch.rand(1, 9, 11, 3)
    stale_mask = torch.ones(1, 4, 4)

    result = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 1.0, 1.0, 0.5,
        source_analysis_mask=stale_mask,
        target_analysis_mask=stale_mask,
    ).result[0]

    assert result.shape == target.shape


def test_property_match_finds_scaled_overlap_from_sharpened_detail():
    generator = torch.Generator().manual_seed(41)
    source = torch.rand((1, 96, 128, 3), generator=generator)
    scaled = torch.nn.functional.interpolate(
        source.permute(0, 3, 1, 2),
        size=(125, 166),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    target = torch.rand((1, 220, 280, 3), generator=generator) * 0.6 + 0.15
    target[:, 47:172, 63:229] = scaled * 0.6 + 0.15

    affine = image_helpers._fit_source_to_target(
        source[0].numpy(),
        target[0].numpy(),
    )

    assert affine is not None
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    assert scale == pytest.approx(1.3, abs=0.08)
    assert affine[0, 2] == pytest.approx(63.0, abs=5.0)
    assert affine[1, 2] == pytest.approx(47.0, abs=5.0)

    result = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 1.0, 1.0, 0.5
    ).result[0]
    before = torch.mean(torch.abs(target[:, 47:172, 63:229] - scaled))
    after = torch.mean(torch.abs(result[:, 47:172, 63:229] - scaled))
    assert after < before * 0.9


def test_property_match_uses_generated_side_of_overlap_edge_for_brightness():
    generator = torch.Generator().manual_seed(53)
    source = torch.rand((1, 120, 150, 3), generator=generator) * 0.55
    target = torch.full((1, 220, 280, 3), 0.9)
    target[:, 52:172, 68:218] = source
    outpaint_mask = torch.ones((1, 220, 280))
    outpaint_mask[:, 52:172, 68:218] = 0.0

    result = image_nodes.UC_ImageMatchPropertiesNode.execute(
        source, target, 1.0, 0.0, 1.0, 1.0,
        mask=outpaint_mask,
        saturation_weight=0.0, contrast_weight=0.0,
    ).result[0]

    assert result[:, :40].mean() < 0.6
    assert torch.equal(result[:, 52:172, 68:218], source)
    transition = result[:, 51, 80:200].mean()
    assert result[:, :40].mean() < transition < 0.9


def test_opencv_edits_preserve_unaffected_fp32_pixels():
    image = torch.rand(1, 20, 20, 3)
    mask = torch.zeros(1, 20, 20)
    mask[:, 8:12, 8:12] = 1.0

    inward = image_nodes.UC_ImageInwardEdgeFill.execute(image, mask, 1, 0).result[0]
    stretched = image_nodes.UC_ImageIterativeStretchFill.execute(
        image, mask, "horizontal", 2, 0, 1, 0
    ).result[0]

    unaffected = mask == 0
    assert torch.equal(inward[unaffected], image[unaffected])
    assert torch.equal(stretched[unaffected], image[unaffected])


def test_text_overlay_preserves_pixels_outside_overlay():
    image = torch.rand(1, 64, 96, 3)

    result = image_nodes.UC_TextOverlayNode.execute(
        image,
        "X",
        12,
        "FFFFFF",
        "000000",
        True,
        2,
        0.5,
        False,
        0,
        -1,
        0,
        -1,
    ).result[0]

    assert torch.equal(result[:, 48:, 48:], image[:, 48:, 48:])


def test_face_foreground_solidification_matches_composite_operation():
    foreground = torch.tensor([0.25, 0.5, 0.625, 0.75, 1.0])
    inverted = 1.0 - foreground
    solid = ((foreground - inverted) * 2.0).clamp(0.0, 1.0)

    assert torch.equal(solid, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_target_warp_keeps_crop_border_fixed():
    y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    target = torch.stack((x, y, x + y), dim=-1).to(torch.float32).div_(30.0)
    source_points = np.array([[5, 5], [11, 5], [11, 11], [5, 11]], dtype=np.float32)
    target_points = source_points + np.array([2, 0], dtype=np.float32)

    warped = background_replace_helpers._warp_target(
        target, source_points, target_points, 1.0, 4
    )

    assert torch.allclose(warped[0], target[0], atol=1e-4)
    assert torch.allclose(warped[-1], target[-1], atol=1e-4)
    assert torch.allclose(warped[:, 0], target[:, 0], atol=1e-4)
    assert torch.allclose(warped[:, -1], target[:, -1], atol=1e-4)
    assert not torch.allclose(warped[5:12, 5:12], target[5:12, 5:12])
    for source_point, target_point in zip(
        source_points.astype(int), target_points.astype(int)
    ):
        expected = target[target_point[1], target_point[0]]
        actual = warped[source_point[1], source_point[0]]
        assert torch.allclose(actual, expected, atol=2e-3)


def test_similarity_transform_allows_rotation_without_source_warp():
    source = np.array([[2, 2], [8, 2], [8, 10], [2, 10]], dtype=np.float32)
    rotation = np.array([[0, -1], [1, 0]], dtype=np.float32)
    target = 1.5 * (source @ rotation.T) + np.array([20, 5], dtype=np.float32)

    scale, solved_rotation, translation = (
        background_replace_helpers._similarity_transform(source, target)
    )
    transformed = scale * (source @ solved_rotation.T) + translation

    assert scale == pytest.approx(1.5)
    assert np.allclose(solved_rotation, rotation, atol=1e-6)
    assert np.allclose(transformed, target, atol=1e-5)


class _FaceModel:
    def __init__(self):
        self.connection_sets = {
            "face_oval": frozenset({(0, 1), (1, 2), (2, 3), (3, 0)})
        }
        self.calls = []

    def detect_batch(self, images, num_faces, score_thresh, variant):
        self.calls.append((images[0].shape, num_faces, score_thresh, variant))
        height, width = images[0].shape[:2]
        if width == 20:
            landmarks = np.array([[4, 8], [8, 4], [12, 8], [8, 12]], dtype=np.float32)
            box = np.array([4, 4, 12, 12], dtype=np.float32)
        else:
            landmarks = np.array(
                [[10, 14], [16, 8], [22, 14], [16, 20]], dtype=np.float32
            )
            box = np.array([10, 8, 22, 20], dtype=np.float32)
        smaller = {"bbox_xyxy": box * 0.5, "landmarks_xy": landmarks * 0.5}
        larger = {"bbox_xyxy": box, "landmarks_xy": landmarks}
        return [[smaller, larger]]


class _BackgroundModel:
    def encode_image(self, image):
        return torch.ones(
            image.shape[0],
            image.shape[1],
            image.shape[2],
            device=image.device,
            dtype=image.dtype,
        )


def test_face_composite_uses_full_detector_and_target_crop_coordinates():
    source = torch.zeros(1, 20, 20, 3)
    source[..., 0] = 1.0
    target = torch.zeros(1, 30, 30, 3)
    face_model = _FaceModel()
    options = {
        "bbox_expansion": 2,
        "mask_expansion": 0,
        "feather_radius": 0,
        "target_warp_strength": 0.0,
        "warp_decay_radius": 4,
    }

    output = composite_nodes.UC_MediaPipeFaceComposite.execute(
        face_model, _BackgroundModel(), source, target, options
    )
    image, crop = output.result

    assert [call[3] for call in face_model.calls] == ["full", "full"]
    assert crop.shape == (1, 16, 16, 3)
    assert image[0, 6:22, 8:24, 0].sum() > 0
    assert image[0, :6].sum() == 0
    assert image[0, :, :8].sum() == 0


def test_face_composite_loads_internal_models_when_disconnected(monkeypatch):
    face_model = _FaceModel()
    background_model = _BackgroundModel()
    monkeypatch.setattr(composite_nodes, "load_face_model", lambda: face_model)
    monkeypatch.setattr(
        composite_nodes,
        "resolve_background_removal_model",
        lambda model: background_model if model is None else model,
    )

    output = composite_nodes.UC_MediaPipeFaceComposite.execute(
        None,
        None,
        torch.zeros(1, 20, 20, 3),
        torch.zeros(1, 30, 30, 3),
        {
            "bbox_expansion": 2,
            "mask_expansion": 0,
            "feather_radius": 0,
            "target_warp_strength": 0.0,
            "warp_decay_radius": 4,
        },
    )

    assert output.result[0].shape == (1, 30, 30, 3)
    assert len(face_model.calls) == 2


def test_face_composite_rejects_batches_and_missing_faces():
    source = torch.zeros(2, 20, 20, 3)
    target = torch.zeros(1, 30, 30, 3)
    with pytest.raises(ValueError, match="requires one source"):
        composite_nodes.UC_MediaPipeFaceComposite.execute(
            _FaceModel(), _BackgroundModel(), source, target
        )

    model = _FaceModel()
    model.detect_batch = lambda *args, **kwargs: [[]]
    with pytest.raises(ValueError, match="No face was detected"):
        composite_nodes.UC_MediaPipeFaceComposite.execute(
            model, _BackgroundModel(), source[:1], target
        )


def test_staged_face_layers_are_stable_ordered_and_intersect_alpha():
    image = torch.ones(1, 20, 20, 4)
    image[..., 3] = 0
    image[:, 4:16, 4:16, 3] = 1
    model = _FaceModel()
    options = composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS
    face_options = composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS | {
        "bbox_expansion": 0,
        "maximum_faces": 2,
    }

    staged = staged_face_helpers._stage_face_foregrounds(
        _BackgroundModel(),
        model,
        {"foreground_0": image},
        options,
        face_options,
    )

    assert [layer["socket"] for layer in staged["layers"]] == [
        "foreground_0",
        "foreground_0_face_0",
        "foreground_0_face_1",
    ]
    assert all(layer["mask"].sum() > 0 for layer in staged["layers"][1:])
    assert staged["layers"][1]["mask"][0, 0, 0] == 0
    assert [call[2] for call in model.calls] == [0.55]


def test_staged_face_detection_failure_is_nonfatal():
    model = _FaceModel()
    model.detect_batch = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("detector")
    )
    staged = staged_face_helpers._stage_face_foregrounds(
        _BackgroundModel(),
        model,
        {"foreground_0": torch.ones(1, 20, 20, 3)},
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS,
        composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS,
    )
    assert [layer["socket"] for layer in staged["layers"]] == ["foreground_0"]
    assert staged["face_warning_count"] == 1


def test_projective_warp_applies_identical_support_to_rgb_and_alpha():
    image = torch.ones(1, 9, 9, 3)
    mask = torch.ones(1, 9, 9)
    warped_image, warped_mask = staged_compositor_helpers.projective_warp(
        image,
        mask,
        [[-0.8, -0.8], [0.8, -1], [0.7, 0.8], [-0.9, 0.7]],
        12,
    )
    assert torch.allclose(warped_image[..., 0], warped_mask, atol=1e-6)


def test_projective_geometry_matches_frontend_parity_fixtures():
    fixtures = json.loads(
        (CUSTOM_NODE_ROOT / "tests/fixtures/staged_transform_geometry.json").read_text()
    )
    for fixture in fixtures:
        background_width, background_height = fixture["background"]
        source_width, source_height = fixture["source"]
        target_longest = max(
            1, round(min(background_width, background_height) * fixture["scale"])
        )
        factor = target_longest / max(source_width, source_height)
        placed_width = max(1, round(source_width * factor))
        placed_height = max(1, round(source_height * factor))
        if staged_compositor_helpers.is_identity_projective_transform(
            fixture["corners"], fixture["rotation"]
        ):
            points = torch.tensor(
                [
                    [0.0, 0.0],
                    [placed_width - 1.0, 0.0],
                    [placed_width - 1.0, placed_height - 1.0],
                    [0.0, placed_height - 1.0],
                ],
                dtype=torch.float64,
            )
            output_width, output_height = placed_width, placed_height
        else:
            points, output_width, output_height = (
                staged_compositor_helpers.projective_geometry(
                    placed_width,
                    placed_height,
                    fixture["corners"],
                    fixture["rotation"],
                    device=torch.device("cpu"),
                    dtype=torch.float64,
                )
            )
        offset = composite_helpers._placement_offsets(
            background_width,
            background_height,
            output_width,
            output_height,
            {
                "_version": 3,
                "center_x": fixture["center"][0],
                "center_y": fixture["center"][1],
            },
            fixture["padding"],
        )
        assert [placed_width, placed_height] == fixture["expected"]["source"]
        assert [output_width, output_height] == fixture["expected"]["output"]
        assert list(offset) == fixture["expected"]["offset"]
        assert list(
            composite_helpers._visible_placement_slices(
                background_width,
                background_height,
                output_width,
                output_height,
                *offset,
            )
        ) == fixture["expected"]["visible"]
        assert torch.allclose(
            points,
            torch.tensor(fixture["expected"]["points"], dtype=torch.float64),
            atol=1e-9,
            rtol=0,
        )


def test_projective_warp_rotates_non_square_layer_without_distortion():
    image = torch.zeros(1, 3, 7, 3)
    mask = torch.zeros(1, 3, 7)
    image[:, 1, :, 0] = 1
    mask[:, 1, :] = 1

    warped_image, warped_mask = staged_compositor_helpers.projective_warp(
        image,
        mask,
        [[-1, -1], [1, -1], [1, 1], [-1, 1]],
        90,
    )

    assert warped_image.shape[1:3] == (7, 3)
    assert warped_mask.shape[1:] == (7, 3)
    assert torch.allclose(warped_image[..., 0], warped_mask, atol=1e-6)
    assert torch.allclose(warped_mask[:, :, 1], torch.ones(1, 7), atol=1e-5)
    assert torch.allclose(
        warped_mask[:, :, (0, 2)],
        torch.zeros(1, 7, 2),
        atol=1e-6,
    )


def test_staged_rotation_uses_expanded_bounds_and_preserves_center():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 3, 7, 3),
                "mask": torch.ones(1, 3, 7),
                "uses_embedded_alpha": True,
                "is_face": False,
            }
        ],
    }
    placement = json.dumps(
        {
            "version": 3,
            "layers": {
                "foreground_0": {
                    "scale": 0.35,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "rotation": 90,
                    "corners": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                },
            },
        }
    )

    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 20, 20, 3),
        staged,
        placement,
        0,
    )
    _, mask, bounding_boxes, layer_masks = output.result
    individual = staged_compositor_helpers._composite_staged_individual_foregrounds(
        torch.zeros(1, 20, 20, 3), staged, placement, 0
    )
    points = torch.nonzero(mask[0] > 0.99)

    assert points[:, 0].min() == 6
    assert points[:, 0].max() == 12
    assert points[:, 1].min() == 8
    assert points[:, 1].max() == 10
    assert bounding_boxes == [[{"x": 8, "y": 6, "width": 3, "height": 7}]]
    assert torch.allclose(layer_masks, mask)
    assert individual.result[0].shape == (1, 20, 20, 3)


def test_staged_layer_geometry_outputs_follow_back_to_front_order():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 2, 2, 3),
                "mask": torch.ones(1, 2, 2),
                "uses_embedded_alpha": True,
            },
            {
                "socket": "foreground_1",
                "image": torch.ones(1, 2, 2, 3),
                "mask": torch.ones(1, 2, 2),
                "uses_embedded_alpha": True,
            },
        ],
    }
    placement = json.dumps(
        {
            "version": 3,
            "workspace_padding": 0,
            "layer_order": ["foreground_1", "foreground_0"],
            "layers": {
                "foreground_0": {"scale": 0.2, "center_x": 0.2, "center_y": 0.5},
                "foreground_1": {"scale": 0.2, "center_x": 0.8, "center_y": 0.5},
            },
        }
    )

    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 20, 3),
        staged,
        placement,
        0,
    )
    _, _, bounding_boxes, layer_masks = output.result
    individual = staged_compositor_helpers._composite_staged_individual_foregrounds(
        torch.zeros(1, 10, 20, 3), staged, placement, 0
    )

    assert bounding_boxes == [
        [
            {"x": 15, "y": 4, "width": 2, "height": 2},
            {"x": 3, "y": 4, "width": 2, "height": 2},
        ]
    ]
    assert layer_masks.shape == (2, 10, 20)
    assert layer_masks[0, 4:6, 15:17].all()
    assert layer_masks[1, 4:6, 3:5].all()
    assert individual.result[0].shape == (2, 10, 20, 3)
    assert individual.result[0][0, 4:6, 15:17].eq(1).all()
    assert torch.count_nonzero(individual.result[0][0, 4:6, 3:5]) == 0
    assert individual.result[0][1, 4:6, 3:5].eq(1).all()
    assert torch.count_nonzero(individual.result[0][1, 4:6, 15:17]) == 0


def test_version_three_warp_applies_to_ordinary_foreground():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 5, 5, 3),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
                "is_face": False,
            }
        ],
    }
    placement = json.dumps(
        {
            "version": 3,
            "layers": {
                "foreground_0": {
                    "scale": 0.5,
                    "center_x": 0.5,
                    "center_y": 0.5,
                    "rotation": 0,
                    "corners": [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]],
                },
            },
        }
    )
    image, mask = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        placement,
        0,
    ).result[:2]
    assert torch.allclose(image[..., 0], mask, atol=1e-6)
    assert mask.sum() < 25


def test_staged_soft_blend_does_not_blend_with_background():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 5, 5, 3),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
                "blend_factor": 0.5,
            }
        ],
    }
    image, mask = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        '{"version":3,"workspace_padding":0,"layers":{"foreground_0":{"scale":0.5}}}',
        0,
    ).result[:2]
    assert torch.allclose(image[..., 0], mask)
    assert mask.max() == 1


def test_staged_soft_blend_maps_zero_to_half_over_underlying_foreground():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.full((1, 5, 5, 3), 0.2),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
            },
            {
                "socket": "foreground_1",
                "image": torch.ones(1, 5, 5, 3),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
                "blend_factor": 0.5,
            },
        ],
    }
    placement = (
        '{"version":3,"workspace_padding":0,"layers":{'
        '"foreground_0":{"scale":0.5},"foreground_1":{"scale":0.5}}}'
    )
    image, mask = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        placement,
        0,
    ).result[:2]
    assert image.max().item() == pytest.approx(0.6)
    assert mask.max() == 1


def test_staged_soft_blend_uses_accumulated_coverage_past_excluded_layer():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.full((1, 5, 5, 3), 0.2),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
            },
            {
                "socket": "foreground_1",
                "image": torch.zeros(1, 5, 5, 3),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
            },
            {
                "socket": "foreground_2",
                "image": torch.ones(1, 5, 5, 3),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
                "blend_factor": 0.5,
            },
        ],
    }
    placement = (
        '{"version":3,"workspace_padding":0,"layers":{'
        '"foreground_0":{"scale":0.5},'
        '"foreground_1":{"scale":0.5,"included":false},'
        '"foreground_2":{"scale":0.5}}}'
    )
    image, mask = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        placement,
        0,
    ).result[:2]
    assert image.max().item() == pytest.approx(0.6)
    assert mask.max() == 1


def test_staged_blend_options_apply_separate_ordinary_and_face_factors():
    staged = {
        "version": 1,
        "layers": [
            {"socket": "foreground_0"},
            {"socket": "foreground_0_face_0", "is_face": True},
        ],
    }
    blended = staged_compositor_helpers._apply_staged_layer_options(
        staged, 0.2, 0.8, 12
    )
    assert [layer["blend_factor"] for layer in blended["layers"]] == pytest.approx(
        [0.6, 0.9]
    )
    assert blended["layers"][1]["feather_radius"] == 12
    assert "blend_factor" not in staged["layers"][0]
    diagnostic = staged_compositor_helpers._apply_staged_layer_options(
        staged, 1.0, -1.0, 4
    )
    assert diagnostic["layers"][1]["blend_factor"] == 0


def test_version_three_exclusion_applies_to_ordinary_foreground():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 5, 5, 3),
                "mask": torch.ones(1, 5, 5),
                "uses_embedded_alpha": True,
                "is_face": False,
            }
        ],
    }
    placement = json.dumps(
        {
            "version": 3,
            "layers": {"foreground_0": {"included": False}},
        }
    )
    image, mask = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 10, 10, 3),
        staged,
        placement,
        0,
    ).result[:2]
    assert torch.count_nonzero(image) == 0
    assert torch.count_nonzero(mask) == 0


def test_face_feather_is_applied_to_final_composite_alpha():
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0_face_0",
                "image": torch.ones(1, 9, 9, 3),
                "mask": torch.ones(1, 9, 9),
                "uses_embedded_alpha": True,
                "is_face": True,
                "feather_radius": 4,
            }
        ],
    }
    image, mask = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 18, 18, 3),
        staged,
        '{"version":3,"workspace_padding":0,"layers":{"foreground_0_face_0":{"scale":0.5}}}',
        0,
    ).result[:2]
    support = mask[0, 5:14, 5:14]
    assert support[4, 4] > support[0, 4]
    assert torch.allclose(image[..., 0], mask)


def test_final_face_feather_scales_with_placed_face(monkeypatch):
    radii = []
    original = composite_helpers._feather_mask

    def capture(mask, radius):
        radii.append(radius)
        return original(mask, radius)

    monkeypatch.setattr(staged_compositor_helpers, "_feather_mask", capture)
    staged = {
        "version": 1,
        "layers": [
            {
                "socket": "foreground_0_face_0",
                "image": torch.ones(1, 9, 9, 3),
                "mask": torch.ones(1, 9, 9),
                "uses_embedded_alpha": True,
                "is_face": True,
                "feather_radius": 4,
            }
        ],
    }
    staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 36, 36, 3),
        staged,
        '{"version":3,"workspace_padding":0,"layers":{"foreground_0_face_0":{"scale":0.5}}}',
        0,
    )
    assert radii[:2] == [-8, -4]


def test_staged_face_reuses_first_raw_background_alpha():
    class CountingBackground:
        def __init__(self):
            self.calls = 0

        def encode_image(self, image):
            self.calls += 1
            return torch.full(
                image.shape[:3], 0.75, device=image.device, dtype=image.dtype
            )

    background_model = CountingBackground()
    staged = staged_face_helpers._stage_face_foregrounds(
        background_model,
        _FaceModel(),
        {"foreground_0": torch.ones(1, 20, 20, 3)},
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS,
        composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS | {"bbox_expansion": 0},
    )

    assert background_model.calls == 1
    assert torch.all(staged["layers"][0]["mask"] == 1)
    assert staged["layers"][1]["mask"].max().item() == pytest.approx(0.75)


def test_staged_face_detection_batches_foregrounds():
    class BatchedFaceModel(_FaceModel):
        def detect_batch(self, images, num_faces, score_thresh, variant):
            self.calls.append((len(images), num_faces, score_thresh, variant))
            results = []
            for image in images:
                height, width = image.shape[:2]
                landmarks = np.array(
                    [
                        [width * 0.25, height * 0.5],
                        [width * 0.5, height * 0.25],
                        [width * 0.75, height * 0.5],
                        [width * 0.5, height * 0.75],
                    ],
                    dtype=np.float32,
                )
                results.append(
                    [
                        {
                            "bbox_xyxy": np.array(
                                [
                                    width * 0.25,
                                    height * 0.25,
                                    width * 0.75,
                                    height * 0.75,
                                ],
                                dtype=np.float32,
                            ),
                            "landmarks_xy": landmarks,
                            "score": 0.9,
                            "presence": 1.0,
                        }
                    ]
                )
            return results

    model = BatchedFaceModel()
    foregrounds = {
        "foreground_0": torch.ones(1, 20, 20, 4),
        "foreground_1": torch.ones(1, 24, 16, 4),
    }
    staged = staged_face_helpers._stage_face_foregrounds(
        _BackgroundModel(),
        model,
        foregrounds,
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS,
        composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS | {"bbox_expansion": 0},
    )

    assert [call[0] for call in model.calls] == [2]
    assert [layer["socket"] for layer in staged["layers"]] == [
        "foreground_0",
        "foreground_0_face_0",
        "foreground_1",
        "foreground_1_face_0",
    ]


def test_staged_face_detection_retries_only_batch_misses():
    class RetryFaceModel(_FaceModel):
        def detect_batch(self, images, num_faces, score_thresh, variant):
            self.calls.append((len(images), score_thresh))
            if len(self.calls) == 1:
                return [
                    [
                        {
                            "bbox_xyxy": np.array([4, 4, 12, 12], dtype=np.float32),
                            "landmarks_xy": np.array(
                                [[4, 8], [8, 4], [12, 8], [8, 12]], dtype=np.float32
                            ),
                            "score": 0.9,
                            "presence": 1.0,
                        }
                    ],
                    [],
                ]
            return [
                [
                    {
                        "bbox_xyxy": np.array([4, 4, 12, 12], dtype=np.float32),
                        "landmarks_xy": np.array(
                            [[4, 8], [8, 4], [12, 8], [8, 12]], dtype=np.float32
                        ),
                        "score": 0.6,
                        "presence": 1.0,
                    }
                ]
            ]

    model = RetryFaceModel()
    staged_face_helpers._stage_face_foregrounds(
        _BackgroundModel(),
        model,
        {
            "foreground_0": torch.ones(1, 20, 20, 4),
            "foreground_1": torch.ones(1, 20, 20, 4),
        },
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS,
        composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS | {"bbox_expansion": 0},
    )

    assert model.calls == [(2, 0.55), (1, 0.385)]


def test_staged_face_rasterizes_crop_local_masks(monkeypatch):
    sizes = []
    original = background_replace_helpers._polygon_mask

    def capture(height, width, points, device, dtype):
        sizes.append((height, width))
        return original(height, width, points, device, dtype)

    monkeypatch.setattr(staged_face_helpers, "_polygon_mask", capture)
    staged_face_helpers._stage_face_foregrounds(
        _BackgroundModel(),
        _FaceModel(),
        {"foreground_0": torch.ones(1, 20, 20, 4)},
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS,
        composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS | {"bbox_expansion": 0},
    )

    assert sizes
    assert all(height < 20 and width < 20 for height, width in sizes)


def test_identity_projective_transform_skips_grid_sample(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("identity transform sampled a grid")

    monkeypatch.setattr(staged_compositor_helpers.F, "grid_sample", fail)
    image = torch.rand(1, 7, 9, 3)
    mask = torch.rand(1, 7, 9)
    warped_image, warped_mask = staged_compositor_helpers.projective_warp(
        image,
        mask,
        [[-1, -1], [1, -1], [1, 1], [-1, 1]],
        0,
    )

    assert warped_image is image
    assert warped_mask is mask


def test_projective_transform_samples_rgb_and_alpha_once(monkeypatch):
    calls = []
    original = staged_compositor_helpers.F.grid_sample

    def capture(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(staged_compositor_helpers.F, "grid_sample", capture)
    staged_compositor_helpers.projective_warp(
        torch.ones(1, 9, 9, 3),
        torch.ones(1, 9, 9),
        [[-0.8, -0.8], [0.8, -1], [0.7, 0.8], [-0.9, 0.7]],
        12,
    )

    assert calls == [(1, 4, 9, 9)]


def test_staged_fingerprint_tracks_only_stage_build_inputs():
    foregrounds = {
        "foreground_0": torch.arange(48, dtype=torch.float32).reshape(1, 4, 4, 3)
    }
    background_options = (
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS.copy()
    )
    face_options = composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS.copy()

    original = staged_compositor_helpers.staged_foreground_fingerprint(
        foregrounds,
        ("internal", "birefnet"),
        background_options,
        face_options,
    )
    composition_only = staged_compositor_helpers.staged_foreground_fingerprint(
        foregrounds,
        ("internal", "birefnet"),
        background_options | {"feather_radius": 19, "foreground_blend": 0.25},
        face_options | {"face_feather_radius": 31, "face_blend": 0.5},
    )
    changed_foregrounds = {"foreground_0": foregrounds["foreground_0"].clone()}
    changed_foregrounds["foreground_0"][0, 0, 0, 0] = 999.0
    changed_pixels = staged_compositor_helpers.staged_foreground_fingerprint(
        changed_foregrounds,
        ("internal", "birefnet"),
        background_options,
        face_options,
    )
    changed_mask_setting = staged_compositor_helpers.staged_foreground_fingerprint(
        foregrounds,
        ("internal", "birefnet"),
        background_options | {"mask_threshold": 0.75},
        face_options,
    )
    changed_face_setting = staged_compositor_helpers.staged_foreground_fingerprint(
        foregrounds,
        ("internal", "birefnet"),
        background_options,
        face_options | {"detection_threshold": 0.8},
    )

    assert composition_only == original
    assert changed_pixels != original
    assert changed_mask_setting != original
    assert changed_face_setting != original


def test_retained_stage_cache_copies_crops_and_evicts_lru():
    source = torch.ones(1, 20, 20, 3)
    source_mask = torch.ones(1, 20, 20)
    cache = staged_compositor_helpers.RetainedStageCache(max_entries=8)

    def stage(index):
        return {
            "version": 1,
            "layers": [
                {
                    "socket": f"foreground_{index}",
                    "image": source[:, 2:8, 3:9],
                    "mask": source_mask[:, 2:8, 3:9],
                }
            ],
        }

    cache["stage-0"] = stage(0)
    cache["stage-1"] = stage(1)
    stored = cache["stage-0"]["layers"][0]
    assert stored["image"].is_contiguous()
    assert stored["mask"].is_contiguous()
    assert (
        stored["image"].untyped_storage().data_ptr()
        != source.untyped_storage().data_ptr()
    )
    assert (
        stored["mask"].untyped_storage().data_ptr()
        != source_mask.untyped_storage().data_ptr()
    )
    assert (
        stored["image"].untyped_storage().nbytes()
        == stored["image"].numel() * stored["image"].element_size()
    )

    for index in range(2, 9):
        cache[f"stage-{index}"] = stage(index)
    assert "stage-1" not in cache
    assert list(cache) == ["stage-0", *(f"stage-{index}" for index in range(2, 9))]


def test_staged_preview_reuses_files_until_feather_changes(monkeypatch):
    saved = []

    def save(image, prefix):
        saved.append((image.clone(), prefix))
        return {"filename": f"{prefix}-{len(saved)}.png"}

    monkeypatch.setattr(staged_compositor_helpers, "_save_editor_preview", save)
    staged = {
        "version": 1,
        "_preview_cache": {},
        "layers": [
            {
                "socket": "foreground_0_face_0",
                "image": torch.ones(1, 9, 9, 3),
                "mask": torch.ones(1, 9, 9),
                "is_face": True,
                "uses_embedded_alpha": True,
                "feather_radius": 2,
            }
        ],
    }
    background = torch.zeros(1, 16, 16, 3)

    first = staged_compositor_helpers._preview_staged_foregrounds(background, staged, 0)
    second = staged_compositor_helpers._preview_staged_foregrounds(
        background, staged, 0
    )
    changed = staged_compositor_helpers._preview_staged_foregrounds(
        background,
        {**staged, "layers": [{**staged["layers"][0], "feather_radius": 3}]},
        0,
    )

    assert len(saved) == 2
    assert (
        first.ui["uc_layered_scene_editor"][0]["layers"][0]["preview"]
        == (second.ui["uc_layered_scene_editor"][0]["layers"][0]["preview"])
    )
    assert (
        changed.ui["uc_layered_scene_editor"][0]["layers"][0]["preview"]
        != (first.ui["uc_layered_scene_editor"][0]["layers"][0]["preview"])
    )


def test_staged_face_compositor_display_name_includes_staged():
    schema = composite_nodes.UC_StagedMediaPipeFaceBackgroundComposite.define_schema()
    assert schema.display_name == "Staged Face Background Composite"


def test_purpose_specific_background_node_schemas():
    alpha_schema = composite_nodes.UC_BackgroundRemovalPreserveAlpha.define_schema()
    individual_schema = composite_nodes.UC_StagedIndividualComposites.define_schema()

    assert alpha_schema.display_name == "Background Removal (Preserve Alpha)"
    assert [value.id for value in alpha_schema.inputs] == [
        "background_removal_model",
        "image",
        "background_removal_model_name",
        "background_options",
    ]
    background_options = alpha_schema.inputs[-1]
    assert background_options.io_type == "UC_STAGED_LAYERED_BACKGROUND_OPTIONS"
    assert background_options.optional is True
    assert background_options.display_name == "Background Options"
    assert individual_schema.display_name == "Staged Individual Composites"
    assert [output.io_type for output in alpha_schema.outputs] == ["IMAGE", "MASK"]
    assert [output.io_type for output in individual_schema.outputs] == [
        "IMAGE",
        "MASK",
        "BOUNDING_BOX",
    ]
    assert individual_schema.is_output_node is True


def test_background_removal_preserve_alpha_uses_soft_model_mask():
    image = torch.rand(2, 6, 8, 3)
    mask = torch.linspace(0, 1, 24).reshape(2, 3, 4)
    rgba, alpha = composite_helpers.background_removal_with_alpha(
        image, _QueuedBackgroundModel([mask])
    )

    assert rgba.shape == (2, 6, 8, 4)
    assert alpha.shape == (2, 6, 8)
    assert torch.equal(rgba[..., :3], image)
    assert torch.equal(rgba[..., 3], alpha)
    assert torch.any((alpha > 0) & (alpha < 1))


def test_background_removal_preserve_alpha_bypasses_model_for_rgba():
    image = torch.rand(2, 5, 7, 4)

    rgba, alpha = composite_helpers.background_removal_with_alpha(
        image,
        None,
        {
            "mask_threshold": 0.9,
            "border_cleanup_width": 12,
            "artifact_cleanup_radius": 12,
            "gap_fill_radius": 12,
            "mask_processing_resolution": 4,
            "feather_radius": 12,
        },
    )

    assert torch.equal(rgba, image)
    assert torch.equal(alpha, image[..., 3])


def test_face_removal_preserves_expanded_crops_and_matching_alpha(monkeypatch):
    image = torch.zeros((1, 6, 8, 4), dtype=torch.float32)
    image[..., 0] = torch.arange(8, dtype=torch.float32)[None, None, :] / 8
    image[..., 3] = 1
    faces = [
        {
            "bbox_xyxy": np.asarray([1, 1, 3, 3], dtype=np.float32),
            "landmarks_xy": np.zeros((1, 2), dtype=np.float32),
        },
        {
            "bbox_xyxy": np.asarray([4, 2, 8, 5], dtype=np.float32),
            "landmarks_xy": np.zeros((1, 2), dtype=np.float32),
        },
    ]
    monkeypatch.setattr(
        staged_face_helpers,
        "detect_many_or_warn",
        lambda *_args, **_kwargs: ({"0": faces}, set()),
    )
    monkeypatch.setattr(
        staged_face_helpers,
        "_polygon_mask",
        lambda height, width, *_args: torch.ones((height, width)),
    )
    face_model = types.SimpleNamespace(connection_sets={"face_oval": [(0, 0)]})
    background_options = {
        "mask_processing_resolution": 0,
    }
    face_options = {
        "maximum_faces": 2,
        "detection_threshold": 0.5,
        "bbox_expansion": 0,
        "mask_expansion": 0,
        "face_feather_radius": 0,
    }

    rgba, alpha = staged_face_helpers.face_removal_with_alpha(
        image, None, face_model, background_options, face_options
    )

    assert rgba.shape == (2, 3, 4, 4)
    assert alpha.shape == (2, 3, 4)
    assert torch.equal(rgba[..., 3], alpha)
    assert torch.equal(alpha[0, 0], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.equal(alpha[0, 1], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.equal(alpha[0, 2], torch.zeros(4))
    assert torch.equal(alpha[1], torch.ones((3, 4)))
    assert torch.equal(rgba[1, ..., 0], image[0, 2:5, 4:8, 0])


def test_face_removal_node_exposes_shared_options_and_lazy_rgba_bypass():
    schema = composite_nodes.UC_FaceRemovalPreserveAlpha.define_schema()

    assert schema.display_name == "Face Removal (Preserve Alpha)"
    assert [output.display_name for output in schema.outputs] == [
        "RGBA Face",
        "Face Alpha Mask",
    ]
    assert composite_nodes.UC_FaceRemovalPreserveAlpha.check_lazy_status(
        torch.zeros((1, 2, 2, 4)),
        background_removal_model=(None, "model_socket"),
    ) == []


def test_background_removal_options_control_processing_and_refinement(monkeypatch):
    image = torch.rand(1, 8, 12, 3)
    captured = {}

    class Model:
        image_size = 4

        def encode_image(self, value):
            captured["input_shape"] = tuple(value.shape)
            mask = value.new_zeros((1, value.shape[1], value.shape[2]))
            mask[:, 1:3, 2:4] = 0.75
            return mask

    original_refine = composite_helpers._refine_foreground_mask

    def capture_refine(mask, threshold, border, artifact, gap):
        captured["refine"] = (threshold, border, artifact, gap)
        return original_refine(mask, threshold, border, artifact, gap)

    monkeypatch.setattr(
        composite_helpers, "_refine_foreground_mask", capture_refine
    )

    rgba, alpha = composite_helpers.background_removal_with_alpha(
        image,
        Model(),
        {
            "mask_threshold": 0.7,
            "border_cleanup_width": 0,
            "artifact_cleanup_radius": 0,
            "gap_fill_radius": 0,
            "mask_processing_resolution": 6,
            "feather_radius": 0,
            "mask_resize_method": "nearest-exact",
        },
    )

    assert captured["input_shape"] == (1, 4, 6, 3)
    assert captured["refine"] == (0.7, 0, 0, 0)
    assert rgba.shape == (1, 8, 12, 4)
    assert alpha.shape == (1, 8, 12)
    assert torch.equal(rgba[..., :3], image)
    assert set(alpha.unique().tolist()) <= {0.0, 1.0}


def test_background_removal_options_apply_inward_feather():
    image = torch.rand(1, 9, 9, 3)
    raw_mask = torch.zeros(1, 9, 9)
    raw_mask[:, 2:7, 2:7] = 1.0

    _, alpha = composite_helpers.background_removal_with_alpha(
        image,
        _QueuedBackgroundModel([raw_mask]),
        {
            "mask_threshold": 0.5,
            "border_cleanup_width": 0,
            "artifact_cleanup_radius": 0,
            "gap_fill_radius": 0,
            "mask_processing_resolution": 0,
            "feather_radius": 2,
            "mask_resize_method": "auto",
        },
    )

    assert torch.any((alpha > 0) & (alpha < 1))
    assert torch.all(alpha <= raw_mask)


def test_existing_staged_nodes_keep_original_four_outputs():
    assert len(composite_nodes.UC_StagedLayeredBackgroundComposite.define_schema().outputs) == 4
    assert len(composite_nodes.UC_StagedMediaPipeFaceBackgroundComposite.define_schema().outputs) == 4


def test_staged_mask_output_is_preallocated_without_stack(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers.torch,
        "stack",
        lambda *args, **kwargs: pytest.fail("layer masks must not be stacked"),
    )
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )

    output = staged_compositor_helpers._composite_staged_foregrounds(
        torch.zeros(1, 8, 8, 3), _paint_test_stage(), '{"version":3,"layers":{}}', 0
    )

    assert output.result[3].shape == (1, 8, 8)


def test_individual_renderer_does_not_enter_accumulated_renderer(monkeypatch):
    monkeypatch.setattr(
        staged_compositor_helpers,
        "_composite_staged_foregrounds",
        lambda *args, **kwargs: pytest.fail("accumulated renderer was called"),
    )
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )

    output = staged_compositor_helpers._composite_staged_individual_foregrounds(
        torch.zeros(1, 8, 8, 3), _paint_test_stage(), '{"version":3,"layers":{}}', 0
    )

    assert output.result[0].shape == (1, 8, 8, 3)
    assert output.result[1].shape == (1, 8, 8)
    assert len(output.result[2][0]) == 1


def test_background_removal_node_bypasses_internal_model_for_rgba(monkeypatch):
    monkeypatch.setattr(
        composite_nodes,
        "_load_internal_background_removal_model",
        lambda *args: pytest.fail("RGBA input must not load a removal model"),
    )
    image = torch.rand(2, 4, 5, 4)

    output = composite_nodes.UC_BackgroundRemovalPreserveAlpha.execute(image)

    assert torch.equal(output.result[0], image)
    assert torch.equal(output.result[1], image[..., 3])


def test_staged_individual_node_owns_staging_and_outputs_aligned_results(monkeypatch):
    node = composite_nodes.UC_StagedIndividualComposites
    node._staged_by_node.clear()
    monkeypatch.setattr(node, "hidden", types.SimpleNamespace(unique_id="individual"))
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    model = _QueuedBackgroundModel(
        [torch.ones(1, 2, 2), torch.ones(1, 2, 2)]
    )

    foregrounds = {
        "foreground_0": torch.ones(1, 2, 2, 3),
        "foreground_1": torch.full((1, 2, 2, 3), 0.5),
    }
    output = node.execute(
        background=torch.zeros(1, 10, 10, 3),
        foreground_images=foregrounds,
        execution_mode="full_run",
        placement_data=json.dumps(
            {
                "version": 3,
                "layer_order": ["foreground_1", "foreground_0"],
                "layers": {},
            }
        ),
        background_removal_model=model,
        background_options={
            "border_cleanup_width": 0,
            "artifact_cleanup_radius": 0,
            "gap_fill_radius": 0,
        },
    )

    assert output.result[0].shape == (2, 10, 10, 3)
    assert output.result[1].shape == (2, 10, 10)
    assert len(output.result[2][0]) == 2
    assert "individual" in node._staged_by_node

    retained = node.execute(
        background=torch.zeros(1, 10, 10, 3),
        foreground_images=foregrounds,
        execution_mode="run_staged",
        placement_data='{"version":3,"layers":{"foreground_0":{"center_x":0.7}}}',
        background_removal_model=model,
        background_options={
            "border_cleanup_width": 0,
            "artifact_cleanup_radius": 0,
            "gap_fill_radius": 0,
        },
    )
    assert retained.ui["uc_layered_scene_editor"][0]["stage_mode"] == "retained"
    assert model.masks == []


def test_face_run_staged_loads_no_models(monkeypatch):
    node = composite_nodes.UC_StagedMediaPipeFaceBackgroundComposite
    node._staged_by_node.clear()
    monkeypatch.setattr(
        node, "hidden", types.SimpleNamespace(unique_id="retained-face")
    )
    monkeypatch.setattr(
        composite_nodes,
        "_load_internal_background_removal_model",
        lambda *args: (_ for _ in ()).throw(AssertionError("loaded removal model")),
    )
    monkeypatch.setattr(
        composite_nodes,
        "load_face_model",
        lambda *args: (_ for _ in ()).throw(AssertionError("loaded face model")),
    )
    monkeypatch.setattr(
        staged_compositor_helpers, "_save_editor_preview", lambda *args: None
    )
    foregrounds = {"foreground_0": torch.ones(1, 4, 4, 3)}
    background_options = (
        composite_nodes.UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS
    )
    face_options = composite_nodes.UC_StagedMediaPipeFaceOptions.DEFAULTS
    fingerprint = staged_compositor_helpers.staged_foreground_fingerprint(
        foregrounds,
        ("internal", "birefnet", "mediapipe_face_fp32"),
        background_options,
        face_options,
    )
    node._staged_by_node["retained-face"] = {
        "version": 1,
        "_stage_fingerprint": fingerprint,
        "layers": [
            {
                "socket": "foreground_0",
                "image": torch.ones(1, 4, 4, 3),
                "mask": torch.ones(1, 4, 4),
                "uses_embedded_alpha": True,
            }
        ],
    }

    output = node.execute(
        background=torch.zeros(1, 12, 12, 3),
        foreground_images=foregrounds,
        execution_mode="run_staged",
        placement_data='{"version":3,"layers":{}}',
    )
    assert output.result[0].sum() > 0
