import pathlib
import sys
import types

import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_parameter_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_parameter_test import parameter_nodes
    from utils_collection_parameter_test.helper_functions import AspectRatio
    from utils_collection_parameter_test.parameter_helpers import select_video_resolution
finally:
    cli_args.cpu = prior_cpu


def test_image_scale_picker_schema_uses_smooth_default_and_positive_scale():
    schema = parameter_nodes.UC_ImageScaleAndResolutionPicker.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    outputs = {value.id: value for value in schema.outputs if value.id}

    assert inputs["upscale_method"].default == "lanczos"
    assert inputs["scale_by"].min > 0
    assert "scale_by" in outputs["upscaled_image"].tooltip
    assert "upscale_by" not in outputs["upscaled_image"].tooltip


def test_video_resolution_selector_uses_nominal_megapixel_target_with_ratio_tolerance():
    width, height = select_video_resolution(
        16,
        9,
        megapixels=0.4,
        multiple=32,
        minimum=256,
        maximum=4096,
    )

    assert (width, height) == (864, 480)
    assert width % 32 == height % 32 == 0


def test_video_resolution_selector_uses_middle_band_video_rungs():
    selected = {
        megapixels: select_video_resolution(
            16,
            9,
            megapixels=megapixels,
            multiple=32,
            minimum=256,
            maximum=4096,
        )
        for megapixels in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    }

    assert selected == {
        0.3: (768, 416),
        0.4: (864, 480),
        0.5: (1024, 576),
        0.6: (1024, 576),
        0.7: (1152, 672),
        0.8: (1152, 672),
    }


def test_video_resolution_selector_schema_defaults_to_video_multiple():
    schema = parameter_nodes.UC_VideoResolutionSelector.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert inputs["multiple"].default == 32
    assert "minimum" not in inputs


def test_video_resolution_selector_returns_resolution_preview():
    output = parameter_nodes.UC_VideoResolutionSelector.execute(
        aspect_ratio=AspectRatio.WIDESCREEN_H,
        megapixels=0.4,
        multiple=32,
        minimum=256,
        duration_seconds=9.3112024,
    )

    assert output.result == (864, 480, 226)
    assert output.ui == {"resolution": ("864×480 · 226 frames",)}


def test_regular_resolution_selector_returns_resolution_preview():
    output = parameter_nodes.UC_ResolutionSelectorExtended.execute(
        aspect_ratio=AspectRatio.WIDESCREEN_H,
        megapixels=0.4,
        multiple=32,
        minimum=256,
    )

    assert output.result == (864, 480)
    assert output.ui == {"resolution": ("864×480",)}


def test_resolution_preview_frontend_is_display_only_and_live():
    frontend = (CUSTOM_NODE_ROOT / "web" / "resolution_preview.js").read_text(
        encoding="utf-8"
    )

    assert "onDrawForeground" in frontend
    assert "onWidgetChanged" in frontend
    assert "prototype.computeSize" in frontend
    assert "prototype.onResize" in frontend
    assert "addWidget" not in frontend


def test_video_resolution_selector_keeps_an_aspect_ratio_axis_exact():
    width, height = select_video_resolution(
        21,
        9,
        megapixels=0.5,
        multiple=32,
        minimum=256,
        maximum=4096,
    )

    assert (width, height) == (1120, 480)
    assert width % 7 == 0


def test_image_scale_picker_keeps_megapixel_fit_and_upscale_factor_separate():
    image = torch.zeros(1, 100, 200, 3)

    output = parameter_nodes.UC_ImageScaleAndResolutionPicker.execute(
        image=image,
        upscale_method="bilinear",
        crop_method="disabled",
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        resolution_steps=256,
        scale_by=2.0,
        multiple=16,
    )
    adjusted, upscaled, width, height, upscaled_width, upscaled_height = output.result

    assert (width, height) == (144, 80)
    assert (upscaled_width, upscaled_height) == (288, 160)
    assert adjusted.shape == (1, 80, 144, 3)
    assert upscaled.shape == (1, 160, 288, 3)


def test_image_scale_picker_center_crop_uses_adjusted_base_for_upscale():
    image = torch.zeros(1, 100, 200, 3)

    output = parameter_nodes.UC_ImageScaleAndResolutionPicker.execute(
        image=image,
        upscale_method="bilinear",
        crop_method="center",
        aspect_ratio=AspectRatio.SQUARE,
        megapixels=0.01,
        resolution_steps=1,
        scale_by=1.5,
        multiple=16,
    )

    assert output.result[2:] == (96, 96, 144, 144)
    assert output.result[0].shape == (1, 96, 96, 3)
    assert output.result[1].shape == (1, 144, 144, 3)
