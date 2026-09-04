import pathlib
import sys
import types

import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_sam3_tiling_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_sam3_tiling_test import model_helpers
from utils_collection_sam3_tiling_test.model_helpers import (
    SAM3_EDGE_PADDING,
    SAM3_WORKING_SIZE,
    _padded_sam3_image,
    _padded_sam3_tile,
    _sam3_long_axis_tile_count,
    _sam3_tile_records,
)


def test_sam3_tiles_split_only_the_longer_axis_with_a_centre_tile():
    records = _sam3_tile_records(4096, 3072, SAM3_EDGE_PADDING)

    assert records == [
        (512, 3648, 0, 3136),
        (0, 2261, 0, 3136),
        (1899, 4160, 0, 3136),
    ]
    assert SAM3_EDGE_PADDING == 32


def test_sam3_without_edge_padding_keeps_dynamic_overlap():
    records = _sam3_tile_records(4096, 3072, 0)

    assert records == [
        (512, 3584, 0, 3072),
        (0, 2219, 0, 3072),
        (1877, 4096, 0, 3072),
    ]


def test_sam3_long_axis_tile_count_is_odd_and_keeps_a_centre_tile():
    assert _sam3_long_axis_tile_count(4096, 3072) == 3
    assert _sam3_long_axis_tile_count(8192, 1024) == 9


def test_sam3_edge_tile_preserves_source_pixels_and_replicates_only_padding():
    image = torch.zeros(1, 3, 500, 750)
    image[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    image[0, :, -1, -1] = torch.tensor([4.0, 5.0, 6.0])

    tile, geometry = _padded_sam3_tile(image, 0, 500, 0, 750, "cpu", torch.float32)

    assert tuple(tile.shape) == (1, 3, 1008, 1008)
    assert geometry == (500, 750, 750, 250, 0)


def test_sam3_downscaling_uses_lanczos(monkeypatch):
    calls = []

    def common_upscale(samples, width, height, method, crop):
        calls.append((tuple(samples.shape), width, height, method, crop))
        return torch.zeros(samples.shape[0], samples.shape[1], height, width)

    monkeypatch.setattr(model_helpers.comfy.utils, "common_upscale", common_upscale)

    tile, _ = _padded_sam3_tile(torch.zeros(1, 3, 1200, 1000), 0, 1200, 0, 1000, "cpu", torch.float32)

    assert tuple(tile.shape) == (1, 3, 1008, 1008)
    assert calls == [((1, 3, 1200, 1200), 1008, 1008, "lanczos", "disabled")]


def test_sam3_image_border_replicates_nearest_edge_pixels():
    image = torch.zeros(1, 2, 3, 3)
    image[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    image[0, 1, -1] = torch.tensor([4.0, 5.0, 6.0])

    padded = _padded_sam3_image(image, SAM3_EDGE_PADDING)

    assert tuple(padded.shape) == (1, 3, 66, 67)
    assert torch.equal(padded[0, :, 0, 0], image[0, 0, 0])
    assert torch.equal(padded[0, :, -1, -1], image[0, 1, -1])
