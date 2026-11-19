import torch

from ..helpers.image_helpers import stitch_image_grid
from ..nodes.image_nodes import UC_StitchImageGrid


def test_stitch_image_grid_empty():
    res = stitch_image_grid([])
    assert res.ndim == 4
    assert res.shape == (1, 64, 64, 3)


def test_stitch_image_grid_single():
    img = torch.rand((1, 50, 50, 3))
    res = stitch_image_grid(img)
    assert res.shape == (1, 50, 50, 3)


def test_stitch_image_grid_batch():
    batch = torch.zeros((6, 64, 64, 3))
    # 2 rows of 3 images, 4px spacing
    res = stitch_image_grid(batch, max_images_per_row=3, max_rows=0, match_image_size=True, spacing_width=4, spacing_color="white")
    # Width: 64*3 + 4*2 = 200
    # Height: 64*2 + 4 = 132
    assert res.shape == (1, 132, 200, 3)


def test_stitch_image_grid_max_rows():
    batch = torch.zeros((6, 64, 64, 3))
    # max_images_per_row=2, max_rows=2 -> truncates to 4 images
    res = stitch_image_grid(batch, max_images_per_row=2, max_rows=2, match_image_size=True, spacing_width=0, spacing_color="black")
    # Width: 64*2 = 128
    # Height: 64*2 = 128
    assert res.shape == (1, 128, 128, 3)


def test_stitch_image_grid_different_sizes():
    imgs = [torch.zeros((1, 64, 64, 3)), torch.zeros((1, 80, 70, 3)), torch.zeros((1, 50, 60, 3)), torch.zeros((1, 90, 90, 3))]
    # match_image_size=True
    res_matched = stitch_image_grid(imgs, max_images_per_row=2, max_rows=0, match_image_size=True, spacing_width=2, spacing_color="black")
    assert res_matched.ndim == 4
    assert res_matched.shape[0] == 1

    # match_image_size=False
    res_native = stitch_image_grid(imgs, max_images_per_row=2, max_rows=0, match_image_size=False, spacing_width=2, spacing_color="black")
    assert res_native.ndim == 4
    assert res_native.shape[0] == 1


def test_stitch_image_grid_schema_and_execute():
    schema = UC_StitchImageGrid.define_schema()
    assert schema.node_id == "UC_StitchImageGrid"
    assert schema.display_name == "Stitch Image Grid"
    assert schema.category == "advanced/image"
    assert schema.is_input_list is True

    input_ids = [inp.id for inp in schema.inputs]
    assert input_ids == ["images", "max_images_per_row", "max_rows", "match_image_size", "spacing_width", "spacing_color", "as_list"]
    assert schema.inputs[-1].default is False
    assert schema.outputs[0].id == "image"
    assert schema.outputs[0].is_output_list is True

    batch = [torch.zeros((1, 32, 32, 3)), torch.zeros((1, 32, 32, 3))]
    out = UC_StitchImageGrid.execute(images=batch, max_images_per_row=[2], max_rows=[0], match_image_size=[True], spacing_width=[0], spacing_color=["white"], as_list=[True])
    assert isinstance(out.args[0], list)
    assert len(out.args[0]) == 1
    assert out.args[0][0].shape == (1, 32, 64, 3)


def test_stitch_image_grid_per_item_default_and_broadcasting():
    batches = [torch.zeros((2, 32, 32, 3)), torch.zeros((3, 32, 32, 3))]
    out = UC_StitchImageGrid.execute(images=batches, max_images_per_row=[2, 3], max_rows=[0], match_image_size=[True], spacing_width=[0], spacing_color=["white"], as_list=[False])
    assert isinstance(out.args[0], list)
    assert [img.shape for img in out.args[0]] == [(1, 32, 64, 3), (1, 32, 96, 3)]


def test_stitch_image_grid_single_batch_and_empty_with_as_list_off():
    single = UC_StitchImageGrid.execute(images=[torch.zeros((4, 32, 32, 3))], max_images_per_row=[2], max_rows=[0], match_image_size=[True], spacing_width=[0], spacing_color=["white"], as_list=[False])
    assert [img.shape for img in single.args[0]] == [(1, 64, 64, 3)]

    empty = UC_StitchImageGrid.execute(images=[], as_list=[False])
    assert [img.shape for img in empty.args[0]] == [(1, 64, 64, 3)]
