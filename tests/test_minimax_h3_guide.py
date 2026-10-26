import pathlib
import sys
import types

import pytest
import torch


PACKAGE_NAME = "utils_collection_h3_guide_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(pathlib.Path(__file__).parents[1])]
sys.modules.setdefault(PACKAGE_NAME, package)

from utils_collection_h3_guide_test.helpers.minimax_h3_guide_helpers import (
    LAYOUT_KEY,
    build_layout,
    layout_from_token_boundary,
    splice_conditioning,
)


def _entry(values, boundary=None, tags=None, **metadata):
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
    metadata["minimax_token_tags"] = torch.tensor(
        tags if tags is not None else [1] * tensor.shape[1], dtype=torch.long
    )
    if boundary is not None:
        metadata[LAYOUT_KEY] = {
            "version": 1, "sequence_length": tensor.shape[1], "prompt_start": boundary
        }
    return [tensor, metadata]






@pytest.mark.parametrize(
    "length,entries,start,expected",
    [
        (3, [(21, 1), (22, 1), (23, 1)], 0, 0),
        # Image expands to four slots; wrapper and following label are media.
        (9, [(11, 1), ({"type": "image"}, 1), (12, 1), (13, 1), (21, 1), (22, 1)], 4, 7),
        # Two video blocks and trailing audio label: final text tags cannot locate this split.
        (16, [({"type": "image"}, 1), ({"type": "image"}, 1), (13, 1), (21, 1)], 3, 15),
        # Already-cleaned weighted prompt; no re-tokenizing the original syntax.
        (7, [({"type": "image"}, 1), (21, 1.5), (22, 0.5)], 1, 5),
        # Textual embedding expands to three positions, plus one literal token.
        (10, [({"type": "image"}, 1), (torch.ones(3, 4), 1), (22, 1)], 1, 6),
        (5, [({"type": "image"}, 1), (13, 1)], 2, 5),
    ],
)
def test_boundary_uses_actual_expanded_prompt_suffix(length, entries, start, expected):
    cond = torch.zeros(1, length, 4)
    tags = torch.ones(length, dtype=torch.long)
    assert layout_from_token_boundary(cond, tags, entries, start) == {
        "version": 1, "sequence_length": length, "prompt_start": expected
    }


def test_empty_input_inserts_before_native_pad():
    cond = torch.zeros(1, 1, 4)
    tags = torch.ones(1, dtype=torch.long)
    assert layout_from_token_boundary(
        cond, tags, [(151643, 1)], 0, empty_prompt_pad=True
    )["prompt_start"] == 0
    base = _entry([0], 0)
    result = splice_conditioning([base], [_entry([8, 9], tags=[1, 0])])
    assert result[0][0].flatten().tolist() == [8, 9, 0]


@pytest.mark.parametrize("entries,start,length", [([(1, 1)], 0, 1), ([(151643, 1)], 1, 1), ([(151643, 1)], 0, 2)])
def test_empty_pad_flag_requires_native_empty_input(entries, start, length):
    with pytest.raises(ValueError, match="pad"):
        layout_from_token_boundary(
            torch.zeros(1, length, 4), torch.ones(length, dtype=torch.long),
            entries, start, empty_prompt_pad=True,
        )


def test_splice_preserves_base_slices_and_auxiliary_metadata_without_mutation():
    pooled = torch.tensor([[42.0]])
    references = [torch.tensor([7.0])]
    base = _entry([10, 11, 12, 20, 21], 3, [1, 0, 1, 1, 1],
                  pooled_output=pooled, ref_latents=references, start_percent=0.25)
    guide = _entry([30, 31, 32], tags=[1, 0, 0], pooled_output=torch.tensor([-1]))
    result = splice_conditioning([base], [guide])[0]
    assert result[0].flatten().tolist() == [10, 11, 12, 30, 31, 32, 20, 21]
    assert result[1]["minimax_token_tags"].tolist() == [1, 0, 1, 1, 0, 0, 1, 1]
    assert result[1][LAYOUT_KEY] == {"version": 1, "sequence_length": 8, "prompt_start": 6}
    assert result[1]["pooled_output"] is pooled
    assert result[1]["ref_latents"] is references
    assert result[1]["start_percent"] == 0.25
    assert result[1] is not base[1]
    assert result[1][LAYOUT_KEY] is not base[1][LAYOUT_KEY]
    assert base[0].flatten().tolist() == [10, 11, 12, 20, 21]
    assert base[1]["minimax_token_tags"].tolist() == [1, 0, 1, 1, 1]
    assert base[1][LAYOUT_KEY]["prompt_start"] == 3
    assert guide[0].flatten().tolist() == [30, 31, 32]
    assert LAYOUT_KEY not in guide[1]


def test_chained_guides_keep_insertion_order():
    first = splice_conditioning([_entry([10, 20], 1)], [_entry([30, 31])])
    result = splice_conditioning(first, [_entry([40, 41, 42])])[0]
    assert result[0].flatten().tolist() == [10, 30, 31, 40, 41, 42, 20]
    assert result[1][LAYOUT_KEY]["prompt_start"] == 6
    assert first[0][1][LAYOUT_KEY]["prompt_start"] == 3


def test_single_guide_broadcasts_over_schedules_and_batches_and_matches_dtype():
    early = _entry([10, 20], 1, start_percent=0.0, end_percent=0.5)
    early[0] = early[0].expand(2, -1, -1).to(torch.float64)
    late = _entry([50, 51, 60], 2, start_percent=0.5, end_percent=1.0)
    result = splice_conditioning([early, late], [_entry([30])])
    assert result[0][0].tolist() == [[[10], [30], [20]], [[10], [30], [20]]]
    assert result[0][0].dtype == torch.float64
    assert result[0][0].device == early[0].device
    assert result[1][0].flatten().tolist() == [50, 51, 30, 60]
    assert result[1][0].dtype == torch.float32
    assert result[0][1]["end_percent"] == 0.5
    assert result[1][1]["start_percent"] == 0.5


@pytest.mark.parametrize("count", [0, 2])
def test_guide_rejects_multiple_or_missing_schedules(count):
    with pytest.raises(ValueError, match="exactly one"):
        splice_conditioning([_entry([10, 20], 1)], [_entry([30])] * count)


@pytest.mark.parametrize("layout", [
    None,
    {"version": 2, "sequence_length": 2, "prompt_start": 1},
    {"version": 1, "sequence_length": 3, "prompt_start": 1},
    {"version": 1, "sequence_length": 2, "prompt_start": 3},
    {"version": 1, "sequence_length": 2, "prompt_start": -1},
    {"version": True, "sequence_length": 2, "prompt_start": 1},
    {"version": 1, "sequence_length": 2, "prompt_start": 1.0},
    {"version": 1, "sequence_length": 2, "prompt_start": 1, "cache": torch.zeros(1)},
])
def test_missing_malformed_or_stale_layout_fails(layout):
    base = _entry([10, 20], 1)
    base[1][LAYOUT_KEY] = layout
    with pytest.raises(ValueError, match="MiniMax H3"):
        splice_conditioning([base], [_entry([30])])


@pytest.mark.parametrize("tags", [None, torch.ones(1, dtype=torch.long), torch.ones(1, 2, dtype=torch.long), torch.ones(2)])
def test_invalid_tags_rejected_for_base_and_guide(tags):
    for invalid_base in (True, False):
        base = _entry([10, 20], 1)
        guide = _entry([30, 31])
        (base if invalid_base else guide)[1]["minimax_token_tags"] = tags
        with pytest.raises(ValueError, match="token tags"):
            splice_conditioning([base], [guide])


@pytest.mark.parametrize("shape,error", [((1, 1, 2), "feature"), ((2, 1, 1), "batch"), ((1, 1), "shape"), ((1, 0, 1), "nonempty")])
def test_guide_shape_must_match_base(shape, error):
    guide = _entry([30])
    guide[0] = torch.zeros(shape)
    with pytest.raises(ValueError, match=error):
        splice_conditioning([_entry([10, 20], 1)], [guide])


@pytest.mark.parametrize("start", [-1, 3, True, 0.5])
def test_invalid_token_boundary_rejected(start):
    with pytest.raises(ValueError, match="token boundary"):
        layout_from_token_boundary(torch.zeros(1, 2, 4), torch.ones(2, dtype=torch.long), [(1, 1)], start)


def test_unknown_suffix_expansion_and_excess_suffix_length_fail():
    cond, tags = torch.zeros(1, 1, 4), torch.ones(1, dtype=torch.long)
    with pytest.raises(ValueError, match="text IDs or embedding tensors"):
        layout_from_token_boundary(cond, tags, [({"type": "image"}, 1)], 0)
    with pytest.raises(ValueError, match="prompt boundary"):
        layout_from_token_boundary(cond, tags, [(torch.ones(3, 4), 1)], 0)


def test_layout_construction_is_scalar_only_and_does_not_change_inputs():
    cond = torch.tensor([[[1.0], [2.0]]])
    tags = torch.tensor([0, 1], dtype=torch.long)
    layout = build_layout(cond, tags, 1)
    assert set(layout) == {"version", "sequence_length", "prompt_start"}
    assert all(type(value) is int for value in layout.values())
    assert cond.tolist() == [[[1.0], [2.0]]]
    assert tags.tolist() == [0, 1]


def test_frame_index_resolution_zero_and_positive():
    from utils_collection_h3_guide_test.helpers.encoder_helpers import resolve_minimax_h3_frame_index
    assert resolve_minimax_h3_frame_index(0) == 0.0
    assert resolve_minimax_h3_frame_index(1) == pytest.approx(1.0 / 24.0)
    assert resolve_minimax_h3_frame_index(24) == 1.0


def test_frame_index_resolution_negative_with_conditioning_metadata():
    from utils_collection_h3_guide_test.helpers.encoder_helpers import resolve_minimax_h3_frame_index
    cond = [[torch.zeros(1, 1, 1), {"minimax_frame_count": 124}]]
    assert resolve_minimax_h3_frame_index(0, conditioning=cond) == 0.0
    assert resolve_minimax_h3_frame_index(1, conditioning=cond) == pytest.approx(1.0 / 24.0)
    assert resolve_minimax_h3_frame_index(-1, conditioning=cond) == pytest.approx(123.0 / 24.0)
    assert resolve_minimax_h3_frame_index(-124, conditioning=cond) == 0.0


def test_frame_index_resolution_negative_with_latent():
    from utils_collection_h3_guide_test.helpers.encoder_helpers import resolve_minimax_h3_frame_index
    # 7 latent steps correspond to 22 frames in MiniMax H3
    samples = torch.zeros(1, 24, 7, 4, 4)
    latent = {"samples": type("NestedStub", (), {"is_nested": True, "tensors": (samples, torch.zeros(1, 32, 2, 4))})()}
    assert resolve_minimax_h3_frame_index(-1, latent=latent) == pytest.approx(21.0 / 24.0)
    assert resolve_minimax_h3_frame_index(0, latent=latent) == 0.0


@pytest.mark.parametrize("bad_index", [0.5, "0", True, False, None])
def test_frame_index_invalid_type_rejected(bad_index):
    from utils_collection_h3_guide_test.helpers.encoder_helpers import resolve_minimax_h3_frame_index
    with pytest.raises(ValueError, match="integer"):
        resolve_minimax_h3_frame_index(bad_index)


def test_frame_index_negative_without_count_fails():
    from utils_collection_h3_guide_test.helpers.encoder_helpers import resolve_minimax_h3_frame_index
    with pytest.raises(ValueError, match="requires video frame count"):
        resolve_minimax_h3_frame_index(-1)


def test_frame_index_out_of_bounds_fails():
    from utils_collection_h3_guide_test.helpers.encoder_helpers import resolve_minimax_h3_frame_index
    cond = [[torch.zeros(1, 1, 1), {"minimax_frame_count": 10}]]
    with pytest.raises(ValueError, match="outside"):
        resolve_minimax_h3_frame_index(10, conditioning=cond)
    with pytest.raises(ValueError, match="outside"):
        resolve_minimax_h3_frame_index(-11, conditioning=cond)


def test_minimax_h3_vlm_frame_guide_subclass_and_schema():
    from utils_collection_h3_guide_test.nodes.encoder_nodes import (
        UC_MiniMaxH3VLMGuide,
        UC_MiniMaxH3VLMFrameGuide,
    )
    assert issubclass(UC_MiniMaxH3VLMFrameGuide, UC_MiniMaxH3VLMGuide)
    schema = UC_MiniMaxH3VLMFrameGuide.define_schema()
    input_ids = [inp.id for inp in schema.inputs]
    assert "frame_index" in input_ids
    assert "timestamp" not in input_ids
    assert "latent" in input_ids
    assert schema.node_id == "UC_MiniMaxH3VLMFrameGuide"
    assert all(inp.tooltip for inp in schema.inputs)
    assert all(out.tooltip for out in schema.outputs)

