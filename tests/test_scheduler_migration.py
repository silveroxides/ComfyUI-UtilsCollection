import types

import pytest
import torch

import comfy.samplers

from .. import node_replacements
from ..helpers import scheduler_helpers
from ..nodes import scheduler_nodes


EXPECTED_NODE_IDS = {
    "UC_SigmoidOffsetScheduler",
    "UC_PowerShiftScheduler",
    "UC_RadianceShiftScheduler",
    "UC_SigmaCurveFromPointsScheduler",
    "UC_SigmaCurvePchipScheduler",
}

EXPECTED_HANDLER_NAMES = {
    "dmad",
    "sigmoid_offset",
    "power_shift",
    "radiance_shift",
    "sigma_curve_from_points",
    "sigma_curve_pchip",
}

OLD_TO_NEW = {
    "LogicIF": "UC_LogicIF",
    "LogicAND": "UC_LogicAND",
    "LogicOR": "UC_LogicOR",
    "LogicNOT": "UC_LogicNOT",
    "LogicXOR": "UC_LogicXOR",
    "MathAdd": "UC_MathAdd",
    "MathSubtract": "UC_MathSubtract",
    "MathMultiply": "UC_MathMultiply",
    "MathDivide": "UC_MathDivide",
    "MathPower": "UC_MathPower",
    "MathFloor": "UC_MathFloor",
    "MathCeil": "UC_MathCeil",
    "MathRound": "UC_MathRound",
    "MathModulo": "UC_MathModulo",
    "MathAbs": "UC_MathAbs",
    "MathSqrt": "UC_MathSqrt",
    "MathSin": "UC_MathSin",
    "MathCos": "UC_MathCos",
    "MathTan": "UC_MathTan",
    "MathMin": "UC_MathMin",
    "MathMax": "UC_MathMax",
    "MathClamp": "UC_MathClamp",
    "MathNumberConvert": "UC_MathNumberConvert",
    "StringToNumber": "UC_StringToNumber",
    "NumberToString": "UC_NumberToString",
    "MathCompare": "UC_MathCompare",
    "MathOperation": "UC_MathOperation",
    "MathAspectRatio": "UC_MathAspectRatio",
    "SigmoidOffsetScheduler": "UC_SigmoidOffsetScheduler",
    "PowerShiftScheduler": "UC_PowerShiftScheduler",
    "RadianceShiftScheduler": "UC_RadianceShiftScheduler",
    "SigmaCurveFromPointsScheduler": "UC_SigmaCurveFromPointsScheduler",
    "SigmaCurvePchipScheduler": "UC_SigmaCurvePchipScheduler",
}


class _Model:
    def __init__(self):
        self.model_sampling = types.SimpleNamespace(
            sigmas=torch.linspace(0.0, 1.0, 101),
            sigma_min=torch.tensor(0.0),
        )

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self.model_sampling


def test_scheduler_node_ids_are_prefixed_and_complete():
    assert {
        node.define_schema().node_id
        for node in scheduler_nodes.MIGRATED_SCHEDULER_NODES
    } == EXPECTED_NODE_IDS


def test_sigma_rescale_is_registered_with_exact_endpoint_controls():
    schemas = {
        node.define_schema().node_id: node.define_schema()
        for node in scheduler_nodes.SCHEDULER_NODES
    }
    schema = schemas["UC_SigmaRescale"]
    inputs = {value.id: value for value in schema.inputs}

    assert inputs["start_sigma"].default == 1.0
    assert inputs["end_sigma"].default == 0.0
    assert [output.id for output in schema.outputs] == ["sigmas"]


def test_sigma_rescale_preserves_shape_count_dtype_and_device():
    sigmas = torch.tensor([14.0, 10.5, 7.0, 3.5, 0.0], dtype=torch.float64)
    result = scheduler_nodes.UC_SigmaRescale.execute(
        sigmas,
        start_sigma=1.0,
        end_sigma=0.0,
    ).args[0]

    assert torch.equal(result, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float64))
    assert result.shape == sigmas.shape
    assert result.dtype == sigmas.dtype
    assert result.device == sigmas.device


def test_sigma_rescale_supports_reversed_endpoints_and_empty_schedules():
    reversed_sigmas = scheduler_helpers.rescale_sigmas(
        torch.tensor([2.0, 1.0, 0.0]),
        start=0.0,
        end=2.0,
    )
    assert torch.equal(reversed_sigmas, torch.tensor([0.0, 1.0, 2.0]))

    empty = torch.empty(0)
    result = scheduler_helpers.rescale_sigmas(empty, start=1.0, end=0.0)
    assert result.numel() == 0
    assert result is not empty


def test_sigma_rescale_rejects_constant_schedules():
    with pytest.raises(ValueError, match="constant sigma schedule"):
        scheduler_helpers.rescale_sigmas(
            torch.ones(4),
            start=1.0,
            end=0.0,
        )


def test_discard_penultimate_sigma_is_registered_and_preserves_terminal():
    schemas = {
        node.define_schema().node_id: node.define_schema()
        for node in scheduler_nodes.SCHEDULER_NODES
    }
    assert "UC_DiscardPenultimateSigma" in schemas

    sigmas = torch.tensor([1.0, 0.6, 0.2, 0.0])
    result = scheduler_nodes.UC_DiscardPenultimateSigma.execute(sigmas).args[0]
    assert torch.equal(result, torch.tensor([1.0, 0.6, 0.0]))
    assert result.dtype == sigmas.dtype
    assert result.device == sigmas.device


def test_scheduler_helpers_produce_cpu_sigmas_with_terminal_zero():
    model_sampling = _Model().model_sampling
    outputs = (
        scheduler_helpers.sigmoid_offset_scheduler(model_sampling, 8),
        scheduler_helpers.power_shift_scheduler(model_sampling, 8),
        scheduler_helpers.radiance_shift_scheduler(model_sampling, 8),
        scheduler_helpers.sigma_curve_scheduler(8),
        scheduler_helpers.sigma_curve_pchip_scheduler(8),
    )
    for sigmas in outputs:
        assert sigmas.device.type == "cpu"
        assert sigmas.dtype == torch.float32
        assert sigmas[-1].item() == 0.0


def test_scheduler_helpers_match_standalone_golden_outputs():
    model_sampling = _Model().model_sampling
    expected = {
        scheduler_helpers.sigmoid_offset_scheduler: [
            1.0,
            0.97,
            0.90,
            0.74,
            0.50,
            0.26,
            0.10,
            0.03,
            0.0,
        ],
        scheduler_helpers.power_shift_scheduler: [
            1.0,
            0.97,
            0.88,
            0.74,
            0.56,
            0.37,
            0.19,
            0.05,
            0.0,
        ],
        scheduler_helpers.radiance_shift_scheduler: [
            1.0,
            0.99,
            0.93,
            0.83,
            0.68,
            0.50,
            0.31,
            0.14,
            0.0,
        ],
    }
    for function, values in expected.items():
        assert torch.allclose(
            function(model_sampling, 8),
            torch.tensor(values),
            atol=1e-6,
        )

    assert torch.allclose(
        scheduler_helpers.sigma_curve_scheduler(5),
        torch.tensor([1.0, 0.9890625, 0.978125, 0.86328125, 0.421875, 0.0]),
        atol=1e-6,
    )
    assert torch.allclose(
        scheduler_helpers.sigma_curve_pchip_scheduler(5),
        torch.tensor([1.0, 0.9890625, 0.9787704, 0.8777022, 0.421875, 0.0]),
        atol=1e-6,
    )


def test_custom_sigma_points_and_dedicated_discard_are_preserved():
    result = scheduler_nodes.UC_SigmaCurveFromPointsScheduler.execute(
        steps=3,
        custom_points="1.0, 0.5, 0.25",
    ).args[0]
    assert torch.equal(result, torch.tensor([1.0, 0.5, 0.25, 0.0]))

    discarded = scheduler_helpers.discard_penultimate_sigma(result)
    assert torch.equal(discarded, torch.tensor([1.0, 0.5, 0.0]))


def test_point_schedulers_expose_the_original_default_curve():
    expected = ", ".join(str(value) for value in scheduler_helpers.BASE_SIGMA_POINTS)
    for node in (
        scheduler_nodes.UC_SigmaCurveFromPointsScheduler,
        scheduler_nodes.UC_SigmaCurvePchipScheduler,
    ):
        assert "model" not in {
            value.id
            for value in node.define_schema().inputs
        }
        custom_points = next(
            value
            for value in node.define_schema().inputs
            if value.id == "custom_points"
        )
        assert custom_points.default == expected


def test_scheduler_registration_is_idempotent_and_excludes_beta(monkeypatch):
    handlers = {}
    names = []
    monkeypatch.setattr(comfy.samplers, "SCHEDULER_HANDLERS", handlers)
    monkeypatch.setattr(comfy.samplers, "SCHEDULER_NAMES", names)

    scheduler_helpers.register_scheduler_handlers()
    scheduler_helpers.register_scheduler_handlers()

    assert set(handlers) == EXPECTED_HANDLER_NAMES
    assert set(names) == EXPECTED_HANDLER_NAMES
    assert len(names) == len(EXPECTED_HANDLER_NAMES)
    assert not any(name.startswith("beta_") for name in names)


def test_migrated_replacements_cover_every_old_node_id():
    replacements = {
        old_node_id: new_node_id
        for new_node_id, old_node_id, _ in node_replacements.REPLACEMENTS
        if old_node_id in OLD_TO_NEW
    }
    replacements.update(
        {
            old_node_id: new_node_id
            for (
                new_node_id,
                old_node_id,
                _,
                _,
            ) in node_replacements.MAPPED_REPLACEMENTS
        }
    )
    assert replacements == OLD_TO_NEW


def test_scheduler_replacement_widget_order_is_preserved():
    widgets = {
        old_node_id: old_widget_ids
        for _, old_node_id, old_widget_ids in node_replacements.REPLACEMENTS
    }
    widgets.update(
        {
            old_node_id: old_widget_ids
            for _, old_node_id, old_widget_ids, _ in node_replacements.MAPPED_REPLACEMENTS
        }
    )
    assert widgets["SigmoidOffsetScheduler"] == [
        "steps",
        "square_k",
        "base_c",
        "start_sigma",
    ]
    assert widgets["PowerShiftScheduler"] == [
        "steps",
        "power",
        "midpoint_shift",
        "discard_penultimate",
        "denoise",
    ]
    assert widgets["SigmaCurvePchipScheduler"] == [
        "steps",
        "discard_penultimate",
        "denoise",
        "custom_points",
    ]


def test_dedicated_schedulers_do_not_expose_postprocessing_controls():
    for node in (
        scheduler_nodes.UC_PowerShiftScheduler,
        scheduler_nodes.UC_RadianceShiftScheduler,
        scheduler_nodes.UC_SigmaCurveFromPointsScheduler,
        scheduler_nodes.UC_SigmaCurvePchipScheduler,
    ):
        schema = node.define_schema()
        input_ids = {value.id for value in schema.inputs}
        assert "denoise" not in input_ids
        assert "discard_penultimate" not in input_ids
        assert "Sigma Rescale" in schema.description
        assert "Sigma Rescale" in schema.outputs[0].tooltip

    sigmoid_inputs = {
        value.id
        for value in scheduler_nodes.UC_SigmoidOffsetScheduler.define_schema().inputs
    }
    assert "start_sigma" in sigmoid_inputs


def test_scheduler_replacements_drop_obsolete_postprocessing_inputs():
    for _, old_node_id, _, input_mapping in node_replacements.MAPPED_REPLACEMENTS:
        assert old_node_id in {
            "PowerShiftScheduler",
            "RadianceShiftScheduler",
            "SigmaCurveFromPointsScheduler",
            "SigmaCurvePchipScheduler",
        }
        mapped_ids = {
            mapping.get(key)
            for mapping in input_mapping
            for key in ("new_id", "old_id")
        }
        assert "denoise" not in mapped_ids
        assert "discard_penultimate" not in mapped_ids
        if old_node_id.startswith("SigmaCurve"):
            assert "model" not in mapped_ids


def test_radiance_node_compensates_before_fixed_penultimate_discard():
    model = _Model()
    result = scheduler_nodes.UC_RadianceShiftScheduler.execute(
        model,
        steps=8,
        power=2.4,
        midpoint_shift=0.98,
    ).args[0]
    expected = scheduler_helpers.radiance_shift_scheduler(
        model.model_sampling,
        steps=8,
        power=2.4,
        midpoint_shift=0.98,
        discard_penultimate=True,
    )
    assert torch.equal(result, expected)
    assert len(result) == 9


def test_sigmas_preview_is_registered_and_schema_is_complete():
    schemas = {
        node.define_schema().node_id: node.define_schema()
        for node in scheduler_nodes.SCHEDULER_NODES
    }
    assert "UC_SigmasPreview" in schemas
    schema = schemas["UC_SigmasPreview"]
    assert schema.node_id == "UC_SigmasPreview"
    assert schema.display_name == "Sigmas Preview"
    assert schema.category == "sampling/custom_sampling/schedulers"
    assert schema.is_output_node is True

    inputs = {value.id: value for value in schema.inputs}
    assert "sigmas" in inputs
    assert inputs["print_as_list"].default is False
    assert inputs["line_color"].default == "blue"
    for inp in schema.inputs:
        assert getattr(inp, "tooltip", None), f"Input {inp.id} missing tooltip"

    assert len(schema.outputs) == 1
    assert schema.outputs[0].id == "image"
    assert getattr(schema.outputs[0], "tooltip", None), "Output image missing tooltip"


def test_sigmas_preview_executes_and_returns_valid_image_and_ui():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float32)
    output = scheduler_nodes.UC_SigmasPreview.execute(
        sigmas,
        print_as_list=True,
        line_color="red",
    )
    assert len(output.args) == 1
    image_tensor = output.args[0]
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.ndim == 4
    assert image_tensor.shape[0] == 1
    assert image_tensor.shape[3] == 3
    assert image_tensor.dtype == torch.float32
    assert 0.0 <= image_tensor.min().item() <= 1.0
    assert 0.0 <= image_tensor.max().item() <= 1.0

    assert "images" in output.ui
    ui_images = output.ui["images"]
    assert len(ui_images) == 1
    assert "filename" in ui_images[0]
    assert ui_images[0]["type"] == "temp"


def test_sigmas_preview_breakdown_handles_edge_cases():
    empty_breakdown = scheduler_helpers.format_sigmas_breakdown(torch.tensor([]), print_to_console=False)
    assert "EMPTY" in empty_breakdown

    constant_sigmas = torch.tensor([1.0, 1.0, 1.0])
    constant_breakdown = scheduler_helpers.format_sigmas_breakdown(constant_sigmas, print_to_console=False)
    assert "Step" in constant_breakdown
    assert "Normalized" in constant_breakdown
