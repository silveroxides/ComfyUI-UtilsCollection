from enum import Enum
from comfy_api.latest import io
from comfy_extras.nodes_ideogram4 import Ideogram4Scheduler

from ..helpers.scheduler_helpers import BASE_SIGMA_POINTS, discard_penultimate_sigma, dmad_scheduler, parse_float_list, power_shift_scheduler, preview_sigmas, radiance_shift_scheduler, rescale_sigmas, sigma_curve_pchip_scheduler, sigma_curve_scheduler, sigmoid_offset_scheduler

SIGMA_RESCALE_I2I_RECOMMENDATION = "For I2I, connect Sigma Rescale."
SIGMA_DISCARD_RECOMMENDATION = "If required, connect Discard Penultimate Sigma."
DEFAULT_SIGMA_POINTS_TEXT = ", ".join(str(value) for value in BASE_SIGMA_POINTS)


class Ideogram4Enum(Enum):
    QUALITY = "Quality"
    HIGH = "High"
    DEFAULT = "Default"
    FAST = "Fast"
    TURBO = "Turbo"


IDEOGRAM4_PRESET_CONFIGS = {
    Ideogram4Enum.QUALITY.value: {"num_steps": 48, "mu": 0.0, "std": 1.5, "preset_id": "V4_QUALITY_48"},
    Ideogram4Enum.HIGH.value: {"num_steps": 34, "mu": 0.0, "std": 1.6875, "preset_id": "V4_HIGH_34"},
    Ideogram4Enum.DEFAULT.value: {"num_steps": 20, "mu": 0.0, "std": 1.75, "preset_id": "V4_DEFAULT_20"},
    Ideogram4Enum.FAST.value: {"num_steps": 16, "mu": 0.25, "std": 1.8375, "preset_id": "V4_FAST_16"},
    Ideogram4Enum.TURBO.value: {"num_steps": 12, "mu": 0.5, "std": 1.75, "preset_id": "V4_TURBO_12"},
}


class Ideogram4SchedulerPreset(Ideogram4Scheduler):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Ideogram4SchedulerPreset",
            display_name="Ideogram 4 Scheduler (Presets)",
            category="sampling/custom_sampling/schedulers",
            description="Schedule Presets for Ideogram 4. They are as follows: Quality=48, High=34, Default=20, Fast=16, Turbo=12",
            inputs=[io.Combo.Input("preset", display_name="ideogram4_scheduler_preset", options=[e.value for e in Ideogram4Enum], default=Ideogram4Enum.DEFAULT.value), io.Int.Input("width", default=1024, min=256, max=8192, step=16), io.Int.Input("height", default=1024, min=256, max=8192, step=16)],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, preset, width, height) -> io.NodeOutput:
        config = IDEOGRAM4_PRESET_CONFIGS.get(preset)
        if not config:
            raise ValueError(f"Invalid preset: {preset}")

        return super().execute(steps=config["num_steps"], width=width, height=height, mu=config["mu"], std=config["std"])


class UC_SigmaRescale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_SigmaRescale",
            display_name="Sigma Rescale",
            category="sampling/custom_sampling/schedulers",
            description=("Rescales a sigma schedule to exact start and end noise levels without changing its shape or step count."),
            inputs=[io.Sigmas.Input("sigmas"), io.Float.Input("start_sigma", default=1.0, min=0.0, max=5000.0, step=0.01, round=False), io.Float.Input("end_sigma", default=0.0, min=0.0, max=5000.0, step=0.01, round=False)],
            outputs=[io.Sigmas.Output("sigmas", display_name="Sigmas")],
        )

    @classmethod
    def execute(cls, sigmas, start_sigma, end_sigma):
        return io.NodeOutput(rescale_sigmas(sigmas, start=start_sigma, end=end_sigma))


class UC_DiscardPenultimateSigma(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_DiscardPenultimateSigma",
            display_name="Discard Penultimate Sigma",
            category="sampling/custom_sampling/schedulers",
            description=("Removes the second-to-last sigma while preserving the terminal sigma."),
            inputs=[io.Sigmas.Input("sigmas")],
            outputs=[io.Sigmas.Output("sigmas", display_name="Sigmas")],
        )

    @classmethod
    def execute(cls, sigmas):
        return io.NodeOutput(discard_penultimate_sigma(sigmas))


class UC_SigmoidOffsetScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_SigmoidOffsetScheduler",
            display_name="Sigmoid Offset Scheduler",
            category="sampling/custom_sampling/schedulers",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=30, min=1, max=10000),
                io.Float.Input("square_k", default=1.0, min=0.0, max=10.0, step=0.01, tooltip="Higher values make the denoising transition sharper."),
                io.Float.Input("base_c", default=0.5, min=-5.0, max=5.0, step=0.01, tooltip="Move more denoising earlier or later."),
                io.Float.Input("start_sigma", default=1.0, min=0.0, max=1.0, step=0.001, tooltip="Adjust initial denoising strength. 1 leaves it unchanged."),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, model, steps, square_k, base_c, start_sigma):
        sigmas = sigmoid_offset_scheduler(model.get_model_object("model_sampling"), steps, square_k=square_k, base_c=base_c)
        if start_sigma != 1.0:
            sigma_min = sigmas.min()
            sigma_max = sigmas.max()
            sigmas = (sigmas - sigma_min) * start_sigma / (sigma_max - sigma_min)
        return io.NodeOutput(sigmas)

    get_sigmas = execute


class UC_PowerShiftScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_PowerShiftScheduler",
            display_name="Power Shift Scheduler",
            category="sampling/custom_sampling/schedulers",
            description=(f"Builds a power-shaped sigma schedule. {SIGMA_RESCALE_I2I_RECOMMENDATION} {SIGMA_DISCARD_RECOMMENDATION}"),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=20, min=3, max=1000),
                io.Float.Input("power", default=2.0, min=0.0, max=5.0, step=0.001, tooltip="Changes how denoising steps are spread. Higher values make the change stronger."),
                io.Float.Input("midpoint_shift", default=1.0, min=0.0, max=5.0, step=0.001, tooltip="Move more denoising toward the start or end of sampling."),
            ],
            outputs=[io.Sigmas.Output(tooltip=(f"{SIGMA_RESCALE_I2I_RECOMMENDATION} {SIGMA_DISCARD_RECOMMENDATION}"))],
        )

    @classmethod
    def execute(cls, model, steps, power, midpoint_shift):
        sigmas = power_shift_scheduler(model.get_model_object("model_sampling"), steps, power, midpoint_shift, discard_penultimate=False).cpu()
        return io.NodeOutput(sigmas)

    get_sigmas = execute


class UC_RadianceShiftScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_RadianceShiftScheduler",
            display_name="Radiance Shift Scheduler",
            category="sampling/custom_sampling/schedulers",
            description=(f"Builds the Radiance power-shift schedule with its compensated step count and penultimate sigma removal. {SIGMA_RESCALE_I2I_RECOMMENDATION}"),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("steps", default=20, min=3, max=1000),
                io.Float.Input("power", default=2.4, min=0.0, max=5.0, step=0.001, tooltip="Changes how Radiance denoising steps are spread. Higher values make the change stronger."),
                io.Float.Input("midpoint_shift", default=0.98, min=0.0, max=5.0, step=0.001, tooltip="Move more Radiance denoising toward the start or end of sampling."),
            ],
            outputs=[io.Sigmas.Output(tooltip=SIGMA_RESCALE_I2I_RECOMMENDATION)],
        )

    @classmethod
    def execute(cls, model, steps, power, midpoint_shift):
        sigmas = radiance_shift_scheduler(model.get_model_object("model_sampling"), steps, power, midpoint_shift, discard_penultimate=True).cpu()
        return io.NodeOutput(sigmas)

    get_sigmas = execute


class UC_SigmaCurveFromPointsScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_SigmaCurveFromPointsScheduler",
            display_name="From Points Scheduler",
            category="sampling/custom_sampling/schedulers",
            description=(f"Interpolates a sigma schedule from configurable points. {SIGMA_RESCALE_I2I_RECOMMENDATION} {SIGMA_DISCARD_RECOMMENDATION}"),
            inputs=[io.Int.Input("steps", default=8, min=1, max=1000), io.String.Input("custom_points", default=DEFAULT_SIGMA_POINTS_TEXT, multiline=False, optional=True, tooltip="Comma-separated points that shape the denoising schedule.")],
            outputs=[io.Sigmas.Output(tooltip=(f"{SIGMA_RESCALE_I2I_RECOMMENDATION} {SIGMA_DISCARD_RECOMMENDATION}"))],
        )

    @classmethod
    def execute(cls, steps, custom_points=None):
        sigma_points = parse_float_list(custom_points) if custom_points is not None else BASE_SIGMA_POINTS
        if len(sigma_points) < 2:
            sigma_points = BASE_SIGMA_POINTS
        sigmas = sigma_curve_scheduler(steps, discard_penultimate=False, sigma_points=sigma_points).cpu()
        return io.NodeOutput(sigmas[-(steps + 1) :])

    get_sigmas = execute


class UC_SigmaCurvePchipScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_SigmaCurvePchipScheduler",
            display_name="PCHIP Scheduler",
            category="sampling/custom_sampling/schedulers",
            description=(f"Interpolates a monotonic PCHIP sigma schedule from configurable points. {SIGMA_RESCALE_I2I_RECOMMENDATION} {SIGMA_DISCARD_RECOMMENDATION}"),
            inputs=[io.Int.Input("steps", default=8, min=1, max=2000), io.String.Input("custom_points", default=DEFAULT_SIGMA_POINTS_TEXT, multiline=False, optional=True, tooltip="Comma-separated points that shape the denoising schedule.")],
            outputs=[io.Sigmas.Output(tooltip=(f"{SIGMA_RESCALE_I2I_RECOMMENDATION} {SIGMA_DISCARD_RECOMMENDATION}"))],
        )

    @classmethod
    def execute(cls, steps, custom_points=None):
        sigma_points = parse_float_list(custom_points) if custom_points is not None else BASE_SIGMA_POINTS
        if len(sigma_points) < 2:
            sigma_points = BASE_SIGMA_POINTS
        sigmas = sigma_curve_pchip_scheduler(steps, discard_penultimate=False, sigma_points=sigma_points).cpu()
        return io.NodeOutput(sigmas[-(steps + 1) :])

    get_sigmas = execute


class UC_DMADSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_DMADSchedule",
            display_name="DMAD Schedule",
            category="sampling/custom_sampling/schedulers",
            description="Shifted linear sigma schedule for DMAD/DMD2 distilled models (e.g. 4-step MiniMax H3).",
            inputs=[io.Int.Input("steps", default=4, min=1, max=1000, tooltip="Number of model evaluations (4 for DMAD student)."), io.Float.Input("shift", default=12.0, min=0.0, max=100.0, step=0.1, tooltip="Time shift for the sigma schedule (12.0 for H3 video).")],
            outputs=[io.Sigmas.Output("sigmas", display_name="Sigmas")],
        )

    @classmethod
    def execute(cls, steps: int = 4, shift: float = 12.0) -> io.NodeOutput:
        sigmas = dmad_scheduler(None, steps=steps, shift=shift).cpu()
        return io.NodeOutput(sigmas)

    get_sigmas = execute


class UC_SigmasPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SigmasPreview",
            display_name="Sigmas Preview",
            category="sampling/custom_sampling/schedulers",
            description=("Renders a graph preview of a sigma schedule and optionally prints a step-by-step table breakdown."),
            is_output_node=True,
            inputs=[
                io.Sigmas.Input("sigmas", tooltip="Sigma schedule to plot and preview."),
                io.Boolean.Input("print_as_list", default=False, tooltip="When true, prints step numbers, sigma values, normalized progress, and step sizes to console."),
                io.String.Input("line_color", default="blue", tooltip="Color of plotted curve and step points on graph (e.g. 'blue', 'red', '#1f77b4')."),
            ],
            outputs=[io.Image.Output("image", display_name="Image", tooltip="Plotted sigma curve preview image as a standard image tensor.")],
        )

    @classmethod
    def execute(cls, sigmas, print_as_list: bool = False, line_color: str = "blue") -> io.NodeOutput:
        images_tensor, ui_images = preview_sigmas(sigmas, print_as_list=print_as_list, line_color=line_color)
        return io.NodeOutput(images_tensor, ui={"images": ui_images})


MIGRATED_SCHEDULER_NODES = [UC_SigmoidOffsetScheduler, UC_PowerShiftScheduler, UC_RadianceShiftScheduler, UC_SigmaCurveFromPointsScheduler, UC_SigmaCurvePchipScheduler]

SCHEDULER_NODES = [Ideogram4SchedulerPreset, UC_SigmaRescale, UC_DiscardPenultimateSigma, UC_DMADSchedule, UC_SigmasPreview, *MIGRATED_SCHEDULER_NODES]
