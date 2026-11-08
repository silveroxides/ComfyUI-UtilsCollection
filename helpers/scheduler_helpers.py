from io import BytesIO
from typing import Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import torch
from scipy.interpolate import PchipInterpolator

from comfy_api.latest import io, ui
import comfy.samplers


BASE_SIGMA_POINTS = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
]


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def rescale_sigmas(sigmas: torch.Tensor, start: float, end: float):
    if sigmas.numel() == 0:
        return sigmas.clone()

    sigma_min = sigmas.min()
    sigma_max = sigmas.max()
    if sigma_max == sigma_min:
        raise ValueError("Cannot rescale a constant sigma schedule.")

    return (
        (sigmas - sigma_min) * (start - end) / (sigma_max - sigma_min)
        + end
    )


def discard_penultimate_sigma(sigmas: torch.Tensor):
    return torch.cat((sigmas[:-2], sigmas[-1:]))


def _sigmoid_fn(
    x_norm,
    k=1.0,
    shift=0.0,
    range_start=-4.0,
    range_end=4.0,
):
    x_mapped = (range_end - range_start) * x_norm + range_start
    shift_param_scaled = shift * 4.0
    val_for_exp = -k * (x_mapped + shift_param_scaled)
    k_x_clipped = numpy.clip(val_for_exp, -700, 700)
    return 1.0 / (1.0 + numpy.exp(k_x_clipped))


def sigmoid_offset_scheduler(
    model_sampling,
    steps: int,
    square_k: float = 1.0,
    base_c: float = 0.5,
):
    total_timesteps = len(model_sampling.sigmas) - 1
    x_norm_values = numpy.linspace(0, 1, steps + 1, endpoint=True)
    sigmoid_shift_for_fn = 2.0 * (base_c - 0.5)
    raw_sigmoid = _sigmoid_fn(
        x_norm_values,
        k=square_k,
        shift=sigmoid_shift_for_fn,
    )
    sig_min = _sigmoid_fn(
        0.0,
        k=square_k,
        shift=sigmoid_shift_for_fn,
    )
    sig_max = _sigmoid_fn(
        1.0,
        k=square_k,
        shift=sigmoid_shift_for_fn,
    )
    normalized_sigmoid = (raw_sigmoid - sig_min) / (sig_max - sig_min)
    transformed_ts_values = 1.0 - normalized_sigmoid
    ts = numpy.rint(transformed_ts_values * total_timesteps).astype(int)
    ts = numpy.clip(ts, 0, total_timesteps)

    sigmas = []
    last_t = -1
    for timestep in ts:
        if timestep != last_t or not sigmas:
            sigmas.append(float(model_sampling.sigmas[timestep].item()))
            last_t = timestep

    sigma_floor = float(model_sampling.sigma_min)
    if sigmas[-1] <= sigma_floor:
        sigmas[-1] = 0.0
    else:
        sigmas.append(0.0)
    return torch.FloatTensor(sigmas)


def power_shift_scheduler(
    model_sampling,
    steps,
    power=2.0,
    midpoint_shift=1.0,
    discard_penultimate=False,
):
    total_timesteps = len(model_sampling.sigmas) - 1
    x = numpy.linspace(0, 1, steps, endpoint=False)
    x = x**midpoint_shift
    ts_normalized = (1 - x**power) ** power
    timesteps = numpy.rint(ts_normalized * total_timesteps)

    sigmas = []
    last_t = -1
    for timestep in timesteps:
        timestep = min(int(timestep), total_timesteps)
        if timestep != last_t:
            sigmas.append(float(model_sampling.sigmas[timestep]))
        last_t = timestep

    sigmas.append(0.0)
    result = torch.FloatTensor(sigmas)
    if discard_penultimate:
        return discard_penultimate_sigma(result)
    return result


def radiance_shift_scheduler(
    model_sampling,
    steps,
    power=2.4,
    midpoint_shift=0.98,
    discard_penultimate=True,
):
    total_timesteps = len(model_sampling.sigmas) - 1
    real_steps = steps + 1
    x = numpy.linspace(0, 1, real_steps, endpoint=False)
    x = x**midpoint_shift
    ts_normalized = (1 - x**power) ** power
    timesteps = numpy.rint(ts_normalized * total_timesteps)

    sigmas = []
    last_t = -1
    for timestep in timesteps:
        timestep = min(int(timestep), total_timesteps)
        if timestep != last_t:
            sigmas.append(float(model_sampling.sigmas[timestep]))
        last_t = timestep

    sigmas.append(0.0)
    result = torch.FloatTensor(sigmas)
    if discard_penultimate:
        return discard_penultimate_sigma(result)
    return result


def sigma_curve_scheduler(
    steps,
    discard_penultimate=False,
    sigma_points=BASE_SIGMA_POINTS,
):
    base = numpy.array(sigma_points, dtype=numpy.float32)
    x_base = numpy.linspace(0.0, float(len(base) - 1), len(base))
    x_new = numpy.linspace(0.0, float(len(base) - 1), steps)
    curve_sigmas = numpy.interp(x_new, x_base, base).astype(numpy.float32)
    sigmas = numpy.concatenate(
        [curve_sigmas, numpy.array([0.0], dtype=numpy.float32)]
    )
    result = torch.from_numpy(sigmas)
    if discard_penultimate:
        return discard_penultimate_sigma(result)
    return result


def sigma_curve_pchip_scheduler(
    steps,
    discard_penultimate=False,
    sigma_points=BASE_SIGMA_POINTS,
):
    base = numpy.array(sigma_points, dtype=numpy.float32)
    x_base = numpy.linspace(0.0, float(len(base) - 1), len(base))
    x_new = numpy.linspace(0.0, float(len(base) - 1), steps)
    curve_sigmas = PchipInterpolator(x_base, base)(x_new).astype(numpy.float32)
    sigmas = numpy.concatenate(
        [curve_sigmas, numpy.array([0.0], dtype=numpy.float32)]
    )
    result = torch.from_numpy(sigmas)
    if discard_penultimate:
        return discard_penultimate_sigma(result)
    return result


def dmad_scheduler(
    model_sampling: Any = None,
    steps: int = 4,
    shift: Optional[float] = None,
) -> torch.Tensor:
    """Shifted linear grid matching DMAD / MiniMax H3 student training.

    Generates steps + 1 sigma values from 1.0 down to 0.0 with time shift applied.
    Defaults to model_sampling.shift (or 12.0 for H3 video).
    """
    if shift is None:
        shift = getattr(model_sampling, "shift", 12.0) if model_sampling is not None else 12.0
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    s = float(shift)
    return s * t / (1.0 + (s - 1.0) * t)


SCHEDULER_HANDLERS = {
    "dmad": dmad_scheduler,
    "sigmoid_offset": sigmoid_offset_scheduler,
    "power_shift": power_shift_scheduler,
    "radiance_shift": radiance_shift_scheduler,
    "sigma_curve_from_points": lambda _model_sampling, steps: sigma_curve_scheduler(
        steps
    ),
    "sigma_curve_pchip": lambda _model_sampling, steps: sigma_curve_pchip_scheduler(
        steps
    ),
}


def register_scheduler_handlers():
    for name, function in SCHEDULER_HANDLERS.items():
        if name not in comfy.samplers.SCHEDULER_HANDLERS:
            comfy.samplers.SCHEDULER_HANDLERS[name] = comfy.samplers.SchedulerHandler(
                handler=function,
                use_ms=True,
            )
        if name not in comfy.samplers.SCHEDULER_NAMES:
            comfy.samplers.SCHEDULER_NAMES.append(name)


def format_sigmas_breakdown(sigmas: torch.Tensor, print_to_console: bool = False) -> str:
    """Format a detailed tabular breakdown of steps, sigma values, normalized progress, and step sizes."""
    sigmas_1d = sigmas.detach().cpu().to(torch.float32).reshape(-1)
    sigmas_list = [round(float(s), 4) for s in sigmas_1d.tolist()]

    if len(sigmas_list) == 0:
        lines = [
            "=" * 60,
            "SIGMAS PREVIEW - PRINT LIST (EMPTY)",
            "=" * 60,
        ]
        text = "\n".join(lines)
        if print_to_console:
            print("\n" + text + "\n")
        return text

    min_s = min(sigmas_list)
    max_s = max(sigmas_list)
    span = max_s - min_s
    if span > 0:
        percentages = [round(float((s - min_s) / span), 4) for s in sigmas_list]
    else:
        percentages = [0.0] * len(sigmas_list)

    lines = [
        "=" * 60,
        "SIGMAS PREVIEW - PRINT LIST",
        "=" * 60,
        f"Total sigma values: {len(sigmas_list)}",
        f"Min sigma:   {min_s:.4f}",
        f"Max sigma:   {max_s:.4f}",
        "",
        "Normalized percentages (0.0-1.0):",
        "-" * 40,
        "Step | Sigma    | Normalized | Step Size",
        "-----|----------|------------|----------",
    ]
    for i, (sigma, pct) in enumerate(zip(sigmas_list, percentages)):
        if i > 0:
            step_size = f"{sigmas_list[i-1] - sigma:8.4f}"
        else:
            step_size = f"{'--':>8}"
        lines.append(f"{i:4d} | {sigma:8.4f} | {pct:10.4f} | {step_size}")
    lines.append("=" * 60)

    text = "\n".join(lines)
    if print_to_console:
        print("\n" + text + "\n")
    return text


def render_sigmas_graph(sigmas: torch.Tensor, line_color: str = "blue") -> torch.Tensor:
    """Render a 2D plot of sigma values against step index and return a normalized float32 [1, H, W, 3] tensor."""
    sigmas_np = sigmas.detach().cpu().to(torch.float32).reshape(-1).numpy()

    fig, ax = plt.subplots()
    ax.plot(sigmas_np, marker="o", linestyle="-", color=line_color)
    ax.set_title("Graph from Tensor")
    ax.set_xlabel("Step Number")
    ax.set_ylabel("Sigma Value")

    with BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf).convert("RGB").copy()
    plt.close(fig)

    numpy_image = numpy.array(image, dtype=numpy.float32) / 255.0
    return torch.from_numpy(numpy_image).unsqueeze(0)


def save_sigmas_preview_image(
    images: torch.Tensor,
    filename_prefix: str = "SigmasPreview",
) -> list[dict[str, Any]]:
    """Save an image batch to ComfyUI's temporary directory for frontend display."""
    saved = ui.ImageSaveHelper.save_images(
        images,
        filename_prefix=filename_prefix,
        folder_type=io.FolderType.temp,
        cls=None,
        compress_level=4,
    )
    return [dict(item) for item in saved]


def preview_sigmas(
    sigmas: torch.Tensor,
    print_as_list: bool = False,
    line_color: str = "blue",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Inspect and render a sigma schedule, returning the output image tensor and frontend UI descriptors."""
    if print_as_list:
        format_sigmas_breakdown(sigmas, print_to_console=True)

    images_tensor = render_sigmas_graph(sigmas, line_color=line_color)
    ui_images = save_sigmas_preview_image(images_tensor, filename_prefix="SigmasPreview")
    return images_tensor, ui_images
