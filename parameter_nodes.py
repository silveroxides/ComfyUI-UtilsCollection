import math

import torch

from comfy_api.latest import ComfyExtension, io
import comfy.model_management as mm
import nodes

from .helper_functions import round_to_nearest, AspectRatio, ASPECT_RATIOS, resize_nchw
from .parameter_helpers import h3_video_length_from_seconds, select_video_resolution


class UC_AdjustedResolutionParameters(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_AdjustedResolutionParameters",
            display_name="Adjusted Resolution",
            category="utils",
            inputs=[
                io.Int.Input(
                    id="width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(
                    id="height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16
                ),
                io.Int.Input(id="batch_size", default=1, min=1, max=4096),
                io.Float.Input(
                    id="scale_by",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Scale used for the larger output resolution.",
                ),
                io.Int.Input(
                    id="multiple",
                    default=16,
                    min=4,
                    max=128,
                    step=4,
                    tooltip="Round the larger resolution to this pixel multiple.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="Width"),
                io.Int.Output(display_name="Height"),
                io.Int.Output(display_name="Upscaled Width"),
                io.Int.Output(display_name="Upscaled Height"),
                io.Int.Output(display_name="Batch"),
            ],
        )

    @classmethod
    def execute(cls, width: int, height: int, batch_size: int, scale_by: float, multiple: int) -> io.NodeOutput:
        adjusted_width = round_to_nearest(width, multiple)
        adjusted_height = round_to_nearest(height, multiple)
        upscaled_width = round_to_nearest(width * scale_by, multiple)
        upscaled_height = round_to_nearest(height * scale_by, multiple)
        return io.NodeOutput(
            adjusted_width,
            adjusted_height,
            upscaled_width,
            upscaled_height,
            batch_size,
        )


class UC_ResolutionSelectorExtended(io.ComfyNode):
    """Calculate width and height from aspect ratio and megapixel target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ResolutionSelectorExtended",
            display_name="Resolution Selector Extended",
            category="utilities",
            description="Calculate width and height from aspect ratio and megapixel target. Useful for setting up Empty Latent Image dimensions.",
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.SQUARE,
                    tooltip="The aspect ratio for the output dimensions.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Target total megapixels. 1.0 MP ≈ 1024×1024 for square.",
                ),
                io.Int.Input(
                    id="multiple",
                    default=8,
                    min=8,
                    max=128,
                    step=4,
                    tooltip="Round the selected resolution to this pixel multiple.",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "width", tooltip="Calculated width in pixels multiplied by the selected multiple."
                ),
                io.Int.Output(
                    "height", tooltip="Calculated height in pixels multiplied by the selected multiple."
                ),
            ],
        )

    @classmethod
    def execute(cls, aspect_ratio: str, megapixels: float, multiple: int, minimum: int = 256) -> io.NodeOutput:
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / multiple) * multiple
        height = round(h_ratio * scale / multiple) * multiple
        if width < minimum or height < minimum:
            step_w = multiple // math.gcd(w_ratio, multiple)
            step_h = multiple // math.gcd(h_ratio, multiple)
            k_step = step_w * step_h // math.gcd(step_w, step_h)
            min_k = math.ceil(max(minimum / w_ratio, minimum / h_ratio))
            k = math.ceil(min_k / k_step) * k_step
            width = w_ratio * k
            height = h_ratio * k
        return io.NodeOutput(width, height, ui={"resolution": (f"{width}×{height}",)})


class UC_VideoResolutionSelector(io.ComfyNode):
    """Select a multiple-aligned video resolution near a nominal megapixel target."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_VideoResolutionSelector",
            display_name="Video Resolution Selector",
            category="utilities",
            description=(
                "Select a video resolution near the target megapixels while keeping "
                "the chosen aspect ratio within the selected tolerance."
            ),
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.WIDESCREEN_H,
                    tooltip="The preferred output aspect ratio.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Nominal target total megapixels used to choose the nearest viable resolution.",
                ),
                io.Int.Input(
                    "multiple",
                    default=32,
                    min=8,
                    max=128,
                    step=4,
                    tooltip="Required pixel multiple for both output dimensions.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=0.0,
                    step=0.1,
                    tooltip="Video duration in seconds at 24 fps. Rounds up to a supported H3 length; 5 seconds gives 124 frames. The minimum is 5 frames.",
                ),
            ],
            outputs=[
                io.Int.Output("width", tooltip="Selected video width in pixels."),
                io.Int.Output("height", tooltip="Selected video height in pixels."),
                io.Int.Output("length", tooltip="H3 generation length in frames at 24 fps, calculated from duration_seconds."),
            ],
        )

    @classmethod
    def execute(
        cls,
        aspect_ratio: str,
        megapixels: float,
        multiple: int,
        minimum: int = 256,
        duration_seconds: float = 5.0,
    ) -> io.NodeOutput:
        ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
        width, height = select_video_resolution(
            ratio_width,
            ratio_height,
            megapixels,
            multiple,
            minimum,
            nodes.MAX_RESOLUTION,
        )
        length = h3_video_length_from_seconds(duration_seconds)
        return io.NodeOutput(width, height, length, ui={"resolution": (f"{width}×{height}",)})


class UC_ImageScaleAndResolutionPicker(io.ComfyNode):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageScaleAndResolutionPicker",
            display_name="Image Scale & Resolution",
            category="utils",
            inputs=[
                io.Image.Input("image", optional=True),
                io.Combo.Input("upscale_method", options=cls.upscale_methods, default="lanczos"),
                io.Combo.Input("crop_method", options=cls.crop_methods, tooltip="If cropping is enabled, the image will be cropped to the target aspect ratio before resizing. Center cropping is used, so the center of the image will be preserved and equal amounts will be cropped from either side."),
                io.Combo.Input(
                    "aspect_ratio",
                    options=AspectRatio,
                    default=AspectRatio.SQUARE,
                    tooltip="The aspect ratio for the output dimensions and cropping.",
                ),
                io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                io.Int.Input(
                    "resolution_steps",
                    default=1,
                    min=1,
                    max=256,
                    advanced=True,
                    tooltip="Does not affect output size. Use Multiple to align dimensions.",
                ),
                io.Float.Input(
                    id="scale_by",
                    default=1.0,
                    min=0.01,
                    max=10.0,
                    step=0.01,
                    tooltip="Scale used for the larger output resolution.",
                ),
                io.Int.Input(
                    id="multiple",
                    default=16,
                    min=4,
                    max=128,
                    step=4,
                    tooltip="Round the larger resolution to this pixel multiple.",
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="The adjusted/cropped base image"),
                io.Image.Output("upscaled_image", display_name="Image", tooltip="The image after applying the scale_by factor"),
                io.Int.Output(display_name="Width"),
                io.Int.Output(display_name="Height"),
                io.Int.Output(display_name="Upscaled Width"),
                io.Int.Output(display_name="Upscaled Height"),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method: str, crop_method: str, aspect_ratio: AspectRatio, megapixels: float, resolution_steps: int, scale_by: float, multiple: int) -> io.NodeOutput:
        total = megapixels * 1024 * 1024

        # Retained in the schema so existing serialized workflows keep their
        # widget layout. The final resolution has one source of alignment:
        # ``multiple``.
        _ = resolution_steps

        if image is not None:
            # B, H, W, C
            samples = image.movedim(-1, 1) # B, C, H, W
            img_h, img_w = samples.shape[2], samples.shape[3]

            if crop_method == "center":
                target_ratio_w, target_ratio_h = ASPECT_RATIOS[aspect_ratio]
                base_scale = math.sqrt(total / (target_ratio_w * target_ratio_h))
                width = target_ratio_w * base_scale
                height = target_ratio_h * base_scale
            else:
                megapixel_scale = math.sqrt(total / (img_w * img_h))
                width = img_w * megapixel_scale
                height = img_h * megapixel_scale
        else:
            target_ratio_w, target_ratio_h = ASPECT_RATIOS[aspect_ratio]
            base_scale = math.sqrt(total / (target_ratio_w * target_ratio_h))
            width = target_ratio_w * base_scale
            height = target_ratio_h * base_scale

            adjusted_width = max(multiple, round_to_nearest(width, multiple))
            adjusted_height = max(multiple, round_to_nearest(height, multiple))

            device = mm.intermediate_device()
            dtype = mm.intermediate_dtype()
            samples = torch.zeros(
                [1, 3, adjusted_height, adjusted_width], dtype=dtype, device=device
            )  # B, C, H, W

        adjusted_width = max(multiple, round_to_nearest(width, multiple))
        adjusted_height = max(multiple, round_to_nearest(height, multiple))
        upscaled_width = max(multiple, round_to_nearest(adjusted_width * scale_by, multiple))
        upscaled_height = max(multiple, round_to_nearest(adjusted_height * scale_by, multiple))

        adjusted_samples = resize_nchw(samples, int(adjusted_width), int(adjusted_height), upscale_method, crop_method)
        upscaled_samples = resize_nchw(adjusted_samples, int(upscaled_width), int(upscaled_height), upscale_method)

        adjusted_image_out = adjusted_samples.movedim(1, -1)
        upscaled_image_out = upscaled_samples.movedim(1, -1)

        return io.NodeOutput(
            adjusted_image_out,
            upscaled_image_out,
            adjusted_width,
            adjusted_height,
            upscaled_width,
            upscaled_height,
        )
