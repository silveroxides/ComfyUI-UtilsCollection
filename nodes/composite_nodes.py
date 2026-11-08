import logging
import math

import numpy as np
import torch

from comfy_api.latest import io
from nodes import MAX_RESOLUTION
from ..helpers.staged_face_helpers import _stage_face_foregrounds, face_removal_with_alpha, load_face_model
from ..helpers.composite_helpers import (
    _COMPOSITE_RESIZE_METHODS,
    _DEFAULT_LAYER_PLACEMENT,
    _DEFAULT_LAYER_PLACEMENT_V2,
    _RESIZE_METHODS,
    _STAGED_BACKGROUND_OPTION_DEFAULTS,
    _blur_mask,
    _broadcast_batch,
    _crop_bounds,
    _expand_mask,
    _feather_mask,
    _flatten_autogrow_images,
    _load_internal_background_removal_model,
    _ordered_layer_keys,
    _ordered_single_foregrounds,
    _parse_layer_payload,
    _parse_layer_placements,
    _placement_offsets,
    _refine_foreground_mask,
    resolve_background_removal_model,
    _resize_composite_image,
    _resize_composite_mask,
    _resize_image,
    _resize_mask,
    _save_editor_preview,
    _visible_placement_slices,
    background_removal_with_alpha,
    build_moss_image_layout_prompt,
    crop_staged_layers_by_indices,
    execute_create_bounding_boxes,
)
from ..helpers.background_replace_helpers import _apply_face_mask_options, _detect_largest_face_adaptive, _expanded_box, _largest_face, _ordered_ring, _polygon_mask, _similarity_transform, _transform_source, _warp_target, _warp_source
from ..helpers.staged_compositor_helpers import RetainedStageCache, _apply_staged_layer_options, _composite_staged_individual_foregrounds, _composite_staged_foregrounds, _preview_staged_foregrounds, _stage_layered_foregrounds, resolve_retained_stage, staged_foreground_fingerprint


FaceDetectionType = io.Custom("FACE_DETECTION_MODEL")
FaceCompositeOptionsType = io.Custom("UC_FACE_COMPOSITE_OPTIONS")
StagedBackgroundOptionsType = io.Custom("UC_STAGED_LAYERED_BACKGROUND_OPTIONS")
StagedFaceOptionsType = io.Custom("UC_STAGED_MEDIAPIPE_FACE_OPTIONS")

_MASK_THRESHOLD_TOOLTIP = "Minimum background-removal confidence retained as foreground before cleanup."
_BORDER_CLEANUP_TOOLTIP = "Source-edge strip width in pixels where weak foreground predictions are removed; 0 disables it."
_ARTIFACT_CLEANUP_TOOLTIP = "Opening radius in pixels used to remove small or thin mask artifacts; 0 disables it."
_GAP_FILL_TOOLTIP = "Closing radius in pixels used to fill small mask cracks and holes; 0 disables it."
_FEATHER_TOOLTIP = "Inward mask-edge softness in pixels; 0 keeps the resized edge unchanged."
_IMAGE_RESIZE_TOOLTIP = "Foreground resampling method. auto uses FP32 area reduction when shrinking and bicubic when enlarging; choose another method to override it."
_MASK_RESIZE_TOOLTIP = "Mask resampling method. auto uses area when shrinking and bilinear when enlarging while preserving soft coverage; nearest-exact produces a hard binary edge."
_MISSING = object()


class UC_CropByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_CropByMask",
            display_name="Crop By Mask",
            category="utils/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input("padding", default=64, min=0, max=MAX_RESOLUTION, step=8, tooltip="Pixels added around the combined nonzero mask bounds before alignment to the selected multiple."),
                io.Int.Input("multiple", default=8, min=4, max=256, step=4, tooltip="Expand the crop dimensions to this pixel multiple without resizing the image or mask."),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask"), io.Int.Output("crop_x", display_name="X"), io.Int.Output("crop_y", display_name="Y"), io.Int.Output("crop_width", display_name="Width"), io.Int.Output("crop_height", display_name="Height")],
        )

    @classmethod
    def execute(cls, image, mask, padding, multiple=8):
        if mask.shape[-2:] != image.shape[1:3]:
            mask = _resize_mask(mask, image.shape[2], image.shape[1], "nearest-exact")
        mask = _broadcast_batch(mask, image.shape[0], "Mask")
        x, y, width, height = _crop_bounds(mask, int(padding), int(multiple))
        return io.NodeOutput(image[:, y : y + height, x : x + width], mask[:, y : y + height, x : x + width], x, y, width, height)


class UC_StagedLayerCrops(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_StagedLayerCrops",
            display_name="Crop Staged Layers by Index",
            category="utils/image",
            inputs=[
                io.Image.Input("image", display_name="Composed Image"),
                io.Mask.Input("layer_masks", display_name="Layer Masks", tooltip="Mask batch in staged compositor layer order; each selected index uses the matching mask to crop the composed image."),
                io.String.Input("layer_indices", multiline=False, default="0", tooltip=("Zero-based layer indices in output order, separated by commas or spaces. Crops use tight bounds with no padding.")),
            ],
            outputs=[io.Image.Output("images", display_name="Images", is_output_list=True)],
        )

    @classmethod
    def execute(cls, image, layer_masks, layer_indices):
        return io.NodeOutput(crop_staged_layers_by_indices(image, layer_masks, layer_indices))


class UC_ImageCropMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ImageCropMerge",
            display_name="Image Crop Merge",
            category="utils/image",
            inputs=[
                io.Image.Input("cropped_image", tooltip="Processed crop to resize and merge into the original image region."),
                io.Image.Input("original_image", tooltip="Full image that receives the processed crop."),
                io.Int.Input("crop_x", default=0, min=0, max=MAX_RESOLUTION, force_input=True, tooltip="Left edge of the crop region in original-image pixels."),
                io.Int.Input("crop_y", default=0, min=0, max=MAX_RESOLUTION, force_input=True, tooltip="Top edge of the crop region in original-image pixels."),
                io.Int.Input("crop_width", default=512, min=1, max=MAX_RESOLUTION, force_input=True, tooltip="Width of the destination crop region; the processed crop is resized to this width."),
                io.Int.Input("crop_height", default=512, min=1, max=MAX_RESOLUTION, force_input=True, tooltip="Height of the destination crop region; the processed crop is resized to this height."),
                io.Combo.Input("resize_method", options=_RESIZE_METHODS, default="lanczos", tooltip="Interpolation used when resizing the processed crop to the destination region."),
                io.Mask.Input("mask", optional=True, tooltip="Optional blend mask for the crop region; disconnected replaces the region completely."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, cropped_image, original_image, crop_x, crop_y, crop_width, crop_height, resize_method, mask=None):
        result = original_image.clone()
        source = _resize_image(cropped_image, int(crop_width), int(crop_height), resize_method).to(result)
        source = _broadcast_batch(source, result.shape[0], "Cropped image")
        x1 = max(0, int(crop_x))
        y1 = max(0, int(crop_y))
        x2 = min(result.shape[2], int(crop_x) + int(crop_width))
        y2 = min(result.shape[1], int(crop_y) + int(crop_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Crop coordinates do not overlap the original image.")
        source = source[:, y1 - int(crop_y) : y2 - int(crop_y), x1 - int(crop_x) : x2 - int(crop_x)]
        if mask is None:
            result[:, y1:y2, x1:x2] = source
        else:
            mask = _resize_mask(mask, int(crop_width), int(crop_height), "bilinear").to(result)
            mask = _broadcast_batch(mask, result.shape[0], "Mask")
            mask = mask[:, y1 - int(crop_y) : y2 - int(crop_y), x1 - int(crop_x) : x2 - int(crop_x)].clamp(0.0, 1.0).unsqueeze(-1)
            result[:, y1:y2, x1:x2] = result[:, y1:y2, x1:x2] * (1.0 - mask) + source * mask
        return io.NodeOutput(result)


class UC_ImageAndMaskResize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ImageAndMaskResize",
            display_name="Image and Mask Resize",
            category="utils/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Image.Input("target", tooltip="Reference image whose width and height are used when explicit dimensions are disconnected."),
                io.Combo.Input("resize_method", options=_RESIZE_METHODS, default="lanczos", tooltip="Interpolation used to resize the image to the target dimensions."),
                io.Combo.Input("crop", options=["disabled", "center"], default="disabled", tooltip="Center crops while resizing when enabled; disabled stretches directly to the target dimensions."),
                io.Int.Input("mask_blur_radius", default=0, min=0, max=256, step=1, tooltip="Gaussian blur radius applied to the resized mask; zero disables blur."),
                io.Int.Input("width", default=512, min=1, max=MAX_RESOLUTION, force_input=True, optional=True, tooltip="Optional output width overriding the target image width."),
                io.Int.Input("height", default=512, min=1, max=MAX_RESOLUTION, force_input=True, optional=True, tooltip="Optional output height overriding the target image height."),
            ],
            outputs=[io.Image.Output(), io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image, mask, target, resize_method, crop, mask_blur_radius, width=None, height=None):
        target_width = int(width) if width is not None else target.shape[2]
        target_height = int(height) if height is not None else target.shape[1]
        if mask.shape[-2:] != image.shape[1:3]:
            mask = _resize_mask(mask, image.shape[2], image.shape[1], "bilinear")
        image = _resize_image(image, target_width, target_height, resize_method, crop)
        mask = _resize_mask(mask, target_width, target_height, "bilinear", crop)
        mask = _blur_mask(mask, int(mask_blur_radius)).clamp(0.0, 1.0)
        return io.NodeOutput(image, mask)


class UC_ResizeMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ResizeMask",
            display_name="Resize Mask",
            category="utils/mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("width", default=512, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("height", default=512, min=0, max=MAX_RESOLUTION, step=1),
                io.Boolean.Input("keep_proportions", default=False, tooltip="Fits the mask inside the requested width and height while preserving its aspect ratio."),
                io.Combo.Input("upscale_method", options=_RESIZE_METHODS, default="bilinear", tooltip="Interpolation used to resize the mask."),
                io.Combo.Input("crop", options=["disabled", "center"], default="disabled", tooltip="Center crops while resizing when enabled; disabled resizes to the calculated dimensions."),
            ],
            outputs=[io.Mask.Output(), io.Int.Output("width"), io.Int.Output("height")],
        )

    @classmethod
    def execute(cls, mask, width, height, keep_proportions, upscale_method, crop):
        original_height, original_width = mask.shape[-2:]
        width = original_width if width == 0 else int(width)
        height = original_height if height == 0 else int(height)
        if keep_proportions:
            ratio = min(width / original_width, height / original_height)
            width = max(1, round(original_width * ratio))
            height = max(1, round(original_height * ratio))
        mask = _resize_mask(mask, width, height, upscale_method, crop)
        return io.NodeOutput(mask, mask.shape[2], mask.shape[1])


class UC_BackgroundRemovalPreserveAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_BackgroundRemovalPreserveAlpha",
            display_name="Background Removal (Preserve Alpha)",
            category="utils/image",
            description=("Removes backgrounds while retaining the source resolution and soft alpha. RGBA inputs preserve their existing alpha without running a model."),
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, lazy=True, tooltip=("Optional Core background-removal model. When connected it overrides the internal BiRefNet/Lucida selector.")),
                io.Image.Input("image"),
                io.Combo.Input("background_removal_model_name", options=["birefnet", "lucida"], default="birefnet", tooltip=("Internal model used when background_removal_model_opt is disconnected.")),
                StagedBackgroundOptionsType.Input(
                    "background_options", display_name="Background Options", optional=True, tooltip=("Optional staged-compositor mask settings. Applies threshold, processing resolution, border cleanup, artifact cleanup, gap fill, feather, and mask resizing. RGBA inputs still preserve their embedded alpha exactly.")
                ),
            ],
            outputs=[io.Image.Output("image", display_name="RGBA Image"), io.Mask.Output("mask", display_name="Alpha Mask")],
        )

    @classmethod
    def check_lazy_status(cls, image, background_removal_model=_MISSING, **kwargs):
        if torch.is_tensor(image) and image.ndim == 4 and image.shape[-1] >= 4:
            return []
        if isinstance(background_removal_model, tuple) and len(background_removal_model) == 2 and background_removal_model[0] is None and background_removal_model[1]:
            return [background_removal_model[1]]
        return []

    @classmethod
    def execute(cls, image, background_removal_model_name="birefnet", background_removal_model=None, background_options=None):
        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError("Background Removal (Preserve Alpha) requires a batched IMAGE.")
        if image.shape[-1] < 4:
            background_removal_model = background_removal_model or (_load_internal_background_removal_model(background_removal_model_name))
        return io.NodeOutput(*background_removal_with_alpha(image, background_removal_model, background_options))


class UC_FaceRemovalPreserveAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_FaceRemovalPreserveAlpha",
            display_name="Face Removal (Preserve Alpha)",
            category="utils/image",
            description=("Keeps detected faces as expanded straight-RGBA crops and returns their matching alpha masks. Differently sized crops are centered on transparent batch padding."),
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, lazy=True, tooltip=("Optional Core background-removal model. When connected it overrides the internal BiRefNet/Lucida selector.")),
                io.Image.Input("image"),
                io.Combo.Input("background_removal_model_name", options=["birefnet", "lucida"], default="birefnet", tooltip=("Internal model used when background_removal_model_opt is disconnected.")),
                StagedBackgroundOptionsType.Input("background_options", display_name="Background Options", optional=True),
                StagedFaceOptionsType.Input("face_options", display_name="Face Options", optional=True),
            ],
            outputs=[io.Image.Output("image", display_name="RGBA Face"), io.Mask.Output("mask", display_name="Face Alpha Mask")],
        )

    @classmethod
    def check_lazy_status(cls, image, background_removal_model=_MISSING, **kwargs):
        if torch.is_tensor(image) and image.ndim == 4 and image.shape[-1] >= 4:
            return []
        if isinstance(background_removal_model, tuple) and len(background_removal_model) == 2 and background_removal_model[0] is None and background_removal_model[1]:
            return [background_removal_model[1]]
        return []

    @classmethod
    def execute(cls, image, background_removal_model_name="birefnet", background_removal_model=None, background_options=None, face_options=None):
        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError("Face Removal (Preserve Alpha) requires a batched IMAGE.")
        if image.shape[-1] < 4:
            background_removal_model = background_removal_model or (_load_internal_background_removal_model(background_removal_model_name))
        background_options = UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS | (background_options or {})
        face_options = UC_StagedMediaPipeFaceOptions.DEFAULTS | (face_options or {})
        return io.NodeOutput(*face_removal_with_alpha(image, background_removal_model, load_face_model(), background_options, face_options))


class UC_UnifiedBackgroundReplace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(io.Image.Input("foreground"), prefix="foreground_", min=1, max=50)
        return io.Schema(
            node_id="UC_UnifiedBackgroundReplace",
            display_name="Unified Background Replace",
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, tooltip="Optional Core background-removal model. Uses the internal BiRefNet model when disconnected."),
                io.Image.Input("background", tooltip="Single image used as the shared output canvas."),
                io.Float.Input("foreground_scale", default=0.90, min=0.05, max=10.0, step=0.01, tooltip="Fraction of the background's shortest side occupied by the foreground's longest bound. Values above 1 overscale and crop at the canvas edges."),
                io.Float.Input("long_axis_shift", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Position along the background's longest axis: -1 is left/up, 0 is centered, and 1 is right/down."),
                io.Float.Input("short_axis_shift", default=0.0, min=-1.0, max=1.0, step=0.01, tooltip="Position along the background's shortest axis: -1 is up/left, 0 is centered, and 1 is down/right."),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01, tooltip="Minimum model confidence retained as solid foreground."),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True, tooltip="Width of the source-edge strip where weak foreground predictions are removed."),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Opening radius used to remove small and thin mask artifacts."),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Closing radius used to fill small cracks and holes in the foreground."),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip="Inward edge softness; the foreground interior remains fully opaque."),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_IMAGE_RESIZE_TOOLTIP),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_MASK_RESIZE_TOOLTIP),
                io.Float.Input("workspace_padding", default=0.5, min=0.0, max=1.0, step=0.05, advanced=True, tooltip="Permitted off-canvas placement margin, up to 25% of each background axis."),
                io.Autogrow.Input("foreground_images", template=foreground_template, tooltip="Images to isolate, resize, and position over the background; each flattened image produces an independent output."),
            ],
            outputs=[io.Image.Output("images"), io.Mask.Output("masks")],
        )

    @classmethod
    def execute(
        cls,
        background_removal_model=None,
        background=None,
        foreground_images=None,
        foreground_scale=0.9,
        long_axis_shift=0.0,
        short_axis_shift=0.0,
        mask_threshold=0.5,
        border_cleanup_width=2,
        artifact_cleanup_radius=2,
        gap_fill_radius=2,
        feather_radius=2,
        image_resize_method="auto",
        mask_resize_method="auto",
        workspace_padding=0.5,
    ):
        if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
            raise ValueError("Unified Background Replace requires exactly one background image.")
        if background.shape[-1] < 3:
            raise ValueError("Background image must have at least three channels.")
        workspace_padding = float(workspace_padding)
        if not math.isfinite(workspace_padding) or not 0.0 <= workspace_padding <= 1.0:
            raise ValueError("Unified Background Replace workspace_padding must be between 0 and 1.")
        foregrounds = _flatten_autogrow_images(foreground_images)
        if not foregrounds:
            raise ValueError("Unified Background Replace requires at least one foreground image.")
        background_removal_model = resolve_background_removal_model(background_removal_model)

        background = background[..., :3]
        background_height, background_width = background.shape[1:3]
        target_longest = max(1, round(min(background_height, background_width) * float(foreground_scale)))
        composites = []
        masks = []

        for index, foreground in enumerate(foregrounds, start=1):
            if foreground.shape[-1] < 3:
                raise ValueError(f"Foreground image {index} must have at least three channels.")
            foreground = foreground[..., :3]
            raw_mask = background_removal_model.encode_image(foreground)
            if not torch.is_tensor(raw_mask):
                raise ValueError(f"Background removal model returned an invalid mask for foreground image {index}.")
            if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
                raw_mask = raw_mask[:, 0]
            elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 3 or raw_mask.shape[0] != 1:
                raise ValueError(f"Background removal model must return one [batch, height, width] mask for foreground image {index}.")
            if raw_mask.shape[-2:] != foreground.shape[1:3]:
                raw_mask = _resize_composite_mask(raw_mask, foreground.shape[2], foreground.shape[1], mask_resize_method)
            refined = _refine_foreground_mask(raw_mask[0], float(mask_threshold), border_cleanup_width, artifact_cleanup_radius, gap_fill_radius)
            points = torch.nonzero(refined > 0, as_tuple=False)
            if points.numel() == 0:
                raise ValueError(f"Background removal produced an empty foreground mask for image {index}.")
            top = int(points[:, 0].min())
            bottom = int(points[:, 0].max()) + 1
            left = int(points[:, 1].min())
            right = int(points[:, 1].max()) + 1
            crop = foreground[:, top:bottom, left:right]
            crop_mask = refined[None, top:bottom, left:right]
            crop_height, crop_width = crop.shape[1:3]
            scale = target_longest / max(crop_height, crop_width)
            placed_height = max(1, round(crop_height * scale))
            placed_width = max(1, round(crop_width * scale))
            resized_foreground = _resize_composite_image(crop, placed_width, placed_height, image_resize_method).to(background)
            resized_mask = _resize_composite_mask(crop_mask, placed_width, placed_height, mask_resize_method).to(background)
            resized_mask = resized_mask[0]
            alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask

            placement = {"long_axis_shift": float(long_axis_shift), "short_axis_shift": float(short_axis_shift)}
            offset_x, offset_y = _placement_offsets(background_width, background_height, placed_width, placed_height, placement, workspace_padding)
            slices = _visible_placement_slices(background_width, background_height, placed_width, placed_height, offset_x, offset_y)
            composite = background.clone()
            canvas_mask = background.new_zeros((1, background_height, background_width))
            if slices is None:
                composites.append(composite)
                masks.append(canvas_mask)
                continue
            (destination_top, destination_bottom, destination_left, destination_right, source_top, source_bottom, source_left, source_right) = slices
            placed_alpha = alpha[source_top:source_bottom, source_left:source_right]
            placed_foreground = resized_foreground[0, source_top:source_bottom, source_left:source_right]
            region = composite[0, destination_top:destination_bottom, destination_left:destination_right]
            composite[0, destination_top:destination_bottom, destination_left:destination_right] = region * (1.0 - placed_alpha.unsqueeze(-1)) + placed_foreground * placed_alpha.unsqueeze(-1)
            canvas_mask[0, destination_top:destination_bottom, destination_left:destination_right] = placed_alpha
            composites.append(composite)
            masks.append(canvas_mask)

        return io.NodeOutput(torch.cat(composites, dim=0), torch.cat(masks, dim=0))


class UC_StagedLayeredBackgroundCompositeOptions(io.ComfyNode):
    DEFAULTS = _STAGED_BACKGROUND_OPTION_DEFAULTS

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_StagedLayeredBackgroundCompositeOptions",
            display_name="Staged Composite Options",
            category="utils/image",
            inputs=[
                io.Float.Input("mask_threshold", default=0.5, min=0, max=1, step=0.01, tooltip=("Threshold removal-model masks and determine tight bounds for embedded alpha. Embedded alpha values inside those bounds remain unchanged.")),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, tooltip=_BORDER_CLEANUP_TOOLTIP),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, tooltip=_ARTIFACT_CLEANUP_TOOLTIP),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, tooltip=_GAP_FILL_TOOLTIP),
                io.Int.Input("feather_radius", default=2, min=0, max=64, tooltip=_FEATHER_TOOLTIP),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", tooltip=_IMAGE_RESIZE_TOOLTIP),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", tooltip=_MASK_RESIZE_TOOLTIP),
                io.Float.Input("foreground_blend", default=1.0, min=0.0, max=1.0, step=0.01, tooltip=("1.0 is fully foreground; 0.0 is a 50/50 normal blend where another foreground or face is underneath. Background-only areas remain fully foreground.")),
                io.Int.Input("mask_processing_resolution", default=0, min=0, max=65536, step=64, advanced=True, tooltip=("Longest edge used for background-removal mask refinement. 0 uses the connected model's native image_size. Original-resolution RGB is preserved.")),
            ],
            outputs=[StagedBackgroundOptionsType.Output(display_name="Background Options")],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(cls.DEFAULTS | kwargs)


class UC_StagedMediaPipeFaceOptions(io.ComfyNode):
    DEFAULTS = {"detection_threshold": 0.55, "maximum_faces": 16, "bbox_expansion": 64, "mask_expansion": 0, "face_feather_radius": 0, "initial_face_scale": 0.25, "face_blend": 1.0}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_StagedMediaPipeFaceOptions",
            display_name="Staged MediaPipe Face Options",
            category="utils/image",
            inputs=[
                io.Float.Input("detection_threshold", default=0.55, min=0, max=1, step=0.01, tooltip="Minimum MediaPipe confidence required to retain a detected face."),
                io.Int.Input("maximum_faces", default=16, min=1, max=16, tooltip="Maximum detected faces retained from each foreground image."),
                io.Int.Input("bbox_expansion", default=64, min=0, max=MAX_RESOLUTION, tooltip="Pixels added around each detected face box before extraction."),
                io.Int.Input("mask_expansion", default=0, min=-MAX_RESOLUTION, max=MAX_RESOLUTION, tooltip="Pixels used to grow the face mask; negative values shrink it."),
                io.Int.Input("face_feather_radius", default=0, min=0, max=512, tooltip="Optional inward softness applied to the extracted face-mask edge; 0 preserves the expanded mask boundary."),
                io.Float.Input("initial_face_scale", default=0.25, min=0.05, max=10, step=0.01, tooltip="Initial face-layer size relative to the background's shortest side."),
                io.Float.Input("face_blend", default=1.0, min=0.0, max=1.0, step=0.01, tooltip=("1.0 is fully face; 0.0 is a 50/50 normal blend where another foreground or face is underneath. Background-only areas remain fully face.")),
            ],
            outputs=[StagedFaceOptionsType.Output(display_name="Face Options")],
        )

    @classmethod
    def execute(cls, **kwargs):
        result = cls.DEFAULTS | kwargs
        result["maximum_faces"] = min(16, int(result["maximum_faces"]))
        return io.NodeOutput(result)


class UC_StagedMediaPipeFaceBackgroundComposite(io.ComfyNode):
    _staged_by_node = RetainedStageCache(max_entries=8)

    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(io.Image.Input("foreground", lazy=True), prefix="foreground_", min=1, max=50)
        return io.Schema(
            node_id="UC_StagedMediaPipeFaceBackgroundComposite",
            display_name="Staged Face Background Composite",
            category="utils/image",
            description=("Automatically retains face-aware foreground cutouts while source pixels and staging settings remain unchanged. Placement edits reuse the retained images and masks without rerunning either model."),
            inputs=[
                io.Image.Input("background"),
                StagedBackgroundOptionsType.Input("background_options", display_name="Background Options", optional=True),
                StagedFaceOptionsType.Input("face_options", display_name="Face Options", optional=True),
                io.Combo.Input("execution_mode", options=["run_staging", "run_staged", "full_run"], default="run_staged", advanced=True, tooltip="Frontend-managed staging request; ordinary composition validates and reuses retained cutouts automatically."),
                io.String.Input("placement_data", default='{"version":3,"workspace_padding":0.5,"layers":{}}', advanced=True),
                io.Combo.Input("background_removal_model_name", options=["birefnet", "lucida"], default="birefnet"),
                io.Autogrow.Input("foreground_images", template=foreground_template),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask"), io.BoundingBox.Output("bounding_boxes", display_name="Boxes"), io.Mask.Output("layer_masks", display_name="Layer Masks")],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, execution_mode, foreground_images=None, **kwargs):
        required = []
        for value in (foreground_images or {}).values():
            if isinstance(value, tuple) and len(value) == 2 and value[0] is None and value[1]:
                required.append(value[1])
        return required

    @classmethod
    def execute(cls, background, foreground_images, execution_mode, placement_data, background_removal_model_name="birefnet", background_options=None, face_options=None):
        node_id = str(cls.hidden.unique_id or "")
        background_options = UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS | (background_options or {})
        face_options = UC_StagedMediaPipeFaceOptions.DEFAULTS | (face_options or {})
        if isinstance(execution_mode, bool):
            execution_mode = "run_staged" if execution_mode else "run_staging"
        if execution_mode not in ("run_staging", "run_staged", "full_run"):
            raise ValueError(f"Unsupported staged compositor execution mode: {execution_mode!r}.")
        fingerprint = staged_foreground_fingerprint(foreground_images, ("internal", str(background_removal_model_name or "birefnet").lower(), "mediapipe_face_fp32"), background_options, face_options)

        def build_stage():
            removal_model = _load_internal_background_removal_model(background_removal_model_name)
            face_model = load_face_model()
            fresh = _stage_face_foregrounds(removal_model, face_model, foreground_images, background_options, face_options, placement_data)
            fresh["background_removal_model_name"] = str(background_removal_model_name).lower()
            return fresh

        staged, reused = resolve_retained_stage(cls._staged_by_node, node_id, fingerprint, build_stage, force=execution_mode in ("run_staging", "full_run"))
        if execution_mode == "run_staging":
            preview_stage = _apply_staged_layer_options(staged, background_options["foreground_blend"], face_options["face_blend"], face_options["face_feather_radius"])
            return _preview_staged_foregrounds(background, preview_stage, background_options["feather_radius"], placement_data, background_options["image_resize_method"], background_options["mask_resize_method"])
        stage_mode = "retained" if reused else "full_run"
        staged = _apply_staged_layer_options(staged, background_options["foreground_blend"], face_options["face_blend"], face_options["face_feather_radius"])
        return _composite_staged_foregrounds(background, staged, placement_data, background_options["feather_radius"], stage_mode=stage_mode, image_resize_method=background_options["image_resize_method"], mask_resize_method=background_options["mask_resize_method"])


class UC_StagedLayeredBackgroundComposite(io.ComfyNode):
    _staged_by_node = RetainedStageCache(max_entries=8)

    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(io.Image.Input("foreground", lazy=True, tooltip="Foreground image to isolate and retain as a placeable layer."), prefix="foreground_", min=1, max=50)
        return io.Schema(
            node_id="UC_StagedLayeredBackgroundComposite",
            display_name="Staged Background Composite",
            description=("Automatically rebuilds retained cutouts when foreground pixels or mask-generation settings change. Background and placement edits reuse the retained images and masks without rerunning background removal."),
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, lazy=True, tooltip=("Optional external Core background-removal model. When connected it overrides the internal BiRefNet/Lucida selector.")),
                io.Image.Input("background", tooltip="Single image used as the scene canvas."),
                StagedBackgroundOptionsType.Input("background_options", display_name="Background Options", optional=True, tooltip=("Optional Staged Composite Options. Uses the option node defaults when disconnected.")),
                io.Combo.Input("execution_mode", options=["run_staging", "run_staged", "full_run"], default="run_staged", advanced=True, tooltip=("Frontend-managed staging request. Ordinary composition fingerprints foregrounds and reuses valid cutouts; the editor's Run Staging action forces a preview refresh.")),
                io.String.Input("placement_data", default='{"version":2,"workspace_padding":0.5,"layers":{}}', advanced=True, tooltip="Versioned per-layer placement data managed by the LiteGraph scene editor."),
                io.Combo.Input("background_removal_model_name", options=["birefnet", "lucida"], default="birefnet", tooltip=("Internal model used when background_removal_model_opt is disconnected. Requires the exact checkpoint filename under models/background_removal.")),
                io.Autogrow.Input("foreground_images", template=foreground_template, tooltip="Foregrounds staged and composited from foreground_0 at the back to the highest socket at the front."),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask"), io.BoundingBox.Output("bounding_boxes", display_name="Boxes"), io.Mask.Output("layer_masks", display_name="Layer Masks")],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, execution_mode, background_removal_model=_MISSING, foreground_images=None, **kwargs):
        required = []
        if background_removal_model is None:
            required.append("background_removal_model")
        for value in (foreground_images or {}).values():
            if isinstance(value, tuple) and len(value) == 2:
                evaluated, original_key = value
            else:
                evaluated, original_key = value, None
            if evaluated is None and original_key:
                required.append(original_key)
        return required

    @classmethod
    def execute(cls, background, foreground_images, execution_mode, placement_data, background_removal_model_name="birefnet", background_removal_model=None, background_options=None):
        node_id = str(cls.hidden.unique_id or "")
        background_options = UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS | (background_options or {})
        if isinstance(execution_mode, bool):
            execution_mode = "run_staged" if execution_mode else "run_staging"
        if execution_mode not in ("run_staging", "run_staged", "full_run"):
            raise ValueError(f"Unsupported staged compositor execution mode: {execution_mode!r}.")
        if background_removal_model is None:
            model_identity = ("internal", str(background_removal_model_name or "birefnet").lower())
        else:
            model_identity = ("external", id(background_removal_model))
        fingerprint = staged_foreground_fingerprint(foreground_images, model_identity, background_options)

        def build_stage():
            if background_removal_model is None:
                removal_model = _load_internal_background_removal_model(background_removal_model_name)
                effective_model_name = str(background_removal_model_name or "birefnet").lower()
            else:
                removal_model = background_removal_model
                effective_model_name = "external"
            fresh = _stage_layered_foregrounds(
                removal_model,
                foreground_images,
                background_options["mask_threshold"],
                background_options["border_cleanup_width"],
                background_options["artifact_cleanup_radius"],
                background_options["gap_fill_radius"],
                background_options["mask_resize_method"],
                placement_data,
                mask_processing_resolution=background_options["mask_processing_resolution"],
            )
            fresh["background_removal_model_name"] = effective_model_name
            return fresh

        staged, reused = resolve_retained_stage(cls._staged_by_node, node_id, fingerprint, build_stage, force=execution_mode in ("run_staging", "full_run"))
        if execution_mode == "run_staging":
            preview_stage = _apply_staged_layer_options(staged, background_options["foreground_blend"], 1.0, 0)
            return _preview_staged_foregrounds(background, preview_stage, background_options["feather_radius"], placement_data, background_options["image_resize_method"], background_options["mask_resize_method"])
        stage_mode = "retained" if reused else "full_run"
        staged = _apply_staged_layer_options(staged, background_options["foreground_blend"], 1.0, 0)
        return _composite_staged_foregrounds(background, staged, placement_data, background_options["feather_radius"], stage_mode=stage_mode, image_resize_method=background_options["image_resize_method"], mask_resize_method=background_options["mask_resize_method"])


class UC_StagedIndividualComposites(io.ComfyNode):
    _staged_by_node = RetainedStageCache(max_entries=8)

    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(io.Image.Input("foreground", lazy=True), prefix="foreground_", min=1, max=50)
        return io.Schema(
            node_id="UC_StagedIndividualComposites",
            display_name="Staged Individual Composites",
            category="utils/image",
            description=("Automatically retains foreground cutouts and returns one full background composite per included foreground. Placement edits reuse cached images and masks."),
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, lazy=True),
                io.Image.Input("background"),
                StagedBackgroundOptionsType.Input("background_options", display_name="Background Options", optional=True, tooltip="foreground_blend is ignored because each result contains one foreground."),
                io.Combo.Input("execution_mode", options=["run_staging", "run_staged", "full_run"], default="run_staged", advanced=True, tooltip="Frontend-managed staging request; retained cutouts are validated automatically."),
                io.String.Input("placement_data", default='{"version":2,"workspace_padding":0.5,"layers":{}}', advanced=True),
                io.Combo.Input("background_removal_model_name", options=["birefnet", "lucida"], default="birefnet"),
                io.Autogrow.Input("foreground_images", template=foreground_template),
            ],
            outputs=[io.Image.Output("individual_composites", display_name="Individual Composites"), io.Mask.Output("masks", display_name="Individual Masks"), io.BoundingBox.Output("bounding_boxes", display_name="Boxes")],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, execution_mode, background_removal_model=_MISSING, foreground_images=None, **kwargs):
        return UC_StagedLayeredBackgroundComposite.check_lazy_status(execution_mode, background_removal_model, foreground_images)

    @classmethod
    def execute(cls, background, foreground_images, execution_mode, placement_data, background_removal_model_name="birefnet", background_removal_model=None, background_options=None):
        node_id = str(cls.hidden.unique_id or "")
        background_options = UC_StagedLayeredBackgroundCompositeOptions.DEFAULTS | (background_options or {})
        if isinstance(execution_mode, bool):
            execution_mode = "run_staged" if execution_mode else "run_staging"
        if execution_mode not in ("run_staging", "run_staged", "full_run"):
            raise ValueError(f"Unsupported staged compositor execution mode: {execution_mode!r}.")
        if background_removal_model is None:
            model_identity = ("internal", str(background_removal_model_name or "birefnet").lower())
        else:
            model_identity = ("external", id(background_removal_model))
        fingerprint = staged_foreground_fingerprint(foreground_images, model_identity, background_options)

        def build_stage():
            if background_removal_model is None:
                removal_model = _load_internal_background_removal_model(background_removal_model_name)
                effective_model_name = str(background_removal_model_name or "birefnet").lower()
            else:
                removal_model = background_removal_model
                effective_model_name = "external"
            fresh = _stage_layered_foregrounds(
                removal_model,
                foreground_images,
                background_options["mask_threshold"],
                background_options["border_cleanup_width"],
                background_options["artifact_cleanup_radius"],
                background_options["gap_fill_radius"],
                background_options["mask_resize_method"],
                placement_data,
                mask_processing_resolution=background_options["mask_processing_resolution"],
            )
            fresh["background_removal_model_name"] = effective_model_name
            return fresh

        staged, reused = resolve_retained_stage(cls._staged_by_node, node_id, fingerprint, build_stage, force=execution_mode in ("run_staging", "full_run"))
        if execution_mode == "run_staging":
            preview = _preview_staged_foregrounds(background, staged, background_options["feather_radius"], placement_data, background_options["image_resize_method"], background_options["mask_resize_method"])
            return io.NodeOutput(preview.result[0], preview.result[3], preview.result[2], ui=preview.ui)
        stage_mode = "retained" if reused else "full_run"
        return _composite_staged_individual_foregrounds(background, staged, placement_data, background_options["feather_radius"], stage_mode=stage_mode, image_resize_method=background_options["image_resize_method"], mask_resize_method=background_options["mask_resize_method"])


class UC_LayeredBackgroundComposite(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        foreground_template = io.Autogrow.TemplatePrefix(io.Image.Input("foreground", tooltip="Foreground image to isolate and add as a placeable layer."), prefix="foreground_", min=1, max=50)
        return io.Schema(
            node_id="UC_LayeredBackgroundComposite",
            display_name="Layered Background Composite",
            category="utils/image",
            inputs=[
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, tooltip="Optional Core background-removal model. Uses the internal BiRefNet model when disconnected."),
                io.Image.Input("background", tooltip="Single image used as the scene canvas."),
                io.Float.Input("mask_threshold", default=0.50, min=0.0, max=1.0, step=0.01, tooltip=_MASK_THRESHOLD_TOOLTIP),
                io.Int.Input("border_cleanup_width", default=2, min=0, max=64, step=1, advanced=True, tooltip=_BORDER_CLEANUP_TOOLTIP),
                io.Int.Input("artifact_cleanup_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_ARTIFACT_CLEANUP_TOOLTIP),
                io.Int.Input("gap_fill_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_GAP_FILL_TOOLTIP),
                io.Int.Input("feather_radius", default=2, min=0, max=64, step=1, advanced=True, tooltip=_FEATHER_TOOLTIP),
                io.Combo.Input("image_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_IMAGE_RESIZE_TOOLTIP),
                io.Combo.Input("mask_resize_method", options=_COMPOSITE_RESIZE_METHODS, default="auto", advanced=True, tooltip=_MASK_RESIZE_TOOLTIP),
                io.String.Input("placement_data", default='{"version":2,"workspace_padding":0.5,"layers":{}}', advanced=True, tooltip="Versioned per-layer placement data managed by the LiteGraph scene editor."),
                io.Autogrow.Input("foreground_images", template=foreground_template, tooltip="One image per socket, composited from foreground_0 at the back to the highest socket at the front."),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask")],
        )

    @classmethod
    def execute(
        cls, background_removal_model=None, background=None, foreground_images=None, placement_data='{"version":2,"workspace_padding":0.5,"layers":{}}', mask_threshold=0.5, border_cleanup_width=2, artifact_cleanup_radius=2, gap_fill_radius=2, feather_radius=2, image_resize_method="auto", mask_resize_method="auto"
    ):
        if not torch.is_tensor(background) or background.ndim != 4 or background.shape[0] != 1:
            raise ValueError("Layered Background Composite requires exactly one background image.")
        if background.shape[-1] < 3:
            raise ValueError("Background image must have at least three channels.")
        foregrounds = _ordered_single_foregrounds(foreground_images)
        if not foregrounds:
            raise ValueError("Layered Background Composite requires at least one foreground image.")
        background_removal_model = resolve_background_removal_model(background_removal_model)
        placements = _parse_layer_placements(placement_data)
        placement_version, _, _, workspace_padding = _parse_layer_payload(placement_data)
        foreground_by_socket = dict(foregrounds)
        foregrounds = [(key, foreground_by_socket[key]) for key in _ordered_layer_keys(placement_data, foreground_by_socket)]

        scene = background[..., :3].clone()
        background_height, background_width = scene.shape[1:3]
        combined_mask = scene.new_zeros((1, background_height, background_width))
        layer_metadata = []

        for key, foreground in foregrounds:
            if foreground.shape[-1] < 3:
                raise ValueError(f"Foreground input {key} must have at least three channels.")
            foreground = foreground[..., :3]
            raw_mask = background_removal_model.encode_image(foreground)
            if not torch.is_tensor(raw_mask):
                raise ValueError(f"Background removal returned an invalid mask for {key}.")
            if raw_mask.ndim == 4 and raw_mask.shape[1] == 1:
                raw_mask = raw_mask[:, 0]
            elif raw_mask.ndim == 4 and raw_mask.shape[-1] == 1:
                raw_mask = raw_mask[..., 0]
            if raw_mask.ndim != 3 or raw_mask.shape[0] != 1:
                raise ValueError(f"Background removal must return one [batch, height, width] mask for {key}.")
            if raw_mask.shape[-2:] != foreground.shape[1:3]:
                raw_mask = _resize_composite_mask(raw_mask, foreground.shape[2], foreground.shape[1], mask_resize_method)

            refined = _refine_foreground_mask(raw_mask[0], float(mask_threshold), border_cleanup_width, artifact_cleanup_radius, gap_fill_radius)
            points = torch.nonzero(refined > 0, as_tuple=False)
            if points.numel() == 0:
                raise ValueError(f"Background removal produced an empty foreground mask for {key}.")
            top = int(points[:, 0].min())
            bottom = int(points[:, 0].max()) + 1
            left = int(points[:, 1].min())
            right = int(points[:, 1].max()) + 1
            crop = foreground[:, top:bottom, left:right]
            crop_mask = refined[None, top:bottom, left:right]
            crop_height, crop_width = crop.shape[1:3]

            placement = placements.get(key, {**(_DEFAULT_LAYER_PLACEMENT_V2 if placement_version == 2 else _DEFAULT_LAYER_PLACEMENT), "_version": placement_version})
            desired_flip = bool(placement.get("flip_horizontal", False))
            desired_flip_vertical = bool(placement.get("flip_vertical", False))
            if desired_flip:
                crop = torch.flip(crop, dims=(2,))
                crop_mask = torch.flip(crop_mask, dims=(2,))
            if desired_flip_vertical:
                crop = torch.flip(crop, dims=(1,))
                crop_mask = torch.flip(crop_mask, dims=(1,))
            target_longest = max(1, round(min(background_height, background_width) * placement["scale"]))
            scale = target_longest / max(crop_height, crop_width)
            placed_height = max(1, round(crop_height * scale))
            placed_width = max(1, round(crop_width * scale))
            resized_foreground = _resize_composite_image(crop, placed_width, placed_height, image_resize_method).to(scene)
            resized_mask = _resize_composite_mask(crop_mask, placed_width, placed_height, mask_resize_method).to(scene)
            resized_mask = resized_mask[0]
            alpha = _feather_mask(resized_mask, -int(feather_radius)) if feather_radius else resized_mask
            offset_x, offset_y = _placement_offsets(background_width, background_height, placed_width, placed_height, placement, workspace_padding if placement_version == 2 else 0.0)

            slices = _visible_placement_slices(background_width, background_height, placed_width, placed_height, offset_x, offset_y)
            if slices is not None:
                (destination_top, destination_bottom, destination_left, destination_right, source_top, source_bottom, source_left, source_right) = slices
                placed_alpha = alpha[source_top:source_bottom, source_left:source_right]
                placed_foreground = resized_foreground[0, source_top:source_bottom, source_left:source_right]
                region = scene[0, destination_top:destination_bottom, destination_left:destination_right]
                scene[0, destination_top:destination_bottom, destination_left:destination_right] = region * (1.0 - placed_alpha.unsqueeze(-1)) + placed_foreground * placed_alpha.unsqueeze(-1)
                mask_region = combined_mask[0, destination_top:destination_bottom, destination_left:destination_right]
                combined_mask[0, destination_top:destination_bottom, destination_left:destination_right] = mask_region + placed_alpha * (1.0 - mask_region)

            preview_alpha = crop_mask[0]
            if feather_radius:
                preview_alpha = _feather_mask(preview_alpha, -int(feather_radius))
            preview_rgba = torch.cat((crop[0], preview_alpha.unsqueeze(-1)), dim=-1).unsqueeze(0)
            layer_metadata.append({"socket": key, "crop_width": crop_width, "crop_height": crop_height, "preview_tensor": preview_rgba, "flip_horizontal": desired_flip, "flip_vertical": desired_flip_vertical})

        editor_metadata = {"version": 1, "background": {"width": background_width, "height": background_height}, "layers": []}
        for layer in layer_metadata:
            entry = {"socket": layer["socket"], "crop_width": layer["crop_width"], "crop_height": layer["crop_height"], "flip_horizontal": layer["flip_horizontal"], "flip_vertical": layer["flip_vertical"]}
            try:
                entry["preview"] = _save_editor_preview(layer["preview_tensor"], f"UC_layered_{layer['socket']}")
            except Exception:
                logging.warning("Unable to create editor cutout preview for %s.", layer["socket"], exc_info=True)
            editor_metadata["layers"].append(entry)

        return io.NodeOutput(scene, combined_mask, ui={"uc_layered_scene_editor": [editor_metadata]})


class UC_MediaPipeFaceCompositeOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MediaPipeFaceCompositeOptions",
            display_name="MediaPipe Face Composite Options",
            category="utils/image",
            inputs=[
                io.Int.Input("bbox_expansion", default=64, min=0, max=MAX_RESOLUTION, step=1),
                io.Int.Input("mask_expansion", default=0, min=-MAX_RESOLUTION, max=MAX_RESOLUTION, step=1, tooltip="Pixels used to expand (positive) or contract (negative) the face mask. Automatically preserves nose and profile features based on distance to the nose landmark."),
                io.Int.Input("feather_radius", default=8, min=-512, max=512, step=1, tooltip="Radius used to feather the face mask boundary. Automatically preserves crisp facial feature edges near the nose landmark."),
                io.Float.Input("source_warp_strength", default=1.0, min=-10.0, max=10.0, step=0.01, tooltip="Strength of thin-plate-spline warping applied to the donor face towards the target facial shape (default 1.0 = donor adapts to target head geometry, negative values reverse the adaptation)."),
                io.Float.Input("target_warp_strength", default=0.0, min=-10.0, max=10.0, step=0.01, tooltip="Strength of thin-plate-spline warping applied to the target recipient face towards the donor face shape (default 0.0 = recipient head/neck undistorted, negative values reverse the adaptation)."),
                io.Int.Input("warp_decay_radius", default=64, min=1, max=MAX_RESOLUTION, step=1),
                io.Float.Input("score_thresh", default=0.25, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[FaceCompositeOptionsType.Output()],
        )

    @classmethod
    def execute(cls, bbox_expansion, mask_expansion, feather_radius, source_warp_strength=1.0, target_warp_strength=0.0, warp_decay_radius=64, score_thresh=0.25):
        return io.NodeOutput(
            {"bbox_expansion": int(bbox_expansion), "mask_expansion": int(mask_expansion), "feather_radius": int(feather_radius), "source_warp_strength": float(source_warp_strength), "target_warp_strength": float(target_warp_strength), "warp_decay_radius": int(warp_decay_radius), "score_thresh": float(score_thresh)}
        )


class UC_MediaPipeFaceComposite(io.ComfyNode):
    DEFAULT_OPTIONS = {"bbox_expansion": 64, "mask_expansion": 0, "feather_radius": 8, "source_warp_strength": 1.0, "target_warp_strength": 0.0, "warp_decay_radius": 64, "score_thresh": 0.25}

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MediaPipeFaceComposite",
            display_name="MediaPipe Face Composite",
            category="utils/image",
            description="Composites the largest source face into the largest target face using full-range MediaPipe detection.",
            inputs=[
                FaceDetectionType.Input("face_detection_model", display_name="face_detection_model_opt", optional=True, tooltip="Optional Core MediaPipe face model. Uses mediapipe_face_fp32.safetensors when disconnected."),
                io.BackgroundRemoval.Input("background_removal_model", display_name="background_removal_model_opt", optional=True, tooltip="Optional Core background-removal model. Uses the internal BiRefNet model when disconnected."),
                io.Image.Input("source"),
                io.Image.Input("target"),
                FaceCompositeOptionsType.Input("options", optional=True),
            ],
            outputs=[io.Image.Output("image"), io.Image.Output("face_crop", display_name="Face Crop")],
        )

    @classmethod
    def execute(cls, face_detection_model=None, background_removal_model=None, source=None, target=None, options=None):
        if source.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("MediaPipe Face Composite currently requires one source and one target image.")
        face_detection_model = face_detection_model or load_face_model()
        background_removal_model = resolve_background_removal_model(background_removal_model)
        options = cls.DEFAULT_OPTIONS | (options or {})
        score_thresh = options["score_thresh"]
        source = source[..., :3]
        target = target[..., :3]
        source_uint8 = source.mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        target_uint8 = target.mul(255.0).add(0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
        source_face = _detect_largest_face_adaptive(face_detection_model, source_uint8, score_thresh, "source")
        target_face = _detect_largest_face_adaptive(face_detection_model, target_uint8, score_thresh, "target")
        ring = _ordered_ring(face_detection_model.connection_sets["face_oval"])

        feature_indices = list(ring)
        nose_indices = set()
        if hasattr(face_detection_model, "connection_sets") and isinstance(face_detection_model.connection_sets, dict):
            for part in ("nose", "lips", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "irises"):
                connections = face_detection_model.connection_sets.get(part)
                if connections:
                    for a, b in connections:
                        if part == "nose":
                            nose_indices.add(a)
                            nose_indices.add(b)
                        if a not in feature_indices:
                            feature_indices.append(a)
                        if b not in feature_indices:
                            feature_indices.append(b)

        max_idx = min(len(source_face["landmarks_xy"]), len(target_face["landmarks_xy"]))
        feature_indices = [idx for idx in feature_indices if idx < max_idx]
        nose_indices = {idx for idx in nose_indices if idx < max_idx}
        if not nose_indices:
            for fallback_idx in (1, 4):
                if fallback_idx < max_idx and fallback_idx in feature_indices:
                    nose_indices.add(fallback_idx)
        oval_count = len([idx for idx in ring if idx < max_idx])

        source = source.to(target)
        source_oval_points = source_face["landmarks_xy"][ring]
        source_mask = _polygon_mask(source.shape[1], source.shape[2], source_oval_points, target.device, target.dtype)
        foreground = background_removal_model.encode_image(source)
        if foreground.shape[-2:] != source.shape[1:3]:
            foreground = _resize_mask(foreground, source.shape[2], source.shape[1], "bilinear")
        foreground = foreground[0].to(target).clamp(0.0, 1.0)

        padding = options["bbox_expansion"]
        sx1, sy1, sx2, sy2 = _expanded_box(source_face["bbox_xyxy"], padding, source.shape[2], source.shape[1])
        tx1, ty1, tx2, ty2 = _expanded_box(target_face["bbox_xyxy"], padding, target.shape[2], target.shape[1])
        source_crop = source[0, sy1:sy2, sx1:sx2]
        source_oval = source_mask[sy1:sy2, sx1:sx2]
        source_foreground = foreground[sy1:sy2, sx1:sx2]
        target_crop = target[0, ty1:ty2, tx1:tx2]

        local_source_points = source_face["landmarks_xy"][feature_indices] - np.array([sx1, sy1], dtype=np.float32)
        local_target_points = target_face["landmarks_xy"][feature_indices] - np.array([tx1, ty1], dtype=np.float32)
        scale, rotation, translation = _similarity_transform(local_source_points, local_target_points)
        placed_source, placed_oval, placed_foreground = _transform_source(source_crop, source_oval, source_foreground, target_crop.shape[0], target_crop.shape[1], scale, rotation, translation)
        placed_source_points = scale * (local_source_points @ rotation.T) + translation

        source_warp_strength = float(options.get("source_warp_strength", 1.0))
        if abs(source_warp_strength) > 1e-4:
            combined_source = torch.cat((placed_source, placed_oval.unsqueeze(-1), placed_foreground.unsqueeze(-1)), dim=-1)
            warped_combined = _warp_source(combined_source, placed_source_points, local_target_points, source_warp_strength, options["warp_decay_radius"], oval_count=oval_count)
            placed_source = warped_combined[..., :3]
            placed_oval = warped_combined[..., 3]
            placed_foreground = warped_combined[..., 4]
            placed_source_points = placed_source_points + (local_target_points - placed_source_points) * source_warp_strength

        target_warp_strength = float(options.get("target_warp_strength", 0.0))
        if abs(target_warp_strength) > 1e-4:
            warped_target = _warp_target(target_crop, placed_source_points, local_target_points, target_warp_strength, options["warp_decay_radius"], oval_count=oval_count)
        else:
            warped_target = target_crop

        alpha = _apply_face_mask_options(placed_oval, placed_foreground, options["mask_expansion"], options["feather_radius"], placed_source_points, feature_indices, nose_indices, oval_count=oval_count)
        completed_crop = warped_target * (1.0 - alpha.unsqueeze(-1)) + placed_source * alpha.unsqueeze(-1)
        result = target.clone()
        result[0, ty1:ty2, tx1:tx2] = completed_crop
        return io.NodeOutput(result, completed_crop.unsqueeze(0))


class UC_CreateBoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_CreateBoundingBoxes",
            display_name="Create Bounding Boxes (Nodes 1.0)",
            category="utils/image",
            description=("Draw, move, resize, and label bounding boxes directly on an interactive Nodes 1.0 canvas. Outputs layout prompt elements for FLUX.3 Layout Prompt and Ideogram JSON prompts, pixel-space bounding boxes, and a rendered preview image."),
            inputs=[
                io.Image.Input("background", optional=True, tooltip=("Optional reference image displayed behind the bounding-box canvas and dimmed in the rendered preview output.")),
                io.MultiType.Input(
                    "bboxes", [io.BoundingBox, io.Array, io.String], optional=True, tooltip=("Optional upstream bounding boxes, prompt layout elements, or JSON string to seed the canvas. When the upstream value changes, it initializes the canvas; manual canvas edits stay active until a new upstream value arrives.")
                ),
                io.Int.Input("width", default=0, min=0, max=16384, step=16, tooltip=("Width of the canvas and pixel grid. Leave at 0 to match the connected background image width (or 1024 when no background is connected).")),
                io.Int.Input("height", default=0, min=0, max=16384, step=16, tooltip=("Height of the canvas and pixel grid. Leave at 0 to match the connected background image height (or 1024 when no background is connected).")),
                io.Combo.Input(
                    "input_coord_space",
                    options=["gemini_0_to_1000", "qwen_0_to_1000", "moss_centered_pixels"],
                    default="gemini_0_to_1000",
                    tooltip=(
                        "Format of incoming JSON bounding boxes connected to bboxes. Use gemini_0_to_1000 for Gemini "
                        "layout JSON outputs ([ymin, xmin, ymax, xmax] on a 0 to 1000 grid), qwen_0_to_1000 for local "
                        "Qwen3-VL / Qwen3.5 JSON outputs ([xmin, ymin, xmax, ymax] on a 0 to 1000 grid), or "
                        "moss_centered_pixels when feeding an already-converted MOSS-Image layout prompt."
                    ),
                ),
                io.String.Input("box_data", default='{"version":1,"boxes":[],"last_incoming":null}', advanced=True, tooltip=("Serialized bounding-box list and metadata managed by the Nodes 1.0 interactive canvas editor.")),
            ],
            outputs=[
                io.Image.Output("preview", display_name="preview", tooltip="Preview image showing the dimmed background or canvas with numbered boxes, labels, and swatches."),
                io.BoundingBox.Output("bboxes", display_name="bboxes", tooltip="Pixel bounding boxes aligned to the resolved canvas width and height."),
                io.Array.Output("elements", display_name="elements", tooltip=("Layout elements ready to connect to FLUX.3 Layout Prompt (referenced as <obj_1>, <obj_2>, ...) or Build JSON Prompt (Ideogram).")),
                io.String.Output("moss_prompt", display_name="moss_prompt", tooltip=("Ready-to-encode MOSS-Image layout JSON prompt with centered pixel box coordinates aligned to the resolved canvas width and height.")),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, width: int = 0, height: int = 0, input_coord_space: str = "gemini_0_to_1000", box_data: str = '{"version":1,"boxes":[],"last_incoming":null}', background=None, bboxes=None) -> io.NodeOutput:
        return execute_create_bounding_boxes(width=width, height=height, box_data=box_data, background=background, bboxes=bboxes, input_coord_space=input_coord_space)


CreateBoundingBoxesNodes1 = UC_CreateBoundingBoxes


class UC_MOSSImageBBoxConvert(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MOSSImageBBoxConvert",
            display_name="MOSS-Image BBox Coordinate Convert",
            category="utils/image",
            description=("Converts Gemini layout JSON or Create Bounding Boxes elements into a ready-to-encode MOSS-Image layout JSON prompt with centered pixel box coordinates matched to your image or target canvas size."),
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="", tooltip=("Connect Gemini's structured JSON text output here (or enter a scene summary caption when connecting boxes through the elements input).")),
                io.Image.Input("image", optional=True, tooltip="Optional reference image used to automatically set width and height when they are left at 0."),
                io.Array.Input("elements", optional=True, tooltip="Optional elements list from Create Bounding Boxes to override or supply the layout boxes."),
                io.Int.Input("width", default=0, min=0, max=16384, step=16, tooltip="Target canvas width in pixels. Leave at 0 to match the connected image width (or 1024 when no image is connected)."),
                io.Int.Input("height", default=0, min=0, max=16384, step=16, tooltip="Target canvas height in pixels. Leave at 0 to match the connected image height (or 1024 when no image is connected)."),
                io.Combo.Input(
                    "input_coord_space",
                    options=["gemini_0_to_1000", "qwen_0_to_1000", "moss_centered_pixels"],
                    default="gemini_0_to_1000",
                    tooltip=(
                        "Format of the incoming JSON boxes in prompt. Use gemini_0_to_1000 for Gemini layout JSON "
                        "outputs ([ymin, xmin, ymax, xmax] on a 0 to 1000 grid), qwen_0_to_1000 for local Qwen3-VL / "
                        "Qwen3.5 JSON outputs ([xmin, ymin, xmax, ymax] on a 0 to 1000 grid), or moss_centered_pixels "
                        "when feeding an already-converted MOSS-Image layout prompt."
                    ),
                ),
                io.String.Input("character_boxes", default="", optional=True, tooltip=("Optional 1-based box numbers (such as '1, 3') to mark as character boxes when boxes come from Create Bounding Boxes elements or Ideogram JSON. Leave empty when feeding Gemini MOSS JSON.")),
            ],
            outputs=[io.String.Output("moss_prompt", display_name="moss_prompt", tooltip="Ready-to-encode MOSS-Image layout JSON prompt with centered pixel box coordinates. Connect directly to your MOSS-Image text encoder.")],
        )

    @classmethod
    def execute(cls, prompt: str = "", width: int = 0, height: int = 0, input_coord_space: str = "gemini_0_to_1000", character_boxes: str = "", image=None, elements=None) -> io.NodeOutput:
        return io.NodeOutput(build_moss_image_layout_prompt(prompt=prompt, elements=elements, width=width, height=height, image=image, input_coord_space=input_coord_space, character_boxes=character_boxes))
