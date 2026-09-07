from comfy_api.latest import io

from .model_helpers import detect_sam3


class UC_SAM3Detect(io.ComfyNode):
    """SAM3 detection over aspect-preserving overlapping tiles."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_SAM3Detect",
            display_name="SAM3 Detect (Tiled)",
            category="image/detection",
            description="Text detection runs centre-first on aspect-preserving SAM3 tiles along the longer image axis; emitted masks always match the input image dimensions.",
            search_aliases=["sam3", "segment anything", "open vocabulary", "text detection", "segment"],
            inputs=[
                io.Model.Input("model", display_name="model"), io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning from CLIPTextEncode"),
                io.BoundingBox.Input("bboxes", display_name="bboxes", force_input=True, optional=True, tooltip="Bounding boxes to segment within"),
                io.String.Input("positive_coords", display_name="positive_coords", force_input=True, optional=True, tooltip="Positive point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                io.String.Input("negative_coords", display_name="negative_coords", force_input=True, optional=True, tooltip="Negative point prompts as JSON [{\"x\": int, \"y\": int}, ...] (pixel coords)"),
                io.Float.Input("threshold", display_name="threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Int.Input("refine_iterations", display_name="refine_iterations", default=2, min=0, max=5, tooltip="SAM decoder refinement passes (0=use raw detector masks)"),
                io.Boolean.Input("edge_padding", display_name="edge_padding", default=True, tooltip="Replicate 32px of the outer image edge before tiling, then discard it from the returned masks. Tile overlap is calculated from the image dimensions."),
                io.Boolean.Input("individual_masks", display_name="individual_masks", default=False, tooltip="Output per-object masks instead of union"),
            ],
            outputs=[io.Mask.Output("masks"), io.BoundingBox.Output("bboxes")],
        )

    @classmethod
    def execute(cls, model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, edge_padding=True, individual_masks=False):
        return io.NodeOutput(*detect_sam3(model, image, conditioning, bboxes, positive_coords, negative_coords, threshold, refine_iterations, individual_masks, edge_padding))
