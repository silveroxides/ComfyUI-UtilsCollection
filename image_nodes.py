import os
import json
from .model_helpers import register_openpose_paths, load_openpose_model, openpose_forward, load_dwpose_model, dwpose_forward
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import scipy
import cv2
import nodes
import folder_paths
from tqdm import tqdm
from PIL import Image, ImageOps, ImageSequence, ImageDraw, ImageFont
import kornia.morphology as morph
from .helper_functions import pil2tensor, math_diag, pct_to_px, composite, fill_mask_from_edges, iterative_directional_stretch_fill, gaussian_blur_nchw, hex_to_rgb, string_to_color, resize_nchw, FLOW_PRESETS
from .tile_helpers import (
    accumulate_tile_images,
    apply_tile_differential_diffusion,
    split_and_encode_tiles,
)
from .color_palette_helpers import extract_prevalent_color_outputs
from .image_helpers import (
    run_openpose_batch,
    run_dwpose_batch,
    VIDEO_FRAME_SAMPLING_STRATEGIES,
    VIDEO_FRAME_TIMESTAMP_FORMATS,
    VIDEO_FRAME_TIMELINE_STYLES,
    VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE,
    VIDEO_TIMELINE_TEXT_STRUCTURE,
    downscale_nohalo_lohalo,
    images_to_video_timeline,
    mask_to_bounding_box,
    match_image_properties,
    sample_video_frames_as_images,
    video_timeline_text,
)


from comfy_api.latest import io
from comfy import model_management
import node_helpers
from nodes import MAX_RESOLUTION

HighResolutionTileLayout = io.Custom("UC_HIGH_RES_TILE_LAYOUT")
PoseKeypoint = io.Custom("POSE_KEYPOINT")
register_openpose_paths()


class UC_BatchedOpenPose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_BatchedOpenPose", display_name="Batched OpenPose Pose", category="image/pose",
            description="Batches body frames and hand/face crops on ComfyUI's selected device. Models belong in models/controlnet/preprocessors.",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("detect_hand", default=True),
                io.Boolean.Input("detect_body", default=True, tooltip="Draw body skeletons. Body inference still locates hands/faces and produces keypoints."),
                io.Boolean.Input("detect_face", default=True),
                io.Int.Input("resolution", default=512, min=64, max=4096, step=64, tooltip="Output shortest edge in pixels."),
                io.Int.Input("batch_size", default=4, min=1, max=64, tooltip="Maximum frames or person crops per network call. Lower this if VRAM is insufficient."),
                io.Boolean.Input("scale_stick_for_xinsr_cn", default=False),
                io.Float.Input("body_threshold", default=0.1, min=0.0, max=1.0, step=0.01, tooltip="Minimum body-joint heatmap peak."),
                io.Float.Input("hand_threshold", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Minimum hand-joint heatmap response."),
                io.Float.Input("face_threshold", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Minimum face-landmark heatmap response."),
                io.Float.Input("limb_threshold", default=0.05, min=0.0, max=1.0, step=0.01, tooltip="Minimum part-affinity score at sampled limb positions."),
                io.Float.Input("limb_support", default=0.8, min=0.0, max=1.0, step=0.01, tooltip="More than this fraction of limb samples must exceed the affinity threshold."),
                io.Int.Input("min_body_parts", default=4, min=1, max=18, tooltip="Minimum connected joints for accepting a person."),
                io.Float.Input("min_body_score", default=0.4, min=0.0, max=10.0, step=0.01, tooltip="Minimum average assembled person score, including limb affinity scores."),
                io.Boolean.Input("temporal_filter", default=False, tooltip="Treat the batch as ordered frames and prune unsupported individual joints before rendering; do not use for unrelated images."),
                io.Int.Input("temporal_radius", default=2, min=1, max=30),
                io.Int.Input("temporal_min_support", default=2, min=1, max=60),
                io.Float.Input("temporal_max_distance", default=0.1, min=0.001, max=1.0, step=0.01, tooltip="Maximum joint movement as a fraction of the person's body-box diagonal."),
                io.Float.Input("temporal_match_iou", default=0.3, min=0.0, max=1.0, step=0.01, tooltip="Minimum body-box overlap for matching people across neighboring frames."),
            ],
            outputs=[io.Image.Output("image"), PoseKeypoint.Output("pose_keypoint")],
        )

    @classmethod
    def execute(cls, image, detect_hand=True, detect_body=True, detect_face=True, resolution=512,
                batch_size=4, scale_stick_for_xinsr_cn=False, body_threshold=0.1, hand_threshold=0.05, face_threshold=0.05,
                limb_threshold=0.05, limb_support=0.8, min_body_parts=4, min_body_score=0.4,
                temporal_filter=False, temporal_radius=2, temporal_min_support=2, temporal_max_distance=0.1, temporal_match_iou=0.3):
        result, poses = run_openpose_batch(
            image, resolution, batch_size, detect_body, detect_hand, detect_face, scale_stick_for_xinsr_cn,
            loader=load_openpose_model, forward=openpose_forward, body_threshold=body_threshold,
            hand_threshold=hand_threshold, face_threshold=face_threshold, limb_threshold=limb_threshold,
            limb_support=limb_support, min_body_parts=min_body_parts, min_body_score=min_body_score,
            temporal_filter=temporal_filter, temporal_radius=temporal_radius, temporal_min_support=temporal_min_support,
            temporal_max_distance=temporal_max_distance, temporal_match_iou=temporal_match_iou,
        )
        return io.NodeOutput(result, poses, ui={"openpose_json": [json.dumps(poses)]})


class UC_DWPoseEstimator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_DWPoseEstimator", display_name="DWPose Estimator", category="image/pose",
            description="Batches eager YOLOX frames and DWPose person crops using safetensors models in models/controlnet/preprocessors.",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("detect_hand", default=True),
                io.Boolean.Input("detect_body", default=True),
                io.Boolean.Input("detect_face", default=True),
                io.Int.Input("resolution", default=512, min=64, max=4096, step=64),
                io.Int.Input("batch_size", default=5, min=1, max=64, tooltip="Maximum frames or person crops per model call; partial batches do not need padding."),
                io.Boolean.Input("scale_stick_for_xinsr_cn", default=False),
                io.Float.Input("detection_threshold", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="Minimum person-detection confidence. Raise to reject weak detections such as shadows; may also remove real people."),
                io.Float.Input("keypoint_threshold", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="Minimum confidence for body, hand, and face keypoints. Raise to hide uncertain joints; this does not track or smooth motion."),
                io.Boolean.Input("temporal_filter", default=False,
                                 tooltip="Treat the IMAGE batch as consecutive frames and prune unsupported individual keypoints before rendering. Person entries are retained."),
                io.Int.Input("temporal_radius", default=2, min=1, max=30, tooltip="Neighbor frames to inspect on each side; spans processing-chunk boundaries."),
                io.Int.Input("temporal_min_support", default=2, min=1, max=60, tooltip="Other frames that must support a joint; limited by available neighbors at clip edges."),
                io.Float.Input("temporal_max_distance", default=0.1, min=0.001, max=1.0, step=0.01,
                               tooltip="Maximum joint movement as a fraction of the person's box diagonal. Increase for faster motion."),
                io.Float.Input("temporal_match_iou", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="Minimum detector-box overlap for matching people across neighboring frames."),
            ],
            outputs=[io.Image.Output("image"), PoseKeypoint.Output("pose_keypoint")],
        )

    @classmethod
    def execute(cls, image, detect_hand=True, detect_body=True, detect_face=True, resolution=512,
                batch_size=5, scale_stick_for_xinsr_cn=False, detection_threshold=0.3, keypoint_threshold=0.3,
                temporal_filter=False, temporal_radius=2, temporal_min_support=2, temporal_max_distance=0.1, temporal_match_iou=0.3):
        result, poses = run_dwpose_batch(image, resolution, batch_size, detect_body, detect_hand, detect_face,
                                        scale_stick_for_xinsr_cn, loader=load_dwpose_model, forward=dwpose_forward,
                                        detection_threshold=detection_threshold, keypoint_threshold=keypoint_threshold,
                                        temporal_filter=temporal_filter, temporal_radius=temporal_radius,
                                        temporal_min_support=temporal_min_support, temporal_max_distance=temporal_max_distance,
                                        temporal_match_iou=temporal_match_iou)
        return io.NodeOutput(result, poses, ui={"openpose_json": [json.dumps(poses)]})




class UC_ExtractPrevalentColors(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ExtractPrevalentColors",
            display_name="Extract Prevalent Colors",
            category="image/color",
            description=(
                "Extracts perceptually distinct prevalent colors from every image "
                "without resizing or spatial sampling."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "color_count",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Maximum number of prevalent colors returned per image.",
                ),
                io.Boolean.Input(
                    "prefix_hash",
                    default=True,
                    tooltip="Prefix every six-digit hexadecimal color with #.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "colors",
                    display_name="colors",
                    tooltip=(
                        "One comma-separated palette ordered from most prevalent "
                        "to least prevalent."
                    ),
                ),
                io.Image.Output(
                    "palette_grid",
                    display_name="palette grid",
                    tooltip=(
                        "RGB color blocks in reading order from the top-left. "
                        "Unused cells are flattened to black."
                    ),
                ),
                io.String.Output(
                    "color_names",
                    display_name="color names",
                    tooltip=(
                        "Nearest human color names in the same order as the "
                        "exact hexadecimal colors."
                    ),
                ),
                io.String.Output(
                    "palette_description",
                    display_name="palette description",
                    tooltip=(
                        "Deterministic palette-level temperature, lightness, "
                        "chroma, neutrality, and contrast description."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, color_count, prefix_hash) -> io.NodeOutput:
        palette, grid, color_names, palette_description = (
            extract_prevalent_color_outputs(
                image, color_count, prefix_hash
            )
        )
        return io.NodeOutput(
            palette,
            grid,
            color_names,
            palette_description,
        )


class UC_Image_Color_Noise(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Image_Color_Noise",
            display_name="Image Color Noise",
            category="utils",
            inputs=[
                io.Int.Input("width", default=512, max=4096, min=64, step=1),
                io.Int.Input("height", default=512, max=4096, min=64, step=1),
                io.Float.Input("frequency", default=0.5, max=100.0, min=0.0, step=0.01),
                io.Float.Input(
                    "attenuation", default=0.5, max=100.0, min=0.0, step=0.01
                ),
                io.Combo.Input(
                    "noise_type",
                    options=["grey", "white", "red", "pink", "green", "blue", "mix"],
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[
                io.Image.Output(display_name="Image"),
            ],
        )

    @classmethod
    def execute(cls, width, height, frequency, attenuation, noise_type, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise_image = cls.generate_power_noise(
            width, height, frequency, attenuation, noise_type, generator
        )
        return io.NodeOutput(pil2tensor(noise_image))

    @classmethod
    def generate_power_noise(
        cls, width, height, frequency, attenuation, noise_type, generator
    ):
        def normalize_array(arr):
            return (255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))).astype(
                np.uint8
            )

        def white_noise(w, h, gen):
            return torch.rand(h, w, generator=gen).numpy()

        def grey_noise_texture(w, h, att, gen):
            return torch.normal(mean=0, std=att, size=(h, w), generator=gen).numpy()

        def fourier_noise(w, h, att, power_modifier, gen):
            noise = grey_noise_texture(w, h, att, gen)
            fy = np.fft.fftfreq(h)[:, np.newaxis]
            fx = np.fft.fftfreq(w)
            f = np.sqrt(fx**2 + fy**2)
            f[0, 0] = 1.0
            power_spectrum = f**power_modifier
            fft_noise = np.fft.fft2(noise)
            fft_modified = fft_noise * power_spectrum
            inv_fft = np.fft.ifft2(fft_modified)
            return np.real(inv_fft)

        noise_array = np.zeros((height, width, 3), dtype=np.uint8)
        zeros_channel = np.zeros((height, width), dtype=np.uint8)

        if noise_type == "grey":
            luma = normalize_array(
                grey_noise_texture(width, height, attenuation, generator)
            )
            noise_array = np.stack([luma, luma, luma], axis=-1)

        elif noise_type == "white":
            r = normalize_array(white_noise(width, height, generator))
            g = normalize_array(white_noise(width, height, generator))
            b = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([r, g, b], axis=-1)

        elif noise_type == "red":
            r = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([r, zeros_channel, zeros_channel], axis=-1)

        elif noise_type == "green":
            g = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([zeros_channel, g, zeros_channel], axis=-1)

        elif noise_type == "blue":
            b = normalize_array(white_noise(width, height, generator))
            noise_array = np.stack([zeros_channel, zeros_channel, b], axis=-1)

        elif noise_type == "pink":
            base_texture = fourier_noise(width, height, attenuation, -1.0, generator)
            r = normalize_array(base_texture)
            g = (r * 0.75).astype(np.uint8)
            b = (r * 0.85).astype(np.uint8)
            noise_array = np.stack([r, g, b], axis=-1)

        elif noise_type == "mix":
            r = normalize_array(
                fourier_noise(width, height, attenuation, -1.0, generator)
            )  # Pink Frequency
            g = normalize_array(
                fourier_noise(width, height, attenuation, 0.5, generator)
            )  # Green Frequency
            b = normalize_array(
                fourier_noise(width, height, attenuation, 1.0, generator)
            )  # Blue Frequency
            noise_array = np.stack([r, g, b], axis=-1)

        else:
            print(f"[ERROR] Unsupported noise type `{noise_type}`")
            return Image.new("RGB", (width, height), color="black")

        return Image.fromarray(noise_array, "RGB")


class UC_LoadImageWithAlpha(io.ComfyNode):
    """Core Load Image compatibility with an additional RGBA IMAGE output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        input_directory = folder_paths.get_input_directory()
        image_files = folder_paths.filter_files_content_types(
            [name for name in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, name))],
            ["image"],
        )
        return io.Schema(
            node_id="UC_LoadImageWithAlpha",
            display_name="Load Image (Preserve Alpha)",
            category="image/loaders",
            inputs=[
                io.Combo.Input(
                    "image",
                    options=sorted(image_files),
                    upload=io.UploadType.image,
                    image_folder=io.FolderType.input,
                    tooltip="Image file to load. Matches Core Load Image while also exposing its alpha as a fourth IMAGE channel.",
                ),
            ],
            outputs=[
                io.Image.Output("image", display_name="IMAGE"),
                io.Mask.Output("mask", display_name="MASK"),
                io.Image.Output("image_rgba", display_name="IMAGE_RGBA"),
            ],
        )

    @classmethod
    def execute(cls, image: str) -> io.NodeOutput:
        rgb, mask = nodes.LoadImage().load_image(image)
        if mask.ndim == 3 and mask.shape[:3] == rgb.shape[:3]:
            alpha = (1.0 - mask).to(device=rgb.device, dtype=rgb.dtype).unsqueeze(-1)
        else:
            alpha = torch.ones((*rgb.shape[:3], 1), device=rgb.device, dtype=rgb.dtype)
        return io.NodeOutput(rgb, mask, torch.cat((rgb, alpha), dim=-1))

    @classmethod
    def fingerprint_inputs(cls, image: str):
        return nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def validate_inputs(cls, image: str):
        return nodes.LoadImage.VALIDATE_INPUTS(image)


class UC_LoadImagePath(io.ComfyNode):
    """
    Load an image from an arbitrary file path with proper mask handling.
    Returns the image and a mask extracted from the alpha channel.
    For images without alpha, returns a full-sized zero mask (not 64x64).
    Supports both absolute and relative paths, with any OS path separator.
    """

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        Normalize a file path to handle:
        - Backslashes (Windows) and forward slashes (Unix)
        - Relative paths (starting with '.', '..', or lowercase letter)
        - Whitespaces in paths and filenames

        Returns an absolute, normalized path.
        """
        if not path:
            return path

        # Strip leading/trailing whitespace but preserve internal whitespace
        path = path.strip()

        # Normalize path separators: replace backslashes with forward slashes
        # Then use os.path.normpath to get the OS-appropriate format
        path = path.replace('\\', '/')
        path = os.path.normpath(path)

        # Convert to absolute path if relative
        # os.path.abspath handles: '.', '..', and paths without drive letter
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        return path

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadImagePath",
            display_name="Load Image (Path)",
            category="advanced/image",
            inputs=[
                io.String.Input(
                    "image_path",
                    multiline=False,
                    placeholder="path/to/image.png or X:/path/to/image.png",
                    tooltip="Path to the image file. Supports absolute or relative paths with any OS format (backslashes or forward slashes). Whitespaces in paths are supported.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Mask.Output(display_name="MASK"),
                io.Mask.Output(display_name="MASK_INVERTED"),
            ],
        )

    @classmethod
    def execute(cls, image_path: str) -> io.NodeOutput:
        # Normalize the path to handle relative paths and different separators
        normalized_path = cls._normalize_path(image_path)

        # Validate path
        if not normalized_path or not os.path.isfile(normalized_path):
            raise ValueError(f"Invalid image path: {image_path} (resolved to: {normalized_path})")

        img = node_helpers.pillow(Image.open, normalized_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            # Handle 16-bit images (mode 'I') - normalize by 65535, not 255
            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 65535))

            image = i.convert("RGB")

            # Set dimensions from first frame
            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            # Skip frames with different dimensions
            if image.size[0] != w or image.size[1] != h:
                continue

            # Convert to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            # Extract mask from alpha channel
            if 'A' in i.getbands():
                # RGBA image - extract alpha
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            elif i.mode == 'P' and 'transparency' in i.info:
                # Palette mode with transparency - convert to RGBA (already transposed)
                rgba = i.convert('RGBA')
                mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
            else:
                # No alpha - return full-sized zero mask (NOT 64x64!)
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))

            # MPO format: only use first frame
            if img.format == "MPO":
                break

        if len(output_images) == 0:
            raise ValueError(f"No valid image frames could be loaded from: {image_path}")

        # Stack frames
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        # Create inverted mask
        output_mask_inverted = 1.0 - output_mask

        return io.NodeOutput(output_image, output_mask, output_mask_inverted)

    @classmethod
    def IS_CHANGED(cls, image_path: str):
        normalized_path = cls._normalize_path(image_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            return ""
        m = hashlib.sha256()
        with open(normalized_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image_path: str):
        if not image_path:
            return "Image path cannot be empty"
        normalized_path = cls._normalize_path(image_path)
        if not os.path.isfile(normalized_path):
            return f"Invalid image file: {image_path} (resolved to: {normalized_path})"
        return True


class UC_LoadImageDirectory(io.ComfyNode):
    """
    Load multiple images from a directory as a batch.
    Supports selecting a range of images using start index and count.
    Images are sorted alphanumerically.
    """

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return path
        path = path.strip()
        path = path.replace('\\', '/')
        path = os.path.normpath(path)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        return path

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadImageDirectory",
            display_name="Load Images (Directory)",
            category="advanced/image",
            inputs=[
                io.String.Input(
                    "directory_path",
                    multiline=False,
                    placeholder="path/to/directory",
                    tooltip="Path to the directory containing images.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Index of the first image to load (sorted alphabetically)."
                ),
                io.Int.Input(
                    "load_count",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Number of images to load."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE", is_output_list=True),
                io.Mask.Output(display_name="MASK", is_output_list=True),
                io.Mask.Output(display_name="MASK_INVERTED", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, directory_path: str, start_index: int, load_count: int) -> io.NodeOutput:
        normalized_path = cls._normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        # Get valid image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mpo'}
        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(normalized_path, f))

        files.sort()

        # Apply slice
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        if not selected_files:
             raise ValueError(f"No images found in range [{start_index}:{end_index}] in directory: {directory_path}")

        # Group images by size to handle batching or provide warning
        output_images = []
        output_masks = []
        output_masks_inverted = []

        for file_path in selected_files:
            w, h = None, None
            try:
                img = node_helpers.pillow(Image.open, file_path)
            except Exception as e:
                print(f"Warning: Could not load image {file_path}: {e}")
                continue

            # Process just the first frame
            i = node_helpers.pillow(ImageOps.exif_transpose, img)

            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 65535))

            image = i.convert("RGB")

            if w is None:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                print(f"Warning: Skipping {file_path} due to dimension mismatch. Expected {w}x{h}, got {image.size}")
                continue

            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
                mask_inverted = 1.0 - mask
            elif i.mode == 'P' and 'transparency' in i.info:
                rgba = i.convert('RGBA')
                mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask_np)
                mask_inverted = 1.0 - mask
            else:
                mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
                mask_inverted = 1.0 - mask


            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))
            output_masks_inverted.append(mask_inverted.unsqueeze(0))

        if not output_images:
            raise ValueError("No valid images loaded (checked dimensions and validity).")


        return io.NodeOutput(output_images, output_masks, output_masks_inverted)

    @classmethod
    def IS_CHANGED(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = cls._normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif', '.mpo'}
        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort()
        end_index = start_index + load_count
        selected_files = files[start_index:end_index]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                # Hash filename and mtime
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        return m.digest().hex()


class UC_SampleVideoFramesAsImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SampleVideoFramesAsImages",
            display_name="Sample Video Frames (Images)",
            category="image/video",
            description=(
                "Samples presentation-timestamp-aligned video frames as both an "
                "IMAGE batch and independently mappable IMAGE list."
            ),
            search_aliases=[
                "sample video frames",
                "extract video frames",
                "video keyframes",
                "video to images",
                "video image batch",
                "video image list",
            ],
            inputs=[
                io.Video.Input("video", tooltip="Core VIDEO input to sample."),
                io.Combo.Input(
                    "sampling_strategy",
                    options=list(VIDEO_FRAME_SAMPLING_STRATEGIES),
                    default="codec keyframes",
                    tooltip=(
                        "Choose which video frames to keep. Focused PTS groups "
                        "frames around the focus settings."
                    ),
                ),
                io.Int.Input(
                    "maximum_frames",
                    default=16,
                    min=0,
                    step=1,
                    tooltip=(
                        "Maximum total output count, including the zero-time frame. "
                        "0 returns every frame eligible after spacing and stride."
                    ),
                ),
                io.Int.Input("focus_areas", default=0, min=0, max=3, step=1, tooltip="Focused PTS only. How many parts to split the video into. 0 spaces frames evenly."),
                io.Float.Input("focus_one", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Focused PTS only. Where frames group in the first part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Float.Input("focus_two", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Focused PTS only. Where frames group in the second part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Float.Input("focus_three", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Focused PTS only. Where frames group in the third part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Boolean.Input(
                    "include_zero_time",
                    default=True,
                    tooltip=(
                        "Place the first visible frame at output position 0 and "
                        "normalize its presentation timestamp to zero."
                    ),
                ),
                io.Float.Input(
                    "minimum_spacing_seconds",
                    default=0.25,
                    min=0.0,
                    step=0.01,
                    round=0.001,
                    tooltip=(
                        "Minimum presentation-time separation between selected "
                        "frames."
                    ),
                ),
                io.Int.Input(
                    "keyframe_stride",
                    default=1,
                    min=1,
                    step=1,
                    tooltip=(
                        "For codec keyframes only, retain every Nth raw codec "
                        "keyframe before spacing and count limiting."
                    ),
                ),
                io.Combo.Input(
                    "timestamp_format",
                    options=list(VIDEO_FRAME_TIMESTAMP_FORMATS),
                    default="00.000s",
                    tooltip="Formatting reused verbatim by every text output.",
                ),
                io.Combo.Input(
                    "timeline_style",
                    options=list(VIDEO_FRAME_TIMELINE_STYLES),
                    default="H3 alignment prefix",
                    tooltip="Built-in timeline syntax, or custom to use the timeline text structure.",
                ),
                io.String.Input(
                    "timeline_text_structure",
                    multiline=True,
                    dynamic_prompts=False,
                    default=VIDEO_TIMELINE_TEXT_STRUCTURE,
                    tooltip=(
                        "One structure repeated for every selected frame. Use "
                        "<<time>> or <<timestamp>>, <<picture>>, and <<shot>>."
                    ),
                ),
                io.String.Input(
                    "structured_timeline_text_structure",
                    multiline=True,
                    dynamic_prompts=False,
                    default=VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
                    tooltip=(
                        "One whole-timeline structure. Use <<duration>>, "
                        "<<segments>>, <<timestamps>>, and <<references>>. One <<shot>> … "
                        "<<timestamp>> section repeats for every selected frame."
                    ),
                ),
                io.Int.Input(
                    "index_offset",
                    default=0,
                    min=0,
                    step=1,
                    tooltip=(
                        "Adds this value to every <Picture N> reference so "
                        "earlier reference images can occupy the first indices."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    "image_batch",
                    display_name="image batch",
                    tooltip="Selected frames as one chronological IMAGE batch.",
                ),
                io.String.Output(
                    "timestamps_text",
                    display_name="timestamps text",
                    tooltip="All formatted timestamps joined into one comma-and-space-separated string.",
                ),
                io.String.Output(
                    "timeline_text",
                    display_name="timeline text",
                    tooltip="One concatenation-ready timeline string.",
                ),
                io.Float.Output(
                    "video_runtime",
                    display_name="video runtime",
                    tooltip="Generation duration in seconds, rounded up to H3's supported length at 24 fps. Use this duration for captioning and generation; sampled frames keep their source timestamps.",
                ),
                io.String.Output(
                    "structured_timeline_text",
                    display_name="structured timeline text",
                    tooltip=(
                        "Video duration, segment count, and chronological "
                        "<Picture N> timestamp references in one sentence."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        sampling_strategy: str,
        maximum_frames: int,
        focus_areas: int,
        focus_one: float,
        focus_two: float,
        focus_three: float,
        include_zero_time: bool,
        minimum_spacing_seconds: float,
        keyframe_stride: int,
        timestamp_format: str,
        timeline_style: str,
        timeline_text_structure: str,
        structured_timeline_text_structure: str,
        index_offset: int,
    ) -> io.NodeOutput:
        sampled = sample_video_frames_as_images(
            video,
            sampling_strategy,
            maximum_frames,
            include_zero_time,
            minimum_spacing_seconds,
            keyframe_stride,
            timestamp_format,
            timeline_style,
            timeline_text_structure,
            structured_timeline_text_structure,
            index_offset,
            focus_areas,
            focus_one,
            focus_two,
            focus_three,
        )
        return io.NodeOutput(
            sampled.image_batch,
            sampled.timestamps_text,
            sampled.timeline_text,
            sampled.video_runtime,
            sampled.structured_timeline_text,
        )


class UC_ImagesToVideoTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplatePrefix(io.Image.Input("image", optional=True), prefix="image", min=1, max=16)
        return io.Schema(
            node_id="UC_ImagesToVideoTimeline",
            display_name="Images to Video Timeline",
            category="image/video",
            description="Gives ordered images timestamps for a video.",
            search_aliases=["image timeline", "images to video", "video timestamps", "image frames"],
            inputs=[
                io.Float.Input(
                    "duration", default=5.0, min=0.01, step=0.01,
                    tooltip="Requested video duration in seconds. Rounds up to H3's supported length at 24 fps before placing image timestamps.",
                ),
                io.Int.Input(
                    "focus_areas", default=0, min=0, max=3, step=1,
                    tooltip="How many parts to split the video into. 0 spaces images evenly.",
                ),
                io.Float.Input("focus_one", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Where images group in the first part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Float.Input("focus_two", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Where images group in the second part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Float.Input("focus_three", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Where images group in the third part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Boolean.Input("last_image_is_final", default=False, tooltip="Turn on only when the last image is the final video frame."),
                io.Boolean.Input(
                    "resize_images", default=True,
                    tooltip="Make all images the same size as the first image. Turn off to keep their original sizes. Image batch becomes black.",
                ),
                io.Combo.Input("timestamp_format", options=list(VIDEO_FRAME_TIMESTAMP_FORMATS), default="00.000s", tooltip="How timestamps look in text outputs."),
                io.Combo.Input("timeline_style", options=list(VIDEO_FRAME_TIMELINE_STYLES), default="H3 alignment prefix", tooltip="Built-in timeline syntax, or custom to use the timeline text structure."),
                io.String.Input("timeline_text_structure", multiline=True, dynamic_prompts=False, default=VIDEO_TIMELINE_TEXT_STRUCTURE, tooltip="One structure repeated for every image when timeline style is custom. Use <<time>> or <<timestamp>>, <<picture>>, and <<shot>>."),
                io.String.Input("structured_timeline_text_structure", multiline=True, dynamic_prompts=False, default=VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE, tooltip="One whole-timeline structure. Use <<duration>>, <<segments>>, <<timestamps>>, and <<references>>. One <<shot>> … <<timestamp>> section repeats for every image."),
                io.Int.Input("index_offset", default=0, min=0, step=1, tooltip="Skip this many Picture numbers before the first image."),
                io.Autogrow.Input("image_inputs", template=image_template, tooltip="Add images in order. Each image in a batch gets its own timestamp."),
            ],
            outputs=[
                io.Image.Output("image_batch", display_name="image batch", tooltip="All images in one batch. Black when Resize Images is off."),
                io.String.Output("timestamps_text", display_name="timestamps text", tooltip="All timestamps in one line."),
                io.String.Output("timeline_text", display_name="timeline text", tooltip="Timeline text ready for a prompt."),
                io.Float.Output("video_runtime", display_name="video runtime", tooltip="Actual H3 generation duration in seconds at 24 fps. The image timeline and duration text use this same rounded value."),
                io.String.Output("structured_timeline_text", display_name="structured timeline text", tooltip="Text with video length and Picture timestamps."),
            ],
        )

    @classmethod
    def execute(cls, duration: float, focus_areas: int, focus_one: float, focus_two: float, focus_three: float, last_image_is_final: bool, resize_images: bool, timestamp_format: str, timeline_style: str, timeline_text_structure: str, structured_timeline_text_structure: str, index_offset: int, image_inputs: io.Autogrow.Type) -> io.NodeOutput:
        sampled = images_to_video_timeline(image_inputs, duration, focus_areas, focus_one, focus_two, focus_three, last_image_is_final, resize_images, timestamp_format, timeline_style, timeline_text_structure, structured_timeline_text_structure, index_offset)
        return io.NodeOutput(
            sampled.image_batch,
            sampled.timestamps_text,
            sampled.timeline_text,
            sampled.video_runtime,
            sampled.structured_timeline_text,
        )


class UC_VideoTimelineText(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_VideoTimelineText",
            display_name="Video Timeline (Text)",
            category="image/video",
            description="Builds text-only video timeline guidance from a manual duration or connected video.",
            inputs=[
                io.Float.Input("duration", default=5.0, min=0.01, step=0.01, optional=True, tooltip="Requested video duration in seconds. Rounds up to H3's supported length at 24 fps. A connected video supplies the starting duration instead."),
                io.Int.Input("segment_count", default=5, min=1, step=1, tooltip="Number of timestamped timeline entries."),
                io.Int.Input("focus_areas", default=0, min=0, max=3, step=1, tooltip="How many parts to split the timeline into. 0 spaces entries evenly."),
                io.Float.Input("focus_one", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Where entries group in the first part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Float.Input("focus_two", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Where entries group in the second part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Float.Input("focus_three", default=0.50, min=0.00, max=1.00, step=0.01, tooltip="Where entries group in the third part. 0 is early, 0.5 is balanced, 1 is late."),
                io.Combo.Input("timestamp_format", options=list(VIDEO_FRAME_TIMESTAMP_FORMATS), default="00.000s", tooltip="Formatting reused verbatim by every text output."),
                io.String.Input("timeline_text_structure", multiline=True, dynamic_prompts=False, default=VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE, tooltip="One structure repeated for every timeline entry. Use <<shot>> and <<time>> or <<timestamp>>."),
                io.String.Input("structured_timeline_text_structure", multiline=True, dynamic_prompts=False, default=VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE, tooltip="One whole-timeline structure. <<duration>> and <<segments>> are scalars; <<timestamps>> is comma-and-space-separated; one <<shot>> … <<timestamp>> section repeats comma-and-space-separated for every timeline entry."),
                io.Video.Input("video", optional=True, tooltip="Optional video whose duration replaces the duration widget, then rounds up to the supported H3 generation length."),
            ],
            outputs=[
                io.String.Output("timestamps_text", display_name="timestamps text", tooltip="All formatted timestamps in one line."),
                io.String.Output("timeline_text", display_name="timeline text", tooltip="Timeline text ready for a prompt."),
                io.Float.Output("video_runtime", display_name="video runtime", tooltip="Actual H3 generation duration in seconds at 24 fps. Timeline timestamps and duration text use this same rounded value."),
                io.String.Output("structured_timeline_text", display_name="structured timeline text", tooltip="Text with video length and Picture timestamps."),
            ],
        )

    @classmethod
    def execute(cls, segment_count: int, focus_areas: int, focus_one: float, focus_two: float, focus_three: float, timestamp_format: str, timeline_text_structure: str, structured_timeline_text_structure: str, duration: float = 5.0, video: io.Video.Type | None = None) -> io.NodeOutput:
        video_duration = video.get_duration() if video is not None else duration
        return io.NodeOutput(*video_timeline_text(video_duration, segment_count, focus_areas, focus_one, focus_two, focus_three, timestamp_format, timeline_text_structure, structured_timeline_text_structure))


class UC_ImageMatchPropertiesNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageMatchProperties",
            display_name="Image Match Properties",
            category="advanced/image",
            inputs=[
                io.Image.Input("original_image", tooltip="Connect the original image. The node automatically finds where it appears inside the larger result, including when its size changed."),
                io.Image.Input("generated_image", tooltip="Connect the outpainted image. Colors near the matched original-image edges are used to correct the whole result."),
                io.Float.Input("overall_weight", default=1.0, min=0.0, max=3.0, step=0.001, tooltip="Strength of the complete correction. 0 keeps the generated image unchanged, 1 applies the measured correction, and values above 1 make every correction stronger."),
                io.Float.Input("color_weight", default=1.0, min=0.0, max=3.0, step=0.001, tooltip="Matches the original image's overall color cast and balance. Values above 1 push the result farther toward those colors. It does not copy colors by pixel position."),
                io.Float.Input("lighting_weight", default=1.0, min=0.0, max=3.0, step=0.001, tooltip="Matches the original image's overall brightness. Values above 1 make the brightness correction stronger."),
                io.Float.Input("texture_preservation", default=0.5, min=0.0, max=1.0, step=0.001, tooltip="Keeps fine detail from the generated image while correcting contrast. Increase it if textures become too harsh or too soft."),
                io.Mask.Input("mask", optional=True, tooltip="Connect the outpaint mask when available. White generated areas are measured and corrected; black original areas stay unchanged. The correction feathers automatically inside the generated edge. Leave disconnected for automatic edge matching and whole-image correction."),
                io.Float.Input("saturation_weight", optional=True, default=1.0, min=0.0, max=3.0, step=0.001, tooltip="Matches how vivid or muted the original image is. Values above 1 strengthen the saturation correction; lower it if the result becomes too colorful."),
                io.Float.Input("contrast_weight", optional=True, default=1.0, min=0.0, max=3.0, step=0.001, tooltip="Matches the difference between dark and bright areas in the original image. Values above 1 strengthen the contrast correction; lower it if shadows or highlights become too strong."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
        overall_weight: float,
        color_weight: float,
        lighting_weight: float,
        texture_preservation: float,
        mask: torch.Tensor = None,
        saturation_weight: float = 1.0,
        contrast_weight: float = 1.0,
        source_analysis_mask: torch.Tensor = None,
        target_analysis_mask: torch.Tensor = None,
    ) -> io.NodeOutput:
        if mask is None and target_analysis_mask is not None:
            mask = target_analysis_mask
        result = match_image_properties(
            original_image,
            generated_image,
            overall_weight,
            color_weight,
            lighting_weight,
            texture_preservation,
            mask,
            saturation_weight,
            contrast_weight,
        )
        return io.NodeOutput(result)


class UC_OpticalFlowComposite(io.ComfyNode):
    """
    Composites a Klein edit onto the original image.

    v2.2: Global Rigid Alignment. Calculates a single global camera shift from
    unchanged background pixels and translates the entire generated image rigidly.
    Eliminates seam distortion while fixing AI background drift.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_OpticalFlowComposite",
            display_name="Optical Flow Composite (Global Align)",
            category="advanced/image",
            inputs=[
                io.Image.Input("original_image"),
                io.Image.Input("generated_image"),
                io.Float.Input(
                    "delta_e_threshold",
                    default=-1.0, min=-1.0, max=100.0, step=1.0,
                    tooltip="How different a pixel's color must be to count as 'edited'. Higher values = only obvious edits are detected (smaller mask, more original preserved). Lower values = subtle changes are also captured (larger mask, more of the generated image used). Set to -1 for automatic tuning."
                ),
                io.Float.Input(
                    "grow_mask_pct",
                    default=0.0, min=-3.0, max=3.0, step=0.1,
                    tooltip="Expands or shrinks the detected edit region. Positive values grow the mask outward, capturing more of the surrounding area (useful if edges of the edit are being clipped). Negative values erode the mask inward, trimming the edges (useful if too much background is being pulled in)."
                ),
                io.Float.Input(
                    "feather_pct",
                    default=2.0, min=0.0, max=10.0, step=0.25,
                    tooltip="How gradually the edit blends into the original at the mask boundary. Higher values create a wider, softer transition (smoother blending, but may wash out fine edges). Lower values create a sharper, more abrupt cutover (crisper edges, but seams may be more visible)."
                ),
                io.Combo.Input(
                    "flow_quality",
                    options=["medium", "fast", "ultrafast"],
                    default="medium",
                    tooltip="Accuracy of the optical flow alignment between original and generated images. Higher quality = more precise change detection and alignment (slower). Lower quality = faster processing but may miss subtle shifts or produce noisier masks."
                ),
                io.Float.Input(
                    "occlusion_threshold",
                    default=-1.0, min=-1.0, max=20.0, step=0.5,
                    tooltip="Sensitivity to pixels that moved so much they can't be reliably matched between images. Higher values ignore more motion discrepancies (fewer false positives from camera jitter, but may miss real edits). Lower values flag more pixels as changed (catches more edits, but may over-detect in noisy areas). Set to -1 for automatic tuning."
                ),
                io.Float.Input(
                    "close_radius_pct",
                    default=0.5, min=0.0, max=5.0, step=0.1,
                    tooltip="Fills small holes and gaps inside the detected edit region. Higher values close larger gaps (creates a more solid, continuous mask). Lower values leave small holes intact (preserves finer mask detail but may leave speckled artifacts inside the edit)."
                ),
                io.Float.Input(
                    "min_region_pct",
                    default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="Removes small isolated blobs from the mask that are likely false positives. Higher values filter out larger stray regions (cleaner mask, but may discard small intentional edits). Lower values keep smaller regions (preserves tiny edits, but may let through noise)."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="composited_image"),
                io.Mask.Output(display_name="change_mask"),
                io.String.Output(display_name="report"),
            ]
        )

    @classmethod
    def execute(cls, original_image, generated_image,
            delta_e_threshold=-1.0, grow_mask_pct=0.0, feather_pct=2.0,
            flow_quality="medium", occlusion_threshold=-1.0,
            close_radius_pct=0.5, min_region_pct=0.05):

        orig_np = original_image[0].cpu().float().numpy()
        gen_np  = generated_image[0].cpu().float().numpy()

        if orig_np.shape != gen_np.shape:
            H, W = gen_np.shape[:2]
            orig_np = cv2.resize(orig_np.astype(np.float32), (W, H), interpolation=cv2.INTER_LANCZOS4)

        H, W = gen_np.shape[:2]
        diag = math_diag(H, W)
        total_area = H * W

        grow_px    = round(grow_mask_pct * diag / 100.0)
        feather_px = abs(feather_pct) * diag / 100.0
        close_px   = pct_to_px(close_radius_pct, diag)
        min_px     = max(0, round(min_region_pct * total_area / 100.0))

        result, change_mask, stats = composite(
            orig_np, gen_np,
            delta_e_threshold   = delta_e_threshold,
            flow_preset         = FLOW_PRESETS[flow_quality],
            occlusion_threshold = occlusion_threshold,
            grow_px             = grow_px,
            close_radius        = close_px,
            min_region_px       = min_px,
            feather_px          = feather_px,
        )

        report_lines =[
            "=== Klein Edit Composite v2.2 (Global Align) ===",
            f"Resolution:       {stats['resolution']}  (diag {stats['diagonal_px']}px)",
            f"",
        ]

        if "auto_delta_e" in stats:
            report_lines.append(f"ΔE threshold:     AUTO → {stats['auto_delta_e']:.1f}")
        else:
            report_lines.append(f"ΔE threshold:     {delta_e_threshold:.1f}")

        if "auto_occlusion" in stats:
            report_lines.append(f"Occlusion thresh: AUTO → {stats['auto_occlusion']:.1f}")
        else:
            report_lines.append(f"Occlusion thresh: {occlusion_threshold:.1f}")

        report_lines +=[
            f"Grow mask:        {grow_mask_pct:+.1f}% → {grow_px:+d}px",
            f"Feather:          {feather_pct:.1f}% → {feather_px:.0f}px",
            f"Close radius:     {close_radius_pct:.1f}% → {close_px}px",
            f"Min region:       {min_region_pct:.2f}% → {min_px}px",
            f"Flow quality:     {flow_quality}",
            f"",
            f"Changed region:   {stats['changed_pct']:.1f}% of image",
            f"Occluded pixels:  {stats['occluded_px']:,}",
            f"Flow mean shift:  {stats['flow_mean_px']:.2f}px",
            f"Flow p99 shift:   {stats['flow_p99_px']:.2f}px",
            f"Median ΔE:        {stats['median_de']:.2f}",
        ]

        return io.NodeOutput(torch.from_numpy(result).unsqueeze(0),
                torch.from_numpy(change_mask).unsqueeze(0),
                "\n".join(report_lines))


class UC_ImageInwardEdgeFill(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageInwardEdgeFill",
            display_name="Image Inward Edge Fill",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Int.Input(
                    "inpaint_radius",
                    default=3,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="How far the algorithm looks for edge pixels. Higher values are slower but better for large holes."
                ),
                io.Int.Input(
                    "edge_blend_blur",
                    default=9,
                    min=0,
                    max=101,
                    step=2,
                    tooltip="Applies a Gaussian blur to the mask to smoothly blend the filled area with the original edges. 0 disables it."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        inpaint_radius: int,
        edge_blend_blur: int,
    ) -> io.NodeOutput:
        result = fill_mask_from_edges(
            image,
            mask,
            inpaint_radius,
            edge_blend_blur,
        )
        return io.NodeOutput(result)


class UC_ImageIterativeStretchFill(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageIterativeStretchFill",
            display_name="Image Iterative Stretch Fill",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Combo.Input(
                    "stretch_axis",
                    default="auto",
                    options=["auto", "horizontal", "vertical"],
                    tooltip="'Auto' stretches across the narrowest dimension of the current mask."
                ),
                io.Int.Input(
                    "sample_thickness",
                    default=32,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="How many pixels of unmasked image to grab from the edges to stretch inwards."
                ),
                io.Int.Input(
                    "edge_blend_blur",
                    default=9,
                    min=0,
                    max=101,
                    step=2,
                    tooltip="Softens the mask boundary to seamlessly blend the stretched fill."
                ),
                io.Int.Input(
                    "iterations",
                    default=5,
                    min=1,
                    max=50,
                    step=1,
                    tooltip="Number of times to repeat the stretch and fill process."
                ),
                io.Int.Input(
                    "mask_decay_pixels",
                    default=4,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Shrinks the mask by this many pixels each iteration, creating a telescoping fill effect."
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        stretch_axis: str,
        sample_thickness: int,
        edge_blend_blur: int,
        iterations: int,
        mask_decay_pixels: int,
    ) -> io.NodeOutput:
        result = iterative_directional_stretch_fill(
            image,
            mask,
            stretch_axis,
            sample_thickness,
            edge_blend_blur,
            iterations,
            mask_decay_pixels,
        )
        return io.NodeOutput(result)


class UC_TextOverlayNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_TextOverlayNode",
            display_name="Text Overlay",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("text", multiline=True, default="Hello World"),
                io.Int.Input("font_size", default=32, min=1, max=1024),
                io.String.Input("text_color", default="FFFFFF"),
                io.String.Input("bg_color", default="000000"),
                io.Boolean.Input("draw_background", default=True),
                io.Int.Input("bg_padding", default=10, min=0, max=1024),
                io.Float.Input("bg_transparency", default=0.5, min=0.0, max=1.0, step=0.05, tooltip="0.0 is fully transparent, 1.0 is fully opaque"),
                io.Boolean.Input("use_percentage", default=False, tooltip="If True, top/bottom/left/right are treated as percentages (0-100) of the image size."),
                io.Int.Input("top", default=-1, min=-1, max=8192, tooltip="-1 for center vertically or use bottom offset"),
                io.Int.Input("bottom", default=-1, min=-1, max=8192, tooltip="-1 for center vertically or use top offset"),
                io.Int.Input("left", default=-1, min=-1, max=8192, tooltip="-1 for center horizontally or use right offset"),
                io.Int.Input("right", default=-1, min=-1, max=8192, tooltip="-1 for center horizontally or use left offset"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        text: str,
        font_size: int,
        text_color: str,
        bg_color: str,
        draw_background: bool,
        bg_padding: int,
        bg_transparency: float,
        use_percentage: bool,
        top: int,
        bottom: int,
        left: int,
        right: int,
    ) -> io.NodeOutput:

        t_color = hex_to_rgb(text_color, (255, 255, 255))

        # Calculate background color with transparency (alpha 0-255)
        b_color_base = hex_to_rgb(bg_color, (0, 0, 0))
        alpha = int(bg_transparency * 255.0)
        # Ensure b_color is exactly 4 elements long for RGBA
        if len(b_color_base) == 3:
            b_color = (b_color_base[0], b_color_base[1], b_color_base[2], alpha)
        else: # Handle case where hex_to_rgb returns a 4-element tuple or default
            b_color = (b_color_base[0], b_color_base[1], b_color_base[2], alpha)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Linux fallback
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

        # Handle batch of images
        batch_count = image.size(0) if len(image.shape) > 3 else 1
        output_images = []

        for i in range(batch_count):
            img_tensor = image[i] if batch_count > 1 else image[0]
            img_height, img_width = img_tensor.shape[:2]
            overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Calculate text size using textbbox
            left_box, top_box, right_box, bottom_box = draw.textbbox((0, 0), text, font=font)
            text_width = right_box - left_box
            text_height = bottom_box - top_box

            # Calculate total width/height including background padding
            total_width = text_width + (bg_padding * 2 if draw_background else 0)
            total_height = text_height + (bg_padding * 2 if draw_background else 0)

            # Resolve coordinates based on mode (pixels vs percentage)
            def resolve_coord(val, max_val):
                if val == -1:
                    return -1
                if use_percentage:
                    return int((val / 100.0) * max_val)
                return val

            l_resolved = resolve_coord(left, img_width)
            r_resolved = resolve_coord(right, img_width)
            t_resolved = resolve_coord(top, img_height)
            b_resolved = resolve_coord(bottom, img_height)

            # Determine X position
            if l_resolved == -1 and r_resolved == -1:
                x_pos = (img_width - total_width) // 2
            elif l_resolved != -1:
                x_pos = l_resolved
            else: # r_resolved != -1
                x_pos = img_width - total_width - r_resolved

            # Determine Y position
            if t_resolved == -1 and b_resolved == -1:
                y_pos = (img_height - total_height) // 2
            elif t_resolved != -1:
                y_pos = t_resolved
            else: # b_resolved != -1
                y_pos = img_height - total_height - b_resolved

            # Draw background
            if draw_background:
                bg_rect = [x_pos, y_pos, x_pos + total_width, y_pos + total_height]
                draw.rectangle(bg_rect, fill=b_color)

            # Draw text
            text_x = x_pos + (bg_padding if draw_background else 0)
            text_y = y_pos + (bg_padding if draw_background else 0)

            # Use textbbox offset for more accurate vertical alignment of text
            draw.text((text_x - left_box, text_y - top_box), text, fill=(*t_color[:3], 255), font=font)

            overlay_tensor = torch.from_numpy(np.asarray(overlay).copy()).to(device=img_tensor.device, dtype=img_tensor.dtype) / 255.0
            overlay_alpha = overlay_tensor[..., 3:4]
            base = img_tensor[..., :3]
            out_tensor = base * (1.0 - overlay_alpha) + overlay_tensor[..., :3] * overlay_alpha
            output_images.append(out_tensor)

        if batch_count > 1:
            out = torch.stack(output_images, dim=0)
        else:
            out = output_images[0].unsqueeze(0)

        return io.NodeOutput(out)


class UC_ModifyMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ModifyMask",
            category="utils/mask",
            display_name="Modify Mask",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input(
                    "expand", default=0, max=MAX_RESOLUTION, min=-MAX_RESOLUTION, step=1, tooltip="Pixels used to grow each mask; negative values shrink it."
                ),
                io.Float.Input(
                    "incremental_expandrate", default=0.0, max=100.0, min=0.0, step=0.01, tooltip="Additional grow or shrink amount applied after each mask in the batch."
                ),
                io.Boolean.Input("tapered_corners", default=True, tooltip="Uses a cross-shaped kernel for more tapered corners; disabled uses a square kernel."),
                io.Boolean.Input("flip_input", default=False, tooltip="Inverts the input mask before all other processing."),
                io.Float.Input(
                    "blur_radius", default=0.0, max=100.0, min=0.0, step=0.01, tooltip="Gaussian blur radius applied after grow or shrink; zero disables blur."
                ),
                io.Float.Input("lerp_alpha", default=1.0, max=1.0, min=0.0, step=0.01, tooltip="Blends each batch mask with the previous processed mask; 1 uses only the current mask."),
                io.Float.Input(
                    "decay_factor", default=1.0, max=1.0, min=0.0, step=0.01, tooltip="Adds a decayed previous batch mask before normalization; 1 preserves its full contribution."
                ),
                io.Boolean.Input("fill_holes", default=False, optional=True, tooltip="Fills enclosed zero-valued holes after grow or shrink."),
                io.Float.Input("lower_clamp", default=0.0, max=100.0, min=0.0, step=0.1, tooltip="Minimum output mask value as a percentage."),
                io.Float.Input("upper_clamp", default=100.0, max=100.0, min=0.0, step=0.1, tooltip="Maximum output mask value as a percentage."),
            ],
            outputs=[
                io.Mask.Output(display_name="Mask"),
                io.Mask.Output(display_name="Inverted Mask"),
            ],
        )

    @classmethod
    def execute(
        self,
        mask,
        expand,
        tapered_corners,
        flip_input,
        blur_radius,
        incremental_expandrate,
        lerp_alpha,
        decay_factor,
        fill_holes=False,
        lower_clamp=0.0,
        upper_clamp=100.0,
    ):

        alpha = lerp_alpha
        decay = decay_factor

        # 1. Clone the original mask to keep a reference to the un-blurred pixels
        original_mask_input = mask.clone()

        if flip_input:
            mask = 1.0 - mask
            original_mask_input = 1.0 - original_mask_input

        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))

        # Prepare original mask for processing loop (match dimensions)
        original_mask_batches = original_mask_input.reshape(
            (-1, mask.shape[-2], mask.shape[-1])
        )

        out = []
        previous_output = None
        current_expand = expand
        for m in tqdm(growmask, desc="Expanding/Contracting Mask"):
            output = (
                m.unsqueeze(0).unsqueeze(0).to(model_management.get_torch_device())
            )  # Add batch and channel dims for kornia
            if abs(round(current_expand)) > 0:
                # Create kernel - kornia expects kernel on same device as input
                if tapered_corners:
                    kernel = torch.tensor(
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        dtype=torch.float32,
                        device=output.device,
                    )
                else:
                    kernel = torch.tensor(
                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        dtype=torch.float32,
                        device=output.device,
                    )

                for _ in range(abs(round(current_expand))):
                    if current_expand < 0:
                        output = morph.erosion(output, kernel)
                    else:
                        output = morph.dilation(output, kernel)

            output = output.squeeze(0).squeeze(0)  # Remove batch and channel dims

            if current_expand < 0:
                current_expand -= abs(incremental_expandrate)
            else:
                current_expand += abs(incremental_expandrate)

            if fill_holes:
                binary_mask = output > 0
                output_np = binary_mask.cpu().numpy()
                filled = scipy.ndimage.binary_fill_holes(output_np)
                output = torch.from_numpy(filled.astype(np.float32)).to(output.device)

            if alpha < 1.0 and previous_output is not None:
                output = alpha * output + (1 - alpha) * previous_output
            if decay < 1.0 and previous_output is not None:
                output += decay * previous_output
                output = output / output.max()
            previous_output = output
            out.append(output.cpu())

        if blur_radius != 0:
            for idx, tensor in enumerate(out):
                blurred_tensor = gaussian_blur_nchw(
                    tensor.unsqueeze(0).unsqueeze(0), float(blur_radius)
                )[0, 0].unsqueeze(0)
                if current_expand > 0:
                    original_slice = original_mask_batches[idx].unsqueeze(0).cpu()
                    blurred_tensor = torch.max(blurred_tensor, original_slice)
                elif current_expand < 0:
                    original_slice = original_mask_batches[idx].unsqueeze(0).cpu()
                    blurred_tensor = torch.min(blurred_tensor, original_slice)
                out[idx] = blurred_tensor
            blurred = torch.cat(out, dim=0)
            if lower_clamp > 0.0:
                blurred = torch.max(blurred, torch.tensor(lower_clamp / 100.0, device=blurred.device))
            if upper_clamp < 100.0:
                blurred = torch.min(blurred, torch.tensor(upper_clamp / 100.0, device=blurred.device))
            mask = blurred
            mask_inverted = 1.0 - blurred
            return io.NodeOutput(mask, mask_inverted)
        else:
            mask = torch.stack(out, dim=0)
            if lower_clamp > 0.0:
                mask = torch.max(mask, torch.tensor(lower_clamp / 100.0, device=mask.device))
            if upper_clamp < 100.0:
                mask = torch.min(mask, torch.tensor(upper_clamp / 100.0, device=mask.device))
            mask_inverted = 1.0 - mask
            return io.NodeOutput(mask, mask_inverted)


class UC_MaskToBoundingBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MaskToBoundingBox",
            display_name="Mask to Bounding Box",
            category="utils/mask",
            description="Finds the smallest bounding box containing all nonzero mask pixels.",
            inputs=[
                io.Mask.Input("mask"),
                io.Boolean.Input("invert", default=False),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional image cropped to the detected bounding box.",
                ),
            ],
            outputs=[
                io.BoundingBox.Output("bounding_box", display_name="bounding box"),
                io.Image.Output("image", display_name="cropped image"),
            ],
        )

    @classmethod
    def execute(cls, mask, invert, image=None) -> io.NodeOutput:
        bounding_box, cropped_image = mask_to_bounding_box(mask, invert, image)
        return io.NodeOutput(bounding_box, cropped_image)


def _blend_luminosity(value):
    return value[..., 0:1] * 0.3 + value[..., 1:2] * 0.59 + value[..., 2:3] * 0.11


def _blend_saturation(value):
    return value.max(dim=-1, keepdim=True).values - value.min(dim=-1, keepdim=True).values


def _clip_blend_color(value):
    luminosity = _blend_luminosity(value)
    minimum = value.min(dim=-1, keepdim=True).values
    maximum = value.max(dim=-1, keepdim=True).values
    below = luminosity + (value - luminosity) * luminosity / (luminosity - minimum).clamp_min(1e-12)
    above = luminosity + (value - luminosity) * (1.0 - luminosity) / (maximum - luminosity).clamp_min(1e-12)
    value = torch.where(minimum < 0.0, below, value)
    return torch.where(maximum > 1.0, above, value)


def _set_blend_luminosity(value, luminosity):
    return _clip_blend_color(value + luminosity - _blend_luminosity(value))


def _set_blend_saturation(value, saturation):
    minimum = value.min(dim=-1, keepdim=True).values
    maximum = value.max(dim=-1, keepdim=True).values
    spread = maximum - minimum
    normalized = (value - minimum) * saturation / spread.clamp_min(1e-12)
    return torch.where(spread > 1e-12, normalized, torch.zeros_like(value))


def _blend_rgb(destination, source, mode):
    destination = destination.clamp(0.0, 1.0)
    source = source.clamp(0.0, 1.0)
    if mode == "add":
        return source
    if mode == "multiply":
        return destination * source
    if mode == "screen":
        return destination + source - destination * source
    if mode == "darken":
        return torch.minimum(destination, source)
    if mode == "lighten":
        return torch.maximum(destination, source)
    if mode == "difference":
        return torch.abs(destination - source)
    if mode == "exclusion":
        return destination + source - 2.0 * destination * source
    if mode == "overlay":
        return torch.where(destination <= 0.5, 2.0 * destination * source, 1.0 - 2.0 * (1.0 - destination) * (1.0 - source))
    if mode == "hard_light":
        return torch.where(source <= 0.5, 2.0 * destination * source, 1.0 - 2.0 * (1.0 - destination) * (1.0 - source))
    if mode == "color_dodge":
        return torch.where(source >= 1.0, torch.ones_like(source), destination / (1.0 - source).clamp_min(1e-12)).clamp(0.0, 1.0)
    if mode == "color_burn":
        return torch.where(source <= 0.0, torch.zeros_like(source), 1.0 - (1.0 - destination) / source.clamp_min(1e-12)).clamp(0.0, 1.0)
    if mode == "soft_light":
        curve = torch.where(
            destination <= 0.25,
            ((16.0 * destination - 12.0) * destination + 4.0) * destination,
            torch.sqrt(destination),
        )
        return torch.where(
            source <= 0.5,
            destination - (1.0 - 2.0 * source) * destination * (1.0 - destination),
            destination + (2.0 * source - 1.0) * (curve - destination),
        )
    if mode == "hue":
        return _set_blend_luminosity(_set_blend_saturation(source, _blend_saturation(destination)), _blend_luminosity(destination))
    if mode == "color":
        return _set_blend_luminosity(source, _blend_luminosity(destination))
    raise ValueError(f"Unsupported image blend mode: {mode!r}.")


class UC_ImageBlendByMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_ImageBlendByMask",
            category="utils/mask",
            display_name="Image Blend by Mask",
            inputs=[
                io.Image.Input("destination", tooltip="Base image retained where the mask or blend strength is zero."),
                io.Image.Input("source", tooltip="Image combined with the destination using the selected blend mode."),
                io.Combo.Input(
                    "mode",
                    options=[
                        "add",
                        "color",
                        "color_burn",
                        "color_dodge",
                        "darken",
                        "difference",
                        "exclusion",
                        "hard_light",
                        "hue",
                        "lighten",
                        "multiply",
                        "overlay",
                        "screen",
                        "soft_light",
                    ],
                    default="add",
                    tooltip="Color blend operation applied between the source and destination images.",
                ),
                io.Float.Input(
                    "blend_percentage", default=1.0, max=1.0, min=0.0, step=0.01, tooltip="Overall blend strength multiplied by the optional mask."
                ),
                io.Boolean.Input("resize_source", default=False, tooltip="Resizes the source to destination dimensions with bicubic interpolation; otherwise dimensions must match."),
                io.Mask.Input("mask", tooltip="Controls where the blend is applied; it is resized and singleton batches broadcast to the image batch."),
            ],
            outputs=[
                io.Image.Output(display_name="blended_image"),
            ],
        )

    @classmethod
    def execute(
        self,
        destination,
        source,
        mode="add",
        blend_percentage=1.0,
        resize_source=False,
        mask=None,
    ):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination[..., :3]
        source = source[..., :3].to(destination)

        if resize_source:
            source = resize_nchw(source.movedim(-1, 1), destination.shape[2], destination.shape[1], "bicubic").movedim(1, -1)
        if source.shape[1:3] != destination.shape[1:3]:
            raise ValueError("Source and destination dimensions must match unless resize_source is enabled.")
        batch_size = max(destination.shape[0], source.shape[0])
        if destination.shape[0] not in (1, batch_size) or source.shape[0] not in (1, batch_size):
            raise ValueError("Source and destination batch sizes must match or be 1.")
        destination = destination.expand(batch_size, -1, -1, -1)
        source = source.expand(batch_size, -1, -1, -1)
        blended = _blend_rgb(destination, source, mode).clamp(0.0, 1.0)
        weight = destination.new_full((batch_size, 1, 1, 1), float(blend_percentage))
        if mask is not None:
            mask = mask.to(destination)
            if mask.shape[-2:] != destination.shape[1:3]:
                mask = resize_nchw(mask.unsqueeze(1), destination.shape[2], destination.shape[1], "bilinear").squeeze(1)
            if mask.shape[0] not in (1, batch_size):
                raise ValueError("Mask batch size must match the image batch size or be 1.")
            weight = weight * mask.expand(batch_size, -1, -1).clamp(0.0, 1.0).unsqueeze(-1)
        return io.NodeOutput(destination * (1.0 - weight) + blended * weight)

class UC_ImagePad(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImagePad",
            display_name="Image Pad",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("left", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Pixels added to the left when target dimensions are not connected."),
                io.Int.Input("right", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Pixels added to the right when target dimensions are not connected."),
                io.Int.Input("top", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Pixels added above the image when target dimensions are not connected."),
                io.Int.Input("bottom", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Pixels added below the image when target dimensions are not connected."),
                io.Int.Input("extra_padding", default=0, min=0, max=MAX_RESOLUTION, step=1, tooltip="Additional padding on every side, or an inset margin when target dimensions are connected."),
                io.Combo.Input(
                    "pad_mode",
                    default="edge",
                    options=["edge", "edge_pixel", "color", "pillarbox_blur"],
                    tooltip="edge: Extend the edge pixels. edge_pixel: Extend the edge pixels but keep the original edge pixel color. color: Fill with a solid color. pillarbox_blur: Fill with a blurred version of the image."
                ),
                io.String.Input("color", multiline=False, default="0, 0, 0", tooltip="Color as RGB values in range 0-255 or 0.0-1.0, or color name or hex code"),
                io.Mask.Input("mask", optional=True, tooltip="Optional source mask padded with value 1 outside the image; disconnected creates a mask for the added border."),
                io.Int.Input("target_width", default=512, min=0, max=MAX_RESOLUTION, step=1, force_input=True, optional=True, tooltip="Optional final width. Connect both target dimensions to proportionally fit and center the image."),
                io.Int.Input("target_height", default=512, min=0, max=MAX_RESOLUTION, step=1, force_input=True, optional=True, tooltip="Optional final height. Connect both target dimensions to proportionally fit and center the image."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )
    # DESCRIPTION = "Pad the input image and optionally mask with the specified padding."

    @classmethod
    def execute(cls, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Parse background color using helper function
        color_list = string_to_color(color)
        bg_color = [x / 255.0 for x in color_list]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            # Rescale to fit target boundaries proportionally
            eff_target_w = max(1, target_width - 2 * extra_padding) if extra_padding > 0 else target_width
            eff_target_h = max(1, target_height - 2 * extra_padding) if extra_padding > 0 else target_height

            # Compute aspect-ratio preserving scaling factor
            scale_factor = min(eff_target_w / W, eff_target_h / H)

            new_W = max(1, round(W * scale_factor))
            new_H = max(1, round(H * scale_factor))

            # Upscale/downscale the image tensor
            image = resize_nchw(image.movedim(-1, 1), new_W, new_H, "lanczos").movedim(1, -1)
            B, H, W, C = image.shape

            # Upscale/downscale the mask tensor if provided
            if mask is not None:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = resize_nchw(src_b, bg_w, bg_h, "bilinear")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
                if pad_mode == "edge":
                    # Pad with edge color (mean)
                    top_edge = image[b, 0, :, :]
                    bottom_edge = image[b, H-1, :, :]
                    left_edge = image[b, :, 0, :]
                    right_edge = image[b, :, W-1, :]
                    out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                    out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                    out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                    out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                elif pad_mode == "edge_pixel":
                    # Pad with exact edge pixel values
                    for y in range(pad_top):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                    for y in range(pad_top+H, padded_height):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, H-1, :, :]
                    for x in range(pad_left):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                    for x in range(pad_left+W, padded_width):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                    out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                    out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                    out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                    out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                else:
                    # Pad with specified background color
                    out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=1.0
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return io.NodeOutput(out_image, out_masks)


class UC_NoHaloLoHaloDownscale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_NoHaloLoHaloDownscale",
            display_name="NoHalo / LoHalo Downscale",
            category="advanced/image",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("method", options=["lohalo", "nohalo"], default="lohalo"),
                io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                io.Int.Input("multiple", default=16, min=4, max=128, step=4),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, method: str, megapixels: float, multiple: int) -> io.NodeOutput:
        return io.NodeOutput(downscale_nohalo_lohalo(image, method, megapixels, multiple))


class UC_HighResolutionTileSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_HighResolutionTileSplit",
            display_name="High Resolution Tile Split & VAE Encode",
            category="image/tiling",
            description=(
                "Splits one pre-upscaled image into overlapping tiles, then "
                "VAE-encodes every tile sequentially for list-mapped sampling."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Vae.Input("vae"),
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Optional diffusion model. Required only when a "
                        "Differential Diffusion mode is enabled."
                    ),
                ),
                io.Combo.Input(
                    "tile_mode",
                    options=["tile_size", "grid"],
                    default="tile_size",
                    tooltip=(
                        "tile_size uses tile width and height. grid uses rows "
                        "and columns. Controls for the other mode are ignored."
                    ),
                ),
                io.Int.Input(
                    "tile_width",
                    default=1024,
                    min=64,
                    max=MAX_RESOLUTION,
                    step=8,
                    tooltip="Processed tile width in tile_size mode.",
                ),
                io.Int.Input(
                    "tile_height",
                    default=1024,
                    min=64,
                    max=MAX_RESOLUTION,
                    step=8,
                    tooltip="Processed tile height in tile_size mode.",
                ),
                io.Int.Input(
                    "rows",
                    default=2,
                    min=1,
                    max=256,
                    step=1,
                    tooltip="Number of tile rows in grid mode.",
                ),
                io.Int.Input(
                    "columns",
                    default=2,
                    min=1,
                    max=256,
                    step=1,
                    tooltip="Number of tile columns in grid mode.",
                ),
                io.Int.Input(
                    "overlap",
                    default=128,
                    min=0,
                    max=MAX_RESOLUTION // 2,
                    step=8,
                    tooltip=(
                        "Actual shared pixels between neighboring tiles in "
                        "both tiling modes."
                    ),
                ),
                io.Combo.Input(
                    "mask_profile",
                    options=["cosine", "linear"],
                    default="cosine",
                    tooltip="Transition profile used for overlap denoising and reconstruction.",
                ),
                io.Float.Input(
                    "feather_width",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Fraction of each overlap occupied by the mask transition.",
                ),
                io.Float.Input(
                    "mask_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Overlap edge protection. 0 leaves the mask flat; "
                        "1 reaches zero at protected internal tile edges."
                    ),
                ),
                io.Combo.Input(
                    "differential_diffusion_mode",
                    options=["off", "core", "advanced"],
                    default="off",
                    tooltip=(
                        "off leaves the model unchanged. core applies "
                        "ComfyUI Core Differential Diffusion. advanced uses "
                        "the threshold multiplier control."
                    ),
                ),
                io.Float.Input(
                    "differential_diffusion_value",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.001,
                    tooltip=(
                        "Mode-dependent value. Core: strength from 0 to 1, "
                        "blending its progressive binary mask with the soft "
                        "mask. Advanced: nonzero threshold divisor from -10 "
                        "to 10, matching KJNodes Differential Diffusion Advanced."
                    ),
                ),
                io.Image.Input(
                    "depth_map",
                    optional=True,
                    tooltip=(
                        "Optional grayscale depth image used to preserve "
                        "structure across solid and feathered denoise-mask "
                        "regions. It is converted to luminance and resized to "
                        "the source image."
                    ),
                ),
                io.Float.Input(
                    "depth_influence",
                    default=1.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Signed depth-mask influence. Positive values use the "
                        "map as supplied, negative values use its inverse, and "
                        "the magnitude controls the effect. 0 disables depth "
                        "modulation."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    "tile_images",
                    display_name="tile images",
                    is_output_list=True,
                    tooltip=(
                        "The exact padded image tensors sent to VAE encoding, "
                        "for spatially matched list-mapped visual conditioning."
                    ),
                ),
                io.Latent.Output(
                    "tile_latents",
                    display_name="tile latents",
                    is_output_list=True,
                    tooltip="Matching VAE latents with Core-compatible noise masks.",
                ),
                HighResolutionTileLayout.Output(
                    "tile_layout",
                    display_name="tile layout",
                    tooltip="Coordinate and overlap metadata for the tile accumulator.",
                ),
                io.Model.Output(
                    "model",
                    display_name="model",
                    tooltip=(
                        "Differential Diffusion model for the sampler guider, "
                        "or the unchanged connected model when mode is off."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        vae,
        tile_mode,
        tile_width,
        tile_height,
        rows,
        columns,
        overlap,
        mask_profile,
        feather_width,
        mask_strength,
        model=None,
        differential_diffusion_mode="off",
        differential_diffusion_value=1.0,
        depth_map=None,
        depth_influence=1.0,
    ) -> io.NodeOutput:
        image_tiles, latent_tiles, layout = split_and_encode_tiles(
            image,
            vae,
            tile_mode,
            tile_width,
            tile_height,
            rows,
            columns,
            overlap,
            mask_profile,
            feather_width,
            mask_strength,
            depth_map,
            depth_influence,
        )
        output_model = apply_tile_differential_diffusion(
            model,
            differential_diffusion_mode,
            differential_diffusion_value,
        )
        return io.NodeOutput(image_tiles, latent_tiles, layout, output_model)


class UC_HighResolutionTileAccumulator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_HighResolutionTileAccumulator",
            display_name="High Resolution Tile Accumulator",
            category="image/tiling",
            description=(
                "Collects a completed decoded tile list and reconstructs the "
                "original canvas with normalized overlap blending."
            ),
            is_input_list=True,
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Decoded image list produced by the mapped sampler path.",
                ),
                HighResolutionTileLayout.Input(
                    "tile_layout",
                    tooltip="Layout from High Resolution Tile Split & VAE Encode.",
                ),
            ],
            outputs=[io.Image.Output("image", display_name="image")],
        )

    @classmethod
    def execute(cls, images, tile_layout) -> io.NodeOutput:
        return io.NodeOutput(accumulate_tile_images(images, tile_layout))


class UC_ListToImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ListToImageBatch",
            display_name="List to Image Batch (High Performance)",
            category="utils",
            is_input_list=True,  # Tells ComfyUI to pass the list itself instead of auto-iterating!
            inputs=[
                io.Image.Input("images"),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ]
        )

    @classmethod
    def execute(cls, images):
        if not images:
            # Fallback: single empty black image
            black_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(black_image)

        # Filter out None values
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            black_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(black_image)

        # Check if all images have the exact same shape.
        # This is the fast-path! It avoids any interpolation/upscaling overhead.
        first_shape = valid_images[0].shape
        same_shape = all(img.shape == first_shape for img in valid_images)

        if same_shape:
            # Extremely fast path: direct concat along batch dimension
            batched = torch.cat(valid_images, dim=0)
            return io.NodeOutput(batched)

        # Slow-path (only if dimensions or channels differ)
        # Pad channels if they differ
        max_channels = max(image.shape[-1] for image in valid_images)
        padded_images = []
        for image in valid_images:
            if image.shape[-1] < max_channels:
                pad_size = max_channels - image.shape[-1]
                padded = torch.nn.functional.pad(image, (0, pad_size), mode='constant', value=1.0)
                padded_images.append(padded)
            else:
                padded_images.append(image)

        # Resize all to match the first image's height and width
        first_h, first_w = first_shape[1], first_shape[2]
        resized_images = []
        for image in padded_images:
            if image.shape[1] != first_h or image.shape[2] != first_w:
                # permute to [B, C, H, W] for interpolate
                img_perm = image.movedim(-1, 1)
                method = "area" if first_h < image.shape[1] or first_w < image.shape[2] else "bicubic"
                img_resized = resize_nchw(img_perm, first_w, first_h, method)
                resized_images.append(img_resized.movedim(1, -1))
            else:
                resized_images.append(image)

        batched = torch.cat(resized_images, dim=0)
        return io.NodeOutput(batched)


