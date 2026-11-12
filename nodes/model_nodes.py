import json

from comfy_api.latest import io

from nodes import MAX_RESOLUTION

from ..helpers.image_helpers import DEPTH_ANYTHING_MULTI_VIEW_MODES, DEPTH_ANYTHING_NORM_SCOPES, DEPTH_ANYTHING_OUTPUT_MODES, DEPTH_ANYTHING_REF_VIEW_STRATEGIES, DEPTH_ANYTHING_RESIZE_METHODS, VIDEO_FRAME_TIMESTAMP_FORMATS, build_yolo_detectors, run_depth_anything, run_yolo_detect
from ..helpers.model_helpers import (
    DEPTH_ANYTHING_MODELS,
    WHISPER_LANGUAGES,
    WHISPER_MODELS,
    YOLO_MODELS,
    apply_minimax_h3_refs_to_conditioning,
    create_minimax_h3_audio_ref,
    create_minimax_h3_image_refs,
    create_minimax_h3_video_ref,
    depth_anything_forward,
    detect_sam3,
    flatten_minimax_h3_ref_collections,
    format_minimax_h3_ref_info,
    format_reference_transcript_phrases,
    get_minimax_h3_latent_upscale_model_names,
    get_minimax_h3_ref_input_fingerprint,
    list_minimax_h3_refs,
    load_depth_anything_model,
    load_minimax_h3_latent_upscale_model,
    load_minimax_h3_ref_collection,
    load_whisper_model,
    load_yolo_model,
    minimax_h3_ref_resolution_grid,
    register_depth_anything_paths,
    register_whisper_paths,
    register_yolo_paths,
    run_whisper,
    save_minimax_h3_ref_collection,
    upscale_minimax_h3_latent,
    yolo_forward,
)

WhisperModel = io.Custom("WHISPER_MODEL")
BBoxDetectorType = io.Custom("BBOX_DETECTOR")
SegmDetectorType = io.Custom("SEGM_DETECTOR")
DepthAnythingModel = io.Custom("DEPTH_ANYTHING_MODEL")
register_whisper_paths()
register_yolo_paths()
register_depth_anything_paths()


class UC_WhisperLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_WhisperLoader",
            display_name="Whisper Loader",
            category="utils/audio",
            description="Loads native Whisper safetensors through UEL and ComfyUI model management.",
            inputs=[io.Combo.Input("model_name", options=list(WHISPER_MODELS), default="base", tooltip="Uses models/whisper. Executing downloads only the selected model if missing.")],
            outputs=[WhisperModel.Output("whisper_model")],
        )

    @classmethod
    def execute(cls, model_name="base"):
        return io.NodeOutput(load_whisper_model(model_name))


class UC_WhisperTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_WhisperTranscribe",
            display_name="Whisper Transcribe",
            category="utils/audio",
            description="Transcribes full recordings or translates speech to English; one aligned result per audio batch item.",
            inputs=[
                WhisperModel.Input("whisper_model"),
                io.Audio.Input("audio"),
                io.Combo.Input("task", options=["transcribe", "translate"], default="transcribe"),
                io.Combo.Input("language", options=["auto", *WHISPER_LANGUAGES], default="auto", tooltip="Spoken language code, or automatic detection. Translation outputs English."),
                io.Boolean.Input("word_timestamps", default=True, optional=True, tooltip="Compute cross-attention alignment. Adds words with start/end seconds and confidence to each segment; required for syllable-counted phrase lines."),
                io.Combo.Input("timestamp_format", options=list(VIDEO_FRAME_TIMESTAMP_FORMATS), default="00.000s", optional=True, tooltip="Timestamp formatting for formatted phrase lines, matching the timeline nodes."),
            ],
            outputs=[
                io.String.Output("text", is_output_list=True),
                io.String.Output("segments", is_output_list=True, tooltip="JSON array of start/end seconds and text. With word timestamps enabled, each segment also has words containing word, start, end, and probability."),
                io.String.Output("language", is_output_list=True),
                io.String.Output("formatted_transcription", is_output_list=True, tooltip="Word-timed phrases split before uppercase words and after punctuation, partitioned into 4–6 syllable lines with [start–end] timestamps matching H3 Reference Video Components."),
            ],
        )

    @classmethod
    def execute(cls, whisper_model, audio, task="transcribe", language="auto", word_timestamps=True, timestamp_format="00.000s"):
        transcripts, segments, languages = run_whisper(whisper_model, audio, task, language, word_timestamps=word_timestamps)
        formatted = [format_reference_transcript_phrases(json.loads(seg), lang, timestamp_format=timestamp_format) for seg, lang in zip(segments, languages)]
        return io.NodeOutput(transcripts, segments, languages, formatted)


MiniMaxH3Ref = io.Custom("MINIMAX_H3_REF")


class UC_MiniMaxH3RefExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        image_compression_options = [
            io.DynamicCombo.Option(
                key="pooled",
                inputs=[
                    io.Int.Input(
                        "reference_resolution",
                        display_name="Reference resolution",
                        default=256,
                        min=32,
                        step=32,
                        tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source.",
                    )
                ],
            ),
            io.DynamicCombo.Option(key="encode", inputs=[]),
            io.DynamicCombo.Option(
                key="refined",
                inputs=[
                    io.Int.Input(
                        "reference_resolution",
                        display_name="Reference resolution",
                        default=256,
                        min=32,
                        step=32,
                        tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source.",
                    ),
                    io.Int.Input("refine_steps", default=100, min=1, step=1, tooltip="Suggested starting range: 50–200 steps. Start at 100; use 50 for a quicker attempt, or try 200 if the compressed reference loses detail. Higher values take longer without guaranteeing an improvement."),
                ],
            ),
        ]
        video_compression_options = [
            io.DynamicCombo.Option(key="encode", inputs=[]),
            io.DynamicCombo.Option(
                key="pooled",
                inputs=[
                    io.Int.Input(
                        "reference_resolution",
                        display_name="Reference resolution",
                        default=256,
                        min=32,
                        step=32,
                        tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source.",
                    ),
                    io.Int.Input(
                        "temporal_density", default=16, min=1, step=1, tooltip="How densely the reference represents time across the video. Higher values keep finer motion detail; lower values blend more moments together. The full clip remains covered. Start with 4–16; try 32 or more for longer or complex clips."
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="refined",
                inputs=[
                    io.Int.Input(
                        "reference_resolution",
                        display_name="Reference resolution",
                        default=256,
                        min=32,
                        step=32,
                        tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source.",
                    ),
                    io.Int.Input(
                        "temporal_density", default=16, min=1, step=1, tooltip="How densely the reference represents time across the video. Higher values keep finer motion detail; lower values blend more moments together. The full clip remains covered. Start with 4–16; try 32 or more for longer or complex clips."
                    ),
                    io.Int.Input("refine_steps", default=100, min=1, step=1, tooltip="Suggested starting range: 50–200 steps. Start at 100; use 50 for a quicker attempt, or try 200 if the compressed reference loses detail. Higher values take longer without guaranteeing an improvement."),
                ],
            ),
        ]
        media_type_options = [
            io.DynamicCombo.Option(key="image", inputs=[io.DynamicCombo.Input("compression", options=image_compression_options, tooltip="encode keeps the most detail. pooled makes a smaller reference but loses detail. refined spends extra time improving the compressed reference.")]),
            io.DynamicCombo.Option(key="video", inputs=[io.DynamicCombo.Input("compression", options=video_compression_options, tooltip="encode keeps the most detail. pooled makes a smaller reference but loses detail. refined spends extra time improving the compressed reference.")]),
        ]
        return io.Schema(
            node_id="UC_MiniMaxH3RefExtract",
            display_name="MiniMax H3 Ref Extract",
            category="model/minimax_h3",
            description="Fuses an image batch into one reusable MiniMax H3 reference, or encodes one video clip.",
            search_aliases=["minimax", "h3", "reference", "ref", "image", "video"],
            inputs=[
                io.Image.Input("images", tooltip="Image mode fuses the batch into one reference. Video mode treats the batch as one 24 fps clip; provide at least five frames."),
                io.Vae.Input("vae", display_name="visual vae", tooltip="Connect the MiniMax H3 video VAE for both images and video."),
                io.DynamicCombo.Input("media_type", options=media_type_options, tooltip="Choose image to fuse multiple stills into one reference, or video for one temporal clip."),
                io.String.Input("description", default="", multiline=True, dynamic_prompts=False, tooltip="Optional notes to save with the reference. These notes do not change your prompt."),
                io.Clip.Input("clip", optional=True, tooltip="Optional MiniMax H3 qwen3vl_32b encoder. Stores independently encoded Qwen conditioning with each visual reference."),
                io.Int.Input("vlm_resolution", default=384, min=0, max=4096, step=32, optional=True, tooltip="Qwen image target. 256–4096 resizes; other values keep original resolution. Used only with clip."),
                io.Int.Input("vlm_reference_start", default=17, min=17, step=1, optional=True, tooltip="Qwen Picture/Video number stored in this fused reference. Use distinct numbers when combining refs extracted in separate runs."),
                io.Boolean.Input("fuse_images", default=True, optional=True, tooltip="When multiple images are provided: True fuses them into a single blended reference; False creates separate references for each image."),
            ],
            outputs=[MiniMaxH3Ref.Output("ref", display_name="ref", tooltip="Connect to Ref Save to keep these references, or Ref Apply to use them."), io.String.Output("info", display_name="info", tooltip="Summary of the created references, including any video frames left out.")],
        )

    @classmethod
    def execute(cls, images, vae, media_type, description="", clip=None, vlm_resolution=384, vlm_reference_start=17, fuse_images=True) -> io.NodeOutput:
        kind = media_type.get("media_type")
        compression = media_type.get("compression", {})
        mode = compression.get("compression")
        grid_long_edge = minimax_h3_ref_resolution_grid(compression["reference_resolution"]) if mode != "encode" else 16
        if kind == "image":
            refs = create_minimax_h3_image_refs(images, vae, mode, grid_long_edge, compression.get("refine_steps", 100), description, clip, vlm_resolution, vlm_reference_start, fuse_images=fuse_images)
        else:
            refs = [create_minimax_h3_video_ref(images, vae, mode, grid_long_edge, compression.get("temporal_density", 16), compression.get("refine_steps", 100), description, clip, vlm_resolution, vlm_reference_start)]
        return io.NodeOutput(refs, format_minimax_h3_ref_info(refs))


class UC_MiniMaxH3AudioRefExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3AudioRefExtract",
            display_name="MiniMax H3 Audio Ref Extract",
            category="model/minimax_h3",
            description="Creates a reusable MiniMax H3 reference from an audio clip.",
            search_aliases=["minimax", "h3", "reference", "ref", "audio"],
            inputs=[
                io.Audio.Input("audio", tooltip="Connect one mono or stereo audio clip. Audio references do not guarantee the same voice in generated output."),
                io.Vae.Input("audio_vae", display_name="audio vae", tooltip="Connect the MiniMax H3 audio VAE, not the video VAE."),
                io.String.Input("description", default="", multiline=True, dynamic_prompts=False, tooltip="Optional notes to save with the audio reference. These notes do not change your prompt."),
            ],
            outputs=[MiniMaxH3Ref.Output("ref", display_name="ref", tooltip="Connect to Ref Save to keep this audio reference, or Ref Apply to use it."), io.String.Output("info", display_name="info", tooltip="Summary of the created audio reference.")],
        )

    @classmethod
    def execute(cls, audio, audio_vae, description="") -> io.NodeOutput:
        refs = [create_minimax_h3_audio_ref(audio, audio_vae, description)]
        return io.NodeOutput(refs, format_minimax_h3_ref_info(refs))


class UC_MiniMaxH3RefLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3RefLoad",
            display_name="MiniMax H3 Ref Load",
            category="model/minimax_h3",
            description="Loads one reference or every member of a saved MiniMax H3 Ref bundle.",
            search_aliases=["minimax", "h3", "reference", "ref", "load"],
            inputs=[io.Combo.Input("filename", options=list_minimax_h3_refs(), tooltip="Choose a saved reference or bundle from ComfyUI/models/minimax_h3_refs or your configured reference folders.")],
            outputs=[MiniMaxH3Ref.Output("ref", display_name="ref", tooltip="Connect to Ref Apply to use this reference, or Ref Save to save another copy."), io.String.Output("info", display_name="info", tooltip="Summary of the loaded reference.")],
        )

    @classmethod
    def fingerprint_inputs(cls, filename):
        return get_minimax_h3_ref_input_fingerprint(filename)

    @classmethod
    def execute(cls, filename) -> io.NodeOutput:
        refs = load_minimax_h3_ref_collection(filename)
        return io.NodeOutput(refs, format_minimax_h3_ref_info(refs))


class UC_MiniMaxH3RefSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3RefSave",
            display_name="MiniMax H3 Ref Save",
            category="model/minimax_h3",
            description="Saves connected references separately or together in one bundle file. No sampler is needed.",
            search_aliases=["minimax", "h3", "reference", "ref", "save"],
            inputs=[
                io.String.Input("filename_prefix", default="ref/MiniMax_H3", tooltip="Name for the saved files. Use a slash to add a subfolder. Files are numbered automatically and existing files are not overwritten."),
                io.Autogrow.Input("refs", template=io.Autogrow.TemplatePrefix(MiniMaxH3Ref.Input("ref", optional=True), prefix="ref_", min=1, max=100), tooltip="Connect one or more references to save. Their input order is preserved in either save layout."),
                io.Combo.Input("save_layout", options=["separate", "bundle"], default="separate", optional=True, tooltip="Separate writes one file per ref. Bundle stores all connected refs in one file, preserving their order and kinds."),
            ],
            outputs=[io.String.Output("saved_paths", display_name="saved paths", is_output_list=True, tooltip="Names of the saved files, including any subfolders.")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, filename_prefix, refs: io.Autogrow.Type | None = None, save_layout="separate") -> io.NodeOutput:
        connected_refs = flatten_minimax_h3_ref_collections(refs)
        if not connected_refs:
            raise ValueError("Connect at least one ref to MiniMax H3 Ref Save.")
        if save_layout not in {"separate", "bundle"}:
            raise ValueError("MiniMax H3 Ref save layout must be separate or bundle.")
        return io.NodeOutput(save_minimax_h3_ref_collection(connected_refs, filename_prefix, bundle=save_layout == "bundle"))


class UC_MiniMaxH3RefApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3RefApply",
            display_name="MiniMax H3 Ref Apply",
            category="advanced/conditioning",
            description="Adds saved or newly created references to your MiniMax H3 conditioning.",
            search_aliases=["minimax", "h3", "reference", "ref", "conditioning"],
            inputs=[
                io.Conditioning.Input("conditioning", tooltip="Connect conditioning from your MiniMax H3 workflow. Existing settings and references are kept."),
                io.Float.Input("retention", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Detail to keep in the references connected here. 1 keeps all stored detail; lower values soften detail; 0 leaves these references out. Existing references are unchanged."),
                io.Int.Input("max_ref_tokens", default=0, min=0, step=1, tooltip="Optional reference-size limit, measured in tokens. Includes existing references. 0 means no limit; exceeding a limit reports an error instead of removing detail."),
                io.Autogrow.Input("refs", template=io.Autogrow.TemplatePrefix(MiniMaxH3Ref.Input("ref", optional=True), prefix="ref_", min=1, max=100), tooltip="Connect one or more references to add. Input order is preserved; references remain separate."),
            ],
            outputs=[io.Conditioning.Output("conditioning", tooltip="Connect to your sampler in place of the original conditioning.")],
        )

    @classmethod
    def execute(cls, conditioning, retention=1.0, max_ref_tokens=0, refs: io.Autogrow.Type | None = None) -> io.NodeOutput:
        connected_refs = flatten_minimax_h3_ref_collections(refs)
        if not connected_refs:
            raise ValueError("Connect at least one ref to MiniMax H3 Ref Apply.")
        return io.NodeOutput(apply_minimax_h3_refs_to_conditioning(conditioning, connected_refs, retention, max_ref_tokens))


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
                io.Model.Input("model", display_name="model"),
                io.Image.Input("image", display_name="image"),
                io.Conditioning.Input("conditioning", display_name="conditioning", optional=True, tooltip="Text conditioning from CLIPTextEncode"),
                io.BoundingBox.Input("bboxes", display_name="bboxes", force_input=True, optional=True, tooltip="Bounding boxes to segment within"),
                io.String.Input("positive_coords", display_name="positive_coords", force_input=True, optional=True, tooltip='Positive point prompts as JSON [{"x": int, "y": int}, ...] (pixel coords)'),
                io.String.Input("negative_coords", display_name="negative_coords", force_input=True, optional=True, tooltip='Negative point prompts as JSON [{"x": int, "y": int}, ...] (pixel coords)'),
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


class UC_YOLODetectorLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_YOLODetectorLoader",
            display_name="YOLO Detector Loader",
            category="image/detection",
            description="Loads a converted YOLOv8 or YOLO11 bounding-box or segmentation safetensors detector for YOLO Detect or detailer workflows. Missing models download on execution to models/yolo/segm or models/yolo/bbox.",
            search_aliases=["yolo", "yolov8", "yolo11", "bbox detector", "segm detector", "adetailer", "impact"],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=list(YOLO_MODELS),
                    default="segm/face_yolov8m-seg_60.safetensors",
                    tooltip="Choose which YOLO detector model to load from models/yolo/segm or models/yolo/bbox. segm/ models produce both bounding boxes and contour masks; bbox/ models produce rectangular bounding boxes. Missing files download on execution.",
                )
            ],
            outputs=[
                BBoxDetectorType.Output("bbox_detector", tooltip="Bounding-box detector for YOLO Detect, FaceDetailer, DetailerForEach, or BboxDetector nodes."),
                SegmDetectorType.Output("segm_detector", tooltip="Silhouette segmentation detector when a segm/ model is selected; empty placeholder when a bbox/ model is selected."),
            ],
        )

    @classmethod
    def execute(cls, model_name="segm/face_yolov8m-seg_60.safetensors"):
        return io.NodeOutput(*build_yolo_detectors(load_yolo_model(model_name), yolo_forward))


class UC_YOLODetect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_YOLODetect",
            display_name="YOLO Detect",
            category="image/detection",
            description="Detects faces, people, hands, skin, hair, or watermarks across single images or video frame batches using native safetensors YOLOv8 and YOLO11 models, returning masks and bounding boxes.",
            search_aliases=["yolo", "yolov8", "yolo11", "detect", "segment", "watermark", "face", "hand", "person", "hair", "skin"],
            inputs=[
                io.Image.Input("image", tooltip="Image or ordered video-frame batch to scan for target objects."),
                BBoxDetectorType.Input("bbox_detector", optional=True, tooltip="Connect bbox_detector from YOLO Detector Loader to run bounding-box detection, or as a fallback when segm_detector is disconnected."),
                SegmDetectorType.Input("segm_detector", optional=True, tooltip="Connect segm_detector from YOLO Detector Loader to run contour segmentation mask detection. Takes priority when both detectors are connected."),
                io.Float.Input("threshold", default=0.3, min=0.0, max=1.0, step=0.01, tooltip="Minimum confidence required to keep a detection. Raise to reject weak detections; lower to recover faint or partly hidden targets."),
                io.Float.Input("nms_threshold", default=0.45, min=0.0, max=1.0, step=0.01, tooltip="Overlap limit for removing duplicate boxes on the same object. Lower removes overlapping duplicates more aggressively; higher keeps nearby targets separate."),
                io.Int.Input("dilation", default=0, min=-512, max=512, step=1, tooltip="Expands or shrinks detected masks in pixels. Positive values grow the mask outward; negative values erode it inward."),
                io.Int.Input("drop_size", default=10, min=1, max=MAX_RESOLUTION, step=1, tooltip="Ignore detections whose width or height is this many pixels or smaller. Raise to filter tiny specks; lower to keep small distant objects."),
                io.Int.Input("max_detections", default=300, min=1, max=1000, step=1, tooltip="Maximum detections kept per frame, highest confidence first."),
                io.Int.Input("batch_size", default=5, min=1, max=64, step=1, tooltip="Maximum images or video frames processed together per network call, including partial tail chunks without padding. Lower if VRAM runs out."),
                io.Boolean.Input("individual_masks", default=False, tooltip="When off, returns one combined mask per input frame. When on, returns a separate mask for each individual detection."),
                io.String.Input("labels", default="all", tooltip="Keep all categories with 'all', or enter comma-separated category names (face, person, hand, skin, hair, watermark) to filter detections."),
                io.Boolean.Input("temporal_filter", default=False, tooltip="Video frames only: fill missed detections on frames where neighboring frames across batch_size chunks confirm the same object. Leave off for unrelated still images."),
                io.Int.Input("temporal_radius", default=2, min=1, max=30, step=1, tooltip="Temporal filter only: frames to inspect before and after each frame when recovering missed detections (2 checks up to four neighbors across batch_size chunks)."),
                io.Int.Input("temporal_min_support", default=2, min=1, max=60, step=1, tooltip="Temporal filter only: neighboring frames within temporal_radius that must detect the object to fill a missed frame. Lower recovers longer dropouts; raise to require stronger neighbor agreement."),
                io.Float.Input("temporal_match_iou", default=0.3, min=0.0, max=1.0, step=0.01, tooltip="Temporal filter only: required box overlap between neighboring frames to match the same object when filling missed frames. Lower for faster movement; raise for stationary objects."),
            ],
            outputs=[
                io.Mask.Output("masks", tooltip="Detected masks at input image dimensions: one combined mask per frame when individual_masks is off, or one mask per detection when on."),
                io.BoundingBox.Output("bboxes", tooltip="Detected pixel bounding boxes (x, y, width, height) for SAM3 Detect, Extract Bounding Box, Adjust Bounding Box, or Ideogram crop workflows."),
            ],
        )

    @classmethod
    def execute(cls, image, bbox_detector=None, segm_detector=None, threshold=0.3, nms_threshold=0.45, dilation=0, drop_size=10, max_detections=300, batch_size=5, individual_masks=False, labels="all", temporal_filter=False, temporal_radius=2, temporal_min_support=2, temporal_match_iou=0.3):
        return io.NodeOutput(
            *run_yolo_detect(
                image,
                bbox_detector=bbox_detector,
                segm_detector=segm_detector,
                threshold=threshold,
                nms_threshold=nms_threshold,
                dilation=dilation,
                drop_size=drop_size,
                max_detections=max_detections,
                batch_size=batch_size,
                individual_masks=individual_masks,
                labels=labels,
                temporal_filter=temporal_filter,
                temporal_radius=temporal_radius,
                temporal_min_support=temporal_min_support,
                temporal_match_iou=temporal_match_iou,
            )
        )


class UC_DepthAnythingV3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_DepthAnythingV3Loader",
            display_name="Depth Anything V3 Loader",
            category="image/depth",
            description="Loads a converted Depth Anything V3 safetensors model from models/depthanything3 through UEL and ComfyUI model management. Missing models download on execution.",
            search_aliases=["depth anything", "depth anything v3", "da3", "depth loader", "monocular depth", "multiview depth"],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=list(DEPTH_ANYTHING_MODELS),
                    default="da3_large.safetensors",
                    tooltip="Choose which Depth Anything V3 checkpoint to load from models/depthanything3. da3_small, da3_base, and da3_large support single-image and multi-view depth, confidence, and camera pose estimation; da3mono_large is specialized for single-image depth and sky detection. Missing files download on execution.",
                )
            ],
            outputs=[DepthAnythingModel.Output("da3_model", tooltip="Loaded Depth Anything V3 model for the Depth Anything V3 estimation node.")],
        )

    @classmethod
    def execute(cls, model_name="da3_large.safetensors"):
        return io.NodeOutput(load_depth_anything_model(model_name))


class UC_DepthAnythingV3(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_DepthAnythingV3",
            display_name="Depth Anything V3",
            category="image/depth",
            description="Estimates depth maps, confidence masks, sky masks, and camera poses across single images, multi-view sets, or video frame batches using Depth Anything V3, with independent lower- and upper-tail clipping for normalized depth outputs.",
            search_aliases=["depth anything", "depth anything v3", "da3", "depth map", "controlnet depth", "disparity", "camera pose", "sky mask"],
            inputs=[
                DepthAnythingModel.Input("da3_model", tooltip="Connect da3_model from Depth Anything V3 Loader."),
                io.Image.Input("image", tooltip="Single image, multi-view image batch, or ordered video-frame batch to estimate depth from."),
                io.Int.Input("process_res", default=504, min=14, max=4096, step=14, tooltip="Target processing size in pixels before rounding or cropping to a multiple of 14. 504 is the model default; raise for finer depth edges at higher memory cost, or lower for faster processing."),
                io.Combo.Input(
                    "resize_method",
                    options=list(DEPTH_ANYTHING_RESIZE_METHODS),
                    default="upper_bound_resize",
                    tooltip="How input dimensions are scaled to process_res and aligned to 14-pixel patches. upper_bound sets the longer edge to process_res; lower_bound sets the shorter edge. resize rounds each edge to the nearest multiple of 14; crop center-crops to a multiple of 14.",
                ),
                io.Combo.Input(
                    "multi_view_mode",
                    options=list(DEPTH_ANYTHING_MULTI_VIEW_MODES),
                    default="independent",
                    tooltip="For image batches on any-view models (da3_small, da3_base, da3_large): independent estimates each frame on its own; multi_view links frames within each batch_size chunk for shared 3D scale and relative camera poses. da3mono_large always processes frames independently.",
                ),
                io.Combo.Input(
                    "ref_view_strategy",
                    options=list(DEPTH_ANYTHING_REF_VIEW_STRATEGIES),
                    default="saddle_balanced",
                    tooltip="Multi-view mode only (3 or more frames per chunk on any-view models): how the anchor frame is chosen. saddle_balanced picks the most balanced view; saddle_sim_range picks the widest similarity span; first or middle locks the anchor to the first or middle frame in each chunk.",
                ),
                io.Combo.Input(
                    "output_mode",
                    options=list(DEPTH_ANYTHING_OUTPUT_MODES),
                    default="disparity",
                    tooltip="disparity converts distance to inverse depth and scales to 0–1 with closer surfaces brighter (white) and distant background darker (black), matching standard ControlNet Depth maps. depth scales linear distance to 0–1 with closer surfaces brighter when invert_depth is off. raw_depth keeps unscaled model distance values.",
                ),
                io.Combo.Input(
                    "normalization_scope",
                    options=list(DEPTH_ANYTHING_NORM_SCOPES),
                    default="per_frame",
                    tooltip="When scaling disparity or depth to 0–1: per_frame scales each frame independently for maximum single-image contrast; per_batch scales across the whole batch together to keep consistent brightness across video frames or multi-view sets.",
                ),
                io.Float.Input(
                    "percentile_clip", default=0.0, min=0.0, max=10.0, step=0.1, tooltip="Symmetric baseline percentage clipped from each end before 0–1 scaling. 0 uses exact minimum and maximum; 2 clips the lowest and highest 2%. The lower- and upper-tail controls can increase clipping on either side independently."
                ),
                io.Boolean.Input("invert_depth", default=False, tooltip="Flips 0–1 depth output so closer surfaces are black (0) and farther surfaces are white (1). Ignored in raw_depth mode."),
                io.Boolean.Input("sky_fill", default=True, tooltip="On da3mono_large, clamps detected sky regions to the 99th-percentile non-sky distance so open sky does not crush foreground depth contrast."),
                io.Boolean.Input("keep_model_size", default=False, tooltip="When off, resizes the output depth map, confidence mask, and sky mask back to the input image dimensions. When on, keeps the 14-pixel-aligned processing dimensions."),
                io.Int.Input("batch_size", default=4, min=1, max=64, step=1, tooltip="Maximum images or video frames processed together per network call, including partial tail chunks without padding. Lower if VRAM runs out; raise in multi_view mode to share scale across more frames at once."),
                io.Float.Input("lower_clip_percentile", default=0.0, min=0.0, max=99.0, step=0.1, tooltip="Percentage of the lowest depth or disparity values to clip before 0–1 scaling. Effective lower-tail clipping is the larger of this value and percentile_clip. Keep at 0 to use only symmetric clipping."),
                io.Float.Input(
                    "upper_clip_percentile", default=0.0, min=0.0, max=99.0, step=0.1, tooltip="Percentage of the highest depth or disparity values to clip before 0–1 scaling. Effective upper-tail clipping is the larger of this value and percentile_clip. Combined lower and upper clipping must stay below 100%."
                ),
            ],
            outputs=[
                io.Image.Output("depth_image", tooltip="Grayscale depth map image ready for ControlNet Depth, MOSS-Image depth control, or preview."),
                io.Mask.Output("depth_mask", tooltip="Single-channel depth mask with the same values as depth_image."),
                io.Mask.Output("confidence", tooltip="Normalized 0–1 per-pixel depth confidence mask from any-view models (da3_small, da3_base, da3_large); all-ones mask on da3mono_large."),
                io.Mask.Output("sky_mask", tooltip="Binary sky mask where 1 marks sky pixels on da3mono_large; all-zeros mask on any-view models."),
                io.String.Output("camera_poses", tooltip="JSON array of per-frame predicted camera extrinsics, intrinsics, and horizontal/vertical field of view in degrees for any-view models; empty JSON array on da3mono_large."),
            ],
        )

    @classmethod
    def execute(
        cls,
        da3_model,
        image,
        process_res=504,
        resize_method="upper_bound_resize",
        multi_view_mode="independent",
        ref_view_strategy="saddle_balanced",
        output_mode="disparity",
        normalization_scope="per_frame",
        percentile_clip=0.0,
        invert_depth=False,
        sky_fill=True,
        keep_model_size=False,
        batch_size=4,
        lower_clip_percentile=0.0,
        upper_clip_percentile=0.0,
    ):
        return io.NodeOutput(
            *run_depth_anything(
                da3_model,
                image,
                process_res=process_res,
                resize_method=resize_method,
                multi_view_mode=multi_view_mode,
                ref_view_strategy=ref_view_strategy,
                output_mode=output_mode,
                normalization_scope=normalization_scope,
                percentile_clip=percentile_clip,
                invert_depth=invert_depth,
                sky_fill=sky_fill,
                keep_model_size=keep_model_size,
                batch_size=batch_size,
                lower_clip_percentile=lower_clip_percentile,
                upper_clip_percentile=upper_clip_percentile,
                forward=depth_anything_forward,
            )
        )


class UC_MiniMaxH3LatentUpscaleLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MiniMaxH3LatentUpscaleLoader",
            display_name="MiniMax H3 Latent Upscale Loader",
            category="model/loaders",
            description="Loads a MiniMax H3 2D or 3D neural latent upscaler .safetensors checkpoint from models/latent_upscale_models through Unified Efficient Loader and ComfyUI model management.",
            search_aliases=["minimax h3 latent upscaler", "h3 latent upscale loader", "minimax upscaler", "latent upscale model"],
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=get_minimax_h3_latent_upscale_model_names(),
                    tooltip="Select a MiniMax H3 2D (2d_conv_v1) or 3D (3d_conv_v1) .safetensors checkpoint placed in ComfyUI/models/latent_upscale_models. 2D checkpoints cover 1.1x to 4.0x scales at half the size and support 6 sharpness levels; 3D checkpoints target 2x, 2.25x, 2.5x, and 4x.",
                )
            ],
            outputs=[io.LatentUpscaleModel.Output("model", tooltip="Loaded MiniMax H3 latent upscale model compatible with MiniMax H3 Latent Upscale and Core MiniMax H3 Latent Upscale With Model.")],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        return io.NodeOutput(load_minimax_h3_latent_upscale_model(model_name))


class UC_MiniMaxH3LatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MiniMaxH3LatentUpscale",
            display_name="MiniMax H3 Latent Upscale",
            category="model/latent/minimax",
            description="Upscales a MiniMax H3 video or joint audio-video latent directly in latent space without decoding through the video VAE, preserving any attached audio track unchanged.",
            search_aliases=["minimax h3 latent upscale", "h3 latent upscaler", "minimax video upscale", "h3 hires latent"],
            inputs=[
                io.LatentUpscaleModel.Input("model", tooltip="Connect model from MiniMax H3 Latent Upscale Loader or Core Load Latent Upscale Model."),
                io.Latent.Input("samples", tooltip="MiniMax H3 joint audio-video latent or standalone video latent to upscale. When an audio stream is present, it passes through untouched."),
                io.DynamicCombo.Input(
                    "mode",
                    tooltip="Choose how the upscaled output resolution is specified: multiply the current size, set exact pixel width and height, or target a total megapixel budget while keeping aspect ratio.",
                    options=[
                        io.DynamicCombo.Option("scale_by", [io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.05, tooltip="Multiplier applied to both width and height (1.0 leaves size unchanged; 2.0 doubles width and height).")]),
                        io.DynamicCombo.Option(
                            "dimensions",
                            [
                                io.Int.Input("width", default=1280, min=64, max=MAX_RESOLUTION, step=32, tooltip="Target output width in pixels before rounding to the alignment grid."),
                                io.Int.Input("height", default=704, min=64, max=MAX_RESOLUTION, step=32, tooltip="Target output height in pixels before rounding to the alignment grid."),
                            ],
                        ),
                        io.DynamicCombo.Option("megapixels", [io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.05, tooltip="Target total pixel area in megapixels while preserving the input aspect ratio (1.0 is roughly 1344x768 at 16:9).")]),
                    ],
                ),
                io.Int.Input("sharpness", default=1, min=1, max=6, step=1, tooltip="Detail crispness for 2D checkpoints trained with sharpness levels: 1 stays closest to the source (recommended default), while 6 adds the strongest edge crispness. Ignored on 3D checkpoints."),
                io.Int.Input("align", default=32, min=16, max=512, step=16, tooltip="Pixel-grid multiple applied independently to output width and height. Keep at 32 so dimensions align cleanly with MiniMax H3's 16x VAE and 2x2 patch grid without border artifacts."),
                io.DynamicCombo.Input(
                    "temporal_chunking",
                    tooltip="Controls temporal windowing on 2D checkpoints for long clips. chunked caps VRAM while keeping exact temporal context; single_pass processes the entire clip at once (3D checkpoints always use built-in slabbing).",
                    options=[
                        io.DynamicCombo.Option("chunked", [io.Int.Input("chunk_frames", default=32, min=4, max=256, step=4, tooltip="Latent frames finalized per pass on 2D checkpoints (each pass automatically includes neighboring frames for seamless temporal context). Lower if VRAM is tight on long clips.")]),
                        io.DynamicCombo.Option("single_pass", []),
                    ],
                ),
            ],
            outputs=[io.Latent.Output("latent", tooltip="Upscaled MiniMax H3 latent ready for second-pass refinement sampling or VAE decoding.")],
        )

    @classmethod
    def execute(cls, model, samples, mode, sharpness: int = 1, align: int = 32, temporal_chunking: dict | None = None) -> io.NodeOutput:
        return io.NodeOutput(upscale_minimax_h3_latent(model, samples, mode, align=align, sharpness=sharpness, temporal_chunking=temporal_chunking))
