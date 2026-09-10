from comfy_api.latest import io
from .model_helpers import (
    WHISPER_MODELS, WHISPER_LANGUAGES, load_whisper_model, register_whisper_paths, run_whisper,
    apply_minimax_h3_refs_to_conditioning,
    create_minimax_h3_audio_ref,
    create_minimax_h3_image_refs,
    create_minimax_h3_video_ref,
    detect_sam3,
    flatten_minimax_h3_ref_collections,
    format_minimax_h3_ref_info,
    get_minimax_h3_ref_input_fingerprint,
    list_minimax_h3_refs,
    load_minimax_h3_ref,
    minimax_h3_ref_resolution_grid,
    save_minimax_h3_ref_collection,
)

WhisperModel = io.Custom("WHISPER_MODEL")
register_whisper_paths()


class UC_WhisperLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_WhisperLoader", display_name="Whisper Loader", category="utils/audio",
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
            node_id="UC_WhisperTranscribe", display_name="Whisper Transcribe", category="utils/audio",
            description="Transcribes full recordings or translates speech to English; one aligned result per audio batch item.",
            inputs=[WhisperModel.Input("whisper_model"), io.Audio.Input("audio"),
                    io.Combo.Input("task", options=["transcribe", "translate"], default="transcribe"),
                    io.Combo.Input("language", options=["auto", *WHISPER_LANGUAGES], default="auto", tooltip="Spoken language code, or automatic detection. Translation outputs English.")],
            outputs=[io.String.Output("text", is_output_list=True),
                     io.String.Output("segments", is_output_list=True, tooltip="JSON array of start/end seconds and text for this recording."),
                     io.String.Output("language", is_output_list=True)],
        )

    @classmethod
    def execute(cls, whisper_model, audio, task="transcribe", language="auto"):
        return io.NodeOutput(*run_whisper(whisper_model, audio, task, language))


MiniMaxH3Ref = io.Custom("MINIMAX_H3_REF")


class UC_MiniMaxH3RefExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        image_compression_options = [
            io.DynamicCombo.Option(key="encode", inputs=[]),
            io.DynamicCombo.Option(
                key="pooled",
                inputs=[
                    io.Int.Input("reference_resolution", display_name="Reference resolution", default=256, min=32, step=32, tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source."),
                ],
            ),
            io.DynamicCombo.Option(
                key="refined",
                inputs=[
                    io.Int.Input("reference_resolution", display_name="Reference resolution", default=256, min=32, step=32, tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source."),
                    io.Int.Input("refine_steps", default=100, min=1, step=1, tooltip="Suggested starting range: 50–200 steps. Start at 100; use 50 for a quicker attempt, or try 200 if the compressed reference loses detail. Higher values take longer without guaranteeing an improvement."),
                ],
            ),
        ]
        video_compression_options = [
            io.DynamicCombo.Option(key="encode", inputs=[]),
            io.DynamicCombo.Option(
                key="pooled",
                inputs=[
                    io.Int.Input("reference_resolution", display_name="Reference resolution", default=256, min=32, step=32, tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source."),
                    io.Int.Input("temporal_density", default=16, min=1, step=1, tooltip="How densely the reference represents time across the video. Higher values keep finer motion detail; lower values blend more moments together. The full clip remains covered. Start with 4–16; try 32 or more for longer or complex clips."),
                ],
            ),
            io.DynamicCombo.Option(
                key="refined",
                inputs=[
                    io.Int.Input("reference_resolution", display_name="Reference resolution", default=256, min=32, step=32, tooltip="Compressed reference size in pixels along the longer edge. Start with 128–512 px; 256 px is the default. Higher values retain more detail but use more memory. The shorter edge follows the source shape. This does not change generation resolution or enlarge the source."),
                    io.Int.Input("temporal_density", default=16, min=1, step=1, tooltip="How densely the reference represents time across the video. Higher values keep finer motion detail; lower values blend more moments together. The full clip remains covered. Start with 4–16; try 32 or more for longer or complex clips."),
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
            description="Creates reusable MiniMax H3 references from images or video frames.",
            search_aliases=["minimax", "h3", "reference", "ref", "image", "video"],
            inputs=[
                io.Image.Input("images", tooltip="Connect images or video frames. Image mode keeps each image separate. Video mode uses the frames as one clip; provide at least five frames at 24 fps."),
                io.Vae.Input("vae", display_name="visual vae", tooltip="Connect the MiniMax H3 video VAE for both images and video."),
                io.DynamicCombo.Input("media_type", options=media_type_options, tooltip="Choose image to keep each image separate, or video to treat the frames as one clip."),
                io.String.Input("description", default="", multiline=True, dynamic_prompts=False, tooltip="Optional notes to save with the reference. These notes do not change your prompt."),
            ],
            outputs=[
                MiniMaxH3Ref.Output("ref", display_name="ref", tooltip="Connect to Ref Save to keep these references, or Ref Apply to use them."),
                io.String.Output("info", display_name="info", tooltip="Summary of the created references, including any video frames left out."),
            ],
        )

    @classmethod
    def execute(cls, images, vae, media_type, description="") -> io.NodeOutput:
        kind = media_type.get("media_type")
        compression = media_type.get("compression", {})
        mode = compression.get("compression")
        grid_long_edge = minimax_h3_ref_resolution_grid(compression["reference_resolution"]) if mode != "encode" else 16
        if kind == "image":
            refs = create_minimax_h3_image_refs(images, vae, mode, grid_long_edge, compression.get("refine_steps", 100), description)
        else:
            refs = [create_minimax_h3_video_ref(images, vae, mode, grid_long_edge, compression.get("temporal_density", 16), compression.get("refine_steps", 100), description)]
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
            outputs=[
                MiniMaxH3Ref.Output("ref", display_name="ref", tooltip="Connect to Ref Save to keep this audio reference, or Ref Apply to use it."),
                io.String.Output("info", display_name="info", tooltip="Summary of the created audio reference."),
            ],
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
            description="Loads a saved MiniMax H3 reference for use with Ref Apply.",
            search_aliases=["minimax", "h3", "reference", "ref", "load"],
            inputs=[
                io.Combo.Input("filename", options=list_minimax_h3_refs(), tooltip="Choose a saved reference from ComfyUI/models/minimax_h3_refs or your configured reference folders."),
            ],
            outputs=[
                MiniMaxH3Ref.Output("ref", display_name="ref", tooltip="Connect to Ref Apply to use this reference, or Ref Save to save another copy."),
                io.String.Output("info", display_name="info", tooltip="Summary of the loaded reference."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, filename):
        return get_minimax_h3_ref_input_fingerprint(filename)

    @classmethod
    def execute(cls, filename) -> io.NodeOutput:
        refs = [load_minimax_h3_ref(filename)]
        return io.NodeOutput(refs, format_minimax_h3_ref_info(refs))


class UC_MiniMaxH3RefSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3RefSave",
            display_name="MiniMax H3 Ref Save",
            category="model/minimax_h3",
            description="Saves each connected reference to its own file. No sampler is needed.",
            search_aliases=["minimax", "h3", "reference", "ref", "save"],
            inputs=[
                io.String.Input("filename_prefix", default="ref/MiniMax_H3", tooltip="Name for the saved files. Use a slash to add a subfolder. Files are numbered automatically and existing files are not overwritten."),
                io.Autogrow.Input("refs", template=io.Autogrow.TemplatePrefix(MiniMaxH3Ref.Input("ref", optional=True), prefix="ref_", min=1, max=100), tooltip="Connect one or more references to save. Each reference is saved separately, in input order."),
            ],
            outputs=[
                io.String.Output("saved_paths", display_name="saved paths", is_output_list=True, tooltip="Names of the saved files, including any subfolders."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, filename_prefix, refs: io.Autogrow.Type | None = None) -> io.NodeOutput:
        connected_refs = flatten_minimax_h3_ref_collections(refs)
        if not connected_refs:
            raise ValueError("Connect at least one ref to MiniMax H3 Ref Save.")
        return io.NodeOutput(save_minimax_h3_ref_collection(connected_refs, filename_prefix))


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
            outputs=[
                io.Conditioning.Output("conditioning", tooltip="Connect to your sampler in place of the original conditioning."),
            ],
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
