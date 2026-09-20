"""Node definitions for sampling workflows."""

from comfy_api.latest import io
from ..helpers.sampling_helpers import (
    split_h3_video_components_into_segments,
    start_sampling_loop,
)


class UC_H3LoopSampler(io.ComfyNode):
    """Looping video/audio sampler for MiniMax H3 joint clips."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_H3LoopSampler",
            display_name="H3 Loop Sampler",
            category="utils/sampling",
            description="Samples long-form MiniMax H3 AV latents by windowed chunks with carry preservation.",
            inputs=[
                io.Noise.Input(
                    "noise",
                    tooltip="Noise generator (such as RandomNoise) used to seed each chunk.",
                ),
                io.Guider.Input(
                    "guider",
                    tooltip="Main guidance settings (such as BasicGuider or CFGGuider). Controls prompt strength and negative conditioning.",
                ),
                io.Sampler.Input(
                    "sampler",
                    tooltip="Sampling method (such as Euler or UniPC) used to denoise each video chunk.",
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Step schedule controlling denoising speed and total step count.",
                ),
                io.Conditioning.Input(
                    "conditioning",
                    tooltip="Text prompt and style guidance for generation. Connect a single prompt to reuse across the entire clip, or a prompt sequence to describe each chunk as the scene progresses.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="Blank joint video and audio canvas (from Empty MiniMax H3 Latent) sized for the total clip length.",
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=124,
                    min=0,
                    max=3600,
                    step=17,
                    tooltip="Length of each generation chunk in frames. Set to 0 to generate the whole video in one single pass.",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=22,
                    min=0,
                    max=720,
                    step=17,
                    tooltip="Number of shared frames between adjacent chunks to ensure seamless transitions.",
                ),
                io.Combo.Input(
                    "carry_mode",
                    options=["mask", "none"],
                    default="mask",
                    tooltip="How to handle seam transitions. 'mask' pins the tail of the previous chunk so motion continues smoothly without sudden jumps.",
                ),
                io.Float.Input(
                    "overlap_strength_video",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How firmly earlier video frames are kept in overlapping seams. 1.0 keeps them exact, while lower values gently re-blend them.",
                ),
                io.Float.Input(
                    "overlap_strength_audio",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How firmly earlier audio is kept across seams. 0.9 allows a smooth crossfade into newly generated audio.",
                ),
                io.Int.Input(
                    "sampling_start_step",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip="First step to begin sampling from. 0 starts from scratch, while higher steps refine existing content.",
                ),
                io.Int.Input(
                    "sampling_end_step",
                    default=1000,
                    min=1,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip="Step where sampling stops. Lower this to pause early and hand off to a second pass.",
                ),
                io.Int.Input(
                    "phase2_start_step",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip="Step number where the second sampler or guider takes over. 0 disables the second phase.",
                ),
                io.Sampler.Input(
                    "phase2_sampler",
                    optional=True,
                    tooltip="Optional second sampler method used after phase2_start_step for two-stage generation.",
                ),
                io.Guider.Input(
                    "phase2_guider",
                    optional=True,
                    tooltip="Optional second guidance configuration used after phase2_start_step.",
                ),
                io.Mask.Input(
                    "denoise_mask",
                    optional=True,
                    tooltip="Optional video mask. White areas generate new video and black areas stay frozen from the input latent.",
                ),
                io.Mask.Input(
                    "audio_denoise_mask",
                    optional=True,
                    tooltip="Optional audio mask. White areas generate new audio and black areas keep the input audio unchanged.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent",
                    tooltip="Finished full-length video and audio clip ready for VAE decoding.",
                ),
                io.Int.Output(
                    "chunks_rendered",
                    tooltip="Total number of chunks rendered to assemble the full clip.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Summary log showing chunk frame ranges, carried frames, and timings.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        noise,
        guider,
        sampler,
        sigmas,
        conditioning,
        latent,
        chunk_frames=124,
        overlap_frames=22,
        carry_mode="mask",
        overlap_strength_video=1.0,
        overlap_strength_audio=0.9,
        sampling_start_step=0,
        sampling_end_step=1000,
        phase2_start_step=0,
        phase2_sampler=None,
        phase2_guider=None,
        denoise_mask=None,
        audio_denoise_mask=None,
    ) -> io.NodeOutput:
        cond_list = conditioning.get("conds") if isinstance(conditioning, dict) and "conds" in conditioning else (
            conditioning if isinstance(conditioning, list) and conditioning and isinstance(conditioning[0], list) and not isinstance(conditioning[0][0], torch.Tensor)
            else [conditioning]
        )

        out_latent, num_chunks, report = start_sampling_loop(
            noise=noise,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            cond_list=cond_list,
            latent=latent,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            carry_mode=carry_mode,
            overlap_strength_video=overlap_strength_video,
            overlap_strength_audio=overlap_strength_audio,
            sampling_start_step=sampling_start_step,
            sampling_end_step=sampling_end_step,
            phase2_start_step=phase2_start_step,
            phase2_sampler=phase2_sampler,
            phase2_guider=phase2_guider,
            denoise_mask=denoise_mask,
            audio_denoise_mask=audio_denoise_mask,
        )
        return io.NodeOutput(out_latent, num_chunks, report)


class UC_H3RefVideoSegments(io.ComfyNode):
    """Splits a reference video into a list of windowed H3 segments for batched conditioning."""

    @classmethod
    def define_schema(cls):
        from ..helpers.image_helpers import VIDEO_FRAME_TIMESTAMP_FORMATS

        return io.Schema(
            node_id="UC_H3RefVideoSegments",
            display_name="H3 Reference Video Segments (List)",
            category="advanced/video",
            description="Prepares all sequential H3 reference segments as list outputs for multi-chunk captioning and conditioning workflows.",
            search_aliases=["minimax", "h3", "reference", "video", "segments", "list"],
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip="Full reference video clip to slice across all segments.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=0.5,
                    min=0.01,
                    max=4.0,
                    step=0.001,
                    tooltip="Target frame size. Scales and center crops frames cleanly to matching standard aspect ratio.",
                ),
                io.Int.Input(
                    "segment_count",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Number of explicit equal segments. 0 uses chunk_frames and overlap_frames to match H3 Loop Sampler windows.",
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip="Frames per chunk window when segment_count is 0.",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=22,
                    min=0,
                    max=720,
                    step=17,
                    tooltip="Overlap frames between consecutive chunk windows.",
                ),
                io.Custom("WHISPER_MODEL").Input(
                    "whisper_model",
                    optional=True,
                    tooltip="Optional Whisper speech recognition model for word-timed transcription.",
                ),
                io.Combo.Input(
                    "timestamp_format",
                    options=list(VIDEO_FRAME_TIMESTAMP_FORMATS),
                    default="00.000s",
                    optional=True,
                    tooltip="Timestamp format for transcript outputs.",
                ),
                io.Boolean.Input(
                    "enable_whisper",
                    default=True,
                    optional=True,
                    tooltip="Transcribe speech in each segment when a Whisper model is connected.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "frames_list",
                    is_output_list=True,
                    tooltip="List of prepared 24 fps video frames, one per segment.",
                ),
                io.Audio.Output(
                    "audio_list",
                    is_output_list=True,
                    tooltip="List of matching audio tracks, one per segment.",
                ),
                io.Int.Output(
                    "width",
                    tooltip="Common width in pixels matching H3 resolution requirements.",
                ),
                io.Int.Output(
                    "height",
                    tooltip="Common height in pixels matching H3 resolution requirements.",
                ),
                io.Int.Output(
                    "lengths",
                    is_output_list=True,
                    tooltip="List of segment frame lengths.",
                ),
                io.Video.Output(
                    "video_list",
                    is_output_list=True,
                    tooltip="List of preview video objects for each segment.",
                ),
                io.String.Output(
                    "transcribed_audio_list",
                    is_output_list=True,
                    tooltip="List of word-timed transcripts aligned to each segment.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        megapixels=0.5,
        segment_count=0,
        chunk_frames=124,
        overlap_frames=22,
        whisper_model=None,
        timestamp_format="00.000s",
        enable_whisper=True,
    ) -> io.NodeOutput:
        (
            frames_list,
            audio_list,
            width,
            height,
            length_list,
            video_list,
            transcript_list,
        ) = split_h3_video_components_into_segments(
            video=video,
            megapixels=megapixels,
            segment_count=segment_count,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            whisper_model=whisper_model,
            timestamp_format=timestamp_format,
            enable_whisper=enable_whisper,
        )
        return io.NodeOutput(
            frames_list,
            audio_list,
            width,
            height,
            length_list,
            video_list,
            transcript_list,
        )
