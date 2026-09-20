"""Node definitions for sampling workflows."""

import torch
from comfy_api.latest import io
from ..helpers.sampling_helpers import (
    H3_CHUNK_SECONDS_OPTIONS,
    H3_OVERLAP_SECONDS_OPTIONS,
    parse_h3_seconds_option,
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
            is_input_list=True,
            description="Samples long-form MiniMax H3 AV latents by windowed chunks with carry preservation.",
            inputs=[
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip="Diffusion model (such as MiniMax H3). When connected, you do not need an external Guider node.",
                ),
                io.Noise.Input(
                    "noise",
                    tooltip="Noise generator (such as RandomNoise) used to seed each chunk.",
                ),
                io.Guider.Input(
                    "guider",
                    optional=True,
                    tooltip="Optional external guider. If model is connected, this can remain disconnected.",
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
                io.AnyType.Input(
                    "segment_lengths",
                    optional=True,
                    tooltip="Optional segment frame lengths list from H3 Reference Video Segments. When connected, the sampler follows these exact segment boundaries automatically and ignores chunk duration.",
                ),
                io.Combo.Input(
                    "chunk_duration",
                    options=list(H3_CHUNK_SECONDS_OPTIONS),
                    default="5.17s (124 frames)",
                    tooltip="Duration of each generation chunk snapped to valid H3 frame grids. Used when segment_lengths is not connected. Choose '0.00s (single pass)' to generate the entire video in one chunk.",
                ),
                io.Combo.Input(
                    "overlap_duration",
                    options=list(H3_OVERLAP_SECONDS_OPTIONS),
                    default="0.92s (22 frames)",
                    tooltip="Shared time window between adjacent chunks to guarantee seamless motion and audio continuity. Used when segment_lengths is not connected.",
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
        sampler,
        sigmas,
        conditioning,
        latent,
        model=None,
        guider=None,
        segment_lengths=None,
        chunk_duration="5.17s (124 frames)",
        overlap_duration="0.92s (22 frames)",
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
        **kwargs,
    ) -> io.NodeOutput:
        # Unwrap any list-wrapped scalar inputs caused by is_input_list=True
        def _first(val):
            return val[0] if isinstance(val, list) and len(val) > 0 else val

        noise_val = _first(noise)
        sampler_val = _first(sampler)
        sigmas_val = _first(sigmas)
        latent_val = _first(latent)
        model_val = _first(model)
        guider_val = _first(guider)
        chunk_dur_val = _first(chunk_duration)
        overlap_dur_val = _first(overlap_duration)
        carry_val = _first(carry_mode)
        str_v_val = _first(overlap_strength_video)
        str_a_val = _first(overlap_strength_audio)
        start_step_val = _first(sampling_start_step)
        end_step_val = _first(sampling_end_step)
        p2_start_val = _first(phase2_start_step)
        p2_sampler_val = _first(phase2_sampler)
        p2_guider_val = _first(phase2_guider)
        d_mask_val = _first(denoise_mask)
        ad_mask_val = _first(audio_denoise_mask)

        if guider_val is None:
            if model_val is None:
                raise ValueError("UC_H3LoopSampler requires either 'model' or 'guider' to be connected.")
            from comfy_extras.nodes_custom_sampler import Guider_Basic
            guider_val = Guider_Basic(model_val)

        # Handle conditioning list
        if isinstance(conditioning, dict) and "conds" in conditioning:
            cond_list = conditioning["conds"]
        elif isinstance(conditioning, list):
            # Check if wrapped in outer list from is_input_list=True: [ [cond0, cond1] ]
            unwrapped = conditioning
            if len(conditioning) == 1 and isinstance(conditioning[0], list):
                if len(conditioning[0]) > 0 and isinstance(conditioning[0][0], list):
                    unwrapped = conditioning[0]

            if len(unwrapped) > 0 and isinstance(unwrapped[0], list):
                if len(unwrapped[0]) > 0 and isinstance(unwrapped[0][0], list):
                    # List of conditionings: [ [[t0, d0]], [[t1, d1]] ]
                    cond_list = unwrapped
                elif len(unwrapped[0]) == 2 and isinstance(unwrapped[0][1], dict):
                    # Single conditioning: [ [t0, d0] ]
                    cond_list = [unwrapped]
                else:
                    cond_list = unwrapped
            else:
                cond_list = [unwrapped]
        else:
            cond_list = [conditioning]

        # Flatten segment_lengths if passed as list of lists
        flat_segments = None
        if segment_lengths is not None:
            if isinstance(segment_lengths, list):
                flat_segments = []
                for item in segment_lengths:
                    if isinstance(item, list):
                        flat_segments.extend([int(x) for x in item])
                    else:
                        flat_segments.append(int(item))
            elif isinstance(segment_lengths, (int, float)):
                flat_segments = [int(segment_lengths)]

        cf = kwargs.get("chunk_frames") if "chunk_frames" in kwargs else parse_h3_seconds_option(chunk_dur_val, 124)
        of = kwargs.get("overlap_frames") if "overlap_frames" in kwargs else parse_h3_seconds_option(overlap_dur_val, 22)

        out_latent, num_chunks, report = start_sampling_loop(
            noise=noise_val,
            guider=guider_val,
            sampler=sampler_val,
            sigmas=sigmas_val,
            cond_list=cond_list,
            latent=latent_val,
            chunk_frames=cf,
            overlap_frames=of,
            segment_lengths=flat_segments,
            carry_mode=carry_val,
            overlap_strength_video=str_v_val,
            overlap_strength_audio=str_a_val,
            sampling_start_step=start_step_val,
            sampling_end_step=end_step_val,
            phase2_start_step=p2_start_val,
            phase2_sampler=p2_sampler_val,
            phase2_guider=p2_guider_val,
            denoise_mask=d_mask_val,
            audio_denoise_mask=ad_mask_val,
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
                    tooltip="Target frame size. Scales and center crops frames cleanly to the nearest standard aspect ratio.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.1,
                    tooltip="Segment duration in seconds when segment_count is 0. 0 uses default 5.17s H3 segment chunks.",
                ),
                io.Float.Input(
                    "start_at_timestamp",
                    default=0.0,
                    min=0.0,
                    step=0.1,
                    tooltip="Seconds to skip at the start of the source video before slicing segments.",
                ),
                io.Int.Input(
                    "segment_count",
                    display_name="Divide into segments (0 = manual)",
                    default=0,
                    min=0,
                    step=1,
                    optional=True,
                    tooltip="Positive count divides the source into that many sequential H3-sized segments. 0 uses duration_seconds and start_at_timestamp.",
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
                    tooltip="Timestamp format for transcript outputs matching the timeline nodes.",
                ),
                io.Boolean.Input(
                    "enable_whisper",
                    default=True,
                    optional=True,
                    tooltip="When false, skips Whisper transcription and returns an empty transcript.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "frames_list",
                    is_output_list=True,
                    tooltip="List of prepared 24 fps video frames with one batch entry per segment.",
                ),
                io.Audio.Output(
                    "audio_list",
                    is_output_list=True,
                    tooltip="List of matching audio soundtracks with one entry per segment.",
                ),
                io.Int.Output(
                    "width",
                    tooltip="Common video width in pixels matching H3 grid requirements.",
                ),
                io.Int.Output(
                    "height",
                    tooltip="Common video height in pixels matching H3 grid requirements.",
                ),
                io.Int.Output(
                    "lengths",
                    is_output_list=True,
                    tooltip="List of segment frame lengths at 24 fps.",
                ),
                io.Video.Output(
                    "video_list",
                    is_output_list=True,
                    tooltip="List of preview video objects for each segment.",
                ),
                io.String.Output(
                    "transcribed_audio_list",
                    is_output_list=True,
                    tooltip="List of word-timed transcripts aligned to each individual segment.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        megapixels=0.5,
        duration_seconds=0.0,
        start_at_timestamp=0.0,
        segment_count=0,
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
            duration_seconds=duration_seconds,
            start_at_timestamp=start_at_timestamp,
            segment_count=segment_count,
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
