"""Node definitions for sampling workflows."""

import comfy.samplers
from comfy_api.latest import io

from ..helpers.model_helpers import apply_minimax_h3_clip_continuation
from ..helpers.sampling_helpers import H3_CHUNK_SECONDS_OPTIONS, H3_OVERLAP_SECONDS_OPTIONS, parse_h3_seconds_option, sample_dmad_renoise, sample_minimax_h3_clip_continuation_chain, start_sampling_loop


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
                io.Model.Input("model", optional=True, tooltip="Diffusion model (such as MiniMax H3). When connected, you do not need an external Guider node."),
                io.Noise.Input("noise", tooltip="Noise generator (such as RandomNoise) used to seed each chunk."),
                io.Guider.Input("guider", optional=True, tooltip="Optional external guider. If model is connected, this can remain disconnected."),
                io.Sampler.Input("sampler", tooltip="Sampling method (such as Euler or UniPC) used to denoise each video chunk."),
                io.Sigmas.Input("sigmas", tooltip="Step schedule controlling denoising speed and total step count."),
                io.Conditioning.Input("conditioning", tooltip="Text prompt and style guidance for generation. Connect a single prompt to reuse across the entire clip, or a prompt sequence to describe each chunk as the scene progresses."),
                io.Latent.Input("latent", tooltip="Blank joint video and audio canvas (from Empty MiniMax H3 Latent) sized for the total clip length."),
                io.Audio.Input("source_audio", optional=True, tooltip="Optional full source soundtrack. With preserve_input, it is encoded and pinned in the H3 latent while this node also returns the clean timeline-aligned track directly."),
                io.Vae.Input("audio_vae", optional=True, lazy=True, tooltip="MiniMax H3 audio VAE. Required only when source_audio is connected."),
                io.AnyType.Input("segment_lengths", optional=True, tooltip="Optional segment frame lengths list from H3 Reference Video Segments. When connected, the sampler follows these exact segment boundaries automatically and ignores chunk duration."),
                io.Combo.Input("chunk_duration", options=list(H3_CHUNK_SECONDS_OPTIONS), default="5.17s (124 frames)", tooltip="Duration of each generation chunk snapped to valid H3 frame grids. Used when segment_lengths is not connected. Choose '0.00s (single pass)' to generate the entire video in one chunk."),
                io.Combo.Input("overlap_duration", options=list(H3_OVERLAP_SECONDS_OPTIONS), default="0.92s (22 frames)", tooltip="Shared time window between adjacent chunks to guarantee seamless motion and audio continuity. Used when segment_lengths is not connected."),
                io.Combo.Input("carry_mode", options=["mask", "none"], default="mask", tooltip="How to handle seam transitions. 'mask' pins the tail of the previous chunk so motion continues smoothly without sudden jumps."),
                io.Float.Input("overlap_strength_video", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="How firmly earlier video frames are kept in overlapping seams. 1.0 keeps them exact, while lower values gently re-blend them."),
                io.Float.Input("overlap_strength_audio", default=0.9, min=0.0, max=1.0, step=0.01, tooltip="How firmly earlier audio is kept across seams when blending. 0.9 allows a smooth crossfade into newly generated audio."),
                io.Combo.Input(
                    "audio_mode",
                    options=["preserve_input", "generate_and_blend", "full_generation"],
                    default="preserve_input",
                    tooltip="Controls audio handling. 'preserve_input' keeps clean original reference audio across the whole clip. 'generate_and_blend' crossfades generated chunk audio with seam carry. 'full_generation' allows the model to synthesize the soundtrack without pinning.",
                ),
                io.Int.Input("sampling_start_step", default=0, min=0, max=1000, step=1, optional=True, tooltip="First step to begin sampling from. 0 starts from scratch, while higher steps refine existing content."),
                io.Int.Input("sampling_end_step", default=1000, min=1, max=1000, step=1, optional=True, tooltip="Step where sampling stops. Lower this to pause early and hand off to a second pass."),
                io.Int.Input("phase2_start_step", default=0, min=0, max=1000, step=1, optional=True, tooltip="Step number where the second sampler or guider takes over. 0 disables the second phase."),
                io.Sampler.Input("phase2_sampler", optional=True, tooltip="Optional second sampler method used after phase2_start_step for two-stage generation."),
                io.Guider.Input("phase2_guider", optional=True, tooltip="Optional second guidance configuration used after phase2_start_step."),
                io.Mask.Input("denoise_mask", optional=True, tooltip="Optional video mask. White areas generate new video and black areas stay frozen from the input latent."),
                io.Mask.Input("audio_denoise_mask", optional=True, tooltip="Optional audio mask. White areas generate new audio and black areas keep the input audio unchanged."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Finished full-length video and audio clip ready for VAE decoding."),
                io.Int.Output("chunks_rendered", tooltip="Total number of chunks rendered to assemble the full clip."),
                io.String.Output("report", tooltip="Summary log showing chunk frame ranges, carried frames, and timings."),
                io.Audio.Output("audio_passthrough", tooltip="Clean source audio aligned to the generated H3 clip. Connect this instead of decoding preserved source audio from the latent."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, source_audio=None, audio_vae=None, **kwargs):
        return ["audio_vae"] if source_audio is not None and audio_vae is None else []

    @classmethod
    def execute(
        cls,
        noise,
        sampler,
        sigmas,
        conditioning,
        latent,
        source_audio=None,
        audio_vae=None,
        model=None,
        guider=None,
        segment_lengths=None,
        chunk_duration="5.17s (124 frames)",
        overlap_duration="0.92s (22 frames)",
        carry_mode="mask",
        overlap_strength_video=1.0,
        overlap_strength_audio=0.9,
        audio_mode="preserve_input",
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
            while isinstance(val, list) and len(val) > 0 and (isinstance(val[0], list) or not isinstance(val, dict)):
                if isinstance(val[0], dict) and "samples" in val[0]:
                    return val[0]
                val = val[0]
            return val

        noise_val = _first(noise)
        sampler_val = _first(sampler)
        sigmas_val = _first(sigmas)
        latent_val = _first(latent)
        source_audio_val = _first(source_audio)
        audio_vae_val = _first(audio_vae)
        model_val = _first(model)
        guider_val = _first(guider)
        chunk_dur_val = _first(chunk_duration)
        overlap_dur_val = _first(overlap_duration)
        carry_val = _first(carry_mode)
        str_v_val = _first(overlap_strength_video)
        str_a_val = _first(overlap_strength_audio)
        audio_m_val = str(_first(audio_mode))
        start_step_val = _first(sampling_start_step)
        end_step_val = _first(sampling_end_step)
        p2_start_val = _first(phase2_start_step)
        p2_sampler_val = _first(phase2_sampler)
        p2_guider_val = _first(phase2_guider)
        d_mask_val = _first(denoise_mask)
        ad_mask_val = _first(audio_denoise_mask)

        # Make sure latent_val is a dict
        if isinstance(latent, list):
            for item in latent:
                if isinstance(item, dict) and "samples" in item:
                    latent_val = item
                    break
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict) and "samples" in sub:
                            latent_val = sub
                            break
                    if isinstance(latent_val, dict) and "samples" in latent_val:
                        break

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

        out_latent, num_chunks, report, passthrough_audio = start_sampling_loop(
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
            audio_mode=audio_m_val,
            sampling_start_step=start_step_val,
            sampling_end_step=end_step_val,
            phase2_start_step=p2_start_val,
            phase2_sampler=p2_sampler_val,
            phase2_guider=p2_guider_val,
            denoise_mask=d_mask_val,
            audio_denoise_mask=ad_mask_val,
            source_audio=source_audio_val,
            audio_vae=audio_vae_val,
        )
        return io.NodeOutput(out_latent, num_chunks, report, passthrough_audio)


from .video_nodes import UC_H3RefVideoSegments


class UC_SamplerDMADReNoise(io.ComfyNode):
    """Few-step re-noise sampler for DMAD/DMD2 distilled models (e.g. 4-step MiniMax H3)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SamplerDMADReNoise",
            display_name="DMAD Re-Noise Sampler",
            category="sampling/custom_sampling/samplers",
            description="Few-step re-noise sampler for DMAD/DMD2 distilled models (e.g. 4-step MiniMax H3).",
            inputs=[io.Float.Input("s_noise", default=1.0, min=0.0, max=10.0, step=0.01, round=False, tooltip="Noise multiplier for intermediate re-noise draws. 1.0 matches the DMAD paper training protocol.")],
            outputs=[io.Sampler.Output("sampler", display_name="Sampler")],
        )

    @classmethod
    def execute(cls, s_noise: float = 1.0) -> io.NodeOutput:
        sampler = comfy.samplers.KSAMPLER(sample_dmad_renoise, extra_options={"s_noise": s_noise})
        return io.NodeOutput(sampler)

    get_sampler = execute


class UC_MiniMaxH3ClipContinuationApply(io.ComfyNode):
    """Pin previous-clip video and audio tail at the head of a MiniMax H3 clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3ClipContinuationApply",
            display_name="MiniMax H3 Clip Continuation Apply",
            category="advanced/conditioning",
            description=("Pins the ending frames and sound of your previous clip onto the start of the new clip so motion, characters, and music continue seamlessly across the cut without re-encoding."),
            inputs=[
                io.Conditioning.Input("conditioning", tooltip="Positive conditioning from your MiniMax H3 encoder."),
                io.Latent.Input("latent", tooltip="Target clip latent from your MiniMax H3 encoder."),
                io.Combo.Input(
                    "tail_frames",
                    options=["22", "5", "39", "56"],
                    default="22",
                    tooltip="Keep at 22 (recommended ~0.9s sweet spot). Controls how many ending video frames of the previous clip are pinned onto the start of the next clip to lock in motion, then removed by Trim. 5 is barely fluid; 39 or 56 spend more of each render on overlap frames that get trimmed away.",
                ),
                io.Int.Input(
                    "audio_tail_frames", default=24, min=0, max=240, step=1, tooltip="Keep at 24 (1 full second of sound) or 48 (2 seconds for musical beats). Controls how much ending audio is carried over across the cut without costing any extra video frames. Multiples of 24 are whole seconds; 0 matches tail_frames."
                ),
                io.Latent.Input("continuation_latent", optional=True, tooltip="Previous clip's sampler output latent (recommended). Slices both picture and sound directly in latent space without any decode or re-encode."),
                io.Custom("MINIMAX_H3_CLIP_CONTINUATION_MEDIA").Input("continuation_media", optional=True, tooltip="Optional output from MiniMax H3 Clip Continuation Load."),
                io.Image.Input("continuation_frames", optional=True, tooltip="Optional decoded frames from the previous clip (used only when continuation_latent and continuation_media are not connected)."),
                io.Audio.Input("continuation_audio", optional=True, tooltip="Optional decoded audio from the previous clip (used only when continuation_latent is not connected)."),
                io.Vae.Input("vae", optional=True, tooltip="MiniMax H3 video VAE (required only when encoding decoded continuation_frames instead of a latent)."),
                io.Vae.Input("audio_vae", optional=True, tooltip="MiniMax H3 audio VAE (required only when encoding decoded continuation_audio instead of a latent)."),
            ],
            outputs=[
                io.Conditioning.Output("conditioning", display_name="conditioning", tooltip="Conditioning with seamless motion and audio continuation attached."),
                io.Int.Output("trim_frames", display_name="trim_frames", tooltip="Number of leading pinned frames to remove with MiniMax H3 Clip Continuation Trim after decoding (0 on the first clip)."),
            ],
        )

    @classmethod
    def execute(cls, conditioning, latent, tail_frames: str = "22", audio_tail_frames: int = 24, continuation_latent=None, continuation_media=None, continuation_frames=None, continuation_audio=None, vae=None, audio_vae=None) -> io.NodeOutput:
        out_cond, trim_frames = apply_minimax_h3_clip_continuation(
            conditioning, latent, tail_frames=int(tail_frames), audio_tail_frames=int(audio_tail_frames), continuation_latent=continuation_latent, continuation_media=continuation_media, continuation_frames=continuation_frames, continuation_audio=continuation_audio, vae=vae, audio_vae=audio_vae
        )
        return io.NodeOutput(out_cond, trim_frames)


class UC_MiniMaxH3ClipContinuationSampler(io.ComfyNode):
    """Drop-in SamplerCustomAdvanced replacement that chains MiniMax H3 segment lists in a single run."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MiniMaxH3ClipContinuationSampler",
            display_name="MiniMax H3 Clip Continuation Sampler",
            category="utils/sampling",
            is_input_list=True,
            description=("Drop-in replacement for SamplerCustomAdvanced that samples a list of MiniMax H3 segments in one run, automatically chaining each finished segment's video and audio tail into the next segment without re-encoding."),
            inputs=[
                io.Noise.Input("noise", tooltip="Noise generator (such as RandomNoise) used to seed each segment."),
                io.Guider.Input("guider", tooltip="Guider (from CFGGuider or BasicGuider). Accepts a single guider or a list of guiders produced from your segment list."),
                io.Sampler.Input("sampler", tooltip="Sampling method used to denoise each segment."),
                io.Sigmas.Input("sigmas", tooltip="Step schedule controlling denoising speed and total step count."),
                io.Latent.Input("latent_image", tooltip="Target latent (or list of segment latents) from your MiniMax H3 encoder."),
                io.Combo.Input(
                    "tail_frames",
                    options=["22", "5", "39", "56"],
                    default="22",
                    tooltip="Keep at 22 (recommended ~0.9s sweet spot). Controls how many ending video frames of the previous segment are pinned onto the start of the next segment to lock in motion, then removed by Trim. 5 is barely fluid; 39 or 56 spend more of each render on overlap frames that get trimmed away.",
                ),
                io.Int.Input(
                    "audio_tail_frames", default=24, min=0, max=240, step=1, tooltip="Keep at 24 (1 full second of sound) or 48 (2 seconds for musical beats). Controls how much ending audio is carried over across the cut without costing any extra video frames. Multiples of 24 are whole seconds; 0 matches tail_frames."
                ),
            ],
            outputs=[
                io.Latent.Output("latent", display_name="latent", is_output_list=True, tooltip="Sampled segment latents (as a list). Connect to VAEDecode and VAEDecodeAudio."),
                io.Int.Output("trim_leading_frames", display_name="trim_leading_frames", is_output_list=True, tooltip="Leading overlap frames to remove for each segment. Connect directly to trim_leading_frames on MiniMax H3 Clip Continuation Trim."),
                io.Int.Output("trim_trailing_frames", display_name="trim_trailing_frames", is_output_list=True, tooltip="Trailing padded frames to remove on the final segment. Connect directly to trim_trailing_frames on MiniMax H3 Clip Continuation Trim."),
                io.Latent.Output("combined_latent", display_name="combined_latent", tooltip="Single stitched full-length video and audio latent with overlapping seams already removed."),
                io.Latent.Output("last_latent", display_name="last_latent", tooltip="Final segment's untrimmed latent, ready to connect to MiniMax H3 Clip Continuation Save to continue the chain in a later run."),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, tail_frames="22", audio_tail_frames=24) -> io.NodeOutput:
        segment_latents, trim_leading_frames, trim_trailing_frames, combined_latent, last_latent = sample_minimax_h3_clip_continuation_chain(noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent=latent_image, tail_frames=tail_frames, audio_tail_frames=audio_tail_frames)
        return io.NodeOutput(segment_latents, trim_leading_frames, trim_trailing_frames, combined_latent, last_latent)


SAMPLING_NODES = [UC_H3LoopSampler, UC_H3RefVideoSegments, UC_SamplerDMADReNoise, UC_MiniMaxH3ClipContinuationApply, UC_MiniMaxH3ClipContinuationSampler]
