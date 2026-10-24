import hashlib
import logging
import os
from typing import Any

import torch
from tqdm import tqdm
from comfy_api.latest import InputImpl, Types, io, ui
from ..helpers.model_helpers import concatenate_audio_list
from ..helpers.video_helpers import (
    is_video_file,
    list_video_files,
    normalize_video_path,
    save_keyframe_optimized_video,
)


class UC_LoadVideoPath(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadVideoPath",
            display_name="Load Video (Path)",
            category="advanced/video",
            search_aliases=["import video path", "open video path", "video file path"],
            inputs=[
                io.String.Input(
                    "video_path",
                    multiline=False,
                    placeholder="path/to/video.mp4 or X:/path/to/video.mp4",
                    tooltip="Video file path. Supports absolute or relative paths and either slash style.",
                ),
            ],
            outputs=[io.Video.Output(display_name="VIDEO")],
        )

    @classmethod
    def execute(cls, video_path: str) -> io.NodeOutput:
        normalized_path = normalize_video_path(video_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            raise ValueError(
                f"Invalid video path: {video_path} (resolved to: {normalized_path})"
            )
        if not is_video_file(normalized_path):
            raise ValueError(f"File does not contain readable video: {video_path}")
        return io.NodeOutput(InputImpl.VideoFromFile(normalized_path))

    @classmethod
    def fingerprint_inputs(cls, video_path: str):
        normalized_path = normalize_video_path(video_path)
        if not normalized_path or not os.path.isfile(normalized_path):
            return ""
        stat = os.stat(normalized_path)
        return f"{normalized_path}:{stat.st_mtime_ns}:{stat.st_size}"

    @classmethod
    def validate_inputs(cls, video_path: str):
        normalized_path = normalize_video_path(video_path)
        if not normalized_path:
            return "Video path cannot be empty"
        if not os.path.isfile(normalized_path):
            return f"Invalid video file: {video_path} (resolved to: {normalized_path})"
        if not is_video_file(normalized_path):
            return f"File does not contain readable video: {video_path}"
        return True


class UC_LoadVideoDirectory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_LoadVideoDirectory",
            display_name="Load Videos (Directory)",
            category="advanced/video",
            search_aliases=["import videos directory", "open videos folder", "video folder"],
            inputs=[
                io.String.Input(
                    "directory_path",
                    multiline=False,
                    placeholder="path/to/video-directory",
                    tooltip="Directory containing video files. Files are sorted alphabetically.",
                ),
                io.Int.Input(
                    "start_index",
                    default=0,
                    min=0,
                    step=1,
                    tooltip="Index of the first video to load in sorted order.",
                ),
                io.Int.Input(
                    "load_count",
                    default=1,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip="Number of videos to load.",
                ),
            ],
            outputs=[io.Video.Output(display_name="VIDEO", is_output_list=True)],
        )

    @classmethod
    def execute(cls, directory_path: str, start_index: int, load_count: int) -> io.NodeOutput:
        normalized_path = normalize_video_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid video directory: {directory_path}")
        files = list_video_files(normalized_path)
        end_index = int(start_index) + int(load_count)
        selected = files[int(start_index):end_index]
        if not selected:
            raise ValueError(
                f"No videos found in range [{start_index}:{end_index}] in directory: {directory_path}"
            )
        return io.NodeOutput([InputImpl.VideoFromFile(path) for path in selected])

    @classmethod
    def IS_CHANGED(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = normalize_video_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""
        selected = list_video_files(normalized_path)[int(start_index):int(start_index) + int(load_count)]
        digest = hashlib.sha256()
        for path in selected:
            stat = os.stat(path)
            digest.update(path.encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(str(stat.st_size).encode("ascii"))
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, directory_path: str, start_index: int, load_count: int):
        normalized_path = normalize_video_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return f"Invalid video directory: {directory_path}"
        if int(start_index) < 0 or int(load_count) < 1:
            return "Video directory start_index must be >= 0 and load_count must be >= 1"
        if not list_video_files(normalized_path)[int(start_index):int(start_index) + int(load_count)]:
            return f"No videos found in requested range: {directory_path}"
        return True


class UC_SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_SaveVideo",
            display_name="Save Video (Keyframe Optimized)",
            category="advanced/video",
            search_aliases=["save video", "export video", "video combine", "nvenc video"],
            description="Saves video with clean seekable keyframes at scene cuts so players and video editors scrub and edit smoothly without black frames or visual glitching.",
            inputs=[
                io.Video.Input("video", optional=True, tooltip="Connect a video to save. Either video or images must be connected."),
                io.Image.Input("images", optional=True, tooltip="Connect image frames to save. Either images or video must be connected."),
                io.Audio.Input("audio", optional=True, tooltip="Optional sound track to include. If empty and video is connected, keeps the video's existing sound."),
                io.Float.Input("fps", default=24.0, min=0.01, max=240.0, step=0.01, tooltip="Playback speed in frames per second. Matches input video speed automatically when video is connected."),
                io.Combo.Input(
                    "codec",
                    options=["auto", "h264_nvenc", "hevc_nvenc", "libx264", "libx265"],
                    default="auto",
                    tooltip="Compression format. 'auto' uses NVIDIA GPU acceleration when available for fast export, or high-quality CPU encoding otherwise.",
                ),
                io.Int.Input(
                    "gop_size",
                    default=100,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Maximum spacing between keyframes. Lower values make seeking and scrubbing smoother in video players and editors.",
                ),
                io.Int.Input(
                    "b_frames",
                    default=0,
                    min=0,
                    max=16,
                    step=1,
                    tooltip="Extra compression between frames. Keep at 0 for smooth scrubbing and frame-by-frame cutting in video editors; raise to 2-4 for smaller file sizes.",
                ),
                io.Float.Input(
                    "bitrate_mbps",
                    default=10.0,
                    min=0.1,
                    max=200.0,
                    step=0.5,
                    tooltip="Target video quality and size for GPU export (in megabits per second). Higher values preserve finer visual detail.",
                ),
                io.Float.Input(
                    "crf",
                    default=19.0,
                    min=0.0,
                    max=51.0,
                    step=1.0,
                    tooltip="Visual quality level for CPU export. Lower numbers mean higher quality (19 is visually near-lossless).",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default",
                    tooltip="Output folder and filename. Supports date stamps like %date:yyyy-MM-dd% to organize files automatically by day.",
                ),
                io.Boolean.Input(
                    "save_workflow_image",
                    default=True,
                    optional=True,
                    tooltip="Saves the first frame as a PNG image with embedded ComfyUI workflow metadata alongside the video.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                io.Image.Output("images", display_name="IMAGES", tooltip="The raw video frames."),
                io.String.Output("filepath", display_name="Filepath", tooltip="Full file path to the saved video on your computer."),
                io.Video.Output("video", display_name="VIDEO", tooltip="The saved video ready to connect to downstream video nodes."),
                io.Audio.Output("audio", display_name="AUDIO", tooltip="The sound track included in the video."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor = None,
        video: Any = None,
        fps: float = 24.0,
        codec: str = "auto",
        gop_size: int = 100,
        b_frames: int = 0,
        bitrate_mbps: float = 10.0,
        crf: float = 19.0,
        audio: Any = None,
        filename_prefix: str = "video/%date:yyyy-MM-dd%/vid_%date:yyyy-MM-dd%_default",
        save_workflow_image: bool = True,
    ) -> io.NodeOutput:
        actual_images = images
        actual_audio = audio
        actual_fps = fps

        if actual_images is None:
            if video is None:
                raise ValueError("UC_SaveVideo requires either 'images' or 'video' to be connected.")
            comps = video.get_components()
            actual_images = comps.images
            if actual_audio is None:
                actual_audio = comps.audio
            if fps == 24.0 and getattr(comps, "frame_rate", None):
                actual_fps = float(comps.frame_rate)

        prompt = cls.hidden.prompt if cls.hidden is not None else None
        extra_pnginfo = cls.hidden.extra_pnginfo if cls.hidden is not None else None
        output_path, file, subfolder = save_keyframe_optimized_video(
            images=actual_images,
            fps=actual_fps,
            codec=codec,
            gop_size=gop_size,
            b_frames=b_frames,
            bitrate_mbps=bitrate_mbps,
            crf=crf,
            audio=actual_audio,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            save_workflow_image=save_workflow_image,
        )
        video_obj = InputImpl.VideoFromFile(output_path)
        return io.NodeOutput(
            actual_images,
            output_path,
            video_obj,
            actual_audio,
            ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]),
        )


class UC_ConcatenateVideoList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ConcatenateVideoList",
            display_name="Concatenate Video (List)",
            category="video",
            is_input_list=True,
            description="Stitches multiple video clips together back-to-back into one continuous video. Connect a list of videos from a queue or multi-clip workflow.",
            inputs=[
                io.Video.Input("videos", tooltip="Connect video clips to stitch together in order. Accepts a list of videos output across a queue run or list generation."),
                io.Audio.Input(
                    "complete_audio",
                    optional=True,
                    tooltip="Optional replacement audio for the whole joined video. If disconnected, automatically stitches the audio from each video clip together so nothing is lost.",
                ),
            ],
            outputs=[
                io.Video.Output("video", display_name="VIDEO", tooltip="The combined continuous video, ready for playback or saving."),
            ],
        )

    @classmethod
    def execute(cls, videos, complete_audio=None) -> io.NodeOutput:
        valid_videos = []
        if isinstance(videos, list):
            for v in videos:
                if v is not None:
                    if isinstance(v, list):
                        valid_videos.extend([item for item in v if item is not None])
                    else:
                        valid_videos.append(v)
        elif videos is not None:
            valid_videos.append(videos)

        if not valid_videos:
            return io.NodeOutput(None)
        if len(valid_videos) == 1 and complete_audio is None:
            return io.NodeOutput(valid_videos[0])

        audio_track = complete_audio[0] if (complete_audio and isinstance(complete_audio, list)) else complete_audio
        total_frames = 0
        total_duration = 0.0
        components_list = []
        for v in tqdm(valid_videos, desc="Inspecting Video Clips for Concatenation", unit="clip"):
            try:
                comp = v.get_components()
                components_list.append(comp)
                fps = float(getattr(comp, "frame_rate", 24.0) or 24.0)
                f_count = len(comp.images) if getattr(comp, "images", None) is not None else 0
                total_frames += f_count
                total_duration += (f_count / fps) if fps > 0 else 0.0
            except Exception:
                components_list.append(None)

        if audio_track is None:
            has_any_audio = any(comp is not None and comp.audio is not None for comp in components_list)
            if has_any_audio:
                collected_audios = []
                for comp in components_list:
                    if comp is None:
                        continue
                    if comp.audio is not None:
                        collected_audios.append(comp.audio)
                    else:
                        fps = float(getattr(comp, "frame_rate", 24.0) or 24.0)
                        duration = len(comp.images) / fps
                        silence = {"waveform": torch.zeros(1, 2, round(duration * 44100)), "sample_rate": 44100}
                        collected_audios.append(silence)
                audio_track = concatenate_audio_list(collected_audios)

        cat_video = InputImpl.VideoFromList(valid_videos, audio_track, Types.VideoCodec.AUTO)
        audio_mode = "override" if complete_audio is not None else ("stitched" if audio_track is not None else "none")
        logging.info(
            "[UC_ConcatenateVideoList] Joined %d video clip(s) -> %d total frame(s) (%.2fs) | audio=%s",
            len(valid_videos), total_frames, total_duration, audio_mode,
        )
        return io.NodeOutput(cat_video)

