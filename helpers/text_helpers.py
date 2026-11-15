import json
import os
import re
from typing import Any, Union

H3_PROMPT_CHANNELS = ("visual", "speech", "sounds", "music")
_H3_TIME = re.compile(r"(?:(\d+):)?(\d+)(?:\.(\d{1,3}))?")


def default_h3_prompt_state() -> dict:
    return {
        "version": 2,
        "precision": 2,
        "active_tab": "subjects",
        "definitions": [],
        "summary": {"task_types": ["reference generation"], "text": ""},
        "retention": [],
        "detailed": {"has_timeline": True, "continuous_text": "", "segment_duration": 2.333},
        "segments": [],
        "overall_soundscape": "",
        "non_diegetic_music": "N/A",
    }


def _format_seconds(val: Any, precision: int = 2) -> str:
    """Format duration in seconds to padded 00.00s or 00.000s per Principle 4."""
    width = precision + 3
    if isinstance(val, (int, float)):
        sec = max(0.0, float(val))
        return f"{sec:0{width}.{precision}f}s"
    s = str(val).strip()
    if s.endswith("s"):
        s = s[:-1]
    if ":" in s:
        parts = s.split(":")
        try:
            sec = max(0.0, int(parts[0]) * 60 + float(parts[1]))
            return f"{sec:0{width}.{precision}f}s"
        except Exception:
            return f"{val}"
    try:
        sec = max(0.0, float(s))
        return f"{sec:0{width}.{precision}f}s"
    except Exception:
        return f"{val}s"


def compile_h3_prompt(serialized: str) -> str:
    """Compile the interactive H3 prompt builder state into MiniMax H3 text adhering to Principle 4."""
    try:
        state = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("prompt_state: invalid JSON.") from error

    if not isinstance(state, dict):
        raise ValueError("prompt_state: expected dictionary.")

    precision = state.get("precision", 2)
    if not isinstance(precision, int) or precision not in (2, 3):
        raise ValueError("prompt_state.precision: choose 2 or 3 decimals.")

    blocks = []

    # 1. subject_definitions:
    raw_defs = state.get("definitions") or state.get("subjects") or []
    subject_lines = []
    for item in raw_defs:
        if isinstance(item, str) and item.strip():
            subject_lines.append(item.strip())
        elif isinstance(item, dict):
            raw = item.get("raw", "").strip()
            if raw:
                subject_lines.append(raw)
                continue

            kind = item.get("kind", item.get("type", "subject"))
            d_id = item.get("id", 1)
            details = item.get("text", "").strip()
            role = item.get("role", "custom")

            if kind == "subject":
                has_ref = item.get("has_ref")
                if has_ref is None:
                    has_ref = item.get("ref_type") not in (None, "none")

                if has_ref:
                    ref_type = item.get("ref_type", "picture")
                    ref_index = item.get("ref_index", 1)
                    ref_tag = f"<{'Video' if ref_type == 'video' else 'Picture'} {ref_index}>"
                    prefix = f"<Subject {d_id}> is fully referenced in {ref_tag}:"
                    subject_lines.append(f"{prefix} {details}".strip())
                else:
                    if details.startswith("is ") or details.startswith("are "):
                        subject_lines.append(f"<Subject {d_id}> {details}")
                    else:
                        subject_lines.append(f"<Subject {d_id}> is {details}" if details else f"<Subject {d_id}> is a primary visual subject.")

            elif kind == "picture":
                if role == "first_frame":
                    subject_lines.append(f"<Picture {d_id}> is the fixed first frame anchor at 00.00s.")
                elif role == "final_frame":
                    subject_lines.append(f"<Picture {d_id}> is the fixed final frame anchor at video endpoint.")
                elif role == "storyboard":
                    subject_lines.append(f"<Picture {d_id}> is a storyboard reference for [Shot 1] and [Shot 2], defining viewpoint and subject placement.")
                else:
                    subject_lines.append(f"<Picture {d_id}>: {details}" if details else f"<Picture {d_id}>:")

            elif kind == "video":
                if role == "edit":
                    subject_lines.append(f"<Video {d_id}> is the source video for the target video edit.")
                elif role == "continue":
                    subject_lines.append(f"<Video {d_id}> is the source video for continuation.")
                elif role == "structure":
                    subject_lines.append(f"<Video {d_id}> provides camera movement, cuts, and temporal structure.")
                else:
                    subject_lines.append(f"<Video {d_id}>: {details}" if details else f"<Video {d_id}>:")

            elif kind == "audio":
                if role == "timbre":
                    spk = item.get("target_subject", 1)
                    subject_lines.append(f"<Audio {d_id}> is the voice-timbre reference for <Subject {spk}> (S{spk}).")
                elif role == "full":
                    subject_lines.append(f"<Audio {d_id}> is reused as the target video's complete final audio track.")
                elif role == "music":
                    subject_lines.append(f"<Audio {d_id}> is the music-style and rhythm reference.")
                else:
                    subject_lines.append(f"<Audio {d_id}>: {details}" if details else f"<Audio {d_id}>:")

    if subject_lines:
        blocks.append("subject_definitions:\n" + "\n".join(subject_lines))

    # 2. summary:
    summary_data = state.get("summary", {})
    if isinstance(summary_data, str):
        summary_text = summary_data.strip()
        task_types = []
    elif isinstance(summary_data, dict):
        summary_text = summary_data.get("text", "").strip()
        task_types = summary_data.get("task_types", [])
    else:
        summary_text = ""
        task_types = []

    if summary_text:
        if task_types:
            ordered_types = []
            if "video editing" in task_types:
                ordered_types.append("video editing")
            for t in task_types:
                if t != "video editing" and t not in ordered_types:
                    ordered_types.append(t)
            prefix = f"[{' + '.join(ordered_types)}]"
            summary_body = f"{prefix} {summary_text}".strip()
        else:
            summary_body = summary_text
        blocks.append(f"summary:\n{summary_body}")

    # 3. retention_analysis:
    retention = state.get("retention", [])
    retention_lines = []
    for item in retention:
        if isinstance(item, str) and item.strip():
            retention_lines.append(item.strip())
        elif isinstance(item, dict):
            label = item.get("label", "").strip()
            marker = item.get("marker", "").strip()
            desc = item.get("text", "") or item.get("descriptor", "")
            desc = desc.strip()
            if label and marker:
                if desc:
                    retention_lines.append(f"{label}: {marker} - {desc}")
                else:
                    retention_lines.append(f"{label}: {marker}")

    if retention_lines:
        blocks.append("retention_analysis:\n" + "\n".join(retention_lines))

    # 4. detailed_description:
    detailed_data = state.get("detailed", {})
    has_timeline = detailed_data.get("has_timeline", True) if isinstance(detailed_data, dict) else True
    continuous_text = detailed_data.get("continuous_text", "").strip() if isinstance(detailed_data, dict) else ""
    segments = state.get("segments", [])

    if has_timeline and segments:
        timeline_segments = []
        for index, seg in enumerate(segments, 1):
            if not isinstance(seg, dict):
                continue
            start_str = _format_seconds(seg.get("start", 0), precision)
            end_str = _format_seconds(seg.get("end", 0), precision)

            lines = []
            visual = seg.get("visual", "").strip()
            has_shot = seg.get("has_shot", False)
            shot_num = seg.get("shot", index)
            if visual or has_shot:
                if has_shot and not visual.startswith(f"[Shot {shot_num}]"):
                    shot_prefix = f"[Shot {shot_num}] "
                else:
                    shot_prefix = ""
                lines.append(f"[VISUAL]: {shot_prefix}{visual}".strip())

            speech = seg.get("speech", {})
            if isinstance(speech, dict) and speech.get("enabled"):
                speaker = speech.get("speaker", "S1").strip()
                lang = speech.get("language", "English").strip()
                s_text = speech.get("text", "").strip()
                subj = speech.get("subject", "").strip()
                subj_prefix = f"{subj} " if subj else ""
                speaker_tag = f"({speaker}) " if speaker else ""
                d_body = f"<d>[{lang}] {s_text}</d>" if s_text else ""
                lines.append(f"[SPEECH]: {subj_prefix}{speaker_tag}{d_body}".strip())
            elif isinstance(speech, str) and speech.strip():
                lines.append(f"[SPEECH]: {speech.strip()}")

            sounds = seg.get("sounds", {})
            if isinstance(sounds, dict) and sounds.get("enabled"):
                snd_text = sounds.get("text", "").strip()
                if snd_text:
                    lines.append(f"[SOUNDS]: {snd_text}")
            elif isinstance(sounds, str) and sounds.strip():
                lines.append(f"[SOUNDS]: {sounds.strip()}")

            music = seg.get("music", {})
            if isinstance(music, dict) and music.get("enabled"):
                mus_text = music.get("text", "").strip()
                if mus_text:
                    lines.append(f"[MUSIC]: {mus_text}")
            elif isinstance(music, str) and music.strip():
                lines.append(f"[MUSIC]: {music.strip()}")

            channels = seg.get("channels", {})
            if isinstance(channels, dict):
                for ch in ("visual", "speech", "sounds", "music"):
                    ch_entry = channels.get(ch, {})
                    if isinstance(ch_entry, dict) and ch_entry.get("enabled") and ch_entry.get("text", "").strip():
                        tag = ch.upper()
                        if not any(l.startswith(f"[{tag}]:") for l in lines):
                            lines.append(f"[{tag}]: {ch_entry['text'].strip()}")

            if lines:
                timeline_segments.append(f"[{start_str}-{end_str}]:\n" + "\n".join(lines))

        if timeline_segments:
            blocks.append("detailed_description:\nTimeline:\n" + "\n\n".join(timeline_segments))
    elif continuous_text:
        blocks.append(f"detailed_description:\n{continuous_text}")

    # 5. overall_soundscape:
    soundscape = state.get("overall_soundscape", "").strip()
    if soundscape:
        blocks.append(f"overall_soundscape:\n{soundscape}")

    # 6. non_diegetic_music:
    music = state.get("non_diegetic_music", "").strip()
    if music and blocks:
        blocks.append(f"non_diegetic_music:\n{music}")
    elif music and music != "N/A":
        blocks.append(f"non_diegetic_music:\n{music}")

    return "\n\n".join(blocks)


def default_h3_base_prompt_state() -> dict[str, Any]:
    return {"version": 1, "task": "T2VA", "duration": 5.0, "final_shot": 1, "auto_final_shot": True, "precision": 2, "active_tab": "task", "description": {"mode": "timeline", "continuous_text": "", "segment_duration": 2.333}, "segments": [], "overall_soundscape": "", "non_diegetic_music": "N/A"}


def compile_h3_base_prompt(prompt_state: Union[str, dict[str, Any]]) -> str:
    if isinstance(prompt_state, str):
        try:
            state = json.loads(prompt_state)
        except Exception:
            return prompt_state
    elif isinstance(prompt_state, dict):
        state = prompt_state
    else:
        return ""

    if not isinstance(state, dict):
        return ""

    task = str(state.get("task", "T2VA")).upper().strip()
    precision = 3 if state.get("precision") == 3 else 2
    duration = float(state.get("duration", 5.0) or 5.0)

    desc_data = state.get("description", {})
    if not isinstance(desc_data, dict):
        desc_data = {}
    mode = desc_data.get("mode", "timeline")
    continuous_text = str(desc_data.get("continuous_text", "")).strip()
    segments = state.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    # Determine final shot index
    final_shot = int(state.get("final_shot", 1) or 1)
    if state.get("auto_final_shot", True) and segments:
        max_shot = 1
        for idx, seg in enumerate(segments, 1):
            if isinstance(seg, dict):
                s = seg.get("shot")
                if s is not None:
                    try:
                        max_shot = max(max_shot, int(s))
                    except (ValueError, TypeError):
                        pass
                elif seg.get("has_shot", False):
                    max_shot = max(max_shot, idx)
        final_shot = max_shot

    # Part One: Keyframe alignment instruction (if any)
    instruction = ""
    dur_str = f"{duration:.2f}"
    if task == "I2VA":
        instruction = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    elif task == "FL2VA":
        instruction = f"How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {final_shot}) aligns with the {dur_str}-second mark of the target video."
    elif task == "L2VA":
        instruction = f"How the reference pictures align with the target video — <Picture 1> (from [Shot {final_shot}]) aligns with the {dur_str}-second mark of the target video."

    blocks: list[str] = []

    # Field 1: integrated_multimodal_description:
    if mode == "timeline" and segments:
        timeline_segments = []
        for index, seg in enumerate(segments, 1):
            if not isinstance(seg, dict):
                continue
            start_str = _format_seconds(seg.get("start", 0), precision)
            end_str = _format_seconds(seg.get("end", 0), precision)

            lines = []
            visual = str(seg.get("visual", "")).strip()
            has_shot = seg.get("has_shot", False)
            shot_num = seg.get("shot", index)
            if visual or has_shot:
                if has_shot and not visual.startswith(f"[Shot {shot_num}]"):
                    shot_prefix = f"[Shot {shot_num}] "
                else:
                    shot_prefix = ""
                lines.append(f"[VISUAL]: {shot_prefix}{visual}".strip())

            speech = seg.get("speech", {})
            if isinstance(speech, dict) and speech.get("enabled", False):
                spk = str(speech.get("speaker", "S1")).strip() or "S1"
                lang = str(speech.get("language", "English")).strip() or "English"
                txt = str(speech.get("text", "")).strip()
                lines.append(f"[SPEECH]: ({spk}) <d>[{lang}] {txt}</d>".strip())

            sounds = seg.get("sounds", {})
            if isinstance(sounds, dict) and sounds.get("enabled", False):
                s_txt = str(sounds.get("text", "")).strip()
                if s_txt:
                    lines.append(f"[SOUNDS]: {s_txt}")

            music_chan = seg.get("music", {})
            if isinstance(music_chan, dict) and music_chan.get("enabled", False):
                m_txt = str(music_chan.get("text", "")).strip()
                if m_txt:
                    lines.append(f"[MUSIC]: {m_txt}")

            if lines:
                timeline_segments.append(f"[{start_str}-{end_str}]:\n" + "\n".join(lines))

        if timeline_segments:
            blocks.append("integrated_multimodal_description:\nTimeline:\n" + "\n\n".join(timeline_segments))
    elif continuous_text:
        blocks.append(f"integrated_multimodal_description:\n{continuous_text}")

    # Field 2: overall_soundscape:
    soundscape = str(state.get("overall_soundscape", "")).strip()
    if soundscape:
        blocks.append(f"overall_soundscape:\n{soundscape}")

    # Field 3: non_diegetic_music:
    music = str(state.get("non_diegetic_music", "")).strip()
    if music and blocks:
        blocks.append(f"non_diegetic_music:\n{music}")
    elif music and music != "N/A":
        blocks.append(f"non_diegetic_music:\n{music}")

    if not blocks and not instruction:
        return ""

    parts = []
    if instruction:
        parts.append(instruction)
    if blocks:
        parts.append("\n\n".join(blocks))

    return "\n\n".join(parts)


def concatenate_aligned_text_inputs(text_inputs: dict[str, list[Any]] | None, delimiter_values: list[Any] | None) -> list[str]:
    ordered_inputs = [values for _, values in sorted((text_inputs or {}).items(), key=lambda item: int(item[0].removeprefix("text_")))]
    output_count = max((len(values) for values in ordered_inputs), default=1)
    delimiters = delimiter_values or [""]

    outputs = []
    for index in range(output_count):
        delimiter = delimiters[min(index, len(delimiters) - 1)]
        parts = [values[min(index, len(values) - 1)] if values else "" for values in ordered_inputs]
        outputs.append(str(delimiter).join(str(value) for value in parts))
    return outputs


TEXT_FILE_DELIMITER_OPTIONS = ["Newline [\n] or [\r\n]", 'Triple Quote ["""]', "Triple Equals [===]", "Triple Dash [---]", "Period [.]", "Comma [,]", "Colon [:]", "Semicolon [;]", "Space [ ]", "Tab [\t]", "Pipe [|]", "Double Newline [\n\n] (Paragraphs)", "None (Single Entry)", "Custom"]

TEXT_FILE_DELIMITER_MAP = {
    "Newline [\n] or [\r\n]": r"\r?\n",
    'Triple Quote ["""]': '"""',
    "Triple Equals [===]": "===",
    "Triple Dash [---]": "---",
    "Period [.]": ".",
    "Comma [,]": ",",
    "Colon [:]": ":",
    "Semicolon [;]": ";",
    "Space [ ]": " ",
    "Tab [\t]": "\t",
    "Pipe [|]": "|",
    "Double Newline [\n\n] (Paragraphs)": r"(?:\r?\n){2,}",
}


def normalize_text_file_path(path: str) -> str:
    if not path:
        return ""
    path = path.strip().strip('"').strip("'")
    path = path.replace("\\", "/")
    path = os.path.normpath(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def read_arbitrary_text_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return raw_bytes.decode("utf-8-sig", errors="replace")
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw_bytes.decode("utf-16", errors="replace")
    if raw_bytes.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return raw_bytes.decode("utf-32", errors="replace")

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("utf-8", errors="replace")


def split_text_content(content: str, delimiter: str, custom_delimiter: str = "", trim_entries: bool = True, skip_empty: bool = True) -> list[str]:
    if delimiter == "Newline [\n] or [\r\n]":
        parts = re.split(r"\r?\n", content)
    elif delimiter == "Double Newline [\n\n] (Paragraphs)":
        parts = re.split(r"(?:\r?\n){2,}", content)
    elif delimiter == "None (Single Entry)":
        parts = [content]
    elif delimiter == "Custom":
        if custom_delimiter:
            parts = content.split(custom_delimiter)
        else:
            parts = [content]
    else:
        delim = TEXT_FILE_DELIMITER_MAP.get(delimiter, delimiter)
        parts = content.split(delim)

    if trim_entries:
        parts = [p.strip() for p in parts]

    if skip_empty:
        parts = [p for p in parts if p]

    return parts


def load_text_file_as_list(file_path: str, delimiter: str = "Newline [\n] or [\r\n]", custom_delimiter: str = "", trim_entries: bool = True, skip_empty: bool = True) -> list[str]:
    normalized = normalize_text_file_path(file_path)
    if not normalized or not os.path.isfile(normalized):
        raise ValueError(f"Invalid file path: {file_path} (resolved to: {normalized})")
    content = read_arbitrary_text_file(normalized)
    return split_text_content(content, delimiter=delimiter, custom_delimiter=custom_delimiter, trim_entries=trim_entries, skip_empty=skip_empty)


TEXT_FILE_EXTENSIONS = {".txt"}


def list_text_files(directory: str, extensions: set[str] | None = None) -> list[str]:
    allowed = extensions if extensions is not None else TEXT_FILE_EXTENSIONS
    return sorted(os.path.join(directory, name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name)) and os.path.splitext(name)[1].lower() in allowed)


def load_text_files_from_directory(directory_path: str, start_index: int = 0, load_count: int = 1) -> list[str]:
    normalized = normalize_text_file_path(directory_path)
    if not normalized or not os.path.isdir(normalized):
        raise ValueError(f"Invalid text directory: {directory_path}")
    start = int(start_index)
    count = int(load_count)
    if start < 0 or count < 1:
        raise ValueError("Text directory start_index must be >= 0 and load_count must be >= 1.")
    end = start + count
    selected = list_text_files(normalized)[start:end]
    if not selected:
        raise ValueError(f"No .txt files found in range [{start}:{end}] in directory: {directory_path}")
    return [read_arbitrary_text_file(path) for path in selected]


def fingerprint_text_directory(directory_path: str, start_index: int = 0, load_count: int = 1) -> str:
    import hashlib

    normalized = normalize_text_file_path(directory_path)
    if not normalized or not os.path.isdir(normalized):
        return ""
    start = int(start_index)
    count = int(load_count)
    if start < 0 or count < 1:
        return ""
    selected = list_text_files(normalized)[start : start + count]
    digest = hashlib.sha256()
    for path in selected:
        stat = os.stat(path)
        digest.update(path.encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def validate_text_directory(directory_path: str, start_index: int = 0, load_count: int = 1):
    normalized = normalize_text_file_path(directory_path)
    if not normalized:
        return "Directory path cannot be empty"
    if not os.path.isdir(normalized):
        return f"Invalid text directory: {directory_path}"
    start = int(start_index)
    count = int(load_count)
    if start < 0 or count < 1:
        return "Text directory start_index must be >= 0 and load_count must be >= 1"
    if not list_text_files(normalized)[start : start + count]:
        return f"No .txt files found in requested range: {directory_path}"
    return True


_MOSS_IMAGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "object",
            "properties": {
                "meta": {"type": "array", "description": "1 to 3 concise medium or genre tags describing the image format or genre.", "items": {"type": "string"}},
                "canvas": {"type": "string", "enum": ["single", "multi_panel", "collage"], "description": "Canvas layout type: single, multi_panel, or collage."},
                "caption": {"type": "string", "description": "Whole-scene description combining subject overview, camera viewpoint, background environment, lighting, medium, and aesthetics."},
            },
            "required": ["meta", "canvas", "caption"],
        },
        "items": {
            "type": "array",
            "description": "Localized subjects, objects, and rendered text ordered from background to foreground.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["character", "object", "text"], "description": "Item category: character for subjects/creatures, object for props/clothing/body-part sub-boxes, text for rendered text."},
                    "caption": {"type": "string", "description": "Detailed visual description of this item (30 to 60 words)."},
                    "text": {"type": "string", "description": "Verbatim rendered characters when label is text."},
                    "palette": {"type": "array", "description": "2 to 5 dominant colors of this item as uppercase #RRGGBB hex codes.", "items": {"type": "string"}},
                    "bbox": {"type": "array", "description": "Four integers [ymin, xmin, ymax, xmax] on a 0..1000 normalized top-left grid.", "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
                },
                "required": ["label", "caption", "palette", "bbox"],
            },
        },
        "bbox_coord_space": {"type": "string", "enum": ["abs_centralized"]},
        "bbox_format": {"type": "string", "enum": ["yxyx"]},
    },
    "required": ["summary", "items", "bbox_coord_space", "bbox_format"],
}

_MOSS_IMAGE_SHORT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "object",
            "properties": {
                "meta": {"type": "array", "description": "1 to 3 concise medium or genre tags describing the image format or genre.", "items": {"type": "string"}},
                "canvas": {"type": "string", "enum": ["single", "multi_panel", "collage"], "description": "Canvas layout type: single, multi_panel, or collage."},
                "caption": {"type": "string", "description": "Concise whole-scene description combining subject overview, camera viewpoint, background environment, lighting, medium, and aesthetics."},
            },
            "required": ["meta", "canvas", "caption"],
        },
        "items": {
            "type": "array",
            "description": "Localized subjects, objects, and rendered text ordered from background to foreground.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["character", "object", "text"]},
                    "caption": {"type": "string", "description": "Detailed visual description of this item (30 to 60 words)."},
                    "text": {"type": "string", "description": "Verbatim rendered characters when label is text."},
                    "bbox": {"type": "array", "description": "Four integers [ymin, xmin, ymax, xmax] on a 0..1000 normalized top-left grid.", "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
                },
                "required": ["label", "caption", "bbox"],
            },
        },
        "bbox_coord_space": {"type": "string", "enum": ["abs_centralized"]},
        "bbox_format": {"type": "string", "enum": ["yxyx"]},
    },
    "required": ["summary", "items", "bbox_coord_space", "bbox_format"],
}

_MOSS_IMAGE_LAYOUT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "object",
            "properties": {
                "meta": {"type": "array", "description": "1 to 3 concise medium or genre tags describing the image format or genre.", "items": {"type": "string"}},
                "canvas": {"type": "string", "enum": ["single", "multi_panel", "collage"], "description": "Canvas layout type: single, multi_panel, or collage."},
                "caption": {"type": "string", "description": "Complete natural language scene description combining viewpoint, spatial arrangement, primary subjects, actions, background environment, lighting, medium, and atmosphere."},
            },
            "required": ["meta", "canvas", "caption"],
        },
        "items": {
            "type": "array",
            "description": "Coherent visible subjects, physical objects, and rendered text regions in the resulting scene ordered from background to foreground and anchored to the first input image's spatial layout.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["character", "object", "text"], "description": "Item category: character for human, humanoid, anthropomorphic, feral, or creature subjects; object for distinct physical objects or props; text for visible rendered text regions."},
                    "caption": {"type": "string", "description": "Detailed natural language visual description of this coherent item's identity, species, gender, anatomy, pose, action, interactions, colors, and surface traits in the resulting scene, or typography and surface treatment for text items."},
                    "text": {"type": "string", "description": "Exact verbatim visible characters when label is text."},
                    "bbox": {"type": "array", "description": "Four integers [ymin, xmin, ymax, xmax] on a 0 to 1000 top left grid measured strictly from the first input image's canvas coordinates.", "items": {"type": "integer", "minimum": 0, "maximum": 1000}},
                },
                "required": ["label", "caption", "bbox"],
            },
        },
        "bbox_coord_space": {"type": "string", "enum": ["abs_centralized"]},
        "bbox_format": {"type": "string", "enum": ["yxyx"]},
    },
    "required": ["summary", "items", "bbox_coord_space", "bbox_format"],
}

_IDEOGRAM_4_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "high_level_description": {"type": "string"},
        "style_description": {
            "type": "object",
            "properties": {"aesthetics": {"type": "string"}, "lighting": {"type": "string"}, "photo": {"type": "string"}, "medium": {"type": "string"}, "color_palette": {"type": "array", "items": {"type": "string"}}},
            "required": ["aesthetics", "lighting", "photo", "medium", "color_palette"],
        },
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"type": {"type": "string", "enum": ["obj", "text"]}, "bbox": {"type": "array", "items": {"type": "integer"}}, "desc": {"type": "string"}, "text": {"type": "string"}, "color_palette": {"type": "array", "items": {"type": "string"}}},
                        "required": ["type", "bbox", "desc", "color_palette"],
                    },
                },
            },
            "required": ["background", "elements"],
        },
    },
    "required": ["high_level_description", "style_description", "compositional_deconstruction"],
}

_IDEOGRAM_4_SHORT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aspect_ratio": {"type": "string"},
        "high_level_description": {"type": "string"},
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "elements": {"type": "array", "items": {"type": "object", "properties": {"type": {"type": "string", "enum": ["obj", "text"]}, "bbox": {"type": "array", "items": {"type": "integer"}}, "text": {"type": "string"}, "desc": {"type": "string"}}, "required": ["type", "bbox", "desc"]}},
            },
            "required": ["background", "elements"],
        },
    },
    "required": ["aspect_ratio", "high_level_description", "compositional_deconstruction"],
}

_IDEOGRAM_4_STYLE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aspect_ratio": {"type": "string"},
        "high_level_description": {"type": "string"},
        "style_description": {"type": "object", "properties": {"aesthetics": {"type": "string"}, "lighting": {"type": "string"}, "photo": {"type": "string"}, "medium": {"type": "string"}, "art_style": {"type": "string"}}, "required": ["aesthetics", "lighting", "medium"]},
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "elements": {"type": "array", "items": {"type": "object", "properties": {"type": {"type": "string", "enum": ["obj", "text"]}, "bbox": {"type": "array", "items": {"type": "integer"}}, "text": {"type": "string"}, "desc": {"type": "string"}}, "required": ["type", "bbox", "desc"]}},
            },
            "required": ["background", "elements"],
        },
    },
    "required": ["aspect_ratio", "high_level_description", "compositional_deconstruction"],
}

_IDEOGRAM_4_COLOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "aspect_ratio": {"type": "string"},
        "high_level_description": {"type": "string"},
        "style_description": {
            "type": "object",
            "properties": {"aesthetics": {"type": "string"}, "lighting": {"type": "string"}, "photo": {"type": "string"}, "medium": {"type": "string"}, "art_style": {"type": "string"}, "color_palette": {"type": "array", "items": {"type": "string"}}},
            "required": ["aesthetics", "lighting", "medium"],
        },
        "compositional_deconstruction": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"type": {"type": "string", "enum": ["obj", "text"]}, "bbox": {"type": "array", "items": {"type": "integer"}}, "text": {"type": "string"}, "desc": {"type": "string"}, "color_palette": {"type": "array", "items": {"type": "string"}}},
                        "required": ["type", "bbox", "desc"],
                    },
                },
            },
            "required": ["background", "elements"],
        },
    },
    "required": ["aspect_ratio", "high_level_description", "compositional_deconstruction"],
}

VLM_PRESET_RESPONSE_SCHEMAS: dict[str, dict[str, Any]] = {
    "moss_image_json_instruction": _MOSS_IMAGE_RESPONSE_SCHEMA,
    "moss_image_json_instruction_short": _MOSS_IMAGE_SHORT_RESPONSE_SCHEMA,
    "moss_image_json_instruction_color": _MOSS_IMAGE_RESPONSE_SCHEMA,
    "moss_image_json_instruction_neutral": _MOSS_IMAGE_LAYOUT_RESPONSE_SCHEMA,
    "moss_image_json_instruction_action": _MOSS_IMAGE_LAYOUT_RESPONSE_SCHEMA,
    "moss_image_json_instruction_qwen": _MOSS_IMAGE_RESPONSE_SCHEMA,
    "moss_image_json_instruction_short_qwen": _MOSS_IMAGE_SHORT_RESPONSE_SCHEMA,
    "moss_image_json_instruction_color_qwen": _MOSS_IMAGE_RESPONSE_SCHEMA,
    "moss_image_json_instruction_neutral_qwen": _MOSS_IMAGE_LAYOUT_RESPONSE_SCHEMA,
    "moss_image_json_instruction_action_qwen": _MOSS_IMAGE_LAYOUT_RESPONSE_SCHEMA,
    "ideogram_4_json_instruction": _IDEOGRAM_4_RESPONSE_SCHEMA,
    "ideogram_4_json_instruction_short": _IDEOGRAM_4_SHORT_RESPONSE_SCHEMA,
    "ideogram_4_json_instruction_style": _IDEOGRAM_4_STYLE_RESPONSE_SCHEMA,
    "ideogram_4_json_instruction_color": _IDEOGRAM_4_COLOR_RESPONSE_SCHEMA,
}


def get_vlm_preset_response_schema(preset: str) -> dict[str, Any] | None:
    schema = VLM_PRESET_RESPONSE_SCHEMAS.get(preset)
    if schema is None:
        return None
    return json.loads(json.dumps(schema))
