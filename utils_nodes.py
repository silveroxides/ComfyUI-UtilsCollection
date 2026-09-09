import sys
import random
from typing import Union
import json
import os
import torch
from comfy_api.latest import io
from comfy_extras.nodes_logic import SwitchNode, SoftSwitchNode
from .helper_functions import to_video_prompt
from .image_helpers import prepare_h3_reference_video_components

_MAX_SEED = 0xFFFFFFFFFFFFFFFF
SeedClusterType = io.Custom("UC_SEED_CLUSTER")


class UC_MiniMaxH3RefVid(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_MiniMaxH3RefVid",
            display_name="H3 Reference Video Components",
            category="advanced/video",
            description="Prepares video frames, audio, and matching generation dimensions for MiniMax H3 reference video workflows.",
            search_aliases=["minimax", "h3", "reference", "video", "components", "24 fps"],
            inputs=[
                io.Video.Input("video", tooltip="Reference clip. Its original frame rate is used to preserve playback speed when preparing 24 fps frames."),
                io.Float.Input("megapixels", default=0.258, min=0.01, max=4.0, step=0.001, tooltip="Target frame size. Matches the video to the nearest standard aspect ratio from Video Resolution Selector, chooses its preferred resolution, and center-crops to fit. Edges may be trimmed; the picture is not stretched."),
                io.Float.Input("duration_seconds", default=0.0, min=0.0, step=0.1, tooltip="Maximum reference duration in seconds. 0 uses the whole clip. The selected duration rounds up to a supported H3 frame count at 24 fps, so it may run slightly longer and repeat the final frame."),
            ],
            outputs=[
                io.Image.Output("frames", tooltip="Prepared 24 fps frames. Connect to H3 Reference to Video's reference-video input."),
                io.Audio.Output("audio", tooltip="Soundtrack matched to the prepared video and adjusted for H3. Missing audio at the end is filled with silence; a tiny silent tail may be added to prevent Core from cutting the start. Leave disconnected if no audio reference is wanted."),
                io.Int.Output("width", tooltip="Matching generation width in pixels."),
                io.Int.Output("height", tooltip="Matching generation height in pixels."),
                io.Int.Output("length", tooltip="Matching generation length in frames at 24 fps, including the H3 length adjustment."),
            ],
        )

    @classmethod
    def execute(cls, video, megapixels=0.258, duration_seconds=0.0):
        return io.NodeOutput(*prepare_h3_reference_video_components(video, megapixels, duration_seconds))


class UC_SeedCluster(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_SeedCluster",
            display_name="SeedCluster",
            category="utils/primitive",
            inputs=[
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=_MAX_SEED,
                    control_after_generate=True,
                    tooltip="Main seed using ComfyUI's generation-time seed control.",
                ),
                io.Int.Input(
                    "increment",
                    default=1,
                    min=1,
                    max=_MAX_SEED,
                    step=1,
                    tooltip="Amount added between consecutive seeds; values wrap within ComfyUI's seed range.",
                ),
            ],
            outputs=[
                io.Int.Output("seed", display_name="seed"),
                SeedClusterType.Output(
                    "seed_cluster",
                    display_name="Cluster",
                ),
            ],
        )

    @classmethod
    def execute(cls, seed: int, increment: int) -> io.NodeOutput:
        seed = int(seed) % (_MAX_SEED + 1)
        cluster = [
            (seed + index * int(increment)) % (_MAX_SEED + 1)
            for index in range(8)
        ]
        return io.NodeOutput(seed, cluster)


class UC_FromSeedCluster(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_FromSeedCluster",
            display_name="FromSeedCluster",
            category="utils/primitive",
            inputs=[
                SeedClusterType.Input(
                    "seed_cluster",
                    display_name="Cluster",
                    tooltip="Seed list produced by SeedCluster.",
                ),
            ],
            outputs=[
                io.Int.Output(f"seed_{index}", display_name=f"Seed {index}")
                for index in range(1, 9)
            ],
        )

    @classmethod
    def execute(cls, seed_cluster: list[int]) -> io.NodeOutput:
        cluster = [int(seed) % (_MAX_SEED + 1) for seed in seed_cluster[:8]]
        if not cluster:
            raise ValueError("FromSeedCluster requires at least one seed.")
        cluster.extend([cluster[-1]] * (8 - len(cluster)))
        return io.NodeOutput(*cluster)


class UC_FromList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        item_type = io.MatchType.Template("item_type")
        return io.Schema(
            node_id="UC_FromList",
            display_name="From List",
            category="utils/list",
            description="Returns a consecutive range from any ComfyUI list.",
            is_input_list=True,
            search_aliases=["List Slice", "Get List Items", "Select From List"],
            inputs=[
                io.MatchType.Input("items", template=item_type, tooltip="Input list retained in its existing order."),
                io.Int.Input("start_index", default=0, min=0, step=1, tooltip="Zero-based index of the first item returned."),
                io.Int.Input("number_of_entries", default=1, min=1, step=1, tooltip="Maximum consecutive items returned from the start index; stops at the end of the list."),
            ],
            outputs=[
                io.MatchType.Output(
                    item_type,
                    id="items",
                    display_name="items",
                    is_output_list=True,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        items: list,
        start_index: list[int],
        number_of_entries: list[int],
    ) -> io.NodeOutput:
        start = int(start_index[0])
        count = int(number_of_entries[0])
        return io.NodeOutput(items[start : start + count])


class UC_SwitchInverseNode(SwitchNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="UC_SwitchInverseNode",
            display_name="Switch (Inverse)",
            category="logic",
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_true", template=template, lazy=True),
                io.MatchType.Input("on_false", template=template, lazy=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )


class UC_SoftSwitchInverseNode(SoftSwitchNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="UC_SoftSwitchInverseNode",
            display_name="Soft Switch (Inverse)",
            category="logic",
            inputs=[
                io.Boolean.Input("switch"),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True),
            ],
            outputs=[
                io.MatchType.Output(template=template, display_name="output"),
            ],
        )

class UC_IntegerRangeRandom(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UC_IntegerRangeRandom",
            display_name="Random Integer in Range",
            category="utils/primitive",
            inputs=[
                io.Int.Input("minimum", min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("maximum", min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("seed", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output(display_name="random_integer")],
        )

    @classmethod
    def execute(cls, minimum: int, maximum: int, seed: int = 0) -> io.NodeOutput:
        min_val = min(minimum, maximum)
        max_val = max(minimum, maximum)
        rng = random.Random(seed)
        return io.NodeOutput(rng.randint(min_val, max_val))


class UC_GetJsonValue(io.ComfyNode):
    """Selects a value from a top-level JSON object without changing its type."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_GetJsonValue",
            display_name="Get JSON Value",
            category="utils/primitive",
            inputs=[
                io.String.Input(
                    "json_path",
                    default="./input/values.json",
                    multiline=False,
                    tooltip="Path to a JSON file containing a top-level object of named values.",
                ),
                io.Combo.Input(
                    "selection_mode",
                    options=["key", "random", "index"],
                    default="key",
                    tooltip="Select a value by its key, a deterministic random choice, or its insertion-order index.",
                ),
                io.String.Input(
                    "key",
                    default="",
                    multiline=False,
                    tooltip="Top-level JSON key to use when selection mode is key.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Zero-based insertion-order index to use when selection mode is index.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed used for deterministic random selection.",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    display_name="value",
                    tooltip="The selected JSON value in its original JSON type.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        json_path: str,
        selection_mode: str,
        key: str,
        index: int,
        seed: int,
    ) -> io.NodeOutput:
        absolute_path = os.path.abspath(json_path)
        try:
            with open(absolute_path, "r", encoding="utf-8") as file:
                values = json.load(file)
        except FileNotFoundError as exc:
            raise ValueError(f"JSON value file was not found: {absolute_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not decode JSON value file {absolute_path}: {exc.msg}.") from exc
        except OSError as exc:
            raise RuntimeError(f"Could not read JSON value file {absolute_path}: {exc}") from exc

        if not isinstance(values, dict):
            raise ValueError(f"JSON value file must contain a top-level object: {absolute_path}")
        if not values:
            raise ValueError(f"JSON value file is empty: {absolute_path}")

        if selection_mode == "key":
            if key not in values:
                raise ValueError(f"JSON key {key!r} was not found in {absolute_path}.")
            value = values[key]
        elif selection_mode == "random":
            value = random.Random(seed).choice(list(values.values()))
        elif selection_mode == "index":
            value_list = list(values.values())
            if index >= len(value_list):
                raise ValueError(
                    f"JSON value index {index} is out of range for {len(value_list)} value(s)."
                )
            value = value_list[index]
        else:
            raise ValueError(f"Unsupported JSON value selection mode: {selection_mode!r}")

        return io.NodeOutput(value)


class UC_ImageToVideoPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ImageToVideoPrompt",
            display_name="Image Prompt to Video Prompt",
            category="advanced/text",
            description=(
                "Converts image-editing prompt language into role-aware video generation "
                "instructions while preserving visual style, identity, and content requirements."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    tooltip="Image-oriented prompt text to convert into video-oriented guidance.",
                ),
                io.Combo.Input(
                    "prompt_role",
                    options=["general", "system", "instruction", "bonus"],
                    default="general",
                    tooltip=(
                        "Selects general, system, instruction, or bonus conversion behavior "
                        "without changing the node interface."
                    ),
                ),
            ],
            outputs=[io.String.Output("text", display_name="text")],
        )

    @classmethod
    def execute(cls, text: str, prompt_role="general") -> io.NodeOutput:
        return io.NodeOutput(to_video_prompt(text, role=prompt_role))


class UC_TagNormalizeCombine(io.ComfyNode):
    """
    Node that normalizes scores in two sets of tags and combines them,
    deduplicating and sorting by the normalized scores.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_TagNormalizeCombine",
            display_name="Tag Normalize and Combine",
            category="advanced/text",
            inputs=[
                io.String.Input("tags_1", multiline=True, default=""),
                io.String.Input("tags_2", multiline=True, default=""),
                io.AnyType.Input(
                    "scores_1",
                    tooltip="Dictionary of scores for tags_1",
                    optional=True,
                ),
                io.AnyType.Input(
                    "scores_2",
                    tooltip="Dictionary of scores for tags_2",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="deduped_tags"),
                io.AnyType.Output(display_name="normalized_scores"),
            ],
        )

    @staticmethod
    def normalize_scores(scores: dict, min_val=0.000001, max_val=0.999999) -> dict:
        if not scores:
            return {}

        current_scores = [float(v) for v in scores.values()]
        current_min = min(current_scores)
        current_max = max(current_scores)

        if current_max == current_min:
            return {k: max_val for k in scores.keys()}

        normalized = {}
        for k, v in scores.items():
            norm = min_val + (float(v) - current_min) / (current_max - current_min) * (
                max_val - min_val
            )
            normalized[k] = norm
        return normalized

    @classmethod
    def execute(
        cls, tags_1: Union[str, list], tags_2: Union[str, list], scores_1: Union[str, dict] = None, scores_2: Union[str, dict] = None
    ) -> io.NodeOutput:
        # Parse tags
        def parse_tags(t_input):
            if isinstance(t_input, list):
                return [str(t).strip() for t in t_input if t]
            if not t_input or not isinstance(t_input, str):
                return []
            # Split by comma and handle potential spaces
            return [t.strip() for t in t_input.split(",") if t.strip()]

        t1_list = parse_tags(tags_1)
        t2_list = parse_tags(tags_2)

        # Handle scores
        def process_scores(s_input, t_list):
            if s_input is None:
                # Generate even distribution
                if not t_list:
                    return {}
                num_tags = len(t_list)
                if num_tags == 1:
                    return {t_list[0]: 0.999999}

                max_v = 0.999999
                min_v = 0.000001
                scores = {}
                for i, tag in enumerate(t_list):
                    # Linearly interpolate from max_v down to min_v
                    score = max_v - i * (max_v - min_v) / (num_tags - 1)
                    scores[tag] = score
                return scores

            # Parse existing scores
            def parse_s(s_in):
                if isinstance(s_in, dict):
                    return s_in
                if not s_in or not isinstance(s_in, str):
                    return {}
                try:
                    return json.loads(s_in)
                except json.JSONDecodeError:
                    return {}

            return cls.normalize_scores(parse_s(s_input))

        norm_s1 = process_scores(scores_1, t1_list)
        norm_s2 = process_scores(scores_2, t2_list)

        # Combine and deduplicate
        combined_scores = {}

        # Process first set
        for t in t1_list:
            score = norm_s1.get(t, 0.000001)
            combined_scores[t] = score

        # Process second set with deduplication logic
        for t in t2_list:
            score = norm_s2.get(t, 0.000001)
            if t in combined_scores:
                # Keep the one with the highest normalized score
                if score > combined_scores[t]:
                    combined_scores[t] = score
            else:
                combined_scores[t] = score

        # Sort tags by normalized scores (descending)
        sorted_tags = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Prepare outputs
        deduped_tags_str = ", ".join(sorted_tags)
        normalized_scores_dict = {tag: combined_scores[tag] for tag in sorted_tags}

        return io.NodeOutput(deduped_tags_str, normalized_scores_dict)


class UC_RandInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_RandInt",
            display_name="RandomInt",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class UC_StaticInt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_StaticInt",
            display_name="StaticInt",
            category="utils/primitive",
            inputs=[
                io.Int.Input("value", min=-sys.maxsize, max=sys.maxsize),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)


class UC_StaticFloat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_StaticFloat",
            display_name="StaticFloat",
            category="utils/primitive",
            inputs=[
                io.Float.Input(
                    "value",
                    default=1.0,
                    min=-sys.float_info.max,
                    max=sys.float_info.max,
                    step=0.01,
                ),
            ],
            outputs=[io.Float.Output()],
        )

    @classmethod
    def execute(cls, value: float) -> io.NodeOutput:
        return io.NodeOutput(value)


class UC_RandIntRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_RandIntRange",
            display_name="RandomIntRange",
            category="utils/primitive",
            inputs=[
                io.Int.Input("min", default=0, min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("max", default=100, min=-sys.maxsize, max=sys.maxsize),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
            ],
            outputs=[io.Int.Output()],
        )

    @classmethod
    def execute(cls, min: int, max: int, seed: int) -> io.NodeOutput:
        rng = random.Random(seed)
        return io.NodeOutput(rng.randint(min, max))


class UC_ColorConvertNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ColorConvertNode",
            display_name="Color Convert",
            category="advanced/color",
            inputs=[
                io.Combo.Input("from_mode", options=["auto", "Manual Hex #FFFFFF", "Int 0-16777215", "Comma separated 255,255,255"], default="auto", tooltip="Select how to interpret the color input. 'Auto' will use the color picker input unless one of the other fields is filled in, in which case it will use the filled-in field based on the other options. The other modes will take precedence over the color picker input when their respective fields are filled in."),
                io.Color.Input("color_hex", default="#FFFFFF", display_name="This is not the Color Selector →", tooltip="Select a color using the color picker. This input is used when 'from_mode' is set to 'auto' and no other manual inputs are provided. The selected color will be converted to hex, int, and string formats for output."),
                io.String.Input("manual_hex", multiline=False, optional=True, display_name="Manual Hex Input", tooltip="Enter a hex color code manually, e.g. #FF00FF. Takes precedence over the color picker input if 'from_mode' is set to 'Manual Hex #FFFFFF'."),
                io.Int.Input("color_int", min=-1, max=16777215, default=-1, optional=True, display_name="Color Int Input", tooltip="Enter a color as an integer (0-16777215). Interpreted as 0xRRGGBB. Takes precedence over the color picker input if 'from_mode' is set to 'Int 0-16777215'."),
                io.String.Input("comma_separated", multiline=False, optional=True, display_name="Comma Separated Input", tooltip="Enter a color as comma-separated RGB values (0-255), e.g. 255,0,255. Takes precedence over the color picker input if 'from_mode' is set to 'Comma separated 255,255,255'."),
            ],
            outputs=[
                io.String.Output(display_name="Hex"),
                io.Int.Output(display_name="Int"),
                io.String.Output(display_name="RGB"),
            ]
        )

    @classmethod
    def _validate_hex(cls, s):
        """Return (r, g, b) if s is a valid 6-digit hex string (case-insensitive, # prefix optional), else None."""
        if s is None:
            return None
        s = s.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            return None
        if not all(c in "0123456789abcdefABCDEF" for c in s):
            return None
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)

    @classmethod
    def _validate_int(cls, n):
        """Return (r, g, b) if n is an int in 0–16777215, else None."""
        if n is None or not isinstance(n, int):
            return None
        if not (0 <= n <= 16777215):
            return None
        return ((n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)

    @classmethod
    def _validate_csv(cls, s):
        """Return (r, g, b) if s is 3 ints 0–255 separated by ',' or ', ', else None."""
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            return None
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None
        if not all(0 <= v <= 255 for v in vals):
            return None
        return tuple(vals)

    @classmethod
    def _rgb_to_outputs(cls, r, g, b):
        return (
            f"#{r:02X}{g:02X}{b:02X}",
            (r << 16) | (g << 8) | b,
            f"{r}, {g}, {b}",
        )

    @classmethod
    def execute(cls, from_mode, color_hex, manual_hex=None, color_int=None, comma_separated=None) -> io.NodeOutput:
        # Sanitize sentinels to None
        if color_int is None or not (0 <= color_int <= 16777215):
            color_int = None
        if not manual_hex or manual_hex.strip() == "":
            manual_hex = None
        if not comma_separated or comma_separated.strip() == "":
            comma_separated = None

        if from_mode == "auto":
            valid_hex = cls._validate_hex(manual_hex)
            valid_int = cls._validate_int(color_int)
            valid_csv = cls._validate_csv(comma_separated)
            valid_inputs = [x for x in [valid_hex, valid_int, valid_csv] if x is not None]

            if len(valid_inputs) == 1:
                r, g, b = valid_inputs[0]
            else:
                if len(valid_inputs) > 1:
                    print("[ColorConvertNode] Warning: multiple optional inputs are valid in auto mode, falling back to color picker.")
                # Use picker (len==0 or len>1)
                picker = cls._validate_hex(color_hex)
                if picker is None:
                    raise ValueError(f"Color picker value '{color_hex}' must be in format #RRGGBB with valid hex digits")
                r, g, b = picker

            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Manual Hex #FFFFFF":
            result = cls._validate_hex(manual_hex)
            if result is None:
                raise ValueError(f"Manual hex input '{manual_hex}' must be 6 valid hex digits (0-9, a-f, A-F), with optional # prefix")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Int 0-16777215":
            result = cls._validate_int(color_int)
            if result is None:
                raise ValueError(f"Color integer input '{color_int}' must be in range 0–16777215")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        elif from_mode == "Comma separated 255,255,255":
            result = cls._validate_csv(comma_separated)
            if result is None:
                raise ValueError(f"Comma-separated input '{comma_separated}' must be 3 integers 0–255 separated by ',' or ', '")
            r, g, b = result
            color_hex_output, color_int_output, color_string_output = cls._rgb_to_outputs(r, g, b)

        else:
            raise ValueError(f"Unknown from_mode '{from_mode}'")

        return io.NodeOutput(color_hex_output, color_int_output, color_string_output)


class UC_ExtractBoundingBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ExtractBoundingBox",
            display_name="Extract Bounding Box",
            category="utils/primitive",
            inputs=[
                io.AnyType.Input(
                    "input_data",
                    tooltip="Input data containing bounding boxes (JSON string, list, dict, or nested structure)"
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Index of the bounding box to extract"
                ),
            ],
            outputs=[
                io.Int.Output(display_name="x"),
                io.Int.Output(display_name="y"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.BoundingBox.Output("bounding_box"),
            ],
        )

    @classmethod
    def find_boxes(cls, data) -> list:
        boxes = []
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                pass

        if isinstance(data, dict):
            if all(k in data for k in ("x", "y", "width", "height")):
                boxes.append(data)
            else:
                for v in data.values():
                    boxes.extend(cls.find_boxes(v))
        elif isinstance(data, (list, tuple)):
            for item in data:
                boxes.extend(cls.find_boxes(item))

        return boxes

    @classmethod
    def select_box(cls, input_data, index: int) -> tuple[int, int, int, int]:
        boxes = cls.find_boxes(input_data)
        if not boxes:
            raise ValueError("No bounding boxes containing 'x', 'y', 'width', and 'height' were found in the input data.")

        if index < 0 or index >= len(boxes):
            raise ValueError(f"Index {index} is out of range. Found {len(boxes)} bounding box(es).")

        box = boxes[index]
        try:
            x = int(float(box["x"]))
            y = int(float(box["y"]))
            width = int(float(box["width"]))
            height = int(float(box["height"]))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Failed to convert bounding box values at index {index} to integers: {exc}") from exc

        return x, y, width, height

    @classmethod
    def execute(cls, input_data: any, index: int) -> io.NodeOutput:
        x, y, width, height = cls.select_box(input_data, index)
        return io.NodeOutput(
            x,
            y,
            width,
            height,
            {"x": x, "y": y, "width": width, "height": height},
        )


class UC_ExtractMask(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ExtractMask",
            display_name="Extract Mask",
            category="utils/primitive",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip="Mask batch ordered by the producing node, such as compositor layer order.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Index of the mask to extract.",
                ),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, masks, index):
        if not torch.is_tensor(masks) or masks.ndim != 3:
            raise ValueError("Extract Mask requires a [count, height, width] mask batch.")
        if index < 0 or index >= masks.shape[0]:
            raise ValueError(
                f"Index {index} is out of range. Found {masks.shape[0]} mask(s)."
            )
        return io.NodeOutput(masks[index:index + 1])


class UC_ExtractImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_ExtractImage",
            display_name="Extract Image",
            category="utils/image",
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Image list ordered by compositor layer order.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Index of the image to extract.",
                ),
            ],
            outputs=[io.Image.Output("image")],
            is_input_list=True,
        )

    @classmethod
    def execute(cls, images, index):
        collection = images
        if not isinstance(collection, list):
            raise ValueError("Extract Image requires an image list.")
        index = index[0] if isinstance(index, list) and index else index
        if index < 0 or index >= len(collection):
            raise ValueError(
                f"Index {index} is out of range. Found {len(collection)} image(s)."
            )
        image = collection[index]
        if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "Image list contains an invalid image; expected [1, height, width, channels]."
            )
        return io.NodeOutput(image)


class UC_Ideogram4BoundingBoxCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Ideogram4BoundingBoxCrop",
            display_name="Ideogram 4 Bounding Box Crop",
            category="utils/image",
            inputs=[
                io.Image.Input("image"),
                io.BoundingBox.Input("bboxes", force_input=True),
                io.Int.Input("index", default=0, min=0, max=sys.maxsize),
            ],
            outputs=[
                io.Image.Output("image"),
                io.String.Output("ig4_bbox", display_name="IG4 Box"),
                io.BoundingBox.Output("bounding_box", display_name="Box"),
            ],
        )

    @staticmethod
    def _ideogram_bbox(box, width, height):
        left = min(max(int(box["x"]), 0), width)
        top = min(max(int(box["y"]), 0), height)
        right = min(max(int(box["x"] + box["width"]), 0), width)
        bottom = min(max(int(box["y"] + box["height"]), 0), height)
        if right <= left or bottom <= top:
            raise ValueError("The selected bounding box has no area inside the image.")
        normalized = (
            round(top / height * 1000),
            round(left / width * 1000),
            round(bottom / height * 1000),
            round(right / width * 1000),
        )
        return (
            {"x": left, "y": top, "width": right - left, "height": bottom - top},
            f"[{','.join(str(value) for value in normalized)}]",
        )

    @classmethod
    def execute(cls, image, bboxes, index):
        if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Ideogram 4 Bounding Box Crop requires exactly one image.")
        x, y, width, height = UC_ExtractBoundingBox.select_box(bboxes, index)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Bounding box at index {index} must have positive width and height."
            )
        box, ig4_bbox = cls._ideogram_bbox(
            {"x": x, "y": y, "width": width, "height": height},
            image.shape[2],
            image.shape[1],
        )

        cropped = image[
            :,
            box["y"]:box["y"] + box["height"],
            box["x"]:box["x"] + box["width"],
            :,
        ]
        return io.NodeOutput(cropped, ig4_bbox, box)


class UC_AdjustBoundingBox(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_AdjustBoundingBox",
            display_name="Adjust Bounding Box",
            category="utils/primitive",
            inputs=[
                io.AnyType.Input(
                    "input_data",
                    tooltip="Bounding-box data accepted by Extract Bounding Box: a native box, JSON string, list, dict, or nested structure.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Index of the bounding box to adjust.",
                ),
                io.Int.Input(
                    "pixel_expansion",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="Pixels added to each selected edge. Both adds this value on all four sides.",
                ),
                io.Combo.Input(
                    "expansion_axis",
                    options=["both", "horizontal", "vertical"],
                    default="both",
                    tooltip="Choose which axes receive the explicit pixel expansion.",
                ),
                io.Combo.Input(
                    "size_multiple",
                    options=["0", "8", "16", "32"],
                    default="0",
                    tooltip="Round width and height upward to this multiple while keeping the box centered. Zero disables alignment.",
                ),
            ],
            outputs=[
                io.BoundingBox.Output("bounding_box"),
            ],
        )

    @staticmethod
    def _expand_centered(origin: int, size: int, amount: int) -> tuple[int, int]:
        return origin - amount, size + 2 * amount

    @staticmethod
    def _align_centered(origin: int, size: int, multiple: int) -> tuple[int, int]:
        if multiple == 0:
            return origin, size
        aligned_size = ((size + multiple - 1) // multiple) * multiple
        added = aligned_size - size
        return origin - added // 2, aligned_size

    @classmethod
    def execute(cls, input_data, index, pixel_expansion, expansion_axis, size_multiple) -> io.NodeOutput:
        x, y, width, height = UC_ExtractBoundingBox.select_box(input_data, index)
        if width <= 0 or height <= 0:
            raise ValueError(f"Bounding box at index {index} must have positive width and height; got {width}x{height}.")

        if expansion_axis in ("both", "horizontal"):
            x, width = cls._expand_centered(x, width, pixel_expansion)
        if expansion_axis in ("both", "vertical"):
            y, height = cls._expand_centered(y, height, pixel_expansion)

        multiple = int(size_multiple)
        x, width = cls._align_centered(x, width, multiple)
        y, height = cls._align_centered(y, height, multiple)

        bounding_box = {"x": x, "y": y, "width": width, "height": height}
        return io.NodeOutput(bounding_box)


class UC_Krea2LayerProbe(io.ComfyNode):
    """
    Krea 2 Text Encoder Activation Probing Node.

    This node unpacks the flattened 12-layer text conditioning tensor of shape (B, seq, 30720)
    back into its original 12 tapped components of shape (B, 12, seq, 2560) representing the
    last 12 layers of Qwen3-VL-4B.

    It calculates layer-wise activation statistics (Mean, Standard Deviation, Max, Min, L2 Norm)
    and saves them in a JSONL file to compare safe and refused prompt dynamics. It can also save
    sequence-averaged raw activation tensors to build offline datasets for pinpoint weight-level
    ablation.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Krea2LayerProbe",
            display_name="Krea 2 Layer Probe",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "prompt_label",
                    default="safe_prompt",
                    tooltip="A unique name/tag to log this prompt in the stats. Tag safe prompts with 'safe_...' and refused prompts with 'refused_...' to perform differential analysis."
                ),
                io.String.Input(
                    "log_dir",
                    default="krea2_stats",
                    tooltip="The directory where JSONL statistical logs and optional raw tensor files (.pt) will be written."
                ),
                io.Boolean.Input(
                    "save_activations",
                    default=False,
                    tooltip="If True, saves raw 12-layer sequence-averaged activation tensors as .pt files in the log directory for difference-vector computations."
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="statistics_summary"),
            ],
        )

    @classmethod
    def execute(cls, conditioning, prompt_label: str = "safe_prompt", log_dir: str = "krea2_stats", save_activations: bool = False) -> io.NodeOutput:
        # Conditioning is a list of tuples: [(tensor, {"pooled_output": pooled})]
        cond_tensor = conditioning[0][0]  # Shape: (B, seq, 30720)

        B, seq, total_dim = cond_tensor.shape
        num_layers = 12
        hidden_size = 2560

        if total_dim != num_layers * hidden_size:
            raise ValueError(f"Expected conditioning dimension {num_layers * hidden_size}, got {total_dim}")

        # Unpack layers: (B, seq, 12, 2560) -> (B, 12, seq, 2560)
        unpacked = cond_tensor.view(B, seq, num_layers, hidden_size).permute(0, 2, 1, 3)

        # Calculate Layer-wise Stats
        stats = {}
        os.makedirs(log_dir, exist_ok=True)

        for i in range(num_layers):
            layer_act = unpacked[:, i, :, :]  # (B, seq, 2560)

            mean_val = torch.mean(layer_act).item()
            std_val = torch.std(layer_act).item()
            max_val = torch.max(layer_act).item()
            min_val = torch.min(layer_act).item()
            l2_norm = torch.norm(layer_act, p=2, dim=-1).mean().item()

            stats[f"layer_{i}"] = {
                "mean": mean_val,
                "std": std_val,
                "max": max_val,
                "min": min_val,
                "l2_norm": l2_norm
            }

        # Write summary to log
        log_file = os.path.join(log_dir, "probe_log.jsonl")
        log_entry = {
            "prompt_label": prompt_label,
            "seq_len": seq,
            "stats": stats
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Optionally save raw tensor activations to calculate average directions later
        if save_activations:
            avg_activation = torch.mean(unpacked, dim=2)  # Shape: (B, 12, 2560)
            save_path = os.path.join(log_dir, f"act_{prompt_label}.pt")
            torch.save(avg_activation.cpu(), save_path)

        summary_str = f"Probed {seq} tokens across {num_layers} layers.\n"
        summary_str += f"Layer 7 (Max weight projection) L2: {stats['layer_7']['l2_norm']:.4f}, Max: {stats['layer_7']['max']:.4f}"

        return io.NodeOutput(conditioning, summary_str)


class UC_Krea2LayerAblator(io.ComfyNode):
    """
    Krea 2 Text Encoder Activation Pinpoint Ablator.

    This node loads pre-computed difference vectors representing the shift in activation spaces
    during safety refusals, and performs a clean orthogonal projection to subtract the refusal
    direction component.

    Warning: As direct activation manipulation (swapping/clamping/orthogonal subtraction) can
    have subtle side-effects on photographic style, this node is primarily an analytical testbed.
    Using the analytical probing results to surgically ablate weights on the diff LoRA is
    the recommended production approach.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_Krea2LayerAblator",
            display_name="Krea 2 Layer Pinpoint Ablator",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "vectors_path",
                    default="krea2_stats/refusal_directions.pt",
                    tooltip="Path to the .pt file containing pre-computed difference vectors for each of the 12 tapped layers."
                ),
                io.Float.Input(
                    "ablation_strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Ablation scale. 1.0 performs pure orthogonal projection (subtraction of the refusal vector component)."
                ),
                io.String.Input(
                    "layers_mask",
                    default="0,0,0,0,0,0,0,1,1,1,1,0",
                    tooltip="12 comma-separated binary integers (0 or 1) selecting which layers undergo orthogonal projection (e.g., '0,0,0,0,0,0,0,1,1,1,1,0' to target deep layers)."
                ),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, conditioning, vectors_path: str = "krea2_stats/refusal_directions.pt", ablation_strength: float = 1.0, layers_mask: str = "0,0,0,0,0,0,0,1,1,1,1,0") -> io.NodeOutput:
        if not os.path.exists(vectors_path):
            print(f"Warning: Refusal vectors file not found at {vectors_path}. Skipping ablation.")
            return io.NodeOutput(conditioning)

        refusal_vectors = torch.load(vectors_path).to(torch.float32)

        mask = [int(x.strip()) for x in layers_mask.split(",")]
        if len(mask) != 12:
            raise ValueError("Layers mask must contain exactly 12 comma-separated binary digits (0 or 1)")

        modified_cond = []
        for cond_tensor, extra in conditioning:
            B, seq, total_dim = cond_tensor.shape
            num_layers = 12
            hidden_size = 2560

            unpacked = cond_tensor.view(B, seq, num_layers, hidden_size).permute(0, 2, 1, 3).clone()

            device = cond_tensor.device
            dtype = cond_tensor.dtype

            for i in range(num_layers):
                if mask[i] == 1:
                    v_refuse = refusal_vectors[0, i, :].to(device=device, dtype=torch.float32)

                    norm = torch.norm(v_refuse, p=2)
                    if norm > 1e-6:
                        v_hat = v_refuse / norm

                        layer_act = unpacked[:, i, :, :].to(dtype=torch.float32)

                        dot_product = torch.sum(layer_act * v_hat, dim=-1, keepdim=True)

                        projection = dot_product * v_hat
                        ablated = layer_act - ablation_strength * projection

                        unpacked[:, i, :, :] = ablated.to(dtype=dtype)

            repacked = unpacked.permute(0, 2, 1, 3).reshape(B, seq, total_dim)

            new_extra = extra.copy()
            modified_cond.append((repacked, new_extra))

        return io.NodeOutput(modified_cond)


class UC_EncoderNodesGuide(io.ComfyNode):
    """
    Detailed Markdown formatted documentation and guide for advanced, plus, and scaled-bias encoder nodes.
    Choose topics to view in any Markdown rendering node.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_EncoderNodesGuide",
            display_name="Encoder Nodes Guide",
            category="utils/documentation",
            inputs=[
                io.Combo.Input(
                    "topic",
                    options=[
                        "node_catalog",
                        "prompt_templates_and_weighting",
                        "image_inputs_and_placeholders",
                        "resolution_and_reference_latents",
                        "visual_fusion",
                        "consensus_settings",
                        "formulas_and_alignment",
                        "embedding_export",
                        "compatibility_nodes",
                    ],
                    default="node_catalog",
                    tooltip="Select the topic you would like to view documentation for.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="markdown"),
            ],
        )

    @classmethod
    def execute(cls, topic: str) -> io.NodeOutput:
        topics = {
            "node_catalog": (
                "## Encoder node catalog\n\n"
                "### Primary encoder: use this by default\n\n"
                "- `UC_AdvancedVisualConditioningEncode`: the recommended encoder for nearly all conditioning workflows. With no image connected it encodes `prompt` and `system_prompt` as text-only conditioning. With images connected it supports one or more visual sources, inline image placement, formula fallback, spatial visual fusion, VLM resolution control, multiplier scaling, and optional VAE reference latents.\n"
                "- `UC_VisualFusionConfig`: optional spatial-fusion configuration for `UC_AdvancedVisualConditioningEncode`. Leave it disconnected when spatial fusion is not required.\n\n"
                "### Visual consensus pipeline\n\n"
                "- `UC_AdvancedVisConEncoder`: specialized encoder for sequential per-resolution spatial fusion followed by complete-conditioning consensus. It is not a replacement for `UC_AdvancedVisualConditioningEncode`; use it when the additional consensus stage is required.\n"
                "- `UC_AdvancedVisConEncoderTokenFusion`: additive token-first equivalent. It fuses visual and DeepStack tokens before one encode at each lane and resolution, then applies the same complete-conditioning consensus across encoded resolution samples.\n"
                "- `UC_VisualConsensusConfiguration`: required two-input joint configuration combining one complete visual-fusion config with one complete consensus config.\n"
                "- `UC_AdvancedConsensusConfiguration`: extends Text Consensus Blend Configurator with `resolution_samples` and 32-aligned `sample_offset`.\n\n"
                "### Standalone conditioning consensus\n\n"
                "- `UC_ConditioningConsensusBlend`: consensus-blends CONDITIONING outputs already produced by separate encoder nodes.\n"
                "- `UC_TextConsensusBlendConfig`: configures `UC_ConditioningConsensusBlend` through `TEXT_BLEND_CONFIG`.\n\n"
                "### Specialized encoders\n\n"
                "- `UC_MiniMaxH3MediaConfig`: packages ordered timestamped images as one Qwen Video timeline and can carry optional native H3 audio conditioning.\n"
                "- `UC_AdvancedMiniMaxH3ImageToVideo`: role-separated MiniMax H3 conditioning for optional first/last frames, native-reference mode, or Qwen visual fusion. Frame and native-reference modes remain mutually exclusive; the node returns CONDITIONING and the matching H3 latent.\n"
                "- `UC_MiniMaxH3VLMGuide`: experimental CONDITIONING-only guide. Independently encodes a timestamp and image together, then splices them before the existing H3 prompt.\n"
                "- `UC_AdvMiniMaxH3ImageToVideoTemporalFusion`: experimental video fusion after separate Qwen encodes. Fuses corresponding video blocks into lane zero; returns CONDITIONING and LATENT.\n"
                "- `UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion`: experimental video fusion of corresponding visual tokens and DeepStack features before Qwen encoding; returns CONDITIONING and LATENT.\n"
                "- `UC_TextEncodeSystemPrompt`: selectable Flux2 Dev, Klein, Krea2, Z-Image, or Z-Image Thinking text template.\n"
                "- `UC_WeightedTextEncodeSystemPrompt`: the same selectable text templates with classical `(text:weight)` parsing and final multiplier scaling.\n"
                "- `UC_TextEncodeLtxv2SystemPrompt`: LTXV2-specific system-prompt encoding with optional image and reference latent.\n"
                "- `UC_TextEncodeSystemEditAdvanced`: explicit system-edit image placement and image expressions.\n"
                "- `UC_TextEncodeGemmaSystemEditAdvanced`: Gemma-specific system-edit image placement.\n"
                "- `UC_AttentionBiasTextEncode`: experimental `<text=strength>` attention-bias encoding.\n"
                "- `UC_Krea2TokenAttentionWeight`: experimental Krea2 phrase-level attention weighting that returns MODEL and CONDITIONING.\n\n"
                "### Embedding export\n\n"
                "- `UC_VLMInputEmbeds`: canonical raw VLM input-embedding export utility. It does not produce generation conditioning."
            ),
            "prompt_templates_and_weighting": (
                "## Prompt templates and weighting\n\n"
                "### Primary encoder\n\n"
                "`UC_AdvancedVisualConditioningEncode` is still the recommended default when a workflow only needs prompt and system-prompt conditioning. With no image connected it follows its text-only path. Its classical `(text:weight)` handling and final `multiplier` remain available without adding a separate text encoder.\n\n"
                "### Specialized selectable templates\n\n"
                "`UC_TextEncodeSystemPrompt` accepts `model_type`, `prompt`, `system_prompt`, and `thinking_content`. Use it when the workflow specifically requires its selectable non-visual template profiles.\n\n"
                "- `flux2dev`, `klein`, `krea2`, `z-image`, and `z-image-thinking` select different serialized templates.\n"
                "- `thinking_content` is inserted only by the Klein or `z-image-thinking` template branches.\n"
                "- An empty `system_prompt` follows the empty-system branch implemented for the selected type.\n\n"
                "### Weighted System Prompt Text Encode\n\n"
                "`UC_WeightedTextEncodeSystemPrompt` accepts the same model types. `(text:weight)` markers are parsed by the classical scaled-bias path. Its final `multiplier` scales sequence conditioning and pooled output when present.\n\n"
                "### Attention Bias Encode\n\n"
                "`UC_AttentionBiasTextEncode` uses `<text=strength>` markers. It removes the markers before encoding and adds the logarithm of each finite non-negative strength to the corresponding key columns of `attention_mask`. Text without `<`, `>`, or `=` uses the ordinary tokenize-and-encode branch.\n\n"
                "### Krea2 Token Attention Weight\n\n"
                "`attention_weights` uses space-separated `(phrase:weight)` entries. Weights must be finite and non-negative. The node converts each weight to a logarithmic attention bias multiplied by `strength`. Phrases not found in the tokenized prompt are skipped with a warning. The node returns both MODEL and CONDITIONING."
            ),
            "image_inputs_and_placeholders": (
                "## Image inputs and placeholders\n\n"
                "### Advanced Visual Conditioning Encode\n\n"
                "`UC_AdvancedVisualConditioningEncode` flattens active autogrow sockets and image batches into one ordered visual-source sequence. Per-image formula variables are assigned contiguously as `a`, `b`, `c`, and onward.\n\n"
                "- With fusion enabled, use `image_input_fusion`; `image_input_1` is accepted as its alias.\n"
                "- With fusion disabled, numbered `image_input_N` placeholders select active images for one native inline encoding pass.\n"
                "- If fusion is disabled and no numbered inline placeholders are present, each image is encoded separately for the formula path.\n"
                "- If no image is connected, the node encodes text only.\n\n"
                "### Advanced Visual Consensus Encoder\n\n"
                "`UC_AdvancedVisConEncoder` does not use global image flattening as its execution architecture; neither does its additive `UC_AdvancedVisConEncoderTokenFusion` alternative. Visual sources, batch lanes, and resolution variants remain separate dimensions.\n\n"
                "- One batched autogrow socket behaves like connecting its images as separate visual sources.\n"
                "- Multiple non-singleton batched sockets pair equal-index images into independent lanes and must have equal batch lengths.\n"
                "- Singleton image sockets broadcast into every batch lane.\n"
                "- Original node spatially fuses completed per-source conditioning inside every lane and resolution. TokenFusion alternative instead fuses visual and DeepStack tokens first and runs one conditioning encode there. Both then apply complete-conditioning consensus across encoded resolution samples.\n\n"
                "### MiniMax H3\n\n"
                "`UC_AdvancedMiniMaxH3ImageToVideo` keeps the existing H3 modes separate through explicit sockets. Connected `first_frame` or `last_frame` inputs select the keyframe path. The `reference_images` autogrow selects native-reference mode and cannot be combined with frame or fusion inputs. The `fusion_images` autogrow is Qwen-only: with Fusion Config disconnected or off its images remain separate numbered pictures, while an active method combines only that group into one picture after any connected frame pictures.\n\n"
                "`UC_MiniMaxH3MediaConfig` accepts a timestamp list or comma-, semicolon-, or newline-delimited string plus an image batch/list or ordered autogrow sockets. It presents those images after existing Pictures as duplicated-frame temporal blocks inside one `<Video 1>` sequence. The encoder applies `vlm_resolution` independently to each image. Timeline images are semantic Qwen conditioning only: they are not VAE encoded, do not become native H3 references, and cannot be fusion targets. Picture fusion remains available, but active mixed Picture and Video fusion requires `grid-deepstack`. User timestamp precision is preserved in Qwen's Video labels; timestamps must remain chronological and within the output duration.\n\n"
                "Optional config audio requires a MiniMax H3 audio VAE. Qwen receives only `<Audio 1>:` before `<Video 1>:`, while the H3 model receives audio-VAE reference rows. This can provide temporal conditioning but does not guarantee hard synchronization with image timestamps. Actual reference-video input remains future work.\n\n"
                "The visible `vlm_resolution` independently prepares every Qwen copy. Frame VAE pixels continue to follow the generation canvas, and native references independently follow `ref_image_size`: `match` limits reference area to the generation canvas and `max` limits the short edge to 2048 pixels. The prompt is passed directly to the native H3 tokenizer without a system template.\n\n"
                "The generic advanced visual encoders also recognize `qwen3vl_32b` and use its raw `<Picture N>:` presentation, but provide Qwen-only visual conditioning. They do not create `minimax_keyframes` or an H3 latent. Their numbered placeholders and fused visual slot retain normal behavior. Keep `ref_latent_mode` off; H3 output must resolve to one conditioning batch.\n\n"
                "### System Edit Text Encode (Advanced)\n\n"
                "- `image_input_N` inserts the matching connected image at that prompt position.\n"
                "- A pipe expression such as `|a + b|` evaluates an image-tensor expression and inserts its result at that position.\n"
                "- If neither numbered placeholders nor pipe expressions are present, all connected images are prepended in socket order.\n\n"
                "### Gemma System Edit Text Encode (Advanced)\n\n"
                "- `image_input_N` inserts the matching connected image at that prompt position.\n"
                "- If no numbered placeholder is present, all connected images are prepended in socket order.\n\n"
                "### Krea2 Token Attention Weight\n\n"
                "This node supports one visual slot in its prompt. With fusion enabled it accepts `image_input_fusion` or `image_input_1`. Numbered multi-image inline placement is not used by its attention-position mapping."
            ),
            "resolution_and_reference_latents": (
                "## VLM resolution\n\n"
                "`UC_AdvancedVisualConditioningEncode`, `UC_Krea2TokenAttentionWeight`, and `UC_AdvancedVisConEncoder` use an integer `vlm_resolution`:\n\n"
                "- Default: `384`\n"
                "- Widget range: `0` through `4096`\n"
                "- Widget step: `32`\n"
                "- Values from `256` through `3584` select an aspect-preserving equivalent-square target aligned to a 32-pixel grid.\n"
                "- Values below `256` or above `3584` preserve the input resolution.\n\n"
                "`UC_TextEncodeSystemEditAdvanced`, `UC_TextEncodeGemmaSystemEditAdvanced`, and `UC_VLMInputEmbeds` retain the preset choices `Fast (384)`, `Balanced (512)`, `Detailed (768)`, `Large (1024)`, `X-Large (1280)`, `XX-Large (1536)`, and `Original`.\n\n"
                "VAE reference resolution choices are `Ultra (512)`, `Turbo (768)`, `Fast (1024)`, `Balanced (1280)`, `Detailed (1536)`, and `Original`.\n\n"
                "`UC_AdvancedVisualConditioningEncode` and `UC_Krea2TokenAttentionWeight` encode only the selected base VLM resolution. `resolution_samples` belongs only to the `UC_AdvancedVisConEncoder` consensus stage and is not part of `UC_VisualFusionConfig`.\n\n"
                "- `UC_AdvancedConsensusConfiguration` adds odd `resolution_samples` values from `1` through `15` with step `2`, plus `sample_offset` from `32` through `512` with step `32`, to the complete Text Consensus Blend Configurator contract.\n"
                "- Samples alternate around the base resolution by `sample_offset`; candidates producing duplicate effective Qwen grids are skipped.\n"
                "- The configured value is exact: `1` remains one resolution sample regardless of visual-source or batch-lane count.\n"
                "- Every sampled resolution remains a complete conditioning until cross-resolution consensus. TokenFusion changes only per-source fusion inside each resolution: visual and DeepStack tokens fuse before that resolution's single conditioning encode.\n"
                "- Original resolution supports one sample but cannot construct multiple adjacent resolution variants.\n\n"
                "## Reference latents\n\n"
                "Nodes exposing reference latents use `vae_resolution`, `ref_latent_mode`, optional `vae`, and a dimension multiple.\n\n"
                "- `off`: adds no reference latent.\n"
                "- `single` and `multi`: append encoded reference latents to conditioning metadata.\n"
                "- `parallel-single` and `parallel-multi`: use a separate conditioning entry for supported non-Krea2 stages.\n"
                "- Krea2 stages use the standard appended metadata path even when a parallel mode is selected.\n"
                "- A reference latent is created only when the required VAE and image inputs are present."
            ),
            "visual_fusion": (
                "## Spatial visual fusion\n\n"
                "### Primary encoder configuration\n\n"
                "`UC_VisualFusionConfig` is the optional spatial-only configuration consumed by `UC_AdvancedVisualConditioningEncode` and `UC_Krea2TokenAttentionWeight`.\n\n"
                "### Fusion methods\n\n"
                "- `off`: disables visual-token fusion and enables the formula fallback path.\n"
                "- `linear`: averages aligned source tokens.\n"
                "- `spatial-checkerboard`: assigns sources by alternating grid coordinates.\n"
                "- `spatial-block-interleave`: assigns sources in blocks controlled by `visual_block_size` from `1` through `8`.\n"
                "- `spatial-dither-random`: uses `seed` and `dither_ratio`; `dither_secondary_pattern` selects `checkerboard`, `block-interleave`, `dither-random-reverse`, or `dither-random-forward`. Reverse starts with the final pair and recursively works toward source 1. Forward starts with sources 1 and 2 and recursively accumulates each later source.\n\n"
                "### Additional controls\n\n"
                "- `dither_mask_cleanup` applies the implemented deterministic paired-cell cleanup pass.\n"
                "- `spatial_perturbation` ranges from `0.0` through `1.0` and exchanges seeded source assignments while retaining source counts.\n"
                "- `visual_encoder_path` selects `grid-deepstack` or `legacy-flat`.\n"
                "- `save_blended_embeds` writes the final isolated visual-token embedding under the relative `save_path` in ComfyUI's embeddings directory.\n\n"
                "### Visual consensus configuration family\n\n"
                "`UC_AdvancedVisConEncoder` instead receives `UC_VisualConsensusConfiguration`, which combines a complete `UC_VisualFusionConfig` and `UC_AdvancedConsensusConfiguration` without duplicating either configuration's controls. Fusion method `off` disables the spatial stage.\n\n"
                "`UC_AdvancedVisConEncoderTokenFusion` receives the same configuration and preserves the same lanes, resolution sampling, and consensus stage. Its spatial stage fuses visual and DeepStack tokens before one conditioning encode per lane and resolution; the original node fuses completed per-source conditionings.\n\n"
                "For both nodes, spatial fusion completes independently at every applicable lane and resolution. Complete resolution conditionings pass through consensus afterward. Spatial fusion and consensus are sequential stages and are never crossfaded or treated as alternatives."
            ),
            "consensus_settings": (
                "## Complete-conditioning consensus\n\n"
                "### Standalone Conditioning Consensus Blend\n\n"
                "`UC_TextConsensusBlendConfig` is consumed directly by `UC_ConditioningConsensusBlend`, which blends CONDITIONING outputs already produced by separate encoder nodes.\n\n"
                "`blend_preset` accepts `off`, `custom`, `baseline`, `power_blend`, `high_clarity`, `smooth`, `varied_merge`, `diverse_concept`, `high_diversity_concept`, and the six `dsc_` presets shown by the widget.\n\n"
                "- `off` bypasses consensus.\n"
                "- A named preset supplies its stored blend method, consensus type, alignment method, similarity threshold, alpha, beta, norm-rescale, DSC, and soft-comfort values.\n"
                "- `power_blend` also supplies its stored alignment threshold. Other named presets retain the `alignment_threshold` widget value.\n"
                "- `custom` uses the manual values directly.\n"
                "- A non-default `global_scale` overrides a named preset's stored scale.\n"
                "- `blend_method`: `linear` or `consensus`.\n"
                "- `consensus_type`: `mean` or `median`.\n"
                "- `alignment_method`: `index` or `similarity` for text conditioning.\n"
                "- `alignment_threshold` controls similarity-based text matching.\n"
                "- `similarity_threshold`, `power_alpha`, and `diversity_beta` control consensus weights.\n"
                "- `rescale_norm`, `global_scale`, `dynamic_similarity_contrast`, and `soft_comfort_bandpass` modify the blended values.\n"
                "- `position_weight` biases text similarity matching toward nearby normalized token positions.\n"
                "- `preserve_common_prefix` copies the longest numerically identical text-conditioning prefix from the first input.\n\n"
                "### Advanced Visual Consensus Encoder\n\n"
                "`UC_AdvancedVisConEncoder` and `UC_AdvancedVisConEncoderTokenFusion` apply equivalent consensus mathematics to complete per-lane and per-resolution conditionings produced by their preceding spatial stage. TokenFusion affects only whether per-source fusion happens before or after that resolution's conditioning encode.\n\n"
                "- `UC_AdvancedConsensusConfiguration` inherits every Text Consensus Blend Configurator input and adds `resolution_samples` plus `sample_offset`.\n"
                "- Its `blend_preset` is authoritative; `off` disables cross-resolution consensus.\n"
                "- A named preset uses its stored mathematics while position weight, common-prefix preservation, global scale, and resolution samples remain available.\n"
                "- Consensus receives complete conditionings; it does not replace, disable, or crossfade against spatial fusion."
            ),
            "formulas_and_alignment": (
                "## Formula fallback\n\n"
                "The `formula` and `padding_method` inputs on `UC_AdvancedVisualConditioningEncode` and `UC_Krea2TokenAttentionWeight` are used only when `UC_VisualFusionConfig` is disconnected or set to `off`. They are not part of `UC_AdvancedVisConEncoder`.\n\n"
                "- Active image passes map to `a`, `b`, `c`, and onward.\n"
                "- An empty formula selects `a`.\n"
                "- Operators: `+`, `-`, `*`, `/`, unary `+`, and unary `-`.\n"
                "- Functions: `abs`, `min`, `max`, and `clamp`.\n"
                "- `(a:1.2)` is normalized to scalar multiplication before restricted expression evaluation.\n"
                "- Attribute access, indexing, comprehensions, imports, and unapproved calls are rejected.\n"
                "- `zero-pad` pads shorter sequence tensors with zero tokens.\n"
                "- `interpolate` resizes shorter sequence tensors along their token dimension.\n\n"
                "## Conditioning Consensus\n\n"
                "`UC_ConditioningConsensusBlend` accepts autogrowing CONDITIONING inputs. One active input is returned unchanged. Multiple inputs must contain the same number of scheduled entries. A disconnected config uses the `baseline` preset; an `off` config returns the first active input."
            ),
            "embedding_export": (
                "## VLM Input Embedding Export\n\n"
                "`UC_VLMInputEmbeds` accepts `clip`, `prompt`, line-separated `image_paths`, preset `vlm_resolution`, line-separated `file_names`, and `slice_visual_tokens`.\n\n"
                "- Every non-empty image path must map to one file name.\n"
                "- Tokenization uses `skip_template=True`.\n"
                "- With `slice_visual_tokens=False`, the full processed token embedding sequence is retained.\n"
                "- With `slice_visual_tokens=True`, the first validated visual-token span is removed.\n"
                "- Each state dictionary is saved as `.safetensors` below ComfyUI's embeddings directory.\n"
                "- The node outputs the final state dictionary and final two-dimensional embedding tensor produced by the input list."
            ),
            "compatibility_nodes": (
                "## Deprecated compatibility nodes\n\n"
                "### Registered interface-preserving replacements\n\n"
                "These migrations are explicitly registered in `node_replacements.py`:\n\n"
                "- `TextEncodeSystemEditPlusAdvanced` → `UC_TextEncodeSystemEditAdvanced`\n"
                "- `TextEncodeGemmaSystemEditPlusAdvanced` → `UC_TextEncodeGemmaSystemEditAdvanced`\n"
                "- `TextEncodeKrea2SystemEditScaledAdv` → `UC_AdvancedVisualConditioningEncode`\n"
                "- `UC_Krea2InputEmbeds` → `UC_VLMInputEmbeds`\n"
                "- `UC_Qwen3VLInputEmbeds` → `UC_VLMInputEmbeds`\n"
                "- `TextEncodeKrea2SysEditScaledAdvAttn` → `UC_Krea2TokenAttentionWeight`\n\n"
                "### Other retained compatibility schemas\n\n"
                "The model-specific system-prompt nodes, model-specific scaled-bias nodes, `UC_ScaledBiasTextEncodeSystemPrompt`, `TextEncodeSystemEditPlus`, `TextEncodeKrea2SystemEditPlusAdvanced`, and `TextEncodeEditPlusAdvanced` remain deprecated and loadable for existing workflows. They are not automatically rewritten because no interface-preserving migration is registered. Do not infer a widget mapping from a similar display name.\n\n"
                "`UC_TextEncodeLtxv2SystemPrompt` remains a current specialized node and is not part of the deprecated compatibility group."
            ),
        }
        markdown = topics.get(topic, "Unknown topic selected.")

        return io.NodeOutput(markdown)


class UC_CompositeNodesGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_CompositeNodesGuide",
            display_name="Composite Nodes Guide",
            category="utils/documentation",
            inputs=[
                io.Combo.Input(
                    "topic",
                    options=[
                        "node_catalog",
                        "model_inputs",
                        "mask_cleanup_and_resize",
                        "background_removal_alpha",
                        "unified_background_replace",
                        "layered_composite",
                        "staged_workflow",
                        "staged_individual_workflow",
                        "staged_face_workflow",
                        "placement_editor",
                        "paint_layer",
                        "mediapipe_face_composite",
                    ],
                    default="node_catalog",
                    tooltip="Select the composite documentation topic.",
                ),
            ],
            outputs=[io.String.Output(display_name="markdown")],
        )

    @classmethod
    def execute(cls, topic: str) -> io.NodeOutput:
        topics = {
            "node_catalog": (
                "## Background and face composite nodes\n\n"
                "- `UC_BackgroundRemovalPreserveAlpha`: returns source-resolution RGBA images and their exact alpha masks in one node.\n"
                "- `UC_UnifiedBackgroundReplace`: produces one independently composited result per flattened foreground image over one background.\n"
                "- `UC_LayeredBackgroundComposite`: removes and places multiple foreground layers in one scene during every execution.\n"
                "- `UC_StagedLayeredBackgroundCompositeOptions`: supplies cleanup, resize, feather, and foreground-blend settings to staged compositors.\n"
                "- `UC_StagedLayeredBackgroundComposite`: retains cutouts between staging and final compositing executions.\n"
                "- `UC_StagedIndividualComposites`: provides its own staging editor and returns one background-plus-one-foreground result per included layer.\n"
                "- `UC_StagedMediaPipeFaceOptions`: supplies detection, extraction, face feather, initial scale, and face-blend settings.\n"
                "- `UC_StagedMediaPipeFaceBackgroundComposite`: stages ordinary foreground layers and independently placeable detected face layers.\n"
                "- `UC_MediaPipeFaceCompositeOptions`: supplies options for direct source-to-target face compositing.\n"
                "- `UC_MediaPipeFaceComposite`: composites the largest detected source face into the largest detected target face."
            ),
            "model_inputs": (
                "## Model inputs and defaults\n\n"
                "Optional model sockets keep their backend IDs and display `_opt` in the node UI.\n\n"
                "- A disconnected `background_removal_model_opt` loads the internal `birefnet.safetensors` checkpoint.\n"
                "- A connected Core BACKGROUND_REMOVAL model overrides the internal default. This permits a connected Lucida model.\n"
                "- A disconnected `face_detection_model_opt` on `UC_MediaPipeFaceComposite` loads `mediapipe_face_fp32.safetensors` from the detection model directory.\n"
                "- A connected FACE_DETECTION_MODEL overrides that internal face model.\n"
                "- The staged face compositor loads its MediaPipe face model internally.\n"
                "- The staged model selectors accept `birefnet` and `lucida`. Missing or invalid internal checkpoint files produce an error containing the expected model location and source URL."
            ),
            "mask_cleanup_and_resize": (
                "## Background-removal mask controls\n\n"
                "- `mask_threshold`: minimum mask value retained before cleanup. Staged default is `0.5`.\n"
                "- `border_cleanup_width`: source-edge strip width where weak predictions are removed. Default `2`; `0` disables it.\n"
                "- `artifact_cleanup_radius`: morphological opening radius. Default `2`; `0` disables it.\n"
                "- `gap_fill_radius`: morphological closing radius. Default `2`; `0` disables it.\n"
                "- `mask_processing_resolution`: longest edge used for removal-mask refinement, face detection, and editor cutout previews. `0` uses the removal model's declared native size (1024 for the installed BiRefNet and Lucida configurations). Original-resolution RGB is cropped only after mask bounds are mapped back.\n"
                "- `feather_radius`: inward mask-edge softness. Staged default `2`; `0` keeps the resized edge unchanged.\n\n"
                "Resize methods are `auto`, `nearest-exact`, `bilinear`, `area`, `bicubic`, and `lanczos`. For images, `auto` uses FP32 area reduction when shrinking and bicubic when enlarging. For masks, `auto` uses area when shrinking and bilinear when enlarging. `nearest-exact` produces a hard resized edge."
            ),
            "background_removal_alpha": (
                "## Background Removal (Preserve Alpha)\n\n"
                "Accepts an IMAGE batch and returns source-resolution RGBA images plus the exact alpha MASK batch. RGB inputs use the connected Core background-removal model or selected internal BiRefNet/Lucida model. The soft model mask is preserved and resized when necessary. RGBA inputs bypass model execution and preserve their supplied alpha. Original RGB values remain stored beneath transparent pixels."
            ),
            "unified_background_replace": (
                "## Unified Background Replace\n\n"
                "This node requires one background image and at least one foreground image. Batched and autogrowing foreground inputs are flattened into separate outputs.\n\n"
                "- `foreground_scale` sets the foreground's longest retained mask bound relative to the background's shortest side.\n"
                "- `long_axis_shift` and `short_axis_shift` position the foreground along the background axes from `-1` through `1`.\n"
                "- `workspace_padding` ranges from `0` through `1` and controls the permitted off-canvas margin.\n"
                "- Cleanup, feather, image-resize, and mask-resize controls are applied independently to every foreground.\n"
                "- Outputs are an IMAGE batch and the corresponding MASK batch."
            ),
            "layered_composite": (
                "## Layered Background Composite\n\n"
                "This node removes backgrounds and builds one scene in a single execution.\n\n"
                "- Foregrounds are composited according to the layer order stored in `placement_data`.\n"
                "- Each layer stores scale, normalized center, horizontal and vertical flips, rotation, local quadrilateral corners, and included or excluded state in the current placement format.\n"
                "- Excluded layers do not contribute to the output.\n"
                "- The node outputs the composited IMAGE and combined MASK.\n"
                "- Unlike the staged nodes, foreground removal runs again whenever this node executes."
            ),
            "staged_workflow": (
                "## Staged Background Composite workflow\n\n"
                "Foreground pixels, the removal-model identity, and mask-generation settings form an automatic stage fingerprint. Matching fingerprints reuse retained cutout and mask tensors; changed or missing stages rebuild automatically. Background and placement-only changes recompose the retained objects without rerunning background removal.\n\n"
                "The editor manages `execution_mode` internally. Ordinary queues validate and reuse the stage. Run Staging queues this output node and its upstream closure as an explicit forced preview refresh. A retained stage belongs to the node instance.\n\n"
                "`Staged Composite Options` is optional. When disconnected, its defaults are used. `foreground_blend=1.0` keeps the layer fully foreground. `0.0` applies the implemented 50/50 normal blend only where an accumulated foreground or face mask already exists; background-only areas remain fully foreground."
            ),
            "staged_individual_workflow": (
                "## Staged Individual Composites\n\n"
                "A complete staged editor with its own automatically validated cutouts and placement state. Each included foreground produces one full-background IMAGE, matching placement MASK, and bounding box; foregrounds are never stacked. Placement changes reuse retained image and mask tensors. `foreground_blend` is unused because every result contains one foreground."
            ),
            "staged_face_workflow": (
                "## Staged Face Background Composite\n\n"
                "Foreground pixels and face-aware staging settings are fingerprinted automatically. A changed or missing stage runs background removal and MediaPipe detection; placement-only changes reuse retained ordinary and face crops without loading either model. Detected face layers are inserted immediately in front of their originating foreground.\n\n"
                "`Staged MediaPipe Face Options` defaults:\n\n"
                "- `detection_threshold=0.55`\n"
                "- `maximum_faces=16`\n"
                "- `bbox_expansion=64`\n"
                "- `mask_expansion=0`\n"
                "- `face_feather_radius=0`\n"
                "- `initial_face_scale=0.25`\n"
                "- `face_blend=1.0`\n\n"
                "Detections are ordered deterministically by vertical and then horizontal center. Overlapping detections are merged by the implemented overlap grouping. A per-image detection failure or empty result leaves the ordinary foreground available. The node outputs IMAGE, MASK, final transformed bounding boxes, and final transformed layer masks."
            ),
            "placement_editor": (
                "## Placement editor\n\n"
                "- Drag a visible included layer to move it.\n"
                "- Corner handles resize by default. Rotate mode changes corner dragging to rotation while ordinary movement remains available.\n"
                "- Warp mode exposes four independently draggable local quadrilateral corners.\n"
                "- Flip H and Flip V are stored per layer.\n"
                "- Excl removes a layer from preview interaction and final compositing while retaining its row and placement data.\n"
                "- Reset restores that layer's placement defaults.\n"
                "- The grip and stacked triangle buttons reorder layers from back to front.\n"
                "- Right-clicking the preview opens the layer context menu.\n"
                "- Workspace Padding applies to each layer's expanded transformed raster, so the preview and final composite use the same off-canvas limits after rotation or warp.\n"
                "- Staged crop dimensions provide exact transform geometry. Before a fresh staging pass, an available source preview is used provisionally and is replaced automatically when crop metadata arrives.\n"
                "- The layer list displays three complete rows before scrolling."
            ),
            "paint_layer": (
                "## Native RGBA paint layer\n\n"
                "Both staged compositors provide one background-sized Paint layer. It starts at the front of the back-to-front layer list and can be reordered or excluded like other layers. Its position controls normal occlusion; painting still targets the Paint layer when a foreground is above it.\n\n"
                "- Enable `Paint` to suspend foreground selection, movement, resizing, rotation, and warping. Disable it to return to placement editing.\n"
                "- Choose a circle or square brush, color, native-pixel radius, opacity, and hardness. Brush opacity controls the alpha written into the RGBA raster; the color selector intentionally has no second alpha control.\n"
                "- The custom HSL selector accepts complete six-digit hex or RGB values, keeps recent colors, and can sample the clean rendered composition with `Pick from composition`. It opens beside the node instead of covering the composition view.\n"
                "- Eraser removes only Paint-layer alpha. Undo and redo retain twenty completed paint operations; Clear is also undoable.\n"
                "- Completed operations autosave losslessly under `input/clipspace`. Leaving Paint mode or queueing waits for the current upload, and upload failure blocks stale execution while preserving the local canvas.\n"
                "- Copying a node shares its current raster until the copy is first edited, when a new asset ID is assigned. Exported workflows must include the referenced input PNG.\n"
                "- A background-size change resizes the complete RGBA raster with premultiplied-alpha bilinear scaling before saving.\n"
                "- Final outputs include Paint alpha in the combined mask and at the Paint layer's exact position in ordered layer masks and bounding boxes. Fully transparent paint is a no-op."
            ),
            "mediapipe_face_composite": (
                "## MediaPipe Face Composite\n\n"
                "The node accepts one source image and one target image. It requests full-range detection, selects the largest detected face from each image, and uses the detector's `face_oval` landmark ring.\n\n"
                "`MediaPipe Face Composite Options`:\n\n"
                "- `bbox_expansion`: expands source and target face boxes. Default `64`.\n"
                "- `mask_expansion`: expands or contracts the source face mask. Default `0`.\n"
                "- `feather_radius`: signed face-mask feather control. Default `8`.\n"
                "- `target_warp_strength`: target deformation strength from `0.0` through `2.0`. Default `1.0`.\n"
                "- `warp_decay_radius`: target-warp falloff radius. Default `64`.\n"
                "- `score_thresh`: detector score threshold. Default `0.25`.\n\n"
                "The source face mask is intersected with the source background-removal mask. Outputs are the composited target IMAGE and the extracted Face Crop."
            ),
        }
        return io.NodeOutput(topics.get(topic, "Unknown topic selected."))


class UC_HighResolutionTilingGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_HighResolutionTilingGuide",
            display_name="High Resolution Tiling Guide",
            category="utils/documentation",
            inputs=[
                io.Combo.Input(
                    "topic",
                    options=[
                        "workflow",
                        "split_settings",
                        "overlap_masks",
                        "depth_structure_mask",
                        "visual_conditioning",
                        "sampling",
                        "accumulation",
                    ],
                    default="workflow",
                    tooltip="Select the high-resolution tiled workflow topic.",
                ),
            ],
            outputs=[io.String.Output(display_name="markdown")],
        )

    @classmethod
    def execute(cls, topic: str) -> io.NodeOutput:
        topics = {
            "workflow": (
                "## High-resolution tiled workflow\n\n"
                "1. Upscale or otherwise prepare one complete source image.\n"
                "2. Connect it and the generation VAE to `UC_HighResolutionTileSplit`.\n"
                "3. Send `tile images` through any list-mapped image processing and visual-conditioning branches.\n"
                "4. Send `tile latents` to the latent input of a Core sampler. ComfyUI pairs image conditioning and latents by list index.\n"
                "5. Decode the sampled latent list with the same VAE.\n"
                "6. Connect the decoded image list and the original `tile layout` to `UC_HighResolutionTileAccumulator`.\n\n"
                "The splitter accepts one source image. It VAE-encodes tiles sequentially and returns true ComfyUI lists so downstream nodes execute once per tile. Every `tile images` item is the exact padded tensor sent to VAE encoding, keeping visual-conditioning grids aligned with their matching latents."
            ),
            "split_settings": (
                "## Split settings\n\n"
                "- `tile_mode=tile_size` uses `tile_width` and `tile_height`; `rows` and `columns` are ignored.\n"
                "- `tile_mode=grid` uses `rows` and `columns`; `tile_width` and `tile_height` are ignored.\n"
                "- `overlap` is the number of source-image pixels shared by adjacent tiles on both axes.\n"
                "- Tile-size mode advances by tile dimension minus overlap.\n"
                "- Grid mode divides the complete image into the requested cells and adds overlap around internal cell boundaries.\n"
                "- In both modes, every real crop is replicate-padded on its bottom and right to the connected VAE's reported compression multiple. That complete padded tensor is both VAE-encoded and emitted through `tile images`; the layout retains the real dimensions for final cropping.\n\n"
                "More overlap provides a wider transition area but processes more pixels. Overlap must remain smaller than the applicable tile or grid-cell dimension."
            ),
            "overlap_masks": (
                "## Overlap masks\n\n"
                "Every latent contains a Core-compatible `noise_mask`. The non-overlapping interior is solid `1.0`. Only edges shared with neighboring tiles are feathered; outer image boundaries remain solid.\n\n"
                "- `mask_profile=cosine` uses an eased transition. `linear` uses a constant-rate transition.\n"
                "- `feather_width` selects the fraction of the overlap occupied by the transition. `1.0` uses the full overlap.\n"
                "- `mask_strength` controls the minimum at a protected shared edge. `1.0` reaches `0`; `0.5` reaches `0.5`; `0.0` leaves the mask solid.\n\n"
                "These masks are also the normalized reconstruction weights used by the accumulator."
            ),
            "depth_structure_mask": (
                "## Depth-guided structure mask\n\n"
                "Connect one grayscale depth-map IMAGE to modulate every real pixel of each tile's latent `noise_mask`, including feathered overlaps. The complete depth map is converted to luminance, clamped to `0–1`, and bilinearly resized to the complete source-image dimensions before tile crops are taken.\n\n"
                "`depth_influence` ranges from `-1` to `1`. Positive values use the supplied depth map, negative values use `1 - depth`, the magnitude controls the effect, and `0` disables depth modulation. The node calculates `depth_factor = 1 - abs(depth_influence) × selected_depth`, then multiplies the existing overlap mask by that factor. This preserves neighboring feather proportions without a boundary between depth and overlap behavior. Zero-valued VAE padding remains unchanged.\n\n"
                "Depth modulation affects denoising only. The accumulator continues using the original overlap feather for reconstruction. Core and Advanced Differential Diffusion consume the depth-modulated latent masks automatically."
            ),
            "visual_conditioning": (
                "## Visual conditioning and image lists\n\n"
                "Connecting `tile images` to `UC_AdvancedVisualConditioningEncode` executes that encoder once per tile. Each image includes the same replicate padding used for its matching VAE latent, so Original-resolution grid/DeepStack encoding sees the complete sampled extent.\n\n"
                "Two image lists connected to two image sockets are paired by index. For example, an original tile list and its radius-2 blurred list become `(original 0, blurred 0)`, `(original 1, blurred 1)`, and so on. With a visual fusion configuration, each pair follows that fusion method.\n\n"
                "Lists are not combined across tile indices. If list lengths differ, standard ComfyUI mapping repeats the final item of the shorter list. A shared text prompt is broadcast to every tile, so a full-image caption can introduce globally described subjects into every tile."
            ),
            "sampling": (
                "## Sampling the tile list\n\n"
                "Connect the conditioning list and `tile latents` to the ordinary Core sampling path. Matching indices are sampled together.\n\n"
                "The sigma schedule controls overall change, including the solid non-overlap interiors. Lowering its start sigma generally preserves more of each encoded tile.\n\n"
                "The splitter can optionally apply Differential Diffusion to a connected model. Connect its `model` output to the sampler guider. `off` returns the connected model unchanged, `core` uses ComfyUI Core Differential Diffusion, and `advanced` uses the KJNodes-compatible threshold multiplier. The model input is required only when either active mode is selected.\n\n"
                "`differential_diffusion_value` is mode-dependent:\n\n"
                "- In `core` mode it is strength from `0` through `1`, blending Core's progressive binary mask with the original soft mask.\n"
                "- In `advanced` mode it is the KJNodes-compatible threshold divisor from `-10` through `10`; it cannot be zero.\n"
                "- Differential Diffusion does not replace sigma selection; start sigma still controls overall change in solid non-overlap interiors.\n\n"
                "The splitter still does not implement a sampler or noise generator. A single noise-provider connection is broadcast to every mapped tile execution."
            ),
            "accumulation": (
                "## Decoding and accumulation\n\n"
                "Decode the sampled latent list with the same VAE used by the splitter. Connect that decoded image list to `images` and connect the splitter's matching `tile layout` directly to the accumulator.\n\n"
                "`UC_HighResolutionTileAccumulator` consumes the complete image list in one execution. It removes VAE padding, places every tile at its recorded source coordinates, applies the stored feather weights, and normalizes overlapping contributions.\n\n"
                "Do not reorder, omit, or insert list items before accumulation. The image list must contain exactly the tiles described by that layout."
            ),
        }
        return io.NodeOutput(topics.get(topic, "Unknown topic selected."))


class UC_MarkdownPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UC_MarkdownPreview",
            display_name="Preview as Markdown",
            category="utils/documentation",
            description="Displays a connected value using the Core Markdown renderer.",
            inputs=[
                io.AnyType.Input(
                    "source",
                    tooltip="Text or another serializable value to render as Markdown.",
                ),
            ],
            outputs=[io.String.Output("text", display_name="text")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source=None) -> io.NodeOutput:
        if isinstance(source, str):
            value = source
        elif isinstance(source, (int, float, bool)) or source is None:
            value = str(source)
        else:
            try:
                value = json.dumps(source, indent=2)
            except (TypeError, ValueError):
                value = str(source)
        return io.NodeOutput(value, ui={"markdown": (value,)})


CompositeNodesGuide = UC_CompositeNodesGuide

