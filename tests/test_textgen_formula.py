import ast
import inspect
import pathlib
import subprocess
import sys
import types

import pytest
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_textgen_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_textgen_test.helpers import textgen_helpers
    from utils_collection_textgen_test.nodes import textgen_nodes
finally:
    cli_args.cpu = prior_cpu


def _images():
    return {
        "a": torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32),
        "b": torch.tensor([[[[0.6, 0.4, 0.1]]]], dtype=torch.float32),
        "image_input_1": torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32),
    }


def test_textgen_formula_preserves_arithmetic_aliases_and_clamping():
    images = _images()
    arithmetic = textgen_nodes.evaluate_formula("-a + (b * 2) ** 1 / 2", images)
    named = textgen_nodes.evaluate_formula("(image_input_1 + b) / 2", images)

    assert arithmetic.dtype == images["a"].dtype
    assert arithmetic.shape == images["a"].shape
    assert torch.allclose(arithmetic, torch.clamp(-images["a"] + images["b"], 0.0, 1.0))
    assert torch.allclose(named, (images["a"] + images["b"]) / 2)
    assert textgen_nodes.evaluate_formula("a * 10", images).max().item() == 1.0


def test_textgen_formula_preserves_named_tensor_functions():
    images = _images()
    result = textgen_nodes.evaluate_formula("clamp(max(abs(a - b), min(a, b)), 0.1, 0.6)", images)
    expected = torch.clamp(
        torch.maximum(torch.abs(images["a"] - images["b"]), torch.minimum(images["a"], images["b"])),
        0.1,
        0.6,
    )
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("expression", [
    "unknown + a",
    "a.__class__",
    "a.mean()",
    "a[0]",
    "sum([a])",
    "clamp(a, min=0.0, max=1.0)",
    "a if True else b",
])
def test_textgen_formula_rejects_unsupported_expression_elements(expression):
    with pytest.raises(RuntimeError, match="Error evaluating textgen visual math expression"):
        textgen_nodes.evaluate_formula(expression, _images())


@pytest.mark.parametrize("expression", ["a +", "a / 0"])
def test_textgen_formula_rejects_invalid_or_nonfinite_results(expression):
    with pytest.raises(RuntimeError, match="Error evaluating textgen visual math expression"):
        textgen_nodes.evaluate_formula(expression, _images())


def test_global_and_inline_formula_routes_evaluate_their_image_expressions(monkeypatch):
    class Clip:
        tokenizer = types.SimpleNamespace(clip_name="qwen3vl_4b")

        @staticmethod
        def tokenize(_prompt, **_kwargs):
            return {"qwen": [[(1, 1.0)]]}

        @staticmethod
        def generate(_tokens, **_kwargs):
            return [7]

        @staticmethod
        def decode(_tokens, **_kwargs):
            return "decoded"

    calls = []
    real_evaluate = textgen_nodes.evaluate_formula

    def record(expression, images):
        calls.append((expression, set(images)))
        return real_evaluate(expression, images)

    monkeypatch.setattr(textgen_nodes, "evaluate_formula", record)
    images = {
        "image_0": torch.full((1, 2, 2, 3), 0.2),
        "image_1": torch.full((1, 2, 2, 3), 0.3),
    }

    global_result = textgen_nodes.UC_TextGenerate.execute(
        Clip(),
        "describe",
        "",
        0,
        12,
        {"sampling_mode": "off"},
        formula="a + b",
        image_inputs=images,
    )
    inline_result = textgen_nodes.UC_TextGenerate.execute(
        Clip(),
        "compare |a + b|",
        "",
        0,
        12,
        {"sampling_mode": "off"},
        image_inputs=images,
    )

    assert global_result.args == inline_result.args == ("decoded", 0)
    assert calls == [
        ("a + b", {"image_input_1", "image_input_2", "a", "b"}),
        ("a + b", {"image_input_1", "image_input_2", "a", "b"}),
    ]


def test_text_generate_schema_has_no_obsolete_blend_config():
    input_ids = [value.id for value in textgen_nodes.UC_TextGenerate.define_schema().inputs]
    assert "blend_config" not in input_ids
    assert "model_type" not in input_ids
    assert "image_inputs" in input_ids
    assert "blend_config" not in inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters
    assert "model_type" not in inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters
    assert not hasattr(textgen_nodes, "BlendConfig")
    assert not hasattr(textgen_nodes, "evaluate_image_consensus_blend")


def test_text_generate_schema_uses_shared_numeric_vlm_resolution_contract():
    inputs = {
        value.id: value for value in textgen_nodes.UC_TextGenerate.define_schema().inputs
    }
    resolution = inputs["vlm_resolution"]
    video_resolution = inputs["vlm_video_resolution"]

    assert resolution.io_type == "INT"
    assert resolution.default == 384
    assert resolution.min == 0
    assert resolution.max == 4096
    assert resolution.step == 32
    assert video_resolution.io_type == "INT"
    assert video_resolution.default == 384
    assert video_resolution.min == 0
    assert video_resolution.max == 4096
    assert video_resolution.step == 32


def test_text_generate_appends_optional_visual_fusion_config():
    inputs = textgen_nodes.UC_TextGenerate.define_schema().inputs
    assert "visual_fusion_config" in {value.id for value in inputs}
    assert "visual_fusion_config" in inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters


def test_text_generate_schema_exposes_token_fusion_capability():
    schema = textgen_nodes.UC_TextGenerate.define_schema()
    assert schema.display_name == "Text Generate (TokenFusion)"
    assert "visual_fusion_config" in [value.id for value in schema.inputs]
    assert [output.id for output in textgen_nodes.UC_TextGenerate.define_schema().outputs] == [
        "generated_text",
        "seed",
    ]


def test_text_generate_schema_accepts_video_components_outputs():
    schema_inputs = textgen_nodes.UC_TextGenerate.define_schema().inputs
    inputs = {value.id: value for value in schema_inputs}

    assert inputs["video"].io_type == "IMAGE"
    assert inputs["video"].optional is True
    assert inputs["video_fps"].io_type == "FLOAT"
    assert inputs["video_fps"].default == 24.0
    assert inputs["sample_fps"].io_type == "FLOAT"
    assert inputs["sample_fps"].default == 2.0
    assert inputs["sample_fps"].max == 1000.0
    signature = inspect.signature(textgen_nodes.UC_TextGenerate.execute).parameters
    assert {"video", "video_fps", "sample_fps"} <= set(signature)
    assert schema_inputs[-1].id == "image_inputs"


def test_qwen3vl_video_sampling_uses_source_fps_and_target_sample_fps(monkeypatch):
    video = torch.arange(24, dtype=torch.float32).view(24, 1, 1, 1).expand(24, 2, 2, 3)
    prepared = []

    def prepare(frame, resolution):
        prepared.append((int(frame[0, 0, 0, 0]), resolution))
        return frame

    monkeypatch.setattr(textgen_helpers, "prepare_vlm_image", prepare)
    frames, indices = textgen_helpers.prepare_qwen3vl_video_frames(
        video, 384, video_fps=24.0, sample_fps=2.0
    )

    assert indices == [0, 12]
    assert prepared == [(0, 384), (12, 384)]
    assert frames.shape == (2, 2, 2, 3)


def test_qwen3vl_video_sampling_repeat_pads_odd_sample_count(monkeypatch):
    video = torch.arange(24, dtype=torch.float32).view(24, 1, 1, 1).expand(24, 2, 2, 3)
    monkeypatch.setattr(textgen_helpers, "prepare_vlm_image", lambda frame, _resolution: frame)

    frames, indices = textgen_helpers.prepare_qwen3vl_video_frames(
        video, 0, video_fps=24.0, sample_fps=5.0
    )

    assert indices == [0, 5, 10, 14, 19, 19]
    assert frames.shape[0] == 6
    assert torch.equal(frames[-1], frames[-2])


def test_qwen3vl_video_sampling_repeat_pads_single_frame(monkeypatch):
    video = torch.ones((1, 2, 2, 3))
    monkeypatch.setattr(textgen_helpers, "prepare_vlm_image", lambda frame, _resolution: frame)

    frames, indices = textgen_helpers.prepare_qwen3vl_video_frames(
        video, 0, video_fps=24.0, sample_fps=2.0
    )

    assert indices == [0, 0]
    assert frames.shape[0] == 2
    assert torch.equal(frames[0], frames[1])


def test_qwen3vl_video_sampling_flattens_image_tensor_lists_in_order(monkeypatch):
    first = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1).expand(2, 2, 2, 3)
    second = torch.tensor([2.0]).view(1, 1, 1, 1).expand(1, 2, 2, 3)
    monkeypatch.setattr(textgen_helpers, "prepare_vlm_image", lambda frame, _resolution: frame)

    frames, indices = textgen_helpers.prepare_qwen3vl_video_frames(
        [first, second], 0, video_fps=3.0, sample_fps=3.0
    )

    assert indices == [0, 1, 2, 2]
    assert frames[:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 2.0]


@pytest.mark.parametrize(
    "video_fps,sample_fps",
    [(0.0, 2.0), (24.0, 0.0), (float("nan"), 2.0), (24.0, True)],
)
def test_qwen3vl_video_sampling_rejects_invalid_rates(video_fps, sample_fps):
    with pytest.raises(ValueError, match="finite value greater than zero"):
        textgen_helpers.prepare_qwen3vl_video_frames(
            torch.zeros((4, 2, 2, 3)),
            0,
            video_fps=video_fps,
            sample_fps=sample_fps,
        )


@pytest.mark.parametrize(
    "video",
    [[], torch.zeros((0, 2, 2, 3)), torch.zeros((2, 2, 2, 4))],
)
def test_qwen3vl_video_sampling_rejects_invalid_frame_batches(video):
    with pytest.raises(ValueError, match="video requires"):
        textgen_helpers.prepare_qwen3vl_video_frames(
            video,
            0,
            video_fps=24.0,
            sample_fps=2.0,
        )


def test_qwen3vl_video_prompt_matches_timestamped_video_token_contract():
    prompt = textgen_helpers.qwen3vl_video_prompt([0, 12, 19, 19], video_fps=24.0)

    assert prompt == (
        "Video 1: <0.2 seconds><|vision_start|><|video_pad|><|vision_end|>"
        "<0.8 seconds><|vision_start|><|video_pad|><|vision_end|>"
    )
    assert "\n" not in prompt
    assert "<Video" not in prompt


def test_qwen3vl_video_preprocessing_bounds_vram_to_one_temporal_pair():
    calls = []

    class Transformer:
        @staticmethod
        def visual(_flatten, grid):
            calls.append(grid.clone())
            token_count = int(grid[0, 0] * grid[0, 1] * grid[0, 2] // 4)
            values = torch.arange(token_count, dtype=torch.float32).view(token_count, 1)
            return values, [values + 100]

    model = types.SimpleNamespace(transformer=Transformer())
    frames = torch.zeros((4, 32, 32, 3))
    blocks = [
        textgen_helpers._preprocess_qwen3vl_video_pair(
            model, torch.device("cpu"), frames[index:index + 2]
        )
        for index in range(0, frames.shape[0], 2)
    ]

    assert len(calls) == 2
    assert [grid.tolist() for grid in calls] == [[[1, 4, 4]], [[1, 4, 4]]]
    assert len(blocks) == 2


def test_text_generate_routes_video_and_images_through_qwen3vl_video_generation(monkeypatch):
    captured = {}
    frames = torch.zeros((2, 4, 4, 3))
    image = torch.ones((1, 4, 4, 3))

    def prepare_video(video, resolution, video_fps, sample_fps):
        captured["video_resolution"] = resolution
        return frames, [0, 12]

    monkeypatch.setattr(textgen_nodes, "prepare_qwen3vl_video_frames", prepare_video)
    monkeypatch.setattr(textgen_nodes, "qwen3vl_video_prompt", lambda indices, fps: "VIDEO_BLOCK")

    def generate(clip, full_prompt, images, video_frames, fusion_config, generation_args, thinking=False):
        captured.update(
            prompt=full_prompt,
            images=images,
            frames=video_frames,
            fusion=fusion_config,
            generation_args=generation_args,
        )
        return [7]

    monkeypatch.setattr(textgen_nodes, "generate_qwen3vl_video", generate)
    result = textgen_nodes.UC_TextGenerate.execute(
        _GenerateClip("qwen3vl_8b"),
        "describe image_input_1",
        "",
        384,
        12,
        {"sampling_mode": "off"},
        image_inputs={"image1": image},
        video=torch.zeros((24, 4, 4, 3)),
        video_fps=24.0,
        sample_fps=2.0,
        vlm_video_resolution=256,
    )

    assert result.args == ("decoded", 0)
    assert captured["prompt"].startswith("<|im_start|>user\nVIDEO_BLOCK")
    assert textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["visual_token"] in captured["prompt"]
    assert len(captured["images"]) == 1
    assert torch.equal(captured["frames"], frames)
    assert captured["fusion"] is None
    assert captured["video_resolution"] == 256


def test_text_generate_labels_and_expands_batched_qwen_images_in_prompt_order():
    clip = _GenerateClip("qwen3vl_8b")
    batch = torch.stack([torch.zeros((2, 2, 3)), torch.ones((2, 2, 3))])

    textgen_nodes.UC_TextGenerate.execute(
        clip,
        "describe image_input_1",
        "",
        0,
        12,
        {"sampling_mode": "off"},
        image_inputs={"image1": batch},
    )

    prompt, kwargs = clip.tokenize_calls[0]
    visual_token = textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["visual_token"]
    assert f"Picture 1: {visual_token}Picture 2: {visual_token}" in prompt
    assert len(kwargs["images"]) == 2
    assert torch.equal(kwargs["images"][0], batch[0:1])
    assert torch.equal(kwargs["images"][1], batch[1:2])


def test_text_generate_fuses_images_but_keeps_video_on_mixed_generation_path(monkeypatch):
    captured = {}
    image_a = torch.zeros((1, 4, 4, 3))
    image_b = torch.ones((1, 4, 4, 3))
    config = {"visual_fusion_method": "linear"}

    monkeypatch.setattr(
        textgen_nodes,
        "prepare_qwen3vl_video_frames",
        lambda video, resolution, video_fps, sample_fps: (video[:2], [0, 12]),
    )
    monkeypatch.setattr(textgen_nodes, "qwen3vl_video_prompt", lambda indices, fps: "VIDEO_BLOCK")

    def generate(clip, full_prompt, images, video_frames, fusion_config, generation_args, thinking=False):
        captured.update(prompt=full_prompt, images=images, fusion=fusion_config)
        return [7]

    monkeypatch.setattr(textgen_nodes, "generate_qwen3vl_video", generate)
    textgen_nodes.UC_TextGenerate.execute(
        _GenerateClip("qwen3vl_32b"),
        "describe",
        "",
        384,
        12,
        {"sampling_mode": "off"},
        image_inputs={"image1": image_a, "image2": image_b},
        visual_fusion_config=config,
        video=torch.zeros((24, 4, 4, 3)),
        video_fps=24.0,
        sample_fps=2.0,
    )

    assert captured["fusion"] is config
    assert len(captured["images"]) == 2
    assert captured["prompt"].count(textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["visual_token"]) == 1
    assert "VIDEO_BLOCK" in captured["prompt"]


def test_text_generate_rejects_video_for_non_qwen3vl_model():
    with pytest.raises(ValueError, match="only by Core Qwen3-VL and Qwen3.5"):
        textgen_nodes.UC_TextGenerate.execute(
            _GenerateClip("gemma3_12b"),
            "describe",
            "",
            384,
            12,
            {"sampling_mode": "off"},
            video=torch.zeros((4, 4, 4, 3)),
        )


def test_qwen35_video_generation_supports_primary_visual_blocks_and_mrope():
    class Transformer:
        def __init__(self):
            self.generated = None

        @staticmethod
        def preprocess_embed(_entry, device):
            return (
                torch.full((1, 4, 3), 2.0, device=device),
                torch.tensor([[1, 4, 4]], device=device),
            )

        @staticmethod
        def visual(_flatten, grid):
            size = int(grid[0, 0] * grid[0, 1] * grid[0, 2] // 4)
            return torch.full((size, 3), 3.0)

        def generate(self, embeds, **kwargs):
            self.generated = (embeds, kwargs)
            return [9]

    class Model:
        def __init__(self):
            self.transformer = Transformer()
            self.execution_device = torch.device("cpu")

        def reset_clip_options(self):
            pass

        def set_clip_options(self, options):
            self.execution_device = options["execution_device"]

        def process_tokens(self, rows, device):
            chunks = []
            info = []
            cursor = 0
            for value in rows[0]:
                if isinstance(value, dict):
                    block = value["data"].reshape(-1, value["data"].shape[-1]).to(device)
                    chunks.append(block)
                    info.append({"type": "embedding", "index": cursor, "size": block.shape[0], "extra": None})
                    cursor += block.shape[0]
                else:
                    chunks.append(torch.zeros((1, 3), device=device))
                    cursor += 1
            return torch.cat(chunks).unsqueeze(0), None, None, info

    model = Model()
    stage = types.SimpleNamespace(clip="qwen35_9b", qwen35_9b=model)

    class Clip:
        cond_stage_model = stage
        tokenizer = types.SimpleNamespace(clip_name="qwen35_9b")
        patcher = types.SimpleNamespace(load_device=torch.device("cpu"))

        @staticmethod
        def load_model():
            pass

        @staticmethod
        def tokenize(_prompt, **_kwargs):
            return {
                "qwen35_9b": [[
                    (248053, 1.0),
                    (textgen_helpers.QWEN35_VIDEO_PAD_ID, 1.0),
                    (248054, 1.0),
                    (248053, 1.0),
                    (textgen_helpers.QWEN35_IMAGE_PAD_ID, 1.0),
                    (248054, 1.0),
                ]]
            }

        @staticmethod
        def decode(ids, **_kwargs):
            return f"decoded:{ids}"

    output, seed = textgen_nodes.UC_TextGenerate.execute(
        Clip(),
        "describe",
        "",
        384,
        12,
        {"sampling_mode": "off"},
        image_inputs={"image_1": torch.ones((1, 32, 32, 3))},
        video=torch.zeros((2, 32, 32, 3)),
        vlm_video_resolution=0,
    ).args

    assert output == "decoded:[9]"
    assert seed == 0
    embeds, forwarded = model.transformer.generated
    assert embeds.shape[1] == 12
    assert embeds[0, 1:5, 0].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert embeds[0, 7:11, 0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert forwarded["position_ids"].shape == (3, 12)
    assert forwarded["mtp"] is True
    assert "deepstack_embeds" not in forwarded


def test_qwen3vl_video_generation_inserts_temporal_and_image_blocks_in_prompt_order():
    class Transformer:
        def __init__(self):
            self.generated = None
            self.built_info = None

        @staticmethod
        def preprocess_embed(_entry, device):
            return (
                torch.full((1, 4, 3), 2.0, device=device),
                {
                    "grid": torch.tensor([[1, 4, 4]], device=device),
                    "deepstack": [torch.full((4, 3), 20.0, device=device)],
                },
            )

        @staticmethod
        def visual(_flatten, grid):
            size = int(grid[0, 0] * grid[0, 1] * grid[0, 2] // 4)
            return torch.full((1, size, 3), 3.0), [torch.full((size, 3), 30.0)]

        def build_image_inputs(self, embeds, embeds_info):
            self.built_info = embeds_info
            visual_mask = torch.zeros((1, embeds.shape[1]), dtype=torch.bool)
            deepstack = [torch.cat([entry["extra"]["deepstack"][0] for entry in embeds_info])]
            return torch.zeros((3, embeds.shape[1])), visual_mask, deepstack

        def generate(self, embeds, **kwargs):
            self.generated = (embeds, kwargs)
            return [9]

    class Model:
        def __init__(self):
            self.transformer = Transformer()
            self.execution_device = torch.device("cpu")

        def reset_clip_options(self):
            pass

        def set_clip_options(self, options):
            self.execution_device = options["execution_device"]

        def process_tokens(self, rows, device):
            chunks = []
            info = []
            cursor = 0
            for value in rows[0]:
                if isinstance(value, dict):
                    block = value["data"].reshape(-1, value["data"].shape[-1]).to(device)
                    chunks.append(block)
                    info.append({"type": "embedding", "index": cursor, "size": block.shape[0], "extra": None})
                    cursor += block.shape[0]
                else:
                    chunks.append(torch.zeros((1, 3), device=device))
                    cursor += 1
            return torch.cat(chunks).unsqueeze(0), None, None, info

    model = Model()
    stage = types.SimpleNamespace(clip="qwen3vl_8b", qwen3vl_8b=model)

    class Clip:
        cond_stage_model = stage
        patcher = types.SimpleNamespace(load_device=torch.device("cpu"))

        @staticmethod
        def load_model():
            pass

        @staticmethod
        def tokenize(_prompt, **_kwargs):
            return {
                "qwen3vl_8b": [[
                    (151652, 1.0),
                    (textgen_helpers.QWEN_VIDEO_PAD_ID, 1.0),
                    (151653, 1.0),
                    (151652, 1.0),
                    (textgen_helpers.QWEN_IMAGE_PAD_ID, 1.0),
                    (151653, 1.0),
                ]]
            }

    result = textgen_helpers.generate_qwen3vl_video(
        Clip(),
        "prompt",
        [torch.ones((1, 32, 32, 3))],
        torch.zeros((2, 32, 32, 3)),
        None,
        {
            "do_sample": False,
            "max_length": 12,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "seed": None,
        },
    )

    assert result == [9]
    assert [entry["type"] for entry in model.transformer.built_info] == ["image", "image"]
    assert model.transformer.built_info[0]["extra"]["grid"].tolist() == [[1, 4, 4]]
    assert model.transformer.built_info[1]["extra"]["grid"].tolist() == [[1, 4, 4]]
    assert model.transformer.built_info[0]["extra"]["deepstack"][0][0, 0] == 30.0
    assert model.transformer.built_info[1]["extra"]["deepstack"][0][0, 0] == 20.0
    embeds, forwarded = model.transformer.generated
    assert embeds.shape[1] == 12
    assert forwarded["max_length"] == 12
    assert forwarded["deepstack_embeds"][0].shape[0] == 8


def test_text_generate_parenthesis_escaping_is_optional_and_final():
    clip = _GenerateClip()
    clip.decode = lambda *args, **kwargs: "plain (Overwatch), (banana)"
    common = (clip, "hello", "", 0, 12, {"sampling_mode": "off"})

    assert textgen_nodes.UC_TextGenerate.execute(*common).args == ("plain (Overwatch), (banana)", 0)
    assert textgen_nodes.UC_TextGenerate.execute(*common, escape_parentheses=True).args == (
        r"plain \(Overwatch\), \(banana\)",
        0,
    )


class _GenerateClip:
    def __init__(self, family="qwen3vl_4b"):
        self.tokenizer = types.SimpleNamespace(clip_name=family)
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, prompt, **kwargs):
        self.tokenize_calls.append((prompt, kwargs))
        return {"qwen": [[(1, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((tokens, kwargs))
        return [7]

    def decode(self, token_ids, skip_special_tokens=True):
        return "decoded"


@pytest.mark.parametrize("resolution", [384, 0])
def test_text_generate_delegates_image_sizing_to_shared_vlm_helper(monkeypatch, resolution):
    calls = []
    image = torch.zeros((1, 32, 64, 3))

    def prepare(source, selected_resolution):
        calls.append((source, selected_resolution))
        return source

    monkeypatch.setattr(textgen_nodes, "prepare_vlm_image", prepare)
    textgen_nodes.UC_TextGenerate.execute(
        _GenerateClip(),
        "describe image_input_1",
        "",
        resolution,
        12,
        {"sampling_mode": "off"},
        image_inputs={"image1": image},
    )

    assert [source is image for source, _ in calls] == [True, True]
    assert [selected_resolution for _, selected_resolution in calls] == [resolution, resolution]


@pytest.mark.parametrize("config", [None, {"visual_fusion_method": "off"}])
def test_text_generate_disconnected_or_off_uses_original_generation_path(config):
    clip = _GenerateClip()
    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "hello", "", 0, 12, {"sampling_mode": "off"},
        image_inputs={}, visual_fusion_config=config,
    )
    assert result.args == ("decoded", 0)
    assert len(clip.tokenize_calls) == len(clip.generate_calls) == 1
    assert clip.generate_calls[0][1]["max_length"] == 12


def test_text_generate_retries_blank_sampled_outputs_and_returns_successful_seed():
    clip = _GenerateClip()
    responses = iter(["", " \n", "caption"])
    clip.decode = lambda *args, **kwargs: next(responses)
    sampling = {
        "sampling_mode": "on",
        "seed": 41,
        "empty_response_retries": 4,
    }

    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "", 0, 12, sampling,
        image_inputs={}, visual_fusion_config=None,
    )

    assert result.args == ("caption", 43)
    assert [call[1]["seed"] for call in clip.generate_calls] == [41, 42, 43]


def test_text_generate_retry_seed_wraps_at_unsigned_64_bit_limit():
    clip = _GenerateClip()
    responses = iter(["", "caption"])
    clip.decode = lambda *args, **kwargs: next(responses)
    sampling = {
        "sampling_mode": "on",
        "seed": 0xffffffffffffffff,
        "empty_response_retries": 1,
    }

    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "", 0, 12, sampling,
        image_inputs={}, visual_fusion_config=None,
    )

    assert result.args == ("caption", 0)
    assert [call[1]["seed"] for call in clip.generate_calls] == [
        0xffffffffffffffff,
        0,
    ]


def test_text_generate_does_not_retry_deterministic_blank_output():
    clip = _GenerateClip()
    clip.decode = lambda *args, **kwargs: ""

    result = textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "", 0, 12,
        {"sampling_mode": "off", "empty_response_retries": 4},
        image_inputs={}, visual_fusion_config=None,
    )

    assert result.args == ("", 0)
    assert len(clip.generate_calls) == 1


def test_qwen3vl_system_prompt_uses_official_assistant_generation_boundary():
    clip = _GenerateClip()
    textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "caption rules", 0, 12, {"sampling_mode": "off"},
        image_inputs={}, visual_fusion_config=None,
    )

    prompt = clip.tokenize_calls[0][0]
    assert prompt == (
        "<|im_start|>system\ncaption rules<|im_end|>\n"
        "<|im_start|>user\ndescribe<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<think>" not in prompt


def test_qwen3vl_fused_system_prompt_uses_official_assistant_generation_boundary(monkeypatch):
    captured = {}

    def generate(clip, full_prompt, images, config, generation_args, thinking=False):
        captured["prompt"] = full_prompt
        captured["images"] = images
        return [7]

    monkeypatch.setattr(textgen_nodes, "generate_fused_qwen3vl", generate)
    clip = _GenerateClip()
    image_a = torch.zeros((1, 2, 2, 3))
    image_b = torch.ones((1, 2, 2, 3))
    textgen_nodes.UC_TextGenerate.execute(
        clip, "describe", "caption rules", 0, 12, {"sampling_mode": "off"},
        formula="a",
        image_inputs={"image1": image_a, "image2": image_b},
        visual_fusion_config={"visual_fusion_method": "linear"},
    )

    assert captured["prompt"].endswith("<|im_start|>assistant\n")
    assert "<think>" not in captured["prompt"]
    assert len(captured["images"]) == 2
    assert torch.equal(captured["images"][0], image_a)
    assert torch.equal(captured["images"][1], image_b)


def test_active_visual_fusion_rejects_unsupported_model_without_tokenizing():
    clip = _GenerateClip("gemma3_12b")
    image = torch.zeros((1, 2, 3, 3))
    with pytest.raises(ValueError, match="only by Core Qwen3-VL and Qwen3.5"):
        textgen_nodes.UC_TextGenerate.execute(
            clip, "describe", "", 0, 12, {"sampling_mode": "off"},
            image_inputs={"image1": image},
            visual_fusion_config={"visual_fusion_method": "linear"},
        )
    assert clip.tokenize_calls == []


def test_qwen35_fused_generation_uses_primary_visual_block_and_mrope():
    class Transformer:
        generated = None

        def generate(self, embeds, **kwargs):
            self.generated = (embeds, kwargs)
            return [9]

    class Model:
        def __init__(self):
            self.transformer = Transformer()
            self.execution_device = torch.device("cpu")

        def reset_clip_options(self):
            pass

        def set_clip_options(self, options):
            self.execution_device = options["execution_device"]

        def process_tokens(self, rows, device):
            image = next(value for value in rows[0] if isinstance(value, dict))
            source = float(image["data"].flatten()[0])
            embeds = torch.zeros((1, 6, 2), device=device)
            embeds[0, 1:5] = source
            info = [{"type": "image", "index": 1, "size": 4, "extra": torch.tensor([[1, 4, 4]])}]
            return embeds, None, None, info

    model = Model()
    stage = types.SimpleNamespace(clip="qwen35_2b", qwen35_2b=model)

    class Clip:
        cond_stage_model = stage
        patcher = types.SimpleNamespace(load_device=torch.device("cpu"))

        @staticmethod
        def load_model():
            pass

        @staticmethod
        def tokenize(prompt, images, **kwargs):
            return {"qwen35": [[(10, 1.0), ({"type": "image", "data": images[0]}, 1.0), (11, 1.0)]]}

    images = [torch.zeros((1, 2, 2, 3)), torch.ones((1, 2, 2, 3))]
    args = {"do_sample": False, "max_length": 4, "temperature": 1.0, "top_k": 50,
            "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "seed": None}
    result = textgen_nodes.generate_fused_qwen35(
        Clip(), "prompt", images, {"visual_fusion_method": "spatial-checkerboard"}, args
    )

    assert result == [9]
    embeds, forwarded = model.transformer.generated
    assert embeds[0, 1:5, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert forwarded["position_ids"].shape == (3, 6)
    assert forwarded["max_length"] == 4
    assert "deepstack_embeds" not in forwarded


@pytest.mark.parametrize("clip_name,expected", [
    ("qwen35_2b", "qwen35"),
    ("qwen3vl_4b", "qwen3vl"),
    ("qwen3vl_8b", "qwen3vl"),
    ("qwen3vl_32b", "qwen3vl"),
    ("gemma3_12b", "gemma"),
    ("llama", "llama3"),
])
def test_textgen_template_detection_uses_clip_tokenizer_identity(clip_name, expected):
    tokenizer = types.SimpleNamespace(clip_name=clip_name)
    clip = types.SimpleNamespace(tokenizer=tokenizer)
    assert textgen_nodes.detect_textgen_template(clip) == expected


def test_textgen_template_detection_uses_inner_tokenizer_class_fallback():
    Qwen3VLTokenizer = type("Qwen3VLTokenizer", (), {})
    tokenizer = types.SimpleNamespace(clip="encoder", encoder=Qwen3VLTokenizer())
    clip = types.SimpleNamespace(tokenizer=tokenizer)
    assert textgen_nodes.detect_textgen_template(clip) == "qwen3vl"


@pytest.mark.parametrize("clip", [types.SimpleNamespace(), types.SimpleNamespace(tokenizer=types.SimpleNamespace())])
def test_textgen_template_detection_rejects_unknown_clip(clip):
    with pytest.raises(ValueError, match="tokenizer"):
        textgen_nodes.detect_textgen_template(clip)


def test_qwen_template_families_use_exact_thinking_suppression():
    assert textgen_nodes.MODEL_TEMPLATES["qwen35"]["suppress_thinking"] == "<think>\n</think>\n"
    assert textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["suppress_thinking"] is None
    assert textgen_nodes.MODEL_TEMPLATES["qwen35"]["visual_token"] == textgen_nodes.MODEL_TEMPLATES["qwen3vl"]["visual_token"]


def test_text_generate_response_schema_conformance_and_node_execution():
    from utils_collection_textgen_test.helpers.text_helpers import get_vlm_preset_response_schema

    schema = get_vlm_preset_response_schema("moss_image_json_instruction_color")
    assert isinstance(schema, dict)

    raw_messy = (
        "<think>\nLet me analyze the layout.\n</think>\n"
        "```json\n"
        '{"summary":{"caption":"Poster scene"},"items":[{"label":"TEXT","caption":"Title","text":"HELLO","bbox":[-10,20,1200,800],"palette":["#112233"]}]}'
        "\n```\nExtra trailing commentary."
    )
    conformed_str, is_valid = textgen_nodes.conform_and_validate_response_schema(raw_messy, schema)
    assert is_valid is True
    import json

    parsed = json.loads(conformed_str)
    assert parsed["summary"]["caption"] == "Poster scene"
    assert parsed["bbox_coord_space"] == "abs_centralized"
    assert parsed["bbox_format"] == "yxyx"
    assert parsed["items"][0]["label"] == "text"
    assert parsed["items"][0]["bbox"] == [0, 20, 1000, 800]
    assert parsed["items"][0]["palette"] == ["#112233"]

    class FakeClip:
        tokenizer = types.SimpleNamespace(clip_name="qwen35_9b")

        @staticmethod
        def tokenize(prompt, **kwargs):
            return {"qwen35": [[(1, 1.0)]]}

        @staticmethod
        def generate(tokens, **kwargs):
            return [10, 11]

        @staticmethod
        def decode(ids, **kwargs):
            return '"summary":{"caption":"Auto prefill"},"items":[{"label":"object","caption":"Box","bbox":[100,200,300,400],"palette":["#abcdef"]}],"bbox_coord_space":"normalized_1000","bbox_format":"yxyx"} trailing'

    out_text, out_seed = textgen_nodes.UC_TextGenerate.execute(
        FakeClip(),
        "analyze",
        "",
        384,
        128,
        {"sampling_mode": "off"},
        thinking=False,
        response_schema=schema,
    ).args
    assert out_seed == 0
    out_obj = json.loads(out_text)
    assert out_obj["summary"]["caption"] == "Auto prefill"
    assert out_obj["items"][0]["bbox"] == [100, 200, 300, 400]


def test_project_source_has_no_builtin_eval_calls():
    offenders = []
    tracked = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=CUSTOM_NODE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_paths = (
        CUSTOM_NODE_ROOT / relative_path
        for relative_path in tracked
        if pathlib.PurePosixPath(relative_path).parent == pathlib.PurePosixPath(".")
    )
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
                offenders.append(f"{source_path.name}:{node.lineno}")
    assert offenders == []
