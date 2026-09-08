import inspect
import copy
import pathlib
import sys
import types
from fractions import Fraction

import pytest
import numpy as np
import torch


CUSTOM_NODE_ROOT = pathlib.Path(__file__).parents[1]
PACKAGE_NAME = "utils_collection_encoder_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(CUSTOM_NODE_ROOT)]
sys.modules.setdefault(PACKAGE_NAME, package)

from comfy.cli_args import args as cli_args

prior_cpu = cli_args.cpu
cli_args.cpu = True
try:
    from utils_collection_encoder_test import encoder_helpers, encoder_nodes
    from utils_collection_encoder_test.encoder_nodes import (
        TextEncodeKrea2SystemEditScaledAdv,
        TextEncodeKrea2SysEditScaledAdvAttn,
        UC_AdvancedMiniMaxH3ImageToVideo,
        UC_MiniMaxH3MediaConfig,
        UC_AdvancedVisualConditioningEncode,
        UC_AttentionBiasTextEncode,
        UC_ConditioningConsensusBlend,
        UC_TextConsensusBlendConfig,
        UC_Krea2TokenAttentionWeight,
        UC_Qwen3VLInputEmbeds,
        UC_VisualFusionConfig,
        UC_VLMInputEmbeds,
    )
finally:
    cli_args.cpu = prior_cpu


VAE_MULTIPLE_ENCODERS = (
    "UC_ScaledBiasTextEncodeLtxv2SystemPrompt",
    "TextEncodeSystemEditPlus",
    "TextEncodeSystemEditPlusAdvanced",
    "TextEncodeKrea2SystemEditPlusAdvanced",
    "TextEncodeEditPlusAdvanced",
    "TextEncodeGemmaSystemEditPlusAdvanced",
    "UC_TextEncodeLtxv2SystemPrompt",
    "TextEncodeKrea2SystemEditScaledAdv",
    "TextEncodeKrea2SysEditScaledAdvAttn",
)


@pytest.fixture(autouse=True)
def isolated_h3_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(encoder_helpers.folder_paths, "get_temp_directory", lambda: str(tmp_path))


def test_power_blend_preset_matches_declared_widget_values():
    preset_input = next(value for value in UC_TextConsensusBlendConfig.define_schema().inputs if value.id == "blend_preset")

    assert "power_blend" in preset_input.options
    assert encoder_helpers.POWER_BLEND_PRESET == {
        "method": "consensus",
        "type": "median",
        "align": "similarity",
        "alignment_threshold": 0.9,
        "thresh": 0.75,
        "alpha": 8.0,
        "beta": 0.0,
        "norm": True,
        "scale": 1.0,
        "dsc": True,
        "soft_comfort": False,
    }


def test_text_blend_config_exposes_position_and_prefix_controls():
    inputs = {value.id: value for value in UC_TextConsensusBlendConfig.define_schema().inputs}
    assert inputs["position_weight"].default == 0.0
    assert inputs["preserve_common_prefix"].default is False
    config = UC_TextConsensusBlendConfig.execute(
        "power_blend", "consensus", "median", "similarity", 0.4, 0.0,
        2.0, 0.0, True, 1.0, position_weight=0.75,
        preserve_common_prefix=True,
    ).args[0]
    assert config["position_weight"] == 0.75
    assert config["preserve_common_prefix"] is True


def test_vae_reference_image_uses_configurable_dimension_multiple():
    samples = torch.zeros(1, 3, 101, 205)

    original = encoder_helpers.prepare_vae_reference_image(samples, None, 32)
    targeted = encoder_helpers.prepare_vae_reference_image(samples, 1024, 64)

    assert original.shape[-2:] == (96, 192)
    assert targeted.shape[-2] % 64 == 0
    assert targeted.shape[-1] % 64 == 0
    with pytest.raises(ValueError, match="at least 4"):
        encoder_helpers.prepare_vae_reference_image(samples, None, 3)


@pytest.mark.parametrize("mode", ["single", "parallel-single"])
def test_krea2_reference_latents_are_attached_to_conditioning(mode):
    krea2_stage = type("Krea2CLIP", (), {})()
    clip = types.SimpleNamespace(cond_stage_model=krea2_stage)
    reference = torch.zeros(1, 16, 8, 8)
    conditioning = [[torch.zeros(1, 2, 4), {}]]

    result = encoder_nodes.apply_parallel_ref_latents(
        clip, conditioning, [reference], mode
    )

    assert len(result) == 1
    assert result[0][1]["reference_latents"] == [reference]


def test_reference_latent_encoders_append_configurable_multiple():
    for class_name in VAE_MULTIPLE_ENCODERS:
        schema = getattr(encoder_nodes, class_name).define_schema()
        control = next(value for value in schema.inputs if value.id == "vae_dimension_multiple")
        assert control.default == 8
        assert control.min == 4
        assert control.step == 4
        assert control.advanced


def test_advanced_visual_encoder_applies_non_default_vae_alignment(monkeypatch):
    encoded_shapes = []

    class VAE:
        @staticmethod
        def encode(image):
            encoded_shapes.append(tuple(image.shape))
            return torch.zeros(1, 4, 1, 1)

    monkeypatch.setattr(
        encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image
    )
    monkeypatch.setattr(
        encoder_nodes,
        "encode_embedding_classical_scaled_bias",
        lambda *_args, **_kwargs: [[torch.ones(1, 2, 3), {}]],
    )

    UC_AdvancedVisualConditioningEncode.execute(
        types.SimpleNamespace(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.zeros(1, 101, 205, 3)},
        visual_fusion_config={"visual_fusion_method": "off"},
        vae_resolution="Original",
        ref_latent_mode="single",
        vae=VAE(),
        vae_dimension_multiple=32,
    )

    assert encoded_shapes == [(1, 96, 192, 3)]


def test_expression_grammar_and_nonfinite_rejection():
    value = torch.tensor([1.0, 2.0])
    assert torch.equal(encoder_helpers.evaluate_tensor_expression("clamp(a * 2, 0, 3)", {"a": value}), torch.tensor([2.0, 3.0]))
    with pytest.raises(ValueError, match="Unsupported expression element"):
        encoder_helpers.evaluate_tensor_expression("a.__class__", {"a": value})
    with pytest.raises(ValueError, match="NaN or infinite"):
        encoder_helpers.evaluate_tensor_expression("a / 0", {"a": value})


@pytest.mark.parametrize(
    "first_property", ["RETURN_TYPES", "RETURN_NAMES", "OUTPUT_IS_LIST", "OUTPUT_TOOLTIPS"]
)
def test_advanced_consensus_output_cache_is_local_after_parent_initialization(
    monkeypatch, first_property
):
    parent = UC_TextConsensusBlendConfig
    child = encoder_nodes.UC_AdvancedConsensusConfiguration
    properties = ("RETURN_TYPES", "RETURN_NAMES", "OUTPUT_IS_LIST", "OUTPUT_TOOLTIPS")
    for name in properties:
        # Reset existing caches while preserving whether the subclass owns them.
        monkeypatch.setattr(parent, f"_{name}", None)
        if f"_{name}" in child.__dict__:
            monkeypatch.setattr(child, f"_{name}", None)
    parent_schema = parent.GET_SCHEMA()
    parent_types = parent.RETURN_TYPES.copy()
    expected = child.FINALIZE_SCHEMA().outputs[0]

    getattr(child, first_property)

    assert child.RETURN_TYPES == [expected.io_type]
    assert child.RETURN_NAMES == [expected.display_name]
    assert child.OUTPUT_IS_LIST == [expected.is_output_list]
    assert child.OUTPUT_TOOLTIPS == [expected.tooltip or None]
    assert parent.RETURN_TYPES == parent_types == [parent_schema.outputs[0].io_type]
    assert child.RETURN_TYPES != parent.RETURN_TYPES


def test_visual_fusion_config_selects_real_encoder_path():
    config = UC_VisualFusionConfig.execute(
        visual_fusion_method="spatial-checkerboard",
        visual_block_size=2,
        dither_ratio=0.5,
        seed=0,
        visual_encoder_path="legacy-flat",
    )[0]
    assert config["visual_encoder_path"] == "legacy-flat"


def test_legacy_flat_temporarily_disables_grid_and_deepstack_inputs():
    class Transformer:
        @staticmethod
        def build_image_inputs(embeds, embeds_info):
            return "grid", "mask", "deepstack"

    transformer = Transformer()
    clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(
            clip_model=types.SimpleNamespace(transformer=transformer),
        ),
    )

    with encoder_helpers.qwen3vl_visual_encoder_path(clip, "legacy-flat"):
        assert transformer.build_image_inputs(None, None) == (None, None, None)

    assert transformer.build_image_inputs(None, None) == ("grid", "mask", "deepstack")


def test_inline_image_placeholders_honor_legacy_flat_encoder_path():
    class Transformer:
        @staticmethod
        def build_image_inputs(embeds, embeds_info):
            return "grid", "mask", "deepstack"

    transformer = Transformer()

    class Clip:
        cond_stage_model = types.SimpleNamespace(
            clip_model=types.SimpleNamespace(transformer=transformer),
        )

        @staticmethod
        def tokenize(*_args, **_kwargs):
            assert transformer.build_image_inputs(None, None) == (None, None, None)
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(_tokens):
            assert transformer.build_image_inputs(None, None) == (None, None, None)
            return [[torch.ones(1, 1, 1), {}]]

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="image_input_1",
        system_prompt="",
        vlm_resolution=0,
        image_inputs={"image_1": torch.ones(1, 2, 2, 3)},
        visual_fusion_config={
            "visual_fusion_method": "off",
            "visual_encoder_path": "legacy-flat",
        },
    )

    assert transformer.build_image_inputs(None, None) == ("grid", "mask", "deepstack")


class _MiniMaxH3TestNamespace:
    def __init__(self, **values):
        self.__dict__.update(values)


class _MiniMaxH3TestPatcher:
    forced_hooks = None
    load_device = torch.device("cpu")

    def __init__(self):
        self.object_patches = {}

    def clone(self):
        cloned = copy.copy(self)
        cloned.object_patches = self.object_patches.copy()
        return cloned

    def get_model_object(self, name):
        return self.object_patches.get(name, self.preprocess_embed)

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    @staticmethod
    def preprocess_embed(embed, device):
        return embed["data"], None


class _MiniMaxH3TestClip:
    clip_name = "qwen3vl_32b"
    clip = clip_name

    def __init__(self):
        self.encoded_tokens = []
        self.tokenize_calls = []
        self.cond_stage_model = self
        self.tokenizer = self
        self.patcher = _MiniMaxH3TestPatcher()

    def clone(self):
        cloned = copy.copy(self)
        cloned.patcher = self.patcher.clone()
        return cloned

    def add_hooks_to_dict(self, metadata):
        pass

    @staticmethod
    def _text_entries(text):
        return [] if not text else [(text, 1.0)]

    def tokenize(self, text, images=None, minimax_ref_items=None, **_kwargs):
        self.tokenize_calls.append(
            {
                "text": text,
                "images": images,
                "minimax_ref_items": minimax_ref_items,
            }
        )
        if minimax_ref_items is not None:
            entries = []
            for item in minimax_ref_items:
                if item["type"] == "image":
                    entries.extend(self._text_entries("<Picture 1>: "))
                    entries.extend([(151652, 1.0), ({"type": "image", "data": item["data"]}, 1.0), (151653, 1.0)])
                elif item["type"] == "video":
                    entries.extend(self._text_entries("<Video 1>: "))
                    frames = item["data"]
                    timestamps = list(item["timestamps"])
                    if frames.shape[0] % 2 == 1:
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps.append(timestamps[-1])
                    for index in range(0, frames.shape[0], 2):
                        timestamp = (timestamps[index] + timestamps[index + 1]) / 2
                        entries.extend(self._text_entries(f"<{float(timestamp):.1f} seconds>"))
                        entries.extend([(151652, 1.0), ({"type": "image", "data": frames[index:index + 2], "minimax_video_block": True}, 1.0), (151653, 1.0)])
            entries.extend(self._text_entries(text))
            return {"qwen3vl_32b": [entries]}
        entries = []
        for index, image in enumerate(images or []):
            entries.extend(self._text_entries(f"<Picture {index + 1}>: "))
            entries.extend(
                [
                    (151652, 1.0),
                    ({"type": "image", "data": image}, 1.0),
                    (151653, 1.0),
                ]
            )
        entries.extend(self._text_entries(text))
        if not entries:
            entries = [(151643, 1.0)]
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded_tokens.append(tokens)
        entries = tokens["qwen3vl_32b"][0]
        tag_values = []
        for entry in entries:
            span = (
                encoder_helpers._qwen3vl_image_span(entry)
                if encoder_helpers.is_image_token(entry)
                else 1
            )
            tag_values.extend(
                [0 if encoder_helpers.is_image_token(entry) or entry[0] in (151652, 151653) else 1] * span
            )
        length = len(tag_values)
        tags = torch.tensor(tag_values)
        return [[torch.ones(1, length, 4), {"minimax_token_tags": tags}]]


class _RecordingMiniMaxVAE:
    def __init__(self):
        self.images = []

    def encode(self, image):
        self.images.append(image)
        return torch.full((1, 4, 1, 1), float(image.mean()))










@pytest.mark.parametrize("mode", ["disabled", "all", "images_only", "video_only"])
def test_h3_cache_modes_preserve_joint_presentation(mode):
    clip = _MiniMaxH3TestClip()
    image = torch.ones(1, 64, 64, 3)
    def execute(prompt):
        return encoder_helpers.execute_advanced_minimax_h3_image_to_video(
            clip, None, prompt, 64, 64, 5, first_frame=image, ref_image_size="none",
            vlm_resolution=0, enable_caching=mode,
        )[0]
    first = execute("subject")
    assert len(clip.encoded_tokens) == 1
    row = clip.encoded_tokens[0]["qwen3vl_32b"][0]
    assert any(encoder_helpers.is_image_token(entry) for entry in row)
    assert row[-1][0] == "subject"
    second = execute("subject")
    assert len(clip.encoded_tokens) == (2 if mode == "disabled" else 1)
    torch.testing.assert_close(first[0][0], second[0][0])
    execute("changed")
    assert len(clip.encoded_tokens) == (3 if mode == "disabled" else 2)


def test_minimax_h3_prompt_tokens_preserve_inline_order_and_raw_syntax():
    clip = _MiniMaxH3TestClip()
    first = torch.tensor([1.0])
    second = torch.tensor([2.0])
    prompt = encoder_helpers.format_minimax_h3_prompt(
        f"first {encoder_helpers.VISION_BLOCK} then {encoder_helpers.VISION_BLOCK}",
        "system",
    )

    tokens = encoder_helpers.tokenize_minimax_h3_prompt(
        clip, prompt, [second, first]
    )["qwen3vl_32b"][0]
    image_entries = [entry[0]["data"] for entry in tokens if encoder_helpers.is_image_token(entry)]
    text = "".join(entry[0] for entry in tokens if isinstance(entry[0], str))

    assert torch.equal(image_entries[0], second)
    assert torch.equal(image_entries[1], first)
    assert text == "system\nfirst <Picture 1>:  then <Picture 2>: "
    assert "<|im_start|>" not in text


def test_minimax_h3_implicit_picture_stays_before_system_text():
    formatted = encoder_helpers.format_minimax_h3_prompt(
        encoder_helpers.VISION_BLOCK + "prompt", "system"
    )
    assert formatted == encoder_helpers.VISION_BLOCK + "system\nprompt"


def test_advanced_visual_encoder_uses_minimax_inline_picture_syntax(monkeypatch):
    clip = _MiniMaxH3TestClip()
    first = torch.ones(1, 2, 2, 3)
    second = torch.full((1, 2, 2, 3), 2.0)
    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)

    output = UC_AdvancedVisualConditioningEncode.execute(
        clip,
        prompt="second image_input_2 then image_input_1",
        system_prompt="system",
        vlm_resolution=0,
        image_inputs={"image_1": first, "image_2": second},
        visual_fusion_config={"visual_fusion_method": "off"},
    ).args[0]

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert torch.equal(images[0], second)
    assert torch.equal(images[1], first)
    assert "<Picture 1>: " in text and "<Picture 2>: " in text
    assert "<|im_start|>" not in text
    assert output[0][1]["minimax_token_tags"].numel() == output[0][0].shape[1]


def test_advanced_visual_encoder_rejects_generic_minimax_reference_latents():
    with pytest.raises(ValueError, match="MiniMax H3 reference latents require Core"):
        UC_AdvancedVisualConditioningEncode.execute(
            _MiniMaxH3TestClip(),
            prompt="prompt",
            system_prompt="",
            vlm_resolution=0,
            image_inputs={},
            visual_fusion_config={"visual_fusion_method": "off"},
            ref_latent_mode="single",
        )


def test_advanced_minimax_h3_node_schema_separates_visual_roles():
    schema = UC_AdvancedMiniMaxH3ImageToVideo.define_schema()
    inputs = {value.id: value for value in schema.inputs}

    assert schema.node_id == "UC_AdvancedMiniMaxH3ImageToVideo"
    assert schema.display_name == "Advanced MiniMax H3 Image to Video"
    assert [value.id for value in schema.inputs] == [
        "clip",
        "vae",
        "first_frame",
        "last_frame",
        "prompt",
        "width",
        "height",
        "length",
        "visual_fusion_config",
        "multiplier",
            "ref_image_size",
            "vlm_resolution",
            "vlm_video_resolution",
            "enable_caching",
            "reference_images",
        "fusion_images",
        "media_config",
        "video",
        "audio",
            "audio_vae",
    ]
    assert inputs["vae"].optional is True
    assert inputs["first_frame"].optional is True
    assert inputs["last_frame"].optional is True
    assert inputs["video"].optional is True
    assert inputs["audio"].optional is True
    assert inputs["audio_vae"].optional is True
    assert "system_prompt" not in inputs
    assert "keyframe_mode" not in inputs
    assert inputs["reference_images"].template.names == [
        f"reference_image_{index}" for index in range(1, 33)
    ]
    assert inputs["fusion_images"].template.names == [
        f"fusion_image_{index}" for index in range(1, 33)
    ]
    assert inputs["reference_images"].template.min == 0
    assert inputs["fusion_images"].template.min == 0
    assert inputs["ref_image_size"].options == ["match", "max", "none"]
    assert inputs["ref_image_size"].default == "match"
    assert inputs["vlm_resolution"].default == 384
    assert "independent" in inputs["vlm_resolution"].tooltip.lower()
    assert inputs["vlm_video_resolution"].default == 384
    assert "more visual tokens" in inputs["vlm_video_resolution"].tooltip
    fusion_tooltip = inputs["fusion_images"].tooltip
    assert "socket N targets Picture N" in fusion_tooltip
    assert "broadcasts to every reference Picture" in fusion_tooltip
    assert "flattened fusion images pair by index" in fusion_tooltip
    assert "Without frames or references" in fusion_tooltip
    assert "native-reference mode ignores them" in fusion_tooltip
    assert "Video blocks are never fusion targets" in fusion_tooltip
    assert "cannot be combined with explicit first/last frame inputs" in (
        inputs["reference_images"].tooltip
    )
    assert [output.display_name for output in schema.outputs] == [
        "positive",
        None,
    ]


def test_minimax_h3_media_config_schema_and_payload():
    schema = UC_MiniMaxH3MediaConfig.define_schema()
    inputs = {value.id: value for value in schema.inputs}
    assert schema.is_input_list is True
    assert inputs["timestamps"].optional is True
    assert inputs["timestamp_format"].default == "0.0s"
    assert inputs["structure"].default == encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE
    assert inputs["structure"].default == "<<picture>>: <<visual>>"
    assert "At <<time>>, <<picture>>: <<visual>> (from <<shot>>)" in inputs["structure"].tooltip
    assert inputs["video_fps"].default == 2
    assert inputs["video_fps"].min == 1
    assert inputs["video_fps"].max == 24
    assert [value.id for value in schema.inputs][-5:] == [
        "video_fps", "video_latent_mode", "video_latent_keyframes", "temporal_density", "temporal_fusion_method"
    ]
    assert inputs["video_latent_mode"].default == "even keyframes"
    assert inputs["video_latent_keyframes"].default == 4
    assert inputs["video_latent_keyframes"].min == 2
    assert inputs["video_latent_keyframes"].max == 213
    assert "video_structure" not in inputs
    assert "audio" not in inputs
    assert "audio_vae" not in inputs
    assert "video_images" not in inputs
    payload = encoder_helpers.build_minimax_h3_media_config([["0; 1.2"]], video_fps=[5])
    assert payload["schema_version"] == 3
    assert payload["timestamps_seconds"] == (Fraction(0), Fraction(6, 5))
    assert payload["video_fps"] == 5
    assert payload["video_latent_mode"] == "even keyframes"
    assert payload["video_latent_keyframes"] == 4
    assert payload["timestamp_format"] == "0.0s"
    assert payload["structure"] == encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE
    assert "video_structure" not in payload
    assert "audio" not in payload
    assert "audio_vae" not in payload
    default_payload = encoder_helpers.build_minimax_h3_media_config(None)
    assert default_payload["timestamps_seconds"] == (Fraction(0),)
    assert default_payload["default_single_visual"] is True


def test_minimax_h3_media_config_normalizes_video_latent_controls():
    payload = encoder_helpers.build_minimax_h3_media_config(
        None,
        video_latent_mode=["even keyframes"],
        video_latent_keyframes=[8],
    )
    assert payload["video_latent_mode"] == "even keyframes"
    assert payload["video_latent_keyframes"] == 8
    legacy_payload = payload.copy()
    legacy_payload.pop("video_latent_mode")
    legacy_payload.pop("video_latent_keyframes")
    validated = encoder_helpers._validate_minimax_h3_media_config(legacy_payload, 124)
    assert validated[-2:] == ("even keyframes", 4)


@pytest.mark.parametrize("value", [True, 1, 214, 2.5])
def test_minimax_h3_media_config_rejects_invalid_video_latent_keyframes(value):
    with pytest.raises(ValueError, match="video latent keyframes"):
        encoder_helpers.build_minimax_h3_media_config(
            None, video_latent_keyframes=value
        )


def test_minimax_h3_media_config_rejects_invalid_video_latent_mode():
    with pytest.raises(ValueError, match="video latent mode"):
        encoder_helpers.build_minimax_h3_media_config(
            None, video_latent_mode="unknown"
        )


def test_minimax_h3_default_media_config_requires_one_visual():
    with pytest.raises(ValueError, match="requires exactly one visual source"):
        encoder_helpers.tokenize_minimax_h3_media_prompt(
            None,
            "prompt",
            [object(), object()],
            [Fraction(0)],
            "0.0s",
            encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
            default_single_visual=True,
        )


@pytest.mark.parametrize("video_fps", [0, 25, 2.5, True])
def test_minimax_h3_media_config_rejects_invalid_video_fps(video_fps):
    with pytest.raises(ValueError, match="video_fps"):
        encoder_helpers.build_minimax_h3_media_config(None, video_fps=video_fps)


def test_minimax_h3_media_config_rejects_timestamp_beyond_output_duration():
    config = {
        "schema_version": 3,
        "timestamps_seconds": (Fraction("1.01"),),
        "timestamp_format": "0.0s",
        "structure": encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        "video_fps": 2,
    }
    with pytest.raises(ValueError, match="output duration"):
        encoder_helpers._validate_minimax_h3_media_config(config, 24)


@pytest.mark.parametrize("audio_vae", [None, object()])
def test_minimax_h3_missing_audio_skips_vae(audio_vae):
    assert encoder_helpers._encode_minimax_h3_audio_reference(None, audio_vae) is None


def test_minimax_h3_audio_requires_vae():
    audio = {"waveform": torch.ones(1, 2, 8), "sample_rate": 32000}
    with pytest.raises(ValueError, match="requires audio_vae when audio is provided"):
        encoder_helpers._encode_minimax_h3_audio_reference(audio, None)


@pytest.mark.parametrize("audio", [
    {},
    {"waveform": torch.ones(1, 2, 0), "sample_rate": 32000},
    {"waveform": torch.full((1, 2, 8), float("nan")), "sample_rate": 32000},
    {"waveform": torch.ones(1, 2, 8), "sample_rate": 0},
])
def test_minimax_h3_malformed_audio_still_fails(audio):
    with pytest.raises(ValueError, match="MiniMax H3 audio"):
        encoder_helpers._encode_minimax_h3_audio_reference(audio, object())


@pytest.mark.parametrize("node_name", [
    "UC_AdvancedMiniMaxH3ImageToVideo",
    "UC_AdvMiniMaxH3ImageToVideoTokenFusion",
])
def test_minimax_h3_audio_vae_is_conditionally_lazy(node_name):
    node = getattr(encoder_nodes, node_name)
    inputs = {value.id: value for value in node.define_schema().inputs}
    assert inputs["audio_vae"].optional is True
    assert inputs["audio_vae"].lazy is True
    assert not inputs["audio"].lazy
    assert node.check_lazy_status() == []
    assert node.check_lazy_status(audio=None, audio_vae=None, clip=object()) == []
    assert node.check_lazy_status(audio=None, audio_vae=object()) == []
    audio = {"waveform": torch.ones(1, 2, 8), "sample_rate": 32000}
    assert node.check_lazy_status(audio=audio, audio_vae=None) == ["audio_vae"]
    assert node.check_lazy_status(audio=audio, audio_vae=object()) == []


def test_minimax_h3_audio_reference_matches_core_contract(monkeypatch):
    class AudioVAE:
        audio_sample_rate = 32000

        def encode(self, waveform):
            assert waveform.shape == (1, 8, 2)
            return torch.ones(1, 32, 2, 4)

    called = []
    monkeypatch.setattr(encoder_helpers.torchaudio.functional, "resample", lambda waveform, source, target: called.append((source, target)) or waveform)
    block = encoder_helpers._encode_minimax_h3_audio_reference(
        {"waveform": torch.ones(2, 2, 8), "sample_rate": 16000},
        AudioVAE(),
    )
    assert called == [(16000, 32000)]
    assert block["kind"] == "audio"
    assert block["ref_audio_t"] == 4


def test_minimax_h3_reference_video_matches_core_resize_trim_and_payload(monkeypatch):
    resized = []

    def upscale(samples, width, height, method, crop):
        resized.append((samples.shape, width, height, method, crop))
        return torch.zeros(samples.shape[0], samples.shape[1], height, width)

    class VideoVAE:
        def encode(self, frames):
            assert frames.shape == (22, 96, 192, 3)
            return torch.ones(1, 4, 3, 6, 12)

    monkeypatch.setattr(encoder_helpers.comfy.utils, "common_upscale", upscale)
    frames, reference = encoder_helpers.prepare_minimax_h3_reference_video(
        torch.ones(30, 100, 200, 3), VideoVAE(), 23
    )
    assert frames.shape == (22, 96, 192, 3)
    assert resized == [((22, 3, 100, 200), 192, 96, "lanczos", "disabled")]
    assert reference["kind"] == "video"
    assert reference["latent_t"] == 3
    assert reference["latent_h"] == 6
    assert reference["latent_w"] == 12
    assert reference["ref_audio_t"] == 0
    assert reference["latent"].shape == (1, 4, 3, 6, 12)
    assert reference["audio_latent"] is None


def test_minimax_h3_reference_video_none_mode_skips_vae(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.utils,
        "common_upscale",
        lambda samples, _width, _height, _method, _crop: samples,
    )
    frames, reference = encoder_helpers.prepare_minimax_h3_reference_video(
        torch.ones(22, 64, 64, 3), None, 22, encode_reference=False
    )
    assert frames.shape == (22, 64, 64, 3)
    assert reference is None


@pytest.mark.parametrize(
    ("source_frames", "expected_indices", "expected_positions"),
    [
        (362, [0, 7, 14, 21], [0, 119, 238, 357]),
        (345, [0, 7, 13, 20], [0, 119, 221, 340]),
    ],
)
def test_minimax_h3_positioned_video_keyframes_cover_complete_duration(
    monkeypatch, source_frames, expected_indices, expected_positions
):
    encoded = {}

    def upscale(samples, width, height, method, crop):
        assert (width, height, method, crop) == (64, 64, "lanczos", "center")
        return torch.zeros(samples.shape[0], 3, height, width)

    class VideoVAE:
        def encode(self, frames):
            latent_t = ((frames.shape[0] - 5) // 17) * 5 + 2
            latent = torch.arange(
                latent_t * 4 * 4, dtype=torch.float32
            ).reshape(1, 1, latent_t, 4, 4)
            encoded["latent"] = latent
            return latent

    monkeypatch.setattr(encoder_helpers.comfy.utils, "common_upscale", upscale)
    keyframes = encoder_helpers.prepare_minimax_h3_positioned_video_keyframes(
        torch.ones(source_frames, 8, 8, 3),
        VideoVAE(),
        source_frames,
        64,
        64,
        4,
    )
    assert [item["resolved_frame_index"] for item in keyframes] == expected_positions
    assert [item["resolved_frame_index"] // 17 for item in keyframes] == expected_indices
    assert [item["latent"].shape[2] for item in keyframes] == [5, 5, 5, 2]
    assert sum(item["latent"].shape[2] for item in keyframes) == 17
    assert all(item["latent"].shape[3:5] == (4, 4) for item in keyframes)
    assert all(item["latent"]._base is None for item in keyframes)
    assert all(
        item["latent"].untyped_storage().data_ptr()
        != encoded["latent"].untyped_storage().data_ptr()
        for item in keyframes
    )


def test_minimax_h3_positioned_video_keyframes_keep_single_final_chunk(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.utils,
        "common_upscale",
        lambda samples, width, height, _method, _crop: torch.zeros(
            samples.shape[0], 3, height, width
        ),
    )

    class VideoVAE:
        def encode(self, _frames):
            return torch.ones(1, 1, 2, 4, 4)

    keyframes = encoder_helpers.prepare_minimax_h3_positioned_video_keyframes(
        torch.ones(5, 8, 8, 3), VideoVAE(), 5, 64, 64, 8
    )
    assert len(keyframes) == 1
    assert keyframes[0]["resolved_frame_index"] == 0
    assert keyframes[0]["latent"].shape == (1, 1, 2, 4, 4)


def test_minimax_h3_positioned_video_keyframes_reject_bad_vae_shape(monkeypatch):
    monkeypatch.setattr(
        encoder_helpers.comfy.utils,
        "common_upscale",
        lambda samples, width, height, _method, _crop: torch.zeros(
            samples.shape[0], 3, height, width
        ),
    )

    class VideoVAE:
        def encode(self, _frames):
            return torch.ones(1, 1, 6, 4, 4)

    with pytest.raises(ValueError, match="unexpected temporal length"):
        encoder_helpers.prepare_minimax_h3_positioned_video_keyframes(
            torch.ones(22, 8, 8, 3), VideoVAE(), 22, 64, 64, 4
        )


@pytest.mark.parametrize("connect_media_config", [False, True])
def test_advanced_minimax_h3_media_config_uses_default_two_fps_presentation(
    connect_media_config,
):
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 7, frames.shape[1] // 16, frames.shape[2] // 16)

    clip = _MiniMaxH3TestClip()
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=torch.ones(22, 64, 64, 3),
        media_config=(
            encoder_helpers.build_minimax_h3_media_config(None)
            if connect_media_config else None
        ),
        enable_caching="disabled",
    )
    video_calls = [
        call for call in clip.tokenize_calls
        if call["minimax_ref_items"]
        and call["minimax_ref_items"][0]["type"] == "video"
    ]
    assert len(video_calls) == 1
    video_item = video_calls[0]["minimax_ref_items"][0]
    assert video_item["data"].shape[0] == 2
    assert video_item["timestamps"] == [Fraction(0), Fraction(1, 2)]
    if connect_media_config:
        assert [
            item["resolved_frame_index"]
            for item in conditioning[0][1]["minimax_keyframes"]
        ] == [0, 17]
        assert "minimax_refs" not in conditioning[0][1]
    else:
        assert conditioning[0][1]["minimax_refs"][0]["kind"] == "video"


def test_minimax_h3_video_sample_indices_support_non_divisor_rates():
    assert encoder_helpers.minimax_h3_video_sample_indices(24, 2) == [0, 12]
    assert encoder_helpers.minimax_h3_video_sample_indices(24, 5) == [0, 5, 10, 14, 19]
    assert encoder_helpers.minimax_h3_video_sample_indices(24, 24) == list(range(24))


def test_advanced_minimax_h3_video_fps_samples_source_and_keeps_latent():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 2, frames.shape[1] // 16, frames.shape[2] // 16)

    clip = _MiniMaxH3TestClip()
    video = (torch.arange(24, dtype=torch.float32) / 23).view(24, 1, 1, 1).expand(24, 64, 64, 3)
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=video,
        media_config=encoder_helpers.build_minimax_h3_media_config(
            None, video_fps=5, video_latent_mode="full video"
        ),
        enable_caching="disabled",
    )
    video_call = next(
        call for call in clip.tokenize_calls
        if call["minimax_ref_items"]
        and call["minimax_ref_items"][0]["type"] == "video"
    )
    video_item = video_call["minimax_ref_items"][0]
    assert video_item["data"].shape[0] == 6
    assert video_item["timestamps"] == [
        Fraction(0), Fraction(5, 24), Fraction(5, 12),
        Fraction(7, 12), Fraction(19, 24), Fraction(19, 24),
    ]
    assert conditioning[0][1]["minimax_refs"][0]["kind"] == "video"


def test_advanced_minimax_h3_explicit_full_video_is_independent_of_ref_image_size():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 2, frames.shape[1] // 16, frames.shape[2] // 16)

    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        _MiniMaxH3TestClip(),
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=torch.ones(22, 64, 64, 3),
        ref_image_size="none",
        media_config=encoder_helpers.build_minimax_h3_media_config(
            None, video_latent_mode="full video"
        ),
        enable_caching="disabled",
    )
    assert conditioning[0][1]["minimax_refs"][0]["kind"] == "video"


def test_advanced_minimax_h3_disconnected_config_preserves_none_video_fallback():
    class FailingVAE:
        def encode(self, _frames):
            raise AssertionError("Disconnected legacy none mode must not encode Video")

    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        _MiniMaxH3TestClip(),
        FailingVAE(),
        "prompt",
        64,
        64,
        22,
        video=torch.ones(22, 64, 64, 3),
        ref_image_size="none",
        media_config=None,
        enable_caching="disabled",
    )
    assert "minimax_refs" not in conditioning[0][1]
    assert "minimax_keyframes" not in conditioning[0][1]


def test_advanced_minimax_h3_video_latent_off_keeps_qwen_video_without_vae():
    class FailingVAE:
        def encode(self, _frames):
            raise AssertionError("Video VAE must not run in off mode")

    clip = _MiniMaxH3TestClip()
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        FailingVAE(),
        "prompt",
        64,
        64,
        22,
        video=torch.ones(22, 64, 64, 3),
        media_config=encoder_helpers.build_minimax_h3_media_config(
            None, video_latent_mode="off"
        ),
        enable_caching="disabled",
    )
    assert "minimax_refs" not in conditioning[0][1]
    assert any(
        call["minimax_ref_items"]
        and call["minimax_ref_items"][0]["type"] == "video"
        for call in clip.tokenize_calls
    )


def test_advanced_minimax_h3_even_video_keyframes_override_none():
    class VideoVAE:
        def encode(self, frames):
            assert frames.shape == (22, 64, 64, 3)
            return torch.ones(1, 4, 7, 4, 4)

    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        _MiniMaxH3TestClip(),
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=torch.ones(22, 64, 64, 3),
        ref_image_size="none",
        media_config=encoder_helpers.build_minimax_h3_media_config(
            None,
            video_latent_mode="even keyframes",
            video_latent_keyframes=4,
        ),
        enable_caching="disabled",
    )
    keyframes = conditioning[0][1]["minimax_keyframes"]
    assert [item["resolved_frame_index"] for item in keyframes] == [0, 17]
    assert [item["latent"].shape[2] for item in keyframes] == [5, 2]
    assert "minimax_refs" not in conditioning[0][1]


def test_minimax_h3_tokenfusion_uses_shared_even_video_keyframes():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 7, frames.shape[1] // 16, frames.shape[2] // 16)

    config = encoder_helpers.build_minimax_h3_media_config(
        None, video_latent_mode="even keyframes"
    )
    standard = encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTokenFusion.execute(
        clip=_MiniMaxH3TestClip(),
        vae=VideoVAE(),
        prompt="prompt",
        width=64,
        height=64,
        length=22,
        ref_image_size="none",
        video=torch.ones(22, 64, 64, 3),
        media_config=config,
    ).args[0]
    keyframes = standard[0][1]["minimax_keyframes"]
    assert [item["resolved_frame_index"] for item in keyframes] == [0, 17]
    assert [item["latent"].shape[2] for item in keyframes] == [5, 2]


def test_minimax_h3_video_latent_modes_do_not_change_qwen_video_presentation():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(1, 4, 7, frames.shape[1] // 16, frames.shape[2] // 16)

    video = (torch.arange(22, dtype=torch.float32) / 21).view(22, 1, 1, 1).expand(22, 64, 64, 3)
    presentations = []
    for mode in ("full video", "even keyframes", "off"):
        clip = _MiniMaxH3TestClip()
        encoder_helpers.execute_advanced_minimax_h3_image_to_video(
            clip,
            VideoVAE(),
            "prompt",
            64,
            64,
            22,
            video=video,
            ref_image_size="none",
            media_config=encoder_helpers.build_minimax_h3_media_config(
                None, video_fps=5, video_latent_mode=mode
            ),
            enable_caching="disabled",
        )
        video_item = next(
            call["minimax_ref_items"][0]
            for call in clip.tokenize_calls
            if call["minimax_ref_items"]
            and call["minimax_ref_items"][0]["type"] == "video"
        )
        presentations.append((video_item["data"], video_item["timestamps"]))
    assert all(torch.equal(data, presentations[0][0]) for data, _timestamps in presentations)
    assert all(timestamps == presentations[0][1] for _data, timestamps in presentations)


@pytest.mark.parametrize("connect_media_config", [False, True])
def test_advanced_minimax_h3_video_qwen_frames_use_vlm_resolution(
    monkeypatch, connect_media_config,
):
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(
                1, 4, 2, max(1, frames.shape[1] // 16), max(1, frames.shape[2] // 16)
            )

    calls = []

    def prepare(image, resolution):
        calls.append((tuple(image.shape), resolution))
        return image

    monkeypatch.setattr(encoder_helpers, "prepare_vlm_image", prepare)
    video = torch.ones(22, 64, 96, 3)
    media_config = (
        encoder_helpers.build_minimax_h3_media_config(
            None, video_latent_mode="full video"
        )
        if connect_media_config else None
    )
    encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        _MiniMaxH3TestClip(),
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        video=video,
        media_config=media_config,
        vlm_resolution=256,
        vlm_video_resolution=512,
        enable_caching="disabled",
    )
    assert calls == [((1, 64, 96, 3), 512), ((1, 64, 96, 3), 512)]


@pytest.mark.parametrize("connect_media_config", [False, True])
def test_advanced_minimax_h3_keeps_reference_pictures_and_video_together(
    connect_media_config,
):
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(
                1, 4, 2, max(1, frames.shape[1] // 16), max(1, frames.shape[2] // 16)
            )

    clip = _MiniMaxH3TestClip()
    reference = torch.full((1, 64, 64, 3), 0.25)
    video = torch.full((22, 64, 64, 3), 0.75)
    media_config = (
        encoder_helpers.build_minimax_h3_media_config(
            [Fraction(0)], video_latent_mode="full video"
        )
        if connect_media_config else None
    )
    conditioning, _latent = encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        reference_images={"reference_image_1": reference},
        video=video,
        media_config=media_config,
        enable_caching="disabled",
    )
    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert text.index("<Picture 1>") < text.index("<Video 1>") < text.index("prompt")
    kinds = [reference["kind"] for reference in conditioning[0][1]["minimax_refs"]]
    assert kinds == ["image", "video"]


def test_advanced_minimax_h3_default_media_keeps_all_pictures_with_video():
    class VideoVAE:
        def encode(self, frames):
            return torch.ones(
                1, 4, 2, max(1, frames.shape[1] // 16), max(1, frames.shape[2] // 16)
            )

    clip = _MiniMaxH3TestClip()
    media_config = encoder_helpers.build_minimax_h3_media_config(
        None,
        timestamp_format="0.00s",
        video_fps=12,
        video_latent_mode="off",
    )
    encoder_helpers.execute_advanced_minimax_h3_image_to_video(
        clip,
        VideoVAE(),
        "prompt",
        64,
        64,
        22,
        reference_images={
            "reference_image_1": torch.full((1, 64, 64, 3), 0.25),
            "reference_image_2": torch.full((1, 64, 64, 3), 0.75),
        },
        video=torch.full((22, 64, 64, 3), 0.5),
        media_config=media_config,
        ref_image_size="none",
        enable_caching="disabled",
    )

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [
        entry[0] for entry in entries if encoder_helpers.is_image_token(entry)
    ]
    assert "<Picture 1>: <Picture 2>: <Video 1>: " in text
    assert [float(visuals[index]["data"].mean()) for index in range(2)] == pytest.approx(
        [0.25, 0.75]
    )


def test_advanced_minimax_h3_even_video_keyframes_reject_native_image_latents():
    class VideoVAE:
        def encode(self, frames):
            latent_t = 7 if frames.shape[0] == 22 else 1
            return torch.ones(
                1, 4, latent_t, frames.shape[1] // 16, frames.shape[2] // 16
            )

    with pytest.raises(ValueError, match="require ref_image_size none"):
        encoder_helpers.execute_advanced_minimax_h3_image_to_video(
            _MiniMaxH3TestClip(),
            VideoVAE(),
            "prompt",
            64,
            64,
            22,
            reference_images={"reference_image_1": torch.ones(1, 64, 64, 3)},
            video=torch.ones(22, 64, 64, 3),
            media_config=encoder_helpers.build_minimax_h3_media_config(
                None, video_latent_mode="even keyframes"
            ),
            enable_caching="disabled",
        )


def test_minimax_h3_media_tokenization_default_matches_core_picture_constructor():
    clip = _MiniMaxH3TestClip()
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [torch.ones(1, 2, 2, 3)],
        [Fraction(0)],
        "0.00s",
        encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert text == "<Picture 1>: prompt"


def test_minimax_h3_media_tokenization_builds_picture_anchors_before_prompt():
    clip = _MiniMaxH3TestClip()
    first = torch.ones(1, 2, 3, 3)
    second = torch.ones(1, 3, 2, 3)
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [first, second],
        [Fraction("1.21"), Fraction("2.46")],
        "0.00s",
        "At <<time>>, <<picture>>: <<visual>> (from <<shot>>) is fully anchored.",
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [entry[0] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert "At 1.21s, <Picture 1>: " in text
    assert "(from [Shot 1]) is fully anchored." in text
    assert "At 2.46s, <Picture 2>: " in text
    assert "(from [Shot 2]) is fully anchored." in text
    assert text.index("At 1.21s") < text.index("At 2.46s") < text.index("prompt")
    assert visuals[0]["data"].shape == (1, 2, 3, 3)
    assert visuals[1]["data"].shape == (1, 3, 2, 3)
    image_calls = [call for call in clip.tokenize_calls if call["images"] is not None]
    assert len(image_calls) == 1
    assert len(image_calls[0]["images"]) == 2
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    tensor, metadata = conditioning[0]
    assert metadata["minimax_token_tags"].numel() == tensor.shape[1]


def test_minimax_h3_media_tokenization_leaves_unanchored_pictures_once():
    clip = _MiniMaxH3TestClip()
    base = torch.zeros(1, 2, 2, 3)
    shot = torch.ones(1, 2, 2, 3)
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [base, shot],
        [Fraction("61.25")],
        "MM:SS.mmm",
        "<<shot>> @ <<time>> uses <<picture>> = <<visual>>",
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    assert "[Shot 1] @ 01:01.250 uses <Picture 1> = " in text
    assert "<Picture 2>: " in text
    visuals = [entry[0] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert len(visuals) == 2


def test_minimax_h3_media_tokenization_rejects_more_timestamps_than_pictures():
    with pytest.raises(ValueError, match="2 timestamps for 1 available Pictures"):
        encoder_helpers.tokenize_minimax_h3_media_prompt(
            _MiniMaxH3TestClip(),
            "prompt",
            [torch.ones(1, 2, 2, 3)],
            [Fraction(0), Fraction(1)],
            "0.0s",
            encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        )


def test_minimax_h3_media_tokenization_combines_picture_and_restricted_video():
    clip = _MiniMaxH3TestClip()
    picture = torch.ones(1, 2, 2, 3)
    frames = [
        torch.full((1, 2, 2, 3), 2.0),
        torch.full((1, 2, 2, 3), 3.0),
    ]
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [picture],
        [Fraction(0)],
        "0.00s",
        "At <<time>>, <<picture>>: <<visual>> (from <<shot>>) is fully anchored.",
        video_frames=frames,
        video_timestamps=[Fraction(1), Fraction(2)],
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [entry[0] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert "At 0.00s, <Picture 1>: " in text
    assert text.count("<Video 1>: ") == 1
    assert "<1.50s>" in text
    assert text.index("<Picture 1>") < text.index("<Video 1>") < text.index("prompt")
    assert visuals[1]["minimax_video_block"] is True
    assert visuals[1]["data"].shape[0] == 2


def test_minimax_h3_default_media_allows_multiple_pictures_with_video():
    clip = _MiniMaxH3TestClip()
    pictures = [
        torch.full((1, 2, 2, 3), 1.0),
        torch.full((1, 2, 2, 3), 2.0),
    ]
    frames = torch.full((3, 2, 2, 3), 3.0)

    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        pictures,
        [Fraction(0)],
        "0.0s",
        encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        default_single_visual=True,
        default_video_frames=frames,
        default_video_timestamps=[Fraction(0), Fraction(1, 2), Fraction(1)],
    )

    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [
        entry[0] for entry in entries if encoder_helpers.is_image_token(entry)
    ]
    assert "<Picture 1>: <Picture 2>: <Video 1>: " in text
    assert text.index("<Picture 1>") < text.index("<Picture 2>") < text.index("<Video 1>")
    assert text.index("<Video 1>") < text.index("prompt")
    assert len(visuals) == 4
    assert [float(visuals[index]["data"].mean()) for index in range(2)] == [1.0, 2.0]


def test_minimax_h3_media_tokenization_formats_derived_video_timestamps():
    clip = _MiniMaxH3TestClip()
    frames = torch.ones(3, 2, 2, 3)
    tokens = encoder_helpers.tokenize_minimax_h3_media_prompt(
        clip,
        "prompt",
        [],
        [Fraction(0)],
        "0.0s",
        encoder_helpers.MINIMAX_H3_MEDIA_STRUCTURE,
        default_single_visual=True,
        default_video_frames=frames,
        default_video_timestamps=[Fraction(0), Fraction(1, 2), Fraction(1)],
    )
    entries = tokens["qwen3vl_32b"][0]
    text = "".join(entry[0] for entry in entries if isinstance(entry[0], str))
    visuals = [
        entry[0] for entry in entries if encoder_helpers.is_image_token(entry)
    ]
    assert "<Video 1>: " in text
    assert text.index("<Video 1>") < text.index("prompt")
    assert "<0.3s>" in text
    assert "<1.0s>" in text
    assert len(visuals) == 2
    assert torch.equal(visuals[0]["data"], frames[:2])
    assert torch.equal(visuals[1]["data"], frames[-1:].expand(2, -1, -1, -1))
    assert all(visual["minimax_video_block"] is True for visual in visuals)


@pytest.mark.parametrize(
    "structure, message",
    [
        ("", "must not be empty"),
        ("<<picture>> <<shot>>", "missing <<visual>>"),
        ("<<visual>> <<shot>>", "missing <<picture>>"),
        ("<<time>> <<picture>> <<visual>> <<shot>> <<unknown>>", "Unknown"),
        ("<<time>> <<picture>> <<visual>> <<visual>> <<shot>>", "exactly one <<visual>>"),
    ],
)
def test_minimax_h3_media_structure_validation(structure, message):
    with pytest.raises(ValueError, match=message):
        encoder_helpers._validate_minimax_h3_media_structure(structure)


def test_minimax_h3_media_structure_allows_optional_time_and_shot_labels():
    structure = "<<picture>>: <<visual>>"
    assert encoder_helpers._validate_minimax_h3_media_structure(structure) == structure


@pytest.mark.parametrize(
    ("image_width", "image_height", "generation_width", "generation_height", "mode", "expected"),
    [
        (128, 64, 64, 32, "match", (64, 32)),
        (100, 50, 1024, 1024, "match", (96, 64)),
        (4096, 4096, 64, 64, "max", (2048, 2048)),
        (4096, 3072, 64, 64, "max", (2720, 2048)),
    ],
)
def test_minimax_h3_reference_size_matches_core(
    image_width,
    image_height,
    generation_width,
    generation_height,
    mode,
    expected,
):
    assert encoder_helpers.minimax_h3_reference_size(
        image_width,
        image_height,
        generation_width,
        generation_height,
        mode,
    ) == expected


def test_advanced_minimax_h3_reference_mode_preserves_flat_order_and_pixels():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.25)
    second = torch.full((1, 32, 64, 3), 0.5)
    third = torch.full((1, 64, 32, 3), 0.75)

    conditioning, latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        ref_image_size="match",
        vlm_resolution=0,
        width=64,
        height=32,
        length=5,
        reference_images={
            "reference_image_1": torch.cat([first, second], dim=0),
            "reference_image_2": third,
        },
        visual_fusion_config=None,
        enable_caching="disabled",
    ).args

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    qwen_images = [
        entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)
    ]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    native_call = clip.tokenize_calls[-1]
    assert native_call["images"] is None
    assert native_call["minimax_ref_items"] is not None
    assert native_call["text"] == "subject"
    assert len(vae.images) == 3
    assert [float(image.mean()) for image in vae.images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )

    metadata = conditioning[0][1]
    references = metadata["minimax_refs"]
    assert [item["kind"] for item in references] == ["image", "image", "image"]
    assert [(item["latent_h"], item["latent_w"]) for item in references] == [
        (2, 4),
        (2, 4),
        (4, 2),
    ]
    assert [float(item["latent"].mean()) for item in references] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    assert "minimax_keyframes" not in metadata
    assert "minimax_frame_count" not in metadata
    video, audio = latent["samples"].tensors
    assert video.shape == (1, 24, 2, 2, 4)
    assert audio.shape == (1, 32, 2, 8)


def test_advanced_minimax_h3_reference_fusion_pairs_flattened_inputs():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = [
        torch.full((1, 4, 6, 3), value)
        for value in (0.25, 0.5, 0.75)
    ]
    fusion = [
        torch.full((1, 4, 6, 3), value)
        for value in (1.0, 0.125, 0.375, 0.625)
    ]

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={
            "reference_image_1": torch.cat(references[:2], dim=0),
            "reference_image_2": references[2],
        },
        fusion_images={"fusion_image_1": torch.cat(fusion, dim=0)},
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
        enable_caching="disabled",
    )

    encoded_images = [
        [float(entry[0]["data"].mean()) for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        encoded_images,
        [
            [0.25, 0.5, 0.75],
            [1.0, 0.5, 0.75],
            [0.25, 0.125, 0.75],
            [0.25, 0.5, 0.375],
        ],
        atol=0.004,
    )
    assert [float(image.mean()) for image in vae.images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )


def test_advanced_minimax_h3_reference_fusion_singleton_broadcasts():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = [
        torch.full((1, 4, 6, 3), value)
        for value in (0.25, 0.5, 0.75)
    ]
    fusion = torch.full((1, 4, 6, 3), 1.0)

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={"reference_image_1": torch.cat(references, dim=0)},
        fusion_images={"fusion_image_1": fusion},
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
        enable_caching="disabled",
    )

    encoded_images = [
        [float(entry[0]["data"].mean()) for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        encoded_images,
        [[0.25, 0.5, 0.75], [1.0, 0.5, 0.75], [0.25, 1.0, 0.75], [0.25, 0.5, 1.0]],
        atol=0.004,
    )


def test_advanced_minimax_h3_reference_fusion_second_socket_disables_broadcast():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = torch.stack(
        [
            torch.full((4, 6, 3), 0.25),
            torch.full((4, 6, 3), 0.5),
            torch.full((4, 6, 3), 0.75),
        ]
    )

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={"reference_image_1": references},
        fusion_images={
            "fusion_image_1": torch.full((1, 4, 6, 3), 1.0),
            "fusion_image_2": torch.full((1, 4, 6, 3), 0.125),
        },
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
        enable_caching="disabled",
    )

    encoded_images = [
        [float(entry[0]["data"].mean()) for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        encoded_images,
        [[0.25, 0.5, 0.75], [1.0, 0.5, 0.75], [0.25, 0.125, 0.75]],
        atol=0.004,
    )


def test_advanced_minimax_h3_reference_fusion_off_ignores_fusion_inputs():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    references = torch.stack(
        [
            torch.full((4, 6, 3), 0.25),
            torch.full((4, 6, 3), 0.5),
        ]
    )
    fusion = torch.full((1, 4, 6, 3), 1.0)

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        width=64,
        height=32,
        length=5,
        vlm_resolution=0,
        reference_images={"reference_image_1": references},
        fusion_images={"fusion_image_1": fusion},
        visual_fusion_config=None,
        enable_caching="disabled",
    )

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    qwen_images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.5], abs=0.004
    )
    assert len(clip.encoded_tokens) == 1


def test_advanced_minimax_h3_reference_save_exports_each_visual_span(monkeypatch):
    clip = _MiniMaxH3TestClip()
    clip.cond_stage_model = _MiniMaxH3TestNamespace(clip_name="qwen3vl_8b", clip="qwen3vl_8b")
    clip.tokenizer = _MiniMaxH3TestNamespace(clip_name="qwen3vl_32b")
    vae = _RecordingMiniMaxVAE()
    exported = []
    monkeypatch.setattr(
        encoder_helpers,
        "save_source_visual_embeddings",
        lambda *args: exported.append(args),
    )

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        ref_image_size="match",
        vlm_resolution=0,
        width=64,
        height=32,
        length=5,
        reference_images={
            "reference_image_1": torch.full((1, 32, 64, 3), 0.25),
            "reference_image_2": torch.full((1, 32, 64, 3), 0.5),
        },
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "save_blended_embeds": True,
        },
        enable_caching="disabled",
    )

    assert len(exported) == 1
    _, tokens, _config, key, _device, visual_indices, _cache = exported[0]
    assert key == "qwen3vl_8b"
    assert visual_indices == [0, 1]
    assert sum(encoder_helpers.is_image_token(entry) for entry in tokens["qwen3vl_32b"][0]) == 2


def test_advanced_minimax_h3_none_keeps_frame_pictures_without_vae_keyframes():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.25)
    last = torch.full((1, 32, 64, 3), 0.75)

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        last_frame=last,
        ref_image_size="none",
        width=64,
        height=32,
        length=5,
        enable_caching="disabled",
    ).args

    qwen_images = [
        entry[0]["data"]
        for entry in clip.encoded_tokens[-1]["qwen3vl_32b"][0]
        if encoder_helpers.is_image_token(entry)
    ]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.75], abs=0.004
    )
    assert vae.images == []
    metadata = conditioning[0][1]
    assert "minimax_keyframes" not in metadata
    assert "minimax_frame_count" not in metadata


def test_advanced_minimax_h3_reference_none_is_ordered_vlm_only():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 32, 64, 3), 0.25)
    second = torch.full((1, 32, 64, 3), 0.5)
    third = torch.full((1, 64, 32, 3), 0.75)

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        ref_image_size="none",
        vlm_resolution=256,
        width=64,
        height=32,
        length=5,
        reference_images={
            "reference_image_1": torch.cat([first, second], dim=0),
            "reference_image_2": third,
        },
        enable_caching="disabled",
    ).args

    native_call = clip.tokenize_calls[-1]
    assert native_call["images"] is None
    assert native_call["minimax_ref_items"] is not None
    assert len(native_call["minimax_ref_items"]) == 3
    qwen_images = [item["data"] for item in native_call["minimax_ref_items"]]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.25, 0.5, 0.75], abs=0.004
    )
    assert [image.shape[1:3] for image in qwen_images] == [
        encoder_helpers.vlm_target_dimensions(32, 64, 256),
        encoder_helpers.vlm_target_dimensions(32, 64, 256),
        encoder_helpers.vlm_target_dimensions(64, 32, 256),
    ]
    assert vae.images == []
    metadata = conditioning[0][1]
    assert "minimax_refs" not in metadata
    assert "minimax_keyframes" not in metadata
    assert "minimax_frame_count" not in metadata


def test_advanced_minimax_h3_vlm_resolution_is_independent_for_every_role():
    first = torch.full((1, 16, 32, 3), 0.125)
    reference = torch.full((1, 32, 64, 3), 0.25)
    fusion = torch.full((1, 64, 32, 3), 0.5)

    keyframe_clip = _MiniMaxH3TestClip()
    keyframe_vae = _RecordingMiniMaxVAE()
    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        keyframe_clip,
        keyframe_vae,
        prompt="subject",
        first_frame=first,
        vlm_resolution=256,
        width=64,
        height=32,
        length=5,
        fusion_images={"fusion_image_1": fusion},
        enable_caching="disabled",
    )
    keyframe_images = [
        entry[0]["data"]
        for entry in keyframe_clip.encoded_tokens[-1]["qwen3vl_32b"][0]
        if encoder_helpers.is_image_token(entry)
    ]
    assert [image.shape[1:3] for image in keyframe_images] == [
        encoder_helpers.vlm_target_dimensions(16, 32, 256),
        encoder_helpers.vlm_target_dimensions(64, 32, 256),
    ]
    assert [image.shape[1:3] for image in keyframe_vae.images] == [(32, 64)]

    reference_clip = _MiniMaxH3TestClip()
    reference_vae = _RecordingMiniMaxVAE()
    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        reference_clip,
        reference_vae,
        prompt="subject",
        ref_image_size="match",
        vlm_resolution=256,
        width=64,
        height=32,
        length=5,
        reference_images={"reference_image_1": reference},
        enable_caching="disabled",
    )
    reference_qwen = next(
        entry[0]["data"]
        for entry in reference_clip.encoded_tokens[-1]["qwen3vl_32b"][0]
        if encoder_helpers.is_image_token(entry)
    )
    assert reference_qwen.shape[1:3] == encoder_helpers.vlm_target_dimensions(
        32, 64, 256
    )
    assert [image.shape[1:3] for image in reference_vae.images] == [(32, 64)]


def test_advanced_minimax_h3_rejects_simultaneous_native_modes_before_encoding():
    first = torch.full((1, 32, 64, 3), 0.125)
    reference = torch.full((1, 32, 64, 3), 0.25)

    for kwargs, message in [
        (
            {"first_frame": first, "reference_images": {"reference_image_1": reference}},
            "frame inputs cannot be combined",
        ),
    ]:
        clip = _MiniMaxH3TestClip()
        vae = _RecordingMiniMaxVAE()
        with pytest.raises(ValueError, match=message):
            UC_AdvancedMiniMaxH3ImageToVideo.execute(
                clip, vae, "subject", 64, 32, 5, **kwargs
            , enable_caching="disabled")
        assert clip.encoded_tokens == []
        assert vae.images == []


def test_advanced_minimax_h3_frame_fusion_targets_matching_picture_slots():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    second = torch.full((1, 4, 6, 3), 0.5)
    third = torch.full((1, 4, 6, 3), 0.75)
    fourth = torch.full((1, 4, 6, 3), 1.0)

    output = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        last_frame=second,
        vlm_resolution=0,
        width=64,
        height=32,
        length=22,
        fusion_images={
            "fusion_image_1": torch.cat([third, fourth], dim=0),
        },
        visual_fusion_config={
            "visual_fusion_method": "linear",
            "visual_encoder_path": "grid-deepstack",
        },
        enable_caching="disabled",
    )

    conditioning, latent = output.args
    encoded_images = []
    for tokens in clip.encoded_tokens:
        entries = tokens["qwen3vl_32b"][0]
        images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
        if images:
            encoded_images.append(images)
    assert len(encoded_images) == 3
    assert [len(images) for images in encoded_images] == [2, 2, 2]
    assert np.allclose(
        [[float(image.mean()) for image in images] for images in encoded_images],
        [[0.25, 0.5], [0.75, 0.5], [1.0, 0.5]],
        atol=0.004,
    )
    assert [float(image.mean()) for image in vae.images] == pytest.approx(
        [0.25, 0.5], abs=0.004
    )
    assert encoded_images[0][0].shape[1:3] == (4, 6)
    assert encoded_images[1][0].shape[1:3] == (4, 6)
    assert vae.images[0].shape[1:3] == (32, 64)
    assert vae.images[1].shape[1:3] == (32, 64)
    image_calls = [call for call in clip.tokenize_calls if call["images"]]
    assert image_calls
    assert all(call["minimax_ref_items"] is None for call in image_calls)
    assert all(call["text"] == "subject" for call in image_calls)

    metadata = conditioning[0][1]
    assert metadata["minimax_frame_count"] == 22
    assert [item["resolved_frame_index"] for item in metadata["minimax_keyframes"]] == [0, 21]
    assert metadata["minimax_token_tags"].numel() == conditioning[0][0].shape[1]
    video, audio = latent["samples"].tensors
    assert video.shape == (1, 24, 7, 2, 4)
    assert audio.shape == (1, 32, 2, 37)


def test_advanced_minimax_h3_first_frame_fusion_uses_only_picture_one():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    fusion = torch.full((1, 4, 6, 3), 0.75)

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        width=64,
        height=32,
        length=5,
        fusion_images={"fusion_image_1": fusion},
        visual_fusion_config={"visual_fusion_method": "linear"},
        enable_caching="disabled",
    ).args

    encoded_images = [
        [entry[0]["data"] for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert [[float(image.mean()) for image in images] for images in encoded_images] == [
        [0.25],
        [0.75],
    ]
    assert [item["resolved_frame_index"] for item in conditioning[0][1]["minimax_keyframes"]] == [0]
    assert [float(image.mean()) for image in vae.images] == pytest.approx([0.25], abs=0.004)


def test_advanced_minimax_h3_fusion_batches_stay_on_their_socket_slots():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    last = torch.full((1, 4, 6, 3), 0.5)
    first_fusion = torch.cat(
        [
            torch.full((1, 4, 6, 3), 0.75),
            torch.full((1, 4, 6, 3), 1.0),
        ]
    )
    last_fusion = torch.cat(
        [
            torch.full((1, 4, 6, 3), 0.125),
            torch.full((1, 4, 6, 3), 0.375),
        ]
    )

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        last_frame=last,
        width=64,
        height=32,
        length=5,
        fusion_images={
            "fusion_image_1": first_fusion,
            "fusion_image_2": last_fusion,
        },
        visual_fusion_config={"visual_fusion_method": "linear"},
        enable_caching="disabled",
    )

    encoded_images = [
        [entry[0]["data"] for entry in tokens["qwen3vl_32b"][0] if encoder_helpers.is_image_token(entry)]
        for tokens in clip.encoded_tokens
    ]
    assert np.allclose(
        [[float(image.mean()) for image in images] for images in encoded_images],
        [
            [0.25, 0.5],
            [0.75, 0.5],
            [1.0, 0.5],
            [0.25, 0.125],
            [0.25, 0.375],
        ],
        atol=0.004,
    )
    assert [float(image.mean()) for image in vae.images] == pytest.approx([0.25, 0.5], abs=0.004)


def test_advanced_minimax_h3_rejects_unpaired_frame_fusion_input():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    first = torch.full((1, 4, 6, 3), 0.25)
    fusion = torch.full((1, 4, 6, 3), 0.75)

    with pytest.raises(ValueError, match="without a matching picture slot"):
        UC_AdvancedMiniMaxH3ImageToVideo.execute(
            clip,
            vae,
            prompt="subject",
            first_frame=first,
            width=64,
            height=32,
            length=5,
            fusion_images={
                "fusion_image_1": fusion,
                "fusion_image_2": fusion,
            },
            visual_fusion_config={"visual_fusion_method": "linear"},
            enable_caching="disabled",
        )
    assert clip.encoded_tokens == []
    assert vae.images == []


def test_advanced_minimax_h3_fusion_off_keeps_all_images_as_separate_pictures():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()
    images = torch.stack(
        [
            torch.full((3, 5, 3), 0.25),
            torch.full((3, 5, 3), 0.5),
            torch.full((3, 5, 3), 0.75),
        ]
    )

    first = torch.full((1, 3, 5, 3), 0.125)
    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        prompt="subject",
        first_frame=first,
        width=64,
        height=32,
        length=5,
        fusion_images={"fusion_image_1": images},
        visual_fusion_config=None,
        enable_caching="disabled",
    ).args

    entries = clip.encoded_tokens[-1]["qwen3vl_32b"][0]
    qwen_images = [entry[0]["data"] for entry in entries if encoder_helpers.is_image_token(entry)]
    assert [float(image.mean()) for image in qwen_images] == pytest.approx(
        [0.125, 0.25, 0.5, 0.75], abs=0.004
    )
    assert len(vae.images) == 1
    assert float(vae.images[0].mean()) == pytest.approx(0.125, abs=0.004)
    assert conditioning[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 0
    native_call = clip.tokenize_calls[-1]
    assert native_call["images"] is not None
    assert native_call["minimax_ref_items"] is None
    assert native_call["text"] == "subject"


def test_advanced_minimax_h3_keeps_placeholder_like_text_raw(monkeypatch):
    def reject_generic_placeholder_path(*_args, **_kwargs):
        raise AssertionError("Dedicated H3 execution used generic placeholder handling.")

    monkeypatch.setattr(
        encoder_helpers,
        "prepare_image_placeholder_prompt",
        reject_generic_placeholder_path,
    )
    clip = _MiniMaxH3TestClip()
    image = torch.ones(1, 4, 4, 3)

    UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        _RecordingMiniMaxVAE(),
        prompt="image_input_fusion image_input_1 image_input_2",
        first_frame=image,
        width=64,
        height=32,
        length=5,
        enable_caching="disabled",
    )

    native_call = clip.tokenize_calls[-1]
    assert native_call["text"] == "image_input_fusion image_input_1 image_input_2"
    assert native_call["images"] is not None


def test_advanced_minimax_h3_validates_encoder():
    class WrongClip:
        tokenizer = types.SimpleNamespace(clip_name="qwen3vl_8b")

    with pytest.raises(ValueError, match="qwen3vl_32b"):
        UC_AdvancedMiniMaxH3ImageToVideo.execute(
            WrongClip(),
            _RecordingMiniMaxVAE(),
            "prompt",
            64,
            32,
            5,
            enable_caching="disabled",
        )


def test_advanced_minimax_h3_accepts_text_only_and_last_only():
    clip = _MiniMaxH3TestClip()
    vae = _RecordingMiniMaxVAE()

    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip, vae, "subject", 64, 32, 5, multiplier=2.0
    , enable_caching="disabled").args

    assert vae.images == []
    assert torch.all(conditioning[0][0] == 2.0)
    assert "minimax_keyframes" not in conditioning[0][1]
    assert "minimax_refs" not in conditioning[0][1]
    assert clip.tokenize_calls[-1]["images"] == []

    last = torch.full((1, 32, 64, 3), 0.75)
    conditioning, _latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(
        clip,
        vae,
        "subject",
        64,
        32,
        5,
        last_frame=last,
        enable_caching="disabled",
    ).args

    assert [item["resolved_frame_index"] for item in conditioning[0][1]["minimax_keyframes"]] == [4]
    assert [float(image.mean()) for image in clip.tokenize_calls[-1]["images"]] == pytest.approx([0.75], abs=0.004)


@pytest.mark.parametrize("prompt", ["subject", "(subject:2)", ""])
@pytest.mark.parametrize("media", ["text", "picture", "video"])
def test_minimax_h3_layout_marks_actual_prompt_suffix(prompt, media):
    clip = _MiniMaxH3TestClip()
    kwargs = {"ref_image_size": "none", "vlm_resolution": 0, "vlm_video_resolution": 0}
    if media == "picture":
        kwargs["first_frame"] = torch.zeros(1, 32, 64, 3)
    elif media == "video":
        kwargs["video"] = torch.zeros(5, 32, 64, 3)
    conditioning, _ = UC_AdvancedMiniMaxH3ImageToVideo.execute(clip, None, prompt, 64, 32, 5, **kwargs, enable_caching="disabled").args
    tensor, metadata = conditioning[0]
    layout = metadata["uc_minimax_h3_vlm_layout"]
    expected = 0 if media == "text" else tensor.shape[1] - int(bool(prompt))
    assert layout == {"version": 1, "sequence_length": tensor.shape[1], "prompt_start": expected}
    assert len(clip.encoded_tokens) == 1
    assert clip.encoded_tokens[0]["qwen3vl_32b"][0][-1][0] == ("subject" if prompt else (151643 if media == "text" else 151653))


def test_minimax_h3_guide_integration_preserves_conditioning_and_native_timestamp():
    clip = _MiniMaxH3TestClip()
    base, _ = UC_AdvancedMiniMaxH3ImageToVideo.execute(clip, None, "subject", 64, 32, 5, ref_image_size="none", enable_caching="disabled").args
    result = encoder_nodes.UC_MiniMaxH3VLMGuide.execute(base, clip, torch.zeros(1, 32, 64, 3), 1.25, 0).args[0]
    assert torch.equal(result[0][0][:, -1:], base[0][0])
    assert base[0][1]["uc_minimax_h3_vlm_layout"]["prompt_start"] == 0
    assert result[0][1]["uc_minimax_h3_vlm_layout"]["prompt_start"] == result[0][0].shape[1] - 1
    assert clip.encoded_tokens[-1]["qwen3vl_32b"][0][0][0] == "<1.2 seconds>"
    assert len(clip.encoded_tokens) == 2


def test_minimax_h3_temporal_media_fields_are_additive_and_standard_ignores_them():
    config = encoder_helpers.build_minimax_h3_media_config(None, temporal_density=[4], temporal_fusion_method=["spatial"])
    assert config["temporal_density"] == 4
    assert config["temporal_fusion_method"] == "spatial"
    legacy = {key: value for key, value in config.items() if not key.startswith("temporal_")}
    image = torch.zeros(1, 32, 64, 3)
    outputs = []
    for payload in (config, legacy):
        clip = _MiniMaxH3TestClip()
        outputs.append(UC_AdvancedMiniMaxH3ImageToVideo.execute(clip, None, "subject", 64, 32, 5, first_frame=image, ref_image_size="none", vlm_resolution=0, media_config=payload, enable_caching="disabled").args[0])
        assert len(clip.encoded_tokens) == 1
    assert torch.equal(outputs[0][0][0], outputs[1][0][0])
    assert outputs[0][0][1]["uc_minimax_h3_vlm_layout"] == outputs[1][0][1]["uc_minimax_h3_vlm_layout"]


@pytest.mark.parametrize("node_name", ["UC_AdvMiniMaxH3ImageToVideoTemporalFusion", "UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion"])
@pytest.mark.parametrize("bypass", ["density_one", "no_video", "consensus_off", "spatial_off"])
def test_temporal_nodes_bypass_without_alternative_encoding(monkeypatch, node_name, bypass):
    def unexpected(*args, **kwargs):
        raise AssertionError("Bypass must not enter temporal lane encoding")
    monkeypatch.setattr(encoder_helpers, "encode_temporal_conditioning", unexpected)
    node = getattr(encoder_nodes, node_name)
    kwargs = {"ref_image_size": "none", "vlm_video_resolution": 0}
    config = encoder_helpers.build_minimax_h3_media_config(None, video_latent_mode="off", temporal_density=1 if bypass == "density_one" else 3, temporal_fusion_method="spatial" if bypass == "spatial_off" else "consensus")
    if bypass != "no_video":
        kwargs.update(video=torch.zeros(25, 32, 64, 3), media_config=config)
    if bypass == "consensus_off":
        kwargs["text_blend_config"] = {"blend_preset": "off"}
    if bypass == "spatial_off":
        kwargs["visual_fusion_config"] = {"visual_fusion_method": "off"}
    clip = _MiniMaxH3TestClip()
    result, latent = node.execute(clip, None, "subject", 64, 32, 25, **kwargs, enable_caching="disabled").args
    assert len(clip.encoded_tokens) == 1
    ordinary = dict(kwargs)
    ordinary.pop("text_blend_config", None)
    expected, expected_latent = UC_AdvancedMiniMaxH3ImageToVideo.execute(_MiniMaxH3TestClip(), None, "subject", 64, 32, 25, **ordinary, enable_caching="disabled").args
    assert torch.equal(result[0][0], expected[0][0])
    assert result[0][1]["uc_minimax_h3_vlm_layout"] == expected[0][1]["uc_minimax_h3_vlm_layout"]
    assert latent.keys() == expected_latent.keys()


def test_temporal_post_node_fuses_only_video_interiors_and_keeps_budget():
    class VideoClip(_MiniMaxH3TestClip):
        def encode_from_tokens_scheduled(self, tokens):
            output = super().encode_from_tokens_scheduled(tokens)
            tensor = output[0][0]
            entries = tokens["qwen3vl_32b"][0]
            spans = encoder_helpers.build_token_to_conditioning_map(entries, tensor)
            for entry, (start, end) in zip(entries, spans):
                if isinstance(entry[0], dict) and entry[0].get("minimax_video_block"):
                    tensor[:, start:end] = float(entry[0]["data"].mean())
            return output

    video = torch.arange(25, dtype=torch.float32)[:, None, None, None].expand(25, 32, 64, 3) / 25
    config = encoder_helpers.build_minimax_h3_media_config(None, video_latent_mode="off", temporal_density=2)
    clip = VideoClip()
    conditioning, _ = encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTemporalFusion.execute(
        clip, None, "(subject:2)", 64, 32, 25, video=video, media_config=config,
        ref_image_size="none", vlm_video_resolution=0,
        text_blend_config={"blend_preset": "custom", "blend_method": "linear", "global_scale": 1.0},
        enable_caching="disabled",
    ).args
    tensor, metadata = conditioning[0]
    assert len(clip.encoded_tokens) == 2
    canonical, alternate = [value["qwen3vl_32b"][0] for value in clip.encoded_tokens]
    spans = encoder_helpers.build_token_to_conditioning_map(canonical, tensor)
    count = 0
    for entry, alt, (start, end) in zip(canonical, alternate, spans):
        if isinstance(entry[0], dict) and entry[0].get("minimax_video_block"):
            expected = (float(entry[0]["data"].mean()) + float(alt[0]["data"].mean())) / 2
            assert torch.allclose(tensor[:, start:end], torch.full_like(tensor[:, start:end], expected))
            count += 1
        else:
            assert entry == alt
            assert torch.all(tensor[:, start:end] == (2.0 if entry[0] == "subject" else 1.0))
    # Native H3 preparation trims 25 input frames to 22 (17n+5), yielding one pair.
    assert count == 1
    assert metadata["uc_minimax_h3_vlm_layout"]["prompt_start"] == tensor.shape[1] - 1
    assert "minimax_refs" not in metadata and "minimax_keyframes" not in metadata


@pytest.mark.parametrize("variant", ["disabled", "token", "temporal_token"])
def test_temporal_pre_node_uses_actual_preprocessed_encode_and_deepstack(variant, tmp_path):
    process_calls, qwen_calls = [], []

    class Transformer:
        model_type = "qwen3vl_32b"

        def __call__(self, _ids, _mask, **kwargs):
            qwen_calls.append(kwargs)
            return kwargs["embeds"], None, None

    class Model:
        transformer = Transformer()
        enable_attention_masks = False
        layer = "last"
        layer_idx = None
        layer_norm_hidden_state = False
        zero_out_masked = False
        return_projected_pooled = True
        return_attention_masks = False

        def process_tokens(self, rows, device):
            process_calls.append(rows)
            vectors, info = [], []
            for value in rows[0]:
                if isinstance(value, dict):
                    pixel = float(value["data"].mean())
                    info.append({"type": "image", "index": len(vectors), "size": 4,
                                 "extra": {"grid": torch.tensor([[1, 4, 4]]),
                                           "deepstack": [torch.full((4, 2), pixel * 10)]}})
                    vectors.extend([[pixel] * 4] * 4)
                else:
                    vectors.append([2. if value == "first" else 3. if value == "second" else 1.] * 4)
            tensor = torch.tensor([vectors])
            return tensor, torch.ones(1, len(vectors)), [len(vectors)], info

    clip = _MiniMaxH3TestClip()
    clip.cond_stage_model = _MiniMaxH3TestNamespace(clip_name="qwen3vl_32b", clip="clip_model", clip_model=Model(), reset_clip_options=lambda: None, set_clip_options=lambda _: None)
    clip.layer_idx = None
    clip.load_model = lambda _: None
    clip.patcher = _MiniMaxH3TestPatcher()
    clip.add_hooks_to_dict = lambda _: None
    video = torch.arange(5, dtype=torch.float32)[:, None, None, None].expand(5, 64, 64, 3) / 5
    config = encoder_helpers.build_minimax_h3_media_config(None, video_latent_mode="off", temporal_density=2)
    if variant != "disabled":
        source = torch.full((1, 64, 64, 3), .8)
        alternative = torch.full((1, 64, 64, 3), .2)
        kwargs = dict(reference_images={"reference_image_1": source}, ref_image_size="none",
                      vlm_resolution=0, vlm_video_resolution=0, enable_caching="all")
        if variant == "token":
            kwargs.update(token_fusion=True, fusion_images={"fusion_image_1": alternative},
                          visual_fusion_config={"visual_fusion_method": "spatial-checkerboard"})
        else:
            kwargs.update(video=video, media_config=config, temporal_fusion=True, temporal_token_fusion=True)
        def execute(prompt):
            return encoder_helpers.execute_advanced_minimax_h3_image_to_video(clip, None, prompt, 64, 64, 5, **kwargs)
        execute("first")
        counts = (len(process_calls), len(qwen_calls))
        assert counts[0] >= 2 and counts[1] == 1
        execute("first")
        assert len(qwen_calls) == counts[1]
        execute("second")
        assert len(qwen_calls) == counts[1] + 1
        assert list(tmp_path.glob("utilscollection_h3_encoder_cache/v2/encoded_section/*.safetensors"))
        (alternative if variant == "token" else video)[0].add_(.1)
        execute("second")
        assert len(qwen_calls) == counts[1] + 2
        return
    conditioning, _ = encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion.execute(
        clip, None, "(subject:2)", 64, 64, 5, video=video, media_config=config,
        reference_images={"reference_image_1": torch.full((1, 64, 64, 3), .8)},
        ref_image_size="none", vlm_resolution=0, vlm_video_resolution=0,
        text_blend_config={"blend_preset": "custom", "blend_method": "linear"},
        enable_caching="disabled",
    ).args
    assert len(process_calls) == 2 and len(qwen_calls) == 1
    assert not clip.encoded_tokens
    info = qwen_calls[0]["embeds_info"]
    tensor, metadata = conditioning[0]
    for entry, expected in zip(info, (.8, .3)):
        start = entry["index"]
        assert torch.allclose(tensor[:, start:start + 4], torch.full((1, 4, 4), expected))
        assert torch.allclose(entry["extra"]["deepstack"][0], torch.full((4, 2), expected * 10))
    assert torch.all(tensor[:, -1] == 2.)
    assert metadata["uc_minimax_h3_vlm_layout"]["prompt_start"] == tensor.shape[1] - 1
    assert metadata["minimax_token_tags"].numel() == tensor.shape[1]


def test_temporal_node_schemas_keep_standard_sockets_except_picture_fusion():
    standard = encoder_nodes.UC_AdvancedMiniMaxH3ImageToVideo.GET_SCHEMA()
    assert not standard.is_experimental
    assert not encoder_nodes.UC_AdvancedMiniMaxH3ImageToVideo.EXPERIMENTAL
    assert not encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTokenFusion.GET_SCHEMA().is_experimental
    expected = [value.id for value in standard.inputs if value.id != "fusion_images"] + ["text_blend_config"]
    for node in (encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTemporalFusion, encoder_nodes.UC_AdvMiniMaxH3ImageToVideoTemporalTokenFusion):
        schema = node.GET_SCHEMA()
        assert schema.is_experimental
        assert node.EXPERIMENTAL
        assert [value.id for value in schema.inputs] == expected
        assert [value.io_type for value in schema.outputs] == ["CONDITIONING", "LATENT"]
        assert "model" not in expected


def test_embedding_output_cannot_escape_root(tmp_path):
    nested = encoder_helpers.resolve_embedding_output_path(str(tmp_path), "nested/item.safetensors")
    assert pathlib.Path(nested).is_relative_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        encoder_helpers.resolve_embedding_output_path(str(tmp_path), "../outside.safetensors")
    with pytest.raises(ValueError, match="relative"):
        encoder_helpers.resolve_embedding_output_path(str(tmp_path), str(tmp_path / "absolute.safetensors"))


def test_krea2_mapping_mirrors_core_prefix_strip():
    tokens = [
        (np.int64(151644), 1.0), (8948, 1.0), (198, 1.0), (42, 1.0), (151645, 1.0),
        (np.int64(151644), 1.0), (872, 1.0), (198, 1.0), (100, 1.0), (101, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 3, 12 * 2560)
    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)
    assert mapping[:8] == [(-1, -1)] * 8
    assert mapping[8:] == [(0, 1), (1, 2), (2, 3)]


def test_token_mapping_expands_prompt_embedding_entries():
    embedding = torch.zeros(3, 5120)
    tokens = [(100, 1.0), (embedding, 1.0), (101, 1.0)]
    conditioning = torch.zeros(1, 5, 5120)

    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)

    assert encoder_helpers.is_image_token(tokens[1]) is False
    assert mapping == [(0, 1), (1, 4), (4, 5)]


def test_mage_flow_mapping_mirrors_core_prefix_strip():
    image = torch.zeros(1, 832, 1248, 3)
    tokens = [(151644, 1.0), (8948, 1.0), (198, 1.0)]
    tokens.extend((token_id, 1.0) for token_id in range(28))
    tokens.extend([
        (np.int64(151644), 1.0), (872, 1.0), (198, 1.0), (74785, 1.0),
        ({"type": "image", "data": image}, 1.0),
        (100, 1.0), (101, 1.0), (102, 1.0), (103, 1.0), (104, 1.0), (151645, 1.0),
    ])
    conditioning = torch.zeros(1, 1021, 2560)

    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)

    assert mapping[:34] == [(-1, -1)] * 34
    assert mapping[34] == (0, 1)
    assert mapping[35] == (1, 1015)
    assert mapping[-1] == (1020, 1021)


def test_krea2_mapping_mirrors_custom_system_prefix_strip():
    image = torch.zeros(1, 32, 32, 3)
    tokens = [
        (151644, 1.0), (872, 1.0), (198, 1.0), (151645, 1.0), (198, 1.0),
        (151644, 1.0), (8948, 1.0), ({"type": "image", "data": image}, 1.0),
        (200, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 8, 12 * 2560)

    mapping = encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)

    assert mapping[:5] == [(-1, -1)] * 5
    assert mapping[5:] == [(0, 1), (1, 2), (2, 6), (6, 7), (7, 8)]


def test_krea2_mapping_rejects_unexplained_length_mismatch():
    image = torch.zeros(1, 32, 32, 3)
    tokens = [
        (151644, 1.0), (872, 1.0), (198, 1.0), (151645, 1.0), (198, 1.0),
        (151644, 1.0), (8948, 1.0), ({"type": "image", "data": image}, 1.0),
        (200, 1.0), (151645, 1.0),
    ]
    conditioning = torch.zeros(1, 6, 12 * 2560)

    with pytest.raises(ValueError, match="refusing to guess a visual range"):
        encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)


def test_minimax_visual_range_uses_core_modality_tags():
    image = torch.zeros(1, 800, 832, 3)
    tokens = {
        "qwen3vl_32b": [[
            (100, 1.0),
            (151652, 1.0),
            ({"type": "image", "data": image}, 1.0),
            (151653, 1.0),
            (101, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 664, 5120)
    tags = torch.ones(664, dtype=torch.long)
    tags[5:657] = 0

    assert encoder_helpers.find_visual_token_range(
        tokens, conditioning, minimax_token_tags=tags
    ) == (6, 656)


def test_minimax_visual_range_selects_one_numbered_block():
    first = torch.zeros(1, 32, 32, 3)
    second = torch.zeros(1, 32, 64, 3)
    tokens = {
        "qwen3vl_32b": [[
            (100, 1.0),
            (151652, 1.0),
            ({"type": "image", "data": first}, 1.0),
            (151653, 1.0),
            (101, 1.0),
            (151652, 1.0),
            ({"type": "image", "data": second}, 1.0),
            (151653, 1.0),
            (102, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 20, 5120)
    tags = torch.ones(20, dtype=torch.long)
    tags[2:6] = 0
    tags[10:15] = 0

    assert encoder_helpers.find_visual_token_range(
        tokens,
        conditioning,
        minimax_token_tags=tags,
        minimax_visual_index=1,
    ) == (11, 14)
    with pytest.raises(ValueError, match="requires a selected visual block"):
        encoder_helpers.find_visual_token_range(
            tokens, conditioning, minimax_token_tags=tags
        )


def test_minimax_visual_range_rejects_tag_run_count_mismatch():
    image = torch.zeros(1, 32, 32, 3)
    tokens = {
        "qwen3vl_32b": [[
            (151652, 1.0),
            ({"type": "image", "data": image}, 1.0),
            (151653, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 12, 5120)
    tags = torch.ones(12, dtype=torch.long)
    tags[1:4] = 0
    tags[7:10] = 0

    with pytest.raises(ValueError, match="numbered visual blocks"):
        encoder_helpers.find_visual_token_range(
            tokens, conditioning, minimax_token_tags=tags
        )


def test_legacy_flat_visual_range_preserves_pre_refactor_spatial_mapping():
    image = torch.zeros(1, 128, 128, 3)
    tokens = {
        "qwen3vl_4b": [[
            (151644, 1.0), (872, 1.0), (198, 1.0), (151645, 1.0), (198, 1.0),
            (151644, 1.0), (8948, 1.0), ({"type": "image", "data": image}, 1.0),
            (200, 1.0), (151645, 1.0),
        ]],
    }
    conditioning = torch.zeros(1, 20, 12 * 2560)

    assert encoder_helpers.find_visual_token_range(
        tokens,
        conditioning,
        legacy_krea_spatial=True,
    ) == (7, 18)


def test_legacy_flat_fusion_layout_uses_retained_visual_span():
    image = torch.zeros(1, 896, 1184, 3)
    assert encoder_helpers.qwen3vl_visual_grid(image) == (28, 37)
    assert encoder_helpers.visual_fusion_grid(image, 1002, legacy_flat=True) == (1, 1002)
    with pytest.raises(ValueError, match="does not match range length"):
        encoder_helpers.visual_fusion_grid(image, 1002)


def test_unknown_visual_expansion_is_rejected_when_length_has_no_solution():
    tokens = [({"type": "image"}, 1.0), (10, 1.0), ({"type": "image"}, 1.0)]
    conditioning = torch.zeros(1, 8, 16)
    with pytest.raises(ValueError, match="no usable Qwen3-VL tensor payload"):
        encoder_helpers.build_token_to_conditioning_map(tokens, conditioning)


def test_klein_visual_range_ignores_core_tail_padding():
    image = torch.zeros(1, 32, 32, 3)
    tokens = {
        "qwen3_4b": [[
            (151652, 1.0),
            ({"type": "image", "data": image}, 1.0),
            (151653, 1.0),
            (151652, 1.0),
            (151655, 1.0),
            (151653, 1.0),
            (10, 1.0),
        ]]
    }
    conditioning = torch.zeros(1, 512, 16)

    assert encoder_helpers.find_visual_token_range(tokens, conditioning) == (1, 5)


def test_klein_vl_detection_does_not_match_z_image_tokenizer():
    klein_type = type(
        "KleinVLTokenizer", (), {"__module__": "comfy.text_encoders.flux"}
    )
    z_image_type = type(
        "ZImageTokenizer", (), {"__module__": "comfy.text_encoders.z_image"}
    )

    assert encoder_helpers.is_klein_vl_text_encoder(
        types.SimpleNamespace(tokenizer=klein_type())
    )
    assert not encoder_helpers.is_klein_vl_text_encoder(
        types.SimpleNamespace(tokenizer=z_image_type())
    )


def test_consensus_off_returns_reference_and_fractional_weights_stay_finite():
    first = torch.tensor([[[1.0, 0.0]]])
    second = torch.tensor([[[-1.0, 0.0]]])
    off, _ = encoder_helpers.blend_text_vectors({"a": first, "b": second}, {"blend_preset": "off"})
    assert off is first
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {
            "blend_preset": "custom",
            "blend_method": "consensus",
            "consensus_type": "mean",
            "alignment_method": "index",
            "power_alpha": 1.5,
            "similarity_threshold": -1.0,
        },
    )
    assert torch.isfinite(blended).all()


def test_consensus_blend_restores_sequence_and_pooled_reference_dtype():
    sequences = {
        "a": torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
        "b": torch.tensor([[[0.0, 1.0]]]),
    }
    pooled = {
        "a": torch.tensor([[1.0, 0.0]], dtype=torch.float16),
        "b": torch.tensor([[0.0, 1.0]]),
    }

    blended, blended_pooled = encoder_helpers.blend_text_vectors(
        sequences,
        {"blend_preset": "baseline"},
        pooled_tensors=pooled,
        device=sequences["a"].device,
        compute_dtype=torch.float32,
    )

    assert blended.device == sequences["a"].device
    assert blended.dtype == sequences["a"].dtype
    assert blended_pooled.device == pooled["a"].device
    assert blended_pooled.dtype == pooled["a"].dtype


def test_common_prefix_is_preserved_before_power_blend():
    prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = torch.cat([prefix, torch.tensor([[1.0, 0.0], [0.0, 1.0]])])[None]
    second = torch.cat([prefix, torch.tensor([[0.8, 0.2], [0.2, 0.8]])])[None]
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {"blend_preset": "power_blend", "preserve_common_prefix": True},
        device=first.device,
        compute_dtype=torch.float32,
    )
    assert torch.equal(blended[0, :2], prefix)
    assert not torch.equal(blended[0, 2:], first[0, 2:])


def test_common_prefix_is_not_scaled_in_linear_mode():
    first = torch.tensor([[[2.0, 3.0], [1.0, 0.0]]])
    second = torch.tensor([[[2.0, 3.0], [3.0, 2.0]]])
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {"blend_preset": "custom", "blend_method": "linear", "global_scale": 2.0,
         "preserve_common_prefix": True},
        device=first.device,
        compute_dtype=torch.float32,
    )
    assert torch.equal(blended[0, 0], first[0, 0])
    assert torch.equal(blended[0, 1], torch.tensor([4.0, 2.0]))


def test_position_bias_prefers_nearby_normalized_positions():
    similarities = torch.tensor([[0.8, 0.9], [0.9, 0.8]])
    unbiased = encoder_helpers._position_biased_similarity_scores(similarities, 0.0)
    biased = encoder_helpers._position_biased_similarity_scores(similarities, 1.0)
    assert torch.equal(unbiased, similarities)
    assert biased[0, 0] > biased[0, 1]
    assert biased[1, 1] > biased[1, 0]


def test_position_bias_does_not_bypass_cosine_alignment_threshold():
    first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    second = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    blended, _ = encoder_helpers.blend_text_vectors(
        {"a": first, "b": second},
        {"blend_preset": "custom", "blend_method": "consensus", "consensus_type": "mean",
         "alignment_method": "similarity", "alignment_threshold": 0.9,
         "similarity_threshold": -1.0, "power_alpha": 1.0, "position_weight": 1.0,
         "rescale_norm": False, "global_scale": 1.0},
        device=first.device,
        compute_dtype=torch.float32,
    )
    assert torch.equal(blended, first)


def test_zero_position_weight_matches_legacy_similarity_alignment():
    sequences = {
        "a": torch.tensor([[[1.0, 0.0], [0.2, 0.8]]]),
        "b": torch.tensor([[[0.9, 0.1], [0.0, 1.0]]]),
    }
    config = {"blend_preset": "baseline"}
    legacy, _ = encoder_helpers.blend_text_vectors(sequences, config)
    explicit_zero, _ = encoder_helpers.blend_text_vectors(sequences, {**config, "position_weight": 0.0})
    assert torch.equal(legacy, explicit_zero)


def test_position_weight_validation_rejects_out_of_range_values():
    sequences = {"a": torch.ones(1, 1, 2), "b": torch.ones(1, 1, 2)}
    with pytest.raises(ValueError, match="Position weight"):
        encoder_helpers.blend_text_vectors(sequences, {"blend_preset": "baseline", "position_weight": 1.1})


def test_consensus_node_passes_original_tensors_to_blender(monkeypatch):
    first = torch.ones(1, 2, 3, dtype=torch.float64)
    second = torch.zeros(1, 2, 3)
    first_pooled = torch.ones(1, 3, dtype=torch.float16)
    seen = {}

    def fake_blend(sequence_tensors, config, pooled_tensors, device, compute_dtype):
        seen["sequence"] = sequence_tensors["a"]
        seen["pooled"] = pooled_tensors["a"]
        return sequence_tensors["a"], pooled_tensors["a"]

    monkeypatch.setattr(encoder_helpers.comfy.model_management, "get_torch_device", lambda: first.device)
    monkeypatch.setattr(encoder_helpers.comfy.model_management, "intermediate_dtype", lambda: torch.float32)
    monkeypatch.setattr("utils_collection_encoder_test.encoder_nodes.blend_text_vectors", fake_blend)

    output = UC_ConditioningConsensusBlend.execute(
        {
            "conditioning_1": [[first, {"pooled_output": first_pooled}]],
            "conditioning_2": [[second, {"pooled_output": torch.zeros(1, 3)}]],
        },
        {"blend_preset": "baseline"},
    ).result[0]

    assert seen["sequence"] is first
    assert seen["pooled"] is first_pooled
    assert output[0][0] is first
    assert output[0][1]["pooled_output"] is first_pooled


@pytest.mark.skipif(
    encoder_helpers.comfy.model_management.is_device_cpu(
        encoder_helpers.comfy.model_management.get_torch_device()
    ),
    reason="No accelerator backend is selected",
)
def test_consensus_accelerator_compute_does_not_change_cpu_output_placement():
    sequences = {"a": torch.ones(1, 2, 3), "b": torch.zeros(1, 2, 3)}
    pooled = {"a": torch.ones(1, 3), "b": torch.zeros(1, 3)}
    compute_device = encoder_helpers.comfy.model_management.get_torch_device()
    compute_dtype = encoder_helpers.comfy.model_management.intermediate_dtype()

    blended, blended_pooled = encoder_helpers.blend_text_vectors(
        sequences,
        {"blend_preset": "baseline"},
        pooled_tensors=pooled,
        device=compute_device,
        compute_dtype=compute_dtype,
    )

    assert blended.device == sequences["a"].device
    assert blended_pooled.device == pooled["a"].device


def test_contextual_weighting_does_not_scale_pooled_output():
    class Clip:
        @staticmethod
        def tokenize(text, **kwargs):
            return {"fake": [[(ord(char), 1.0) for char in text]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            length = len(tokens["fake"][0])
            sequence = torch.ones(1, length, 2)
            pooled = torch.full((1, 2), 7.0)
            return [[sequence, {"pooled_output": pooled}]]

    conditioning = encoder_helpers.encode_embedding_classical_scaled_bias(Clip(), "(ab:2)c")
    sequence, metadata = conditioning[0]
    assert torch.equal(sequence[0, :2], torch.full((2, 2), 2.0))
    assert torch.equal(sequence[0, 2:], torch.ones(1, 2))
    assert torch.equal(metadata["pooled_output"], torch.full((1, 2), 7.0))


def test_contextual_weight_syntax_clean_text_matches_encoder_input():
    assert encoder_helpers.strip_contextual_weight_syntax("a (painting:-1) and ((light:2):0.5)") == "a painting and light"


def test_contextual_weight_syntax_preserves_backslash_escaped_parentheses():
    assert encoder_helpers.strip_contextual_weight_syntax(r"pop culture \(Overwatch\) and \(banana\)") == (
        "pop culture (Overwatch) and (banana)"
    )


def test_advanced_visual_text_only_path_preserves_custom_system_prompt():
    class Clip:
        tokenized_text = None

        @classmethod
        def tokenize(cls, text, **kwargs):
            cls.tokenized_text = text
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            return [[torch.ones(1, 1, 1), {}]]

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="subject",
        system_prompt="custom rules",
        vlm_resolution=384,
        image_inputs={},
    )

    assert Clip.tokenized_text.startswith("<|im_start|>user\n<|im_end|>\n<|im_start|>system\ncustom rules")
    assert "<|im_start|>user\nsubject<|im_end|>" in Clip.tokenized_text

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="subject",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={},
    )

    assert Clip.tokenized_text.startswith("<|im_start|>system\nDescribe the image")
    assert not Clip.tokenized_text.startswith("<|im_start|>user\n<|im_end|>")


def test_advanced_visual_image_only_path_uses_anti_stripping_template(monkeypatch):
    class Clip:
        tokenized_text = None

        @classmethod
        def tokenize(cls, text, **kwargs):
            cls.tokenized_text = text
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(tokens):
            return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(
        encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image
    )

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.ones(1, 2, 2, 3)},
    )

    assert Clip.tokenized_text.startswith(
        "<|im_start|>user\n<|im_end|>\n<|im_start|>system\n<|im_end|>"
    )
    assert "Describe the image by detailing" not in Clip.tokenized_text
    assert "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" in Clip.tokenized_text


def test_advanced_visual_semantic_anchor_numbers_unfused_inputs(monkeypatch):
    encoded_prompts = []

    def encode(_clip, prompt, **_kwargs):
        encoded_prompts.append(prompt)
        return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)
    monkeypatch.setattr(encoder_nodes, "encode_embedding_classical_scaled_bias", encode)

    UC_AdvancedVisualConditioningEncode.execute(
        object(),
        prompt="describe",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={
            "image_1": torch.zeros(1, 2, 2, 3),
            "image_2": torch.ones(1, 2, 2, 3),
        },
        semantic_anchor=True,
    )

    assert f"<Picture 1>: {encoder_nodes.VISION_BLOCK}" in encoded_prompts[0]
    assert f"<Picture 2>: {encoder_nodes.VISION_BLOCK}" in encoded_prompts[1]


def test_advanced_visual_semantic_anchor_is_disabled_by_default(monkeypatch):
    encoded_prompts = []

    def encode(_clip, prompt, **_kwargs):
        encoded_prompts.append(prompt)
        return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)
    monkeypatch.setattr(encoder_nodes, "encode_embedding_classical_scaled_bias", encode)

    UC_AdvancedVisualConditioningEncode.execute(
        object(),
        prompt="describe",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={"image_1": torch.ones(1, 2, 2, 3)},
    )

    assert "<Picture 1>:" not in encoded_prompts[0]


def test_advanced_visual_semantic_anchor_preserves_inline_image_numbers(monkeypatch):
    class Clip:
        cond_stage_model = object()
        tokenized_text = None

        @classmethod
        def tokenize(cls, text, **_kwargs):
            cls.tokenized_text = text
            return {"fake": [[(1, 1.0)]]}

        @staticmethod
        def encode_from_tokens_scheduled(_tokens):
            return [[torch.ones(1, 1, 1), {}]]

    monkeypatch.setattr(encoder_nodes, "prepare_vlm_image", lambda image, _resolution: image)

    UC_AdvancedVisualConditioningEncode.execute(
        Clip(),
        prompt="image_input_2 then image_input_1",
        system_prompt="",
        vlm_resolution=384,
        image_inputs={
            "image_1": torch.zeros(1, 2, 2, 3),
            "image_2": torch.ones(1, 2, 2, 3),
        },
        semantic_anchor=True,
    )

    picture_two = f"<Picture 2>: {encoder_nodes.VISION_BLOCK}"
    picture_one = f"<Picture 1>: {encoder_nodes.VISION_BLOCK}"
    assert picture_two in Clip.tokenized_text, Clip.tokenized_text
    assert picture_one in Clip.tokenized_text, Clip.tokenized_text
    assert Clip.tokenized_text.index(picture_two) < Clip.tokenized_text.index(picture_one)


def test_numbered_image_placeholders_preserve_prompt_order_and_strip_invalid(caplog):
    prompt, numbers = encoder_helpers.prepare_image_placeholder_prompt(
        "first image_input_2 then IMAGE_INPUT_1 repeat image_input_2 missing image_input_3 image_input_fusion",
        image_count=2,
        fusion_active=False,
        context="test",
    )

    assert numbers == (2, 1, 2)
    assert prompt.count(encoder_helpers.VISION_BLOCK) == 3
    assert "image_input_" not in prompt.lower()
    assert "stripped unavailable or fusion-only" in caplog.text


def test_fusion_placeholder_uses_one_slot_and_strips_the_rest(caplog):
    prompt, numbers = encoder_helpers.prepare_image_placeholder_prompt(
        "ignored image_input_1 chosen image_input_fusion removed image_input_2",
        image_count=2,
        fusion_active=True,
        context="test",
    )

    assert numbers == ()
    assert prompt.count(encoder_helpers.VISION_BLOCK) == 1
    assert "image_input_" not in prompt.lower()
    assert "stripped 2 additional" in caplog.text


def test_fusion_placeholder_accepts_image_one_alias_and_logs_fallback(caplog):
    prompt, _ = encoder_helpers.prepare_image_placeholder_prompt(
        "near image_input_1 subject",
        image_count=3,
        fusion_active=True,
        context="test",
    )

    assert prompt == f"near {encoder_helpers.VISION_BLOCK} subject"
    assert "treating image_input_1 as image_input_fusion" in caplog.text


def test_canonical_and_compatibility_schema_flags():
    assert UC_AttentionBiasTextEncode.define_schema().is_experimental
    for name in (
        "UC_AdvancedVisualConditioningEncode",
        "UC_AdvancedVisualConditioningEncodeTokenFusion",
        "UC_MiniMaxH3MediaConfig",
        "UC_AdvancedVisConEncoder",
        "UC_AdvancedVisConEncoderTokenFusion",
        "UC_VisualConsensusConfiguration",
        "UC_TextConsensusBlendConfig",
        "UC_ConditioningConsensusBlend",
    ):
        node = getattr(encoder_nodes, name)
        assert not node.GET_SCHEMA().is_experimental, name
        assert not node.EXPERIMENTAL, name
    assert encoder_nodes.UC_AdvancedConsensusConfiguration.GET_SCHEMA().is_experimental
    assert encoder_nodes.UC_AdvancedConsensusConfiguration.EXPERIMENTAL
    assert not UC_AdvancedVisualConditioningEncode.define_schema().is_deprecated
    assert TextEncodeKrea2SystemEditScaledAdv.define_schema().is_deprecated
    assert UC_Krea2TokenAttentionWeight.define_schema().is_experimental
    assert TextEncodeKrea2SysEditScaledAdvAttn.define_schema().is_deprecated
    assert UC_Qwen3VLInputEmbeds.define_schema().is_deprecated
    assert not UC_VLMInputEmbeds.define_schema().is_deprecated


def test_visual_fusion_encoder_formula_defaults_are_blank():
    for node in (
        encoder_nodes.UC_AdvancedVisualConditioningEncode,
        encoder_nodes.UC_Krea2TokenAttentionWeight,
    ):
        inputs = {value.id: value for value in node.define_schema().inputs}
        assert inputs["formula"].default == ""
        assert inspect.signature(node.execute).parameters["formula"].default == ""

    schema = encoder_nodes.UC_AdvancedVisualConditioningEncode.define_schema()
    assert [value.id for value in schema.inputs][-2:] == [
        "semantic_anchor",
        "image_inputs",
    ]
    assert {value.id: value for value in schema.inputs}["semantic_anchor"].default is False
    assert list(inspect.signature(encoder_nodes.UC_AdvancedVisualConditioningEncode.execute).parameters)[-1] == "semantic_anchor"
