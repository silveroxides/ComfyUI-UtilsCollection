"""Whisper regressions for batch boundaries, window seeking, and cache cleanup."""
import json
import builtins
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).parents[1]
PACKAGE = "utils_collection_whisper_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules.setdefault(PACKAGE, package)
from utils_collection_whisper_test import model_helpers as helpers
from utils_collection_whisper_test.models.whisper import ModelDimensions, Whisper


def test_missing_tiktoken_does_not_prevent_model_helpers_import(monkeypatch):
    original_import = builtins.__import__

    def without_tiktoken(name, *args, **kwargs):
        if name == "tiktoken":
            raise ModuleNotFoundError("No module named 'tiktoken'", name="tiktoken")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tiktoken)
    module_name = f"{PACKAGE}.model_helpers_without_tiktoken"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "model_helpers.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    assert callable(module.load_openpose_model)
    with pytest.raises(RuntimeError, match="Install tiktoken"):
        module.whisper_get_encoding()
    # Must fail before reading audio or loading the model onto the device.
    with pytest.raises(RuntimeError, match="Install tiktoken"):
        module.run_whisper(None, None, "transcribe", "auto")


def test_stereo_resampling_and_batch_output_alignment(monkeypatch):
    samples = torch.arange(8000, dtype=torch.float32) / 8000
    waveform = torch.stack([torch.stack([samples, -samples]), torch.stack([samples, samples])])
    audio = {"waveform": waveform, "sample_rate": 8000}
    prepared = helpers.prepare_whisper_audio(audio)
    assert prepared.shape == (2, 16000)
    assert torch.count_nonzero(prepared[0]) == 0
    assert prepared[1, 1000:15000].mean() > 0.4
    seen = []
    def transcribe(model, waveform, task, language):
        seen.append((waveform.clone(), task, language))
        return {"text": str(len(seen)), "segments": [{"start": 0, "end": 1, "text": str(len(seen))}], "language": "sv"}
    monkeypatch.setattr(helpers, "transcribe_whisper", transcribe)
    monkeypatch.setattr(helpers.comfy.model_management, "load_models_gpu", lambda models: None)
    output = helpers.run_whisper(types.SimpleNamespace(model=types.SimpleNamespace(num_languages=99)), audio, "translate", "sv")
    assert output[0] == ["1", "2"]
    assert [json.loads(value)[0]["text"] for value in output[1]] == output[0]
    assert output[2] == ["sv", "sv"]
    assert [(task, language) for _, task, language in seen] == [("translate", "sv"), ("translate", "sv")]
    torch.testing.assert_close(seen[0][0], prepared[0])
    torch.testing.assert_close(seen[1][0], prepared[1])


def test_multiwindow_timestamp_offsets_and_prompt_continuity(monkeypatch):
    tokenizer = helpers.whisper_get_tokenizer(True, language="en")
    text_tokens = tokenizer.encode(" hello")
    prompts = []
    def decode(model, mel, language, task, prompt):
        prompts.append(list(prompt))
        seconds = 30 if len(prompts) == 1 else 5
        return types.SimpleNamespace(tokens=[tokenizer.timestamp_begin, *text_tokens, tokenizer.timestamp_begin + seconds * 50], no_speech_prob=0, avg_logprob=0, temperature=0)
    monkeypatch.setattr(helpers, "whisper_decode_with_fallback", decode)
    model = types.SimpleNamespace(dims=types.SimpleNamespace(n_mels=80, n_audio_ctx=1500), device=torch.device("cpu"), compute_dtype=torch.float32, is_multilingual=True, num_languages=99)
    result = helpers.transcribe_whisper(model, torch.zeros(35 * 16000), "transcribe", "en")
    assert [(segment["start"], segment["end"]) for segment in result["segments"]] == [(0, 30), (30, 35)]
    assert result["text"] == " hello hello"
    assert prompts[0] == [] and text_tokens[0] in prompts[1]


def test_interruption_removes_decoder_hooks(monkeypatch):
    dims = ModelDimensions(4, 6, 8, 2, 1, 51865, 8, 8, 2, 1)
    model = Whisper(dims)
    model.device = torch.device("cpu")
    task = helpers.WhisperDecodingTask(model, helpers.WhisperDecodingOptions(language="en"))
    def interrupted(tokens, features):
        task.inference.kv_cache, task.inference.hooks = model.install_kv_cache_hooks()
        raise RuntimeError("test interruption")
    monkeypatch.setattr(task.inference, "logits", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        task._main_loop(torch.zeros(1, 6, 8), torch.tensor([[50258, 50259, 50359]]))
    assert task.inference.hooks == [] and task.inference.kv_cache == {}
    assert all(not layer._forward_hooks for layer in model.decoder.modules())
