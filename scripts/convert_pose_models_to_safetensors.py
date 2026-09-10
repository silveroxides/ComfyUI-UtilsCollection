"""Extract trusted pose checkpoints into verified eager-model safetensors."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import types
import uuid

import torch
from unifiedefficientloader import IncrementalSafetensorsWriter

from convert_big_lama_to_safetensors import sha256_file, verify_output


MANIFEST = json.loads((Path(__file__).resolve().parents[1] / "models" / "migrations.json").read_text(encoding="utf-8"))
MODELS = MANIFEST["models"]


def model_type(kind):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root.parent.parent))
    package = types.ModuleType("_uc_pose_conversion_models")
    package.__path__ = [str(root / "models")]
    sys.modules.setdefault(package.__name__, package)
    specification = MODELS[kind]
    return getattr(importlib.import_module(f"{package.__name__}.{specification['module']}"), specification["architecture"])


def convert(kind, source, destination):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("Destination must end in .safetensors")
    specification = MODELS[kind]
    architecture = model_type(kind)
    if specification["source_format"] == "torchscript":
        scripted = torch.jit.load(str(source), map_location="cpu")
        weights = dict(scripted.state_dict())
        del scripted
    else:
        weights = torch.load(source, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = architecture()
    expected = model.state_dict()
    if specification["mapping"] == "strip_model_prefix":
        mapping = {key: ".".join(key.split(".")[1:]) for key in expected}
    else:
        module = sys.modules[architecture.__module__]
        prefix = specification.get("map_prefix", "")
        roots = getattr(module, f"{prefix}TORCHSCRIPT_STATE_DICT_ROOT_MAP", {})
        key_map = getattr(module, f"{prefix}TORCHSCRIPT_STATE_DICT_KEY_MAP", {})
        mapping = {}
        for key in weights:
            root, separator, suffix = key.rpartition(".")
            target = key_map.get(key, f"{roots.get(root, root)}.{suffix}" if separator else key)
            if target in mapping:
                raise ValueError(f"Duplicate converted key: {target}")
            mapping[target] = key
    if set(mapping) != set(expected):
        raise ValueError(f"Converted keys do not match {architecture.__name__}: missing={sorted(set(expected) - set(mapping))[:5]}, unexpected={sorted(set(mapping) - set(expected))[:5]}")
    if set(mapping.values()) != set(weights):
        raise ValueError(f"Checkpoint keys do not match {architecture.__name__}")
    converted = {}
    for key, source_key in mapping.items():
        value = weights[source_key]
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Checkpoint tensor does not match {architecture.__name__}: {key}")
        converted[key] = value.detach().cpu().contiguous()
    del weights, expected, model
    gc.collect()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    metadata = {"architecture": architecture.__name__, "format": "pt", "source_filename": source.name,
                "source_sha256": sha256_file(source), "kind": kind}
    try:
        with IncrementalSafetensorsWriter(str(temporary), metadata=metadata, max_workers=1) as writer:
            for key, value in converted.items():
                writer.write(key, value)
        verify_output(temporary, converted)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Verified {destination}: {len(converted)} tensors, {destination.stat().st_size} bytes")  # noqa: T201


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=MODELS)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.kind, args.source, args.destination)


if __name__ == "__main__":
    main()
