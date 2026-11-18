import hashlib
import json
import logging
import math
import numbers
import re
from contextlib import ExitStack, contextmanager
from typing import Any

import torch

try:
    import jsonschema
except ImportError:
    jsonschema = None

import comfy.ops
from comfy import model_management
from comfy.text_encoders.minimax import process_video_block
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids

from .encoder_helpers import fuse_deepstack_layers, fuse_visual_token_sources, prepare_vlm_image

QWEN_IMAGE_PAD_ID = 151655
QWEN_VIDEO_PAD_ID = 151656
QWEN35_IMAGE_PAD_ID = 248056
QWEN35_VIDEO_PAD_ID = 248057
QWEN_IMAGE_PAD_IDS = frozenset({QWEN_IMAGE_PAD_ID, QWEN35_IMAGE_PAD_ID})
QWEN_VIDEO_PAD_IDS = frozenset({QWEN_VIDEO_PAD_ID, QWEN35_VIDEO_PAD_ID})
QWEN_VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"
QWEN_MAX_VIDEO_FRAMES = 768


def _positive_fps(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"Qwen3-VL {name} must be a finite value greater than zero.")
    return float(value)


def prepare_qwen3vl_video_frames(video, resolution, video_fps, sample_fps, max_frames: int = QWEN_MAX_VIDEO_FRAMES) -> tuple[torch.Tensor, list[int]]:
    sources = list(video) if isinstance(video, (list, tuple)) else [video]
    source_frames = []
    for source in sources:
        if not torch.is_tensor(source) or source.ndim != 4 or source.shape[-1] != 3:
            raise ValueError("Qwen3-VL video requires BHWC RGB IMAGE tensors.")
        source_frames.extend(source[index : index + 1] for index in range(source.shape[0]))
    frame_count = len(source_frames)
    if frame_count < 1:
        raise ValueError("Qwen3-VL video requires at least one source frame.")
    source_fps = _positive_fps(video_fps, "video_fps")
    target_fps = min(_positive_fps(sample_fps, "sample_fps"), source_fps)
    indices = []
    sample_index = 0
    while True:
        source_index = int(math.floor(sample_index * source_fps / target_fps + 0.5))
        if source_index >= frame_count:
            break
        if not indices or source_index != indices[-1]:
            indices.append(source_index)
        sample_index += 1
    if max_frames and len(indices) > max_frames:
        logging.warning("[UC_TextGenerate] Sampled %d video frames at %.2f fps exceeds the %d-frame Qwen video limit; uniformly subsampling to %d frames.", len(indices), target_fps, max_frames, max_frames)
        indices = [indices[round(i * (len(indices) - 1) / (max_frames - 1))] for i in range(max_frames)]
    prepared = [prepare_vlm_image(source_frames[index], resolution) for index in indices]
    if len({tuple(frame.shape[1:3]) for frame in prepared}) != 1:
        raise ValueError("Qwen3-VL video frames must resolve to one common spatial size.")
    frames = torch.cat(prepared, dim=0)
    motion_delta = float((frames[1:] - frames[:-1]).abs().mean()) if frames.shape[0] > 1 else 0.0
    logging.info("UC_TextGenerate Qwen3-VL video: source_frames=%d sampled_frames=%d temporal_blocks=%d source_fps=%.4f sample_fps=%.4f first_index=%d last_index=%d motion_delta=%.8f", frame_count, len(indices), (len(indices) + 1) // 2, source_fps, target_fps, indices[0], indices[-1], motion_delta)
    if len(indices) % 2 == 1:
        frames = torch.cat([frames, frames[-1:]], dim=0)
        indices.append(indices[-1])
    return frames, indices


def qwen3vl_video_prompt(indices: list[int], video_fps, video_number: int = 1) -> str:
    source_fps = _positive_fps(video_fps, "video_fps")
    if not indices or len(indices) % 2 != 0:
        raise ValueError("Qwen3-VL video sampling must produce paired frame indices.")
    return f"Video {int(video_number)}: " + "".join(f"<{((indices[index] + indices[index + 1]) / 2) / source_fps:.1f} seconds>{QWEN_VIDEO_TOKEN}" for index in range(0, len(indices), 2))


def _qwen_clip_model(clip):
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None:
        return None
    for name in (getattr(stage, "clip", None), getattr(stage, "clip_name", None)):
        candidate = getattr(stage, name, None) if isinstance(name, str) else None
        if candidate is not None and hasattr(candidate, "process_tokens") and hasattr(candidate, "transformer"):
            return candidate
    if hasattr(stage, "process_tokens") and hasattr(stage, "transformer"):
        return stage
    return None


def _token_rows(tokens):
    if isinstance(tokens, dict):
        tokens = next(iter(tokens.values()))
    return [[item[0] for item in batch] for batch in tokens]


def _visual_grid_shape(block):
    grid = block["extra"].get("grid")
    if grid is None:
        raise ValueError("Qwen3-VL visual source is missing its exact grid metadata.")
    values = grid.reshape(-1).tolist() if torch.is_tensor(grid) else list(grid)
    if len(values) < 3:
        raise ValueError(f"Qwen3-VL visual source received a malformed grid: {grid!r}.")
    shape = (int(values[-2]) // 2, int(values[-1]) // 2)
    size = block["embedding"].reshape(-1, block["embedding"].shape[-1]).shape[0]
    if shape[0] * shape[1] != size:
        raise ValueError(f"Qwen3-VL visual grid {shape} does not match its {size} embeddings.")
    return shape


def _load_qwen_generation_model(clip):
    model = _qwen_clip_model(clip)
    if model is None:
        raise ValueError("Qwen multimodal generation requires a supported Core Qwen model wrapper.")
    if hasattr(model, "reset_clip_options"):
        model.reset_clip_options()
    if hasattr(clip, "load_model"):
        clip.load_model()
    device = getattr(getattr(clip, "patcher", None), "load_device", None) or getattr(model, "execution_device", None)
    if device is None:
        device = model.transformer.get_input_embeddings().weight.device
    if hasattr(model, "set_clip_options"):
        model.set_clip_options({"layer": None, "execution_device": device})
    return model, device


def _qwen_execution_context(clip, device):
    stack = ExitStack()
    if hasattr(model_management, "cuda_device_context"):
        stack.enter_context(model_management.cuda_device_context(device))
    stage = getattr(clip, "cond_stage_model", None)
    if stage is not None and hasattr(stage, "modules") and getattr(device, "type", None) == "cuda" and hasattr(comfy.ops, "use_quantized_matmul"):
        stack.enter_context(comfy.ops.use_quantized_matmul(stage, device))
    return stack


def _preprocess_qwen3vl_image(model, device, image):
    embedding, extra = model.transformer.preprocess_embed({"type": "image", "data": image, "original_type": "image"}, device=device)
    if hasattr(model.transformer, "build_image_inputs"):
        if embedding is None or not isinstance(extra, dict) or extra.get("grid") is None or not extra.get("deepstack"):
            raise ValueError("Qwen3-VL image preprocessing did not return grid and DeepStack metadata.")
        return {"kind": "image", "embedding": embedding, "extra": extra}
    grid = extra.get("grid") if isinstance(extra, dict) else extra
    if embedding is None or grid is None:
        raise ValueError("Qwen3.5 image preprocessing did not return grid metadata.")
    return {"kind": "image", "embedding": embedding, "extra": {"grid": grid}}


def _preprocess_qwen3vl_video_pair(model, device, frames):
    flatten, grid = process_video_block(frames)
    visual_out = model.transformer.visual(flatten.to(device, dtype=torch.float32), grid)
    if hasattr(model.transformer, "build_image_inputs"):
        if not isinstance(visual_out, tuple) or len(visual_out) != 2:
            raise ValueError("Qwen3-VL video preprocessing did not return temporal and DeepStack embeddings.")
        embedding, deepstack = visual_out
        if embedding is None or not deepstack:
            raise ValueError("Qwen3-VL video preprocessing did not return temporal and DeepStack embeddings.")
        return {"kind": "video", "embedding": embedding, "extra": {"grid": grid, "deepstack": deepstack}}
    embedding = visual_out[0] if isinstance(visual_out, tuple) else visual_out
    if embedding is None:
        raise ValueError("Qwen3.5 video preprocessing did not return temporal embeddings.")
    return {"kind": "video", "embedding": embedding, "extra": {"grid": grid}}


def _fuse_preprocessed_qwen3vl_images(blocks, config, device):
    if not blocks:
        raise ValueError("Active visual fusion with video requires at least one image source.")
    shapes = [_visual_grid_shape(block) for block in blocks]
    size = blocks[0]["embedding"].reshape(-1, blocks[0]["embedding"].shape[-1]).shape[0]
    primary = [block["embedding"].reshape(-1, block["embedding"].shape[-1]).to(device) for block in blocks]
    mask_cache = {}
    fused = fuse_visual_token_sources(primary, config, device, mask_cache, size, shapes)
    extra = dict(blocks[0]["extra"])
    deepstacks = {index: block["extra"].get("deepstack") for index, block in enumerate(blocks)}
    if all(deepstacks.values()):
        extra["deepstack"] = fuse_deepstack_layers(deepstacks, config, device, mask_cache, size, shapes)
    return {"kind": "image", "embedding": fused, "extra": extra}


def _attach_preprocessed_visuals(rows, image_blocks, video_blocks):
    if len(rows) != 1:
        raise ValueError("Qwen3-VL mixed image/video generation requires one tokenizer batch.")
    image_queue = list(image_blocks)
    video_queue = list(video_blocks)
    tagged = []
    for index, value in enumerate(rows[0]):
        block = None
        if isinstance(value, numbers.Integral) and int(value) in QWEN_IMAGE_PAD_IDS and image_queue:
            block = image_queue.pop(0)
        elif isinstance(value, numbers.Integral) and int(value) in QWEN_VIDEO_PAD_IDS and video_queue:
            block = video_queue.pop(0)
        if block is not None:
            marker = {"type": "embedding", "data": block["embedding"], "_textgen_visual": block}
            rows[0][index] = marker
            tagged.append(marker)
    if image_queue or video_queue:
        raise ValueError("Qwen3-VL prompt did not contain every required image and video placeholder.")
    return tagged


def generate_qwen3vl_video(clip, full_prompt, images, video_frames, fusion_config, generation_args, thinking=False):
    model, device = _load_qwen_generation_model(clip)
    has_deepstack = hasattr(model.transformer, "build_image_inputs")
    if not has_deepstack and (not hasattr(model.transformer, "visual") or not callable(getattr(model.transformer, "preprocess_embed", None))):
        raise ValueError("Video input requires a Core Qwen3-VL or Qwen3.5 model wrapper.")

    with _qwen_execution_context(clip, device):
        image_blocks = [_preprocess_qwen3vl_image(model, device, image) for image in images]
        if fusion_config is not None:
            if fusion_config.get("save_blended_embeds", False):
                logging.warning("UC_TextGenerate ignores save_blended_embeds in mixed image/video generation.")
            image_blocks = [_fuse_preprocessed_qwen3vl_images(image_blocks, fusion_config, device)]
        video_batches = list(video_frames) if isinstance(video_frames, (list, tuple)) else [video_frames]
        video_blocks = [_preprocess_qwen3vl_video_pair(model, device, batch[index : index + 2]) for batch in video_batches for index in range(0, batch.shape[0], 2)]

        tokens = clip.tokenize(full_prompt, skip_template=True, min_length=1, thinking=thinking)
        rows = _token_rows(tokens)
        tagged = _attach_preprocessed_visuals(rows, image_blocks, video_blocks)
        other_values = [value for row in rows for value in row if not isinstance(value, numbers.Integral)]
        embeds, _, _, embeds_info = model.process_tokens(rows, device)
        if len(embeds_info) != len(other_values):
            raise ValueError("Qwen3-VL could not preserve the prepared multimodal token layout.")

        tagged_ids = {id(marker) for marker in tagged}
        mapped = 0
        for source, info in zip(other_values, embeds_info, strict=True):
            if id(source) not in tagged_ids:
                continue
            block = source["_textgen_visual"]
            info["type"] = "image"
            info["extra"] = block["extra"]
            info["textgen_source_kind"] = block["kind"]
            mapped += 1
        if mapped != len(tagged):
            raise ValueError("Qwen3-VL could not map every prepared visual embedding into the prompt.")

        logging.info("UC_TextGenerate Qwen3-VL media blocks: %s", [(info.get("textgen_source_kind"), int(info["index"]), int(info["size"])) for info in embeds_info if info.get("textgen_source_kind") is not None])

        if has_deepstack:
            position_ids, visual_mask, deepstack = model.transformer.build_image_inputs(embeds, embeds_info)
            transformer_args = {k: v for k, v in generation_args.items() if k != "mtp"}
            return model.transformer.generate(embeds, **transformer_args, position_ids=position_ids, visual_pos_masks=visual_mask, deepstack_embeds=deepstack)
        position_ids = qwen2vl_mrope_position_ids(embeds_info, embeds.shape[1], embeds.device)
        if position_ids is None:
            raise ValueError("Qwen3.5 video generation could not rebuild multimodal MRoPE positions.")
        return model.transformer.generate(embeds, **generation_args, position_ids=position_ids)


def _strip_thinking_and_fences(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, maxsplit=1, flags=re.IGNORECASE)[1].strip()
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        cleaned = re.split(r"<think>", cleaned, flags=re.IGNORECASE)[0].strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    return cleaned


def _extract_balanced_json(text: str, expect_array: bool = False) -> Any:
    cleaned = _strip_thinking_and_fences(text)
    if not cleaned:
        return [] if expect_array else {}
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    open_ch = "[" if expect_array else "{"
    close_ch = "]" if expect_array else "}"
    start = cleaned.find(open_ch)
    if start < 0:
        alt_start = cleaned.find("{" if expect_array else "[")
        if alt_start >= 0:
            start = alt_start
            open_ch = cleaned[start]
            close_ch = "}" if open_ch == "{" else "]"
        else:
            return [] if expect_array else {}

    stack: list[str] = []
    in_str = False
    escape = False
    end = -1
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    end = idx + 1
                    break

    candidate = cleaned[start:end] if end > start else cleaned[start:]
    if end < 0 and stack:
        if in_str:
            candidate += '"'
        candidate = re.sub(r",\s*$", "", candidate)
        candidate += "".join(reversed(stack))

    try:
        return json.loads(candidate)
    except Exception:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except Exception:
            return [] if expect_array else {}


def _default_for_schema(schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return ""
    if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    stype = schema.get("type", "object")
    if isinstance(stype, list):
        stype = next((t for t in stype if t != "null"), stype[0] if stype else "string")
    if stype == "object":
        out: dict[str, Any] = {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        req = schema.get("required") if isinstance(schema.get("required"), list) else list(props.keys())
        for key in req:
            if key in props:
                out[key] = _default_for_schema(props[key])
        return out
    if stype == "array":
        min_items = int(schema.get("minItems", 0) or 0)
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [_default_for_schema(item_schema) for _ in range(max(0, min_items))]
    if stype == "integer":
        low = schema.get("minimum")
        return int(low) if isinstance(low, (int, float)) else 0
    if stype == "number":
        low = schema.get("minimum")
        return float(low) if isinstance(low, (int, float)) else 0.0
    if stype == "boolean":
        return False
    return ""


def _conform_value_to_schema(value: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return value
    enum_vals = schema.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        if value in enum_vals:
            return value
        str_val = str(value or "").strip().lower()
        for candidate in enum_vals:
            if str(candidate).lower() == str_val:
                return candidate
        return enum_vals[0]

    stype = schema.get("type")
    if isinstance(stype, list):
        if value is None and "null" in stype:
            return None
        stype = next((t for t in stype if t != "null"), stype[0] if stype else "string")

    if stype == "object" or (stype is None and "properties" in schema):
        source = value if isinstance(value, dict) else {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        req = schema.get("required") if isinstance(schema.get("required"), list) else []
        out: dict[str, Any] = {}
        alias_map = {"caption": ("desc", "description", "summary"), "desc": ("caption", "description"), "label": ("type", "category"), "type": ("label", "category"), "palette": ("color_palette", "colors"), "color_palette": ("palette", "colors"), "bbox": ("box_2d", "box")}
        for key, prop_schema in props.items():
            raw_val = source.get(key)
            if raw_val is None and key in alias_map:
                for alt in alias_map[key]:
                    if alt in source and source[alt] is not None:
                        raw_val = source[alt]
                        break
            if raw_val is not None:
                conformed_val = _conform_value_to_schema(raw_val, prop_schema)
                if key == "text" and key not in req and str(out.get("label", source.get("label", "")) or "").strip().lower() not in ("", "text") and not str(conformed_val or "").strip():
                    continue
                out[key] = conformed_val
            elif key in req:
                out[key] = _default_for_schema(prop_schema)
        if not props:
            for k, v in source.items():
                out[str(k)] = v
        return out

    if stype == "array" or (stype is None and "items" in schema):
        raw_list = value if isinstance(value, list) else ([value] if value is not None and value != "" else [])
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else None
        items = [_conform_value_to_schema(item, item_schema) if item_schema else item for item in raw_list]
        if isinstance(item_schema, dict) and item_schema.get("type") == "object":
            item_req = item_schema.get("required") if isinstance(item_schema.get("required"), list) else []
            if "bbox" in item_req:
                items = [it for it in items if isinstance(it, dict) and isinstance(it.get("bbox"), list) and len(it["bbox"]) == 4]
        elif isinstance(item_schema, dict) and item_schema.get("type") == "string" and "hex" in str(schema.get("description", "")).lower():
            items = [s for s in items if isinstance(s, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", s.strip())]
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and max_items >= 0:
            items = items[:max_items]
        min_items = int(schema.get("minItems", 0) or 0)
        while len(items) < min_items:
            items.append(_default_for_schema(item_schema or {}))
        return items

    if stype == "integer":
        try:
            num = int(round(float(value)))
        except (TypeError, ValueError):
            num = _default_for_schema(schema)
        low = schema.get("minimum")
        high = schema.get("maximum")
        if isinstance(low, (int, float)):
            num = max(int(math.ceil(low)), num)
        if isinstance(high, (int, float)):
            num = min(int(math.floor(high)), num)
        return num

    if stype == "number":
        try:
            num_f = float(value)
            if not math.isfinite(num_f):
                raise ValueError
        except (TypeError, ValueError):
            num_f = float(_default_for_schema(schema))
        low = schema.get("minimum")
        high = schema.get("maximum")
        if isinstance(low, (int, float)):
            num_f = max(float(low), num_f)
        if isinstance(high, (int, float)):
            num_f = min(float(high), num_f)
        return num_f

    if stype == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    if stype == "string":
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value)

    return value


def conform_and_validate_response_schema(text: str, response_schema: dict[str, Any] | None) -> tuple[str, bool]:
    if not isinstance(response_schema, dict) or not response_schema:
        return text, True
    expect_array = response_schema.get("type") == "array"
    extracted = _extract_balanced_json(text, expect_array=expect_array)
    had_content = bool(extracted)
    conformed = _conform_value_to_schema(extracted, response_schema)
    valid = had_content
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=conformed, schema=response_schema)
        except Exception:
            valid = False
    serialized = json.dumps(conformed, ensure_ascii=False, separators=(",", ":"))
    return serialized, valid


def _resolve_eos_token_id(clip: Any, transformer: Any) -> int:
    cfg = getattr(getattr(transformer, "model", None), "config", None)
    stop_tokens = getattr(cfg, "stop_tokens", None)
    if isinstance(stop_tokens, (list, tuple)):
        int_stops = [int(tok) for tok in stop_tokens if isinstance(tok, int)]
        if int_stops:
            return int_stops[-1]

    stage_tok = getattr(clip, "tokenizer", None)
    inner_name = getattr(stage_tok, "clip", None) if stage_tok is not None else None
    inner_tok = getattr(stage_tok, inner_name, None) if isinstance(inner_name, str) else None
    for candidate in (inner_tok, stage_tok):
        if candidate is None:
            continue
        hf_tok = getattr(candidate, "tokenizer", None)
        if hf_tok is not None:
            vocab = hf_tok.get_vocab() if callable(getattr(hf_tok, "get_vocab", None)) else None
            if isinstance(vocab, dict):
                for tok_text in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>"):
                    tok_val = vocab.get(tok_text)
                    if isinstance(tok_val, int):
                        return tok_val
            eos_id = getattr(hf_tok, "eos_token_id", None)
            if isinstance(eos_id, int):
                return eos_id
        end_tok = getattr(candidate, "end_token", None)
        if isinstance(end_tok, int):
            return end_tok
    return 151645


@contextmanager
def schema_guided_generation_context(clip: Any, response_schema: dict[str, Any] | None, prefilled_open: str = "", thinking_prefilled: bool = False, mtp: Any = False):
    if not isinstance(response_schema, dict) or not response_schema:
        yield
        return

    model = _qwen_clip_model(clip)
    transformer = getattr(model, "transformer", None) if model is not None else None
    orig_sample_token = getattr(transformer, "sample_token", None) if transformer is not None else None
    if not callable(orig_sample_token):
        yield
        return

    # Qwen3.5 speculative MTP drafts tokens without calling sample_token on accepted draft rows
    # or when sampling is off, so per-token brace tracking via sample_token cannot mirror ids.
    mtp_active = mtp not in (False, None, "off", 0) and getattr(transformer, "mtp", None) is not None
    if mtp_active:
        yield
        return

    eos_token_id = _resolve_eos_token_id(clip, transformer)

    history_ids: list[int] = []
    finished = False

    def _is_json_complete(decoded_body: str) -> bool:
        body = _strip_thinking_and_fences(decoded_body)
        if not body:
            return False
        start_ch = "[" if response_schema.get("type") == "array" else "{"
        start_idx = body.find(start_ch)
        if start_idx < 0:
            return False
        stack: list[str] = []
        in_str = False
        esc = False
        for idx in range(start_idx, len(body)):
            ch = body[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
                    if not stack:
                        return True
        return False

    def guided_sample_token(logits, *args, **kwargs):
        nonlocal finished
        if finished:
            batch = int(logits.shape[0]) if getattr(logits, "ndim", 0) >= 2 else 1
            return torch.full((batch, 1), eos_token_id, dtype=torch.long, device=logits.device)
        next_tok = orig_sample_token(logits, *args, **kwargs)
        tok_id = int(next_tok.reshape(-1)[0].item())
        history_ids.append(tok_id)
        try:
            decoded = prefilled_open + clip.decode(history_ids, skip_special_tokens=True)
            if "</think>" in decoded:
                if _is_json_complete(decoded.partition("</think>")[2]):
                    finished = True
            elif not thinking_prefilled and "<think>" not in decoded:
                if _is_json_complete(decoded):
                    finished = True
        except Exception:
            pass
        return next_tok

    transformer.sample_token = guided_sample_token
    try:
        yield
    finally:
        transformer.sample_token = orig_sample_token


def fingerprint_textgen_inputs(prompt="", system_prompt="", vlm_resolution=384, vlm_video_resolution=384, max_length=512, formula="", thinking=False, mtp="auto", escape_parentheses=False, sampling_mode=None, seed=None, video_fps=24.0, sample_fps=2.0, visual_fusion_config=None, response_schema=None) -> str:
    sampling = sampling_mode if isinstance(sampling_mode, dict) else {}
    mode = sampling.get("sampling_mode", "off")
    if mode == "on":
        resolved_seed = seed if seed is not None else sampling.get("seed", 0)
        normalized_sampling = {
            "sampling_mode": "on",
            "temperature": sampling.get("temperature", 0.7),
            "top_k": sampling.get("top_k", 64),
            "top_p": sampling.get("top_p", 0.95),
            "min_p": sampling.get("min_p", 0.05),
            "repetition_penalty": sampling.get("repetition_penalty", 1.05),
            "presence_penalty": sampling.get("presence_penalty", 0.0),
            "empty_response_retries": sampling.get("empty_response_retries", 4),
            "seed": int(resolved_seed) if resolved_seed is not None else 0,
        }
    else:
        normalized_sampling = {"sampling_mode": mode}

    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "vlm_resolution": vlm_resolution,
        "vlm_video_resolution": vlm_video_resolution,
        "max_length": max_length,
        "formula": formula,
        "thinking": bool(thinking),
        "mtp": mtp,
        "escape_parentheses": bool(escape_parentheses),
        "sampling": normalized_sampling,
        "video_fps": video_fps,
        "sample_fps": sample_fps,
        "visual_fusion_config": visual_fusion_config if isinstance(visual_fusion_config, dict) else None,
        "response_schema": response_schema if isinstance(response_schema, dict) else None,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
