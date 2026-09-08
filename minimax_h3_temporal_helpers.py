"""Bounded temporal sampling for experimental MiniMax H3 encoders."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

import comfy.model_management


def minimax_h3_temporal_frame_pairs(
    frame_count: int,
    canonical_indices: Sequence[int],
    density: int = 1,
) -> list[list[tuple[int, int]]]:
    """Return unique source pairs per canonical block, canonical pair first.

    The caller supplies ordinary sampling indices, preserving both configured
    FPS sampling and the native no-config path exactly. Lane k samples k/density
    of the interval from each canonical position to the next; the final interval
    ends at frame_count. Round to nearest integer, with half ties upward. Omit a
    lane's pair if rounding reaches either interval's exclusive endpoint, rather
    than borrowing a frame from the next canonical position or outside the video.
    An odd final sample repeats its shifted frame, matching native two-frame
    padding. Deduplication is local to a block and preserves lane order.

    This plans Python indices only; no tensors, encoder imports, or model work.
    Density one produces only native pairs and performs no alternative sampling.
    """
    if type(frame_count) is not int or frame_count < 0:
        raise ValueError("MiniMax H3 source frame count must be a nonnegative integer.")
    if type(density) is not int or not 1 <= density <= 24:
        raise ValueError("MiniMax H3 temporal density must be an integer from 1 to 24.")
    indices = list(canonical_indices)
    previous = -1
    for index in indices:
        if type(index) is not int or not previous < index < frame_count:
            raise ValueError("MiniMax H3 canonical indices must be increasing source frame indices.")
        previous = index

    blocks = []
    for position in range(0, len(indices), 2):
        second_position = min(position + 1, len(indices) - 1)
        pair = (indices[position], indices[second_position])
        pairs = [pair]
        if density == 1:
            blocks.append(pairs)
            continue
        seen = {pair}
        for lane in range(1, density):
            shifted = []
            for sample_position in (position, second_position):
                start = indices[sample_position]
                stop = (
                    indices[sample_position + 1]
                    if sample_position + 1 < len(indices)
                    else frame_count
                )
                index = start + (2 * lane * (stop - start) + density) // (2 * density)
                if index >= stop:
                    break
                shifted.append(index)
            if len(shifted) == 2:
                pair = (shifted[0], shifted[1])
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        blocks.append(pairs)
    return blocks


def _validate_temporal_sources(sources, token_count=None, source_count=None):
    if not sources or any(source.ndim != 2 for source in sources):
        raise ValueError("Temporal sources must have [tokens, dimensions] shape.")
    first = sources[0]
    if first.shape[0] == 0 or first.shape[1] == 0:
        raise ValueError("Temporal blocks must contain tokens and dimensions.")
    if any(source.shape != first.shape or source.device != first.device
           or source.dtype != first.dtype for source in sources):
        raise ValueError("Temporal sources must match shape, device, and dtype.")
    if token_count is not None and first.shape[0] != token_count:
        raise ValueError("Temporal DeepStack token count must match the primary block.")
    if source_count is not None and len(sources) != source_count:
        raise ValueError("Temporal DeepStack source count must match the primary block.")


def _temporal_consensus_weights(stacked, settings):
    consensus = (torch.median(stacked, dim=0).values
                 if settings["consensus_type"] == "median" else stacked.mean(dim=0))
    similarities = torch.mv(
        F.normalize(stacked, p=2, dim=1, eps=1e-8),
        F.normalize(consensus, p=2, dim=0, eps=1e-8),
    )
    stretched = similarities
    similarity_alignment = settings["alignment_method"] == "similarity"
    if similarity_alignment and settings["dynamic_similarity_contrast"]:
        low, high = similarities.min(), similarities.max()
        if high > low:
            stretched = 0.7 + 0.3 * (similarities - low) / (high - low + 1e-8)
    weights = torch.zeros_like(similarities)
    mask = similarities >= settings["similarity_threshold"]
    if mask.any():
        if settings["diversity_beta"] > 0.0:
            distance_base = (1.5 if similarity_alignment and settings["soft_comfort_bandpass"]
                             else 1.001)
            safe = stretched[mask].clamp(min=0.0, max=1.0)
            weights[mask] = safe.pow(settings["power_alpha"]) * (
                distance_base - safe
            ).clamp(min=0.0).pow(settings["diversity_beta"])
        else:
            weights[mask] = stretched[mask].clamp(min=0.0).pow(settings["power_alpha"])
        total = weights.sum()
        if total > 0:
            return weights / total
    return torch.ones_like(similarities) / len(similarities)


def _apply_temporal_consensus_row(stacked, weights, settings):
    merged = (stacked * weights.to(device=stacked.device, dtype=stacked.dtype)[:, None]).sum(dim=0)
    if settings["rescale_norm"]:
        average_norm = torch.norm(stacked, p=2, dim=1).mean()
        merged_norm = torch.norm(merged, p=2)
        if merged_norm > 0:
            merged = (merged / merged_norm) * average_norm
    if settings["global_scale"] != 1.0:
        merged *= settings["global_scale"]
    return merged


def fuse_temporal_block(
    sources,
    method,
    resolved_consensus_settings=None,
    visual_config=None,
    grids=None,
    spatial_fuse_callback=None,
    position_score_callback=None,
    deepstack_layers=None,
):
    """Fuse one corresponding video block; reuse primary choices on DeepStack.

    Sources are [tokens, dimensions], canonical first, with identical shapes,
    devices and dtypes. DeepStack is layer-major: each layer contains one tensor
    per source. Each layer can have its own feature width/device/dtype. Caller
    supplies resolved existing consensus settings and existing spatial/position
    callbacks, keeping this leaf independent of encoder_helpers and model loading.
    No matching can escape this block. Returns (primary, layer list or None).
    """
    _validate_temporal_sources(sources)
    layers = [] if deepstack_layers is None else list(deepstack_layers)
    for layer in layers:
        _validate_temporal_sources(layer, sources[0].shape[0], len(sources))
    if method not in ("consensus", "spatial"):
        raise ValueError(f"Unsupported temporal fusion method: {method}")
    settings = resolved_consensus_settings
    if len(sources) == 1 or (method == "consensus" and settings["blend_preset"] == "off"):
        return sources[0], ([layer[0] for layer in layers] if deepstack_layers is not None else None)
    if method == "spatial":
        if spatial_fuse_callback is None:
            raise ValueError("Temporal spatial fusion requires the existing visual fusion callback.")
        primary, weights = spatial_fuse_callback(
            sources, visual_config, sources[0].device,
            expected_length=sources[0].shape[0], source_grids=grids, return_weights=True,
        )
        deepstack = [spatial_fuse_callback(
            layer, visual_config, layer[0].device,
            expected_length=layer[0].shape[0], source_grids=grids, weights_override=weights,
        ) for layer in layers]
        return primary, deepstack if deepstack_layers is not None else None

    prefix_length = 0
    if settings["preserve_common_prefix"]:
        common = torch.ones(sources[0].shape[0], dtype=torch.bool, device=sources[0].device)
        for source in sources[1:]:
            common &= torch.isclose(sources[0], source, rtol=1e-5, atol=1e-6).all(dim=-1)
        mismatch = (~common).nonzero(as_tuple=False)
        prefix_length = int(mismatch[0].item()) if mismatch.numel() else len(common)
    output = sources[0].clone()
    deepstack = [layer[0].clone() for layer in layers]
    if prefix_length == output.shape[0]:
        return output, deepstack if deepstack_layers is not None else None
    trimmed = [source[prefix_length:] for source in sources]
    if settings["blend_method"] == "linear":
        output[prefix_length:] = torch.stack(trimmed).mean(dim=0) * settings["global_scale"]
        for target, layer in zip(deepstack, layers):
            target[prefix_length:] = torch.stack(
                [source[prefix_length:] for source in layer]
            ).mean(dim=0) * settings["global_scale"]
        return output, deepstack if deepstack_layers is not None else None

    length = trimmed[0].shape[0]
    selections = [[(0, row)] for row in range(length)]
    if settings["alignment_method"] == "similarity":
        reference = F.normalize(trimmed[0], p=2, dim=1)
        for source_index, source in enumerate(trimmed[1:], start=1):
            similarities = torch.mm(reference, F.normalize(source, p=2, dim=1).t())
            scores = similarities.clone()
            if settings["position_weight"] > 0.0:
                if position_score_callback is None:
                    raise ValueError("Temporal position alignment requires the existing score callback.")
                scores = position_score_callback(scores, settings["position_weight"])
                scores = scores.masked_fill(similarities < settings["alignment_threshold"], -100.0)
            for _ in range(length):
                flat_index = torch.argmax(scores)
                maximum = scores.flatten()[flat_index].item()
                if (settings["position_weight"] == 0.0 and maximum < settings["alignment_threshold"]) or maximum <= -100.0:
                    break
                row, column = divmod(int(flat_index.item()), length)
                selections[row].append((source_index, column))
                scores[row, :] = -100.0
                scores[:, column] = -100.0
    else:
        selections = [[(source, row) for source in range(len(sources))] for row in range(length)]
    for row, selected in enumerate(selections):
        stacked = torch.stack([trimmed[source][index] for source, index in selected])
        weights = _temporal_consensus_weights(stacked, settings)
        output[prefix_length + row] = _apply_temporal_consensus_row(stacked, weights, settings)
        for target, layer in zip(deepstack, layers):
            selected_features = torch.stack([
                layer[source][prefix_length + index] for source, index in selected
            ])
            target[prefix_length + row] = _apply_temporal_consensus_row(selected_features, weights, settings)
    return output, deepstack if deepstack_layers is not None else None


def _temporal_token_layout(tokens, frame_pairs):
    if set(tokens) != {"qwen3vl_32b"} or len(tokens["qwen3vl_32b"]) != 1:
        raise ValueError("Temporal fusion requires one MiniMax H3 token row.")
    row = tokens["qwen3vl_32b"][0]
    image_positions = [index for index, value in enumerate(row)
                       if isinstance(value[0], dict) and value[0].get("type") == "image"]
    videos = [(ordinal, position) for ordinal, position in enumerate(image_positions)
              if row[position][0].get("minimax_video_block", False)]
    if len(videos) != len(frame_pairs) or any(not pairs for pairs in frame_pairs):
        raise ValueError("Temporal frame pairs must match canonical video blocks.")
    return row, image_positions, videos


def _temporal_lane_tokens(tokens, videos, frame_pairs, lane, prepare_pair_callback):
    row = list(tokens["qwen3vl_32b"][0])
    for (_, position), pairs in zip(videos, frame_pairs):
        if lane < len(pairs):
            value = row[position]
            entry = dict(value[0])
            entry["data"] = prepare_pair_callback(pairs[lane])
            row[position] = (entry, *value[1:])
    return {"qwen3vl_32b": [row]}


def _temporal_tag_spans(conditioning, metadata, row, image_positions, token_spans_callback):
    tags = metadata.get("minimax_token_tags")
    if (conditioning.ndim != 3 or not torch.is_tensor(tags)
            or tags.ndim != 1 or len(tags) != conditioning.shape[1]):
        raise ValueError("Temporal conditioning requires matching MiniMax H3 tags.")
    mapped = token_spans_callback(row, conditioning)
    if len(mapped) != len(row):
        raise ValueError("Temporal token spans do not match canonical token entries.")
    spans = []
    for position in image_positions:
        start, end = mapped[position]
        if (not 0 < start < end < len(tags)
                or not torch.all(tags[start - 1:end + 1] == 0)):
            raise ValueError("Temporal mapped visual spans require native vision wrapper tags.")
        spans.append((start, end))
    return spans


def _temporal_processed_sources(processed, videos, image_count):
    embeds, _, _, info = processed
    images = [entry for entry in info if entry.get("type") == "image"]
    if embeds.ndim != 3 or embeds.shape[0] != 1 or len(images) != image_count:
        raise ValueError("Temporal processing changed the canonical image layout.")
    blocks = []
    for ordinal, _ in videos:
        entry = images[ordinal]
        start, size = entry["index"], entry["size"]
        if start < 0 or size < 1 or start + size > embeds.shape[1]:
            raise ValueError("Temporal visual span exceeds processed embeddings.")
        extra = entry.get("extra") or {}
        grid = extra.get("grid")
        values = grid.reshape(-1).tolist() if torch.is_tensor(grid) else list(grid or [])
        if len(values) != 3 or values[0] != 1:
            raise ValueError("Temporal source requires one native two-frame visual grid.")
        spatial_grid = (int(values[1]) // 2, int(values[2]) // 2)
        if spatial_grid[0] * spatial_grid[1] != size:
            raise ValueError("Temporal visual grid does not match expanded tokens.")
        layers = extra.get("deepstack")
        blocks.append((embeds[0, start:start + size].clone(), spatial_grid,
                       None if layers is None else [layer.clone() for layer in layers]))
    return blocks




def encode_temporal_conditioning(
    clip, canonical_tokens, frame_pairs, prepare_pair_callback, *, token_fusion,
    fusion_callback, encode_tokens_callback, active_clip_model_callback=None,
    encode_preprocessed_callback=None, visual_context_callback=None,
    video_grid_callback=None, token_spans_callback=None,
    cache=None,
):
    """Encode bounded full lanes, fusing only corresponding video interiors.

    Callbacks own existing encoding, geometry and fusion settings. Alternatives
    preserve canonical token tuples and all metadata except video-entry data.
    Rank-wise processing retains copied block features, never full alternate
    sequences. Post mode encodes each lane; pre mode runs Qwen once per schedule.
    """
    row, image_positions, videos = _temporal_token_layout(canonical_tokens, frame_pairs)
    lane_count = max((len(pairs) for pairs in frame_pairs), default=1)
    if lane_count == 1:
        return encode_tokens_callback(canonical_tokens)
    if not token_fusion:
        canonical = encode_tokens_callback(canonical_tokens)
        if not canonical:
            raise ValueError("Temporal encoding produced no conditioning schedules.")
        canonical_spans = [_temporal_tag_spans(tensor, metadata, row, image_positions, token_spans_callback)
                           for tensor, metadata in canonical]
        sources = [[[tensor[:, spans[ordinal][0]:spans[ordinal][1]].clone()]
                    for ordinal, _ in videos]
                   for (tensor, _), spans in zip(canonical, canonical_spans)]
        for lane in range(1, lane_count):
            lane_tokens = _temporal_lane_tokens(
                canonical_tokens, videos, frame_pairs, lane, prepare_pair_callback,
            )
            encoded = encode_tokens_callback(lane_tokens)
            if len(encoded) != len(canonical):
                raise ValueError("Temporal lanes produced different conditioning schedules.")
            for schedule, ((base, base_meta), (tensor, metadata)) in enumerate(zip(canonical, encoded)):
                if any(base_meta.get(key) != metadata.get(key) for key in ("clip_start_percent", "clip_end_percent")):
                    raise ValueError("Temporal lanes produced different conditioning schedule boundaries.")
                spans = _temporal_tag_spans(
                    tensor, metadata, lane_tokens["qwen3vl_32b"][0], image_positions, token_spans_callback,
                )
                if tensor.shape != base.shape or spans != canonical_spans[schedule]:
                    raise ValueError("Temporal lanes changed canonical conditioning layout.")
                for block, (ordinal, _) in enumerate(videos):
                    if lane < len(frame_pairs[block]):
                        start, end = spans[ordinal]
                        sources[schedule][block].append(tensor[:, start:end].clone())
            del encoded, tensor
        output = []
        for schedule, ((tensor, metadata), spans) in enumerate(zip(canonical, canonical_spans)):
            fused = tensor.clone()
            for block, (ordinal, position) in enumerate(videos):
                block_sources = sources[schedule][block]
                if len(block_sources) == 1:
                    continue
                start, end = spans[ordinal]
                grid = video_grid_callback(row[position][0]["data"], end - start)
                if grid[0] * grid[1] != end - start:
                    raise ValueError("Temporal post-encoder grid does not match visual span.")
                for batch in range(tensor.shape[0]):
                    merged, _ = fusion_callback(
                        [source[batch] for source in block_sources], [grid] * len(block_sources), None,
                    )
                    if merged.shape != fused[batch, start:end].shape:
                        raise ValueError("Temporal fusion changed canonical conditioning shape.")
                    fused[batch, start:end] = merged
            output.append([fused, metadata.copy()])
        return output

    clip.cond_stage_model.reset_clip_options()
    if clip.layer_idx is not None:
        clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})
    clip.load_model(canonical_tokens)
    device = clip.patcher.load_device
    clip.cond_stage_model.set_clip_options({"execution_device": device})
    clip_model = active_clip_model_callback(clip)

    def encode_once():
        processed = clip_model.process_tokens([[value[0] for value in row]], device)
        embeds, attention, num_tokens, info = processed
        canonical_blocks = _temporal_processed_sources(processed, videos, len(image_positions))
        block_sources = [[block] for block in canonical_blocks]
        for lane in range(1, lane_count):
            lane_tokens = _temporal_lane_tokens(
                canonical_tokens, videos, frame_pairs, lane, prepare_pair_callback,
            )
            alternate = clip_model.process_tokens(
                [[value[0] for value in lane_tokens["qwen3vl_32b"][0]]], device,
            )
            blocks = _temporal_processed_sources(alternate, videos, len(image_positions))
            for block, value in enumerate(blocks):
                if lane < len(frame_pairs[block]):
                    block_sources[block].append(value)
            del alternate, blocks
        fused = embeds.clone()
        fused_info = [dict(entry) for entry in info]
        images = [entry for entry in fused_info if entry.get("type") == "image"]
        for (ordinal, _), values in zip(videos, block_sources):
            if len(values) == 1:
                continue
            source_layers = [value[2] for value in values]
            counts = [None if layers is None else len(layers) for layers in source_layers]
            if len(set(counts)) != 1:
                raise ValueError("Temporal sources produced different DeepStack layers.")
            layers = (None if counts[0] is None else
                      [[source[layer] for source in source_layers] for layer in range(counts[0])])
            merged, deepstack = fusion_callback(
                [value[0] for value in values], [value[1] for value in values], layers,
            )
            entry = images[ordinal]
            start, size = entry["index"], entry["size"]
            if merged.shape != embeds[0, start:start + size].shape:
                raise ValueError("Temporal fusion changed canonical embedding shape.")
            fused[0, start:start + size] = merged
            entry["extra"] = dict(entry.get("extra") or {})
            if deepstack is not None:
                entry["extra"]["deepstack"] = deepstack
        with visual_context_callback():
            return encode_preprocessed_callback(
                clip_model, fused, attention, num_tokens, fused_info,
                **({"cache": cache, "hooks": clip.patcher.forced_hooks} if cache is not None else {}),
            )

    hooks = clip.patcher.forced_hooks
    schedules = hooks.get_hooks_for_clip_schedule() if hooks is not None and clip.use_clip_schedule else None
    output = []
    with comfy.model_management.cuda_device_context(device):
        if schedules is None:
            tensor, metadata = encode_once()
            clip.add_hooks_to_dict(metadata)
            output.append([tensor, metadata])
        else:
            hooks.reset()
            clip.patcher.patch_hooks(None)
            try:
                for time_range, scheduled_hooks in schedules:
                    for hook, keyframe in scheduled_hooks:
                        hook.hook_keyframe._current_keyframe = keyframe
                    clip.patcher.patch_hooks(hooks)
                    tensor, metadata = encode_once()
                    metadata["clip_start_percent"], metadata["clip_end_percent"] = time_range
                    clip.add_hooks_to_dict(metadata)
                    output.append([tensor, metadata])
            finally:
                hooks.reset()
    return output
