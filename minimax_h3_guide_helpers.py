"""MiniMax H3 prompt boundaries and independently encoded guide insertion."""

import torch


LAYOUT_KEY = "uc_minimax_h3_vlm_layout"
LAYOUT_VERSION = 1
_LAYOUT_FIELDS = {"version", "sequence_length", "prompt_start"}


def _validate_tensor_and_tags(cond_tensor, tags):
    if not torch.is_tensor(cond_tensor) or cond_tensor.ndim != 3:
        raise ValueError("MiniMax H3 conditioning must have shape [batch, sequence, features].")
    if any(size < 1 for size in cond_tensor.shape):
        raise ValueError("MiniMax H3 conditioning dimensions must be nonempty.")
    if not cond_tensor.is_floating_point():
        raise ValueError("MiniMax H3 conditioning must use a floating-point dtype.")
    if not torch.is_tensor(tags) or tags.ndim != 1 or tags.shape[0] != cond_tensor.shape[1]:
        raise ValueError("MiniMax H3 token tags must be one-dimensional and match the sequence length.")
    if tags.dtype != torch.long:
        raise ValueError("MiniMax H3 token tags must use torch.long.")


def build_layout(cond_tensor, tags, prompt_start):
    """Build scalar metadata from a known final embedding boundary, without caching tensors."""
    _validate_tensor_and_tags(cond_tensor, tags)
    if type(prompt_start) is not int or not 0 <= prompt_start <= cond_tensor.shape[1]:
        raise ValueError("MiniMax H3 prompt boundary must be an integer inside the sequence.")
    return {
        "version": LAYOUT_VERSION,
        "sequence_length": cond_tensor.shape[1],
        "prompt_start": prompt_start,
    }


def layout_from_token_boundary(
    cond_tensor, tags, token_entries, prompt_token_start, *, empty_prompt_pad=False
):
    """Resolve the prompt suffix in the final sequence without expanding media again.

    The caller supplies the actual tokenizer entries and the first prompt entry
    from presentation assembly (after any weighted-prompt cleaning). Media can
    expand arbitrarily before that boundary. The text suffix must contain only
    literal IDs or textual embedding tensors. A wholly empty native H3 input has
    one pad token: explicitly mark that case to insert before its pad. An empty
    prompt following media instead has an empty suffix and inserts at the end.
    """
    if type(prompt_token_start) is not int or not 0 <= prompt_token_start <= len(token_entries):
        raise ValueError("MiniMax H3 prompt token boundary is outside the token entries.")
    if empty_prompt_pad:
        if (
            len(token_entries) != 1
            or not isinstance(token_entries[0], (tuple, list))
            or not token_entries[0]
            or type(token_entries[0][0]) is not int
            or token_entries[0][0] != 151643
            or prompt_token_start != 0
        ):
            raise ValueError("MiniMax H3 empty-input layout requires its single native pad token.")
        layout = build_layout(cond_tensor, tags, 0)
        if layout["sequence_length"] != 1:
            raise ValueError("MiniMax H3 empty-input pad must encode to one token.")
        return layout

    suffix_length = 0
    for entry in token_entries[prompt_token_start:]:
        if not isinstance(entry, (tuple, list)) or not entry:
            raise ValueError("MiniMax H3 prompt suffix contains an invalid token entry.")
        value = entry[0]
        if type(value) is int:
            suffix_length += 1
        elif torch.is_tensor(value) and value.ndim >= 1 and value.shape[-1] > 0 and value.numel() > 0:
            suffix_length += value.numel() // value.shape[-1]
        else:
            raise ValueError("MiniMax H3 prompt suffix must contain text IDs or embedding tensors.")
    _validate_tensor_and_tags(cond_tensor, tags)
    return build_layout(cond_tensor, tags, cond_tensor.shape[1] - suffix_length)


def _validated_entry(entry, *, require_layout):
    if not isinstance(entry, (tuple, list)) or len(entry) != 2 or not isinstance(entry[1], dict):
        raise ValueError("MiniMax H3 conditioning entries must contain a tensor and metadata dictionary.")
    tensor, metadata = entry
    tags = metadata.get("minimax_token_tags")
    _validate_tensor_and_tags(tensor, tags)
    if not require_layout:
        return tensor, metadata, tags, None
    layout = metadata.get(LAYOUT_KEY)
    if (
        not isinstance(layout, dict)
        or set(layout) != _LAYOUT_FIELDS
        or any(type(layout[key]) is not int for key in _LAYOUT_FIELDS)
        or layout["version"] != LAYOUT_VERSION
    ):
        raise ValueError("MiniMax H3 guide requires supported VLM layout metadata; re-encode the conditioning.")
    if layout["sequence_length"] != tensor.shape[1]:
        raise ValueError("MiniMax H3 VLM layout is stale; re-encode the conditioning.")
    build_layout(tensor, tags, layout["prompt_start"])
    return tensor, metadata, tags, layout["prompt_start"]




def splice_conditioning(base_conditioning, guide_conditioning):
    """Insert one complete guide encoding before every compatible base prompt.

    Base schedule and auxiliary metadata remain authoritative. A guide batch of
    one broadcasts across the base batch; multiple guide schedules are rejected.
    Input tensors and dictionaries are never modified.
    """
    if not isinstance(base_conditioning, (tuple, list)) or not base_conditioning:
        raise ValueError("MiniMax H3 guide requires nonempty base conditioning.")
    if not isinstance(guide_conditioning, (tuple, list)) or len(guide_conditioning) != 1:
        raise ValueError("MiniMax H3 guide encoding must produce exactly one conditioning schedule.")
    guide_tensor, _, guide_tags, _ = _validated_entry(guide_conditioning[0], require_layout=False)
    output = []
    for entry in base_conditioning:
        base_tensor, metadata, base_tags, boundary = _validated_entry(entry, require_layout=True)
        if guide_tensor.shape[2] != base_tensor.shape[2]:
            raise ValueError("MiniMax H3 guide and base conditioning feature dimensions must match.")
        if guide_tensor.shape[0] not in (1, base_tensor.shape[0]):
            raise ValueError("MiniMax H3 guide batch must be one or match the base conditioning batch.")
        guide = guide_tensor.to(device=base_tensor.device, dtype=base_tensor.dtype)
        if guide.shape[0] != base_tensor.shape[0]:
            guide = guide.expand(base_tensor.shape[0], -1, -1)
        tensor = torch.cat((base_tensor[:, :boundary], guide, base_tensor[:, boundary:]), dim=1)
        tags = torch.cat((base_tags[:boundary], guide_tags.to(device=base_tags.device), base_tags[boundary:]))
        updated = metadata.copy()
        updated["minimax_token_tags"] = tags
        updated[LAYOUT_KEY] = build_layout(tensor, tags, boundary + guide.shape[1])
        output.append([tensor, updated])
    return output
