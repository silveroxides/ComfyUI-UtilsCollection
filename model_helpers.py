import json

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils


SAM3_WORKING_SIZE = 1008
SAM3_EDGE_PADDING = 32


def _extract_text_prompts(conditioning, device, dtype):
    cond_meta = conditioning[0][1]
    multi = cond_meta.get("sam3_multi_cond")
    prompts = []
    if multi is not None:
        for entry in multi:
            embedding = entry["cond"].to(device=device, dtype=dtype)
            mask = entry["attention_mask"]
            if mask is None:
                mask = torch.ones(embedding.shape[:2], dtype=torch.int64, device=device)
            else:
                mask = mask.to(device)
            prompts.append((embedding, mask, entry.get("max_detections", 1)))
    else:
        embedding = conditioning[0][0].to(device=device, dtype=dtype)
        mask = cond_meta.get("attention_mask")
        if mask is None:
            mask = torch.ones(embedding.shape[:2], dtype=torch.int64, device=device)
        else:
            mask = mask.to(device)
        prompts.append((embedding, mask, 1))
    return prompts


def _sam3_axis_ranges(length, count, tile_size):
    if count == 1:
        return [(0, length)]
    center_start = (length - tile_size) / 2
    stride = length / count
    offsets = [0]
    for offset in range(1, count // 2 + 1):
        offsets.extend((-offset, offset))
    return [
        (
            max(0, round(center_start + offset * stride)),
            min(length, round(center_start + offset * stride) + tile_size),
        )
        for offset in offsets
    ]


def _sam3_long_axis_tile_count(longest, shortest):
    count = max(3, -(-longest // shortest))
    return count if count % 2 else count + 1


def _sam3_tile_records(height, width, edge_padding):
    if height <= width:
        rows, columns = 1, _sam3_long_axis_tile_count(width, height)
    else:
        rows, columns = _sam3_long_axis_tile_count(height, width), 1
    padded_height = height + edge_padding * 2
    padded_width = width + edge_padding * 2
    tile_size = min(padded_height, padded_width)
    return [
        (y0, y1, x0, x1)
        for y0, y1 in _sam3_axis_ranges(padded_height, rows, tile_size)
        for x0, x1 in _sam3_axis_ranges(padded_width, columns, tile_size)
    ]


def _padded_sam3_tile(image, y0, y1, x0, x1, device, dtype):
    tile = image[:, :, y0:y1, x0:x1]
    tile_height, tile_width = tile.shape[-2:]
    square_size = max(tile_height, tile_width)
    pad_height = square_size - tile_height
    pad_width = square_size - tile_width
    pad_top = pad_height if y0 == 0 else 0 if y1 == image.shape[-2] else pad_height // 2
    pad_left = pad_width if x0 == 0 else 0 if x1 == image.shape[-1] else pad_width // 2
    padded = F.pad(
        tile,
        (pad_left, square_size - tile_width - pad_left, pad_top, square_size - tile_height - pad_top),
        mode="replicate",
    )
    if square_size > SAM3_WORKING_SIZE:
        frame = comfy.utils.common_upscale(padded, SAM3_WORKING_SIZE, SAM3_WORKING_SIZE, "lanczos", crop="disabled")
    else:
        frame = F.interpolate(padded, size=(SAM3_WORKING_SIZE, SAM3_WORKING_SIZE), mode="bilinear", align_corners=False)
    return frame.to(device=device, dtype=dtype), (tile_height, tile_width, square_size, pad_top, pad_left)


def _padded_sam3_image(image, edge_padding):
    image = image[..., :3].movedim(-1, 1)
    return F.pad(image, (edge_padding,) * 4, mode="replicate") if edge_padding else image


def _refine_mask(sam3_model, frame, coarse_mask, iterations):
    mask_logit = coarse_mask[None, None]
    for _ in range(iterations):
        mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
    return (mask_logit[0] > 0).float()


def detect_sam3(model, image, conditioning=None, bboxes=None, positive_coords=None, negative_coords=None, threshold=0.5, refine_iterations=2, individual_masks=False, edge_padding=True):
    batch, height, width, _ = image.shape

    def boxes_to_tensor(box_list):
        return torch.tensor([[
            [(box["x"] + box["width"] / 2) / width, (box["y"] + box["height"] / 2) / height, box["width"] / width, box["height"] / height]
            for box in box_list
        ]], dtype=torch.float32)

    per_frame_boxes = None
    if bboxes is not None:
        if isinstance(bboxes, dict):
            per_frame_boxes = [boxes_to_tensor([bboxes])] * batch
        elif isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], list):
            per_frame_boxes = [boxes_to_tensor(frame_boxes) if frame_boxes else None for frame_boxes in bboxes]
            per_frame_boxes.extend([per_frame_boxes[-1] if per_frame_boxes else None] * max(0, batch - len(per_frame_boxes)))
        elif isinstance(bboxes, list) and bboxes:
            per_frame_boxes = [boxes_to_tensor(bboxes)] * batch

    positive_points = json.loads(positive_coords) if positive_coords else []
    negative_points = json.loads(negative_coords) if negative_coords else []
    comfy.model_management.load_model_gpu(model)
    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()
    sam3_model = model.model.diffusion_model
    point_inputs = None
    if positive_points or negative_points:
        coordinates = [[point["x"] / width * SAM3_WORKING_SIZE, point["y"] / height * SAM3_WORKING_SIZE] for point in positive_points + negative_points]
        point_inputs = {
            "point_coords": torch.tensor([coordinates], dtype=dtype, device=device),
            "point_labels": torch.tensor([[1] * len(positive_points) + [0] * len(negative_points)], dtype=torch.int32, device=device),
        }
    conditionings = _extract_text_prompts(conditioning, device, dtype) if conditioning else []
    tile_edge_padding = SAM3_EDGE_PADDING if edge_padding else 0
    image_in = None
    if point_inputs is not None or (per_frame_boxes is not None and not conditionings):
        image_in = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), SAM3_WORKING_SIZE, SAM3_WORKING_SIZE, "bilinear", crop="disabled")
    tiled_images = None
    if conditionings:
        tiled_images = _padded_sam3_image(image, tile_edge_padding)
    all_bboxes, all_masks = [], []
    progress = comfy.utils.ProgressBar(batch)
    for index in range(batch):
        frame = image_in[index:index + 1].to(device=device, dtype=dtype) if image_in is not None else None
        frame_boxes = per_frame_boxes[index].to(device=device, dtype=dtype) if per_frame_boxes is not None and per_frame_boxes[index] is not None else None
        bbox_dicts, frame_masks = [], []
        if point_inputs is not None:
            mask_logit = sam3_model.forward_segment(frame, point_inputs=point_inputs)
            for _ in range(max(0, refine_iterations - 1)):
                mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
            frame_masks.append((F.interpolate(mask_logit, size=(height, width), mode="bilinear", align_corners=False)[0] > 0).float())
        if frame_boxes is not None and not conditionings:
            for cx, cy, box_width, box_height in frame_boxes[0].tolist():
                box = torch.tensor([[[(cx - box_width / 2) * SAM3_WORKING_SIZE, (cy - box_height / 2) * SAM3_WORKING_SIZE], [(cx + box_width / 2) * SAM3_WORKING_SIZE, (cy + box_height / 2) * SAM3_WORKING_SIZE]]], device=device, dtype=dtype)
                mask_logit = sam3_model.forward_segment(frame, box_inputs=box)
                for _ in range(max(0, refine_iterations - 1)):
                    mask_logit = sam3_model.forward_segment(frame, mask_inputs=mask_logit)
                frame_masks.append((F.interpolate(mask_logit, size=(height, width), mode="bilinear", align_corners=False)[0] > 0).float())
        for embedding, text_mask, max_detections in conditionings:
            candidates = []
            for y0, y1, x0, x1 in _sam3_tile_records(height, width, tile_edge_padding):
                tile, tile_geometry = _padded_sam3_tile(tiled_images[index:index + 1], y0, y1, x0, x1, device, dtype)
                results = sam3_model(tile, text_embeddings=embedding, text_mask=text_mask, threshold=threshold, orig_size=(SAM3_WORKING_SIZE, SAM3_WORKING_SIZE))
                scores = results["scores"][0].sigmoid()
                kept = scores > threshold
                tile_boxes, tile_scores, tile_masks = results["boxes"][0][kept], scores[kept], results["masks"][0][kept]
                order = tile_scores.argsort(descending=True)[:max_detections]
                for box, score, mask in zip(tile_boxes[order], tile_scores[order], tile_masks[order]):
                    candidates.append((float(score), box, mask, tile, tile_geometry, y0, y1, x0, x1))

            for score, box, mask, tile, tile_geometry, y0, y1, x0, x1 in candidates:
                tile_height, tile_width, square_size, pad_top, pad_left = tile_geometry
                box_x0 = max(0.0, min(float(box[0]) * square_size / SAM3_WORKING_SIZE - pad_left, tile_width))
                box_y0 = max(0.0, min(float(box[1]) * square_size / SAM3_WORKING_SIZE - pad_top, tile_height))
                box_x1 = max(0.0, min(float(box[2]) * square_size / SAM3_WORKING_SIZE - pad_left, tile_width))
                box_y1 = max(0.0, min(float(box[3]) * square_size / SAM3_WORKING_SIZE - pad_top, tile_height))
                if box_x1 <= box_x0 or box_y1 <= box_y0:
                    continue
                global_x0, global_y0 = x0 + box_x0, y0 + box_y0
                global_x1, global_y1 = x0 + box_x1, y0 + box_y1
                clipped_x0 = max(tile_edge_padding, global_x0)
                clipped_y0 = max(tile_edge_padding, global_y0)
                clipped_x1 = min(width + tile_edge_padding, global_x1)
                clipped_y1 = min(height + tile_edge_padding, global_y1)
                if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                    continue
                bbox_dicts.append({"x": clipped_x0 - tile_edge_padding, "y": clipped_y0 - tile_edge_padding, "width": clipped_x1 - clipped_x0, "height": clipped_y1 - clipped_y0, "score": score})
                refined = _refine_mask(sam3_model, tile, mask, refine_iterations)
                refined = F.interpolate(refined[None], size=(square_size, square_size), mode="bilinear", align_corners=False)[0]
                refined = refined[:, pad_top:pad_top + tile_height, pad_left:pad_left + tile_width]
                source_y0 = max(tile_edge_padding, y0)
                source_y1 = min(height + tile_edge_padding, y1)
                source_x0 = max(tile_edge_padding, x0)
                source_x1 = min(width + tile_edge_padding, x1)
                full_mask = torch.zeros(1, height, width, device=device, dtype=dtype)
                full_mask[:, source_y0 - tile_edge_padding:source_y1 - tile_edge_padding, source_x0 - tile_edge_padding:source_x1 - tile_edge_padding] = refined[:, source_y0 - y0:source_y1 - y0, source_x0 - x0:source_x1 - x0]
                frame_masks.append(full_mask)
        all_bboxes.append(bbox_dicts)
        if frame_masks:
            combined = torch.cat(frame_masks)
            all_masks.append(combined if individual_masks else (combined > 0).any(dim=0).float())
        elif individual_masks:
            all_masks.append(torch.zeros(0, height, width, device=comfy.model_management.intermediate_device()))
        else:
            all_masks.append(torch.zeros(height, width, device=comfy.model_management.intermediate_device()))
        progress.update(1)
    intermediate_device = comfy.model_management.intermediate_device()
    masks = [mask.to(intermediate_device) for mask in all_masks]
    return (torch.cat(masks) if individual_masks else torch.stack(masks), all_bboxes)
