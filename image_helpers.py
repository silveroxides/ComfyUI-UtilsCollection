from __future__ import annotations

import io
import colorsys
import comfy.model_management
import comfy.utils
import math
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterator, Sequence

import av
import cv2
import numpy as np
import torch
import torchaudio
from nodes import MAX_RESOLUTION

from .helper_functions import ASPECT_RATIOS, resize_nchw
from .parameter_helpers import h3_video_length_from_seconds, select_video_resolution


BODY_LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
              (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
              (0, 15), (15, 17), (2, 16), (5, 17))
PAF_CHANNELS = ((12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25),
                (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (28, 29),
                (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27))
BODY_COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
               (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
               (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
               (255, 0, 255), (255, 0, 170), (255, 0, 85))
HAND_LIMBS = tuple(edge for start in (1, 5, 9, 13, 17)
                   for edge in ((0, start), (start, start + 1), (start + 1, start + 2), (start + 2, start + 3)))


def resize_pose_map(image, width, height):
    """Resize feature maps in channel groups supported by OpenCV's area kernel."""
    if image.shape[:2] == (height, width):
        return image
    interpolation = cv2.INTER_AREA if width + height < sum(image.shape[:2]) else cv2.INTER_LANCZOS4
    if interpolation == cv2.INTER_AREA and image.ndim == 3 and image.shape[2] > 4:
        return np.concatenate([resize_pose_map(image[..., start:start + 4], width, height)
                               for start in range(0, image.shape[2], 4)], axis=2)
    result = cv2.resize(image, (int(width), int(height)), interpolation=interpolation)
    return result[..., None] if result.ndim == 2 and image.ndim == 3 else result


def score_limb_pairs(first, second, paf, channels, limb_threshold=0.05, limb_support=0.8):
    """Score every candidate pair using the original ten-point PAF criterion."""
    if not len(first) or not len(second):
        return []
    delta = second[None, :, :2] - first[:, None, :2]
    distance = np.maximum(np.linalg.norm(delta, axis=-1), 0.001)
    direction = delta / distance[..., None]
    fraction = np.linspace(0, 1, 10)
    samples = first[:, None, None, :2] + delta[:, :, None, :] * fraction[None, None, :, None]
    points = np.rint(samples).astype(np.intp)
    x, y = points[..., 0], points[..., 1]
    projection = paf[y, x, channels[0]] * direction[..., 0, None] + paf[y, x, channels[1]] * direction[..., 1, None]
    scores = projection.mean(axis=-1) + np.minimum(0.5 * paf.shape[0] / distance - 1, 0)
    valid = (np.count_nonzero(projection > limb_threshold, axis=-1) > limb_support * 10) & (scores > 0)
    rows, columns = np.nonzero(valid)
    order = np.argsort(-scores[rows, columns], kind="stable")
    used_first, used_second, matches = set(), set(), []
    for index in order:
        i, j = int(rows[index]), int(columns[index])
        if i not in used_first and j not in used_second:
            used_first.add(i)
            used_second.add(j)
            matches.append((int(first[i, 3]), int(second[j, 3]), float(scores[i, j])))
    return matches


def decode_body(heatmap, paf, body_threshold=0.1, limb_threshold=0.05, limb_support=0.8, min_body_parts=4, min_body_score=0.4):
    smoothed = cv2.GaussianBlur(heatmap[..., :18], (25, 25), 3, borderType=cv2.BORDER_REFLECT)
    padded = np.pad(smoothed, ((1, 1), (1, 1), (0, 0)), mode="constant")
    peaks = ((smoothed > body_threshold) & (smoothed >= padded[:-2, 1:-1]) & (smoothed >= padded[2:, 1:-1])
             & (smoothed >= padded[1:-1, :-2]) & (smoothed >= padded[1:-1, 2:]))
    candidates, by_part = [], []
    for part in range(18):
        y, x = np.nonzero(peaks[..., part])
        entries = np.column_stack((x, y, heatmap[y, x, part], np.arange(len(candidates), len(candidates) + len(x))))
        candidates.extend(entries)
        by_part.append(entries)
    if not candidates:
        return []
    candidate = np.asarray(candidates)
    people = []
    for limb, ((a, b), channels) in enumerate(zip(BODY_LIMBS, PAF_CHANNELS)):
        for first, second, score in score_limb_pairs(by_part[a], by_part[b], paf, channels, limb_threshold, limb_support):
            owners = [i for i, person in enumerate(people) if person[a] == first or person[b] == second]
            if len(owners) == 1:
                person = people[owners[0]]
                if person[b] != second:
                    person[b] = second
                    person[-1] += 1
                    person[-2] += candidate[second, 2] + score
            elif len(owners) >= 2:
                i, j = owners[:2]
                first_person, second_person = people[i], people[j]
                if not np.any((first_person[:18] >= 0) & (second_person[:18] >= 0)):
                    first_person[:18] += second_person[:18] + 1
                    first_person[-2:] += second_person[-2:]
                    first_person[-2] += score
                    people.pop(j)
                else:
                    first_person[b] = second
                    first_person[-1] += 1
                    first_person[-2] += candidate[second, 2] + score
            elif limb < 17:
                person = np.full(20, -1.0)
                person[a], person[b] = first, second
                person[-1] = 2
                person[-2] = candidate[first, 2] + candidate[second, 2] + score
                people.append(person)
    return [
        [None if index < 0 else tuple(candidate[int(index), :2]) for index in person[:18]]
        for person in people if person[-1] >= min_body_parts and person[-2] / person[-1] >= min_body_score
    ]


def hand_boxes(body, width, height):
    boxes = []
    for side, joints in (("hand_left_keypoints_2d", (5, 6, 7)), ("hand_right_keypoints_2d", (2, 3, 4))):
        if any(body[joint] is None for joint in joints):
            continue
        shoulder, elbow, wrist = (np.asarray(body[joint]) for joint in joints)
        center = wrist + 0.33 * (wrist - elbow)
        size = 1.5 * max(np.linalg.norm(wrist - elbow), 0.9 * np.linalg.norm(elbow - shoulder))
        x, y = np.maximum(center - size / 2, 0)
        size = min(size, width - x, height - y)
        if size >= 20:
            boxes.append((side, (int(x), int(y), int(size))))
    return boxes


def face_box(body, width, height):
    if body[0] is None:
        return None
    head = np.asarray(body[0])
    radius = max((max(abs(head - body[index])) * factor
                  for index, factor in ((14, 3), (15, 3), (16, 1.5), (17, 1.5))
                  if body[index] is not None), default=0)
    x, y = np.maximum(head - radius, 0)
    size = min(radius * 2, width - x, height - y)
    return (int(x), int(y), int(size)) if size >= 20 else None


def decode_hand(heatmap, box, width, height, threshold=0.05):
    x, y, size = box
    result = []
    for part in range(21):
        raw = heatmap[..., part]
        blurred = cv2.GaussianBlur(raw, (25, 25), 3, borderType=cv2.BORDER_REFLECT)
        count, labels = cv2.connectedComponents((blurred > threshold).astype(np.uint8), connectivity=8)
        if count <= 1:
            result.extend((-1.0, -1.0, 1.0))
            continue
        scores = np.bincount(labels.ravel(), weights=raw.ravel(), minlength=count)
        component = int(np.argmax(scores[1:])) + 1
        py, px = np.unravel_index(np.argmax(np.where(labels == component, raw, 0)), raw.shape)
        local_x, local_y = int(px * size / raw.shape[1]), int(py * size / raw.shape[0])
        result.extend(((x + local_x) / width if local_x else -1.0, (y + local_y) / height if local_y else -1.0, 1.0))
    return result


def decode_face(heatmap, box, width, height, threshold=0.05):
    x, y, size = box
    result = []
    for part in range(heatmap.shape[-1]):
        plane = heatmap[..., part]
        index = int(plane.argmax())
        if plane.flat[index] <= threshold:
            result.extend((0.0, 0.0, 0.0))
            continue
        py, px = divmod(index, plane.shape[1])
        local_x, local_y = px * size / plane.shape[1], py * size / plane.shape[0]
        result.extend(((x + local_x) / width if local_x else -1.0, (y + local_y) / height if local_y else -1.0, 1.0))
    return result if any(result[2::3]) else None


def encode_body(body, width, height):
    return [value for point in body for value in
            ((float(point[0]) / width, float(point[1]) / height, 1.0) if point is not None else (0.0, 0.0, 0.0))]


def pose_overlay_mask(pose_maps):
    """Select rendered pose pixels, not person boxes or body silhouettes."""
    return torch.any(pose_maps != 0, dim=-1).to(dtype=torch.float32)


def draw_pose_frame(people, height, width, draw_body=True, draw_hands=True, draw_face=True, scale_stick=False, drawing_scale=1.0):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    stick_scale = (1 if max(height, width) < 500 else min(2 + max(height, width) // 1000, 7)) if scale_stick else 1
    for person in people:
        if draw_body:
            points = np.asarray(person["pose_keypoints_2d"]).reshape(-1, 3)
            for (a, b), color in zip(BODY_LIMBS[:17], BODY_COLORS):
                if not points[a, 2] or not points[b, 2]:
                    continue
                start, end = points[[a, b], :2] * [width, height]
                center = (start + end) / 2
                delta = start - end
                polygon = cv2.ellipse2Poly(tuple(center.astype(int)), (int(np.linalg.norm(delta) / 2), max(1, round(4 * stick_scale * drawing_scale))),
                                          int(math.degrees(math.atan2(delta[1], delta[0]))), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, tuple(int(channel * 0.6) for channel in color))
            for point, color in zip(points, BODY_COLORS):
                if point[2]:
                    cv2.circle(canvas, tuple((point[:2] * [width, height]).astype(int)), max(1, round(4 * drawing_scale)), color, -1)
        if draw_hands:
            for name in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
                if not person.get(name):
                    continue
                points = np.asarray(person[name]).reshape(-1, 3)
                pixels = (points[:, :2] * [width, height]).astype(int)
                for index, (a, b) in enumerate(HAND_LIMBS):
                    if np.all(pixels[[a, b]] > 0):
                        color = tuple(channel * 255 for channel in colorsys.hsv_to_rgb(index / 20, 1, 1))
                        cv2.line(canvas, tuple(pixels[a]), tuple(pixels[b]), color, max(1, round(2 * drawing_scale)))
                for point in pixels:
                    if np.all(point > 0):
                        cv2.circle(canvas, tuple(point), max(1, round(4 * drawing_scale)), (0, 0, 255), -1)
        if draw_face and person.get("face_keypoints_2d"):
            points = np.asarray(person["face_keypoints_2d"]).reshape(-1, 3)
            for point in points:
                pixel = (point[:2] * [width, height]).astype(int)
                if np.all(pixel > 0):
                    cv2.circle(canvas, tuple(pixel), max(1, round(3 * drawing_scale)), (255, 255, 255), -1)
    return canvas


def overlay_pose_keypoints(images, documents, draw_body=True, draw_hands=True, draw_face=True, opacity=1.0, drawing_scale=1.0):
    if images.ndim != 4 or images.shape[-1] not in (3, 4) or not len(images):
        raise ValueError("Pose overlay requires a nonempty RGB or RGBA image batch.")
    if not isinstance(documents, list) or len(documents) not in (1, len(images)):
        raise ValueError("Supply one pose frame to reuse, or one pose frame per image.")
    if not 0 <= opacity <= 1 or drawing_scale <= 0:
        raise ValueError("Pose opacity must be 0–1 and drawing size must be positive.")
    height, width = images.shape[1:3]
    output = images.clone()
    masks = torch.empty((len(images), height, width), dtype=torch.float32, device=images.device)
    for index in range(len(images)):
        comfy.model_management.throw_exception_if_processing_interrupted()
        document = documents[0 if len(documents) == 1 else index]
        if "people" not in document:
            raise ValueError("Pose overlay requires human OpenPose/DWPose keypoints, not animal keypoints.")
        canvas = draw_pose_frame(document["people"], height, width, draw_body, draw_hands, draw_face, drawing_scale=drawing_scale)
        pose = torch.from_numpy(canvas).to(device=images.device, dtype=images.dtype).div_(255)
        masks[index] = pose_overlay_mask(pose) * opacity
        alpha = masks[index].unsqueeze(-1)
        output[index, ..., :3] = images[index, ..., :3] * (1 - alpha) + pose * alpha
    return output, masks


def prepare_pose_frames(images, resolution):
    height, width = images.shape[1:3]
    factor = resolution / min(height, width) if resolution else 1.0
    target_height, target_width = max(1, round(height * factor)), max(1, round(width * factor))
    frames, body_inputs = [], []
    for image in images:
        pixels = (image[..., :3].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        resized = cv2.resize(pixels, (target_width, target_height), interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
        frame = np.pad(resized[..., ::-1], ((0, -target_height % 64), (0, -target_width % 64), (0, 0)), mode="edge")
        body_scale = 184 / frame.shape[0]
        body = resize_pose_map(frame, int(frame.shape[1] * body_scale), 184)
        body = np.pad(body, ((0, -body.shape[0] % 8), (0, -body.shape[1] % 8), (0, 0)), constant_values=128)
        frames.append(frame)
        body_inputs.append(body)
    return frames, body_inputs, (target_height, target_width)


def restore_body_maps(heatmap, paf, body_input, frame):
    height, width = frame.shape[:2]
    scaled_width = int(width * 184 / height)
    results = []
    for value in (heatmap, paf):
        expanded = resize_pose_map(value, body_input.shape[1], body_input.shape[0])
        results.append(resize_pose_map(expanded[:184, :scaled_width], width, height))
    return results


def estimate_hand_jobs(patcher, jobs, frames, people, batch_size, forward, threshold=0.05):
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        crops = [cv2.GaussianBlur(frames[frame][y:y + size, x:x + size], (0, 0), 0.8)
                 for frame, person, side, (x, y, size) in chunk]
        average = np.zeros((len(chunk), 128, 128, 22), dtype=np.float32)
        for side_length in (184, 368, 552, 736):
            inputs = [resize_pose_map(crop, side_length, side_length) for crop in crops]
            heatmaps = forward(patcher, inputs)
            for index, heatmap in enumerate(heatmaps):
                expanded = resize_pose_map(heatmap, side_length, side_length)
                average[index] += resize_pose_map(expanded, 128, 128) * 0.25
        for index, (frame, person, side, box) in enumerate(chunk):
            height, width = frames[frame].shape[:2]
            people[frame][person][side] = decode_hand(average[index], box, width, height, threshold)


def estimate_face_jobs(patcher, jobs, frames, people, batch_size, forward, threshold=0.05):
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        inputs = [resize_pose_map(frames[frame][y:y + size, x:x + size], 384, 384)
                  for frame, person, (x, y, size) in chunk]
        heatmaps = forward(patcher, inputs)
        for heatmap, (frame, person, box) in zip(heatmaps, chunk):
            height, width = frames[frame].shape[:2]
            # Match the face decoder's threshold/argmax at crop resolution.
            tensor = torch.from_numpy(heatmap).movedim(-1, 0)[None]
            heatmap = torch.nn.functional.interpolate(tensor, size=(box[2], box[2]), mode="bilinear", align_corners=True)[0].movedim(0, -1).numpy()
            people[frame][person]["face_keypoints_2d"] = decode_face(heatmap, box, width, height, threshold)


def run_openpose_batch(images, resolution=0, batch_size=4, detect_body=True, detect_hand=True, detect_face=True,
                       scale_stick=False, *, loader, forward, body_decoder=decode_body,
                       body_threshold=0.45, hand_threshold=0.45, face_threshold=0.45, limb_threshold=0.45,
                       limb_support=0.8, min_body_parts=4, min_body_score=0.4,
                       temporal_filter=False, temporal_radius=2, temporal_min_support=2, temporal_max_distance=0.1,
                       temporal_match_iou=0.3):
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] not in (3, 4):
        raise ValueError("OpenPose requires a nonempty IMAGE batch with RGB or RGBA channels.")
    if batch_size < 1 or resolution < 0:
        raise ValueError("OpenPose batch_size must be positive and resolution must be zero or positive.")
    models = {"body": loader("body")}
    if detect_hand:
        models["hand"] = loader("hand")
    if detect_face:
        models["face"] = loader("face")
    if not all(0 <= value <= 1 for value in (body_threshold, hand_threshold, face_threshold, limb_threshold, limb_support)):
        raise ValueError("OpenPose confidence/support thresholds must be between zero and one.")
    if min_body_parts < 1 or min_body_parts > 18 or min_body_score < 0:
        raise ValueError("OpenPose requires 1–18 minimum body parts and a nonnegative minimum body score.")
    output, documents, all_boxes = None, [], []
    progress = comfy.utils.ProgressBar(len(images))
    for start in range(0, len(images), batch_size):
        comfy.model_management.throw_exception_if_processing_interrupted()
        frames, body_inputs, (target_height, target_width) = prepare_pose_frames(images[start:start + batch_size], resolution)
        pafs, heatmaps = forward(models["body"], body_inputs)
        people, hand_jobs, face_jobs, chunk_boxes = [], [], [], []
        for frame_index, (frame, body_input, heatmap, paf) in enumerate(zip(frames, body_inputs, heatmaps, pafs)):
            heatmap, paf = restore_body_maps(heatmap, paf, body_input, frame)
            bodies = body_decoder(heatmap[:target_height, :target_width], paf[:target_height, :target_width],
                                  body_threshold=body_threshold, limb_threshold=limb_threshold, limb_support=limb_support,
                                  min_body_parts=min_body_parts, min_body_score=min_body_score)
            frame_people, frame_boxes = [], []
            for person_index, body in enumerate(bodies):
                points = np.asarray([point for point in body if point is not None])
                lower, upper = points.min(axis=0), points.max(axis=0)
                margin = max(float((upper - lower).max()) * 0.05, 1)
                frame_boxes.append(np.concatenate((lower - margin, upper + margin)))
                frame_people.append({
                    "pose_keypoints_2d": encode_body(body, target_width, target_height),
                    "hand_left_keypoints_2d": None, "hand_right_keypoints_2d": None, "face_keypoints_2d": None,
                })
                if detect_hand:
                    hand_jobs.extend((frame_index, person_index, side, box) for side, box in hand_boxes(body, target_width, target_height))
                if detect_face:
                    box = face_box(body, target_width, target_height)
                    if box:
                        face_jobs.append((frame_index, person_index, box))
            people.append(frame_people)
            chunk_boxes.append(frame_boxes)
        visible_frames = [frame[:target_height, :target_width] for frame in frames]
        if hand_jobs:
            estimate_hand_jobs(models["hand"], hand_jobs, visible_frames, people, batch_size, forward, hand_threshold)
        if face_jobs:
            estimate_face_jobs(models["face"], face_jobs, visible_frames, people, batch_size, forward, face_threshold)
        if output is None:
            output = torch.empty((len(images), target_height, target_width, 3), dtype=torch.float32, device="cpu")
        for index, frame_people in enumerate(people):
            documents.append({"people": frame_people, "canvas_height": target_height, "canvas_width": target_width})
            all_boxes.append(chunk_boxes[index])
            progress.update(1)
    if temporal_filter:
        documents = prune_temporal_pose_keypoints(documents, all_boxes, temporal_radius, temporal_min_support, temporal_max_distance, temporal_match_iou)
    for index, document in enumerate(documents):
        comfy.model_management.throw_exception_if_processing_interrupted()
        rendered = draw_pose_frame(document["people"], target_height, target_width, detect_body, detect_hand, detect_face, scale_stick)
        output[index] = torch.from_numpy(rendered).float().div_(255)
    return output, documents


def dwpose_detector_input(frame):
    height, width = frame.shape[:2]
    ratio = min(640 / height, 640 / width)
    resized = cv2.resize(frame, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return padded.astype(np.float32), ratio


def decode_dwpose_boxes(prediction, ratio, classes=(0,), detection_threshold=0.3, nms_threshold=0.45):
    grids, strides = [], []
    for stride in (8, 16, 32):
        y, x = np.mgrid[:640 // stride, :640 // stride]
        grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        strides.append(np.full((x.size, 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    if prediction.ndim != 2 or prediction.shape != (len(grid), 85):
        raise ValueError(f"Unsupported YOLOX output shape: {prediction.shape}; expected {(len(grid), 85)}.")
    results = []
    for category in classes:
        scores = prediction[:, 4] * prediction[:, 5 + category]
        chosen = scores > detection_threshold
        if not np.any(chosen):
            continue
        center = (prediction[chosen, :2] + grid[chosen]) * stride[chosen]
        size = np.exp(prediction[chosen, 2:4]) * stride[chosen]
        boxes = np.concatenate((center - size / 2, center + size / 2), axis=-1) / ratio
        results.append(suppress_pose_boxes(boxes, scores[chosen], nms_threshold))
    return np.concatenate(results) if results else np.empty((0, 4), dtype=np.float32)


def suppress_pose_boxes(boxes, scores, nms_threshold=0.45):
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    order, keep = np.argsort(scores)[::-1], []
    while len(order):
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        top_left = np.maximum(boxes[current, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection = np.maximum(bottom_right - top_left + 1, 0).prod(axis=-1)
        overlap = intersection / (area[current] + area[remaining] - intersection)
        order = remaining[overlap <= nms_threshold]
    return boxes[keep]


def prepare_dwpose_crop(frame, box, input_size=(288, 384)):
    box = np.asarray(box, dtype=np.float32)
    center = (box[:2] + box[2:]) * 0.5
    scale = (box[2:] - box[:2]) * 1.25
    if np.any(scale <= 0):
        raise ValueError("DWPose detector returned an empty person box.")
    aspect = input_size[0] / input_size[1]
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    factor = np.array(input_size, dtype=np.float32) / scale
    matrix = np.array([[factor[0], 0, input_size[0] / 2 - center[0] * factor[0]],
                       [0, factor[1], input_size[1] / 2 - center[1] * factor[1]]], dtype=np.float32)
    crop = cv2.warpAffine(frame, matrix, input_size, flags=cv2.INTER_LINEAR)
    crop = (crop.astype(np.float32) - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
    return crop, center, scale


def decode_dwpose_person(simcc_x, simcc_y, center, scale, width, height, keypoint_threshold=0.3):
    if simcc_x.shape != (133, 576) or simcc_y.shape != (133, 768):
        raise ValueError(f"Unsupported DWPose SimCC shapes: {simcc_x.shape}, {simcc_y.shape}.")
    points = np.stack((simcc_x.argmax(axis=-1), simcc_y.argmax(axis=-1)), axis=-1).astype(np.float32) / 2
    points = points / [288, 384] * scale + center - scale / 2
    scores = np.minimum(simcc_x.max(axis=-1), simcc_y.max(axis=-1))
    joints = np.column_stack((points, scores))
    neck = joints[[5, 6]].mean(axis=0)
    neck[2] = float(np.all(scores[[5, 6]] > keypoint_threshold))
    joints = np.insert(joints, 17, neck, axis=0)
    joints[[1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]] = joints[[17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]]
    def encode(part):
        if not np.any((part[:, 2] >= keypoint_threshold) & (part[:, 2] > 0)):
            return None
        return [value for x, y, score in part for value in
                ((float(x) / width, float(y) / height, 1.0) if score >= keypoint_threshold and score > 0 else (0.0, 0.0, 0.0))]
    face = np.concatenate((joints[24:92], joints[[14, 15]]))
    return {
        "pose_keypoints_2d": encode(joints[:18]) or [0.0] * 54,
        "face_keypoints_2d": encode(face),
        "hand_left_keypoints_2d": encode(joints[92:113]),
        "hand_right_keypoints_2d": encode(joints[113:134]),
    }


def decode_animal_pose(simcc_x, simcc_y, center, scale, keypoint_threshold=0.3):
    if simcc_x.shape != (17, 512) or simcc_y.shape != (17, 512):
        raise ValueError(f"Unsupported AP10K SimCC shapes: {simcc_x.shape}, {simcc_y.shape}")
    coordinates = np.stack((simcc_x.argmax(axis=-1), simcc_y.argmax(axis=-1)), axis=-1).astype(np.float32) / 2
    coordinates = coordinates / 256 * scale + center - scale / 2
    scores = np.minimum(simcc_x.max(axis=-1), simcc_y.max(axis=-1)).clip(0, 1)
    result = np.column_stack((coordinates, scores))
    result[scores < keypoint_threshold] = 0
    return result.tolist()


AP10K_LIMBS = ((1, 2), (2, 3), (1, 3), (3, 4), (4, 9), (9, 10), (10, 11), (4, 6),
              (6, 7), (7, 8), (4, 5), (5, 15), (15, 16), (16, 17), (5, 12), (12, 13), (13, 14))
AP10K_COLORS = ((255, 255, 255), (100, 255, 100), (150, 255, 255), (100, 50, 255), (50, 150, 200),
               (0, 255, 255), (0, 150, 0), (0, 0, 255), (0, 0, 150), (255, 50, 255), (255, 0, 255),
               (255, 0, 0), (150, 0, 0), (255, 255, 100), (0, 150, 0), (255, 255, 0), (150, 150, 150))


def draw_animal_pose_frame(animals, height, width):
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for animal in animals:
        for (a, b), color in zip(AP10K_LIMBS, AP10K_COLORS):
            if animal[a - 1][2] <= 0 or animal[b - 1][2] <= 0:
                continue
            start = tuple(int(value) for value in animal[a - 1][:2])
            end = tuple(int(value) for value in animal[b - 1][:2])
            cv2.line(output, start, end, color, 5)
    return output


def prune_temporal_pose_keypoints(documents, boxes, radius=2, min_support=2, max_distance=0.1, match_iou=0.3):
    """Zero unsupported joints; preserve person entries and use unfiltered neighbors."""
    if radius < 1 or min_support < 1 or max_distance <= 0 or not 0 <= match_iou <= 1:
        raise ValueError("Temporal filtering requires positive radius/support/distance and matching IoU between zero and one.")
    if len(documents) < 2:
        return documents
    matches = [[{} for _ in frame["people"]] for frame in documents]
    for first in range(len(documents)):
        for second in range(first + 1, min(len(documents), first + radius + 1)):
            a, b = np.asarray(boxes[first]), np.asarray(boxes[second])
            if not len(a) or not len(b):
                continue
            extent = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0)
            intersection = extent.prod(-1)
            area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
            area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
            overlap = intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-8)
            used_a, used_b = set(), set()
            for flat in np.argsort(-overlap.ravel(), kind="stable"):
                i, j = np.unravel_index(flat, overlap.shape)
                if overlap[i, j] < match_iou or overlap[i, j] <= 0:
                    break
                if i in used_a or j in used_b:
                    continue
                used_a.add(i)
                used_b.add(j)
                matches[first][i][second] = j
                matches[second][j][first] = i
    filtered = [{**frame, "people": [dict(person) for person in frame["people"]]} for frame in documents]
    fields = ("pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d")
    for frame_index, frame in enumerate(documents):
        available = min(radius, frame_index) + min(radius, len(documents) - frame_index - 1)
        required = min(min_support, available)
        dimensions = np.array([frame["canvas_width"], frame["canvas_height"]])
        for person_index, person in enumerate(frame["people"]):
            box = np.asarray(boxes[frame_index][person_index])
            distance_limit = max_distance * max(float(np.linalg.norm(box[2:] - box[:2])), 1)
            for field in fields:
                values = person.get(field)
                if not values:
                    continue
                points = np.asarray(values).reshape(-1, 3)
                support = np.zeros(len(points), dtype=np.int32)
                for neighbor, matched_person in matches[frame_index][person_index].items():
                    other = documents[neighbor]["people"][matched_person].get(field)
                    if not other or len(other) != len(values):
                        continue
                    other = np.asarray(other).reshape(-1, 3)
                    distance = np.linalg.norm((points[:, :2] - other[:, :2]) * dimensions, axis=-1)
                    support += (other[:, 2] > 0) & (distance <= distance_limit)
                unsupported = (points[:, 2] > 0) & (support < required)
                if np.any(unsupported):
                    retained = points.copy()
                    retained[unsupported] = 0
                    filtered[frame_index]["people"][person_index][field] = retained.ravel().tolist()
    return filtered


def prune_temporal_animal_keypoints(documents, boxes, radius=2, min_support=2, max_distance=0.1, match_iou=0.3):
    normalized = []
    for frame in documents:
        dimensions = np.array([frame["canvas_width"], frame["canvas_height"], 1])
        people = [{"pose_keypoints_2d": (np.asarray(animal) / dimensions).ravel().tolist()} for animal in frame["animals"]]
        normalized.append({"people": people, "canvas_width": frame["canvas_width"], "canvas_height": frame["canvas_height"]})
    filtered = prune_temporal_pose_keypoints(normalized, boxes, radius, min_support, max_distance, match_iou)
    return [{**original, "animals": [(np.asarray(person["pose_keypoints_2d"]).reshape(-1, 3)
                                      * [frame["canvas_width"], frame["canvas_height"], 1]).tolist() for person in frame["people"]]}
            for original, frame in zip(documents, filtered)]


def run_dwpose_batch(images, resolution=0, batch_size=5, detect_body=True, detect_hand=True, detect_face=True,
                     scale_stick=False, *, loader, forward, animal=False, detection_threshold=0.3, keypoint_threshold=0.3, nms_threshold=0.45,
                     temporal_filter=False, temporal_radius=2, temporal_min_support=2, temporal_max_distance=0.1, temporal_match_iou=0.3):
    if images.ndim != 4 or not len(images) or images.shape[-1] not in (3, 4):
        raise ValueError("DWPose requires a nonempty RGB or RGBA IMAGE batch.")
    if batch_size < 1 or resolution < 0:
        raise ValueError("DWPose batch_size must be positive and resolution must be zero or positive.")
    if not all(0 <= value <= 1 for value in (detection_threshold, keypoint_threshold, nms_threshold)):
        raise ValueError("Detection and keypoint thresholds must be between zero and one.")
    if temporal_filter and (temporal_radius < 1 or temporal_min_support < 1 or temporal_max_distance <= 0):
        raise ValueError("Temporal keypoint filtering requires positive radius, support and distance values.")
    models = {"detector": loader("detector")}
    factor = resolution / min(images.shape[1:3]) if resolution else 1.0
    height, width = (max(1, round(value * factor)) for value in images.shape[1:3])
    output = torch.empty((len(images), height, width, 3), dtype=torch.float32, device="cpu")
    documents, all_boxes = [], []
    progress = comfy.utils.ProgressBar(len(images))
    for start in range(0, len(images), batch_size):
        comfy.model_management.throw_exception_if_processing_interrupted()
        frames = []
        for image in images[start:start + batch_size]:
            pixels = (image[..., :3].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            model_pixels = pixels if animal else pixels[..., ::-1]
            frames.append(cv2.resize(model_pixels, (width, height), interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA))
        inputs, ratios = zip(*(dwpose_detector_input(frame) for frame in frames))
        detections = forward(models["detector"], inputs)
        if len(detections) != len(frames):
            raise ValueError("The YOLOX export did not preserve the detector batch. Use a batch-capable TorchScript export.")
        jobs, people = [], [[] for frame in frames]
        chunk_boxes = [[] for frame in frames]
        for index, (frame, prediction, ratio) in enumerate(zip(frames, detections, ratios)):
            for box in decode_dwpose_boxes(prediction, ratio, range(14, 24) if animal else (0,), detection_threshold, nms_threshold):
                crop, center, scale = prepare_dwpose_crop(frame, box, (256, 256) if animal else (288, 384))
                jobs.append((index, crop, center, scale))
                chunk_boxes[index].append(box)
        if jobs and "pose" not in models:
            models["pose"] = loader("pose")
        for offset in range(0, len(jobs), batch_size):
            chunk = jobs[offset:offset + batch_size]
            crops = [item[1] for item in chunk]
            x, y = forward(models["pose"], crops)
            for item, simcc_x, simcc_y in zip(chunk, x, y):
                frame, _, center, scale = item
                people[frame].append(decode_animal_pose(simcc_x, simcc_y, center, scale, keypoint_threshold) if animal
                                     else decode_dwpose_person(simcc_x, simcc_y, center, scale, width, height, keypoint_threshold))
        for index, frame_people in enumerate(people):
            document = {"canvas_height": height, "canvas_width": width}
            document.update({"version": "ap10k", "animals": frame_people} if animal else {"people": frame_people})
            documents.append(document)
            all_boxes.append(chunk_boxes[index])
            progress.update(1)
    if temporal_filter:
        filter_keypoints = prune_temporal_animal_keypoints if animal else prune_temporal_pose_keypoints
        documents = filter_keypoints(documents, all_boxes, temporal_radius, temporal_min_support, temporal_max_distance, temporal_match_iou)
    for index, document in enumerate(documents):
        comfy.model_management.throw_exception_if_processing_interrupted()
        rendered = draw_animal_pose_frame(document["animals"], height, width) if animal else draw_pose_frame(document["people"], height, width, detect_body, detect_hand, detect_face, scale_stick)
        output[index] = torch.from_numpy(rendered).float().div_(255)
    return output, documents


def render_densepose_frame(result, height, width, cmap="viridis"):
    if cmap not in ("viridis", "parula"):
        raise ValueError("DensePose colormap must be viridis or parula")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if cmap == "viridis":
        canvas[:] = [68, 1, 84]
    boxes, coarse, fine, _, _ = result
    for index, box in enumerate(boxes.detach().cpu().tolist()):
        x, y = int(box[0]), int(box[1])
        box_width, box_height = max(1, int(box[2] - box[0])), max(1, int(box[3] - box[1]))
        foreground = torch.nn.functional.interpolate(coarse[index:index + 1], (box_height, box_width), mode="bilinear", align_corners=False).argmax(1)[0] > 0
        labels = torch.nn.functional.interpolate(fine[index:index + 1], (box_height, box_width), mode="bilinear", align_corners=False).argmax(1)[0]
        labels = (labels * foreground).detach().cpu().numpy()
        left, top, right, bottom = max(0, x), max(0, y), min(width, x + box_width), min(height, y + box_height)
        if right <= left or bottom <= top:
            continue
        labels = labels[top - y:bottom - y, left - x:right - x]
        color = cv2.applyColorMap((labels.astype(np.float32) * (255 / 24)).clip(0, 255).astype(np.uint8),
                                 cv2.COLORMAP_VIRIDIS if cmap == "viridis" else cv2.COLORMAP_PARULA)[..., ::-1]
        region = canvas[top:bottom, left:right]
        mask = labels > 0
        region[mask] = color[mask]
    return canvas


def run_densepose_batch(images, resolution=0, batch_size=2, cmap="viridis", *, loader, forward,
                        score_threshold=0.05, detection_nms_threshold=0.5, max_detections=100,
                        rpn_pre_nms_topk=1000, rpn_post_nms_topk=1000, rpn_nms_threshold=0.7):
    if images.ndim != 4 or not len(images) or images.shape[-1] not in (3, 4):
        raise ValueError("DensePose requires a nonempty RGB or RGBA IMAGE batch")
    if resolution < 0 or batch_size < 1 or cmap not in ("viridis", "parula"):
        raise ValueError("Invalid DensePose resolution, batch size or colormap")
    if not all(0 <= value <= 1 for value in (score_threshold, detection_nms_threshold, rpn_nms_threshold)):
        raise ValueError("DensePose thresholds must be between zero and one")
    if min(max_detections, rpn_pre_nms_topk, rpn_post_nms_topk) < 1:
        raise ValueError("DensePose detection/proposal limits must be positive")
    patcher = loader()
    scale = resolution / min(images.shape[1:3]) if resolution else 1.0
    height, width = (max(1, round(size * scale)) for size in images.shape[1:3])
    output = torch.empty((len(images), height, width, 3), dtype=torch.float32, device="cpu")
    progress = comfy.utils.ProgressBar(len(images))
    for start in range(0, len(images), batch_size):
        frames = []
        for image in images[start:start + batch_size]:
            pixels = (image[..., :3].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            frames.append(cv2.resize(pixels, (width, height), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA))
        predictions = forward(patcher, frames, score_threshold=score_threshold, detection_nms_threshold=detection_nms_threshold,
                              max_detections=max_detections, rpn_pre_nms_topk=rpn_pre_nms_topk,
                              rpn_post_nms_topk=rpn_post_nms_topk, rpn_nms_threshold=rpn_nms_threshold)
        if len(predictions) != len(frames):
            raise ValueError("DensePose model did not preserve frame batch order")
        for index, result in enumerate(predictions):
            comfy.model_management.throw_exception_if_processing_interrupted()
            output[start + index] = torch.from_numpy(render_densepose_frame(result, height, width, cmap)).float().div_(255)
            progress.update(1)
    return output


def prepare_h3_reference_video_components(video, megapixels: float, duration_seconds: float = 0.0, start_at_timestamp: float = 0.0):
    components = video.get_components()
    source_frames = components.images
    source_rate = float(components.frame_rate)
    if source_frames.ndim != 4 or min(source_frames.shape[:3]) < 1:
        raise ValueError("Reference video must contain non-empty frames.")
    if not math.isfinite(source_rate) or source_rate <= 0:
        raise ValueError("Reference video must have a positive frame rate.")
    if not math.isfinite(megapixels) or megapixels <= 0:
        raise ValueError("Megapixels must be positive.")
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("Duration must be zero or a positive number of seconds.")
    if not math.isfinite(start_at_timestamp) or start_at_timestamp < 0:
        raise ValueError("Start timestamp must be zero or a positive number of seconds.")

    source_count = source_frames.shape[0]
    source_seconds = source_count / source_rate
    start_frame = h3_video_length_from_seconds(start_at_timestamp) if start_at_timestamp > 0 else 0
    start_seconds = start_frame / 24
    if start_seconds >= source_seconds:
        raise ValueError("Start timestamp rounds past the end of the reference video.")
    selected_seconds = source_seconds - start_seconds
    if duration_seconds > 0:
        selected_seconds = min(selected_seconds, duration_seconds)
    frame_count = h3_video_length_from_seconds(selected_seconds)
    frame_indices = [min(round((start_frame + index) * source_rate / 24), source_count - 1) for index in range(frame_count)]
    video_frames = source_frames[frame_indices]
    prepared_frames = video_frames

    source_height, source_width = source_frames.shape[1:3]
    source_aspect = source_width / source_height
    ratio_width, ratio_height = min(
        ASPECT_RATIOS.values(), key=lambda ratio: abs(source_aspect - ratio[0] / ratio[1]),
    )
    output_width, output_height = select_video_resolution(
        ratio_width, ratio_height, megapixels, 32, 32, MAX_RESOLUTION,
    )
    if (output_height, output_width) != (source_height, source_width):
        prepared_frames = resize_nchw(
            prepared_frames.movedim(-1, 1), output_width, output_height, "lanczos", "center",
        ).clamp(0.0, 1.0).movedim(1, -1).contiguous()

    soundtrack = components.audio
    audio_window = round(frame_count / 24 * 32000)
    aligned_samples = ((audio_window + 799) // 800) * 800
    if soundtrack is not None and soundtrack.get("waveform") is not None and soundtrack["waveform"].numel() > 0:
        sample_rate = int(soundtrack["sample_rate"])
        if sample_rate <= 0:
            raise ValueError("Reference audio must have a positive sample rate.")
        sample_count = round(frame_count / 24 * sample_rate)
        start_sample = round(start_seconds * sample_rate)
        audio_samples = soundtrack["waveform"][..., start_sample:start_sample + sample_count]
        if sample_rate != 32000 and audio_samples.numel():
            audio_samples = torchaudio.functional.resample(audio_samples, sample_rate, 32000)
        audio_samples = audio_samples[..., :audio_window]
        # Align before Core's generic VAE crop; preserve the start of the audio.
        audio_samples = torch.nn.functional.pad(audio_samples, (0, aligned_samples - audio_samples.shape[-1]))
    else:
        audio_samples = torch.zeros(1, 2, aligned_samples)
    prepared_audio = {"waveform": audio_samples, "sample_rate": 32000}
    preview = {"start_frame": start_frame, "length": frame_count, "source_seconds": source_seconds}
    return prepared_frames, prepared_audio, output_width, output_height, frame_count, video_frames, preview


def _robust_channel_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, median, high = np.percentile(values, (10.0, 50.0, 90.0), axis=0)
    return median.astype(np.float32), np.maximum(high - low, 1e-4).astype(np.float32)


def _covariance_shape(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    clipped = np.clip(
        values,
        np.percentile(values, 1.0, axis=0),
        np.percentile(values, 99.0, axis=0),
    )
    center = np.median(clipped, axis=0).astype(np.float32)
    covariance = (
        np.cov(clipped - center, rowvar=False).astype(np.float32)
        if len(clipped) > 1
        else np.zeros((2, 2), dtype=np.float32)
    )
    covariance += np.eye(2, dtype=np.float32) * 1e-5
    spread = max(float(np.trace(covariance)), 1e-5)
    return center, covariance / spread, spread


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-5) ** power
    return (vectors * values) @ vectors.T


def _prepare_mask(mask: torch.Tensor | None, index: int, height: int, width: int) -> np.ndarray | None:
    if mask is None:
        return None
    selected = mask[min(index, mask.shape[0] - 1)].detach().float().cpu().numpy().squeeze()
    if selected.shape != (height, width):
        selected = cv2.resize(selected, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(selected, 0.0, 1.0).astype(np.float32)


def _feather_outpaint_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    binary = (mask > 0.5).astype(np.uint8)
    if not np.any(binary) or np.all(binary):
        return mask
    feather_width = max(2.0, math.hypot(*mask.shape) * 0.005)
    distance_inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    inward_feather = np.clip(distance_inside / feather_width, 0.0, 1.0)
    return mask * inward_feather


def _alignment_detail_image(image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor((image_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 1024.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sigma = max(0.8, math.hypot(*gray.shape) * 0.0015)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(gray, 1.75, blurred, -0.75, 0.0), scale


def _fit_source_to_target(source_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray | None:
    source_gray, source_scale = _alignment_detail_image(source_rgb)
    target_gray, target_scale = _alignment_detail_image(target_rgb)
    detector = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.01, edgeThreshold=20)
    source_points, source_descriptors = detector.detectAndCompute(source_gray, None)
    target_points, target_descriptors = detector.detectAndCompute(target_gray, None)
    if source_descriptors is None or target_descriptors is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.8 * second.distance]
    if len(matches) < 4:
        return None
    source_xy = np.float32([source_points[match.queryIdx].pt for match in matches])
    target_xy = np.float32([target_points[match.trainIdx].pt for match in matches])
    threshold = max(2.0, math.hypot(*target_rgb.shape[:2]) * 0.003)
    affine, inliers = cv2.estimateAffinePartial2D(
        source_xy,
        target_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    inlier_count = 0 if inliers is None else int(inliers.sum())
    if affine is None or inliers is None or inlier_count < 4 or inlier_count / len(matches) < 0.6:
        return None
    affine_full = np.eye(3, dtype=np.float64)
    affine_full[:2] = affine
    affine_full = (
        np.diag([1.0 / target_scale, 1.0 / target_scale, 1.0])
        @ affine_full
        @ np.diag([source_scale, source_scale, 1.0])
    )
    affine = affine_full[:2]
    scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
    if not 0.2 <= scale <= 5.0:
        return None
    source_h, source_w = source_rgb.shape[:2]
    corners = np.float32([[[0, 0], [source_w, 0], [source_w, source_h], [0, source_h]]])
    mapped = cv2.transform(corners, affine)[0]
    target_h, target_w = target_rgb.shape[:2]
    visible = cv2.intersectConvexConvex(
        mapped.astype(np.float32),
        np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]]),
    )[0]
    mapped_area = abs(float(cv2.contourArea(mapped)))
    if mapped_area <= 1.0 or visible / mapped_area < 0.5:
        return None
    return affine.astype(np.float32)


def _matched_overlap_edge_pixels(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    outpaint_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    affine = _fit_source_to_target(source_rgb, target_rgb)
    if affine is None:
        return None
    target_h, target_w = target_rgb.shape[:2]
    warped_source = cv2.warpAffine(
        source_lab,
        affine,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    source_mask = np.ones(source_rgb.shape[:2], dtype=np.uint8)
    overlap = cv2.warpAffine(
        source_mask,
        affine,
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    distance = cv2.distanceTransform(overlap, cv2.DIST_L2, 3)
    overlap_area = int(np.count_nonzero(overlap))
    edge_width = max(3.0, math.sqrt(overlap_area) * 0.04)
    source_edge = (overlap > 0) & (distance <= edge_width)
    outside = (overlap == 0).astype(np.uint8)
    outside_distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    target_edge = (outside > 0) & (outside_distance <= edge_width)
    if outpaint_mask is not None:
        target_edge &= outpaint_mask > 1e-4
        radius = max(1, int(math.ceil(edge_width)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        adjacent = cv2.dilate(target_edge.astype(np.uint8), kernel) > 0
        source_edge &= adjacent
    if np.count_nonzero(source_edge) < 16 or np.count_nonzero(target_edge) < 16:
        return None
    return warped_source[source_edge], target_lab[target_edge]


def match_image_properties(
    source: torch.Tensor,
    target: torch.Tensor,
    overall_weight: float,
    color_weight: float,
    lighting_weight: float,
    texture_preservation: float,
    mask: torch.Tensor | None = None,
    saturation_weight: float = 1.0,
    contrast_weight: float = 1.0,
) -> torch.Tensor:
    """Transfer global color and lighting statistics without spatial correspondence."""
    if overall_weight <= 0.0 or (color_weight <= 0.0 and lighting_weight <= 0.0 and saturation_weight <= 0.0 and contrast_weight <= 0.0):
        return target.clone()

    outputs = []
    for index in range(target.shape[0]):
        source_index = min(index, source.shape[0] - 1)
        source_rgb = np.clip(source[source_index, ..., :3].detach().float().cpu().numpy(), 0.0, 1.0)
        target_rgb = np.clip(target[index, ..., :3].detach().float().cpu().numpy(), 0.0, 1.0)
        source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        apply_mask = _feather_outpaint_mask(_prepare_mask(mask, index, *target_rgb.shape[:2]))

        matched_pixels = _matched_overlap_edge_pixels(
            source_rgb,
            target_rgb,
            source_lab,
            target_lab,
            apply_mask,
        )
        if matched_pixels is None:
            source_pixels = source_lab.reshape(-1, 3)
            target_pixels = target_lab.reshape(-1, 3)
        else:
            source_pixels, target_pixels = matched_pixels

        source_l_center, source_l_range = _robust_channel_stats(source_pixels[:, :1])
        target_l_center, target_l_range = _robust_channel_stats(target_pixels[:, :1])
        contrast = float(np.clip(source_l_range[0] / target_l_range[0], 0.25, 4.0))
        contrast = 1.0 + (contrast - 1.0) * contrast_weight * overall_weight
        lighting = lighting_weight * overall_weight

        target_l = target_lab[..., 0]
        sigma = max(1.0, math.hypot(*target_l.shape) * 0.01)
        base = cv2.GaussianBlur(target_l, (0, 0), sigmaX=sigma, sigmaY=sigma)
        detail = target_l - base
        full_transfer = (target_l - target_l_center[0]) * contrast + target_l_center[0]
        detail_preserving = (base - target_l_center[0]) * contrast + target_l_center[0] + detail
        transferred_l = full_transfer * (1.0 - texture_preservation) + detail_preserving * texture_preservation
        transferred_l += (source_l_center[0] - target_l_center[0]) * lighting

        source_center, source_shape, source_spread = _covariance_shape(source_pixels[:, 1:3])
        target_center, target_shape, target_spread = _covariance_shape(target_pixels[:, 1:3])
        shape_transform = _symmetric_matrix_power(source_shape, 0.5) @ _symmetric_matrix_power(target_shape, -0.5)
        centered_chroma = target_lab[..., 1:3] - target_center
        shaped_chroma = centered_chroma @ shape_transform.T
        saturation = math.sqrt(source_spread / target_spread)
        saturation = float(np.clip(saturation, 0.25, 4.0))
        saturation = 1.0 + (saturation - 1.0) * saturation_weight * overall_weight
        color = color_weight * overall_weight
        color_matched = shaped_chroma + source_center
        transferred_chroma = target_lab[..., 1:3] * (1.0 - color) + color_matched * color
        transferred_chroma *= saturation

        result_lab = target_lab.copy()
        result_lab[..., 0] = transferred_l
        result_lab[..., 1:3] = transferred_chroma
        result_lab[..., 0] = np.clip(result_lab[..., 0], 0.0, 100.0)
        result_lab[..., 1:3] = np.clip(result_lab[..., 1:3], -127.0, 127.0)
        result_rgb = np.clip(cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0)

        if apply_mask is not None:
            result_rgb = target_rgb * (1.0 - apply_mask[..., None]) + result_rgb * apply_mask[..., None]
        if target.shape[-1] > 3:
            extra = target[index, ..., 3:].detach().float().cpu().numpy()
            result_rgb = np.concatenate((result_rgb, extra), axis=-1)
        outputs.append(torch.from_numpy(result_rgb.astype(np.float32)))

    return torch.stack(outputs).to(device=target.device, dtype=target.dtype)


def mask_to_bounding_box(
    mask: torch.Tensor,
    invert: bool = False,
    image: torch.Tensor | None = None,
) -> tuple[dict[str, int], torch.Tensor | None]:
    """Return one Core bounding box covering all nonzero mask pixels."""
    if mask.ndim < 2:
        raise ValueError("Mask to Bounding Box requires a mask with at least two dimensions.")

    active_mask = 1.0 - mask if invert else mask
    nonzero = torch.nonzero(active_mask)
    if nonzero.numel() == 0:
        raise ValueError("Mask to Bounding Box requires at least one nonzero pixel.")

    y_min = int(nonzero[:, -2].min().item())
    y_max = int(nonzero[:, -2].max().item())
    x_min = int(nonzero[:, -1].min().item())
    x_max = int(nonzero[:, -1].max().item())
    bounding_box = {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min + 1,
        "height": y_max - y_min + 1,
    }
    cropped_image = (
        image[:, y_min:y_max + 1, x_min:x_max + 1, :]
        if image is not None
        else None
    )
    return bounding_box, cropped_image


_HALO_CONTEXT_RADIUS = 13
_HALO_WORKSPACE_BYTES = 128 * 1024 * 1024
_LOHALO_CONTRAST = 3.38589


def halo_downscale_dimensions(width: int, height: int, megapixels: float, multiple: int) -> tuple[int, int, float, float, float]:
    """Return aligned output dimensions and a uniform, centered source transform."""
    target_pixels = megapixels * 1024 * 1024
    scale = math.sqrt(target_pixels / (width * height))
    output_width = max(multiple, int((width * scale + multiple / 2) // multiple) * multiple)
    output_height = max(multiple, int((height * scale + multiple / 2) // multiple) * multiple)
    cover_scale = max(output_width / width, output_height / height)
    if cover_scale >= 1.0:
        raise ValueError("NoHalo/LoHalo Downscale requires an output smaller than the input.")
    view_width = output_width / cover_scale
    view_height = output_height / cover_scale
    return output_width, output_height, cover_scale, (width - view_width) * 0.5, (height - view_height) * 0.5


def _mitchell_kernel(distance: torch.Tensor) -> torch.Tensor:
    distance = distance.abs()
    inner = (7.0 / 6.0) * distance**3 - 2.0 * distance**2 + 8.0 / 9.0
    outer = (-7.0 / 18.0) * distance**3 + 2.0 * distance**2 - (10.0 / 3.0) * distance + 16.0 / 9.0
    return torch.where(distance < 1.0, inner, torch.where(distance < 2.0, outer, 0.0))


def _robidoux_kernel(radius_squared: torch.Tensor) -> torch.Tensor:
    sqrt_two = math.sqrt(2.0)
    radius = torch.sqrt(radius_squared.clamp_min(0.0))
    inner = radius_squared * (-3.0 * radius + (45739.0 + 7164.0 * sqrt_two) / 10319.0) + (-8926.0 - 14328.0 * sqrt_two) / 10319.0
    outer = (radius + (-103.0 - 36.0 * sqrt_two) / (7.0 + 72.0 * sqrt_two)) * (radius - 2.0) ** 2
    return torch.where(radius_squared < 1.0, inner, torch.where(radius_squared < 4.0, outer, 0.0))


def _inverse_sigmoidal(value: torch.Tensor) -> torch.Tensor:
    sig1 = math.tanh(0.25 * _LOHALO_CONTRAST)
    slope = (1.0 / sig1 - sig1) * 0.25 * _LOHALO_CONTRAST
    middle = torch.atanh(((2.0 * sig1) * value - sig1).clamp(-0.999999, 0.999999)) * (2.0 / _LOHALO_CONTRAST) + 0.5
    return torch.where(value <= 0.0, value / slope, torch.where(value >= 1.0, value / slope + 1.0 - 1.0 / slope, middle))


def _extended_sigmoidal(value: torch.Tensor) -> torch.Tensor:
    sig1 = math.tanh(0.25 * _LOHALO_CONTRAST)
    slope = (1.0 / sig1 - sig1) * 0.25 * _LOHALO_CONTRAST
    middle = (0.5 / sig1) * torch.tanh(0.5 * _LOHALO_CONTRAST * value - 0.25 * _LOHALO_CONTRAST) + 0.5
    return torch.where(value <= 0.0, slope * value, torch.where(value >= 1.0, slope * value + 1.0 - slope, middle))


def _minmod(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.where(first * second >= 0.0, torch.where(first.square() <= first * second, first, second), 0.0)


def _nohalo_subdivision(p: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Vectorized GEGL/libvips NoHalo level-one subdivision for an oriented 5x5 stencil."""
    u2, u3, u4 = p[..., 0, 1], p[..., 0, 2], p[..., 0, 3]
    d1, d2, d3, d4, d5 = (p[..., 1, index] for index in range(5))
    t1, t2, t3, t4, t5 = (p[..., 2, index] for index in range(5))
    q1, q2, q3, q4, q5 = (p[..., 3, index] for index in range(5))
    c2, c3, c4 = p[..., 4, 1], p[..., 4, 2], p[..., 4, 3]

    du2, dt2, tq2, qc2 = d2-u2, t2-d2, q2-t2, c2-q2
    du3, dt3, tq3, qc3 = d3-u3, t3-d3, q3-t3, c3-q3
    du4, dt4, tq4, qc4 = d4-u4, t4-d4, q4-t4, c4-q4
    d12, d23, d34, d45 = d2-d1, d3-d2, d4-d3, d5-d4
    t12, t23, t34, t45 = t2-t1, t3-t2, t4-t3, t5-t4
    q12, q23, q34, q45 = q2-q1, q3-q2, q4-q3, q5-q4

    d3y, t3y = _minmod(dt3, du3), _minmod(dt3, tq3)
    q3y = _minmod(qc3, tq3)
    t4y, q4y, d4y = _minmod(dt4, tq4), _minmod(qc4, tq4), _minmod(dt4, du4)
    t2x, t3x, t4x = _minmod(t23, t12), _minmod(t23, t34), _minmod(t45, t34)
    q3x, q4x, q2x = _minmod(q23, q34), _minmod(q45, q34), _minmod(q23, q12)
    d3x, d4x, d2x = _minmod(d23, d34), _minmod(d45, d34), _minmod(d23, d12)
    t2y, q2y, d2y = _minmod(dt2, tq2), _minmod(qc2, tq2), _minmod(dt2, du2)

    a12 = 0.5*(d3+t3) + 0.25*(d3y-t3y)
    a32 = 0.5*(t3+q3) + 0.25*(t3y-q3y)
    a34 = 0.5*(t4+q4) + 0.25*(t4y-q4y)
    a14 = 0.5*(d4+t4) + 0.25*(d4y-t4y)
    a21 = 0.5*(t2+t3) + 0.25*(t2x-t3x)
    a23 = 0.5*(t3+t4) + 0.25*(t3x-t4x)
    a43 = 0.5*(q3+q4) + 0.25*(q3x-q4x)
    a41 = 0.5*(q2+q3) + 0.25*(q2x-q3x)
    a33 = 0.125*((t3x-t4x)+(q3x-q4x)) + 0.5*(a32+a34)
    a13 = 0.25*(d4-t3) + 0.125*(d4y-t4y+d3x-d4x) + 0.5*(a12+a23)
    a31 = 0.25*(q2-t3) + 0.125*(q2x-q3x+t2y-q2y) + 0.5*(a21+a32)
    a11 = 0.25*(d2+d3+t2+t3) + 0.125*(d2x-d3x+t2x-t3x+d2y+d3y-t2y-t3y)
    return a11, a12, a13, a14, a21, t3, a23, t4, a31, a32, a33, a34, a41, q3, a43, q4


def _lbb(stencil: tuple[torch.Tensor, ...], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    u1,u2,u3,u4,d1,d2,d3,d4,t1,t2,t3,t4,q1,q2,q3,q4 = stencil
    min00 = torch.minimum(torch.minimum(torch.minimum(u1,u2),u3), torch.minimum(torch.minimum(d1,d2), torch.minimum(d3, torch.minimum(t1,torch.minimum(t2,t3)))))
    max00 = torch.maximum(torch.maximum(torch.maximum(u1,u2),u3), torch.maximum(torch.maximum(d1,d2), torch.maximum(d3, torch.maximum(t1,torch.maximum(t2,t3)))))
    min10 = torch.minimum(torch.minimum(torch.minimum(u2,u3),u4), torch.minimum(torch.minimum(d2,d3), torch.minimum(d4, torch.minimum(t2,torch.minimum(t3,t4)))))
    max10 = torch.maximum(torch.maximum(torch.maximum(u2,u3),u4), torch.maximum(torch.maximum(d2,d3), torch.maximum(d4, torch.maximum(t2,torch.maximum(t3,t4)))))
    min01 = torch.minimum(torch.minimum(torch.minimum(d1,d2),d3), torch.minimum(torch.minimum(t1,t2), torch.minimum(t3, torch.minimum(q1,torch.minimum(q2,q3)))))
    max01 = torch.maximum(torch.maximum(torch.maximum(d1,d2),d3), torch.maximum(torch.maximum(t1,t2), torch.maximum(t3, torch.maximum(q1,torch.maximum(q2,q3)))))
    min11 = torch.minimum(torch.minimum(torch.minimum(d2,d3),d4), torch.minimum(torch.minimum(t2,t3), torch.minimum(t4, torch.minimum(q2,torch.minimum(q3,q4)))))
    max11 = torch.maximum(torch.maximum(torch.maximum(d2,d3),d4), torch.maximum(torch.maximum(t2,t3), torch.maximum(t4, torch.maximum(q2,torch.maximum(q3,q4)))))

    values = (d2,d3,t2,t3)
    mins, maxs = (min00,min10,min01,min11), (max00,max10,max01,max11)
    dx0 = (d3-d1,d4-d2,t3-t1,t4-t2)
    dy0 = (t2-u2,t3-u3,q2-d2,q3-d3)
    cross0 = (u1-u3+t3-t1, u2-u4+t4-t2, q3-q1-d3+d1, q4-q2-d4+d2)
    dx, dy, cross = [], [], []
    for value, minimum, maximum, ddx, ddy, dcross in zip(values, mins, maxs, dx0, dy0, cross0):
        limit = 6.0 * torch.minimum(value-minimum, maximum-value)
        ddx = ddx.sign() * torch.minimum(ddx.abs(), limit)
        ddy = ddy.sign() * torch.minimum(ddy.abs(), limit)
        sum12, dif12 = 6.0*(ddx+ddy), 6.0*(ddx-ddy)
        lower = torch.maximum(sum12.abs()-36.0*(value-minimum), dif12.abs()-36.0*(maximum-value))
        upper = torch.minimum(36.0*(maximum-value)-sum12.abs(), 36.0*(value-minimum)-dif12.abs())
        dx.append(ddx)
        dy.append(ddy)
        cross.append(dcross.clamp(min=lower, max=upper))

    hx0, hx1 = 2*x**3-3*x**2+1, -2*x**3+3*x**2
    hdx0, hdx1 = x**3-2*x**2+x, x**3-x**2
    hy0, hy1 = 2*y**3-3*y**2+1, -2*y**3+3*y**2
    hdy0, hdy1 = y**3-2*y**2+y, y**3-y**2
    c = (hx0*hy0,hx1*hy0,hx0*hy1,hx1*hy1)
    cx = (hdx0*hy0,hdx1*hy0,hdx0*hy1,hdx1*hy1)
    cy = (hx0*hdy0,hx1*hdy0,hx0*hdy1,hx1*hdy1)
    cxy = (hdx0*hdy0,hdx1*hdy0,hdx0*hdy1,hdx1*hdy1)
    return sum(a*b for a,b in zip(c,values)) + 0.5*sum(a*b for a,b in zip(cx,dx)) + 0.5*sum(a*b for a,b in zip(cy,dy)) + 0.25*sum(a*b for a,b in zip(cxy,cross))


def _gather_halo_patch(image: torch.Tensor, x_index: torch.Tensor, y_index: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = image.shape
    yy = (y_index[..., None, None] + offsets[:, None]).clamp(0, height-1)
    xx = (x_index[..., None, None] + offsets[None, :]).clamp(0, width-1)
    linear = (yy * width + xx).reshape(1, 1, -1).expand(batch, channels, -1)
    return torch.gather(image.reshape(batch, channels, -1), 2, linear).reshape(batch, channels, *x_index.shape, offsets.numel(), offsets.numel())


def downscale_nohalo_lohalo(image: torch.Tensor, method: str, megapixels: float, multiple: int) -> torch.Tensor:
    """Downscale BHWC images using axis-aligned GEGL NoHalo or LoHalo sampling."""
    original_dtype = image.dtype
    batch, height, width, channels = image.shape
    out_width, out_height, scale, offset_x, offset_y = halo_downscale_dimensions(width, height, megapixels, multiple)
    source = image.movedim(-1, 1).float()
    support = min(_HALO_CONTEXT_RADIUS, max(2, math.ceil((2.0 if method == "lohalo" else 1.0) / scale + 0.5)))
    kernel_size = 2 * support + 1
    bytes_per_pixel = batch * kernel_size * kernel_size * (channels * 4 + 24)
    tile_rows = max(1, min(out_height, _HALO_WORKSPACE_BYTES // max(1, out_width * bytes_per_pixel)))
    x = offset_x + (torch.arange(out_width, device=image.device, dtype=torch.float32) + 0.5) / scale
    x_anchor = torch.floor(x).long()
    x_fraction = x - (x_anchor.float() + 0.5)
    offsets = torch.arange(-support, support+1, device=image.device)
    output = torch.empty((batch, channels, out_height, out_width), device=image.device, dtype=torch.float32)

    for y_start in range(0, out_height, tile_rows):
        y_stop = min(out_height, y_start + tile_rows)
        y = offset_y + (torch.arange(y_start, y_stop, device=image.device, dtype=torch.float32) + 0.5) / scale
        y_anchor = torch.floor(y).long()
        y_fraction = y - (y_anchor.float() + 0.5)
        xi = x_anchor.unsqueeze(0).expand(y.numel(), -1)
        yi = y_anchor.unsqueeze(1).expand(-1, out_width)
        patch = _gather_halo_patch(source, xi, yi, offsets)
        dx = x_fraction[None, :, None, None] - offsets.float()[None, None, None, :]
        dy = y_fraction[:, None, None, None] - offsets.float()[None, None, :, None]

        if method == "lohalo":
            weights = _mitchell_kernel(dx) * _mitchell_kernel(dy)
            sigmoid_patch = _inverse_sigmoidal(patch)
            mitchell = (sigmoid_patch * weights).sum((-1,-2))
            if channels == 4:
                mitchell[:, :3] = _extended_sigmoidal(mitchell[:, :3])
                mitchell[:, 3] = (patch[:, 3] * weights).sum((-1,-2))
            else:
                mitchell = _extended_sigmoidal(mitchell)
            ewa_weights = _robidoux_kernel(scale*scale*(dx*dx+dy*dy))
            ewa_total = ewa_weights.sum((-1,-2))
            ewa_total = torch.where(ewa_total.abs() < 1e-8, torch.ones_like(ewa_total), ewa_total)
            ewa = (patch * ewa_weights).sum((-1,-2)) / ewa_total
            tile = scale*scale*mitchell + (1.0-scale*scale)*ewa
        else:
            signs_x = torch.where(x_fraction >= 0, 1, -1)
            signs_y = torch.where(y_fraction >= 0, 1, -1)
            oriented = torch.empty((*patch.shape[:-2],5,5), device=image.device, dtype=patch.dtype)
            center = support
            patch_flat = patch.flatten(-2)
            for row in range(5):
                for column in range(5):
                    source_row = center + (row-2) * signs_y[:, None]
                    source_column = center + (column-2) * signs_x[None, :]
                    source_index = source_row * kernel_size + source_column
                    gather_index = source_index[None,None].expand(batch, channels, -1, -1).unsqueeze(-1)
                    oriented[..., row, column] = torch.gather(patch_flat, -1, gather_index).squeeze(-1)
            subdivision = _nohalo_subdivision(oriented)
            local_x = 2.0*x_fraction.abs()
            local_y = 2.0*y_fraction.abs()
            lbb = _lbb(subdivision, local_x[None,None,None,:], local_y[None,None,:,None])
            ewa_weights = (1.0-torch.sqrt((scale*scale*(dx*dx+dy*dy)).clamp_min(0.0))).clamp_min(0.0)
            ewa = (patch * ewa_weights).sum((-1,-2)) / ewa_weights.sum((-1,-2)).clamp_min(1e-8)
            tile = scale*scale*lbb + (1.0-scale*scale)*ewa
        output[:,:,y_start:y_stop] = tile
    return output.movedim(1,-1).to(original_dtype)


VIDEO_FRAME_SAMPLING_STRATEGIES = (
    "codec keyframes",
    "uniform PTS",
    "focused PTS",
)

VIDEO_FRAME_TIMESTAMP_FORMATS = (
    "HH:MM:SS.mmm",
    "HH:MM:SS:mmm",
    "MM:SS.mmm",
    "MM:SS:mmm",
    "00.000s",
    "0.0s",
    "0.00s",
)

VIDEO_FRAME_TIMELINE_STYLES = (
    "H3 alignment prefix",
    "H3 pictures",
    "indexed",
    "timestamps only",
    "custom",
)

VIDEO_TIMELINE_TEXT_STRUCTURE = (
    "For the target video, at <<time>> into the target video, "
    "<<picture>> (from <<shot>>) is fully referenced."
)
VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE = (
    "Target video duration is <<duration>> seconds divided into "
    "<<segments>> segments. Reference each image with <<references>>."
)
VIDEO_TEXT_TIMELINE_TEXT_STRUCTURE = "Shot <<shot>> at <<timestamp>>."
VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_STRUCTURE = (
    "Target video duration is <<duration>> seconds divided into "
    "<<segments>> segments. <<shot>> at <<timestamp>>."
)

_VIDEO_TIMELINE_TEXT_MARKERS = frozenset(
    {"time", "timestamp", "picture", "shot"}
)
_VIDEO_STRUCTURED_TIMELINE_TEXT_MARKERS = frozenset(
    {"duration", "segments", "timestamps", "references", "shot", "timestamp"}
)
_VIDEO_TEXT_TIMELINE_TEXT_MARKERS = frozenset({"shot", "time", "timestamp"})
_VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_MARKERS = frozenset(
    {"duration", "segments", "timestamps", "shot", "timestamp"}
)

_VIDEO_TIMESTAMP_COLON_PATTERN = re.compile(
    r"^(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d+)(?P<fraction>[.:]\d+)?$"
)


def parse_video_timestamp(value) -> Fraction:
    """Parse one supported video timestamp into exact nonnegative seconds."""
    if isinstance(value, bool):
        raise ValueError("boolean values are not timestamps")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        result = Fraction(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp is empty")
        suffix = re.fullmatch(r"(.+?)\s*(?:s|seconds?)", text, re.IGNORECASE)
        if suffix:
            text = suffix.group(1).strip()
        colon_parts = text.split(":")
        if len(colon_parts) in (3, 4) and all(part.isdigit() for part in colon_parts):
            if len(colon_parts) == 4:
                hours, minutes, seconds, milliseconds = map(int, colon_parts)
            else:
                hours = 0
                minutes, seconds, milliseconds = map(int, colon_parts)
            if minutes >= 60 and hours:
                raise ValueError("minute component must be below 60")
            if seconds >= 60:
                raise ValueError("second component must be below 60")
            result = Fraction(hours * 3600 + minutes * 60 + seconds) + Fraction(milliseconds, 10 ** len(colon_parts[-1]))
            if result < 0:
                raise ValueError("timestamp must not be negative")
            return result
        match = _VIDEO_TIMESTAMP_COLON_PATTERN.fullmatch(text)
        if match:
            hours = int(match.group("hours") or 0)
            minutes = int(match.group("minutes"))
            seconds = int(match.group("seconds"))
            if minutes >= 60 and match.group("hours") is not None:
                raise ValueError("minute component must be below 60")
            if seconds >= 60:
                raise ValueError("second component must be below 60")
            fraction = match.group("fraction")
            fractional = Fraction(0)
            if fraction:
                digits = fraction[1:]
                fractional = Fraction(int(digits), 10 ** len(digits))
            result = Fraction(hours * 3600 + minutes * 60 + seconds) + fractional
        else:
            try:
                result = Fraction(text)
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"unsupported timestamp {value!r}") from exc
    else:
        raise ValueError(f"unsupported timestamp type {type(value).__name__}")
    if result < 0:
        raise ValueError("timestamp must not be negative")
    return result


def parse_video_timestamps(value) -> list[Fraction]:
    """Flatten and parse timestamp containers while preserving source order."""
    raw = []

    def collect(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        elif isinstance(item, str) and re.search(r"[,;\n]", item):
            parts = re.split(r"[,;\n]", item)
            if any(not part.strip() for part in parts):
                raise ValueError("timestamp list contains an empty item")
            raw.extend(parts)
        else:
            raw.append(item)

    collect(value)
    if not raw:
        raise ValueError("at least one timestamp is required")
    parsed = []
    for index, item in enumerate(raw, start=1):
        try:
            parsed.append(parse_video_timestamp(item))
        except ValueError as exc:
            raise ValueError(f"timestamp {index}: {exc}") from exc
    for index in range(1, len(parsed)):
        if parsed[index] < parsed[index - 1]:
            raise ValueError(f"timestamp {index + 1} is earlier than timestamp {index}")
    return parsed


@dataclass(frozen=True)
class VideoFrameRecord:
    """Presentation metadata for one frame in the active VIDEO input."""

    frame_index: int
    timestamp: Fraction
    key_frame: bool


@dataclass(frozen=True)
class SampledVideoFrames:
    image_batch: torch.Tensor
    image_list: list[torch.Tensor]
    timestamps: list[str]
    timestamps_text: str
    timeline_text: str
    video_runtime: float
    structured_timeline_text: str


def _as_fraction(value: Fraction | float | int) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _video_source_factory(video) -> Callable[[], str | io.BytesIO]:
    source = video.get_stream_source()
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        return lambda: path

    seek = getattr(source, "seek", None)
    if callable(seek):
        seek(0)
    data = source.read()
    return lambda: io.BytesIO(data)


def _active_video_frames(
    video,
    source_factory: Callable[[], str | io.BytesIO] | None = None,
) -> Iterator[tuple[int, Fraction, av.VideoFrame]]:
    """Yield active frames in presentation order with clip-relative PTS."""

    if source_factory is None:
        source_factory = _video_source_factory(video)
    source = source_factory()
    start_seconds, duration_seconds = video.get_active_trim_window()
    trim_start = _as_fraction(start_seconds)
    trim_duration = _as_fraction(duration_seconds)

    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The VIDEO input contains no video stream.")

        stream = container.streams.video[0]
        if stream.time_base is None:
            raise ValueError("The video stream has no time base for PTS conversion.")

        stream_time_base = Fraction(stream.time_base)
        stream_start_pts = stream.start_time if stream.start_time is not None else 0
        stream_origin = Fraction(stream_start_pts) * stream_time_base
        active_start = stream_origin + trim_start
        active_end = active_start + trim_duration if trim_duration > 0 else None

        if trim_start > 0:
            seek_pts = stream_start_pts + int(trim_start / stream_time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        relative_origin = None
        previous_time = None
        frame_index = 0

        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(
                    "A decoded video frame has no PTS; exact timestamp sampling is unavailable."
                )

            frame_time_base = frame.time_base or stream.time_base
            if frame_time_base is None:
                raise ValueError(
                    "A decoded video frame has no time base for PTS conversion."
                )

            presentation_time = Fraction(frame.pts) * Fraction(frame_time_base)
            if presentation_time < active_start:
                continue
            if active_end is not None and presentation_time >= active_end:
                break
            if previous_time is not None and presentation_time < previous_time:
                raise ValueError("Video presentation timestamps are not monotonic.")

            if relative_origin is None:
                relative_origin = presentation_time
            relative_time = presentation_time - relative_origin
            previous_time = presentation_time
            yield frame_index, relative_time, frame
            frame_index += 1


def scan_video_frame_records(
    video,
    source_factory: Callable[[], str | io.BytesIO] | None = None,
) -> list[VideoFrameRecord]:
    records = [
        VideoFrameRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            key_frame=bool(frame.key_frame),
        )
        for frame_index, timestamp, frame in _active_video_frames(
            video, source_factory
        )
    ]
    if not records:
        raise ValueError("The active VIDEO input contains no decodable frames.")
    return records


def _spacing_filter(
    records: Sequence[VideoFrameRecord], minimum_spacing: Fraction
) -> list[VideoFrameRecord]:
    selected: list[VideoFrameRecord] = []
    for record in sorted(records, key=lambda item: (item.timestamp, item.frame_index)):
        if not selected or record.timestamp - selected[-1].timestamp >= minimum_spacing:
            selected.append(record)
    return selected


def _evenly_thin(
    records: Sequence[VideoFrameRecord],
    count: int,
    *,
    single_from_end: bool,
) -> list[VideoFrameRecord]:
    if count <= 0 or not records:
        return []
    if count >= len(records):
        return list(records)
    if count == 1:
        index = len(records) - 1 if single_from_end else len(records) // 2
        return [records[index]]

    denominator = count - 1
    last_index = len(records) - 1
    indices = [
        (position * last_index + denominator // 2) // denominator
        for position in range(count)
    ]
    return [records[index] for index in indices]


def _select_uniform_samples(
    records: Sequence[VideoFrameRecord],
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing: Fraction,
    timestamp_format: str | None,
) -> tuple[list[VideoFrameRecord], list[Fraction]]:
    zero_record = records[0]
    candidates = [record for record in records if record.frame_index != zero_record.frame_index]

    if maximum_frames == 0:
        combined = ([zero_record] if include_zero_time else []) + candidates
        selected = _spacing_filter(combined, minimum_spacing)
        return selected, [record.timestamp for record in selected]

    if not candidates:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]

    if include_zero_time and maximum_frames == 1:
        return [zero_record], [Fraction(0)]

    duration = records[-1].timestamp
    if duration <= 0:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]

    if include_zero_time:
        target_count = maximum_frames - 1
        targets = [
            duration * Fraction(position, target_count)
            for position in range(1, target_count + 1)
        ]
        selected_pairs = [(zero_record, Fraction(0))]
    else:
        target_count = maximum_frames
        targets = [
            duration * Fraction(position, target_count + 1)
            for position in range(1, target_count + 1)
        ]
        selected_pairs = []

    if timestamp_format is not None:
        targets = [
            round_video_timestamp(target, timestamp_format)
            for target in targets
        ]

    for target in targets:
        selected_pairs.append(
            (
                min(
                    candidates,
                    key=lambda record: (
                        abs(record.timestamp - target),
                        record.timestamp,
                        record.frame_index,
                    ),
                ),
                target,
            )
        )

    unique = {
        record.frame_index: (record, timestamp)
        for record, timestamp in selected_pairs
    }
    spaced = _spacing_filter(
        [record for record, _ in unique.values()],
        minimum_spacing,
    )
    timestamps_by_index = {
        record.frame_index: timestamp
        for record, timestamp in unique.values()
    }
    return spaced, [timestamps_by_index[record.frame_index] for record in spaced]


def _select_uniform_records(
    records: Sequence[VideoFrameRecord],
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing: Fraction,
) -> list[VideoFrameRecord]:
    selected, _ = _select_uniform_samples(
        records,
        maximum_frames,
        include_zero_time,
        minimum_spacing,
        timestamp_format=None,
    )
    return selected


def _select_focused_samples(records: Sequence[VideoFrameRecord], maximum_frames: int, include_zero_time: bool, minimum_spacing: Fraction, timestamp_format: str | None, focus_areas: int, focus_one: float, focus_two: float, focus_three: float) -> tuple[list[VideoFrameRecord], list[Fraction]]:
    if maximum_frames == 0:
        return _select_uniform_samples(records, maximum_frames, include_zero_time, minimum_spacing, timestamp_format)

    zero_record = records[0]
    candidates = [record for record in records if record.frame_index != zero_record.frame_index]
    if not candidates:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]
    if include_zero_time and maximum_frames == 1:
        return [zero_record], [Fraction(0)]

    duration = records[-1].timestamp
    if duration <= 0:
        selected = [zero_record] if include_zero_time else []
        return selected, [record.timestamp for record in selected]

    targets = [
        _as_fraction(target) for target in focused_timeline_timestamps(
            maximum_frames, float(duration), focus_areas, focus_one, focus_two, focus_three, include_zero_time, include_zero_time
        )
    ]
    if timestamp_format is not None:
        targets = [round_video_timestamp(target, timestamp_format) for target in targets]

    selected_pairs = []
    for target in targets:
        if include_zero_time and target == 0:
            selected_pairs.append((zero_record, Fraction(0)))
        else:
            selected_pairs.append((min(candidates, key=lambda record: (abs(record.timestamp - target), record.timestamp, record.frame_index)), target))

    unique = {record.frame_index: (record, timestamp) for record, timestamp in selected_pairs}
    spaced = _spacing_filter([record for record, _ in unique.values()], minimum_spacing)
    timestamps_by_index = {record.frame_index: timestamp for record, timestamp in unique.values()}
    return spaced, [timestamps_by_index[record.frame_index] for record in spaced]


def _select_keyframe_records(
    records: Sequence[VideoFrameRecord],
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing: Fraction,
    keyframe_stride: int,
) -> list[VideoFrameRecord]:
    zero_record = records[0]
    raw_keyframes = [record for record in records if record.key_frame]
    candidates = raw_keyframes[::keyframe_stride]

    if include_zero_time:
        combined = [zero_record] + [
            record for record in candidates if record.frame_index != zero_record.frame_index
        ]
    else:
        combined = [record for record in candidates if record.timestamp > 0]

    spaced = _spacing_filter(combined, minimum_spacing)
    if maximum_frames == 0 or len(spaced) <= maximum_frames:
        return spaced

    if include_zero_time:
        zero = spaced[0]
        remaining = _evenly_thin(
            spaced[1:],
            maximum_frames - 1,
            single_from_end=True,
        )
        return [zero] + remaining

    return _evenly_thin(
        spaced,
        maximum_frames,
        single_from_end=False,
    )


def _select_video_frame_records_and_timestamps(
    records: Sequence[VideoFrameRecord],
    strategy: str,
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing_seconds: float,
    keyframe_stride: int,
    timestamp_format: str | None,
    focus_areas: int = 0,
    focus_one: float = 0.5,
    focus_two: float = 0.5,
    focus_three: float = 0.5,
) -> tuple[list[VideoFrameRecord], list[Fraction]]:
    if strategy not in VIDEO_FRAME_SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported video-frame sampling strategy: {strategy}")
    if maximum_frames < 0:
        raise ValueError("maximum_frames must be zero or greater.")
    if minimum_spacing_seconds < 0:
        raise ValueError("minimum_spacing_seconds must be zero or greater.")
    if keyframe_stride < 1:
        raise ValueError("keyframe_stride must be at least one.")
    if not records:
        raise ValueError("No video-frame records are available for selection.")

    ordered = sorted(records, key=lambda item: (item.timestamp, item.frame_index))
    minimum_spacing = _as_fraction(minimum_spacing_seconds)

    if strategy == "uniform PTS":
        selected, output_timestamps = _select_uniform_samples(
            ordered,
            maximum_frames,
            include_zero_time,
            minimum_spacing,
            timestamp_format,
        )
    elif strategy == "focused PTS":
        selected, output_timestamps = _select_focused_samples(
            ordered, maximum_frames, include_zero_time, minimum_spacing, timestamp_format, focus_areas, focus_one, focus_two, focus_three
        )
    else:
        selected = _select_keyframe_records(
            ordered,
            maximum_frames,
            include_zero_time,
            minimum_spacing,
            keyframe_stride,
        )
        output_timestamps = [record.timestamp for record in selected]

    if not selected:
        raise ValueError("No video frames satisfy the selected sampling controls.")
    return selected, output_timestamps


def select_video_frame_records(
    records: Sequence[VideoFrameRecord],
    strategy: str,
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing_seconds: float,
    keyframe_stride: int,
    focus_areas: int = 0,
    focus_one: float = 0.5,
    focus_two: float = 0.5,
    focus_three: float = 0.5,
) -> list[VideoFrameRecord]:
    selected, _ = _select_video_frame_records_and_timestamps(
        records,
        strategy,
        maximum_frames,
        include_zero_time,
        minimum_spacing_seconds,
        keyframe_stride,
        timestamp_format=None,
        focus_areas=focus_areas,
        focus_one=focus_one,
        focus_two=focus_two,
        focus_three=focus_three,
    )
    return selected


def _frame_to_image(frame: av.VideoFrame) -> torch.Tensor:
    image = frame.to_ndarray(format="rgb24")
    rotation = getattr(frame, "rotation", 0) or 0
    if rotation:
        quarter_turns = int(round(rotation / 90.0))
        image = np.rot90(image, k=quarter_turns, axes=(0, 1)).copy()
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).to(dtype=torch.float32).div_(255.0)


def decode_selected_video_frames(
    video,
    records: Sequence[VideoFrameRecord],
    source_factory: Callable[[], str | io.BytesIO] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    expected = {record.frame_index: record for record in records}
    selected_images: dict[int, torch.Tensor] = {}
    last_index = max(expected)

    for frame_index, timestamp, frame in _active_video_frames(
        video, source_factory
    ):
        record = expected.get(frame_index)
        if record is not None:
            if timestamp != record.timestamp:
                raise ValueError("Video timestamps changed between selection and decoding.")
            selected_images[frame_index] = _frame_to_image(frame)
        if frame_index >= last_index:
            break

    missing = [record.frame_index for record in records if record.frame_index not in selected_images]
    if missing:
        raise ValueError("Selected video frames could not be decoded.")

    ordered_images = [selected_images[record.frame_index] for record in records]
    first_shape = ordered_images[0].shape
    if any(image.shape != first_shape for image in ordered_images[1:]):
        raise ValueError("Selected video frames have inconsistent image dimensions.")

    image_batch = torch.stack(ordered_images, dim=0)
    image_list = [image_batch[index : index + 1] for index in range(image_batch.shape[0])]
    return image_batch, image_list


def _rounded_units(timestamp: Fraction, units_per_second: int) -> int:
    if timestamp < 0:
        raise ValueError("Relative video timestamps cannot be negative.")
    numerator = timestamp.numerator * units_per_second
    denominator = timestamp.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def round_video_timestamp(
    timestamp: Fraction | float,
    timestamp_format: str,
) -> Fraction:
    if timestamp_format not in VIDEO_FRAME_TIMESTAMP_FORMATS:
        raise ValueError(f"Unsupported video timestamp format: {timestamp_format}")
    units_per_second = 10 if timestamp_format == "0.0s" else (
        100 if timestamp_format == "0.00s" else 1000
    )
    return Fraction(
        _rounded_units(_as_fraction(timestamp), units_per_second),
        units_per_second,
    )


def format_video_timestamp(timestamp: Fraction | float, timestamp_format: str) -> str:
    if timestamp_format not in VIDEO_FRAME_TIMESTAMP_FORMATS:
        raise ValueError(f"Unsupported video timestamp format: {timestamp_format}")

    value = _as_fraction(timestamp)
    if timestamp_format == "0.0s":
        total_deciseconds = _rounded_units(value, 10)
        seconds, deciseconds = divmod(total_deciseconds, 10)
        return f"{seconds}.{deciseconds}s"

    if timestamp_format == "0.00s":
        total_centiseconds = _rounded_units(value, 100)
        seconds, centiseconds = divmod(total_centiseconds, 100)
        return f"{seconds}.{centiseconds:02d}s"

    total_milliseconds = _rounded_units(value, 1000)
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)

    if timestamp_format == "00.000s":
        return f"{total_seconds:02d}.{milliseconds:03d}s"

    total_minutes, seconds = divmod(total_seconds, 60)
    if timestamp_format == "MM:SS.mmm":
        return f"{total_minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    if timestamp_format == "MM:SS:mmm":
        return f"{total_minutes:02d}:{seconds:02d}:{milliseconds:03d}"

    hours, minutes = divmod(total_minutes, 60)
    separator = "." if timestamp_format == "HH:MM:SS.mmm" else ":"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def build_video_timeline_text(
    timestamps: Sequence[str],
    timeline_style: str,
    timeline_text_structure: str,
    index_offset: int = 0,
) -> str:
    if timeline_style not in VIDEO_FRAME_TIMELINE_STYLES:
        raise ValueError(f"Unsupported video timeline style: {timeline_style}")
    if timeline_style == "timestamps only":
        return "\n".join(timestamps)
    if timeline_style == "indexed":
        return "\n".join(
            f"{index}: {timestamp}"
            for index, timestamp in enumerate(timestamps)
        )
    if timeline_style == "H3 pictures":
        return "\n".join(
            f"<Picture {index + index_offset}> at {timestamp}"
            for index, timestamp in enumerate(timestamps, start=1)
        )
    if timeline_style == "H3 alignment prefix":
        timeline_text_structure = VIDEO_TIMELINE_TEXT_STRUCTURE
    _validate_video_timeline_structure(
        timeline_text_structure,
        _VIDEO_TIMELINE_TEXT_MARKERS,
        "timeline text structure",
    )
    return "\n".join(
        timeline_text_structure.replace("<<time>>", timestamp)
        .replace("<<timestamp>>", timestamp)
        .replace("<<picture>>", f"<Picture {index + index_offset}>")
        .replace("<<shot>>", f"[Shot {index}]")
        for index, timestamp in enumerate(timestamps, start=1)
    )


def _expand_structured_shot_timestamps(
    structure: str, timestamps: Sequence[str], structure_name: str
) -> str:
    shot_count = structure.count("<<shot>>")
    timestamp_count = structure.count("<<timestamp>>")
    if shot_count != timestamp_count:
        raise ValueError(
            f"{structure_name} must use <<shot>> and <<timestamp>> together."
        )
    if shot_count > 1:
        raise ValueError(
            f"{structure_name} may contain one <<shot>> and <<timestamp>> pair."
        )
    if not shot_count:
        return structure

    before, repeated = structure.split("<<shot>>")
    between, after = repeated.split("<<timestamp>>")
    return before + ", ".join(
        f"Shot {index}{between}{timestamp}"
        for index, timestamp in enumerate(timestamps, start=1)
    ) + after


def build_structured_video_timeline_text(
    video_runtime: float,
    timestamps: Sequence[str],
    structured_timeline_text_structure: str = VIDEO_STRUCTURED_TIMELINE_TEXT_STRUCTURE,
    index_offset: int = 0,
) -> str:
    _validate_video_timeline_structure(
        structured_timeline_text_structure,
        _VIDEO_STRUCTURED_TIMELINE_TEXT_MARKERS,
        "structured timeline text structure",
    )
    references = [
        f"<Picture {index + index_offset}> at {timestamp}"
        for index, timestamp in enumerate(timestamps, start=1)
    ]
    reference_text = (
        references[0]
        if len(references) == 1
        else f"{', '.join(references[:-1])} and {references[-1]}"
        if references
        else ""
    )
    structured_timeline_text_structure = _expand_structured_shot_timestamps(
        structured_timeline_text_structure,
        timestamps,
        "Structured timeline text structure",
    )
    return (
        structured_timeline_text_structure.replace(
            "<<duration>>", f"{video_runtime:g}"
        )
        .replace("<<segments>>", str(len(timestamps)))
        .replace("<<timestamps>>", ", ".join(timestamps))
        .replace("<<references>>", reference_text)
    )


def build_text_video_timeline_text(timestamps: Sequence[str], structure: str) -> str:
    _validate_video_timeline_structure(
        structure, _VIDEO_TEXT_TIMELINE_TEXT_MARKERS, "text timeline structure"
    )
    return "\n".join(
        structure.replace("<<shot>>", f"Shot {index}").replace(
            "<<timestamp>>", timestamp
        ).replace("<<time>>", timestamp)
        for index, timestamp in enumerate(timestamps, start=1)
    )


def build_text_structured_video_timeline_text(
    video_runtime: float, timestamps: Sequence[str], structure: str
) -> str:
    _validate_video_timeline_structure(
        structure,
        _VIDEO_TEXT_STRUCTURED_TIMELINE_TEXT_MARKERS,
        "text structured timeline structure",
    )
    structure = _expand_structured_shot_timestamps(
        structure, timestamps, "Text structured timeline structure"
    )
    return (
        structure.replace("<<duration>>", f"{video_runtime:g}")
        .replace("<<segments>>", str(len(timestamps)))
        .replace("<<timestamps>>", ", ".join(timestamps))
    )


def _validate_video_timeline_structure(structure, markers, label):
    if not isinstance(structure, str) or not structure.strip():
        raise ValueError(f"Video {label} must not be empty.")
    unknown = sorted(set(re.findall(r"<<([^<>]+)>>", structure)) - markers)
    if unknown:
        raise ValueError(f"Unknown video {label} marker: <<{unknown[0]}>>.")
    return structure


def _timeline_input_images(image_inputs) -> list[torch.Tensor]:
    """Flatten autogrow IMAGE inputs in numeric socket and batch order."""
    if not isinstance(image_inputs, dict):
        raise ValueError("Images to Video Timeline requires at least one connected image.")

    def socket_number(name):
        match = re.search(r"\d+", name)
        return int(match.group()) if match else 0

    images = []
    for name in sorted(image_inputs, key=socket_number):
        value = image_inputs[name]
        if value is None:
            continue
        if not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] < 1:
            raise ValueError("Images to Video Timeline inputs must be nonempty BHWC IMAGE batches.")
        images.extend(value[index:index + 1] for index in range(value.shape[0]))
    if not images:
        raise ValueError("Images to Video Timeline requires at least one connected image.")
    return images


def _timeline_image_outputs(image_inputs, resize_images: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Optionally normalize images for batching while preserving the ordered image list."""
    images = _timeline_input_images(image_inputs)
    if not resize_images:
        return torch.zeros((1, 64, 64, 3), dtype=images[0].dtype, device=images[0].device), images
    first_height, first_width = images[0].shape[1:3]
    max_channels = max(image.shape[-1] for image in images)
    normalized = []
    for image in images:
        if image.shape[-1] < max_channels:
            image = torch.nn.functional.pad(image, (0, max_channels - image.shape[-1]), value=1.0)
        if image.shape[1:3] != (first_height, first_width):
            method = "area" if first_height < image.shape[1] or first_width < image.shape[2] else "bicubic"
            image = resize_nchw(image.movedim(-1, 1), first_width, first_height, method).movedim(1, -1)
        normalized.append(image)
    image_batch = torch.cat(normalized, dim=0)
    return image_batch, [image_batch[index:index + 1] for index in range(image_batch.shape[0])]


def _truncated_normal_quantile(quantile: float, center: float) -> float:
    """Return a unit-interval Gaussian quantile centered at a local focus value."""
    deviation = 0.18
    root_two = math.sqrt(2.0)
    lower = 0.5 * (1.0 + math.erf(-center / (deviation * root_two)))
    upper = 0.5 * (1.0 + math.erf((1.0 - center) / (deviation * root_two)))
    target = lower + quantile * (upper - lower)
    low, high = 0.0, 1.0
    for _ in range(48):
        midpoint = (low + high) / 2.0
        value = 0.5 * (1.0 + math.erf((midpoint - center) / (deviation * root_two)))
        if value < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def focused_timeline_timestamps(count: int, duration: float, focus_areas: int, focus_one: float, focus_two: float, focus_three: float, anchor_start: bool = True, anchor_end: bool = True) -> list[float]:
    """Place ordered timestamps across a duration with optional local focus peaks."""
    if count < 1:
        raise ValueError("Focused timeline requires at least one timestamp.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Images to Video Timeline duration must be finite and greater than zero.")
    if isinstance(focus_areas, bool) or focus_areas not in range(4):
        raise ValueError("Images to Video Timeline focus_areas must be an integer from 0 to 3.")
    focuses = (focus_one, focus_two, focus_three)
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in focuses):
        raise ValueError("Images to Video Timeline focus values must be finite values from 0 to 1.")
    if anchor_start and count == 1:
        return [0.0]
    if focus_areas == 0:
        anchors = int(anchor_start) + int(anchor_end)
        denominator = count - anchors + 1
        start = 0 if anchor_start else 1
        return [duration * index / denominator for index in range(start, start + count)]

    movable_count = count - int(anchor_start) - int(anchor_end)
    timestamps = [0.0] if anchor_start else []
    for index in range(1, movable_count + 1):
        global_quantile = index / (movable_count + 1)
        section = min(int(global_quantile * focus_areas), focus_areas - 1)
        local_quantile = global_quantile * focus_areas - section
        local_position = _truncated_normal_quantile(local_quantile, focuses[section])
        timestamps.append(duration * (section + local_position) / focus_areas)
    if anchor_end:
        timestamps.append(duration)
    return timestamps


def images_to_video_timeline(image_inputs, duration: float, focus_areas: int, focus_one: float, focus_two: float, focus_three: float, last_image_is_final: bool, resize_images: bool, timestamp_format: str, timeline_style: str, timeline_text_structure: str, structured_timeline_text_structure: str, index_offset: int = 0) -> SampledVideoFrames:
    """Normalize supplied images and assign their manual video timeline timestamps."""
    duration = h3_video_length_from_seconds(duration) / 24
    image_batch, image_list = _timeline_image_outputs(image_inputs, resize_images)
    raw_timestamps = focused_timeline_timestamps(len(image_list), duration, focus_areas, focus_one, focus_two, focus_three, anchor_end=last_image_is_final)
    timestamps = [format_video_timestamp(timestamp, timestamp_format) for timestamp in raw_timestamps]
    return SampledVideoFrames(
        image_batch=image_batch,
        image_list=image_list,
        timestamps=timestamps,
        timestamps_text=", ".join(timestamps),
        timeline_text=build_video_timeline_text(timestamps, timeline_style, timeline_text_structure, index_offset),
        video_runtime=duration,
        structured_timeline_text=build_structured_video_timeline_text(duration, timestamps, structured_timeline_text_structure, index_offset),
    )


def sample_video_frames_as_images(
    video,
    sampling_strategy: str,
    maximum_frames: int,
    include_zero_time: bool,
    minimum_spacing_seconds: float,
    keyframe_stride: int,
    timestamp_format: str,
    timeline_style: str,
    timeline_text_structure: str,
    structured_timeline_text_structure: str,
    index_offset: int = 0,
    focus_areas: int = 0,
    focus_one: float = 0.5,
    focus_two: float = 0.5,
    focus_three: float = 0.5,
) -> SampledVideoFrames:
    video_runtime = h3_video_length_from_seconds(float(video.get_duration())) / 24
    source_factory = _video_source_factory(video)
    records = scan_video_frame_records(video, source_factory)
    selected, output_timestamps = _select_video_frame_records_and_timestamps(
        records,
        sampling_strategy,
        maximum_frames,
        include_zero_time,
        minimum_spacing_seconds,
        keyframe_stride,
        timestamp_format,
        focus_areas,
        focus_one,
        focus_two,
        focus_three,
    )
    image_batch, image_list = decode_selected_video_frames(
        video, selected, source_factory
    )
    timestamps = [
        format_video_timestamp(timestamp, timestamp_format)
        for timestamp in output_timestamps
    ]

    return SampledVideoFrames(
        image_batch=image_batch,
        image_list=image_list,
        timestamps=timestamps,
        timestamps_text=", ".join(timestamps),
        timeline_text=build_video_timeline_text(
            timestamps, timeline_style, timeline_text_structure, index_offset
        ),
        video_runtime=video_runtime,
        structured_timeline_text=build_structured_video_timeline_text(
            video_runtime, timestamps, structured_timeline_text_structure, index_offset
        ),
    )


def video_timeline_text(
    duration: float,
    segment_count: int,
    focus_areas: int,
    focus_one: float,
    focus_two: float,
    focus_three: float,
    timestamp_format: str,
    timeline_text_structure: str,
    structured_timeline_text_structure: str,
) -> tuple[str, str, float, str]:
    duration = h3_video_length_from_seconds(duration) / 24
    raw_timestamps = focused_timeline_timestamps(
        segment_count, duration, focus_areas, focus_one, focus_two, focus_three
    )
    timestamps = [
        format_video_timestamp(timestamp, timestamp_format)
        for timestamp in raw_timestamps
    ]
    return (
        ", ".join(timestamps),
        build_text_video_timeline_text(timestamps, timeline_text_structure),
        duration,
        build_text_structured_video_timeline_text(
            duration, timestamps, structured_timeline_text_structure
        ),
    )
