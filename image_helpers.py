from __future__ import annotations

import io
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


def prepare_h3_reference_video_components(video, megapixels: float, duration_seconds: float = 0.0):
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

    source_count = source_frames.shape[0]
    selected_seconds = source_count / source_rate
    if duration_seconds > 0:
        selected_seconds = min(selected_seconds, duration_seconds)
    frame_count = h3_video_length_from_seconds(selected_seconds)
    frame_indices = [min(round(index * source_rate / 24), source_count - 1) for index in range(frame_count)]
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
            prepared_frames.movedim(-1, 1), output_width, output_height, "bicubic", "center",
        ).clamp(0.0, 1.0).movedim(1, -1).contiguous()

    soundtrack = components.audio
    audio_window = round(frame_count / 24 * 32000)
    aligned_samples = ((audio_window + 799) // 800) * 800
    if soundtrack is not None and soundtrack.get("waveform") is not None and soundtrack["waveform"].numel() > 0:
        sample_rate = int(soundtrack["sample_rate"])
        if sample_rate <= 0:
            raise ValueError("Reference audio must have a positive sample rate.")
        sample_count = round(frame_count / 24 * sample_rate)
        audio_samples = soundtrack["waveform"][..., :sample_count]
        if sample_rate != 32000:
            audio_samples = torchaudio.functional.resample(audio_samples, sample_rate, 32000)
        audio_samples = audio_samples[..., :audio_window]
        # Align before Core's generic VAE crop; preserve the start of the audio.
        audio_samples = torch.nn.functional.pad(audio_samples, (0, aligned_samples - audio_samples.shape[-1]))
    else:
        audio_samples = torch.zeros(1, 2, aligned_samples)
    prepared_audio = {"waveform": audio_samples, "sample_rate": 32000}
    return prepared_frames, prepared_audio, output_width, output_height, frame_count, video_frames


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
