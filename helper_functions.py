from typing import List, Union
from PIL import Image
from enum import Enum
import numpy as np
import re
import torch
import torch.nn.functional as F
import cv2
import math
import json
import codecs


def resize_nchw(samples: torch.Tensor, width: int, height: int, method: str, crop: str = "disabled") -> torch.Tensor:
    """Resize NCHW tensors without reducing image data to uint8."""
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("Resize dimensions must be positive.")
    source_height, source_width = samples.shape[-2:]
    if crop == "center":
        source_aspect = source_width / source_height
        target_aspect = width / height
        x = round((source_width - source_width * (target_aspect / source_aspect)) / 2) if source_aspect > target_aspect else 0
        y = round((source_height - source_height * (source_aspect / target_aspect)) / 2) if source_aspect < target_aspect else 0
        samples = samples[..., y:source_height - y, x:source_width - x]
        source_height, source_width = samples.shape[-2:]
    elif crop != "disabled":
        raise ValueError(f"Unsupported resize crop mode: {crop!r}.")
    if (source_width, source_height) == (width, height):
        return samples
    if method == "lanczos":
        original_device, original_dtype = samples.device, samples.dtype
        cpu = samples.detach().to(device="cpu", dtype=torch.float32)
        batches = []
        for batch in cpu:
            channels = [
                torch.from_numpy(np.asarray(Image.fromarray(channel.numpy()).resize((width, height), Image.Resampling.LANCZOS)).copy())
                for channel in batch
            ]
            batches.append(torch.stack(channels))
        return torch.stack(batches).to(device=original_device, dtype=original_dtype)
    if method not in ("nearest-exact", "bilinear", "area", "bicubic"):
        raise ValueError(f"Unsupported resize method: {method!r}.")
    options = {}
    if method in ("bilinear", "bicubic"):
        options["align_corners"] = False
        options["antialias"] = width < source_width or height < source_height
    return F.interpolate(samples, size=(height, width), mode=method, **options)

def round_to_nearest(n, m):
    return int((n + (m / 2)) // m) * m


# Tensor to PIL
def simpletensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# PIL to Tensor
def simplepil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2tensor(image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> List[Image.Image]:
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    return [
        Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    ]

def hex_to_rgb(hex_str: str, default=(255, 255, 255)):
    hex_str = hex_str.lstrip('#')
    try:
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        elif len(hex_str) == 8:
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4, 6))
    except ValueError:
        pass
    return default

def math_diag(H: int, W: int) -> float:
    return math.sqrt(H * H + W * W)

def pct_to_px(pct: float, diag: float) -> int:
    return max(0, round(abs(pct) * diag / 100.0))

def blur_kernel_for_diag(diag: float) -> tuple:
    k = max(3, int(round(diag / 724.0 * 3)))
    if k % 2 == 0: k += 1
    return (k, k)

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    lin = np.where(rgb <= 0.04045,
                   rgb / 12.92,
                   ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],[0.2126729, 0.7151522, 0.0721750],[0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T / np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    def f(t):
        return np.where(t > (6/29)**3,
                        t ** (1/3),
                        t / (3 * (6/29)**2) + 4/29)

    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])
    return np.stack([116*fy - 16, 500*(fx - fy), 200*(fy - fz)], axis=-1).astype(np.float32)

def string_to_color(color_string: str) -> List[int]:
    color_list = [0, 0, 0]  # Default fallback (black)

    if ',' in color_string:
        # Handle CSV format (e.g., "255, 0, 0" or "255, 0, 0, 128" or "1.0, 0.5, 0.0")
        try:
            values = [float(channel.strip()) for channel in color_string.split(',')]
            # Convert to 0-255 range if values are in 0-1 range
            if all(0 <= v <= 1 for v in values):
                color_list = [int(v * 255) for v in values]
            else:
                color_list = [int(v) for v in values]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")
    elif color_string.startswith('#') or (color_string.lstrip('#').isalnum() and not color_string.lstrip('#').replace('.', '', 1).isdigit()):
        # Could be Hex format or color name
        color_string_stripped = color_string.lstrip('#')
        # Try hex first
        if len(color_string_stripped) in [6, 8] and all(c in '0123456789ABCDEFabcdef' for c in color_string_stripped):
            if len(color_string_stripped) == 6:  # #RRGGBB
                color_list = [int(color_string_stripped[i:i+2], 16) for i in (0, 2, 4)]
            elif len(color_string_stripped) == 8:  # #RRGGBBAA
                color_list = [int(color_string_stripped[i:i+2], 16) for i in (0, 2, 4, 6)]
        else:
            # Try color name (e.g., "red", "blue", "cyan")
            try:
                rgb = ImageColor.getrgb(color_string)
                color_list = list(rgb)
            except ValueError:
                logging.warning(f"Invalid color name or hex format: {color_string}. Using default black.")
    else:
        # Handle single value (grayscale) - can be int or float
        try:
            value = float(color_string.strip())
            # Convert to 0-255 range if it's a float between 0-1
            if 0 <= value <= 1:
                value = int(value * 255)
            else:
                value = int(value)
            color_list = [value, value, value]
        except ValueError:
            logging.warning(f"Invalid color format: {color_string}. Using default black.")

    # Clip values to valid range
    color_list = np.clip(color_list, 0, 255).tolist()

    return color_list

def dis_flow(gray_a: np.ndarray, gray_b: np.ndarray, preset: int) -> np.ndarray:
    return cv2.DISOpticalFlow_create(preset).calc(gray_a, gray_b, None)

def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    H, W = flow.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xx + flow[..., 0]).astype(np.float32)
    map_y = (yy + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

def occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, threshold: float) -> np.ndarray:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    return (err > threshold).astype(np.float32)

def grow_mask(mask: np.ndarray, grow_px: int) -> np.ndarray:
    if grow_px == 0: return mask
    radius = abs(grow_px)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    op = cv2.MORPH_DILATE if grow_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), op, k).astype(np.float32)

def auto_delta_e_threshold(delta_e: np.ndarray) -> float:
    p75 = float(np.percentile(delta_e, 75))
    p90 = float(np.percentile(delta_e, 90))
    spread = p90 - p75
    threshold = p75 + max(spread * 0.4, 3.0) if spread > 5.0 else p75 + max(spread * 0.6, 4.0)
    return float(np.clip(threshold, 4.0, 60.0))

def auto_occlusion_threshold(flow_fwd: np.ndarray, flow_bwd: np.ndarray) -> float:
    H, W = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bwd_x = cv2.remap(flow_bwd[..., 0], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    bwd_y = cv2.remap(flow_bwd[..., 1], xx + flow_fwd[..., 0], yy + flow_fwd[..., 1],
                      cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    err = np.sqrt((flow_fwd[..., 0] + bwd_x)**2 + (flow_fwd[..., 1] + bwd_y)**2)
    p85 = float(np.percentile(err, 85))
    p95 = float(np.percentile(err, 95))
    threshold = p95 + max((p95 - p85) * 0.5, 0.5)
    return float(np.clip(threshold, 1.0, 15.0))

def composite(original_np: np.ndarray,
               generated_np: np.ndarray,
               delta_e_threshold: float,
               flow_preset: int,
               occlusion_threshold: float,
               grow_px: int,
               close_radius: int,
               min_region_px: int,
               feather_px: float) -> tuple:

    H, W = original_np.shape[:2]
    diag = math_diag(H, W)

    orig_u8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
    gen_u8  = (np.clip(generated_np, 0, 1) * 255).astype(np.uint8)
    gray_orig = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2GRAY)
    gray_gen  = cv2.cvtColor(gen_u8,  cv2.COLOR_RGB2GRAY)

    flow_fwd = dis_flow(gray_orig, gray_gen, flow_preset)
    flow_bwd = dis_flow(gray_gen, gray_orig, flow_preset)

    warped_gen_dense = warp(generated_np.astype(np.float32), flow_fwd)

    blur_kernel = blur_kernel_for_diag(diag)
    orig_blur = cv2.GaussianBlur(original_np, blur_kernel, 0)
    wgen_blur = cv2.GaussianBlur(warped_gen_dense, blur_kernel, 0)

    orig_lab = rgb_to_lab(orig_blur.reshape(-1, 3)).reshape(H, W, 3)
    wgen_lab = rgb_to_lab(wgen_blur.reshape(-1, 3)).reshape(H, W, 3)

    lab_diff = orig_lab - wgen_lab
    lab_diff[..., 0] *= 0.7
    delta_e = np.sqrt((lab_diff**2).sum(axis=2))

    sk = max(blur_kernel_for_diag(diag)[0], 5)
    if sk % 2 == 0: sk += 1
    delta_e_smooth = cv2.GaussianBlur(delta_e, (sk, sk), 0)

    auto_report = {}
    if delta_e_threshold < 0:
        delta_e_threshold = auto_delta_e_threshold(delta_e_smooth)
        auto_report["auto_delta_e"] = delta_e_threshold

    if occlusion_threshold < 0:
        occlusion_threshold = auto_occlusion_threshold(flow_fwd, flow_bwd)
        auto_report["auto_occlusion"] = occlusion_threshold

    occluded = occlusion_mask(flow_fwd, flow_bwd, occlusion_threshold)

    changed = np.maximum((delta_e_smooth > delta_e_threshold).astype(np.float32), occluded)

    if grow_px != 0:
        changed = grow_mask(changed, grow_px)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(np.float32)
    if min_region_px > 0:
        n, labeled, stats_cc, _ = cv2.connectedComponentsWithStats((changed > 0.5).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats_cc[i, cv2.CC_STAT_AREA] < min_region_px:
                changed[labeled == i] = 0

    sharp_mask = changed.copy()

    if feather_px > 0:
        inv_mask = (sharp_mask < 0.5).astype(np.uint8)
        if inv_mask.min() == 0:
            dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)
            fade_dist = feather_px * 3.0
            t = np.clip(1.0 - (dist / fade_dist), 0.0, 1.0)
            composite_mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        else:
            composite_mask = sharp_mask
    else:
        composite_mask = sharp_mask

    y_grid, x_grid = np.mgrid[0:H:10, 0:W:10]
    pts_orig = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2).astype(np.float32)

    flow_sub = flow_fwd[0:H:10, 0:W:10].reshape(-1, 2)
    mask_sub = sharp_mask[0:H:10, 0:W:10].reshape(-1)

    bg_idx = mask_sub < 0.1
    M = None
    if bg_idx.sum() > 10:
        src_pts = pts_orig[bg_idx]
        dst_pts = src_pts + flow_sub[bg_idx]

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if M is not None and not np.allclose(M, identity, atol=1e-6, rtol=0.0):
        final_aligned_gen = cv2.warpAffine(
            generated_np.astype(np.float32),
            M.astype(np.float64),
            (W, H),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT
        )
    else:
        final_aligned_gen = generated_np

    m3 = composite_mask[..., np.newaxis]
    result = np.clip(original_np * (1.0 - m3) + final_aligned_gen * m3, 0, 1)

    flow_mag = np.sqrt((flow_fwd**2).sum(axis=2))
    n_changed = int((sharp_mask > 0.5).sum())
    stats = {
        "changed_pct":    100 * n_changed / (H * W),
        "occluded_px":    int(occluded.sum()),
        "flow_mean_px":   float(flow_mag.mean()),
        "flow_p99_px":    float(np.percentile(flow_mag, 99)),
        "median_de":      float(np.median(delta_e)),
        "resolution":     f"{W}x{H}",
        "diagonal_px":    round(diag),
    }
    stats.update(auto_report)

    return result, composite_mask, stats


def fill_mask_from_edges(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    inpaint_radius: int,
    edge_blend_blur: int,
) -> torch.Tensor:

    batch_size = image_tensor.size(0)
    out_tensors = []
    mask_batch = mask_tensor.size(0)

    for i in range(batch_size):
        # ComfyUI images are typically [B, H, W, C], float32, 0.0-1.0
        # Convert to numpy uint8 for OpenCV
        original_np = np.clip(image_tensor[i].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        img_np = np.clip(255.0 * original_np, 0, 255).astype(np.uint8)

        # If the image has an alpha channel, we only inpaint the RGB channels
        has_alpha = img_np.shape[-1] == 4
        if has_alpha:
            alpha_channel = original_np[:, :, 3]
            img_np = img_np[:, :, :3]
            original_rgb = original_np[:, :, :3]
        else:
            original_rgb = original_np

        # Extract and format the mask
        mask_i = i if i < mask_batch else 0
        m_t = mask_tensor[mask_i].cpu().numpy()

        if m_t.ndim > 2:
            m_t = m_t.squeeze()
        if m_t.shape[:2] != img_np.shape[:2]:
            m_t = cv2.resize(m_t, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        # OpenCV inpaint requires a strictly binary 8-bit mask
        mask_np = np.clip(255.0 * m_t, 0, 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        # Apply Navier-Stokes Inpainting (pulls edge pixels inward)
        inpainted = cv2.inpaint(img_np, mask_binary, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)

        # Smooth boundary blending
        if edge_blend_blur > 0:
            # Ensure blur kernel size is odd
            blur_size = edge_blend_blur if edge_blend_blur % 2 == 1 else edge_blend_blur + 1

            # Create a soft mask for alpha blending the inpainted result back onto the original
            soft_mask = cv2.GaussianBlur(mask_np.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
            soft_mask = soft_mask[:, :, np.newaxis]  # Reshape to [H, W, 1] for broadcasting

            inpainted_f = inpainted.astype(np.float32) / 255.0

            # Composite: Original image where mask is 0, Inpainted image where mask is 1
            final_np = original_rgb * (1.0 - soft_mask) + inpainted_f * soft_mask
        else:
            hard_mask = (mask_binary.astype(np.float32) / 255.0)[:, :, np.newaxis]
            final_np = original_rgb * (1.0 - hard_mask) + (inpainted.astype(np.float32) / 255.0) * hard_mask

        # Restore alpha channel if it existed
        if has_alpha:
            final_np = np.dstack((final_np, alpha_channel))

        # Convert back to ComfyUI tensor [1, H, W, C]
        out_tensor = torch.from_numpy(np.clip(final_np, 0.0, 1.0).astype(np.float32)).unsqueeze(0)
        out_tensors.append(out_tensor)

    return torch.cat(out_tensors, dim=0).to(image_tensor)


def create_stretched_patch(img: np.ndarray, mask_binary: np.ndarray, axis: str, sample_thickness: int) -> np.ndarray:
    """
    Scans rows/cols to find the exact organic mask edges, samples a patch of 'sample_thickness',
    mirrors it, stretches it across the gap, and cross-fades.
    """
    H, W_img, C = img.shape
    out = np.copy(img).astype(np.float32)
    m_bool = (mask_binary > 127).astype(np.int8)

    if axis == 'horizontal':
        for y in range(H):
            row_mask = m_bool[y]
            if not np.any(row_mask): continue

            padded = np.pad(row_mask, (1, 1), mode='constant', constant_values=0)
            diff = np.diff(padded)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            for start, end in zip(starts, ends):
                w = end - start + 1

                L_start = max(0, start - sample_thickness)
                L_width = start - L_start
                R_end = min(W_img, end + 1 + sample_thickness)
                R_width = R_end - (end + 1)

                L_stretch, R_stretch = None, None

                if L_width > 0:
                    L_strip = img[y, L_start:start].reshape(1, L_width, C)
                    L_strip = L_strip[:, ::-1, :] # Mirror horizontally
                    L_stretch = cv2.resize(L_strip, (w, 1)).reshape(w, C).astype(np.float32)

                if R_width > 0:
                    R_strip = img[y, end+1:R_end].reshape(1, R_width, C)
                    R_strip = R_strip[:, ::-1, :] # Mirror horizontally
                    R_stretch = cv2.resize(R_strip, (w, 1)).reshape(w, C).astype(np.float32)

                weights = np.linspace(1.0, 0.0, w).reshape(w, 1)

                if L_stretch is not None and R_stretch is not None:
                    out[y, start:end+1] = L_stretch * weights + R_stretch * (1.0 - weights)
                elif L_stretch is not None:
                    out[y, start:end+1] = L_stretch
                elif R_stretch is not None:
                    out[y, start:end+1] = R_stretch

    elif axis == 'vertical':
        for x in range(W_img):
            col_mask = m_bool[:, x]
            if not np.any(col_mask): continue

            padded = np.pad(col_mask, (1, 1), mode='constant', constant_values=0)
            diff = np.diff(padded)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            for start, end in zip(starts, ends):
                h_seg = end - start + 1

                T_start = max(0, start - sample_thickness)
                T_height = start - T_start
                B_end = min(H, end + 1 + sample_thickness)
                B_height = B_end - (end + 1)

                T_stretch, B_stretch = None, None

                if T_height > 0:
                    T_strip = img[T_start:start, x].reshape(T_height, 1, C)
                    T_strip = T_strip[::-1, :, :] # Mirror vertically
                    T_stretch = cv2.resize(T_strip, (1, h_seg)).reshape(h_seg, C).astype(np.float32)

                if B_height > 0:
                    B_strip = img[end+1:B_end, x].reshape(B_height, 1, C)
                    B_strip = B_strip[::-1, :, :] # Mirror vertically
                    B_stretch = cv2.resize(B_strip, (1, h_seg)).reshape(h_seg, C).astype(np.float32)

                weights = np.linspace(1.0, 0.0, h_seg).reshape(h_seg, 1)

                if T_stretch is not None and B_stretch is not None:
                    out[start:end+1, x] = T_stretch * weights + B_stretch * (1.0 - weights)
                elif T_stretch is not None:
                    out[start:end+1, x] = T_stretch
                elif B_stretch is not None:
                    out[start:end+1, x] = B_stretch

    return out


def iterative_directional_stretch_fill(
    image_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    stretch_axis: str,
    sample_thickness: int,
    edge_blend_blur: int,
    iterations: int,
    mask_decay_pixels: int,
) -> torch.Tensor:

    batch_size = image_tensor.size(0)
    out_tensors = []
    mask_batch = mask_tensor.size(0)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for i in range(batch_size):
        img_np = np.clip(image_tensor[i].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)

        has_alpha = img_np.shape[-1] == 4
        if has_alpha:
            alpha_channel = img_np[:, :, 3].copy()
            img_np = img_np[:, :, :3]

        mask_i = i if i < mask_batch else 0
        m_t = mask_tensor[mask_i].cpu().numpy()

        if m_t.ndim > 2:
            m_t = m_t.squeeze()
        if m_t.shape[:2] != img_np.shape[:2]:
            m_t = cv2.resize(m_t, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_np = np.clip(255.0 * m_t, 0, 255).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

        working_img = np.copy(img_np)
        working_mask = np.copy(mask_binary)

        for step in range(iterations):
            if cv2.countNonZero(working_mask) == 0:
                break

            current_axis = stretch_axis
            if current_axis == 'auto':
                x, y, w, h = cv2.boundingRect(working_mask)
                current_axis = 'horizontal' if w < h else 'vertical'

            # Pass the full image and mask into our stretch function
            canvas = create_stretched_patch(working_img, working_mask, current_axis, sample_thickness)

            if edge_blend_blur > 0:
                blur_size = edge_blend_blur if edge_blend_blur % 2 == 1 else edge_blend_blur + 1
                soft_mask = cv2.GaussianBlur(working_mask.astype(np.float32) / 255.0, (blur_size, blur_size), 0)
                soft_mask = soft_mask[:, :, np.newaxis]
            else:
                soft_mask = (working_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

            working_f = working_img.astype(np.float32)
            merged = working_f * (1.0 - soft_mask) + canvas * soft_mask
            working_img = np.clip(merged, 0.0, 1.0).astype(np.float32)

            if step < iterations - 1 and mask_decay_pixels > 0:
                working_mask = cv2.erode(working_mask, erosion_kernel, iterations=mask_decay_pixels)

        if has_alpha:
            working_img = np.dstack((working_img, alpha_channel))

        out_tensor = torch.from_numpy(working_img.astype(np.float32)).unsqueeze(0)
        out_tensors.append(out_tensor)

    return torch.cat(out_tensors, dim=0).to(image_tensor)

def gaussian_blur_nchw(img_nchw, sigma_px):
    if sigma_px <= 0:
        return img_nchw
    radius = max(1, int(3.0 * float(sigma_px)))
    k = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
    k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
    k1 = k1 / k1.sum()
    kx = k1.view(1, 1, 1, k)
    ky = k1.view(1, 1, k, 1)
    c = img_nchw.shape[1]
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
    img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
    return img_nchw




def get_token_count(clip, text):
        """
        Robustly tokenizes a text segment and returns the number of its content tokens.
        """
        if not text:
            return 0

        tokens = clip.tokenize(text)

        max_content_len = 0
        for key in tokens:
            if len(tokens[key]) > 0 and len(tokens[key][0]) > 0:

                content_len = len(tokens[key][0]) - 2
                if content_len > max_content_len:
                    max_content_len = content_len

        return max(0, max_content_len)


def get_token_count_scaled(clip, text, **kwargs):
    """
    Robustly tokenizes a text segment and returns the number of its content tokens.
    Calculates true length by ignoring padding tokens added by fixed-length tokenizers (like Qwen3).
    """
    # Only return 0 if no text AND no llama_template (which adds tokens)
    if not text and not kwargs.get("llama_template"):
        return 0

    tokens = clip.tokenize(text, **kwargs)

    max_content_len = 0
    for key in tokens:
        if len(tokens[key]) > 0 and len(tokens[key][0]) > 0:
            token_list = tokens[key][0]

            # Count tokens that aren't padding.
            # In ComfyUI, padding tokens are usually 0 or the end-of-text token repeated.
            # We find the first occurrence of the end-of-text token or look for the padding.

            # For robust counting across models:
            # 1. Start with the full length
            # 2. Subtract 2 (for start and end tokens)
            # 3. If the length is still suspiciously large (like 510 or 254),
            #    it's likely a padded tokenizer. We try to find the actual content.

            raw_len = len(token_list)
            content_len = raw_len - 2

            # If it looks like a fixed-length padded result (common for T5/Qwen in ComfyUI)
            if raw_len >= 77:
                # Find the actual used length.
                # Most tokenizers pad with 0 or the last token (EOS).
                if isinstance(token_list, torch.Tensor):
                    ids = token_list.tolist()
                else:
                    ids = token_list

                # Qwen/Llama usually have a start token at 0.
                # Let's count how many IDs are actually distinct from the last token in the list
                # (which is usually the padding token)
                pad_id = ids[-1]
                actual_count = 0
                for i in range(1, len(ids) - 1): # Skip start token at 0 and end token
                    if ids[i] != pad_id:
                        actual_count += 1
                    else:
                        break # Hit padding
                content_len = actual_count

            if content_len > max_content_len:
                max_content_len = content_len

    return max(0, max_content_len)

VIDEO_PROMPT_ROLES = ("general", "system", "instruction", "bonus")

_VIDEO_PROMPT_COMMON_REPLACEMENTS = (
    # Reference-relative opening state and continuity constraints. Keep the
    # specific multi-clause forms ahead of the shorter directive forms.
    (
        r"(?i)\b(?:keep(?:ing)?|maintain(?:ing)?|ensure|make\s+sure)\s+(?:the\s+)?subject(?:'s)?\s+position\s+and\s+(?:their\s+)?pose\s+(?:the\s+)?same\s+as\s+(?:the\s+)?reference\b",
        "Use the subject's referenced placement and pose as the opening state, then allow natural movement while preserving identity and anatomical continuity",
    ),
    (
        r"(?i)\b(?:make\s+sure|ensure)\s+(?:that\s+)?the\s+subject\s+is\s+in\s+the\s+same\s+pose\s+and\s+angle\b",
        "Use the referenced pose and camera angle as the opening state, then develop coherent subject and camera motion while preserving identity and anatomical continuity",
    ),
    (
        r"(?i)\b(?:make\s+sure|ensure)\s+(?:that\s+)?the\s+subject\s+is\s+in\s+the\s+same\s+position\b",
        "Use the referenced subject placement as the opening position, then allow motivated movement through the scene",
    ),
    (
        r"(?i)\b(?:keep(?:ing)?|maintain(?:ing)?|ensure|make\s+sure)\s+(?:that\s+)?(?:the\s+)?composition\s+(?:of\s+the\s+image\s+)?(?:is\s+|the\s+)?same\s+as\s+(?:the\s+)?reference\b",
        "Use the reference composition as the opening framing, then preserve spatial continuity while allowing coherent subject and camera motion",
    ),
    (
        r"(?i)\bkeeping\s+(?:the\s+)?composition\s+and\s+structure\s+of\s+the\s+image\s+(?:the\s+)?same\s+as\s+(?:the\s+)?reference\b",
        "using the reference composition and scene structure as the opening state while maintaining spatial and temporal continuity through motion",
    ),
    (
        r"(?i)\b(?:make\s+sure|ensure)\s+(?:that\s+)?the\s+background\s+is\s+the\s+same\b",
        "Preserve the referenced environment identity and layout while allowing camera parallax and physically consistent environmental motion",
    ),
    (
        r"(?i)\bnot\s+changing\s+the\s+positioning\s+of\s+subjects\s+in\s+(?:the\s+)?image\b",
        "using their referenced placement as the opening state before motivated subject movement",
    ),
    (
        r"(?i)\bkeeping\s+the\s+structure\s+of\s+the\s+image\s+intact\b",
        "preserving subject identity, scene structure, and temporal continuity throughout the video",
    ),
    (
        r"(?i)\bstrictly\s+maintaining\s+their\s+original\s+appearance\b",
        "strictly preserving their visual identity throughout motion",
    ),
    # Static body, gaze, framing, focus, and illumination constraints.
    (
        r"(?i)\bkeep\s+leg\s+positions?\b",
        "Use the referenced leg arrangement as the opening state, then animate coherent leg movement with anatomical continuity",
    ),
    (
        r"(?i)\bkeep\s+arm\s+positions?\b",
        "Use the referenced arm arrangement as the opening state, then animate coherent arm movement with anatomical continuity",
    ),
    (
        r"(?i)\bkeep\s+(?:the\s+)?pose\b",
        "Use the referenced pose as the opening pose, then allow natural movement with anatomical continuity",
    ),
    (
        r"(?i)\bkeep\s+(?:the\s+)?angle\b",
        "Use the referenced camera angle as the opening view, then allow coherent camera development",
    ),
    (
        r"(?i)\bkeep\s+(?:the\s+)?viewing\s+direction\b",
        "Use the referenced viewing direction initially, then allow motivated gaze changes",
    ),
    (
        r"(?i)\bkeep\s+(?:the\s+)?eyes\b",
        "Preserve the eyes' identity and appearance while allowing natural blinking and gaze motion",
    ),
    (
        r"(?i)\bkeep\s+in\s+focus\b",
        "Maintain sharp subject focus continuously through motion",
    ),
    (
        r"(?i)\bensure\s+(?:that\s+)?lighting\s+is\s+accurate(?:\s+for\s+the\s+scene)?\b",
        "Maintain temporally consistent scene lighting throughout the motion",
    ),
    (
        r"(?i)\bensure\s+(?:that\s+)?shadows\s+are\s+displayed\s+correctly\b",
        "Maintain physically consistent shadows that respond to subject, object, and camera motion",
    ),
    # Contextual task conversions. These patterns intentionally retain the
    # captured visual style instead of replacing medium nouns globally.
    (
        r"(?i)\bediting\s+requests\b",
        "image-to-video requests",
    ),
    (
        r"(?i)\bediting\s+request\b",
        "image-to-video request",
    ),
    (
        r"(?i)\bediting\s+instructions\b",
        "image-to-video instructions",
    ),
    (
        r"(?i)\bediting\s+instruction\b",
        "image-to-video instruction",
    ),
    (
        r"(?i)\bedit\s+requests\b",
        "image-to-video requests",
    ),
    (
        r"(?i)\bedit\s+request\b",
        "image-to-video request",
    ),
    (
        r"(?i)\bmodify\s+the\s+subject(?:'s)?\s+appearance,\s*pose,\s*position,\s*or\s*composition\s+as\s+requested\b",
        "Generate the requested subject appearance, use the reference pose, placement, and composition as the opening state, and then develop coherent motion",
    ),
    (
        r"(?i)\bedit\s+(?:the\s+)?image\s+style\s+into\s+([^\r\n.]+)",
        r"Generate the moving video in \1",
    ),
    (
        r"(?i)\bmodifying\s+the\s+style\s+to\s+look\s+like\s+([^\r\n.]+)",
        r"rendering the moving video with motion and physics authentic to \1",
    ),
    (
        r"(?i)\bmodifying\s+the\s+style\s+to\s+look\s+real\b",
        "rendering the moving video with realistic motion, physics, and appearance",
    ),
    (
        r"(?i)\bmodify\s+the\s+style\s+to\s+look\s+like\s+([^\r\n.]+)",
        r"Render the moving video with motion and physics authentic to \1",
    ),
    (
        r"(?i)\bmodify\s+the\s+style\s+to\s+look\s+real\b",
        "Render the moving video with realistic motion, physics, and appearance",
    ),
    (
        r"(?i)\bfocuses\s+on\s+edits\s+to\s+look\s+like\s+([^\r\n.]+)",
        r"focuses on generating video with motion and appearance authentic to \1",
    ),
    (
        r"(?i)\bfocus\s+on\s+edits\s+to\s+look\s+like\s+([^\r\n.]+)",
        r"focus on generating video with motion and appearance authentic to \1",
    ),
    (
        r"(?i)\bedit\s+this\s+into\s+(?:an?\s+)?([^\r\n.]+)",
        r"Generate a moving video with the appearance and content of \1",
    ),
    (
        r"(?i)\bedit\s+the\s+primary\s+subject\s+so\s+that\b",
        "Depict the primary subject throughout the video so that",
    ),
    (
        r"(?i)\bmodify\s+(?:(?:any\s+)?subjects?'?\s+)?appearance\s+to\s+show\s+real\s+details\b",
        "Render every subject with consistent real detail throughout the video",
    ),
    (
        r"(?i)\bmake\s+(?:it|the\s+image)\s+look\s+like\s+(?:an?\s+)?professional\s+([^\r\n.]*?photograph)\b",
        r"Maintain the appearance of professional \1 throughout the video",
    ),
    (
        r"(?i)\bthe\s+result\s+should\s+be\s+(?:an?\s+)?sharply\s+focused\s+photograph\b",
        "The video should maintain sharp focus and photographic realism throughout motion",
    ),
)

_VIDEO_PROMPT_ROLE_REPLACEMENTS = {
    "general": (),
    "system": (
        (
            r"(?i)\bimage-editing\s+expert\b",
            "image-to-video prompt expert",
        ),
        (
            r"(?i)\bimage\s+editing\s+descriptions\b",
            "image-to-video prompt descriptions",
        ),
        (
            r"(?i)\bconverting\s+simple\s+(?:editing|image-to-video)\s+requests?\s+into\s+clear\s+and\s+structured\s+instructions?\b",
            "converting simple image-to-video requests into clear and structured video instructions",
        ),
        (
            r"(?i)\bconvert\s+(?:editing|image-to-video)\s+requests?\s+into\s+one\s+concise\s+instruction\b",
            "convert image-to-video requests into one concise video instruction",
        ),
        (
            r"(?i)\bconverts\s+(?:editing|image-to-video)\s+requests?\s+into\s+one\s+concise\s+instruction\b",
            "converts image-to-video requests into one concise video instruction",
        ),
        (
            r"(?i)\bclothing\s+removal,\s*nude\s+body\s+edit\s+and\s+nude\s+details\s+edit\s+expert\b",
            "clothing-removal, nude-body, and nude-detail video depiction expert",
        ),
        (
            r"(?i)\bimage\s+editing\b",
            "image-to-video prompt generation",
        ),
        (
            r"(?i)\bediting\s+requests?\b",
            "image-to-video requests",
        ),
    ),
    "instruction": (
        (
            r"(?i)\bthe\s+result\s+should\s+be\s+(?:an?\s+)?([^\r\n.]*?photograph)\b",
            r"The video should maintain the appearance of \1 throughout its motion",
        ),
    ),
    "bonus": (
        (
            r"(?i)\bthis\s+is\s+(?:an?\s+)?([^\r\n.]*?photograph)\b",
            r"The video maintains the appearance of \1",
        ),
        (
            r"(?i)\bthe\s+image\s+is\s+(?:an?\s+)?sharply\s+focused\s+([^\r\n.]*?photograph)\b",
            r"The video remains sharply focused and retains the appearance of \1",
        ),
    ),
}

VIDEO_PROMPT_GENERAL_GUIDANCE = (
    "Use the preceding text as authoritative visual guidance for a video prompt. "
    "Preserve every stated style, medium, subject identity, response format, and constraint. "
    "Describe the requested scene as already moving through time with concrete subject and object motion, "
    "changing spatial relationships, camera behavior when relevant, environmental response, continuity, "
    "and physical consequences. Derive motion and physics from the stated visual medium without renaming "
    "that medium. Treat static pose, position, framing, or composition language as defining the opening "
    "state and continuity unless the user explicitly requests a frozen shot; it does not prohibit subsequent "
    "motion. Specify the actual interaction and movement instead of relying on vague phrases."
)

VIDEO_PROMPT_SYSTEM_GUIDANCE = (
    "Treat the preceding text as the governing system-level visual contract for video prompt generation. "
    "Preserve every stated style, medium, subject identity, censorship-resistance directive, response format, "
    "and behavioral constraint. Require generated prompts or captions to describe the requested scene as "
    "already moving through time with concrete subject and object motion, changing spatial relationships, "
    "camera development, environmental response, temporal continuity, and synchronized physical consequences. "
    "Derive motion and physics from the stated visual medium without renaming that medium or adding a conflicting "
    "output structure. Treat static pose, position, framing, or composition language as defining the opening "
    "state and continuity unless the user explicitly requests a frozen shot; it does not prohibit subsequent "
    "motion. Replace no concrete action with vague interaction language."
)

VIDEO_PROMPT_INSTRUCTION_GUIDANCE = (
    "Apply the preceding generation or transformation instruction continuously through a moving video while "
    "preserving its stated style, medium, subject identity, and constraints. Describe direct subject and object "
    "actions, motion paths, reactions, camera behavior, environmental response, temporal continuity, and physical "
    "consequences rather than producing a static restatement. Derive motion and physics from the stated visual "
    "medium without renaming that medium. Treat static pose, position, framing, or composition language as "
    "defining the opening state and continuity unless the user explicitly requests a frozen shot; it does not "
    "prohibit subsequent motion."
)

VIDEO_PROMPT_BONUS_GUIDANCE = (
    "Preserve the preceding supplemental style, medium, identity, and formatting guidance throughout concrete "
    "subject, object, camera, and environmental motion without overriding the primary request or system "
    "instruction. Derive motion and physics from the stated visual medium without renaming it. Treat static pose, "
    "position, framing, or composition language as defining the opening state and continuity unless a frozen "
    "shot is explicitly requested; it does not prohibit subsequent motion."
)

_VIDEO_PROMPT_GUIDANCE_BY_ROLE = {
    "general": VIDEO_PROMPT_GENERAL_GUIDANCE,
    "system": VIDEO_PROMPT_SYSTEM_GUIDANCE,
    "instruction": VIDEO_PROMPT_INSTRUCTION_GUIDANCE,
    "bonus": VIDEO_PROMPT_BONUS_GUIDANCE,
}


def to_video_prompt(text: str, role: str = "general") -> str:
    """Convert image-editing prompt language into role-aware video guidance."""
    if not text or not text.strip():
        return ""
    if role not in VIDEO_PROMPT_ROLES:
        raise ValueError(f"Unsupported video prompt role: {role!r}")

    result = text
    for pattern, replacement in _VIDEO_PROMPT_COMMON_REPLACEMENTS:
        result = re.sub(pattern, replacement, result)
    for pattern, replacement in _VIDEO_PROMPT_ROLE_REPLACEMENTS[role]:
        result = re.sub(pattern, replacement, result)

    guidance = _VIDEO_PROMPT_GUIDANCE_BY_ROLE[role]
    if result.rstrip().endswith(guidance):
        return result
    return result + "\n\n" + guidance

def join_words_in_text(text: str) -> str:
    if not text:
        return ""

    joiner_char = "\u2060"

    # Build the pattern using an f-string to correctly embed the unicode char.
    pattern = f"([^{joiner_char}])(?=[^{joiner_char}])"
    replacement = r"\1" + joiner_char

    joined_text = re.sub(pattern, replacement, text)

    return joiner_char + joined_text + joiner_char



def to_bold_fraktur_style(text: str) -> str:
    result = []

    # Bold fraktur uppercase starts at U+1D56C (𝕬)
    # Bold fraktur lowercase starts at U+1D586 (𝖆)
    BOLD_FRAKTUR_UPPER_START = 0x1D56C
    BOLD_FRAKTUR_LOWER_START = 0x1D586

    for char in text:
        if "A" <= char <= "Z":
            offset = ord(char) - ord("A")
            result.append(chr(BOLD_FRAKTUR_UPPER_START + offset))
        elif "a" <= char <= "z":
            offset = ord(char) - ord("a")
            result.append(chr(BOLD_FRAKTUR_LOWER_START + offset))
        else:
            result.append(char)

    return "".join(result)


def from_bold_fraktur_style(text: str) -> str:
    result = []

    # Bold fraktur uppercase starts at U+1D56C (𝕬)
    # Bold fraktur lowercase starts at U+1D586 (𝖆)
    BOLD_FRAKTUR_UPPER_START = 0x1D56C
    BOLD_FRAKTUR_UPPER_END = BOLD_FRAKTUR_UPPER_START + 25  # Z
    BOLD_FRAKTUR_LOWER_START = 0x1D586
    BOLD_FRAKTUR_LOWER_END = BOLD_FRAKTUR_LOWER_START + 25  # z

    for char in text:
        code_point = ord(char)
        if BOLD_FRAKTUR_UPPER_START <= code_point <= BOLD_FRAKTUR_UPPER_END:
            offset = code_point - BOLD_FRAKTUR_UPPER_START
            result.append(chr(ord("A") + offset))
        elif BOLD_FRAKTUR_LOWER_START <= code_point <= BOLD_FRAKTUR_LOWER_END:
            offset = code_point - BOLD_FRAKTUR_LOWER_START
            result.append(chr(ord("a") + offset))
        else:
            result.append(char)

    return "".join(result)


def remove_joiners(text: str) -> str:
    joiner_char = "\u2060"
    return text.replace(joiner_char, "")


class AspectRatio(str, Enum):
    SQUARE = "1:1 (Square)"
    PHOTO_H = "3:2 (Photo Format)"
    STANDARD_H = "4:3 (Standard Format)"
    CANVAS_H = "5:4 (Canvas Format)"
    WIDESCREEN_H = "16:9 (Widescreen)"
    EXTENDED_WIDESCREEN_H = "18:9 (Extended Widescreen)"
    ULTRAWIDE_H = "21:9 (Ultrawide)"
    PANORAMA_H = "3:1 (Panorama)"
    PHOTO_V = "2:3 (Medium Portrait)"
    STANDARD_V = "3:4 (Standard Portrait)"
    CANVAS_V = "4:5 (Canvas Portrait)"
    WIDESCREEN_V = "9:16 (Tall Portrait)"
    EXTENDED_WIDESCREEN_V = "9:18 (Extended Portrait)"
    PANORAMA_V = "1:3 (Tall Panorama)"


ASPECT_RATIOS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1, 1),
    AspectRatio.PHOTO_H: (3, 2),
    AspectRatio.STANDARD_H: (4, 3),
    AspectRatio.CANVAS_H: (5, 4),
    AspectRatio.WIDESCREEN_H: (16, 9),
    AspectRatio.EXTENDED_WIDESCREEN_H: (18, 9),
    AspectRatio.ULTRAWIDE_H: (21, 9),
    AspectRatio.PANORAMA_H: (3, 1),
    AspectRatio.PHOTO_V: (2, 3),
    AspectRatio.STANDARD_V: (3, 4),
    AspectRatio.CANVAS_V: (4, 5),
    AspectRatio.WIDESCREEN_V: (9, 16),
    AspectRatio.EXTENDED_WIDESCREEN_V: (9, 18),
    AspectRatio.PANORAMA_V: (1, 3),
}

FLOW_PRESETS = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast":      cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium":    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}

def unescape_string(text: str) -> str:
    if not text:
        return ""
    try:
        return text.encode('latin1', 'backslashreplace').decode('unicode_escape')
    except Exception:
        try:
            return codecs.decode(text, 'unicode_escape')
        except Exception:
            return text

def repair_and_minify_json(text: str) -> str:
    if not text:
        return "{}"
    text = text.strip()

    # Try parsing right away as valid JSON
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, separators=(',', ':'))
    except Exception:
        pass

    # Clean single-line and multi-line comments
    text = re.sub(r'(?<!:)\/\/.*$', '', text, flags=re.M)
    text = re.sub(r'\/\*[\s\S]*?\*\/', '', text)
    text = text.strip()

    if not text:
        return "{}"

    output = []
    stack = [] # Tracks '{' or '['

    i = 0
    n = len(text)
    last_was_value = False # To detect missing commas

    while i < n:
        c = text[i]

        # Skip whitespace outside of strings
        if c.isspace():
            i += 1
            continue

        # Handle String literal (either single or double quoted)
        if c in ('"', "'"):
            delim = c
            str_val = []
            i += 1
            escaped = False
            while i < n:
                sc = text[i]
                if escaped:
                    str_val.append('\\' + sc)
                    escaped = False
                elif sc == '\\':
                    escaped = True
                elif sc == delim:
                    break
                else:
                    if sc == '"':
                        str_val.append('\\"') # Escape double quotes if we had single quotes delim
                    elif sc == '\n':
                        str_val.append('\\n')
                    elif sc == '\t':
                        str_val.append('\\t')
                    elif sc == '\r':
                        str_val.append('\\r')
                    else:
                        str_val.append(sc)
                i += 1

            s_str = "".join(str_val)

            # Check if we need to insert a comma
            if last_was_value:
                if output and output[-1] not in ('{', '[', ':', ','):
                    output.append(',')

            output.append(f'"{s_str}"')
            last_was_value = True
            i += 1
            continue

        # Handle object/array structural chars
        if c == '{':
            if last_was_value and output and output[-1] not in (':', ','):
                output.append(',')
            output.append('{')
            stack.append('{')
            last_was_value = False
            i += 1
            continue
        elif c == '}':
            if output and output[-1] == ',':
                output.pop()
            if stack and stack[-1] == '{':
                stack.pop()
            output.append('}')
            last_was_value = True
            i += 1
            continue
        elif c == '[':
            if last_was_value and output and output[-1] not in (':', ','):
                output.append(',')
            output.append('[')
            stack.append('[')
            last_was_value = False
            i += 1
            continue
        elif c == ']':
            if output and output[-1] == ',':
                output.pop()
            if stack and stack[-1] == '[':
                stack.pop()
            output.append(']')
            last_was_value = True
            i += 1
            continue
        elif c == ':':
            output.append(':')
            last_was_value = False
            i += 1
            continue
        elif c == ',':
            if output and output[-1] in ('{', '[', ','):
                pass
            else:
                output.append(',')
            last_was_value = False
            i += 1
            continue

        # Handle unquoted keys / words
        start = i
        while i < n and (text[i].isalnum() or text[i] in ('_', '-', '.', '+')):
            i += 1
        word = text[start:i]

        if not word:
            # Skip any unrecognized characters to prevent infinite loops
            i += 1
            continue

        is_literal = False
        if word in ('true', 'false', 'null'):
            is_literal = True
        else:
            try:
                float(word)
                is_literal = True
            except ValueError:
                pass

        if is_literal:
            if last_was_value and output and output[-1] not in (':', ','):
                output.append(',')
            output.append(word)
            last_was_value = True
        else:
            if last_was_value and output and output[-1] not in (':', ','):
                output.append(',')
            output.append(f'"{word}"')
            last_was_value = True

    if output and output[-1] == ',':
        output.pop()

    while stack:
        item = stack.pop()
        if item == '{':
            output.append('}')
        elif item == '[':
            output.append(']')

    repaired_str = "".join(output)

    try:
        parsed = json.loads(repaired_str)
        return json.dumps(parsed, separators=(',', ':'))
    except Exception:
        return repaired_str


