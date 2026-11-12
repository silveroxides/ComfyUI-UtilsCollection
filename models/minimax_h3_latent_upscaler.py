"""MiniMax H3 2D+temporal neural latent upscaler architecture."""

from collections.abc import Sequence

import comfy.ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD

FRAME_SLAB = 16
DEFAULT_CHUNK_FRAMES = 32


def _norm2d_silu(x: torch.Tensor, norm: nn.Module, scale: torch.Tensor | None = None, shift: torch.Tensor | None = None) -> torch.Tensor:
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(norm, x, offloadable=True)
    try:
        w, b = weight, bias
        if scale is not None and shift is not None:
            one_plus_scale = 1.0 + scale
            w = weight * one_plus_scale if weight is not None else None
            b = torch.addcmul(shift, bias, one_plus_scale) if bias is not None else shift
        return F.silu(F.group_norm(x, norm.num_groups, w, b, norm.eps))
    finally:
        comfy.ops.uncast_bias_weight(norm, weight, bias, offload_stream)


def _conv2d_cl(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(conv, x, offloadable=True)
    try:
        if conv.padding_mode != "zeros":
            padded = F.pad(x, conv._reversed_padding_repeated_twice, mode=conv.padding_mode)
            return F.conv2d(padded, weight.contiguous(memory_format=torch.channels_last), bias, conv.stride, (0, 0), conv.dilation, conv.groups)
        return F.conv2d(x, weight.contiguous(memory_format=torch.channels_last), bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    finally:
        comfy.ops.uncast_bias_weight(conv, weight, bias, offload_stream)


class FourierScaleTierEmbedding2D(nn.Module):
    """Fourier scale (scale_h, scale_w) and optional sharpness-tier conditioning."""

    def __init__(self, embed_dim: int = 128, num_frequencies: int = 16, tier_dim: int = 0, dtype=None, device=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.tier_dim = int(tier_dim)
        in_dim = 4 * self.num_frequencies + self.tier_dim
        self.linear = nn.Sequential(operations.Linear(in_dim, embed_dim, dtype=dtype, device=device), nn.SiLU(), operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device))

    def forward(self, scale_h: float, scale_w: float, tier: int | float | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(self.num_frequencies, device=device, dtype=torch.float32)).unsqueeze(0)
        sh = float(scale_h) * freqs
        sw = float(scale_w) * freqs
        parts = [torch.sin(sh), torch.cos(sh), torch.sin(sw), torch.cos(sw)]
        if self.tier_dim > 0:
            tier_freqs = (2.0 ** torch.arange(self.tier_dim // 2, device=device, dtype=torch.float32)).unsqueeze(0)
            t_val = 1.0 if tier is None else float(tier)
            tt = (t_val - 1.0) * tier_freqs
            parts.extend((torch.sin(tt), torch.cos(tt)))
        emb = torch.cat(parts, dim=1).to(dtype=dtype)
        return self.linear(emb)


class ResBlockEmb2D(nn.Module):
    """Scale-conditioned 2D residual block preserving checkpoint module indices."""

    def __init__(self, channels: int, emb_channels: int, out_channels: int | None = None, dtype=None, device=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        out_ch = out_channels or channels
        self.channels = channels
        self.out_channels = out_ch
        groups_in = 32 if channels >= 32 else channels
        groups_out = 32 if out_ch >= 32 else out_ch
        self.in_layers = nn.Sequential(operations.GroupNorm(groups_in, channels, dtype=dtype, device=device), nn.Identity(), operations.Conv2d(channels, out_ch, 3, padding=1, dtype=dtype, device=device))
        self.emb_layers = nn.Sequential(nn.SiLU(), operations.Linear(emb_channels, 2 * out_ch, dtype=dtype, device=device))
        self.out_norm = operations.GroupNorm(groups_out, out_ch, dtype=dtype, device=device)
        self.out_layers = nn.Sequential(nn.Identity(), nn.Identity(), operations.Conv2d(out_ch, out_ch, 3, padding=1, dtype=dtype, device=device))
        self.skip = operations.Conv2d(channels, out_ch, 1, dtype=dtype, device=device) if out_ch != channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.emb_layers(emb)[0].chunk(2)
        if isinstance(self.skip, nn.Identity):
            for f0 in range(0, x.shape[0], FRAME_SLAB):
                f1 = min(f0 + FRAME_SLAB, x.shape[0])
                slab = x[f0:f1]
                h = _conv2d_cl(self.in_layers[2], _norm2d_silu(slab, self.in_layers[0]))
                delta = _conv2d_cl(self.out_layers[2], _norm2d_silu(h, self.out_norm, scale, shift))
                slab.add_(delta)
            return x

        out = torch.empty((x.shape[0], self.out_channels, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        for f0 in range(0, x.shape[0], FRAME_SLAB):
            f1 = min(f0 + FRAME_SLAB, x.shape[0])
            slab = x[f0:f1]
            h = _conv2d_cl(self.in_layers[2], _norm2d_silu(slab, self.in_layers[0]))
            delta = _conv2d_cl(self.out_layers[2], _norm2d_silu(h, self.out_norm, scale, shift))
            out[f0:f1] = _conv2d_cl(self.skip, slab).add_(delta)
        return out


class PSUpsample2D(nn.Module):
    """Exclusive learned PixelShuffle(k) branch with replicate padding."""

    def __init__(self, channels: int, k: int = 2, dtype=None, device=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.k = int(k)
        self.conv = operations.Conv2d(channels, self.k * self.k * channels, 3, padding=1, padding_mode="replicate", dtype=dtype, device=device)
        self.ps = nn.PixelShuffle(self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], x.shape[1], x.shape[2] * self.k, x.shape[3] * self.k), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        for f0 in range(0, x.shape[0], FRAME_SLAB):
            f1 = min(f0 + FRAME_SLAB, x.shape[0])
            out[f0:f1] = self.ps(_conv2d_cl(self.conv, x[f0:f1]))
        return out


class PSUpsampleBank2D(nn.Module):
    """Bank of learned PixelShuffle branches indexed by integer scale k."""

    def __init__(self, channels: int, ks: Sequence[int], dtype=None, device=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.ks = sorted({int(k) for k in ks})
        self.branches = nn.ModuleDict({str(k): PSUpsample2D(channels, k, dtype=dtype, device=device, operations=operations) for k in self.ks})

    def select_k(self, scale: float) -> int:
        for k in self.ks:
            if abs(k - scale) <= 1e-3:
                return k
        for k in self.ks:
            if k > scale:
                return k
        return self.ks[-1]

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return self.branches[str(self.select_k(scale))](x)


class TemporalConv2D(nn.Module):
    """Per-frame GroupNorm + 1D temporal depthwise Conv3d + pointwise 1x1 Conv3d."""

    def __init__(self, channels: int, kernel_size: int = 5, dtype=None, device=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        groups = 32 if channels >= 32 else channels
        self.norm = operations.GroupNorm(groups, channels, dtype=dtype, device=device)
        self.dwconv = operations.Conv3d(channels, channels, (kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0), groups=channels, dtype=dtype, device=device)
        self.pwconv = operations.Conv3d(channels, channels, 1, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, batch_size: int, frames: int) -> torch.Tensor:
        _, channels, height, width = x.shape
        x_bt = x.view(batch_size, frames, channels, height, width)
        dw_weight, dw_bias, dw_stream = comfy.ops.cast_bias_weight(self.dwconv, x, offloadable=True)
        try:
            pw_weight, pw_bias, pw_stream = comfy.ops.cast_bias_weight(self.pwconv, x, offloadable=True)
            try:
                k = dw_weight.shape[2]
                pad = k // 2
                pw_w2d = pw_weight[:, :, 0].contiguous(memory_format=torch.channels_last)
                prev_tail: torch.Tensor | None = None
                prev_tail_start = 0

                for t0 in range(0, frames, FRAME_SLAB):
                    t1 = min(t0 + FRAME_SLAB, frames)
                    a = max(0, t0 - pad)
                    b = min(frames, t1 + pad)

                    if prev_tail is not None and a < t0:
                        head = prev_tail[:, a - prev_tail_start :]
                        window_in = torch.cat((head, x_bt[:, t0:b]), dim=1)
                    else:
                        window_in = x_bt[:, a:b]

                    next_need = max(t0, t1 - pad)
                    next_tail = x_bt[:, next_need:t1].clone() if t1 < frames and pad > 0 else None

                    h = _norm2d_silu(window_in.reshape(-1, channels, height, width), self.norm).view(batch_size, b - a, channels, height, width)
                    n_slab = t1 - t0
                    acc = torch.empty((batch_size, n_slab, channels, height, width), device=x.device, dtype=x.dtype)
                    if dw_bias is not None:
                        acc.copy_(dw_bias.view(1, 1, channels, 1, 1))
                    else:
                        acc.zero_()

                    for i in range(k):
                        s = i - pad
                        dst_lo = max(0, -s - t0)
                        dst_hi = min(n_slab, frames - s - t0)
                        if dst_hi > dst_lo:
                            src_lo = (t0 + dst_lo + s) - a
                            src_hi = src_lo + (dst_hi - dst_lo)
                            acc[:, dst_lo:dst_hi].addcmul_(h[:, src_lo:src_hi], dw_weight[:, 0, i, 0, 0].view(1, 1, channels, 1, 1))

                    acc_2d = acc.reshape(batch_size * n_slab, channels, height, width).contiguous(memory_format=torch.channels_last)
                    pw_out = F.conv2d(acc_2d, pw_w2d, pw_bias)
                    x_bt[:, t0:t1].add_(pw_out.view(batch_size, n_slab, channels, height, width))

                    prev_tail = next_tail
                    prev_tail_start = next_need
            finally:
                comfy.ops.uncast_bias_weight(self.pwconv, pw_weight, pw_bias, pw_stream)
        finally:
            comfy.ops.uncast_bias_weight(self.dwconv, dw_weight, dw_bias, dw_stream)
        return x


class LatentResizer2D(nn.Module):
    """Inner 2D resizer matching checkpoint `resizer.*` parameter hierarchy."""

    def __init__(self, in_channels: int = 24, channels: int = 640, in_blocks: int = 12, out_blocks: int = 12, embed_dim: int = 128, dual_embed_freq: int = 16, tier_dim: int = 32, learned_upsample_ks: Sequence[int] = (), dtype=None, device=None, operations=comfy.ops.disable_weight_init):
        super().__init__()
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.tier_dim = int(tier_dim)
        ops = {"dtype": dtype, "device": device, "operations": operations}
        self.embed = FourierScaleTierEmbedding2D(embed_dim=embed_dim, num_frequencies=dual_embed_freq, tier_dim=self.tier_dim, **ops)
        self.conv_in = operations.Conv2d(in_channels, channels, 3, padding=1, dtype=dtype, device=device)
        self.learned_up = PSUpsampleBank2D(channels, learned_upsample_ks, **ops) if learned_upsample_ks else None
        self.in_blocks = nn.ModuleList([ResBlockEmb2D(channels, embed_dim, **ops) for _ in range(in_blocks)])
        self.out_blocks = nn.ModuleList([ResBlockEmb2D(channels, embed_dim, **ops) for _ in range(out_blocks)])
        groups = 32 if channels >= 32 else channels
        self.norm_out = operations.GroupNorm(groups, channels, dtype=dtype, device=device)
        self.conv_out = operations.Conv2d(channels, in_channels, 3, padding=1, dtype=dtype, device=device)

    def upsample_mid(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if self.learned_up is not None:
            s = min(target_hw[0] / x.shape[-2], target_hw[1] / x.shape[-1])
            k = self.learned_up.select_k(s)
            if abs(k - s) <= 1e-3 and x.shape[-2] * k >= target_hw[0] and x.shape[-1] * k >= target_hw[1]:
                y = self.learned_up(x, s)
                if y.shape[-2] != target_hw[0] or y.shape[-1] != target_hw[1]:
                    y = y[..., : target_hw[0], : target_hw[1]].contiguous(memory_format=torch.channels_last)
                return y
        out = torch.empty((x.shape[0], x.shape[1], target_hw[0], target_hw[1]), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        for f0 in range(0, x.shape[0], FRAME_SLAB):
            f1 = min(f0 + FRAME_SLAB, x.shape[0])
            out[f0:f1] = F.interpolate(x[f0:f1], size=target_hw, mode="bilinear", align_corners=False)
        return out


class MiniMaxH3LatentUpscaler2D(nn.Module):
    """2D-conv + interleaved temporal-conv upscaler for 24-channel MiniMax H3 latents."""

    def __init__(
        self,
        in_channels: int = 24,
        channels: int = 640,
        in_blocks: int = 12,
        out_blocks: int = 12,
        embed_dim: int = 128,
        dual_embed_freq: int = 16,
        tier_dim: int = 32,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
        learned_upsample_ks: Sequence[int] = (),
        dtype=None,
        device=None,
        operations=comfy.ops.disable_weight_init,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.temporal_every = int(temporal_every)
        self.temporal_kernel = int(temporal_kernel)
        ops = {"dtype": dtype, "device": device, "operations": operations}
        self.resizer = LatentResizer2D(in_channels=in_channels, channels=channels, in_blocks=in_blocks, out_blocks=out_blocks, embed_dim=embed_dim, dual_embed_freq=dual_embed_freq, tier_dim=tier_dim, learned_upsample_ks=learned_upsample_ks, **ops)
        self.temporal_blocks = nn.ModuleList()
        if self.temporal_every > 0:
            self.temporal_blocks.append(TemporalConv2D(channels, temporal_kernel, **ops))
            self.temporal_blocks.append(TemporalConv2D(channels, temporal_kernel, **ops))

    @property
    def supports_sharpness_tier(self) -> bool:
        return self.resizer.tier_dim > 0

    def get_dtype(self):
        return self.dtype

    def _temporal_receptive_radius(self) -> int:
        if not self.temporal_blocks or self.temporal_every <= 0:
            return 0
        radius = max(self.temporal_kernel // 2, 1)
        n_in = sum(1 for i in range(len(self.resizer.in_blocks)) if i % self.temporal_every == 0)
        n_out = sum(1 for i in range(len(self.resizer.out_blocks)) if i % self.temporal_every == 0)
        return radius * max(n_in + n_out, 1)

    def memory_required(self, shape: Sequence[int], size: Sequence[int], chunk_frames: int | None = DEFAULT_CHUNK_FRAMES) -> int:
        n, _, t, h, w = shape
        h_out, w_out = int(size[-2]), int(size[-1])
        overlap = self._temporal_receptive_radius()
        eff_t = t if not chunk_frames or chunk_frames <= 0 or t <= chunk_frames else min(t, int(chunk_frames) + 2 * overlap)
        max_area = max(h * w, h_out * w_out)
        itemsize = self.dtype.itemsize if self.dtype is not None else 2
        io_bytes = n * self.in_channels * t * (h * w + h_out * w_out) * itemsize
        act_bytes = n * self.channels * (eff_t + 3 * FRAME_SLAB) * max_area * itemsize
        return io_bytes + act_bytes

    def _forward_segment(self, x: torch.Tensor, target_hw: tuple[int, int], tier: int | None) -> torch.Tensor:
        b, c, t, h, w = x.shape
        scale_h = target_hw[0] / h
        scale_w = target_hw[1] / w
        emb = self.resizer.embed(scale_h, scale_w, tier, x.device, x.dtype)

        x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        feat = torch.empty((b * t, self.channels, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        for f0 in range(0, b * t, FRAME_SLAB):
            f1 = min(f0 + FRAME_SLAB, b * t)
            feat[f0:f1] = _conv2d_cl(self.resizer.conv_in, x_2d[f0:f1])

        for i, block in enumerate(self.resizer.in_blocks):
            feat = block(feat, emb)
            if self.temporal_every > 0 and i % self.temporal_every == 0 and len(self.temporal_blocks) > 0:
                feat = self.temporal_blocks[0](feat, b, t)

        feat = self.resizer.upsample_mid(feat, target_hw)

        for i, block in enumerate(self.resizer.out_blocks):
            feat = block(feat, emb)
            if self.temporal_every > 0 and i % self.temporal_every == 0 and len(self.temporal_blocks) > 1:
                feat = self.temporal_blocks[1](feat, b, t)

        out_2d = torch.empty((b * t, self.in_channels, target_hw[0], target_hw[1]), device=x.device, dtype=x.dtype)
        for f0 in range(0, b * t, FRAME_SLAB):
            f1 = min(f0 + FRAME_SLAB, b * t)
            out_2d[f0:f1] = _conv2d_cl(self.resizer.conv_out, _norm2d_silu(feat[f0:f1], self.resizer.norm_out))

        return out_2d.view(b, t, self.in_channels, target_hw[0], target_hw[1]).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: torch.Tensor, size: Sequence[int], *, sharpness: int | None = 1, chunk_frames: int | None = DEFAULT_CHUNK_FRAMES) -> torch.Tensor:
        """Upscale normalized H3 video latent [B, 24, T, H, W] to target (H', W') while preserving T."""
        target_hw = (int(size[-2]), int(size[-1]))
        if target_hw == (x.shape[-2], x.shape[-1]):
            return x

        mean = torch.tensor(LATENTS_MEAN, device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        x_norm = (x - mean) / std

        tier = int(sharpness) if self.supports_sharpness_tier and sharpness is not None else None
        b, c, t, _, _ = x_norm.shape
        overlap = self._temporal_receptive_radius()

        if not chunk_frames or chunk_frames <= 0 or t <= chunk_frames or overlap == 0:
            out = self._forward_segment(x_norm, target_hw, tier)
            return torch.addcmul(mean, out, std)

        chunk = max(1, int(chunk_frames))
        out_full = torch.empty((b, c, t, target_hw[0], target_hw[1]), device=x.device, dtype=x.dtype)
        for seg_start in range(0, t, chunk):
            seg_end = min(t, seg_start + chunk)
            lo = max(0, seg_start - overlap)
            hi = min(t, seg_end + overlap)
            seg_out = self._forward_segment(x_norm[:, :, lo:hi], target_hw, tier)
            s0 = seg_start - lo
            out_full[:, :, seg_start:seg_end] = seg_out[:, :, s0 : s0 + (seg_end - seg_start)]

        return torch.addcmul(mean, out_full, std)
