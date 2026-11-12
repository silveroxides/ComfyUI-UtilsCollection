"""Reusable model architectures owned by UtilsCollection."""

from .depth_anything import DA3Base, DA3Large, DA3MetricLarge, DA3MonoLarge, DA3Small, DepthAnything3Net
from .minimax_h3_latent_upscaler import MiniMaxH3LatentUpscaler2D

__all__ = ["DA3Base", "DA3Large", "DA3MetricLarge", "DA3MonoLarge", "DA3Small", "DepthAnything3Net", "MiniMaxH3LatentUpscaler2D"]
