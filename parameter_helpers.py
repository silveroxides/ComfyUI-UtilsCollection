import math


VIDEO_MIDDLE_BAND_MIN_MEGAPIXELS = 0.3
VIDEO_MIDDLE_BAND_MAX_MEGAPIXELS = 0.8
VIDEO_MIDDLE_BAND_RESOLUTIONS = {
    (7, 3): ((896, 384), (1120, 480), (1280, 544)),
    (16, 9): ((768, 416), (864, 480), (1024, 576), (1152, 672)),
    (16, 10): ((768, 480), (1024, 640)),
    (4, 3): ((640, 480), (768, 576), (864, 672), (1024, 768)),
}


def h3_video_length_from_seconds(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Duration must be zero or a positive number of seconds.")
    frames = max(5, round(seconds * 24))
    return frames + (5 - frames % 17) % 17


def _middle_band_resolution(
    ratio_width: int,
    ratio_height: int,
    megapixels: float,
    multiple: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int] | None:
    if not VIDEO_MIDDLE_BAND_MIN_MEGAPIXELS <= megapixels <= VIDEO_MIDDLE_BAND_MAX_MEGAPIXELS:
        return None

    ratio_divisor = math.gcd(ratio_width, ratio_height)
    ratio = (ratio_width // ratio_divisor, ratio_height // ratio_divisor)
    landscape = ratio[0] >= ratio[1]
    landscape_ratio = ratio if landscape else (ratio[1], ratio[0])
    candidates = VIDEO_MIDDLE_BAND_RESOLUTIONS.get(landscape_ratio, ())
    if not landscape:
        candidates = tuple((height, width) for width, height in candidates)
    candidates = tuple(
        (width, height)
        for width, height in candidates
        if width % multiple == 0
        and height % multiple == 0
        and minimum <= width <= maximum
        and minimum <= height <= maximum
    )
    if not candidates:
        return None

    target_pixels = megapixels * 1024 * 1024
    return min(candidates, key=lambda dimensions: abs(dimensions[0] * dimensions[1] - target_pixels))


def select_video_resolution(
    ratio_width: int,
    ratio_height: int,
    megapixels: float,
    multiple: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    """Select a multiple-aligned video resolution near a nominal MP target.

    The dominant axis remains an exact multiple of the simplified aspect-ratio
    side. Its companion axis is the nearest required multiple for that axis.
    """
    if ratio_width <= 0 or ratio_height <= 0:
        raise ValueError("Aspect ratio dimensions must be positive.")
    if megapixels <= 0:
        raise ValueError("Megapixels must be positive.")
    if multiple <= 0:
        raise ValueError("Dimension multiple must be positive.")
    if minimum > maximum:
        raise ValueError("Minimum resolution cannot exceed maximum resolution.")

    middle_band = _middle_band_resolution(
        ratio_width, ratio_height, megapixels, multiple, minimum, maximum
    )
    if middle_band is not None:
        return middle_band

    target_pixels = megapixels * 1024 * 1024
    maximum_aligned = maximum // multiple * multiple
    ratio_divisor = math.gcd(ratio_width, ratio_height)
    ratio_width //= ratio_divisor
    ratio_height //= ratio_divisor
    landscape = ratio_width >= ratio_height
    anchor_ratio = ratio_width if landscape else ratio_height
    companion_ratio = ratio_height if landscape else ratio_width
    anchor_step = math.lcm(multiple, anchor_ratio)
    minimum_anchor = math.ceil(minimum / anchor_step) * anchor_step
    candidates: set[tuple[int, int]] = set()

    for anchor in range(minimum_anchor, maximum_aligned + 1, anchor_step):
        ideal_companion = anchor * companion_ratio / anchor_ratio
        lower_companion = math.floor(ideal_companion / multiple) * multiple
        upper_companion = math.ceil(ideal_companion / multiple) * multiple
        for companion in {lower_companion, upper_companion}:
            if not minimum <= companion <= maximum_aligned:
                continue
            dimensions = (anchor, companion) if landscape else (companion, anchor)
            candidates.add(dimensions)

    if not candidates:
        raise ValueError(
            "No multiple-aligned resolution fits the selected aspect ratio."
        )

    target_ratio = ratio_width / ratio_height

    def candidate_key(dimensions: tuple[int, int]) -> tuple[float, float, int, int]:
        width, height = dimensions
        megapixel_error = abs(width * height - target_pixels) / target_pixels
        ratio_error = abs((width / height) / target_ratio - 1.0)
        return megapixel_error, ratio_error, width, height

    width, height = min(candidates, key=candidate_key)
    return width, height
