// Mirrors h3_video_length_from_seconds in parameter_helpers.py.
export function h3VideoLengthFromSeconds(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  const value = seconds * 24;
  const rounded = Math.round(value);
  const halfEven = Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
  const frames = Math.max(5, halfEven);
  return frames + ((((5 - (frames % 17)) % 17) + 17) % 17);
}

export function h3ReferenceFrameRange(startSeconds, durationSeconds, sourceSeconds = null) {
  const start = startSeconds === 0 ? 0 : h3VideoLengthFromSeconds(startSeconds);
  if (start === null) return { start: null, end: null, length: null };
  let duration = durationSeconds;
  if (sourceSeconds !== null) {
    const remaining = sourceSeconds - start / 24;
    if (remaining <= 0) return { start, end: null, length: null };
    duration = duration > 0 ? Math.min(duration, remaining) : duration === 0 ? remaining : null;
  }
  const length = duration > 0 ? h3VideoLengthFromSeconds(duration) : null;
  return { start, end: length === null ? null : start + length - 1, length };
}
