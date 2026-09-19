// Mirrors h3_video_length_from_seconds in parameter_helpers.py.
export function h3VideoLengthFromSeconds(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  const value = seconds * 24;
  const rounded = Math.round(value);
  const halfEven = Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
  const frames = Math.max(5, halfEven);
  return frames + ((((5 - (frames % 17)) % 17) + 17) % 17);
}

export function h3ReferenceFrameRange(startSeconds, durationSeconds, sourceSeconds = null, segmentCount = 0, segmentIndex = 0, prependedFrames = 0) {
  if (segmentCount !== 0) {
    if (!Number.isInteger(segmentCount) || segmentCount < 1
        || !Number.isInteger(segmentIndex) || segmentIndex < 0 || segmentIndex >= segmentCount
        || !Number.isFinite(sourceSeconds) || sourceSeconds <= 0) {
      return { start: null, end: null, length: null };
    }
    const value = sourceSeconds * 24;
    const rounded = Math.round(value);
    const total = Math.max(1, Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded);
    const targetF = total / segmentCount;
    const prependedT = Math.max(0, Number.isInteger(prependedFrames) ? prependedFrames : 0);
    const R = ((((5 - (prependedT % 17)) % 17) + 17) % 17);
    let k = Math.max(0, Math.round((targetF - R) / 17));
    if (R === 0 && k === 0) k = 1;
    while (((R > 0 && k > 0) || (R === 0 && k > 1)) && (segmentCount - 1) * (R + 17 * k) >= total) {
      k--;
    }
    const nonFinalLength = R + 17 * k;
    const start = segmentIndex * nonFinalLength;
    if (segmentIndex < segmentCount - 1) {
      const length = nonFinalLength;
      const stop = Math.min(total, start + length);
      return { start, end: stop - 1, length: length + prependedT, padding: length - (stop - start) };
    }
    const remaining = Math.max(1, total - start);
    const totalH3 = h3VideoLengthFromSeconds((remaining + prependedT) / 24);
    const length = totalH3 - prependedT;
    return { start, end: total - 1, length: totalH3, padding: length - remaining };
  }
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
