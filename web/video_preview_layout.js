const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;

export const EMPTY_PREVIEW_ASPECT = 16 / 9;
export const MAX_PREVIEW_HEIGHT = 200;
export const MIN_PREVIEW_HEIGHT = 80;

export function videoPreviewHeight(width, videoWidth = 0, videoHeight = 0, previewMode = "none") {
  if (previewMode === "none") return 0;
  const previewWidth = Math.max(0, finite(width));
  if (previewWidth <= 0) return 0;

  const aspect = finite(videoWidth) > 0 && finite(videoHeight) > 0
    ? finite(videoWidth) / finite(videoHeight)
    : EMPTY_PREVIEW_ASPECT;

  const rawHeight = Math.ceil(previewWidth / aspect);
  const ceiling = Math.ceil(Math.min(previewWidth / EMPTY_PREVIEW_ASPECT, MAX_PREVIEW_HEIGHT));
  return Math.max(MIN_PREVIEW_HEIGHT, Math.min(rawHeight, ceiling));
}

export function clampVideoPreviewSize(size, minimum, collapsed = false) {
  if (!size || collapsed) return size;
  size[0] = Math.max(size[0], minimum[0]);
  size[1] = Math.max(size[1], minimum[1]);
  return size;
}

export function growNodeSize(current = [0, 0], required = [0, 0]) {
  return [
    Math.max(finite(current[0]), finite(required[0])),
    Math.max(finite(current[1]), finite(required[1])),
  ];
}
