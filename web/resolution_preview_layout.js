export const RESOLUTION_PREVIEW_HEIGHT = 26;
export const RESOLUTION_PREVIEW_SIDE_PADDING = 8;

export function resolutionPreviewMinimumSize(baseSize, textWidth = 0) {
  return [
    Math.max(Number(baseSize?.[0]) || 0, textWidth > 0 ? Math.ceil(textWidth) + 2 * RESOLUTION_PREVIEW_SIDE_PADDING : 0),
    (Number(baseSize?.[1]) || 0) + RESOLUTION_PREVIEW_HEIGHT,
  ];
}

export function clampResolutionPreviewSize(size, minimum, collapsed = false) {
  if (!size || collapsed) return size;
  size[0] = Math.max(size[0], minimum[0]);
  size[1] = Math.max(size[1], minimum[1]);
  return size;
}
