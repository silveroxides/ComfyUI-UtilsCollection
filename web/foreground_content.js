export const DEFAULT_TEXT = Object.freeze({
  value: "", x: 0.1, y: 0.1, size: 48, color: "#000000", opacity: 1,
  family: "monospace", bold: false, italic: false, align: "left",
});

const number = (value, fallback, min, max) => Number.isFinite(Number(value))
  ? Math.max(min, Math.min(max, Number(value))) : fallback;

export function normalizeText(value = {}) {
  return {
    value: typeof value.value === "string" ? value.value : "",
    x: number(value.x, DEFAULT_TEXT.x, 0, 1), y: number(value.y, DEFAULT_TEXT.y, 0, 1),
    size: number(value.size, DEFAULT_TEXT.size, 1, 4096),
    color: /^#[0-9a-f]{6}$/i.test(value.color || "") ? value.color : DEFAULT_TEXT.color,
    opacity: number(value.opacity, 1, 0, 1),
    family: ["sans-serif", "serif", "monospace"].includes(value.family) ? value.family : DEFAULT_TEXT.family,
    bold: value.bold === true, italic: value.italic === true,
    align: ["left", "center", "right"].includes(value.align) ? value.align : "left",
  };
}

export function normalizeForegroundContent(value = {}) {
  const result = {};
  for (const [key, item] of Object.entries(value || {}).sort(([a], [b]) => a.localeCompare(b))) {
    if (!item || typeof item !== "object" || !/foreground_\d+(?:_face_\d+)?$/.test(key)) continue;
    result[key] = {
      width: number(item.width, 1, 1, Number.MAX_SAFE_INTEGER),
      height: number(item.height, 1, 1, Number.MAX_SAFE_INTEGER),
      brush: { visible: item.brush?.visible !== false, ...(item.brush?.asset ? { asset: item.brush.asset } : {}) },
      object_erase: item.object_erase?.asset ? { asset: item.object_erase.asset } : {},
      text: {
        ...normalizeText(item.text), visible: item.text?.visible !== false,
        ...(item.text?.asset ? { asset: item.text.asset } : {}),
      },
    };
  }
  return result;
}

// Invert the same homography used by placement previews, not its bounding box.
function homography(points) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = points.map((p) => Array.isArray(p) ? p : [p.x, p.y]);
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  const determinant = dx1 * dy2 - dx2 * dy1;
  const g = Math.abs(determinant) < 1e-12 ? 0 : (dx3 * dy2 - dx2 * dy3) / determinant;
  const h = Math.abs(determinant) < 1e-12 ? 0 : (dx1 * dy3 - dx3 * dy1) / determinant;
  return { a: x1 - x0 + g * x1, b: x3 - x0 + h * x3, c: x0,
    d: y1 - y0 + g * y1, e: y3 - y0 + h * y3, f: y0, g, h };
}

export function foregroundLocalPoint(points, point, width, height, flipH = false, flipV = false) {
  const m = homography(points);
  const a = m.a - point.x * m.g, b = m.b - point.x * m.h;
  const d = m.d - point.y * m.g, e = m.e - point.y * m.h;
  const det = a * e - b * d;
  if (Math.abs(det) < 1e-12) return null;
  const u = ((point.x - m.c) * e - b * (point.y - m.f)) / det;
  const v = (a * (point.y - m.f) - (point.x - m.c) * d) / det;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  return { x: (flipH ? 1 - u : u) * width, y: (flipV ? 1 - v : v) * height };
}

export function foregroundDisplayPoint(points, point, width, height, flipH = false, flipV = false) {
  const m = homography(points);
  const u = flipH ? 1 - point.x / width : point.x / width;
  const v = flipV ? 1 - point.y / height : point.y / height;
  const divisor = m.g * u + m.h * v + 1;
  return [(m.a * u + m.b * v + m.c) / divisor, (m.d * u + m.e * v + m.f) / divisor];
}

export function foregroundBrushScale(width, height, placedWidth, placedHeight) {
  // One uniform brush-size scale; compensate only the canvas's aspect ratio.
  // Existing pixels retain their attachment, but new stamps are not stretched.
  const uniform = Math.max(placedWidth, placedHeight) / Math.max(width, height);
  return { x: width * uniform / placedWidth, y: height * uniform / placedHeight };
}

export function foregroundBrushQuad(geometry) {
  if (!geometry.transformed.identity) return geometry.points;
  const { x, y, width, height } = geometry.frame;
  // Unwarped drawImage uses pixel edges, rather than projective pixel centers.
  return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]];
}

export function renderForegroundText(canvas, text, width, height) {
  canvas.width = width; canvas.height = height;
  const context = canvas.getContext("2d");
  context.font = `${text.italic ? "italic " : ""}${text.bold ? "bold " : ""}${text.size}px ${text.family}`;
  context.fillStyle = text.color;
  context.globalAlpha = text.opacity;
  context.textBaseline = "top";
  context.textAlign = text.align;
  text.value.split(/\r?\n/).forEach((line, index) => {
    context.fillText(line, text.x * width, text.y * height + index * text.size * 1.2);
  });
  return canvas;
}
