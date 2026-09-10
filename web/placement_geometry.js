import { normalizeForegroundContent } from "./foreground_content.js";

export const DEFAULT_PLACEMENT = Object.freeze({
  scale: 0.9,
  center_x: 0.5,
  center_y: 0.5,
  flip_horizontal: false,
  flip_vertical: false,
});
export const LEGACY_DEFAULT_PLACEMENT = Object.freeze({ scale: 0.9, long_axis_shift: 0, short_axis_shift: 0 });
export const DEFAULT_WORKSPACE_PADDING = 0.5;
export const DEFAULT_FACE_CORNERS = Object.freeze([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
export const PAINT_LAYER_KEY = "__uc_paint__";

export function normalizePaintLayer(value) {
  const source = value && typeof value === "object" ? value : {};
  const asset = source.asset && typeof source.asset === "object"
    ? {
        filename: typeof source.asset.filename === "string" ? source.asset.filename : "",
        subfolder: typeof source.asset.subfolder === "string" ? source.asset.subfolder : "clipspace",
        type: source.asset.type === "input" ? "input" : "input",
      }
    : null;
  const width = Math.max(0, Math.round(finite(source.width, 0)));
  const height = Math.max(0, Math.round(finite(source.height, 0)));
  const revision = Math.max(0, Math.round(finite(source.revision, 0)));
  return {
    included: source.included !== false,
    asset_id: typeof source.asset_id === "string" ? source.asset_id : "",
    owner_node_id: source.owner_node_id == null ? "" : String(source.owner_node_id),
    revision,
    width,
    height,
    ...(asset?.filename ? { asset } : {}),
  };
}

const finite = (value, fallback) => Number.isFinite(Number(value)) ? Number(value) : fallback;
export const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
export const normalizeWorkspacePadding = (value) => clamp(finite(value, DEFAULT_WORKSPACE_PADDING), 0, 1);

export function stepPlacementValue(field, value, delta) {
  const minimum = field === "scale" ? 0.05 : -10;
  const fallback = field === "scale" ? DEFAULT_PLACEMENT.scale : 0.5;
  const adjusted = finite(value, fallback) + finite(delta, 0);
  return clamp(Math.round(adjusted * 1e10) / 1e10, minimum, 10);
}

export function normalizePlacement(value = {}, version = 2, face = false) {
  const scale = clamp(finite(value.scale, DEFAULT_PLACEMENT.scale), 0.05, 10);
  if (version === 1 && value.center_x === undefined && value.center_y === undefined) {
    return {
      scale,
      long_axis_shift: clamp(finite(value.long_axis_shift, 0), -1, 1),
      short_axis_shift: clamp(finite(value.short_axis_shift, 0), -1, 1),
      flip_horizontal: value.flip_horizontal === true,
      flip_vertical: value.flip_vertical === true,
      ...(value.locked === true ? { locked: true } : {}),
    };
  }
  const normalized = {
    scale,
    center_x: clamp(finite(value.center_x, 0.5), -10, 10),
    center_y: clamp(finite(value.center_y, 0.5), -10, 10),
    flip_horizontal: value.flip_horizontal === true,
    flip_vertical: value.flip_vertical === true,
  };
  if (version === 3) {
    normalized.rotation = normalizeRotation(value.rotation);
    normalized.corners = normalizeQuad(value.corners);
  }
  if (version === 3) {
    normalized.included = value.included !== false;
  }
  if (value.locked === true) normalized.locked = true;
  return normalized;
}

export const normalizeRotation = (value) => ((finite(value, 0) + 180) % 360 + 360) % 360 - 180;

export function rotationFromPointer(startRotation, center, startPointer, pointer) {
  const startAngle = Math.atan2(
    startPointer.y - center.y,
    startPointer.x - center.x,
  );
  const angle = Math.atan2(pointer.y - center.y, pointer.x - center.x);
  return normalizeRotation(startRotation + (angle - startAngle) * 180 / Math.PI);
}

export function quadArea(points) {
  return Math.abs(points.reduce((sum, point, index) => {
    const next = points[(index + 1) % 4];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0)) / 2;
}

export function isValidQuad(points) {
  if (!Array.isArray(points) || points.length !== 4 || points.some(
    (point) => !Array.isArray(point) || point.length !== 2 || point.some((v) => !Number.isFinite(Number(v))),
  )) return false;
  const cross = points.map((a, index) => {
    const b = points[(index + 1) % 4], c = points[(index + 2) % 4];
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
  });
  return quadArea(points) >= 1e-4 && (cross.every((v) => v > 1e-6) || cross.every((v) => v < -1e-6));
}

export function pointInConvexQuad(point, corners, epsilon = 1e-6) {
  if (!point || !Array.isArray(corners) || corners.length !== 4) return false;
  const cross = corners.map((corner, index) => {
    const next = corners[(index + 1) % corners.length];
    return (
      (next.x - corner.x) * (point.y - corner.y)
      - (next.y - corner.y) * (point.x - corner.x)
    );
  });
  return cross.every((value) => value >= -epsilon)
    || cross.every((value) => value <= epsilon);
}

export function frontmostLayerAtPoint(layers, point, quadFor, includedFor) {
  for (const key of [...layers].reverse()) {
    if (includedFor && !includedFor(key)) continue;
    if (pointInConvexQuad(point, quadFor(key))) return key;
  }
  return null;
}

export function normalizeQuad(value) {
  const points = Array.isArray(value) ? value.map((point) => [
    clamp(finite(point?.[0], 0), -1, 1), clamp(finite(point?.[1], 0), -1, 1),
  ]) : DEFAULT_FACE_CORNERS.map((point) => [...point]);
  return isValidQuad(points) ? points : DEFAULT_FACE_CORNERS.map((point) => [...point]);
}

export function dragQuadCorner(corners, index, x, y) {
  const candidate = normalizeQuad(corners).map((point) => [...point]);
  candidate[index] = [clamp(finite(x, candidate[index][0]), -1, 1), clamp(finite(y, candidate[index][1]), -1, 1)];
  return isValidQuad(candidate) ? candidate : normalizeQuad(corners);
}

export function deformQuadCentroidLocked(corners, point, deltaX, deltaY) {
  const original = normalizeQuad(corners);
  const weights = original.map((corner) => 1 / Math.max(1e-6, Math.hypot(corner[0] - point[0], corner[1] - point[1])));
  const total = weights.reduce((a, b) => a + b, 0);
  const movement = original.map((_, index) => [
    deltaX * weights[index] / total,
    deltaY * weights[index] / total,
  ]);
  const horizontalDirection = Math.sign(deltaX);
  const verticalDirection = Math.sign(deltaY);
  for (let index = 0; index < original.length; index++) {
    const [cornerX, cornerY] = original[index];
    const horizontalInfluence = horizontalDirection ? (1 + horizontalDirection * cornerX) / 2 : 0;
    const verticalInfluence = verticalDirection ? (1 + verticalDirection * cornerY) / 2 : 0;
    movement[index][1] -= cornerY * Math.abs(deltaX) * 0.8 * horizontalInfluence;
    movement[index][0] -= cornerX * Math.abs(deltaY) * 0.8 * verticalInfluence;
  }
  const average = movement.reduce(
    (sum, value) => [sum[0] + value[0] / 4, sum[1] + value[1] / 4],
    [0, 0],
  );
  const centered = movement.map(([x, y]) => [x - average[0], y - average[1]]);
  for (let axis = 0; axis < 2; axis++) {
    const lower = original.map((corner) => -1 - corner[axis]);
    const upper = original.map((corner) => 1 - corner[axis]);
    let low = -4, high = 4;
    for (let iteration = 0; iteration < 48; iteration++) {
      const lambda = (low + high) / 2;
      const sum = centered.reduce(
        (value, movementValue, index) => value + clamp(movementValue[axis] - lambda, lower[index], upper[index]),
        0,
      );
      if (sum > 0) low = lambda;
      else high = lambda;
    }
    const lambda = (low + high) / 2;
    for (let index = 0; index < centered.length; index++) {
      centered[index][axis] = clamp(centered[index][axis] - lambda, lower[index], upper[index]);
    }
  }
  const at = (amount) => original.map((corner, index) => [
    corner[0] + centered[index][0] * amount,
    corner[1] + centered[index][1] * amount,
  ]);
  let amount = 1;
  let candidate = at(amount);
  if (isValidQuad(candidate)) return candidate;
  for (let iteration = 0; iteration < 20; iteration++) {
    amount *= 0.5;
    candidate = at(amount);
    if (isValidQuad(candidate)) return candidate;
  }
  return original;
}

export function projectQuadPoint(corners, u, v) {
  const [topLeft, topRight, bottomRight, bottomLeft] = normalizeQuad(corners);
  const dx1 = topRight[0] - bottomRight[0];
  const dx2 = bottomLeft[0] - bottomRight[0];
  const dx3 = topLeft[0] - topRight[0] + bottomRight[0] - bottomLeft[0];
  const dy1 = topRight[1] - bottomRight[1];
  const dy2 = bottomLeft[1] - bottomRight[1];
  const dy3 = topLeft[1] - topRight[1] + bottomRight[1] - bottomLeft[1];
  const denominator = dx1 * dy2 - dx2 * dy1;
  const g = Math.abs(denominator) < 1e-12 ? 0 : (dx3 * dy2 - dx2 * dy3) / denominator;
  const h = Math.abs(denominator) < 1e-12 ? 0 : (dx1 * dy3 - dx3 * dy1) / denominator;
  const a = topRight[0] - topLeft[0] + g * topRight[0];
  const b = bottomLeft[0] - topLeft[0] + h * bottomLeft[0];
  const d = topRight[1] - topLeft[1] + g * topRight[1];
  const e = bottomLeft[1] - topLeft[1] + h * bottomLeft[1];
  const scale = g * u + h * v + 1;
  return [
    (a * u + b * v + topLeft[0]) / scale,
    (d * u + e * v + topLeft[1]) / scale,
  ];
}

export function parsePlacementData(value) {
  try {
    const data = typeof value === "string" ? JSON.parse(value || "{}") : value;
    const version = data?.version ?? 1;
    if (!data || typeof data !== "object" || ![1, 2, 3].includes(version)) throw new Error();
    const layers = {};
    for (const [key, placement] of Object.entries(data.layers || {})) {
      if (placement && typeof placement === "object") layers[key] = normalizePlacement(placement, version, key.includes("_face_"));
    }
    const layer_order = Array.isArray(data.layer_order)
      ? [...new Set(data.layer_order.filter((key) => typeof key === "string"))]
      : [];
    return {
      version,
      workspace_padding: normalizeWorkspacePadding(data.workspace_padding),
      layer_order,
      layers,
      ...(data.foreground_content ? { foreground_content: normalizeForegroundContent(data.foreground_content) } : {}),
      ...(data.paint_layer && typeof data.paint_layer === "object"
        ? { paint_layer: normalizePaintLayer(data.paint_layer) }
        : {}),
    };
  } catch {
    return {
      version: 2,
      workspace_padding: DEFAULT_WORKSPACE_PADDING,
      layer_order: [],
      layers: {},
    };
  }
}

export function serializePlacementData(data) {
  const layers = {};
  for (const key of Object.keys(data.layers || {}).sort(layerKeyCompare)) {
    layers[key] = normalizePlacement(data.layers[key], data.version ?? 2, key.includes("_face_"));
  }
  const paintLayer = normalizePaintLayer(data.paint_layer);
  return JSON.stringify({
    version: data.version ?? 2,
    workspace_padding: normalizeWorkspacePadding(data.workspace_padding),
    layer_order: Array.isArray(data.layer_order)
      ? [...new Set(data.layer_order.filter((key) => typeof key === "string"))]
      : [],
    layers,
    ...(data.foreground_content && Object.keys(data.foreground_content).length
      ? { foreground_content: normalizeForegroundContent(data.foreground_content) } : {}),
    ...(paintLayer.asset || paintLayer.asset_id ? { paint_layer: paintLayer } : {}),
  });
}

export function layerKeyCompare(a, b) {
  const ai = Number((a.match(/\d+/) || [0])[0]);
  const bi = Number((b.match(/\d+/) || [0])[0]);
  return ai - bi || a.localeCompare(b);
}

export function sizeFromScale(backgroundWidth, backgroundHeight, aspect, scale) {
  const longest = Math.max(0, Number(scale)) * Math.min(backgroundWidth, backgroundHeight);
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  return ratio >= 1
    ? { width: longest, height: longest / ratio }
    : { width: longest * ratio, height: longest };
}

function physicalShifts(backgroundWidth, backgroundHeight, placement) {
  if (backgroundWidth >= backgroundHeight) {
    return { x: placement.long_axis_shift, y: placement.short_axis_shift };
  }
  return { x: placement.short_axis_shift, y: placement.long_axis_shift };
}

export function placementToRect(backgroundWidth, backgroundHeight, aspect, value) {
  const legacy = value?.center_x === undefined && value?.center_y === undefined;
  const placement = normalizePlacement(value, legacy ? 1 : 2);
  const size = sizeFromScale(backgroundWidth, backgroundHeight, aspect, placement.scale);
  if (!legacy) {
    return {
      x: placement.center_x * backgroundWidth - size.width / 2,
      y: placement.center_y * backgroundHeight - size.height / 2,
      width: size.width,
      height: size.height,
    };
  }
  const shifts = physicalShifts(backgroundWidth, backgroundHeight, placement);
  const travelX = backgroundWidth - size.width;
  const travelY = backgroundHeight - size.height;
  return {
    x: ((shifts.x + 1) / 2) * travelX,
    y: ((shifts.y + 1) / 2) * travelY,
    width: size.width,
    height: size.height,
  };
}

export function rectToPlacement(backgroundWidth, backgroundHeight, rect, prior = {}) {
  const longest = Math.max(rect.width, rect.height);
  const scale = longest / Math.min(backgroundWidth, backgroundHeight);
  const placement = {
    scale,
    center_x: (rect.x + rect.width / 2) / backgroundWidth,
    center_y: (rect.y + rect.height / 2) / backgroundHeight,
    flip_horizontal: prior.flip_horizontal === true,
    flip_vertical: prior.flip_vertical === true,
  };
  for (const field of ["rotation", "corners", "included", "locked"]) {
    if (prior[field] !== undefined) placement[field] = prior[field];
  }
  return placement;
}

export function pythonRound(value) {
  const numeric = finite(value, 0);
  const lower = Math.floor(numeric);
  const fraction = numeric - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

export function backendPlacedSize(backgroundWidth, backgroundHeight, sourceWidth, sourceHeight, scale) {
  const width = Math.max(1, pythonRound(sourceWidth));
  const height = Math.max(1, pythonRound(sourceHeight));
  const targetLongest = Math.max(1, pythonRound(
    Math.min(backgroundWidth, backgroundHeight) * Math.max(0, finite(scale, DEFAULT_PLACEMENT.scale)),
  ));
  const factor = targetLongest / Math.max(width, height);
  return {
    width: Math.max(1, pythonRound(width * factor)),
    height: Math.max(1, pythonRound(height * factor)),
  };
}

const identityCorners = (corners) => normalizeQuad(corners).every(
  (point, index) => point[0] === DEFAULT_FACE_CORNERS[index][0]
    && point[1] === DEFAULT_FACE_CORNERS[index][1],
);

export function transformedPixelGeometry(width, height, corners, rotation = 0) {
  const normalizedCorners = normalizeQuad(corners);
  const normalizedRotation = normalizeRotation(rotation);
  if (normalizedRotation === 0 && identityCorners(normalizedCorners)) {
    return {
      identity: true,
      width,
      height,
      minimum: {
        x: -Math.max(width - 1, 1) / 2,
        y: -Math.max(height - 1, 1) / 2,
      },
      points: [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
    };
  }
  const halfWidth = Math.max(width - 1, 1) / 2;
  const halfHeight = Math.max(height - 1, 1) / 2;
  const angle = normalizedRotation * Math.PI / 180;
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  const rotated = normalizedCorners.map(([x, y]) => {
    const px = x * halfWidth, py = y * halfHeight;
    return [px * cosine - py * sine, px * sine + py * cosine];
  });
  const minimumX = Math.min(...rotated.map((point) => point[0]));
  const minimumY = Math.min(...rotated.map((point) => point[1]));
  const maximumX = Math.max(...rotated.map((point) => point[0]));
  const maximumY = Math.max(...rotated.map((point) => point[1]));
  return {
    identity: false,
    width: Math.max(1, Math.ceil(maximumX - minimumX - 1e-6) + 1),
    height: Math.max(1, Math.ceil(maximumY - minimumY - 1e-6) + 1),
    minimum: { x: minimumX, y: minimumY },
    points: rotated.map(([x, y]) => [x - minimumX, y - minimumY]),
  };
}

export function backendPlacementOffset(
  backgroundWidth, backgroundHeight, placedWidth, placedHeight, placement, workspacePadding = 0,
) {
  let x = pythonRound(finite(placement?.center_x, 0.5) * backgroundWidth - placedWidth / 2);
  let y = pythonRound(finite(placement?.center_y, 0.5) * backgroundHeight - placedHeight / 2);
  const padding = normalizeWorkspacePadding(workspacePadding);
  const paddingX = backgroundWidth * 0.25 * padding;
  const paddingY = backgroundHeight * 0.25 * padding;
  const xLimits = [-paddingX, backgroundWidth + paddingX - placedWidth, 0, backgroundWidth - placedWidth];
  const yLimits = [-paddingY, backgroundHeight + paddingY - placedHeight, 0, backgroundHeight - placedHeight];
  x = pythonRound(clamp(x, Math.min(...xLimits), Math.max(...xLimits)));
  y = pythonRound(clamp(y, Math.min(...yLimits), Math.max(...yLimits)));
  return { x, y };
}

export function resolveLayerGeometry({
  backgroundWidth,
  backgroundHeight,
  sourceWidth,
  sourceHeight,
  placement,
  workspacePadding = 0,
}) {
  const normalized = normalizePlacement(placement, 3);
  const source = backendPlacedSize(
    backgroundWidth, backgroundHeight, sourceWidth, sourceHeight, normalized.scale,
  );
  const transformed = transformedPixelGeometry(
    source.width, source.height, normalized.corners, normalized.rotation,
  );
  const offset = backendPlacementOffset(
    backgroundWidth, backgroundHeight, transformed.width, transformed.height,
    normalized, workspacePadding,
  );
  const left = Math.max(0, offset.x), top = Math.max(0, offset.y);
  const right = Math.min(backgroundWidth, offset.x + transformed.width);
  const bottom = Math.min(backgroundHeight, offset.y + transformed.height);
  const visible = right > left && bottom > top ? {
    destination: { x: left, y: top, width: right - left, height: bottom - top },
    source: { x: left - offset.x, y: top - offset.y, width: right - left, height: bottom - top },
  } : null;
  return {
    source,
    transformed,
    offset,
    frame: { x: offset.x, y: offset.y, width: transformed.width, height: transformed.height },
    points: transformed.points.map(([x, y]) => ({ x: offset.x + x, y: offset.y + y })),
    visible,
  };
}

export function resolvedWorkspaceBounds(backgroundWidth, backgroundHeight, workspacePadding, frames = []) {
  const padding = normalizeWorkspacePadding(workspacePadding);
  const paddingX = backgroundWidth * 0.25 * padding;
  const paddingY = backgroundHeight * 0.25 * padding;
  const bounds = {
    left: -paddingX,
    top: -paddingY,
    right: backgroundWidth + paddingX,
    bottom: backgroundHeight + paddingY,
  };
  for (const frame of frames) {
    bounds.left = Math.min(bounds.left, frame.x);
    bounds.top = Math.min(bounds.top, frame.y);
    bounds.right = Math.max(bounds.right, frame.x + frame.width);
    bounds.bottom = Math.max(bounds.bottom, frame.y + frame.height);
  }
  return bounds;
}

export function moveRect(backgroundWidth, backgroundHeight, rect, deltaX, deltaY, workspacePadding = 0) {
  const padding = normalizeWorkspacePadding(workspacePadding);
  const paddingX = backgroundWidth * 0.25 * padding;
  const paddingY = backgroundHeight * 0.25 * padding;
  const travelX = backgroundWidth + paddingX * 2 - rect.width;
  const travelY = backgroundHeight + paddingY * 2 - rect.height;
  const xLimits = [-paddingX, -paddingX + travelX, 0, backgroundWidth - rect.width];
  const yLimits = [-paddingY, -paddingY + travelY, 0, backgroundHeight - rect.height];
  return {
    ...rect,
    x: clamp(rect.x + deltaX, Math.min(...xLimits), Math.max(...xLimits)) || 0,
    y: clamp(rect.y + deltaY, Math.min(...yLimits), Math.max(...yLimits)) || 0,
  };
}

export function moveResolvedPlacement(
  backgroundWidth, backgroundHeight, geometry, placement, deltaX, deltaY, workspacePadding = 0,
) {
  const frame = moveRect(
    backgroundWidth, backgroundHeight, geometry.frame, deltaX, deltaY, workspacePadding,
  );
  return {
    ...placement,
    center_x: (frame.x + frame.width / 2) / backgroundWidth,
    center_y: (frame.y + frame.height / 2) / backgroundHeight,
  };
}

export function drawRect(start, end, aspect) {
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const availableWidth = Math.abs(end.x - start.x);
  const availableHeight = Math.abs(end.y - start.y);
  const width = Math.min(availableWidth, availableHeight * ratio);
  const height = width / ratio;
  return {
    x: Math.min(start.x, end.x) + (availableWidth - width) / 2,
    y: Math.min(start.y, end.y) + (availableHeight - height) / 2,
    width,
    height,
  };
}

export function resizeRectFromDelta(rect, handle, deltaX, deltaY, aspect, minimumLongest, maximumLongest) {
  const ratio = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const widthPerLongest = ratio >= 1 ? 1 : ratio;
  const heightPerLongest = ratio >= 1 ? 1 / ratio : 1;
  const signX = handle.includes("w") ? -1 : 1;
  const signY = handle.includes("n") ? -1 : 1;
  const projected = (
    deltaX * signX * widthPerLongest + deltaY * signY * heightPerLongest
  ) / (widthPerLongest ** 2 + heightPerLongest ** 2);
  const startLongest = Math.max(rect.width, rect.height);
  const longest = clamp(startLongest + projected, minimumLongest, maximumLongest);
  const width = longest * widthPerLongest;
  const height = longest * heightPerLongest;
  const fixedX = handle.includes("w") ? rect.x + rect.width : rect.x;
  const fixedY = handle.includes("n") ? rect.y + rect.height : rect.y;
  return {
    x: handle.includes("w") ? fixedX - width : fixedX,
    y: handle.includes("n") ? fixedY - height : fixedY,
    width,
    height,
  };
}
