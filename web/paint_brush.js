const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, Number(value)));

export const DEFAULT_BRUSH_SETTINGS = Object.freeze({
  shape: "circle",
  color: "#000000",
  size: 5,
  opacity: 1,
  hardness: 1,
  erasing: false,
});

export function paintLayerVisible(included, editing = false) {
  return editing || included !== false;
}

export function normalizeBrushSettings(value = {}) {
  return {
    shape: value.shape === "square" ? "square" : "circle",
    color: /^#[0-9a-f]{6}$/i.test(value.color || "") ? value.color.toLowerCase() : DEFAULT_BRUSH_SETTINGS.color,
    size: clamp(Number.isFinite(Number(value.size)) ? value.size : DEFAULT_BRUSH_SETTINGS.size, 1, 250),
    opacity: clamp(Number.isFinite(Number(value.opacity)) ? value.opacity : DEFAULT_BRUSH_SETTINGS.opacity, 0, 1),
    hardness: clamp(Number.isFinite(Number(value.hardness)) ? value.hardness : DEFAULT_BRUSH_SETTINGS.hardness, 0, 1),
    erasing: value.erasing === true,
  };
}

export function brushTextureKey(settings) {
  const value = normalizeBrushSettings(settings);
  return [value.shape, value.color, value.size, value.opacity, value.hardness].join(":");
}

export function interpolatedPoints(start, end, spacing, scale = { x: 1, y: 1 }) {
  const distance = Math.hypot((end.x - start.x) / scale.x, (end.y - start.y) / scale.y);
  if (!distance) return [{ ...end }];
  const count = Math.max(1, Math.ceil(distance / Math.max(1e-6, spacing)));
  return Array.from({ length: count }, (_, index) => {
    const amount = (index + 1) / count;
    return {
      x: start.x + (end.x - start.x) * amount,
      y: start.y + (end.y - start.y) * amount,
    };
  });
}

function unionRect(rect, x, y, radiusX, radiusY, width, height) {
  const left = Math.max(0, Math.floor(x - radiusX - 2));
  const top = Math.max(0, Math.floor(y - radiusY - 2));
  const right = Math.min(width, Math.ceil(x + radiusX + 2));
  const bottom = Math.min(height, Math.ceil(y + radiusY + 2));
  if (right <= left || bottom <= top) return rect;
  if (!rect) return { left, top, right, bottom };
  return {
    left: Math.min(rect.left, left),
    top: Math.min(rect.top, top),
    right: Math.max(rect.right, right),
    bottom: Math.max(rect.bottom, bottom),
  };
}

export function clipBrushSegment(start, end, width, height, radius) {
  const dx = end.x - start.x, dy = end.y - start.y;
  let near = 0, far = 1;
  for (const [p, q] of [
    [-dx, start.x + radius], [dx, width + radius - start.x],
    [-dy, start.y + radius], [dy, height + radius - start.y],
  ]) {
    if (p === 0) { if (q < 0) return null; else continue; }
    const t = q / p;
    if (p < 0) near = Math.max(near, t); else far = Math.min(far, t);
    if (near > far) return null;
  }
  return [{ x: start.x + dx * near, y: start.y + dy * near }, { x: start.x + dx * far, y: start.y + dy * far }];
}

export function imageDataAlphaBounds(imageData) {
  let left = imageData.width, top = imageData.height, right = 0, bottom = 0;
  for (let y = 0; y < imageData.height; y++) for (let x = 0; x < imageData.width; x++) {
    if (!imageData.data[(y * imageData.width + x) * 4 + 3]) continue;
    left = Math.min(left, x); top = Math.min(top, y);
    right = Math.max(right, x + 1); bottom = Math.max(bottom, y + 1);
  }
  return right > left && bottom > top ? { left, top, right, bottom } : null;
}

class PatchHistory {
  constructor(limit = 20) {
    this.limit = limit;
    this.entries = [];
    this.index = 0;
  }

  push(entry) {
    this.entries.splice(this.index);
    this.entries.push(entry);
    if (this.entries.length > this.limit) this.entries.shift();
    this.index = this.entries.length;
  }

  apply(context, direction) {
    if (direction < 0) {
      if (this.index <= 0) return false;
      const entry = this.entries[--this.index];
      context.putImageData(entry.before, entry.left, entry.top);
      return true;
    }
    if (this.index >= this.entries.length) return false;
    const entry = this.entries[this.index++];
    context.putImageData(entry.after, entry.left, entry.top);
    return true;
  }

  clear() {
    this.entries = [];
    this.index = 0;
  }

  get canUndo() { return this.index > 0; }
  get canRedo() { return this.index < this.entries.length; }
}

export class PaintLayerCanvas {
  constructor(createCanvas = () => document.createElement("canvas")) {
    this.createCanvas = createCanvas;
    this.canvas = createCanvas();
    this.strokeCanvas = createCanvas();
    this.canvas.width = this.canvas.height = 0;
    this.strokeCanvas.width = this.strokeCanvas.height = 0;
    this.context = this.canvas.getContext("2d", { willReadFrequently: true });
    this.strokeContext = this.strokeCanvas.getContext("2d", { willReadFrequently: true });
    this.history = new PatchHistory(20);
    this.textureCache = new Map();
    this.stroke = null;
  }

  resize(width, height) {
    width = Math.max(1, Math.round(width));
    height = Math.max(1, Math.round(height));
    if (this.canvas.width === width && this.canvas.height === height) return false;
    const prior = this.createCanvas();
    prior.width = this.canvas.width;
    prior.height = this.canvas.height;
    if (this.canvas.width && this.canvas.height) prior.getContext("2d").drawImage(this.canvas, 0, 0);
    this.canvas.width = this.strokeCanvas.width = width;
    this.canvas.height = this.strokeCanvas.height = height;
    this.context = this.canvas.getContext("2d", { willReadFrequently: true });
    this.strokeContext = this.strokeCanvas.getContext("2d", { willReadFrequently: true });
    if (prior.width && prior.height) this.context.drawImage(prior, 0, 0, width, height);
    this.history.clear();
    this.stroke = null;
    return prior.width > 0 && prior.height > 0;
  }

  load(image, width, height) {
    this.resize(width, height);
    this.context.clearRect(0, 0, width, height);
    this.context.drawImage(image, 0, 0, width, height);
    this.history.clear();
  }

  texture(settings) {
    const normalized = normalizeBrushSettings(settings);
    const key = brushTextureKey(normalized);
    if (this.textureCache.has(key)) return this.textureCache.get(key);
    const radius = Math.max(1, normalized.size);
    const side = Math.max(2, Math.ceil(radius * 2 + 4));
    const texture = this.createCanvas();
    texture.width = texture.height = side;
    const context = texture.getContext("2d");
    const center = side / 2;
    const alpha = normalized.opacity;
    if (normalized.shape === "circle") {
      const gradient = context.createRadialGradient(center, center, 0, center, center, radius);
      gradient.addColorStop(0, `${normalized.color}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`);
      gradient.addColorStop(normalized.hardness, `${normalized.color}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`);
      gradient.addColorStop(1, `${normalized.color}00`);
      context.fillStyle = gradient;
      context.beginPath();
      context.arc(center, center, radius, 0, Math.PI * 2);
      context.fill();
    } else {
      const image = context.createImageData(side, side);
      const rgb = normalized.color.match(/[0-9a-f]{2}/gi).map((part) => Number.parseInt(part, 16));
      const hard = radius * normalized.hardness;
      const fade = Math.max(1e-6, radius - hard);
      for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
        const distance = Math.max(Math.abs(x + 0.5 - center), Math.abs(y + 0.5 - center));
        const amount = distance <= hard ? alpha : distance <= radius
          ? alpha * Math.pow(1 - (distance - hard) / fade, 2)
          : 0;
        const offset = (y * side + x) * 4;
        image.data[offset] = rgb[0]; image.data[offset + 1] = rgb[1]; image.data[offset + 2] = rgb[2];
        image.data[offset + 3] = Math.round(amount * 255);
      }
      context.putImageData(image, 0, 0);
    }
    this.textureCache.set(key, texture);
    if (this.textureCache.size > 24) this.textureCache.delete(this.textureCache.keys().next().value);
    return texture;
  }

  stamp(point) {
    const texture = this.texture(this.stroke.settings);
    const width = texture.width * this.stroke.scale.x;
    const height = texture.height * this.stroke.scale.y;
    const x = point.x - width / 2;
    const y = point.y - height / 2;
    this.strokeContext.drawImage(texture, x, y, width, height);
    this.stroke.dirty = unionRect(
      this.stroke.dirty, point.x, point.y,
      this.stroke.settings.size * this.stroke.scale.x, this.stroke.settings.size * this.stroke.scale.y,
      this.canvas.width, this.canvas.height,
    );
  }

  begin(point, settings, scale = { x: 1, y: 1 }) {
    if (!this.canvas.width || !this.canvas.height) return false;
    const normalized = normalizeBrushSettings(settings);
    this.strokeContext.clearRect(0, 0, this.strokeCanvas.width, this.strokeCanvas.height);
    this.stroke = { settings: normalized, scale, last: point, dirty: null };
    this.stamp(point);
    return true;
  }

  move(points) {
    if (!this.stroke) return false;
    const spacing = Math.max(1, this.stroke.settings.size * 2 * 0.1);
    for (const point of points) {
      // Inverse perspective can map an off-canvas pointer extremely far away.
      // Only interpolate the portion whose brush can touch this canvas.
      const radius = this.stroke.settings.size * Math.max(this.stroke.scale.x, this.stroke.scale.y) + 2;
      const segment = clipBrushSegment(this.stroke.last, point, this.canvas.width, this.canvas.height, radius);
      if (segment) for (const interpolated of interpolatedPoints(...segment, spacing, this.stroke.scale)) this.stamp(interpolated);
      this.stroke.last = point;
    }
    return true;
  }

  end(cancelled = false) {
    if (!this.stroke) return false;
    const { dirty, settings } = this.stroke;
    this.stroke = null;
    if (cancelled || !dirty) {
      this.strokeContext.clearRect(0, 0, this.strokeCanvas.width, this.strokeCanvas.height);
      return false;
    }
    const width = dirty.right - dirty.left;
    const height = dirty.bottom - dirty.top;
    const before = this.context.getImageData(dirty.left, dirty.top, width, height);
    this.context.save();
    this.context.globalCompositeOperation = settings.erasing ? "destination-out" : "source-over";
    this.context.drawImage(
      this.strokeCanvas, dirty.left, dirty.top, width, height,
      dirty.left, dirty.top, width, height,
    );
    this.context.restore();
    const after = this.context.getImageData(dirty.left, dirty.top, width, height);
    this.history.push({ ...dirty, before, after });
    this.strokeContext.clearRect(dirty.left, dirty.top, width, height);
    return true;
  }

  undo() { return this.history.apply(this.context, -1); }
  redo() { return this.history.apply(this.context, 1); }

  clear() {
    if (!this.canvas.width || !this.canvas.height) return false;
    const bounds = imageDataAlphaBounds(
      this.context.getImageData(0, 0, this.canvas.width, this.canvas.height),
    );
    if (!bounds) return false;
    const width = bounds.right - bounds.left, height = bounds.bottom - bounds.top;
    const before = this.context.getImageData(bounds.left, bounds.top, width, height);
    this.context.clearRect(bounds.left, bounds.top, width, height);
    const after = this.context.getImageData(bounds.left, bounds.top, width, height);
    this.history.push({ ...bounds, before, after });
    return true;
  }

  draw(context, x, y, width, height) {
    context.drawImage(this.canvas, x, y, width, height);
    if (!this.stroke) return;
    context.save();
    context.globalCompositeOperation = this.stroke.settings.erasing ? "destination-out" : "source-over";
    context.drawImage(this.strokeCanvas, x, y, width, height);
    context.restore();
  }
}
