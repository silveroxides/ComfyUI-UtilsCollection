import test from "node:test";
import assert from "node:assert/strict";
import { foregroundBrushScale } from "../web/foreground_content.js";

import {
  brushTextureKey,
  interpolatedPoints,
  imageDataAlphaBounds,
  normalizeBrushSettings,
  paintLayerVisible,
  PaintLayerCanvas,
} from "../web/paint_brush.js";

function mockCanvas() {
  const calls = [];
  const gradient = { addColorStop: (...args) => calls.push(["stop", ...args]) };
  const context = {
    calls,
    globalCompositeOperation: "source-over",
    save: () => calls.push(["save"]), restore: () => calls.push(["restore"]),
    clearRect: (...args) => calls.push(["clearRect", ...args]),
    drawImage: (...args) => {
      if (!args[0].width || !args[0].height) throw new Error("Cannot draw a zero-sized source canvas");
      calls.push(["drawImage", ...args]);
    },
    getImageData: (_x, _y, width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) }),
    putImageData: (...args) => calls.push(["putImageData", ...args]),
    createRadialGradient: () => gradient,
    beginPath: () => calls.push(["beginPath"]), arc: (...args) => calls.push(["arc", ...args]),
    fill: () => calls.push(["fill"]),
    createImageData: (width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) }),
  };
  return { width: 0, height: 0, calls, getContext: () => context };
}

test("paint remains visible while editing even when excluded from final composition", () => {
  assert.equal(paintLayerVisible(false, false), false);
  assert.equal(paintLayerVisible(false, true), true);
  assert.equal(paintLayerVisible(true, false), true);
  assert.equal(paintLayerVisible(undefined, false), true);
});

test("brush settings clamp without losing supported shapes", () => {
  assert.deepEqual(normalizeBrushSettings({
    shape: "square", color: "#ABCDEF", size: 999, opacity: -1, hardness: 2, erasing: true,
  }), {
    shape: "square", color: "#abcdef", size: 250, opacity: 0, hardness: 1, erasing: true,
  });
});

test("missing opacity defaults to solid without replacing a saved value", () => {
  assert.equal(normalizeBrushSettings({}).opacity, 1);
  assert.equal(normalizeBrushSettings({ opacity: 0.7 }).opacity, 0.7);
});

test("alpha bounds isolate the retained undo patch for clear", () => {
  const data = new Uint8ClampedArray(5 * 4 * 4);
  data[(1 * 5 + 2) * 4 + 3] = 255;
  data[(3 * 5 + 4) * 4 + 3] = 1;
  assert.deepEqual(imageDataAlphaBounds({ data, width: 5, height: 4 }), {
    left: 2, top: 1, right: 5, bottom: 4,
  });
  assert.equal(imageDataAlphaBounds({ data: new Uint8ClampedArray(16), width: 2, height: 2 }), null);
});

test("brush cache identity includes opacity and every rendered control", () => {
  const base = { shape: "circle", color: "#123456", size: 10, opacity: 0.7, hardness: 0.5 };
  for (const changed of [
    { shape: "square" }, { color: "#654321" }, { size: 11 }, { opacity: 0.6 }, { hardness: 0.4 },
  ]) assert.notEqual(brushTextureKey(base), brushTextureKey({ ...base, ...changed }));
});

test("fast strokes receive stamps no farther apart than requested spacing", () => {
  const points = interpolatedPoints({ x: 0, y: 0 }, { x: 19, y: 0 }, 2);
  assert.deepEqual(points.at(-1), { x: 19, y: 0 });
  let prior = { x: 0, y: 0 };
  for (const point of points) {
    assert.ok(Math.hypot(point.x - prior.x, point.y - prior.y) <= 2);
    prior = point;
  }
});

test("paint canvas allocates, previews, commits, and traverses history", () => {
  const created = [];
  const paint = new PaintLayerCanvas(() => {
    const canvas = mockCanvas();
    created.push(canvas);
    return canvas;
  });
  paint.resize(64, 48);
  assert.equal(paint.begin({ x: 8, y: 9 }, { color: "#123456", size: 4 }), true);
  assert.equal(paint.move([{ x: 20, y: 9 }]), true);
  const preview = mockCanvas().getContext("2d");
  paint.draw(preview, 0, 0, 64, 48);
  assert.ok(preview.calls.some(([name]) => name === "drawImage"));
  assert.equal(paint.end(false), true);
  assert.equal(paint.history.canUndo, true);
  assert.equal(paint.undo(), true);
  assert.equal(paint.redo(), true);
});

test("brush stamps keep equal displayed axes on rectangular foregrounds and after aspect changes", () => {
  for (const shape of ["circle", "square"]) for (const [width, height, placedWidth, placedHeight] of [
    [300, 300, 600, 200], [300, 300, 200, 600], [300, 100, 600, 200], [100, 300, 200, 600],
  ]) {
    const paint = new PaintLayerCanvas(mockCanvas);
    paint.resize(width, height);
    paint.begin({ x: width / 2, y: height / 2 }, { shape, size: 5 }, foregroundBrushScale(width, height, placedWidth, placedHeight));
    const [, , , , stampWidth, stampHeight] = paint.strokeContext.calls.find(([name]) => name === "drawImage");
    assert.ok(Math.abs(stampWidth * placedWidth / width - stampHeight * placedHeight / height) < 1e-9);
  }
});
