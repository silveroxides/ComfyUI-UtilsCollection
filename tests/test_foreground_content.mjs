import test from "node:test";
import assert from "node:assert/strict";
import { foregroundLocalPoint, foregroundDisplayPoint, normalizeForegroundContent } from "../web/foreground_content.js";
import { parsePlacementData, serializePlacementData, resolveLayerGeometry } from "../web/placement_geometry.js";
import { uploadForegroundCanvas } from "../web/paint_persistence.js";
import { updatePromptNodeInputs } from "../web/staged_queue.js";
import { clipBrushSegment } from "../web/paint_brush.js";

test("foreground painting maps rotated and warped points back to the source canvas, including flips", () => {
  for (const rotation of [0, 45, -90, 179]) for (const flip of [false, true]) {
    const geometry = resolveLayerGeometry({
      backgroundWidth: 1024, backgroundHeight: 768, sourceWidth: 300, sourceHeight: 180,
      placement: { scale: 0.8, center_x: 0.4, center_y: 0.6, rotation, corners: [[-0.8, -1], [0.9, -0.5], [1, 1], [-1, 0.7]] },
      workspacePadding: 0.5,
    });
    for (const point of [{ x: 0, y: 0 }, { x: 300, y: 180 }, { x: 92, y: 74 }]) {
      const [x, y] = foregroundDisplayPoint(geometry.points, point, 300, 180, flip, !flip);
      const actual = foregroundLocalPoint(geometry.points, { x, y }, 300, 180, flip, !flip);
      assert.ok(Math.abs(actual.x - point.x) < 1e-7);
      assert.ok(Math.abs(actual.y - point.y) < 1e-7);
    }
  }
  const trapezoid = [[10, 10], [110, 10], [90, 70], [30, 70]];
  assert.deepEqual(foregroundLocalPoint(trapezoid, { x: 10, y: 10 }, 100, 60), { x: 0, y: 0 });
  const corner = foregroundLocalPoint(trapezoid, { x: 90, y: 70 }, 100, 60);
  assert.ok(Math.abs(corner.x - 100) < 1e-9 && Math.abs(corner.y - 60) < 1e-9);
});

test("foreground content survives placement reset and serialization without creating layer indices", () => {
  const content = normalizeForegroundContent({ foreground_0_face_0: {
    width: 800, height: 600, brush: { visible: false, asset: { filename: "brush.png", type: "input", subfolder: "clipspace" } },
    text: { value: "hello\nworld", size: 28, x: 0.2, y: 0.4 },
  } });
  const data = parsePlacementData({ version: 3, layers: {}, foreground_content: content });
  data.layers.foreground_0_face_0 = { scale: 1, center_x: 0.5, center_y: 0.5 };
  const restored = parsePlacementData(serializePlacementData(data));
  assert.deepEqual(restored.foreground_content, content);
  assert.deepEqual(restored.layer_order, []);
  assert.equal(JSON.parse(serializePlacementData(parsePlacementData("{}"))).foreground_content, undefined);
});

test("off-canvas perspective strokes are clipped before allocating interpolated samples", () => {
  const segment = clipBrushSegment({ x: 50, y: 50 }, { x: 1e12, y: 50 }, 100, 100, 5);
  assert.deepEqual(segment, [{ x: 50, y: 50 }, { x: 105, y: 50 }]);
  assert.equal(clipBrushSegment({ x: -20, y: -20 }, { x: -20, y: 1e12 }, 100, 100, 5), null);
});

test("foreground PNG saves are immutable, fail on upload errors, and synchronize queued workflow snapshots", async () => {
  const requests = [];
  const api = { fetchApi: async (_url, options) => {
    requests.push(options.body);
    return { ok: true, json: async () => ({ name: options.body.get("image").name, subfolder: "clipspace" }) };
  } };
  const canvas = { toBlob: (callback) => callback(new Blob(["pixels"], { type: "image/png" })) };
  const first = await uploadForegroundCanvas(api, canvas);
  const second = await uploadForegroundCanvas(api, canvas);
  assert.notEqual(first.filename, second.filename);
  assert.equal(requests[0].get("overwrite"), "false");
  await assert.rejects(uploadForegroundCanvas({ fetchApi: async () => ({ ok: false, status: 500 }) }, canvas), /500/);
  const prompt = { output: { 1: { inputs: { placement_data: "old" } } }, workflow: { nodes: [{ id: 1, widgets_values: ["old"] }] } };
  const updated = updatePromptNodeInputs(prompt, [{ nodeId: 1, inputs: { placement_data: "saved" }, serializedNode: { widgets_values: ["saved"] } }]);
  assert.equal(updated.output[1].inputs.placement_data, "saved");
  assert.deepEqual(updated.workflow.nodes[0].widgets_values, ["saved"]);
  assert.deepEqual(prompt.workflow.nodes[0].widgets_values, ["old"]);
});
