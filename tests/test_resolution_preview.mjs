import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { h3VideoLengthFromSeconds, h3ReferenceFrameRange } from "../web/h3_video_length.js";

test("H3 reference range preserves zero start and resolves remaining duration", () => {
  assert.deepEqual(h3ReferenceFrameRange(0, 5), { start: 0, end: 123, length: 124 });
  assert.deepEqual(h3ReferenceFrameRange(1, 0), { start: 39, end: null, length: null });
  assert.deepEqual(h3ReferenceFrameRange(1, 2, 10), { start: 39, end: 94, length: 56 });
  assert.deepEqual(h3ReferenceFrameRange(1, 0, 10), { start: 39, end: 247, length: 209 });
  assert.deepEqual(h3ReferenceFrameRange(1, 100, 10), { start: 39, end: 247, length: 209 });
  assert.deepEqual(h3ReferenceFrameRange(null, 5, 10), { start: null, end: null, length: null });
  assert.deepEqual(h3ReferenceFrameRange(1, null, 10), { start: 39, end: null, length: null });
  assert.equal(h3VideoLengthFromSeconds(22.5 / 24), 22);
});

test("indexed H3 segments share exact boundaries and report padding", () => {
  assert.deepEqual(h3ReferenceFrameRange(99, 99, 32.4, 3, 0), { start: 0, end: 259, length: 260, padding: 0 });
  assert.deepEqual(h3ReferenceFrameRange(99, 99, 32.4, 3, 1), { start: 260, end: 519, length: 260, padding: 0 });
  assert.deepEqual(h3ReferenceFrameRange(99, 99, 32.4, 3, 2), { start: 520, end: 777, length: 260, padding: 2 });
  assert.deepEqual(h3ReferenceFrameRange(99, 99, 32.4, 3, 1, 22), { start: 260, end: 514, length: 277, padding: 0 });
  assert.deepEqual(h3ReferenceFrameRange(0, 0, null, 3, 0), { start: null, end: null, length: null });
  assert.deepEqual(h3ReferenceFrameRange(0, 0, 32.4, 3, 3), { start: null, end: null, length: null });
});

import {
  clampResolutionPreviewSize,
  resolutionPreviewMinimumSize,
} from "../web/resolution_preview_layout.js";


test("resolution preview reserves height and expands only for measured text", () => {
  assert.deepEqual(resolutionPreviewMinimumSize([180, 220]), [180, 246]);
  assert.deepEqual(resolutionPreviewMinimumSize([180, 220], 100), [180, 246]);
  assert.deepEqual(resolutionPreviewMinimumSize([180, 220], 200.5), [217, 246]);
  assert.deepEqual(resolutionPreviewMinimumSize([320, 180], 200), [320, 206]);
});


test("resolution preview prevents manual resize below its computed minimum", () => {
  const size = [160, 190];
  assert.equal(clampResolutionPreviewSize(size, [220, 246]), size);
  assert.deepEqual(size, [220, 246]);
});


test("collapsed nodes keep their collapsed size", () => {
  const size = [90, 30];
  clampResolutionPreviewSize(size, [220, 246], true);
  assert.deepEqual(size, [90, 30]);
});

test("video widget previews agree with backend middle-band resolutions before execution", async () => {
  const source = await readFile(new URL("../web/resolution_preview.js", import.meta.url), "utf8");
  let extension;
  vm.runInNewContext(source.replace(/^import[\s\S]*?;\r?$/gm, ""), {
    app: { registerExtension(value) { extension = value; } },
    document: { createElement() { return { getContext() { return { measureText(text) { return { width: text.length * 6 }; } }; } }; } },
    h3VideoLengthFromSeconds,
    h3ReferenceFrameRange,
    clampResolutionPreviewSize,
    resolutionPreviewMinimumSize,
  });
  class Node {
    constructor() {
      this.size = [180, 220];
      this.widgets = [
        { name: "aspect_ratio", value: "4:3" },
        { name: "megapixels", value: 0.5 },
        { name: "multiple", value: 32 },
        { name: "duration_seconds", value: 5 },
      ];
    }
    computeSize() { return [180, 220]; }
    setSize(size) { this.size = size; }
    setDirtyCanvas() {}
  }
  await extension.beforeRegisterNodeDef(Node, { name: "UC_VideoResolutionSelector" });
  const node = new Node();
  node.onNodeCreated();
  assert.equal(node.__ucResolutionPreview, "864×672 · 124 frames");
  for (const [ratio, megapixels, backendResolution] of [
    ["3:4", 0.5, "672×864"],
    ["21:9", 0.7, "1280×544"],
    ["9:21", 0.7, "544×1280"],
  ]) {
    node.widgets[0].value = ratio;
    node.widgets[1].value = megapixels;
    node.onWidgetChanged("aspect_ratio");
    const previewBeforeExecution = node.__ucResolutionPreview;
    node.onExecuted({ resolution: [`${backendResolution} · 124 frames`] });
    assert.equal(previewBeforeExecution, node.__ucResolutionPreview);
  }
});
