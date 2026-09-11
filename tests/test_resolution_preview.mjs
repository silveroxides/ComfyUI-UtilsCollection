import test from "node:test";
import assert from "node:assert/strict";
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
