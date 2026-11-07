import test from "node:test";
import assert from "node:assert/strict";

import {
  EMPTY_PREVIEW_ASPECT,
  MAX_PREVIEW_HEIGHT,
  MIN_PREVIEW_HEIGHT,
  clampVideoPreviewSize,
  growNodeSize,
  videoPreviewHeight,
} from "../web/video_preview_layout.js";

test("video preview height is zero when preview_mode is none", () => {
  assert.equal(videoPreviewHeight(320, 1920, 1080, "none"), 0);
  assert.equal(videoPreviewHeight(320, 1080, 1920, "none"), 0);
  assert.equal(videoPreviewHeight(0, 1920, 1080, "thumbnail"), 0);
});

test("video preview height bounds landscape, square, and portrait aspect ratios", () => {
  // 16:9 landscape video at width 320 -> 320 / (16/9) = 180 (within MAX 200)
  const landscape = videoPreviewHeight(320, 1920, 1080, "thumbnail");
  assert.equal(landscape, 180);

  // 9:16 portrait video at width 320: raw aspect height would be 320 / (9/16) = 569,
  // but must be clamped to ceiling 180 and never exceed MAX_PREVIEW_HEIGHT (200)
  const portrait = videoPreviewHeight(320, 1080, 1920, "thumbnail");
  assert.equal(portrait, 180);
  assert.ok(portrait <= MAX_PREVIEW_HEIGHT);

  // 1:1 square video at width 320: raw is 320, must be clamped to ceiling 180
  const square = videoPreviewHeight(320, 1000, 1000, "thumbnail");
  assert.equal(square, 180);
  assert.ok(square <= MAX_PREVIEW_HEIGHT);

  // Narrow width enforces minimum preview height floor
  const narrow = videoPreviewHeight(100, 1920, 1080, "reduced_proxy");
  assert.equal(narrow, MIN_PREVIEW_HEIGHT);
});

test("clampVideoPreviewSize preserves collapsed node dimensions", () => {
  const size = [80, 30];
  clampVideoPreviewSize(size, [300, 400], true);
  assert.deepEqual(size, [80, 30]);
});

test("clampVideoPreviewSize clamps uncollapsed dimensions to minimums", () => {
  const size = [200, 150];
  clampVideoPreviewSize(size, [260, 320], false);
  assert.deepEqual(size, [260, 320]);
});

test("growNodeSize preserves manual resize floors", () => {
  assert.deepEqual(growNodeSize([400, 500], [250, 300]), [400, 500]);
  assert.deepEqual(growNodeSize([200, 200], [300, 350]), [300, 350]);
});
