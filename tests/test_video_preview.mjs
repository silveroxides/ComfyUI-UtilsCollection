import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../web/video_preview.js", import.meta.url), "utf8");
let extension;
globalThis.__videoPreviewTest = {
  api: { apiURL(route) { return `/comfy${route}`; } },
  app: { registerExtension(value) { extension = value; } },
};
const moduleSource = source.replace(/^import .*;\r?\n/gm, "")
  .replace(/^/, "const { api, app } = globalThis.__videoPreviewTest;\n");
await import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString("base64")}`);
extension.setup();
const { api } = globalThis.__videoPreviewTest;
delete globalThis.__videoPreviewTest;

test("video view requests omit image conversion but preserve file routing", () => {
  const url = new URL(api.apiURL("/view?filename=ref-segment-01.mp4&type=input&subfolder=clips&preview=webp%3B90&rand=42"), "http://localhost");
  assert.equal(url.pathname, "/comfy/view");
  assert.equal(url.searchParams.has("preview"), false);
  assert.equal(url.searchParams.get("filename"), "ref-segment-01.mp4");
  assert.equal(url.searchParams.get("type"), "input");
  assert.equal(url.searchParams.get("subfolder"), "clips");
  assert.equal(url.searchParams.get("rand"), "42");
});

test("images, unrelated routes and plain video URLs stay unchanged", () => {
  for (const route of ["/view?filename=frame.png&preview=webp", "/other?filename=clip.mp4&preview=webp", "/view?filename=clip.mp4&type=input"]) {
    assert.equal(api.apiURL(route), `/comfy${route}`);
  }
});
