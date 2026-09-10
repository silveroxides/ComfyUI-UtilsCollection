import test from "node:test";
import assert from "node:assert/strict";
import { ForegroundContentEditor } from "../web/foreground_content_editor.js";
import { PaintColorPicker } from "../web/paint_color_picker.js";

// DOM/canvas doubles exercise interaction and persistence, not raster quality.
class Element {
  constructor(tag) {
    this.tag = tag; this.style = {}; this.children = []; this.listeners = {};
    this.width = 300; this.height = 150; this.value = "";
    this.context = {
      clearRect() {}, putImageData() {}, save() {}, restore() {}, translate() {}, scale() {},
      drawImage(source) { if (!source.width || !source.height) throw new Error("Empty source"); },
      getImageData: (_x, _y, w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
      createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
      fillText() {},
    };
  }
  append(...children) { for (const child of children) { child.remove(); child.parentElement = this; this.children.push(child); } }
  replaceChildren(...children) { for (const child of [...this.children]) child.remove(); this.append(...children); }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter((child) => child !== this); this.parentElement = null; }
  addEventListener(type, listener) { (this.listeners[type] ??= []).push(listener); }
  removeEventListener(type, listener) { this.listeners[type] = (this.listeners[type] || []).filter((item) => item !== listener); }
  setAttribute(name, value) { (this.attributes ??= {})[name] = value; }
  getContext() { return this.context; }
  focus() {}
  blur() {}
  hasPointerCapture() { return false; }
  setPointerCapture() {}
  toBlob(callback) { callback(new Blob(["test pixels"], { type: "image/png" })); }
}

function harness() {
  globalThis.document = { createElement: (tag) => new Element(tag), addEventListener() {}, removeEventListener() {} };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  globalThis.window = new EventTarget();
  window.devicePixelRatio = 1;
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: { getItem: () => null, setItem() {} } });
  const placements = { foreground_0: {}, foreground_1: {} };
  const editor = {
    data: { foreground_content: {} }, disposed: false, canvas: new Element("canvas"),
    node: { graph: {} }, status: {},
    layerSourceSize: () => ({ width: 64, height: 48 }),
    layerMetadata: () => ({ crop_width: 64, crop_height: 48 }),
    layerPlacement: (key) => placements[key],
    dimensions: () => ({ width: 1600, height: 900 }),
    resolvedGeometry: () => ({ source: { width: 64, height: 48 }, transformed: { identity: true }, frame: { x: 0, y: 0, width: 64, height: 48 } }),
    requestDraw() {}, scheduleLayout() {}, persistPlacement() {}, flushPlacement() {},
    selectLayer(key) { editor.selected = key; },
    layerContextActions: () => [],
    paintSettings: { color: "#000000", size: 5, shape: "circle", opacity: 1, hardness: 1, erasing: false },
    savePaintSettings() {},
  };
  const requests = [];
  const api = { fetchApi: async (_url, options) => {
    const filename = options.body.get("image").name;
    requests.push(filename);
    return { ok: true, json: async () => ({ name: filename, subfolder: "clipspace" }) };
  } };
  const content = new ForegroundContentEditor(editor, api);
  return { content, editor, api, requests, placements };
}

test("text reset, visibility and tool history are independent of brush and placement", async () => {
  const { content, editor } = harness();
  const state = content.get("foreground_0", true);
  await state.loading;
  const brush = state.brush;
  content.changeText(state, { value: "first\nsecond", size: 44 });
  content.toggleVisible("foreground_0", "text");
  await content.reset("foreground_0", "text");
  assert.equal(state.item.text.value, "");
  assert.equal(state.item.text.visible, false);
  assert.equal(state.brush, brush);
  content.active = { key: "foreground_0", kind: "text" };
  content.buildControls = () => {};
  content.history(-1);
  assert.equal(state.item.text.value, "first\nsecond");
  assert.equal(state.item.text.size, 44);
  assert.equal(state.item.text.visible, false);
  assert.equal(editor.data.foreground_content.foreground_1, undefined);
  content.dispose();
});

test("activating text exits transforms, hidden stays hidden, and keyboard/reset controls are isolated", async () => {
  const { content, editor, placements } = harness();
  editor.rotateLayer = "foreground_0";
  content.toggleVisible("foreground_0", "text");
  await content.activate("foreground_0", "text");
  assert.equal(editor.rotateLayer, null);
  assert.equal(content.active.kind, "text");
  assert.equal(content.get("foreground_0").item.text.visible, false);
  assert.match(content.message.textContent, /Hidden/);
  let stopped = false, prevented = false;
  content.root.listeners.keydown[0]({ key: "Backspace", stopPropagation() { stopped = true; } });
  assert.ok(stopped);
  content.keyDown({ key: "Delete", stopPropagation() {}, preventDefault() { prevented = true; } });
  assert.ok(prevented);
  const labels = content.menuActions("foreground_0").map((action) => action.label).filter(Boolean);
  assert.deepEqual(labels, ["Brush", "Show Brush", "Reset Brush", "Text", "Show Text", "Reset Text"]);
  placements.foreground_0.locked = true;
  content.sync();
  assert.equal(content.active, null);
  assert.ok(content.menuActions("foreground_0").find((a) => a.label === "Brush").disabled);
  assert.ok(content.menuActions("foreground_0").find((a) => a.label === "Reset Text").disabled);
  content.dispose();
});

test("text steppers commit values, clamp opacity, and cancel numeric typing without exiting text mode", async () => {
  const { content } = harness();
  await content.activate("foreground_0", "text");
  const state = content.get("foreground_0");
  const sizeRow = content.root.children.find((row) => row.children[0]?.textContent === "Size");
  const [, , sizeInput, increaseSize] = sizeRow.children;
  sizeInput.value = "70";
  sizeInput.listeners.change[0]();
  assert.equal(state.item.text.size, 70);
  increaseSize.listeners.click[0]({ stopPropagation() {} });
  assert.equal(state.item.text.size, 71);
  sizeInput.value = "not a number";
  sizeInput.listeners.change[0]();
  assert.equal(sizeInput.value, "71");
  sizeInput.value = "100";
  sizeInput.listeners.keydown[0]({ key: "Escape", preventDefault() {}, stopPropagation() {} });
  assert.equal(sizeInput.value, "71");
  assert.equal(content.active.kind, "text");
  const opacityRow = content.root.children.find((row) => row.children[0]?.textContent === "Opacity");
  const [, decreaseOpacity, opacityInput, increaseOpacity] = opacityRow.children;
  opacityInput.value = "0";
  opacityInput.listeners.change[0]();
  decreaseOpacity.listeners.click[0]({ stopPropagation() {} });
  assert.equal(state.item.text.opacity, 0);
  increaseOpacity.listeners.click[0]({ stopPropagation() {} });
  assert.equal(state.item.text.opacity, 0.01);
  content.dispose();
});

test("brush and text share the embedded picker without mixing colors or moving text during eyedropper sampling", async () => {
  const { content, editor } = harness();
  await content.activate("foreground_0", "brush");
  const brushPicker = content.picker;
  assert.ok(brushPicker instanceof PaintColorPicker);
  assert.equal(brushPicker.panel.parentElement, content.root);
  brushPicker.setColor("#123456");
  assert.equal(editor.paintSettings.color, "#123456");
  await content.activate("foreground_0", "text");
  assert.equal(brushPicker.panel.parentElement, null);
  assert.ok(content.picker instanceof PaintColorPicker);
  assert.equal(content.picker.panel.parentElement, content.root);
  const state = content.get("foreground_0");
  content.picker.setColor("#abcdef");
  assert.equal(state.item.text.color, "#abcdef");
  assert.equal(editor.paintSettings.color, "#123456");
  const anchor = [state.item.text.x, state.item.text.y];
  window.devicePixelRatio = 1;
  editor.canvasPoint = () => ({ x: 4, y: 4 });
  editor.sampleCanvas = new Element("canvas");
  editor.sampleCanvas.context.getImageData = () => ({ data: new Uint8ClampedArray([9, 8, 7, 255]) });
  content.local = () => ({ x: 4, y: 4 });
  content.eyedropper = true;
  content.down({ button: 0, pointerId: 1, preventDefault() {}, stopPropagation() {} });
  assert.equal(state.item.text.color, "#090807");
  assert.deepEqual([state.item.text.x, state.item.text.y], anchor);
  assert.equal(editor.paintSettings.color, "#123456");
  content.dispose();
});

test("persistent idle sidebar reuses the selected foreground's context actions and disabled state", () => {
  const { content, editor } = harness();
  const called = [];
  let locked = false;
  editor.layerContextActions = (key) => [
    { label: "Brush", disabled: locked, callback: () => called.push(key) },
    { separator: true },
    { label: locked ? "Unlock" : "Lock", checked: locked, callback: () => { locked = !locked; } },
  ];
  content.sync();
  assert.match(content.root.children[0].textContent, /Select a foreground/);
  editor.selected = "foreground_0";
  content.sync();
  assert.equal(content.root.style.display, "flex");
  const buttons = () => content.root.children.filter((element) => element.tag === "button");
  buttons()[0].listeners.click[0]({ stopPropagation() {} });
  assert.deepEqual(called, ["foreground_0"]);
  buttons()[1].listeners.click[0]({ stopPropagation() {} });
  assert.equal(buttons()[0].disabled, true);
  assert.equal(buttons()[1].textContent, "Unlock");
  editor.selected = "foreground_1";
  locked = false;
  content.sync();
  buttons()[0].listeners.click[0]({ stopPropagation() {} });
  assert.deepEqual(called, ["foreground_0", "foreground_1"]);
  content.exit();
  assert.equal(content.root.style.display, "flex");
  assert.match(content.root.children[0].textContent, /Foreground 1/);
  content.dispose();
});

test("queue save drains concurrent edits and reset cannot be overwritten by an earlier upload", async () => {
  const { content, api, requests } = harness();
  const state = content.get("foreground_0", true);
  await state.loading;
  content.changeText(state, { value: "before" });
  const original = api.fetchApi;
  let release, started;
  const waitStarted = new Promise((resolve) => { started = resolve; });
  api.fetchApi = async (...args) => {
    if (!release) { await new Promise((resolve) => { release = resolve; started(); }); }
    return original(...args);
  };
  const saving = content.flush();
  await waitStarted;
  await content.reset("foreground_0", "text");
  release();
  await saving;
  assert.equal(requests.length, 2);
  assert.notEqual(requests[0], requests[1]);
  assert.equal(state.item.text.value, "");
  assert.equal(state.item.text.asset.filename, requests[1]);
  assert.equal(content.hasPending(), false);
  content.dispose();
});

test("failed saves remain pending and reject queueing until retry succeeds", async () => {
  const { content, api } = harness();
  const state = content.get("foreground_0", true);
  await state.loading;
  content.changeText(state, { value: "keep me" });
  const original = api.fetchApi;
  api.fetchApi = async () => ({ ok: false, status: 503 });
  await assert.rejects(content.flush(), /503/);
  assert.ok(content.hasPending());
  assert.equal(state.item.text.asset, undefined);
  api.fetchApi = original;
  await content.flush();
  assert.equal(content.hasPending(), false);
  content.dispose();
});

test("queued save awaits restored assets instead of saving an uninitialized canvas", async () => {
  const { content, editor, api } = harness();
  editor.data.foreground_content.foreground_0 = {
    width: 64, height: 48, brush: { visible: true, asset: { filename: "stored.png", type: "input", subfolder: "clipspace" } },
    text: { visible: true, value: "" },
  };
  api.apiURL = (url) => url;
  let image;
  globalThis.Image = class {
    constructor() { this.width = 64; this.height = 48; image = this; }
  };
  let finished = false;
  const saving = content.flush().then(() => { finished = true; });
  await Promise.resolve();
  assert.equal(finished, false);
  image.onload();
  await saving;
  assert.equal(content.get("foreground_0").ready, true);
  assert.equal(content.get("foreground_0").item.brush.asset.filename, "stored.png");
  content.dispose();
});

test("background autosave leaves an in-progress brush stroke captured; queue save finishes it", async () => {
  const { content } = harness();
  const state = content.get("foreground_0", true);
  await state.loading;
  state.brush.end = () => true;
  content.pointer = { id: 7, state };
  await content.flush({ finishStroke: false });
  assert.equal(content.pointer.id, 7);
  assert.equal(state.item.brush.asset, undefined);
  await content.flush();
  assert.equal(content.pointer, null);
  assert.ok(state.item.brush.asset.filename);
  content.dispose();
});

test("queue save revisits an earlier foreground changed while a later foreground uploads", async () => {
  const { content, api, requests } = harness();
  const first = content.get("foreground_0", true), second = content.get("foreground_1", true);
  await Promise.all([first.loading, second.loading]);
  content.changeText(first, { value: "first" });
  content.changeText(second, { value: "second" });
  const original = api.fetchApi;
  api.fetchApi = async (...args) => {
    if (requests.length === 1) content.changeText(first, { value: "changed during second upload" });
    return original(...args);
  };
  await content.flush();
  assert.equal(requests.length, 3);
  assert.equal(first.item.text.asset.filename, requests[2]);
  assert.equal(second.item.text.asset.filename, requests[1]);
  assert.equal(content.hasPending(), false);
  content.dispose();
});

test("object eraser routes strokes to a saved mask and shares chronological brush undo and reset", async () => {
  const { content, editor } = harness();
  const state = content.get("foreground_0", true);
  await state.loading;
  content.active = { key: "foreground_0", kind: "brush" };
  content.local = () => ({ x: 4, y: 4 });
  content.buildControls = () => {};
  editor.paintSettings = { color: "#123456", size: 5, opacity: 0.5, hardness: 1, erasing: true };
  const calls = [];
  for (const kind of ["brush", "object_erase"]) {
    state[kind].begin = (_point, settings) => { calls.push([kind, "begin", settings.erasing]); return true; };
    for (const operation of ["end", "undo", "redo", "clear"]) state[kind][operation] = () => { calls.push([kind, operation]); return true; };
  }
  const event = { button: 0, pointerId: 1, preventDefault() {}, stopPropagation() {} };
  content.down(event); content.end(false, event);
  content.objectErasing = true;
  content.down(event); content.end(false, event);
  assert.deepEqual(calls.filter((call) => call[1] === "begin"), [["brush", "begin", true], ["object_erase", "begin", false]]);
  content.history(-1); content.history(-1);
  assert.deepEqual(calls.filter((call) => call[1] === "undo"), [["object_erase", "undo"], ["brush", "undo"]]);
  content.history(1); content.history(1);
  await content.flush();
  assert.ok(state.item.object_erase.asset.filename);
  assert.notEqual(state.item.object_erase.asset.filename, state.item.brush.asset.filename);
  await content.reset("foreground_0", "brush");
  assert.deepEqual(calls.filter((call) => call[1] === "clear"), [["brush", "clear"], ["object_erase", "clear"]]);
  assert.equal(state.generation.text, 0);
  content.history(-1);
  assert.deepEqual(calls.slice(-2), [["object_erase", "undo"], ["brush", "undo"]]);
  content.dispose();
});
