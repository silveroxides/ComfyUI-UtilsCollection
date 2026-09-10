import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

import {
  BACKGROUND_PREVIEW_ALPHA,
  editorWidgetHeight,
  flexContentWidth,
  growNodeSize,
  inlinePanelLayout,
  naturalEditorWidth,
  previewHeight,
  visibleLayerListHeight,
  wrapConnectionRefresh,
} from "../web/staged_editor_layout.js";

test("background preview is not arbitrarily dimmed", () => {
  assert.equal(BACKGROUND_PREVIEW_ALPHA, 1);
});

test("natural editor width follows its widest complete control row", () => {
  assert.equal(naturalEditorWidth({
    coreWidth: 300, toolbarWidth: 620, rowWidths: [480, 840], overlayWidths: [410], chromeWidth: 20,
  }), 860);
});

test("complete flex rows accumulate controls, gaps, chrome, and rounding allowance", () => {
  assert.equal(flexContentWidth([28, 24, 48, 32, 820], 3, 8), 977);
  assert.equal(naturalEditorWidth({
    coreWidth: 300, toolbarWidth: 620, rowWidths: [977], overlayWidths: [410], chromeWidth: 20,
  }), 997);
});

test("inline picker expands node without changing preview width", () => {
  assert.deepEqual(inlinePanelLayout(900, 16, 0, 0), {
    extraWidth: 0, nodeWidth: 900, previewWidth: 884,
  });
  assert.deepEqual(inlinePanelLayout(900, 16, 220, 6), {
    extraWidth: 226, nodeWidth: 1126, previewWidth: 884,
  });
});

test("editor applies picker expansion above manual width and does not retain it after resizing", () => {
  // Execute the actual editor methods; the geometry helper alone cannot catch a
  // caller calculating the correct width but passing a different width to setSize.
  const source = readFileSync(new URL("../web/layered_background_editor.js", import.meta.url), "utf8");
  const methods = source.slice(source.indexOf("  updateLayout() {"), source.indexOf("  connectedLayers() {"));
  const prototype = runInNewContext(`(class {${methods}}).prototype`, {
    getComputedStyle: (element) => element.style,
    naturalEditorWidth, inlinePanelLayout, previewHeight, editorWidgetHeight, growNodeSize,
    wrapConnectionRefresh() {},
  });
  let open = false;
  let inlineOpen = false;
  const foregroundPanel = { root: { offsetWidth: 220, style: {} }, picker: {
    isOpen: () => inlineOpen, panel: { offsetWidth: 214 },
    setPanelHeight() { assert.fail("embedded picker must not be stretched to canvas height"); },
  } };
  const node = {
    size: [900, 700], computeSize: () => [600, 420],
    setSize(size) { this.size = [...size]; this.onResize(size); },
    graph: { setDirtyCanvas() {} },
  };
  const editor = Object.assign(Object.create(prototype), {
    node, root: { isConnected: true, style: { gap: "6px" } },
    previewRow: { style: { gap: "6px" } }, stage: { style: {} },
    layerList: { children: [] }, layerListGroup: { offsetHeight: 80 },
    paintColorPicker: { isOpen: () => open, panel: { offsetWidth: 220 }, setPanelHeight() {} },
    foregroundContent: null,
    manualNodeSize: [900, 700], layoutPanelWidth: 0, layoutMinWidth: 0,
    updateLayerListHeight() {}, measuredFlexWidth: (target) => target?.measuredWidth || 0, stageOverlayWidths: () => [],
    stylePixels: (_target, names) => names.includes("paddingLeft") ? 16 : 8,
    dimensions: () => ({ width: 1600, height: 900 }), scheduleLayout() {},
  });
  editor.wrapNodeLifecycle();
  editor.updateLayout();
  const closedHeight = editor.previewRow.style.minHeight;
  const closedMinimumHeight = editor.layoutMinHeight;
  open = true;
  editor.updateLayout();
  assert.equal(node.size[0], 1126);
  assert.equal(node.size[0] - 16 - 220 - 6, 884);
  assert.equal(editor.previewRow.style.minHeight, closedHeight);
  editor.updateLayout();
  assert.equal(node.size[0], 1126, "repeated layout must not accumulate picker width");
  node.setSize([1226, 700]);
  editor.updateLayout();
  assert.equal(node.size[0], 1226, "manual resizing with picker open must not add picker width again");
  open = false;
  editor.updateLayout();
  assert.equal(node.size[0], 1000);
  open = true;
  editor.updateLayout();
  assert.equal(node.size[0], 1226);
  editor.foregroundContent = foregroundPanel;
  editor.foregroundContent.active = { key: "foreground_0", kind: "brush" };
  editor.updateLayout();
  assert.equal(node.size[0], 1452);
  assert.equal(node.size[0] - 16 - 2 * (220 + 6), 984);
  assert.equal(editor.layoutMinHeight - closedMinimumHeight, previewHeight(984, 1600, 900) - previewHeight(884, 1600, 900));
  const bothPanelsHeight = editor.previewRow.style.minHeight;
  open = false;
  editor.updateLayout();
  assert.equal(node.size[0], 1226);
  assert.equal(editor.previewRow.style.minHeight, bothPanelsHeight);
  const sidebarMinimumHeight = editor.layoutMinHeight;
  inlineOpen = true;
  editor.updateLayout();
  assert.equal(node.size[0], 1226, "embedded picker must not add node width");
  assert.equal(editor.previewRow.style.minHeight, bothPanelsHeight);
  assert.equal(editor.layoutMinHeight, sidebarMinimumHeight, "embedded picker must not push down the layer list");
  inlineOpen = false;
  editor.foregroundContent.active = { key: "foreground_0", kind: "text" };
  editor.updateLayout();
  assert.equal(node.size[0], 1226);
  assert.equal(editor.previewRow.style.minHeight, bothPanelsHeight);
  editor.foregroundContent.active = null;
  editor.updateLayout();
  assert.equal(node.size[0], 1226, "idle actions keep the same left-panel width as the tools");
  assert.equal(editor.previewRow.style.minHeight, bothPanelsHeight);
  editor.layerList.children.push({ measuredWidth: 1120 });
  editor.updateLayout();
  assert.equal(node.size[0], 1362, "the complete layer-order row plus chrome and persistent sidebar sets the minimum width");
  const wideListHeight = editor.previewRow.style.minHeight;
  editor.foregroundContent.active = { key: "foreground_0", kind: "brush" };
  open = true;
  editor.updateLayout();
  assert.equal(node.size[0], 1588);
  assert.equal(node.size[0] - 16 - 2 * (220 + 6), 1120);
  assert.equal(editor.previewRow.style.minHeight, wideListHeight);
  editor.foregroundContent.active = null;
  open = false;
  editor.updateLayout();
  assert.equal(node.size[0], 1362, "idle sidebar must not squeeze the layer-order list below its minimum");
});

test("preview contains square and portrait images without increasing minimum height", () => {
  assert.equal(previewHeight(800), 450);
  assert.equal(previewHeight(800, 1600, 800), 400);
  assert.equal(previewHeight(800, 800, 1600), 450);
  assert.equal(previewHeight(800, 800, 800), 450);
});

test("layer list exposes three complete measured rows", () => {
  assert.equal(visibleLayerListHeight([40, 42, 44, 50], 3, 6), 138);
  assert.equal(visibleLayerListHeight([40, 42], 3, 6), 91);
});

test("widget height is composed from measured sections", () => {
  assert.equal(editorWidgetHeight({
    stageHeight: 450, toolbarHeight: 34, layerGroupHeight: 150, chromeHeight: 10, gaps: 12,
  }), 656);
});

test("required sizing grows from the manual size floor rather than previous automatic growth", () => {
  assert.deepEqual(growNodeSize([900, 700], [800, 600]), [900, 700]);
  assert.deepEqual(growNodeSize([700, 500], [800, 600]), [800, 600]);
  assert.deepEqual(growNodeSize([800, 600], [800, 600]), [800, 600]);
});

test("connection changes preserve existing handlers and force source refresh", async () => {
  const calls = [];
  const node = {
    onConnectionsChange(...args) {
      calls.push(["original", ...args]);
      return "preserved";
    },
  };
  wrapConnectionRefresh(node, (force) => calls.push(["refresh", force]));

  assert.equal(node.onConnectionsChange("input", 1), "preserved");
  assert.deepEqual(calls, [["original", "input", 1]]);
  await new Promise((resolve) => queueMicrotask(resolve));
  assert.deepEqual(calls, [
    ["original", "input", 1],
    ["refresh", true],
  ]);
});
