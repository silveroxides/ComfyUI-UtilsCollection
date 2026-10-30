import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  BBOX_DEFAULT_COLORS,
  drawNormalizedBox,
  moveNormalizedBox,
  normalizeBoxRegion,
  parseBoxEditorData,
  resizeNormalizedBox,
  serializeBoxEditorData,
} from "./placement_geometry.js";
import { PaintColorPicker, rgbToHex } from "./paint_color_picker.js";
import { createStagedAction } from "./staged_editor_controls.js";
import {
  editorWidgetHeight,
  flexContentWidth,
  growNodeSize,
  inlinePanelLayout,
  naturalEditorWidth,
  previewHeight,
  visibleLayerListHeight,
  wrapConnectionRefresh,
} from "./staged_editor_layout.js";
import { updatePromptNodeInputs } from "./staged_queue.js";

const BBOX_NODE_TYPES = new Set([
  "UC_CreateBoundingBoxes",
  "CreateBoundingBoxesNodes1",
]);
const bboxEditors = new Set();
let latestOutputs = {};

const originalApiQueuePrompt = api.queuePrompt;
api.queuePrompt = async function(index, prompt, ...args) {
  const relevantBboxEditors = prompt?.output
    ? [...bboxEditors].filter((editor) => prompt.output[String(editor.node.id)])
    : [...bboxEditors];
  for (const bboxEditor of relevantBboxEditors) {
    bboxEditor.flushState();
  }
  const editorUpdates = relevantBboxEditors.map((editor) => ({
    nodeId: editor.node.id,
    inputs: { box_data: editor.boxDataWidget.value },
    serializedNode: editor.node.serialize(),
  }));
  const currentPrompt = editorUpdates.length
    ? updatePromptNodeInputs(prompt, editorUpdates)
    : prompt;
  return originalApiQueuePrompt.call(this, index, currentPrompt, ...args);
};

function descriptorUrl(descriptor) {
  if (!descriptor?.filename) return null;
  const query = new URLSearchParams({
    filename: descriptor.filename,
    subfolder: descriptor.subfolder || "",
    type: descriptor.type || "temp",
  });
  return api.apiURL(`/view?${query}`);
}

function style(target, values) {
  Object.assign(target.style, values);
  return target;
}

function element(tag, values = {}) {
  return style(document.createElement(tag), values);
}

function nextBoxOutlineColor(boxes = []) {
  const used = new Set(
    boxes.map((box) => String(box?.palette?.[0] || box?.outline_color || "").toLowerCase()).filter(Boolean),
  );
  for (const candidate of BBOX_DEFAULT_COLORS) {
    if (!used.has(candidate.toLowerCase())) return candidate;
  }
  return BBOX_DEFAULT_COLORS[boxes.length % BBOX_DEFAULT_COLORS.length];
}

export class BoundingBoxesEditor {
  constructor(node) {
    this.node = node;
    this.boxDataWidget = node.widgets?.find((widget) => widget.name === "box_data");
    if (!this.boxDataWidget) throw new Error("Bounding boxes editor box_data widget was not created.");
    this.widthWidget = node.widgets?.find((widget) => widget.name === "width");
    this.heightWidget = node.widgets?.find((widget) => widget.name === "height");
    this.data = parseBoxEditorData(this.boxDataWidget.value);
    this.selectedIndex = this.data.boxes.length ? 0 : -1;
    this.backgroundImage = null;
    this.backgroundSource = null;
    this.backgroundGeneration = 0;
    this.outputBackground = null;
    this.gesture = null;
    this.drawFrame = 0;
    this.layoutFrame = 0;
    this.layoutMinWidth = 0;
    this.layoutPanelWidth = 0;
    this.layoutMinHeight = 0;
    this.manualNodeSize = [...(node.size || [0, 0])];
    this.applyingLayoutSize = false;
    this.baseComputeSize = null;
    this.layoutSignature = "";
    this.boxListSignature = "";
    this.disposed = false;
    this.createDom();
    this.installWidget();
    this.wrapNodeLifecycle();
    this.refreshSources(true);
    this.pollTimer = window.setInterval(() => this.refreshSources(false), 500);
  }

  createDom() {
    this.root = element("div", {
      boxSizing: "border-box",
      display: "flex",
      flexDirection: "column",
      gap: "6px",
      width: "100%",
      height: "100%",
      minHeight: "0",
      padding: "4px",
      color: "var(--input-text, #ddd)",
      font: "12px sans-serif",
    });
    this.stage = element("div", {
      position: "relative",
      minHeight: "0",
      flex: "1 1 auto",
      width: "auto",
      aspectRatio: "16 / 9",
      overflow: "hidden",
      border: "1px solid rgba(255,255,255,.18)",
      borderRadius: "5px",
      background: "#17191d",
    });
    this.previewRow = element("div", {
      display: "flex",
      flexDirection: "row",
      alignItems: "stretch",
      gap: "6px",
      minWidth: "0",
      minHeight: "0",
      flex: "1 1 auto",
    });
    this.previewRow.append(this.stage);
    this.canvas = element("canvas", {
      display: "block",
      width: "100%",
      height: "100%",
      outline: "none",
      touchAction: "none",
      cursor: "crosshair",
    });
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", "Bounding boxes editor canvas");
    this.status = element("div", {
      position: "absolute",
      left: "8px",
      bottom: "7px",
      padding: "3px 6px",
      borderRadius: "4px",
      color: "#fff",
      background: "rgba(0,0,0,.65)",
      pointerEvents: "none",
      fontSize: "11px",
    });
    this.stage.append(this.canvas, this.status);
    this.activeColorBoxIndex = -1;
    this.eyedropperIndex = -1;
    this.rowSwatchSyncers = new Map();
    this.colorPicker = new PaintColorPicker({
      color: BBOX_DEFAULT_COLORS[0],
      panelHost: this.previewRow,
      onOpenChange: () => this.scheduleLayout(),
      onChange: (color) => {
        const idx = this.activeColorBoxIndex >= 0 ? this.activeColorBoxIndex : this.selectedIndex;
        if (idx < 0 || idx >= this.data.boxes.length) return;
        const rest = (this.data.boxes[idx]?.palette || []).slice(1);
        this.updateBoxField(idx, { palette: [color, ...rest], outline_color: color }, false);
        this.rowSwatchSyncers.get(idx)?.(color);
      },
      onEyedropper: () => {
        const idx = this.activeColorBoxIndex >= 0 ? this.activeColorBoxIndex : this.selectedIndex;
        if (idx < 0 || idx >= this.data.boxes.length) return;
        this.eyedropperIndex = idx;
        this.canvas.style.cursor = "crosshair";
      },
      onClear: () => {
        const idx = this.activeColorBoxIndex >= 0 ? this.activeColorBoxIndex : this.selectedIndex;
        if (idx < 0 || idx >= this.data.boxes.length) return;
        this.updateBoxField(idx, { palette: [] }, false);
        this.rowSwatchSyncers.get(idx)?.("");
      },
    });

    this.toolbar = element("div", {
      display: "flex",
      flex: "0 0 auto",
      flexDirection: "row",
      flexWrap: "nowrap",
      alignItems: "center",
      gap: "5px",
      padding: "4px 6px",
      border: "1px solid rgba(255,255,255,.16)",
      borderRadius: "4px",
      background: "rgba(0,0,0,.18)",
    });
    this.addObjButton = createStagedAction("+ Object Box", "Add a new object bounding box", () => this.addBox("obj"));
    this.addTextButton = createStagedAction("+ Text Box", "Add a new text bounding box", () => this.addBox("text"));
    this.duplicateButton = createStagedAction("Duplicate", "Duplicate the selected bounding box", () => this.duplicateSelected());
    this.deleteButton = createStagedAction("Delete", "Delete the selected bounding box", () => this.deleteSelected());
    this.clearButton = createStagedAction("Clear All", "Remove all bounding boxes", () => this.clearBoxes());
    this.toolbar.append(
      this.addObjButton,
      this.addTextButton,
      this.duplicateButton,
      this.deleteButton,
      this.clearButton,
    );

    this.boxListGroup = element("div", {
      display: "flex",
      flex: "0 0 auto",
      overflow: "hidden",
      flexDirection: "column",
      gap: "3px",
    });
    const caption = document.createElement("span");
    caption.textContent = "Boxes (back → front • cite in FLUX.3 Layout Prompt as <obj_1>, <obj_2>, …)";
    caption.style.opacity = ".78";
    this.boxList = element("div", {
      display: "flex",
      flex: "0 1 auto",
      flexDirection: "column",
      gap: "3px",
      overflowY: "auto",
      overflowX: "hidden",
      padding: "2px",
      border: "1px solid rgba(255,255,255,.16)",
      borderRadius: "4px",
      background: "rgba(0,0,0,.18)",
    });
    this.boxListGroup.append(caption, this.boxList);
    this.root.append(this.previewRow, this.toolbar, this.boxListGroup);

    for (const eventName of ["pointerdown", "pointermove", "pointerup", "click", "dblclick"]) {
      this.root.addEventListener(eventName, (event) => event.stopPropagation());
    }
    this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this.pointerEnd(event, false));
    this.canvas.addEventListener("pointercancel", (event) => this.pointerEnd(event, true));
    this.canvas.addEventListener("keydown", (event) => this.keyDown(event));
    this.resizeObserver = new ResizeObserver(() => {
      this.requestDraw();
      this.scheduleLayout();
    });
    this.resizeObserver.observe(this.stage);
    this.resizeObserver.observe(this.toolbar);
    this.resizeObserver.observe(this.boxList);
  }

  installWidget() {
    this.boxDataWidget.computeSize = () => [0, -4];
    this.boxDataWidget.draw = () => {};
    const widget = this.node.addDOMWidget("bounding_boxes_editor", "uc_bounding_boxes_editor", this.root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => this.layoutMinHeight,
    });
    this.editorWidget = widget;
    widget.serialize = false;
    this.scheduleLayout();
  }

  wrapNodeLifecycle() {
    const originalComputeSize = this.node.computeSize;
    this.baseComputeSize = (out) => originalComputeSize?.call(this.node, out) || [...(this.node.size || [0, 0])];
    this.node.computeSize = (out) => {
      const size = this.baseComputeSize(out);
      if (this.layoutMinWidth > 0) size[0] = Math.max(size[0], this.layoutMinWidth);
      return size;
    };
    const originalResize = this.node.onResize;
    this.node.onResize = (size) => {
      const resized = size || this.node.size;
      const minimum = this.node.computeSize?.() || [0, 0];
      if (resized) {
        resized[0] = Math.max(resized[0], minimum[0]);
        resized[1] = Math.max(resized[1], minimum[1]);
        if (!this.applyingLayoutSize) {
          this.manualNodeSize = [resized[0] - this.layoutPanelWidth, resized[1]];
        }
      }
      const result = originalResize?.call(this.node, resized);
      this.scheduleLayout();
      return result;
    };
    const originalExecuted = this.node.onExecuted;
    this.node.onExecuted = (output) => {
      const result = originalExecuted?.call(this.node, output);
      this.setOutput(output);
      return result;
    };
    wrapConnectionRefresh(this.node, (force) => this.refreshSources(force));
    for (const widget of [this.widthWidget, this.heightWidget]) {
      if (!widget) continue;
      const prevCallback = widget.callback;
      widget.callback = (...args) => {
        const res = prevCallback?.apply(widget, args);
        this.scheduleLayout();
        this.requestDraw();
        return res;
      };
    }
    const originalRemoved = this.node.onRemoved;
    this.node.onRemoved = (...args) => {
      this.dispose();
      return originalRemoved?.apply(this.node, args);
    };
  }

  stylePixels(target, names) {
    const computed = getComputedStyle(target);
    return names.reduce((total, name) => total + (Number.parseFloat(computed[name]) || 0), 0);
  }

  measuredFlexWidth(target) {
    if (!target) return 0;
    const computed = getComputedStyle(target);
    const gap = Number.parseFloat(computed.columnGap || computed.gap) || 0;
    const chrome = this.stylePixels(target, ["paddingLeft", "paddingRight", "borderLeftWidth", "borderRightWidth"]);
    return flexContentWidth(
      [...target.children].map((child) => Math.max(child.offsetWidth || 0, child.scrollWidth || 0)),
      gap,
      chrome,
    );
  }

  dimensions() {
    const wWidget = Number(this.widthWidget?.value) || 0;
    const hWidget = Number(this.heightWidget?.value) || 0;
    const bgWidth = this.backgroundImage?.naturalWidth || this.backgroundImage?.width || this.outputDims?.[0] || 0;
    const bgHeight = this.backgroundImage?.naturalHeight || this.backgroundImage?.height || this.outputDims?.[1] || 0;
    const width = wWidget > 0 ? wWidget : (bgWidth || 1024);
    const height = hWidget > 0 ? hWidget : (bgHeight || 1024);
    return { width: Math.max(64, width), height: Math.max(64, height) };
  }

  scheduleLayout() {
    if (this.layoutFrame || this.disposed) return;
    this.layoutFrame = requestAnimationFrame(() => {
      this.layoutFrame = 0;
      this.updateLayout();
      this.draw();
    });
  }

  updateLayout() {
    if (this.disposed || !this.root.isConnected) return;
    const allRows = [...this.boxList.children];
    const listStyle = getComputedStyle(this.boxList);
    const rowGap = Number.parseFloat(listStyle.rowGap || listStyle.gap) || 0;
    const listChrome = this.stylePixels(this.boxList, ["paddingTop", "paddingBottom", "borderTopWidth", "borderBottomWidth"]);
    this.boxList.style.maxHeight = `${visibleLayerListHeight(allRows.map((row) => row.offsetHeight || 40), rowGap, listChrome)}px`;

    const measuredRows = this.data.boxes.length ? allRows : [];
    const rootHorizontalChrome = this.stylePixels(this.root, ["paddingLeft", "paddingRight", "borderLeftWidth", "borderRightWidth"]);
    const widgetMargins = (this.editorWidget?.margin || 0) * 2;
    const computed = this.baseComputeSize?.() || this.node.computeSize?.() || this.node.size || [0, 0];
    const requiredWidth = naturalEditorWidth({
      coreWidth: Math.max(computed[0], 560),
      toolbarWidth: this.measuredFlexWidth(this.toolbar),
      rowWidths: measuredRows.map((row) => this.measuredFlexWidth(row)),
      overlayWidths: [340],
      chromeWidth: rootHorizontalChrome + widgetMargins,
    });
    const colorPicker = this.colorPicker;
    const pickerWidth = colorPicker?.isOpen() ? (colorPicker.panel.offsetWidth || 220) : 0;
    const panelGap = Number.parseFloat(getComputedStyle(this.previewRow).gap) || 0;
    const sidePanelCount = Number(pickerWidth > 0);
    const panelLayout = inlinePanelLayout(
      Math.max(requiredWidth, this.manualNodeSize[0] || 0),
      rootHorizontalChrome + widgetMargins,
      pickerWidth,
      sidePanelCount * panelGap,
    );
    const pickerExtra = panelLayout.extraWidth;
    this.layoutPanelWidth = pickerExtra;
    this.layoutMinWidth = requiredWidth + pickerExtra;
    const nodeWidth = panelLayout.nodeWidth;
    const stageWidth = panelLayout.previewWidth;
    const dims = this.dimensions();
    const stageHeight = previewHeight(stageWidth, dims.width, dims.height);
    this.previewRow.style.minHeight = `${stageHeight}px`;
    this.stage.style.height = "auto";
    this.stage.style.aspectRatio = "auto";
    colorPicker?.setPanelHeight(stageHeight);

    const rootStyle = getComputedStyle(this.root);
    const rootVerticalChrome = this.stylePixels(this.root, ["paddingTop", "paddingBottom", "borderTopWidth", "borderBottomWidth"]);
    const gap = Number.parseFloat(rootStyle.rowGap || rootStyle.gap) || 0;
    const minimumHeight = editorWidgetHeight({
      stageHeight,
      toolbarHeight: this.toolbar.offsetHeight || 36,
      layerGroupHeight: this.boxListGroup.offsetHeight || 44,
      chromeHeight: rootVerticalChrome,
      gaps: 2 * gap,
    });
    this.layoutMinHeight = minimumHeight;
    const requiredNode = this.baseComputeSize?.() || this.node.computeSize?.() || [requiredWidth, minimumHeight];
    const next = growNodeSize(this.manualNodeSize, [nodeWidth, requiredNode[1]]);
    const signature = `${next[0]}:${next[1]}:${minimumHeight}:${stageHeight}`;
    if (signature === this.layoutSignature) return;
    this.layoutSignature = signature;
    if (next[0] !== this.node.size?.[0] || next[1] !== this.node.size?.[1]) {
      this.applyingLayoutSize = true;
      try {
        this.node.setSize?.(next);
      } finally {
        this.applyingLayoutSize = false;
      }
    }
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  inputOrigin(name) {
    const slot = (this.node.inputs || []).findIndex((input) => (
      input.name === name || input.name.endsWith(`.${name}`) || input.label === name
    ));
    const linkId = slot >= 0 ? this.node.inputs?.[slot]?.link : null;
    const link = linkId != null ? this.node.graph?.links?.[linkId] : null;
    return link ? { link, node: this.node.graph?._nodes_by_id?.[link.origin_id] } : null;
  }

  loadImage(url, callback, isCurrent) {
    const image = new Image();
    image.onload = () => {
      if (!this.disposed && isCurrent()) callback(image);
    };
    image.onerror = () => this.requestDraw();
    image.src = url;
  }

  resolveBackground(force) {
    const origin = this.inputOrigin("background");
    if (!origin) {
      this.outputBackground = null;
      this.outputDims = null;
    }
    let source = null;
    let image = null;
    if (origin?.node && ![2, 4].includes(origin.node.mode)) {
      const candidate = origin.node.imgs?.[0];
      if (candidate?.complete && (candidate.naturalWidth || candidate.width)) {
        source = `element:${origin.link.origin_id}:${candidate.currentSrc || candidate.src || candidate.width}`;
        image = candidate;
      } else {
        const descriptor = latestOutputs[String(origin.link.origin_id)]?.images?.[0];
        const url = descriptorUrl(descriptor);
        if (url) source = url;
      }
    }
    if (!source && this.outputBackground) {
      source = descriptorUrl(this.outputBackground);
    }
    if (!force && source === this.backgroundSource) return;
    this.backgroundSource = source;
    if (image) {
      this.backgroundGeneration++;
      this.backgroundImage = image;
      this.scheduleLayout();
      this.requestDraw();
    } else if (source) {
      const generation = ++this.backgroundGeneration;
      this.loadImage(source, (loaded) => {
        this.backgroundImage = loaded;
        this.scheduleLayout();
        this.requestDraw();
      }, () => generation === this.backgroundGeneration);
    } else {
      this.backgroundGeneration++;
      this.backgroundImage = null;
      this.scheduleLayout();
      this.requestDraw();
    }
  }

  refreshSources(force = false) {
    if (this.disposed) return;
    this.resolveBackground(force);
    this.syncBoxList();
    this.syncToolbar();
    this.requestDraw();
  }

  upstreamUpdated(nodeId) {
    const bgOrigin = this.inputOrigin("background");
    const bboxOrigin = this.inputOrigin("bboxes");
    if (
      String(bgOrigin?.link?.origin_id) === String(nodeId)
      || String(bboxOrigin?.link?.origin_id) === String(nodeId)
    ) {
      this.backgroundSource = null;
      this.refreshSources(true);
    }
  }

  setOutput(output) {
    const payload = output?.uc_bounding_boxes_editor?.[0];
    if (!payload || payload.version !== 1) return;
    this.outputBackground = payload.background || null;
    this.outputDims = Array.isArray(payload.dims) && payload.dims.length === 2 ? payload.dims : null;
    if (Array.isArray(payload.input_regions) && (payload.upstream_changed || !this.data.boxes.length)) {
      this.data.boxes = payload.input_regions.map((item, idx) => normalizeBoxRegion(item, idx));
      this.data.last_incoming = payload.input_regions;
      this.selectedIndex = this.data.boxes.length ? 0 : -1;
      this.persistState();
      this.boxListSignature = "";
      this.syncBoxList();
      this.syncToolbar();
    }
    this.backgroundSource = null;
    this.resolveBackground(true);
    this.scheduleLayout();
    this.requestDraw();
  }

  persistState() {
    this.boxDataWidget.value = serializeBoxEditorData(this.data);
    this.boxDataWidget.callback?.(this.boxDataWidget.value, app.canvas, this.node);
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  flushState() {
    this.persistState();
  }

  syncToolbar() {
    const hasSelection = this.selectedIndex >= 0 && this.selectedIndex < this.data.boxes.length;
    if (this.duplicateButton) this.duplicateButton.disabled = !hasSelection;
    if (this.deleteButton) this.deleteButton.disabled = !hasSelection;
    if (this.clearButton) this.clearButton.disabled = this.data.boxes.length === 0;
  }

  selectBox(index) {
    const next = index >= 0 && index < this.data.boxes.length ? index : -1;
    if (this.selectedIndex === next) return;
    this.selectedIndex = next;
    this.boxListSignature = "";
    this.syncBoxList();
    this.syncToolbar();
    this.requestDraw();
  }

  addBox(type = "obj") {
    this.node.graph?.beforeChange?.();
    const index = this.data.boxes.length;
    const offset = (index % 5) * 0.04;
    const box = normalizeBoxRegion({
      x: Math.min(0.65, 0.2 + offset),
      y: Math.min(0.65, 0.2 + offset),
      w: 0.35,
      h: 0.35,
      type,
      desc: "",
      text: "",
      palette: [],
      outline_color: nextBoxOutlineColor(this.data.boxes),
      included: true,
      locked: false,
    }, index);
    this.data.boxes.push(box);
    this.selectedIndex = this.data.boxes.length - 1;
    this.persistState();
    this.node.graph?.afterChange?.();
    this.boxListSignature = "";
    this.syncBoxList();
    this.syncToolbar();
    this.scheduleLayout();
    this.requestDraw();
  }

  duplicateSelected() {
    if (this.selectedIndex < 0 || this.selectedIndex >= this.data.boxes.length) return;
    this.node.graph?.beforeChange?.();
    const source = this.data.boxes[this.selectedIndex];
    const nextIdx = this.data.boxes.length;
    const copy = normalizeBoxRegion({
      ...source,
      outline_color: source.palette?.length ? source.outline_color : nextBoxOutlineColor(this.data.boxes),
      x: Math.min(1 - source.w, source.x + 0.04),
      y: Math.min(1 - source.h, source.y + 0.04),
      locked: false,
    }, nextIdx);
    this.data.boxes.splice(this.selectedIndex + 1, 0, copy);
    this.selectedIndex += 1;
    this.persistState();
    this.node.graph?.afterChange?.();
    this.boxListSignature = "";
    this.syncBoxList();
    this.syncToolbar();
    this.scheduleLayout();
    this.requestDraw();
  }

  deleteBoxAt(index) {
    if (index < 0 || index >= this.data.boxes.length) return;
    this.node.graph?.beforeChange?.();
    this.data.boxes.splice(index, 1);
    if (this.selectedIndex >= this.data.boxes.length) {
      this.selectedIndex = this.data.boxes.length - 1;
    }
    this.persistState();
    this.node.graph?.afterChange?.();
    this.boxListSignature = "";
    this.syncBoxList();
    this.syncToolbar();
    this.scheduleLayout();
    this.requestDraw();
  }

  deleteSelected() {
    this.deleteBoxAt(this.selectedIndex);
  }

  clearBoxes() {
    if (!this.data.boxes.length) return;
    this.node.graph?.beforeChange?.();
    this.data.boxes = [];
    this.selectedIndex = -1;
    this.persistState();
    this.node.graph?.afterChange?.();
    this.boxListSignature = "";
    this.syncBoxList();
    this.syncToolbar();
    this.scheduleLayout();
    this.requestDraw();
  }

  moveBoxOrder(index, delta) {
    const target = index + delta;
    if (index < 0 || index >= this.data.boxes.length || target < 0 || target >= this.data.boxes.length) {
      return false;
    }
    this.node.graph?.beforeChange?.();
    const [moved] = this.data.boxes.splice(index, 1);
    this.data.boxes.splice(target, 0, moved);
    this.selectedIndex = target;
    this.persistState();
    this.node.graph?.afterChange?.();
    this.boxListSignature = "";
    this.syncBoxList();
    this.requestDraw();
    return true;
  }

  updateBoxField(index, patch, rebuildList = false) {
    if (index < 0 || index >= this.data.boxes.length) return;
    this.node.graph?.beforeChange?.();
    this.data.boxes[index] = normalizeBoxRegion({
      ...this.data.boxes[index],
      ...patch,
    }, index);
    this.persistState();
    this.node.graph?.afterChange?.();
    if (rebuildList) {
      this.boxListSignature = "";
      this.syncBoxList();
      this.scheduleLayout();
    }
    this.requestDraw();
  }

  syncBoxList() {
    const signature = `${this.data.boxes.map((b, i) => `${i}:${b.type}:${b.included}:${b.locked}:${b.outline_color || ""}:${(b.palette || []).join(",")}`).join("|")}::${this.selectedIndex}`;
    if (signature === this.boxListSignature) return;
    this.boxListSignature = signature;

    this.rowSwatchSyncers?.clear();
    if (!this.data.boxes.length) {
      this.colorPicker?.setOpen(false);
      const empty = element("div", {
        padding: "8px",
        opacity: ".68",
        textAlign: "center",
      });
      empty.textContent = "No bounding boxes yet — click and drag on the canvas or press + Object Box / + Text Box.";
      this.boxList.replaceChildren(empty);
      return;
    }

    this.boxList.replaceChildren(...this.data.boxes.map((box, index) => {
      const selected = index === this.selectedIndex;
      const hasCustomColor = Boolean(box.palette?.length);
      const fallbackOutline = box.outline_color || BBOX_DEFAULT_COLORS[index % BBOX_DEFAULT_COLORS.length];
      const primaryColor = box.palette?.[0] || fallbackOutline;
      const row = element("div", {
        boxSizing: "border-box",
        display: "flex",
        flex: "0 0 auto",
        flexWrap: "nowrap",
        alignItems: "center",
        gap: "4px",
        minHeight: "36px",
        padding: "2px 4px",
        border: `1px solid ${selected ? "#65c9ff" : "rgba(255,255,255,.14)"}`,
        borderRadius: "4px",
        background: selected ? "rgba(64,180,255,.16)" : "rgba(255,255,255,.04)",
        opacity: box.included === false ? ".52" : "1",
      });
      row.addEventListener("click", () => this.selectBox(index));

      const reorderGroup = element("div", { display: "flex", flex: "0 0 auto", gap: "2px" });
      const makeOrderBtn = (label, title, delta) => {
        const disabled = delta < 0 ? index === 0 : index === this.data.boxes.length - 1;
        const btn = element("button", {
          width: "24px",
          height: "26px",
          padding: "0",
          border: "1px solid rgba(255,255,255,.2)",
          borderRadius: "3px",
          color: "inherit",
          background: "rgba(0,0,0,.25)",
          cursor: disabled ? "default" : "pointer",
          opacity: disabled ? ".35" : "1",
        });
        btn.type = "button";
        btn.textContent = label;
        btn.title = title;
        btn.disabled = disabled;
        btn.addEventListener("click", (event) => {
          event.stopPropagation();
          this.moveBoxOrder(index, delta);
        });
        return btn;
      };
      reorderGroup.append(
        makeOrderBtn("▲", "Move earlier (toward background)", -1),
        makeOrderBtn("▼", "Move later (toward foreground)", 1),
      );

      const tagBadge = element("span", {
        flex: "0 0 auto",
        padding: "2px 6px",
        borderRadius: "3px",
        fontWeight: "600",
        fontSize: "11px",
        color: "#111",
        background: primaryColor,
        whiteSpace: "nowrap",
      });
      tagBadge.textContent = `<obj_${index + 1}>`;
      tagBadge.title = `Cite as <obj_${index + 1}> in FLUX.3 Layout Prompt`;

      const typeButton = createStagedAction(
        box.type === "text" ? "Text" : "Obj",
        "Toggle box type between Object ('obj') and Text ('text')",
        () => this.updateBoxField(index, { type: box.type === "text" ? "obj" : "text" }, true),
      );

      const colorInput = element("button", {
        boxSizing: "border-box",
        flex: "0 0 34px",
        width: "34px",
        height: "28px",
        padding: "3px",
        border: hasCustomColor ? "1px solid rgba(255,255,255,.55)" : "1px dashed rgba(255,255,255,.35)",
        borderRadius: "4px",
        background: "rgba(0,0,0,.25)",
        cursor: "pointer",
      });
      colorInput.type = "button";
      const colorSwatch = element("span", {
        display: "block",
        width: "100%",
        height: "100%",
        borderRadius: "2px",
        background: primaryColor,
        opacity: hasCustomColor ? "1" : "0.35",
      });
      colorInput.append(colorSwatch);
      const syncColorSwatchStyle = (activeColor) => {
        const isActive = Boolean(activeColor);
        const currentBox = this.data.boxes[index];
        const displayColor = activeColor || currentBox?.outline_color || BBOX_DEFAULT_COLORS[index % BBOX_DEFAULT_COLORS.length];
        colorSwatch.style.background = displayColor;
        colorSwatch.style.opacity = isActive ? "1" : "0.35";
        colorInput.style.border = isActive ? "1px solid rgba(255,255,255,.55)" : "1px dashed rgba(255,255,255,.35)";
        colorInput.title = isActive
          ? `Prompt color active (${activeColor}). Click to edit; right-click or Shift+click to clear.`
          : "No color sent to prompt (outline only). Click to open color picker; right-click or Shift+click to clear.";
        tagBadge.style.background = displayColor;
      };
      syncColorSwatchStyle(hasCustomColor ? box.palette[0] : "");
      this.rowSwatchSyncers.set(index, syncColorSwatchStyle);
      colorInput.addEventListener("click", (event) => {
        event.stopPropagation();
        if (event.shiftKey) {
          event.preventDefault();
          this.updateBoxField(index, { palette: [] }, false);
          syncColorSwatchStyle("");
          if (this.activeColorBoxIndex === index) this.colorPicker?.setOpen(false);
          return;
        }
        const currentBox = this.data.boxes[index];
        const currentColor = currentBox?.palette?.[0] || currentBox?.outline_color || BBOX_DEFAULT_COLORS[index % BBOX_DEFAULT_COLORS.length];
        const sameRowOpen = this.colorPicker?.isOpen() && this.activeColorBoxIndex === index;
        this.activeColorBoxIndex = index;
        this.selectBox(index);
        if (this.colorPicker) {
          this.colorPicker.trigger = colorInput;
          this.colorPicker.previous = currentColor;
          this.colorPicker.setColor(currentColor, false, false);
          this.colorPicker.setOpen(!sameRowOpen);
        }
      });
      colorInput.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        this.updateBoxField(index, { palette: [] }, false);
        syncColorSwatchStyle("");
        if (this.activeColorBoxIndex === index) this.colorPicker?.setOpen(false);
      });

      const descInput = element("input", {
        boxSizing: "border-box",
        flex: "0 0 140px",
        width: "140px",
        minWidth: "140px",
        height: "28px",
        padding: "2px 6px",
        border: "1px solid rgba(255,255,255,.2)",
        borderRadius: "4px",
        color: "inherit",
        background: "rgba(0,0,0,.28)",
      });
      descInput.type = "text";
      descInput.placeholder = "Description (e.g. a red sports car)…";
      descInput.value = box.desc || "";
      descInput.title = `Description for <obj_${index + 1}>`;
      descInput.addEventListener("click", (event) => {
        event.stopPropagation();
        this.selectBox(index);
      });
      descInput.addEventListener("change", () => {
        this.updateBoxField(index, { desc: descInput.value }, false);
      });
      descInput.addEventListener("input", () => {
        this.data.boxes[index].desc = descInput.value;
        this.persistState();
        this.requestDraw();
      });

      let textInput = null;
      if (box.type === "text") {
        textInput = element("input", {
          boxSizing: "border-box",
          flex: "0 0 100px",
          width: "100px",
          minWidth: "100px",
          height: "28px",
          padding: "2px 6px",
          border: "1px solid rgba(101,201,255,.45)",
          borderRadius: "4px",
          color: "inherit",
          background: "rgba(0,0,0,.28)",
        });
        textInput.type = "text";
        textInput.placeholder = "Rendered text…";
        textInput.value = box.text || "";
        textInput.title = "Exact text string for text-type bounding box";
        textInput.addEventListener("click", (event) => {
          event.stopPropagation();
          this.selectBox(index);
        });
        textInput.addEventListener("input", () => {
          this.data.boxes[index].text = textInput.value;
          this.persistState();
          this.requestDraw();
        });
      }

      const lockBtn = createStagedAction(
        box.locked ? "Unlock" : "Lock",
        "Lock or unlock canvas movement/resizing for this box",
        () => this.updateBoxField(index, { locked: !box.locked }, true),
      );
      const inclBtn = createStagedAction(
        box.included === false ? "Incl" : "Excl",
        "Include or exclude this box from output elements and bboxes",
        () => this.updateBoxField(index, { included: box.included === false }, true),
      );
      const removeBtn = createStagedAction(
        "✕",
        "Remove this bounding box",
        () => this.deleteBoxAt(index),
      );

      row.append(reorderGroup, tagBadge, typeButton, colorInput, descInput);
      if (textInput) row.append(textInput);
      row.append(lockBtn, inclBtn, removeBtn);
      return row;
    }));
  }

  viewFor(stageWidth, stageHeight, dims) {
    const pad = 10;
    const availW = Math.max(1, stageWidth - pad * 2);
    const availH = Math.max(1, stageHeight - pad * 2);
    const scale = Math.min(availW / dims.width, availH / dims.height);
    const width = dims.width * scale;
    const height = dims.height * scale;
    return {
      x: (stageWidth - width) / 2,
      y: (stageHeight - height) / 2,
      width,
      height,
      scale,
    };
  }

  boxCanvasRect(box, view = this.view) {
    return {
      x: view.x + box.x * view.width,
      y: view.y + box.y * view.height,
      width: box.w * view.width,
      height: box.h * view.height,
    };
  }

  handlePointsForBox(box, view = this.view) {
    const r = this.boxCanvasRect(box, view);
    const cx = r.x + r.width / 2;
    const cy = r.y + r.height / 2;
    return {
      nw: { x: r.x, y: r.y },
      n: { x: cx, y: r.y },
      ne: { x: r.x + r.width, y: r.y },
      e: { x: r.x + r.width, y: cy },
      se: { x: r.x + r.width, y: r.y + r.height },
      s: { x: cx, y: r.y + r.height },
      sw: { x: r.x, y: r.y + r.height },
      w: { x: r.x, y: cy },
    };
  }

  requestDraw() {
    if (!this.drawFrame && !this.disposed) {
      this.drawFrame = requestAnimationFrame(() => {
        this.drawFrame = 0;
        this.draw();
      });
    }
  }

  draw() {
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    if (width <= 0 || height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#17191d";
    ctx.fillRect(0, 0, width, height);

    const dims = this.dimensions();
    this.view = this.viewFor(width, height, dims);

    if (this.backgroundImage) {
      ctx.drawImage(this.backgroundImage, this.view.x, this.view.y, this.view.width, this.view.height);
      ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
      ctx.fillRect(this.view.x, this.view.y, this.view.width, this.view.height);
    } else {
      ctx.fillStyle = "#22262e";
      ctx.fillRect(this.view.x, this.view.y, this.view.width, this.view.height);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const gx = this.view.x + (this.view.width * i) / 4;
        const gy = this.view.y + (this.view.height * i) / 4;
        ctx.beginPath();
        ctx.moveTo(gx, this.view.y);
        ctx.lineTo(gx, this.view.y + this.view.height);
        ctx.moveTo(this.view.x, gy);
        ctx.lineTo(this.view.x + this.view.width, gy);
        ctx.stroke();
      }
    }

    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,.55)";
    ctx.strokeRect(this.view.x, this.view.y, this.view.width, this.view.height);

    this.data.boxes.forEach((box, index) => {
      if (box.included === false) return;
      const rect = this.boxCanvasRect(box);
      const selected = index === this.selectedIndex;
      const color = box.palette?.[0] || box.outline_color || BBOX_DEFAULT_COLORS[index % BBOX_DEFAULT_COLORS.length];

      ctx.save();
      ctx.fillStyle = selected ? `${color}33` : `${color}18`;
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
      ctx.lineWidth = selected ? 2.5 : 1.75;
      ctx.strokeStyle = color;
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);

      const tag = `${String(index + 1).padStart(2, "0")}${box.type === "text" ? " T" : ""}`;
      ctx.font = "bold 11px sans-serif";
      const tagW = Math.ceil(ctx.measureText(tag).width) + 8;
      const tagH = 16;
      ctx.fillStyle = color;
      ctx.fillRect(rect.x, rect.y, Math.min(tagW, rect.width), Math.min(tagH, rect.height));
      ctx.fillStyle = "#111";
      ctx.fillText(tag, rect.x + 4, rect.y + 12);

      const labelText = box.type === "text" && box.text
        ? `"${box.text}"${box.desc ? ` — ${box.desc}` : ""}`
        : (box.desc || "");
      if (labelText && rect.width > 24 && rect.height > 26) {
        ctx.beginPath();
        ctx.rect(rect.x + 2, rect.y + tagH + 2, Math.max(0, rect.width - 4), Math.max(0, rect.height - tagH - 4));
        ctx.clip();
        ctx.font = "12px sans-serif";
        ctx.fillStyle = "#fff";
        ctx.shadowColor = "rgba(0,0,0,0.85)";
        ctx.shadowBlur = 3;
        ctx.fillText(labelText, rect.x + 5, rect.y + tagH + 14);
      }
      ctx.restore();
    });

    if (
      this.selectedIndex >= 0
      && this.selectedIndex < this.data.boxes.length
      && this.data.boxes[this.selectedIndex].included !== false
      && !this.data.boxes[this.selectedIndex].locked
    ) {
      const handles = this.handlePointsForBox(this.data.boxes[this.selectedIndex]);
      for (const point of Object.values(handles)) {
        ctx.fillStyle = "#111";
        ctx.fillRect(point.x - 4.5, point.y - 4.5, 9, 9);
        ctx.fillStyle = "#fff";
        ctx.fillRect(point.x - 3, point.y - 3, 6, 6);
      }
    }

    const activeCount = this.data.boxes.filter((b) => b.included !== false).length;
    this.status.textContent = `${activeCount} box${activeCount === 1 ? "" : "es"} (${dims.width}×${dims.height}) • drag empty canvas to draw • Shift+drag draws over boxes`;
  }

  canvasPoint(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * (this.canvas.clientWidth / bounds.width),
      y: (event.clientY - bounds.top) * (this.canvas.clientHeight / bounds.height),
    };
  }

  normalizedPoint(canvasPoint, view = this.view) {
    return {
      x: (canvasPoint.x - view.x) / Math.max(1, view.width),
      y: (canvasPoint.y - view.y) / Math.max(1, view.height),
    };
  }

  boxAtPoint(normPoint) {
    for (let i = this.data.boxes.length - 1; i >= 0; i--) {
      const box = this.data.boxes[i];
      if (box.included === false || box.locked) continue;
      if (
        normPoint.x >= box.x
        && normPoint.x <= box.x + box.w
        && normPoint.y >= box.y
        && normPoint.y <= box.y + box.h
      ) {
        return i;
      }
    }
    return -1;
  }

  pointerDown(event) {
    if (!this.view || event.button !== 0) return;
    event.preventDefault();
    this.canvas.focus();
    const cPt = this.canvasPoint(event);
    if (this.eyedropperIndex >= 0 && this.eyedropperIndex < this.data.boxes.length) {
      const ctx = this.canvas.getContext("2d");
      const px = Math.max(0, Math.min(this.canvas.width - 1, Math.round(cPt.x)));
      const py = Math.max(0, Math.min(this.canvas.height - 1, Math.round(cPt.y)));
      const sample = ctx.getImageData(px, py, 1, 1).data;
      this.activeColorBoxIndex = this.eyedropperIndex;
      this.eyedropperIndex = -1;
      this.colorPicker?.setColor(rgbToHex(sample[0], sample[1], sample[2]), true);
      return;
    }
    const nPt = this.normalizedPoint(cPt);

    let action = null;
    let handle = null;
    let targetIndex = this.selectedIndex;

    if (
      !event.shiftKey
      && targetIndex >= 0
      && targetIndex < this.data.boxes.length
      && this.data.boxes[targetIndex].included !== false
      && !this.data.boxes[targetIndex].locked
    ) {
      const handles = this.handlePointsForBox(this.data.boxes[targetIndex]);
      for (const [name, hPt] of Object.entries(handles)) {
        if (Math.hypot(cPt.x - hPt.x, cPt.y - hPt.y) <= 10) {
          action = "resize";
          handle = name;
          break;
        }
      }
    }

    if (!action && !event.shiftKey) {
      const hit = this.boxAtPoint(nPt);
      if (hit >= 0) {
        targetIndex = hit;
        this.selectBox(hit);
        action = "move";
      }
    }

    if (!action) {
      if (nPt.x < 0 || nPt.x > 1 || nPt.y < 0 || nPt.y > 1) return;
      this.node.graph?.beforeChange?.();
      const newIndex = this.data.boxes.length;
      const newBox = {
        ...drawNormalizedBox(nPt, nPt, newIndex, "obj", 0.02),
        outline_color: nextBoxOutlineColor(this.data.boxes),
      };
      this.data.boxes.push(newBox);
      targetIndex = newIndex;
      this.selectedIndex = newIndex;
      action = "draw";
    } else {
      this.node.graph?.beforeChange?.();
    }

    this.gesture = {
      action,
      handle,
      index: targetIndex,
      startNorm: nPt,
      startBox: { ...this.data.boxes[targetIndex] },
      pointerId: event.pointerId,
      view: { ...this.view },
    };
    this.canvas.setPointerCapture(event.pointerId);
    this.requestDraw();
  }

  pointerMove(event) {
    if (!this.gesture || event.pointerId !== this.gesture.pointerId) return;
    event.preventDefault();
    const cPt = this.canvasPoint(event);
    const nPt = this.normalizedPoint(cPt, this.gesture.view);
    const dx = nPt.x - this.gesture.startNorm.x;
    const dy = nPt.y - this.gesture.startNorm.y;
    const idx = this.gesture.index;
    if (idx < 0 || idx >= this.data.boxes.length) return;

    if (this.gesture.action === "move") {
      this.data.boxes[idx] = moveNormalizedBox(this.gesture.startBox, dx, dy);
    } else if (this.gesture.action === "resize") {
      this.data.boxes[idx] = resizeNormalizedBox(this.gesture.startBox, this.gesture.handle, dx, dy);
    } else if (this.gesture.action === "draw") {
      const drawn = drawNormalizedBox(this.gesture.startNorm, nPt, idx, this.gesture.startBox.type, 0.02);
      this.data.boxes[idx] = {
        ...this.gesture.startBox,
        x: drawn.x,
        y: drawn.y,
        w: drawn.w,
        h: drawn.h,
      };
    }
    this.requestDraw();
  }

  pointerEnd(event, cancelled = false) {
    if (!this.gesture || event.pointerId !== this.gesture.pointerId) return;
    if (this.canvas.hasPointerCapture?.(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    if (cancelled && this.gesture.action === "draw") {
      this.data.boxes.splice(this.gesture.index, 1);
      this.selectedIndex = this.data.boxes.length - 1;
    }
    this.gesture = null;
    this.persistState();
    this.node.graph?.afterChange?.();
    this.boxListSignature = "";
    this.syncBoxList();
    this.syncToolbar();
    this.scheduleLayout();
    this.requestDraw();
  }

  keyDown(event) {
    if (this.selectedIndex < 0 || this.selectedIndex >= this.data.boxes.length) return;
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      event.stopPropagation();
      this.deleteSelected();
      return;
    }
    const deltas = {
      ArrowLeft: [-0.005, 0],
      ArrowRight: [0.005, 0],
      ArrowUp: [0, -0.005],
      ArrowDown: [0, 0.005],
    };
    if (!deltas[event.key]) return;
    event.preventDefault();
    event.stopPropagation();
    const box = this.data.boxes[this.selectedIndex];
    if (box.locked || box.included === false) return;
    const mult = event.shiftKey ? 5 : 1;
    const [dx, dy] = deltas[event.key];
    this.node.graph?.beforeChange?.();
    this.data.boxes[this.selectedIndex] = moveNormalizedBox(box, dx * mult, dy * mult);
    this.persistState();
    this.node.graph?.afterChange?.();
    this.requestDraw();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    clearInterval(this.pollTimer);
    this.resizeObserver?.disconnect();
    this.colorPicker?.dispose();
    if (this.layoutFrame) cancelAnimationFrame(this.layoutFrame);
    if (this.drawFrame) cancelAnimationFrame(this.drawFrame);
    bboxEditors.delete(this);
  }
}

function installBboxEditor(node) {
  if (node.__ucBoundingBoxesEditor) return node.__ucBoundingBoxesEditor;
  const editor = new BoundingBoxesEditor(node);
  node.__ucBoundingBoxesEditor = editor;
  bboxEditors.add(editor);
  const output = latestOutputs[String(node.id)];
  if (output) editor.setOutput(output);
  return editor;
}

app.registerExtension({
  name: "UtilsCollection.BoundingBoxesEditor",
  nodeCreated(node) {
    if (BBOX_NODE_TYPES.has(node.comfyClass) || BBOX_NODE_TYPES.has(node.type)) {
      installBboxEditor(node);
    }
  },
  loadedGraphNode(node) {
    if (BBOX_NODE_TYPES.has(node.comfyClass) || BBOX_NODE_TYPES.has(node.type)) {
      const editor = installBboxEditor(node);
      editor.data = parseBoxEditorData(editor.boxDataWidget.value);
      editor.selectedIndex = editor.data.boxes.length ? 0 : -1;
      editor.boxListSignature = "";
      editor.refreshSources(true);
      editor.scheduleLayout();
    }
  },
  onNodeOutputsUpdated(outputs) {
    latestOutputs = { ...latestOutputs, ...outputs };
    for (const [nodeId] of Object.entries(outputs || {})) {
      for (const editor of bboxEditors.values()) editor.upstreamUpdated(nodeId);
    }
    for (const [nodeId, output] of Object.entries(outputs || {})) {
      for (const editor of bboxEditors.values()) {
        if (String(editor.node.id) === String(nodeId)) editor.setOutput(output);
      }
    }
  },
});
