import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  DEFAULT_FACE_CORNERS,
  DEFAULT_PLACEMENT,
  DEFAULT_WORKSPACE_PADDING,
  deformQuadCentroidLocked,
  dragQuadCorner,
  frontmostLayerAtPoint,
  layerKeyCompare,
  moveRect,
  moveResolvedPlacement,
  normalizePlacement,
  PAINT_LAYER_KEY,
  normalizeWorkspacePadding,
  parsePlacementData,
  placementToRect,
  rectToPlacement,
  resolveLayerGeometry,
  resolvedWorkspaceBounds,
  resizeRectFromDelta,
  rotationFromPointer,
  serializePlacementData,
  stepPlacementValue,
} from "./placement_geometry.js";
import {
  DEFAULT_BRUSH_SETTINGS,
  normalizeBrushSettings,
  paintLayerVisible,
  PaintLayerCanvas,
} from "./paint_brush.js";
import { PaintColorPicker, rgbToHex } from "./paint_color_picker.js";
import { uploadPaintCanvas } from "./paint_persistence.js";
import { ForegroundContentEditor } from "./foreground_content_editor.js";
import { bindHoldRepeat, createStagedAction, createStagedNumericControl } from "./staged_editor_controls.js";
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
} from "./staged_editor_layout.js";
import {
  buildLayerContextActions,
  LayerContextMenu,
} from "./layer_context_menu.js";
import {
  buildEditorPromptInputs,
  buildStagingPrompt,
  updatePromptNodeInputs,
} from "./staged_queue.js";
import {
  backgroundModelAppearance,
  normalizeBackgroundModel,
  toggleBackgroundModel,
} from "./background_model_selection.js";
import { drawProjectiveImage } from "./perspective_preview.js";

const NODE_TYPES = new Set([
  "UC_LayeredBackgroundComposite",
  "UC_StagedLayeredBackgroundComposite",
  "UC_StagedMediaPipeFaceBackgroundComposite",
  "UC_StagedIndividualComposites",
]);
const PAINT_EDITOR_ENABLED = false;
const editors = new Set();
let latestOutputs = {};
let activeStagingQueue = null;
const COMPOSITE_RESIZE_METHODS = new Set(["auto", "nearest-exact", "bilinear", "area", "bicubic", "lanczos"]);

const originalApiQueuePrompt = api.queuePrompt;
api.queuePrompt = async function(index, prompt, ...args) {
  const relevantEditors = prompt?.output
    ? [...editors].filter((editor) => prompt.output[String(editor.node.id)])
    : [...editors];
  do {
    await Promise.all(relevantEditors.map((editor) => editor.flushPaint()));
  } while (relevantEditors.some((editor) => editor.foregroundContent?.hasPending()));
  const currentPrompt = updatePromptNodeInputs(prompt, relevantEditors.map((editor) => ({
    nodeId: editor.node.id,
    inputs: buildEditorPromptInputs(editor.placementWidget.value, editor.isStagedComposite()),
    serializedNode: editor.node.serialize(),
  })));
  const queuedPrompt = activeStagingQueue
    ? buildStagingPrompt(currentPrompt, activeStagingQueue.nodeId)
    : currentPrompt;
  return originalApiQueuePrompt.call(this, index, queuedPrompt, ...args);
};

async function queueStagingNode(node) {
  if (activeStagingQueue) throw new Error("Another direct staging request is already being queued.");
  activeStagingQueue = { nodeId: String(node.id) };
  try {
    return await app.queuePrompt(0);
  } finally {
    activeStagingQueue = null;
  }
}

function descriptorUrl(descriptor) {
  if (!descriptor?.filename) return null;
  const query = new URLSearchParams({
    filename: descriptor.filename,
    subfolder: descriptor.subfolder || "",
    type: descriptor.type || "temp",
  });
  return api.apiURL(`/view?${query}`);
}

function style(element, values) {
  Object.assign(element.style, values);
  return element;
}

function element(tag, values = {}) {
  return style(document.createElement(tag), values);
}

class LayeredPlacementEditor {
  constructor(node) {
    this.node = node;
    this.placementWidget = node.widgets?.find((widget) => widget.name === "placement_data");
    if (!this.placementWidget) throw new Error("Layered compositor placement_data widget was not created.");
    this.modelWidget = node.widgets?.find((widget) => widget.name === "background_removal_model_name");
    if (this.modelWidget) this.modelWidget.value = normalizeBackgroundModel(this.modelWidget.value);
    this.modelSelectionStale = false;
    this.migrateLegacyWidgetOrder();
    this.executionWidget = node.widgets?.find((widget) => widget.name === "execution_mode");
    if (typeof this.executionWidget?.value === "boolean") {
      this.executionWidget.value = this.executionWidget.value ? "run_staged" : "run_staging";
    }
    if (this.isStagedComposite() && this.executionWidget) this.executionWidget.value = "run_staged";
    this.data = parsePlacementData(this.placementWidget.value);
    this.selected = null;
    this.rotateLayer = null;
    this.backgroundImage = null;
    this.backgroundSource = null;
    this.metadata = null;
    this.metadataSignature = null;
    this.cutouts = new Map();
    this.preliminaryLayers = new Map();
    this.preliminarySources = new Map();
    this.preliminaryGenerations = new Map();
    this.backgroundGeneration = 0;
    this.metadataGeneration = 0;
    this.drawFrame = 0;
    this.commitFrame = 0;
    this.pendingPlacement = null;
    this.gesture = null;
    this.keyTransaction = false;
    this.paintLayer = PAINT_EDITOR_ENABLED && this.isStagedComposite()
      ? new PaintLayerCanvas()
      : null;
    this.paintMode = false;
    this.paintEyedropper = false;
    this.paintDirty = false;
    this.paintChangeGeneration = 0;
    this.paintSaveTimer = 0;
    this.paintSavePromise = null;
    this.paintAssetSignature = null;
    this.paintAssetGeneration = 0;
    this.paintPointerId = null;
    this.paintCursor = null;
    this.layoutFrame = 0;
    this.layoutMinWidth = 0;
    this.layoutPanelWidth = 0;
    this.layoutMinHeight = 0;
    this.manualNodeSize = [...(node.size || [0, 0])];
    this.applyingLayoutSize = false;
    this.baseComputeSize = null;
    this.layoutSignature = "";
    this.paintSettings = this.loadPaintSettings();
    this.sampleCanvas = document.createElement("canvas");
    this.layerBuffer = document.createElement("canvas");
    this.layerOutside = document.createElement("canvas");
    this.layerScratch = document.createElement("canvas");
    this.layerCoverage = document.createElement("canvas");
    this.contextMenu = new LayerContextMenu();
    this.disposed = false;
    this.createDom();
    this.installWidget();
    this.wrapNodeLifecycle();
    this.refreshSources();
    this.pollTimer = window.setInterval(() => this.refreshSources(), 500);
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
      flex: "1 0 auto",
      aspectRatio: "16 / 9",
      overflow: "hidden",
      border: "1px solid rgba(255,255,255,.18)",
      borderRadius: "5px",
      background: "#17191d",
    });
    this.previewRow = element("div", {
      display: "flex", flexDirection: "row", alignItems: "stretch", gap: "6px",
      minWidth: "0", minHeight: "0", flex: "1 1 auto",
    });
    this.previewRow.append(this.stage);
    Object.assign(this.stage.style, { flex: "1 1 auto", width: "auto" });
    this.canvas = element("canvas", {
      display: "block",
      width: "100%",
      height: "100%",
      outline: "none",
      touchAction: "none",
    });
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", "Layered foreground placement canvas");
    this.status = element("div", {
      position: "absolute",
      left: "8px",
      bottom: "7px",
      padding: "3px 6px",
      borderRadius: "4px",
      color: "#fff",
      background: "rgba(0,0,0,.65)",
      pointerEvents: "none",
    });
    this.paddingOverlay = element("div", {
      position: "absolute",
      right: "8px",
      bottom: "7px",
      display: "flex",
      alignItems: "center",
      gap: "6px",
      padding: "5px 7px",
      borderRadius: "5px",
      color: "#fff",
      background: "rgba(0,0,0,.72)",
    });
    const paddingLabel = document.createElement("span");
    paddingLabel.textContent = "Padding";
    this.paddingSlider = element("input", { width: "120px", cursor: "pointer" });
    this.paddingSlider.type = "range";
    this.paddingSlider.min = "0";
    this.paddingSlider.max = "1";
    this.paddingSlider.step = "0.05";
    this.paddingSlider.title = "Visible and permitted placement padding";
    this.paddingValueButton = element("button", {
      minWidth: "48px",
      height: "28px",
      padding: "2px 7px",
      border: "1px solid rgba(255,255,255,.25)",
      borderRadius: "4px",
      color: "inherit",
      background: "rgba(255,255,255,.08)",
      cursor: "pointer",
    });
    this.paddingValueButton.type = "button";
    this.paddingValueButton.title = "Enter an exact workspace padding value";
    this.paddingExactInput = element("input", {
      display: "none",
      width: "54px",
      height: "28px",
      boxSizing: "border-box",
      border: "1px solid #65c9ff",
      borderRadius: "4px",
      color: "inherit",
      background: "#20242a",
      textAlign: "center",
    });
    this.paddingExactInput.type = "text";
    this.paddingExactInput.inputMode = "decimal";
    this.paddingOverlay.append(paddingLabel, this.paddingSlider, this.paddingValueButton, this.paddingExactInput);

    if (this.isStagedComposite()) {
      this.modelButton = element("button", {
        position: "absolute",
        left: "8px",
        top: "8px",
        height: "30px",
        minWidth: "82px",
        padding: "3px 11px",
        borderRadius: "5px",
        color: "#fff",
        cursor: "pointer",
      });
      this.modelButton.type = "button";
      this.modelButton.title = (
        "Select the internally loaded background-removal model. A connected "
        + "background_removal_model_opt input overrides this selection."
      );
      this.stagingButton = element("button", {
        position: "absolute",
        right: "8px",
        top: "8px",
        height: "30px",
        padding: "3px 11px",
        border: "1px solid rgba(101,201,255,.8)",
        borderRadius: "5px",
        color: "#fff",
        background: "rgba(25,105,145,.88)",
        cursor: "pointer",
      });
      this.stagingButton.type = "button";
      this.stagingButton.textContent = "Run Staging";
      this.stagingButton.title = "Refresh this node's retained cutouts and placement previews";
    }
    this.stage.append(this.canvas, this.status, this.paddingOverlay);
    if (this.modelButton) this.stage.append(this.modelButton);
    if (this.stagingButton) this.stage.append(this.stagingButton);

    this.layerListGroup = element("div", {
      display: "flex",
      flex: "0 0 auto",
      overflow: "hidden",
      flexDirection: "column",
      gap: "3px",
    });
    const layerCaption = document.createElement("span");
    layerCaption.textContent = "Layer order (back → front)";
    layerCaption.style.opacity = ".78";
    this.layerList = element("div", {
      display: "flex",
      flex: "0 1 auto",
      flexDirection: "column",
      gap: "2px",
      overflowY: "auto",
      overflowX: "hidden",
      padding: "2px",
      border: "1px solid rgba(255,255,255,.16)",
      borderRadius: "4px",
      background: "rgba(0,0,0,.18)",
    });
    this.layerListGroup.append(layerCaption, this.layerList);
    this.rowControls = new Map();
    this.paintToolbar = this.paintLayer ? this.createPaintToolbar() : null;
    this.foregroundContent = this.isStagedComposite() ? new ForegroundContentEditor(this, api) : null;
    this.root.append(this.previewRow);
    if (this.paintToolbar) this.root.append(this.paintToolbar);
    if (this.foregroundContent) this.previewRow.prepend(this.foregroundContent.root);
    this.root.append(this.layerListGroup);

    this.paddingSlider.addEventListener("pointerdown", () => this.beginPaddingEdit());
    this.paddingSlider.addEventListener("input", () => this.updatePadding(this.paddingSlider.value));
    this.paddingSlider.addEventListener("change", () => this.endPaddingEdit());
    this.paddingSlider.addEventListener("pointerup", () => this.endPaddingEdit());
    this.paddingSlider.addEventListener("pointercancel", () => this.endPaddingEdit());
    this.paddingSlider.addEventListener("blur", () => this.endPaddingEdit());
    this.paddingValueButton.addEventListener("click", () => this.openPaddingEditor());
    this.paddingExactInput.addEventListener("blur", () => this.commitPaddingEditor());
    this.paddingExactInput.addEventListener("keydown", (event) => this.paddingEditorKeyDown(event));
    this.modelButton?.addEventListener("click", () => this.toggleBackgroundModel());
    this.stagingButton?.addEventListener("click", () => this.runStaging());
    for (const eventName of ["pointerdown", "pointermove", "pointerup", "click", "dblclick"]) {
      this.root.addEventListener(eventName, (event) => event.stopPropagation());
    }
    this.canvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    this.canvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    this.canvas.addEventListener("pointerup", (event) => this.pointerEnd(event, false));
    this.canvas.addEventListener("pointercancel", (event) => this.pointerEnd(event, true));
    this.canvas.addEventListener("lostpointercapture", (event) => this.foregroundContent?.end(true, event));
    this.canvas.addEventListener("pointerleave", () => {
      if (this.foregroundContent && !this.foregroundContent.pointer) {
        this.foregroundContent.cursor = null;
        this.requestDraw();
      }
    });
    this.canvas.addEventListener("contextmenu", (event) => this.openLayerContextMenu(event));
    this.canvas.addEventListener("keydown", (event) => this.keyDown(event));
    this.canvas.addEventListener("keyup", () => this.finishKeyTransaction());
    this.canvas.addEventListener("blur", () => this.finishKeyTransaction());
    this.resizeObserver = new ResizeObserver(() => {
      this.requestDraw();
      this.scheduleLayout();
    });
    this.resizeObserver.observe(this.stage);
    if (this.paintToolbar) this.resizeObserver.observe(this.paintToolbar);
    if (this.foregroundContent) this.resizeObserver.observe(this.foregroundContent.root);
    this.resizeObserver.observe(this.layerList);
    this.syncModelButton();
  }

  loadPaintSettings() {
    try {
      return normalizeBrushSettings(JSON.parse(localStorage.getItem("uc_staged_paint_brush_settings") || "{}"));
    } catch {
      return { ...DEFAULT_BRUSH_SETTINGS };
    }
  }

  savePaintSettings() {
    try { localStorage.setItem("uc_staged_paint_brush_settings", JSON.stringify(this.paintSettings)); } catch {}
  }

  createPaintToolbar() {
    const root = element("div", {
      display: "flex", flexDirection: "row", flexWrap: "nowrap", alignItems: "center", gap: "5px", padding: "5px",
      border: "1px solid rgba(255,255,255,.16)", borderRadius: "4px", background: "rgba(0,0,0,.18)",
    });
    const action = (text, title, callback) => {
      const button = element("button", { height: "28px", padding: "2px 7px", color: "inherit", cursor: "pointer", background: "rgba(0,0,0,.25)", border: "1px solid rgba(255,255,255,.25)", borderRadius: "4px" });
      button.type = "button"; button.textContent = text; button.title = title;
      button.addEventListener("click", (event) => { event.stopPropagation(); callback(); });
      return button;
    };
    this.paintModeButton = action("Paint", "Toggle exclusive paint interaction", () => this.togglePaintMode());
    this.paintShapeButton = action(this.paintSettings.shape === "circle" ? "Circle" : "Square", "Toggle circle or square brush", () => {
      this.paintSettings.shape = this.paintSettings.shape === "circle" ? "square" : "circle";
      this.paintShapeButton.textContent = this.paintSettings.shape === "circle" ? "Circle" : "Square";
      this.savePaintSettings();
    });
    this.paintEraseButton = action("Eraser", "Erase only the RGBA paint layer", () => {
      this.paintSettings.erasing = !this.paintSettings.erasing;
      this.syncPaintControls(); this.savePaintSettings();
    });
    this.paintUndoButton = action("Undo", "Undo the previous paint operation", () => this.paintHistory(-1));
    this.paintRedoButton = action("Redo", "Redo the next paint operation", () => this.paintHistory(1));
    this.paintClearButton = action("Clear", "Clear the RGBA paint layer", () => {
      if (this.paintLayer.clear()) this.paintChanged();
    });
      this.paintColorPicker = new PaintColorPicker({
        color: this.paintSettings.color,
        panelHost: this.previewRow,
        onOpenChange: () => this.scheduleLayout(),
      onChange: (color) => { this.paintSettings.color = color; this.savePaintSettings(); },
      onEyedropper: () => { this.paintEyedropper = true; if (!this.paintMode) this.togglePaintMode(true); this.syncPaintControls(); },
    });
    const numeric = (label, field, minimum, maximum, step) => {
      const wrapper = element("label", { display: "inline-flex", alignItems: "center", gap: "3px", minWidth: "0", opacity: ".9" });
      const caption = document.createElement("span"); caption.textContent = label;
      const input = element("input", { width: "46px", minWidth: "0" }); input.type = "number";
      input.min = String(minimum); input.max = String(maximum); input.step = String(step); input.value = String(this.paintSettings[field]);
      input.addEventListener("change", () => {
        this.paintSettings = normalizeBrushSettings({ ...this.paintSettings, [field]: Number(input.value) });
        input.value = String(this.paintSettings[field]); this.savePaintSettings();
      });
      wrapper.append(caption, input); return { wrapper, input };
    };
    this.paintSizeControl = numeric("Size", "size", 1, 250, 1);
    this.paintOpacityControl = numeric("Opacity", "opacity", 0, 1, 0.01);
    this.paintHardnessControl = numeric("Hard", "hardness", 0, 1, 0.01);
    this.paintSaveStatus = element("span", { minWidth: "44px", opacity: ".7" });
    root.append(
      this.paintModeButton, this.paintColorPicker.root, this.paintShapeButton,
      this.paintSizeControl.wrapper, this.paintOpacityControl.wrapper, this.paintHardnessControl.wrapper,
      this.paintEraseButton, this.paintUndoButton, this.paintRedoButton,
      this.paintClearButton, this.paintSaveStatus,
    );
    this.syncPaintControls();
    return root;
  }

  syncPaintControls() {
    if (!this.paintLayer) return;
    this.paintModeButton?.setAttribute("aria-pressed", String(this.paintMode));
    if (this.paintModeButton) this.paintModeButton.style.background = this.paintMode ? "rgba(64,180,255,.30)" : "rgba(0,0,0,.25)";
    this.paintEraseButton?.setAttribute("aria-pressed", String(this.paintSettings.erasing));
    if (this.paintEraseButton) this.paintEraseButton.style.background = this.paintSettings.erasing ? "rgba(255,165,64,.30)" : "rgba(0,0,0,.25)";
    if (this.paintUndoButton) this.paintUndoButton.disabled = !this.paintLayer.history.canUndo;
    if (this.paintRedoButton) this.paintRedoButton.disabled = !this.paintLayer.history.canRedo;
    if (this.paintSaveStatus) this.paintSaveStatus.textContent = this.paintDirty ? "Unsaved" : this.paintSavePromise ? "Saving…" : "Saved";
    this.canvas.style.cursor = this.paintEyedropper ? "crosshair" : this.paintMode ? "none" : "default";
  }

  installWidget() {
    this.placementWidget.computeSize = () => [0, -4];
    this.placementWidget.draw = () => {};
    if (this.modelWidget) {
      this.modelWidget.computeSize = () => [0, -4];
      this.modelWidget.draw = () => {};
    }
    if (this.executionWidget) {
      this.executionWidget.computeSize = () => [0, -4];
      this.executionWidget.draw = () => {};
    }
    const widget = this.node.addDOMWidget("layered_scene_editor", "uc_layered_scene_editor", this.root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => this.layoutMinHeight,
    });
    this.editorWidget = widget;
    widget.serialize = false;
    this.scheduleLayout();
  }

  scheduleLayout() {
    if (this.layoutFrame || this.disposed) return;
    this.layoutFrame = requestAnimationFrame(() => {
      this.layoutFrame = 0;
      this.updateLayout();
      // Source changes can alter both aspect and layer rows. Draw only after
      // those dimensions are committed so first connection needs no resize.
      this.draw();
    });
  }

  stylePixels(target, names) {
    const style = getComputedStyle(target);
    return names.reduce((total, name) => total + (Number.parseFloat(style[name]) || 0), 0);
  }

  measuredFlexWidth(target) {
    if (!target) return 0;
    const style = getComputedStyle(target);
    const gap = Number.parseFloat(style.columnGap || style.gap) || 0;
    const chrome = this.stylePixels(target, ["paddingLeft", "paddingRight", "borderLeftWidth", "borderRightWidth"]);
    return flexContentWidth(
      [...target.children].map((child) => Math.max(child.offsetWidth || 0, child.scrollWidth || 0)),
      gap,
      chrome,
    );
  }

  updateLayerListHeight() {
    const rows = [...this.layerList.children];
    const style = getComputedStyle(this.layerList);
    const gap = Number.parseFloat(style.rowGap || style.gap) || 0;
    const chrome = this.stylePixels(this.layerList, ["paddingTop", "paddingBottom", "borderTopWidth", "borderBottomWidth"]);
    this.layerList.style.maxHeight = `${visibleLayerListHeight(rows.map((row) => row.offsetHeight), gap, chrome)}px`;
  }

  stageOverlayWidths() {
    const inset = (target, property) => target ? (Number.parseFloat(getComputedStyle(target)[property]) || 0) : 0;
    const gap = Number.parseFloat(getComputedStyle(this.root).gap) || 0;
    const stageChrome = this.stylePixels(this.stage, ["borderLeftWidth", "borderRightWidth"]);
    const top = (this.modelButton?.offsetWidth || 0) + (this.stagingButton?.offsetWidth || 0)
      + inset(this.modelButton, "left") + inset(this.stagingButton, "right") + gap + stageChrome;
    const bottom = (this.status?.offsetWidth || 0) + (this.paddingOverlay?.scrollWidth || 0)
      + inset(this.status, "left") + inset(this.paddingOverlay, "right") + gap + stageChrome;
    return [top, bottom];
  }

  updateLayout() {
    if (this.disposed || !this.root.isConnected) return;
    this.updateLayerListHeight();
    const rootStyle = getComputedStyle(this.root);
    const rootHorizontalChrome = this.stylePixels(this.root, ["paddingLeft", "paddingRight", "borderLeftWidth", "borderRightWidth"]);
    const widgetMargins = (this.editorWidget?.margin || 0) * 2;
    const computed = this.baseComputeSize?.() || this.node.computeSize?.() || this.node.size || [0, 0];
    const requiredWidth = naturalEditorWidth({
      coreWidth: computed[0],
      toolbarWidth: this.measuredFlexWidth(this.paintToolbar),
      rowWidths: [...this.layerList.children].map((row) => this.measuredFlexWidth(row)),
      overlayWidths: this.stageOverlayWidths(),
      chromeWidth: rootHorizontalChrome + widgetMargins,
    });
    // Foreground color controls are embedded below the tools in the left panel.
    // Only the legacy standalone picker needs additional horizontal space.
    const colorPicker = this.paintColorPicker;
    const pickerWidth = colorPicker?.isOpen() ? (colorPicker.panel.offsetWidth || 220) : 0;
    const controlsWidth = this.foregroundContent ? (this.foregroundContent.root.offsetWidth || 220) : 0;
    const panelGap = Number.parseFloat(getComputedStyle(this.previewRow).gap) || 0;
    const sidePanelCount = Number(pickerWidth > 0) + Number(controlsWidth > 0);
    const panelLayout = inlinePanelLayout(
      Math.max(requiredWidth, this.manualNodeSize[0] || 0),
      rootHorizontalChrome + widgetMargins,
      pickerWidth + controlsWidth,
      sidePanelCount * panelGap,
    );
    const pickerExtra = panelLayout.extraWidth;
    this.layoutPanelWidth = pickerExtra;
    this.layoutMinWidth = requiredWidth + pickerExtra;
    const nodeWidth = panelLayout.nodeWidth;
    const stageWidth = panelLayout.previewWidth;
    const dimensions = this.dimensions();
    const stageHeight = previewHeight(stageWidth, dimensions?.width, dimensions?.height);
    this.previewRow.style.minHeight = `${stageHeight}px`;
    this.stage.style.height = "auto";
    this.stage.style.aspectRatio = "auto";
    colorPicker?.setPanelHeight(stageHeight);
    if (this.foregroundContent) this.foregroundContent.root.style.height = `${stageHeight}px`;

    const rootVerticalChrome = this.stylePixels(this.root, ["paddingTop", "paddingBottom", "borderTopWidth", "borderBottomWidth"]);
    const gap = Number.parseFloat(rootStyle.rowGap || rootStyle.gap) || 0;
    const sectionCount = 2 + (this.paintToolbar ? 1 : 0);
    const minimumHeight = editorWidgetHeight({
      stageHeight,
      toolbarHeight: this.paintToolbar?.offsetHeight || 0,
      layerGroupHeight: this.layerListGroup.offsetHeight,
      chromeHeight: rootVerticalChrome,
      gaps: Math.max(0, sectionCount - 1) * gap,
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
    const originalRemoved = this.node.onRemoved;
    this.node.onRemoved = (...args) => {
      this.dispose();
      return originalRemoved?.apply(this.node, args);
    };
  }

  connectedLayers() {
    const direct = (this.node.inputs || [])
      .filter((input) => /foreground_\d+$/.test(input.name) && input.link != null)
      .map((input) => input.name.match(/foreground_\d+$/)[0]);
    const detectedFaces = (this.metadata?.layers || [])
      .filter((layer) => layer.is_face)
      .map((layer) => layer.socket);
    if (direct.length) return this.orderLayers([...direct, ...detectedFaces]);
    return this.orderLayers((this.metadata?.layers || []).map((layer) => layer.socket));
  }

  compositionLayers() {
    const foregrounds = this.connectedLayers();
    if (!this.paintLayer) return foregrounds;
    const available = [...foregrounds, PAINT_LAYER_KEY];
    const availableSet = new Set(available);
    const requested = (this.data.layer_order || []).filter((key) => availableSet.has(key));
    for (const key of foregrounds) if (!requested.includes(key)) requested.push(key);
    if (!requested.includes(PAINT_LAYER_KEY)) requested.push(PAINT_LAYER_KEY);
    return requested;
  }

  persistPlacement() {
    this.placementWidget.value = serializePlacementData(this.data);
    this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    this.node.graph?.setDirtyCanvas?.(true, true);
  }

  ensurePaintOrdered() {
    if (!this.paintLayer || this.data.layer_order.includes(PAINT_LAYER_KEY)) return;
    this.data.layer_order = this.compositionLayers();
    this.persistPlacement();
  }

  togglePaintMode(force) {
    if (!this.paintLayer) return;
    const enabling = force === true || (force !== false && !this.paintMode);
    if (enabling) {
      const dimensions = this.dimensions();
      if (!dimensions) {
        this.status.textContent = "A resolved background preview is required before painting";
        this.status.hidden = false;
        return;
      }
      this.syncPaintCanvas();
      this.ensurePaintOrdered();
      this.paintMode = true;
      this.selected = PAINT_LAYER_KEY;
      this.rotateLayer = null;
      this.warpLayer = null;
    } else {
      if (this.paintPointerId != null) {
        const pointerId = this.paintPointerId;
        const changed = this.paintLayer.end(false);
        if (this.canvas.hasPointerCapture?.(pointerId)) this.canvas.releasePointerCapture(pointerId);
        this.paintPointerId = null;
        if (changed) this.paintChanged();
      }
      this.paintMode = false;
      this.paintEyedropper = false;
      void this.flushPaint().catch(() => {});
    }
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
    this.syncPaintControls();
    this.requestDraw();
  }

  paintHistory(direction) {
    const changed = direction < 0 ? this.paintLayer.undo() : this.paintLayer.redo();
    if (changed) this.paintChanged();
  }

  paintChanged() {
    this.ensurePaintOrdered();
    this.paintDirty = true;
    this.paintChangeGeneration++;
    if (this.paintSaveTimer) clearTimeout(this.paintSaveTimer);
    this.paintSaveTimer = window.setTimeout(() => {
      this.paintSaveTimer = 0;
      void this.flushPaint().catch(() => {});
    }, 450);
    this.syncPaintControls();
    this.requestDraw();
  }

  async flushPaint() {
    this.flushPlacement();
    await this.foregroundContent?.flush();
    if (this.paintLayer && this.paintPointerId != null) {
      const pointerId = this.paintPointerId;
      const changed = this.paintLayer.end(false);
      if (this.canvas.hasPointerCapture?.(pointerId)) this.canvas.releasePointerCapture(pointerId);
      this.paintPointerId = null;
      if (changed) this.paintChanged();
    }
    if (!this.paintLayer || !this.paintDirty) return this.paintSavePromise;
    if (this.paintSaveTimer) { clearTimeout(this.paintSaveTimer); this.paintSaveTimer = 0; }
    if (this.paintSavePromise) await this.paintSavePromise;
    if (!this.paintDirty) return;
    this.paintSaveStatus.textContent = "Saving…";
    const savingGeneration = this.paintChangeGeneration;
    this.paintSavePromise = uploadPaintCanvas(
      api, this.paintLayer.canvas, this.data.paint_layer, this.node.id,
    ).then((paint) => {
      this.data.paint_layer = paint;
      this.paintAssetSignature = JSON.stringify(paint.asset);
      if (savingGeneration === this.paintChangeGeneration) this.paintDirty = false;
      this.persistPlacement();
      this.paintSaveStatus.textContent = "Saved";
    }).catch((error) => {
      this.paintDirty = true;
      this.paintSaveStatus.textContent = "Save failed";
      this.status.textContent = error?.message || String(error);
      this.status.hidden = false;
      throw error;
    }).finally(() => {
      this.paintSavePromise = null;
      this.syncPaintControls();
    });
    return this.paintSavePromise;
  }

  syncPaintCanvas() {
    if (!this.paintLayer) return;
    const dimensions = this.dimensions();
    if (!dimensions) return;
    const asset = this.data.paint_layer?.asset;
    const signature = asset?.filename ? JSON.stringify(asset) : null;
    if (signature && signature !== this.paintAssetSignature) {
      this.paintAssetSignature = signature;
      const generation = ++this.paintAssetGeneration;
      const url = descriptorUrl(asset);
      this.loadImage(url, (image) => {
        if (generation !== this.paintAssetGeneration) return;
        this.paintLayer.load(image, dimensions.width, dimensions.height);
        this.syncPaintControls();
        this.requestDraw();
      }, () => generation === this.paintAssetGeneration);
      return;
    }
    const resized = this.paintLayer.resize(dimensions.width, dimensions.height);
    if (resized && signature) this.paintChanged();
  }

  orderLayers(keys) {
    const available = [...new Set(keys)].sort(layerKeyCompare);
    const availableSet = new Set(available);
    const ordered = (this.data.layer_order || []).filter((key) => availableSet.has(key));
    ordered.push(...available.filter((key) => !ordered.includes(key)));
    return ordered;
  }

  isStagedComposite() {
    return [
      "UC_StagedLayeredBackgroundComposite",
      "UC_StagedMediaPipeFaceBackgroundComposite",
      "UC_StagedIndividualComposites",
    ]
      .includes(this.node.comfyClass || this.node.type);
  }

  usesRetainedStage() {
    return this.node.widgets?.find((widget) => widget.name === "execution_mode")?.value === "run_staged";
  }

  inputOrigin(name) {
    const slot = (this.node.inputs || []).findIndex((input) => (
      input.name === name || input.name.endsWith(`.${name}`) || input.label === name
    ));
    const linkId = slot >= 0 ? this.node.inputs?.[slot]?.link : null;
    const link = linkId != null ? this.node.graph?.links?.[linkId] : null;
    return link ? { link, node: this.node.graph?._nodes_by_id?.[link.origin_id] } : null;
  }

  semanticSignature() {
    // Layer order is placement state, not source state. Keep this list canonical so
    // reordering back-to-front never invalidates retained cutouts.
    const sourceNames = ["background", ...this.connectedLayers().slice().sort(layerKeyCompare)];
    const links = sourceNames.map((name) => {
      const origin = this.inputOrigin(name);
      const preview = origin?.node?.imgs?.[0];
      const previewIdentity = preview?.currentSrc || preview?.src || "";
      return `${name}:${origin?.link?.id ?? origin?.link?.origin_id ?? "-"}:${origin?.node?.mode ?? "-"}:${previewIdentity}`;
    });
    const maskValues = ["mask_threshold", "border_cleanup_width", "artifact_cleanup_radius", "gap_fill_radius", "feather_radius"]
      .map((name) => `${name}:${this.node.widgets?.find((widget) => widget.name === name)?.value}`);
    return [...links, ...maskValues].join("|");
  }

  refreshSources(force = false) {
    if (this.disposed) return;
    this.foregroundContent?.sync();
    const foregroundLayers = this.connectedLayers();
    const layers = this.compositionLayers();
    if (this.foregroundContent?.active && !foregroundLayers.includes(this.foregroundContent.active.key)) this.foregroundContent.exit();
    if (!this.selected || !layers.includes(this.selected)) this.selected = foregroundLayers[0] || (this.paintLayer ? PAINT_LAYER_KEY : null);
    this.syncLayerList(layers);
    if (this.lastLayerCount !== layers.length) {
      this.lastLayerCount = layers.length;
      this.scheduleLayout();
    }
    const signature = this.semanticSignature();
    if (this.metadataSignature && signature !== this.metadataSignature && !this.isStagedComposite()) {
        this.metadata = null;
        this.metadataSignature = null;
      this.cutouts.clear();
      this.metadataGeneration++;
    }
    this.resolveBackground(force);
    this.resolvePreliminaryLayers(foregroundLayers, force);
    this.syncPaintCanvas();
    this.syncNumericControls();
    this.requestDraw();
  }

  syncLayerList(layers) {
    const signature = `${layers.join("|")}::${this.selected || ""}`;
    if (signature === this.layerListSignature) return;
    this.layerListSignature = signature;
    this.rowControls.clear();
    this.layerList.replaceChildren(...layers.map((key, index) => {
      const selected = key === this.selected;
      const row = element("div", {
        display: "flex",
        flexWrap: "nowrap",
        alignItems: "center",
        gap: "3px",
        minHeight: "40px",
        padding: "2px 3px",
        border: `1px solid ${selected ? "#65c9ff" : "rgba(255,255,255,.14)"}`,
        borderRadius: "3px",
        background: selected ? "rgba(64,180,255,.18)" : "rgba(255,255,255,.04)",
        cursor: "pointer",
      });
      row.dataset.layer = key;
      const grip = element("span", {
        alignSelf: "stretch",
        display: "flex",
        flex: "0 0 28px",
        alignItems: "center",
        justifyContent: "center",
        color: "rgba(255,255,255,.65)",
        cursor: "grab",
        fontSize: "20px",
        lineHeight: "20px",
        borderRight: "1px solid rgba(255,255,255,.12)",
      });
      grip.textContent = "⠿";
      grip.title = "Drag to change stacking order";
      grip.draggable = true;
      const reorderButtons = element("div", {
        display: "flex",
        flex: "0 0 67px",
        flexDirection: "row",
        gap: "3px",
      });
      const reorderButton = (text, title, delta) => {
        const disabled = delta < 0 ? index === 0 : index === layers.length - 1;
        const button = element("button", {
          width: "32px",
          height: "30px",
          padding: "0",
          border: "1px solid rgba(255,255,255,.2)",
          borderRadius: "3px",
          color: "inherit",
          background: "rgba(0,0,0,.25)",
          cursor: disabled ? "default" : "pointer",
          fontSize: "16px",
          lineHeight: "1",
          opacity: disabled ? ".35" : "1",
        });
        button.type = "button";
        button.textContent = text;
        button.title = title;
        button.disabled = disabled;
        bindHoldRepeat(button, () => this.moveLayerBy(key, delta), {
          captureTarget: this.layerList,
          enabled: () => {
            const current = this.compositionLayers();
            const position = current.indexOf(key);
            return !this.disposed && position >= 0 && position + delta >= 0 && position + delta < current.length;
          },
        });
        return button;
      };
      reorderButtons.append(
        reorderButton("▲", "Move one layer toward the back", -1),
        reorderButton("▼", "Move one layer toward the front", 1),
      );
      const name = element("span", {
        flex: "0 0 48px",
        width: "48px",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
      });
      name.textContent = key === PAINT_LAYER_KEY ? "Paint" : key.replace(/^foreground_/, "").replace("_face_", "_f_");
      name.title = key;
      const position = element("span", { flex: "0 0 auto", opacity: ".65", fontSize: "11px" });
      position.textContent = index === 0 ? "back" : index === layers.length - 1 ? "front" : String(index + 1);
      const controls = element("div", {
        display: "flex",
        flex: "0 0 auto",
        flexWrap: "nowrap",
        gap: "3px",
        alignItems: "center",
        overflow: "visible",
      });
      const numericInputs = {};
      const transformButtons = [];
      if (key === PAINT_LAYER_KEY) {
        const paint = createStagedAction("Paint", "Toggle exclusive paint interaction", () => this.togglePaintMode());
        const include = createStagedAction("Incl", "Include or exclude the paint layer", () => this.togglePaintIncluded());
        controls.append(paint, include);
        this.rowControls.set(key, { inputs: numericInputs, include, paint, row });
      } else {
        for (const [field, label, tooltip] of [
          ["scale", "Scale", "Foreground size relative to the background"],
          ["center_x", "X", "Horizontal center divided by background width"],
          ["center_y", "Y", "Vertical center divided by background height"],
        ]) {
          const numeric = this.createLayerNumericControl(key, field, label, tooltip);
          numericInputs[field] = numeric.input;
          transformButtons.push(numeric.decrement, numeric.increment);
          controls.append(numeric.root);
        }
        const flip = createStagedAction("Flip H", "Mirror this foreground horizontally", () => this.toggleHorizontalFlip(key));
        const flipVertical = createStagedAction("Flip V", "Mirror this foreground vertically", () => this.toggleVerticalFlip(key));
        const reset = createStagedAction("Reset", "Reset this foreground placement", () => this.resetLayer(key));
        transformButtons.push(flip, flipVertical, reset);
        controls.append(flip, flipVertical, reset);
        const rotation = this.createLayerNumericControl(key, "rotation", "Rot°", "Layer rotation in degrees");
        const toggleRotation = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (this.paintMode
            || this.layerPlacement(key).locked === true
            || this.layerPlacement(key).included === false) return;
          this.selectLayer(key);
          this.toggleRotate(key);
        };
        Object.assign(rotation.caption.style, {
          padding: "2px",
          borderRadius: "3px",
          cursor: "pointer",
          userSelect: "none",
        });
        rotation.caption.title = "Toggle canvas rotation mode for this layer";
        rotation.caption.setAttribute("role", "button");
        rotation.caption.setAttribute("aria-label", rotation.caption.title);
        rotation.caption.setAttribute("aria-pressed", "false");
        rotation.caption.tabIndex = 0;
        rotation.caption.addEventListener("click", toggleRotation);
        rotation.caption.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") toggleRotation(event);
        });
        for (const button of [rotation.decrement, rotation.increment]) {
          Object.assign(button.style, { flex: "0 0 20px", width: "20px", fontSize: "13px" });
        }
        Object.assign(rotation.input.style, { flex: "0 0 48px", minWidth: "48px", width: "48px" });
        rotation.decrement.title = "Rotate left one degree";
        rotation.increment.title = "Rotate right one degree";
        numericInputs.rotation = rotation.input;
        transformButtons.push(rotation.decrement, rotation.increment);
        const warp = createStagedAction("Warp", "Toggle four-corner warp editing", () => this.toggleWarp(key));
        transformButtons.push(warp);
        const include = createStagedAction("Incl", "Include or exclude this layer", () => this.toggleLayerIncluded(key));
        const lock = createStagedAction("🔓", "Lock this layer against transforms", () => this.toggleLayerLocked(key));
        Object.assign(lock.style, { flex: "0 0 28px", width: "28px", padding: "0" });
        controls.append(rotation.root, warp, include, lock);
        this.rowControls.set(key, {
          inputs: numericInputs,
          transformButtons,
          flip,
          flipVertical,
          reset,
          include,
          lock,
          warp,
          rotate: rotation.caption,
          row,
        });
      }
      row.append(grip, reorderButtons, name, position, controls);
      row.addEventListener("click", () => this.selectLayer(key));
      grip.addEventListener("dragstart", (event) => {
        this.draggedLayer = key;
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", key);
        row.style.opacity = ".45";
      });
      row.addEventListener("dragover", (event) => {
        if (!this.draggedLayer || this.draggedLayer === key) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        const bounds = row.getBoundingClientRect();
        const edge = event.clientY > bounds.top + bounds.height / 2 ? "inset 0 -2px #65c9ff" : "inset 0 2px #65c9ff";
        row.style.boxShadow = edge;
      });
      row.addEventListener("dragleave", () => { row.style.boxShadow = "none"; });
      row.addEventListener("drop", (event) => {
        event.preventDefault();
        row.style.boxShadow = "none";
        if (!this.draggedLayer || this.draggedLayer === key) return;
        const bounds = row.getBoundingClientRect();
        this.dropLayer(this.draggedLayer, key, event.clientY > bounds.top + bounds.height / 2);
      });
      grip.addEventListener("dragend", () => {
        this.draggedLayer = null;
        row.style.opacity = "1";
      });
      return row;
    }));
    this.syncNumericControls();
    this.scheduleLayout();
  }

  syncModelButton() {
    if (!this.modelButton) return;
    const appearance = backgroundModelAppearance(this.modelWidget?.value);
    this.modelButton.textContent = appearance.text;
    this.modelButton.style.border = `1px solid ${appearance.border}`;
    this.modelButton.style.background = appearance.background;
  }

  toggleBackgroundModel() {
    if (!this.modelWidget) return;
    this.node.graph?.beforeChange?.();
    this.modelWidget.value = toggleBackgroundModel(this.modelWidget.value);
    this.modelWidget.callback?.(this.modelWidget.value, app.canvas, this.node);
    this.modelSelectionStale = true;
    this.node.graph?.setDirtyCanvas?.(true, true);
    this.node.graph?.afterChange?.();
    this.syncModelButton();
    this.requestDraw();
  }

  createLayerNumericControl(key, field, label, tooltip) {
    const step = field === "rotation" ? 1 : 0.002;
    const { root, caption, input, decrement, increment } = createStagedNumericControl(label, tooltip, `${key} ${label}`, step);
    input.dataset.field = field;
    input.setAttribute("aria-label", `${key} ${label}`);
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("change", () => this.numericChanged(key, field, input.value));
    input.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Enter") {
        event.preventDefault();
        input.blur();
      } else if (event.key === "Escape") {
        event.preventDefault();
        const dimensions = this.dimensions();
        const placement = dimensions
          ? rectToPlacement(dimensions.width, dimensions.height, this.rectFor(key, dimensions), this.layerPlacement(key))
          : DEFAULT_PLACEMENT;
        input.value = Number(placement[field]).toFixed(4);
        input.blur();
      }
    });
    for (const [button, direction] of [[decrement, -1], [increment, 1]]) {
      bindHoldRepeat(button, () => field === "rotation"
        ? this.stepLayerRotation(key, direction) : this.stepLayerValue(key, field, direction * step), {
        captureTarget: this.layerList,
        enabled: () => !this.disposed && !this.paintMode && !this.layerPlacement(key).locked && this.compositionLayers().includes(key),
      });
    }
    root.addEventListener("click", (event) => event.stopPropagation());
    return { root, caption, input, decrement, increment };
  }

  resolveBackground(force) {
    const origin = this.inputOrigin("background");
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
    if (!source && this.metadata && this.metadataSignature === this.semanticSignature()) {
      source = descriptorUrl(this.metadata.background?.preview);
    }
    if (!force && source === this.backgroundSource) return;
    this.backgroundSource = source;
    if (image) {
      this.backgroundGeneration++;
      this.backgroundImage = image;
      this.syncPaintCanvas();
      this.scheduleLayout();
      this.requestDraw();
    } else if (source) {
      const generation = ++this.backgroundGeneration;
      this.loadImage(source, (loaded) => {
        this.backgroundImage = loaded;
        this.syncPaintCanvas();
        this.scheduleLayout();
        this.requestDraw();
      }, () => generation === this.backgroundGeneration);
    } else {
      this.backgroundGeneration++;
      this.backgroundImage = null;
      this.scheduleLayout();
    }
  }

  namedWidget(name) {
    return this.node.widgets?.find((widget) => widget.name === name);
  }

  migrateLegacyWidgetOrder() {
    const placement = this.placementWidget;
    const maskThreshold = this.namedWidget("mask_threshold");
    const borderCleanup = this.namedWidget("border_cleanup_width");
    const artifactCleanup = this.namedWidget("artifact_cleanup_radius");
    const gapFill = this.namedWidget("gap_fill_radius");
    const feather = this.namedWidget("feather_radius");
    const imageResize = this.namedWidget("image_resize_method");
    const maskResize = this.namedWidget("mask_resize_method");
    if (!placement || !maskThreshold || !borderCleanup || !artifactCleanup || !gapFill || !feather || !imageResize || !maskResize) return false;

    const isPlacementJson = (value) => typeof value === "string" && value.trim().startsWith("{");
    const isResizeMethod = (value) => COMPOSITE_RESIZE_METHODS.has(String(value));
    if (this.isStagedComposite()) {
      // Old staged order: ... gap, placement, feather, image resize, mask resize.
      if (!isPlacementJson(feather.value) || !isResizeMethod(placement.value)) return false;
      const oldPlacement = feather.value;
      const oldFeather = imageResize.value;
      const oldImageResize = maskResize.value;
      const oldMaskResize = placement.value;
      placement.value = oldPlacement;
      feather.value = oldFeather;
      imageResize.value = oldImageResize;
      maskResize.value = oldMaskResize;
      return true;
    } else {
      // Old layered order: placement preceded all cleanup and resize widgets.
      if (!isPlacementJson(maskThreshold.value) || !isResizeMethod(placement.value)) return false;
      const oldPlacement = maskThreshold.value;
      const oldMaskThreshold = borderCleanup.value;
      const oldBorderCleanup = artifactCleanup.value;
      const oldArtifactCleanup = gapFill.value;
      const oldGapFill = feather.value;
      const oldFeather = imageResize.value;
      const oldImageResize = maskResize.value;
      const oldMaskResize = placement.value;
      placement.value = oldPlacement;
      maskThreshold.value = oldMaskThreshold;
      borderCleanup.value = oldBorderCleanup;
      artifactCleanup.value = oldArtifactCleanup;
      gapFill.value = oldGapFill;
      feather.value = oldFeather;
      imageResize.value = oldImageResize;
      maskResize.value = oldMaskResize;
      return true;
    }
  }

  resolvePreliminaryLayers(layers, force) {
    const available = new Set(layers);
    for (const key of this.preliminaryLayers.keys()) {
      if (!available.has(key)) this.preliminaryLayers.delete(key);
    }
    for (const key of this.preliminarySources.keys()) {
      if (!available.has(key)) this.preliminarySources.delete(key);
    }
    for (const key of layers) {
      const origin = this.inputOrigin(key);
      let source = null;
      let image = null;
      if (origin?.node && ![2, 4].includes(origin.node.mode)) {
        const candidate = origin.node.imgs?.[0];
        if (candidate?.complete && (candidate.naturalWidth || candidate.width)) {
          source = `element:${origin.link.origin_id}:${candidate.currentSrc || candidate.src || candidate.width}`;
          image = candidate;
        } else {
          source = descriptorUrl(latestOutputs[String(origin.link.origin_id)]?.images?.[0]);
        }
      }
      if (!force && source === this.preliminarySources.get(key)) continue;
      this.preliminarySources.set(key, source);
      const generation = (this.preliminaryGenerations.get(key) || 0) + 1;
      this.preliminaryGenerations.set(key, generation);
      if (image) {
        this.preliminaryLayers.set(key, image);
      } else if (source) {
        this.loadImage(source, (loaded) => {
          this.preliminaryLayers.set(key, loaded);
          this.syncNumericControls();
          this.requestDraw();
        }, () => generation === this.preliminaryGenerations.get(key));
      } else {
        this.preliminaryLayers.delete(key);
      }
    }
  }

  loadImage(url, callback, isCurrent) {
    const image = new Image();
    image.onload = () => {
      if (!this.disposed && isCurrent()) callback(image);
    };
    image.onerror = () => this.requestDraw();
    image.src = url;
  }

  setOutput(output) {
    const metadata = output?.uc_layered_scene_editor?.[0];
    if (!metadata || metadata.version !== 1) return;
    this.metadata = metadata;
    this.metadataSignature = this.semanticSignature();
    if (metadata.stage_mode !== "retained") this.modelSelectionStale = false;
    this.cutouts.clear();
    const generation = ++this.metadataGeneration;
    for (const layer of metadata.layers || []) {
      const url = descriptorUrl(layer.preview);
      if (!url) continue;
      this.loadImage(url, (image) => {
        this.cutouts.set(layer.socket, image);
        this.requestDraw();
      }, () => generation === this.metadataGeneration);
    }
    this.backgroundSource = null;
    this.resolveBackground(true);
    this.scheduleLayout();
    this.requestDraw();
  }

  upstreamUpdated(nodeId) {
    const sourceNames = ["background", ...this.connectedLayers()];
    const origins = sourceNames.map((name) => this.inputOrigin(name));
    if (!origins.some((origin) => String(origin?.link?.origin_id) === String(nodeId))) return;
    if (!this.isStagedComposite() && sourceNames.slice(1).some((name) => String(this.inputOrigin(name)?.link?.origin_id) === String(nodeId))) {
      this.metadata = null;
      this.metadataSignature = null;
      this.cutouts.clear();
      this.metadataGeneration++;
    }
    this.backgroundSource = null;
    this.refreshSources(true);
  }

  layerMetadata(key) {
    return this.metadata?.layers?.find((layer) => layer.socket === key);
  }

  layerAspect(key) {
    const metadata = this.layerMetadata(key);
    if (metadata?.crop_width > 0 && metadata?.crop_height > 0) {
      return metadata.crop_width / metadata.crop_height;
    }
    const preliminary = this.preliminaryLayers.get(key);
    const width = preliminary?.naturalWidth || preliminary?.width;
    const height = preliminary?.naturalHeight || preliminary?.height;
    return width > 0 && height > 0 ? width / height : 1;
  }

  layerPlacement(key) {
    const face = key.includes("_face_");
    const fallback = face
      ? { ...DEFAULT_PLACEMENT, scale: 0.25, rotation: 0, corners: [[-1, -1], [1, -1], [1, 1], [-1, 1]], included: true }
      : this.data.version === 1
      ? { scale: DEFAULT_PLACEMENT.scale, long_axis_shift: 0, short_axis_shift: 0 }
      : DEFAULT_PLACEMENT;
    return normalizePlacement(this.data.layers[key] || fallback, face ? 3 : this.data.version, face);
  }

  dimensions() {
    const exact = this.metadataSignature === this.semanticSignature() ? this.metadata?.background : null;
    const width = exact?.width || this.backgroundImage?.naturalWidth || this.backgroundImage?.width;
    const height = exact?.height || this.backgroundImage?.naturalHeight || this.backgroundImage?.height;
    return width > 0 && height > 0 ? { width, height } : null;
  }

  viewFor(width, height, dimensions, layers) {
    const paddingLevel = normalizeWorkspacePadding(this.data.workspace_padding);
    const bounds = resolvedWorkspaceBounds(
      dimensions.width,
      dimensions.height,
      paddingLevel,
      layers.filter((key) => key !== PAINT_LAYER_KEY)
        .map((key) => this.resolvedGeometry(key, dimensions).frame),
    );
    let { left, top, right, bottom } = bounds;
    const outerPadding = Math.max(right - left, bottom - top) * 0.04 * paddingLevel;
    left -= outerPadding;
    top -= outerPadding;
    right += outerPadding;
    bottom += outerPadding;
    const padding = 8 * paddingLevel;
    const scale = Math.min((width - padding * 2) / (right - left), (height - padding * 2) / (bottom - top));
    return {
      x: (width - (right - left) * scale) / 2 - left * scale,
      y: (height - (bottom - top) * scale) / 2 - top * scale,
      width: dimensions.width * scale,
      height: dimensions.height * scale,
      scale,
    };
  }

  requestDraw() {
    if (!this.drawFrame) this.drawFrame = requestAnimationFrame(() => {
      this.drawFrame = 0;
      this.draw();
    });
  }

  draw() {
    this.foregroundContent?.sync();
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    if (width <= 0 || height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
      this.canvas.width = Math.round(width * dpr);
      this.canvas.height = Math.round(height * dpr);
    }
    const context = this.canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#17191d";
    context.fillRect(0, 0, width, height);
    const dimensions = this.dimensions();
    const foregroundLayers = this.connectedLayers();
    const layers = this.compositionLayers();
    if (!dimensions || !this.backgroundImage) {
      this.view = null;
      this.status.textContent = this.inputOrigin("background")
        ? "Background preview unavailable"
        : "Connect a background image";
      this.status.hidden = false;
      return;
    }
    const visibleLayers = layers.filter((key) => key === PAINT_LAYER_KEY
      ? paintLayerVisible(this.data.paint_layer?.included, this.paintMode)
      : this.layerPlacement(key).included !== false);
    this.view = this.gesture?.view
      || this.viewFor(width, height, dimensions, visibleLayers);
    context.globalAlpha = BACKGROUND_PREVIEW_ALPHA;
    context.drawImage(this.backgroundImage, this.view.x, this.view.y, this.view.width, this.view.height);
    context.globalAlpha = 1;
    this.drawLayers(context, visibleLayers, dimensions, width, height, dpr);
    this.updateSampleCanvas(visibleLayers, dimensions, width, height, dpr);
    context.lineWidth = 2;
    context.strokeStyle = "rgba(255,255,255,.65)";
    context.strokeRect(this.view.x, this.view.y, this.view.width, this.view.height);
    if (
      this.selected && this.selected !== PAINT_LAYER_KEY && !this.paintMode && !this.foregroundContent?.active
      && this.layerPlacement(this.selected).locked !== true
      && this.layerPlacement(this.selected).included !== false
    ) {
      this.drawHandles(context, this.selected, dimensions);
    }
    if (this.paintMode && this.paintCursor) this.drawPaintCursor(context);
    this.foregroundContent?.drawCursor(context);
    const pending = foregroundLayers.filter((key) => !this.layerMetadata(key)).length;
    if (this.isStagedComposite() && this.modelSelectionStale) {
      this.status.textContent = "Removal model changed • next queue rebuilds stage • Run Staging refreshes previews now";
    } else if (this.isStagedComposite() && !layers.length) {
      this.status.textContent = this.usesRetainedStage()
        ? "No retained stage • next queue builds it • Run Staging refreshes previews now"
        : "Run Staging refreshes foreground previews";
    } else if (pending) {
      this.status.textContent = `${pending} foreground${pending === 1 ? "" : "s"} pending removal pass`;
    } else {
      const stage = this.isStagedComposite()
        ? ` • ${this.metadata?.stage_mode === "retained" ? "retained stage" : this.metadata?.stage_mode === "full_run" ? "full run" : "fresh stage"}`
        : "";
      this.status.textContent = `${layers.length} layer${layers.length === 1 ? "" : "s"} • back → front${stage}`;
    }
    this.status.hidden = false;
    if (this.foregroundContent?.error) this.status.textContent = this.foregroundContent.error;
  }

  rectFor(key, dimensions) {
    if (key === PAINT_LAYER_KEY) {
      return { x: 0, y: 0, width: dimensions.width, height: dimensions.height };
    }
    return placementToRect(dimensions.width, dimensions.height, this.layerAspect(key), this.layerPlacement(key));
  }

  layerSourceSize(key) {
    const metadata = this.layerMetadata(key);
    if (metadata?.crop_width > 0 && metadata?.crop_height > 0) {
      return { width: metadata.crop_width, height: metadata.crop_height };
    }
    const preview = this.cutouts.get(key) || this.preliminaryLayers.get(key);
    return {
      width: preview?.naturalWidth || preview?.width || 1,
      height: preview?.naturalHeight || preview?.height || 1,
    };
  }

  resolvedGeometry(key, dimensions, placement = this.layerPlacement(key)) {
    const source = this.layerSourceSize(key);
    return resolveLayerGeometry({
      backgroundWidth: dimensions.width,
      backgroundHeight: dimensions.height,
      sourceWidth: source.width,
      sourceHeight: source.height,
      placement,
      workspacePadding: this.data.workspace_padding,
    });
  }

  updateSampleCanvas(layers, dimensions, width, height, dpr) {
    const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
    if (this.sampleCanvas.width !== pixelWidth || this.sampleCanvas.height !== pixelHeight) {
      this.sampleCanvas.width = pixelWidth;
      this.sampleCanvas.height = pixelHeight;
    }
    const context = this.sampleCanvas.getContext("2d", { willReadFrequently: true });
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.drawImage(this.backgroundImage, this.view.x, this.view.y, this.view.width, this.view.height);
    this.drawLayers(context, layers, dimensions, width, height, dpr, false);
  }

  drawPaintCursor(context) {
    const radius = this.paintSettings.size * this.view.scale;
    context.save();
    context.beginPath();
    if (this.paintSettings.shape === "square") {
      context.rect(this.paintCursor.x - radius, this.paintCursor.y - radius, radius * 2, radius * 2);
    } else {
      context.arc(this.paintCursor.x, this.paintCursor.y, radius, 0, Math.PI * 2);
    }
    context.lineWidth = 1;
    context.strokeStyle = "rgba(0,0,0,.9)";
    context.stroke();
    context.setLineDash([3, 3]);
    context.strokeStyle = "rgba(255,255,255,.95)";
    context.stroke();
    context.restore();
  }

  toCanvasRect(rect) {
    return {
      x: this.view.x + rect.x * this.view.scale,
      y: this.view.y + rect.y * this.view.scale,
      width: rect.width * this.view.scale,
      height: rect.height * this.view.scale,
    };
  }

  drawLayers(context, layers, dimensions, width, height, dpr, showOverlays = true) {
    const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
    for (const canvas of [this.layerBuffer, this.layerOutside, this.layerScratch, this.layerCoverage]) {
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
    }
    const coverage = this.layerCoverage.getContext("2d");
    coverage.setTransform(1, 0, 0, 1, 0, 0);
    coverage.clearRect(0, 0, pixelWidth, pixelHeight);
    for (const key of layers) {
      const layer = this.layerBuffer.getContext("2d");
      layer.setTransform(1, 0, 0, 1, 0, 0);
      layer.clearRect(0, 0, pixelWidth, pixelHeight);
      layer.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.drawLayer(layer, key, dimensions, "content");

      const blendFactor = Math.max(0, Math.min(1, Number(this.layerMetadata(key)?.blend_factor ?? 1)));
      const scratch = this.layerScratch.getContext("2d");
      scratch.setTransform(1, 0, 0, 1, 0, 0);
      scratch.clearRect(0, 0, pixelWidth, pixelHeight);
      scratch.globalCompositeOperation = "source-over";
      scratch.globalAlpha = blendFactor;
      scratch.drawImage(this.layerBuffer, 0, 0);
      if (blendFactor < 1) {
        const outside = this.layerOutside.getContext("2d");
        outside.setTransform(1, 0, 0, 1, 0, 0);
        outside.clearRect(0, 0, pixelWidth, pixelHeight);
        outside.globalCompositeOperation = "source-over";
        outside.globalAlpha = 1;
        outside.drawImage(this.layerBuffer, 0, 0);
        outside.globalCompositeOperation = "destination-out";
        outside.drawImage(this.layerCoverage, 0, 0);
        outside.globalCompositeOperation = "source-over";
        scratch.globalCompositeOperation = "lighter";
        scratch.globalAlpha = 1 - blendFactor;
        scratch.drawImage(this.layerOutside, 0, 0);
      }
      context.save();
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.globalAlpha = 1;
      context.drawImage(this.layerScratch, 0, 0);
      context.restore();
      coverage.globalCompositeOperation = "source-over";
      coverage.drawImage(this.layerBuffer, 0, 0);
      if (showOverlays) this.drawLayer(context, key, dimensions, "overlay");
    }
  }

  drawLayer(context, key, dimensions, mode = "all") {
    const rect = this.toCanvasRect(this.rectFor(key, dimensions));
    if (key === PAINT_LAYER_KEY) {
      if (mode !== "overlay" && paintLayerVisible(this.data.paint_layer?.included, this.paintMode)) {
        this.paintLayer.draw(context, rect.x, rect.y, rect.width, rect.height);
      }
      return;
    }
    const sourcePreview = this.cutouts.get(key) || this.preliminaryLayers.get(key);
    const preview = mode !== "overlay" ? (this.foregroundContent?.preview(key, sourcePreview) || sourcePreview) : sourcePreview;
    const placement = this.layerPlacement(key);
    const geometry = this.resolvedGeometry(key, dimensions, placement);
    const destinationPoints = geometry.points.map((point) => [
      this.view.x + point.x * this.view.scale,
      this.view.y + point.y * this.view.scale,
    ]);
    const destinationFrame = this.toCanvasRect(geometry.frame);
    const defaultCorners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    const warped = Array.isArray(placement.corners) && placement.corners.some(
      (point, index) => point.some((value, axis) => value !== defaultCorners[index][axis]),
    );
    const transformed = warped || Number(placement.rotation || 0) !== 0;
    context.save();
    if (mode !== "overlay" && preview && placement.included !== false) {
      const desiredFlip = placement.flip_horizontal === true;
      const desiredFlipVertical = placement.flip_vertical === true;
      const previewFlip = this.layerMetadata(key)?.flip_horizontal === true;
      const previewFlipVertical = this.layerMetadata(key)?.flip_vertical === true;
      const flipHorizontal = desiredFlip !== previewFlip;
      const flipVertical = desiredFlipVertical !== previewFlipVertical;
      if (transformed) {
        drawProjectiveImage(context, preview, destinationPoints, flipHorizontal, flipVertical);
      } else if (flipHorizontal || flipVertical) {
        context.save();
        context.translate(
          destinationFrame.x + (flipHorizontal ? destinationFrame.width : 0),
          destinationFrame.y + (flipVertical ? destinationFrame.height : 0),
        );
        context.scale(flipHorizontal ? -1 : 1, flipVertical ? -1 : 1);
        context.drawImage(preview, 0, 0, destinationFrame.width, destinationFrame.height);
        context.restore();
      } else {
        context.drawImage(
          preview, destinationFrame.x, destinationFrame.y,
          destinationFrame.width, destinationFrame.height,
        );
      }
    }
    if (mode === "content") {
      context.restore();
      return;
    }
    const outline = transformed ? destinationPoints : [
      [destinationFrame.x, destinationFrame.y],
      [destinationFrame.x + destinationFrame.width, destinationFrame.y],
      [destinationFrame.x + destinationFrame.width, destinationFrame.y + destinationFrame.height],
      [destinationFrame.x, destinationFrame.y + destinationFrame.height],
    ];
    context.beginPath();
    context.moveTo(...outline[0]);
    for (const point of outline.slice(1)) context.lineTo(...point);
    context.closePath();
    context.fillStyle = key === this.selected ? "rgba(64,180,255,.16)" : "rgba(255,255,255,.055)";
    if (this.foregroundContent?.active?.key !== key) context.fill();
    context.save();
    context.setLineDash(this.layerMetadata(key) ? [] : [6, 4]);
    context.lineWidth = key === this.selected ? 4 : 3;
    context.strokeStyle = "rgba(0,0,0,.8)";
    context.stroke();
    context.lineWidth = key === this.selected ? 2 : 1;
    context.strokeStyle = key === this.selected ? "#65c9ff" : "rgba(255,255,255,.88)";
    context.stroke();
    context.restore();
    context.restore();
  }

  handlePoints(key, dimensions) {
    const [nw, ne, se, sw] = this.layerQuadPoints(key, dimensions);
    return { nw, ne, sw, se };
  }

  layerQuadPoints(key, dimensions) {
    const geometry = this.resolvedGeometry(key, dimensions);
    if (geometry.transformed.identity) {
      const frame = this.toCanvasRect(geometry.frame);
      return [
        { x: frame.x, y: frame.y },
        { x: frame.x + frame.width, y: frame.y },
        { x: frame.x + frame.width, y: frame.y + frame.height },
        { x: frame.x, y: frame.y + frame.height },
      ];
    }
    return geometry.points.map((point) => ({
      x: this.view.x + point.x * this.view.scale,
      y: this.view.y + point.y * this.view.scale,
    }));
  }

  faceQuadPoints(key, dimensions) {
    return this.layerQuadPoints(key, dimensions);
  }

  layerAtCanvasPoint(canvasPoint, dimensions) {
    return frontmostLayerAtPoint(
      this.connectedLayers(),
      canvasPoint,
      (key) => this.layerQuadPoints(key, dimensions),
      (key) => this.layerPlacement(key).included !== false
        && this.layerPlacement(key).locked !== true,
    );
  }

  openLayerContextMenu(event) {
    event.preventDefault();
    event.stopPropagation();
    this.contextMenu.close();
    if (this.paintMode) return;
    const dimensions = this.dimensions();
    if (!this.view || !dimensions) return;
    const key = this.layerAtCanvasPoint(this.canvasPoint(event), dimensions) || this.foregroundContent?.active?.key;
    if (!key) return;
    this.selectLayer(key);
    this.contextMenu.open(event.clientX, event.clientY, this.layerContextActions(key));
  }

  layerContextActions(key) {
    const layers = this.compositionLayers();
    const index = layers.indexOf(key);
    if (index < 0) return [];
    return buildLayerContextActions({
      index,
      count: layers.length,
      placement: this.layerPlacement(key),
      warpActive: this.warpLayer === key,
      rotateActive: this.rotateLayer === key,
      moveBack: () => this.moveLayerBy(key, -1),
      moveForward: () => this.moveLayerBy(key, 1),
      sendToBack: () => this.moveLayerToEdge(key, false),
      bringToFront: () => this.moveLayerToEdge(key, true),
      flipHorizontal: () => this.toggleHorizontalFlip(key),
      flipVertical: () => this.toggleVerticalFlip(key),
      toggleWarp: () => this.toggleWarp(key),
      toggleRotate: () => this.toggleRotate(key),
      toggleLock: () => this.toggleLayerLocked(key),
      exclude: () => this.toggleLayerIncluded(key),
      reset: () => this.resetLayer(key),
      contentActions: this.foregroundContent?.menuActions(key) || [],
    });
  }

  drawHandles(context, key, dimensions) {
    const points = this.warpLayer === key
      ? this.faceQuadPoints(key, dimensions)
      : Object.values(this.handlePoints(key, dimensions));
    if (this.warpLayer === key) {
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (const point of points.slice(1)) context.lineTo(point.x, point.y);
      context.closePath();
      context.strokeStyle = "#ff9bd2";
      context.lineWidth = 2;
      context.stroke();
    }
    for (const point of points) {
      context.fillStyle = "#111";
      context.fillRect(point.x - 5, point.y - 5, 10, 10);
      context.fillStyle = "#fff";
      context.fillRect(point.x - 3, point.y - 3, 6, 6);
    }
  }

  canvasPoint(event) {
    const bounds = this.canvas.getBoundingClientRect();
    return {
      x: (event.clientX - bounds.left) * (this.canvas.clientWidth / bounds.width),
      y: (event.clientY - bounds.top) * (this.canvas.clientHeight / bounds.height),
    };
  }

  backgroundPoint(canvasPoint, view = this.view) {
    return { x: (canvasPoint.x - view.x) / view.scale, y: (canvasPoint.y - view.y) / view.scale };
  }

  pointerDown(event) {
    if (this.foregroundContent?.down(event)) return;
    const dimensions = this.dimensions();
    if (this.paintMode) {
      if (!this.view || !dimensions || event.button !== 0) return;
      event.preventDefault();
      this.canvas.focus();
      const canvasPoint = this.canvasPoint(event);
      const point = this.backgroundPoint(canvasPoint);
      const inside = point.x >= 0 && point.x <= dimensions.width
        && point.y >= 0 && point.y <= dimensions.height;
      if (!inside) return;
      if (this.paintEyedropper) {
        const dpr = window.devicePixelRatio || 1;
        const sampleX = Math.max(0, Math.min(this.sampleCanvas.width - 1, Math.floor(canvasPoint.x * dpr)));
        const sampleY = Math.max(0, Math.min(this.sampleCanvas.height - 1, Math.floor(canvasPoint.y * dpr)));
        const sample = this.sampleCanvas.getContext("2d", { willReadFrequently: true })
          .getImageData(sampleX, sampleY, 1, 1).data;
        this.paintSettings.color = rgbToHex(sample[0], sample[1], sample[2]);
        this.paintColorPicker.setColor(this.paintSettings.color, true);
        this.paintEyedropper = false;
        this.savePaintSettings();
        this.syncPaintControls();
        return;
      }
      if (this.paintLayer.begin(point, this.paintSettings)) {
        this.paintPointerId = event.pointerId;
        this.canvas.setPointerCapture(event.pointerId);
        this.requestDraw();
      }
      return;
    }
    const layers = this.connectedLayers().filter(
      (key) => this.layerPlacement(key).included !== false,
    );
    if (!this.view || !dimensions || !layers.length || event.button !== 0) return;
    event.preventDefault();
    this.canvas.focus();
    const canvasPoint = this.canvasPoint(event);
    const backgroundPoint = this.backgroundPoint(canvasPoint);
    let action = null;
    let handle = null;
    if (this.selected && this.layerPlacement(this.selected).locked !== true && this.warpLayer === this.selected) {
      const cornerIndex = this.faceQuadPoints(this.selected, dimensions).findIndex((point) => (
        Math.hypot(canvasPoint.x - point.x, canvasPoint.y - point.y) <= 12
      ));
      if (cornerIndex >= 0) {
        action = "warp_corner";
        handle = cornerIndex;
      }
    }
    if (this.selected && this.layerPlacement(this.selected).locked !== true && this.warpLayer !== this.selected) {
      for (const [name, point] of Object.entries(this.handlePoints(this.selected, dimensions))) {
        if (Math.hypot(canvasPoint.x - point.x, canvasPoint.y - point.y) <= 12) {
          action = this.rotateLayer === this.selected ? "rotate" : "resize";
          handle = name;
          break;
        }
      }
    }
    if (!action) {
      const key = this.layerAtCanvasPoint(canvasPoint, dimensions);
      if (key) {
        this.selectLayer(key);
        action = this.warpLayer === key ? "warp_deform" : "move";
      }
    }
    if (!action) return;
    this.flushPlacement();
    this.node.graph?.beforeChange?.();
    this.gesture = {
      action,
      handle,
      key: this.selected,
      startCanvas: canvasPoint,
      startPoint: backgroundPoint,
      startRect: this.rectFor(this.selected, dimensions),
      startGeometry: this.resolvedGeometry(this.selected, dimensions),
      originalValue: this.placementWidget.value,
      pointerId: event.pointerId,
      changed: false,
      view: { ...this.view },
      startPlacement: this.layerPlacement(this.selected),
    };
    this.canvas.setPointerCapture(event.pointerId);
  }

  pointerMove(event) {
    if (this.foregroundContent?.move(event)) return;
    if (this.paintMode) {
      if (!this.view) return;
      const cursorPoint = this.canvasPoint(event);
      this.paintCursor = cursorPoint;
      if (this.paintPointerId === event.pointerId) {
        event.preventDefault();
        const events = event.getCoalescedEvents?.() || [event];
        this.paintLayer.move(events.map((sample) => this.backgroundPoint(this.canvasPoint(sample))));
      }
      this.requestDraw();
      return;
    }
    if (!this.gesture || !this.view) return;
    event.preventDefault();
    const dimensions = this.dimensions();
    const canvasPoint = this.canvasPoint(event);
    const point = this.backgroundPoint(canvasPoint, this.gesture.view);
    const deltaX = point.x - this.gesture.startPoint.x;
    const deltaY = point.y - this.gesture.startPoint.y;
    let rect;
    if (this.gesture.action === "rotate") {
      const start = this.gesture.startPlacement;
      const canvasRect = this.toCanvasRect(this.gesture.startGeometry.frame);
      const center = {
        x: canvasRect.x + canvasRect.width / 2,
        y: canvasRect.y + canvasRect.height / 2,
      };
      const rotation = rotationFromPointer(
        Number(start.rotation || 0),
        center,
        this.gesture.startCanvas,
        canvasPoint,
      );
      this.gesture.changed = true;
      this.queuePlacement(this.gesture.key, { ...start, rotation });
      return;
    } else if (this.gesture.action === "warp_corner" || this.gesture.action === "warp_deform") {
      const start = this.gesture.startPlacement;
      const geometry = this.gesture.startGeometry;
      const localPoint = point;
      const angle = -Number(start.rotation || 0) * Math.PI / 180;
      const cosine = Math.cos(angle), sine = Math.sin(angle);
      const rotatedX = localPoint.x - geometry.offset.x + geometry.transformed.minimum.x;
      const rotatedY = localPoint.y - geometry.offset.y + geometry.transformed.minimum.y;
      const sourceX = rotatedX * cosine - rotatedY * sine;
      const sourceY = rotatedX * sine + rotatedY * cosine;
      const halfWidth = Math.max(geometry.source.width - 1, 1) / 2;
      const halfHeight = Math.max(geometry.source.height - 1, 1) / 2;
      const localX = sourceX / halfWidth;
      const localY = sourceY / halfHeight;
      let corners;
      if (this.gesture.action === "warp_corner") {
        corners = dragQuadCorner(start.corners, this.gesture.handle, localX, localY);
      } else {
        const dx = (deltaX * cosine - deltaY * sine) / halfWidth;
        const dy = (deltaX * sine + deltaY * cosine) / halfHeight;
        corners = deformQuadCentroidLocked(start.corners, [localX - dx, localY - dy], dx, dy);
      }
      this.gesture.changed = true;
      this.queuePlacement(this.gesture.key, { ...start, corners });
      return;
    } else if (this.gesture.action === "move") {
      this.gesture.changed = true;
      this.queuePlacement(this.gesture.key, moveResolvedPlacement(
        dimensions.width, dimensions.height, this.gesture.startGeometry,
        this.gesture.startPlacement, deltaX, deltaY, this.data.workspace_padding,
      ));
      return;
    } else if (this.gesture.action === "resize") {
      const shortest = Math.min(dimensions.width, dimensions.height);
      const angle = -Number(this.gesture.startPlacement.rotation || 0) * Math.PI / 180;
      const cosine = Math.cos(angle), sine = Math.sin(angle);
      rect = resizeRectFromDelta(
        this.gesture.startRect,
        this.gesture.handle,
        deltaX * cosine - deltaY * sine,
        deltaX * sine + deltaY * cosine,
        this.layerAspect(this.gesture.key),
        shortest * 0.05,
        shortest * 10,
      );
    } else return;
    this.gesture.changed = true;
    this.queuePlacement(this.gesture.key, rectToPlacement(
      dimensions.width,
      dimensions.height,
      rect,
      this.layerPlacement(this.gesture.key),
    ));
  }

  pointerEnd(event, cancelled) {
    if (this.foregroundContent?.end(cancelled, event)) return;
    if (this.paintPointerId != null && event.pointerId === this.paintPointerId) {
      const changed = this.paintLayer.end(cancelled);
      if (this.canvas.hasPointerCapture?.(this.paintPointerId)) {
        this.canvas.releasePointerCapture(this.paintPointerId);
      }
      this.paintPointerId = null;
      if (changed) this.paintChanged();
      else this.requestDraw();
      return;
    }
    if (!this.gesture) return;
    if (cancelled || (this.gesture.action === "draw" && !this.gesture.changed)) {
      this.pendingPlacement = null;
      if (this.commitFrame) cancelAnimationFrame(this.commitFrame);
      this.commitFrame = 0;
      this.placementWidget.value = this.gesture.originalValue;
      this.data = parsePlacementData(this.gesture.originalValue);
      this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    } else {
      this.flushPlacement();
    }
    const pointerId = Number.isInteger(event.pointerId) ? event.pointerId : this.gesture.pointerId;
    if (Number.isInteger(pointerId) && this.canvas.hasPointerCapture?.(pointerId)) {
      this.canvas.releasePointerCapture(pointerId);
    }
    this.gesture = null;
    this.node.graph?.afterChange?.();
    this.syncNumericControls();
    this.requestDraw();
  }

  queuePlacement(key, placement) {
    this.pendingPlacement = { key, placement };
    if (!this.commitFrame) this.commitFrame = requestAnimationFrame(() => this.flushPlacement());
  }

  flushPlacement() {
    if (this.commitFrame) cancelAnimationFrame(this.commitFrame);
    this.commitFrame = 0;
    if (!this.pendingPlacement) return;
    const { key, placement } = this.pendingPlacement;
    this.pendingPlacement = null;
    if (this.data.version === 1) {
      const dimensions = this.dimensions();
      if (dimensions) {
        const migrated = {};
        for (const layerKey of Object.keys(this.data.layers)) {
          const rect = placementToRect(
            dimensions.width, dimensions.height, this.layerAspect(layerKey),
            normalizePlacement(this.data.layers[layerKey], 1),
          );
          migrated[layerKey] = rectToPlacement(
            dimensions.width, dimensions.height, rect, this.data.layers[layerKey]
          );
        }
        this.data.layers = migrated;
      }
    }
    const hasTransform = placement.rotation !== undefined || placement.corners !== undefined || placement.included !== undefined;
    this.data.version = key.includes("_face_") || hasTransform || this.data.version === 3 ? 3 : 2;
    this.data.layers[key] = normalizePlacement(placement, this.data.version, key.includes("_face_"));
    this.placementWidget.value = serializePlacementData(this.data);
    this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    this.node.graph?.setDirtyCanvas?.(true, true);
    this.syncNumericControls();
    this.requestDraw();
  }

  selectLayer(key) {
    if (!this.compositionLayers().includes(key)) return;
    if (this.foregroundContent?.active && this.foregroundContent.active.key !== key) this.foregroundContent.exit();
    if (this.paintMode && key !== PAINT_LAYER_KEY) return;
    this.selected = key;
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
    this.syncNumericControls();
    this.requestDraw();
  }

  syncNumericControls() {
    const dimensions = this.dimensions();
    for (const [key, controls] of this.rowControls.entries()) {
      if (key === PAINT_LAYER_KEY) {
        const included = this.data.paint_layer?.included !== false;
        controls.include.textContent = included ? "Excl" : "Incl";
        controls.paint.setAttribute("aria-pressed", String(this.paintMode));
        controls.paint.style.background = this.paintMode ? "rgba(64,180,255,.30)" : "rgba(0,0,0,.25)";
        controls.row.style.opacity = included ? "1" : ".48";
        continue;
      }
      const placement = dimensions
        ? rectToPlacement(
            dimensions.width, dimensions.height, this.rectFor(key, dimensions),
            this.layerPlacement(key),
          )
        : DEFAULT_PLACEMENT;
      for (const [field, input] of Object.entries(controls.inputs)) {
        const value = field === "rotation" ? this.layerPlacement(key).rotation ?? 0 : placement[field];
        if (document.activeElement !== input) input.value = Number(value).toFixed(field === "rotation" ? 0 : 4);
      }
      controls.flip.setAttribute("aria-pressed", String(placement.flip_horizontal === true));
      controls.flip.style.background = placement.flip_horizontal
        ? "rgba(64,180,255,.30)"
        : "rgba(0,0,0,.25)";
      controls.flipVertical.setAttribute("aria-pressed", String(placement.flip_vertical === true));
      controls.flipVertical.style.background = placement.flip_vertical
        ? "rgba(64,180,255,.30)"
        : "rgba(0,0,0,.25)";
      if (controls.include) {
        const included = this.layerPlacement(key).included !== false;
        controls.include.textContent = included ? "Excl" : "Incl";
        controls.row.style.opacity = included ? "1" : ".48";
      }
      const locked = this.layerPlacement(key).locked === true;
      if (controls.lock) {
        controls.lock.textContent = locked ? "🔒" : "🔓";
        controls.lock.title = locked ? "Unlock this layer for transforms" : "Lock this layer against transforms";
        controls.lock.setAttribute("aria-label", controls.lock.title);
        controls.lock.setAttribute("aria-pressed", String(locked));
        controls.lock.style.background = locked ? "rgba(255,165,64,.30)" : "rgba(0,0,0,.25)";
      }
      if (controls.warp) {
        controls.warp.setAttribute("aria-pressed", String(this.warpLayer === key));
        controls.warp.style.background = this.warpLayer === key ? "rgba(64,180,255,.30)" : "rgba(0,0,0,.25)";
      }
      if (controls.rotate) {
        const active = this.rotateLayer === key;
        const disabled = this.paintMode || locked || this.layerPlacement(key).included === false;
        controls.rotate.setAttribute("aria-pressed", String(active));
        controls.rotate.setAttribute("aria-disabled", String(disabled));
        controls.rotate.tabIndex = disabled ? -1 : 0;
        controls.rotate.style.background = active ? "rgba(64,180,255,.30)" : "transparent";
        controls.rotate.style.cursor = disabled ? "default" : "pointer";
        controls.rotate.style.opacity = disabled ? ".55" : ".78";
      }
      for (const input of Object.values(controls.inputs)) input.disabled = this.paintMode || locked;
      for (const control of controls.transformButtons || []) control.disabled = this.paintMode || locked;
    }
    const padding = normalizeWorkspacePadding(
      this.data.workspace_padding ?? DEFAULT_WORKSPACE_PADDING,
    );
    this.paddingSlider.value = String(padding);
    this.paddingValueButton.textContent = padding.toFixed(2);
  }

  dropLayer(dragged, target, insertAfter) {
    const layers = this.compositionLayers();
    const draggedIndex = layers.indexOf(dragged);
    if (draggedIndex < 0) return;
    layers.splice(draggedIndex, 1);
    let targetIndex = layers.indexOf(target);
    if (targetIndex < 0) return;
    if (insertAfter) targetIndex += 1;
    layers.splice(targetIndex, 0, dragged);
    this.node.graph?.beforeChange?.();
    this.data.layer_order = layers;
    this.placementWidget.value = serializePlacementData(this.data);
    this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    this.node.graph?.setDirtyCanvas?.(true, true);
    this.node.graph?.afterChange?.();
    this.selected = this.paintMode ? PAINT_LAYER_KEY : dragged;
    this.layerListSignature = null;
    this.syncLayerList(layers);
    this.syncNumericControls();
    this.rowControls.get(dragged)?.row.scrollIntoView?.({ block: "nearest" });
    this.requestDraw();
  }

  moveLayerBy(key, delta) {
    const layers = this.compositionLayers();
    const index = layers.indexOf(key);
    const targetIndex = index + delta;
    if (index < 0 || targetIndex < 0 || targetIndex >= layers.length) return;
    this.dropLayer(key, layers[targetIndex], delta > 0);
  }

  moveLayerToEdge(key, front) {
    const layers = this.compositionLayers();
    const index = layers.indexOf(key);
    if (index < 0) return;
    const targetIndex = front ? layers.length - 1 : 0;
    if (index === targetIndex) return;
    this.dropLayer(key, layers[targetIndex], front);
  }

  beginPaddingEdit() {
    if (this.paddingEditActive) return;
    this.paddingEditActive = true;
    this.node.graph?.beforeChange?.();
  }

  updatePadding(value) {
    if (!this.paddingEditActive) this.beginPaddingEdit();
    this.data.workspace_padding = normalizeWorkspacePadding(value);
    this.placementWidget.value = serializePlacementData(this.data);
    this.placementWidget.callback?.(this.placementWidget.value, app.canvas, this.node);
    this.node.graph?.setDirtyCanvas?.(true, true);
    this.syncNumericControls();
    this.requestDraw();
  }

  endPaddingEdit() {
    if (!this.paddingEditActive) return;
    this.paddingEditActive = false;
    this.node.graph?.afterChange?.();
  }

  paddingChanged(value) {
    this.beginPaddingEdit();
    this.updatePadding(value);
    this.endPaddingEdit();
  }

  openPaddingEditor() {
    this.paddingValueButton.style.display = "none";
    this.paddingExactInput.style.display = "block";
    this.paddingExactInput.value = normalizeWorkspacePadding(this.data.workspace_padding).toFixed(2);
    this.paddingExactInput.focus();
    this.paddingExactInput.select();
  }

  commitPaddingEditor() {
    if (this.paddingExactInput.style.display === "none") return;
    const value = this.paddingExactInput.value;
    this.paddingExactInput.style.display = "none";
    this.paddingValueButton.style.display = "block";
    this.paddingChanged(value);
  }

  paddingEditorKeyDown(event) {
    event.stopPropagation();
    if (event.key === "Enter") {
      event.preventDefault();
      this.commitPaddingEditor();
    } else if (event.key === "Escape") {
      event.preventDefault();
      this.paddingExactInput.style.display = "none";
      this.paddingValueButton.style.display = "block";
      this.syncNumericControls();
      this.paddingValueButton.focus();
    }
  }

  numericChanged(key, field, value) {
    if (this.paintMode || this.layerPlacement(key).locked === true) return;
    const dimensions = this.dimensions();
    if (!dimensions) return;
    const normalizedValue = Number(value);
    if (!Number.isFinite(normalizedValue)) {
      this.syncNumericControls();
      return;
    }
    const placement = {
      ...rectToPlacement(
        dimensions.width, dimensions.height, this.rectFor(key, dimensions), this.layerPlacement(key),
      ),
      flip_horizontal: this.layerPlacement(key).flip_horizontal,
      flip_vertical: this.layerPlacement(key).flip_vertical,
      [field]: normalizedValue,
    };
    this.node.graph?.beforeChange?.();
    this.selected = key;
    this.queuePlacement(key, placement);
    this.flushPlacement();
    this.node.graph?.afterChange?.();
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
  }

  stepLayerValue(key, field, delta) {
    if (this.paintMode || this.layerPlacement(key).locked === true) return;
    const dimensions = this.dimensions();
    if (!dimensions) return;
    const placement = rectToPlacement(
      dimensions.width, dimensions.height, this.rectFor(key, dimensions), this.layerPlacement(key),
    );
    this.numericChanged(key, field, stepPlacementValue(field, placement[field], delta));
  }

  resetLayer(key) {
    this.foregroundContent?.exit();
    if (key === PAINT_LAYER_KEY) {
      if (this.paintLayer.clear()) this.paintChanged();
      return;
    }
    if (this.paintMode || this.layerPlacement(key).locked === true) return;
    this.node.graph?.beforeChange?.();
    this.selected = key;
    this.queuePlacement(key, {
      ...DEFAULT_PLACEMENT,
      ...(key.includes("_face_") ? { scale: 0.25 } : {}),
      rotation: 0,
      corners: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
      included: true,
    });
    this.flushPlacement();
    this.node.graph?.afterChange?.();
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
  }

  resetSelected() {
    if (this.selected) this.resetLayer(this.selected);
  }

  toggleHorizontalFlip(key = this.selected) {
    this.toggleFlip(key, "flip_horizontal");
  }

  toggleVerticalFlip(key = this.selected) {
    this.toggleFlip(key, "flip_vertical");
  }

  toggleFlip(key, field) {
    this.foregroundContent?.exit();
    if (!key || this.paintMode || this.layerPlacement(key).locked === true) return;
    const placement = this.layerPlacement(key);
    this.node.graph?.beforeChange?.();
    this.selected = key;
    this.queuePlacement(key, {
      ...placement,
      [field]: !placement[field],
    });
    this.flushPlacement();
    this.node.graph?.afterChange?.();
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
  }

  updateLayerTransform(key, changes, allowLocked = false) {
    if (this.paintMode || (!allowLocked && this.layerPlacement(key).locked === true)) return;
    this.node.graph?.beforeChange?.();
    this.selected = key;
    this.queuePlacement(key, { ...this.layerPlacement(key), ...changes });
    this.flushPlacement();
    this.node.graph?.afterChange?.();
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
  }

  stepLayerRotation(key, delta) {
    const rotation = Number(this.layerPlacement(key).rotation || 0);
    this.updateLayerTransform(key, { rotation: ((rotation + delta + 180) % 360 + 360) % 360 - 180 });
  }

  toggleWarp(key) {
    this.foregroundContent?.exit();
    if (this.paintMode || this.layerPlacement(key).locked === true) return;
    const enabling = this.warpLayer !== key;
    if (enabling && this.layerPlacement(key).included === false) return;
    if (enabling && this.data.version !== 3) {
      this.updateLayerTransform(key, {
        rotation: 0,
        corners: DEFAULT_FACE_CORNERS.map((corner) => [...corner]),
      });
    }
    this.warpLayer = enabling ? key : null;
    if (enabling) this.rotateLayer = null;
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
    this.requestDraw();
  }

  toggleRotate(key) {
    this.foregroundContent?.exit();
    if (this.paintMode || this.layerPlacement(key).locked === true) return;
    const enabling = this.rotateLayer !== key;
    if (enabling && this.layerPlacement(key).included === false) return;
    this.rotateLayer = enabling ? key : null;
    if (enabling) this.warpLayer = null;
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
    this.requestDraw();
  }

  toggleLayerIncluded(key) {
    if (this.foregroundContent?.active?.key === key) this.foregroundContent.exit();
    const excluding = this.layerPlacement(key).included !== false;
    if (excluding) {
      if (this.selected === key) this.selected = null;
      if (this.warpLayer === key) this.warpLayer = null;
      if (this.rotateLayer === key) this.rotateLayer = null;
    }
    this.updateLayerTransform(key, { included: !excluding }, true);
  }

  toggleLayerLocked(key) {
    if (this.foregroundContent?.active?.key === key) this.foregroundContent.exit();
    if (!key || key === PAINT_LAYER_KEY || this.paintMode) return;
    const locked = this.layerPlacement(key).locked === true;
    if (!locked) {
      if (this.warpLayer === key) this.warpLayer = null;
      if (this.rotateLayer === key) this.rotateLayer = null;
      if (this.gesture?.key === key) this.pointerEnd({ pointerId: this.gesture.pointerId }, true);
    }
    this.updateLayerTransform(key, { locked: !locked }, true);
  }

  togglePaintIncluded() {
    this.ensurePaintOrdered();
    this.node.graph?.beforeChange?.();
    this.data.paint_layer = {
      ...this.data.paint_layer,
      included: this.data.paint_layer?.included === false,
    };
    this.persistPlacement();
    this.node.graph?.afterChange?.();
    this.layerListSignature = null;
    this.syncLayerList(this.compositionLayers());
    this.requestDraw();
  }

  async runStaging() {
    if (!this.stagingButton || this.stagingButton.disabled) return;
    const originalText = this.stagingButton.textContent;
    this.stagingButton.disabled = true;
    this.stagingButton.textContent = "Staging…";
    this.stagingButton.style.cursor = "wait";
    try {
      await queueStagingNode(this.node);
    } catch (error) {
      console.error("Unable to queue staged compositor", error);
      this.status.textContent = `Staging queue failed: ${error?.message || error}`;
      this.status.hidden = false;
    } finally {
      this.stagingButton.disabled = false;
      this.stagingButton.textContent = originalText;
      this.stagingButton.style.cursor = "pointer";
    }
  }

  keyDown(event) {
    if (this.foregroundContent?.keyDown(event)) return;
    if (!this.selected || !this.view) return;
    if (this.paintMode) {
      const command = event.ctrlKey || event.metaKey;
      if (command && event.key.toLowerCase() === "z") {
        event.preventDefault();
        this.paintHistory(event.shiftKey ? 1 : -1);
      } else if (command && event.key.toLowerCase() === "y") {
        event.preventDefault();
        this.paintHistory(1);
      } else if (event.key === "Escape") {
        event.preventDefault();
        if (this.paintPointerId != null) {
          this.pointerEnd({ pointerId: this.paintPointerId }, true);
        } else {
          this.togglePaintMode(false);
        }
      }
      return;
    }
    if (event.key === "Escape" && this.gesture) {
      event.preventDefault();
      this.pointerEnd({ pointerId: event.pointerId }, true);
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      event.stopPropagation();
      this.resetSelected();
      return;
    }
    const deltas = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!deltas[event.key]) return;
    event.preventDefault();
    event.stopPropagation();
    if (!this.keyTransaction) {
      this.node.graph?.beforeChange?.();
      this.keyTransaction = true;
    }
    const dimensions = this.dimensions();
    const amount = event.shiftKey ? 10 : 1;
    const [dx, dy] = deltas[event.key];
    const geometry = this.resolvedGeometry(this.selected, dimensions);
    this.queuePlacement(this.selected, moveResolvedPlacement(
      dimensions.width, dimensions.height, geometry, this.layerPlacement(this.selected),
      dx * amount, dy * amount, this.data.workspace_padding,
    ));
  }

  finishKeyTransaction() {
    if (!this.keyTransaction) return;
    this.flushPlacement();
    this.keyTransaction = false;
    this.node.graph?.afterChange?.();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.foregroundContent?.dispose();
    this.endPaddingEdit();
    clearInterval(this.pollTimer);
    if (this.paintSaveTimer) clearTimeout(this.paintSaveTimer);
    if (this.paintPointerId != null) this.paintLayer.end(true);
    this.resizeObserver?.disconnect();
    this.paintColorPicker?.dispose();
    this.contextMenu.dispose();
    if (this.layoutFrame) cancelAnimationFrame(this.layoutFrame);
    if (this.drawFrame) cancelAnimationFrame(this.drawFrame);
    if (this.commitFrame) cancelAnimationFrame(this.commitFrame);
    editors.delete(this);
  }
}

function install(node) {
  if (node.__ucLayeredPlacementEditor) return node.__ucLayeredPlacementEditor;
  const editor = new LayeredPlacementEditor(node);
  node.__ucLayeredPlacementEditor = editor;
  editors.add(editor);
  const output = latestOutputs[String(node.id)];
  if (output) editor.setOutput(output);
  return editor;
}

app.registerExtension({
  name: "UtilsCollection.LayeredBackgroundEditor",
  nodeCreated(node) {
    if (NODE_TYPES.has(node.comfyClass) || NODE_TYPES.has(node.type)) install(node);
  },
  loadedGraphNode(node) {
    if (NODE_TYPES.has(node.comfyClass) || NODE_TYPES.has(node.type)) {
      const editor = install(node);
      const migrated = editor.migrateLegacyWidgetOrder();
      editor.foregroundContent?.exit();
      editor.data = parsePlacementData(editor.placementWidget.value);
      if (migrated) {
        editor.node.graph?.setDirtyCanvas?.(true, true);
      }
      editor.refreshSources(true);
      editor.scheduleLayout();
    }
  },
  onNodeOutputsUpdated(outputs) {
    latestOutputs = { ...latestOutputs, ...outputs };
    for (const [nodeId] of Object.entries(outputs || {})) {
      for (const editor of editors.values()) editor.upstreamUpdated(nodeId);
    }
    for (const [nodeId, output] of Object.entries(outputs || {})) {
      for (const editor of editors.values()) {
        if (String(editor.node.id) === String(nodeId)) editor.setOutput(output);
      }
    }
  },
});
