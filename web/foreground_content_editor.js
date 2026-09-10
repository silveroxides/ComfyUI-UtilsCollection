import { PaintLayerCanvas } from "./paint_brush.js";
import { PaintColorPicker } from "./paint_color_picker.js";
import { uploadForegroundCanvas } from "./paint_persistence.js";
import { STAGED_INPUT_STYLE, bindHoldRepeat, createStagedAction, createStagedControlRow, createStagedNumericControl } from "./staged_editor_controls.js";
import {
  DEFAULT_TEXT, foregroundLocalPoint, foregroundDisplayPoint,
  foregroundBrushScale, foregroundBrushQuad,
  normalizeForegroundContent, normalizeText, renderForegroundText,
} from "./foreground_content.js";

const canvas = () => document.createElement("canvas");
const CONTENT_KINDS = ["brush", "object_erase", "text"];

export class ForegroundContentEditor {
  constructor(editor, api) {
    this.editor = editor;
    this.api = api;
    this.states = new Map();
    this.active = null;
    this.pointer = null;
    this.objectErasing = false;
    this.cursor = null;
    this.timer = 0;
    this.saving = null;
    this.error = "";
    this.activation = 0;
    this.stopRepeat = [];
    this.root = document.createElement("div");
    Object.assign(this.root.style, {
      display: "flex", flex: "0 0 220px", width: "220px", boxSizing: "border-box",
      flexDirection: "column", gap: "3px", alignItems: "stretch",
      minHeight: "0", overflowY: "auto", padding: "2px", color: "inherit", font: "inherit",
      border: "1px solid rgba(255,255,255,.16)", borderRadius: "4px", background: "rgba(0,0,0,.18)",
    });
    for (const type of ["keydown", "keyup", "pointerdown", "pointermove", "pointerup", "wheel"]) {
      this.root.addEventListener(type, (event) => event.stopPropagation());
    }
    this.root.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !event.isComposing) { event.preventDefault(); this.exit(); }
    });
  }

  fail(error) {
    this.error = error.message || String(error);
    if (this.message) this.message.textContent = this.error;
    this.editor.status.textContent = this.error;
    this.editor.status.hidden = false;
    this.editor.requestDraw();
  }

  current(state) {
    return !this.editor.disposed && this.editor.data.foreground_content?.[state.key] === state.item;
  }

  get(key, create = false) {
    let item = this.editor.data.foreground_content?.[key];
    if (!item && !create) return null;
    if (!item) {
      const { width, height } = this.editor.layerSourceSize(key);
      item = normalizeForegroundContent({ [key]: { width, height } })[key];
      this.editor.data.foreground_content ??= {};
      this.editor.data.foreground_content[key] = item;
    }
    let state = this.states.get(key);
    if (state?.item === item) return state;
    if (state && JSON.stringify(state.item) === JSON.stringify(item)) {
      state.item = item;
      return state;
    }
    state = {
      key, item, brush: new PaintLayerCanvas(), object_erase: new PaintLayerCanvas(), text: canvas(), brushPreview: canvas(), composite: canvas(),
      generation: { brush: 0, object_erase: 0, text: 0 }, saved: { brush: 0, object_erase: 0, text: 0 }, ready: false,
      textHistory: [], textFuture: [], brushHistory: [], brushFuture: [], previewRevision: 0,
    };
    this.states.set(key, state);
    state.brush.resize(item.width, item.height);
    state.object_erase.resize(item.width, item.height);
    state.text.width = item.width; state.text.height = item.height;
    state.loading = Promise.all(CONTENT_KINDS.map(async (kind) => {
      const asset = item[kind]?.asset;
      if (!asset) {
        if (kind === "text" && item.text.value) {
          renderForegroundText(state.text, item.text, item.width, item.height);
          state.generation.text++;
        }
        return;
      }
      const image = new Image();
      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = () => reject(new Error(`Unable to load ${key} ${kind}: ${asset.filename}`));
        image.src = this.api.apiURL(`/view?${new URLSearchParams(asset)}`);
      });
      if (!this.current(state)) return;
      if (kind !== "text") state[kind].load(image, item.width, item.height);
      else state.text.getContext("2d").drawImage(image, 0, 0, item.width, item.height);
    })).then(() => {
      state.ready = true;
      this.editor.requestDraw();
    });
    void state.loading.catch((error) => this.fail(error));
    return state;
  }

  async activate(key, kind) {
    if (this.active?.key === key && this.active.kind === kind) { this.exit(); return; }
    this.exit();
    const placement = this.editor.layerPlacement(key);
    if (placement.locked || placement.included === false) return;
    if (!this.editor.layerMetadata(key)) {
      this.fail(new Error("Run Staging to resolve the foreground canvas before editing."));
      return;
    }
    const generation = ++this.activation;
    const state = this.get(key, true);
    try { await state.loading; } catch (error) { this.fail(error); return; }
    if (generation !== this.activation || !this.current(state)) return;
    if (this.editor.gesture) this.editor.pointerEnd({ pointerId: this.editor.gesture.pointerId }, false);
    this.editor.flushPlacement();
    this.editor.rotateLayer = null;
    this.editor.warpLayer = null;
    this.editor.selectLayer(key);
    this.active = { key, kind };
    this.error = "";
    this.buildControls();
    this.editor.canvas.focus();
    this.editor.requestDraw();
  }

  exit() {
    this.activation++;
    this.end(false);
    this.active = null;
    this.cursor = null;
    this.eyedropper = false;
    this.picker?.dispose();
    this.picker = null;
    for (const stop of this.stopRepeat) stop();
    this.stopRepeat = [];
    this.idleSignature = null;
    this.showIdleActions();
    this.editor.canvas.style.cursor = "default";
    this.editor.scheduleLayout();
    this.editor.requestDraw();
  }

  sync() {
    if (!this.active) { this.showIdleActions(); return; }
    const state = this.states.get(this.active.key);
    const placement = this.editor.layerPlacement(this.active.key);
    if (!state || !this.current(state) || placement.locked || placement.included === false) this.exit();
  }

  showIdleActions() {
    const key = this.editor.selected;
    const actions = key ? this.editor.layerContextActions(key) : [];
    const signature = JSON.stringify([key, actions.map(({ label, disabled, checked }) => [label, disabled, checked])]);
    if (signature === this.idleSignature) return;
    this.idleSignature = signature;
    this.root.replaceChildren();
    const caption = document.createElement("span");
    caption.textContent = key ? `Foreground ${key.replace(/^foreground_/, "").replace("_face_", " • face ")}` : "Select a foreground in the layer list.";
    Object.assign(caption.style, { opacity: ".78", padding: "4px 3px", overflowWrap: "anywhere" });
    this.root.append(caption);
    for (const action of actions) {
      if (action.separator) {
        const divider = document.createElement("div");
        Object.assign(divider.style, { flex: "0 0 1px", background: "rgba(255,255,255,.14)", margin: "2px 0" });
        this.root.append(divider);
        continue;
      }
      const button = createStagedAction(action.label, action.label, () => {
        if (!action.disabled) action.callback();
        this.sync();
      });
      button.disabled = Boolean(action.disabled);
      Object.assign(button.style, {
        textAlign: "left", opacity: action.disabled ? ".35" : "1",
        background: action.checked ? "rgba(64,180,255,.18)" : "rgba(0,0,0,.25)",
      });
      if (action.checked !== undefined) button.setAttribute("aria-pressed", String(Boolean(action.checked)));
      this.root.append(button);
    }
    this.message = null;
  }

  buildControls() {
    for (const stop of this.stopRepeat) stop();
    this.stopRepeat = [];
    this.idleSignature = null;
    this.picker?.dispose();
    this.root.replaceChildren();
    this.root.style.display = "flex";
    const { key, kind } = this.active;
    const state = this.get(key);
    const actions = document.createElement("div");
    Object.assign(actions.style, { display: "flex", flex: "0 0 auto", gap: "3px", padding: "2px 3px" });
    this.root.append(actions);
    const button = (label, callback) => {
      const control = createStagedAction(label, `${kind === "brush" ? "Brush" : "Text"}: ${label}`, callback);
      actions.append(control);
      return control;
    };
    const row = (label) => {
      const result = createStagedControlRow(label, `${key} ${label}`);
      result.caption.style.flex = "1 1 auto";
      result.root.style.width = "100%";
      this.root.append(result.root);
      return result;
    };
    const numeric = (label, getValue, change, min, max, step) => {
      const control = createStagedNumericControl(label, `${key} ${label}`, `${key} ${label}`, step);
      control.caption.style.flex = "1 1 auto";
      control.root.style.width = "100%";
      control.input.value = String(getValue());
      const commit = (value) => {
        let changed = false;
        if (String(value).trim() && Number.isFinite(Number(value))) {
          const next = Math.max(min, Math.min(max, Number(Number(value).toFixed(6))));
          if (next !== getValue()) { change(next); changed = true; }
        }
        control.input.value = String(getValue());
        return changed;
      };
      control.input.addEventListener("change", () => commit(control.input.value));
      control.input.addEventListener("keydown", (event) => {
        event.stopPropagation();
        if (event.key === "Enter") { event.preventDefault(); commit(control.input.value); control.input.blur(); }
        else if (event.key === "Escape") { event.preventDefault(); control.input.value = String(getValue()); control.input.blur(); }
      });
      this.stopRepeat.push(
        bindHoldRepeat(control.decrement, () => commit(getValue() - step)),
        bindHoldRepeat(control.increment, () => commit(getValue() + step)),
      );
      this.root.append(control.root);
    };
    const field = (label, type, value, change, attributes = {}) => {
      const { root: wrapper } = row(label);
      const input = document.createElement(type === "textarea" ? "textarea" : "input");
      if (type !== "textarea") input.type = type;
      input.value = value;
      Object.assign(input, attributes);
      Object.assign(input.style, STAGED_INPUT_STYLE, { width: type === "checkbox" ? "16px" : "64px" });
      input.setAttribute("aria-label", `${key} ${label}`);
      if (type === "checkbox") input.style.accentColor = "#65c9ff";
      if (type === "textarea") {
        Object.assign(wrapper.style, { flexDirection: "column", height: "auto", alignItems: "stretch", gap: "3px", padding: "3px" });
        Object.assign(input.style, { width: "100%", height: "76px", minHeight: "48px", resize: "vertical", padding: "4px" });
        input.rows = 3;
      }
      input.addEventListener("input", () => change(input.type === "checkbox" ? input.checked : input.value));
      wrapper.append(input);
      return input;
    };
    const select = (label, values, value, change) => {
      const { root: wrapper } = row(label);
      const input = document.createElement("select");
      for (const value of values) { const option = document.createElement("option"); option.value = option.textContent = value; input.append(option); }
      Object.assign(input.style, STAGED_INPUT_STYLE, { minWidth: "0", maxWidth: "136px", padding: "1px 3px" });
      input.setAttribute("aria-label", `${key} ${label}`);
      input.value = value; input.addEventListener("change", () => change(input.value));
      wrapper.append(input);
    };
    button("Done", () => this.exit());
    button("Undo", () => this.history(-1));
    button("Redo", () => this.history(1));
    if (kind === "brush") {
      const settings = this.editor.paintSettings;
      const change = (name, value) => {
        settings[name] = value;
        this.editor.savePaintSettings();
        this.editor.requestDraw();
      };
      select("Shape", ["circle", "square"], settings.shape, (value) => change("shape", value));
      numeric("Radius", () => settings.size, (v) => change("size", v), 1, 250, 1);
      numeric("Opacity", () => settings.opacity, (v) => change("opacity", v), 0, 1, 0.01);
      numeric("Hardness", () => settings.hardness, (v) => change("hardness", v), 0, 1, 0.01);
      const brushEraser = field("Eraser", "checkbox", "", (v) => {
        this.end(false);
        change("erasing", v);
        if (v) { this.objectErasing = false; objectEraser.checked = false; }
      }, { checked: settings.erasing && !this.objectErasing });
      const objectEraser = field("Object Eraser", "checkbox", "", (v) => {
        this.end(false);
        this.objectErasing = v;
        if (v) { change("erasing", false); brushEraser.checked = false; }
      }, { checked: this.objectErasing });
      objectEraser.title = "Erase original foreground pixels. Undo or Reset Brush restores them; text is unaffected.";
      this.editor.canvas.style.cursor = "crosshair";
    } else {
      const change = (name, value) => this.changeText(state, { [name]: value });
      this.textInput = field("Text", "textarea", state.item.text.value, (v) => change("value", v));
      select("Font", ["sans-serif", "serif", "monospace"], state.item.text.family, (v) => change("family", v));
      select("Align", ["left", "center", "right"], state.item.text.align, (v) => change("align", v));
      numeric("Size", () => state.item.text.size, (v) => change("size", v), 1, 4096, 1);
      numeric("Opacity", () => state.item.text.opacity, (v) => change("opacity", v), 0, 1, 0.01);
      field("Bold", "checkbox", "", (v) => change("bold", v), { checked: state.item.text.bold });
      field("Italic", "checkbox", "", (v) => change("italic", v), { checked: state.item.text.italic });
    }
    const colorRow = row("Color");
    this.picker = new PaintColorPicker({
      color: kind === "brush" ? this.editor.paintSettings.color : state.item.text.color,
      onChange: (color) => {
        if (kind === "text") this.changeText(state, { color });
        else {
          this.editor.paintSettings.color = color;
          this.editor.savePaintSettings();
          this.editor.requestDraw();
        }
      },
      panelHost: this.root,
      onOpenChange: () => this.editor.scheduleLayout(),
      onEyedropper: () => { this.eyedropper = true; this.editor.canvas.style.cursor = "crosshair"; },
    });
    colorRow.root.append(this.picker.root);
    Object.assign(this.picker.panel.style, {
      flex: "0 0 auto", width: "100%", height: "auto", padding: "6px",
      color: "inherit", font: "inherit", background: "rgba(0,0,0,.16)",
      border: "1px solid rgba(255,255,255,.22)", borderRadius: "6px", boxShadow: "none",
    });
    this.picker.sl.style.flex = "0 0 96px";
    Object.assign(this.picker.hexInput.style, STAGED_INPUT_STYLE, { width: "70px", minWidth: "0" });
    for (const input of this.picker.rgbInputs) Object.assign(input.style, STAGED_INPUT_STYLE, { width: "34px", minWidth: "0" });
    this.message = document.createElement("span"); this.root.append(this.message);
    Object.assign(this.message.style, { opacity: ".65", fontSize: "11px", padding: "2px 3px", overflowWrap: "anywhere" });
    this.updateStatus();
    this.editor.scheduleLayout();
  }

  changeText(state, changes) {
    state.textHistory.push({ ...state.item.text });
    if (state.textHistory.length > 20) state.textHistory.shift();
    state.textFuture = [];
    state.item.text = { ...state.item.text, ...normalizeText({ ...state.item.text, ...changes }) };
    renderForegroundText(state.text, state.item.text, state.item.width, state.item.height);
    this.changed(state, "text");
  }

  changed(state, kind) {
    state.generation[kind]++;
    state.previewRevision++;
    this.error = "";
    this.editor.persistPlacement();
    this.updateStatus();
    this.editor.requestDraw();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => { void this.flush({ finishStroke: false }).catch((error) => this.fail(error)); }, 450);
  }

  updateStatus() {
    if (!this.active || !this.message) return;
    const state = this.get(this.active.key);
    const hidden = state.item[this.active.kind].visible === false ? "Hidden • " : "";
    this.message.textContent = this.error || `${hidden}${this.saving ? "Saving…" : Object.keys(state.generation).some((k) => state.generation[k] !== state.saved[k]) ? "Unsaved" : "Saved"}`;
  }

  local(event, state) {
    const dimensions = this.editor.dimensions();
    if (!dimensions || !this.editor.view) return null;
    const placement = this.editor.layerPlacement(state.key);
    return foregroundLocalPoint(
      foregroundBrushQuad(this.editor.resolvedGeometry(state.key, dimensions)),
      this.editor.backgroundPoint(this.editor.canvasPoint(event)), state.item.width, state.item.height,
      placement.flip_horizontal, placement.flip_vertical,
    );
  }

  down(event) {
    if (!this.active) return false;
    if (event.button !== 0) return true;
    event.preventDefault(); event.stopPropagation();
    const state = this.get(this.active.key);
    const point = this.local(event, state);
    if (!state.ready || !point || point.x < 0 || point.y < 0 || point.x > state.item.width || point.y > state.item.height) return true;
    this.editor.canvas.focus();
    if (this.eyedropper) {
      const p = this.editor.canvasPoint(event), dpr = window.devicePixelRatio || 1;
      const sample = this.editor.sampleCanvas.getContext("2d").getImageData(Math.floor(p.x * dpr), Math.floor(p.y * dpr), 1, 1).data;
      const color = `#${[...sample.slice(0, 3)].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
      this.picker.setColor(color, true);
      this.eyedropper = false;
    } else if (this.active.kind === "text") {
      this.changeText(state, { x: point.x / state.item.width, y: point.y / state.item.height });
      this.textInput?.focus();
    } else {
      const kind = this.objectErasing ? "object_erase" : "brush";
      const settings = this.objectErasing ? { ...this.editor.paintSettings, color: "#ffffff", erasing: false } : this.editor.paintSettings;
      const geometry = this.editor.resolvedGeometry(state.key, this.editor.dimensions());
      const scale = foregroundBrushScale(state.item.width, state.item.height, geometry.source.width, geometry.source.height);
      if (state[kind].begin(point, settings, scale)) {
        state.previewRevision++;
        this.pointer = { id: event.pointerId, state, kind };
        this.editor.canvas.setPointerCapture(event.pointerId);
        this.editor.requestDraw();
      }
    }
    return true;
  }

  move(event) {
    if (!this.active) return false;
    const state = this.get(this.active.key);
    this.cursor = this.local(event, state);
    if (this.pointer?.id === event.pointerId) {
      event.preventDefault();
      const coalesced = event.getCoalescedEvents?.();
      const events = coalesced?.length ? coalesced : [event];
      state[this.pointer.kind || "brush"].move(events.map((sample) => this.local(sample, state)).filter(Boolean));
      state.previewRevision++;
    }
    this.editor.requestDraw();
    return true;
  }

  end(cancelled, event) {
    if (!this.pointer || (event && this.pointer.id !== event.pointerId)) return false;
    const { id, state, kind = "brush" } = this.pointer;
    this.pointer = null;
    const changed = state[kind].end(cancelled);
    state.previewRevision++;
    if (this.editor.canvas.hasPointerCapture?.(id)) this.editor.canvas.releasePointerCapture(id);
    if (changed) { this.recordBrushOperation(state, [kind]); this.changed(state, kind); }
    this.editor.requestDraw();
    return true;
  }

  keyDown(event) {
    if (!this.active) return false;
    event.stopPropagation();
    if (event.key === "Escape") {
      event.preventDefault();
      if (this.pointer) this.end(true); else this.exit();
    } else if ((event.ctrlKey || event.metaKey) && ["z", "y"].includes(event.key.toLowerCase())) {
      event.preventDefault();
      this.history(event.shiftKey || event.key.toLowerCase() === "y" ? 1 : -1);
    } else if (["Backspace", "Delete", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
    }
    return true;
  }

  history(direction) {
    if (!this.active) return;
    this.end(false);
    const state = this.get(this.active.key);
    if (this.active.kind === "brush") {
      const from = direction < 0 ? state.brushHistory : state.brushFuture;
      const to = direction < 0 ? state.brushFuture : state.brushHistory;
      if (!from.length) return;
      const operation = from.pop();
      for (const kind of direction < 0 ? [...operation].reverse() : operation) {
        if (direction < 0 ? state[kind].undo() : state[kind].redo()) this.changed(state, kind);
      }
      to.push(operation);
    } else {
      const from = direction < 0 ? state.textHistory : state.textFuture;
      const to = direction < 0 ? state.textFuture : state.textHistory;
      if (!from.length) return;
      to.push({ ...state.item.text });
      const visible = state.item.text.visible;
      state.item.text = { ...from.pop(), visible };
      renderForegroundText(state.text, state.item.text, state.item.width, state.item.height);
      this.changed(state, "text");
      this.buildControls();
    }
  }

  async reset(key, kind) {
    const state = this.get(key);
    if (!state || this.editor.layerPlacement(key).locked) return;
    await state.loading;
    if (!this.current(state)) return;
    this.end(false);
    if (kind === "brush") {
      const operation = [];
      for (const target of ["brush", "object_erase"]) {
        if (state[target].clear()) { operation.push(target); this.changed(state, target); }
      }
      if (operation.length) this.recordBrushOperation(state, operation);
    } else this.changeText(state, DEFAULT_TEXT);
    if (this.active?.key === key) this.buildControls();
  }

  toggleVisible(key, kind) {
    const state = this.get(key, true);
    this.editor.node.graph?.beforeChange?.();
    state.item[kind].visible = !state.item[kind].visible;
    this.editor.persistPlacement();
    this.editor.node.graph?.afterChange?.();
    this.updateStatus(); this.editor.requestDraw();
  }

  recordBrushOperation(state, operation) {
    state.brushHistory.push(operation);
    if (state.brushHistory.length > 20) state.brushHistory.shift();
    state.brushFuture = [];
  }

  menuActions(key) {
    const placement = this.editor.layerPlacement(key);
    const item = this.editor.data.foreground_content?.[key];
    const actions = [{ separator: true }];
    for (const kind of ["brush", "text"]) {
      const label = kind === "brush" ? "Brush" : "Text";
      actions.push(
        { label, checked: this.active?.key === key && this.active.kind === kind,
          disabled: placement.locked === true || placement.included === false,
          callback: () => { void this.activate(key, kind).catch((error) => this.fail(error)); } },
        { label: `Show ${label}`, checked: item?.[kind]?.visible !== false,
          disabled: !item && !this.editor.layerMetadata(key),
          callback: () => this.toggleVisible(key, kind) },
        { label: `Reset ${label}`, disabled: !item || placement.locked === true,
          callback: () => { void this.reset(key, kind).catch((error) => this.fail(error)); } },
      );
    }
    actions.push({ separator: true });
    return actions;
  }

  preview(key, base) {
    const state = this.get(key);
    if (!state?.ready || !base) return base;
    const { item } = state;
    if (!item.brush.visible && !item.text.visible) return base;
    const target = state.composite;
    const width = base.naturalWidth || base.width, height = base.naturalHeight || base.height;
    const metadata = this.editor.layerMetadata(key);
    const signature = `${state.previewRevision}:${item.brush.visible}:${item.text.visible}:${metadata?.flip_horizontal}:${metadata?.flip_vertical}:${width}:${height}`;
    if (state.previewBase === base && state.previewSignature === signature) return target;
    if (target.width !== width || target.height !== height) { target.width = width; target.height = height; }
    const context = target.getContext("2d");
    context.clearRect(0, 0, width, height);
    context.drawImage(base, 0, 0, width, height);
    // Base previews may already be flipped by staging. Match that orientation;
    // drawLayer later applies the difference between staged and desired flips.
    context.save();
    context.translate(metadata?.flip_horizontal ? width : 0, metadata?.flip_vertical ? height : 0);
    context.scale(metadata?.flip_horizontal ? -1 : 1, metadata?.flip_vertical ? -1 : 1);
    if (item.brush.visible) {
      const brush = state.brushPreview;
      if (brush.width !== width || brush.height !== height) { brush.width = width; brush.height = height; }
      const ctx = brush.getContext("2d"); ctx.clearRect(0, 0, width, height);
      state.object_erase.draw(ctx, 0, 0, width, height);
      context.globalCompositeOperation = "destination-out";
      context.drawImage(brush, 0, 0);
      context.globalCompositeOperation = "source-over";
      ctx.clearRect(0, 0, width, height);
      state.brush.draw(ctx, 0, 0, width, height);
      context.drawImage(brush, 0, 0);
    }
    if (item.text.visible) context.drawImage(state.text, 0, 0, width, height);
    context.restore();
    state.previewBase = base;
    state.previewSignature = signature;
    return target;
  }

  drawCursor(context) {
    if (this.active?.kind !== "brush" || !this.cursor || this.eyedropper) return;
    const state = this.get(this.active.key), editor = this.editor;
    const placement = editor.layerPlacement(state.key);
    const geometry = editor.resolvedGeometry(state.key, editor.dimensions());
    const points = foregroundBrushQuad(geometry);
    const scale = this.pointer?.state === state
      ? state[this.pointer.kind || "brush"].stroke.scale
      : foregroundBrushScale(state.item.width, state.item.height, geometry.source.width, geometry.source.height);
    const size = editor.paintSettings.size;
    context.save(); context.beginPath();
    for (let i = 0; i <= 40; i++) {
      const angle = i / 40 * 2 * Math.PI;
      let x = Math.cos(angle), y = Math.sin(angle);
      if (editor.paintSettings.shape === "square") { const scale = Math.max(Math.abs(x), Math.abs(y)); x /= scale; y /= scale; }
      const p = foregroundDisplayPoint(points, { x: this.cursor.x + x * size * scale.x, y: this.cursor.y + y * size * scale.y },
        state.item.width, state.item.height, placement.flip_horizontal, placement.flip_vertical);
      const screen = [editor.view.x + p[0] * editor.view.scale, editor.view.y + p[1] * editor.view.scale];
      if (i) context.lineTo(...screen); else context.moveTo(...screen);
    }
    context.strokeStyle = "#000"; context.lineWidth = 3; context.stroke();
    context.strokeStyle = "#fff"; context.lineWidth = 1; context.stroke(); context.restore();
  }

  async flush({ finishStroke = true } = {}) {
    if (finishStroke) this.end(false);
    clearTimeout(this.timer);
    while (this.saving) await this.saving;
    const save = async () => {
      do {
        for (const key of Object.keys(this.editor.data.foreground_content || {})) {
          const state = this.get(key);
          await state.loading;
          for (const kind of CONTENT_KINDS) {
            while (this.current(state) && state.generation[kind] !== state.saved[kind]) {
              const generation = state.generation[kind];
              const source = kind === "text" ? state.text : state[kind].canvas;
              const snapshot = canvas(); snapshot.width = source.width; snapshot.height = source.height;
              snapshot.getContext("2d").drawImage(source, 0, 0);
              const asset = await uploadForegroundCanvas(this.api, snapshot);
              if (!this.current(state)) break;
              if (generation !== state.generation[kind]) continue;
              state.item[kind] ??= {};
              state.item[kind].asset = asset;
              state.saved[kind] = generation;
              this.editor.persistPlacement();
            }
          }
        }
      } while ([...this.states.values()].some((state) => this.current(state)
        && CONTENT_KINDS.some((kind) => state.generation[kind] !== state.saved[kind])));
    };
    this.saving = save(); this.updateStatus();
    try { await this.saving; this.error = ""; }
    catch (error) { this.fail(error); throw error; }
    finally { this.saving = null; this.updateStatus(); }
  }

  hasPending() {
    return this.pointer !== null || [...this.states.values()].some((state) => this.current(state)
      && CONTENT_KINDS.some((kind) => state.generation[kind] !== state.saved[kind]));
  }

  dispose() {
    for (const stop of this.stopRepeat) stop();
    this.stopRepeat = [];
    this.activation++;
    this.end(true);
    clearTimeout(this.timer);
    this.picker?.dispose();
    this.states.clear();
  }
}
