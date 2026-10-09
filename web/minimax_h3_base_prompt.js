import { app } from "../../scripts/app.js";
import {
  BASE_TASKS,
  BASE_TABS,
  BASE_TAB_LABELS,
  VISUAL_STYLES,
  CAMERA_MOTIONS,
  formatSeconds,
  parseSeconds,
  h3StepDuration,
  createBaseState,
  restoreBaseState,
  compileBasePrompt,
  addBaseSegment,
  getFinalShot,
  baseTimelineWarnings,
} from "./minimax_h3_base_prompt_model.js";

class H3BaseCanvasPromptEditor {
  constructor(node, name, data) {
    this.node = node;
    this.raw = data[1]?.default ?? JSON.stringify(createBaseState());
    this.state = null;
    this.error = null;
    this.scroll = 0;
    this.contentHeight = 0;
    this.viewportY = 0;
    this.viewportHeight = 380;
    this.hitRegions = [];
    this.hoveredRegion = null;
    this.textEditor = null;
    this.outsidePointer = null;
    this.abort = new AbortController();
    this.restore(this.raw);

    this.widget = node.addCustomWidget({
      name,
      type: "custom",
      value: this.raw,
      options: { socketless: true },
      computeSize: () => [node.size[0], Math.max(380, node.size[1] - 40)],
      draw: (ctx, _node, width, y) => this.draw(ctx, width, y),
      mouse: (event, position) => this.mouse(event, position),
      serializeValue: () => this.serialize(),
    });
    this.widget.options.socketless = true;

    const getWidgetOnPos = node.getWidgetOnPos;
    node.getWidgetOnPos = (graphX, graphY, includeDisabled) => {
      const x = graphX - node.pos[0];
      const y = graphY - node.pos[1];
      if (this.hitRegions.some(region => this.contains(region, x, y))) return this.widget;
      const found = getWidgetOnPos.call(node, graphX, graphY, includeDisabled);
      return found === this.widget ? undefined : found;
    };

    app.canvas?.canvas?.addEventListener("pointermove", event => this.onPointerMove(event), {
      passive: true, signal: this.abort.signal,
    });
    app.canvas?.canvas?.addEventListener("pointerleave", () => this.clearHoveredTooltip(), {
      passive: true, signal: this.abort.signal,
    });
    app.canvas?.canvas?.addEventListener("wheel", event => this.onWheel(event), {
      capture: true, passive: false, signal: this.abort.signal,
    });

    const onRemoved = node.onRemoved;
    node.onRemoved = (...args) => {
      this.abort.abort();
      this.closeTextEditor();
      node.getWidgetOnPos = getWidgetOnPos;
      onRemoved?.apply(node, args);
    };

    requestAnimationFrame(() => {
      if (!this.abort.signal.aborted) {
        node.setSize([Math.max(480, node.size[0] || 480), Math.max(500, node.size[1] || 500)]);
      }
    });
  }

  restore(raw) {
    try {
      this.state = restoreBaseState(raw);
      this.error = null;
    } catch (err) {
      this.error = String(err?.message || err);
      this.state = createBaseState();
    }
  }

  serialize() {
    return JSON.stringify(this.state);
  }

  sync() {
    const raw = typeof this.widget.value === "string" ? this.widget.value : JSON.stringify(this.widget.value);
    if (raw !== this.serialize()) {
      this.restore(raw);
    }
  }

  change(mutator) {
    mutator();
    this.widget.value = this.serialize();
    if (this.node.graph) {
      this.node.graph.setDirtyCanvas(true, true);
    }
    app.canvas?.setDirty(true, true);
  }

  box(ctx, x, y, w, h, fill = null, stroke = null, radius = 0) {
    if (w <= 0 || h <= 0) return;
    ctx.save();
    ctx.beginPath();
    if (radius && ctx.roundRect) {
      ctx.roundRect(x, y, w, h, radius);
    } else {
      ctx.rect(x, y, w, h);
    }
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.stroke();
    }
    ctx.restore();
  }

  text(ctx, value, x, y, color = "#ddd", font = "12px sans-serif", maxWidth = null) {
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const line = String(value ?? "").replace(/\r?\n/g, " ");
    if (!maxWidth || ctx.measureText(line).width <= maxWidth) {
      ctx.fillText(line, x, y);
      return;
    }
    let end = line.length;
    while (end > 0 && ctx.measureText(line.slice(0, end) + "…").width > maxWidth) end--;
    ctx.fillText(line.slice(0, end) + "…", x, y);
  }

  drawMultilineText(ctx, text, x, y, maxWidth, lineHeight = 15, color = "#cbd5e1", font = "11px sans-serif") {
    ctx.font = font;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const words = String(text ?? "").split(/\s+/);
    let line = "";
    let curY = y;
    for (let n = 0; n < words.length; n++) {
      const testLine = line ? `${line} ${words[n]}` : words[n];
      const metrics = ctx.measureText(testLine);
      if (metrics.width > maxWidth && line) {
        ctx.fillText(line, x, curY);
        line = words[n];
        curY += lineHeight;
      } else {
        line = testLine;
      }
    }
    if (line) {
      ctx.fillText(line, x, curY);
      curY += lineHeight;
    }
    return curY;
  }

  button(ctx, x, y, w, h, label, action, accent = false, align = "center", active = false, danger = false, tooltip = null, isFixed = false) {
    let fill = accent ? "#26394e" : "#202329";
    let stroke = accent ? "#709ecc" : "#59606a";
    let textColor = accent ? "#d7ebff" : "#ddd";

    if (active) {
      fill = "#1d4ed8";
      stroke = "#93c5fd";
      textColor = "#ffffff";
    } else if (danger) {
      fill = "#7f1d1d";
      stroke = "#f87171";
      textColor = "#fecaca";
    }

    this.box(ctx, x, y, w, h, fill, stroke, 4);
    ctx.font = "11px sans-serif";
    const tw = ctx.measureText(label).width;
    let tx = x + (w - tw) / 2;
    if (align === "left") tx = x + 8;
    else if (align === "right") tx = x + w - tw - 8;

    this.text(ctx, label, tx, y + h / 2, textColor, "11px sans-serif", w - 8);
    if (isFixed) {
      this.hitFixed(x, y, w, h, action, tooltip);
    } else {
      this.hit(x, y, w, h, action, tooltip);
    }
  }

  hitFixed(x, y, w, h, action, tooltip = null) {
    this.hitRegions.push({ x, y, w, h, action, tooltip });
  }

  hit(x, y, w, h, action, tooltip = null) {
    const top = Math.max(y, this.viewportY || 0);
    const bottom = Math.min(y + h, (this.viewportY || 0) + (this.viewportHeight || 9999));
    if (bottom > top) {
      this.hitRegions.push({ x, y: top, w, h: bottom - top, action, tooltip });
    }
  }

  onPointerMove(event) {
    if (this.textEditor) {
      this.clearHoveredTooltip();
      return;
    }
    const canvas = app.canvas;
    if (!canvas?.graph) return;
    const bounds = canvas.canvas.getBoundingClientRect();
    const scale = canvas.ds.scale;
    const graphX = (event.clientX - bounds.left) / scale - canvas.ds.offset[0];
    const graphY = (event.clientY - bounds.top) / scale - canvas.ds.offset[1];
    const x = graphX - this.node.pos[0];
    const y = graphY - this.node.pos[1];

    if (x < 0 || x > (this.node.size[0] || 480) || y < 0 || y > (this.node.size[1] || 500)) {
      this.clearHoveredTooltip();
      return;
    }

    const matched = this.hitRegions.find(r => r.tooltip && this.contains(r, x, y));
    if (matched !== this.hoveredRegion) {
      this.hoveredRegion = matched || null;
      canvas.setDirty(true, false);
    }
  }

  clearHoveredTooltip() {
    if (this.hoveredRegion) {
      this.hoveredRegion = null;
      app.canvas?.setDirty(true, false);
    }
  }

  drawCanvasTooltip(ctx, region, nodeWidth, yTop, nodeHeight) {
    if (!region?.tooltip) return;
    const text = region.tooltip;
    const maxWidth = Math.min(280, nodeWidth - 28);
    const font = "11px sans-serif";
    const lineHeight = 15;

    ctx.save();
    ctx.font = font;

    const words = String(text).split(/\s+/);
    const lines = [];
    let currentLine = "";
    for (let i = 0; i < words.length; i++) {
      const testLine = currentLine ? `${currentLine} ${words[i]}` : words[i];
      if (ctx.measureText(testLine).width > maxWidth && currentLine) {
        lines.push(currentLine);
        currentLine = words[i];
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) lines.push(currentLine);

    let maxMeasured = 0;
    for (const l of lines) {
      maxMeasured = Math.max(maxMeasured, ctx.measureText(l).width);
    }

    const paddingX = 8;
    const paddingY = 6;
    const boxW = Math.max(60, maxMeasured + paddingX * 2);
    const boxH = lines.length * lineHeight + paddingY * 2;

    let bx = region.x + region.w / 2 - boxW / 2;
    bx = Math.max(10, Math.min(nodeWidth - boxW - 10, bx));

    let by = region.y - boxH - 8;
    if (by < yTop + 4) {
      by = region.y + region.h + 8;
    }

    ctx.shadowColor = "rgba(0, 0, 0, 0.7)";
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 3;

    this.box(ctx, bx, by, boxW, boxH, "#0f1218", "#475569", 4);

    ctx.shadowColor = "transparent";
    ctx.shadowBlur = 0;

    ctx.fillStyle = "#f1f5f9";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], bx + paddingX, by + paddingY + i * lineHeight);
    }

    ctx.restore();
  }

  openMultilineEditor(rect, value, apply, tagSpawner = null) {
    this.closeTextEditor();
    this.clearHoveredTooltip();

    const container = document.createElement("div");
    container.dataset.testid = "h3-base-prompt-editor-container";
    Object.assign(container.style, {
      position: "fixed", zIndex: "1000", boxSizing: "border-box", margin: "0",
      display: "flex", flexDirection: "column", background: "#1f2228",
      border: "1px solid #709ecc", borderRadius: "4px", padding: "4px", gap: "4px",
    });

    const element = document.createElement("textarea");
    element.className = "comfy-multiline-input";
    element.dataset.testid = "h3-base-prompt-textarea";
    element.value = value;
    element.spellcheck = true;
    Object.assign(element.style, {
      width: "100%", height: "100%", minHeight: "80px", boxSizing: "border-box",
      background: "#121418", color: "#eee", border: "1px solid #3d434d", borderRadius: "3px",
      outline: "none", resize: "none", fontFamily: "Inter, Arial, sans-serif", fontSize: "12px",
      lineHeight: "1.35", padding: "4px",
    });

    if (tagSpawner && tagSpawner.length) {
      const tagBar = document.createElement("div");
      Object.assign(tagBar.style, {
        display: "flex", flexWrap: "wrap", gap: "4px", maxHeight: "50px", overflowY: "auto",
      });
      tagSpawner.forEach(tag => {
        const btn = document.createElement("button");
        btn.textContent = tag;
        Object.assign(btn.style, {
          background: "#26394e", color: "#d7ebff", border: "1px solid #709ecc",
          borderRadius: "3px", fontSize: "11px", padding: "2px 6px", cursor: "pointer",
        });
        btn.onmousedown = e => e.preventDefault();
        btn.onclick = e => {
          e.preventDefault();
          const start = element.selectionStart ?? element.value.length;
          const end = element.selectionEnd ?? element.value.length;
          const prev = element.value;
          element.value = prev.slice(0, start) + tag + prev.slice(end);
          element.selectionStart = element.selectionEnd = start + tag.length;
          element.focus();
          this.change(() => apply(element.value));
        };
        tagBar.appendChild(btn);
      });
      container.appendChild(tagBar);
    }

    container.appendChild(element);
    this.textEditor = { element, container, rect };

    this.outsidePointer = (event) => {
      if (this.textEditor?.container && !this.textEditor.container.contains(event.target)) {
        this.closeTextEditor();
      }
    };
    requestAnimationFrame(() => {
      document.addEventListener("pointerdown", this.outsidePointer, true);
    });

    element.addEventListener("input", () => this.change(() => apply(element.value)));
    element.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Escape" || (event.key === "Enter" && (event.ctrlKey || event.metaKey))) {
        event.preventDefault();
        this.closeTextEditor();
      }
    });

    element.addEventListener("blur", (e) => {
      if (this.textEditor?.container && e.relatedTarget && this.textEditor.container.contains(e.relatedTarget)) {
        return;
      }
      this.closeTextEditor();
    });

    document.body.appendChild(container);
    this.positionTextEditor();
    requestAnimationFrame(() => element.focus());
  }

  openSingleLineEditor(rect, value, apply) {
    this.closeTextEditor();
    this.clearHoveredTooltip();

    const container = document.createElement("div");
    container.dataset.testid = "h3-base-prompt-singleline-container";
    Object.assign(container.style, {
      position: "fixed", zIndex: "1000", boxSizing: "border-box", margin: "0",
      display: "flex", background: "#1f2228",
      border: "1px solid #709ecc", borderRadius: "4px", padding: "2px",
    });

    const element = document.createElement("input");
    element.type = "text";
    element.value = String(value ?? "");
    Object.assign(element.style, {
      width: "100%", height: "100%", boxSizing: "border-box",
      background: "#121418", color: "#f8fafc", border: "none", outline: "none",
      borderRadius: "2px", padding: "2px 6px", fontSize: "11px", fontFamily: "sans-serif",
    });

    container.appendChild(element);
    this.textEditor = { element, container, rect, isSingle: true };

    this.outsidePointer = (event) => {
      if (this.textEditor?.container && !this.textEditor.container.contains(event.target)) {
        this.closeTextEditor();
      }
    };
    requestAnimationFrame(() => {
      document.addEventListener("pointerdown", this.outsidePointer, true);
    });

    element.addEventListener("input", () => this.change(() => apply(element.value)));
    element.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Enter" || event.key === "Escape") {
        event.preventDefault();
        this.closeTextEditor();
      }
    });

    element.addEventListener("blur", () => this.closeTextEditor());

    document.body.appendChild(container);
    this.positionTextEditor();
    requestAnimationFrame(() => {
      element.focus();
      element.select();
    });
  }

  closeTextEditor() {
    if (this.outsidePointer) {
      document.removeEventListener("pointerdown", this.outsidePointer, true);
      this.outsidePointer = null;
    }
    if (this.textEditor) {
      this.textEditor.container?.remove();
      this.textEditor.element?.remove();
      this.textEditor = null;
    }
  }

  positionTextEditor() {
    if (!this.textEditor || !app.canvas?.canvas) return;
    const { container, rect, isSingle } = this.textEditor;
    const canvas = app.canvas.canvas;
    const canvasRect = canvas.getBoundingClientRect();
    const scale = app.canvas.ds.scale;
    const x = canvasRect.left + (this.node.pos[0] + rect.x + app.canvas.ds.offset[0]) * scale;
    const y = canvasRect.top + (this.node.pos[1] + rect.y + app.canvas.ds.offset[1]) * scale;
    const w = isSingle ? Math.max(120, rect.w * scale) : Math.max(340, rect.w * scale);
    const h = isSingle ? Math.max(26, rect.h * scale) : Math.max(140, rect.h * scale + 60);

    Object.assign(container.style, {
      left: `${Math.max(10, Math.min(window.innerWidth - w - 10, x))}px`,
      top: `${Math.max(10, Math.min(window.innerHeight - h - 10, y))}px`,
      width: `${w}px`,
      height: `${h}px`,
    });
  }

  draw(ctx, width, y) {
    if (this.textEditor) this.positionTextEditor();
    this.sync();
    this.hitRegions = [];

    const nodeHeight = this.node.size[1] || 500;
    const totalHeight = Math.max(380, nodeHeight - y - 10);
    this.viewportY = y;
    this.viewportHeight = totalHeight;
    this.box(ctx, 4, y, width - 8, totalHeight, "#181a1f", "#333842", 5);

    const left = 10;
    const right = width - 10;
    const available = right - left;

    if (this.error) {
      this.text(ctx, this.error, left, y + 20, "#ff8080", "bold 12px sans-serif");
      return;
    }

    // Top Header: Title + Precision
    this.text(ctx, "MiniMax H3 Base Prompt Builder", left, y + 14, "#f3f5f7", "bold 12px sans-serif");
    const prec = this.state.precision || 2;
    this.button(ctx, right - 80, y + 4, 80, 20, `${prec} Decimals`, () => {
      this.change(() => {
        this.state.precision = this.state.precision === 2 ? 3 : 2;
      });
    }, true, "center", false, false, "Switches timestamps between two decimal places (00.00s) and three decimal places (00.000s) across the entire prompt.", true);

    // Tab Navigation Bar
    const tabY = y + 28;
    const tabWidth = Math.floor(available / BASE_TABS.length);
    const tabTooltips = {
      task: "Select your video generation mode (text-to-video or keyframe pictures) and configure frame alignment.",
      description: "Write what happens in the video, either as chronological timed segments or as continuous shot-by-shot text.",
      soundscape: "Describe the background ambience, environment noise, and physical action sounds heard across the entire video.",
      music: "Describe the audience-only background music, including instruments, tempo, and rhythm.",
    };

    BASE_TABS.forEach((tab, index) => {
      const active = (this.state.activeTab || "task") === tab;
      const tx = left + index * tabWidth;
      const tw = index === BASE_TABS.length - 1 ? (right - tx) : (tabWidth - 2);

      let countStr = "";
      if (tab === "description") {
        countStr = this.state.description?.mode === "continuous" ? " (Cont)" : ` (${this.state.segments?.length || 0})`;
      }

      const label = `${BASE_TAB_LABELS[tab].split(" ")[1]}${countStr}`;
      this.button(ctx, tx, tabY, tw, 26, label, () => {
        this.closeTextEditor();
        this.clearHoveredTooltip();
        this.change(() => {
          this.state.activeTab = tab;
          this.scroll = 0;
        });
      }, false, "center", active, false, tabTooltips[tab], true);
    });

    const contentTopY = tabY + 32;
    const previewHeight = 58;
    const previewY = y + totalHeight - previewHeight - 4;
    const contentHeight = Math.max(150, previewY - contentTopY - 6);
    this.viewportY = contentTopY;
    this.viewportHeight = contentHeight;

    ctx.save();
    ctx.beginPath();
    ctx.rect(left, contentTopY, available, contentHeight);
    ctx.clip();

    let cy = 0;
    const sy = yCoord => contentTopY + yCoord - this.scroll;

    switch (this.state.activeTab || "task") {
      case "task":
        cy = this.drawTaskTab(ctx, left, cy, available, sy);
        break;
      case "description":
        cy = this.drawDescriptionTab(ctx, left, cy, available, sy);
        break;
      case "soundscape":
        cy = this.drawSoundscapeTab(ctx, left, cy, available, sy);
        break;
      case "music":
        cy = this.drawMusicTab(ctx, left, cy, available, sy);
        break;
    }

    this.contentHeight = Math.max(contentHeight, cy);
    ctx.restore();

    // Scrollbar if needed
    if (this.contentHeight > contentHeight) {
      const barH = Math.max(20, (contentHeight / this.contentHeight) * contentHeight);
      const barY = contentTopY + (this.scroll / (this.contentHeight - contentHeight)) * (contentHeight - barH);
      this.box(ctx, right - 4, barY, 4, barH, "#4b5563", null, 2);
    }

    // Bottom Preview Area
    this.box(ctx, left, previewY, available, previewHeight, "#121418", "#2d323b", 3);
    const warnings = baseTimelineWarnings(this.state);
    const statusText = warnings.length ? `⚠ ${warnings[0]}` : `Prompt compiled: ${compileBasePrompt(this.state).length} chars`;
    const statusColor = warnings.length ? "#fbbf24" : "#4ade80";
    this.text(ctx, statusText, left + 6, previewY + 12, statusColor, "11px sans-serif");

    const promptPreview = compileBasePrompt(this.state) || "Prompt preview will appear here...";
    const previewLine = promptPreview.replace(/\n+/g, " ❚ ");
    this.text(ctx, previewLine, left + 6, previewY + 34, "#94a3b8", "11px monospace", available - 12);

    // Canvas Draw-Loop Tooltip (Offset upwards on vertical axis)
    if (this.hoveredRegion?.tooltip) {
      this.drawCanvasTooltip(ctx, this.hoveredRegion, width, y, totalHeight);
    }
  }

  drawTaskTab(ctx, left, cy, available, sy) {
    this.text(ctx, "Base Mode Task Selection", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
    cy += 24;

    const taskW = Math.floor((available - 12) / 4);
    const curTask = this.state.task || "T2VA";
    const taskTooltips = {
      T2VA: "Text-to-Video: Generates a complete video purely from text description without using any starting image.",
      I2VA: "Image-to-Video: Uses Picture 1 as the exact opening first frame (at 00.00s) and develops action forward from it.",
      FL2VA: "First & Last Frame: Uses Picture 1 as the opening frame and Picture 2 as the ending frame, generating the motion connecting them.",
      L2VA: "Last Frame Landing: Begins with action that gradually leads up to and settles into Picture 1 at the end of the video.",
    };

    BASE_TASKS.forEach((t, i) => {
      this.button(ctx, left + i * (taskW + 4), sy(cy), taskW, 26, t, () => {
        this.change(() => { this.state.task = t; });
      }, false, "center", curTask === t, false, taskTooltips[t]);
    });
    cy += 34;

    // Task description card
    const cardH = 58;
    this.box(ctx, left, sy(cy), available, cardH, "#1e232a", "#374151", 4);
    let desc = "";
    if (curTask === "T2VA") {
      desc = "T2VA: Direct text-to-video with audio. No image alignment instruction. Core fields begin directly with integrated_multimodal_description.";
    } else if (curTask === "I2VA") {
      desc = "I2VA: First-frame image-to-video. Emits standard instruction anchoring <Picture 1> at 0.00s. Shot 1 develops forward from Picture 1.";
    } else if (curTask === "FL2VA") {
      desc = "FL2VA: First-and-last frame interpolation. Picture 1 anchors 0.00s and Picture 2 anchors the final duration. Connects both frames across shots.";
    } else if (curTask === "L2VA") {
      desc = "L2VA: Last-frame landing. Picture 1 anchors the final endpoint. Shot 1 infers an earlier state and lands on Picture 1 at the end.";
    }
    this.drawMultilineText(ctx, desc, left + 8, sy(cy + 8), available - 16, 15, "#cbd5e1", "11px sans-serif");
    cy += cardH + 12;

    // Parameters for FL2VA / L2VA alignment instruction
    if (curTask === "FL2VA" || curTask === "L2VA") {
      this.text(ctx, "Keyframe Alignment Parameters", left, sy(cy + 8), "#93c5fd", "bold 12px sans-serif");
      cy += 22;

      // Duration
      this.text(ctx, `Target Duration: ${Number(this.state.duration || 5.0).toFixed(2)}s`, left, sy(cy + 10), "#e2e8f0", "11px sans-serif");
      this.button(ctx, left + 140, sy(cy), 42, 20, "-1.0s", () => this.change(() => {
        this.state.duration = Math.max(1.0, (Number(this.state.duration) || 5.0) - 1.0);
      }), false, "center", false, false, "Shortens target video duration by 1 second.");
      this.button(ctx, left + 186, sy(cy), 42, 20, "+1.0s", () => this.change(() => {
        this.state.duration = (Number(this.state.duration) || 5.0) + 1.0;
      }), false, "center", false, false, "Lengthens target video duration by 1 second.");
      [5.0, 6.0, 8.0, 10.0].forEach((d, i) => {
        this.button(ctx, left + 236 + i * 38, sy(cy), 34, 20, `${d}s`, () => this.change(() => {
          this.state.duration = d;
        }), false, "center", Math.abs((this.state.duration || 5.0) - d) < 0.05, false, `Sets total video duration to ${d} seconds.`);
      });
      cy += 28;

      // Final shot
      const autoShot = this.state.autoFinalShot !== false;
      const finalShotNum = getFinalShot(this.state);
      this.text(ctx, `Final Shot: Shot ${finalShotNum}`, left, sy(cy + 10), "#e2e8f0", "11px sans-serif");
      this.button(ctx, left + 140, sy(cy), 130, 20, autoShot ? "Auto (From Segments)" : "Manual", () => this.change(() => {
        this.state.autoFinalShot = !autoShot;
      }), false, "center", autoShot, false, "When Auto is on, the final keyframe automatically locks to your highest shot number. Turn off to set a custom shot number.");

      if (!autoShot) {
        this.button(ctx, left + 276, sy(cy), 30, 20, "-1", () => this.change(() => {
          this.state.finalShot = Math.max(1, (Number(this.state.finalShot) || 1) - 1);
        }), false, "center", false, false, "Decreases the landing shot number.");
        this.button(ctx, left + 310, sy(cy), 30, 20, "+1", () => this.change(() => {
          this.state.finalShot = (Number(this.state.finalShot) || 1) + 1;
        }), false, "center", false, false, "Increases the landing shot number.");
      }
      cy += 32;
    }

    // Alignment Instruction Preview
    this.text(ctx, "Instruction Header Preview", left, sy(cy + 8), "#94a3b8", "bold 11px sans-serif");
    cy += 20;

    const headerBoxH = 50;
    this.box(ctx, left, sy(cy), available, headerBoxH, "#121418", "#333842", 3);
    let instr = "(No alignment instruction header for T2VA)";
    const durStr = Number(this.state.duration || 5.0).toFixed(2);
    const finalShot = getFinalShot(this.state);
    if (curTask === "I2VA") {
      instr = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
    } else if (curTask === "FL2VA") {
      instr = `How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot ${finalShot}) aligns with the ${durStr}-second mark of the target video.`;
    } else if (curTask === "L2VA") {
      instr = `How the reference pictures align with the target video — <Picture 1> (from [Shot ${finalShot}]) aligns with the ${durStr}-second mark of the target video.`;
    }
    this.drawMultilineText(ctx, instr, left + 8, sy(cy + 8), available - 16, 15, curTask === "T2VA" ? "#64748b" : "#67e8f9", "11px sans-serif");
    cy += headerBoxH + 16;

    return cy;
  }

  drawDescriptionTab(ctx, left, cy, available, sy) {
    const isTimeline = this.state.description?.mode !== "continuous";

    // Mode Selector: Timeline vs Continuous
    this.text(ctx, "Description Mode:", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
    this.button(ctx, left + 120, sy(cy), 110, 22, "Timeline Mode", () => this.change(() => {
      this.closeTextEditor();
      if (!this.state.description) this.state.description = {};
      this.state.description.mode = "timeline";
    }), false, "center", isTimeline, false, "Divides your video into timed chronological segments with separate visual, dialogue, and sound controls.");
    this.button(ctx, left + 236, sy(cy), 110, 22, "Continuous Text", () => this.change(() => {
      this.closeTextEditor();
      if (!this.state.description) this.state.description = {};
      this.state.description.mode = "continuous";
    }), false, "center", !isTimeline, false, "Allows writing freely in continuous paragraphs using shot labels like [Shot 1] and camera cuts.");
    cy += 30;

    // Quick tag chips bar
    this.text(ctx, "Quick Insert Helpers:", left, sy(cy + 8), "#94a3b8", "10px sans-serif");
    cy += 18;
    const task = this.state.task || "T2VA";
    const tags = [];
    if (task !== "T2VA") {
      tags.push({ label: "<Picture 1>", tip: "Inserts the Picture 1 reference tag where the first keyframe applies." });
      if (task === "FL2VA") tags.push({ label: "<Picture 2>", tip: "Inserts the Picture 2 reference tag where the ending keyframe applies." });
    }
    tags.push(
      { label: "(S1)", tip: "Labels the speaker so the character's vocal identity remains consistent." },
      { label: "(S2)", tip: "Labels the speaker so the character's vocal identity remains consistent." },
      { label: "<d>[English] ...</d>", tip: "Inserts dialogue tags. Characters speak the exact words written inside." },
      { label: "<scenetrans>", tip: "Marks that spoken dialogue continues seamlessly across a camera cut without pausing." },
      { label: "<cutoff>", tip: "Marks that speech is suddenly cut off by the end of the video clip." }
    );

    let chipX = left;
    tags.forEach(item => {
      const tagW = Math.max(50, ctx.measureText ? ctx.measureText(item.label).width + 16 : 60);
      if (chipX + tagW > left + available) {
        chipX = left;
        cy += 24;
      }
      this.button(ctx, chipX, sy(cy), tagW, 20, item.label, () => {
        if (!isTimeline) {
          const cur = this.state.description?.continuousText || "";
          this.change(() => { this.state.description.continuousText = cur ? `${cur} ${item.label}` : item.label; });
        } else {
          if (!this.state.segments?.length) {
            addBaseSegment(this.state);
          }
          const lastSeg = this.state.segments[this.state.segments.length - 1];
          const cur = lastSeg.visual || "";
          this.change(() => { lastSeg.visual = cur ? `${cur} ${item.label}` : item.label; });
        }
      }, true, "center", false, false, item.tip);
      chipX += tagW + 4;
    });
    cy += 28;

    if (!isTimeline) {
      // Continuous Mode View
      this.text(ctx, "Continuous Multimodal Description (Single or Multi-Shot paragraph):", left, sy(cy + 8), "#e2e8f0", "11px sans-serif");
      cy += 20;

      const boxH = 140;
      this.box(ctx, left, sy(cy), available, boxH, "#121418", "#333842", 4);
      const textVal = this.state.description?.continuousText || "";
      const displayTxt = textVal || "Click to write continuous description, e.g. [Shot 1] Live-action, cinematic... [Shot 2] At 00:03.500, the camera cuts to...";
      this.drawMultilineText(ctx, displayTxt, left + 8, sy(cy + 8), available - 16, 16, textVal ? "#f8fafc" : "#64748b", "11px sans-serif");

      const editorRect = { x: left, y: sy(cy), w: available, h: boxH };
      this.hit(left, sy(cy), available, boxH, () => {
        this.openMultilineEditor(
          editorRect,
          this.state.description?.continuousText || "",
          (val) => { this.state.description.continuousText = val; },
          ["[Shot 1]", "[Shot 2] At 00:03.500, the camera cuts to ", ...tags.map(t => t.label)]
        );
      });
      cy += boxH + 16;
      return cy;
    }

    // Timeline Mode View
    const segDur = Number(this.state.description?.segmentDuration) || 2.333;
    this.text(ctx, `Timeline Segments (${this.state.segments?.length || 0})`, left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");

    this.button(ctx, left + 140, sy(cy), 96, 22, "+ Add Segment", () => this.change(() => {
      addBaseSegment(this.state);
    }), true, "center", false, false, "Adds a new timed scene segment starting right after the previous one.");

    // Duration Stepper Controls
    this.text(ctx, "Dur:", left + 246, sy(cy + 10), "#94a3b8", "11px sans-serif");
    this.button(ctx, left + 274, sy(cy), 36, 22, "-17f", () => this.change(() => {
      const nextSec = h3StepDuration(segDur, -1);
      if (!this.state.description) this.state.description = {};
      this.state.description.segmentDuration = nextSec;
    }), false, "center", false, false, "Shortens segment duration by 17 frames (about 0.71 seconds), matching H3 native steps.");
    this.button(ctx, left + 314, sy(cy), 36, 22, "+17f", () => this.change(() => {
      const nextSec = h3StepDuration(segDur, 1);
      if (!this.state.description) this.state.description = {};
      this.state.description.segmentDuration = nextSec;
    }), false, "center", false, false, "Lengthens segment duration by 17 frames (about 0.71 seconds), matching H3 native steps.");
    const durRect = { x: left + 354, y: sy(cy), w: 56, h: 22 };
    this.button(ctx, durRect.x, durRect.y, durRect.w, durRect.h, `${segDur.toFixed(2)}s`, () => {
      this.openSingleLineEditor(durRect, String(segDur.toFixed(2)), (val) => {
        const s = parseFloat(val);
        if (s > 0) {
          if (!this.state.description) this.state.description = {};
          this.state.description.segmentDuration = s;
        }
      });
    }, true, "center", false, false, "Current default segment duration. Click to edit value.");
    cy += 30;

    // Segment List
    const segments = this.state.segments || [];
    if (!segments.length) {
      this.box(ctx, left, sy(cy), available, 60, "#1a1d24", "#2d323b", 3);
      this.text(ctx, "No segments yet. Click '+ Add Segment' above to start.", left + 12, sy(cy + 30), "#64748b", "12px sans-serif");
      return cy + 70;
    }

    segments.forEach((seg, idx) => {
      const cardH = 150;
      this.box(ctx, left, sy(cy), available, cardH, "#1a1e26", "#2d3442", 4);

      // Card Header
      const headerText = `Segment ${idx + 1}: [${seg.start} - ${seg.end}]`;
      this.text(ctx, headerText, left + 8, sy(cy + 14), "#60a5fa", "bold 11px sans-serif");

      // Shot Toggle & Counter
      const hasShot = seg.hasShot !== false;
      const shotNum = seg.shot || idx + 1;
      this.button(ctx, left + available - 130, sy(cy + 6), 70, 20, hasShot ? `[Shot ${shotNum}]` : "No Cut", () => this.change(() => {
        seg.hasShot = !hasShot;
        if (seg.hasShot && !seg.shot) seg.shot = idx + 1;
      }), false, "center", hasShot, false, "Toggles whether this segment introduces an instant camera cut or smoothly continues previous movement.");

      // Delete Segment
      this.button(ctx, left + available - 26, sy(cy + 6), 20, 20, "✕", () => this.change(() => {
        this.closeTextEditor();
        this.clearHoveredTooltip();
        this.state.segments.splice(idx, 1);
      }), false, "center", false, true, "Deletes this scene segment from the timeline.");

      // Visual line
      const visualY = cy + 30;
      this.text(ctx, "[VISUAL]:", left + 8, sy(visualY + 10), "#94a3b8", "bold 10px sans-serif");
      const visBoxW = available - 80;
      this.box(ctx, left + 68, sy(visualY), visBoxW, 36, "#121418", "#333842", 3);
      const visText = seg.visual || "Click to describe visual action and camera motion...";
      this.drawMultilineText(ctx, visText, left + 72, sy(visualY + 4), visBoxW - 8, 14, seg.visual ? "#f8fafc" : "#64748b", "10px sans-serif");

      const visRect = { x: left + 68, y: sy(visualY), w: visBoxW, h: 36 };
      this.hit(left + 68, sy(visualY), visBoxW, 36, () => {
        this.openMultilineEditor(visRect, seg.visual || "", (v) => { seg.visual = v; }, [
          "[Shot 1]", "[Shot 2] At 00:03.500, the camera cuts to ",
          ...CAMERA_MOTIONS.map(m => `The camera ${m.toLowerCase()} with small amplitude at slow speed`),
          ...tags.map(t => t.label),
        ]);
      }, "Click to edit visual description and camera action for this segment.");

      // Speech Channel
      const speechY = visualY + 42;
      const speechEnabled = seg.speech?.enabled;
      this.button(ctx, left + 8, sy(speechY), 54, 20, "Speech", () => this.change(() => {
        if (!seg.speech) seg.speech = { enabled: false, speaker: "S1", language: "English", text: "" };
        seg.speech.enabled = !speechEnabled;
      }), false, "center", speechEnabled, false, "Enables spoken dialogue, speaker identity, and spoken language for this segment.");

      if (speechEnabled) {
        this.text(ctx, `(${seg.speech?.speaker || "S1"}) <d>[${seg.speech?.language || "English"}]`, left + 68, sy(speechY + 10), "#cbd5e1", "10px monospace");
        const spkBoxW = available - 210;
        this.box(ctx, left + 180, sy(speechY), spkBoxW, 20, "#121418", "#333842", 3);
        const spkText = seg.speech?.text || "Spoken dialogue...";
        this.text(ctx, spkText, left + 184, sy(speechY + 10), seg.speech?.text ? "#67e8f9" : "#64748b", "10px sans-serif", spkBoxW - 8);

        const spkRect = { x: left + 180, y: sy(speechY), w: spkBoxW, h: 20 };
        this.hit(left + 180, sy(speechY), spkBoxW, 20, () => {
          this.openMultilineEditor(spkRect, seg.speech?.text || "", (v) => {
            if (!seg.speech) seg.speech = { enabled: true, speaker: "S1", language: "English", text: "" };
            seg.speech.text = v;
          });
        }, "Click to edit spoken dialogue words.");
      }

      // Sounds Channel
      const audioY = speechY + 26;
      const soundsEnabled = seg.sounds?.enabled;
      this.button(ctx, left + 8, sy(audioY), 54, 20, "Sounds", () => this.change(() => {
        if (!seg.sounds) seg.sounds = { enabled: false, text: "" };
        seg.sounds.enabled = !soundsEnabled;
      }), false, "center", soundsEnabled, false, "Enables synchronized physical action sounds, impacts, and ambient effects for this segment.");

      if (soundsEnabled) {
        const sndBoxW = available - 80;
        this.box(ctx, left + 68, sy(audioY), sndBoxW, 20, "#121418", "#333842", 3);
        const sndText = seg.sounds?.text || "Diegetic sound events...";
        this.text(ctx, sndText, left + 72, sy(audioY + 10), seg.sounds?.text ? "#fed7aa" : "#64748b", "10px sans-serif", sndBoxW - 8);

        const sndRect = { x: left + 68, y: sy(audioY), w: sndBoxW, h: 20 };
        this.hit(left + 68, sy(audioY), sndBoxW, 20, () => {
          this.openMultilineEditor(sndRect, seg.sounds?.text || "", (v) => {
            if (!seg.sounds) seg.sounds = { enabled: true, text: "" };
            seg.sounds.text = v;
          });
        }, "Click to edit synchronized physical action sounds.");
      }

      cy += cardH + 8;
    });

    return cy;
  }

  drawSoundscapeTab(ctx, left, cy, available, sy) {
    this.text(ctx, "3. overall_soundscape", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
    cy += 22;

    this.text(ctx, "1–4 English sentences summarizing ambient, action, and physical sounds across the whole video.", left, sy(cy + 8), "#94a3b8", "11px sans-serif");
    cy += 20;

    // Quick snippets
    this.button(ctx, left, sy(cy), 105, 20, "Rain & Ambience", () => this.change(() => {
      this.state.overall_soundscape = "Steady rain taps against the café windows while low room ambience continues underneath. Wet footsteps echo softly.";
    }), false, "center", false, false, "Fills in a ready-made description for gentle indoor rain ambience and quiet room tone.");
    this.button(ctx, left + 110, sy(cy), 78, 20, "Room Tone", () => this.change(() => {
      this.state.overall_soundscape = "Quiet indoor room tone and a low ventilation hum continue throughout the video.";
    }), false, "center", false, false, "Fills in a ready-made description for subtle room tone and ventilation hum.");
    this.button(ctx, left + 192, sy(cy), 86, 20, "Urban Traffic", () => this.change(() => {
      this.state.overall_soundscape = "Distant city traffic hums with intermittent car horns and passing footsteps on the pavement.";
    }), false, "center", false, false, "Fills in a ready-made description for distant city traffic and street footsteps.");
    this.button(ctx, left + 282, sy(cy), 54, 20, "Set N/A", () => this.change(() => {
      this.state.overall_soundscape = "N/A";
    }), false, "center", false, false, "Sets soundscape to N/A for silent background audio.");
    cy += 28;

    const boxH = 120;
    this.box(ctx, left, sy(cy), available, boxH, "#121418", "#333842", 4);
    const textVal = this.state.overall_soundscape || "";
    const displayTxt = textVal || "Click to write overall soundscape summary...";
    this.drawMultilineText(ctx, displayTxt, left + 8, sy(cy + 8), available - 16, 16, textVal ? "#f8fafc" : "#64748b", "11px sans-serif");

    const rect = { x: left, y: sy(cy), w: available, h: boxH };
    this.hit(left, sy(cy), available, boxH, () => {
      this.openMultilineEditor(rect, this.state.overall_soundscape || "", (v) => { this.state.overall_soundscape = v; });
    }, "Click to edit overall background soundscape summary.");
    cy += boxH + 16;

    return cy;
  }

  drawMusicTab(ctx, left, cy, available, sy) {
    this.text(ctx, "4. non_diegetic_music", left, sy(cy + 10), "#93c5fd", "bold 12px sans-serif");
    cy += 22;

    this.text(ctx, "1–3 English sentences describing audience-only BGM (instrumentation, tempo, dynamics) or N/A.", left, sy(cy + 8), "#94a3b8", "11px sans-serif");
    cy += 20;

    // Quick snippets
    this.button(ctx, left, sy(cy), 100, 20, "Acoustic Guitar", () => this.change(() => {
      this.state.non_diegetic_music = "A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.";
    }), false, "center", false, false, "Fills in a ready-made description for gentle acoustic guitar background music.");
    this.button(ctx, left + 105, sy(cy), 100, 20, "Piano & Strings", () => this.change(() => {
      this.state.non_diegetic_music = "Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out.";
    }), false, "center", false, false, "Fills in a ready-made description for slow, emotive solo piano with subtle background strings.");
    this.button(ctx, left + 210, sy(cy), 78, 20, "Synth Pulse", () => this.change(() => {
      this.state.non_diegetic_music = "A low electronic pulse at a slow tempo, ending immediately after the action.";
    }), false, "center", false, false, "Fills in a ready-made description for a low atmospheric synthesizer rhythm.");
    this.button(ctx, left + 292, sy(cy), 54, 20, "Set N/A", () => this.change(() => {
      this.state.non_diegetic_music = "N/A";
    }), false, "center", false, false, "Sets background music to N/A when no music should play.");
    cy += 28;

    const boxH = 120;
    this.box(ctx, left, sy(cy), available, boxH, "#121418", "#333842", 4);
    const textVal = this.state.non_diegetic_music || "";
    const displayTxt = textVal || "Click to write non-diegetic music or set N/A...";
    this.drawMultilineText(ctx, displayTxt, left + 8, sy(cy + 8), available - 16, 16, textVal ? "#f8fafc" : "#64748b", "11px sans-serif");

    const rect = { x: left, y: sy(cy), w: available, h: boxH };
    this.hit(left, sy(cy), available, boxH, () => {
      this.openMultilineEditor(rect, this.state.non_diegetic_music || "", (v) => { this.state.non_diegetic_music = v; });
    }, "Click to edit audience background music description.");
    cy += boxH + 16;

    return cy;
  }

  mouse(event, position) {
    if (event.button !== 0 || !/up$/.test(event.type)) return true;
    this.clearHoveredTooltip();
    const region = this.hitRegions.find(item => this.contains(item, position[0], position[1]));
    region?.action(event, position);
    return true;
  }

  onWheel(event) {
    if (this.textEditor) this.closeTextEditor();
    this.clearHoveredTooltip();
    if (event.ctrlKey || event.metaKey || !this.state) return;
    const canvas = app.canvas;
    if (!canvas?.graph || this.contentHeight <= this.viewportHeight) return;
    const bounds = canvas.canvas.getBoundingClientRect();
    const graphX = (event.clientX - bounds.left) / canvas.ds.scale - canvas.ds.offset[0];
    const graphY = (event.clientY - bounds.top) / canvas.ds.scale - canvas.ds.offset[1];
    if (canvas.graph.getNodeOnPos(graphX, graphY, canvas.visible_nodes) !== this.node) return;
    const x = graphX - this.node.pos[0];
    const y = graphY - this.node.pos[1];
    if (x < 4 || x > this.node.size[0] - 4 || y < this.viewportY || y > this.viewportY + this.viewportHeight) return;
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.viewportHeight : 1);
    const next = Math.max(0, Math.min(this.contentHeight - this.viewportHeight, this.scroll + delta));
    if (next === this.scroll) return;
    this.scroll = next;
    event.preventDefault();
    event.stopImmediatePropagation();
    canvas.setDirty(true, true);
  }

  contains(region, x, y) {
    return x >= region.x && x <= region.x + region.w && y >= region.y && y <= region.y + region.h;
  }
}

app.registerExtension({
  name: "utils_collection.minimax_h3_base_prompt",
  getCustomWidgets() {
    return {
      UC_MINIMAX_H3_BASE_PROMPT_BUILDER(node, name, data) {
        new H3BaseCanvasPromptEditor(node, name, data);
        return { widget: node.widgets[node.widgets.length - 1] };
      },
    };
  },
});

export { H3BaseCanvasPromptEditor };
