import { app } from "../../scripts/app.js";

let mermaidPromise = null;

function getMermaid() {
  if (window.mermaid) return Promise.resolve(window.mermaid);
  if (mermaidPromise) return mermaidPromise;
  mermaidPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = new URL("./vendor/mermaid.min.js", import.meta.url).href;
    script.onload = () => {
      try {
        window.mermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          securityLevel: "loose",
          fontFamily: "sans-serif",
          fontSize: 12,
        });
        resolve(window.mermaid);
      } catch (err) {
        reject(err);
      }
    };
    script.onerror = reject;
    document.head.appendChild(script);
  });
  return mermaidPromise;
}

const MERMAID_DIAGRAM_KEYWORDS = [
  "flowchart", "graph", "sequencediagram", "classdiagram",
  "erdiagram", "gantt", "statediagram", "pie", "mindmap",
  "journey", "quadrantchart", "requirementdiagram", "gitgraph",
  "c4context", "c4container", "c4component", "c4dynamic", "c4deployment",
  "timeline", "zenuml", "sankey", "xychart", "block", "packet",
  "kanban", "architecture",
];

function isMermaidSyntax(text) {
  if (!text || typeof text !== "string") return false;
  const firstLine = text.trim().split("\n")[0].trim().toLowerCase();
  return MERMAID_DIAGRAM_KEYWORDS.some(kw => firstLine.startsWith(kw));
}

function extractMermaidCode(text) {
  if (!text || typeof text !== "string") return "";
  const trimmed = text.trim();
  const fenceMatch = trimmed.match(/```mermaid\s*([\s\S]*?)```/i);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }
  const genericFence = trimmed.match(/```(?:[\w-]*)\s*([\s\S]*?)```/);
  if (genericFence) {
    const inside = genericFence[1].trim();
    if (isMermaidSyntax(inside)) return inside;
  }
  if (isMermaidSyntax(trimmed)) {
    return trimmed;
  }
  return "";
}

let renderCounter = 0;

async function renderMermaidToImage(code) {
  const mermaid = await getMermaid();
  const id = `uc_mermaid_${Date.now()}_${++renderCounter}`;
  const { svg } = await mermaid.render(id, code);

  const stray = document.getElementById(id) || document.getElementById(`d${id}`);
  if (stray) stray.remove();

  return new Promise((resolve, reject) => {
    const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ img, svg, width: img.naturalWidth || 800, height: img.naturalHeight || 600 });
    };
    img.onerror = e => {
      URL.revokeObjectURL(url);
      reject(e);
    };
    img.src = url;
  });
}

class MermaidCanvasWidget {
  constructor(node, name, isNote, initialText = "") {
    this.node = node;
    this.name = name;
    this.isNote = isNote;
    this.text = initialText;
    this.renderedImg = null;
    this.imgWidth = 0;
    this.imgHeight = 0;
    this.error = null;
    this.isRendering = false;
    this.renderSeq = 0;

    this.pan = { x: 0, y: 0 };
    this.zoom = 1.0;
    this.isDragging = false;
    this.dragStart = { x: 0, y: 0 };

    this.hitRegions = [];
    this.textEditor = null;
    this.outsidePointer = null;
    this.abort = new AbortController();

    this.widget = node.addCustomWidget({
      name,
      type: "custom",
      value: this.text,
      options: { socketless: true },
      computeSize: () => [node.size[0], Math.max(280, node.size[1] - 40)],
      draw: (ctx, _node, width, y) => this.draw(ctx, width, y),
      mouse: (event, position) => this.mouse(event, position),
      serializeValue: () => this.text,
    });
    this.widget.options.socketless = true;

    const getWidgetOnPos = node.getWidgetOnPos;
    node.getWidgetOnPos = (graphX, graphY, includeDisabled) => {
      const x = graphX - node.pos[0];
      const y = graphY - node.pos[1];
      if (this.hitRegions.some(region => this.contains(region, x, y))) return this.widget;
      if (this.viewport && this.contains(this.viewport, x, y)) return this.widget;
      const found = getWidgetOnPos?.call(node, graphX, graphY, includeDisabled);
      return found === this.widget ? undefined : found;
    };

    const canvasEl = app.canvas?.canvas;
    if (canvasEl) {
      canvasEl.addEventListener("pointermove", event => this.onPointerMove(event), {
        passive: true,
        signal: this.abort.signal,
      });
      canvasEl.addEventListener("pointerup", event => this.onPointerUp(event), {
        passive: true,
        signal: this.abort.signal,
      });
      canvasEl.addEventListener("wheel", event => this.onWheel(event), {
        capture: true,
        passive: false,
        signal: this.abort.signal,
      });
    }

    const onRemoved = node.onRemoved;
    node.onRemoved = (...args) => {
      this.abort.abort();
      this.closeTextEditor();
      node.getWidgetOnPos = getWidgetOnPos;
      onRemoved?.apply(node, args);
    };

    this.requestRender();
  }

  setText(newText) {
    if (this.text === newText) return;
    this.text = newText;
    this.widget.value = newText;
    if (this.isNote) {
      const textWidget = this.node.widgets?.find(w => w.name === "text");
      if (textWidget) textWidget.value = newText;
    }
    this.requestRender();
  }

  async requestRender() {
    const seq = ++this.renderSeq;
    const code = extractMermaidCode(this.text);
    if (!code) {
      this.renderedImg = null;
      this.error = this.text.trim()
        ? "No valid Mermaid diagram found (use ```mermaid ... ``` or raw syntax)."
        : null;
      this.node.setDirtyCanvas(true, true);
      return;
    }
    this.isRendering = true;
    this.error = null;
    this.node.setDirtyCanvas(true, true);
    try {
      const result = await renderMermaidToImage(code);
      if (seq !== this.renderSeq) return;
      this.renderedImg = result.img;
      this.imgWidth = result.width;
      this.imgHeight = result.height;
      this.error = null;
      this.autoFit();
    } catch (err) {
      if (seq !== this.renderSeq) return;
      this.renderedImg = null;
      this.error = (err?.message || String(err)).replace(/Error:\s*/, "");
    } finally {
      if (seq === this.renderSeq) {
        this.isRendering = false;
        this.node.setDirtyCanvas(true, true);
      }
    }
  }

  autoFit() {
    if (!this.imgWidth || !this.imgHeight || !this.viewport) return;
    const padding = 16;
    const availW = Math.max(20, this.viewport.w - padding * 2);
    const availH = Math.max(20, this.viewport.h - padding * 2);
    const scaleX = availW / this.imgWidth;
    const scaleY = availH / this.imgHeight;
    this.zoom = Math.min(scaleX, scaleY, 2.0);
    this.pan = {
      x: (this.viewport.w - this.imgWidth * this.zoom) / 2,
      y: (this.viewport.h - this.imgHeight * this.zoom) / 2,
    };
    this.node.setDirtyCanvas(true, true);
  }

  getDiagramTypeLabel() {
    const code = extractMermaidCode(this.text);
    if (!code) return "Mermaid Viewer";
    const firstWord = code.trim().split(/[\s\n(]/)[0].toLowerCase();
    const map = {
      flowchart: "Flowchart",
      graph: "Flowchart",
      sequencediagram: "Sequence Diagram",
      classdiagram: "Class Diagram",
      erdiagram: "ER Diagram",
      gantt: "Gantt Chart",
      statediagram: "State Diagram",
      "statediagram-v2": "State Diagram",
      pie: "Pie Chart",
      mindmap: "Mind Map",
      journey: "User Journey",
      quadrantchart: "Quadrant Chart",
      requirementdiagram: "Requirement Diagram",
      gitgraph: "Git Graph",
      c4context: "C4 Diagram",
      c4container: "C4 Container",
      c4component: "C4 Component",
      timeline: "Timeline Diagram",
      zenuml: "ZenUML Diagram",
      "sankey-beta": "Sankey Diagram",
      "xychart-beta": "XY Chart",
      "block-beta": "Block Diagram",
      "packet-beta": "Packet Diagram",
      kanban: "Kanban Board",
      "architecture-beta": "Architecture Diagram",
    };
    return map[firstWord] || "Mermaid Diagram";
  }

  contains(rect, x, y) {
    return x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h;
  }

  hit(x, y, w, h, action) {
    this.hitRegions.push({ x, y, w, h, action });
  }

  box(ctx, x, y, w, h, fill, stroke, r = 0, topRound = true, botRound = true) {
    ctx.save();
    ctx.beginPath();
    if (r > 0) {
      const rtl = topRound ? r : 0;
      const rtr = topRound ? r : 0;
      const rbr = botRound ? r : 0;
      const rbl = botRound ? r : 0;
      ctx.moveTo(x + rtl, y);
      ctx.lineTo(x + w - rtr, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + rtr);
      ctx.lineTo(x + w, y + h - rbr);
      ctx.quadraticCurveTo(x + w, y + h, x + w - rbr, y + h);
      ctx.lineTo(x + rbl, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - rbl);
      ctx.lineTo(x, y + rtl);
      ctx.quadraticCurveTo(x, y, x + rtl, y);
    } else {
      ctx.rect(x, y, w, h);
    }
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    ctx.restore();
  }

  button(ctx, x, y, w, h, label, action, accent = false, align = "center") {
    this.hit(x, y, w, h, action);
    const bg = accent ? "#2563eb" : "#242933";
    const stroke = accent ? "#3b82f6" : "#3b4252";
    const textCol = accent ? "#ffffff" : "#cbd5e1";
    this.box(ctx, x, y, w, h, bg, stroke, 3);
    ctx.save();
    ctx.font = "bold 10px sans-serif";
    ctx.fillStyle = textCol;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    const tx = align === "center" ? x + w / 2 : x + 6;
    ctx.fillText(label, tx, y + h / 2);
    ctx.restore();
  }

  drawMultilineText(ctx, text, x, y, maxWidth, lineHeight = 16, color = "#f87171") {
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const words = String(text).split(/\s+/);
    let line = "";
    let curY = y;
    for (const word of words) {
      const testLine = line + (line ? " " : "") + word;
      if (ctx.measureText(testLine).width > maxWidth && line) {
        ctx.fillText(line, x, curY);
        line = word;
        curY += lineHeight;
      } else {
        line = testLine;
      }
    }
    if (line) ctx.fillText(line, x, curY);
    ctx.restore();
  }

  draw(ctx, width, y) {
    if (this.textEditor) this.positionTextEditor();
    this.hitRegions = [];

    const nodeHeight = this.node.size[1] || 360;
    const headerHeight = 28;
    const totalHeight = Math.max(240, nodeHeight - y - 10);
    const vpX = 6;
    const vpY = y + headerHeight;
    const vpW = width - 12;
    const vpH = totalHeight - headerHeight - 4;
    this.viewport = { x: vpX, y: vpY, w: vpW, h: vpH };

    this.box(ctx, 4, y, width - 8, totalHeight, "#131519", "#2a2e38", 6);
    this.box(ctx, 4, y, width - 8, headerHeight, "#1a1d24", "#2a2e38", 6, true, false);

    ctx.save();
    ctx.font = "bold 11px sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const diagramType = this.getDiagramTypeLabel();
    ctx.fillText(diagramType, 12, y + headerHeight / 2);
    ctx.restore();

    let btnRight = width - 12;

    const fitW = 34;
    this.button(ctx, btnRight - fitW, y + 4, fitW, 20, "Fit", () => this.autoFit(), false);
    btnRight -= fitW + 4;

    const zoomW = 30;
    this.button(ctx, btnRight - zoomW, y + 4, zoomW, 20, "1:1", () => {
      this.zoom = 1.0;
      if (this.imgWidth && this.viewport) {
        this.pan = {
          x: (this.viewport.w - this.imgWidth) / 2,
          y: (this.viewport.h - this.imgHeight) / 2,
        };
      }
      this.node.setDirtyCanvas(true, true);
    }, false);
    btnRight -= zoomW + 4;

    if (this.isNote) {
      const editW = 40;
      this.button(ctx, btnRight - editW, y + 4, editW, 20, "Edit", () => this.openTextEditor(), true);
      btnRight -= editW + 4;
    }

    ctx.save();
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`${Math.round(this.zoom * 100)}%`, btnRight - 8, y + headerHeight / 2);
    ctx.restore();

    this.box(ctx, vpX, vpY, vpW, vpH, "#0d0f12", "#1f232b", 4);

    ctx.save();
    ctx.beginPath();
    ctx.rect(vpX, vpY, vpW, vpH);
    ctx.clip();

    if (this.renderedImg) {
      ctx.translate(vpX + this.pan.x, vpY + this.pan.y);
      ctx.scale(this.zoom, this.zoom);
      ctx.drawImage(this.renderedImg, 0, 0);
    } else if (this.isRendering) {
      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#38bdf8";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Rendering Mermaid diagram...", vpX + vpW / 2, vpY + vpH / 2);
    } else if (this.error) {
      this.drawMultilineText(ctx, this.error, vpX + 16, vpY + vpH / 2 - 20, vpW - 32, 16, "#f87171");
    } else {
      ctx.font = "11px sans-serif";
      ctx.fillStyle = "#475569";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const hint = this.isNote
        ? "Click 'Edit' to enter Markdown or Mermaid code."
        : "Waiting for incoming diagram text...";
      ctx.fillText(hint, vpX + vpW / 2, vpY + vpH / 2);
    }

    ctx.restore();
  }

  mouse(event, position) {
    const x = position[0];
    const y = position[1];

    if (event.type === "pointerdown" || event.type === "mousedown") {
      const hit = this.hitRegions.find(r => this.contains(r, x, y));
      if (hit) {
        hit.action(event, position);
        return true;
      }
      if (this.viewport && this.contains(this.viewport, x, y)) {
        this.isDragging = true;
        this.dragStart = { x, y };
        return true;
      }
    }
    return true;
  }

  onPointerMove(event) {
    if (!this.isDragging) return;
    const canvas = app.canvas;
    if (!canvas?.graph) return;
    const bounds = canvas.canvas.getBoundingClientRect();
    const scale = canvas.ds.scale;
    const graphX = (event.clientX - bounds.left) / scale - canvas.ds.offset[0];
    const graphY = (event.clientY - bounds.top) / scale - canvas.ds.offset[1];
    const x = graphX - this.node.pos[0];
    const y = graphY - this.node.pos[1];

    const dx = x - this.dragStart.x;
    const dy = y - this.dragStart.y;
    this.pan.x += dx;
    this.pan.y += dy;
    this.dragStart = { x, y };
    this.node.setDirtyCanvas(true, true);
  }

  onPointerUp() {
    if (this.isDragging) {
      this.isDragging = false;
      this.node.setDirtyCanvas(true, true);
    }
  }

  onWheel(event) {
    if (this.textEditor) return;
    const canvas = app.canvas;
    if (!canvas?.graph) return;
    const bounds = canvas.canvas.getBoundingClientRect();
    const scale = canvas.ds.scale;
    const graphX = (event.clientX - bounds.left) / scale - canvas.ds.offset[0];
    const graphY = (event.clientY - bounds.top) / scale - canvas.ds.offset[1];
    if (canvas.graph.getNodeOnPos(graphX, graphY, canvas.visible_nodes) !== this.node) return;
    const x = graphX - this.node.pos[0];
    const y = graphY - this.node.pos[1];

    if (this.viewport && this.contains(this.viewport, x, y)) {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();

      const factor = event.deltaY < 0 ? 1.15 : 0.85;
      const newZoom = Math.max(0.05, Math.min(10.0, this.zoom * factor));
      const relX = x - (this.viewport.x + this.pan.x);
      const relY = y - (this.viewport.y + this.pan.y);
      this.pan.x -= relX * (factor - 1);
      this.pan.y -= relY * (factor - 1);
      this.zoom = newZoom;
      this.node.setDirtyCanvas(true, true);
    }
  }

  openTextEditor() {
    if (this.textEditor) return;
    const container = document.createElement("div");
    Object.assign(container.style, {
      position: "fixed",
      zIndex: "1000",
      background: "#181a1f",
      border: "1px solid #3b82f6",
      borderRadius: "6px",
      boxShadow: "0 10px 25px rgba(0,0,0,0.6)",
      display: "flex",
      flexDirection: "column",
      padding: "8px",
      boxSizing: "border-box",
    });

    const header = document.createElement("div");
    Object.assign(header.style, {
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      marginBottom: "6px",
      color: "#94a3b8",
      fontSize: "11px",
      fontFamily: "sans-serif",
    });
    header.innerHTML = "<span><b>Mermaid / Markdown Editor</b> (Ctrl+Enter to close, Esc to cancel)</span>";

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    Object.assign(closeBtn.style, {
      background: "none",
      border: "none",
      color: "#94a3b8",
      cursor: "pointer",
      fontSize: "14px",
      padding: "0 4px",
    });
    closeBtn.onclick = () => this.closeTextEditor();
    header.appendChild(closeBtn);
    container.appendChild(header);

    const textarea = document.createElement("textarea");
    textarea.value = this.text;
    Object.assign(textarea.style, {
      width: "100%",
      height: "100%",
      flex: "1",
      boxSizing: "border-box",
      background: "#0d0f12",
      color: "#f8fafc",
      border: "1px solid #2a2e38",
      outline: "none",
      borderRadius: "4px",
      padding: "8px",
      fontSize: "12px",
      fontFamily: "monospace",
      resize: "none",
      lineHeight: "1.4",
    });
    container.appendChild(textarea);

    this.textEditor = { container, textarea };

    this.outsidePointer = event => {
      if (this.textEditor?.container && !this.textEditor.container.contains(event.target)) {
        this.closeTextEditor();
      }
    };
    requestAnimationFrame(() => {
      document.addEventListener("pointerdown", this.outsidePointer, true);
    });

    textarea.addEventListener("input", () => {
      this.setText(textarea.value);
    });

    textarea.addEventListener("keydown", event => {
      event.stopPropagation();
      if (event.key === "Escape") {
        event.preventDefault();
        this.closeTextEditor();
      } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        this.closeTextEditor();
      }
    });

    document.body.appendChild(container);
    this.positionTextEditor();
    requestAnimationFrame(() => textarea.focus());
  }

  closeTextEditor() {
    if (this.outsidePointer) {
      document.removeEventListener("pointerdown", this.outsidePointer, true);
      this.outsidePointer = null;
    }
    if (this.textEditor) {
      this.textEditor.container?.remove();
      this.textEditor = null;
      this.node.setDirtyCanvas(true, true);
    }
  }

  positionTextEditor() {
    if (!this.textEditor || !app.canvas?.canvas) return;
    const { container } = this.textEditor;
    const canvas = app.canvas.canvas;
    const canvasRect = canvas.getBoundingClientRect();
    const scale = app.canvas.ds.scale;
    const x = canvasRect.left + (this.node.pos[0] + app.canvas.ds.offset[0]) * scale;
    const y = canvasRect.top + (this.node.pos[1] + app.canvas.ds.offset[1]) * scale;
    const w = Math.max(380, this.node.size[0] * scale);
    const h = Math.max(260, this.node.size[1] * scale);

    Object.assign(container.style, {
      left: `${Math.max(10, Math.min(window.innerWidth - w - 10, x))}px`,
      top: `${Math.max(10, Math.min(window.innerHeight - h - 10, y))}px`,
      width: `${w}px`,
      height: `${h}px`,
    });
  }
}

app.registerExtension({
  name: "UtilsCollection.MermaidCanvas",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UC_MermaidNote" && nodeData.name !== "UC_MermaidPreview") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const isNote = nodeData.name === "UC_MermaidNote";

      if (isNote) {
        const textWidget = this.widgets?.find(w => w.name === "text");
        if (textWidget) {
          textWidget.type = "converted-widget";
          textWidget.computeSize = () => [0, -4];
        }
      }

      const initialText = isNote
        ? (this.widgets?.find(w => w.name === "text")?.value ?? "")
        : "";

      const editor = new MermaidCanvasWidget(this, isNote ? "mermaid_note" : "mermaid_preview", isNote, initialText);
      this.__mermaidEditor = editor;
      this.setSize([
        Math.max(this.size[0], 420),
        Math.max(this.size[1], 360),
      ]);
      return result;
    };

    if (nodeData.name === "UC_MermaidPreview") {
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const mermaidText = message?.mermaid_text;
        const val = Array.isArray(mermaidText) ? mermaidText[0] : mermaidText;
        if (this.__mermaidEditor && val !== undefined) {
          this.__mermaidEditor.setText(String(val));
        }
      };
    }
  },
});
