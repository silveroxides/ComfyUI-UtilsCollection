import { app } from "../../scripts/app.js";

let mermaidPromise = null;

function getMermaid() {
  if (globalThis.mermaid) return Promise.resolve(globalThis.mermaid);
  if (mermaidPromise) return mermaidPromise;
  mermaidPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = new URL("./vendor/mermaid.min.js", import.meta.url).href;
    script.onload = () => {
      try {
        globalThis.mermaid.initialize({
          startOnLoad: false,
          theme: "dark",
          securityLevel: "strict",
          fontFamily: "sans-serif",
          fontSize: 12,
        });
        resolve(globalThis.mermaid);
      } catch (err) {
        mermaidPromise = null;
        reject(err);
      }
    };
    script.onerror = err => {
      mermaidPromise = null;
      reject(err);
    };
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

const DIAGRAM_LABELS = {
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

function getDiagramTypeLabel(text) {
  const code = extractMermaidCode(text);
  if (!code) {
    return text && text.trim() ? "Text / Markdown" : "Mermaid Viewer";
  }
  const firstWord = code.trim().split(/[\s\n(]/)[0].toLowerCase();
  return DIAGRAM_LABELS[firstWord] || "Mermaid Diagram";
}

let renderCounter = 0;

class MermaidDOMWidget {
  constructor(node, name, isNote, textWidget = null, initialText = "") {
    this.node = node;
    this.name = name;
    this.isNote = isNote;
    this.textWidget = textWidget;
    this.text = String(initialText ?? "");
    this.editing = false;
    this.disposed = false;
    this.renderSeq = 0;

    this.buildDOM();
    this.installWidget();
    this.requestRender();
  }

  buildDOM() {
    const root = document.createElement("div");
    Object.assign(root.style, {
      display: "flex",
      flexDirection: "column",
      width: "100%",
      height: "100%",
      boxSizing: "border-box",
      background: "#131519",
      border: "1px solid #2a2e38",
      borderRadius: "6px",
      overflow: "hidden",
      color: "#e2e8f0",
      fontFamily: "sans-serif",
    });

    const header = document.createElement("div");
    Object.assign(header.style, {
      height: "28px",
      flexShrink: "0",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 8px",
      background: "#1a1d24",
      borderBottom: "1px solid #2a2e38",
      boxSizing: "border-box",
    });

    const titleEl = document.createElement("span");
    Object.assign(titleEl.style, {
      fontSize: "11px",
      fontWeight: "600",
      color: "#94a3b8",
      overflow: "hidden",
      textOverflow: "ellipsis",
      whiteSpace: "nowrap",
    });
    titleEl.textContent = getDiagramTypeLabel(this.text);
    header.appendChild(titleEl);

    let toggleBtn = null;
    if (this.isNote) {
      toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      toggleBtn.textContent = "Edit";
      Object.assign(toggleBtn.style, {
        background: "#2563eb",
        border: "1px solid #3b82f6",
        borderRadius: "3px",
        color: "#ffffff",
        fontSize: "10px",
        fontWeight: "600",
        padding: "2px 8px",
        cursor: "pointer",
      });
      toggleBtn.addEventListener("pointerdown", event => event.stopPropagation());
      toggleBtn.addEventListener("click", event => {
        event.stopPropagation();
        this.setEditing(!this.editing);
      });
      header.appendChild(toggleBtn);
    }

    const stageEl = document.createElement("div");
    Object.assign(stageEl.style, {
      flex: "1",
      minHeight: "0",
      overflow: "auto",
      padding: "10px",
      boxSizing: "border-box",
      background: "#0d0f12",
      display: "flex",
      flexDirection: "column",
    });

    let editorEl = null;
    if (this.isNote) {
      editorEl = document.createElement("textarea");
      editorEl.value = this.text;
      editorEl.spellcheck = false;
      Object.assign(editorEl.style, {
        display: "none",
        flex: "1",
        width: "100%",
        height: "100%",
        minHeight: "0",
        resize: "none",
        border: "none",
        outline: "none",
        background: "#0d0f12",
        color: "#f8fafc",
        fontFamily: "monospace",
        fontSize: "12px",
        lineHeight: "1.45",
        padding: "10px",
        boxSizing: "border-box",
      });
      editorEl.addEventListener("pointerdown", event => event.stopPropagation());
      editorEl.addEventListener("input", () => {
        this.syncTextValue(editorEl.value);
      });
      editorEl.addEventListener("keydown", event => {
        event.stopPropagation();
        if (event.key === "Escape" || (event.key === "Enter" && (event.ctrlKey || event.metaKey))) {
          event.preventDefault();
          this.setEditing(false);
        }
      });
    }

    root.addEventListener(
      "wheel",
      event => {
        if (event.ctrlKey || event.metaKey) return;
        const activeScroller = this.editing && editorEl ? editorEl : stageEl;
        const canScrollY = activeScroller.scrollHeight > activeScroller.clientHeight;
        const canScrollX = activeScroller.scrollWidth > activeScroller.clientWidth;
        if (canScrollY || canScrollX) {
          event.stopPropagation();
        }
      },
      { passive: true },
    );

    root.appendChild(header);
    root.appendChild(stageEl);
    if (editorEl) root.appendChild(editorEl);

    this.root = root;
    this.titleEl = titleEl;
    this.toggleBtn = toggleBtn;
    this.stageEl = stageEl;
    this.editorEl = editorEl;
  }

  installWidget() {
    if (this.textWidget) {
      this.textWidget.computeSize = () => [0, -4];
      this.textWidget.draw = () => {};
      if (this.textWidget.element) {
        this.textWidget.element.style.display = "none";
      }
    }

    const serializePreview = !this.isNote;
    this.domWidget = this.node.addDOMWidget(this.name, "uc_mermaid_dom", this.root, {
      serialize: serializePreview,
      hideOnZoom: false,
      getMinHeight: () => 220,
      getValue: () => this.text,
      setValue: value => this.setText(String(value ?? "")),
    });
    if (this.domWidget) {
      this.domWidget.serialize = serializePreview;
    }
    if (serializePreview) {
      this.node.serialize_widgets = true;
    }

    const onRemoved = this.node.onRemoved;
    this.node.onRemoved = (...args) => {
      this.dispose();
      return onRemoved?.apply(this.node, args);
    };
  }

  dispose() {
    this.disposed = true;
    this.renderSeq++;
  }

  syncTextValue(newText) {
    const normalized = String(newText ?? "");
    this.text = normalized;
    if (this.textWidget && this.textWidget.value !== normalized) {
      this.textWidget.value = normalized;
    }
    if (!this.isNote && this.domWidget) {
      this.domWidget.value = normalized;
    }
    this.titleEl.textContent = getDiagramTypeLabel(normalized);
  }

  setEditing(nextEditing) {
    if (!this.isNote || !this.editorEl || this.editing === nextEditing) return;
    this.editing = nextEditing;
    if (this.editing) {
      this.editorEl.value = this.text;
      this.stageEl.style.display = "none";
      this.editorEl.style.display = "block";
      if (this.toggleBtn) this.toggleBtn.textContent = "Preview";
      this.editorEl.focus?.();
    } else {
      this.syncTextValue(this.editorEl.value);
      this.editorEl.style.display = "none";
      this.stageEl.style.display = "flex";
      if (this.toggleBtn) this.toggleBtn.textContent = "Edit";
      this.requestRender();
    }
  }

  setText(newText, forceRender = false) {
    const normalized = String(newText ?? "");
    const changed = normalized !== this.text;
    this.syncTextValue(normalized);
    if (this.editorEl && this.editorEl.value !== normalized) {
      this.editorEl.value = normalized;
    }
    if ((changed || forceRender) && !this.editing) {
      return this.requestRender();
    }
    return Promise.resolve();
  }

  renderPlainBlock(content, color = "#e2e8f0", centered = false) {
    this.stageEl.replaceChildren();
    const block = document.createElement("pre");
    Object.assign(block.style, {
      margin: centered ? "auto" : "0",
      whiteSpace: "pre-wrap",
      wordBreak: "break-word",
      color,
      fontFamily: centered ? "sans-serif" : "monospace",
      fontSize: centered ? "11px" : "12px",
      lineHeight: "1.45",
      textAlign: centered ? "center" : "left",
    });
    block.textContent = content;
    this.stageEl.appendChild(block);
  }

  async requestRender() {
    if (this.disposed) return;
    const seq = ++this.renderSeq;
    this.titleEl.textContent = getDiagramTypeLabel(this.text);

    const trimmed = this.text.trim();
    if (!trimmed) {
      const hint = this.isNote
        ? "Click 'Edit' to enter Mermaid diagram code or Markdown notes."
        : "Waiting for incoming diagram text...";
      this.renderPlainBlock(hint, "#64748b", true);
      return;
    }

    const code = extractMermaidCode(this.text);
    if (!code) {
      this.renderPlainBlock(this.text, "#e2e8f0", false);
      return;
    }

    this.renderPlainBlock("Rendering Mermaid diagram...", "#38bdf8", true);
    try {
      const mermaid = await getMermaid();
      if (this.disposed || seq !== this.renderSeq) return;
      const id = `uc_mermaid_${Date.now()}_${++renderCounter}`;
      const { svg } = await mermaid.render(id, code);
      const stray = document.getElementById?.(id) || document.getElementById?.(`d${id}`);
      stray?.remove?.();
      if (this.disposed || seq !== this.renderSeq) return;

      this.stageEl.innerHTML = svg;
      const svgEl = this.stageEl.querySelector?.("svg");
      if (svgEl?.style) {
        svgEl.style.maxWidth = "100%";
        svgEl.style.height = "auto";
        svgEl.style.display = "block";
        svgEl.style.margin = "auto";
      }
    } catch (err) {
      if (this.disposed || seq !== this.renderSeq) return;
      const message = (err?.message || String(err)).replace(/^Error:\s*/, "");
      this.renderPlainBlock(message, "#f87171", false);
    }
  }
}

app.registerExtension({
  name: "UtilsCollection.MermaidCanvas",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UC_MermaidNote" && nodeData.name !== "UC_MermaidPreview") return;

    const isNote = nodeData.name === "UC_MermaidNote";

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const textWidget = isNote ? (this.widgets?.find(w => w.name === "text") ?? null) : null;
      const initialText = textWidget ? (textWidget.value ?? "") : "";

      this.__mermaidEditor = new MermaidDOMWidget(
        this,
        isNote ? "mermaid_note" : "mermaid_preview",
        isNote,
        textWidget,
        initialText,
      );
      this.setSize([
        Math.max(this.size?.[0] || 0, 420),
        Math.max(this.size?.[1] || 0, 320),
      ]);
      return result;
    };

    if (isNote) {
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function (info) {
        const result = onConfigure?.apply(this, arguments);
        const savedText = this.widgets?.find(w => w.name === "text")?.value;
        if (this.__mermaidEditor && savedText !== undefined) {
          this.__mermaidEditor.setText(String(savedText));
        }
        return result;
      };
    } else {
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const result = onExecuted?.apply(this, arguments);
        const mermaidText = message?.mermaid_text;
        const val = Array.isArray(mermaidText) ? mermaidText[0] : mermaidText;
        if (this.__mermaidEditor && val !== undefined) {
          this.__mermaidEditor.setText(String(val));
        }
        return result;
      };
    }
  },
});
