import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

class MockElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.style = {};
    this.children = [];
    this.listeners = new Map();
    this.textContent = "";
    this._innerHTML = "";
    this.value = "";
    this.scrollHeight = 100;
    this.clientHeight = 100;
    this.scrollWidth = 100;
    this.clientWidth = 100;
    this._svgChild = null;
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set innerHTML(html) {
    this._innerHTML = String(html);
    this.children = [];
    this._svgChild = this._innerHTML.includes("<svg") ? new MockElement("svg") : null;
  }

  appendChild(child) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...nodes) {
    this.children = [...nodes];
    this._innerHTML = "";
    this._svgChild = null;
  }

  querySelector(selector) {
    if (selector === "svg") return this._svgChild;
    return null;
  }

  addEventListener(type, handler, options) {
    const list = this.listeners.get(type) || [];
    list.push({ handler, options });
    this.listeners.set(type, list);
  }

  dispatch(type, event = {}) {
    const list = this.listeners.get(type) || [];
    for (const entry of list) {
      entry.handler(event);
    }
  }

  focus() {
    this.focused = true;
  }
}

async function loadMermaidExtension(mermaidImpl) {
  const source = await readFile(new URL("../web/mermaid_canvas.js", import.meta.url), "utf8");
  let extension = null;
  const canvasListeners = [];
  const strayNodes = new Map();

  const sandbox = {
    URL,
    Promise,
    Date,
    Math,
    String,
    Array,
    Object,
    mermaid: mermaidImpl,
    app: {
      canvas: {
        canvas: {
          addEventListener(type) {
            canvasListeners.push(type);
          },
          getBoundingClientRect() {
            throw new Error("getBoundingClientRect should never be called on graph canvas");
          },
        },
      },
      registerExtension(ext) {
        extension = ext;
      },
    },
    document: {
      head: new MockElement("head"),
      createElement(tag) {
        return new MockElement(tag);
      },
      getElementById(id) {
        return strayNodes.get(id) || null;
      },
    },
  };
  sandbox.globalThis = sandbox;

  const moduleUrl = JSON.stringify(new URL("../web/mermaid_canvas.js", import.meta.url).href);
  const instrumented =
    source
      .replace(/^import[\s\S]*?;\r?$/gm, "")
      .replace(/import\.meta\.url/g, moduleUrl) +
    "\nglobalThis.__exports = { isMermaidSyntax, extractMermaidCode, getDiagramTypeLabel, MermaidDOMWidget };";
  vm.runInNewContext(instrumented, sandbox);

  return { extension, canvasListeners, strayNodes, exports: sandbox.__exports };
}

function createNodeClass(initialWidgets = []) {
  return class MockNode {
    constructor() {
      this.size = [300, 200];
      this.widgets = initialWidgets.map(w => ({ ...w }));
      this.domWidgets = [];
    }

    setSize(size) {
      this.size = size;
    }

    addDOMWidget(name, type, element, options) {
      const widget = { name, type, element, options };
      this.domWidgets.push(widget);
      return widget;
    }
  };
}

test("extractMermaidCode handles fenced blocks, raw diagrams, and plain Markdown notes", async () => {
  const { exports } = await loadMermaidExtension({ async render() { return { svg: "<svg></svg>" }; } });
  const { extractMermaidCode, isMermaidSyntax, getDiagramTypeLabel } = exports;

  assert.equal(isMermaidSyntax("flowchart TD\nA --> B"), true);
  assert.equal(isMermaidSyntax("## Plain Markdown Heading"), false);
  assert.equal(extractMermaidCode("Notes\n```mermaid\ngraph LR\nA-->B\n```\nFooter"), "graph LR\nA-->B");
  assert.equal(extractMermaidCode("## Plain wiring guide\nA -> B"), "");
  assert.equal(getDiagramTypeLabel("sequenceDiagram\nA->>B: ping"), "Sequence Diagram");
  assert.equal(getDiagramTypeLabel("## Plain wiring guide"), "Text / Markdown");
});

test("UC_MermaidNote mounts static DOM widget with zero graph canvas listeners and syncs edits", async () => {
  const renderedCodes = [];
  const { extension, canvasListeners } = await loadMermaidExtension({
    async render(id, code) {
      renderedCodes.push({ id, code });
      return { svg: `<svg data-id="${id}"><g>${code}</g></svg>` };
    },
  });

  const NoteNode = createNodeClass([
    { name: "text", type: "customtext", value: "```mermaid\nflowchart TD\nA-->B\n```", element: new MockElement("textarea") },
  ]);
  await extension.beforeRegisterNodeDef(NoteNode, { name: "UC_MermaidNote" });

  const node = new NoteNode();
  node.onNodeCreated();
  await new Promise(resolve => setImmediate(resolve));

  assert.deepEqual(canvasListeners, []);
  assert.equal(node.domWidgets.length, 1);
  assert.equal(node.domWidgets[0].type, "uc_mermaid_dom");
  assert.equal(node.domWidgets[0].options.serialize, false);
  assert.equal(node.widgets[0].type, "customtext");
  assert.deepEqual(Array.from(node.widgets[0].computeSize()), [0, -4]);
  assert.equal(node.widgets[0].element.style.display, "none");
  assert.equal(renderedCodes.length, 1);
  assert.equal(renderedCodes[0].code, "flowchart TD\nA-->B");
  assert.ok(node.__mermaidEditor.stageEl.innerHTML.includes("<svg"));

  // Entering Edit mode and typing updates hidden text widget without re-running mermaid.render per keystroke
  node.__mermaidEditor.toggleBtn.dispatch("click", { stopPropagation() {} });
  assert.equal(node.__mermaidEditor.editing, true);
  node.__mermaidEditor.editorEl.value = "graph LR\nX-->Y";
  node.__mermaidEditor.editorEl.dispatch("input");
  assert.equal(node.widgets[0].value, "graph LR\nX-->Y");
  assert.equal(renderedCodes.length, 1);

  // Returning to Preview mode renders once
  node.__mermaidEditor.toggleBtn.dispatch("click", { stopPropagation() {} });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(node.__mermaidEditor.editing, false);
  assert.equal(renderedCodes.length, 2);
  assert.equal(renderedCodes[1].code, "graph LR\nX-->Y");

  // Reloading saved workflow via onConfigure restores text and renders plain Markdown fallback cleanly
  node.widgets[0].value = "## Saved plain note\nStep 1 -> Step 2";
  node.onConfigure({});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(renderedCodes.length, 2);
  assert.equal(node.__mermaidEditor.stageEl.children[0].textContent, "## Saved plain note\nStep 1 -> Step 2");
});

test("UC_MermaidPreview updates on execution and ignores late renders after node removal", async () => {
  let resolveRender;
  const { extension, canvasListeners } = await loadMermaidExtension({
    render(_id, code) {
      return new Promise(resolve => {
        resolveRender = () => resolve({ svg: `<svg>${code}</svg>` });
      });
    },
  });

  const PreviewNode = createNodeClass();
  await extension.beforeRegisterNodeDef(PreviewNode, { name: "UC_MermaidPreview" });

  const node = new PreviewNode();
  node.onNodeCreated();
  assert.deepEqual(canvasListeners, []);
  assert.equal(node.domWidgets[0].options.serialize, true);

  node.onExecuted({ mermaid_text: ["pie title Load\n\"GPU\" : 80"] });
  await new Promise(resolve => setImmediate(resolve));
  node.onRemoved();
  resolveRender();
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(node.__mermaidEditor.disposed, true);
  assert.equal(node.__mermaidEditor.stageEl.innerHTML, "");
});
