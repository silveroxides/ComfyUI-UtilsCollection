function action(label, callback, { disabled = false, checked = false } = {}) {
  return { label, callback, disabled, checked };
}

export function buildLayerContextActions({
  index,
  count,
  placement,
  warpActive,
  rotateActive,
  moveBack,
  moveForward,
  sendToBack,
  bringToFront,
  flipHorizontal,
  flipVertical,
  toggleWarp,
  toggleRotate,
  toggleLock,
  exclude,
  reset,
  contentActions = [],
}) {
  return [
    action("Move Back", moveBack, { disabled: index <= 0 }),
    action("Move Forward", moveForward, { disabled: index >= count - 1 }),
    action("Send to Back", sendToBack, { disabled: index <= 0 }),
    action("Bring to Front", bringToFront, { disabled: index >= count - 1 }),
    { separator: true },
    action("Flip H", flipHorizontal, { checked: placement.flip_horizontal === true, disabled: placement.locked === true }),
    action("Flip V", flipVertical, { checked: placement.flip_vertical === true, disabled: placement.locked === true }),
    action("Warp", toggleWarp, { checked: warpActive, disabled: placement.locked === true }),
    action("Rotate", toggleRotate, { checked: rotateActive, disabled: placement.locked === true }),
    ...contentActions,
    action(placement.locked === true ? "Unlock" : "Lock", toggleLock, { checked: placement.locked === true }),
    action(placement.included === false ? "Include" : "Exclude", exclude),
    { separator: true },
    action("Reset", reset, { disabled: placement.locked === true }),
  ];
}

export class LayerContextMenu {
  constructor() {
    this.element = document.createElement("div");
    Object.assign(this.element.style, {
      position: "fixed",
      zIndex: "100000",
      display: "none",
      minWidth: "138px",
      padding: "3px",
      border: "1px solid rgba(255,255,255,.24)",
      borderRadius: "4px",
      color: "var(--input-text, #ddd)",
      background: "#202328",
      boxShadow: "0 4px 14px rgba(0,0,0,.45)",
      font: "12px sans-serif",
    });
    this.outsidePointer = (event) => {
      if (!this.element.contains(event.target)) this.close();
    };
    this.keyDown = (event) => {
      if (event.key === "Escape") this.close();
    };
    this.closeBound = () => this.close();
  }

  open(clientX, clientY, actions) {
    this.close();
    this.element.replaceChildren(...actions.map((item) => {
      if (item.separator) {
        const separator = document.createElement("div");
        Object.assign(separator.style, {
          height: "1px",
          margin: "3px 2px",
          background: "rgba(255,255,255,.14)",
        });
        return separator;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = item.disabled === true;
      button.textContent = `${item.checked ? "✓ " : ""}${item.label}`;
      Object.assign(button.style, {
        display: "block",
        width: "100%",
        height: "25px",
        padding: "1px 7px",
        border: "0",
        borderRadius: "3px",
        color: "inherit",
        background: "transparent",
        textAlign: "left",
        cursor: item.disabled ? "default" : "pointer",
        opacity: item.disabled ? ".4" : "1",
      });
      button.addEventListener("pointerenter", () => {
        if (!button.disabled) button.style.background = "rgba(101,201,255,.18)";
      });
      button.addEventListener("pointerleave", () => {
        button.style.background = "transparent";
      });
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        if (button.disabled) return;
        this.close();
        item.callback();
      });
      return button;
    }));
    document.body.append(this.element);
    this.element.style.display = "block";
    const bounds = this.element.getBoundingClientRect();
    const left = Math.max(2, Math.min(clientX, window.innerWidth - bounds.width - 2));
    const top = Math.max(2, Math.min(clientY, window.innerHeight - bounds.height - 2));
    this.element.style.left = `${left}px`;
    this.element.style.top = `${top}px`;
    document.addEventListener("pointerdown", this.outsidePointer, true);
    document.addEventListener("keydown", this.keyDown, true);
    window.addEventListener("resize", this.closeBound);
    window.addEventListener("scroll", this.closeBound, true);
  }

  close() {
    this.element.remove();
    this.element.style.display = "none";
    document.removeEventListener("pointerdown", this.outsidePointer, true);
    document.removeEventListener("keydown", this.keyDown, true);
    window.removeEventListener("resize", this.closeBound);
    window.removeEventListener("scroll", this.closeBound, true);
  }

  dispose() {
    this.close();
  }
}
