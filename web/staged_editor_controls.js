export const STAGED_INPUT_STYLE = Object.freeze({
  boxSizing: "border-box", height: "28px", border: "1px solid rgba(255,255,255,.2)",
  borderRadius: "4px", color: "inherit", background: "rgba(0,0,0,.25)", font: "inherit",
});

export function bindHoldRepeat(button, change, {
  captureTarget = button, enabled = () => !button.disabled,
} = {}) {
  let timer = 0, pointerId = null, pointerClick = false;
  const stop = () => {
    clearTimeout(timer); timer = 0;
    const id = pointerId; pointerId = null;
    captureTarget.removeEventListener("pointerup", finish);
    captureTarget.removeEventListener("pointercancel", finish);
    captureTarget.removeEventListener("lostpointercapture", finish);
    window.removeEventListener("blur", stop);
    if (id !== null && captureTarget.hasPointerCapture?.(id)) captureTarget.releasePointerCapture(id);
  };
  const finish = (event) => { if (event.pointerId === pointerId) stop(); };
  const tick = () => {
    if (pointerId === null || captureTarget.isConnected === false || !enabled()) { stop(); return; }
    if (change() === false) { stop(); return; }
    if (pointerId !== null) timer = setTimeout(tick, 65);
  };
  button.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.isPrimary === false || !enabled()) return;
    event.preventDefault(); event.stopPropagation();
    stop();
    pointerId = event.pointerId; pointerClick = true;
    captureTarget.setPointerCapture(pointerId);
    captureTarget.addEventListener("pointerup", finish);
    captureTarget.addEventListener("pointercancel", finish);
    captureTarget.addEventListener("lostpointercapture", finish);
    window.addEventListener("blur", stop);
    if (change() === false) { stop(); return; }
    if (pointerId !== null) timer = setTimeout(tick, 350);
  });
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    if (event.detail > 0 && pointerClick) { pointerClick = false; return; }
    if (enabled()) change();
  });
  return stop;
}

export function createStagedControlRow(label, tooltip) {
  const root = document.createElement("div");
  Object.assign(root.style, {
    boxSizing: "border-box", display: "flex", flex: "0 0 auto", height: "34px",
    alignItems: "center", gap: "2px", padding: "1px 2px",
    border: "1px solid rgba(255,255,255,.22)", borderRadius: "6px", background: "rgba(0,0,0,.16)",
  });
  root.title = tooltip;
  const caption = document.createElement("span");
  Object.assign(caption.style, { flex: "0 0 auto", opacity: ".78", whiteSpace: "nowrap" });
  caption.textContent = label;
  root.append(caption);
  return { root, caption };
}

export function createStagedAction(text, title, callback) {
  const button = document.createElement("button");
  Object.assign(button.style, STAGED_INPUT_STYLE, { flex: "0 0 auto", padding: "1px 7px", cursor: "pointer" });
  button.type = "button";
  button.textContent = text;
  button.title = title;
  button.setAttribute("aria-label", title);
  button.addEventListener("click", (event) => { event.stopPropagation(); callback(); });
  return button;
}

export function createStagedNumericControl(label, tooltip, ariaLabel, step) {
  const { root, caption } = createStagedControlRow(label, tooltip);
  const stepButton = (text, title) => {
    const button = document.createElement("button");
    Object.assign(button.style, STAGED_INPUT_STYLE, {
      flex: "0 0 22px", width: "22px", padding: "0", cursor: "pointer", fontSize: "14px", lineHeight: "1",
    });
    button.type = "button"; button.textContent = text; button.title = title;
    button.setAttribute("aria-label", title);
    return button;
  };
  const decrement = stepButton("◀", `${label}: decrease by ${step}`);
  const increment = stepButton("▶", `${label}: increase by ${step}`);
  const input = document.createElement("input");
  Object.assign(input.style, STAGED_INPUT_STYLE, { flex: "0 0 56px", minWidth: "56px", width: "56px", textAlign: "center" });
  input.type = "text"; input.inputMode = "decimal";
  input.setAttribute("aria-label", ariaLabel);
  root.append(decrement, input, increment);
  return { root, caption, input, decrement, increment };
}
