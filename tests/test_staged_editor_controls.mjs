import test from "node:test";
import assert from "node:assert/strict";
import { bindHoldRepeat } from "../web/staged_editor_controls.js";

class Target extends EventTarget {
  isConnected = true;
  disabled = false;
  setPointerCapture(id) { this.pointer = id; }
  hasPointerCapture(id) { return this.pointer === id; }
  releasePointerCapture() { this.pointer = null; }
}
const fire = (target, type, values = {}) => target.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), values));

test("held arrows repeat after a delay and release does not add a duplicate click", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  globalThis.window = new EventTarget();
  const button = new Target();
  let count = 0;
  const stop = bindHoldRepeat(button, () => { count++; });
  t.after(stop);
  fire(button, "pointerdown", { button: 0, pointerId: 1 });
  assert.equal(count, 1);
  t.mock.timers.tick(349);
  assert.equal(count, 1);
  t.mock.timers.tick(1);
  t.mock.timers.tick(65);
  assert.equal(count, 3);
  fire(button, "pointerup", { pointerId: 1 });
  fire(button, "click", { detail: 1 });
  t.mock.timers.tick(1000);
  assert.equal(count, 3);
  fire(button, "click", { detail: 0 });
  assert.equal(count, 4, "keyboard activation still performs one step");
});

test("hold survives replaced layer-row buttons and stops at limits, cancellation, or focus loss", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  globalThis.window = new EventTarget();
  const button = new Target(), layerList = new Target();
  let count = 0;
  const stop = bindHoldRepeat(button, () => { count++; button.isConnected = false; }, { captureTarget: layerList, enabled: () => count < 3 });
  t.after(stop);
  fire(button, "pointerdown", { button: 0, pointerId: 1 });
  t.mock.timers.tick(350);
  t.mock.timers.tick(65);
  t.mock.timers.tick(65);
  assert.equal(count, 3);
  t.mock.timers.tick(1000);
  assert.equal(count, 3);
  count = 0;
  fire(button, "pointerdown", { button: 0, pointerId: 2 });
  fire(layerList, "pointercancel", { pointerId: 2 });
  t.mock.timers.tick(500);
  assert.equal(count, 1);
  fire(button, "pointerdown", { button: 0, pointerId: 3 });
  window.dispatchEvent(new Event("blur"));
  t.mock.timers.tick(500);
  assert.equal(count, 2);
});
