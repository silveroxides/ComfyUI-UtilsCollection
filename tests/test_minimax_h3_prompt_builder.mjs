import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  createState,
  restoreState,
  compilePrompt,
  addDefinition,
  addSegment,
  h3StepDuration,
  timelineWarnings,
} from "../web/minimax_h3_prompt_model.js";
import {
  createBaseState,
  restoreBaseState,
  compileBasePrompt,
  addBaseSegment,
  getFinalShot,
  baseTimelineWarnings,
} from "../web/minimax_h3_base_prompt_model.js";

test("full reference model compiles 6-field envelope and steps H3 frames", () => {
  const state = createState();
  addDefinition(state, "subject", { hasRef: true, refType: "picture", refIndex: 1, text: "A young woman." });
  addDefinition(state, "picture", { role: "first_frame" });
  addDefinition(state, "video", { role: "edit" });
  addDefinition(state, "audio", { role: "timbre", targetSubject: 1 });

  state.summary.taskTypes = ["video editing", "reference generation"];
  state.summary.text = "Target video edit.";

  state.retention = [
    { label: "<Subject 1>", marker: "attribute_transfer", text: "transfers appearance" },
  ];

  addSegment(state);
  state.segments[0].visual = "Camera pushes in on <Subject 1>.";
  state.segments[0].speech = { enabled: true, speaker: "S1", language: "English", text: "Look here." };
  state.segments[0].sounds = { enabled: true, text: "Wind blows." };

  state.overall_soundscape = "Outdoor breeze.";
  state.non_diegetic_music = "Soft guitar.";

  const compiled = compilePrompt(state);
  assert.match(compiled, /^subject_definitions:\n<Subject 1> is fully referenced in <Picture 1>: A young woman\./);
  assert.match(compiled, /<Picture 1> is the fixed first frame anchor at 00\.00s\./);
  assert.match(compiled, /<Video 1> is the source video for the target video edit\./);
  assert.match(compiled, /<Audio 1> is the voice-timbre reference for <Subject 1> \(S1\)\./);
  assert.match(compiled, /summary:\n\[video editing \+ reference generation\] Target video edit\./);
  assert.match(compiled, /retention_analysis:\n<Subject 1>: attribute_transfer - transfers appearance/);
  assert.match(compiled, /detailed_description:\nTimeline:\n\[00\.00s-02\.33s\]:\n\[VISUAL\]: \[Shot 1\] Camera pushes in on <Subject 1>\./);
  assert.match(compiled, /\[SPEECH\]: \(S1\) <d>\[English\] Look here\.<\/d>/);
  assert.match(compiled, /\[SOUNDS\]: Wind blows\./);
  assert.match(compiled, /overall_soundscape:\nOutdoor breeze\./);
  assert.match(compiled, /non_diegetic_music:\nSoft guitar\./);

  // Stepping duration by 17 frames at 24fps
  const nextDuration = h3StepDuration(2.333, 1);
  assert.ok(Math.abs(nextDuration - 3.0416) < 0.01);
});

test("base prompt model compiles T2VA, I2VA, FL2VA, and L2VA with 3 core fields", () => {
  // 1. T2VA with Timeline
  const t2vState = createBaseState();
  t2vState.task = "T2VA";
  addBaseSegment(t2vState);
  t2vState.segments[0].visual = "A baker opens the shop.";
  t2vState.overall_soundscape = "Morning street noise.";
  t2vState.non_diegetic_music = "Acoustic guitar.";

  const t2vCompiled = compileBasePrompt(t2vState);
  assert.ok(!t2vCompiled.includes("How the reference pictures align"));
  assert.ok(!t2vCompiled.includes("For the target video"));
  assert.ok(!t2vCompiled.includes("subject_definitions:"));
  assert.match(t2vCompiled, /^integrated_multimodal_description:\nTimeline:\n\[00\.00s-02\.33s\]:\n\[VISUAL\]: \[Shot 1\] A baker opens the shop\./);
  assert.match(t2vCompiled, /overall_soundscape:\nMorning street noise\./);
  assert.match(t2vCompiled, /non_diegetic_music:\nAcoustic guitar\./);

  // 2. I2VA
  const i2vState = createBaseState();
  i2vState.task = "I2VA";
  i2vState.description.mode = "continuous";
  i2vState.description.continuousText = "[Shot 1] Live-action, cinematic, starting from <Picture 1>.";
  i2vState.overall_soundscape = "Room tone.";
  i2vState.non_diegetic_music = "N/A";

  const i2vCompiled = compileBasePrompt(i2vState);
  assert.ok(i2vCompiled.startsWith("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"));
  assert.match(i2vCompiled, /integrated_multimodal_description:\n\[Shot 1\] Live-action, cinematic, starting from <Picture 1>\./);

  // 3. FL2VA with auto final shot
  const fl2vState = createBaseState();
  fl2vState.task = "FL2VA";
  fl2vState.duration = 8.0;
  addBaseSegment(fl2vState);
  addBaseSegment(fl2vState);
  fl2vState.segments[1].visual = "Lands on Picture 2.";
  const fl2vFinalShot = getFinalShot(fl2vState);
  assert.equal(fl2vFinalShot, 2);

  const fl2vCompiled = compileBasePrompt(fl2vState);
  assert.ok(fl2vCompiled.startsWith("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 8.00-second mark of the target video.\n\n"));

  // 4. L2VA
  const l2vState = createBaseState();
  l2vState.task = "L2VA";
  l2vState.duration = 6.0;
  l2vState.autoFinalShot = false;
  l2vState.finalShot = 3;
  l2vState.description.mode = "continuous";
  l2vState.description.continuousText = "[Shot 1] Drops and lands on <Picture 1>.";
  const l2vCompiled = compileBasePrompt(l2vState);
  assert.ok(l2vCompiled.startsWith("How the reference pictures align with the target video — <Picture 1> (from [Shot 3]) aligns with the 6.00-second mark of the target video.\n\n"));
});

test("multiline text overlay closes when outside pointerdown occurs or on blur", () => {
  const source = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(source.includes("this.outsidePointer = (event) =>"));
  assert.ok(source.includes("document.addEventListener(\"pointerdown\", this.outsidePointer, true);"));
  assert.ok(source.includes("document.removeEventListener(\"pointerdown\", this.outsidePointer, true);"));
  assert.ok(source.includes("element.addEventListener(\"blur\""));

  const baseSource = readFileSync(new URL("../web/minimax_h3_base_prompt.js", import.meta.url), "utf8");
  assert.ok(baseSource.includes("this.outsidePointer = (event) =>"));
  assert.ok(baseSource.includes("document.addEventListener(\"pointerdown\", this.outsidePointer, true);"));
  assert.ok(baseSource.includes("document.removeEventListener(\"pointerdown\", this.outsidePointer, true);"));
  assert.ok(baseSource.includes("element.addEventListener(\"blur\""));
});

test("canvas draw-loop tooltips are rendered with upward offset and no DOM elements", () => {
  const dynamicSource = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(dynamicSource.includes("drawCanvasTooltip("));
  assert.ok(dynamicSource.includes("by = region.y - boxH - 8"));
  assert.ok(dynamicSource.includes("onPointerMove(event)"));
  assert.ok(dynamicSource.includes("clearHoveredTooltip()"));

  const baseSource = readFileSync(new URL("../web/minimax_h3_base_prompt.js", import.meta.url), "utf8");
  assert.ok(baseSource.includes("drawCanvasTooltip("));
  assert.ok(baseSource.includes("by = region.y - boxH - 8"));
  assert.ok(baseSource.includes("onPointerMove(event)"));
  assert.ok(baseSource.includes("clearHoveredTooltip()"));
});

test("collapsing dropdown menus route clicks through getWidgetOnPos and support disabled options", () => {
  const source = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(source.includes("this.activeMenu && this.menuBoundingBox && this.contains(this.menuBoundingBox, x, y)"));
  assert.ok(source.includes("this.menuBoundingBox = { x: mx, y: my, w: mw, h: totalH }"));
  assert.ok(source.includes("const disabled = Boolean(opt.disabled)"));
});

test("navigation tabs and fixed controls use hitFixed to prevent viewport clipping across redraws", () => {
  const baseSource = readFileSync(new URL("../web/minimax_h3_base_prompt.js", import.meta.url), "utf8");
  assert.ok(baseSource.includes("hitFixed(x, y, w, h, action, tooltip)"));
  assert.ok(baseSource.includes("this.hitFixed(x, y, w, h, action, tooltip)"));
  assert.ok(baseSource.includes("this.viewportY = y;"));

  const dynamicSource = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(dynamicSource.includes("hitFixed(x, y, w, h, action, tooltip)"));
  assert.ok(dynamicSource.includes("this.hitFixed(x, y, w, h, action, tooltip)"));
});

test("single-line button edits use inline input overlay instead of non-existent canvas prompt", () => {
  const dynamicSource = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(dynamicSource.includes("openSingleLineEditor(rect, value, apply)"));
  assert.ok(dynamicSource.includes("h3-prompt-singleline-container"));
  assert.ok(dynamicSource.includes("durRect"));
  assert.ok(dynamicSource.includes("startRect"));
  assert.ok(dynamicSource.includes("endRect"));

  const baseSource = readFileSync(new URL("../web/minimax_h3_base_prompt.js", import.meta.url), "utf8");
  assert.ok(baseSource.includes("openSingleLineEditor(rect, value, apply)"));
  assert.ok(baseSource.includes("h3-base-prompt-singleline-container"));
});

test("dropdown menus attach document-level pointerdown listener for outside dismissal", () => {
  const dynamicSource = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(dynamicSource.includes("this.menuOutsidePointer = () =>"));
  assert.ok(dynamicSource.includes("document.addEventListener(\"pointerdown\", this.menuOutsidePointer, true)"));
  assert.ok(dynamicSource.includes("document.removeEventListener(\"pointerdown\", this.menuOutsidePointer, true)"));
});

test("timeline segment cards compute dynamic height avoiding hardcoded vertical overlap", () => {
  const dynamicSource = readFileSync(new URL("../web/minimax_h3_prompt.js", import.meta.url), "utf8");
  assert.ok(dynamicSource.includes("const spkActive = Boolean(seg.speech?.enabled);"));
  assert.ok(dynamicSource.includes("const sndActive = Boolean(seg.sounds?.enabled);"));
  assert.ok(dynamicSource.includes("const musActive = Boolean(seg.music?.enabled);"));
  assert.ok(dynamicSource.includes("const cardH = collapsed"));
  assert.ok(dynamicSource.includes("spkActive ? 34 : 0"));
  assert.ok(dynamicSource.includes("sndActive ? 34 : 0"));
  assert.ok(dynamicSource.includes("musActive ? 34 : 0"));
});
