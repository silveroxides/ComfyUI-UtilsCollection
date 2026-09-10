function addUpstreamNode(nodeId, source, filtered) {
  const id = String(nodeId);
  if (filtered[id]) return;
  const node = source[id];
  if (!node) throw new Error(`Cannot queue staged compositor: prompt node ${id} is missing.`);

  filtered[id] = { ...node, inputs: { ...(node.inputs || {}) } };
  for (const value of Object.values(node.inputs || {})) {
    if (!Array.isArray(value) || value.length < 2) continue;
    const upstreamId = String(value[0]);
    if (source[upstreamId]) addUpstreamNode(upstreamId, source, filtered);
  }
}

export function buildOutputClosure(output, targetNodeId, inputOverrides = {}) {
  const source = output || {};
  const targetId = String(targetNodeId);
  const filtered = {};
  addUpstreamNode(targetId, source, filtered);
  filtered[targetId] = {
    ...filtered[targetId],
    inputs: { ...filtered[targetId].inputs, ...inputOverrides },
  };
  return filtered;
}

export function buildStagingPrompt(prompt, targetNodeId) {
  if (!prompt?.output) throw new Error("Cannot queue staged compositor: prompt output is unavailable.");
  return {
    ...prompt,
    output: buildOutputClosure(prompt.output, targetNodeId, { execution_mode: "run_staging" }),
  };
}

export function buildEditorPromptInputs(placementData, staged) {
  return {
    placement_data: placementData,
    ...(staged ? { execution_mode: "run_staged" } : {}),
  };
}

export function updatePromptNodeInputs(prompt, updates) {
  if (!prompt?.output || !updates?.length) return prompt;
  const output = { ...prompt.output };
  let workflow = prompt.workflow;
  let changed = false;
  for (const { nodeId, inputs, serializedNode } of updates) {
    const id = String(nodeId);
    if (!output[id]) continue;
    output[id] = { ...output[id], inputs: { ...(output[id].inputs || {}), ...inputs } };
    changed = true;
    if (serializedNode && workflow?.nodes) {
      workflow = { ...workflow, nodes: workflow.nodes.map((node) => String(node.id) === id
        ? { ...node, widgets_values: serializedNode.widgets_values,
          ...(serializedNode.widgets_values_named ? { widgets_values_named: serializedNode.widgets_values_named } : {}) }
        : node) };
    }
  }
  return changed ? { ...prompt, output, ...(workflow ? { workflow } : {}) } : prompt;
}
