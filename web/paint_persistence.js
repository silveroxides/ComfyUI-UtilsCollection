function newAssetId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

export function paintAssetForNode(paint = {}, nodeId) {
  const owner = String(nodeId);
  const shared = paint.owner_node_id && String(paint.owner_node_id) !== owner;
  const assetId = !paint.asset_id || shared ? newAssetId() : paint.asset_id;
  return {
    ...paint,
    asset_id: assetId,
    owner_node_id: owner,
    asset: {
      filename: `uc-staged-paint-${assetId}.png`,
      subfolder: "clipspace",
      type: "input",
    },
  };
}

export function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => canvas.toBlob((blob) => {
    if (blob) resolve(blob);
    else reject(new Error("The staged paint canvas could not be encoded as PNG."));
  }, "image/png"));
}

export async function uploadPaintCanvas(api, canvas, paint, nodeId) {
  const next = paintAssetForNode(paint, nodeId);
  const blob = await canvasToPngBlob(canvas);
  const form = new FormData();
  form.append("image", blob, next.asset.filename);
  form.append("type", "input");
  form.append("subfolder", next.asset.subfolder);
  form.append("overwrite", "true");
  const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`Paint upload failed (${response.status}${detail ? `: ${detail}` : ""}).`);
  }
  const uploaded = await response.json();
  if (!uploaded?.name) throw new Error("Paint upload returned no filename.");
  return {
    ...next,
    asset: {
      filename: uploaded.name,
      subfolder: uploaded.subfolder || next.asset.subfolder,
      type: "input",
    },
    width: canvas.width,
    height: canvas.height,
    revision: Math.max(0, Number(paint.revision) || 0) + 1,
  };
}

// Every foreground save is immutable: queued prompts and workflow copies retain
// their original pixels even when the same editor keeps drawing.
export async function uploadForegroundCanvas(api, canvas) {
  const filename = `uc-foreground-${newAssetId()}.png`;
  const blob = await canvasToPngBlob(canvas);
  const form = new FormData();
  form.append("image", blob, filename);
  form.append("type", "input");
  form.append("subfolder", "clipspace");
  form.append("overwrite", "false");
  const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
  if (!response.ok) throw new Error(`Foreground content upload failed (${response.status}).`);
  const uploaded = await response.json();
  if (!uploaded?.name) throw new Error("Foreground content upload returned no filename.");
  return { filename: uploaded.name, subfolder: uploaded.subfolder ?? "clipspace", type: "input" };
}
