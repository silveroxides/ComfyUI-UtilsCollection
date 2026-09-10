import { app } from "../../scripts/app.js";
import { h3VideoLengthFromSeconds, h3ReferenceFrameRange } from "./h3_video_length.js";
import {
  clampResolutionPreviewSize,
  resolutionPreviewMinimumSize,
} from "./resolution_preview_layout.js";

const VIDEO_MIDDLE_BAND_RESOLUTIONS = {
  "21:9": [[896, 384], [1120, 480]],
  "16:9": [[768, 416], [864, 480], [1024, 576], [1152, 672]],
  "16:10": [[768, 480], [1024, 640]],
  "4:3": [[640, 480], [768, 576], [1024, 768]],
};

function gcd(left, right) {
  while (right) [left, right] = [right, left % right];
  return left;
}

function lcm(left, right) {
  return (left * right) / gcd(left, right);
}

function ratioFromValue(value) {
  const match = String(value).match(/^(\d+):(\d+)/);
  return match ? [Number(match[1]), Number(match[2])] : null;
}

function widgetValue(node, name) {
  return node.widgets?.find((widget) => widget.name === name)?.value;
}

function widgetIsLinked(node, name) {
  return node.inputs?.some((input) => input.widget?.name === name && input.link != null) ?? false;
}

function compareKeys(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
}

function regularResolution(ratioWidth, ratioHeight, megapixels, multiple, minimum) {
  const scale = Math.sqrt((megapixels * 1024 * 1024) / (ratioWidth * ratioHeight));
  let width = Math.round((ratioWidth * scale) / multiple) * multiple;
  let height = Math.round((ratioHeight * scale) / multiple) * multiple;
  if (width < minimum || height < minimum) {
    const widthStep = multiple / gcd(ratioWidth, multiple);
    const heightStep = multiple / gcd(ratioHeight, multiple);
    const ratioStep = lcm(widthStep, heightStep);
    const minimumRatio = Math.ceil(Math.max(minimum / ratioWidth, minimum / ratioHeight));
    const ratioScale = Math.ceil(minimumRatio / ratioStep) * ratioStep;
    width = ratioWidth * ratioScale;
    height = ratioHeight * ratioScale;
  }
  return [width, height];
}

function videoResolution(ratioWidth, ratioHeight, megapixels, multiple, minimum) {
  const divisor = gcd(ratioWidth, ratioHeight);
  ratioWidth /= divisor;
  ratioHeight /= divisor;
  const landscape = ratioWidth >= ratioHeight;
  const landscapeRatio = landscape
    ? [ratioWidth, ratioHeight]
    : [ratioHeight, ratioWidth];
  const targetPixels = megapixels * 1024 * 1024;
  const middleBand = VIDEO_MIDDLE_BAND_RESOLUTIONS[landscapeRatio.join(":")];
  if (megapixels >= 0.3 && megapixels <= 0.8 && middleBand) {
    const candidates = middleBand
      .map(([width, height]) => (landscape ? [width, height] : [height, width]))
      .filter(([width, height]) => (
        width % multiple === 0
        && height % multiple === 0
        && width >= minimum
        && height >= minimum
      ));
    if (candidates.length) {
      return candidates.reduce((best, candidate) => (
        Math.abs(candidate[0] * candidate[1] - targetPixels)
          < Math.abs(best[0] * best[1] - targetPixels)
          ? candidate
          : best
      ));
    }
  }

  const anchorRatio = landscape ? ratioWidth : ratioHeight;
  const companionRatio = landscape ? ratioHeight : ratioWidth;
  const anchorStep = lcm(multiple, anchorRatio);
  const minimumAnchor = Math.ceil(minimum / anchorStep) * anchorStep;
  const maximum = 8192;
  let best;
  let bestKey;
  for (let anchor = minimumAnchor; anchor <= maximum; anchor += anchorStep) {
    const idealCompanion = (anchor * companionRatio) / anchorRatio;
    const companions = new Set([
      Math.floor(idealCompanion / multiple) * multiple,
      Math.ceil(idealCompanion / multiple) * multiple,
    ]);
    for (const companion of companions) {
      if (companion < minimum || companion > maximum) continue;
      const [width, height] = landscape ? [anchor, companion] : [companion, anchor];
      const megapixelError = Math.abs(width * height - targetPixels) / targetPixels;
      const ratioError = Math.abs((width / height) / (ratioWidth / ratioHeight) - 1);
      const key = [megapixelError, ratioError, width, height];
      if (!bestKey || compareKeys(key, bestKey) < 0) {
        best = [width, height];
        bestKey = key;
      }
    }
  }
  return best;
}

function updatePreview(node, backendValue) {
  if (node.__ucH3ReferenceVideo) {
    const range = backendValue ?? h3ReferenceFrameRange(
      widgetIsLinked(node, "start_at_timestamp") ? null : Number(widgetValue(node, "start_at_timestamp")),
      widgetIsLinked(node, "duration_seconds") ? null : Number(widgetValue(node, "duration_seconds")),
      node.__ucH3SourceSeconds ?? null,
    );
    node.__ucResolutionPreview = `start frame ${range.start ?? "…"} · end frame ${range.end ?? "…"} · ${range.length ?? "…"} frames`;
    node.setDirtyCanvas(true, true);
    return;
  }
  if (backendValue !== undefined) {
    node.__ucResolutionPreview = String(Array.isArray(backendValue) ? backendValue[0] : backendValue);
  } else {
    const ratio = ratioFromValue(widgetValue(node, "aspect_ratio"));
    const megapixels = Number(widgetValue(node, "megapixels"));
    const multiple = Number(widgetValue(node, "multiple"));
    const minimum = Number(widgetValue(node, "minimum")) || 256;
    if (!ratio || !megapixels || !multiple) return;
    const [width, height] = node.__ucVideoResolutionSelector
      ? videoResolution(...ratio, megapixels, multiple, minimum)
      : regularResolution(...ratio, megapixels, multiple, minimum);
    const length = node.__ucVideoResolutionSelector && !widgetIsLinked(node, "duration_seconds")
      ? h3VideoLengthFromSeconds(Number(widgetValue(node, "duration_seconds")))
      : null;
    node.__ucResolutionPreview = length === null
      ? `${width}×${height}`
      : `${width}×${height} · ${length} frames`;
  }
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "ComfyUI.UtilsCollection.ResolutionPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["UC_ResolutionSelectorExtended", "UC_VideoResolutionSelector", "UC_MiniMaxH3RefVid"].includes(nodeData.name)) return;

    const computeSize = nodeType.prototype.computeSize;
    nodeType.prototype.computeSize = function (out) {
      const baseSize = computeSize?.call(this, out ? [...out] : undefined) || [...(out || this.size || [0, 0])];
      const size = resolutionPreviewMinimumSize(baseSize);
      if (nodeData.name === "UC_MiniMaxH3RefVid") size[0] = Math.max(size[0], 380);
      return size;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      clampResolutionPreviewSize(size, this.computeSize(), this.flags?.collapsed);
      return onResize?.apply(this, arguments);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.__ucVideoResolutionSelector = nodeData.name === "UC_VideoResolutionSelector";
      this.__ucH3ReferenceVideo = nodeData.name === "UC_MiniMaxH3RefVid";
      this.__ucResolutionPreview = "";
      const minimum = this.computeSize();
      this.setSize([
        Math.max(this.size[0], minimum[0]),
        Math.max(this.size[1], minimum[1]),
      ]);
      updatePreview(this);
      return result;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      onDrawForeground?.apply(this, arguments);
      if (this.flags.collapsed || !this.__ucResolutionPreview) return;
      ctx.save();
      ctx.fillStyle = "#bbb";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this.__ucResolutionPreview, this.size[0] / 2, this.size[1] - 9);
      ctx.restore();
    };

    const onWidgetChanged = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (name) {
      const result = onWidgetChanged?.apply(this, arguments);
      if (["aspect_ratio", "megapixels", "multiple", "minimum", "duration_seconds", "start_at_timestamp"].includes(name)) updatePreview(this);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (this.__ucH3ReferenceVideo) {
        const range = message?.h3_reference_range?.[0];
        if (range) {
          this.__ucH3SourceSeconds = range.source_seconds;
          updatePreview(this, { start: range.start_frame, end: range.start_frame + range.length - 1, length: range.length });
        }
        return;
      }
      updatePreview(this, message?.resolution);
    };

    if (nodeData.name === "UC_MiniMaxH3RefVid") {
      const onConnectionsChange = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (type, slot) {
        const result = onConnectionsChange?.apply(this, arguments);
        if (type === 1) {
          if (this.inputs?.[slot]?.name === "video") this.__ucH3SourceSeconds = null;
          updatePreview(this);
        }
        return result;
      };
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        updatePreview(this);
        return result;
      };
    }
  },
});
