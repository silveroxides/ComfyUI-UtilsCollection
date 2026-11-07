import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  clampVideoPreviewSize,
  growNodeSize,
  videoPreviewHeight,
} from "./video_preview_layout.js";

const ALLOWED_VIDEO_EXTS = new Set([
  "mp4", "mov", "mkv", "webm", "avi", "m4v", "gif",
  "mpg", "mpeg", "wmv", "flv", "ogv", "ts",
]);

export function isVideoFile(file) {
  if (!file) return false;
  if (String(file.type || "").startsWith("video/")) return true;
  const ext = String(file.name || "").split(".").pop()?.toLowerCase();
  return ext ? ALLOWED_VIDEO_EXTS.has(ext) : false;
}

export function droppedVideoFiles(event) {
  return Array.from(event?.dataTransfer?.files || []).filter(isVideoFile);
}

export function retainComboValue(widget, value) {
  const values = widget?.options?.values;
  if (Array.isArray(values) && !values.includes(value)) values.push(value);
}

export function applyVideoWidgetValue(node, widget, value, invokeCallback = true) {
  if (!widget || typeof value !== "string" || !value) return;
  retainComboValue(widget, value);
  const oldValue = widget.value;
  widget.value = value;
  const widgetIndex = node.widgets?.indexOf(widget) ?? -1;
  if (widgetIndex >= 0 && Array.isArray(node.widgets_values)) {
    node.widgets_values[widgetIndex] = value;
  }
  if (invokeCallback && oldValue !== value) {
    widget.callback?.(value);
    node.onWidgetChanged?.(widget.name, value, oldValue, widget);
  }
  node.graph?.setDirtyCanvas?.(true, true);
}

export async function uploadVideoFile(node, widget, file, buttonWidget) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file, file.name);

  const prevText = buttonWidget?.label || "choose video to upload";
  if (buttonWidget) buttonWidget.label = `Uploading ${file.name.slice(0, 16)}...`;
  node.graph?.setDirtyCanvas?.(true, true);

  try {
    const response = await fetch("/uc/api/upload_video", {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      let errorMsg = `Upload failed (HTTP ${response.status})`;
      try {
        const errorJson = await response.json();
        if (errorJson.error) errorMsg = errorJson.error;
      } catch {}
      throw new Error(errorMsg);
    }
    const uploaded = await response.json();
    const value = uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name;
    applyVideoWidgetValue(node, widget, value);
  } finally {
    if (buttonWidget) buttonWidget.label = prevText;
    node.graph?.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: "ComfyUI.UtilsCollection.UCLoadVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "UC_LoadVideo") return;

    // Preserve user manual resize while clamping to computed minimums
    const originalResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const minimum = this.computeSize();
      clampVideoPreviewSize(size, minimum, this.flags?.collapsed);
      if (!this.__ucApplyingLayout && size && !this.flags?.collapsed) {
        this.__ucManualSize = [size[0], size[1]];
      }
      return originalResize?.apply(this, arguments);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const fileWidget = this.widgets?.find((w) => w.name === "file");
      const modeWidget = this.widgets?.find((w) => w.name === "preview_mode");
      if (!fileWidget || !modeWidget) return result;

      const nodeRef = this;
      this.__ucManualSize = [...(this.size || [260, 120])];
      this.__ucApplyingLayout = false;
      this.__ucPreviewHeight = 0;
      this.__ucMediaWidth = 0;
      this.__ucMediaHeight = 0;
      this.__ucLayoutSignature = "";
      this.__ucIsPlaying = false;

      // Offscreen media elements (never mounted into DOM tree directly)
      const imgEl = new Image();
      const videoEl = document.createElement("video");
      videoEl.playsInline = true;
      videoEl.muted = true;
      videoEl.loop = true;
      videoEl.autoplay = false;

      let isVideoActive = false;
      let lastFrameTime = 0;

      const hasMedia = () => {
        return isVideoActive ? (videoEl.readyState >= 1 && Boolean(videoEl.src)) : (imgEl.complete && imgEl.naturalWidth > 0 && Boolean(imgEl.src));
      };

      const startPlaybackLoop = () => {
        if (!isVideoActive || videoEl.paused || videoEl.ended) return;
        nodeRef.graph?.setDirtyCanvas?.(true, false);
        requestAnimationFrame(startPlaybackLoop);
      };

      // 1. Upload button widget (stacked directly below combo widgets)
      const uploadBtn = this.addWidget("button", "choose video to upload", "upload", () => {
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*," + Array.from(ALLOWED_VIDEO_EXTS).map((e) => "." + e).join(",");
        fileInput.style.display = "none";
        fileInput.addEventListener("cancel", () => fileInput.remove());
        fileInput.onchange = async () => {
          const file = fileInput.files?.[0];
          fileInput.remove();
          if (!file) return;
          try {
            await uploadVideoFile(nodeRef, fileWidget, file, uploadBtn);
          } catch (err) {
            console.error("[UC_LoadVideo] Upload error:", err);
            alert(`Video upload failed: ${err.message}`);
          }
        };
        document.body.appendChild(fileInput);
        fileInput.click();
      });
      uploadBtn.serialize = false;

      // 2. Pure Canvas Preview Widget (rendered directly into LiteGraph's canvas draw loop)
      const previewWidget = this.addCustomWidget({
        name: "preview",
        type: "custom",
        options: { socketless: true },
        computeSize: (width) => {
          const mode = modeWidget.value || "none";
          if (mode === "none" || !hasMedia() || nodeRef.flags?.collapsed) {
            return [width, -4];
          }
          const availW = Math.max(120, width - 20);
          const stageH = videoPreviewHeight(availW, nodeRef.__ucMediaWidth, nodeRef.__ucMediaHeight, mode);
          return [width, stageH > 0 ? stageH + 8 : -4];
        },
        draw: (ctx, node, widgetWidth, y, widgetHeight) => {
          const mode = modeWidget.value || "none";
          if (mode === "none" || !hasMedia() || node.flags?.collapsed) return;

          const padX = 10;
          const padY = y + 2;
          const availW = Math.max(10, widgetWidth - padX * 2);
          const stageH = videoPreviewHeight(availW, nodeRef.__ucMediaWidth, nodeRef.__ucMediaHeight, mode);
          if (stageH <= 0) return;

          ctx.save();
          // Draw bounded stage container
          ctx.beginPath();
          if (ctx.roundRect) {
            ctx.roundRect(padX, padY, availW, stageH, 4);
          } else {
            ctx.rect(padX, padY, availW, stageH);
          }
          ctx.fillStyle = "#121418";
          ctx.fill();
          ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.clip();

          // Render media fitted (object-fit: contain)
          const source = isVideoActive ? (videoEl.readyState >= 1 ? videoEl : null) : (imgEl.complete && imgEl.naturalWidth > 0 ? imgEl : null);
          if (source) {
            const sw = source.videoWidth || source.naturalWidth || availW;
            const sh = source.videoHeight || source.naturalHeight || stageH;
            const scale = Math.min(availW / sw, stageH / sh);
            const dw = sw * scale;
            const dh = sh * scale;
            const dx = padX + (availW - dw) / 2;
            const dy = padY + (stageH - dh) / 2;
            ctx.drawImage(source, dx, dy, dw, dh);

            // If video is paused, draw subtle play indicator
            if (isVideoActive && videoEl.paused) {
              const cx = padX + availW / 2;
              const cy = padY + stageH / 2;
              ctx.beginPath();
              ctx.arc(cx, cy, 18, 0, Math.PI * 2);
              ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
              ctx.fill();
              ctx.beginPath();
              ctx.moveTo(cx - 5, cy - 8);
              ctx.lineTo(cx + 8, cy);
              ctx.lineTo(cx - 5, cy + 8);
              ctx.closePath();
              ctx.fillStyle = "#ffffff";
              ctx.fill();
            }
          }

          ctx.restore();
        },
        mouse: (event, position, node) => {
          if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
          if (!isVideoActive || !hasMedia() || (modeWidget.value || "none") === "none") return false;

          const widgetY = previewWidget.last_y || (node.size[1] - (nodeRef.__ucPreviewHeight || 0) - 10);
          const stageH = nodeRef.__ucPreviewHeight || 0;
          if (position[1] >= widgetY && position[1] <= widgetY + stageH) {
            if (videoEl.paused) {
              videoEl.play().then(() => {
                startPlaybackLoop();
              }).catch(() => {});
            } else {
              videoEl.pause();
              nodeRef.graph?.setDirtyCanvas?.(true, true);
            }
            return true;
          }
          return false;
        },
      });

      const updateLayout = () => {
        if (nodeRef.flags?.collapsed) {
          videoEl.pause();
          return;
        }
        const mode = modeWidget.value || "none";
        const mediaActive = hasMedia();

        if (mode === "none" || !mediaActive) {
          nodeRef.__ucPreviewHeight = 0;
          videoEl.pause();
        } else {
          const availW = Math.max(120, (nodeRef.size?.[0] || 260) - 20);
          const stageH = videoPreviewHeight(availW, nodeRef.__ucMediaWidth, nodeRef.__ucMediaHeight, mode);
          nodeRef.__ucPreviewHeight = stageH;
        }

        const computedMin = nodeRef.computeSize();
        const next = growNodeSize(nodeRef.__ucManualSize, computedMin);

        const signature = `${next[0]}:${next[1]}:${nodeRef.__ucPreviewHeight}`;
        if (signature === nodeRef.__ucLayoutSignature) return;
        nodeRef.__ucLayoutSignature = signature;

        if (!nodeRef.__ucApplyingLayout && (next[0] !== nodeRef.size[0] || next[1] !== nodeRef.size[1])) {
          nodeRef.__ucApplyingLayout = true;
          try {
            nodeRef.setSize(next);
          } finally {
            nodeRef.__ucApplyingLayout = false;
          }
        }
        nodeRef.graph?.setDirtyCanvas?.(true, true);
      };

      const clearMedia = () => {
        videoEl.pause();
        videoEl.src = "";
        imgEl.src = "";
        isVideoActive = false;
        nodeRef.__ucMediaWidth = 0;
        nodeRef.__ucMediaHeight = 0;
      };

      const updateSource = (srcUrl, isVideo = false) => {
        if (!srcUrl || (modeWidget.value || "none") === "none") {
          clearMedia();
          updateLayout();
          return;
        }
        isVideoActive = isVideo;
        if (isVideo) {
          imgEl.src = "";
          videoEl.onloadedmetadata = () => {
            nodeRef.__ucMediaWidth = videoEl.videoWidth || 0;
            nodeRef.__ucMediaHeight = videoEl.videoHeight || 0;
            updateLayout();
            videoEl.play().then(() => {
              startPlaybackLoop();
            }).catch(() => {});
          };
          videoEl.src = srcUrl;
        } else {
          videoEl.pause();
          videoEl.src = "";
          imgEl.onload = () => {
            nodeRef.__ucMediaWidth = imgEl.naturalWidth || 0;
            nodeRef.__ucMediaHeight = imgEl.naturalHeight || 0;
            updateLayout();
          };
          imgEl.src = srcUrl;
        }
      };

      // Handle dropdown changes
      const origFileCallback = fileWidget.callback;
      fileWidget.callback = function (val) {
        origFileCallback?.apply(this, arguments);
        const mode = modeWidget.value || "none";
        if (mode === "full" && val) {
          updateSource(api.apiURL(`/view?filename=${encodeURIComponent(val)}&type=input`), true);
        } else {
          clearMedia();
          updateLayout();
        }
      };

      const origModeCallback = modeWidget.callback;
      modeWidget.callback = function (mode) {
        origModeCallback?.apply(this, arguments);
        if (mode === "full" && fileWidget.value) {
          updateSource(api.apiURL(`/view?filename=${encodeURIComponent(fileWidget.value)}&type=input`), true);
        } else {
          clearMedia();
          updateLayout();
        }
      };

      // Drag & drop video file onto node
      this.onDragOver = (event) => {
        const items = Array.from(event?.dataTransfer?.items || []);
        return items.some((item) => item.kind === "file");
      };

      this.onDragDrop = async (event) => {
        const files = droppedVideoFiles(event);
        if (!files.length) return false;
        try {
          await uploadVideoFile(nodeRef, fileWidget, files[0], uploadBtn);
          return true;
        } catch (err) {
          console.error("[UC_LoadVideo] Drag & drop upload error:", err);
          alert(`Video upload failed: ${err.message}`);
          return false;
        }
      };

      // On execution result, load thumbnail still or proxy video if available
      const origExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        origExecuted?.apply(this, arguments);
        const mode = modeWidget.value || "none";
        if (mode === "none") {
          clearMedia();
          updateLayout();
          return;
        }
        const imgItem = message?.images?.[0];
        if (imgItem) {
          const query = new URLSearchParams({
            filename: imgItem.filename,
            subfolder: imgItem.subfolder || "",
            type: imgItem.type || "temp",
          });
          const url = api.apiURL(`/view?${query}`);
          const isVideo = imgItem.filename.endsWith(".mp4") || imgItem.filename.endsWith(".webm");
          updateSource(url, isVideo);
        }
      };

      clearMedia();
      updateLayout();
      return result;
    };
  },
});
