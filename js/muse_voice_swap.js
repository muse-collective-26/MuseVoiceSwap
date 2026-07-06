const { app } = window.comfyAPI.app;

// --- UI constants (reused from Muse Director for visual consistency) ---
const RULER_HEIGHT = 24;
const VOCAL_TRACK_HEIGHT = 100;
const BG_TRACK_HEIGHT = 60;
const CANVAS_HEIGHT = RULER_HEIGHT + VOCAL_TRACK_HEIGHT + BG_TRACK_HEIGHT;
const HANDLE_HIT_PX = 10;
const PANEL_HEIGHT = 130;
const PREVIEW_HEIGHT = 130;

const NODE_COLOR = "#2D1B69";
const NODE_BGCOLOR = "#0d0818";
const CANVAS_BG = "#2a2a2a";
const SELECTION_HIGHLIGHT = "#4fff8f";
const SEGMENT_FILL = "#3a4a6a";
const SEGMENT_FILL_LOCKED = "#5a4a3a";
const PEAK_COLOR = "#8fb7ff";
const BG_PEAK_COLOR = "#777";
const LABEL_FONT = "12px sans-serif";

// Speaker color-coding (1=blue, 2=orange, 3=green) — locked overrides speaker fill.
const SPEAKER_FILL = { 1: "#3a4a6a", 2: "#7a5a2a", 3: "#2a6a4a" };
const SPEAKER_BORDER = { 1: "#8899bb", 2: "#e0b070", 3: "#7fd9a8" };

const HIDDEN_WIDGET_NAMES = ["timeline_data", "timeline_ui"];

// Advanced/tuning widgets collapsed behind the "Settings" toggle by default —
// same show/hide-on-demand pattern as Muse Director's MUSE_SETTINGS_WIDGET_NAMES.
const SETTINGS_WIDGET_NAMES = [
  "sep_chunk_length", "sep_chunk_overlap", "sep_chunk_fade_shape",
  "vad_top_db", "min_segment_seconds", "min_gap_seconds",
  "whisper_model_size", "whisper_language",
  "emotion_tag_temperature",
  "fish_language", "fish_device", "fish_precision", "fish_attention",
  "fish_max_new_tokens", "fish_chunk_length", "fish_temperature", "fish_top_p",
  "fish_repetition_penalty", "fish_seed", "fish_keep_model_loaded",
  "time_match_mode", "stretch_fft_size", "bg_extend_mode", "bg_volume",
  "reference_text_2", "reference_text_3",
];

// Multiline STRING widgets that need their default (oversized) height capped.
const CLAMPED_TEXT_WIDGET_NAMES = ["reference_text", "reference_text_2", "reference_text_3"];

const STYLES = `
  .mvs-wrapper {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 100%;
    box-sizing: border-box;
  }
  .mvs-canvas { background: ${CANVAS_BG}; border-radius: 4px; cursor: pointer; display: block; }
  .mvs-toolbar { display: flex; gap: 6px; align-items: center; }
  .mvs-btn {
    background: #222; color: #e0e0e0; border: 1px solid #111; border-radius: 4px;
    padding: 4px 10px; font-size: 12px; cursor: pointer;
  }
  .mvs-btn:hover { background: #333; border-color: #555; }
  .mvs-panel {
    background: #1c1c1c; border: 1px solid #333; border-radius: 4px;
    padding: 6px; display: flex; flex-direction: column; gap: 4px;
  }
  .mvs-panel textarea {
    width: 100%; box-sizing: border-box; background: #111; color: #e0e0e0;
    border: 1px solid #333; border-radius: 3px; font-family: inherit; font-size: 12px;
    resize: vertical; min-height: 40px;
  }
  .mvs-panel input[type="text"] {
    width: 100%; box-sizing: border-box; background: #111; color: #e0e0e0;
    border: 1px solid #333; border-radius: 3px; font-size: 11px; padding: 2px 4px;
  }
  .mvs-panel-row { display: flex; gap: 6px; align-items: center; font-size: 11px; color: #aaa; }
  .mvs-empty { color: #777; font-size: 12px; padding: 8px; text-align: center; }
  .mvs-settings-btn { font-weight: 600; margin-left: auto; }
  .mvs-preview {
    background: #161616; border: 1px solid #2a2a2a; border-radius: 4px;
    max-height: 120px; overflow-y: auto; font-size: 11px;
  }
  .mvs-preview-row {
    display: flex; gap: 8px; padding: 4px 8px; border-bottom: 1px solid #222; cursor: pointer;
  }
  .mvs-preview-row:hover { background: #222; }
  .mvs-preview-row.selected { background: #24352a; }
  .mvs-preview-time { color: #888; flex: 0 0 auto; white-space: nowrap; }
  .mvs-preview-text { color: #ddd; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mvs-preview-tag { color: #b39ddb; font-size: 10px; flex: 0 0 auto; }
`;

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  if (!w.options) w.options = {};
  w.options.hidden = true;
  w.computeSize = () => [0, -4];
  w.draw = () => {};
}

function showWidget(w) {
  if (!w) return;
  w.hidden = false;
  if (w.options) w.options.hidden = false;
  delete w.computeSize;
  delete w.draw;
}

function fmtTime(t) {
  const s = Math.max(0, t);
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(2);
  return `${m}:${sec.padStart(5, "0")}`;
}

function uid() {
  return "seg_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

class VoiceSwapTimelineEditor {
  constructor(node, container, widget) {
    this.node = node;
    this.widget = widget;
    this.container = container;

    this.timeline = {
      vocalSegments: [],
      bgWaveformPeaks: [],
      source_duration: 0,
    };
    this.selectedId = null;
    this.drag = null; // {mode: "move"|"resize-l"|"resize-r"|"create", id, startX, ...}

    this._buildDom();
    this._loadFromWidgetValue();
    this.render();
  }

  _buildDom() {
    if (!document.getElementById("mvs-styles")) {
      const style = document.createElement("style");
      style.id = "mvs-styles";
      style.textContent = STYLES;
      document.head.appendChild(style);
    }

    this.wrapper = document.createElement("div");
    this.wrapper.className = "mvs-wrapper";

    const toolbar = document.createElement("div");
    toolbar.className = "mvs-toolbar";

    const addBtn = document.createElement("button");
    addBtn.className = "mvs-btn";
    addBtn.textContent = "+ Add Segment";
    addBtn.onclick = () => this._addSegmentAtEnd();
    toolbar.appendChild(addBtn);

    const delBtn = document.createElement("button");
    delBtn.className = "mvs-btn";
    delBtn.textContent = "Delete Selected";
    delBtn.onclick = () => this._deleteSelected();
    toolbar.appendChild(delBtn);

    this.statusLabel = document.createElement("span");
    this.statusLabel.style.cssText = "color:#888;font-size:11px;margin-left:8px;";
    toolbar.appendChild(this.statusLabel);

    const settingsBtn = document.createElement("button");
    settingsBtn.className = "mvs-btn mvs-settings-btn";
    let settingsVisible = false;
    const updateSettingsBtn = () => {
      settingsBtn.textContent = settingsVisible ? "▲ Settings" : "▼ Settings";
    };
    updateSettingsBtn();
    settingsBtn.onclick = () => {
      settingsVisible = !settingsVisible;
      for (const w of this.node.widgets || []) {
        if (SETTINGS_WIDGET_NAMES.includes(w.name)) {
          if (settingsVisible) {
            showWidget(w);
            if (CLAMPED_TEXT_WIDGET_NAMES.includes(w.name)) {
              w.computeSize = function (width) { return [width || 200, 64]; };
            }
          } else {
            hideWidget(w);
          }
        }
      }
      updateSettingsBtn();
      this.node.setDirtyCanvas(true, true);
      this.syncLayoutToNode();
    };
    toolbar.appendChild(settingsBtn);

    this.canvas = document.createElement("canvas");
    this.canvas.className = "mvs-canvas";
    this.canvas.height = CANVAS_HEIGHT;

    this.previewList = document.createElement("div");
    this.previewList.className = "mvs-preview";
    this._renderPreviewList();

    this.panel = document.createElement("div");
    this.panel.className = "mvs-panel";
    this._renderPanel();

    this.wrapper.appendChild(toolbar);
    this.wrapper.appendChild(this.canvas);
    this.wrapper.appendChild(this.previewList);
    this.wrapper.appendChild(this.panel);
    this.container.appendChild(this.wrapper);

    this.canvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
    window.addEventListener("mousemove", (e) => this._onMouseMove(e));
    window.addEventListener("mouseup", () => this._onMouseUp());
  }

  _loadFromWidgetValue() {
    const raw = this.widget?.value;
    if (!raw || raw === "{}") return;
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        this.timeline.vocalSegments = parsed.vocalSegments || [];
        this.timeline.bgWaveformPeaks = parsed.bgWaveformPeaks || [];
        this.timeline.source_duration = parsed.source_duration || 0;
      }
    } catch (err) {
      console.warn("[MuseVoiceSwap] failed to parse existing timeline_data:", err);
    }
  }

  applyExecutionResult(message) {
    const payloadList = message?.muse_voice_swap;
    if (!payloadList || !payloadList.length) return;
    const payload = payloadList[0];

    this.timeline.vocalSegments = (payload.segments || []).map((s) => ({ ...s, id: s.id || uid() }));
    this.timeline.vocalPeaks = payload.vocal_peaks || [];
    this.timeline.bgWaveformPeaks = payload.bg_peaks || [];
    this.timeline.source_duration = payload.source_duration || 0;
    this.timeline.final_duration = payload.final_duration || payload.source_duration || 0;

    this.selectedId = null;
    this._commitChanges();
    this.render();
    this._renderPanel();
  }

  _totalDuration() {
    const fromSegs = this.timeline.vocalSegments.reduce(
      (max, s) => Math.max(max, (s.placed_start ?? s.start) + (s.placed_duration ?? (s.end - s.start))),
      0
    );
    return Math.max(this.timeline.final_duration || 0, this.timeline.source_duration || 0, fromSegs, 1);
  }

  _timeToX(t, width) {
    const total = this._totalDuration();
    return (t / total) * width;
  }

  _xToTime(x, width) {
    const total = this._totalDuration();
    return Math.max(0, (x / width) * total);
  }

  render() {
    const width = Math.max(200, this.node.size?.[0] ? this.node.size[0] - 30 : 700);
    this.canvas.width = width;
    const ctx = this.canvas.getContext("2d");
    ctx.clearRect(0, 0, width, CANVAS_HEIGHT);

    // Ruler
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, width, RULER_HEIGHT);
    ctx.fillStyle = "#888";
    ctx.font = LABEL_FONT;
    const total = this._totalDuration();
    const step = total > 20 ? 5 : total > 8 ? 2 : 1;
    for (let t = 0; t <= total; t += step) {
      const x = this._timeToX(t, width);
      ctx.fillRect(x, RULER_HEIGHT - 6, 1, 6);
      ctx.fillText(fmtTime(t), x + 2, RULER_HEIGHT - 8);
    }

    // Vocal track
    const vocalY = RULER_HEIGHT;
    ctx.fillStyle = "#232338";
    ctx.fillRect(0, vocalY, width, VOCAL_TRACK_HEIGHT);
    this._drawPeaks(ctx, this.timeline.vocalPeaks, 0, vocalY, width, VOCAL_TRACK_HEIGHT, PEAK_COLOR);

    for (const seg of this.timeline.vocalSegments) {
      const start = seg.placed_start ?? seg.start;
      const dur = seg.placed_duration ?? (seg.end - seg.start);
      const x = this._timeToX(start, width);
      const w = Math.max(4, this._timeToX(start + dur, width) - x);
      const selected = seg.id === this.selectedId;

      const speaker = seg.speaker || 1;
      ctx.fillStyle = seg.locked ? SEGMENT_FILL_LOCKED : (SPEAKER_FILL[speaker] || SEGMENT_FILL);
      ctx.globalAlpha = 0.55;
      ctx.fillRect(x, vocalY + 4, w, VOCAL_TRACK_HEIGHT - 8);
      ctx.globalAlpha = 1.0;

      ctx.strokeStyle = selected ? SELECTION_HIGHLIGHT : (SPEAKER_BORDER[speaker] || "#8899bb");
      ctx.lineWidth = selected ? 2 : 1;
      ctx.strokeRect(x, vocalY + 4, w, VOCAL_TRACK_HEIGHT - 8);

      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, vocalY, Math.max(0, w - 4), VOCAL_TRACK_HEIGHT);
      ctx.clip();

      ctx.fillStyle = "#ddd";
      ctx.font = LABEL_FONT;
      const label = `S${speaker}: ${seg.text || "(no transcript)"}`.slice(0, 44);
      ctx.fillText(label, x + 4, vocalY + 16);

      if (seg.emotion_text && seg.emotion_text.trim()) {
        ctx.fillStyle = "#b39ddb";
        ctx.font = "italic 11px sans-serif";
        ctx.fillText(seg.emotion_text.slice(0, 44), x + 4, vocalY + 30);
      }

      ctx.restore();
    }

    if (this.timeline.vocalSegments.length === 0) {
      ctx.fillStyle = "#666";
      ctx.font = LABEL_FONT;
      ctx.fillText("No segments yet — queue the node once to auto-detect, or click + Add Segment.", 8, vocalY + VOCAL_TRACK_HEIGHT / 2);
    }

    // BG reference track (read-only)
    const bgY = vocalY + VOCAL_TRACK_HEIGHT;
    ctx.fillStyle = "#1c1c1c";
    ctx.fillRect(0, bgY, width, BG_TRACK_HEIGHT);
    ctx.fillStyle = "#666";
    ctx.font = "10px sans-serif";
    ctx.fillText("Background (read-only)", 4, bgY + 10);
    this._drawPeaks(ctx, this.timeline.bgWaveformPeaks, 0, bgY + 12, width, BG_TRACK_HEIGHT - 12, BG_PEAK_COLOR);

    this.statusLabel.textContent = `${this.timeline.vocalSegments.length} segment(s) · total ${fmtTime(total)}`;
    this._renderPreviewList();
  }

  _renderPreviewList() {
    if (!this.previewList) return;
    this.previewList.innerHTML = "";

    const sorted = [...this.timeline.vocalSegments].sort((a, b) => a.start - b.start);
    if (sorted.length === 0) {
      const empty = document.createElement("div");
      empty.className = "mvs-empty";
      empty.textContent = "No segments to preview yet.";
      this.previewList.appendChild(empty);
      return;
    }

    for (const seg of sorted) {
      const row = document.createElement("div");
      row.className = "mvs-preview-row" + (seg.id === this.selectedId ? " selected" : "");
      row.onclick = () => {
        this.selectedId = seg.id;
        this.render();
        this._renderPanel();
      };

      const time = document.createElement("span");
      time.className = "mvs-preview-time";
      time.textContent = fmtTime(seg.start);
      row.appendChild(time);

      const speakerBadge = document.createElement("span");
      speakerBadge.className = "mvs-preview-tag";
      speakerBadge.style.color = SPEAKER_BORDER[seg.speaker || 1] || "#8899bb";
      speakerBadge.textContent = `S${seg.speaker || 1}`;
      row.appendChild(speakerBadge);

      const hasEmotion = seg.emotion_text && seg.emotion_text.trim();
      if (hasEmotion) {
        const tag = document.createElement("span");
        tag.className = "mvs-preview-tag";
        tag.textContent = "LLM";
        row.appendChild(tag);
      }

      const text = document.createElement("span");
      text.className = "mvs-preview-text";
      text.textContent = hasEmotion ? seg.emotion_text : (seg.text || "(no transcript)");
      row.appendChild(text);

      this.previewList.appendChild(row);
    }
  }

  _drawPeaks(ctx, peaks, x0, y0, w, h, color) {
    if (!peaks || peaks.length === 0) return;
    const mid = y0 + h / 2;
    const barW = w / peaks.length;
    ctx.fillStyle = color;
    for (let i = 0; i < peaks.length; i++) {
      const amp = Math.min(1, peaks[i]) * (h / 2 - 2);
      const x = x0 + i * barW;
      ctx.fillRect(x, mid - amp, Math.max(1, barW - 1), amp * 2);
    }
  }

  _hitTest(x, y, width) {
    const vocalY = RULER_HEIGHT;
    if (y < vocalY || y > vocalY + VOCAL_TRACK_HEIGHT) return null;
    for (const seg of this.timeline.vocalSegments) {
      const start = seg.placed_start ?? seg.start;
      const dur = seg.placed_duration ?? (seg.end - seg.start);
      const sx = this._timeToX(start, width);
      const ex = this._timeToX(start + dur, width);
      if (x >= sx - HANDLE_HIT_PX && x <= sx + HANDLE_HIT_PX) return { seg, mode: "resize-l" };
      if (x >= ex - HANDLE_HIT_PX && x <= ex + HANDLE_HIT_PX) return { seg, mode: "resize-r" };
      if (x >= sx && x <= ex) return { seg, mode: "move" };
    }
    return null;
  }

  _onMouseDown(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const width = this.canvas.width;

    const hit = this._hitTest(x, y, width);
    if (hit) {
      this.selectedId = hit.seg.id;
      this.drag = { mode: hit.mode, seg: hit.seg, startX: x, origStart: hit.seg.start, origEnd: hit.seg.end };
    } else if (y >= RULER_HEIGHT && y <= RULER_HEIGHT + VOCAL_TRACK_HEIGHT) {
      const t = this._xToTime(x, width);
      const newSeg = { id: uid(), start: t, end: t + 0.5, text: "", locked: false };
      this.timeline.vocalSegments.push(newSeg);
      this.selectedId = newSeg.id;
      this.drag = { mode: "resize-r", seg: newSeg, startX: x, origStart: newSeg.start, origEnd: newSeg.end };
    } else {
      this.selectedId = null;
    }
    this.render();
    this._renderPanel();
  }

  _onMouseMove(e) {
    if (!this.drag) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = this.canvas.width;
    const t = this._xToTime(x, width);
    const seg = this.drag.seg;

    if (this.drag.mode === "move") {
      const dur = this.drag.origEnd - this.drag.origStart;
      const deltaT = this._xToTime(x, width) - this._xToTime(this.drag.startX, width);
      seg.start = Math.max(0, this.drag.origStart + deltaT);
      seg.end = seg.start + dur;
    } else if (this.drag.mode === "resize-l") {
      seg.start = Math.min(t, seg.end - 0.05);
    } else if (this.drag.mode === "resize-r") {
      seg.end = Math.max(t, seg.start + 0.05);
    }
    this.render();
  }

  _onMouseUp() {
    if (this.drag) {
      this.drag = null;
      this._commitChanges();
      this._renderPanel();
    }
  }

  _addSegmentAtEnd() {
    const total = this._totalDuration();
    const newSeg = { id: uid(), start: total, end: total + 1.0, text: "", locked: false, speaker: 1 };
    this.timeline.vocalSegments.push(newSeg);
    this.selectedId = newSeg.id;
    this._commitChanges();
    this.render();
    this._renderPanel();
  }

  _deleteSelected() {
    if (!this.selectedId) return;
    this.timeline.vocalSegments = this.timeline.vocalSegments.filter((s) => s.id !== this.selectedId);
    this.selectedId = null;
    this._commitChanges();
    this.render();
    this._renderPanel();
  }

  _renderPanel() {
    this.panel.innerHTML = "";
    const seg = this.timeline.vocalSegments.find((s) => s.id === this.selectedId);
    if (!seg) {
      const empty = document.createElement("div");
      empty.className = "mvs-empty";
      empty.textContent = "Click a segment to edit its transcript.";
      this.panel.appendChild(empty);
      return;
    }

    const info = document.createElement("div");
    info.className = "mvs-panel-row";
    info.textContent = `${fmtTime(seg.start)} → ${fmtTime(seg.end)} (${(seg.end - seg.start).toFixed(2)}s)`;

    const speakerLabel = document.createElement("span");
    speakerLabel.textContent = "Speaker:";
    speakerLabel.style.marginLeft = "auto";
    const speakerSelect = document.createElement("select");
    speakerSelect.style.cssText = "background:#111;color:#e0e0e0;border:1px solid #333;border-radius:3px;font-size:11px;";
    for (const n of [1, 2, 3]) {
      const opt = document.createElement("option");
      opt.value = String(n);
      opt.textContent = `Speaker ${n}`;
      if ((seg.speaker || 1) === n) opt.selected = true;
      speakerSelect.appendChild(opt);
    }
    speakerSelect.onchange = () => {
      seg.speaker = parseInt(speakerSelect.value, 10);
      this._commitChanges();
      this.render();
    };
    info.appendChild(speakerLabel);
    info.appendChild(speakerSelect);
    this.panel.appendChild(info);

    const textArea = document.createElement("textarea");
    textArea.placeholder = "Transcript (auto-filled by Whisper after first run — edit to correct)";
    textArea.value = seg.text || "";
    textArea.oninput = () => {
      seg.text = textArea.value;
      this._commitChanges();
      this.render(); // refreshes canvas label + preview list without rebuilding the panel (keeps focus)
    };
    this.panel.appendChild(textArea);

    const emotionArea = document.createElement("textarea");
    emotionArea.placeholder = "Emotion-tagged version used for cloning (auto-filled by Gemma when use_llm_emotion_tags is on — edit to adjust tags)";
    emotionArea.style.borderColor = "#4a3a5a";
    emotionArea.value = seg.emotion_text || "";
    emotionArea.oninput = () => {
      seg.emotion_text = emotionArea.value;
      this._commitChanges();
      this.render();
    };
    this.panel.appendChild(emotionArea);

    const overrideRow = document.createElement("div");
    overrideRow.className = "mvs-panel-row";
    const overrideLabel = document.createElement("span");
    overrideLabel.textContent = "Voice override (optional):";
    const overrideInput = document.createElement("input");
    overrideInput.type = "text";
    overrideInput.placeholder = "leave blank to use node-level reference_audio";
    overrideInput.value = seg.reference_audio_override || "";
    overrideInput.oninput = () => {
      seg.reference_audio_override = overrideInput.value;
      this._commitChanges();
    };
    overrideRow.appendChild(overrideLabel);
    overrideRow.appendChild(overrideInput);
    this.panel.appendChild(overrideRow);

    const lockRow = document.createElement("div");
    lockRow.className = "mvs-panel-row";
    const lockCheck = document.createElement("input");
    lockCheck.type = "checkbox";
    lockCheck.checked = !!seg.locked;
    lockCheck.onchange = () => {
      seg.locked = lockCheck.checked;
      this._commitChanges();
      this.render();
    };
    const lockLabel = document.createElement("span");
    lockLabel.textContent = "Lock (exempt this segment from re-detect/re-transcribe/auto-speaker-assignment)";
    lockRow.appendChild(lockCheck);
    lockRow.appendChild(lockLabel);
    this.panel.appendChild(lockRow);
  }

  _commitChanges() {
    const toSave = {
      schema_version: 1,
      source_duration: this.timeline.source_duration || 0,
      vocalSegments: this.timeline.vocalSegments.map((s) => ({
        id: s.id,
        start: s.start,
        end: s.end,
        text: s.text || "",
        emotion_text: s.emotion_text || "",
        speaker: s.speaker || 1,
        reference_audio_override: s.reference_audio_override || "",
        reference_text_override: s.reference_text_override || "",
        locked: !!s.locked,
      })),
      bgWaveformPeaks: this.timeline.bgWaveformPeaks || [],
      last_run: this.timeline.last_run || {},
    };
    const jsonStr = JSON.stringify(toSave);
    if (this.widget) {
      this.widget.value = jsonStr;
      if (this.node.onWidgetChanged) {
        this.node.onWidgetChanged(this.widget.name, jsonStr, this.widget.value, this.widget);
      }
      if (this.widget.callback) {
        try {
          this.widget.callback(jsonStr);
        } catch (err) {
          // ignore
        }
      }
    }
  }

  syncLayoutToNode() {
    this.render();
  }

  destroy() {
    // canvas/panel are children of container, which LiteGraph removes on node
    // removal — nothing to explicitly tear down beyond dropping references.
    this.node = null;
  }
}

app.registerExtension({
  name: "MuseCollective.VoiceSwapTimeline",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MuseVoiceSwapV1") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);

      this.color = NODE_COLOR;
      this.bgcolor = NODE_BGCOLOR;
      if (!this.size || this.size[0] < 760) this.size = [760, this.size?.[1] || 400];

      for (const w of this.widgets || []) {
        if (HIDDEN_WIDGET_NAMES.includes(w.name)) hideWidget(w);
        else if (SETTINGS_WIDGET_NAMES.includes(w.name)) hideWidget(w);
      }

      // reference_text (always visible, not behind the Settings toggle) is a
      // multiline STRING widget — ComfyUI's default sizing for those grows
      // quite tall with nothing capping it. Clamp it to a fixed height.
      // reference_text_2/_3 get the same clamp applied when the Settings
      // toggle reveals them (see settingsBtn.onclick below) — applying it here
      // too would fight with hideWidget's own computeSize override.
      const alwaysVisibleTextWidget = this.widgets?.find((w) => w.name === "reference_text");
      if (alwaysVisibleTextWidget) {
        alwaysVisibleTextWidget.computeSize = function (width) {
          return [width || 200, 64];
        };
      }

      const timelineDataWidget = this.widgets?.find((w) => w.name === "timeline_data");

      const container = document.createElement("div");
      const self = this;
      const widget = this.addDOMWidget("timeline_ui", "timeline_ui", container, {
        getValue: () => "",
        setValue: () => {},
      });
      widget.computeSize = function (width) {
        const nodeWidth = self.size?.[0] || width || 760;
        return [Math.max(10, nodeWidth - 30), CANVAS_HEIGHT + PREVIEW_HEIGHT + PANEL_HEIGHT + 60];
      };

      setTimeout(() => {
        try {
          self._timelineEditor = new VoiceSwapTimelineEditor(self, container, timelineDataWidget || widget);
        } catch (err) {
          console.error("[MuseVoiceSwap] timeline editor init failed:", err);
        }
      }, 0);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      if (onExecuted) onExecuted.apply(this, arguments);
      if (this._timelineEditor) {
        this._timelineEditor.applyExecutionResult(message);
      }
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const out = onResize?.apply(this, arguments);
      if (this._timelineEditor) {
        requestAnimationFrame(() => this._timelineEditor?.syncLayoutToNode());
      }
      return out;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this._timelineEditor?.destroy();
      return onRemoved?.apply(this, arguments);
    };
  },
});
