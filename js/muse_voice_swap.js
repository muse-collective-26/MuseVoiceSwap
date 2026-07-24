const { app } = window.comfyAPI.app;

// --- UI constants (reused from Muse Director for visual consistency) ---
const RULER_HEIGHT = 24;
const VOCAL_TRACK_HEIGHT = 100;
const BG_TRACK_HEIGHT = 60;
const CANVAS_HEIGHT = RULER_HEIGHT + VOCAL_TRACK_HEIGHT + BG_TRACK_HEIGHT;
const HANDLE_HIT_PX = 10;
const BOX_HANDLE_HIT_PX = 16; // generous -- the selection window's edges are the primary drag target now
const BOX_HANDLE_DRAW_PX = 8; // visual width of its handle bars, drawn so what you see matches what you can grab
const PANEL_HEIGHT = 130;
const PANEL_HEIGHT_IDLE = 40; // just the "click a segment" hint — no segment selected
const PREVIEW_HEIGHT = 130;
const OG_CANVAS_HEIGHT = 40; // secondary scrubber -- just peaks + its own red line, no ruler/segments
const OG_TOOLBAR_HEIGHT = 30;
const SETTINGS_PANEL_HEIGHT = 66; // ~2 wrapped rows of section buttons at a ~760px node width

const NODE_COLOR = "#2D1B69";
const NODE_BGCOLOR = "#0d0818";
const CANVAS_BG = "#2a2a2a";
const SELECTION_HIGHLIGHT = "#4fff8f";
const SELECTION_BOX_COLOR = "#ffcc33"; // the independent merge-selection window (distinct from segment-selected green)
const SEGMENT_FILL = "#3a4a6a";
const SEGMENT_FILL_LOCKED = "#5a4a3a";
const PEAK_COLOR = "#8fb7ff";
const BG_PEAK_COLOR = "#777";
const PLAYHEAD_COLOR = "#ff4444";
const LABEL_FONT = "12px sans-serif";

// Speaker color-coding (1=blue, 2=orange, 3=green) — locked overrides speaker fill.
const SPEAKER_FILL = { 1: "#3a4a6a", 2: "#7a5a2a", 3: "#2a6a4a" };
const SPEAKER_BORDER = { 1: "#8899bb", 2: "#e0b070", 3: "#7fd9a8" };

const HIDDEN_WIDGET_NAMES = ["timeline_data", "timeline_ui"];

// Permanently hidden -- no button ever reveals these, INPUT_TYPES defaults
// just apply. Each is either the dead side of a fallback pair whose live
// side has already won (vad_top_db only matters for energy_threshold VAD,
// which lost to silero; stretch_fft_size only matters for
// time_match_mode=stretch_to_fit, which lost to ripple) or a hardware/infra
// knob "auto"/"linear" already resolves correctly on this fixed single-GPU
// install (fish_device/precision/attention/keep_model_loaded, sep_chunk_fade_shape).
const ALWAYS_HIDDEN_WIDGET_NAMES = [
  "vad_top_db", "stretch_fft_size",
  "fish_device", "fish_precision", "fish_attention", "fish_keep_model_loaded",
  "sep_chunk_fade_shape",
];

// Advanced/tuning widgets collapsed behind the "Settings" toggle by default,
// grouped into named sections -- each renders as its own small DOM button in
// the settings panel (see _buildDom), NOT as a native LiteGraph widget, so
// none of this touches the node's serialized widgets_values positional array.
const SETTINGS_GROUPS = [
  { label: "Separation", names: ["sep_chunk_length", "sep_chunk_overlap"] },
  { label: "Segmentation (VAD)", names: ["vad_method", "min_segment_seconds", "min_gap_seconds", "silero_threshold", "silero_speech_pad_ms"] },
  { label: "Transcription", names: ["whisper_model_size", "whisper_language"] },
  { label: "Emotion Tags", names: ["emotion_tag_temperature"] },
  { label: "Fish S2", names: ["fish_language", "fish_max_new_tokens", "fish_chunk_length", "fish_temperature", "fish_top_p", "fish_repetition_penalty", "fish_seed"] },
  { label: "Time & Mix", names: ["time_match_mode", "bg_extend_mode", "bg_volume"] },
];

// reference_text/_2/_3 are forceInput sockets now (see voice_swap_node.py),
// not widgets -- they render as a plain connection point with no inline box
// regardless, so they need neither hiding/showing nor height-clamping here.

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
  .mvs-btn:disabled { opacity: 0.4; cursor: default; }
  .mvs-btn:disabled:hover { background: #222; border-color: #111; }
  .mvs-time { color: #aaa; font-size: 11px; font-variant-numeric: tabular-nums; margin-left: 4px; }
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
  .mvs-preview-row.checked { background: #2a230f; }
  .mvs-preview-row input[type="checkbox"] { margin: 0; cursor: pointer; flex: 0 0 auto; }
  .mvs-preview-time { color: #888; flex: 0 0 auto; white-space: nowrap; }
  .mvs-preview-text { color: #ddd; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mvs-preview-tag { color: #b39ddb; font-size: 10px; flex: 0 0 auto; }
  .mvs-batch-bar {
    display: none; align-items: center; gap: 8px; padding: 6px 8px;
    background: #241d0a; border: 1px solid #4a3a10; border-radius: 4px;
    font-size: 11px; color: #e0c07a;
  }
  .mvs-batch-bar .mvs-btn { padding: 3px 8px; font-size: 11px; }
  .mvs-og-label { color: #888; font-size: 11px; }
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
    // Segments checked in the transcript list below -- an alternate way to
    // pick what "+ Add Segment" merges (by identity, not by dragging a time
    // range), and a target for bulk speaker (re)assignment.
    this.checkedIds = new Set();

    // The independent "merge selection" window -- an amber overlay, decoupled
    // from any specific segment, always visible and resizable via its own two
    // edge handles. "+ Add Segment" merges whatever it currently covers into
    // one new segment. Never written into timeline_data -- purely ephemeral
    // client-side editing state.
    this.selectionBox = { start: 0, end: 1 };
    this.boxDrag = null; // "left" | "right" | null

    // --- Playback state (main scrubber -- previews the cloned/final mix) ---
    this.playheadTime = 0;
    this.isPlaying = false;
    this.audioBuffer = null; // decoded Web Audio buffer of the cloned/final mix
    this.audioContext = null;
    this._sourceNode = null;
    this._rulerDrag = false;
    this._isHovering = false;

    // --- Playback state (secondary scrubber -- previews the ORIGINAL
    // separated vocals, untouched by segmentation/cloning, so it has real
    // dialogue everywhere including gaps with no segment yet. Completely
    // independent play/pause + red line from the main one above.) ---
    this.ogPlayheadTime = 0;
    this.ogIsPlaying = false;
    this.ogAudioBuffer = null;
    this._ogSourceNode = null;
    this._ogRulerDrag = false;

    this._buildDom();
    this._loadFromWidgetValue();
    this._initSelectionBoxDefault();
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

    this.playBtn = document.createElement("button");
    this.playBtn.className = "mvs-btn";
    this.playBtn.textContent = "▶";
    this.playBtn.title = "Play/Pause (Space)";
    this.playBtn.onclick = () => this._togglePlay();
    toolbar.appendChild(this.playBtn);

    this.timeLabel = document.createElement("span");
    this.timeLabel.className = "mvs-time";
    this.timeLabel.textContent = "0:00.00";
    toolbar.appendChild(this.timeLabel);

    const addBtn = document.createElement("button");
    addBtn.className = "mvs-btn";
    addBtn.textContent = "+ Add Segment";
    addBtn.title = "Merges the amber window's range -- or, if you've ticked rows in the list below, merges those instead";
    addBtn.onclick = () => this._mergeSelectionIntoSegment();
    toolbar.appendChild(addBtn);

    const delBtn = document.createElement("button");
    delBtn.className = "mvs-btn";
    delBtn.textContent = "Delete Selected";
    delBtn.onclick = () => this._deleteSelected();
    toolbar.appendChild(delBtn);

    this.statusLabel = document.createElement("span");
    this.statusLabel.style.cssText = "color:#888;font-size:11px;margin-left:8px;";
    toolbar.appendChild(this.statusLabel);

    // Settings panel -- one small button per SETTINGS_GROUPS section (e.g.
    // "Fish S2", "Segmentation (VAD)") instead of dumping 20+ raw params in
    // one flat list. Clicking a section button reveals just that section's
    // real native widgets; the master "Settings" toggle below just shows/
    // hides this whole panel of section buttons. Built as plain DOM here
    // (not native LiteGraph widgets) specifically so it never touches the
    // node's serialized widgets_values array.
    this.settingsPanel = document.createElement("div");
    this.settingsPanel.style.cssText = "display:none; flex-wrap:wrap; gap:4px; padding:2px 0;";
    this._settingsGroupOpen = {};
    for (const group of SETTINGS_GROUPS) {
      const memberWidgets = group.names
        .map((n) => this.node.widgets.find((w) => w.name === n))
        .filter(Boolean);
      if (!memberWidgets.length) continue;

      const groupBtn = document.createElement("button");
      groupBtn.className = "mvs-btn";
      groupBtn.style.fontSize = "11px";
      const updateGroupBtn = () => {
        groupBtn.textContent = `${this._settingsGroupOpen[group.label] ? "▾" : "▸"} ${group.label}`;
      };
      updateGroupBtn();
      groupBtn.onclick = () => {
        this._settingsGroupOpen[group.label] = !this._settingsGroupOpen[group.label];
        for (const w of memberWidgets) {
          if (this._settingsGroupOpen[group.label]) showWidget(w);
          else hideWidget(w);
        }
        updateGroupBtn();
        this.node.setDirtyCanvas(true, true);
        this.syncLayoutToNode();
      };
      this.settingsPanel.appendChild(groupBtn);
    }

    const settingsBtn = document.createElement("button");
    settingsBtn.className = "mvs-btn mvs-settings-btn";
    let settingsVisible = false;
    const updateSettingsBtn = () => {
      settingsBtn.textContent = settingsVisible ? "▲ Settings" : "▼ Settings";
    };
    updateSettingsBtn();
    settingsBtn.onclick = () => {
      settingsVisible = !settingsVisible;
      this._settingsPanelVisible = settingsVisible;
      this.settingsPanel.style.display = settingsVisible ? "flex" : "none";
      updateSettingsBtn();
      this.node.setDirtyCanvas(true, true);
      this.syncLayoutToNode();
    };
    toolbar.appendChild(settingsBtn);

    this.canvas = document.createElement("canvas");
    this.canvas.className = "mvs-canvas";
    this.canvas.height = CANVAS_HEIGHT;

    // Secondary scrubber -- original (uncloned) vocals, own play/pause + own
    // red line only, no segments/editing. Purely for listening to find what's
    // actually in a gap the main timeline hasn't turned into a segment yet.
    const ogToolbar = document.createElement("div");
    ogToolbar.className = "mvs-toolbar";

    this.ogPlayBtn = document.createElement("button");
    this.ogPlayBtn.className = "mvs-btn";
    this.ogPlayBtn.textContent = "▶";
    this.ogPlayBtn.title = "Play/Pause the original (uncloned) audio";
    this.ogPlayBtn.onclick = () => this._ogTogglePlay();
    ogToolbar.appendChild(this.ogPlayBtn);

    this.ogTimeLabel = document.createElement("span");
    this.ogTimeLabel.className = "mvs-time";
    this.ogTimeLabel.textContent = "0:00.00";
    ogToolbar.appendChild(this.ogTimeLabel);

    const ogLabel = document.createElement("span");
    ogLabel.className = "mvs-og-label";
    ogLabel.textContent = "Original audio (reference only, before cloning)";
    ogToolbar.appendChild(ogLabel);

    this.ogCanvas = document.createElement("canvas");
    this.ogCanvas.className = "mvs-canvas";
    this.ogCanvas.height = OG_CANVAS_HEIGHT;

    this.previewList = document.createElement("div");
    this.previewList.className = "mvs-preview";
    this._renderPreviewList();

    this.batchBar = document.createElement("div");
    this.batchBar.className = "mvs-batch-bar";
    this._renderBatchBar();

    this.panel = document.createElement("div");
    this.panel.className = "mvs-panel";
    this._renderPanel();

    this.wrapper.appendChild(toolbar);
    this.wrapper.appendChild(this.settingsPanel);
    this.wrapper.appendChild(this.canvas);
    this.wrapper.appendChild(ogToolbar);
    this.wrapper.appendChild(this.ogCanvas);
    this.wrapper.appendChild(this.previewList);
    this.wrapper.appendChild(this.batchBar);
    this.wrapper.appendChild(this.panel);
    this.container.appendChild(this.wrapper);

    this.canvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
    this.ogCanvas.addEventListener("mousedown", (e) => this._onOgMouseDown(e));
    window.addEventListener("mousemove", (e) => this._onMouseMove(e));
    window.addEventListener("mousemove", (e) => this._onOgMouseMove(e));
    window.addEventListener("mouseup", () => this._onMouseUp());

    this.wrapper.addEventListener("mouseenter", () => { this._isHovering = true; });
    this.wrapper.addEventListener("mouseleave", () => { this._isHovering = false; });

    this._onKeyDown = (e) => {
      if (!this._isHovering) return;
      const activeTag = document.activeElement ? document.activeElement.tagName : "";
      if (activeTag === "INPUT" || activeTag === "TEXTAREA") return;

      if (e.key === " " || e.code === "Space") {
        this._togglePlay();
        e.preventDefault();
        e.stopPropagation();
      }
    };
    window.addEventListener("keydown", this._onKeyDown);
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

  // Positions the merge-selection window somewhere immediately useful: over
  // the first existing segment if there is one, else a small window at the
  // very start of the clip.
  _initSelectionBoxDefault() {
    const segs = [...this.timeline.vocalSegments].sort((a, b) => a.start - b.start);
    if (segs.length > 0) {
      this.selectionBox = { start: segs[0].start, end: segs[0].end };
    } else {
      const total = this._totalDuration();
      this.selectionBox = { start: 0, end: Math.min(1.0, total) };
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
    this.checkedIds.clear();
    this._initSelectionBoxDefault();
    this._pauseAudio();
    this.playheadTime = 0;
    this.audioBuffer = null;
    if (payload.audio_preview_b64) {
      this._decodeAudioPreview(payload.audio_preview_b64, "audioBuffer");
    }

    this._ogPauseAudio();
    this.ogPlayheadTime = 0;
    this.ogAudioBuffer = null;
    if (payload.og_audio_preview_b64) {
      this._decodeAudioPreview(payload.og_audio_preview_b64, "ogAudioBuffer").then(() => this._renderOgCanvas());
    }

    this._commitChanges();
    this.render();
    this._renderPanel();
    this._renderOgCanvas();
  }

  // Decodes a base64 WAV into a Web Audio buffer and stores it on `this[targetProp]`.
  async _decodeAudioPreview(b64, targetProp) {
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      const binary = window.atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      this[targetProp] = await this.audioContext.decodeAudioData(bytes.buffer);
    } catch (err) {
      console.warn(`[MuseVoiceSwap] failed to decode audio preview (${targetProp}):`, err);
      this[targetProp] = null;
    }
  }

  _totalDuration() {
    // Deliberately source-space only (seg.start/seg.end), never placed_start/
    // placed_duration (the segment's position in the ripple-placed FINAL
    // output, which drifts from the source once a clone runs longer/shorter
    // than its original line) -- this editor only ever describes positions in
    // the source audio; final_duration below already accounts for any output
    // drift without needing the boxes themselves to track it.
    const fromSegs = this.timeline.vocalSegments.reduce(
      (max, s) => Math.max(max, s.end),
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

  render(opts) {
    const light = !!(opts && opts.light); // skip previewList DOM rebuild during playback ticking / box dragging
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
    // A corrupt/runaway segment duration (bad upstream data) must never be able
    // to hang the tab by turning this into a million-iteration draw loop --
    // cap the tick count outright regardless of how large `total` gets.
    let step = total > 20 ? 5 : total > 8 ? 2 : 1;
    const MAX_RULER_TICKS = 300;
    if (total / step > MAX_RULER_TICKS) step = total / MAX_RULER_TICKS;
    for (let t = 0; t <= total; t += step) {
      const x = this._timeToX(t, width);
      ctx.fillRect(x, RULER_HEIGHT - 6, 1, 6);
      ctx.fillText(fmtTime(t), x + 2, RULER_HEIGHT - 8);
    }

    // vocal_peaks/bg_peaks are always computed from the ORIGINAL source-length
    // audio (400 fixed samples spanning source_duration), but `total` can be
    // larger than source_duration once ripple placement drifts the final
    // output longer than the source. _drawPeaks stretches its fixed sample
    // count across whatever width it's given, so without this the waveform
    // would visually stretch out to fill the drifted total instead of
    // stopping where the real source audio actually ends.
    const sourcePeaksWidth = this._timeToX(this.timeline.source_duration || total, width);

    // Vocal track
    const vocalY = RULER_HEIGHT;
    ctx.fillStyle = "#232338";
    ctx.fillRect(0, vocalY, width, VOCAL_TRACK_HEIGHT);
    this._drawPeaks(ctx, this.timeline.vocalPeaks, 0, vocalY, sourcePeaksWidth, VOCAL_TRACK_HEIGHT, PEAK_COLOR);

    for (const seg of this.timeline.vocalSegments) {
      // Source-space position (see _totalDuration) -- always matches the
      // waveform peaks behind it, which are the original source audio.
      const start = seg.start;
      const dur = seg.end - seg.start;
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
      ctx.fillText("No segments yet — queue the node once to auto-detect, or drag the amber window and click + Add Segment.", 8, vocalY + VOCAL_TRACK_HEIGHT / 2);
    }

    // BG reference track (read-only)
    const bgY = vocalY + VOCAL_TRACK_HEIGHT;
    ctx.fillStyle = "#1c1c1c";
    ctx.fillRect(0, bgY, width, BG_TRACK_HEIGHT);
    ctx.fillStyle = "#666";
    ctx.font = "10px sans-serif";
    ctx.fillText("Background (read-only)", 4, bgY + 10);
    this._drawPeaks(ctx, this.timeline.bgWaveformPeaks, 0, bgY + 12, sourcePeaksWidth, BG_TRACK_HEIGHT - 12, BG_PEAK_COLOR);

    // Independent merge-selection window (amber) -- always visible, decoupled
    // from any specific segment. "+ Add Segment" merges whatever it covers.
    // Drawn only over the vocal track (matches segment height -- doesn't
    // visually intrude on the read-only bg track below it), drawn after the
    // segments so it's never painted over. Its HIT-TEST zone still spans the
    // full canvas height (see _hitSelectionBoxHandle) -- the ruler strip above
    // remains a clear, uncrowded place to grab an edge from even when it
    // visually lines up with an existing segment's own border.
    {
      const bx = this._timeToX(this.selectionBox.start, width);
      const bw = Math.max(4, this._timeToX(this.selectionBox.end, width) - bx);
      ctx.fillStyle = SELECTION_BOX_COLOR;
      ctx.globalAlpha = 0.12;
      ctx.fillRect(bx, vocalY, bw, VOCAL_TRACK_HEIGHT);
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = SELECTION_BOX_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(bx, vocalY, bw, VOCAL_TRACK_HEIGHT);
      // Large, always-visible grab targets sized to match BOX_HANDLE_HIT_PX
      // (what you see is what you can grab).
      ctx.fillRect(bx - BOX_HANDLE_DRAW_PX / 2, vocalY, BOX_HANDLE_DRAW_PX, VOCAL_TRACK_HEIGHT);
      ctx.fillRect(bx + bw - BOX_HANDLE_DRAW_PX / 2, vocalY, BOX_HANDLE_DRAW_PX, VOCAL_TRACK_HEIGHT);
    }

    // Playhead (drawn over ruler + both tracks)
    const playX = this._timeToX(Math.min(this.playheadTime, total), width);
    ctx.strokeStyle = PLAYHEAD_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, CANVAS_HEIGHT);
    ctx.stroke();

    const selDur = this.selectionBox.end - this.selectionBox.start;
    this.statusLabel.textContent =
      `${this.timeline.vocalSegments.length} segment(s) · total ${fmtTime(total)} · ` +
      `selection ${fmtTime(this.selectionBox.start)}–${fmtTime(this.selectionBox.end)} (${selDur.toFixed(2)}s)`;
    if (this.timeLabel) this.timeLabel.textContent = fmtTime(this.playheadTime);
    if (this.playBtn) this.playBtn.textContent = this.isPlaying ? "⏸" : "▶";
    if (!light) this._renderPreviewList();
    this._renderOgCanvas();
  }

  _renderPreviewList() {
    if (!this.previewList) return;
    this.previewList.innerHTML = "";

    // Drop checks for segments that no longer exist (deleted, merged away, or
    // replaced by a fresh run).
    const validIds = new Set(this.timeline.vocalSegments.map((s) => s.id));
    for (const id of [...this.checkedIds]) {
      if (!validIds.has(id)) this.checkedIds.delete(id);
    }

    const sorted = [...this.timeline.vocalSegments].sort((a, b) => a.start - b.start);
    if (sorted.length === 0) {
      const empty = document.createElement("div");
      empty.className = "mvs-empty";
      empty.textContent = "No segments to preview yet.";
      this.previewList.appendChild(empty);
      this._renderBatchBar();
      return;
    }

    for (const seg of sorted) {
      const row = document.createElement("div");
      const isChecked = this.checkedIds.has(seg.id);
      row.className = "mvs-preview-row" + (seg.id === this.selectedId ? " selected" : "") + (isChecked ? " checked" : "");
      row.onclick = () => {
        this.selectedId = seg.id;
        this.render();
        this._renderPanel();
      };

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = isChecked;
      checkbox.title = "Tick to include in a batch merge or bulk speaker assignment";
      checkbox.addEventListener("click", (e) => e.stopPropagation()); // don't also trigger row's select-on-click
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) this.checkedIds.add(seg.id);
        else this.checkedIds.delete(seg.id);
        row.classList.toggle("checked", checkbox.checked);
        this._renderBatchBar();
      });
      row.appendChild(checkbox);

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

    this._renderBatchBar();
  }

  // Shown only while 1+ rows are ticked -- lets you bulk-relabel their
  // speaker without merging, and reminds you that "+ Add Segment" will act
  // on this set instead of the amber window while any are ticked.
  _renderBatchBar() {
    if (!this.batchBar) return;
    const n = this.checkedIds.size;
    if (n === 0) {
      this.batchBar.style.display = "none";
      this.batchBar.innerHTML = "";
      return;
    }

    this.batchBar.style.display = "flex";
    this.batchBar.innerHTML = "";

    const label = document.createElement("span");
    label.textContent = `${n} ticked — "+ Add Segment" merges these · Assign speaker:`;
    this.batchBar.appendChild(label);

    for (const spk of [1, 2, 3]) {
      const btn = document.createElement("button");
      btn.className = "mvs-btn";
      btn.textContent = `S${spk}`;
      btn.style.color = SPEAKER_BORDER[spk] || "#e0e0e0";
      btn.onclick = () => this._assignSpeakerToChecked(spk);
      this.batchBar.appendChild(btn);
    }

    const clearBtn = document.createElement("button");
    clearBtn.className = "mvs-btn";
    clearBtn.textContent = "Clear";
    clearBtn.onclick = () => {
      this.checkedIds.clear();
      this._renderPreviewList();
    };
    this.batchBar.appendChild(clearBtn);
  }

  _assignSpeakerToChecked(speaker) {
    for (const seg of this.timeline.vocalSegments) {
      if (this.checkedIds.has(seg.id)) seg.speaker = speaker;
    }
    this._commitChanges();
    this.render();
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

  // Closest matching EXISTING segment under (x, y) -- pure selection, no drag
  // affordance of its own anymore (dragging is now the selection window's job).
  _hitSegment(x, y, width) {
    const vocalY = RULER_HEIGHT;
    if (y < vocalY || y > vocalY + VOCAL_TRACK_HEIGHT) return null;
    let best = null;
    let bestDist = Infinity;
    for (const seg of this.timeline.vocalSegments) {
      const start = seg.start;
      const dur = seg.end - seg.start;
      const sx = this._timeToX(start, width);
      const ex = this._timeToX(start + dur, width);
      if (x >= sx && x <= ex) {
        const dist = Math.abs(x - (sx + ex) / 2);
        if (dist < bestDist) {
          bestDist = dist;
          best = seg;
        }
      }
    }
    return best;
  }

  // "left" / "right" if (x, y) is near one of the merge-selection window's
  // own edges, else null. Checked across the FULL canvas height (not just the
  // vocal track) and with a generous tolerance -- this is the one thing in
  // the timeline that's actually draggable now (besides the playhead), so it
  // needs to be very easy to grab even when its edge lines up with an
  // existing segment's own border.
  _hitSelectionBoxHandle(x, y, width) {
    if (y < 0 || y > CANVAS_HEIGHT) return null;
    const bx = this._timeToX(this.selectionBox.start, width);
    const bw = Math.max(4, this._timeToX(this.selectionBox.end, width) - bx);
    if (Math.abs(x - bx) <= BOX_HANDLE_HIT_PX) return "left";
    if (Math.abs(x - (bx + bw)) <= BOX_HANDLE_HIT_PX) return "right";
    return null;
  }

  // Converts a mouse event's screen-space position into a canvas's own
  // internal pixel coordinate space (the one _timeToX/_xToTime/hit-testing
  // all operate in). These only match 1:1 when ComfyUI's graph is at 100%
  // zoom -- at any other zoom level, getBoundingClientRect() returns the
  // on-screen (zoomed) size while the canvas's width/height stay fixed, so
  // every drag target was silently off by the zoom ratio without this.
  _getCanvasXY(e, canvasEl) {
    const el = canvasEl || this.canvas;
    const rect = el.getBoundingClientRect();
    const scaleX = el.width / rect.width;
    const scaleY = el.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  _onMouseDown(e) {
    const { x, y } = this._getCanvasXY(e);
    const width = this.canvas.width;

    // Grabbing the red playhead line itself, anywhere along its full height
    // (not just the top ruler strip) -- takes priority so it's always
    // directly draggable, in place of "play, overshoot, restart from 0".
    const playX = this._timeToX(Math.min(this.playheadTime, this._totalDuration()), width);
    if (Math.abs(x - playX) <= 6 && y >= 0 && y <= CANVAS_HEIGHT) {
      this._rulerDrag = true;
      this._seekTo(this._xToTime(x, width));
      return;
    }

    // The selection window's handles are checked next, across the FULL
    // canvas height (ruler + vocal track + bg track) -- not just where the
    // handle bar is drawn -- so there's always a clear, uncrowded strip
    // (e.g. the ruler above, or the bg track below) to grab it from even
    // when its edge lines up with an existing segment's own border.
    const boxHit = this._hitSelectionBoxHandle(x, y, width);
    if (boxHit) {
      this.boxDrag = boxHit;
      return;
    }

    if (y < RULER_HEIGHT) {
      this._rulerDrag = true;
      this._seekTo(this._xToTime(x, width));
      return;
    }

    // Plain segment click: pure selection (for viewing/editing transcript,
    // speaker, lock in the panel below) -- no drag of its own.
    const seg = this._hitSegment(x, y, width);
    this.selectedId = seg ? seg.id : null;
    this.render();
    this._renderPanel();
  }

  _onMouseMove(e) {
    if (this._rulerDrag) {
      const { x } = this._getCanvasXY(e);
      this._seekTo(this._xToTime(x, this.canvas.width));
      return;
    }

    if (this.boxDrag) {
      const { x } = this._getCanvasXY(e);
      const width = this.canvas.width;
      const t = this._xToTime(x, width);
      const MIN_WIDTH = 0.1;
      if (this.boxDrag === "left") {
        this.selectionBox.start = Math.max(0, Math.min(t, this.selectionBox.end - MIN_WIDTH));
      } else {
        const total = this._totalDuration();
        this.selectionBox.end = Math.min(total, Math.max(t, this.selectionBox.start + MIN_WIDTH));
      }
      this.render({ light: true });
      return;
    }

    // Hover-only pass: surface the drag affordance via cursor shape.
    const { x, y } = this._getCanvasXY(e);
    if (x >= 0 && x <= this.canvas.width && y >= 0 && y <= this.canvas.height) {
      const playX = this._timeToX(Math.min(this.playheadTime, this._totalDuration()), this.canvas.width);
      if (Math.abs(x - playX) <= 6) {
        this.canvas.style.cursor = "col-resize";
      } else if (this._hitSelectionBoxHandle(x, y, this.canvas.width)) {
        this.canvas.style.cursor = "ew-resize";
      } else {
        this.canvas.style.cursor = "pointer";
      }
    }
  }

  _onMouseUp() {
    this._rulerDrag = false;
    this.boxDrag = null;
    this._ogRulerDrag = false;
  }

  // --- Secondary scrubber (original/uncloned audio) -- click/drag its own
  // red line only, no segments, no editing.

  _onOgMouseDown(e) {
    const { x } = this._getCanvasXY(e, this.ogCanvas);
    this._ogRulerDrag = true;
    this._ogSeekTo(this._xToTime(x, this.ogCanvas.width));
  }

  _onOgMouseMove(e) {
    if (!this._ogRulerDrag) return;
    const { x } = this._getCanvasXY(e, this.ogCanvas);
    this._ogSeekTo(this._xToTime(x, this.ogCanvas.width));
  }

  // --- Playback ---

  _seekTo(t) {
    const total = this._totalDuration();
    this.playheadTime = Math.max(0, Math.min(total, t));
    if (this.isPlaying) {
      this._playAudio(); // restart the buffer source from the new offset
    } else {
      this.render({ light: true });
    }
  }

  async _togglePlay() {
    if (this.isPlaying) {
      this._pauseAudio();
    } else {
      await this._playAudio();
    }
  }

  async _playAudio() {
    if (!this.audioBuffer) return;
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioContext.state !== "running") {
      try { await this.audioContext.resume(); } catch (err) { /* ignore */ }
    }
    this._stopSourceNode();

    const total = this._totalDuration();
    if (this.playheadTime >= total) this.playheadTime = 0;

    const source = this.audioContext.createBufferSource();
    source.buffer = this.audioBuffer;
    source.connect(this.audioContext.destination);
    const offset = Math.min(this.playheadTime, Math.max(0, this.audioBuffer.duration - 0.01));
    source.start(0, offset);
    source.onended = () => {
      if (this._sourceNode === source) {
        this._sourceNode = null;
        this.isPlaying = false;
        this.render({ light: true });
      }
    };

    this._sourceNode = source;
    this._playbackStartCtxTime = this.audioContext.currentTime;
    this._playbackStartOffset = offset;
    this.isPlaying = true;
    this.render({ light: true });
    this._tickPlayback();
  }

  _stopSourceNode() {
    if (this._sourceNode) {
      const node = this._sourceNode;
      node.onended = null;
      try { node.stop(); } catch (err) { /* already stopped */ }
      this._sourceNode = null;
    }
    if (this._playbackRAF) {
      cancelAnimationFrame(this._playbackRAF);
      this._playbackRAF = null;
    }
  }

  _pauseAudio() {
    this._stopSourceNode();
    this.isPlaying = false;
    this.render({ light: true });
  }

  _tickPlayback() {
    if (!this.isPlaying) return;
    const elapsed = this.audioContext.currentTime - this._playbackStartCtxTime;
    this.playheadTime = this._playbackStartOffset + elapsed;

    const total = this._totalDuration();
    if (this.playheadTime >= total) {
      this.playheadTime = total;
      this._pauseAudio();
      return;
    }

    this.render({ light: true });
    this._playbackRAF = requestAnimationFrame(() => this._tickPlayback());
  }

  // --- Secondary scrubber's own playback engine -- entirely independent
  // isPlaying/playhead/source-node state from the main one above, sharing
  // only the AudioContext (harmless -- it's not tied to a specific buffer).

  _ogSeekTo(t) {
    const total = this._totalDuration();
    this.ogPlayheadTime = Math.max(0, Math.min(total, t));
    if (this.ogIsPlaying) {
      this._ogPlayAudio();
    } else {
      this._renderOgCanvas();
    }
  }

  async _ogTogglePlay() {
    if (this.ogIsPlaying) {
      this._ogPauseAudio();
    } else {
      await this._ogPlayAudio();
    }
  }

  async _ogPlayAudio() {
    if (!this.ogAudioBuffer) return;
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.audioContext.state !== "running") {
      try { await this.audioContext.resume(); } catch (err) { /* ignore */ }
    }
    this._ogStopSourceNode();

    const total = this._totalDuration();
    if (this.ogPlayheadTime >= total) this.ogPlayheadTime = 0;

    const source = this.audioContext.createBufferSource();
    source.buffer = this.ogAudioBuffer;
    source.connect(this.audioContext.destination);
    const offset = Math.min(this.ogPlayheadTime, Math.max(0, this.ogAudioBuffer.duration - 0.01));
    source.start(0, offset);
    source.onended = () => {
      if (this._ogSourceNode === source) {
        this._ogSourceNode = null;
        this.ogIsPlaying = false;
        this._renderOgCanvas();
      }
    };

    this._ogSourceNode = source;
    this._ogPlaybackStartCtxTime = this.audioContext.currentTime;
    this._ogPlaybackStartOffset = offset;
    this.ogIsPlaying = true;
    this._renderOgCanvas();
    this._ogTickPlayback();
  }

  _ogStopSourceNode() {
    if (this._ogSourceNode) {
      const node = this._ogSourceNode;
      node.onended = null;
      try { node.stop(); } catch (err) { /* already stopped */ }
      this._ogSourceNode = null;
    }
    if (this._ogPlaybackRAF) {
      cancelAnimationFrame(this._ogPlaybackRAF);
      this._ogPlaybackRAF = null;
    }
  }

  _ogPauseAudio() {
    this._ogStopSourceNode();
    this.ogIsPlaying = false;
    this._renderOgCanvas();
  }

  _ogTickPlayback() {
    if (!this.ogIsPlaying) return;
    const elapsed = this.audioContext.currentTime - this._ogPlaybackStartCtxTime;
    this.ogPlayheadTime = this._ogPlaybackStartOffset + elapsed;

    const total = this._totalDuration();
    if (this.ogPlayheadTime >= total) {
      this.ogPlayheadTime = total;
      this._ogPauseAudio();
      return;
    }

    this._renderOgCanvas();
    this._ogPlaybackRAF = requestAnimationFrame(() => this._ogTickPlayback());
  }

  // Minimal draw: waveform peaks (reuses the same vocal peaks the main
  // canvas draws) + its own red playhead line. No ruler, no segments, no box.
  _renderOgCanvas() {
    if (!this.ogCanvas) return;
    const width = Math.max(200, this.node.size?.[0] ? this.node.size[0] - 30 : 700);
    this.ogCanvas.width = width;
    const ctx = this.ogCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, OG_CANVAS_HEIGHT);

    ctx.fillStyle = "#232338";
    ctx.fillRect(0, 0, width, OG_CANVAS_HEIGHT);
    const total = this._totalDuration();
    // Same fix as the main canvas: vocal_peaks only ever spans source_duration,
    // which can be shorter than `total` once ripple placement drifts the
    // final output longer than the source.
    const sourcePeaksWidth = this._timeToX(this.timeline.source_duration || total, width);
    this._drawPeaks(ctx, this.timeline.vocalPeaks, 0, 0, sourcePeaksWidth, OG_CANVAS_HEIGHT, PEAK_COLOR);

    const playX = this._timeToX(Math.min(this.ogPlayheadTime, total), width);
    ctx.strokeStyle = PLAYHEAD_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, OG_CANVAS_HEIGHT);
    ctx.stroke();

    if (this.ogTimeLabel) this.ogTimeLabel.textContent = fmtTime(this.ogPlayheadTime);
    if (this.ogPlayBtn) this.ogPlayBtn.textContent = this.ogIsPlaying ? "⏸" : "▶";
  }

  // Merges either (a) whatever segments are ticked in the list below, if any
  // are, or (b) everything the amber selection window currently covers, into
  // one new segment. Only the audio ACTUALLY INSIDE that range is consumed --
  // if the range only partially overlaps an existing segment, whatever part
  // of that segment falls outside the range is preserved as its own leftover
  // segment rather than silently discarded. Listen with Play, either drag the
  // window over a group of small/misdetected segments OR tick them in the
  // list, hit "+ Add Segment", then assign the new segment's speaker below.
  _mergeSelectionIntoSegment() {
    let start, end, inheritedSpeaker = 1;

    if (this.checkedIds.size > 0) {
      const checked = this.timeline.vocalSegments.filter((s) => this.checkedIds.has(s.id));
      if (checked.length === 0) return;
      start = Math.min(...checked.map((s) => s.start));
      end = Math.max(...checked.map((s) => s.end));
      // If they were already bulk-assigned the same speaker, keep it rather
      // than silently resetting to Speaker 1.
      inheritedSpeaker = checked[0].speaker || 1;
    } else {
      start = this.selectionBox.start;
      end = this.selectionBox.end;
    }
    if (end - start < 0.05) return;

    const EPS = 0.01;
    const survivors = [];
    const leftovers = [];
    for (const seg of this.timeline.vocalSegments) {
      const overlaps = seg.end > start && seg.start < end;
      if (!overlaps) {
        survivors.push(seg);
        continue;
      }
      // Consumed by the merge -- but its transcript no longer matches either
      // trimmed piece, so leftovers get blank text (auto-retranscribed on the
      // next run, same as a manual split).
      if (seg.start < start - EPS) {
        leftovers.push({ ...seg, id: uid(), start: seg.start, end: start, text: "", emotion_text: "", locked: false });
      }
      if (seg.end > end + EPS) {
        leftovers.push({ ...seg, id: uid(), start: end, end: seg.end, text: "", emotion_text: "", locked: false });
      }
    }

    const merged = { id: uid(), start, end, text: "", emotion_text: "", speaker: inheritedSpeaker, locked: false };
    this.timeline.vocalSegments = [...survivors, ...leftovers, merged].sort((a, b) => a.start - b.start);

    this.selectedId = merged.id;
    this.checkedIds.clear();

    // Auto-advance the window to right after the merged range either way, so
    // the drag-based flow keeps working seamlessly after a checkbox-based merge.
    const total = this._totalDuration();
    const nextStart = Math.min(end, total);
    const nextWidth = Math.min(1.0, Math.max(0.2, total - nextStart));
    this.selectionBox = { start: nextStart, end: Math.min(total, nextStart + nextWidth) };

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
      // No segment selected -- the node's reserved height for this widget
      // (see computeSize) shrinks to just this hint, so nudge LiteGraph to
      // re-measure and shrink the node now rather than on the next
      // unrelated redraw.
      this.node.setDirtyCanvas(true, true);
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

    // A segment just got selected -- reserved height grows back to
    // PANEL_HEIGHT (see computeSize); nudge LiteGraph to re-measure now.
    this.node.setDirtyCanvas(true, true);
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
    this._stopSourceNode();
    this._ogStopSourceNode();
    if (this._onKeyDown) window.removeEventListener("keydown", this._onKeyDown);
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

      const allGroupedSettingsNames = SETTINGS_GROUPS.flatMap((g) => g.names);
      for (const w of this.widgets || []) {
        if (HIDDEN_WIDGET_NAMES.includes(w.name)) hideWidget(w);
        else if (ALWAYS_HIDDEN_WIDGET_NAMES.includes(w.name)) hideWidget(w);
        else if (allGroupedSettingsNames.includes(w.name)) hideWidget(w);
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
        const panelH = self._timelineEditor?.selectedId ? PANEL_HEIGHT : PANEL_HEIGHT_IDLE;
        const settingsPanelH = self._timelineEditor?._settingsPanelVisible ? SETTINGS_PANEL_HEIGHT : 0;
        return [
          Math.max(10, nodeWidth - 30),
          CANVAS_HEIGHT + OG_TOOLBAR_HEIGHT + OG_CANVAS_HEIGHT + PREVIEW_HEIGHT + panelH + settingsPanelH + 60,
        ];
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
