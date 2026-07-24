"""
Pure signal-processing functions for MuseVoiceSwap. No ComfyUI imports here —
these are exercised standalone (via the embedded python interpreter) before
ever being wired into the node, so keep this file free of `comfy.*`/`folder_paths`
dependencies.

AUDIO dict convention (matches ComfyUI's own AUDIO type):
    {"waveform": torch.Tensor[1, C, S], "sample_rate": int}
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Segment detection
# ---------------------------------------------------------------------------

def _speaker_aware_merge(
    intervals_sec: list[tuple[float, float]],
    vocals_audio: dict,
    min_gap_seconds: float,
    merge_similarity_threshold: float,
) -> list[tuple[float, float]]:
    """Same job as the plain gap-based merge, but a short gap only merges two
    intervals when a quick speaker-embedding check says they're likely the
    same voice. A brief interjection from someone else wedged into a small
    gap (an "ouch!", "help!", a quick reaction) stays as its own interval
    instead of being silently absorbed into the surrounding speaker's segment,
    which is what plain gap-based merging always did regardless of who was
    actually talking.

    Embeddings on very short intervals are inherently noisy (documented
    elsewhere in this file — ECAPA-TDNN needs more than a word or two of
    audio to be reliable), so this is a best-effort signal, not a guarantee.
    Whenever the check itself fails (too short to embed, encoder unavailable,
    etc.) this falls back to merging — the original, always-merge behavior —
    rather than guessing a split that might not be real.
    """
    merged: list[tuple[float, float]] = []
    for s, e in intervals_sec:
        if merged and (s - merged[-1][1]) < min_gap_seconds:
            prev_s, prev_e = merged[-1]
            same_speaker = True
            try:
                emb_a = compute_speaker_embedding(crop_audio(vocals_audio, prev_s, prev_e))
                emb_b = compute_speaker_embedding(crop_audio(vocals_audio, s, e))
                sim = float(
                    np.dot(emb_a, emb_b)
                    / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
                )
                same_speaker = sim >= merge_similarity_threshold
            except Exception as ex:
                print(f"[MuseVoiceSwap] Speaker-aware merge check failed, defaulting to merge: {ex}")
                same_speaker = True
            if same_speaker:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        else:
            merged.append((s, e))
    return merged


def _intervals_to_segments(
    intervals_sec: list[tuple[float, float]],
    total_dur: float,
    min_segment_seconds: float,
    min_gap_seconds: float,
    vocals_audio: dict | None = None,
    speaker_aware_merge: bool = False,
    merge_similarity_threshold: float = 0.3,
) -> list[dict]:
    """Shared merge/filter/fallback logic for turning a raw list of (start, end)
    speech intervals (in seconds, from whichever VAD method) into the segment
    list format the rest of the pipeline expects. Falls back to treating the
    whole clip as one segment if nothing is detected, so 1-segment and
    N-segment sources share one code path."""
    if not intervals_sec:
        return [{"id": "seg_1", "start": 0.0, "end": round(total_dur, 3), "text": ""}]

    # Merge intervals separated by gaps shorter than min_gap_seconds — unless
    # speaker_aware_merge is on, in which case a short gap only merges when
    # both sides sound like the same voice (see _speaker_aware_merge above).
    if speaker_aware_merge and vocals_audio is not None and len(intervals_sec) > 1:
        merged = _speaker_aware_merge(intervals_sec, vocals_audio, min_gap_seconds, merge_similarity_threshold)
    else:
        merged = []
        for s, e in intervals_sec:
            if merged and (s - merged[-1][1]) < min_gap_seconds:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

    # Drop segments shorter than min_segment_seconds (breath/click artifacts).
    segments = []
    for s, e in merged:
        if (e - s) < min_segment_seconds:
            continue
        segments.append({"id": f"seg_{len(segments) + 1}", "start": round(s, 3), "end": round(e, 3), "text": ""})

    if not segments:
        return [{"id": "seg_1", "start": 0.0, "end": round(total_dur, 3), "text": ""}]

    return segments


def detect_speech_segments(
    vocals_audio: dict,
    top_db: float = 35.0,
    min_segment_seconds: float = 0.3,
    min_gap_seconds: float = 0.2,
    speaker_aware_merge: bool = False,
    merge_similarity_threshold: float = 0.3,
) -> list[dict]:
    """Energy-threshold VAD (librosa.effects.split) -- the original/legacy
    method, kept as the "energy_threshold" vad_method option so there's an
    instant revert path if detect_speech_segments_silero doesn't work out for
    some source material. Its main weakness: it can't tell quiet-but-real
    speech apart from actual silence, since it's just a relative dB threshold."""
    import librosa

    waveform = vocals_audio["waveform"].squeeze(0)  # [C, S]
    mono = waveform.mean(dim=0).cpu().numpy().astype(np.float32)
    sr = vocals_audio["sample_rate"]

    total_dur = mono.shape[-1] / sr if sr else 0.0
    if mono.shape[-1] == 0 or total_dur <= 0:
        return []

    intervals = librosa.effects.split(mono, top_db=top_db)
    intervals_sec = [(float(s) / sr, float(e) / sr) for s, e in intervals]
    return _intervals_to_segments(
        intervals_sec, total_dur, min_segment_seconds, min_gap_seconds,
        vocals_audio=vocals_audio, speaker_aware_merge=speaker_aware_merge,
        merge_similarity_threshold=merge_similarity_threshold,
    )


_SILERO_VAD_CACHE: dict[str, object] = {}


def _load_silero_vad():
    if "model" in _SILERO_VAD_CACHE:
        return _SILERO_VAD_CACHE["model"], _SILERO_VAD_CACHE["get_speech_timestamps"]

    import torch as _torch

    model, utils = _torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False, trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    _SILERO_VAD_CACHE["model"] = model
    _SILERO_VAD_CACHE["get_speech_timestamps"] = get_speech_timestamps
    return model, get_speech_timestamps


def detect_speech_segments_silero(
    vocals_audio: dict,
    min_segment_seconds: float = 0.3,
    min_gap_seconds: float = 0.2,
    silero_threshold: float = 0.5,
    speech_pad_ms: int = 30,
    speaker_aware_merge: bool = False,
    merge_similarity_threshold: float = 0.3,
) -> list[dict]:
    """Neural VAD (Silero, auto-downloaded via torch.hub on first use, cached
    after) in place of an energy threshold -- correctly distinguishes quiet-
    but-real speech from actual silence, which is exactly where
    detect_speech_segments (energy-threshold) tends to miss/merge segments.

    silero_threshold: speech-probability cutoff Silero uses per audio chunk
    before calling it "speech" at all (its own default is 0.5). Lower = more
    sensitive, catches quieter/less confident speech that would otherwise be
    dropped entirely -- this is the direct fix for "missing some talking",
    separate from min_segment_seconds/min_gap_seconds below, which only
    filter/merge segments Silero already decided WERE speech.
    speech_pad_ms: padding added to both ends of each detected speech chunk,
    so word onsets/tails right at a boundary don't get clipped."""
    import torchaudio

    waveform = vocals_audio["waveform"].squeeze(0)  # [C, S]
    mono = waveform.mean(dim=0)  # [S], stays a torch tensor -- Silero wants one
    sr = vocals_audio["sample_rate"]

    total_dur = mono.shape[-1] / sr if sr else 0.0
    if mono.shape[-1] == 0 or total_dur <= 0:
        return []

    silero_sr = 16000
    if sr != silero_sr:
        resample = torchaudio.transforms.Resample(sr, silero_sr)
        mono_16k = resample(mono)
    else:
        mono_16k = mono

    model, get_speech_timestamps = _load_silero_vad()
    timestamps = get_speech_timestamps(
        mono_16k, model, sampling_rate=silero_sr, return_seconds=True,
        threshold=silero_threshold, speech_pad_ms=speech_pad_ms,
    )
    intervals_sec = [(float(t["start"]), float(t["end"])) for t in timestamps]
    return _intervals_to_segments(
        intervals_sec, total_dur, min_segment_seconds, min_gap_seconds,
        vocals_audio=vocals_audio, speaker_aware_merge=speaker_aware_merge,
        merge_similarity_threshold=merge_similarity_threshold,
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def crop_audio(audio: dict, start: float, end: float) -> dict:
    """Sample-accurate crop (unlike AudioCrop's whole-second-only 'MM:SS' parser)."""
    waveform = audio["waveform"].squeeze(0)  # [C, S]
    sr = audio["sample_rate"]
    s_sample = max(0, int(start * sr))
    e_sample = min(waveform.shape[-1], int(end * sr))
    if e_sample <= s_sample:
        e_sample = min(waveform.shape[-1], s_sample + 1)
    return {"waveform": waveform[:, s_sample:e_sample].unsqueeze(0), "sample_rate": sr}


# ---------------------------------------------------------------------------
# Speaker identification (auto-assign segments to a known reference voice)
# ---------------------------------------------------------------------------

_SPEAKER_ENCODER = None
_SPEECHBRAIN_LAZY_IMPORT_PATCHED = False


def _patch_speechbrain_windows_recursion_bug():
    """speechbrain's lazy module loader (speechbrain/utils/importutils.py) guards
    against infinite recursion by checking whether the immediate caller of
    ensure_module() is `inspect.py` itself. That guard is too narrow: once
    speechbrain is imported, any later code that walks the call stack via
    `inspect` (e.g. hydra's `initialize()`, which Fish Audio S2's decoder loader
    calls) can bounce between `inspect.getmodule()`'s sys.modules scan and
    `LazyModule.__getattr__` through OTHER monkeypatched layers in between
    (observed: torch's own `inspect.getfile` patch in
    torch/package/package_importer.py) that the original guard never anticipated,
    so the immediate-caller check never fires and recursion runs until
    RecursionError. Fix: a re-entrancy guard — if we're already resolving a lazy
    module anywhere on the current call stack, any nested resolution attempt
    bails out immediately (as a normal "attribute not found") instead of
    recursing further, regardless of which frames sit in between. Separately,
    any lazy submodule that genuinely fails to import (e.g. an optional
    integration like speechbrain.integrations.k2_fsa when the optional `k2`
    dependency isn't installed) is reported as a plain AttributeError rather
    than letting ImportError escape — the same reasoning applies: this function
    can be invoked incidentally by unrelated stack-walking code that has no
    interest in that submodule and shouldn't be crashed by an irrelevant,
    optional-dependency failure. Modules this package actually uses are always
    imported directly (`from speechbrain.x import Y`), never through this lazy
    path, so nothing we rely on is silently masked by this."""
    global _SPEECHBRAIN_LAZY_IMPORT_PATCHED
    if _SPEECHBRAIN_LAZY_IMPORT_PATCHED:
        return
    import importlib
    from speechbrain.utils import importutils as sb_importutils

    _reentrancy_guard = {"active": False}

    def _patched_ensure_module(self, stacklevel):
        if _reentrancy_guard["active"]:
            raise AttributeError()
        _reentrancy_guard["active"] = True
        try:
            if self.lazy_module is None:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(f".{self.target}", self.package)
            return self.lazy_module
        except Exception as e:
            raise AttributeError(f"Lazy import of {self!r} failed") from e
        finally:
            _reentrancy_guard["active"] = False

    sb_importutils.LazyModule.ensure_module = _patched_ensure_module
    _SPEECHBRAIN_LAZY_IMPORT_PATCHED = True


def _get_speaker_encoder():
    """speechbrain's ECAPA-TDNN speaker-verification embedding (spkrec-ecapa-voxceleb).
    Much better speaker discrimination than resemblyzer in practice (verified against
    real reference clips: ~0.96 same-speaker similarity vs ~0.1-0.15 cross-speaker,
    vs resemblyzer's much muddier separation on the same clips)."""
    global _SPEAKER_ENCODER
    if _SPEAKER_ENCODER is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        _patch_speechbrain_windows_recursion_bug()

        _SPEAKER_ENCODER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="speechbrain_models/spkrec-ecapa-voxceleb",
            # Windows portable Python usually lacks the privilege to create
            # symlinks (huggingface_hub's default caching strategy) — copy
            # the model files into savedir instead.
            local_strategy=LocalStrategy.COPY,
        )
    return _SPEAKER_ENCODER


def compute_speaker_embedding(audio: dict) -> np.ndarray:
    """ECAPA-TDNN speaker embedding for an AUDIO dict. Raises if the clip is
    empty — callers should catch and treat that as 'couldn't determine speaker'."""
    import librosa

    waveform = audio["waveform"].squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
    sr = audio["sample_rate"]
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    if waveform.shape[-1] == 0:
        raise ValueError("empty clip (too short)")

    encoder = _get_speaker_encoder()
    wav_tensor = torch.from_numpy(waveform).unsqueeze(0)
    emb = encoder.encode_batch(wav_tensor)
    return emb.squeeze().detach().cpu().numpy()


def assign_speakers_by_similarity(
    segments: list[dict],
    vocals_audio: dict,
    reference_embeddings: dict[int, np.ndarray],
    min_confidence_margin: float = 0.1,
) -> list[dict]:
    """
    Mutates each non-locked segment's "speaker" field to whichever reference
    embedding it's most cosine-similar to — but only if the top match clearly
    beats the runner-up by min_confidence_margin. Ambiguous segments (close
    call between two reference voices) are left on their current speaker
    rather than forcing a guess. Still a best-guess classifier overall —
    intended to be reviewed/corrected in the timeline, same trust level as
    Whisper's transcript or Gemma's emotion tags, not a guaranteed-correct
    auto-decision.
    """
    if len(reference_embeddings) < 2:
        return segments

    speaker_nums = list(reference_embeddings.keys())
    ref_matrix = np.stack([reference_embeddings[s] for s in speaker_nums])  # [N, D]
    ref_norms = np.linalg.norm(ref_matrix, axis=1)

    # Tried padding the embedding-only crop window here (to give ECAPA-TDNN
    # more signal on the many sub-1s segments where its raw similarity scores
    # measure as pure noise, ~0.13 max either direction). Measured directly
    # against real segments: mixed results, most margins got WORSE, and two
    # segments' classifications outright flipped speaker with no ground truth
    # to say the flip was correct. Reverted -- not a verified improvement, and
    # the padded audio pulling in adjacent breath/silence/background adds as
    # much noise as signal. Left as a plain per-segment crop; the real fix for
    # this scenario is manual review in the timeline, not a smarter crop.
    skipped_low_confidence = 0
    for seg in segments:
        if seg.get("locked"):
            continue
        clip = crop_audio(vocals_audio, seg["start"], seg["end"])
        try:
            emb = compute_speaker_embedding(clip)
        except Exception as e:
            print(f"[MuseVoiceSwap] Speaker ID skipped for segment {seg.get('id')}: {e}")
            continue
        sims = ref_matrix @ emb / (ref_norms * np.linalg.norm(emb) + 1e-8)
        order = np.argsort(sims)[::-1]
        top_sim, second_sim = sims[order[0]], sims[order[1]]
        if (top_sim - second_sim) < min_confidence_margin:
            skipped_low_confidence += 1
            continue
        seg["speaker"] = speaker_nums[int(order[0])]

    if skipped_low_confidence:
        print(
            f"[MuseVoiceSwap] {skipped_low_confidence} segment(s) left unchanged — "
            f"speaker match too close to call (margin < {min_confidence_margin})."
        )

    return segments


_WHISPER_CACHE: dict[str, object] = {}


def _load_whisper_cached(model_size: str):
    if model_size in _WHISPER_CACHE:
        return _WHISPER_CACHE[model_size]

    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    _WHISPER_CACHE.clear()  # keep only the most recently used size loaded
    _WHISPER_CACHE[model_size] = model
    return model


def transcribe_segment(
    vocals_audio: dict,
    start: float,
    end: float,
    model_size: str = "small",
    language: str = "",
) -> str:
    import librosa

    model = _load_whisper_cached(model_size)

    waveform = vocals_audio["waveform"].squeeze(0)  # [C, S]
    sr = vocals_audio["sample_rate"]
    s_sample, e_sample = int(start * sr), int(end * sr)
    s_sample = max(0, s_sample)
    e_sample = min(waveform.shape[-1], e_sample)
    if e_sample <= s_sample:
        return ""

    clip = waveform[:, s_sample:e_sample].mean(dim=0).cpu().numpy().astype(np.float32)

    if sr != 16000:
        clip = librosa.resample(clip, orig_sr=sr, target_sr=16000)

    result = model.transcribe(
        clip,
        language=(language or None),
        fp16=torch.cuda.is_available(),
    )
    return result.get("text", "").strip()


def transcribe_whole_with_words(
    vocals_audio: dict,
    model_size: str = "small",
    language: str = "",
) -> list[dict]:
    """"Transcribe first" half of transcribe-first-then-align: runs Whisper ONCE
    over the ENTIRE vocals track (full sentence context) instead of on each
    tiny VAD-cropped fragment in isolation. Whisper is dramatically less
    reliable given a lone half-second fragment with no surrounding context --
    that isolation is what was producing garbled nonsense like "search." /
    "magnetic surge." for segments that should have been complete sentences.
    Returns word-level timestamps: [{"word", "start", "end"}, ...] in
    chronological order, to be mapped onto segment boundaries afterward via
    assign_words_to_segments()."""
    model = _load_whisper_cached(model_size)

    waveform = vocals_audio["waveform"].squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
    sr = vocals_audio["sample_rate"]
    if waveform.shape[-1] == 0:
        return []
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

    result = model.transcribe(
        waveform,
        language=(language or None),
        fp16=torch.cuda.is_available(),
        word_timestamps=True,
    )

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({"word": text, "start": float(w["start"]), "end": float(w["end"])})
    return words


def assign_words_to_segments(segments: list[dict], words: list[dict]) -> None:
    """"Align" half of transcribe-first-then-align: fills in each segment's
    "text" from whichever already-transcribed words (from
    transcribe_whole_with_words) fall inside its [start, end) boundary, keyed
    by each word's midpoint. Mutates segments in place. A segment with no
    overlapping words is left with empty text (same as before: nothing to
    clone there, or gets picked up on a later manual edit)."""
    for seg in segments:
        s, e = seg["start"], seg["end"]
        matched = [w["word"] for w in words if s <= (w["start"] + w["end"]) / 2.0 < e]
        seg["text"] = " ".join(matched).strip()


# ---------------------------------------------------------------------------
# LLM emotion-tag annotation (local Gemma model via ComfyUI's own CLIPLoader +
# TextGenerate nodes — no API key, no quota, fully local/free)
# ---------------------------------------------------------------------------

_EMOTION_TAG_PROMPT = """You are annotating a spoken-line transcript for a text-to-speech voice cloning \
system (Fish Audio S2) that supports inline emotion/prosody tags in square brackets, e.g. [excited], \
[whisper], [laughing], [sad], [angry], [sighs], [pause], [volume up], [volume down], [pitch up], \
[pitch down], [nervous], [sarcastic], [curious], [confident].

Listen to the attached audio clip carefully — the actual tone, pacing, emphasis, pauses, and emotion \
of the delivery. Then reproduce the exact transcript below verbatim, inserting 1-4 bracketed tags at \
the points in the sentence where they match what you actually hear (not just what the words say).

Rules:
- Do NOT change, add, or remove any words from the transcript.
- Only insert bracketed tags, placed naturally before the phrase they apply to.
- Do not over-tag — a short line rarely needs more than 1-2 tags.
- If the delivery is genuinely flat/neutral throughout, return the transcript unchanged with no tags.
- Return ONLY the annotated line, no explanation, no quotes, no markdown.

Transcript: "{text}"
"""

_GEMMA_CLIP_CACHE: dict[str, object] = {}


def _load_gemma_clip_cached(clip_name: str):
    if clip_name in _GEMMA_CLIP_CACHE:
        return _GEMMA_CLIP_CACHE[clip_name]

    import nodes

    (clip,) = nodes.CLIPLoader().load_clip(clip_name, type="stable_diffusion", device="default")
    _GEMMA_CLIP_CACHE.clear()  # keep only the most recently used checkpoint loaded
    _GEMMA_CLIP_CACHE[clip_name] = clip
    return clip


def _looks_degenerate(generated_text: str, original_transcript: str) -> bool:
    """Guards against LLM repetition-loop failures — observed in practice with
    longer reference-audio inputs, where the model regurgitates a fragment of
    its own instruction prompt dozens of times instead of annotating the line.
    Two checks: the result is implausibly long relative to the source line, or
    it contains the same 6-word phrase (normalized, so minor token variance
    like "emphasis"/"emphis" between repeats still matches) 3+ times."""
    import re

    if len(generated_text) > max(120, len(original_transcript) * 4):
        return True

    words = re.findall(r"[a-z0-9']+", generated_text.lower())
    shingle_len = 6
    if len(words) >= shingle_len * 3:
        seen: dict[str, int] = {}
        for i in range(len(words) - shingle_len + 1):
            shingle = " ".join(words[i : i + shingle_len])
            seen[shingle] = seen.get(shingle, 0) + 1
            if seen[shingle] >= 3:
                return True
    return False


def annotate_emotion_tags(
    audio: dict,
    transcript_text: str,
    clip_name: str,
    max_length: int = 512,
    temperature: float = 0.7,
    seed: int = 0,
) -> str:
    """Feeds the actual segment audio to a local Gemma model (loaded via
    ComfyUI's own CLIPLoader, run via its own TextGenerate node) and asks it
    to reproduce the transcript with Fish S2-style inline emotion tags
    inserted based on what it hears. Fully local — no API key, no quota.
    Falls back to the plain transcript on any failure, and also on a
    degenerate/repetitive generation (see `_looks_degenerate`), so a bad
    generation never gets forwarded to Fish S2 to actually speak."""
    if not clip_name or not transcript_text.strip():
        return transcript_text

    try:
        from comfy_extras.nodes_textgen import TextGenerate

        clip = _load_gemma_clip_cached(clip_name)
        prompt = _EMOTION_TAG_PROMPT.format(text=transcript_text.strip())
        # Cap generation length to roughly the size of the source line (plus
        # room for a handful of bracket tags) so a degenerate/looping
        # generation can't run on for hundreds of tokens in the first place.
        word_count = max(1, len(transcript_text.split()))
        capped_max_length = max(48, min(max_length, word_count * 4 + 40))
        sampling_mode = {
            "sampling_mode": "on",
            "temperature": temperature,
            "top_k": 64,
            "top_p": 0.95,
            "min_p": 0.05,
            "repetition_penalty": 1.15,
            "seed": seed,
            "presence_penalty": 0.0,
        }

        output = TextGenerate.execute(
            clip, prompt, capped_max_length, sampling_mode,
            audio=audio, use_default_template=True,
        )
        generated_text = output.result[0] if output.result else ""
        # Some Gemma checkpoints emit an explicit <think>...</think> reasoning
        # block before the actual answer. If it never closed, the model was
        # still mid-thought when the token cap hit - there's no real answer to
        # use. If it did close, keep only what follows it: sending the
        # reasoning itself to Fish S2 as "emotion_text" makes it literally
        # speak paragraphs of internal monologue instead of the line (slow
        # and wrong - this is what was blowing up generation time).
        if "<think>" in generated_text and "</think>" not in generated_text:
            generated_text = ""
        elif "</think>" in generated_text:
            generated_text = generated_text.rsplit("</think>", 1)[-1]
        result = (generated_text or "").strip().strip('"').strip()
        if not result:
            return transcript_text
        if _looks_degenerate(result, transcript_text):
            print(
                f"[MuseVoiceSwap] Gemma emotion-tag annotation looked degenerate/repetitive, "
                f"using plain transcript instead: {result[:80]!r}..."
            )
            return transcript_text
        return result
    except Exception as e:
        print(f"[MuseVoiceSwap] Gemma emotion-tag annotation failed, using plain transcript: {e}")
        return transcript_text


# ---------------------------------------------------------------------------
# Time-match / placement
# ---------------------------------------------------------------------------

def apply_ripple_placement(cloned_segments: list[dict]) -> list[dict]:
    """
    cloned_segments: list of {"start","end","placed_duration",...} (already
    sorted by original start time). Mutates and returns the list with a
    "placed_start" key added, preserving each segment's *original* silence gap
    to the previous segment rather than its original absolute start time.
    """
    cursor_delta = 0.0
    prev_original_end = 0.0
    for seg in cloned_segments:
        original_gap = seg["start"] - prev_original_end
        seg["placed_start"] = prev_original_end + cursor_delta + original_gap
        prev_original_end = seg["end"]
        cursor_delta += seg["placed_duration"] - (seg["end"] - seg["start"])
    return cloned_segments


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def new_silent_buffer(duration_seconds: float, sample_rate: int, channels: int = 1) -> torch.Tensor:
    num_samples = max(1, int(round(duration_seconds * sample_rate)))
    return torch.zeros(channels, num_samples)


def overlay_at(buffer: torch.Tensor, sample_rate: int, clip_audio: dict, start_seconds: float) -> torch.Tensor:
    """Add clip_audio's waveform into buffer at start_seconds, resampling if sample
    rates differ. buffer is [C, S] at `sample_rate`; clip_audio is a ComfyUI AUDIO
    dict. Returns the buffer, growing it (a new tensor) if the clip runs past its
    current end — callers should still size the buffer correctly upfront via
    new_silent_buffer(final_length_seconds, ...) to avoid repeated growth."""
    import torchaudio

    clip_wave = clip_audio["waveform"].squeeze(0)  # [C, S]
    clip_sr = clip_audio["sample_rate"]

    if clip_sr != sample_rate:
        resample = torchaudio.transforms.Resample(clip_sr, sample_rate)
        clip_wave = resample(clip_wave)

    if clip_wave.shape[0] != buffer.shape[0]:
        if clip_wave.shape[0] == 1:
            clip_wave = clip_wave.repeat(buffer.shape[0], 1)
        else:
            clip_wave = clip_wave.mean(dim=0, keepdim=True).repeat(buffer.shape[0], 1)

    start_sample = max(0, int(round(start_seconds * sample_rate)))
    end_sample = start_sample + clip_wave.shape[-1]

    if end_sample > buffer.shape[-1]:
        pad = end_sample - buffer.shape[-1]
        buffer = torch.cat([buffer, torch.zeros(buffer.shape[0], pad)], dim=-1)

    buffer[:, start_sample:end_sample] += clip_wave
    return buffer


def match_bg_length(bg_audio: dict, target_seconds: float, mode: str = "hold_last") -> dict:
    """Extend or trim bg_audio's waveform to target_seconds via loop/hold_last/trim."""
    waveform = bg_audio["waveform"].squeeze(0)  # [C, S]
    sr = bg_audio["sample_rate"]
    target_samples = max(1, int(round(target_seconds * sr)))
    current_samples = waveform.shape[-1]

    if current_samples == target_samples:
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}

    if current_samples > target_samples:
        trimmed = waveform[:, :target_samples]
        return {"waveform": trimmed.unsqueeze(0), "sample_rate": sr}

    # Need to extend.
    needed = target_samples - current_samples
    if mode == "trim":
        # "trim" mode with a too-short bg just means: don't extend, pad silence.
        pad = torch.zeros(waveform.shape[0], needed)
        extended = torch.cat([waveform, pad], dim=-1)
        return {"waveform": extended.unsqueeze(0), "sample_rate": sr}

    if mode == "loop":
        reps = int(np.ceil(needed / max(1, current_samples)))
        tail = waveform.repeat(1, reps + 1)[:, :needed]
        extended = torch.cat([waveform, tail], dim=-1)
        return {"waveform": extended.unsqueeze(0), "sample_rate": sr}

    # "hold_last" (default): repeat a short tail window with a fade to avoid a click.
    fade_samples = min(current_samples, int(0.5 * sr))
    if fade_samples <= 0:
        pad = torch.zeros(waveform.shape[0], needed)
        extended = torch.cat([waveform, pad], dim=-1)
        return {"waveform": extended.unsqueeze(0), "sample_rate": sr}

    tail_window = waveform[:, -fade_samples:]
    reps = int(np.ceil(needed / fade_samples))
    held = tail_window.repeat(1, reps)[:, :needed]

    fade_len = min(fade_samples, needed)
    fade_curve = torch.linspace(1.0, 0.0, fade_len)
    held[:, :fade_len] *= fade_curve

    extended = torch.cat([waveform, held], dim=-1)
    return {"waveform": extended.unsqueeze(0), "sample_rate": sr}


# ---------------------------------------------------------------------------
# Waveform peaks (for the JS timeline's "ui" payload)
# ---------------------------------------------------------------------------

def compute_peaks(audio: dict, num_peaks: int = 400) -> list[float]:
    waveform = audio["waveform"].squeeze(0).mean(dim=0)  # mono [S]
    total = waveform.shape[-1]
    if total == 0:
        return []
    step = max(1, total // num_peaks)
    usable = (total // step) * step
    if usable == 0:
        return [float(waveform.abs().max().item())]
    trimmed = waveform[:usable]
    peaks = trimmed.abs().unfold(0, step, step).amax(dim=1)
    return [round(v, 5) for v in peaks.cpu().tolist()]


def audio_duration_seconds(audio: dict) -> float:
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    if sr <= 0:
        return 0.0
    return waveform.shape[-1] / sr


def cap_runaway_clone_duration(
    clone_audio: dict, orig_duration: float,
    max_multiplier: float = 6.0, hard_cap_seconds: float = 30.0,
) -> tuple[dict, float]:
    """Guards against a TTS runaway-generation failure mode (Fish S2 occasionally
    keeps generating far more audio than a short line calls for). Left uncapped,
    one such segment poisons apply_ripple_placement's cumulative cursor_delta,
    silently shifting every later segment's placement by that same absurd amount
    -- the actual audio (not just the reported duration) is truncated so the two
    always agree."""
    duration = audio_duration_seconds(clone_audio)
    allowed = max(orig_duration * max_multiplier, hard_cap_seconds)
    if duration <= allowed:
        return clone_audio, duration

    sr = clone_audio["sample_rate"]
    max_samples = max(1, int(round(allowed * sr)))
    waveform = clone_audio["waveform"][..., :max_samples]
    print(
        f"[MuseVoiceSwap] Clone audio for a {orig_duration:.2f}s line came back "
        f"{duration:.2f}s long -- truncating to {allowed:.2f}s to avoid corrupting "
        f"segment placement (likely a TTS runaway-generation glitch)."
    )
    return {"waveform": waveform, "sample_rate": sr}, allowed


def encode_audio_preview_b64(audio: dict, target_sample_rate: int = 22050) -> str:
    """Mono, downsampled, base64-encoded WAV for the timeline widget's in-browser
    playhead/scrub playback — a review aid only, never fed back into the clone/mix
    math, so quality is traded for a small payload in the node's `ui` message.

    Encodes via Python's stdlib `wave` module rather than torchaudio.save(), which
    on newer torchaudio defaults to a torchcodec-backed encoder -- torchcodec ships
    prebuilt libtorchcodec DLLs pinned to specific FFmpeg/PyTorch versions, and a
    mismatch there (observed on this install: PyTorch 2.9.1+cu130) hard-crashes the
    whole node well after the actual clone/mix work has already succeeded. The
    resample step below is unrelated to that encoder and stays on torchaudio."""
    import base64
    import io
    import wave

    waveform = audio["waveform"].squeeze(0)  # [C, S]
    sr = audio["sample_rate"]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        import torchaudio
        resample = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resample(waveform)

    samples = waveform.squeeze(0).clamp(-1.0, 1.0)
    pcm16 = (samples * 32767.0).to(torch.int16).cpu().numpy()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sample_rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")
