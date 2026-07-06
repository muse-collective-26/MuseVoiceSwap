"""Muse Voice Swap V1 — swaps the speaking voice in an already-generated AV
clip's audio track while preserving background/ambient sound, exposing a
Muse-Director-style timeline for reviewing/editing detected segments."""

import json

import folder_paths
from comfy_extras.nodes_audio import load as _load_audio_file

from . import voice_swap_pipeline as vsp
from .voice_swap_bridge import (
    get_audio_combine_class,
    get_audio_separation_class,
    get_fish_model_names_fn,
    get_fish_voice_clone_class,
    get_time_shift_fn,
)

LANGUAGES = [
    "auto", "en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de",
    "it", "tr", "nl", "sv", "no", "da", "fi", "pl", "hi", "vi", "th",
    "id", "ms", "uk", "bg", "hr", "cs", "sk", "sl", "ro", "hu", "et",
    "lv", "lt", "el", "he", "fa", "bn", "ta", "te", "kn", "ml", "si",
]

WHISPER_LANGUAGES = ["auto", "en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de", "it"]


def _get_model_names_safe():
    try:
        return get_fish_model_names_fn()()
    except Exception as e:
        print(f"[MuseVoiceSwap] Could not resolve Fish S2 model list yet: {e}")
        return ["(FishAudioS2 not loaded)"]


def _get_text_encoder_names_safe():
    try:
        names = folder_paths.get_filename_list("text_encoders")
        return names if names else ["(no text_encoders found)"]
    except Exception as e:
        print(f"[MuseVoiceSwap] Could not list text_encoders: {e}")
        return ["(no text_encoders found)"]


def _default_gemma_name(names):
    """Prefer the exact checkpoint used by the reference Fish S2 Pro workflow
    (gemma4_e4b_it_fp8_scaled.safetensors), since text_encoders/ commonly holds
    several other Gemma checkpoints too (e.g. Gemma3-12B variants used for
    unrelated image/video workflows) — a plain 'gemma' substring match would
    otherwise pick whichever one happens to sort first and silently load the
    wrong (much larger, unintended) model. Falls back to substring match only
    if that exact file isn't present in this install."""
    for name in names:
        if name == "gemma4_e4b_it_fp8_scaled.safetensors":
            return name
    for name in names:
        if "gemma" in name.lower():
            return name
    return names[0] if names else ""


def _parse_timeline_json(timeline_data: str) -> dict:
    try:
        if timeline_data and timeline_data.strip() not in ("", "{}"):
            data = json.loads(timeline_data)
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"[MuseVoiceSwap] Failed to parse timeline_data JSON, resetting: {e}")
    return {}


def _load_reference_override(filename: str):
    """Resolve a per-segment reference_audio_override filename (from the
    ComfyUI input directory, same convention as the core LoadAudio node) into
    an AUDIO dict, or None if not set/not found."""
    if not filename or not filename.strip():
        return None
    try:
        audio_path = folder_paths.get_annotated_filepath(filename.strip())
        waveform, sample_rate = _load_audio_file(audio_path)
        return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
    except Exception as e:
        print(f"[MuseVoiceSwap] Could not load reference_audio_override '{filename}': {e}")
        return None


def _sum_stems(*stems):
    waveforms = [s["waveform"] for s in stems]
    sample_rate = stems[0]["sample_rate"]
    min_len = min(w.shape[-1] for w in waveforms)
    total = sum(w[..., :min_len] for w in waveforms)
    return {"waveform": total, "sample_rate": sample_rate}


def _segment_to_json(seg: dict) -> dict:
    return {
        "id": seg["id"],
        "start": seg["start"],
        "end": seg["end"],
        "text": seg.get("text", ""),
        "emotion_text": seg.get("emotion_text", ""),
        "speaker": seg.get("speaker", 1),
        "reference_audio_override": seg.get("reference_audio_override", ""),
        "reference_text_override": seg.get("reference_text_override", ""),
        "locked": seg.get("locked", False),
        "placed_start": seg.get("placed_start", seg["start"]),
        "placed_duration": seg.get("placed_duration", seg["end"] - seg["start"]),
    }


class MuseVoiceSwapV1:
    @classmethod
    def INPUT_TYPES(cls):
        model_names = _get_model_names_safe()
        gemma_names = _get_text_encoder_names_safe()
        return {
            "required": {
                "source_audio": ("AUDIO", {"tooltip": "The LTX-generated mixed audio (speech + ambient)."}),
                "reference_audio": ("AUDIO", {"tooltip": "Voice-clone target reference, 5-30s."}),
                "reference_text": ("STRING", {"multiline": True, "default": "", "tooltip": (
                    "Transcript of reference_audio. Leave blank to auto-transcribe it "
                    "internally via Whisper — no need to wire in a separate transcript node."
                )}),

                "timeline_data": ("STRING", {"default": "{}"}),
                "use_timeline_override": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ON = keep existing segment boundaries/text from timeline_data; "
                        "skip auto-detect and auto-transcribe, only re-clone + remix."
                    ),
                }),
                "multi_speaker_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ON = assign each segment to speaker 1/2/3 (set per-segment in the "
                        "timeline panel) and clone it using that speaker's reference voice "
                        "(reference_audio / reference_audio_2 / reference_audio_3). Also "
                        "produces isolated audio tracks for speakers 2 and 3."
                    ),
                }),
                "auto_assign_speakers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ON (with multi_speaker_mode) = auto-guess each segment's speaker by "
                        "comparing its voice against reference_audio/_2/_3 using a speaker-"
                        "embedding similarity match. This is a best-guess classifier, not a "
                        "guarantee — review/correct assignments in the timeline afterward, "
                        "same as you would Whisper's transcript. Only runs on a fresh (non-"
                        "override) pass — once use_timeline_override is on, your manual "
                        "corrections are trusted as-is and this is skipped entirely, so you "
                        "don't need to lock a segment just to protect a manual speaker fix."
                    ),
                }),

                "sep_chunk_length": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 60.0, "step": 0.5}),
                "sep_chunk_overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sep_chunk_fade_shape": (["linear", "half_sine", "logarithmic", "exponential"], {"default": "linear"}),

                "vad_top_db": ("FLOAT", {"default": 35.0, "min": 5.0, "max": 60.0, "step": 1.0}),
                "min_segment_seconds": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 5.0, "step": 0.05}),
                "min_gap_seconds": ("FLOAT", {"default": 0.2, "min": 0.02, "max": 5.0, "step": 0.02}),

                "whisper_model_size": (["tiny", "base", "small", "medium", "large-v3"], {"default": "small"}),
                "whisper_language": (WHISPER_LANGUAGES, {"default": "auto"}),

                "use_llm_emotion_tags": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "ON = feed each segment's actual audio to a local Gemma model and ask it "
                        "to insert Fish S2-style emotion/prosody tags ([excited], [whisper], "
                        "[pause], etc.) based on the real delivery, before cloning. Fixes flat/ "
                        "monotone clones. Fully local — no API key, no quota/cost."
                    ),
                }),
                "gemma_clip_name": (gemma_names, {
                    "default": _default_gemma_name(gemma_names),
                    "tooltip": (
                        "Gemma checkpoint from ComfyUI/models/text_encoders/, loaded via the same "
                        "CLIPLoader + TextGenerate nodes ComfyUI uses natively. e.g. "
                        "gemma4_e4b_it_fp8_scaled.safetensors"
                    ),
                }),
                "emotion_tag_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),

                "fish_model_path": (model_names,),
                "fish_language": (LANGUAGES, {"default": "auto"}),
                "fish_device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "fish_precision": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "fish_attention": (["auto", "sdpa", "sage_attention", "flash_attention"], {"default": "auto"}),
                "fish_max_new_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                "fish_chunk_length": ("INT", {"default": 200, "min": 100, "max": 400, "step": 10}),
                "fish_temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "fish_top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "fish_repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.9, "max": 2.0, "step": 0.05}),
                "fish_seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "fish_keep_model_loaded": ("BOOLEAN", {"default": True}),

                "time_match_mode": (["ripple", "stretch_to_fit", "off"], {"default": "ripple"}),
                "stretch_fft_size": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512}),

                "bg_extend_mode": (["hold_last", "loop", "trim"], {"default": "hold_last"}),
                "bg_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),

                "timeline_ui": ("STRING", {"default": ""}),
            },
            "optional": {
                "reference_audio_2": ("AUDIO", {"tooltip": "Reference voice for segments assigned speaker 2 (multi_speaker_mode)."}),
                "reference_text_2": ("STRING", {"multiline": True, "default": ""}),
                "reference_audio_3": ("AUDIO", {"tooltip": "Reference voice for segments assigned speaker 3 (multi_speaker_mode)."}),
                "reference_text_3": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = (
        "final_audio", "new_vocals_only", "bg_audio_only",
        "speaker_2_vocals_only", "speaker_3_vocals_only", "timeline_data",
    )
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"
    DESCRIPTION = (
        "Separates vocals from ambient in an LTX-generated audio track, "
        "re-voices the detected speech segment(s) via Fish S2 cloning, and "
        "remixes with the untouched ambient bed."
    )

    def execute(
        self,
        source_audio,
        reference_audio,
        reference_text,
        timeline_data,
        use_timeline_override,
        multi_speaker_mode,
        auto_assign_speakers,
        sep_chunk_length,
        sep_chunk_overlap,
        sep_chunk_fade_shape,
        vad_top_db,
        min_segment_seconds,
        min_gap_seconds,
        whisper_model_size,
        whisper_language,
        use_llm_emotion_tags,
        gemma_clip_name,
        emotion_tag_temperature,
        fish_model_path,
        fish_language,
        fish_device,
        fish_precision,
        fish_attention,
        fish_max_new_tokens,
        fish_chunk_length,
        fish_temperature,
        fish_top_p,
        fish_repetition_penalty,
        fish_seed,
        fish_keep_model_loaded,
        time_match_mode,
        stretch_fft_size,
        bg_extend_mode,
        bg_volume,
        timeline_ui="",
        reference_audio_2=None,
        reference_text_2="",
        reference_audio_3=None,
        reference_text_3="",
    ):
        tdata = _parse_timeline_json(timeline_data)

        # 1. SEPARATE ----------------------------------------------------
        AudioSeparation = get_audio_separation_class()
        bass, drums, other, vocals = AudioSeparation().main(
            source_audio,
            chunk_fade_shape=sep_chunk_fade_shape,
            chunk_length=sep_chunk_length,
            chunk_overlap=sep_chunk_overlap,
        )
        bg_stem = _sum_stems(bass, drums, other)

        # 2. DETECT SEGMENTS (skippable) -----------------------------------
        if use_timeline_override and tdata.get("vocalSegments"):
            segments = [dict(s) for s in tdata["vocalSegments"]]
        else:
            segments = vsp.detect_speech_segments(
                vocals, top_db=vad_top_db,
                min_segment_seconds=min_segment_seconds,
                min_gap_seconds=min_gap_seconds,
            )

        # 2b. AUTO-ASSIGN SPEAKERS (skipped entirely under override — same "trust
        # my edits" contract as segment detection above, otherwise every re-run
        # would silently re-guess and overwrite manual speaker corrections made
        # in the timeline unless every corrected segment was also locked) -----
        if (not use_timeline_override) and multi_speaker_mode and auto_assign_speakers:
            reference_embeddings = {}
            for spk_num, ref_audio_candidate in (
                (1, reference_audio), (2, reference_audio_2), (3, reference_audio_3),
            ):
                if ref_audio_candidate is None:
                    continue
                try:
                    reference_embeddings[spk_num] = vsp.compute_speaker_embedding(ref_audio_candidate)
                except Exception as e:
                    print(f"[MuseVoiceSwap] Could not embed reference voice for speaker {spk_num}: {e}")
            if len(reference_embeddings) >= 2:
                segments = vsp.assign_speakers_by_similarity(segments, vocals, reference_embeddings)
            else:
                print("[MuseVoiceSwap] auto_assign_speakers needs at least 2 reference voices — skipping.")

        # 3. TRANSCRIBE (skippable per-segment) -----------------------------
        for seg in segments:
            # A locked segment keeps its text even on a fresh auto-detect pass;
            # otherwise skipping re-transcription only applies under override mode.
            already_has_text = seg.get("text", "").strip()
            if already_has_text and (seg.get("locked") or use_timeline_override):
                continue
            seg["text"] = vsp.transcribe_segment(
                vocals, seg["start"], seg["end"],
                model_size=whisper_model_size,
                language=("" if whisper_language == "auto" else whisper_language),
            )

        # 3b. ANNOTATE emotion tags (optional, skippable per-segment) ----------
        if use_llm_emotion_tags:
            for seg in segments:
                if not seg.get("text", "").strip():
                    continue
                already_annotated = seg.get("emotion_text", "").strip()
                if already_annotated and (seg.get("locked") or use_timeline_override):
                    continue
                segment_audio = vsp.crop_audio(vocals, seg["start"], seg["end"])
                seg["emotion_text"] = vsp.annotate_emotion_tags(
                    segment_audio, seg["text"],
                    clip_name=gemma_clip_name,
                    temperature=emotion_tag_temperature,
                )

        # 4. CLONE each segment ---------------------------------------------
        # If reference_text is left blank, auto-transcribe reference_audio via
        # the same Whisper pass used for segments, so no external transcript
        # node needs to be wired in (unlike the plain FishS2VoiceCloneTTS node).
        # Cached by id() so the shared global reference_audio is only ever
        # transcribed once, even across many segments.
        _ref_text_cache: dict[int, str] = {}

        def _resolve_reference_text(ref_audio_dict, ref_text_override):
            if ref_text_override and ref_text_override.strip():
                return ref_text_override.strip()
            cache_key = id(ref_audio_dict)
            if cache_key not in _ref_text_cache:
                dur = vsp.audio_duration_seconds(ref_audio_dict)
                _ref_text_cache[cache_key] = vsp.transcribe_segment(
                    ref_audio_dict, 0.0, dur,
                    model_size=whisper_model_size,
                    language=("" if whisper_language == "auto" else whisper_language),
                )
            return _ref_text_cache[cache_key]

        FishS2VoiceCloneTTS = get_fish_voice_clone_class()
        fish_node = FishS2VoiceCloneTTS()
        cloned_segments = []
        for seg in segments:
            if not seg.get("text", "").strip():
                continue
            override_audio = _load_reference_override(seg.get("reference_audio_override", ""))
            if override_audio is not None:
                ref_audio, base_ref_text = override_audio, reference_text
            else:
                speaker = seg.get("speaker", 1) or 1
                if multi_speaker_mode and speaker == 2 and reference_audio_2 is not None:
                    ref_audio, base_ref_text = reference_audio_2, reference_text_2
                elif multi_speaker_mode and speaker == 3 and reference_audio_3 is not None:
                    ref_audio, base_ref_text = reference_audio_3, reference_text_3
                else:
                    ref_audio, base_ref_text = reference_audio, reference_text
            ref_text = _resolve_reference_text(ref_audio, seg.get("reference_text_override") or base_ref_text)
            clone_text = seg["text"]
            if use_llm_emotion_tags and seg.get("emotion_text", "").strip():
                clone_text = seg["emotion_text"]
            (clone_audio,) = fish_node.generate(
                model_path=fish_model_path,
                text=clone_text,
                reference_audio=ref_audio,
                language=fish_language,
                device=fish_device,
                precision=fish_precision,
                attention=fish_attention,
                max_new_tokens=fish_max_new_tokens,
                chunk_length=fish_chunk_length,
                temperature=fish_temperature,
                top_p=fish_top_p,
                repetition_penalty=fish_repetition_penalty,
                seed=fish_seed,
                keep_model_loaded=fish_keep_model_loaded,
                offload_to_cpu=False,
                compile_model=False,
                reference_text=ref_text,
            )
            orig_duration = seg["end"] - seg["start"]
            clone_duration = vsp.audio_duration_seconds(clone_audio)
            cloned_segments.append({
                **seg,
                "clone_audio": clone_audio,
                "orig_duration": orig_duration,
                "placed_duration": clone_duration,
            })

        # 5. TIME-MATCH each clone -------------------------------------------
        if time_match_mode == "stretch_to_fit":
            time_shift = get_time_shift_fn()
            for cseg in cloned_segments:
                clone_duration = cseg["placed_duration"]
                orig_duration = cseg["orig_duration"] or 0.001
                rate = clone_duration / orig_duration if orig_duration > 0 else 1.0
                rate = max(0.1, min(10.0, rate))
                waveform = cseg["clone_audio"]["waveform"].squeeze(0)
                stretched = time_shift(waveform, rate, fft_size=stretch_fft_size)
                cseg["clone_audio"] = {
                    "waveform": stretched.unsqueeze(0),
                    "sample_rate": cseg["clone_audio"]["sample_rate"],
                }
                cseg["placed_start"] = cseg["start"]
                cseg["placed_duration"] = cseg["orig_duration"]
        elif time_match_mode == "off":
            for cseg in cloned_segments:
                cseg["placed_start"] = cseg["start"]
        else:  # ripple (default)
            cloned_segments.sort(key=lambda s: s["start"])
            cloned_segments = vsp.apply_ripple_placement(cloned_segments)

        # 6. ASSEMBLE new vocal-only track ------------------------------------
        source_duration = vsp.audio_duration_seconds(source_audio)
        if cloned_segments:
            final_length_seconds = max(
                cseg["placed_start"] + cseg["placed_duration"] for cseg in cloned_segments
            )
        else:
            final_length_seconds = source_duration
        final_length_seconds = max(final_length_seconds, source_duration)

        out_sample_rate = vocals["sample_rate"]
        buffer = vsp.new_silent_buffer(
            final_length_seconds, out_sample_rate, channels=vocals["waveform"].shape[1]
        )
        for cseg in cloned_segments:
            buffer = vsp.overlay_at(buffer, out_sample_rate, cseg["clone_audio"], cseg["placed_start"])
        new_vocals_only = {"waveform": buffer.unsqueeze(0), "sample_rate": out_sample_rate}

        # Isolated per-speaker tracks (speaker 1 is already the bulk of
        # new_vocals_only above; these two mirror FishS2MultiSpeakerSplitTTS's
        # "silence when not speaking" per-speaker outputs, but placed using our
        # own VAD-derived timing rather than a fixed pause-per-turn).
        speaker2_buffer = vsp.new_silent_buffer(
            final_length_seconds, out_sample_rate, channels=vocals["waveform"].shape[1]
        )
        speaker3_buffer = vsp.new_silent_buffer(
            final_length_seconds, out_sample_rate, channels=vocals["waveform"].shape[1]
        )
        for cseg in cloned_segments:
            spk = cseg.get("speaker", 1) or 1
            if spk == 2:
                speaker2_buffer = vsp.overlay_at(speaker2_buffer, out_sample_rate, cseg["clone_audio"], cseg["placed_start"])
            elif spk == 3:
                speaker3_buffer = vsp.overlay_at(speaker3_buffer, out_sample_rate, cseg["clone_audio"], cseg["placed_start"])
        speaker_2_vocals_only = {"waveform": speaker2_buffer.unsqueeze(0), "sample_rate": out_sample_rate}
        speaker_3_vocals_only = {"waveform": speaker3_buffer.unsqueeze(0), "sample_rate": out_sample_rate}

        drift = final_length_seconds - source_duration
        if abs(drift) > 0.5:
            print(f"[MuseVoiceSwap] Note: final audio duration drifted {drift:+.2f}s vs source ({source_duration:.2f}s -> {final_length_seconds:.2f}s)")

        # 7. EXTEND/TRIM background stem --------------------------------------
        bg_audio_only = vsp.match_bg_length(bg_stem, final_length_seconds, mode=bg_extend_mode)
        if bg_volume != 1.0:
            bg_audio_only = {
                "waveform": bg_audio_only["waveform"] * bg_volume,
                "sample_rate": bg_audio_only["sample_rate"],
            }

        # 8. MIX ----------------------------------------------------------------
        AudioCombine = get_audio_combine_class()
        (final_audio,) = AudioCombine().main(new_vocals_only, bg_audio_only, method="add")

        # 9. Serialize timeline_data back out -------------------------------------
        updated_segments = [_segment_to_json(cseg) for cseg in cloned_segments]
        updated_timeline_data = json.dumps({
            "schema_version": 1,
            "source_duration": round(source_duration, 3),
            "vocalSegments": updated_segments,
            "bgWaveformPeaks": vsp.compute_peaks(bg_stem),
            "last_run": {
                "time_match_mode": time_match_mode,
                "used_override": bool(use_timeline_override),
                "final_duration": round(final_length_seconds, 3),
            },
        })

        return {
            "ui": {
                "muse_voice_swap": [{
                    "vocal_peaks": vsp.compute_peaks(vocals),
                    "bg_peaks": vsp.compute_peaks(bg_stem),
                    "segments": updated_segments,
                    "source_duration": round(source_duration, 3),
                    "final_duration": round(final_length_seconds, 3),
                }],
            },
            "result": (
                final_audio, new_vocals_only, bg_audio_only,
                speaker_2_vocals_only, speaker_3_vocals_only, updated_timeline_data,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "MuseVoiceSwapV1": MuseVoiceSwapV1,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseVoiceSwapV1": "Muse Voice Swap V1",
}
