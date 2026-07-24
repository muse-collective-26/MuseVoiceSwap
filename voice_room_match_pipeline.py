"""
Pure DSP functions for MuseVoiceRoomMatch -- takes several separately-cloned
voice clips and makes them sound like they were captured together in one
physical space: a shared loudness target, ONE synthesized reverb impulse
response convolved identically into every voice, and an optional shared
low-level room-tone bed. No ComfyUI imports here -- exercised standalone via
the embedded interpreter before ever being wired into a node.

The "sounds like the same room" goal is deliberately scoped to shared
ACOUSTIC characteristics (reverb, room tone, loudness) rather than anything
about vocal identity (pitch/timbre) -- matching pitch/timbre between two
different people's voices isn't achievable without literally changing whose
voice it sounds like, which defeats the point of cloning each person's real
voice in the first place.

AUDIO dict convention (matches ComfyUI's own AUDIO type):
    {"waveform": torch.Tensor[1, C, S], "sample_rate": int}
"""

from __future__ import annotations

import numpy as np
import torch

from .voice_swap_pipeline import audio_duration_seconds

# Each preset controls three independently-audible things, not just "how much
# reverb": (1) predelay_ms + early reflections (er_count/er_spread_ms/er_decay)
# -- the discrete echo pattern BEFORE the diffuse wash, which is what actually
# reads as "size and shape of the room" (a plain length/brightness knob alone
# produces generic-sounding wash no matter how it's tuned); (2) low_decay_tc vs
# high_decay_tc -- separate decay speeds for low vs high frequency content, so
# a preset can be "warm" (highs die fast, dark rumbling tail: jazz_club,
# cathedral) or "bright/ringy" (highs sustain nearly as long as lows: rock_arena)
# instead of every preset just fading uniformly; (3) cutoff_hz/wet as before.
# "outdoor" is intentionally near-dry -- open air has no enclosing surfaces to
# reflect sound, so its character comes from the (separately generated)
# room-tone bed, not reverb.
ENVIRONMENT_PRESETS = {
    "studio_dry": dict(predelay_ms=0,  er_count=0,  er_spread_ms=20,  er_decay=0.5, ir_seconds=0.05, low_decay_tc=0.02, high_decay_tc=0.01, cutoff_hz=10000, wet=0.05),
    "room":       dict(predelay_ms=5,  er_count=4,  er_spread_ms=35,  er_decay=0.55, ir_seconds=0.5,  low_decay_tc=0.15, high_decay_tc=0.08, cutoff_hz=7000, wet=0.18),
    "jazz_club":  dict(predelay_ms=8,  er_count=7,  er_spread_ms=45,  er_decay=0.45, ir_seconds=0.7,  low_decay_tc=0.28, high_decay_tc=0.10, cutoff_hz=3500, wet=0.24),
    "hall":       dict(predelay_ms=25, er_count=6,  er_spread_ms=80,  er_decay=0.6,  ir_seconds=2.2,  low_decay_tc=0.9,  high_decay_tc=0.40, cutoff_hz=7000, wet=0.33),
    "auditorium": dict(predelay_ms=35, er_count=8,  er_spread_ms=100, er_decay=0.6,  ir_seconds=2.8,  low_decay_tc=1.3,  high_decay_tc=0.50, cutoff_hz=6200, wet=0.37),
    "rock_arena": dict(predelay_ms=15, er_count=4,  er_spread_ms=140, er_decay=0.8,  ir_seconds=2.4,  low_decay_tc=1.0,  high_decay_tc=0.9,  cutoff_hz=9000, wet=0.42),
    "cathedral":  dict(predelay_ms=60, er_count=10, er_spread_ms=160, er_decay=0.5,  ir_seconds=5.0,  low_decay_tc=2.6,  high_decay_tc=0.8,  cutoff_hz=4200, wet=0.48),
    "outdoor":    dict(predelay_ms=0,  er_count=0,  er_spread_ms=20,  er_decay=0.5,  ir_seconds=0.08, low_decay_tc=0.02, high_decay_tc=0.02, cutoff_hz=5000, wet=0.06),
}
ENVIRONMENT_NAMES = list(ENVIRONMENT_PRESETS.keys())


def _generate_early_reflections(sr: int, predelay_ms: float, er_count: int, er_spread_ms: float, er_decay: float, seed: int) -> np.ndarray:
    """Sparse discrete echo taps after `predelay_ms` of silence, spread across
    `er_spread_ms` with each successive tap `er_decay`x quieter than the last.
    This discrete pattern -- not the diffuse tail -- is what gives a preset
    its distinct "size/shape" character (e.g. rock_arena's few, widely-spaced,
    slowly-decaying taps read as a slapback echo; cathedral's many closely-
    packed taps read as a wash building into its long tail)."""
    predelay_samples = int(predelay_ms / 1000.0 * sr)
    spread_samples = max(1, int(er_spread_ms / 1000.0 * sr))
    total = predelay_samples + spread_samples + 1
    taps = np.zeros(total, dtype=np.float64)
    if er_count <= 0:
        return taps

    rng = np.random.default_rng(seed + 1000)
    times = np.sort(rng.integers(0, spread_samples, size=er_count))
    for i, t in enumerate(times):
        gain = (er_decay ** i) * (1.0 if i == 0 else 0.6)
        idx = predelay_samples + int(t)
        if idx < total:
            taps[idx] += gain * float(rng.choice([-1.0, 1.0]))
    return taps


def _schroeder_allpass_diffuse(x: np.ndarray, sr: int, seed: int) -> np.ndarray:
    """Cascaded Schroeder allpass filters -- the textbook fix for exactly the
    'metal cylinder' artifact: a handful of sparse, hard-edged reflections
    (or a single short noise realization) has strong, uneven comb-filter-like
    peaks/notches in its spectrum, which reads as metallic ringing rather
    than a natural diffuse room reflection. This structure (H(z) = (z^-M - g)
    / (1 - g*z^-M)) is mathematically flat in magnitude at every frequency --
    it only smears/decorrelates the TIME-domain arrival pattern -- so it
    breaks up that periodicity without darkening or brightening the sound."""
    import scipy.signal as sps

    rng = np.random.default_rng(seed + 5000)
    # Mutually-prime-ish delays (in ms) so the cascaded stages don't reinforce
    # each other's periodicity; small jitter per seed keeps different presets/
    # seeds from all sharing identical diffuser coloration.
    base_delays_ms = [4.7, 7.3, 11.9]
    y = x.astype(np.float64)
    for i, dm in enumerate(base_delays_ms):
        jitter = rng.uniform(0.9, 1.1)
        m = max(1, int(dm * jitter / 1000.0 * sr))
        g = 0.6
        b = np.zeros(m + 1)
        a = np.zeros(m + 1)
        b[0] = -g
        b[m] = 1.0
        a[0] = 1.0
        a[m] = -g
        y = sps.lfilter(b, a, y)
    return y


def synthesize_room_ir(sr: int, preset: dict, seed: int = 0) -> np.ndarray:
    """Synthetic reverb impulse response: predelay + discrete early
    reflections, followed by a two-band (low/high) exponentially-decaying
    diffuse tail so brightness can fade at a different rate than body/warmth,
    then run through an allpass diffuser to smooth out the comb-filter/
    metallic ringing that sparse taps + a single noise realization otherwise
    produce. The SAME seed always produces a byte-identical IR -- that's what
    actually makes convolving several different voices with this one IR
    sound like a shared physical space, rather than each voice getting its
    own independently-generated (merely similar-flavored) reverb."""
    import scipy.signal as sps

    early = _generate_early_reflections(
        sr, preset["predelay_ms"], preset["er_count"], preset["er_spread_ms"], preset["er_decay"], seed
    )

    n = max(len(early), int(preset["ir_seconds"] * sr))
    t = np.arange(n) / sr

    # Two decorrelated noise draws per band, averaged, instead of one --
    # a single short noise realization has large random spectral peaks/
    # notches (tens of dB); averaging independent draws cancels much of that
    # randomness the way averaging any independent noisy measurements does.
    def _dual_draw(base_seed: int) -> np.ndarray:
        a = np.random.default_rng(base_seed).standard_normal(n).astype(np.float64)
        b = np.random.default_rng(base_seed + 1).standard_normal(n).astype(np.float64)
        return (a + b) / np.sqrt(2.0)

    noise_low = _dual_draw(seed)
    noise_high = _dual_draw(seed + 2000)

    env_low = np.exp(-t / max(preset["low_decay_tc"], 1e-4))
    env_high = np.exp(-t / max(preset["high_decay_tc"], 1e-4))

    nyquist = sr / 2.0
    band_split_hz = 1200.0
    low_cut = min(0.99, min(preset["cutoff_hz"], band_split_hz) / nyquist)
    b_lo, a_lo = sps.butter(2, low_cut, btype="low")
    tail_low = sps.lfilter(b_lo, a_lo, noise_low * env_low)

    high_lo_norm = min(0.99, band_split_hz / nyquist)
    high_hi_norm = min(0.99, preset["cutoff_hz"] / nyquist)
    if high_hi_norm <= high_lo_norm:
        high_hi_norm = min(0.99, high_lo_norm + 0.05)
    b_hi, a_hi = sps.butter(2, [high_lo_norm, high_hi_norm], btype="band")
    tail_high = sps.lfilter(b_hi, a_hi, noise_high * env_high)

    reflected = np.zeros(n, dtype=np.float64)
    reflected[: len(early)] += early
    reflected += tail_low + tail_high
    reflected = _schroeder_allpass_diffuse(reflected, sr, seed)

    ir = reflected
    # A direct impulse at t=0, added AFTER diffusion so the first (loudest)
    # arrival stays a crisp, undiffused transient rather than being smeared
    # by the allpass chain along with the reflections.
    ir[0] += 1.0

    peak = np.max(np.abs(ir))
    if peak > 0:
        ir = ir / peak
    return ir.astype(np.float32)


def apply_shared_reverb(audio: dict, ir: np.ndarray, wet: float, dry: float | None = None) -> dict:
    """Convolves `audio` with the given impulse response and blends wet/dry.
    `dry` defaults to (1 - wet) if not given."""
    from scipy.signal import fftconvolve

    if dry is None:
        dry = 1.0 - wet

    waveform = audio["waveform"].squeeze(0)  # [C, S]
    sr = audio["sample_rate"]
    np_wave = waveform.cpu().numpy().astype(np.float32)

    channels = []
    for ch in range(np_wave.shape[0]):
        wet_signal = fftconvolve(np_wave[ch], ir, mode="full")[: np_wave.shape[-1]]
        channels.append(np_wave[ch] * dry + wet_signal * wet)
    mixed_np = np.stack(channels, axis=0)

    peak = np.max(np.abs(mixed_np))
    if peak > 0.98:
        mixed_np = mixed_np / peak * 0.98

    return {"waveform": torch.from_numpy(mixed_np.astype(np.float32)).unsqueeze(0), "sample_rate": sr}


def normalize_lufs(audio: dict, target_lufs: float, max_db_change: float = 12.0) -> dict:
    """Approximate (RMS-based, not true K-weighted) LUFS normalization -- same
    formula convention as ComfyUI-AudioTools' AudioNormalizeLUFS node, so
    results stay consistent with that node if both are used in the same graph."""
    waveform = audio["waveform"].squeeze(0)
    sr = audio["sample_rate"]
    np_wave = waveform.cpu().numpy().astype(np.float64)

    rms = np.sqrt(np.mean(np_wave ** 2))
    if rms < 1e-10:
        return audio  # silent/empty -- nothing to normalize

    current_lufs = 20 * np.log10(rms) - 0.691
    gain_db = float(np.clip(target_lufs - current_lufs, -max_db_change, max_db_change))
    gain_linear = 10 ** (gain_db / 20)

    normalized = np_wave * gain_linear
    peak = np.max(np.abs(normalized))
    if peak > 0.98:
        normalized = normalized / peak * 0.98

    return {"waveform": torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0), "sample_rate": sr}


def generate_shared_room_tone(sr: int, duration_seconds: float, cutoff_hz: float, level: float, seed: int = 1) -> np.ndarray:
    """Very quiet, filtered noise bed generated ONCE and mixed identically
    into every voice -- the classic post-production "room tone" trick for
    masking the fact separate takes were never actually recorded together,
    since every voice now shares literally the same background texture."""
    import scipy.signal as sps

    n = max(1, int(duration_seconds * sr))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float64)

    nyquist = sr / 2.0
    cutoff_norm = min(0.99, cutoff_hz / nyquist)
    if cutoff_norm > 0.01:
        b, a = sps.butter(2, cutoff_norm, btype="low")
        noise = sps.lfilter(b, a, noise)

    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak
    return (noise * level).astype(np.float32)


def mix_in_room_tone(audio: dict, room_tone: np.ndarray) -> dict:
    """Adds the shared room-tone bed under `audio`, looping it to cover the
    full length if the bed is shorter than the voice clip."""
    waveform = audio["waveform"].squeeze(0)
    sr = audio["sample_rate"]
    np_wave = waveform.cpu().numpy().astype(np.float32)

    n = np_wave.shape[-1]
    if room_tone.shape[-1] < n:
        reps = int(np.ceil(n / room_tone.shape[-1]))
        tone = np.tile(room_tone, reps)[:n]
    else:
        tone = room_tone[:n]

    mixed = np_wave + tone[np.newaxis, :]
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed / peak * 0.98

    return {"waveform": torch.from_numpy(mixed.astype(np.float32)).unsqueeze(0), "sample_rate": sr}


def match_room(
    voices: list[dict],
    environment: str,
    target_lufs: float = -20.0,
    wet_override: float | None = None,
    room_tone_level: float = 0.0,
    seed: int = 0,
) -> list[dict]:
    """Processes a list of AUDIO dicts so they sound like they share one
    physical space: same loudness target, convolved with the SAME synthesized
    reverb impulse response, and (optionally) the same low-level room-tone
    bed mixed under all of them. Returns a new list, same order/length as
    input. All voices are resampled to the first voice's sample rate first,
    so one impulse response applies consistently to every one of them."""
    if not voices:
        return []

    preset = ENVIRONMENT_PRESETS.get(environment, ENVIRONMENT_PRESETS["room"])
    wet = wet_override if wet_override is not None else preset["wet"]

    sr = voices[0]["sample_rate"]
    aligned = []
    for v in voices:
        if v["sample_rate"] != sr:
            import torchaudio
            resample = torchaudio.transforms.Resample(v["sample_rate"], sr)
            aligned.append({"waveform": resample(v["waveform"].squeeze(0)).unsqueeze(0), "sample_rate": sr})
        else:
            aligned.append(v)

    ir = synthesize_room_ir(sr, preset, seed=seed)

    room_tone = None
    if room_tone_level > 0:
        max_dur = max(audio_duration_seconds(v) for v in aligned)
        room_tone = generate_shared_room_tone(
            sr, max_dur + 1.0, preset["cutoff_hz"] * 0.5, room_tone_level, seed=seed + 1
        )

    results = []
    for v in aligned:
        out = normalize_lufs(v, target_lufs)
        out = apply_shared_reverb(out, ir, wet)
        if room_tone is not None:
            out = mix_in_room_tone(out, room_tone)
        results.append(out)
    return results
