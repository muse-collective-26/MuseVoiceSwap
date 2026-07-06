"""
Cross-package lookups for classes/functions that live in other custom_nodes
packages whose folder names contain hyphens (so they can't be `import`ed
normally). Mirrors the pattern used by _get_encode_relay() in
comfyui-muse-collective/prompt_relay_muse.py.

All lookups happen lazily at execute() time, never at module import time,
since ComfyUI's custom_nodes load order is not guaranteed.
"""

import inspect
import sys


def _snapshot_module_keys():
    # sys.modules can change size while we iterate (other threads loading
    # modules concurrently). Retry until we get a clean copy.
    for _ in range(50):
        try:
            return list(sys.modules.keys())
        except RuntimeError:
            continue
    return []


def _find_in_loaded_modules(key_substrings, attr_name, want="any"):
    """
    Scan sys.modules for a module whose key contains ALL of key_substrings
    (case-insensitive), then return getattr(module, attr_name) if it passes
    the `want` check ("class" -> inspect.isclass, "function" -> inspect.isfunction,
    "any" -> just not None).

    Returns (value, matched_module_key) or (None, None) if nothing matched.
    """
    if isinstance(key_substrings, str):
        key_substrings = [key_substrings]
    needles = [s.lower() for s in key_substrings]

    for key in _snapshot_module_keys():
        key_lower = key.lower()
        if not all(n in key_lower for n in needles):
            continue
        mod = sys.modules.get(key)
        if mod is None:
            continue
        value = getattr(mod, attr_name, None)
        if value is None:
            continue
        if want == "class" and not inspect.isclass(value):
            continue
        if want == "function" and not inspect.isfunction(value):
            continue
        return value, key

    return None, None


def get_audio_separation_class():
    value, matched_key = _find_in_loaded_modules(
        ["audio-separation-nodes", "separation"], "AudioSeparation", want="class"
    )
    if value is None:
        # fallback: some installs may expose it under a differently-named module key
        value, matched_key = _find_in_loaded_modules(
            "audio_separation", "AudioSeparation", want="class"
        )
    if value is None:
        raise RuntimeError(
            "[MuseVoiceSwap] AudioSeparation not found — is "
            "audio-separation-nodes-comfyui installed and loaded?"
        )
    print(f"[MuseVoiceSwap] AudioSeparation resolved from module: {matched_key}")
    return value


def get_audio_combine_class():
    value, matched_key = _find_in_loaded_modules(
        ["audio-separation-nodes", "combine"], "AudioCombine", want="class"
    )
    if value is None:
        value, matched_key = _find_in_loaded_modules(
            "audio_separation", "AudioCombine", want="class"
        )
    if value is None:
        raise RuntimeError(
            "[MuseVoiceSwap] AudioCombine not found — is "
            "audio-separation-nodes-comfyui installed and loaded?"
        )
    print(f"[MuseVoiceSwap] AudioCombine resolved from module: {matched_key}")
    return value


def get_time_shift_fn():
    value, matched_key = _find_in_loaded_modules(
        ["audio-separation-nodes", "utils"], "time_shift", want="function"
    )
    if value is None:
        value, matched_key = _find_in_loaded_modules(
            ["audio-separation-nodes", "time_shift"], "time_shift", want="function"
        )
    if value is None:
        value, matched_key = _find_in_loaded_modules(
            "audio_separation", "time_shift", want="function"
        )
    if value is None:
        raise RuntimeError(
            "[MuseVoiceSwap] time_shift() not found — is "
            "audio-separation-nodes-comfyui installed and loaded?"
        )
    print(f"[MuseVoiceSwap] time_shift resolved from module: {matched_key}")
    return value


def get_fish_model_names_fn():
    value, matched_key = _find_in_loaded_modules(
        "fishaudios2", "get_model_names", want="function"
    )
    if value is None:
        raise RuntimeError(
            "[MuseVoiceSwap] get_model_names() not found — is "
            "ComfyUI-FishAudioS2 installed and loaded?"
        )
    print(f"[MuseVoiceSwap] get_model_names resolved from module: {matched_key}")
    return value


def get_fish_voice_clone_class():
    # Both ComfyUI-FishAudioS2 and ComfyUI-fish-audio-s2 may be installed side
    # by side — require the specific "fishaudios2" (no hyphen/case) needle plus
    # a class-type check on the exact attribute name to avoid grabbing the wrong
    # same-named class from an unrelated/duplicate package.
    value, matched_key = _find_in_loaded_modules(
        "fishaudios2", "FishS2VoiceCloneTTS", want="class"
    )
    if value is None:
        raise RuntimeError(
            "[MuseVoiceSwap] FishS2VoiceCloneTTS not found — is "
            "ComfyUI-FishAudioS2 installed and loaded?"
        )
    print(f"[MuseVoiceSwap] FishS2VoiceCloneTTS resolved from module: {matched_key}")
    return value
