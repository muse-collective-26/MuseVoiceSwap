"""
ComfyUI node wrapping voice_room_match_pipeline -- makes multiple voice
clips (e.g. MuseVoiceSwap's per-speaker cloned outputs) sound like they
were captured together in the same physical space, with selectable
environment presets (Hall, Auditorium, Jazz Club, Rock Arena, Room,
Outdoor, etc.) analogous to a consumer audio system's reverb menu.
"""

from . import voice_room_match_pipeline as vrm


class MuseVoiceRoomMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_1": ("AUDIO", {"tooltip": "First voice to blend into the shared space."}),
                "environment": (vrm.ENVIRONMENT_NAMES, {
                    "default": "room",
                    "tooltip": (
                        "Shared acoustic space applied IDENTICALLY to every connected "
                        "voice -- same reverb impulse response, same room tone -- which "
                        "is what actually sells 'recorded together', not matching pitch/"
                        "tone (not possible between different voices without changing "
                        "who it sounds like)."
                    ),
                }),
                "target_lufs": ("FLOAT", {
                    "default": -20.0, "min": -40.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Common loudness target all voices are matched to before the shared reverb is applied.",
                }),
                "wet_level": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Reverb wet mix override. -1 = use the environment preset's own default wet level.",
                }),
                "room_tone_level": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.1, "step": 0.005,
                    "tooltip": "Shared, very quiet background noise bed mixed identically into every voice (0 = off).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31 - 1,
                    "tooltip": "Same seed = same synthesized room impulse response/room-tone every run.",
                }),
            },
            "optional": {
                "voice_2": ("AUDIO", {"tooltip": "Second voice, blended into the same shared space as voice_1."}),
                "voice_3": ("AUDIO", {"tooltip": "Third voice, blended into the same shared space as voice_1."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("voice_1_out", "voice_2_out", "voice_3_out")
    FUNCTION = "execute"
    CATEGORY = "Muse Collective"
    DESCRIPTION = (
        "Makes multiple separately-cloned voices sound like they were recorded "
        "together in the same physical space: shared loudness target, one "
        "synthesized reverb impulse response convolved identically into every "
        "voice, and an optional shared room-tone bed. Pick an environment preset "
        "(Room, Hall, Auditorium, Jazz Club, Rock Arena, Cathedral, Outdoor, "
        "Studio Dry) the same way you'd pick a reverb preset on a consumer audio system."
    )

    def execute(self, voice_1, environment, target_lufs, wet_level, room_tone_level, seed, voice_2=None, voice_3=None):
        voices = [voice_1]
        indices = [0]
        if voice_2 is not None:
            voices.append(voice_2)
            indices.append(1)
        if voice_3 is not None:
            voices.append(voice_3)
            indices.append(2)

        wet_override = None if wet_level < 0 else wet_level
        processed = vrm.match_room(
            voices, environment,
            target_lufs=target_lufs,
            wet_override=wet_override,
            room_tone_level=room_tone_level,
            seed=seed,
        )

        outputs = [None, None, None]
        for idx, audio in zip(indices, processed):
            outputs[idx] = audio

        # Unconnected voice slots still need a valid AUDIO dict returned (a
        # node output can't be None) -- give them a single silent sample
        # rather than duplicating a real voice's audio into an unused output.
        for i in range(3):
            if outputs[i] is None:
                sr = voice_1["sample_rate"]
                outputs[i] = {"waveform": voice_1["waveform"][:, :, :1] * 0.0, "sample_rate": sr}

        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "MuseVoiceRoomMatch": MuseVoiceRoomMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MuseVoiceRoomMatch": "Muse Voice Room Match",
}
