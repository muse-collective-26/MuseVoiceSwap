# MODELS AND NODES USED IN THIS WORKFLOW

---

### Put all models in the models folder, organised into their respective subfolders.
### If a required folder doesn't exist, create it.

---

## 🟣 Gemma Text Encoder (for emotion-tag annotation)
Download **gemma4_e4b_it_fp8_scaled.safetensors** from [HERE](https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e4b_it_fp8_scaled.safetensors)

📁 **Place in:** `ComfyUI/models/text_encoders`

---

## 🟣 Fish Audio S2 Pro (voice cloning)
Download the whole **s2-pro-fp8** repo from [HERE](https://huggingface.co/drbaph/s2-pro-fp8) (`model.safetensors`, `codec.pth`, and its config/tokenizer files)

📁 **Place in:** `ComfyUI/models/fishaudioS2/s2-pro-fp8`

💡 Easier option: skip the manual download — in the Muse Voice Swap V1 node, set `fish_model_path` to **`s2-pro-fp8 (auto download)`** and it fetches this automatically on first run.

---

## 🟣 Demucs (vocal/background separation)
No manual download — **hdemucs_high_trained.pt** auto-downloads via torchaudio's own model hub the first time the node runs.

📁 **Auto-cached to:** `%USERPROFILE%\.cache\torch\hub\torchaudio\models`

---

## 🟣 ECAPA-TDNN Speaker Embedding (auto speaker-assignment)
No manual download — **speechbrain/spkrec-ecapa-voxceleb** auto-downloads from HuggingFace the first time `auto_assign_speakers` is used.

📁 **Auto-cached to:** `speechbrain_models/spkrec-ecapa-voxceleb` (inside your ComfyUI install root)

---

## 🟣 Whisper (transcription)
No manual download — the model size selected in `whisper_model_size` (e.g. `small`) auto-downloads via the `openai-whisper` package's own cache on first use.

---

## 🧩 Nodes Installed from ComfyUI Manager

- **ComfyUI-FishAudioS2** — voice cloning ([Saganaki22/ComfyUI-FishAudioS2](https://github.com/Saganaki22/ComfyUI-FishAudioS2))
- **audio-separation-nodes-comfyui** — Demucs separation, time-stretch, mixing ([christian-byrne/audio-separation-nodes-comfyui](https://github.com/christian-byrne/audio-separation-nodes-comfyui))
- **ComfyUI-VideoHelperSuite** — audio/video loading (`VHS_LoadAudioUpload`, `VHS_LoadVideo`)

Search for each by name in Manager → Custom Nodes → search

---

## 🟣 Muse Collective Voice Swap Node

Install via Manager → **Install via Git URL:**

```
https://github.com/muse-collective-26/MuseVoiceSwap.git
```

Python dependencies (`librosa`, `openai-whisper`, `soundfile`, `speechbrain`, `hyperpyyaml`) install automatically via Manager's `requirements.txt` handling.
