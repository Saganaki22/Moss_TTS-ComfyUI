# Moss_TTS-ComfyUI

**English** | **[简体中文](./README_zh.md)**

**Version: v0.1.1**

[![Version](https://img.shields.io/badge/version-v0.1.1-blue)](https://github.com/Saganaki22/Moss_TTS-ComfyUI/releases/tag/v0.1.1)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-2d7dd2)](https://github.com/comfyanonymous/ComfyUI)
[![ComfyUI Manager](https://img.shields.io/badge/ComfyUI_Manager-compatible-6f42c1)](https://github.com/ltdrdata/ComfyUI-Manager)
[![MOSS-TTS v1.5](https://img.shields.io/badge/Hugging_Face-MOSS--TTS--Local--Transformer--v1.5-ffd21e)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
[![Audio Tokenizer](https://img.shields.io/badge/Hugging_Face-MOSS--Audio--Tokenizer--v2-ffd21e)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)
[![Upstream](https://img.shields.io/badge/Upstream-OpenMOSS%2FMOSS--TTS-111111)](https://github.com/OpenMOSS/MOSS-TTS)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)

ComfyUI nodes for [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5), a 48 kHz stereo local Transformer TTS model with direct generation, zero-shot voice cloning, continuation, duration control, pronunciation markup, multilingual synthesis, and code-switching.

This node pack keeps the small Hugging Face remote-code, tokenizer, and config assets inside the custom node folder. Large model files are stored under `ComfyUI/models/mosstts/`, and downloads only happen when `download_if_missing` is enabled.

> Use voice cloning only with voices you own or have explicit permission to use. Do not use this project for impersonation, fraud, harassment, consent evasion, or other harmful use.

## Highlights

- Native ComfyUI `AUDIO` input and 48 kHz stereo `AUDIO` output.
- Direct TTS without reference audio.
- Zero-shot voice cloning from a ComfyUI audio reference.
- Speech continuation for finishing or extending an existing sentence with prefix audio plus transcript.
- Optional Whisper transcription node for continuation prefix text.
- Language tags for all 31 languages listed by the v1.5 model card.
- Inline pause markers such as `[pause 0.5s]`, plus Pinyin and IPA pronunciation control.
- `auto`, `bf16`, and `fp16` dtype selection.
- `auto`, `sdpa`, `flash_attention`, and `eager` attention selection.
- AIMDO DynamicVRAM integration: when AIMDO is active, the MOSS model, MOSS-Audio-Tokenizer, and optional Whisper ASR model are registered with native dynamic patchers.
- Native ComfyUI generation progress plus terminal `tqdm` frame progress.
- No separate unload node. Reloading with different model, dtype, attention, or tokenizer files unloads the previous bundle internally.

## Installation

### ComfyUI Manager

Once this node is listed in ComfyUI Manager, search for **MOSS-TTS** and install it normally.

To install immediately through Manager:

1. Open **Manager**.
2. Choose **Install via Git URL**.
3. Enter:

```text
https://github.com/Saganaki22/Moss_TTS-ComfyUI
```

4. Restart ComfyUI.

### Manual Install

From `ComfyUI/custom_nodes`:

```powershell
git clone https://github.com/Saganaki22/Moss_TTS-ComfyUI.git
..\venv\Scripts\python.exe Moss_TTS-ComfyUI\install.py
```

Linux or portable venv installs can run:

```bash
git clone https://github.com/Saganaki22/Moss_TTS-ComfyUI.git
../venv/bin/python Moss_TTS-ComfyUI/install.py
```

For a `uv`-managed ComfyUI environment:

```bash
uv run python Moss_TTS-ComfyUI/install.py
```

`install.py` installs only missing lightweight dependencies such as `accelerate`, `huggingface-hub`, `numpy`, `safetensors`, and `tqdm`. It does not replace ComfyUI's `torch`, `torchaudio`, `torchvision`, or `transformers`.

Restart ComfyUI after installing or updating.

## Example Workflows

Example workflow files and preview images are included under:

```text
ComfyUI/custom_nodes/Moss_TTS-ComfyUI/example_workflows/
```

## Model Files

The loader uses this structure:

```text
ComfyUI/
├── custom_nodes/
│   └── Moss_TTS-ComfyUI/
│       ├── assets/
│       │   ├── moss-tts-local-transformer-v1.5/
│       │   │   ├── config.json
│       │   │   ├── configuration_moss_tts.py
│       │   │   ├── modeling_moss_tts.py
│       │   │   ├── processing_moss_tts.py
│       │   │   ├── qwen3_decoder.py
│       │   │   ├── gpt2_decoder.py
│       │   │   └── tokenizer files
│       │   └── moss-audio-tokenizer-v2/
│       │       ├── config.json
│       │       ├── configuration_moss_audio_tokenizer.py
│       │       └── modeling_moss_audio_tokenizer.py
│       ├── README.md
│       ├── README_zh.md
│       └── pyproject.toml
└── models/
    └── mosstts/
        ├── moss-tts-local-transformer-v1.5/
        │   └── model.safetensors
        └── moss-audio-tokenizer-v2/
            ├── model.safetensors.index.json
            ├── model-00001-of-00003.safetensors
            ├── model-00002-of-00003.safetensors
            └── model-00003-of-00003.safetensors
```

Model sources:

| Component | Source | Local destination |
| --- | --- | --- |
| MOSS-TTS Local Transformer v1.5 | [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | `ComfyUI/models/mosstts/moss-tts-local-transformer-v1.5/model.safetensors` |
| MOSS-Audio-Tokenizer-v2 | [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) | `ComfyUI/models/mosstts/moss-audio-tokenizer-v2/` |
| Optional Whisper ASR | [OpenAI Whisper models](https://huggingface.co/openai) | `ComfyUI/models/audio_encoders/` |

The main MOSS repository is not snapshot-downloaded into `models/`. Only `model.safetensors` is downloaded there. The small remote-code/config/tokenizer files are bundled under `assets/`, then linked or copied beside the local weights at load time so Transformers can load from a normal local folder.

## Nodes

| Node | Purpose |
| --- | --- |
| `MOSS-TTS Load Model` | Loads the MOSS model and stereo audio tokenizer. Exposes model catalog, dtype, attention backend, and `download_if_missing`. |
| `MOSS-TTS Generate Speech` | Reference-less TTS with language hints, style fields, duration tokens, pause tags, Pinyin/IPA, and sampling controls. |
| `MOSS-TTS Voice Clone` | Zero-shot voice cloning from a ComfyUI `AUDIO` reference. No reference transcript is required for this local v1.5 path. |
| `MOSS-TTS Continue Speech` | Continues prefix audio. Requires prefix audio and its transcript, then generates continuation text. |
| `MOSS-TTS Whisper Transcribe` | Optional ASR helper for producing `prefix_text` for continuation workflows. |

<details>
<summary><strong>Node Settings</strong></summary>

### MOSS-TTS Load Model

| Setting | Default | Notes |
| --- | --- | --- |
| `model` | `MOSS-TTS Local Transformer v1.5 BF16 - OpenMOSS-Team` | Catalog entry for the v1.5 checkpoint. |
| `dtype` | `auto` | `auto` reads the model config, currently BF16. Manual options: `bf16`, `fp16`. |
| `attention` | `auto` | `auto` prefers FlashAttention 2 when installed and compatible, falls back to SDPA on CUDA and eager on CPU. Manual options: `sdpa`, `flash_attention`, `eager`. |
| `download_if_missing` | `true` | Downloads missing main weight and audio-tokenizer model files only when enabled. |

### Generation, Voice Clone, and Continuation

| Setting | Notes |
| --- | --- |
| `text` / `continuation_text` | Target text. Inline `[pause X.Ys]`, Pinyin, IPA, and code-switching text are passed through. |
| `language` | Optional language hint. `auto` omits the field. |
| `instruction` | Optional style or delivery instruction. |
| `duration_tokens` | Optional duration target. `0` omits it. MOSS runs at about 12.5 acoustic frames per second. |
| `quality`, `sound_event`, `ambient_sound` | Extra fields supported by the MOSS user-message schema. |
| `max_new_tokens` | Generation budget. Increase for long-form speech. |
| `do_sample` | Enables stochastic sampling. |
| `text_temperature`, `text_top_p`, `text_top_k` | Sampling controls for assistant/control tokens. |
| `audio_temperature`, `audio_top_p`, `audio_top_k`, `audio_repetition_penalty` | Sampling controls for acoustic codebooks. Defaults follow the v1.5 model card recommendations. |
| `seed` | `0` leaves generation unseeded; positive values make sampling repeatable. |
| `reference_audio` | Voice Clone only. ComfyUI `AUDIO` reference encoded by MOSS-Audio-Tokenizer-v2. |
| `prefix_audio` | Continue Speech only. Audio prefix to continue from. |
| `prefix_text` | Continue Speech only. Transcript of the prefix audio. Required for continuation. |
| `return_full_audio` | Continue Speech only. Returns prefix plus continuation when enabled. |

### Whisper Transcribe

| Setting | Notes |
| --- | --- |
| `audio` | Input audio to transcribe. |
| `model` | Whisper model catalog under `ComfyUI/models/audio_encoders/`. |
| `dtype` | `auto`, `bf16`, or `fp32`. |
| `language` | Optional Whisper language hint. |
| `task` | `transcribe` or `translate`. |
| `chunk_length_s` | Chunk size for longer clips. |
| `download_if_missing` | Downloads the selected Whisper model when missing. |

</details>

<details>
<summary><strong>Languages</strong></summary>

MOSS-TTS Local Transformer v1.5 lists 31 supported languages:

| Language | Code | Language | Code | Language | Code |
| --- | --- | --- | --- | --- | --- |
| Chinese | `zh` | Cantonese | `yue` | English | `en` |
| Arabic | `ar` | Czech | `cs` | Danish | `da` |
| Dutch | `nl` | Finnish | `fi` | French | `fr` |
| German | `de` | Greek | `el` | Hebrew | `he` |
| Hindi | `hi` | Hungarian | `hu` | Italian | `it` |
| Japanese | `ja` | Korean | `ko` | Macedonian | `mk` |
| Malay | `ms` | Persian (Farsi) | `fa` | Polish | `pl` |
| Portuguese | `pt` | Romanian | `ro` | Russian | `ru` |
| Spanish | `es` | Swahili | `sw` | Swedish | `sv` |
| Tagalog | `tl` | Thai | `th` | Turkish | `tr` |
| Vietnamese | `vi` |  |  |  |  |

</details>

<details>
<summary><strong>Advanced Notes</strong></summary>

- This node uses `trust_remote_code=True` against local files copied from the bundled `assets/` folders.
- The v1.5 codec is stereo. `processor.decode(...)` returns `[channels, samples]`; the runtime saves that tensor shape directly.
- Audio encoding and decoding use MOSS-Audio-Tokenizer-v2.
- v1.5 config sets `sampling_rate=48000` and `n_vq=12`.
- Leave `attention=auto` unless you are debugging. FlashAttention 2 is optional; SDPA is the normal CUDA fallback.
- When AIMDO DynamicVRAM is enabled in ComfyUI, the loader forces native dynamic patchers for the main model, audio tokenizer, and optional Whisper ASR model. Whisper is CPU-staged first so CUDA allocation is handed to ComfyUI/AIMDO.
- Generation reports progress to both the ComfyUI progress bar and a terminal `tqdm` bar, based on generated acoustic frames up to `max_new_tokens`.
- If `download_if_missing=false`, all model files must already exist in the folder structure above.

</details>

<details>
<summary><strong>Troubleshooting</strong></summary>

### Hugging Face downloads use the wrong mirror

Some custom nodes set `HF_ENDPOINT` globally. This node does not force an endpoint. If downloads go to an unwanted mirror, inspect the ComfyUI process environment and other custom nodes that mutate `os.environ["HF_ENDPOINT"]`.

### `Unexpected keyword argument local_files_only`

Older MOSS processor remote code can leak `local_files_only` into `ProcessorMixin`. This node avoids passing that keyword to `AutoProcessor.from_pretrained(...)` while still loading from local folders.

### `property 'device' ... has no setter`

MOSS remote-code models expose a read-only `device` property, while ComfyUI patchers write `model.device` during load and unload. This node installs a writable instance-local compatibility property before registering the model with ComfyUI.

</details>

## Project Links

- Repository: [Saganaki22/Moss_TTS-ComfyUI](https://github.com/Saganaki22/Moss_TTS-ComfyUI)
- Upstream project: [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- Model card: [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
- Audio tokenizer: [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)
- Technical report: [MOSS-TTS Technical Report](https://arxiv.org/abs/2603.18090)

## Citation

If you use this model, please cite the MOSS-TTS technical report:

```bibtex
@misc{gong2026mossttstechnicalreport,
  title         = {MOSS-TTS Technical Report},
  author        = {Yitian Gong and Botian Jiang and Yiwei Zhao and Yucheng Yuan and Kuangwei Chen and Yaozhou Jiang and Cheng Chang and Dong Hong and Mingshu Chen and Ruixiao Li and Yiyang Zhang and Yang Gao and Hanfu Chen and Ke Chen and Songlin Wang and Xiaogui Yang and Yuqian Zhang and Kexin Huang and ZhengYuan Lin and Kang Yu and Ziqi Chen and Jin Wang and Zhaoye Fei and Qinyuan Cheng and Shimin Li and Xipeng Qiu},
  year          = {2026},
  eprint        = {2603.18090},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2603.18090}
}
```
