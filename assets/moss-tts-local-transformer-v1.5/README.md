---
license: apache-2.0
library_name: transformers
pipeline_tag: text-to-speech
tags:
- text-to-speech
- voice-cloning
- custom_code
- moss-tts
- moss-tts-local
- arxiv:2603.18090
language:
- zh
- yue
- en
- ar
- cs
- da
- de
- nl
- es
- fr
- fi
- el
- he
- hi
- hu
- ja
- it
- ko
- mk
- ms
- ru
- fa
- pl
- pt
- sv
- ro
- sw
- tl
- th
- tr
- vi
---
# MOSS-TTS Family

<br>

<p align="center">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_imgaes_demo/openmoss_x_mosi" height="50" align="middle" />
</p>

<div align="center">
  <a href="https://github.com/OpenMOSS/MOSS-TTS/tree/main"><img src="https://img.shields.io/badge/Project%20Page-GitHub-blue"></a>
  <a href="https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface"></a>
  <a href="https://modelscope.cn/collections/OpenMOSS-Team/MOSS-TTS"><img src="https://img.shields.io/badge/ModelScope-Models-lightgrey?logo=modelscope&amp"></a>
  <a href="https://mosi.cn/#models"><img src="https://img.shields.io/badge/Blog-View-blue?logo=internet-explorer&amp"></a>

  <a href="https://arxiv.org/abs/2603.18090"><img src="https://img.shields.io/badge/Arxiv-2603.18090-red?logo=Arxiv&amp"></a>
  <a href="https://studio.mosi.cn"><img src="https://img.shields.io/badge/AIStudio-Try-green?logo=internet-explorer&amp"></a>
  <a href="https://studio.mosi.cn/docs/moss-tts"><img src="https://img.shields.io/badge/API-Docs-00A3FF?logo=fastapi&amp"></a>
  <a href="https://x.com/Open_MOSS"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
  <a href="https://discord.gg/fvm5TaWjU3"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&amp"></a>
</div>

# MOSS-TTS-Local-Transformer-v1.5

**MOSS-TTS-Local-Transformer-v1.5** is continued from [MOSS-TTS-Local-Transformer-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer). It preserves the main 1.0 capabilities, including zero-shot voice cloning, long-form speech generation, token-level duration control, Pinyin/IPA pronunciation control, multilingual synthesis, and code-switching. For the full 1.0 feature walkthrough, input schema, and evaluation tables, please refer to the [MOSS-TTS-Local-Transformer-v1.0 README](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer).

Compared with [MOSS-TTS-Local-Transformer-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer), v1.5 focuses on the following improvements:
- **Higher-fidelity stereo audio modeling**: v1.5 uses [MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) as the audio tokenizer, supporting native 48 kHz stereo input and output for richer spatial detail and more natural perceived audio quality. Since the codec output is stereo, save the `[channels, samples]` tensor returned by `processor.decode(...)` directly.
- **Stronger multilingual synthesis with language tags**: when the `language` field is omitted, v1.5 may improve some languages and regress slightly on others compared with 1.0. When the language is specified, v1.5 is stronger than 1.0 on almost all supported languages. Set the tag when building the user message, for example `processor.build_user_message(text=text_fr, language="French")`.
- **More stable voice cloning**: v1.5 improves speaker similarity and reduces cloning variance, making repeated generations more consistent.
- **Better long-reference, short-text cloning**: v1.5 handles scenarios where the reference audio is much longer than the target text more reliably than 1.0.
- **More stable punctuation-following prosody**: v1.5 follows punctuation-driven pauses more closely, especially in long sentences.
- **Explicit pause control**: v1.5 supports inline pause markers such as `"[pause 3.2s]"`. For example, `我今天学习了一首中国的古诗，它的名字是[pause 3.2s]静夜思！` inserts an explicit 3.2s pause before `静夜思`.

## Supported Languages

MOSS-TTS Local Transformer v1.5 supports **31 languages**. It keeps the 20 languages supported by [MOSS-TTS-Local-Transformer-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer) and extends multilingual continued training to additional languages including Cantonese, Dutch, Finnish, Hindi, Macedonian, Malay, Romanian, Swahili, Tagalog, Thai, and Vietnamese.

| Language | Code | Flag | Language | Code | Flag | Language | Code | Flag |
|---|---|---|---|---|---|---|---|---|
| Chinese | zh | 🇨🇳 | Cantonese | yue | 🇭🇰 | English | en | 🇺🇸 |
| Arabic | ar | 🇸🇦 | Czech | cs | 🇨🇿 | Danish | da | 🇩🇰 |
| Dutch | nl | 🇳🇱 | Finnish | fi | 🇫🇮 | French | fr | 🇫🇷 |
| German | de | 🇩🇪 | Greek | el | 🇬🇷 | Hebrew | he | 🇮🇱 |
| Hindi | hi | 🇮🇳 | Hungarian | hu | 🇭🇺 | Italian | it | 🇮🇹 |
| Japanese | ja | 🇯🇵 | Korean | ko | 🇰🇷 | Macedonian | mk | 🇲🇰 |
| Malay | ms | 🇲🇾 | Persian (Farsi) | fa | 🇮🇷 | Polish | pl | 🇵🇱 |
| Portuguese | pt | 🇵🇹 | Romanian | ro | 🇷🇴 | Russian | ru | 🇷🇺 |
| Spanish | es | 🇪🇸 | Swahili | sw | 🇹🇿 | Swedish | sv | 🇸🇪 |
| Tagalog | tl | 🇵🇭 | Thai | th | 🇹🇭 | Turkish | tr | 🇹🇷 |
| Vietnamese | vi | 🇻🇳 | | | | | | |

## Quick Start

### Environment Setup

We recommend a clean, isolated Python environment with **Transformers 5.0.0**, or a recent Transformers version with Qwen3 support, to avoid dependency conflicts.

```bash
conda create -n moss-tts python=3.12 -y
conda activate moss-tts
```

Install all required dependencies:

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime]"
```

#### (Optional) Install FlashAttention 2
For better speed and lower GPU memory usage, you can install FlashAttention 2 if your hardware supports it.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]" --no-build-isolation
```

If your machine has limited RAM and many CPU cores, you can cap build parallelism:

```bash
MAX_JOBS=4 pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[flash-attn]" --no-build-isolation
```

Notes:
- Dependencies are managed in `pyproject.toml`, which currently pins `torch==2.9.1+cu128` and `torchaudio==2.9.1+cu128`.
- If FlashAttention 2 fails to build on your machine, you can skip it and use the default attention backend.
- FlashAttention 2 is only available on supported GPUs and is typically used with `torch.float16` or `torch.bfloat16`.

### Basic Usage

> Tip: MOSS-TTS-Local-Transformer-v1.5 uses a fixed 12-codebook RVQ depth. Do not set `n_vq_for_inference` to a value different from `config.n_vq`.

MOSS-TTS-Local-Transformer-v1.5 provides the standard Hugging Face `AutoProcessor` and `AutoModel` interface. The examples below cover:
1. Direct generation with language tags
2. Voice cloning
3. Duration control
4. Explicit pause control with `[pause X.Ys]`

```python
from pathlib import Path
from tqdm import tqdm
import importlib.util

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

# Disable the broken cuDNN SDPA backend on some CUDA/PyTorch combinations.
torch.backends.cuda.enable_cudnn_sdp(False)
# Keep these enabled as fallbacks.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


def resolve_attn_implementation() -> str:
    # Prefer FlashAttention 2 when package + device conditions are met.
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    # CUDA fallback: use PyTorch SDPA kernels.
    if device == "cuda":
        return "sdpa"
    # CPU fallback.
    return "eager"


attn_implementation = resolve_attn_implementation()
print(f"[INFO] Using attn_implementation={attn_implementation}")

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

text_zh = "亲爱的你，愿你的每一天都值得被记住，也值得被珍惜。"
text_en = "We stand on the threshold of the AI era, where intelligence becomes an extension of human creativity."
text_fr = "Bonjour, je voudrais essayer une voix francaise naturelle et stable."
text_pause = "我今天学习了一首中国的古诗，它的名字是[pause 3.2s]静夜思！"

# Use remote demo audio to avoid requiring local assets.
ref_audio_zh = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_zh.wav"
ref_audio_en = "https://speech-demo.oss-cn-shanghai.aliyuncs.com/moss_tts_demo/tts_readme_demo/reference_en.m4a"

conversations = [
    # Direct TTS. Language tags are recommended in v1.5 when the language is known.
    [processor.build_user_message(text=text_zh, language="Chinese")],
    [processor.build_user_message(text=text_en, language="English")],
    [processor.build_user_message(text=text_fr, language="French")],
    # Explicit pause control. Use [pause X.Ys], such as [pause 3.2s].
    [processor.build_user_message(text=text_pause, language="Chinese")],
    # Voice cloning with a reference audio.
    [processor.build_user_message(text=text_zh, reference=[ref_audio_zh], language="Chinese")],
    [processor.build_user_message(text=text_en, reference=[ref_audio_en], language="English")],
    # Duration control. At 12.5 frames per second, 125 frames is about 10 seconds.
    [processor.build_user_message(text=text_en, tokens=125, language="English")],
]

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=attn_implementation,
    torch_dtype=dtype,
).to(device)
model.eval()

batch_size = 1
save_dir = Path("inference_root_moss_tts_local_v1_5")
save_dir.mkdir(exist_ok=True, parents=True)
sample_idx = 0

with torch.no_grad():
    for start in tqdm(range(0, len(conversations), batch_size)):
        batch_conversations = conversations[start : start + batch_size]
        batch = processor(batch_conversations, mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            do_sample=True,
            audio_temperature=1.7,
            audio_top_p=0.8,
            audio_top_k=25,
            audio_repetition_penalty=1.0,
        )

        for message in processor.decode(outputs):
            if message is None:
                continue
            audio = message.audio_codes_list[0]
            out_path = save_dir / f"sample{sample_idx}.wav"
            sample_idx += 1
            # MOSS-TTS Local v1.5 codec returns stereo audio as [channels, samples].
            # Save the two-channel tensor directly.
            torchaudio.save(str(out_path), audio, processor.model_config.sampling_rate)
```

## Generation Parameters

| Parameter | Recommended | Description |
|---|---:|---|
| `audio_temperature` | `1.7` | Sampling temperature for audio RVQ layers. |
| `audio_top_p` | `0.8` | Nucleus sampling cutoff for audio RVQ layers. |
| `audio_top_k` | `25` | Top-k sampling cutoff for audio RVQ layers. |
| `audio_repetition_penalty` | `1.0` | Penalty for repeated acoustic token patterns. |
| `n_vq_for_inference` | `12` | Fixed by this release. Values other than `config.n_vq` are rejected. |

## Notes

- This repository uses Hugging Face remote code. Load it with `trust_remote_code=True`.
- The MOSS-TTS-Local-Transformer-v1.5 codec is stereo. `processor.decode(...)` returns audio tensors shaped as `[channels, samples]`, so save them directly with `torchaudio.save(path, audio, sampling_rate)`.
- Audio encoding and decoding use `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`.
- The model configuration sets `sampling_rate` to 48000 and `n_vq` to 12.
- If FlashAttention 2 is unavailable, the example falls back to SDPA on CUDA and eager attention on CPU.

## SGLang Usage

You can serve MOSS-TTS-Local-Transformer-v1.5 with [SGLang-Omni](https://github.com/sgl-project/sglang-omni), which exposes an OpenAI-compatible `/v1/audio/speech` API for reference-less synthesis, zero-shot voice cloning, streaming, duration control, and language/style hints.

See the [MOSS-TTS-Local cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts_local.md) for installation, full API details, deployment config, benchmarking, and limitations.

### Install and Serve

Install `sglang-omni` by following the [SGLang-Omni installation guide](https://sgl-project.github.io/sglang-omni/get_started/installation.html), then download and serve the model:

```bash
hf download OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5

sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --port 8000
```

A matching config file is available in SGLang-Omni at `examples/configs/moss_tts_local.yaml`.

### Basic Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "SGLang-Omni is a great project!"}' \
  --output output.wav
```

### Voice Cloning

Provide a reference clip and its transcript for better speaker similarity. `audio_path` may be a local path readable by the server, an HTTP(S) URL, or a base64 data URI.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are accepted as shorthand for `references[0].audio_path` and `references[0].text`.

### Streaming

Set `"stream": true`, `"response_format": "pcm"`, and `"stream_format": "audio"` to receive raw 48 kHz PCM chunks. Pipe the stream through `ffmpeg` to write a playable WAV file:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "stream": true,
    "response_format": "pcm",
    "stream_format": "audio"
  }' \
  | ffmpeg -f s16le -ar 48000 -ac 1 -i pipe:0 output_stream.wav
```

### Duration, Markup, and Language

Duration can be guided with an inline `${token:N}` prefix or with `token_count` / `duration_tokens`. Inline markup such as `[pause 0.5s]`, Pinyin, and IPA is passed through unchanged. Use `language` to hint the target language and `instructions` for free-form style guidance.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "${token:150}今天天气不错 [pause 0.5s] 就该出去晒晒太阳。",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "language": "Chinese"
  }' \
  --output output_markup.wav
```

## More Usage

MOSS-TTS-Local-Transformer-v1.5 is API-compatible with MOSS-TTS-Local-Transformer-v1.0. For continuation with prefix audio, detailed `UserMessage` and `AssistantMessage` fields, generation hyperparameters, Pinyin/IPA preprocessing examples, and evaluation results, see the [MOSS-TTS-Local-Transformer-v1.0](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer).

## Citation

If you use this model, please cite the [MOSS-TTS Technical Report](https://arxiv.org/abs/2603.18090).
