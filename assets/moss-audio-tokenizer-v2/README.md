---
license: apache-2.0
library_name: transformers
tags:
  - audio
  - audio-tokenizer
  - neural-codec
  - moss-tts-family
  - MOSS Audio Tokenizer
  - speech-tokenizer
  - trust-remote-code
  - arxiv:2602.10934
---

# MOSS-Audio-Tokenizer-v2

This is the code for the 48khz stereo version of MOSS-Audio-Tokenizer presented in [MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models](https://arxiv.org/abs/2602.10934).

**MOSS-Audio-Tokenizer-v2** is a unified discrete audio tokenizer based on the **Cat** (**C**ausal **A**udio **T**okenizer with **T**ransformer) architecture. Scaling to 2 billion parameters, it functions as a unified discrete interface, delivering both lossless-quality reconstruction and high-level semantic alignment.

**Key Features:**

*   **Extreme Compression & Variable Bitrate**: It compresses 48kHz stereo audio into a remarkably low frame rate of 12.5Hz. Utilizing a 32-layer Residual Vector Quantization stack, it supports high-fidelity reconstruction across a wide range of bitrates.
*   **Pure Transformer Architecture**: The model features a "CNN-free" homogeneous architecture built entirely from Causal Transformer blocks. With 2B combined parameters (Encoder + Decoder), it ensures exceptional scalability and supports low-latency streaming inference.
*   **Large-Scale General Audio Training**: Trained on 3 million hours of diverse audio data, the model excels at encoding and reconstructing all audio domains, including speech, sound effects, and music.
*   **Unified Semantic-Acoustic Representation**: While achieving state-of-the-art reconstruction quality, Cat produces discrete tokens that are "semantic-rich," making them ideal for downstream tasks like speech understanding (ASR) and generation (TTS).
*   **Fully Trained From Scratch**: Cat does not rely on any pretrained encoders (such as HuBERT or Whisper) or distillation from teacher models. All representations are learned autonomously from raw data.
*   **End-to-End Joint Optimization**: All components—including the encoder, quantizer, decoder, discriminator, and a decoder-only LLM for semantic alignment—are optimized jointly in a single unified training pipeline.

**Summary:**
By combining a simple, scalable architecture with massive-scale data, the Cat architecture overcomes the bottlenecks of traditional audio tokenizers. It provides a robust, high-fidelity, and semantically grounded interface for the next generation of native audio foundation models.

This repository contains a lightweight remote-code implementation that mirrors the current 🤗 Transformers
`transformers.models.moss_audio_tokenizer` module. It is hosted as a Hugging Face Hub model repository and should be
loaded with `trust_remote_code=True`.

<br>
<p align="center">
    <img src="images/arch.png" width="95%"> <br>
    Architecture of MossAudioTokenizer
</p>
<br>

## Usage

### Quickstart

```python
import torch
from transformers import AutoModel
import torchaudio

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval().to(device)

audio_path = "demo/demo_gt.wav"  # replace with your own 48 kHz stereo audio path if needed
wav, sr = torchaudio.load(audio_path)
if sr != model.sampling_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sampling_rate)
if wav.shape[0] == 1:
    wav = wav.repeat(model.config.number_channels, 1)
else:
    wav = wav[: model.config.number_channels]
wav = wav.unsqueeze(0).to(device)
enc = model.encode(wav, return_dict=True)
print(f"enc.audio_codes.shape: {enc.audio_codes.shape}")
dec = model.decode(enc.audio_codes, return_dict=True)
print(f"dec.audio.shape: {dec.audio.shape}")
wav = dec.audio.squeeze(0)
torchaudio.save("demo/demo_rec.wav", wav.cpu(), sample_rate=model.sampling_rate)

# Decode using only the first 8 layers of the RVQ
dec_rvq8 = model.decode(enc.audio_codes[:8], return_dict=True)
wav_rvq8 = dec_rvq8.audio.squeeze(0)
torchaudio.save("demo/demo_rec_rvq8.wav", wav_rvq8.cpu(), sample_rate=model.sampling_rate)
```

For production use with `trust_remote_code=True`, pin `revision` to a reviewed commit hash.

### Attention Backend And Compute Dtype

`config.attention_implementation` controls whether transformer layers prefer `sdpa` or `flash_attention_2`.
`config.compute_dtype` controls the non-quantizer autocast dtype and supports `fp32`, `bf16`.
`config.codec_weight_dtype` controls encoder/decoder parameter dtype and defaults to `fp32`.
The quantizer is always kept in fp32.

GPU bf16 loading:

```python
import torch
from transformers import AutoModel

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, low_cpu_mem_usage=True, codec_weight_dtype="bf16").eval().to(device)
```
Passing codec_weight_dtype="bf16" at load time avoids first materializing encoder/decoder weights as fp32 on GPU and then converting them to bf16, which would increase peak GPU memory.

You can also switch an already loaded model:

```python
model.set_attention_implementation("flash_attention_2")
model.set_compute_dtype("bf16")
model.set_codec_weight_dtype("bf16")  # encoder/decoder bf16, quantizer fp32
```

Avoid calling plain `model.to(torch.bfloat16)` on the whole codec; that also casts quantizer weights and can cause dtype mismatches or serious precision loss.

### Streaming

`MossAudioTokenizerModel.encode`, `decode`, `batch_encode`, and `batch_decode` all support streaming through a
`chunk_duration` argument.

- `chunk_duration` is expressed in seconds.
- `chunk_duration * MossAudioTokenizerConfig.sampling_rate` must be divisible by `MossAudioTokenizerConfig.downsample_rate`.
- Streaming batch inference is supported.
- The public waveform interface expects stereo inputs shaped `(2, T)` or batched stereo inputs shaped `(B, 2, T)`.

```python
import torch
from transformers import AutoModel

repo_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval().to(device)
audio = torch.randn(2, 48000 * 6).to(device)  # dummy stereo waveform

# 6.0s @ 48kHz = 288000 samples, divisible by downsample_rate=3840
enc = model.encode(audio.unsqueeze(0), return_dict=True, chunk_duration=0.08)
dec = model.decode(enc.audio_codes, return_dict=True, chunk_duration=0.08)

batch_enc = model.batch_encode([audio, audio[:, : 48000 * 3]], chunk_duration=0.08)
codes_list = [
    batch_enc.audio_codes[:, i, : batch_enc.audio_codes_lengths[i]]
    for i in range(batch_enc.audio_codes.shape[1])
]
batch_dec = model.batch_decode(codes_list, chunk_duration=0.08)
```

## Repository layout

- `configuration_moss_audio_tokenizer.py`
- `modeling_moss_audio_tokenizer.py`
- `__init__.py`
- `config.json`
- `model.safetensors.index.json`
- sharded model weights: `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`,
  `model-00003-of-00003.safetensors`
- `demo/demo_gt.wav`

## Evaluation Metrics

The table below compares the reconstruction quality of open-source audio tokenizers with MOSS-Audio-Tokenizer-v2 on speech and audio/music data.

- Speech metrics are evaluated on LibriSpeech test-clean (English) and AISHELL-2 (Chinese), reported as EN/ZH.
- Audio metrics are evaluated on the AudioSet evaluation subset, while music metrics are evaluated on MUSDB, reported as audio/music.
- STFT-Dist. denotes the STFT distance.
- Higher is better for speech metrics, while lower is better for audio/music metrics (Mel-Loss, STFT-Dist.).
- Nq denotes the number of quantizers.

| Model | bps | Frame rate | Nq | Speech: SIM ↑ (EN/ZH) | Speech: STOI ↑ (EN/ZH) | Speech: PESQ-NB ↑ (EN/ZH) | Speech: PESQ-WB ↑ (EN/ZH) | Audio/Music: Mel-Loss ↓ | Audio/Music: STFT-Dist. ↓ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XCodec2.0** | 800 | 50 | 1 | 0.82 / 0.74 | 0.92 / 0.86 | 3.04 / 2.46 | 2.43 / 1.96 | -- / -- | -- / -- |
| **MiMo Audio Tokenizer** | 850 | 25 | 4 | 0.80 / 0.74 | 0.91 / 0.87 | 2.94 / 2.62 | 2.39 / 2.14 | **0.82** / 0.81 | 2.33 / 2.23 |
| **Higgs Audio Tokenizer** | 1000 | 25 | 4 | 0.77 / 0.68 | 0.83 / 0.82 | 3.03 / 2.61 | 2.48 / 2.14 | 0.83 / **0.80** | **2.20** / **2.05** |
| **SpeechTokenizer** | 1000 | 50 | 2 | 0.36 / 0.25 | 0.77 / 0.68 | 1.59 / 1.38 | 1.25 / 1.17 | -- / -- | -- / -- |
| **XY-Tokenizer** | 1000 | 12.5 | 8 | 0.85 / 0.79 | 0.92 / 0.87 | 3.10 / 2.63 | 2.50 / 2.12 | -- / -- | -- / -- |
| **BigCodec** | 1040 | 80 | 1 | 0.84 / 0.69 | 0.93 / 0.88 | 3.27 / 2.55 | 2.68 / 2.06 | -- / -- | -- / -- |
| **Mimi** | 1100 | 12.5 | 8 | 0.74 / 0.59 | 0.91 / 0.85 | 2.80 / 2.24 | 2.25 / 1.78 | 1.24 / 1.19 | 2.62 / 2.49 |
| **MOSS-Audio-Tokenizer-v2 (Ours)** | 750 | 12.5 | 6 | 0.82 / 0.75 | 0.92 / 0.88 | 3.14 / 2.68 | 2.59 / 2.19 | 0.93 / 0.91 | 2.28 / 2.14 |
| **MOSS-Audio-Tokenizer-v2 (Ours)** | 1000 | 12.5 | 8 | **0.88** / **0.80** | **0.94** / **0.90** | **3.39** / **2.93** | **2.88** / **2.43** | 0.88 / 0.86 | 2.22 / 2.07 |
| **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |
| **DAC** | 1500 | 75 | 2 | 0.48 / 0.41 | 0.83 / 0.79 | 1.87 / 1.67 | 1.48 / 1.37 | -- / -- | -- / -- |
| **Encodec** | 1500 | 75 | 2 | 0.60 / 0.45 | 0.85 / 0.81 | 1.94 / 1.80 | 1.56 / 1.48 | 1.12 / 1.04 | 2.60 / 2.42 |
| **Higgs Audio Tokenizer** | 2000 | 25 | 8 | 0.90 / 0.83 | 0.85 / 0.85 | 3.59 / 3.22 | 3.11 / 2.73 | 0.74 / 0.70 | **2.07** / **1.92** |
| **SpeechTokenizer** | 2000 | 50 | 4 | 0.66 / 0.50 | 0.88 / 0.80 | 2.38 / 1.79 | 1.92 / 1.49 | -- / -- | -- / -- |
| **Qwen3 TTS Tokenizer** | 2200 | 12.5 | 16 | **0.95** / 0.88 | **0.96** / 0.93 | 3.66 / 3.10 | 3.19 / 2.62 | -- / -- | -- / -- |
| **MiMo Audio Tokenizer** | 2250 | 25 | 12 | 0.89 / 0.83 | 0.95 / 0.92 | 3.57 / 3.25 | 3.05 / 2.71 | **0.70** / **0.68** | 2.21 / 2.10 |
| **Mimi** | 2475 | 12.5 | 18 | 0.89 / 0.76 | 0.94 / 0.91 | 3.49 / 2.90 | 2.97 / 2.35 | 1.10 / 1.06 | 2.45 / 2.32 |
| **MOSS-Audio-Tokenizer-v2 (Ours)** | 1500 | 12.5 | 12 | 0.93 / 0.86 | 0.95 / 0.92 | 3.66 / 3.24 | 3.23 / 2.77 | 0.83 / 0.79 | 2.15 / 1.98 |
| **MOSS-Audio-Tokenizer-v2 (Ours)** | 2000 | 12.5 | 16 | **0.95** / **0.89** | **0.96** / **0.94** | **3.80** / **3.44** | **3.45** / **3.01** | 0.79 / 0.75 | 2.10 / 1.93 |
| **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** | **—** |
| **DAC** | 3000 | 75 | 4 | 0.74 / 0.67 | 0.90 / 0.88 | 2.76 / 2.47 | 2.31 / 2.07 | 0.86 / 0.83 | 2.23 / 2.10 |
| **MiMo Audio Tokenizer** | 3650 | 25 | 20 | 0.91 / 0.85 | 0.95 / 0.93 | 3.73 / 3.44 | 3.25 / 2.89 | 0.66 / 0.65 | 2.17 / 2.06 |
| **SpeechTokenizer** | 4000 | 50 | 8 | 0.85 / 0.69 | 0.92 / 0.85 | 3.05 / 2.20 | 2.60 / 1.87 | -- / -- | -- / -- |
| **Mimi** | 4400 | 12.5 | 32 | 0.94 / 0.83 | 0.96 / 0.94 | 3.80 / 3.31 | 3.43 / 2.78 | 1.02 / 0.98 | 2.34 / 2.21 |
| **Encodec** | 4500 | 75 | 6 | 0.86 / 0.75 | 0.92 / 0.91 | 2.91 / 2.63 | 2.46 / 2.15 | 0.91 / 0.84 | 2.33 / 2.17 |
| **DAC** | 6000 | 75 | 8 | 0.89 / 0.84 | 0.95 / 0.94 | 3.75 / 3.57 | 3.41 / 3.20 | **0.65** / **0.63** | **1.97** / 1.87 |
| **MOSS-Audio-Tokenizer-v2 (Ours)** | 3000 | 12.5 | 24 | 0.96 / 0.92 | **0.97** / 0.95 | 3.94 / 3.64 | 3.66 / 3.28 | 0.75 / 0.71 | 2.04 / 1.87 |
| **MOSS-Audio-Tokenizer-v2 (Ours)** | 4000 | 12.5 | 32 | **0.97** / **0.93** | **0.97** / **0.96** | **3.98** / **3.72** | **3.75** / **3.39** | 0.73 / 0.69 | 2.02 / **1.84** |

### LibriSpeech Speech Metrics (MOSS-Audio-Tokenizer-v2 vs. Open-source Tokenizers)

The plots below compare our MOSS-Audio-Tokenizer-v2 model with other open-source speech tokenizers on the LibriSpeech dataset, evaluated with SIM, STOI, PESQ-NB, and PESQ-WB (higher is better).
We control the bps of the same model by adjusting the number of RVQ codebooks used during inference.

<table>
  <tr>
    <td align="center"><br><img src="images/metric.png" width="100%"></td>
  </tr>
</table>


## Citation
If you use this code or result in your paper, please cite our work as:
```tex
@misc{gong2026mossaudiotokenizerscaling,
  title={MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models},
  author={Yitian Gong and Kuangwei Chen and Zhaoye Fei and Xiaogui Yang and Ke Chen and Yang Wang and Kexin Huang and Mingshu Chen and Ruixiao Li and Qingyuan Cheng and Shimin Li and Xipeng Qiu},
  year={2026},
  eprint={2602.10934},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2602.10934}
}
```

## License
MOSS-Audio-Tokenizer-v2 is released under the Apache 2.0 license. See `LICENSE` for the full license text.
