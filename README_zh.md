# Moss_TTS-ComfyUI

**[English](./README.md)** | **简体中文**

**版本：v0.1.1**

[![Version](https://img.shields.io/badge/version-v0.1.1-blue)](https://github.com/Saganaki22/Moss_TTS-ComfyUI/releases/tag/v0.1.1)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-2d7dd2)](https://github.com/comfyanonymous/ComfyUI)
[![ComfyUI Manager](https://img.shields.io/badge/ComfyUI_Manager-compatible-6f42c1)](https://github.com/ltdrdata/ComfyUI-Manager)
[![MOSS-TTS v1.5](https://img.shields.io/badge/Hugging_Face-MOSS--TTS--Local--Transformer--v1.5-ffd21e)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
[![Audio Tokenizer](https://img.shields.io/badge/Hugging_Face-MOSS--Audio--Tokenizer--v2-ffd21e)](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)
[![Upstream](https://img.shields.io/badge/Upstream-OpenMOSS%2FMOSS--TTS-111111)](https://github.com/OpenMOSS/MOSS-TTS)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)

这是用于 [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) 的 ComfyUI 自定义节点。模型支持 48 kHz 立体声输出、普通文本生成、零样本音色克隆、语音续写、时长控制、发音标记、多语言合成和中英等语言混合。

本节点会把 Hugging Face 仓库中的小型 remote-code、tokenizer 和 config 文件放在自定义节点的 `assets/` 目录中；大型模型权重放在 `ComfyUI/models/mosstts/`。只有启用 `download_if_missing` 时才会下载缺失模型文件。

> 请只克隆你拥有或已获得明确授权的声音。不要将本项目用于冒充、欺诈、骚扰、规避同意或其他有害用途。

## 主要特性

- 标准 ComfyUI `AUDIO` 输入，48 kHz 立体声 `AUDIO` 输出。
- 无参考音频的普通 TTS。
- 使用 ComfyUI 音频参考进行零样本音色克隆。
- 语音续写：用前缀音频和前缀转写继续生成后续句子。
- 可选 Whisper 转写节点，用于生成续写所需的 `prefix_text`。
- 支持 v1.5 模型卡列出的 31 种语言标签。
- 支持 `[pause 0.5s]` 等显式停顿标记，以及 Pinyin、IPA 发音控制。
- `auto`、`bf16`、`fp16` 精度选项。
- `auto`、`sdpa`、`flash_attention`、`eager` 注意力后端选项。
- AIMDO DynamicVRAM 集成：当 ComfyUI 已启用 AIMDO 时，主模型、MOSS-Audio-Tokenizer 和可选 Whisper ASR 模型都会使用原生动态 patcher。
- 原生 ComfyUI 生成进度条，以及终端 `tqdm` 帧进度条。
- 不提供单独 unload 节点。切换模型、精度、注意力或 tokenizer 文件时，加载器会自动卸载旧 bundle。

## 安装

### ComfyUI Manager

当本节点进入 ComfyUI Manager 列表后，可以搜索 **MOSS-TTS** 并直接安装。

也可以现在通过 Git URL 安装：

1. 打开 **Manager**。
2. 选择 **Install via Git URL**。
3. 输入：

```text
https://github.com/Saganaki22/Moss_TTS-ComfyUI
```

4. 重启 ComfyUI。

### 手动安装

在 `ComfyUI/custom_nodes` 目录下运行：

```powershell
git clone https://github.com/Saganaki22/Moss_TTS-ComfyUI.git
..\venv\Scripts\python.exe Moss_TTS-ComfyUI\install.py
```

Linux 或普通 venv 环境：

```bash
git clone https://github.com/Saganaki22/Moss_TTS-ComfyUI.git
../venv/bin/python Moss_TTS-ComfyUI/install.py
```

如果 ComfyUI 使用 `uv` 管理：

```bash
uv run python Moss_TTS-ComfyUI/install.py
```

`install.py` 只会安装缺失的轻量依赖，例如 `accelerate`、`huggingface-hub`、`numpy`、`safetensors` 和 `tqdm`。它不会替换 ComfyUI 的 `torch`、`torchaudio`、`torchvision` 或 `transformers`。

安装或更新后请重启 ComfyUI。

## 示例工作流

示例工作流文件和预览图放在：

```text
ComfyUI/custom_nodes/Moss_TTS-ComfyUI/example_workflows/
```

## 模型文件结构

加载器使用以下目录结构：

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

模型来源：

| 组件 | 来源 | 本地目录 |
| --- | --- | --- |
| MOSS-TTS Local Transformer v1.5 | [OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | `ComfyUI/models/mosstts/moss-tts-local-transformer-v1.5/model.safetensors` |
| MOSS-Audio-Tokenizer-v2 | [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) | `ComfyUI/models/mosstts/moss-audio-tokenizer-v2/` |
| 可选 Whisper ASR | [OpenAI Whisper models](https://huggingface.co/openai) | `ComfyUI/models/audio_encoders/` |

主 MOSS 仓库不会完整 snapshot 到 `models/`。主模型只下载 `model.safetensors`。小型 remote-code、config、tokenizer 文件随节点放在 `assets/` 中，加载时会硬链接或复制到权重旁边，让 Transformers 从正常本地目录加载。

## 节点

| 节点 | 用途 |
| --- | --- |
| `MOSS-TTS Load Model` | 加载 MOSS 主模型和立体声音频 tokenizer。提供模型目录、精度、注意力后端和 `download_if_missing`。 |
| `MOSS-TTS Generate Speech` | 无参考 TTS。支持语言标签、风格字段、时长 token、停顿标记、Pinyin/IPA 和采样参数。 |
| `MOSS-TTS Voice Clone` | 从 ComfyUI `AUDIO` 参考音频进行零样本音色克隆。本地 v1.5 路径不需要参考文本。 |
| `MOSS-TTS Continue Speech` | 继续前缀音频。需要前缀音频和它的转写文本，然后生成续写文本。 |
| `MOSS-TTS Whisper Transcribe` | 可选 ASR 辅助节点，用于续写工作流中的 `prefix_text`。 |

<details>
<summary><strong>节点参数</strong></summary>

### MOSS-TTS Load Model

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `MOSS-TTS Local Transformer v1.5 BF16 - OpenMOSS-Team` | v1.5 checkpoint 目录项。 |
| `dtype` | `auto` | `auto` 读取模型 config，目前为 BF16。手动选项：`bf16`、`fp16`。 |
| `attention` | `auto` | `auto` 会在可用时使用 FlashAttention 2，在 CUDA 上回退到 SDPA，在 CPU 上使用 eager。手动选项：`sdpa`、`flash_attention`、`eager`。 |
| `download_if_missing` | `true` | 只在启用时下载缺失的主模型权重和音频 tokenizer 权重。 |

### 生成、克隆和续写

| 参数 | 说明 |
| --- | --- |
| `text` / `continuation_text` | 目标文本。`[pause X.Ys]`、Pinyin、IPA 和混合语言文本会原样传给 MOSS。 |
| `language` | 可选语言标签。`auto` 表示不传语言字段。 |
| `instruction` | 可选风格或朗读方式指令。 |
| `duration_tokens` | 可选时长目标。`0` 表示不传。MOSS 约为每秒 12.5 个声学帧。 |
| `quality`、`sound_event`、`ambient_sound` | MOSS user-message schema 支持的额外字段。 |
| `max_new_tokens` | 生成预算。长文本可适当增大。 |
| `do_sample` | 是否启用随机采样。 |
| `text_temperature`、`text_top_p`、`text_top_k` | assistant/control token 的采样参数。 |
| `audio_temperature`、`audio_top_p`、`audio_top_k`、`audio_repetition_penalty` | 声学 codebook 的采样参数。默认值遵循 v1.5 模型卡建议。 |
| `seed` | `0` 表示不固定随机种子；正数可复现采样。 |
| `reference_audio` | Voice Clone 专用。由 MOSS-Audio-Tokenizer-v2 编码的 ComfyUI `AUDIO` 参考音频。 |
| `prefix_audio` | Continue Speech 专用。要继续的前缀音频。 |
| `prefix_text` | Continue Speech 专用。前缀音频的转写文本，续写必需。 |
| `return_full_audio` | Continue Speech 专用。启用后返回前缀加续写音频。 |

### Whisper Transcribe

| 参数 | 说明 |
| --- | --- |
| `audio` | 需要转写的输入音频。 |
| `model` | 存放在 `ComfyUI/models/audio_encoders/` 下的 Whisper 模型目录项。 |
| `dtype` | `auto`、`bf16` 或 `fp32`。 |
| `language` | 可选 Whisper 语言提示。 |
| `task` | `transcribe` 或 `translate`。 |
| `chunk_length_s` | 长音频分块大小。 |
| `download_if_missing` | 缺失时下载所选 Whisper 模型。 |

</details>

<details>
<summary><strong>支持语言</strong></summary>

MOSS-TTS Local Transformer v1.5 模型卡列出 31 种支持语言：

| 语言 | 代码 | 语言 | 代码 | 语言 | 代码 |
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
<summary><strong>高级说明</strong></summary>

- 本节点对本地 `assets/` 中的 remote-code 文件使用 `trust_remote_code=True`。
- v1.5 codec 是立体声。`processor.decode(...)` 返回 `[channels, samples]`，运行时会直接保留该张量形状。
- 音频编码和解码使用 MOSS-Audio-Tokenizer-v2。
- v1.5 config 设置 `sampling_rate=48000` 和 `n_vq=12`。
- 除非调试，否则建议保持 `attention=auto`。FlashAttention 2 是可选项；CUDA 上的常规回退是 SDPA。
- 当 ComfyUI 启用 AIMDO DynamicVRAM 时，加载器会强制主模型、音频 tokenizer 和可选 Whisper ASR 模型使用原生动态 patcher。Whisper 会先在 CPU 暂存，再把 CUDA 加载交给 ComfyUI/AIMDO。
- 生成时会同时更新 ComfyUI 进度条和终端 `tqdm` 进度条，进度基于生成的声学帧并以 `max_new_tokens` 为上限。
- 如果 `download_if_missing=false`，上方目录结构中的模型文件必须已经存在。

</details>

<details>
<summary><strong>故障排查</strong></summary>

### Hugging Face 下载到了错误镜像

有些自定义节点会全局设置 `HF_ENDPOINT`。本节点不会强制 Hugging Face endpoint。如果下载跳到不想要的镜像，请检查 ComfyUI 进程环境变量以及其他会修改 `os.environ["HF_ENDPOINT"]` 的自定义节点。

### `Unexpected keyword argument local_files_only`

部分 MOSS processor remote-code 会把 `local_files_only` 传入 `ProcessorMixin`。本节点不会把该参数传给 `AutoProcessor.from_pretrained(...)`，但仍然从本地目录加载。

### `property 'device' ... has no setter`

MOSS remote-code 模型暴露只读 `device` 属性，而 ComfyUI patcher 在加载和卸载时会写入 `model.device`。本节点在注册模型前会安装一个实例级可写兼容属性。

</details>

## 项目链接

- 仓库：[Saganaki22/Moss_TTS-ComfyUI](https://github.com/Saganaki22/Moss_TTS-ComfyUI)
- 上游项目：[OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- 模型卡：[OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)
- 音频 tokenizer：[OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)
- 技术报告：[MOSS-TTS Technical Report](https://arxiv.org/abs/2603.18090)

## 引用

如果你使用该模型，请引用 MOSS-TTS 技术报告：

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
