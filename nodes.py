"""ComfyUI node definitions for MOSS-TTS Local Transformer."""

from __future__ import annotations

from .loader import (
    ATTENTION_OPTIONS,
    DEFAULT_MODEL,
    DTYPE_OPTIONS,
    get_model_choices,
    load_mosstts_bundle,
)
from .runtime import (
    clone_mosstts_audio,
    continue_mosstts_audio,
    generate_mosstts_audio,
)
from .whisper import MossTTSWhisperTranscribe

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


MODEL_TYPE = "MOSSTTS_MODEL"

LANGUAGES = [
    "auto",
    "Chinese",
    "Cantonese",
    "English",
    "Arabic",
    "Czech",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Italian",
    "Japanese",
    "Korean",
    "Macedonian",
    "Malay",
    "Persian (Farsi)",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Thai",
    "Turkish",
    "Vietnamese",
]


def _progress(total: int):
    total = max(1, int(total))
    pbar = ProgressBar(total) if ProgressBar is not None else None

    def update(current: int, total: int) -> None:
        if pbar is not None:
            pbar.update_absolute(max(0, int(current)), max(1, int(total)))

    return update


def _text_input(default: str) -> tuple:
    return (
        "STRING",
        {
            "multiline": True,
            "default": default,
            "tooltip": "Text to synthesize. Inline [pause 0.5s], Pinyin, and IPA are passed through to MOSS-TTS.",
        },
    )


def _conditioning_controls() -> dict:
    return {
        "language": (
            LANGUAGES,
            {
                "default": "auto",
                "tooltip": "Language hint. v1.5 performs best when the language is specified.",
            },
        ),
        "instruction": (
            "STRING",
            {
                "multiline": True,
                "default": "",
                "tooltip": "Optional free-form style or delivery instruction.",
            },
        ),
        "duration_tokens": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 45000,
                "step": 1,
                "tooltip": "Optional duration target in audio tokens. 0 omits the field. MOSS runs around 12.5 tokens per second.",
            },
        ),
        "quality": (
            "STRING",
            {
                "multiline": False,
                "default": "",
                "tooltip": "Optional quality hint field exposed by the MOSS user-message schema.",
            },
        ),
        "sound_event": (
            "STRING",
            {
                "multiline": False,
                "default": "",
                "tooltip": "Optional sound-event hint field exposed by the MOSS user-message schema.",
            },
        ),
        "ambient_sound": (
            "STRING",
            {
                "multiline": False,
                "default": "",
                "tooltip": "Optional ambient-sound hint field exposed by the MOSS user-message schema.",
            },
        ),
    }


def _sampling_controls() -> dict:
    return {
        "max_new_tokens": (
            "INT",
            {
                "default": 4096,
                "min": 1,
                "max": 45000,
                "step": 1,
                "tooltip": "Generation budget in audio frames. At 12.5 frames per second, 4096 is roughly 5.5 minutes.",
            },
        ),
        "do_sample": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": "Use stochastic sampling. Disable for deterministic greedy decoding.",
            },
        ),
        "text_temperature": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "Sampling temperature for assistant text/audio-control tokens.",
            },
        ),
        "text_top_p": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Nucleus sampling for assistant text/audio-control tokens.",
            },
        ),
        "text_top_k": (
            "INT",
            {
                "default": 50,
                "min": 0,
                "max": 4096,
                "step": 1,
                "tooltip": "Top-K sampling for assistant text/audio-control tokens.",
            },
        ),
        "audio_temperature": (
            "FLOAT",
            {
                "default": 1.7,
                "min": 0.0,
                "max": 3.0,
                "step": 0.01,
                "tooltip": "Recommended v1.5 audio sampling temperature.",
            },
        ),
        "audio_top_p": (
            "FLOAT",
            {
                "default": 0.8,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Recommended v1.5 nucleus sampling cutoff for audio codebooks.",
            },
        ),
        "audio_top_k": (
            "INT",
            {
                "default": 25,
                "min": 0,
                "max": 1024,
                "step": 1,
                "tooltip": "Recommended v1.5 Top-K cutoff for audio codebooks.",
            },
        ),
        "audio_repetition_penalty": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 1.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "Penalty for repeated acoustic code patterns.",
            },
        ),
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 2**63 - 1,
                "tooltip": "0 leaves sampling unseeded. Positive values make identical settings repeatable.",
            },
        ),
    }


class MossTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "Cataloged OpenMOSS MOSS-TTS Local Transformer v1.5. Main weights are stored in ComfyUI/models/mosstts/moss-tts-local-transformer-v1.5/.",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "auto uses the dtype declared by the bundled model config. Manual options are bf16 and fp16.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "auto uses FlashAttention 2 when compatible, SDPA on CUDA fallback, and eager on CPU. Manual eager forces the main model's plain attention path; the codec uses SDPA.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download missing main model and required audio-tokenizer model files. Disable for offline operation.",
                    },
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("mosstts_model",)
    FUNCTION = "load_model"
    CATEGORY = "MOSS-TTS"
    DESCRIPTION = "Load MOSS-TTS Local Transformer v1.5 and its 48 kHz stereo audio tokenizer with ComfyUI/AIMDO memory tracking."

    def load_model(self, model: str, dtype: str, attention: str, download_if_missing: bool):
        bundle = load_mosstts_bundle(
            model_choice=model,
            dtype_name=dtype,
            attention=attention,
            download_if_missing=bool(download_if_missing),
        )
        return (bundle,)


class MossTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {"tooltip": "Loaded MOSS-TTS model bundle."}),
            "text": _text_input("Hello! This is MOSS-TTS Local Transformer v1.5 running inside ComfyUI."),
        }
        required.update(_conditioning_controls())
        required.update(_sampling_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "MOSS-TTS"
    DESCRIPTION = "Generate 48 kHz stereo speech without a voice reference."

    def generate(self, mosstts_model, text: str, language: str, instruction: str, duration_tokens: int, quality: str, sound_event: str, ambient_sound: str, max_new_tokens: int, do_sample: bool, text_temperature: float, text_top_p: float, text_top_k: int, audio_temperature: float, audio_top_p: float, audio_top_k: int, audio_repetition_penalty: float, seed: int):
        audio = generate_mosstts_audio(
            mosstts_model,
            text=text,
            language=language,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            seed=seed,
            progress_callback=_progress(max_new_tokens),
        )
        return (audio,)


class MossTTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {"tooltip": "Loaded MOSS-TTS model bundle."}),
            "reference_audio": (
                "AUDIO",
                {"tooltip": "Reference speaker audio. The node encodes it with MOSS-Audio-Tokenizer-v2 before generation."},
            ),
            "text": _text_input("This line will be spoken in the reference voice."),
        }
        required.update(_conditioning_controls())
        required.update(_sampling_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "MOSS-TTS"
    DESCRIPTION = "Zero-shot voice cloning from a ComfyUI AUDIO reference."

    def clone(self, mosstts_model, reference_audio: dict, text: str, language: str, instruction: str, duration_tokens: int, quality: str, sound_event: str, ambient_sound: str, max_new_tokens: int, do_sample: bool, text_temperature: float, text_top_p: float, text_top_k: int, audio_temperature: float, audio_top_p: float, audio_top_k: int, audio_repetition_penalty: float, seed: int):
        audio = clone_mosstts_audio(
            mosstts_model,
            text=text,
            reference_audio=reference_audio,
            language=language,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            seed=seed,
            progress_callback=_progress(max_new_tokens),
        )
        return (audio,)


class MossTTSContinueSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "mosstts_model": (MODEL_TYPE, {"tooltip": "Loaded MOSS-TTS model bundle."}),
            "prefix_audio": (
                "AUDIO",
                {"tooltip": "Prefix audio to continue from. Provide a matching transcript below."},
            ),
            "prefix_text": _text_input("This is the transcript of the prefix audio. "),
            "continuation_text": _text_input("This is the sentence ending or continuation to generate."),
            "return_full_audio": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "When enabled, concatenate the original prefix audio with the generated continuation.",
                },
            ),
        }
        required.update(_conditioning_controls())
        required.update(_sampling_controls())
        return {"required": required}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "continue_speech"
    CATEGORY = "MOSS-TTS"
    DESCRIPTION = "Continuation-based cloning for finishing a sentence or extending prefix audio."

    def continue_speech(self, mosstts_model, prefix_audio: dict, prefix_text: str, continuation_text: str, return_full_audio: bool, language: str, instruction: str, duration_tokens: int, quality: str, sound_event: str, ambient_sound: str, max_new_tokens: int, do_sample: bool, text_temperature: float, text_top_p: float, text_top_k: int, audio_temperature: float, audio_top_p: float, audio_top_k: int, audio_repetition_penalty: float, seed: int):
        audio = continue_mosstts_audio(
            mosstts_model,
            prefix_audio=prefix_audio,
            prefix_text=prefix_text,
            continuation_text=continuation_text,
            language=language,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            return_full_audio=bool(return_full_audio),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            seed=seed,
            progress_callback=_progress(max_new_tokens),
        )
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "MossTTSModelLoader": MossTTSModelLoader,
    "MossTTSGenerate": MossTTSGenerate,
    "MossTTSVoiceClone": MossTTSVoiceClone,
    "MossTTSContinueSpeech": MossTTSContinueSpeech,
    "MossTTSWhisperTranscribe": MossTTSWhisperTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossTTSModelLoader": "MOSS-TTS Load Model",
    "MossTTSGenerate": "MOSS-TTS Generate Speech",
    "MossTTSVoiceClone": "MOSS-TTS Voice Clone",
    "MossTTSContinueSpeech": "MOSS-TTS Continue Speech",
    "MossTTSWhisperTranscribe": "MOSS-TTS Whisper Transcribe",
}
