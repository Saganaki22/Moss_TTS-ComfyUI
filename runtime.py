"""Runtime helpers for MOSS-TTS generation nodes."""

from __future__ import annotations

import logging
from typing import Any

import torch


logger = logging.getLogger("Moss_TTS-ComfyUI")


def seed_everything(seed: int) -> None:
    if seed <= 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)


def comfy_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav.contiguous(), sample_rate


def tensor_to_comfy_audio(audio: torch.Tensor, sample_rate: int) -> dict:
    wav = audio.detach().float().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim == 2:
        wav = wav.unsqueeze(0)
    return {
        "waveform": wav.contiguous(),
        "sample_rate": int(sample_rate),
    }


def _field(value: str) -> str | None:
    value = str(value or "").strip()
    return value if value else None


def _language(value: str) -> str | None:
    value = str(value or "").strip()
    if not value or value == "auto":
        return None
    return value


def _duration_tokens(value: int) -> int | None:
    value = int(value)
    return value if value > 0 else None


def _build_user_kwargs(
    *,
    text: str,
    language: str,
    instruction: str,
    duration_tokens: int,
    quality: str,
    sound_event: str,
    ambient_sound: str,
    reference: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"text": text}
    if reference:
        kwargs["reference"] = reference
    language_value = _language(language)
    if language_value is not None:
        kwargs["language"] = language_value
    tokens_value = _duration_tokens(duration_tokens)
    if tokens_value is not None:
        kwargs["tokens"] = tokens_value
    for key, value in (
        ("instruction", instruction),
        ("quality", quality),
        ("sound_event", sound_event),
        ("ambient_sound", ambient_sound),
    ):
        normalized = _field(value)
        if normalized is not None:
            kwargs[key] = normalized
    return kwargs


def _encode_reference_audio(bundle, reference_audio: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    from .loader import resume_bundle_to_device

    resume_bundle_to_device(bundle)
    wav, sample_rate = comfy_audio_to_tensor(reference_audio)
    codes = bundle.processor.encode_audios_from_wav([wav], sample_rate)[0]
    target_sr = int(bundle.processor.model_config.sampling_rate)
    if sample_rate != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sample_rate, target_sr)
    return codes, wav, target_sr


def _run_model_generate(
    bundle,
    conversations: list[list[dict[str, Any]]],
    *,
    mode: str,
    max_new_tokens: int,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    seed: int,
    progress_callback=None,
) -> torch.Tensor:
    from .loader import resume_bundle_to_device

    resume_bundle_to_device(bundle)
    seed_everything(int(seed))

    batch = bundle.processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(bundle.device)
    attention_mask = batch["attention_mask"].to(bundle.device)
    progress_total = max(1, int(max_new_tokens))
    if progress_callback is not None:
        progress_callback(0, progress_total)
    with torch.inference_mode():
        outputs = bundle.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            text_temperature=float(text_temperature),
            text_top_p=float(text_top_p),
            text_top_k=int(text_top_k),
            audio_temperature=float(audio_temperature),
            audio_top_p=float(audio_top_p),
            audio_top_k=int(audio_top_k),
            audio_repetition_penalty=float(audio_repetition_penalty),
            progress_callback=progress_callback,
            show_progress=True,
            progress_description=f"MOSS-TTS {mode}",
        )
        messages = bundle.processor.decode(outputs, return_stereo=True)
    if progress_callback is not None:
        progress_callback(progress_total, progress_total)
    for message in messages:
        if message is not None and message.audio_codes_list:
            return message.audio_codes_list[0]
    raise RuntimeError("MOSS-TTS did not return audio.")


def generate_mosstts_audio(
    bundle,
    *,
    text: str,
    language: str,
    instruction: str,
    duration_tokens: int,
    quality: str,
    sound_event: str,
    ambient_sound: str,
    max_new_tokens: int,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    seed: int,
    progress_callback=None,
) -> dict:
    user = bundle.processor.build_user_message(
        **_build_user_kwargs(
            text=text,
            language=language,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
        )
    )
    audio = _run_model_generate(
        bundle,
        [[user]],
        mode="generation",
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
        progress_callback=progress_callback,
    )
    return tensor_to_comfy_audio(audio, bundle.processor.model_config.sampling_rate)


def clone_mosstts_audio(
    bundle,
    *,
    text: str,
    reference_audio: dict,
    language: str,
    instruction: str,
    duration_tokens: int,
    quality: str,
    sound_event: str,
    ambient_sound: str,
    max_new_tokens: int,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    seed: int,
    progress_callback=None,
) -> dict:
    reference_codes, _reference_wav, _sample_rate = _encode_reference_audio(bundle, reference_audio)
    user = bundle.processor.build_user_message(
        **_build_user_kwargs(
            text=text,
            language=language,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
            reference=[reference_codes],
        )
    )
    audio = _run_model_generate(
        bundle,
        [[user]],
        mode="generation",
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
        progress_callback=progress_callback,
    )
    return tensor_to_comfy_audio(audio, bundle.processor.model_config.sampling_rate)


def continue_mosstts_audio(
    bundle,
    *,
    prefix_audio: dict,
    prefix_text: str,
    continuation_text: str,
    language: str,
    instruction: str,
    duration_tokens: int,
    quality: str,
    sound_event: str,
    ambient_sound: str,
    return_full_audio: bool,
    max_new_tokens: int,
    do_sample: bool,
    text_temperature: float,
    text_top_p: float,
    text_top_k: int,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    seed: int,
    progress_callback=None,
) -> dict:
    prefix_codes, prefix_wav, sample_rate = _encode_reference_audio(bundle, prefix_audio)
    full_text = f"{prefix_text}{continuation_text}"
    user = bundle.processor.build_user_message(
        **_build_user_kwargs(
            text=full_text,
            language=language,
            instruction=instruction,
            duration_tokens=duration_tokens,
            quality=quality,
            sound_event=sound_event,
            ambient_sound=ambient_sound,
        )
    )
    assistant = bundle.processor.build_assistant_message(audio_codes_list=[prefix_codes])
    generated = _run_model_generate(
        bundle,
        [[user, assistant]],
        mode="continuation",
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
        progress_callback=progress_callback,
    )
    if return_full_audio:
        generated = torch.cat([prefix_wav.cpu(), generated.cpu()], dim=-1)
    return tensor_to_comfy_audio(generated, sample_rate)
