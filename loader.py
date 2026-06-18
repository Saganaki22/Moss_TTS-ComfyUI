"""MOSS-TTS model discovery, downloads, loading, and ComfyUI/AIMDO registration."""

from __future__ import annotations

import gc
import filecmp
import importlib
import importlib.util
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


logger = logging.getLogger("Moss_TTS-ComfyUI")

MODEL_FOLDER_NAME = "mosstts"
MAIN_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
MAIN_SUBDIR = "moss-tts-local-transformer-v1.5"
CODEC_SUBDIR = "moss-audio-tokenizer-v2"
MAIN_WEIGHT_FILE = "model.safetensors"
CODEC_MODEL_FILES = [
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]
DTYPE_OPTIONS = ["auto", "bf16", "fp16"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention", "eager"]
DEFAULT_MODEL = "MOSS-TTS Local Transformer v1.5 BF16 - OpenMOSS-Team (auto-download)"

_ACTIVE_BUNDLE: "MossTTSBundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class ModelPreset:
    repo_id: str
    subdir: str
    filename: str


MODEL_CATALOG = {
    DEFAULT_MODEL: ModelPreset(
        repo_id=MAIN_REPO_ID,
        subdir=MAIN_SUBDIR,
        filename=MAIN_WEIGHT_FILE,
    ),
}


@dataclass
class MossTTSBundle:
    model: Any
    processor: Any
    model_dir: Path
    model_path: Path
    codec_dir: Path
    device: torch.device
    torch_dtype: torch.dtype
    dtype_name: str
    attention: str
    attn_implementation: str
    patchers: list[Any] = field(default_factory=list)


def node_dir() -> Path:
    return Path(__file__).resolve().parent


def assets_dir() -> Path:
    return node_dir() / "assets"


def main_assets_dir() -> Path:
    return assets_dir() / MAIN_SUBDIR


def codec_assets_dir() -> Path:
    return assets_dir() / CODEC_SUBDIR


def model_root() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    except Exception:
        base = Path(__file__).resolve().parents[2] / "models" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        extensions = {".safetensors", ".json", ".py", ".txt", ".jinja"}
        if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(model_root()))
            return

        paths, known_extensions = folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME]
        normalized = list(paths)
        target = str(model_root())
        if target not in normalized:
            normalized.append(target)
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = (
            normalized,
            set(known_extensions) | extensions,
        )
    except Exception as exc:
        logger.debug("Could not register mosstts model folder: %s", exc)


def get_model_choices() -> list[str]:
    register_model_folder()
    return list(MODEL_CATALOG)


def _require_asset_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing bundled {label} assets at {path}. Reinstall or update this custom node."
        )
    required = ["config.json", "__init__.py"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Bundled {label} assets are incomplete at {path}: {missing}."
        )


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except Exception:
        return False


def _link_or_copy(src: Path, dst: Path, *, allow_large_copy: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if _same_file(src, dst):
            return
        if src.name != MAIN_WEIGHT_FILE and filecmp.cmp(src, dst, shallow=False):
            return
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    if not allow_large_copy:
        raise RuntimeError(
            f"Could not hardlink or symlink {src} to {dst}. "
            "Move the model file directly into the target folder."
        )
    shutil.copy2(src, dst)


def _link_asset_tree(asset_path: Path, target_path: Path) -> None:
    for src in asset_path.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(asset_path)
        _link_or_copy(src, target_path / rel)


def _hf_download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s/%s to %s", repo_id, filename, local_dir)
    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(local_dir),
    }
    return Path(hf_hub_download(**kwargs))


def resolve_main_model_dir(model_choice: str, download_if_missing: bool) -> tuple[Path, Path]:
    preset = MODEL_CATALOG.get(model_choice)
    if preset is None:
        raise ValueError(f"Unknown MOSS-TTS model catalog entry: {model_choice}")

    _require_asset_dir(main_assets_dir(), "MOSS-TTS")
    runtime_dir = model_root() / preset.subdir
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _link_asset_tree(main_assets_dir(), runtime_dir)

    model_path = runtime_dir / preset.filename
    root_model_path = model_root() / preset.filename
    if not model_path.is_file() and root_model_path.is_file():
        _link_or_copy(root_model_path, model_path, allow_large_copy=False)

    if not model_path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Missing MOSS-TTS model file at {model_path}. "
                "Enable download_if_missing or place model.safetensors there."
            )
        _hf_download_file(preset.repo_id, preset.filename, runtime_dir)

    return runtime_dir, model_path


def resolve_codec_dir(download_if_missing: bool) -> Path:
    _require_asset_dir(codec_assets_dir(), "MOSS-Audio-Tokenizer-v2")
    codec_dir = model_root() / CODEC_SUBDIR
    codec_dir.mkdir(parents=True, exist_ok=True)
    _link_asset_tree(codec_assets_dir(), codec_dir)

    missing = [filename for filename in CODEC_MODEL_FILES if not (codec_dir / filename).is_file()]
    if missing:
        if not download_if_missing:
            raise FileNotFoundError(
                f"Missing MOSS audio tokenizer model files in {codec_dir}: {missing}. "
                "Enable download_if_missing or place the tokenizer shards there."
            )
        for filename in missing:
            _hf_download_file(CODEC_REPO_ID, filename, codec_dir)
    return codec_dir


def _config_dtype_name() -> str:
    config_path = main_assets_dir() / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return "bfloat16"
    for key in ("torch_dtype", "dtype"):
        value = data.get(key)
        if value:
            return str(value)
    for section in ("qwen3_config", "language_config"):
        value = (data.get(section) or {}).get("dtype")
        if value:
            return str(value)
    return "bfloat16"


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower().replace("torch.", "")
    if normalized in {"auto", "bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported MOSS-TTS dtype: {dtype_name}")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        dtype = _dtype_from_name(_config_dtype_name())
        logger.info("MOSS-TTS dtype auto resolved to %s from bundled model config.", dtype)
    else:
        dtype = _dtype_from_name(dtype_name)
    if device.type == "cpu" and dtype is torch.float16:
        logger.warning("FP16 on CPU is poorly supported; using FP32 instead.")
        return torch.float32
    return dtype


def resolve_device() -> torch.device:
    try:
        import comfy.model_management as mm

        return normalize_device(torch.device(mm.get_torch_device()))
    except Exception:
        return normalize_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def normalize_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(f"cuda:{index}")
    return device


def _flash_attention_available(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type != "cuda":
        return False
    if dtype not in {torch.float16, torch.bfloat16}:
        return False
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
        return major >= 8
    except Exception:
        return True


def resolve_attention(attention: str, device: torch.device, dtype: torch.dtype) -> tuple[str, str]:
    if attention == "auto":
        if _flash_attention_available(device, dtype):
            logger.info("MOSS-TTS attention auto resolved to flash_attention.")
            return "flash_attention", "flash_attention_2"
        if device.type == "cuda":
            logger.info("MOSS-TTS attention auto resolved to sdpa.")
            return "sdpa", "sdpa"
        logger.info("MOSS-TTS attention auto resolved to eager.")
        return "eager", "eager"
    if attention == "sdpa":
        return "sdpa", "sdpa"
    if attention == "flash_attention":
        if not _flash_attention_available(device, dtype):
            raise RuntimeError(
                "flash_attention requires CUDA, BF16/FP16, FlashAttention 2, and a compatible GPU."
            )
        return "flash_attention", "flash_attention_2"
    if attention == "eager":
        return "eager", "eager"
    raise ValueError(f"Unsupported attention mode: {attention}")


def _configure_torch_attention(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


try:
    import comfy.model_patcher as _model_patcher

    _ComfyCorePatcher = _model_patcher.CoreModelPatcher
except Exception:
    _ComfyCorePatcher = None


def dynamic_vram_active(device: torch.device) -> bool:
    if torch.device(device).type == "cpu":
        return False
    try:
        import comfy.memory_management

        if not bool(comfy.memory_management.aimdo_enabled):
            return False
        try:
            import comfy_aimdo.control
            import comfy_aimdo.host_buffer
            import comfy_aimdo.model_vbar

            return (
                comfy_aimdo.control.lib is not None
                and comfy_aimdo.host_buffer.lib is not None
                and comfy_aimdo.model_vbar.lib is not None
            )
        except Exception:
            return False
    except Exception:
        return False


def _register_many_with_comfy(patchers: list[Any]) -> None:
    patchers = [
        patcher
        for patcher in patchers
        if patcher is not None and patcher.load_device.type != "cpu"
    ]
    if not patchers:
        return
    try:
        import comfy.model_management as mm

        already_loaded = {
            id(loaded.model)
            for loaded in mm.current_loaded_models
            if loaded.model is not None
        }
        to_load = [p for p in patchers if id(p) not in already_loaded]
        if not to_load:
            return
        mm.load_models_gpu(to_load)
        for patcher in to_load:
            logger.info(
                "Loaded %s through ComfyUI%s memory management.",
                patcher.model.__class__.__name__,
                "/AIMDO" if patcher.is_dynamic() else "",
            )
    except Exception as exc:
        raise RuntimeError("Could not load MOSS-TTS through ComfyUI memory management.") from exc


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                try:
                    finalizer = getattr(loaded, "_patcher_finalizer", None)
                    if finalizer is not None:
                        finalizer.detach()
                    loaded._patcher_finalizer = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def _set_module_device_if_writable(module: torch.nn.Module, device: torch.device) -> None:
    try:
        module.device = torch.device(device)
    except (AttributeError, RuntimeError, TypeError):
        # Some HF remote-code models expose device as a read-only property.
        pass


def _ensure_writable_device_property(module: torch.nn.Module) -> None:
    cls = module.__class__
    prop = getattr(cls, "device", None)
    if not isinstance(prop, property) or prop.fset is not None:
        return
    if getattr(module, "_mosstts_writable_device_property", False):
        return

    def _get_device(self):
        runtime_device = self.__dict__.get("_mosstts_runtime_device")
        if runtime_device is not None:
            return runtime_device
        return prop.fget(self)

    def _set_device(self, value):
        self.__dict__["_mosstts_runtime_device"] = torch.device(value)

    writable_cls = type(
        f"{cls.__name__}ComfyWritableDevice",
        (cls,),
        {
            "device": property(_get_device, _set_device),
            "_mosstts_device_base_class": cls,
        },
    )
    module.__class__ = writable_cls
    module._mosstts_writable_device_property = True


def register_runtime_module(module: torch.nn.Module, device: torch.device, *, dynamic: bool | None = None) -> Any:
    device = normalize_device(device)
    _ensure_writable_device_property(module)
    if _ComfyCorePatcher is None or device.type == "cpu":
        module.to(device)
        return None

    import comfy.model_patcher as model_patcher

    use_dynamic = dynamic_vram_active(device) and dynamic is not False
    patcher_class = (
        model_patcher.ModelPatcherDynamic
        if use_dynamic
        else model_patcher.ModelPatcher
    )
    patcher = patcher_class(
        module,
        load_device=torch.device(device),
        offload_device=torch.device("cpu"),
    )
    module.model_loaded_weight_memory = 0
    if not patcher.is_dynamic() and hasattr(module, "device"):
        _set_module_device_if_writable(module, device)
    _register_many_with_comfy([patcher])
    logger.info(
        "Registered %s with ComfyUI%s memory management.",
        module.__class__.__name__,
        "/AIMDO" if patcher.is_dynamic() else "",
    )
    return patcher


def resume_bundle_to_device(bundle: MossTTSBundle) -> None:
    _register_many_with_comfy(bundle.patchers)


def resume_runtime_module(patcher: Any, device: torch.device | None = None) -> None:
    del device
    if patcher is not None:
        _register_many_with_comfy([patcher])


def unload_runtime_module(patcher: Any) -> None:
    if patcher is None:
        return
    _unregister_from_comfy(patcher)
    try:
        patcher.detach()
    except Exception:
        pass


def unload_mosstts_bundle(bundle: MossTTSBundle | None, reason: str = "model changed") -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading MOSS-TTS bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        unload_runtime_module(patcher)
    bundle.patchers.clear()
    for module in (bundle.model, getattr(bundle.processor, "audio_tokenizer", None)):
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            module.model_loaded_weight_memory = 0
            if hasattr(module, "dynamic_vbars"):
                module.dynamic_vbars.clear()
            if hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
            else:
                module.to("cpu")
        except Exception:
            pass
    bundle.model = None
    if getattr(bundle.processor, "audio_tokenizer", None) is not None:
        bundle.processor.audio_tokenizer = None
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def _dtype_to_codec_name(dtype: torch.dtype) -> str:
    if dtype in {torch.bfloat16, torch.float16}:
        return "bf16"
    return "fp32"


def _load_model_and_processor(
    model_dir: Path,
    model_path: Path,
    codec_dir: Path,
    dtype: torch.dtype,
    attn_implementation: str,
    weight_device: torch.device,
    progress_callback: Any = None,
):
    """Build both models on meta device, stream weights from safetensors,
    and assemble the processor — no trust_remote_code."""
    from . import native

    codec_attn_implementation = "sdpa" if attn_implementation == "eager" else attn_implementation
    codec_weight_dtype = _dtype_to_codec_name(dtype)
    codec_compute_dtype = "bf16" if dtype in {torch.bfloat16, torch.float16} else "fp32"

    tts_config = native.read_tts_config(model_dir)
    codec_config = native.read_codec_config(codec_dir)

    model = native.build_tts_model(tts_config, attn_implementation)
    codec = native.build_codec_model(
        codec_config,
        codec_weight_dtype=codec_weight_dtype,
        codec_compute_dtype=codec_compute_dtype,
        codec_attn_implementation=codec_attn_implementation,
    )

    native.load_tts_weights(model, model_path, dtype, weight_device, progress_callback)
    native.load_codec_weights(codec, codec_dir, dtype, weight_device, progress_callback)

    native.convert_modules_for_comfy(model)
    native.convert_modules_for_comfy(codec)
    native.set_runtime_dtype(model, dtype)
    native.set_runtime_dtype(codec, dtype)

    tokenizer = native.build_tokenizer(model_dir)
    processor = native.build_processor(tokenizer, codec, tts_config)
    return model, processor, codec


def _file_stat_key(paths: list[Path]) -> tuple[Any, ...]:
    values: list[Any] = []
    for path in paths:
        stat = path.stat()
        values.extend([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return tuple(values)


def load_mosstts_bundle(
    model_choice: str,
    dtype_name: str,
    attention: str,
    download_if_missing: bool,
) -> MossTTSBundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    register_model_folder()
    model_dir, model_path = resolve_main_model_dir(model_choice, download_if_missing)
    codec_dir = resolve_codec_dir(download_if_missing)
    device = resolve_device()
    dtype = resolve_dtype(dtype_name, device)
    runtime_attention, attn_implementation = resolve_attention(attention, device, dtype)
    _configure_torch_attention(device)

    codec_paths = [codec_dir / filename for filename in CODEC_MODEL_FILES]
    load_key = (
        *_file_stat_key([model_path, *codec_paths]),
        str(device),
        str(dtype),
        runtime_attention,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_mosstts_bundle(_ACTIVE_BUNDLE, reason="model, dtype, attention, or codec changed")

    logger.info(
        "Loading MOSS-TTS from %s with codec %s on %s dtype=%s attention=%s.",
        model_dir,
        codec_dir,
        device,
        dtype,
        runtime_attention,
    )

    weight_device = torch.device("cpu") if device.type != "cpu" else device
    model, processor, codec = _load_model_and_processor(
        model_dir=model_dir,
        model_path=model_path,
        codec_dir=codec_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
        weight_device=weight_device,
    )

    patchers: list[Any] = []
    use_dynamic = dynamic_vram_active(device)
    if use_dynamic:
        logger.info("AIMDO DynamicVRAM is active; using dynamic patchers for MOSS-TTS model and codec.")
    else:
        logger.info("AIMDO not active; using static ComfyUI memory management.")
    main_patcher = register_runtime_module(model, device, dynamic=use_dynamic)
    if main_patcher is not None:
        patchers.append(main_patcher)

    if isinstance(codec, torch.nn.Module):
        codec_patcher = register_runtime_module(codec, device, dynamic=use_dynamic)
        if codec_patcher is not None:
            patchers.append(codec_patcher)
    else:
        raise RuntimeError("MOSS-TTS processor did not expose an audio_tokenizer module.")

    bundle = MossTTSBundle(
        model=model,
        processor=processor,
        model_dir=model_dir,
        model_path=model_path,
        codec_dir=codec_dir,
        device=device,
        torch_dtype=dtype,
        dtype_name=dtype_name,
        attention=runtime_attention,
        attn_implementation=attn_implementation,
        patchers=patchers,
    )
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    install_comfy_unload_hook()
    return bundle


def install_comfy_unload_hook() -> None:
    """Patch ComfyUI's unload_all_models so the native unload button also
    hard-releases the MOSS-TTS bundle (VRAM + RAM + dynamic VBAR state)."""
    try:
        import comfy.model_management as mm
    except Exception:
        return

    if getattr(mm, "_mosstts_unload_hook_installed", False):
        return

    original_unload_all_models = mm.unload_all_models

    def unload_all_models_with_mosstts(*args, **kwargs):
        try:
            return original_unload_all_models(*args, **kwargs)
        finally:
            unload_mosstts_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_all_models")

    mm.unload_all_models = unload_all_models_with_mosstts

    original_unload_model_and_clones = getattr(mm, "unload_model_and_clones", None)
    if original_unload_model_and_clones is not None:
        def unload_model_and_clones_with_mosstts(model, *args, **kwargs):
            try:
                return original_unload_model_and_clones(model, *args, **kwargs)
            finally:
                if _ACTIVE_BUNDLE is not None and model is not None:
                    owned = list(_ACTIVE_BUNDLE.patchers) + [_ACTIVE_BUNDLE.model]
                    codec = getattr(_ACTIVE_BUNDLE.processor, "audio_tokenizer", None)
                    if codec is not None:
                        owned.append(codec)
                    if any(existing is model or existing is getattr(model, "model", None) for existing in owned if existing is not None):
                        unload_mosstts_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_model_and_clones")

        mm.unload_model_and_clones = unload_model_and_clones_with_mosstts

    mm._mosstts_unload_hook_installed = True
    logger.debug("Installed MOSS-TTS unload hook for ComfyUI native unload.")
