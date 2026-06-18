"""Native MOSS-TTS model construction and weight loading.

Imports the bundled asset scripts directly as packages, builds the model on
``torch.device("meta")``, and streams weights from safetensors into the
materialised parameters.  No ``trust_remote_code``, no HF
``transformers_modules`` cache — mirrors the Zonos2 native-loading pattern.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

try:
    from comfy.ops import cast_bias_weight, uncast_bias_weight
except Exception:
    def cast_bias_weight(model, input=None, **kwargs):
        return model.weight, getattr(model, "bias", None), None
    def uncast_bias_weight(model, weight, bias, offload_stream):
        return None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logger = logging.getLogger("Moss_TTS-ComfyUI")


# ---------------------------------------------------------------------------
# Asset package registration
#
# The bundled scripts under ``assets/`` use relative imports
# (``from .configuration_moss_tts import ...``), so they must be exposed as
# proper packages.  The on-disk directory names contain hyphens/dots which are
# not valid Python identifiers, so we register them under synthetic names and
# point ``__path__`` at the real directories.
# ---------------------------------------------------------------------------

def _assets_root() -> Path:
    return Path(__file__).resolve().parent / "assets"


_PACKAGE_MAP = {
    "_mosstts_native_tts": "moss-tts-local-transformer-v1.5",
    "_mosstts_native_codec": "moss-audio-tokenizer-v2",
}


def _register_asset_packages() -> None:
    for pkg_name, subdir in _PACKAGE_MAP.items():
        if pkg_name in sys.modules:
            continue
        pkg_dir = _assets_root() / subdir
        if not pkg_dir.is_dir():
            raise FileNotFoundError(
                f"Bundled asset directory missing: {pkg_dir}. "
                "Reinstall or update this custom node."
            )
        init_file = pkg_dir / "__init__.py"
        if not init_file.is_file():
            raise FileNotFoundError(
                f"Bundled asset package missing __init__.py: {init_file}."
            )
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(init_file),
            submodule_search_locations=[str(pkg_dir)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = module
        spec.loader.exec_module(module)


def tts_classes():
    """Return (Config, Model, Processor) classes for the main MOSS-TTS model."""
    _register_asset_packages()
    from _mosstts_native_tts.configuration_moss_tts import MossTTSLocalConfig
    from _mosstts_native_tts.modeling_moss_tts import MossTTSLocalModel
    from _mosstts_native_tts.processing_moss_tts import MossTTSLocalProcessor

    return MossTTSLocalConfig, MossTTSLocalModel, MossTTSLocalProcessor


def codec_classes():
    """Return (Config, Model) classes for MOSS-Audio-Tokenizer-v2."""
    _register_asset_packages()
    from _mosstts_native_codec.configuration_moss_audio_tokenizer import (
        MossAudioTokenizerConfig,
    )
    from _mosstts_native_codec.modeling_moss_audio_tokenizer import (
        MossAudioTokenizerModel,
    )

    return MossAudioTokenizerConfig, MossAudioTokenizerModel


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------

def read_tts_config(model_dir: Path):
    MossTTSLocalConfig, _, _ = tts_classes()
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    raw.pop("auto_map", None)
    raw.pop("architectures", None)
    return MossTTSLocalConfig(**raw)


def read_codec_config(codec_dir: Path):
    MossAudioTokenizerConfig, _ = codec_classes()
    raw = json.loads((codec_dir / "config.json").read_text(encoding="utf-8"))
    raw.pop("auto_map", None)
    raw.pop("architectures", None)
    raw.pop("model_type", None)
    return MossAudioTokenizerConfig(**raw)


# ---------------------------------------------------------------------------
# Model construction on meta device
# ---------------------------------------------------------------------------

def build_tts_model(config, attn_implementation: str):
    """Build MossTTSLocalModel under ``torch.device('meta')``."""
    _, MossTTSLocalModel, _ = tts_classes()

    config.attn_implementation = attn_implementation
    if hasattr(config, "qwen3_config"):
        config.qwen3_config._attn_implementation = attn_implementation
    config.local_transformer_attn_implementation = (
        "sdpa" if attn_implementation == "eager" else attn_implementation
    )

    with torch.device("meta"):
        model = MossTTSLocalModel(config)
    return model


def build_codec_model(
    config,
    codec_weight_dtype: str,
    codec_compute_dtype: str,
    codec_attn_implementation: str,
):
    """Build MossAudioTokenizerModel under ``torch.device('meta')``."""
    _, MossAudioTokenizerModel = codec_classes()

    config.attention_implementation = codec_attn_implementation
    config.codec_weight_dtype = codec_weight_dtype
    config.compute_dtype = codec_compute_dtype

    with torch.device("meta"):
        model = MossAudioTokenizerModel(config)
    return model


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _remap_weight_norm_keys(state_dict: dict[str, torch.Tensor], prefix: str = "") -> None:
    """Remap legacy weight_norm parameter names to the new parametrizations API.

    ``nn.utils.parametrizations.weight_norm`` stores originals as
    ``parametrizations.weight.original0`` / ``original1`` while older
    checkpoints (and ``nn.utils.weight_norm``) use ``weight_g`` / ``weight_v``.
    """
    replacements = (
        (".weight_g", ".parametrizations.weight.original0"),
        (".weight_v", ".parametrizations.weight.original1"),
    )
    for key in list(state_dict.keys()):
        if prefix and not key.startswith(prefix):
            continue
        new_key = key
        for src, dst in replacements:
            new_key = new_key.replace(src, dst)
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)


def _set_parameter(
    model: nn.Module,
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Assign ``tensor`` to parameter ``name`` on ``model`` with dtype/device."""
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    current = getattr(parent, leaf)
    if tuple(current.shape) != tuple(tensor.shape):
        tensor = tensor.reshape(current.shape)
    value = tensor
    if value.is_floating_point() and value.dtype != dtype:
        value = value.to(dtype=dtype)
    value = value.to(device=device).contiguous()
    setattr(parent, leaf, nn.Parameter(value, requires_grad=False))


def _ensure_no_meta_tensors(model: nn.Module, label: str) -> None:
    meta_params = [n for n, t in model.state_dict().items() if t.device.type == "meta"]
    meta_bufs = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params or meta_bufs:
        raise RuntimeError(
            f"{label} load left meta tensors: "
            f"params={meta_params[:5]}, buffers={meta_bufs[:5]}"
        )


def _materialize_non_persistent_buffers(model: nn.Module, device: torch.device) -> None:
    """Re-create non-persistent buffers left on meta after ``torch.device('meta')`` construction.

    Non-persistent buffers (registered with ``persistent=False``) are excluded
    from ``state_dict`` / ``load_state_dict``, so they remain on meta when the
    model is built under ``torch.device('meta')``.  Known buffer families are
    re-computed; anything unrecognized falls back to an empty tensor.
    """
    for module in model.modules():
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type != "meta":
                continue
            if name == "inv_freq" and hasattr(module, "_compute_inv_freq"):
                try:
                    new_buf = module._compute_inv_freq(device=device)
                    module.register_buffer(name, new_buf.to(device=device, dtype=buf.dtype), persistent=False)
                    continue
                except Exception:
                    pass
            new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
            module.register_buffer(name, new_buf, persistent=False)


def load_tts_weights(
    model: nn.Module,
    model_path: Path,
    dtype: torch.dtype,
    weight_device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Stream MOSS-TTS weights from a single safetensors file into ``model``.

    Each parameter is assigned individually with ``setattr`` (Zonos2 pattern)
    instead of ``load_state_dict(assign=True)`` — the latter can interact
    poorly with HF ``PreTrainedModel`` internals and tied-weight bookkeeping.
    """
    # Keys that are tied to another parameter; skip them — tie_weights()
    # re-aliases them after loading.
    tied_keys: set[str] = {"text_lm_head.weight"}
    tied_keys.update(
        f"audio_lm_heads.{i}.weight"
        for i in range(int(model.config.n_vq))
    )

    model_state_keys = set(model.state_dict().keys())

    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        checkpoint_keys = sorted(handle.keys())
        total = len(checkpoint_keys)
        pbar = (
            tqdm(total=total, desc="Loading MOSS-TTS weights", unit="tensor",
                 dynamic_ncols=True, leave=True)
            if tqdm is not None else None
        )
        try:
            for index, name in enumerate(checkpoint_keys, start=1):
                tensor = handle.get_tensor(name)

                # Skip tied / derived keys — they'll be handled by tie_weights()
                # and initialize_local_text_lm_head_from_text_lm_head().
                if name in tied_keys:
                    if pbar is not None:
                        pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(index, total)
                    continue

                # Skip keys that don't exist in the model (e.g. metadata).
                if name not in model_state_keys:
                    logger.debug("Skipping unexpected checkpoint key: %s", name)
                    if pbar is not None:
                        pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(index, total)
                    continue

                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                tensor = tensor.to(device=weight_device).contiguous()
                _set_parameter(model, name, tensor, dtype, weight_device)

                if pbar is not None:
                    pbar.update(1)
                if progress_callback is not None:
                    progress_callback(index, total)
        finally:
            if pbar is not None:
                pbar.close()

    # Verify every non-tied model key got a real tensor.
    still_meta = [
        n for n, t in model.state_dict().items()
        if t.device.type == "meta" and n not in tied_keys
    ]
    if still_meta:
        raise RuntimeError(
            f"MOSS-TTS checkpoint did not cover model keys: {still_meta[:10]}"
        )

    # Re-establish tied weights (text_lm_head ↔ embed_tokens,
    # audio_lm_heads[i] ↔ audio_embeddings[i]).  Do NOT call
    # initialize_local_text_lm_head_from_text_lm_head() here — the
    # checkpoint already contains the trained local_text_lm_head weights.
    model.tie_weights()

    _materialize_non_persistent_buffers(model, weight_device)
    _ensure_no_meta_tensors(model, "MOSS-TTS")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info("Loaded MOSS-TTS checkpoint tensors from %s.", model_path)


def load_codec_weights(
    model: nn.Module,
    codec_dir: Path,
    dtype: torch.dtype,
    weight_device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Stream MOSS-Audio-Tokenizer-v2 weights from sharded safetensors."""
    index_path = codec_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing codec safetensors index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(str(shard), []).append(key)

    # The quantizer always runs in fp32 (codebook numerics); encoder/decoder
    # use the runtime dtype requested by the caller.
    def _is_quantizer_key(key: str) -> bool:
        return key.startswith(("quantizer.input_proj", "quantizer.output_proj", "quantizer.quantizers"))

    names = sorted(weight_map.keys())
    total = len(names)
    pbar = (
        tqdm(total=total, desc="Loading MOSS-Codec weights", unit="tensor",
             dynamic_ncols=True, leave=True)
        if tqdm is not None else None
    )

    state_dict: dict[str, torch.Tensor] = {}
    loaded = 0
    try:
        for shard_name, shard_keys in sorted(shards.items()):
            shard_path = codec_dir / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    if key not in handle.keys():
                        continue
                    tensor = handle.get_tensor(key)
                    target_dtype = torch.float32 if _is_quantizer_key(key) else dtype
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype=target_dtype)
                    state_dict[key] = tensor.to(device=weight_device).contiguous()
                    loaded += 1
                    if pbar is not None:
                        pbar.update(1)
                    if progress_callback is not None:
                        progress_callback(loaded, total)
    finally:
        if pbar is not None:
            pbar.close()

    # Codec conv1d layers use weight_norm; legacy checkpoints store
    # weight_g/weight_v while the model expects parametrizations.weight.originalN.
    _remap_weight_norm_keys(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected_real = [k for k in incompatible.unexpected_keys if k]
    missing_real = [k for k in incompatible.missing_keys if k]
    if unexpected_real:
        raise RuntimeError(
            f"Unexpected MOSS-Codec checkpoint keys: {unexpected_real[:10]}"
        )
    if missing_real:
        raise RuntimeError(
            f"Missing MOSS-Codec checkpoint keys: {missing_real[:10]}"
        )

    _materialize_non_persistent_buffers(model, weight_device)
    _ensure_no_meta_tensors(model, "MOSS-Audio-Tokenizer")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    logger.info("Loaded %d MOSS-Codec checkpoint tensors.", total)


# ---------------------------------------------------------------------------
# Tokenizer + Processor
# ---------------------------------------------------------------------------

def build_tokenizer(model_dir: Path):
    """Load the Qwen3-based tokenizer from standard tokenizer files."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


def build_processor(tokenizer, codec, config):
    """Construct MossTTSLocalProcessor directly, no AutoProcessor."""
    _, _, MossTTSLocalProcessor = tts_classes()
    return MossTTSLocalProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=codec,
        model_config=config,
    )


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def estimate_module_bytes(module: nn.Module) -> int:
    """Total byte size of all unique parameters and buffers."""
    seen: set[int] = set()
    total = 0
    for value in list(module.parameters()) + list(module.buffers()):
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        total += value.numel() * value.element_size()
    return total


def set_runtime_dtype(module: nn.Module, dtype: torch.dtype) -> None:
    """Tag every floating-point parameter/buffer with a ComfyUI dtype hint.

    ComfyUI's lowvram / AIMDO machinery reads ``<name>_comfy_model_dtype`` to
    decide the compute precision for offloaded weights.
    """
    for module_name, sub in module.named_modules():
        target_dtype = torch.float32 if module_name == "quantizer" or module_name.startswith("quantizer.") else dtype
        for name, value in sub.named_parameters(recurse=False):
            if value.is_floating_point():
                setattr(sub, f"{name}_comfy_model_dtype", target_dtype)
        for name, value in sub.named_buffers(recurse=False):
            if value.is_floating_point():
                setattr(sub, f"{name}_comfy_model_dtype", target_dtype)


# ---------------------------------------------------------------------------
# Comfy-castable module conversion
#
# ModelPatcherDynamic.load() only creates VBAR page allocations for modules
# that have the ``comfy_cast_weights`` attribute (comfy/model_patcher.py:1862).
# Plain nn.Linear / nn.Embedding lack it, so all their weights go through the
# "force pre-loaded" path — straight to VRAM with no VBAR residency tracking,
# which is why the MemoryVisualization heatmap stays grey.
#
# We patch __class__ in-place after loading so the instances keep their
# parameters/state but gain the comfy_cast_weights attribute and a forward
# that calls cast_bias_weight / uncast_bias_weight.
# ---------------------------------------------------------------------------

class _ComfyLinear(nn.Linear):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, x):
        if not hasattr(self, "_v") and self.weight.device == x.device:
            return F.linear(x, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, x, offloadable=True)
        try:
            return F.linear(x, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyEmbedding(nn.Embedding):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []
    bias = None

    def _weight_dtype(self):
        return getattr(self, "weight_comfy_model_dtype", None) or self.weight.dtype

    def forward(self, input):
        if not hasattr(self, "_v") and self.weight.device == input.device:
            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        weight, _bias, stream = cast_bias_weight(
            self,
            dtype=self._weight_dtype(),
            device=input.device,
            offloadable=True,
        )
        try:
            return F.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        finally:
            uncast_bias_weight(self, weight, _bias, stream)


class _ComfyLayerNorm(nn.LayerNorm):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    @staticmethod
    def _match_input_dtype(value, input):
        if value is not None and value.is_floating_point() and value.dtype != input.dtype:
            return value.to(dtype=input.dtype)
        return value

    def forward(self, input):
        if (
            not hasattr(self, "_v")
            and (self.weight is None or self.weight.device == input.device)
            and (self.bias is None or self.bias.device == input.device)
        ):
            weight = self._match_input_dtype(self.weight, input)
            bias = self._match_input_dtype(self.bias, input)
            return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        if self.weight is not None:
            weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        else:
            weight, bias, stream = None, None, None
        weight = self._match_input_dtype(weight, input)
        bias = self._match_input_dtype(bias, input)
        try:
            return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConv1d(nn.Conv1d):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input):
        if (
            not hasattr(self, "_v")
            and self.weight.device == input.device
            and (self.bias is None or self.bias.device == input.device)
        ):
            return self._conv_forward(input, self.weight, self.bias)
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return self._conv_forward(input, weight, bias)
        finally:
            uncast_bias_weight(self, weight, bias, stream)


class _ComfyConvTranspose1d(nn.ConvTranspose1d):
    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def forward(self, input, output_size=None):
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        if (
            not hasattr(self, "_v")
            and self.weight.device == input.device
            and (self.bias is None or self.bias.device == input.device)
        ):
            return F.conv_transpose1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        weight, bias, stream = cast_bias_weight(self, input, offloadable=True)
        try:
            return F.conv_transpose1d(
                input,
                weight,
                bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
        finally:
            uncast_bias_weight(self, weight, bias, stream)


def convert_modules_for_comfy(model: nn.Module) -> None:
    """Patch castable torch modules in-place so ModelPatcherDynamic
    routes them through the VBAR-backed dynamic-loading path."""
    for module in model.modules():
        if isinstance(module, _ComfyLinear):
            continue
        if isinstance(module, nn.Linear):
            module.__class__ = _ComfyLinear
        elif type(module) is nn.Embedding:
            module.__class__ = _ComfyEmbedding
        elif isinstance(module, nn.LayerNorm):
            module.__class__ = _ComfyLayerNorm
        elif isinstance(module, nn.Conv1d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConv1d
        elif isinstance(module, nn.ConvTranspose1d) and not hasattr(module, "parametrizations"):
            module.__class__ = _ComfyConvTranspose1d
