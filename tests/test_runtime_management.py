from __future__ import annotations

import importlib.util
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load_loader_module():
    spec = importlib.util.spec_from_file_location("mosstts_test_loader", ROOT / "loader.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_test_package():
    package_name = "mosstts_test_package"
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class ReadOnlyDeviceModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    @property
    def device(self):
        return next(self.parameters()).device


def _initialize_aimdo_if_possible():
    import comfy.model_management as mm
    import comfy_aimdo.control

    try:
        comfy_aimdo.control.init()
    except TypeError:
        comfy_aimdo.control.init(None)

    if comfy_aimdo.control.lib is None:
        return False

    import comfy_aimdo.host_buffer
    import comfy_aimdo.model_vbar
    import comfy_aimdo.vram_buffer

    importlib.reload(comfy_aimdo.host_buffer)
    importlib.reload(comfy_aimdo.model_vbar)
    importlib.reload(comfy_aimdo.vram_buffer)

    devices = [device for device in mm.get_all_torch_devices() if device.type == "cuda"]
    if not devices:
        return False

    try:
        return comfy_aimdo.control.init_devices((device.index, 0) for device in devices)
    except TypeError:
        return comfy_aimdo.control.init_devices(device.index for device in devices)


@pytest.fixture()
def loader():
    return _load_loader_module()


@pytest.fixture(autouse=True)
def clean_comfy_loaded_models():
    import comfy.model_management as mm

    original = list(mm.current_loaded_models)
    mm.current_loaded_models[:] = []
    yield
    for loaded in list(mm.current_loaded_models):
        try:
            loaded.model.detach()
        except Exception:
            pass
    mm.current_loaded_models[:] = original


def test_static_comfy_registration_handles_read_only_device(loader, monkeypatch):
    import comfy.memory_management
    import comfy.model_management as mm
    import comfy.model_patcher

    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)
    model = ReadOnlyDeviceModule()

    patcher = loader.register_runtime_module(model, torch.device("cpu"), dynamic=False)

    assert patcher is None
    assert model.device == torch.device("cpu")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for ComfyUI GPU patcher smoke coverage.")

    model = ReadOnlyDeviceModule()
    patcher = loader.register_runtime_module(model, torch.device("cuda"), dynamic=False)

    assert isinstance(patcher, comfy.model_patcher.ModelPatcher)
    assert not patcher.is_dynamic()
    assert any(loaded.model is patcher for loaded in mm.current_loaded_models)

    loader._unregister_from_comfy(patcher)
    assert not any(loaded.model is patcher for loaded in mm.current_loaded_models)


def test_comfy_progress_bar_uses_generation_budget(monkeypatch):
    package = _load_test_package()
    nodes = importlib.import_module(f"{package.__name__}.nodes")
    events = []

    class FakeProgressBar:
        def __init__(self, total):
            events.append(("init", total))

        def update_absolute(self, current, total):
            events.append(("update", current, total))

    monkeypatch.setattr(nodes, "ProgressBar", FakeProgressBar)

    callback = nodes._progress(123)
    callback(8, 123)

    assert events == [("init", 123), ("update", 8, 123)]


def test_runtime_forwards_progress_callback_to_model_generate(monkeypatch):
    package = _load_test_package()
    runtime = importlib.import_module(f"{package.__name__}.runtime")
    loader_module = importlib.import_module(f"{package.__name__}.loader")
    monkeypatch.setattr(loader_module, "resume_bundle_to_device", lambda bundle: None)

    progress_events = []
    generate_kwargs = {}

    class FakeProcessor:
        model_config = type("Config", (), {"sampling_rate": 48000})()

        def __call__(self, conversations, mode):
            return {
                "input_ids": torch.zeros((1, 1, 13), dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.bool),
            }

        def decode(self, outputs, return_stereo=True):
            return [type("Message", (), {"audio_codes_list": [torch.zeros((2, 4))]})()]

    class FakeModel:
        def generate(self, **kwargs):
            generate_kwargs.update(kwargs)
            kwargs["progress_callback"](2, 10)
            return object()

    bundle = type(
        "Bundle",
        (),
        {
            "processor": FakeProcessor(),
            "model": FakeModel(),
            "device": torch.device("cpu"),
        },
    )()

    runtime._run_model_generate(
        bundle,
        [[{"role": "user"}]],
        mode="generation",
        max_new_tokens=10,
        do_sample=True,
        text_temperature=1.0,
        text_top_p=1.0,
        text_top_k=50,
        audio_temperature=1.7,
        audio_top_p=0.8,
        audio_top_k=25,
        audio_repetition_penalty=1.0,
        seed=0,
        progress_callback=lambda current, total: progress_events.append((current, total)),
    )

    assert generate_kwargs["progress_callback"] is not None
    assert generate_kwargs["show_progress"] is True
    assert generate_kwargs["progress_description"] == "MOSS-TTS generation"
    assert (0, 10) in progress_events
    assert (2, 10) in progress_events
    assert progress_events[-1] == (10, 10)


def test_whisper_uses_cpu_staging_and_dynamic_registration_when_aimdo_active(monkeypatch, tmp_path):
    package = _load_test_package()
    whisper = importlib.import_module(f"{package.__name__}.whisper")
    whisper._PIPELINE_CACHE.clear()

    pipeline_calls = []
    registrations = []
    resumed = []

    class FakeWhisperModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

    class FakePipe:
        def __init__(self):
            self.model = FakeWhisperModel()
            self.device = torch.device("cpu")

    def fake_pipeline(model_path, torch_dtype, device):
        pipeline_calls.append((model_path, torch_dtype, device))
        return FakePipe()

    monkeypatch.setattr(whisper, "_build_asr_pipeline", fake_pipeline)
    monkeypatch.setattr(whisper, "register_audio_encoders_folder", lambda: None)
    monkeypatch.setattr(whisper, "_resolve_device", lambda: "cuda:0")
    monkeypatch.setattr(whisper, "_resolve_whisper_path", lambda model, download: tmp_path)
    monkeypatch.setattr(whisper, "_resolve_dtype", lambda dtype, device: torch.bfloat16)
    monkeypatch.setattr(whisper, "dynamic_vram_active", lambda device: True)
    monkeypatch.setattr(
        whisper,
        "register_runtime_module",
        lambda module, device, dynamic=None: registrations.append((module, torch.device(device), dynamic)) or object(),
    )
    monkeypatch.setattr(
        whisper,
        "resume_runtime_module",
        lambda patcher, device: resumed.append((patcher, torch.device(device))),
    )

    pipe = whisper.get_whisper_pipeline("whisper-large-v3-turbo (auto-download)", "auto", False)

    assert pipeline_calls[0] == (tmp_path, torch.bfloat16, "cpu")
    assert registrations[0][1] == torch.device("cuda:0")
    assert registrations[0][2] is True
    assert pipe.device == torch.device("cuda:0")
    assert getattr(pipe.model.proj, "comfy_cast_weights", False) is True

    cached = whisper.get_whisper_pipeline("whisper-large-v3-turbo (auto-download)", "auto", False)
    assert cached is pipe
    assert resumed and resumed[0][1] == torch.device("cuda:0")
    whisper._PIPELINE_CACHE.clear()


def test_whisper_filters_punctuation_only_hallucinations():
    package = _load_test_package()
    whisper = importlib.import_module(f"{package.__name__}.whisper")

    assert whisper._normalize_transcript("........") == ""
    assert whisper._normalize_transcript("   ") == ""
    assert whisper._normalize_transcript("Hello.") == "Hello."


def test_audio_tokenizer_device_prefers_comfy_runtime_device():
    package = _load_test_package()
    native = importlib.import_module(f"{package.__name__}.native")
    native.tts_classes()
    from _mosstts_native_tts.processing_moss_tts import MossTTSLocalProcessor

    class FakeTokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.device = torch.device("cuda:0")

    processor = types.SimpleNamespace(audio_tokenizer=FakeTokenizer())

    assert MossTTSLocalProcessor._get_audio_tokenizer_device(processor) == torch.device("cuda:0")


def test_codec_autocast_prefers_comfy_runtime_device(monkeypatch):
    package = _load_test_package()
    native = importlib.import_module(f"{package.__name__}.native")
    native.codec_classes()
    from _mosstts_native_codec.modeling_moss_audio_tokenizer import MossAudioTokenizerModel

    class FakeCodec:
        compute_dtype = torch.bfloat16
        _mosstts_runtime_device = torch.device("cuda:0")

        def __init__(self):
            self.weight = nn.Parameter(torch.zeros(1))

        def parameters(self):
            return iter([self.weight])

    calls = []

    class FakeAutocast:
        def __init__(self, *, device_type, dtype):
            calls.append((device_type, dtype))

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch, "autocast", FakeAutocast)

    with MossAudioTokenizerModel._codec_inference_autocast(FakeCodec()):
        pass

    assert calls == [("cuda", torch.bfloat16)]


def test_native_conversion_adds_comfy_cast_modules():
    package = _load_test_package()
    native = importlib.import_module(f"{package.__name__}.native")

    class CustomEmbedding(nn.Embedding):
        def forward(self, input, past_key_values_length=0):
            return super().forward(input + past_key_values_length)

    model = nn.Module()
    model.linear = nn.Linear(4, 4)
    model.embedding = nn.Embedding(8, 4)
    model.custom_embedding = CustomEmbedding(8, 4)
    model.norm = nn.LayerNorm(4)
    model.conv = nn.Conv1d(2, 2, 3)
    model.deconv = nn.ConvTranspose1d(2, 2, 3)
    model.weight_norm_conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(2, 2, 3))

    native.convert_modules_for_comfy(model)

    assert getattr(model.linear, "comfy_cast_weights", False) is True
    assert getattr(model.embedding, "comfy_cast_weights", False) is True
    assert not getattr(model.custom_embedding, "comfy_cast_weights", False)
    model.custom_embedding(torch.zeros(1, dtype=torch.long), past_key_values_length=1)
    assert getattr(model.norm, "comfy_cast_weights", False) is True
    assert getattr(model.conv, "comfy_cast_weights", False) is True
    assert getattr(model.deconv, "comfy_cast_weights", False) is True
    assert not getattr(model.weight_norm_conv, "comfy_cast_weights", False)


def test_comfy_embedding_cast_uses_weight_dtype_for_integer_ids(monkeypatch):
    package = _load_test_package()
    native = importlib.import_module(f"{package.__name__}.native")

    model = nn.Module()
    model.embedding = nn.Embedding(8, 4)
    native.convert_modules_for_comfy(model)
    model.embedding.weight_comfy_model_dtype = torch.bfloat16
    model.embedding._v = object()

    captured = {}

    def fake_cast_bias_weight(module, input=None, **kwargs):
        captured.update(kwargs)
        return module.weight.to(dtype=kwargs["dtype"]), None, None

    monkeypatch.setattr(native, "cast_bias_weight", fake_cast_bias_weight)
    monkeypatch.setattr(native, "uncast_bias_weight", lambda *args, **kwargs: None)

    model.embedding(torch.zeros(1, dtype=torch.long))

    assert captured["dtype"] == torch.bfloat16
    assert captured["device"] == torch.device("cpu")


def test_aimdo_dynamic_registration_and_unload(loader, monkeypatch):
    import comfy.memory_management
    import comfy.model_management as mm
    import comfy.model_patcher

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native AIMDO dynamic VRAM smoke coverage.")
    if not hasattr(comfy.model_patcher, "ModelPatcherDynamic"):
        pytest.skip("This ComfyUI build does not expose ModelPatcherDynamic.")
    if not _initialize_aimdo_if_possible():
        pytest.skip("comfy-aimdo native library could not be initialized in this process.")

    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)
    monkeypatch.setattr(
        comfy.model_patcher,
        "CoreModelPatcher",
        comfy.model_patcher.ModelPatcherDynamic,
    )
    model = ReadOnlyDeviceModule()

    patcher = loader.register_runtime_module(model, torch.device("cuda"), dynamic=True)

    assert isinstance(patcher, comfy.model_patcher.ModelPatcherDynamic)
    assert patcher.is_dynamic()
    assert any(loaded.model is patcher for loaded in mm.current_loaded_models)

    bundle = loader.MossTTSBundle(
        model=model,
        processor=type("Processor", (), {"audio_tokenizer": None})(),
        model_dir=ROOT,
        model_path=ROOT / "model.safetensors",
        codec_dir=ROOT,
        device=torch.device("cuda"),
        torch_dtype=torch.bfloat16,
        dtype_name="bf16",
        attention="sdpa",
        attn_implementation="sdpa",
        patchers=[patcher],
    )

    loader.resume_bundle_to_device(bundle)
    assert any(loaded.model is patcher for loaded in mm.current_loaded_models)

    loader.unload_mosstts_bundle(bundle, reason="test")
    assert bundle.patchers == []
    assert not any(loaded.model is patcher for loaded in mm.current_loaded_models)


def test_bundle_load_forces_dynamic_for_main_model_and_codec_when_aimdo_active(loader, monkeypatch, tmp_path):
    model_dir = tmp_path / "moss"
    codec_dir = tmp_path / "codec"
    model_dir.mkdir()
    codec_dir.mkdir()
    model_path = model_dir / "model.safetensors"
    model_path.write_bytes(b"model")
    for filename in loader.CODEC_MODEL_FILES:
        (codec_dir / filename).write_bytes(b"codec")

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

    class FakeProcessor:
        def __init__(self):
            self.audio_tokenizer = FakeModel()

    registrations = []

    monkeypatch.setattr(loader, "_ACTIVE_BUNDLE", None)
    monkeypatch.setattr(loader, "_ACTIVE_LOAD_KEY", None)
    monkeypatch.setattr(loader, "register_model_folder", lambda: None)
    monkeypatch.setattr(loader, "resolve_main_model_dir", lambda model_choice, download_if_missing: (model_dir, model_path))
    monkeypatch.setattr(loader, "resolve_codec_dir", lambda download_if_missing: codec_dir)
    monkeypatch.setattr(loader, "resolve_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(loader, "resolve_dtype", lambda dtype_name, device: torch.bfloat16)
    monkeypatch.setattr(loader, "resolve_attention", lambda attention, device, dtype: ("sdpa", "sdpa"))
    monkeypatch.setattr(loader, "_configure_torch_attention", lambda device: None)
    monkeypatch.setattr(
        loader,
        "_load_model_and_processor",
        lambda **kwargs: (FakeModel(), FakeProcessor(), FakeModel()),
    )
    monkeypatch.setattr(loader, "install_comfy_unload_hook", lambda: None)
    monkeypatch.setattr(loader, "dynamic_vram_active", lambda device: True)
    monkeypatch.setattr(
        loader,
        "register_runtime_module",
        lambda module, device, dynamic=None: registrations.append((module, torch.device(device), dynamic)) or object(),
    )

    bundle = loader.load_mosstts_bundle(loader.DEFAULT_MODEL, "auto", "auto", False)

    assert len(bundle.patchers) == 2
    assert [entry[2] for entry in registrations] == [True, True]
