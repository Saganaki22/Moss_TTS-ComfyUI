"""Dependency check/install helper for Moss_TTS-ComfyUI."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil
import subprocess
import sys


PREFIX = "[Moss_TTS-ComfyUI]"
CRITICAL_IMPORTS = ["torch", "torchaudio", "transformers"]
LIGHTWEIGHT_IMPORTS = {
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface-hub",
    "numpy": "numpy",
    "safetensors": "safetensors",
    "tqdm": "tqdm",
}
MIN_TRANSFORMERS = (4, 57, 0)
RECOMMENDED_TRANSFORMERS = "5.0.0"


def _release_tuple(raw_version: str) -> tuple[int, int, int]:
    release = raw_version.split("+", 1)[0].split("-", 1)[0].split(".")
    values = []
    for part in release[:3]:
        digits = "".join(character for character in part if character.isdigit())
        values.append(int(digits or 0))
    return tuple((values + [0, 0, 0])[:3])


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _print_transformers_status() -> None:
    try:
        installed = version("transformers")
    except PackageNotFoundError:
        return
    parsed = _release_tuple(installed)
    if parsed >= MIN_TRANSFORMERS:
        print(f"{PREFIX} Transformers {installed} is new enough for MOSS/Qwen3 remote code.")
        return
    print(
        f"{PREFIX} WARNING: Transformers {installed} may be too old for MOSS-TTS v1.5. "
        f"Recommended: {RECOMMENDED_TRANSFORMERS}. This helper will not replace it."
    )


def _install_command(packages: list[str]) -> list[str]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *packages]
    return [sys.executable, "-m", "pip", "install", *packages]


def main() -> int:
    missing_critical = [name for name in CRITICAL_IMPORTS if not _has_module(name)]
    if missing_critical:
        print(f"{PREFIX} Missing ComfyUI/runtime dependency: {', '.join(missing_critical)}")
        print(f"{PREFIX} This helper will not modify torch, torchaudio, or transformers.")
        return 1

    _print_transformers_status()
    missing = [
        package
        for module, package in LIGHTWEIGHT_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        print(f"{PREFIX} Dependencies are already present.")
        return 0

    print(f"{PREFIX} Installing missing lightweight dependencies: {', '.join(missing)}")
    print(f"{PREFIX} Torch, torchaudio, and transformers are not modified.")
    command = _install_command(missing)
    installer = "uv" if Path(command[0]).stem.lower() == "uv" else "pip"
    print(f"{PREFIX} Using {installer} with active Python: {sys.executable}")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
