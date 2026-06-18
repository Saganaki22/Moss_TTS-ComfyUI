"""ComfyUI custom nodes for MOSS-TTS Local Transformer."""

from __future__ import annotations

__version__ = "0.1.1"

import logging
from typing import Any


logger = logging.getLogger("Moss_TTS-ComfyUI")
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Moss_TTS-ComfyUI] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


NODE_CLASS_MAPPINGS: dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

if __package__:
    try:
        from .nodes import NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS
        from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS

        NODE_CLASS_MAPPINGS.update(_NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_DISPLAY_NAME_MAPPINGS)
        logger.info("Registered %d node(s).", len(NODE_CLASS_MAPPINGS))
    except Exception as exc:
        logger.error("Failed to register MOSS-TTS nodes: %s", exc, exc_info=True)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
