# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared prompt/overlay helpers for the Falcon Perception examples.

Kept in one place so ``end2end.py`` and ``batch_inference.py`` cannot drift:
the prompt string is brittle (the model is sensitive to rewording) and a fix
to the overlay colours/resampling should not need to land twice.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# The reference stops on EOS (11) and <|end_of_query|> (263).
STOP_TOKEN_IDS = [11, 263]

PALETTE = [
    (255, 59, 48),
    (52, 199, 89),
    (0, 122, 255),
    (255, 149, 0),
    (175, 82, 222),
    (255, 204, 0),
    (90, 200, 250),
    (255, 45, 85),
    (162, 132, 94),
    (48, 209, 88),
]


def build_prompt(query: str) -> str:
    """The exact string the model expects. Deviating here silently degrades output."""
    return f"<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"


def overlay(image: Image.Image, masks: np.ndarray) -> Image.Image:
    """Draw each instance mask over the original image so a reviewer can eyeball it."""
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    width, height = canvas.size
    for i, mask in enumerate(masks):
        colour = PALETTE[i % len(PALETTE)]
        binary = np.asarray(mask) > 0
        if binary.shape != (height, width):
            rows = (np.arange(height) * binary.shape[0] / height).astype(int).clip(0, binary.shape[0] - 1)
            cols = (np.arange(width) * binary.shape[1] / width).astype(int).clip(0, binary.shape[1] - 1)
            binary = binary[rows][:, cols]
        stencil = Image.fromarray((binary * 255).astype(np.uint8))
        layer = Image.composite(Image.new("RGBA", canvas.size, (*colour, 115)), layer, stencil)
    return Image.alpha_composite(canvas, layer).convert("RGB")
