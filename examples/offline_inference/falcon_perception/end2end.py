#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception: image + text query -> bounding boxes and instance masks.

Two stages run behind one ``Omni`` call:

  0. thinker      — autoregressive; emits one box per detected instance
  1. segmentation — AnyUp-upsamples the image features and contracts them with
                    each ``<|seg|>`` hidden state to produce per-instance masks

The prompt format is load-bearing and is *not* a chat template. Use as ti is. The image
placeholder must be the literal ``<|image|>`` and the query must be wrapped in
``<|start_of_query|> ... <|REF_SEG|>``; the processor replaces ``<|image|>``
with the structural token run plus one token per patch.

Usage:
    python end2end.py --model tiiuae/Falcon-Perception --image photo.jpg \\
        --query "the red pepper" --out-dir /tmp/falcon_perception
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Bare import, not relative: this script is run directly (``python end2end.py``),
# which puts its own directory on ``sys.path`` but does not treat it as a
# package -- a relative import would fail with "attempted relative import with
# no known parent package".
from _common import STOP_TOKEN_IDS, build_prompt, overlay
from PIL import Image
from vllm import SamplingParams

import vllm_omni
from vllm_omni.entrypoints.omni import Omni

# Passed as a full path, not a bare name: ``Omni`` resolves a bare deploy-config
# name against ``vllm_omni/deploy/`` on one code path but reads it as a literal
# path on another, so a bare name only works when the CWD happens to hold the file.
DEPLOY_CONFIG = Path(vllm_omni.__file__).parent / "deploy" / "falcon_perception.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="tiiuae/Falcon-Perception", help="HF id or local snapshot path")
    parser.add_argument("--image", required=True, help="path to the input image")
    parser.add_argument("--query", default="the objects", help="what to segment, in plain English")
    parser.add_argument("--out-dir", default="/tmp/falcon_perception", help="where to write the overlay")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omni = Omni(model=args.model, deploy_config=str(DEPLOY_CONFIG), seed=0)

    # One SamplingParams per stage. Greedy is required, not a preference: the
    # geometry (i.e. bounding box for each object) is decoded from the hidden state that produced each <|coord|> /
    # <|size|> token, so sampling with temperature > 0 might desynchronise boxes from tokens.
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            detokenize=True,
            stop_token_ids=STOP_TOKEN_IDS,
        ),
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False),
    ]

    masks = np.zeros((0, 1, 1), dtype=np.uint8)
    boxes: list[list[float]] = []
    for stage_output in omni.generate(
        [{"prompt": build_prompt(args.query), "multi_modal_data": {"image": image}}],
        params,
    ):
        completion = stage_output.request_output.outputs[0]
        multimodal = getattr(completion, "multimodal_output", None)
        if not multimodal:
            continue
        if "masks" in multimodal and hasattr(multimodal["masks"], "shape"):
            masks = np.asarray(multimodal["masks"].to(torch.uint8).cpu())
        if "boxes" in multimodal and hasattr(multimodal["boxes"], "shape"):
            boxes = np.asarray(multimodal["boxes"].float().cpu()).tolist()

    print(f"query   : {args.query!r}")
    print(f"image   : {image.size[0]}x{image.size[1]}")
    print(f"instances: {masks.shape[0]}")
    for i, box in enumerate(boxes[:10]):
        x, y, w, h = box
        print(f"  box {i:>3}: centre=({x:.3f}, {y:.3f}) size=({w:.3f}, {h:.3f})  [normalised]")
    if len(boxes) > 10:
        print(f"  ... {len(boxes) - 10} more")

    # save the segm masks overlaid on the original image
    if masks.shape[0]:
        out_path = out_dir / f"{Path(args.image).stem}_masks.png"
        overlay(image, masks).save(out_path)
        print(f"\nwrote overlay to {out_path}")
    else:
        print("\nno instances matched the query — nothing to draw")


# Required: the deploy YAML uses distributed_executor_backend: mp with spawn, and
# a script without this guard hangs in _check_not_importing_main.
if __name__ == "__main__":
    main()
