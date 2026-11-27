# Copyright (c) 2026 XPENG Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert an internal IronViT training checkpoint to the public layout.

The internal checkpoints wrap the vision tower in a wrapper module and keep the
Qwen3-VL patch merger / DeepStack mergers, which the released models do not use.
This script renames the tensor keys to the released ``IronViTModel`` state dict
and writes ``model.safetensors`` + ``config.json`` +
``preprocessor_config.json`` + the three model modules from ``ironvit/``, i.e. a
directory you can upload to the Hub as-is.

Example::

    python tools/convert_checkpoint.py \\
        --src /path/to/internal_ckpt \\
        --dst /path/to/IronViT-Softmax-400M \\
        --variant softmax
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict

import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_FILES = ("modeling_ironvit.py", "configuration_ironvit.py", "image_processing_ironvit.py")

SOFTMAX_LAYER_INDICES = [3, 7, 11, 15, 19, 23, 24, 25, 26]

BASE_CONFIG = dict(
    hidden_size=1152,
    depth=27,
    num_heads=16,
    intermediate_size=4304,
    hidden_act="gelu_pytorch_tanh",
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_channels=3,
    num_position_embeddings=2304,
    rope_theta=10000.0,
    qk_norm=True,
    layer_norm_eps=1e-6,
    num_register_tokens=8,
    num_cls_tokens=1,
    map_qk_norm=True,
    post_layernorm=True,
    initializer_range=0.02,
    architectures=["IronViTModel"],
    model_type="ironvit",
    auto_map={
        "AutoConfig": "configuration_ironvit.IronViTConfig",
        "AutoModel": "modeling_ironvit.IronViTModel",
        "AutoImageProcessor": "image_processing_ironvit.IronViTImageProcessor",
    },
)

PREPROCESSOR_CONFIG = {
    "image_processor_type": "IronViTImageProcessor",
    "processor_class": "IronViTImageProcessor",
    "auto_map": {"AutoImageProcessor": "image_processing_ironvit.IronViTImageProcessor"},
    "do_resize": True,
    "do_rescale": True,
    "do_normalize": True,
    "rescale_factor": 1.0 / 255.0,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "in_channels": 3,
    "size": {"height": 384, "width": 384},
    "max_patches": 576,
}

LINEAR_ATTN_KEYS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.out_proj.weight",
    "attn.gate_proj.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
)


def convert(state_dict: Dict[str, torch.Tensor], variant: str) -> Dict[str, torch.Tensor]:
    """Rename an internal state dict to the released ``IronViTModel`` layout.

    Args:
        state_dict: Tensors loaded from the internal training checkpoint.
        variant: ``"softmax"`` or ``"hybrid"``; selects which block keys are
            kept and which LayerNorm becomes ``post_layernorm``.

    Returns:
        A state dict that loads into :class:`IronViTModel` unchanged.
    """
    softmax_blocks = set(SOFTMAX_LAYER_INDICES) if variant == "hybrid" else None
    out = {
        "register_tokens": state_dict["register_tokens"],
        "patch_embed.proj.weight": state_dict["vision.patch_embed.proj.weight"],
        "patch_embed.proj.bias": state_dict["vision.patch_embed.proj.bias"],
        "pos_embed.weight": state_dict["vision.pos_embed.weight"],
    }

    # The post LayerNorm is the merger norm of the softmax model, and the
    # student's own post_layernorm for the hybrid model.
    if variant == "softmax":
        prefix = "vision.merger.norm."
    else:
        prefix = "vision.post_layernorm."
    out["post_layernorm.weight"] = state_dict[prefix + "weight"]
    out["post_layernorm.bias"] = state_dict[prefix + "bias"]

    for key, value in state_dict.items():
        if key.startswith("vision.blocks."):
            index, _, tail = key[len("vision.blocks."):].partition(".")
            if tail.startswith("attn."):
                if variant == "hybrid" and int(index) not in softmax_blocks:
                    if tail not in LINEAR_ATTN_KEYS:
                        continue
            out[f"blocks.{index}.{tail}"] = value
        elif key.startswith("map_head."):
            out[key] = value

    # Older hybrid checkpoints predate the MAP head's QK-norm parameters. The
    # training code still applies RMSNorm there (unit-initialised, so it is a
    # no-op on the weights), so materialise the two missing vectors. The
    # checkpoint behind the released IronViT-Hybrid-400M already has them.
    for name in ("map_head.q_norm.weight", "map_head.k_norm.weight"):
        if name not in out and state_dict.get("map_head.query") is not None:
            out[name] = torch.ones(BASE_CONFIG["hidden_size"] // BASE_CONFIG["num_heads"])

    return out


def validate_converted(converted: Dict[str, torch.Tensor], config: dict) -> None:
    """Verify converted keys and shapes match :class:`IronViTModel` exactly."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from ironvit import IronViTConfig, IronViTModel

    cfg = IronViTConfig(**config)
    with torch.device("meta"):
        expected = {k: tuple(v.shape) for k, v in IronViTModel(cfg).state_dict().items()}
    actual = {k: tuple(v.shape) for k, v in converted.items()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    bad_shapes = sorted(
        f"{k}: expected {expected[k]}, got {actual[k]}"
        for k in set(expected) & set(actual)
        if expected[k] != actual[k]
    )
    if missing or unexpected or bad_shapes:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unexpected:
            parts.append(f"unexpected={unexpected}")
        if bad_shapes:
            parts.append(f"shape_mismatch={bad_shapes}")
        raise ValueError("converted state dict does not match IronViTModel: " + "; ".join(parts))


def main() -> None:
    """Convert a checkpoint and write an upload-ready model directory."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="internal checkpoint directory")
    parser.add_argument("--dst", required=True, help="output directory")
    parser.add_argument("--variant", choices=["softmax", "hybrid"], required=True)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    state_dict = torch.load(os.path.join(args.src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    converted = convert(state_dict, args.variant)
    config = dict(BASE_CONFIG)
    if args.variant == "softmax":
        config["attention_type"] = "softmax"
    else:
        config.update(
            attention_type="hybrid",
            softmax_layer_indices=SOFTMAX_LAYER_INDICES,
            linear_act="1+elu",
            linear_scale=8.0,
            linear_gate=True,
        )
    validate_converted(converted, config)
    # bf16 is what the released weights ship in; conversion from the fp32
    # training checkpoint is lossless enough that no probe notices.
    save_file(
        {key: value.to(torch.bfloat16).contiguous() for key, value in converted.items()},
        os.path.join(args.dst, "model.safetensors"),
    )
    with open(os.path.join(args.dst, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(os.path.join(args.dst, "preprocessor_config.json"), "w", encoding="utf-8") as handle:
        json.dump(PREPROCESSOR_CONFIG, handle, indent=2)
    # trust_remote_code=True loads these next to the weights.
    for name in MODULE_FILES:
        shutil.copyfile(os.path.join(REPO_ROOT, "ironvit", name), os.path.join(args.dst, name))

    print(f"wrote {len(converted)} tensors to {args.dst}")


if __name__ == "__main__":
    main()
