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
"""Configuration for IronViT vision encoders.

IronViT follows the Qwen3-VL vision-tower architecture (patch embedding, 2-D RoPE,
learned position embedding, gelu-tanh MLP, 27 layers x 1152 hidden) and changes
two things:

* attention: all blocks use softmax (``IronViT-Softmax``), or 9 blocks use
  softmax while the remaining 18 use linear attention (``IronViT-Hybrid``);
* read-out: 8 register tokens plus a learned-query attention-pooling head (MAP)
  produce one global feature vector per image.
"""

from __future__ import annotations

from typing import Any, List, Optional

from transformers import PretrainedConfig


class IronViTConfig(PretrainedConfig):
    """Configuration object for :class:`IronViTModel`.

    Args:
        hidden_size: Width of the transformer blocks.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads; ``hidden_size % num_heads`` must
            be zero.
        intermediate_size: MLP hidden width.
        hidden_act: MLP activation. ``gelu_pytorch_tanh`` matches Qwen3-VL.
        patch_size: Spatial size of a patch, in pixels.
        temporal_patch_size: Number of frames a patch spans. Still images are
            repeated along this axis to satisfy the Conv3d kernel.
        spatial_merge_size: Edge length of the block used by the patch merger
            and by the blocked token order.
        in_channels: Number of image channels.
        num_position_embeddings: Size of the learned position-embedding table;
            the released models use 2304, a 48x48 grid.
        rope_theta: Base of the 2-D rotary embedding.
        qk_norm: Apply RMSNorm to per-head queries and keys in the backbone.
        layer_norm_eps: Epsilon of every LayerNorm / RMSNorm.
        attention_type: ``"softmax"`` or ``"hybrid"``.
        softmax_layer_indices: Block indices that keep softmax attention in the
            hybrid model. Required for ``"hybrid"``, forbidden for
            ``"softmax"``.
        linear_act: Kernel of the linear attention. The released hybrid model
            uses ``"1+elu"``.
        linear_scale: Constant folded into the linear-attention denominator.
        linear_gate: Multiply the linear-attention output by
            ``sigmoid(gate_proj(x))`` before the output projection.
        num_register_tokens: Number of learnable register tokens prepended to
            each image.
        num_cls_tokens: Number of learned pooling queries; the MAP head is
            omitted when this is 0.
        map_qk_norm: Apply RMSNorm to per-head queries and keys in the MAP head.
        post_layernorm: LayerNorm applied to the patch tokens before they are
            returned.
        initializer_range: Standard deviation of the truncated-normal weight
            initialisation.

    Raises:
        ValueError: If ``attention_type`` and ``softmax_layer_indices``
            contradict each other.
    """

    model_type = "ironvit"

    def __init__(
        self,
        hidden_size: int = 1152,
        depth: int = 27,
        num_heads: int = 16,
        intermediate_size: int = 4304,
        hidden_act: str = "gelu_pytorch_tanh",
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
        num_position_embeddings: int = 2304,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        layer_norm_eps: float = 1e-6,
        # --- attention mix ---
        attention_type: str = "softmax",
        softmax_layer_indices: Optional[List[int]] = None,
        linear_act: str = "1+elu",
        linear_scale: float = 8.0,
        linear_gate: bool = True,
        # --- read-out ---
        num_register_tokens: int = 8,
        num_cls_tokens: int = 1,
        map_qk_norm: bool = True,
        post_layernorm: bool = True,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.in_channels = in_channels
        self.num_position_embeddings = num_position_embeddings
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.layer_norm_eps = layer_norm_eps

        self.attention_type = attention_type
        self.softmax_layer_indices = sorted(softmax_layer_indices or [])
        self.linear_act = linear_act
        self.linear_scale = linear_scale
        self.linear_gate = linear_gate

        self.num_register_tokens = num_register_tokens
        self.num_cls_tokens = num_cls_tokens
        self.map_qk_norm = map_qk_norm
        self.post_layernorm = post_layernorm

        self.initializer_range = initializer_range

        if self.attention_type not in ("softmax", "hybrid"):
            raise ValueError(f"attention_type must be 'softmax' or 'hybrid', got {attention_type!r}")
        if self.attention_type == "softmax" and self.softmax_layer_indices:
            raise ValueError("attention_type='softmax' must not set softmax_layer_indices")
        if self.attention_type == "hybrid" and not self.softmax_layer_indices:
            raise ValueError("attention_type='hybrid' requires softmax_layer_indices")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        if (self.hidden_size // self.num_heads) % 4 != 0:
            raise ValueError(
                f"head_dim ({self.hidden_size // self.num_heads}) must be divisible by 4 for 2-D RoPE"
            )
        if any(i < 0 or i >= self.depth for i in self.softmax_layer_indices):
            raise ValueError(
                f"softmax_layer_indices must be in [0, {self.depth - 1}], got {self.softmax_layer_indices}"
            )
