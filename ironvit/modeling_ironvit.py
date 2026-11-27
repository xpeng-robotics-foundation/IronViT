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
"""IronViT model definition.

Pure PyTorch: no Triton, flash-attention or hardware-specific kernels, so the
file loads unchanged through ``trust_remote_code=True`` on any backend that
runs ``torch.nn.functional.scaled_dot_product_attention``.

Architecture
------------
A ViT that follows the Qwen3-VL vision-tower architecture with two changes:

1. **Attention mix.** ``IronViT-Softmax`` uses softmax attention in every
   block. ``IronViT-Hybrid`` keeps softmax attention in
   ``config.softmax_layer_indices`` (9 of 27 blocks) and uses linear attention
   elsewhere (18 blocks). The linear operator is non-causal and length
   normalised::

       out_i = act(W_q x_i) · (Σ_j act(W_k x_j) act(W_v x_j)ᵀ) / (L · scale)

   with ``act = 1 + elu`` and ``scale = 8``. It is O(L) instead of O(L²), which
   pays off at high resolution. An output gate ``sigmoid(W_g x)`` multiplies
   the attention result before ``out_proj``.

2. **Read-out.** 8 learnable register tokens are prepended to every image's
   token sequence (they attend, then are dropped). A learned-query multi-head
   attention pooling head (MAP) turns the patch tokens into one global feature
   vector per image.

Positions
---------
Two encodings, as in Qwen3-VL: a learned 48x48 position embedding bilinearly
interpolated to the image grid and added to the patch embeddings, and a 2-D
rotary position embedding applied inside attention. Register tokens get zero
RoPE and no additive position embedding.

Token order
-----------
Patches are in Qwen3-VL "blocked" order — ``(block_h, block_w, intra_h,
intra_w)`` — so each ``spatial_merge_size x spatial_merge_size`` block is
contiguous and the patch merger can collapse it directly.
:meth:`IronViTModel.to_raster_order` restores the usual row-major grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_ironvit import IronViTConfig


@dataclass
class IronViTOutput(ModelOutput):
    """Output of :class:`IronViTModel`.

    Attributes:
        patch_features: ``[num_patches, hidden_size]`` layer-normed patch
            tokens, packed across the batch in blocked order. Input to the
            segmentation and depth probes.
        global_features: ``[batch, num_cls_tokens * hidden_size]`` MAP-pooled
            image summaries. Input to the classification probes.
    """

    patch_features: Optional[torch.Tensor] = None
    global_features: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half a turn, as in RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE over ``[seq, heads, head_dim]`` with per-token ``cos``/``sin``.

    Args:
        x: Queries or keys of shape ``[seq, heads, head_dim]``.
        cos, sin: Per-token rotation factors of shape ``[seq, head_dim]``.

    Returns:
        The rotated tensor, in the input dtype. Rotation runs in float32 and
        the result is cast back, matching Qwen3-VL.
    """
    x_float = x.float()
    out = x_float * cos[:, None, :] + _rotate_half(x_float) * sin[:, None, :]
    return out.type_as(x)


class IronViTPatchEmbed(nn.Module):
    """Conv3d patch embedding over ``(temporal_patch_size, patch, patch)``."""

    def __init__(self, config: IronViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed a packed batch of patches.

        Args:
            pixel_values: ``[num_patches, C * T * P * P]`` flattened patches.

        Returns:
            ``[num_patches, hidden_size]`` patch embeddings.
        """
        x = pixel_values.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class IronViTMLP(nn.Module):
    """Two-layer MLP with the activation named in ``config.hidden_act``."""

    def __init__(self, config: IronViTConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.hidden_act = config.hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``fc1 -> activation -> fc2`` to ``[seq, hidden_size]`` tokens."""
        x = self.linear_fc1(x)
        if self.hidden_act in ("gelu_pytorch_tanh", "gelu"):
            x = F.gelu(x, approximate="tanh")
        elif self.hidden_act == "silu":
            x = F.silu(x)
        else:
            raise ValueError(f"unsupported hidden_act={self.hidden_act!r}")
        return self.linear_fc2(x)


class IronViTSoftmaxAttention(nn.Module):
    """Qwen3-VL softmax attention: fused ``qkv``, RMSNorm on per-head Q/K, RoPE."""

    def __init__(self, config: IronViTConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        # nn.Identity() when disabled, so the parameter names never change.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps) if config.qk_norm else nn.Identity()
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Run softmax attention over a packed batch.

        Args:
            x: ``[seq, dim]`` packed tokens of every image in the batch.
            cu_seqlens: ``[batch + 1]`` cumulative image lengths.
            cos, sin: ``[seq, head_dim]`` RoPE factors.

        Returns:
            ``[seq, dim]`` attention output.
        """
        seq_len = x.shape[0]
        q, k, v = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        out = _packed_softmax_attention(q, k, v, cu_seqlens, self.scale)
        return self.proj(out.reshape(seq_len, -1))


class IronViTLinearAttention(nn.Module):
    """Length-normalised, non-causal linear attention with the ``1+elu`` kernel.

    Computed per image in a packed batch::

        kv        = Σ_j act(k_j) act(v_j)ᵀ / (L · scale)
        out_i     = act(q_i) · kv

    The state is cast to float32 before accumulation (unless overridden by an
    outer autocast context).
    """

    def __init__(self, config: IronViTConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = config.linear_scale
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps) if config.qk_norm else nn.Identity()
        self.gate_proj = nn.Linear(self.dim, self.dim, bias=False) if config.linear_gate else None
        self.act_fn = {"1+elu": lambda t: 1.0 + F.elu(t), "2+elu": lambda t: 2.0 + F.elu(t),
                       "relu6": F.relu6, "relu": F.relu, "silu": F.silu}[config.linear_act]

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Run linear attention over a packed batch.

        Args:
            x: ``[seq, dim]`` packed tokens of every image in the batch.
            cu_seqlens: ``[batch + 1]`` cumulative image lengths.
            cos, sin: ``[seq, head_dim]`` RoPE factors.

        Returns:
            ``[seq, dim]`` attention output.
        """
        seq_len = x.shape[0]
        shape = (seq_len, self.num_heads, self.head_dim)
        q = self.act_fn(self.q_proj(x)).reshape(shape)
        k = self.act_fn(self.k_proj(x)).reshape(shape)
        v = self.act_fn(self.v_proj(x)).reshape(shape)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        out = torch.empty_like(v)
        bounds = cu_seqlens.tolist()
        for start, end in zip(bounds[:-1], bounds[1:]):
            length = end - start
            qs, ks, vs = q[start:end].float(), k[start:end].float(), v[start:end].float()
            kv = torch.einsum("lhd,lhe->hde", ks, vs) / (length * self.scale)
            out[start:end] = torch.einsum("lhd,hde->lhe", qs, kv).to(v.dtype)

        out = out.reshape(seq_len, -1)
        if self.gate_proj is not None:
            out = torch.sigmoid(self.gate_proj(x)) * out
        return self.out_proj(out)


def _packed_softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              cu_seqlens: torch.Tensor, scale: float) -> torch.Tensor:
    """Non-causal attention within each image of a packed ``[seq, heads, dim]`` batch."""
    out = torch.empty_like(v)
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        qi, ki, vi = (t[start:end].transpose(0, 1)[None] for t in (q, k, v))
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale, is_causal=False)
        out[start:end] = oi[0].transpose(0, 1)
    return out


class IronViTBlock(nn.Module):
    """Pre-norm transformer block; ``linear`` selects the attention operator."""

    def __init__(self, config: IronViTConfig, linear: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = IronViTLinearAttention(config) if linear else IronViTSoftmaxAttention(config)
        self.mlp = IronViTMLP(config)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP to a packed batch of tokens.

        Args:
            x: ``[seq, dim]`` packed tokens.
            cu_seqlens: ``[batch + 1]`` cumulative image lengths.
            cos, sin: ``[seq, head_dim]`` RoPE factors.

        Returns:
            ``[seq, dim]`` block output.
        """
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.norm2(x))


class IronViTMapHead(nn.Module):
    """Learned-query multi-head attention pooling, one vector per image.

    Composition: ``LayerNorm(query + attention(query, patches))``.
    """

    def __init__(self, config: IronViTConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.query = nn.Parameter(torch.empty(1, config.num_cls_tokens, self.hidden_size))
        self.attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, batch_first=True)
        eps = config.layer_norm_eps
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps) if config.map_qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps) if config.map_qk_norm else nn.Identity()
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, patch_tokens: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Pool every image in a packed batch to one vector each.

        Args:
            patch_tokens: ``[num_patches, dim]`` packed patch tokens.
            grid_thw: ``[batch, 3]`` integer ``(t, h, w)`` patch grids.

        Returns:
            ``[batch, num_cls_tokens * dim]`` pooled image summaries.
        """
        lengths = grid_thw.prod(-1).tolist()
        # Images in a packed batch have different token counts, so pool each
        # image separately instead of padding and masking.
        pooled = [self._pool_one(tokens[None]) for tokens in torch.split(patch_tokens, lengths, dim=0)]
        return torch.cat([p.reshape(1, -1) for p in pooled], dim=0)

    def _pool_one(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool a single image's tokens, ``[1, L, dim]`` -> ``[1, num_cls, dim]``."""
        query = self.query
        if isinstance(self.q_norm, nn.Identity):
            pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        else:
            pooled = self._normed_attention(query, tokens)
        return self.norm(query + pooled)

    def _normed_attention(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``nn.MultiheadAttention`` forward with RMSNorm on per-head Q and K."""
        d, h = self.hidden_size, self.num_heads
        weight, bias = self.attn.in_proj_weight, self.attn.in_proj_bias

        def project(x: torch.Tensor, part: int) -> torch.Tensor:
            """Slice rows ``part`` of the fused in-projection and split the heads."""
            rows = slice(part * d, (part + 1) * d)
            out = F.linear(x, weight[rows], None if bias is None else bias[rows])
            return out.view(1, -1, h, d // h).transpose(1, 2)

        q = self.q_norm(project(query, 0))
        k = self.k_norm(project(tokens, 1))
        v = project(tokens, 2)
        pooled = F.scaled_dot_product_attention(q, k, v)
        return self.attn.out_proj(pooled.transpose(1, 2).reshape(1, -1, d))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class IronViTPreTrainedModel(PreTrainedModel):
    """Base class wiring :class:`IronViTModel` into the ``transformers`` API."""

    config_class = IronViTConfig
    base_model_prefix = "ironvit"
    # Attention always goes through F.scaled_dot_product_attention; declaring it
    # keeps `attn_implementation="sdpa"` (the transformers default) accepted.
    _supports_sdpa = True
    _no_split_modules = ["IronViTBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, IronViTModel):
            nn.init.trunc_normal_(module.register_tokens, std=std)
        elif isinstance(module, IronViTMapHead):
            nn.init.trunc_normal_(module.query, std=std)


class IronViTModel(IronViTPreTrainedModel):
    """IronViT vision encoder.

    Args:
        config: An :class:`IronViTConfig` describing the released variant.

    Example::

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "IRONVIT/IronViT-Softmax-400M",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).eval()
        outputs = model(**inputs)
        outputs.patch_features   # [num_patches, 1152]
        outputs.global_features  # [batch, 1152]
    """

    def __init__(self, config: IronViTConfig) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.patch_embed = IronViTPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))

        head_dim = config.hidden_size // config.num_heads
        self.head_dim = head_dim

        softmax_set = set(config.softmax_layer_indices)
        self.blocks = nn.ModuleList(
            IronViTBlock(config, linear=(config.attention_type == "hybrid" and i not in softmax_set))
            for i in range(config.depth)
        )

        # Same LayerNorm Qwen3-VL's patch merger applies: the feature space the
        # probes were distilled in, and what a VLM projector expects.
        if config.post_layernorm:
            self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = nn.Identity()

        self.map_head = IronViTMapHead(config) if config.num_cls_tokens else None
        self.post_init()

    # -- position helpers --------------------------------------------------

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        """RoPE inverse frequencies, recomputed from the config on every call.

        Deliberately *not* a registered buffer. ``persistent=False`` buffers are
        absent from the checkpoint, and transformers>=5 builds the module on the
        meta device before materialising it, which leaves such a buffer holding
        uninitialised memory (observed as ``-1.5e+25`` in the first slot). The
        value is eight floats; recomputing it is free and cannot go stale.
        """
        half = self.head_dim // 2
        return 1.0 / (self.config.rope_theta ** (
            torch.arange(0, half, 2, dtype=torch.float, device=device) / half
        ))

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """2-D rotary frequencies per patch, in blocked order.

        Args:
            grid_thw: ``[batch, 3]`` integer ``(t, h, w)`` patch grids.

        Returns:
            ``[seq, head_dim / 2]`` RoPE angles.
        """
        merge = self.spatial_merge_size
        device = grid_thw.device
        max_hw = max(max(h, w) for _, h, w in grid_thw.tolist())
        inv_freq = self._inv_freq(device=device)
        freqs = torch.outer(torch.arange(max_hw, device=device, dtype=inv_freq.dtype), inv_freq)
        pos_ids = torch.empty((int(grid_thw.prod(-1).sum().item()), 2), dtype=torch.long, device=device)

        offset = 0
        for frames, height, width in grid_thw.tolist():
            bh, bw = height // merge, width // merge
            # Blocked order (block_h, block_w, intra_h, intra_w): the row index
            # varies on dim 2 and the column index on dim 3.
            rows = torch.arange(bh, device=device)[:, None, None, None] * merge + torch.arange(merge, device=device)[None, None, :, None]
            cols = torch.arange(bw, device=device)[None, :, None, None] * merge + torch.arange(merge, device=device)[None, None, None, :]
            rows = rows.expand(bh, bw, merge, merge).reshape(-1)
            cols = cols.expand(bh, bw, merge, merge).reshape(-1)
            n = rows.numel()
            pos_ids[offset:offset + frames * n] = torch.stack((rows, cols), dim=-1).repeat(frames, 1)
            offset += frames * n
        return freqs[pos_ids].flatten(1)

    def pos_embed_interpolate(self, grid_thw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Bilinear interpolation of the learned 48x48 grid to each image grid.

        Args:
            grid_thw: ``[batch, 3]`` integer ``(t, h, w)`` patch grids.
            dtype: Output dtype.

        Returns:
            ``[seq, hidden_size]`` additive position embeddings in blocked order.
        """
        n = self.num_grid_per_side
        device = self.pos_embed.weight.device
        idx_list, weight_list = [[] for _ in range(4)], [[] for _ in range(4)]

        for _, h, w in grid_thw.tolist():
            h_idx, w_idx = torch.linspace(0, n - 1, h), torch.linspace(0, n - 1, w)
            h_floor, w_floor = h_idx.int(), w_idx.int()
            h_ceil = (h_floor + 1).clip(max=n - 1)
            w_ceil = (w_floor + 1).clip(max=n - 1)
            dh, dw = h_idx - h_floor, w_idx - w_floor
            base_h, base_h_ceil = h_floor * n, h_ceil * n
            idx_list[0].extend((base_h[None].T + w_floor[None]).flatten().tolist())
            idx_list[1].extend((base_h[None].T + w_ceil[None]).flatten().tolist())
            idx_list[2].extend((base_h_ceil[None].T + w_floor[None]).flatten().tolist())
            idx_list[3].extend((base_h_ceil[None].T + w_ceil[None]).flatten().tolist())
            weight_list[0].extend(((1 - dh)[None].T * (1 - dw)[None]).flatten().tolist())
            weight_list[1].extend(((1 - dh)[None].T * dw[None]).flatten().tolist())
            weight_list[2].extend((dh[None].T * (1 - dw)[None]).flatten().tolist())
            weight_list[3].extend((dh[None].T * dw[None]).flatten().tolist())

        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        embeds = self.pos_embed(idx) * weight[:, :, None]
        patch_embeds = embeds[0] + embeds[1] + embeds[2] + embeds[3]

        sizes = [h * w for _, h, w in grid_thw.tolist()]
        merge = self.spatial_merge_size
        out = []
        for embed, (frames, h, w) in zip(patch_embeds.split(sizes), grid_thw.tolist()):
            embed = embed.view(h // merge, merge, w // merge, merge, -1).permute(0, 2, 1, 3, 4).flatten(0, 3)
            out.append(embed.repeat(frames, 1))
        return torch.cat(out).to(dtype)

    # -- sequence packing --------------------------------------------------

    def _with_prefix(self, x: torch.Tensor, grid_thw: torch.Tensor, prefix_rows: torch.Tensor) -> torch.Tensor:
        """Insert ``prefix_rows`` (one per image, ``[num_prefix, dim]``) before each image's tokens."""
        chunks, offset = [], 0
        for length in grid_thw.prod(-1).tolist():
            chunks.append(prefix_rows)
            chunks.append(x[offset:offset + length])
            offset += length
        return torch.cat(chunks)

    def _without_prefix(self, x: torch.Tensor, grid_thw: torch.Tensor, num_prefix: int) -> torch.Tensor:
        """Drop ``num_prefix`` leading rows per image."""
        chunks, offset = [], 0
        for length in grid_thw.prod(-1).tolist():
            chunks.append(x[offset + num_prefix:offset + num_prefix + length])
            offset += num_prefix + length
        return torch.cat(chunks)

    # -- forward -----------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None,
                image_grid_thw: Optional[torch.Tensor] = None) -> IronViTOutput:
        """Run the encoder.

        Args:
            pixel_values: ``[num_patches, C * T * patch * patch]`` normalized
                patches in blocked order.
            grid_thw: ``[batch, 3]`` integer ``(t, h, w)`` patch grids.
            image_grid_thw: Alias of ``grid_thw``, so the processor's output
                dict can be splatted straight in: ``model(**inputs)``.

        Returns:
            An :class:`IronViTOutput` with ``patch_features`` and, when the
            config enables the MAP head, ``global_features``.

        Raises:
            ValueError: If neither ``grid_thw`` nor ``image_grid_thw`` is given.
        """
        if grid_thw is None:
            grid_thw = image_grid_thw
        if grid_thw is None:
            raise ValueError("IronViTModel.forward needs grid_thw / image_grid_thw")
        grid_thw = grid_thw.to(pixel_values.device, torch.long)
        if grid_thw.ndim == 3:
            grid_thw = grid_thw.squeeze(1)
        dtype = self.patch_embed.proj.weight.dtype

        hidden = self.patch_embed(pixel_values) + self.pos_embed_interpolate(grid_thw, dtype)
        rope = self.rot_pos_emb(grid_thw)
        rope_emb = torch.cat((rope, rope), dim=-1)

        num_prefix = self.config.num_register_tokens
        if num_prefix:
            prefix = self.register_tokens.to(hidden.dtype)[0]
            hidden = self._with_prefix(hidden, grid_thw, prefix)
            # Registers are position-free: zero RoPE (cos=1, sin=0).
            rope_emb = self._with_prefix(rope_emb, grid_thw, rope_emb.new_zeros(num_prefix, rope_emb.shape[-1]))

        cos, sin = rope_emb.cos().to(hidden.dtype), rope_emb.sin().to(hidden.dtype)
        lengths = [length + num_prefix for length in grid_thw.prod(-1).tolist()]
        cu_seqlens = F.pad(torch.tensor(lengths, dtype=torch.int32).cumsum(0).to(torch.int32), (1, 0), value=0)
        cu_seqlens = cu_seqlens.to(hidden.device)

        for block in self.blocks:
            hidden = block(hidden, cu_seqlens, cos, sin)

        if num_prefix:
            hidden = self._without_prefix(hidden, grid_thw, num_prefix)

        patch_features = self.post_layernorm(hidden)
        return IronViTOutput(
            patch_features=patch_features,
            global_features=self.map_head(patch_features, grid_thw) if self.map_head is not None else None,
        )

    # -- convenience -------------------------------------------------------

    def to_raster_order(self, tokens: torch.Tensor, grid_thw: torch.Tensor) -> List[torch.Tensor]:
        """Reorder blocked patch tokens to row-major per-image grids.

        Args:
            tokens: ``[num_patches, dim]`` tokens in blocked order.
            grid_thw: ``[batch, 3]`` integer ``(t, h, w)`` patch grids.

        Returns:
            One tensor per image: ``[gh, gw, dim]`` for a single frame, or
            ``[t, gh, gw, dim]`` when the grid spans more than one temporal
            patch.
        """
        merge = self.spatial_merge_size
        grids = []
        offset = 0
        for frames, gh, gw in grid_thw.tolist():
            n = frames * gh * gw
            chunk = tokens[offset:offset + n]
            chunk = chunk.view(frames, gh // merge, gw // merge, merge, merge, -1)
            chunk = chunk.permute(0, 1, 3, 2, 4, 5).reshape(frames, gh, gw, -1)
            grids.append(chunk[0] if frames == 1 else chunk)
            offset += n
        return grids
