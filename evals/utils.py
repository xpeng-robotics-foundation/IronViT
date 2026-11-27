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
"""Shared helpers for the frozen-backbone probes.

Every probe in this directory follows the same recipe: an image, sampled over a
fixed resolution grid, is turned into ``pixel_values`` + ``image_grid_thw``, the
released IronViT model produces ``patch_features`` in blocked order, and the
probe reorders them to a raster ``[gh, gw, D]`` grid.

The one thing worth knowing: IronViT keeps Qwen3-VL's *blocked* token order
``(block_h, block_w, intra_h, intra_w)``, and :func:`to_raster` converts it.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T


def build_eval_transform(image_size: int) -> T.Compose:
    """ImageNet linear-probe transform: resize the short side, then center-crop.

    Args:
        image_size: Side length of the square crop, in pixels.

    Returns:
        A transform producing a ``[3, image_size, image_size]`` float tensor in
        ``[0, 1]``. ``256 / 224`` is the standard ratio; scaling it keeps the
        crop fraction of the image identical at any ``image_size``.
    """
    return T.Compose(
        [
            T.Resize(int(image_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


def load_backbone(model_id: str, device: torch.device, frozen: bool = True):
    """Load a released IronViT checkpoint in bfloat16 and eval mode.

    Args:
        model_id: HF repo id or local checkpoint directory.
        device: Device to place the model on.
        frozen: Set ``requires_grad=False`` on every parameter.

    Returns:
        The model, ready to be called inside ``torch.no_grad()``.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(device).eval()
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


@torch.no_grad()
def encode(model, images_01: torch.Tensor, patch_size: int = 16,
           temporal_patch_size: int = 2, merge_size: int = 2,
           device: torch.device | None = None,
           autocast: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of ``[0, 1]`` images with the frozen backbone.

    Args:
        model: An :class:`IronViTModel` in eval mode.
        images_01: ``[B, 3, H, W]`` images in ``[0, 1]``; the model's own
            mean = std = 0.5 normalization is applied here.
        patch_size, temporal_patch_size, merge_size: Must match the model.
        device: Device to run on. Defaults to the model's own device.
        autocast: Run the forward under ``torch.autocast(bfloat16)``. The
            weights are already bf16, but autocast additionally keeps the
            reductions (LayerNorm, softmax, the MAP normaliser) in fp32.
            The probe that produced the reported dense numbers enables this;
            the global probes leave it off, so they match their recipe.

    Returns:
        A ``(tokens, pooled)`` pair:

        * ``tokens``: ``[B, gh * gw, D]`` patch features in **raster** order.
        * ``pooled``: ``[B, D]`` MAP-pooled global features.

        Both are float32, so the probe head is dtype-agnostic.

    Raises:
        AssertionError: If the patch grid is not divisible by ``merge_size``;
            the blocked token order cannot represent it.
    """
    device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch, _, height, width = images_01.shape
    gh, gw = height // patch_size, width // patch_size
    assert gh % merge_size == 0 and gw % merge_size == 0, (
        f"grid {gh}x{gw} must be divisible by spatial_merge_size={merge_size}"
    )

    normed = ((images_01.to(device) - 0.5) / 0.5).to(dtype)
    # [B, 3, H, W] -> duplicate over time to satisfy the Conv3d temporal kernel
    # -> [B, gh, gw, C, T, p, p] -> blocks of (merge, merge) -> flat patches.
    frames = normed.unsqueeze(2).repeat(1, 1, temporal_patch_size, 1, 1)
    tiles = frames.view(batch, 3, temporal_patch_size, gh, patch_size, gw, patch_size)
    tiles = tiles.permute(0, 3, 5, 1, 2, 4, 6).contiguous()  # [B, gh, gw, C, T, p, p]
    m = merge_size
    tiles = tiles.view(batch, gh // m, m, gw // m, m, 3, temporal_patch_size, patch_size, patch_size)
    tiles = tiles.permute(0, 1, 3, 2, 4, 5, 6, 7, 8).contiguous()  # blocked order
    pixel_values = tiles.reshape(batch * gh * gw, 3 * temporal_patch_size * patch_size * patch_size)

    grid_thw = torch.tensor([[1, gh, gw]] * batch, dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=autocast and device.type == "cuda"):
        out = model(pixel_values, grid_thw)
    tokens = to_raster(out.patch_features, grid_thw, gh, gw, m)
    return tokens.float(), out.global_features.float()


def to_raster(tokens: torch.Tensor, grid_thw: torch.Tensor, gh: int, gw: int,
              merge_size: int = 2) -> torch.Tensor:
    """Convert blocked patch tokens to a raster grid.

    Args:
        tokens: ``[B * gh * gw, D]`` patch tokens in blocked order.
        grid_thw: ``[B, 3]`` integer grids; only the batch size is read here.
        gh, gw: Patch-grid height and width of every image in the batch.
        merge_size: Block edge used by the token order.

    Returns:
        ``[B, gh * gw, D]`` tokens in row-major order.

    Note:
        The inverse of the blocked patchify: tokens are grouped
        ``(block_h, block_w, intra_h, intra_w)``, so expand them back to
        ``(gh, gw)`` and flatten spatially.
    """
    batch = grid_thw.shape[0]
    dim = tokens.shape[-1]
    x = tokens.view(batch, gh // merge_size, gw // merge_size, merge_size, merge_size, dim)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, bh, m, bw, m, D]
    return x.reshape(batch, gh * gw, dim)


@torch.no_grad()
def extract_features(model, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the backbone over ``loader`` and collect pooled features and labels.

    Args:
        model: Frozen IronViT model in eval mode.
        loader: Yields ``(images, targets)`` batches.
        device: Device the model lives on.

    Returns:
        ``(features, labels)`` of shape ``[N, D]`` and ``[N]`` on the CPU.
    """
    feats, labels = [], []
    for index, (images, targets) in enumerate(loader):
        _, pooled = encode(model, images, device=device)
        feats.append(pooled.float().cpu())
        labels.append(targets)
        if index % 50 == 0:
            print(f"  {index * loader.batch_size + len(targets)} images", flush=True)
    return torch.cat(feats), torch.cat(labels)


def cached_features(model, loader: DataLoader, split: str, cache_dir: str,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load features from ``cache_dir`` if possible, otherwise extract and save them.

    Args:
        model: Frozen IronViT model in eval mode.
        loader: Yields ``(images, targets)`` batches for ``split``.
        split: Cache key, e.g. ``"train"`` or ``"val"``.
        cache_dir: Directory holding ``<split>.pt``.
        device: Device the model lives on.

    Returns:
        ``(features, labels)`` of shape ``[N, D]`` and ``[N]``.
    """
    path = os.path.join(cache_dir, f"{split}.pt")
    if os.path.isfile(path):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["features"].shape[0] == len(loader.dataset):
            print(f"  [{split}] loaded {blob['features'].shape[0]} cached features")
            return blob["features"], blob["labels"]
    print(f"  [{split}] extracting {len(loader.dataset)} features")
    feats, labels = extract_features(model, loader, device)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save({"features": feats, "labels": labels}, tmp)
    os.replace(tmp, path)
    return feats, labels


def _cache_dir_for(cache_dir: str, model_id: str, image_size: int) -> str:
    """Build a collision-resistant cache directory for ``model_id`` and ``image_size``."""
    key_src = os.path.realpath(model_id) if os.path.exists(model_id) else model_id.rstrip("/")
    key_hash = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:8]
    return os.path.join(cache_dir, f"{os.path.basename(model_id.rstrip('/'))}_{image_size}_{key_hash}")


def imagenet_features(model_id: str, data_root: str, image_size: int, batch_size: int,
                      num_workers: int, cache_dir: str, device: torch.device) -> Dict:
    """Encode ImageNet-1k once and return both splits, from cache when available.

    Args:
        model_id: HF repo id or local checkpoint directory.
        data_root: ``ImageFolder`` root holding ``train/`` and ``val/``.
        image_size: Square crop fed to the backbone.
        batch_size: Backbone batch size during extraction.
        num_workers: DataLoader workers.
        cache_dir: Root of the feature cache; the model name is appended.
        device: Device to run on.

    Returns:
        ``{"model", "train_features", "train_labels", "val_features",
        "val_labels", "num_classes"}``, with features as float32 CPU tensors.
    """
    model = load_backbone(model_id, device)
    transform = build_eval_transform(image_size)
    train_set = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
    val_set = datasets.ImageFolder(os.path.join(data_root, "val"), transform=transform)
    if train_set.classes != val_set.classes:
        raise ValueError("ImageNet train/ and val/ class folders do not match")
    print(f"ImageNet-1k: {len(train_set)} train / {len(val_set)} val, {len(train_set.classes)} classes")

    cache = _cache_dir_for(cache_dir, model_id, image_size)
    splits = {}
    for split, dataset in (("train", train_set), ("val", val_set)):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        splits[split] = cached_features(model, loader, split, cache, device)

    return {
        "model": model,
        "train_features": splits["train"][0],
        "train_labels": splits["train"][1],
        "val_features": splits["val"][0],
        "val_labels": splits["val"][1],
        "num_classes": len(train_set.classes),
    }


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Top-``k`` accuracy of ``logits`` against ``target``, in ``[0, 1]``."""
    k = min(k, logits.shape[-1])
    return (logits.topk(k, dim=-1).indices == target.unsqueeze(1)).any(dim=1).float().mean().item()


__all__ = [
    "build_eval_transform",
    "cached_features",
    "encode",
    "extract_features",
    "imagenet_features",
    "load_backbone",
    "to_raster",
    "topk_accuracy",
]
