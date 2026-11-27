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
"""NYUv2 monocular-depth linear probe (Eigen split, d_max = 10 m).

Frozen patch features are concatenated with their spatial mean broadcast over
the grid, upsampled 4x, and fed to a ``Conv1x1 -> 256 bins`` head. Depth is
treated as classification into 256 bins over ``[1e-3, 10]`` m: the bin logits are
bilinearly resized to the ground-truth resolution and then collapsed into a
prediction by a softmax-weighted mean over uniformly spaced bin centres.
Training uses SigLoss + 0.5 * gradient loss with AdamW 1e-4 / weight decay 0.01
for 50k iterations, after a 16k linear warm-up followed by cosine decay.
Evaluation is on the native 480x640 frame with the Eigen crop
``[45:471, 41:601]`` restricting the metric region and flip TTA; the reported
RMSE is the one at the checkpoint with the highest delta-1 accuracy, as in
Appendix A.3.4 of the paper.

Example::

    python evals/depth.py \\
        --model IRONVIT/IronViT-Softmax-400M \\
        --data-root /path/to/nyu

The data root follows the BTS / Monocular-Depth-Estimation-Toolbox layout:
``<data-root>/nyu_{train,test}.txt``, each line ``<rgb> <depth> <focal>``, with
uint16 depth PNGs where ``depth_m = value / 1000``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import utils

D_MIN, D_MAX = 1e-3, 10.0
NUM_BINS = 256
EIGEN_CROP = (45, 471, 41, 601)  # top, bottom, left, right on the 480x640 frame
PATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="HF repo id or local checkpoint directory")
    parser.add_argument("--data-root", required=True,
                        help="directory holding nyu_train.txt / nyu_test.txt")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="training batch size")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50_000,
                        help="head training iterations")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="AdamW learning rate after warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay")
    parser.add_argument("--warmup-iterations", type=int, default=16_000,
                        help="linear warm-up before the cosine decay")
    parser.add_argument("--eval-every", type=int, default=10_000,
                        help="validate every N iterations; the best delta-1 is reported")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the head initialisation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _read_split_file(path: str) -> List[Tuple[str, str]]:
    """Parse ``<rgb> <depth> <focal>`` lines into data-root-relative paths.

    Args:
        path: A BTS split file.

    Returns:
        ``(rgb, depth)`` pairs, relative to the data root.

    Note:
        ``nyu_train.txt`` writes leading slashes (``/kitchen_0028b/...``) while
        ``nyu_test.txt`` does not; stripping them keeps ``os.path.join`` from
        discarding the data root.
    """
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0].lstrip("/"), parts[1].lstrip("/")))
    return entries


class NYUv2(Dataset):
    """BTS-format NYUv2 pairs at the native 480x640 resolution.

    Args:
        data_root: Directory holding ``nyu_{split}.txt``.
        split: ``"train"`` or ``"test"``.
        mode: ``"train"`` enables the DINOv2 augmentation pipeline.
    """

    def __init__(self, data_root: str, split: str, mode: str) -> None:
        self.entries = _read_split_file(os.path.join(data_root, f"nyu_{split}.txt"))
        self.rgb = [os.path.join(data_root, rgb) for rgb, _ in self.entries]
        self.depth = [os.path.join(data_root, depth) for _, depth in self.entries]
        self.mode = mode

    def __len__(self) -> int:
        return len(self.rgb)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, depth)`` in ``[0, 1]`` and metres."""
        rgb = np.asarray(Image.open(self.rgb[index]).convert("RGB"))
        depth = np.asarray(Image.open(self.depth[index]), dtype=np.float32) / 1000.0
        # np.array copies: PIL hands back a read-only buffer, which
        # torch.from_numpy refuses to wrap without a warning.
        image = TF.to_tensor(np.array(rgb))
        target = torch.from_numpy(np.array(depth, dtype=np.float32))[None]  # [1, H, W]

        if self.mode == "train":
            # Per-sample RNG mixed with the worker id, so a sample is
            # augmented identically regardless of which worker loads it.
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else 0
            rng = random.Random((worker_id * 1_000_003) ^ (index * 2654435761))
            image, target = _augment(image, target[0], rng)
            target = target[None]
        return image, target[0]


def _augment(image: torch.Tensor, target: torch.Tensor, rng: random.Random,
             crop_size: Tuple[int, int] = (480, 640)) -> Tuple[torch.Tensor, torch.Tensor]:
    """DINOv2 NYU train pipeline: rotate, flip, crop, colour jitter.

    The colour step is not `adjust_saturation`: the reference draws one gamma,
    one brightness and three per-channel multipliers and applies them
    directly, which is a different operation from a saturation adjustment on
    the HSV wheel. Rotating RGB bilinearly and depth with nearest-neighbour
    keeps depth discontinuities from being smeared; the corners the rotation
    leaves empty become 0 and drop out of the loss.
    """
    if rng.random() < 0.5:
        angle = rng.uniform(-2.5, 2.5)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        target = TF.rotate(target[None], angle, interpolation=TF.InterpolationMode.NEAREST,
                           fill=0.0)[0]
    if rng.random() < 0.5:
        image, target = TF.hflip(image), TF.hflip(target)

    # RandomCrop to crop_size; a no-op at the native 480x640 with the TIPS
    # protocol, but it keeps the random stream identical to the reference.
    crop_h, crop_w = crop_size
    height, width = image.shape[-2:]
    top = 0 if height == crop_h else rng.randint(0, height - crop_h)
    left = 0 if width == crop_w else rng.randint(0, width - crop_w)
    image = image[:, top:top + crop_h, left:left + crop_w]
    target = target[top:top + crop_h, left:left + crop_w]

    if rng.random() < 0.5:
        gamma = rng.uniform(0.9, 1.1)
        brightness = rng.uniform(0.75, 1.25)
        colour = [rng.uniform(0.9, 1.1) for _ in range(3)]
        image = image.clamp_min(0.0).pow(gamma) * brightness
        channel = torch.tensor(colour, dtype=image.dtype, device=image.device).view(3, 1, 1)
        image = (image * channel).clamp_(0.0, 1.0)
    return image, target


class DepthHead(nn.Module):
    """``[B, gh * gw, 2D]`` -> 4x upsample -> ``Conv1x1 -> 256 bins``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2 * dim, NUM_BINS, kernel_size=1)
        nn.init.trunc_normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, tokens: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """Map raster patch features to per-bin depth logits.

        The global vector that is concatenated in is the mean of the patch
        tokens themselves, which is what Appendix A.3.4 specifies.

        Args:
            tokens: ``[B, gh * gw, D]`` raster patch features.
            grid: ``(gh, gw)`` patch-grid shape.

        Returns:
            ``[B, 256, 4 * gh, 4 * gw]`` bin logits at a quarter of the pixel
            resolution.
        """
        gh, gw = grid
        batch, _, dim = tokens.shape
        dense = tokens.transpose(1, 2).reshape(batch, dim, gh, gw)
        pooled = dense.mean(dim=(2, 3))
        dense = torch.cat([dense, pooled[:, :, None, None].expand(-1, -1, gh, gw)], dim=1)
        return self.conv(F.interpolate(dense, scale_factor=4, mode="bilinear", align_corners=False))


def predict_depth(logits: torch.Tensor) -> torch.Tensor:
    """Softmax-weighted bin centres -> metres, ``[B, H, W]``.

    The bins are the ``NUM_BINS`` equal intervals spanning ``[D_MIN, D_MAX]``;
    the value each one contributes is its midpoint, as in Appendix A.3.4.
    """
    edges = torch.linspace(D_MIN, D_MAX, NUM_BINS + 1, device=logits.device, dtype=logits.dtype)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return (F.softmax(logits, dim=1) * centres.view(1, -1, 1, 1)).sum(dim=1)


def _decode(logits: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bin logits at head resolution -> depth ``[B, size[0], size[1]]`` in metres.

    The resize to ground-truth resolution happens on the 256-channel logits,
    *before* the softmax collapses them. Doing it after is not the same
    operation: the weighted sum is non-linear in the bins, so interpolating a
    single depth map and interpolating each bin's probability then combining
    give different answers. This order is the one that produced the number in
    the model card.

    Args:
        logits: ``[B, NUM_BINS, 4 * gh, 4 * gw]`` from :class:`DepthHead`.
        size: Target ``(height, width)``.
    """
    if tuple(logits.shape[-2:]) != tuple(size):
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return predict_depth(logits)


def sig_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
             warm: bool) -> torch.Tensor:
    """Scale-invariant log loss.

    Args:
        prediction, target: ``[B, H, W]`` depth in metres.
        valid: Boolean mask of pixels that take part in the loss.
        warm: Use the plain mean-squared log error (the first 100 iterations),
            as in BTS, instead of the scale-invariant form.

    Returns:
        ``sqrt(var(g) + 0.15 * mean(g)^2)`` with ``g = log(prediction) - log(target)``.
    """
    g = torch.log(prediction[valid].clamp_min(1e-3)) - torch.log(target[valid].clamp_min(1e-3))
    if warm:
        return torch.sqrt(g.pow(2).mean())
    # ``var + lambda * mean^2`` rather than ``mean^2 - lambda * mean^2``: the
    # two are algebraically equal, but this form cannot go negative from
    # cancellation when ``mean(g)`` is large.
    return torch.sqrt((g.var(unbiased=False) + 0.15 * g.mean().pow(2)).clamp_min(0.0))


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """L1 on the spatial gradient of (predicted depth, GT depth).

    Encourages depth edges to line up with image edges, at full resolution
    only -- the reference implementation is single-scale, and adding coarser
    scales would change the loss magnitude and so the effective weight
    against SigLoss.

    Args:
        prediction, target: ``[B, H, W]`` depth in metres.
        valid: Boolean mask of valid pixels.

    Returns:
        The horizontal and vertical mean absolute gradient residuals, summed.
        A pixel pair counts only when both of its endpoints are valid.
    """
    diff_pred_x = prediction[:, :, 1:] - prediction[:, :, :-1]
    diff_pred_y = prediction[:, 1:, :] - prediction[:, :-1, :]
    diff_gt_x = target[:, :, 1:] - target[:, :, :-1]
    diff_gt_y = target[:, 1:, :] - target[:, :-1, :]
    mask_x = valid[:, :, 1:] & valid[:, :, :-1]
    mask_y = valid[:, 1:, :] & valid[:, :-1, :]
    loss_x = (diff_pred_x - diff_gt_x).abs()
    loss_y = (diff_pred_y - diff_gt_y).abs()
    nx = mask_x.sum().clamp_min(1)
    ny = mask_y.sum().clamp_min(1)
    return loss_x[mask_x].sum() / nx + loss_y[mask_y].sum() / ny


@torch.no_grad()
def evaluate(model, head: DepthHead, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Run the standard NYUv2 depth metrics on the Eigen crop with flip TTA.

    Args:
        model: Frozen IronViT model in eval mode.
        head: Trained :class:`DepthHead`.
        loader: Yields ``(image, depth)`` batches at native resolution.
        device: Device to run on.

    Returns:
        ``{"rmse", "abs_rel", "log10", "d1", "d2", "d3"}``.
    """
    head.eval()
    totals = {k: 0.0 for k in ("rmse", "abs_rel", "log10", "d1", "d2", "d3")}
    count = 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        tokens, _ = utils.encode(model, image, device=device)
        gh, gw = image.shape[-2] // PATCH_SIZE, image.shape[-1] // PATCH_SIZE
        # Flip TTA: encode the mirrored image, decode it, and mirror the depth
        # map back before averaging the two views. Exactly one mirror per
        # view: un-mirroring the tokens and then the decoded depth cancels
        # out, leaving a second view that is misaligned with the first and
        # with the labels. Averaging depth maps rather than logits matters
        # once the softmax has been applied; the resize to ground-truth
        # resolution happens per view, before averaging, as in DINOv2's
        # MultiScaleFlipAug.
        def view(features: torch.Tensor) -> torch.Tensor:
            return _decode(head(features, (gh, gw)), target.shape[-2:])

        flipped, _ = utils.encode(model, image.flip(-1), device=device)
        prediction = 0.5 * (view(tokens) + view(flipped).flip(-1))

        top, bottom, left, right = EIGEN_CROP
        eigen = torch.zeros_like(target, dtype=torch.bool)
        eigen[..., top:bottom, left:right] = True
        for i in range(target.shape[0]):
            p = prediction[i].clamp(D_MIN, D_MAX)
            g = target[i]
            valid = (g > D_MIN) & (g < D_MAX) & eigen[i]
            if not valid.any():
                continue
            error = (p - g)[valid]
            ratio = torch.maximum(p[valid] / g[valid], g[valid] / p[valid])
            totals["rmse"] += error.pow(2).mean().sqrt().item()
            totals["abs_rel"] += (error.abs() / g[valid]).mean().item()
            totals["log10"] += (torch.log10(p[valid]) - torch.log10(g[valid])).abs().mean().item()
            totals["d1"] += (ratio < 1.25).float().mean().item()
            totals["d2"] += (ratio < 1.25 ** 2).float().mean().item()
            totals["d3"] += (ratio < 1.25 ** 3).float().mean().item()
            count += 1
    return {k: v / max(1, count) for k, v in totals.items()}


def lr_at(step: int, args: argparse.Namespace) -> float:
    """16k linear warm-up then cosine decay to zero, as in Appendix A.3.4.

    Warm-up starts at ``1e-6 * lr`` rather than at 0, matching mmseg's
    ``LinearLR`` with ``start_factor=1e-6``.
    """
    warmup_ratio = 1e-6
    if step < args.warmup_iterations:
        progress = (step + 1) / max(1, args.warmup_iterations)
        return args.lr * (warmup_ratio + (1.0 - warmup_ratio) * progress)
    progress = (step - args.warmup_iterations) / max(1, args.iterations - args.warmup_iterations)
    return 0.5 * args.lr * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    """Train the depth head and report the metrics at the best delta-1 checkpoint."""
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = utils.load_backbone(args.model, device)

    train_set = NYUv2(args.data_root, "train", "train")
    val_set = NYUv2(args.data_root, "test", "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"NYUv2: {len(train_set)} train / {len(val_set)} test; "
          f"iters={args.iterations} warmup={args.warmup_iterations} "
          f"batch={args.batch_size} lr={args.lr} wd={args.weight_decay}", flush=True)

    head = DepthHead(model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, started = 0, time.time()
    head.train()

    best: Dict[str, float] = {}
    best_step = 0
    while step < args.iterations:
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            lr = lr_at(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            tokens, _ = utils.encode(model, image, device=device)
            gh, gw = image.shape[-2] // PATCH_SIZE, image.shape[-1] // PATCH_SIZE
            prediction = _decode(head(tokens, (gh, gw)), target.shape[-2:])
            valid = (target > D_MIN) & (target < D_MAX)
            loss = sig_loss(prediction, target, valid, warm=step < 100)
            loss = loss + 0.5 * gradient_loss(prediction, target, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 35.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                print(f"  iter {step}/{args.iterations} loss={loss.item():.4f} "
                      f"lr={lr:.2e} ({time.time() - started:.0f}s)", flush=True)
            if step % args.eval_every == 0 or step == args.iterations:
                results = evaluate(model, head, val_loader, device)
                print("  iter %d EVAL " % step
                      + "  ".join(f"{k}={v:.4f}" for k, v in results.items()), flush=True)
                if results["d1"] > best.get("d1", -1.0):
                    best, best_step = results, step
                head.train()
            if step >= args.iterations:
                break

    print(f"\nNYUv2 depth (iter {best_step}): "
          + "  ".join(f"{k}={v:.4f}" for k, v in best.items()))


if __name__ == "__main__":
    main()
