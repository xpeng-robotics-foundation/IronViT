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
"""ADE20K semantic-segmentation linear probe.

Frozen backbone. The global feature is broadcast over the patch grid and
concatenated to every patch token; the head is ``BatchNorm2d(2D) -> Conv1x1``.
The head trains for 40k iterations (AdamW 1e-3, weight decay 1e-4, 1.5k linear
warm-up then linear decay to zero) on 512x512 crops drawn with random short-side
scaling in [0.5, 2.0], horizontal flips and photometric distortion, and is
evaluated with a 512/341 sliding window. The reported number is the highest
validation mIoU, as in Appendix A.3.4 of the paper.

Example::

    python evals/segmentation.py \\
        --model IRONVIT/IronViT-Softmax-400M \\
        --data-root /path/to/ADE20K

The data root must hold
``<data-root>/ADEChallengeData2016/{images,annotations}/{training,validation}``.
Download: https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import utils

NUM_CLASSES = 150
IGNORE_INDEX = -1


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="HF repo id or local checkpoint directory")
    parser.add_argument("--data-root", required=True,
                        help="directory holding ADEChallengeData2016/")
    parser.add_argument("--crop-size", type=int, default=512,
                        help="training crop and sliding-window size")
    parser.add_argument("--eval-stride", type=int, default=341,
                        help="sliding-window stride at evaluation (2/3 of the crop)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="training batch size")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40_000,
                        help="head training iterations")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="peak AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--warmup-iterations", type=int, default=1500,
                        help="linear warm-up before the linear decay to zero")
    parser.add_argument("--eval-every", type=int, default=5000,
                        help="validate every N iterations; the best mIoU is reported")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the head initialisation and augmentation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class ADE20K(Dataset):
    """ADE20K images and masks; class 0 is "unlabeled" and becomes ``-1``.

    Args:
        data_root: Directory holding ``ADEChallengeData2016/``.
        split: ``"training"`` or ``"validation"``.
        mode: ``"train"`` applies the mmseg train pipeline, anything else the
            test pipeline.
        crop_size: Side length of the square training crop, which is also the
            short-edge target of both pipelines.
        seed: Seed mixed into the per-sample RNG, so augmentation is
            deterministic in the sample index.
    """

    def __init__(self, data_root: str, split: str, mode: str, crop_size: int,
                 seed: int = 0) -> None:
        root = os.path.join(data_root, "ADEChallengeData2016")
        images = sorted(f for f in os.listdir(os.path.join(root, "images", split)) if f.endswith(".jpg"))
        self.images = [os.path.join(root, "images", split, f) for f in images]
        self.masks = [os.path.join(root, "annotations", split, os.path.splitext(f)[0] + ".png") for f in images]
        self.mode = mode
        self.crop_size = crop_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image, mask)``; the mask is shifted so 0 means "unlabeled"."""
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        # Seed mix from the reference implementation: the worker id takes part,
        # so a sample augmented in worker 3 is not augmented like the same
        # sample in worker 0.
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random((worker_id * 1_000_003) ^ (index * 2654435761) ^ self.seed)

        if self.mode == "train":
            # mmseg train pipeline: random short-side scaling, random crop with
            # cat_max_ratio rejection, flip, then photometric distortion.
            scale = rng.uniform(0.5, 2.0)
            image, mask = _resize_short_edge(image, mask, int(round(self.crop_size * scale)))
            image, mask = _crop(image, mask, self.crop_size, rng)
            if rng.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            image = TF.to_tensor(image)
            # mmseg wraps PhotoMetricDistortion in a RandomApply with p=0.5, so
            # half the samples get no colour jitter at all; each individual op
            # inside it is drawn at p=0.5 as well.
            if rng.random() < 0.5:
                image = _photometric_distortion(image, rng)
            return image, torch.from_numpy(np.array(mask, dtype=np.int64)) - 1

        # mmseg test pipeline: fixed short side, no crop (sliding window).
        image, mask = _resize_short_edge(image, mask, self.crop_size)
        return TF.to_tensor(image), torch.from_numpy(np.array(mask, dtype=np.int64)) - 1


def _resize_short_edge(image: Image.Image, mask: Image.Image,
                       short: int) -> Tuple[Image.Image, Image.Image]:
    """Resize a pair so its short side is ``short``, keeping the aspect ratio."""
    width, height = image.size
    if height < width:
        size = (int(round(width * short / height)), short)
    else:
        size = (short, int(round(height * short / width)))
    return (image.resize(size, Image.BICUBIC), mask.resize(size, Image.NEAREST))


def _crop(image: Image.Image, mask: Image.Image, size: int,
          rng: random.Random, cat_max_ratio: float = 0.75,
          max_tries: int = 10) -> Tuple[Image.Image, Image.Image]:
    """Random square crop, re-rolled when one class dominates it.

    Args:
        image, mask: The resized pair.
        size: Side length of the square crop.
        rng: Per-sample random source.
        cat_max_ratio: Reject a crop in which one class covers more than this
            fraction of the labelled pixels; matches mmseg's ``RandomCrop`` and
            DINOv2's ADE20K recipe.
        max_tries: Give up and accept the last candidate after this many tries.
    """
    width, height = image.size
    if width < size or height < size:  # pad up before cropping
        pad_w, pad_h = max(0, size - width), max(0, size - height)
        image = TF.pad(image, (0, 0, pad_w, pad_h))
        mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=0)
        width, height = image.size

    mask_np = np.array(mask, dtype=np.int64)
    top = left = 0
    for _ in range(max_tries):
        # Draw top before left, and skip the draw on an axis with no freedom,
        # so the random stream matches the reference implementation.
        top = 0 if height == size else rng.randint(0, height - size)
        left = 0 if width == size else rng.randint(0, width - size)
        patch = mask_np[top:top + size, left:left + size]
        labelled = patch[patch != 0]  # 0 is "unlabeled" before the shift
        if labelled.size == 0:
            continue
        counts = np.bincount(labelled, minlength=256)
        if counts.max() / labelled.size < cat_max_ratio:
            break

    box = (left, top, left + size, top + size)
    return image.crop(box), mask.crop(box)


def _photometric_distortion(image: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """mmseg's ``PhotoMetricDistortion`` on a ``[3, H, W]`` tensor in ``[0, 1]``.

    The operations and the number of random draws they consume follow mmseg
    exactly: brightness, then a *contrast-first* coin, then saturation, hue,
    contrast again on the other branch, and finally a channel swap. Each
    operation is applied with probability 0.5, as in Appendix A.3.4.

    The draw count matters: it decides which draws saturation and hue receive,
    so a version that always applies contrast in the same place augments with a
    different distribution even though every marginal is still 0.5.
    """
    image = image.clamp(0.0, 1.0)
    # 1. Brightness: additive shift in [-32/255, +32/255].
    if rng.random() < 0.5:
        image = image + (rng.random() * 2 - 1) * (32.0 / 255.0)
    # 2. Contrast is applied before the colour ops half the time, after them
    #    the other half -- mirroring mmseg's ``contrast_first`` coin.
    contrast_first = rng.random() < 0.5
    if contrast_first and rng.random() < 0.5:
        image = image * rng.uniform(0.5, 1.5)
    # 3. Saturation.
    if rng.random() < 0.5:
        image = TF.adjust_saturation(image.clamp(0.0, 1.0), rng.uniform(0.5, 1.5))
    # 4. Hue.
    if rng.random() < 0.5:
        image = TF.adjust_hue(image.clamp(0.0, 1.0), (rng.random() * 2 - 1) * (18.0 / 360.0))
    # 5. Contrast again, on the branch that did not apply it above.
    if not contrast_first and rng.random() < 0.5:
        image = image * rng.uniform(0.5, 1.5)
    # 6. Random channel swap.
    if rng.random() < 0.5:
        order = [0, 1, 2]
        rng.shuffle(order)
        image = image[order]
    return image.clamp(0.0, 1.0)


class SegHead(nn.Module):
    """``BatchNorm2d(2D) -> Conv1x1(2D, 150)`` over patch tokens + global feature.

    Args:
        dim: Width of the *patch* features; the head sees ``2 * dim`` channels
            once the global vector is concatenated.
        num_classes: Number of ADE20K classes.
    """

    def __init__(self, dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(2 * dim, eps=1e-5)
        self.conv = nn.Conv2d(2 * dim, num_classes, kernel_size=1)
        # BatchNorm already centres and scales the inputs, so a small
        # truncated-normal weight keeps the initial loss near ln(num_classes).
        nn.init.trunc_normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor,
                grid: Tuple[int, int]) -> torch.Tensor:
        """Map patch tokens and their global vector to per-patch class logits.

        Args:
            tokens: ``[B, gh * gw, D]`` raster patch features.
            pooled: ``[B, D]`` global feature, broadcast over the patch grid.
            grid: ``(gh, gw)`` patch-grid shape.

        Returns:
            ``[B, num_classes, gh, gw]`` logits.
        """
        gh, gw = grid
        batch, _, dim = tokens.shape
        x = tokens.transpose(1, 2).reshape(batch, dim, gh, gw)
        x = torch.cat([x, pooled[:, :, None, None].expand(-1, -1, gh, gw)], dim=1)
        return self.conv(self.norm(x))


def confusion_update(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> None:
    """Accumulate a ``[C, C]`` confusion matrix from flat predictions and labels.

    Args:
        confusion: ``[C, C]`` matrix to add into, indexed ``[true, pred]``.
        logits: ``[B, C, H, W]`` class scores.
        target: ``[B, H, W]`` labels; ``IGNORE_INDEX`` entries are dropped.
    """
    pred = logits.argmax(dim=1).reshape(-1)
    target = target.reshape(-1)
    keep = target != IGNORE_INDEX
    indices = target[keep] * NUM_CLASSES + pred[keep]
    confusion += torch.bincount(indices, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def mean_iou(confusion: torch.Tensor) -> float:
    """Mean IoU over the classes that appear in ``confusion``, in ``[0, 1]``."""
    inter = torch.diag(confusion).float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - inter
    valid = union > 0
    return (inter[valid] / union[valid]).mean().item()


def _window_starts(length: int, crop: int, stride: int) -> List[int]:
    """Sliding-window offsets, with the last window flush against the edge.

    Args:
        length: Extent being covered, in pixels.
        crop: Window size.
        stride: Distance between consecutive windows.

    Returns:
        Offsets at which the window starts.

    Note:
        Without the final flush window the stride-341 grid leaves up to
        ``stride - 1`` pixels of a 640-wide image uncovered, and hence
        unnormalised in ``logits_sum / count``.
    """
    last = max(0, length - crop)
    starts = list(range(0, last, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def slide_evaluate(model, head: SegHead, loader: DataLoader, crop: int, stride: int,
                   device: torch.device, label: str = "") -> float:
    """Sliding-window inference, averaged over overlapping windows.

    Args:
        model: Frozen IronViT model in eval mode.
        head: Trained :class:`SegHead`.
        loader: Yields ``(image, mask)`` pairs of variable size, one image at a
            time.
        crop: Window size.
        stride: Distance between consecutive windows.
        device: Device to run on.
        label: Prefix for the progress line.

    Returns:
        Mean IoU over the validation split, in ``[0, 1]``.
    """
    head.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    started = time.time()

    for index, (image, mask) in enumerate(loader):
        image, mask = image[0].to(device), mask[0].to(device)  # batch of one, variable size
        _, height, width = image.shape
        # Pad images smaller than the window so at least one crop fits.
        pad_h, pad_w = max(0, crop - height), max(0, crop - width)
        image = F.pad(image, (0, pad_w, 0, pad_h))
        height, width = image.shape[-2:]
        logits_sum = torch.zeros(NUM_CLASSES, height, width, device=device)
        count = torch.zeros(1, height, width, device=device)

        for top in _window_starts(height, crop, stride):
            for left in _window_starts(width, crop, stride):
                crop_image = image[:, top:top + crop, left:left + crop][None]
                tokens, pooled = utils.encode(model, crop_image, device=device, autocast=True)
                logits = head(tokens, pooled, (crop // 16, crop // 16))
                logits = F.interpolate(logits, size=(crop, crop), mode="bilinear")
                logits_sum[:, top:top + crop, left:left + crop] += logits[0]
                count[:, top:top + crop, left:left + crop] += 1

        logits = (logits_sum / count)[:, :mask.shape[0], :mask.shape[1]]
        confusion_update(confusion, logits[None], mask[None])
        if index == 0 or (index + 1) % 200 == 0:
            print(f"  {label}[{index + 1}/{len(loader)}] mIoU so far "
                  f"{mean_iou(confusion) * 100:.2f} ({time.time() - started:.0f}s)", flush=True)

    return mean_iou(confusion)


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warm-up then linear decay to zero, as in Appendix A.3.4.

    Warm-up starts at ``warmup_ratio * lr`` rather than at 0, matching mmseg's
    ``LinearLR`` with ``start_factor=1e-6``.
    """
    warmup_ratio = 1e-6
    if step < args.warmup_iterations:
        progress = (step + 1) / max(1, args.warmup_iterations)
        return args.lr * (warmup_ratio + (1.0 - warmup_ratio) * progress)
    progress = (step - args.warmup_iterations) / max(1, args.iterations - args.warmup_iterations)
    return args.lr * max(0.0, 1.0 - progress)


def main() -> None:
    """Train the segmentation head and report the best sliding-window mIoU."""
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = utils.load_backbone(args.model, device)

    train_set = ADE20K(args.data_root, "training", "train", args.crop_size, seed=args.seed)
    val_set = ADE20K(args.data_root, "validation", "val", args.crop_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"ADE20K: {len(train_set)} train / {len(val_set)} val; "
          f"iters={args.iterations} warmup={args.warmup_iterations} "
          f"batch={args.batch_size} lr={args.lr} wd={args.weight_decay}", flush=True)

    head = SegHead(model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    head.train()
    step, best, best_step, started = 0, 0.0, 0, time.time()
    while step < args.iterations:
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            lr = lr_at(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            tokens, pooled = utils.encode(model, image, device=device, autocast=True)
            grid = args.crop_size // 16
            logits = head(tokens, pooled, (grid, grid))
            # Upsample the per-patch logits to the crop and apply CE there.
            logits = F.interpolate(logits, size=(args.crop_size, args.crop_size), mode="bilinear")
            loss = F.cross_entropy(logits, mask, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                remaining = (time.time() - started) / step * (args.iterations - step)
                print(f"  iter {step}/{args.iterations} loss={loss.item():.4f} "
                      f"lr={lr:.2e} eta {remaining / 60:.0f} min", flush=True)
            if step % args.eval_every == 0 or step == args.iterations:
                miou = slide_evaluate(model, head, val_loader, args.crop_size,
                                      args.eval_stride, device, label=f"iter {step} ")
                print(f"  iter {step} EVAL val_mIoU={miou * 100:.2f}%", flush=True)
                if miou > best:
                    best, best_step = miou, step
                head.train()
            if step >= args.iterations:
                break

    print(f"\nADE20K mIoU: {best * 100:.2f} (best of iter {best_step})")


if __name__ == "__main__":
    main()
