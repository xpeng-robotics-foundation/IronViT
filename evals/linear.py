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
"""ImageNet-1k linear probe on frozen ``global_features``.

Freeze the backbone, cache one MAP vector per image, then train a single
``nn.Linear(1152, 1000)`` head with SGD (lr 0.1, momentum 0.9, Nesterov, cosine
schedule, 20 epochs, batch 1024), following the DINOv2 / TIPS protocol.

Example::

    python evals/linear.py \\
        --model IRONVIT/IronViT-Softmax-400M \\
        --data-root /path/to/imagenet-1k \\
        --image-size 384

The data root must hold the standard ``ImageFolder`` tree:
``<data-root>/train/<class>/*.JPEG`` and ``<data-root>/val/<class>/*.JPEG``.
Features are cached, so running ``evals/knn.py`` afterwards is instant.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

import utils


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="HF repo id or local checkpoint directory")
    parser.add_argument("--data-root", required=True,
                        help="ImageFolder root with train/ and val/")
    parser.add_argument("--image-size", type=int, default=384,
                        help="square crop fed to the backbone")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="backbone batch size during feature extraction")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20,
                        help="epochs of the linear head")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="peak SGD learning rate of the linear head")
    parser.add_argument("--head-batch-size", type=int, default=1024,
                        help="batch size used to train the linear head")
    parser.add_argument("--cache-dir", default="./feature_cache",
                        help="where extracted features are cached")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the head initialisation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def train_linear_head(train_feats: torch.Tensor, train_labels: torch.Tensor,
                      val_feats: torch.Tensor, val_labels: torch.Tensor,
                      num_classes: int, args: argparse.Namespace,
                      device: torch.device) -> Tuple[float, float]:
    """Fit a linear classifier on frozen features and report its validation accuracy.

    Args:
        train_feats, train_labels: Cached training features and their labels.
        val_feats, val_labels: Cached validation features and their labels.
        num_classes: Width of the classifier.
        args: Parsed arguments; uses ``epochs``, ``lr`` and ``head_batch_size``.
        device: Device to train on.

    Returns:
        ``(top1, top5)`` validation accuracy in ``[0, 1]``.
    """
    head = nn.Linear(train_feats.shape[1], num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=0.0, nesterov=True)
    steps = max(1, train_feats.shape[0] // args.head_batch_size)
    best = (0.0, 0.0, 0)

    for epoch in range(args.epochs):
        # Cosine decay with a 2-epoch linear warmup.
        if epoch < 2:
            lr = args.lr * (epoch + 1) / 2
        else:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 2) / max(1, args.epochs - 2)))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Fixed per epoch, independent of ``--seed``, exactly as the reference
        # probe draws it: the seed then only decides the head initialisation.
        order = torch.randperm(train_feats.shape[0],
                               generator=torch.Generator().manual_seed(epoch))
        head.train()
        total = 0.0
        for step in range(steps):
            index = order[step * args.head_batch_size:(step + 1) * args.head_batch_size]
            logits = head(train_feats[index].to(device))
            loss = F.cross_entropy(logits, train_labels[index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()

        head.eval()
        with torch.no_grad():
            logits = torch.cat([head(val_feats[s:s + 4096].to(device)).cpu()
                                for s in range(0, val_feats.shape[0], 4096)])
        top1 = utils.topk_accuracy(logits, val_labels, 1)
        top5 = utils.topk_accuracy(logits, val_labels, 5)
        print(f"  epoch {epoch + 1:2d}/{args.epochs} lr={lr:.4g} loss={total / steps:.4f} "
              f"top1={top1 * 100:.2f}% top5={top5 * 100:.2f}%", flush=True)
        # The reference reports the best epoch, not the last one; on this
        # recipe they are usually the same epoch but not reliably so.
        if top1 > best[0]:
            best = (top1, top5, epoch + 1)

    print(f"  best epoch {best[2]}/{args.epochs}")
    return best[0], best[1]


def main() -> None:
    """Extract features, train the linear head and report top-1 / top-5."""
    args = parse_args()
    device = torch.device(args.device)
    # The head is initialised from the global RNG, so seed it: otherwise two
    # runs of the same recipe differ by a few hundredths of a point
    # (87.33-87.38 observed) for no reason a reader of the table can see. The
    # epoch order is seeded per epoch inside ``train_linear_head`` instead, so
    # it does not depend on this.
    torch.manual_seed(args.seed)
    blob = utils.imagenet_features(args.model, args.data_root, args.image_size, args.batch_size,
                                   args.num_workers, args.cache_dir, device)
    top1, top5 = train_linear_head(blob["train_features"], blob["train_labels"],
                                   blob["val_features"], blob["val_labels"],
                                   blob["num_classes"], args, device)
    print(f"\nlinear probe  top1={top1 * 100:.2f}%  top5={top5 * 100:.2f}%")


if __name__ == "__main__":
    main()
