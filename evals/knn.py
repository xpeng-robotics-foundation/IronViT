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
"""ImageNet-1k kNN evaluation on frozen ``global_features``.

Cosine-similarity weighted vote over the top-k training neighbours, k in
``{10, 20, 100}`` at temperature 0.07, following the DINOv2 / TIPS protocol.

Example::

    python evals/knn.py \\
        --model IRONVIT/IronViT-Softmax-400M \\
        --data-root /path/to/imagenet-1k \\
        --image-size 384

The feature cache written by ``evals/linear.py`` is reused, so running the
probes in either order extracts the backbone features only once.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

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
    parser.add_argument("--cache-dir", default="./feature_cache",
                        help="where extracted features are cached")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def knn(train_feats: torch.Tensor, train_labels: torch.Tensor,
        val_feats: torch.Tensor, val_labels: torch.Tensor, num_classes: int,
        device: torch.device, ks: Tuple[int, ...] = (10, 20, 100),
        temperature: float = 0.07, chunk: int = 512) -> Dict[str, float]:
    """Cosine-similarity weighted kNN, DINOv2 style.

    Args:
        train_feats, train_labels: Cached training features and their labels.
        val_feats, val_labels: Cached validation features and their labels.
        num_classes: Number of classes the vote is accumulated over.
        device: Device to run the similarity search on.
        ks: Neighbour counts to report.
        temperature: Softmax temperature over the cosine similarities.
        chunk: Validation queries processed per step, to bound memory.

    Returns:
        ``{"knn_top1@<k>": accuracy}`` for every ``k`` in ``ks``.
    """
    train = F.normalize(train_feats.to(device=device, dtype=torch.float16), dim=-1)
    correct = {k: 0 for k in ks}
    for start in range(0, val_feats.shape[0], chunk):
        v = F.normalize(val_feats[start:start + chunk].to(device=device, dtype=torch.float16), dim=-1)
        sims, idx = (v @ train.T).topk(min(max(ks), train.shape[0]), dim=-1)
        weights = (sims.float() / temperature).softmax(dim=-1)
        labels = train_labels.to(device)[idx]
        target = val_labels[start:start + chunk].to(device)
        for k in ks:
            votes = torch.zeros((labels.shape[0], num_classes), device=device)
            votes.scatter_add_(1, labels[:, :k], weights[:, :k])
            correct[k] += (votes.argmax(dim=-1) == target).sum().item()
    return {f"knn_top1@{k}": correct[k] / val_feats.shape[0] for k in ks}


def main() -> None:
    """Report top-1 kNN accuracy for every neighbour count."""
    args = parse_args()
    device = torch.device(args.device)
    blob = utils.imagenet_features(args.model, args.data_root, args.image_size, args.batch_size,
                                   args.num_workers, args.cache_dir, device)
    results = knn(blob["train_features"], blob["train_labels"],
                  blob["val_features"], blob["val_labels"], blob["num_classes"], device)
    for name, value in results.items():
        print(f"{name}: {value * 100:.2f}%")


if __name__ == "__main__":
    main()
