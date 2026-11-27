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
"""ImageNet-1k zero-shot classification through a language-aligned readout.

IronViT is a pure vision encoder: it has no text tower, so it cannot be read out
zero-shot on its own. These checkpoints were distilled against multi-teacher
targets that include Meta's Perception Encoder, and each carries a small MLP head
trained to map its pooled vector into PE-Core-G14-448's image embedding. Applying
that head and then a fixed text classifier gives a CLIP-style zero-shot score.

Everything except the backbone ships as one file, ``--head``::

    {
      "adapter": {fc1.weight, norm.weight, norm.bias, fc2.weight},  # 1152 -> 1280
      "mean": [1280], "std": [1280],   # teacher normalization the adapter inverts
      "classifier": [1280, 1000],       # PE-Core-G14-448 text embeddings
      "classnames": [...],              # 1000 ImageNet-1k names, classifier order
      "logit_scale": 100.0,
    }

The classifier columns are the text embeddings averaged over the 80 standard
prompt templates and L2-normalized. Shipping the matrix rather than the text
tower keeps this repository free of a vendored CLIP implementation.

Scoring is ``100 * normalize(adapter(pooled) * std + mean) @ classifier``,
matching the reference evaluation.

Example::

    python evals/zeroshot.py \\
        --model IRONVIT/IronViT-Softmax-400M \\
        --data-root /path/to/imagenet-1k

The data root is the standard ``ImageFolder`` tree, ``<data-root>/val/<class>/``.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

import utils


def build_transform(image_size: int) -> T.Compose:
    """Squash each image to ``image_size`` square, keeping the aspect ratio distorted.

    Zero-shot follows PE-Core, whose encoders see whole images resized to a
    square (SigLIP-style). This is deliberately *not*
    :func:`utils.build_eval_transform`, the crop-based linear-probe transform:
    cropping throws away the border context the text embeddings were matched
    against, and costs a few tenths of a point.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ]
    )


class PEAdapter(nn.Module):
    """Pooled IronViT feature -> PE-Core-G14-448 image embedding (1280-d)."""

    def __init__(self, in_dim: int = 1152, hidden_dim: int = 1536, out_dim: int = 1280) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.norm(self.fc1(x))))


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="HF repo id or local checkpoint directory")
    parser.add_argument("--data-root", required=True,
                        help="ImageFolder root holding val/ (and train/)")
    parser.add_argument("--head", default=None,
                        help="path to zero-shot head file (defaults to pe_zeroshot_head.pt from --model)")
    parser.add_argument("--image-size", type=int, default=384,
                        help="images are squashed to this square size")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_head_path(model_id: str, head: Optional[str] = None) -> str:
    """Resolve a local path or download ``pe_zeroshot_head.pt`` from ``model_id``."""
    if head is not None:
        if os.path.isfile(head):
            return head
        if os.path.isabs(head) or os.sep in head or "/" in head:
            raise FileNotFoundError(f"zero-shot head not found: {head}")
    filename = head or "pe_zeroshot_head.pt"
    if os.path.isdir(model_id):
        candidate = os.path.join(model_id, filename)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"zero-shot head not found at {candidate}")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_id, filename=filename)


def load_head(path: str) -> dict:
    """Load the adapter, its inverse normalization, and the text classifier."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    missing = {"adapter", "mean", "std", "classifier"} - set(blob)
    if missing:
        raise KeyError(f"{path} is missing {sorted(missing)}")

    state = blob["adapter"]
    adapter = PEAdapter(
        in_dim=int(state["fc1.weight"].shape[1]),
        hidden_dim=int(state["fc1.weight"].shape[0]),
        out_dim=int(state["fc2.weight"].shape[0]),
    )
    adapter.load_state_dict(state, strict=True)

    classifier = blob["classifier"].float()
    if classifier.shape[0] != adapter.fc2.weight.shape[0]:
        raise ValueError(
            f"adapter outputs {adapter.fc2.weight.shape[0]}-d but the classifier "
            f"has {classifier.shape[0]} rows -- mismatched head file"
        )
    return {
        "adapter": adapter.float().eval().requires_grad_(False),
        "mean": torch.as_tensor(blob["mean"], dtype=torch.float32).clone(),
        "std": torch.as_tensor(blob["std"], dtype=torch.float32).clone().clamp_min(1e-6),
        "classifier": classifier.requires_grad_(False),
        "logit_scale": float(blob.get("logit_scale", 100.0)),
        "classnames": blob.get("classnames"),
    }


@torch.no_grad()
def evaluate(model, head: dict, loader: DataLoader, device: torch.device) -> dict:
    """Score the validation split, returning ``acc1``, ``acc5`` and ``n``."""
    adapter = head["adapter"].to(device)
    mean, std = head["mean"].to(device), head["std"].to(device)
    classifier = head["classifier"].to(device)
    scale = head["logit_scale"]

    correct1 = correct5 = total = 0
    for index, (images, labels) in enumerate(loader):
        _, pooled = utils.encode(model, images, device=device)
        # The adapter was trained on raw post-norm MAP vectors, so it consumes
        # them un-normalized; normalization happens on its PE-space output,
        # which is the space the text classifier was built in.
        embedding = adapter(pooled.float()) * std + mean
        logits = scale * F.normalize(embedding, dim=-1) @ classifier
        labels = labels.to(device)
        top5 = logits.topk(5, dim=-1).indices
        correct1 += int((top5[:, 0] == labels).sum())
        correct5 += int((top5 == labels.unsqueeze(1)).any(dim=1).sum())
        total += labels.numel()
        if index % 50 == 0:
            print(f"  {total} images", flush=True)
    return {"acc1": correct1 / total, "acc5": correct5 / total, "n": total}


def main() -> None:
    """Run zero-shot ImageNet-1k and print top-1 / top-5."""
    args = parse_args()
    device = torch.device(args.device)

    model = utils.load_backbone(args.model, device)
    head = load_head(resolve_head_path(args.model, args.head))

    dataset = datasets.ImageFolder(
        os.path.join(args.data_root, "val"),
        transform=build_transform(args.image_size),
    )
    classnames = head["classnames"]
    # The head's names are the readable IN-1k class names; the val folders are
    # named by WNID, so only the count is comparable. Both follow the canonical
    # IN-1k order, which is what makes the column order line up.
    if classnames is not None and len(classnames) != len(dataset.classes):
        raise ValueError(
            f"the head has {len(classnames)} class names but the val tree has "
            f"{len(dataset.classes)} folders -- wrong dataset or head file"
        )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"ImageNet-1k zero-shot: {len(dataset)} images, {len(dataset.classes)} classes")

    metrics = evaluate(model, head, loader, device)
    print(f"\nzero-shot  top1={metrics['acc1'] * 100:.2f}%  "
          f"top5={metrics['acc5'] * 100:.2f}%  (n={metrics['n']})")


if __name__ == "__main__":
    main()
