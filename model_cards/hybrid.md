---
license: apache-2.0
tags:
  - vision
  - vision-transformer
  - linear-attention
  - multi-teacher-distillation
  - image-classification
  - semantic-segmentation
  - depth-estimation
library_name: transformers
pipeline_tag: image-feature-extraction
---

# IronViT-Hybrid-400M

A 445M-parameter vision encoder with mixed attention: 9 of 27 blocks use softmax
attention and 18 use linear attention (ELU+1 kernel), reducing high-resolution
prefill cost for long sequences while keeping much of the softmax representation
quality.

Trained in two stages of multi-teacher distillation. Stage II distils from the
softmax capability bridge — the 90k refinement checkpoint of the same run that
produced [IronViT-Softmax-400M](https://huggingface.co/IRONVIT/IronViT-Softmax-400M).

Part of the [IronViT](https://github.com/xpeng-robotics-foundation/IronViT) family.

[ :scroll: [`Paper`](https://arxiv.org/abs/2609.29252)] [ :globe_with_meridians: [`Project page`](https://xpeng-robotics.github.io/iron-fm/?model=vit)] [ 🤗 [`Softmax-400M`](https://huggingface.co/IRONVIT/IronViT-Softmax-400M)]

## Usage

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

model_id = "IRONVIT/IronViT-Hybrid-400M"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

inputs = processor(Image.open("example.jpg").convert("RGB"), return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model(**inputs)

out.patch_features   # [num_patches, 1152], blocked order (block_h, block_w, intra_h, intra_w)
out.global_features  # [1, 1152], MAP-pooled image summary
```

For dense tasks, reorder the patch tokens to a raster grid with
`model.to_raster_order(out.patch_features, inputs["image_grid_thw"])`.

The interface is identical to the softmax variant, so the two are drop-in
substitutes in a VLM or a probe script.

## Architecture

| | |
|---|---|
| hidden / depth / heads | 1152 / 27 / 16 |
| MLP | 4304, `gelu_pytorch_tanh` |
| patch / temporal / merge | 16 / 2 / 2 |
| positions | learned 48×48 grid + 2-D RoPE |
| attention | softmax in blocks `[3, 7, 11, 15, 19, 23, 24, 25, 26]`, linear elsewhere |
| linear kernel | `act(x) = 1 + elu(x)`, `kv = Σ act(k) act(v)ᵀ / (L · 8)`, gated by `sigmoid(gate_proj(x))` |
| read-out | 8 registers + learned-query attention pooling (MAP) |
| parameters | 445M |

## Results

| Benchmark | IronViT-Hybrid-400M | IronViT-Softmax-400M | SigLIP2 NaFlex SO400M |
|---|---|---|---|
| IN-1k linear probe | 87.12 | 87.35 | 87.45 |
| IN-1k kNN | 85.37 | 85.82 | 85.28 |
| IN-1k zero-shot | 82.88 | **83.27** | — |
| ADE20K mIoU | 52.60 | 52.97 | 44.48 |
| NYUv2 RMSE ↓ | 0.343 | 0.339 | 0.459 |

Zero-shot reads the pooled feature out through a small adapter into
PE-Core-G14-448's text space; the adapter and the fixed text classifier ship as
`pe_zeroshot_head.pt` next to this checkpoint, so `evals/zeroshot.py` needs no
text tower.

Probe scripts are in `evals/` of the [GitHub repo](https://github.com/xpeng-robotics-foundation/IronViT).

## Citation

```bibtex
@misc{huang2026ironvit,
  title         = {IronViT: Toward Efficient Generalist Visual Representation Learning},
  author        = {Jiaxi Huang and Yueqi Hu and Xin Zhu and Xiaopeng Zhang and Huiting Qiao and
                   Yanglin Zhang and Zefeng Ji and Rongxue Li and Yifei Xu and Huiying Yu and
                   Wei Liu and Jiayin Zheng and Yinggan Xu and Peipeng Chen and Yin Zhang and
                   Jian Yao},
  year          = {2026},
  eprint        = {2609.29252},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.29252}
}
```

## License

Apache 2.0.
