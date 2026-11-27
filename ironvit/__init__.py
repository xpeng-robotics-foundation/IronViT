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
"""IronViT: multi-teacher distilled vision encoders for robotics and VLMs.

Two released variants share one architecture and differ only in the attention
operator used by each block:

``IronViT-Softmax-400M``
    All 27 blocks use softmax attention. Highest quality.

``IronViT-Hybrid-400M``
    Blocks listed in ``config.softmax_layer_indices`` (9 of 27) keep softmax
    attention; the other 18 use linear attention, reducing high-resolution
    prefill cost. Within half a point of the softmax model on classification,
    segmentation and depth probes.

Both variants return

* ``patch_features``: ``[num_patches, hidden_size]`` post-LayerNorm patch
  tokens in Qwen3-VL blocked order (``(block_h, block_w, intra_h, intra_w)``);
* ``global_features``: ``[batch, num_cls_tokens * hidden_size]`` learned-query
  attention-pooled (MAP) summary built from those tokens.

Example::

    from transformers import AutoImageProcessor, AutoModel

    model_id = "IRONVIT/IronViT-Softmax-400M"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype="bfloat16")
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    outputs = model(**processor(images, return_tensors="pt"))
"""

from __future__ import annotations

from .configuration_ironvit import IronViTConfig
from .image_processing_ironvit import IronViTImageProcessor
from .modeling_ironvit import IronViTModel

__all__ = [
    "IronViTConfig",
    "IronViTImageProcessor",
    "IronViTModel",
]

__version__ = "0.1.0"
