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
"""Image preprocessing for IronViT.

Reproduces Qwen3-VL's vision preprocessing:

1. optional resize to a fixed square, or an aspect-preserving resize that caps
   the number of patches (``max_patches``);
2. rescale to ``[0, 1]`` and normalize with mean = std = ``0.5``;
3. split into ``patch_size`` patches in Qwen3-VL "blocked" order and flatten
   each patch to ``C * temporal_patch_size * patch_size**2`` values.

The result feeds straight into :meth:`IronViTModel.forward`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.utils import TensorType

from .configuration_ironvit import IronViTConfig


def _as_image_list(images: ImageInput) -> list:
    """Wrap single 2-D arrays/tensors as 1-element lists before ``make_list_of_images``."""
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 2:
        return [images]
    return make_list_of_images(images)


def _normalize_channels(arr: np.ndarray) -> np.ndarray:
    """Convert a 2-D or 3-D array to HWC with 3 RGB channels."""
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            return np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] == 4:
            return arr[..., :3]
        if arr.shape[-1] == 3:
            return arr
    raise ValueError(f"expected 2-D or 3-D RGB/grayscale image array, got shape {arr.shape}")


class IronViTImageProcessor(BaseImageProcessor):
    """Prepare images for IronViT.

    Args:
        size: Fixed output ``{"height": H, "width": W}`` in pixels. Images are
            resized to this size (aspect ratio ignored) and rounded down to a
            multiple of ``grid_unit``. ``None`` keeps the dynamic path, which
            preserves the aspect ratio and caps the patch count with
            ``max_patches``.
        max_patches: Cap on ``gh * gw`` when ``size`` is ``None``; the image is
            downscaled until it fits.
        patch_size: Spatial size of a patch. Must match the model config.
        temporal_patch_size: Number of frames per patch. Must match the model
            config; still images are repeated along this axis.
        spatial_merge_size: Block edge used by the token order and the merger.
        in_channels: Number of image channels.
        do_resize / do_rescale / do_normalize: Toggle each preprocessing step.
        resample: PIL resampling filter used by the resize.
        rescale_factor: Multiplier applied to the ``[0, 255]`` pixel values.
        image_mean: Per-channel mean subtracted after rescaling. Defaults to
            ``[0.5, 0.5, 0.5]``, as in Qwen3-VL.
        image_std: Per-channel standard deviation divided out after rescaling.
            Defaults to ``[0.5, 0.5, 0.5]``.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        size: Optional[dict] = None,
        max_patches: int = 576,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        resample: int = Image.BICUBIC,
        rescale_factor: float = 1.0 / 255.0,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_patches < spatial_merge_size ** 2:
            raise ValueError(
                f"max_patches ({max_patches}) must be at least spatial_merge_size**2 ({spatial_merge_size ** 2})"
            )
        self.size = size
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.in_channels = in_channels
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

    @classmethod
    def from_config(cls, config: IronViTConfig) -> "IronViTImageProcessor":
        """Build a processor with the patch geometry of ``config``."""
        return cls(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            in_channels=config.in_channels,
        )

    @property
    def grid_unit(self) -> int:
        """Resize granularity: patch x merge. Both sides of the image must be a multiple."""
        return self.patch_size * self.spatial_merge_size

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[TensorType] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """Turn one or more images into model inputs.

        Args:
            images: A PIL image, a list of PIL images, or an array; each is
                converted to RGB.
            return_tensors: Tensor type of the returned features.

        Returns:
            A :class:`~transformers.BatchFeature` with

            * ``pixel_values``: ``[num_patches, C * T * P * P]`` patches in
              blocked order;
            * ``image_grid_thw``: ``[batch, 3]`` integer ``(t, h, w)`` grids.

        Raises:
            ValueError: If ``do_resize=False`` leaves a patch grid that
                ``spatial_merge_size`` does not divide, which the blocked token
                order and the MAP head cannot handle.
        """
        images = _as_image_list(images)
        arrays, grids = [], []
        for image in images:
            arr = self._prepare_array(image)
            if self.do_normalize:
                arr = (arr - np.array(self.image_mean, dtype=np.float32)) / np.array(self.image_std, dtype=np.float32)
            height, width = arr.shape[0], arr.shape[1]
            if height % self.grid_unit or width % self.grid_unit:
                raise ValueError(
                    f"image of size {width}x{height} is not divisible by grid_unit={self.grid_unit} "
                    f"(patch_size={self.patch_size} * spatial_merge_size={self.spatial_merge_size}). "
                    f"Either keep do_resize=True or pad/crop the input so both sides are multiples of {self.grid_unit}."
                )
            gh, gw = height // self.patch_size, width // self.patch_size
            arrays.append(torch.from_numpy(arr).permute(2, 0, 1))  # [C, H, W]
            grids.append([1, gh, gw])

        pixel_values, image_grid_thw = self._patchify(arrays, grids)
        return BatchFeature(
            {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def _prepare_array(self, image: object) -> np.ndarray:
        """Convert one input image to an optionally resized/rescaled ``[H, W, 3]`` float32 array."""
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
            if self.do_resize:
                pil = self._resize(pil)
            arr = np.asarray(pil, dtype=np.float32)
            return arr * self.rescale_factor if self.do_rescale else arr

        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().float().numpy() if image.is_floating_point() else image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            arr = image
        else:
            raise TypeError(f"unsupported image type: {type(image).__name__}")

        arr = _normalize_channels(arr)
        # Preserve floating-point values directly when rescaling is disabled.
        if np.issubdtype(arr.dtype, np.floating) and not self.do_rescale:
            arr = arr.astype(np.float32, copy=False)
            if self.do_resize:
                new_w, new_h = self._target_size(arr.shape[0], arr.shape[1])
                if (new_w, new_h) != (arr.shape[1], arr.shape[0]):
                    resized = F.interpolate(
                        torch.from_numpy(arr).permute(2, 0, 1)[None],
                        size=(new_h, new_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )[0].permute(1, 2, 0).numpy()
                    arr = np.clip(resized, float(arr.min()), float(arr.max())) if arr.size else resized
            return arr

        if np.issubdtype(arr.dtype, np.floating):
            if arr.size and arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        pil = Image.fromarray(arr).convert("RGB")
        if self.do_resize:
            pil = self._resize(pil)
        out = np.asarray(pil, dtype=np.float32)
        return out * self.rescale_factor if self.do_rescale else out

    def _target_size(self, height: int, width: int) -> Tuple[int, int]:
        """Compute ``(new_w, new_h)`` aligned to ``grid_unit`` and capped at ``max_patches``."""
        unit = self.grid_unit
        m = self.spatial_merge_size
        if self.size is not None:
            return max(unit, self.size["width"] // unit * unit), max(unit, self.size["height"] // unit * unit)
        gh, gw = max(1, height // self.patch_size), max(1, width // self.patch_size)
        if gh * gw > self.max_patches:
            scale = (self.max_patches / (gh * gw)) ** 0.5
            gh, gw = max(1, int(gh * scale)), max(1, int(gw * scale))
        bh, bw = max(1, gh // m), max(1, gw // m)
        max_blocks = self.max_patches // (m * m)
        if bh * bw > max_blocks:
            if bw >= bh:
                bw = max(1, max_blocks // bh)
            else:
                bh = max(1, max_blocks // bw)
        return bw * unit, bh * unit

    def _resize(self, image: Image.Image) -> Image.Image:
        """Resize to ``size``, or to the largest grid under ``max_patches``."""
        width, height = image.size
        new_w, new_h = self._target_size(height, width)
        if (new_w, new_h) == (width, height):
            return image
        return image.resize((new_w, new_h), resample=self.resample)

    def _patchify(self, images: List[torch.Tensor], grids: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten every image into blocked-order patches.

        Args:
            images: ``[C, H, W]`` normalized images.
            grids: ``[t, gh, gw]`` per image.

        Returns:
            ``pixel_values`` of shape ``[num_patches, C * T * P * P]`` and
            ``image_grid_thw`` of shape ``[batch, 3]``.
        """
        patches = []
        for image, (_, gh, gw) in zip(images, grids):
            p, t, m = self.patch_size, self.temporal_patch_size, self.spatial_merge_size
            c, h, w = image.shape
            frames = image.unsqueeze(1).repeat(1, t, 1, 1)  # [C, T, H, W]; still images repeat
            # [C, T, H, W] -> [block_h, block_w, intra_h, intra_w, C, T, p, p]
            frames = frames.view(c, t, gh // m, m, p, gw // m, m, p)
            frames = frames.permute(2, 5, 3, 6, 0, 1, 4, 7).reshape(-1, c * t * p * p)
            patches.append(frames)
        return torch.cat(patches, dim=0), torch.tensor(grids, dtype=torch.int64)
