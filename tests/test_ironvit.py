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
"""Lightweight CPU smoke tests for IronViT model, processor, and eval helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS_DIR = os.path.join(REPO_ROOT, "evals")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if EVALS_DIR not in sys.path:
    sys.path.insert(0, EVALS_DIR)

from ironvit import IronViTConfig, IronViTImageProcessor, IronViTModel  # noqa: E402
import utils as eval_utils  # noqa: E402
import zeroshot as eval_zeroshot  # noqa: E402


def _tiny_config(variant: str = "softmax") -> IronViTConfig:
    return IronViTConfig(
        hidden_size=64,
        depth=4,
        num_heads=4,
        intermediate_size=128,
        num_position_embeddings=16,
        attention_type=variant,
        softmax_layer_indices=[1, 3] if variant == "hybrid" else None,
    )


class ConfigAndModelTests(unittest.TestCase):
    def test_invalid_configs(self) -> None:
        with self.assertRaises(ValueError):
            IronViTConfig(attention_type="invalid")
        with self.assertRaises(ValueError):
            IronViTConfig(attention_type="softmax", softmax_layer_indices=[0])
        with self.assertRaises(ValueError):
            IronViTConfig(attention_type="hybrid", softmax_layer_indices=None)
        with self.assertRaises(ValueError):
            IronViTConfig(hidden_size=65, num_heads=4)
        with self.assertRaises(ValueError):
            IronViTConfig(hidden_size=40, num_heads=4)  # head_dim=10, not divisible by 4
        with self.assertRaises(ValueError):
            IronViTConfig(depth=2, attention_type="hybrid", softmax_layer_indices=[2])

    def test_init_weights_and_no_fake_gradient_checkpointing(self) -> None:
        cfg = _tiny_config("softmax")
        model = IronViTModel(cfg)
        self.assertFalse(model.supports_gradient_checkpointing)
        with torch.no_grad():
            model.patch_embed.proj.weight.fill_(float("nan"))
            model.register_tokens.fill_(float("nan"))
        model.apply(model._init_weights)
        self.assertTrue(torch.isfinite(model.patch_embed.proj.weight).all())
        self.assertTrue(torch.isfinite(model.register_tokens).all())
        self.assertTrue(torch.isfinite(model.map_head.query).all())

    def test_forward_variants_and_shapes(self) -> None:
        pil_sq = Image.new("RGB", (64, 64), (10, 20, 30))
        pil_rect = Image.new("RGB", (96, 64), (40, 50, 60))

        for variant in ("softmax", "hybrid"):
            cfg = _tiny_config(variant)
            model = IronViTModel(cfg).eval()
            proc = IronViTImageProcessor.from_config(cfg)

            inputs = proc([pil_sq, pil_rect], return_tensors="pt")
            with torch.inference_mode():
                out = model(**inputs)
                out_grid = model(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])

            self.assertEqual(tuple(out.patch_features.shape), (4 * 4 + 4 * 6, 64))
            self.assertEqual(tuple(out.global_features.shape), (2, 64))
            self.assertTrue(torch.isfinite(out.patch_features).all())
            self.assertTrue(torch.isfinite(out.global_features).all())
            self.assertTrue(torch.allclose(out.patch_features, out_grid.patch_features))

            raster = model.to_raster_order(out.patch_features, inputs["image_grid_thw"])
            self.assertEqual(len(raster), 2)
            self.assertEqual(tuple(raster[0].shape), (4, 4, 64))
            self.assertEqual(tuple(raster[1].shape), (4, 6, 64))


class ImageProcessorTests(unittest.TestCase):
    def test_pil_numpy_and_tensor_inputs(self) -> None:
        proc = IronViTImageProcessor(size={"height": 64, "width": 64})
        pil_img = Image.new("RGB", (64, 64), (128, 64, 32))
        arr_hwc = np.full((64, 64, 3), 128, dtype=np.uint8)
        arr_chw = np.full((3, 64, 64), 0.5, dtype=np.float32)
        arr_gray = np.full((64, 64), 128, dtype=np.uint8)
        tensor_chw = torch.full((3, 64, 64), 0.5, dtype=torch.float32)
        tensor_bf16 = torch.full((3, 64, 64), 0.5, dtype=torch.bfloat16)

        for inp, batch in (
            (pil_img, 1),
            ([pil_img, pil_img], 2),
            (arr_hwc, 1),
            (arr_chw, 1),
            (arr_gray, 1),
            (tensor_chw, 1),
            (tensor_bf16, 1),
        ):
            out = proc(inp, return_tensors="pt")
            self.assertEqual(tuple(out["pixel_values"].shape), (batch * 16, 3 * 2 * 16 * 16))

        proc_raw = IronViTImageProcessor(size={"height": 64, "width": 64}, do_rescale=False, do_normalize=False)
        raw_out = proc_raw(np.full((64, 64, 3), 0.5, dtype=np.float32), return_tensors="pt")
        self.assertTrue(torch.allclose(raw_out["pixel_values"], torch.full_like(raw_out["pixel_values"], 0.5)))

    def test_dynamic_resize_budget_and_invalid_grid(self) -> None:
        proc_dyn = IronViTImageProcessor(size=None, max_patches=16)
        for w, h in ((128, 64), (4096, 16)):
            out_dyn = proc_dyn(Image.new("RGB", (w, h)), return_tensors="pt")
            _, gh, gw = out_dyn["image_grid_thw"][0].tolist()
            self.assertLessEqual(gh * gw, 16)
            self.assertEqual(gh % proc_dyn.spatial_merge_size, 0)
            self.assertEqual(gw % proc_dyn.spatial_merge_size, 0)

        proc_strict = IronViTImageProcessor(do_resize=False)
        for bad_w, bad_h in ((48, 48), (65, 64)):
            with self.assertRaises(ValueError):
                proc_strict(Image.new("RGB", (bad_w, bad_h)), return_tensors="pt")


class EvalHelperTests(unittest.TestCase):
    def test_eval_transform_and_numerical_parity(self) -> None:
        tf = eval_utils.build_eval_transform(384)
        tensor = tf(Image.new("RGB", (500, 400)))
        self.assertEqual(tuple(tensor.shape), (3, 384, 384))

        cfg = _tiny_config("softmax")
        model = IronViTModel(cfg).eval()
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        images_01 = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)[None]

        tokens, pooled = eval_utils.encode(model, images_01, device=torch.device("cpu"))
        proc = IronViTImageProcessor(size={"height": 64, "width": 96})
        model_inputs = proc(pil_img, return_tensors="pt")
        with torch.inference_mode():
            out = model(**model_inputs)
        raster = model.to_raster_order(out.patch_features, model_inputs["image_grid_thw"])[0]
        self.assertEqual(tuple(raster.shape), (4, 6, 64))
        self.assertTrue(torch.allclose(tokens[0], raster.reshape(24, 64), atol=1e-5))
        self.assertTrue(torch.allclose(pooled, out.global_features.float(), atol=1e-5))

    def test_cache_dir_isolation(self) -> None:
        c1 = eval_utils._cache_dir_for("./cache", "/expA/ckpt", 384)
        c2 = eval_utils._cache_dir_for("./cache", "/expB/ckpt", 384)
        c3 = eval_utils._cache_dir_for("./cache", "/expA/ckpt", 224)
        self.assertNotEqual(c1, c2)
        self.assertNotEqual(c1, c3)

    def test_zeroshot_resolve_and_load_head_without_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            head_path = os.path.join(tmpdir, "pe_zeroshot_head.pt")
            blob = {
                "adapter": {
                    "fc1.weight": torch.zeros(8, 16),
                    "norm.weight": torch.ones(8),
                    "norm.bias": torch.zeros(8),
                    "fc2.weight": torch.zeros(12, 8),
                },
                "mean": torch.zeros(12),
                "std": torch.ones(12),
                "classifier": torch.zeros(12, 5),
                "classnames": ["a", "b", "c", "d", "e"],
                "logit_scale": 100.0,
            }
            torch.save(blob, head_path)

            self.assertEqual(eval_zeroshot.resolve_head_path("unused", head_path), head_path)
            self.assertEqual(eval_zeroshot.resolve_head_path(tmpdir, None), head_path)
            with self.assertRaises(FileNotFoundError):
                eval_zeroshot.resolve_head_path(tmpdir, "missing.pt")
            with self.assertRaises(FileNotFoundError):
                eval_zeroshot.resolve_head_path("IRONVIT/IronViT-Softmax-400M", "/nonexistent/head.pt")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = eval_zeroshot.load_head(head_path)
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(user_warnings, [])
            self.assertEqual(tuple(loaded["classifier"].shape), (12, 5))


if __name__ == "__main__":
    unittest.main()
